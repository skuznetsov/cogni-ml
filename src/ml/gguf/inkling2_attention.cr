require "./inkling2_matmul"
require "./inkling2_weights"

module ML::GGUF
  # Causal channelwise short convolution used by Inkling 2 attention and FFN
  # outputs. The object owns its recent inputs, so each instance is one
  # recurrent sequence/layer stream.
  class Inkling2Convolution
    getter width : Int32
    getter taps : Int32

    @history : Array(Array(Float32)?)
    @history_count = 0
    @next_history = 0

    def initialize(@width : Int32, @taps : Int32)
      raise ArgumentError.new("Inkling 2 convolution width must be positive") unless @width > 0
      raise ArgumentError.new("Inkling 2 convolution needs at least one tap") unless @taps > 0
      @history = Array(Array(Float32)?).new(@taps - 1, nil)
    end

    # `weights` is channel-major with taps ordered from oldest to current,
    # matching ggml_ssm_conv. Optional dynamic taps are head-major with tap 0
    # multiplying the current input, matching Inkling 2's graph loop.
    def apply(input : Array(Float32), weights : Array(Float32), dynamic : Array(Float32)? = nil) : Array(Float32)
      unless input.size == @width
        raise ArgumentError.new("Inkling 2 convolution input width mismatch: expected #{@width}, got #{input.size}")
      end
      unless weights.size == @width * @taps
        raise ArgumentError.new("Inkling 2 convolution weight size mismatch: expected #{@width * @taps}, got #{weights.size}")
      end

      dynamic_heads = 0
      channels_per_head = 0
      if dynamic_values = dynamic
        unless dynamic_values.size > 0 && dynamic_values.size % @taps == 0
          raise ArgumentError.new("Inkling 2 dynamic convolution taps must be a non-empty multiple of tap count")
        end
        dynamic_heads = dynamic_values.size // @taps
        unless @width % dynamic_heads == 0
          raise ArgumentError.new("Inkling 2 dynamic convolution heads must divide channel width")
        end
        channels_per_head = @width // dynamic_heads
      end

      result = Array(Float32).new(@width, 0_f32)
      @width.times do |channel|
        static_sum = 0_f32
        @taps.times do |tap|
          sample = if tap == @taps - 1
                     input[channel]
                   else
                     history_sample(tap, channel)
                   end
          static_sum += sample * weights[channel * @taps + tap]
        end

        dynamic_sum = 0_f32
        if dynamic_values = dynamic
          head = channel // channels_per_head
          @taps.times do |lag|
            window_tap = @taps - 1 - lag
            sample = if window_tap == @taps - 1
                       input[channel]
                     else
                       history_sample(window_tap, channel)
                     end
            dynamic_sum += sample * dynamic_values[head * @taps + lag]
          end
        end

        scale_terms = (dynamic.nil? ? @taps : 2 * @taps) + 1
        result[channel] = (input[channel] + static_sum + dynamic_sum) / Math.sqrt(scale_terms.to_f64).to_f32
      end

      remember(input)
      result
    end

    private def history_sample(window_tap : Int32, channel : Int32) : Float32
      history_size = @taps - 1
      missing = history_size - @history_count
      prior_index = window_tap - missing
      return 0_f32 if prior_index < 0

      slot = if @history_count < history_size
               prior_index
             else
               (@next_history + prior_index) % history_size
             end
      @history[slot].not_nil![channel]
    end

    private def remember(input : Array(Float32)) : Nil
      history_size = @taps - 1
      return if history_size == 0

      @history[@next_history] = input.dup
      @next_history = (@next_history + 1) % history_size
      @history_count += 1 if @history_count < history_size
    end

    def reset : Nil
      @history.fill(nil)
      @history_count = 0
      @next_history = 0
    end
  end

  # Per-layer attention and convolution state. Full attention grows its KV
  # arrays by one token at a time up to max_seq; sliding attention overwrites
  # a fixed-size ring of the most recent window.
  class Inkling2AttentionState
    getter sliding : Bool
    getter capacity : Int32

    @keys : Array(Array(Float32)?)
    @values : Array(Array(Float32)?)
    @positions : Array(Int32)
    @count = 0
    @next_slot = 0
    @last_position : Int32? = nil
    @key_convolution : Inkling2Convolution
    @value_convolution : Inkling2Convolution
    @post_convolution : Inkling2Convolution? = nil

    def initialize(config : Inkling2Config, @sliding : Bool, max_seq : Int32)
      raise ArgumentError.new("Inkling 2 max_seq must be positive") unless max_seq > 0
      @capacity = @sliding ? config.sliding_window : max_seq
      raise ArgumentError.new("Inkling 2 attention window must be positive") unless @capacity > 0

      kv_width = config.n_head_kv * config.head_dim
      @key_convolution = Inkling2Convolution.new(kv_width, config.conv_size)
      @value_convolution = Inkling2Convolution.new(kv_width, config.conv_size)
      @keys = [] of Array(Float32)?
      @values = [] of Array(Float32)?
      @positions = [] of Int32
      if @sliding
        @keys = Array(Array(Float32)?).new(@capacity, nil)
        @values = Array(Array(Float32)?).new(@capacity, nil)
        @positions = Array(Int32).new(@capacity, -1)
      end
    end

    def validate_position!(position : Int32) : Nil
      raise ArgumentError.new("Inkling 2 positions must be non-negative") if position < 0
      if previous = @last_position
        raise ArgumentError.new("Inkling 2 positions must increase within an attention state") if position <= previous
      end
      if !@sliding && @count >= @capacity
        raise ArgumentError.new("Inkling 2 full-attention cache exceeded max_seq #{@capacity}")
      end
    end

    def convolve_key(input : Array(Float32), weights : Array(Float32), dynamic : Array(Float32)?) : Array(Float32)
      @key_convolution.apply(input, weights, dynamic)
    end

    def convolve_value(input : Array(Float32), weights : Array(Float32), dynamic : Array(Float32)?) : Array(Float32)
      @value_convolution.apply(input, weights, dynamic)
    end

    def convolve_output(input : Array(Float32), weights : Array(Float32)) : Array(Float32)
      convolution = @post_convolution
      unless convolution
        unless input.size > 0 && weights.size % input.size == 0
          raise ArgumentError.new("Inkling 2 post-convolution weight size must be a multiple of output width")
        end
        convolution = Inkling2Convolution.new(input.size.to_i32, weights.size.to_i32 // input.size)
        @post_convolution = convolution
      else
        unless convolution.width == input.size && weights.size == convolution.width * convolution.taps
          raise ArgumentError.new("Inkling 2 post-convolution dimensions changed within an attention state")
        end
      end
      convolution.apply(input, weights)
    end

    def append(key : Array(Float32), value : Array(Float32), position : Int32) : Nil
      @last_position = position
      if @sliding
        @keys[@next_slot] = key
        @values[@next_slot] = value
        @positions[@next_slot] = position
        @count += 1 if @count < @capacity
        @next_slot = (@next_slot + 1) % @capacity
      else
        @keys << key
        @values << value
        @positions << position
        @count += 1
      end
    end

    def count : Int32
      @count
    end

    def relative_extent(config : Inkling2Config) : Int32
      @sliding ? config.sliding_window : config.rel_extent
    end

    def reset : Nil
      if @sliding
        @keys.fill(nil)
        @values.fill(nil)
        @positions.fill(-1)
      else
        @keys.clear
        @values.clear
        @positions.clear
      end
      @count = 0
      @next_slot = 0
      @last_position = nil
      @key_convolution.reset
      @value_convolution.reset
      @post_convolution.try(&.reset)
    end

    def key_value_at(index : Int32) : Tuple(Array(Float32), Array(Float32), Int32)
      raise IndexError.new("Inkling 2 attention cache index out of bounds") unless 0 <= index < @count
      slot = if @sliding && @count == @capacity
               (@next_slot + index) % @capacity
             else
               index
             end
      {@keys[slot].not_nil!, @values[slot].not_nil!, @positions[slot]}
    end
  end

  module Inkling2Attention
    # The input vectors are already normalized by the layer builder. This
    # method returns the gated attention projection after its post-convolution,
    # before the caller adds the attention residual.
    def self.forward(
      weights : Inkling2AttentionWeights,
      config : Inkling2Config,
      state : Inkling2AttentionState,
      matmul : Inkling2Matmul,
      q_in : Array(Float32),
      k_in : Array(Float32),
      v_in : Array(Float32),
      position : Int32,
    ) : Array(Float32)
      state.validate_position!(position)

      jobs = [] of {QuantWeight, Array(Float32)}
      jobs << {weights.q, q_in}
      jobs << {weights.k, k_in}
      jobs << {weights.v, v_in}
      jobs << {weights.gate, q_in}
      jobs << {weights.sink, q_in}
      jobs << {weights.rel_proj, q_in}

      dynamic_k_index : Int32? = nil
      dynamic_v_index : Int32? = nil
      unless state.sliding
        if k_projection = weights.k_conv_proj
          dynamic_k_index = jobs.size.to_i32
          jobs << {k_projection, k_in}
        else
          raise ArgumentError.new("Inkling 2 full-attention layer is missing K convolution projection")
        end
        if v_projection = weights.v_conv_proj
          dynamic_v_index = jobs.size.to_i32
          jobs << {v_projection, v_in}
        else
          raise ArgumentError.new("Inkling 2 full-attention layer is missing V convolution projection")
        end
      end

      projections = matmul.multiply_many(jobs)
      unless projections.size == jobs.size
        raise ArgumentError.new("Inkling 2 matmul returned a different number of projections than requested")
      end

      n_head = config.n_head
      n_head_kv = config.n_head_kv
      head_dim = config.head_dim
      rel_dim = config.rel_dim
      rel_extent = state.relative_extent(config)
      conv_size = config.conv_size
      kv_width = n_head_kv * head_dim
      q_width = n_head * head_dim
      validate_projection!(projections[0], q_width, "Q")
      validate_projection!(projections[1], kv_width, "K")
      validate_projection!(projections[2], kv_width, "V")
      validate_projection!(projections[3], q_width, "gate")
      validate_projection!(projections[4], n_head, "sink")
      validate_projection!(projections[5], n_head * rel_dim, "relative position")

      dynamic_k = dynamic_k_index.try { |index| projections[index] }
      dynamic_v = dynamic_v_index.try { |index| projections[index] }
      dynamic_k.try { |values| validate_projection!(values, conv_size * n_head_kv, "dynamic K convolution") }
      dynamic_v.try { |values| validate_projection!(values, conv_size * n_head_kv, "dynamic V convolution") }

      validate_projection!(weights.q_clip, q_width, "Q clip")
      validate_projection!(weights.rel_clip, n_head * rel_dim, "relative clip")
      validate_projection!(weights.rel_bias, rel_extent * rel_dim, "relative bias")
      validate_projection!(weights.k_conv, conv_size * kv_width, "K convolution")
      validate_projection!(weights.v_conv, conv_size * kv_width, "V convolution")
      validate_projection!(weights.post_conv, conv_size * weights.output.out_dim, "post convolution")
      unless weights.output.in_dim == q_width && weights.output.out_dim == config.n_embd
        raise ArgumentError.new("Inkling 2 output matrix dimensions do not match attention width")
      end

      q = projections[0]
      k = state.convolve_key(projections[1], weights.k_conv, dynamic_k)
      v = state.convolve_value(projections[2], weights.v_conv, dynamic_v)

      n_head.times do |head|
        clip_head!(q, head * head_dim, head_dim, weights.q_clip, head * head_dim, config.qk_clip)
        clip_head!(projections[5], head * rel_dim, rel_dim, weights.rel_clip, head * rel_dim, config.rel_clip)
      end
      n_head_kv.times do |head|
        clip_head!(k, head * head_dim, head_dim, nil, 0, config.qk_clip)
      end

      n_head_kv.times do |head|
        rms_norm_head!(v, head * head_dim, head_dim, config.rms_eps)
      end

      tau = if state.sliding
              1_f32
            else
              ratio = (position.to_f64 + 1.0) / config.log_scale_floor.to_f64
              1_f32 + config.log_scale_alpha * Math.log(Math.max(1.0, ratio)).to_f32
            end
      if !state.sliding
        q.map! { |value| value * tau }
        projections[5].map! { |value| value * tau }
      end

      state.append(k, v, position)
      attended = Array(Float32).new(q_width, 0_f32)
      group_size = n_head // n_head_kv
      score_scale = 1.0_f32 / Math.sqrt(head_dim.to_f64).to_f32
      key_count = state.count

      n_head.times do |head|
        kv_head = head // group_size
        head_q_offset = head * head_dim
        head_r_offset = head * rel_dim
        logits = [] of Float32
        values = [] of Array(Float32)
        key_count.times do |key_index|
          key, value, key_position = state.key_value_at(key_index)
          distance = position.to_i64 - key_position.to_i64
          next if distance < 0
          next if state.sliding && distance >= config.sliding_window

          qk = 0_f64
          head_dim.times do |dim|
            qk += q[head_q_offset + dim].to_f64 * key[kv_head * head_dim + dim].to_f64
          end
          score = (qk.to_f32 * score_scale).to_f32

          if distance < rel_extent
            bias_offset = distance.to_i32 * rel_dim
            rel_dot = 0_f64
            rel_dim.times do |dim|
              rel_dot += projections[5][head_r_offset + dim].to_f64 * weights.rel_bias[bias_offset + dim].to_f64
            end
            score += rel_dot.to_f32
          end

          score -= projections[4][head]
          logits << score
          values << value
        end

        # The learned sink is represented as a zero logit in the denominator.
        # `sink` has already been subtracted from every real-key logit above.
        max_logit = 0_f32
        logits.each { |logit| max_logit = logit if logit > max_logit }
        denominator = Math.exp(-max_logit.to_f64)
        logits.each { |logit| denominator += Math.exp((logit - max_logit).to_f64) }
        probabilities = Array(Float64).new(logits.size) do |index|
          Math.exp((logits[index] - max_logit).to_f64) / denominator
        end

        head_dim.times do |dim|
          context = 0_f64
          probabilities.each_with_index do |probability, index|
            context += probability * values[index][kv_head * head_dim + dim].to_f64
          end
          gate = projections[3][head_q_offset + dim]
          sigmoid = (1.0 / (1.0 + Math.exp(-gate.to_f64))).to_f32
          attended[head_q_offset + dim] = (context.to_f32 * sigmoid).to_f32
        end
      end

      projected = matmul.multiply(weights.output, attended)
      validate_projection!(projected, weights.output.out_dim, "output")
      state.convolve_output(projected, weights.post_conv)
    end

    private def self.validate_projection!(values : Array(Float32), expected : Int32, name : String) : Nil
      unless values.size == expected
        raise ArgumentError.new("Inkling 2 #{name} projection size mismatch: expected #{expected}, got #{values.size}")
      end
    end

    private def self.clip_head!(
      values : Array(Float32),
      offset : Int32,
      width : Int32,
      channel_scale : Array(Float32)?,
      scale_offset : Int32,
      threshold : Float32,
    ) : Nil
      norm_sq = 0_f64
      width.times do |index|
        scale = channel_scale ? channel_scale[scale_offset + index] : 1_f32
        value = values[offset + index] * scale
        values[offset + index] = value
        norm_sq += value.to_f64 * value.to_f64
      end
      norm = Math.sqrt(norm_sq).to_f32 + 1.0e-6_f32
      factor = Math.min(1_f32, threshold / norm)
      width.times { |index| values[offset + index] *= factor }
    end

    private def self.rms_norm_head!(values : Array(Float32), offset : Int32, width : Int32, epsilon : Float32) : Nil
      mean_square = 0_f64
      width.times do |index|
        value = values[offset + index].to_f64
        mean_square += value * value
      end
      mean_square /= width
      scale = (1.0 / Math.sqrt(mean_square + epsilon.to_f64)).to_f32
      width.times { |index| values[offset + index] *= scale }
    end
  end
end
