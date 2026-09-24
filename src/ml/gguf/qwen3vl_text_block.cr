# CPU reference slice for one Qwen3-VL text decoder block.
#
# Every tensor value is a BF16 value decoded into Float32. Row-major weight
# arrays use PyTorch [out, in] layout; hidden states are flattened [1, tokens,
# hidden]. This deliberately covers the text-only path only: positions are the
# raw sequence arange, RoPE axes are identical, and no visual embeddings or KV
# cache are admitted here. Attention defaults to eager CPU equations, with an
# optional F32-intermediate SDPA-like mode for matching the PyTorch CPU math
# backend's BF16 attention arithmetic. This remains a single-block reference,
# not a full-model checkpoint parity claim.

module ML::GGUF
  struct Qwen3VLTextBlockConfig
    enum AttentionArithmetic
      # Preserve the explicit BF16 materialization boundaries of the eager
      # reference path.
      Eager

      # Use the F32 intermediates of the PyTorch SDPA math backend for scores,
      # softmax, and the weighted value sum; the attention output remains BF16.
      SdpaF32
    end

    getter hidden_dim : Int32
    getter heads : Int32
    getter kv_heads : Int32
    getter head_dim : Int32
    getter intermediate_dim : Int32
    getter eps : Float32
    getter rope_theta : Float32
    getter attention_arithmetic : AttentionArithmetic

    def initialize(@hidden_dim : Int32, @heads : Int32, @kv_heads : Int32,
                   @head_dim : Int32, @intermediate_dim : Int32,
                   @eps : Float32 = 1e-6_f32, @rope_theta : Float32 = 5_000_000.0_f32,
                   @attention_arithmetic : AttentionArithmetic = AttentionArithmetic::Eager)
      raise ArgumentError.new("hidden_dim must be positive") unless @hidden_dim > 0
      raise ArgumentError.new("heads and head_dim must be positive") unless @heads > 0 && @head_dim > 0
      raise ArgumentError.new("heads * head_dim must equal hidden_dim") unless @heads * @head_dim == @hidden_dim
      raise ArgumentError.new("kv_heads must be positive and divide heads") unless @kv_heads > 0 && @heads.divisible_by?(@kv_heads)
      raise ArgumentError.new("head_dim must be positive and even for text RoPE") unless @head_dim.even?
      raise ArgumentError.new("intermediate_dim must be positive") unless @intermediate_dim > 0
      raise ArgumentError.new("eps must be positive") unless @eps > 0.0_f32
      raise ArgumentError.new("rope_theta must be greater than one") unless @rope_theta > 1.0_f32
    end
  end

  # Explicit single-block weights, with projection matrices row-major [out,in].
  # The arrays are owned by the caller and are expected to contain values
  # decoded from BF16 (the forward path quantizes each consumed value to BF16
  # again so synthetic Float32 inputs follow the same arithmetic contract).
  class Qwen3VLTextBlockWeights
    getter input_layernorm : Array(Float32)
    getter q_proj : Array(Float32)
    getter k_proj : Array(Float32)
    getter v_proj : Array(Float32)
    getter q_norm : Array(Float32)
    getter k_norm : Array(Float32)
    getter o_proj : Array(Float32)
    getter post_attention_layernorm : Array(Float32)
    getter gate_proj : Array(Float32)
    getter up_proj : Array(Float32)
    getter down_proj : Array(Float32)

    def initialize(*, @input_layernorm : Array(Float32), @q_proj : Array(Float32),
                   @k_proj : Array(Float32), @v_proj : Array(Float32),
                   @q_norm : Array(Float32), @k_norm : Array(Float32),
                   @o_proj : Array(Float32),
                   @post_attention_layernorm : Array(Float32),
                   @gate_proj : Array(Float32), @up_proj : Array(Float32),
                   @down_proj : Array(Float32))
    end
  end

  module Qwen3VLTextBlock
    # TextModel.forward constructs arange(seq_len) and repeats it over all four
    # axes when position_ids is absent. In this no-vision route that wins over
    # the attention mask; masked left-pad tokens still occupy sequence slots.
    def self.text_only_position_ids(sequence_length : Int32) : Array(Int32)
      raise ArgumentError.new("sequence_length must not be negative") if sequence_length < 0
      Array(Int32).new(sequence_length) { |index| index }
    end

    # One text-only decoder block for one batch. `attention_mask` is a key mask
    # (true means visible); causality additionally restricts key <= query.
    # Position ids are never compacted according to this mask.
    def self.forward(
      hidden_states : Array(Float32),
      attention_mask : Array(Bool),
      weights : Qwen3VLTextBlockWeights,
      config : Qwen3VLTextBlockConfig,
      *,
      trace : Hash(String, Array(Float32))? = nil,
    ) : Array(Float32)
      hidden_dim = config.hidden_dim
      unless hidden_states.size > 0 && hidden_states.size.divisible_by?(hidden_dim)
        raise ArgumentError.new("hidden_states size mismatch")
      end
      token_count = hidden_states.size // hidden_dim
      unless attention_mask.size == token_count
        raise ArgumentError.new("attention_mask size mismatch")
      end
      validate_weights!(weights, config)

      # The text encoder's hidden states, linear outputs, RMSNorm outputs, and
      # residuals are BF16 module values. Decode-to-F32 inputs are rounded here.
      residual = quantize_bf16(hidden_states)
      trace_boundary(trace, "layer0_input", residual)
      normalized = rms_norm_rows(residual, token_count, hidden_dim, weights.input_layernorm, config.eps)
      trace_boundary(trace, "layers.0.input_layernorm", normalized)

      q = linear(normalized, token_count, hidden_dim, config.heads * config.head_dim, weights.q_proj)
      trace_boundary(trace, "layers.0.self_attn.q_proj", q)
      k = linear(normalized, token_count, hidden_dim, config.kv_heads * config.head_dim, weights.k_proj)
      trace_boundary(trace, "layers.0.self_attn.k_proj", k)
      v = linear(normalized, token_count, hidden_dim, config.kv_heads * config.head_dim, weights.v_proj)
      trace_boundary(trace, "layers.0.self_attn.v_proj", v)
      q = rms_norm_heads(q, token_count, config.heads, config.head_dim, weights.q_norm, config.eps)
      trace_boundary(trace, "layers.0.self_attn.q_norm", q)
      k = rms_norm_heads(k, token_count, config.kv_heads, config.head_dim, weights.k_norm, config.eps)
      trace_boundary(trace, "layers.0.self_attn.k_norm", k)

      positions = text_only_position_ids(token_count)
      q = apply_text_rope(q, token_count, config.heads, config.head_dim, positions, config.rope_theta)
      trace_boundary(trace, "post_rope_q", q)
      k = apply_text_rope(k, token_count, config.kv_heads, config.head_dim, positions, config.rope_theta)
      trace_boundary(trace, "post_rope_k", k)
      attended = causal_gqa_attention(q, k, v, attention_mask, token_count, config)
      trace_boundary(trace, "attended", attended)
      attention_branch = linear(
        attended, token_count, config.hidden_dim, hidden_dim, weights.o_proj
      )
      trace_boundary(trace, "layers.0.self_attn.o_proj", attention_branch)
      after_attention = residual_add(residual, attention_branch)

      residual = after_attention
      normalized = rms_norm_rows(
        residual, token_count, hidden_dim, weights.post_attention_layernorm, config.eps
      )
      trace_boundary(trace, "layers.0.post_attention_layernorm", normalized)
      gate = linear(normalized, token_count, hidden_dim, config.intermediate_dim, weights.gate_proj)
      trace_boundary(trace, "layers.0.mlp.gate_proj", gate)
      up = linear(normalized, token_count, hidden_dim, config.intermediate_dim, weights.up_proj)
      trace_boundary(trace, "layers.0.mlp.up_proj", up)
      activated = Array(Float32).new(gate.size, 0.0_f32)
      gate.size.times do |index|
        silu = silu_bf16(gate[index])
        activated[index] = bf16(silu * up[index])
      end
      mlp_branch = linear(
        activated, token_count, config.intermediate_dim, hidden_dim, weights.down_proj
      )
      trace_boundary(trace, "layers.0.mlp.down_proj", mlp_branch)
      output = residual_add(residual, mlp_branch)
      trace_boundary(trace, "layers.0", output)
      output
    end

    # Diagnostic consumers receive snapshots, never aliases to arrays used by
    # the forward path. The nil default performs no array copy.
    private def self.trace_boundary(trace : Hash(String, Array(Float32))?,
                                    name : String, values : Array(Float32)) : Nil
      if sink = trace
        sink[name] = values.dup
      end
      nil
    end

    private def self.validate_weights!(weights : Qwen3VLTextBlockWeights,
                                       config : Qwen3VLTextBlockConfig) : Nil
      hidden_dim = config.hidden_dim
      query_dim = config.heads * config.head_dim
      kv_dim = config.kv_heads * config.head_dim
      intermediate_dim = config.intermediate_dim
      validate_vector!(weights.input_layernorm, hidden_dim, "input_layernorm")
      validate_matrix!(weights.q_proj, query_dim, hidden_dim, "q_proj")
      validate_matrix!(weights.k_proj, kv_dim, hidden_dim, "k_proj")
      validate_matrix!(weights.v_proj, kv_dim, hidden_dim, "v_proj")
      validate_vector!(weights.q_norm, config.head_dim, "q_norm")
      validate_vector!(weights.k_norm, config.head_dim, "k_norm")
      validate_matrix!(weights.o_proj, hidden_dim, query_dim, "o_proj")
      validate_vector!(weights.post_attention_layernorm, hidden_dim, "post_attention_layernorm")
      validate_matrix!(weights.gate_proj, intermediate_dim, hidden_dim, "gate_proj")
      validate_matrix!(weights.up_proj, intermediate_dim, hidden_dim, "up_proj")
      validate_matrix!(weights.down_proj, hidden_dim, intermediate_dim, "down_proj")
    end

    private def self.validate_vector!(weight : Array(Float32), length : Int32, name : String) : Nil
      raise ArgumentError.new("#{name} weight shape mismatch") unless weight.size == length
    end

    private def self.validate_matrix!(weight : Array(Float32), rows : Int32,
                                      columns : Int32, name : String) : Nil
      raise ArgumentError.new("#{name} weight shape mismatch") unless weight.size == rows * columns
    end

    private def self.quantize_bf16(values : Array(Float32)) : Array(Float32)
      values.map { |value| bf16(value) }
    end

    # IEEE-754 round-to-nearest, ties-to-even conversion, returned as Float32
    # for the host-side reference arithmetic.
    private def self.bf16(value : Float32) : Float32
      bits = value.unsafe_as(UInt32)
      exponent = bits & 0x7f800000_u32
      return (bits & 0xffff0000_u32).unsafe_as(Float32) if exponent == 0x7f800000_u32
      rounded = bits + 0x7fff_u32 + ((bits >> 16) & 1_u32)
      (rounded & 0xffff0000_u32).unsafe_as(Float32)
    end

    private def self.linear(input : Array(Float32), rows : Int32, input_dim : Int32,
                            output_dim : Int32, weight : Array(Float32)) : Array(Float32)
      output = Array(Float32).new(rows * output_dim, 0.0_f32)
      rows.times do |row|
        output_dim.times do |out_index|
          weight_offset = out_index * input_dim
          input_offset = row * input_dim
          sum = 0.0_f32
          input_dim.times do |in_index|
            sum += bf16(weight[weight_offset + in_index]) * input[input_offset + in_index]
          end
          output[row * output_dim + out_index] = bf16(sum)
        end
      end
      output
    end

    private def self.rms_norm_rows(input : Array(Float32), rows : Int32, dim : Int32,
                                   weight : Array(Float32), eps : Float32) : Array(Float32)
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        mean_square = 0.0_f32
        dim.times do |column|
          value = input[offset + column]
          mean_square += value * value
        end
        variance = mean_square / dim.to_f32
        inverse_rms = (1.0_f32 / Math.sqrt((variance + eps).to_f64)).to_f32
        dim.times do |column|
          normalized = bf16(input[offset + column] * inverse_rms)
          output[offset + column] = bf16(normalized * bf16(weight[column]))
        end
      end
      output
    end

    private def self.rms_norm_heads(input : Array(Float32), tokens : Int32, heads : Int32,
                                    head_dim : Int32, weight : Array(Float32),
                                    eps : Float32) : Array(Float32)
      rows = tokens * heads
      rms_norm_rows(input, rows, head_dim, weight, eps)
    end

    private def self.apply_text_rope(input : Array(Float32), tokens : Int32, heads : Int32,
                                     head_dim : Int32, positions : Array(Int32),
                                     theta : Float32) : Array(Float32)
      half_dim = head_dim // 2
      output = Array(Float32).new(input.size, 0.0_f32)
      tokens.times do |token|
        heads.times do |head|
          base = (token * heads + head) * head_dim
          half_dim.times do |pair|
            exponent = (2 * pair).to_f64 / head_dim.to_f64
            inverse_frequency = Math.exp(-exponent * Math.log(theta.to_f64)).to_f32
            angle = (positions[token].to_f32 * inverse_frequency).to_f32
            cos = bf16(Math.cos(angle.to_f64).to_f32)
            sin = bf16(Math.sin(angle.to_f64).to_f32)
            first = bf16(input[base + pair])
            second = bf16(input[base + pair + half_dim])
            output[base + pair] = bf16(
              bf16(first * cos) + bf16(-second * sin)
            )
            output[base + pair + half_dim] = bf16(
              bf16(second * cos) + bf16(first * sin)
            )
          end
        end
      end
      output
    end

    private def self.causal_gqa_attention(q : Array(Float32), k : Array(Float32),
                                          v : Array(Float32), attention_mask : Array(Bool),
                                          tokens : Int32,
                                          config : Qwen3VLTextBlockConfig) : Array(Float32)
      case config.attention_arithmetic
      when .eager?
        eager_causal_gqa_attention(q, k, v, attention_mask, tokens, config)
      when .sdpa_f32?
        sdpa_f32_causal_gqa_attention(q, k, v, attention_mask, tokens, config)
      else
        raise ArgumentError.new("unsupported Qwen3VL attention arithmetic")
      end
    end

    private def self.eager_causal_gqa_attention(q : Array(Float32), k : Array(Float32),
                                                v : Array(Float32), attention_mask : Array(Bool),
                                                tokens : Int32,
                                                config : Qwen3VLTextBlockConfig) : Array(Float32)
      hidden_dim = config.hidden_dim
      heads = config.heads
      kv_heads = config.kv_heads
      head_dim = config.head_dim
      groups = heads // kv_heads
      output = Array(Float32).new(tokens * hidden_dim, 0.0_f32)
      scale = 1.0_f32 / Math.sqrt(head_dim.to_f64).to_f32

      tokens.times do |query_index|
        heads.times do |head|
          kv_head = head // groups
          query_offset = (query_index * heads + head) * head_dim
          visible_keys = [] of Int32
          scores = [] of Float32
          (query_index + 1).times do |key_index|
            next unless attention_mask[key_index]
            key_offset = (key_index * kv_heads + kv_head) * head_dim
            dot = 0.0_f32
            head_dim.times do |column|
              dot += q[query_offset + column] * k[key_offset + column]
            end
            # eager_attention_forward materializes BF16 scores, applies the
            # scalar scale in that dtype, then casts softmax output to BF16.
            scores << bf16(bf16(dot) * scale)
            visible_keys << key_index
          end

          next if visible_keys.empty?
          maximum = scores.max
          exponentials = scores.map { |score| Math.exp((score - maximum).to_f64).to_f32 }
          denominator = 0.0_f32
          exponentials.each { |value| denominator += value }
          attended = Array(Float32).new(head_dim, 0.0_f32)
          visible_keys.each_with_index do |key_index, score_index|
            probability = bf16(exponentials[score_index] / denominator)
            value_offset = (key_index * kv_heads + kv_head) * head_dim
            head_dim.times do |column|
              attended[column] += probability * v[value_offset + column]
            end
          end
          head_dim.times do |column|
            output[query_index * hidden_dim + head * head_dim + column] = bf16(attended[column])
          end
        end
      end
      output
    end

    # Mirrors the CPU SDPA math path for BF16 Q/K/V: Q/K dot products, scaled
    # scores, softmax, and the weighted V sum stay Float32. Inputs and the
    # materialized attention result remain BF16 module values.
    private def self.sdpa_f32_causal_gqa_attention(q : Array(Float32), k : Array(Float32),
                                                   v : Array(Float32), attention_mask : Array(Bool),
                                                   tokens : Int32,
                                                   config : Qwen3VLTextBlockConfig) : Array(Float32)
      hidden_dim = config.hidden_dim
      heads = config.heads
      kv_heads = config.kv_heads
      head_dim = config.head_dim
      groups = heads // kv_heads
      output = Array(Float32).new(tokens * hidden_dim, 0.0_f32)
      scale = 1.0_f32 / Math.sqrt(head_dim.to_f64).to_f32

      tokens.times do |query_index|
        heads.times do |head|
          kv_head = head // groups
          query_offset = (query_index * heads + head) * head_dim
          visible_keys = [] of Int32
          scores = [] of Float32
          (query_index + 1).times do |key_index|
            next unless attention_mask[key_index]
            key_offset = (key_index * kv_heads + kv_head) * head_dim
            dot = 0.0_f32
            head_dim.times do |column|
              dot += q[query_offset + column] * k[key_offset + column]
            end
            scores << dot * scale
            visible_keys << key_index
          end

          next if visible_keys.empty?
          maximum = scores.max
          exponentials = scores.map { |score| Math.exp((score - maximum).to_f64).to_f32 }
          denominator = 0.0_f32
          exponentials.each { |value| denominator += value }
          attended = Array(Float32).new(head_dim, 0.0_f32)
          visible_keys.each_with_index do |key_index, score_index|
            probability = exponentials[score_index] / denominator
            value_offset = (key_index * kv_heads + kv_head) * head_dim
            head_dim.times do |column|
              attended[column] += probability * v[value_offset + column]
            end
          end
          head_dim.times do |column|
            output[query_index * hidden_dim + head * head_dim + column] = bf16(attended[column])
          end
        end
      end
      output
    end

    private def self.silu_bf16(value : Float32) : Float32
      sigmoid = if value >= 0.0_f32
                  1.0_f32 / (1.0_f32 + Math.exp(-value.to_f64).to_f32)
                else
                  exponential = Math.exp(value.to_f64).to_f32
                  exponential / (1.0_f32 + exponential)
                end
      bf16(value * sigmoid)
    end

    private def self.residual_add(residual : Array(Float32), branch : Array(Float32)) : Array(Float32)
      residual.zip(branch).map { |left, right| bf16(left + right) }
    end
  end
end
