require "./codec"
require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  # Device-side DAC decoder. Activations stay in Resident::Buffer allocations;
  # the only host copies are explicit trace reads or the caller's terminal read.
  class CodecResident
    SOURCE = {{ read_file("#{__DIR__}/kernels/codec_resident.metal") }}

    private struct Activation
      getter rows : Int32
      getter channels : Int32
      getter data : Resident::Buffer

      def initialize(@rows : Int32, @channels : Int32, @data : Resident::Buffer)
      end
    end

    getter config : CodecConfig
    property trace : Proc(String, Vector, Nil)?
    # Diagnostic observers can enqueue comparisons or retain immutable device
    # activations without introducing a host read or completion barrier.
    property device_trace : Proc(String, Resident::Buffer, Int32, Int32, Nil)?

    @store : WeightStore
    @context : Resident::Context
    @prefix : String

    def initialize(@store : WeightStore, @context : Resident::Context,
                   @config : CodecConfig = CodecConfig.default, @prefix : String = "codec.")
      @trace = nil
      @device_trace = nil
    end

    # `codes` is either frame-major [frames,main,c0,...,cN] or, when
    # codebook_major is true, [c0,c1,...,cN,frames]. The native Generator's
    # codebooks method consumes c0 as semantic and c1..cN as residual codes;
    # the main token is outside the codec codebooks.
    def decode(codes : Resident::Buffer, frames : Int32, codebook_major : Bool = false) : Resident::Buffer
      Profile.measure("codec_resident.decode", category: "codec") do
        decode_impl(codes, frames, codebook_major)
      end
    rescue ex
      begin
        @context.abort
      rescue
        # Keep the original inference failure if error-path draining fails.
      end
      raise ex
    end

    private def decode_impl(codes : Resident::Buffer, frames : Int32,
                            codebook_major : Bool) : Resident::Buffer
      raise ArgumentError.new("codec input must contain at least one frame") unless frames > 0
      books = @config.residual_codebooks + 1
      input_width = codebook_major ? books : books + 1
      expected_codes = checked_count(frames, input_width)
      raise ArgumentError.new("codec token buffer shape mismatch") unless codes.count == expected_codes

      latent = decode_codebooks(codes, frames, input_width, codebook_major)
      trace_activation("quantizer.codebook_sum", latent)
      latent = post_transformer(latent)
      trace_activation("quantizer.post_module", latent)
      @context.guard(latent.data)
      @context.submit

      @config.quantizer_upsample_rates.reverse.each_with_index do |rate, stage|
        prefix = "quantizer.upsample.#{stage}"
        latent = conv_transpose1d(latent, "#{prefix}.0.conv", rate, rate, false)
        trace_activation("#{prefix}.deconv", latent)
        @context.submit
        latent = convnext(latent, "#{prefix}.1")
        trace_activation("#{prefix}.convnext", latent)
        @context.guard(latent.data)
        @context.submit
      end

      waveform = decode_waveform(latent)
      expected_frames = frames
      @config.quantizer_upsample_rates.each { |rate| expected_frames = checked_count(expected_frames, rate) }
      @config.decoder_rates.each { |rate| expected_frames = checked_count(expected_frames, rate) }
      raise ArgumentError.new("DAC output length mismatch") unless waveform.rows == expected_frames && waveform.channels == 1
      @context.guard(waveform.data)
      trace_activation("waveform", waveform)
      # Include the final GPU completion in codec wall time. `read` also
      # synchronizes for terminal/diagnostic callers, but ordinary callers
      # should not need a host readback to ensure completion.
      @context.finish
      waveform.data
    end

    private def decode_codebooks(codes : Resident::Buffer, frames : Int32, input_width : Int32,
                                 codebook_major : Bool) : Activation
      channels = @config.latent_channels
      latent = Activation.new(frames, channels, @context.zeros(checked_count(frames, channels)))
      (@config.residual_codebooks + 1).times do |book|
        prefix, vocab = if book == 0
                          {"quantizer.semantic_quantizer.quantizers.0", @config.semantic_codebook_size}
                        else
                          {"quantizer.quantizer.quantizers.#{book - 1}", @config.residual_codebook_size}
                        end
        table_tensor = tensor("#{prefix}.codebook.weight")
        validate_shape!(table_tensor, [vocab, @config.codebook_dim], "#{prefix}.codebook.weight")
        embedded = @context.alloc(checked_count(frames, @config.codebook_dim))
        @context.dispatch(
          "dr3c_embedding", SOURCE, [codes, @context.tensor(table_tensor), embedded],
          [frames.to_u64, input_width.to_u64, book.to_u64, vocab.to_u64,
           @config.codebook_dim.to_u64, codebook_major ? 1_u64 : 0_u64], embedded.count,
        )
        if callback = @device_trace
          callback.call("quantizer.book.#{book}.embedding", embedded, frames, @config.codebook_dim)
        end
        projected = conv1d(Activation.new(frames, @config.codebook_dim, embedded),
          "#{prefix}.out_proj", 1, 1, 1, 1, true)
        if callback = @device_trace
          callback.call("quantizer.book.#{book}.projection", projected.data, frames, channels)
        end
        latent = Activation.new(frames, channels, @context.add(latent.data, projected.data))
      end
      @context.submit
      latent
    end

    private def post_transformer(input : Activation) : Activation
      hidden = input
      @config.post_layers.times do |layer_index|
        prefix = "quantizer.post_module.layers.#{layer_index}"
        hidden = transformer_layer(hidden, prefix)
        trace_activation("#{prefix}", hidden)
        @context.guard(hidden.data)
        @context.submit
      end
      weight = tensor("quantizer.post_module.norm.weight")
      validate_shape!(weight, [input.channels], "quantizer.post_module.norm.weight")
      normalized = @context.rms(hidden.data, @context.tensor(weight), @config.norm_eps, heads: hidden.rows)
      Activation.new(hidden.rows, hidden.channels, normalized)
    end

    private def transformer_layer(input : Activation, prefix : String) : Activation
      time = input.rows
      channels = input.channels
      heads = @config.transformer_heads
      head_dim = @config.transformer_head_dim
      raise ArgumentError.new("transformer input channel mismatch") unless channels == heads * head_dim

      attention_norm_weight = tensor("#{prefix}.attention_norm.weight")
      validate_shape!(attention_norm_weight, [channels], "#{prefix}.attention_norm.weight")
      normalized = @context.rms(input.data, @context.tensor(attention_norm_weight), @config.norm_eps, heads: time)
      qkv_weight = tensor("#{prefix}.attention.wqkv.weight")
      validate_shape!(qkv_weight, [channels * 3, channels], "#{prefix}.attention.wqkv.weight")
      qkv = linear(normalized, time, channels, "#{prefix}.attention.wqkv.weight")
      query = @context.alloc(checked_count(time, channels))
      key = @context.alloc(query.count)
      value = @context.alloc(query.count)
      @context.dispatch("dr3c_qkv_split", SOURCE, [qkv, query, key, value],
        [time.to_u64, channels.to_u64], query.count)
      rope_params = [time.to_u64, heads.to_u64, head_dim.to_u64, @config.rope_base.unsafe_as(UInt32).to_u64]
      @context.dispatch("dr3c_rope_all", SOURCE, [query], rope_params, time * heads * (head_dim // 2))
      @context.dispatch("dr3c_rope_all", SOURCE, [key], rope_params, time * heads * (head_dim // 2))

      window = @config.transformer_window
      score_count = checked_count(time, checked_count(heads, window))
      scores = @context.alloc(score_count)
      scale = (1.0_f32 / Math.sqrt(head_dim.to_f32)).to_f32
      @context.dispatch("dr3c_attention_scores", SOURCE, [query, key, scores],
        [time.to_u64, heads.to_u64, head_dim.to_u64, window.to_u64, scale.unsafe_as(UInt32).to_u64], score_count)
      probabilities = @context.alloc(score_count)
      @context.dispatch("dr3c_attention_probs", SOURCE, [scores, probabilities],
        [time.to_u64, heads.to_u64, window.to_u64], checked_count(time, heads))
      attended = @context.alloc(checked_count(time, channels))
      @context.dispatch("dr3c_attention_values", SOURCE, [probabilities, value, attended],
        [time.to_u64, heads.to_u64, head_dim.to_u64, window.to_u64], attended.count)

      wo_weight = tensor("#{prefix}.attention.wo.weight")
      validate_shape!(wo_weight, [channels, channels], "#{prefix}.attention.wo.weight")
      attention_projection = linear(attended, time, channels, "#{prefix}.attention.wo.weight")
      attention_scale = tensor("#{prefix}.attention_layer_scale.gamma")
      validate_shape!(attention_scale, [channels], "#{prefix}.attention_layer_scale.gamma")
      attention_projection = channel_scale(attention_projection, time, channels, @context.tensor(attention_scale))
      after_attention = Activation.new(time, channels, @context.add(input.data, attention_projection))

      ffn_norm_weight = tensor("#{prefix}.ffn_norm.weight")
      validate_shape!(ffn_norm_weight, [channels], "#{prefix}.ffn_norm.weight")
      normalized_ffn = @context.rms(after_attention.data, @context.tensor(ffn_norm_weight), @config.norm_eps, heads: time)
      ff = @config.transformer_ff_dim
      w1_weight = tensor("#{prefix}.feed_forward.w1.weight")
      w3_weight = tensor("#{prefix}.feed_forward.w3.weight")
      validate_shape!(w1_weight, [ff, channels], "#{prefix}.feed_forward.w1.weight")
      validate_shape!(w3_weight, [ff, channels], "#{prefix}.feed_forward.w3.weight")
      w1 = linear(normalized_ffn, time, channels, "#{prefix}.feed_forward.w1.weight")
      w3 = linear(normalized_ffn, time, channels, "#{prefix}.feed_forward.w3.weight")
      gated = @context.alloc(checked_count(time, ff))
      @context.dispatch("dr3c_silu_product", SOURCE, [w1, w3, gated], [gated.count.to_u64], gated.count)
      w2_weight = tensor("#{prefix}.feed_forward.w2.weight")
      validate_shape!(w2_weight, [channels, ff], "#{prefix}.feed_forward.w2.weight")
      projected = linear(gated, time, ff, "#{prefix}.feed_forward.w2.weight")
      ffn_scale = tensor("#{prefix}.ffn_layer_scale.gamma")
      validate_shape!(ffn_scale, [channels], "#{prefix}.ffn_layer_scale.gamma")
      projected = channel_scale(projected, time, channels, @context.tensor(ffn_scale))
      Activation.new(time, channels, @context.add(after_attention.data, projected))
    end

    private def convnext(input : Activation, prefix : String) : Activation
      channels = input.channels
      x = conv1d(input, "#{prefix}.dwconv.conv", 7, 1, channels, 1, false)
      gamma = tensor("#{prefix}.norm.weight")
      beta = tensor("#{prefix}.norm.bias")
      validate_shape!(gamma, [channels], "#{prefix}.norm.weight")
      validate_shape!(beta, [channels], "#{prefix}.norm.bias")
      normalized = @context.alloc(x.data.count)
      @context.dispatch("dr3c_layer_norm", SOURCE, [x.data, @context.tensor(gamma), @context.tensor(beta), normalized],
        [x.rows.to_u64, channels.to_u64, 1.0e-6_f32.unsafe_as(UInt32).to_u64], x.rows)

      hidden = checked_count(channels, 4)
      pw1_weight = tensor("#{prefix}.pwconv1.weight")
      validate_shape!(pw1_weight, [hidden, channels], "#{prefix}.pwconv1.weight")
      projected = linear(normalized, x.rows, channels, "#{prefix}.pwconv1.weight", "#{prefix}.pwconv1.bias")
      activated = @context.alloc(projected.count)
      @context.dispatch("dr3c_gelu", SOURCE, [projected, activated], [projected.count.to_u64], projected.count)
      pw2_weight = tensor("#{prefix}.pwconv2.weight")
      validate_shape!(pw2_weight, [channels, hidden], "#{prefix}.pwconv2.weight")
      residual = linear(activated, x.rows, hidden, "#{prefix}.pwconv2.weight", "#{prefix}.pwconv2.bias")
      gamma_scale = tensor("#{prefix}.gamma")
      validate_shape!(gamma_scale, [channels], "#{prefix}.gamma")
      residual = channel_scale(residual, x.rows, channels, @context.tensor(gamma_scale))
      Activation.new(x.rows, channels, @context.add(input.data, residual))
    end

    private def decode_waveform(input : Activation) : Activation
      x = conv1d(input, "decoder.model.0", 7, 1, 1, 1, true)
      trace_activation("decoder.model.0", x)
      @context.submit
      channels = @config.decoder_channels
      @config.decoder_rates.each_with_index do |rate, stage|
        output_dim = channels >> (stage + 1)
        prefix = "decoder.model.#{stage + 1}.block"
        x = snake(x, "#{prefix}.0.alpha")
        trace_activation("decoder.model.#{stage + 1}.block.0", x)
        x = conv_transpose1d(x, "#{prefix}.1", rate * 2, rate, true)
        trace_activation("decoder.model.#{stage + 1}.block.1", x)
        @context.submit
        3.times do |unit_index|
          unit_prefix = "#{prefix}.#{unit_index + 2}.block"
          x = residual_unit(x, unit_prefix, output_dim, [1, 3, 9][unit_index])
          trace_activation("decoder.model.#{stage + 1}.block.#{unit_index + 2}", x)
          @context.submit
        end
        raise ArgumentError.new("decoder stage output width mismatch") unless x.channels == output_dim
      end

      final_index = @config.decoder_rates.size + 1
      x = snake(x, "decoder.model.#{final_index}.alpha")
      trace_activation("decoder.model.#{final_index}", x)
      x = conv1d(x, "decoder.model.#{final_index + 1}", 7, 1, 1, 1, true)
      trace_activation("decoder.model.#{final_index + 1}", x)
      mono = @context.alloc(x.rows)
      @context.dispatch("dr3c_tanh_mono", SOURCE, [x.data, mono], [x.rows.to_u64, x.channels.to_u64], x.rows)
      Activation.new(x.rows, 1, mono)
    end

    private def residual_unit(input : Activation, prefix : String, channels : Int32, dilation : Int32) : Activation
      raise ArgumentError.new("residual unit input width mismatch") unless input.channels == channels
      y = snake(input, "#{prefix}.0.alpha")
      y = conv1d(y, "#{prefix}.1", 7, 1, 1, dilation, true)
      y = snake(y, "#{prefix}.2.alpha")
      y = conv1d(y, "#{prefix}.3", 1, 1, 1, 1, true)
      Activation.new(input.rows, input.channels, @context.add(input.data, y.data))
    end

    private def conv1d(input : Activation, prefix : String, kernel : Int32, stride : Int32,
                       groups : Int32, dilation : Int32, weight_normalized : Bool) : Activation
      raise ArgumentError.new("invalid Conv1d parameters") unless kernel > 0 && stride > 0 && dilation > 0 && groups > 0 && input.channels % groups == 0
      weight_name = weight_normalized ? "#{prefix}.weight_v" : "#{prefix}.weight"
      weight = tensor(weight_name)
      raise ArgumentError.new("invalid Conv1d weight rank for #{weight_name}") unless weight.shape.size == 3
      out_channels = weight.shape[0]
      in_per_group = input.channels // groups
      validate_shape!(weight, [out_channels, in_per_group, kernel], weight_name)
      raise ArgumentError.new("Conv1d output channels not divisible by groups") unless out_channels % groups == 0
      bias_tensor = tensor("#{prefix}.bias")
      validate_shape!(bias_tensor, [out_channels], "#{prefix}.bias")
      resident_weight = @context.tensor(weight)
      if weight_normalized
        gain = tensor("#{prefix}.weight_g")
        validate_element_count!(gain, out_channels, "#{prefix}.weight_g")
        resident_weight = effective_weight(resident_weight, @context.tensor(gain), out_channels,
          checked_count(in_per_group, kernel))
      end
      output = @context.alloc(checked_count(input.rows, out_channels))
      @context.dispatch("dr3c_conv1d", SOURCE,
        [input.data, resident_weight, @context.tensor(bias_tensor), output],
        [input.rows.to_u64, input.channels.to_u64, out_channels.to_u64, kernel.to_u64,
         stride.to_u64, groups.to_u64, dilation.to_u64], output.count)
      Activation.new(input.rows, out_channels, output)
    end

    private def conv_transpose1d(input : Activation, prefix : String, kernel : Int32,
                                 stride : Int32, weight_normalized : Bool) : Activation
      raise ArgumentError.new("invalid ConvTranspose1d parameters") unless kernel > 0 && stride > 0
      weight_name = weight_normalized ? "#{prefix}.weight_v" : "#{prefix}.weight"
      weight = tensor(weight_name)
      raise ArgumentError.new("invalid ConvTranspose1d weight rank") unless weight.shape.size == 3
      raise ArgumentError.new("invalid ConvTranspose1d weight shape") unless weight.shape[0] == input.channels && weight.shape[2] == kernel
      out_channels = weight.shape[1]
      bias_tensor = tensor("#{prefix}.bias")
      validate_shape!(bias_tensor, [out_channels], "#{prefix}.bias")
      resident_weight = @context.tensor(weight)
      if weight_normalized
        gain = tensor("#{prefix}.weight_g")
        validate_element_count!(gain, input.channels, "#{prefix}.weight_g")
        resident_weight = effective_weight(resident_weight, @context.tensor(gain), input.channels,
          checked_count(out_channels, kernel))
      end
      rows = checked_count(input.rows, stride)
      output = @context.alloc(checked_count(rows, out_channels))
      @context.dispatch("dr3c_conv_transpose1d", SOURCE,
        [input.data, resident_weight, @context.tensor(bias_tensor), output],
        [input.rows.to_u64, input.channels.to_u64, out_channels.to_u64, kernel.to_u64, stride.to_u64], output.count)
      Activation.new(rows, out_channels, output)
    end

    private def effective_weight(weight : Resident::Buffer, gain : Resident::Buffer,
                                  channels : Int32, per_channel : Int32) : Resident::Buffer
      raise ArgumentError.new("invalid weight-normalization shape") unless channels > 0 && per_channel > 0 && weight.count == checked_count(channels, per_channel)
      scales = @context.alloc(channels)
      @context.dispatch("dr3c_wn_scales", SOURCE, [weight, gain, scales],
        [channels.to_u64, per_channel.to_u64], channels)
      effective = @context.alloc(weight.count)
      @context.dispatch("dr3c_wn_apply", SOURCE, [weight, scales, effective],
        [weight.count.to_u64, per_channel.to_u64], weight.count)
      effective
    end

    private def snake(input : Activation, alpha_name : String) : Activation
      alpha = tensor(alpha_name)
      validate_shape!(alpha, [input.channels], alpha_name)
      output = @context.alloc(input.data.count)
      @context.dispatch("dr3c_snake", SOURCE, [input.data, @context.tensor(alpha), output],
        [input.rows.to_u64, input.channels.to_u64], output.count)
      Activation.new(input.rows, input.channels, output)
    end

    private def linear(input : Resident::Buffer, rows : Int32, in_channels : Int32,
                       weight_name : String, bias_name : String? = nil) : Resident::Buffer
      weight = tensor(weight_name)
      raise ArgumentError.new("invalid dense weight rank for #{weight_name}") unless weight.shape.size == 2
      out_channels = weight.shape[0]
      validate_shape!(weight, [out_channels, in_channels], weight_name)
      raise ArgumentError.new("dense input width mismatch") unless input.count == checked_count(rows, in_channels)
      output = @context.alloc(checked_count(rows, out_channels))
      resident_weight = @context.tensor(weight)
      if name = bias_name
        bias = tensor(name)
        validate_shape!(bias, [out_channels], name)
        @context.dispatch("dr3c_linear_bias", SOURCE, [input, resident_weight, @context.tensor(bias), output],
          [rows.to_u64, in_channels.to_u64, out_channels.to_u64], output.count)
      else
        @context.dispatch("dr3c_linear", SOURCE, [input, resident_weight, output],
          [rows.to_u64, in_channels.to_u64, out_channels.to_u64], output.count)
      end
      output
    end

    private def channel_scale(input : Resident::Buffer, rows : Int32, channels : Int32,
                              scale : Resident::Buffer) : Resident::Buffer
      raise ArgumentError.new("channel scale shape mismatch") unless input.count == checked_count(rows, channels) && scale.count == channels
      output = @context.alloc(input.count)
      @context.dispatch("dr3c_channel_scale", SOURCE, [input, scale, output],
        [rows.to_u64, channels.to_u64], output.count)
      output
    end

    private def tensor(name : String) : Tensor
      candidates = ["#{@prefix}#{name}"]
      if name.ends_with?(".weight_v") || name.ends_with?(".weight_g")
        stem = name[0...(name.size - 9)]
        original = name.ends_with?(".weight_v") ? "original1" : "original0"
        candidates << "#{@prefix}#{stem}.conv.parametrizations.weight.#{original}"
        candidates << "#{@prefix}#{stem}.parametrizations.weight.#{original}"
      elsif name.ends_with?(".bias")
        stem = name[0...(name.size - 5)]
        candidates << "#{@prefix}#{stem}.conv.bias"
      end
      candidates.each do |candidate|
        begin
          return @store.tensor(candidate)
        rescue ex : ArgumentError
          # Try the remaining checkpoint spellings used by PyTorch/MLX.
        end
      end
      raise ArgumentError.new("missing codec tensor #{@prefix}#{name}; tried #{candidates.join(", ")}")
    end

    private def validate_shape!(value : Tensor, expected : Array(Int32), name : String) : Nil
      if expected.size == 1
        return validate_element_count!(value, expected[0], name)
      end
      raise ArgumentError.new("#{name} shape #{value.shape} does not match #{expected}") unless value.shape == expected
    end

    # safetensors checkpoints may retain PyTorch's singleton dimensions for
    # weight_g (for example [channels, 1, 1]); the CPU codec consumes this as a
    # flat channel vector, so validate its element count while preserving the
    # original resident buffer layout.
    private def validate_element_count!(value : Tensor, expected : Int32, name : String) : Nil
      raise ArgumentError.new("#{name} contains #{value.count} elements, expected #{expected}") unless value.count == expected
    end

    private def checked_count(a : Int32, b : Int32) : Int32
      count = a.to_i64 * b.to_i64
      raise ArgumentError.new("resident codec allocation exceeds Int32 capacity") unless a > 0 && b > 0 && count <= Int32::MAX
      count.to_i32
    end

    private def trace_activation(name : String, activation : Activation) : Nil
      @device_trace.try(&.call(name, activation.data, activation.rows, activation.channels))
      if callback = @trace
        callback.call(name, @context.read(activation.data, "diagnostic"))
      end
    end
  end
end
{% end %}
