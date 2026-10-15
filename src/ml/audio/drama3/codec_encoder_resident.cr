require "./codec"
require "./codec_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  # Architecture controls for the causal Fish DAC analysis encoder and its
  # reference-audio quantizer. Defaults are pinned to fish_s1_dac.py.
  class CodecEncoderConfig
    getter encoder_dim : Int32
    getter encoder_rates : Array(Int32)
    getter encoder_transformer_layers : Array(Int32)
    getter quantizer_downsample_rates : Array(Int32)
    getter pre_layers : Int32
    getter encoder_window : Int32

    def initialize(
      @encoder_dim : Int32 = 64,
      @encoder_rates : Array(Int32) = [2, 4, 8, 8],
      @encoder_transformer_layers : Array(Int32) = [0, 0, 0, 4],
      @quantizer_downsample_rates : Array(Int32) = [2, 2],
      @pre_layers : Int32 = 8,
      @encoder_window : Int32 = 512,
    )
      raise ArgumentError.new("encoder_dim must be positive") unless @encoder_dim > 0
      raise ArgumentError.new("encoder rates and transformer layers must be non-empty and aligned") unless !@encoder_rates.empty? && @encoder_rates.size == @encoder_transformer_layers.size
      raise ArgumentError.new("encoder rates must be positive") if @encoder_rates.any? { |rate| rate <= 0 }
      raise ArgumentError.new("encoder transformer layers must be non-negative") if @encoder_transformer_layers.any? { |layers| layers < 0 }
      raise ArgumentError.new("quantizer downsample rates must be positive") if @quantizer_downsample_rates.empty? || @quantizer_downsample_rates.any? { |rate| rate <= 0 }
      raise ArgumentError.new("pre_layers must be non-negative") if @pre_layers < 0
      raise ArgumentError.new("transformer windows must be positive") unless @encoder_window > 0
    end

    def latent_channels : Int32
      checked_product(@encoder_dim, 1 << @encoder_rates.size)
    end

    def frame_length : Int32
      rates = @encoder_rates + @quantizer_downsample_rates
      rates.reduce(1) { |stride, rate| checked_product(stride, rate) }
    end

    def self.default : CodecEncoderConfig
      new
    end

    private def checked_product(left : Int32, right : Int32) : Int32
      product = left.to_i64 * right.to_i64
      raise ArgumentError.new("codec encoder dimensions exceed Int32 capacity") unless product > 0 && product <= Int32::MAX
      product.to_i32
    end
  end

  struct CodecEncoderResult
    getter codes : Resident::Buffer
    getter frames : Int32

    def initialize(@codes : Resident::Buffer, @frames : Int32)
    end
  end

  # GPU-resident waveform encoder for reference voice conditioning. Input is
  # mono 44.1 kHz, normalized [-1, 1) F32 audio already resident on the device;
  # its output is codebook-major [10, frames] F32 exact integer IDs.
  class CodecEncoderResident
    FRAME_LENGTH = 2048
    SOURCE = {{ read_file("#{__DIR__}/kernels/codec_resident.metal") + read_file("#{__DIR__}/kernels/codec_encoder_resident.metal") }}

    private struct Activation
      getter rows : Int32
      getter channels : Int32
      getter data : Resident::Buffer

      def initialize(@rows : Int32, @channels : Int32, @data : Resident::Buffer)
      end
    end

    private struct VQResult
      getter codes : Resident::Buffer
      getter projected : Activation

      def initialize(@codes : Resident::Buffer, @projected : Activation)
      end
    end

    getter codec_config : CodecConfig
    getter encoder_config : CodecEncoderConfig
    property device_trace : Proc(String, Resident::Buffer, Int32, Int32, Nil)?

    @store : WeightStore
    @context : Resident::Context
    @prefix : String
    @split_k_threshold : Int32

    def initialize(@store : WeightStore, @context : Resident::Context,
                   @codec_config : CodecConfig = CodecConfig.default,
                   @encoder_config : CodecEncoderConfig = CodecEncoderConfig.default,
                   @prefix : String = "codec.")
      @device_trace = nil
      # MLX dispatches larger split-K grids on Max/Ultra devices. This is
      # device metadata only; no activation or result is read from the GPU.
      device_name = ML::Metal::Device.instance.name
      @split_k_threshold = device_name.includes?("Max") || device_name.includes?("Ultra") ? 2048 : 1024
      raise ArgumentError.new("encoder and codec latent channel counts differ") unless @encoder_config.latent_channels == @codec_config.latent_channels
      raise ArgumentError.new("reference encoder frame length must be 2048 samples") unless @encoder_config.frame_length == FRAME_LENGTH
      if @encoder_config.encoder_transformer_layers.any? { |layers| layers > 0 }
        final_dim = @encoder_config.latent_channels
        raise ArgumentError.new("encoder transformer requires 64-wide heads") unless final_dim % 64 == 0
      end
      raise ArgumentError.new("pre-module transformer channel mismatch") unless @codec_config.latent_channels == @codec_config.transformer_heads * @codec_config.transformer_head_dim
    end

    # Enqueues the entire reference pipeline without a finish or readback.
    # Bounded `context.submit` calls may wait for scratch-buffer reclamation;
    # the encoder-to-prompt handoff adds no drain or activation readback.
    def encode(audio : Resident::Buffer, sample_count : Int32) : CodecEncoderResult
      validate_input!(audio, sample_count)
      begin
        Profile.measure("codec_encoder_resident.encode", category: "codec") do
          encode_impl(audio, sample_count)
        end
      rescue ex
        begin
          @context.abort
        rescue
          # Keep the original inference failure if error-path draining fails.
        end
        raise ex
      end
    end

    private def encode_impl(audio : Resident::Buffer, sample_count : Int32) : CodecEncoderResult
      frames = (sample_count.to_i64 + FRAME_LENGTH - 1) // FRAME_LENGTH
      raise ArgumentError.new("reference frame count exceeds Int32 capacity") if frames > Int32::MAX
      padded_samples = checked_count(frames.to_i32, FRAME_LENGTH)
      normalized = @context.alloc(padded_samples)
      @context.dispatch("dr3e_pad_audio", SOURCE, [audio, normalized],
        [sample_count.to_u64, padded_samples.to_u64], padded_samples)
      trace_activation("audio.padded", Activation.new(padded_samples, 1, normalized))

      encoded = encode_dac(Activation.new(padded_samples, 1, normalized))
      raise ArgumentError.new("reference encoder frame count mismatch") unless encoded.rows == frames
      raise ArgumentError.new("reference quantizer output width mismatch") unless encoded.channels == @codec_config.latent_channels
      @context.guard(encoded.data)
      @context.submit

      vq = encode_quantizer(encoded)
      raise ArgumentError.new("reference code count mismatch") unless vq.size == @codec_config.residual_codebooks + 1
      code_major = @context.alloc(checked_count(vq.size.to_i32, frames.to_i32))
      vq.each_with_index do |codes, book|
        @context.dispatch("dr3e_store_codebook", SOURCE, [codes, code_major],
          [frames.to_u64, book.to_u64, vq.size.to_u64], frames.to_i32)
      end
      time_major = @context.alloc(code_major.count)
      @context.dispatch("dr3e_transpose_codes", SOURCE, [code_major, time_major],
        [frames.to_u64, vq.size.to_u64], code_major.count)
      trace_activation("final.codes", Activation.new(frames.to_i32, vq.size.to_i32, time_major))
      @context.guard(code_major)
      @context.submit
      CodecEncoderResult.new(code_major, frames.to_i32)
    end

    private def validate_input!(audio : Resident::Buffer, sample_count : Int32) : Nil
      raise ArgumentError.new("reference audio must be non-empty") unless sample_count > 0
      raise ArgumentError.new("reference audio sample count exceeds 10 seconds") if sample_count > 441_000
      raise ArgumentError.new("reference audio buffer sample count mismatch") unless audio.count == sample_count
    end

    private def encode_dac(input : Activation) : Activation
      hidden = conv1d(input, "encoder.block.0", 7, 1, 1, 1, true)
      trace_activation("encoder.block.0", hidden)
      @context.submit

      @encoder_config.encoder_rates.each_with_index do |stride, stage|
        block = stage + 1
        output_channels = @encoder_config.encoder_dim * (1 << block)
        residual_channels = output_channels // 2
        prefix = "encoder.block.#{block}.block"
        raise ArgumentError.new("encoder block input channel mismatch") unless hidden.channels == residual_channels

        [1, 3, 9].each_with_index do |dilation, unit_index|
          unit_prefix = "#{prefix}.#{unit_index}.block"
          residual = snake(hidden, "#{unit_prefix}.0.alpha")
          residual = conv1d(residual, "#{unit_prefix}.1", 7, 1, 1, dilation, true)
          residual = snake(residual, "#{unit_prefix}.2.alpha")
          residual = conv1d(residual, "#{unit_prefix}.3", 1, 1, 1, 1, true)
          hidden = add(hidden, residual)
          trace_activation("encoder.block.#{block}.block.#{unit_index}", hidden)
          @context.guard(hidden.data)
          @context.submit
        end

        hidden = snake(hidden, "#{prefix}.3.alpha")
        trace_activation("encoder.block.#{block}.block.3", hidden)
        hidden = conv1d(hidden, "#{prefix}.4", stride * 2, stride, 1, 1, true)
        trace_activation("encoder.block.#{block}.block.4", hidden)
        layers = @encoder_config.encoder_transformer_layers[stage]
        if layers > 0
          transformer_prefix = "#{prefix}.5"
          heads = output_channels // 64
          layers.times do |layer|
            layer_prefix = "#{transformer_prefix}.layers.#{layer}"
            rope = tensor("#{transformer_prefix}.freqs_cis")
            hidden = transformer_layer(hidden, layer_prefix, heads, 64, output_channels * 3,
              @encoder_config.encoder_window, 1.0e-5_f32, rope)
            @context.guard(hidden.data)
            @context.submit
          end

          # Fish's Transformer applies its final RMSNorm after the layer stack.
          # WindowLimitedTransformer then applies output_proj; the encoder
          # config sets input_dim == dim, so that projection is Identity.
          final_norm_weight = tensor("#{transformer_prefix}.norm.weight")
          validate_shape!(final_norm_weight, [output_channels], "#{transformer_prefix}.norm.weight")
          normalized = encoder_rms(hidden.data, @context.tensor(final_norm_weight),
            1.0e-5_f32, hidden.rows)
          hidden = Activation.new(hidden.rows, hidden.channels, normalized)
        end
        trace_activation("#{prefix}.5", hidden)
        trace_activation("encoder.block.#{block}", hidden)
        @context.guard(hidden.data)
        @context.submit
      end

      hidden = snake(hidden, "encoder.block.5.alpha")
      trace_activation("encoder.block.5", hidden)
      hidden = conv1d(hidden, "encoder.block.6", 3, 1, 1, 1, true)
      trace_activation("encoder.block.6", hidden)
      @context.guard(hidden.data)
      @context.submit

      trace_activation("encoder.output", hidden)

      @encoder_config.quantizer_downsample_rates.each_with_index do |rate, stage|
        prefix = "quantizer.downsample.#{stage}"
        hidden = conv1d(hidden, "#{prefix}.0.conv", rate, rate, 1, 1, false)
        trace_activation("#{prefix}.conv", hidden)
        @context.submit
        hidden = convnext(hidden, "#{prefix}.1")
        trace_activation("#{prefix}.convnext", hidden)
        @context.guard(hidden.data)
        @context.submit
      end
      hidden
    end

    private def encode_quantizer(input : Activation) : Array(Resident::Buffer)
      hidden = input
      trace_activation("quantizer.pre_module.input_proj", hidden)
      trace_activation("quantizer.pre_module.look_ahead_conv", hidden)
      @encoder_config.pre_layers.times do |layer|
        prefix = "quantizer.pre_module.layers.#{layer}"
        rope = tensor("quantizer.pre_module.freqs_cis")
        hidden = transformer_layer(hidden, prefix, @codec_config.transformer_heads,
          @codec_config.transformer_head_dim, @codec_config.transformer_ff_dim,
          @codec_config.transformer_window, @codec_config.norm_eps, rope)
        trace_activation("quantizer.pre_module.layer.#{layer}", hidden)
        @context.guard(hidden.data)
        @context.submit
      end
      norm_weight = tensor("quantizer.pre_module.norm.weight")
      validate_shape!(norm_weight, [input.channels], "quantizer.pre_module.norm.weight")
      normalized = encoder_rms(hidden.data, @context.tensor(norm_weight), @codec_config.norm_eps, hidden.rows)
      hidden = Activation.new(hidden.rows, hidden.channels, normalized)
      trace_activation("quantizer.pre_module.norm", hidden)
      trace_activation("quantizer.pre_module.output_proj", hidden)
      trace_activation("quantizer.pre_module", hidden)
      trace_activation("quantizer.semantic.input", hidden)

      codes = [] of Resident::Buffer
      semantic_prefix = "quantizer.semantic_quantizer.quantizers.0"
      semantic = vector_quantize(hidden, semantic_prefix, @codec_config.semantic_codebook_size, "quantizer.semantic.0")
      codes << semantic.codes
      residual = subtract(hidden, semantic.projected)

      @codec_config.residual_codebooks.times do |index|
        prefix = "quantizer.quantizer.quantizers.#{index}"
        label = "quantizer.residual.#{index}"
        result = vector_quantize(residual, prefix, @codec_config.residual_codebook_size, label)
        codes << result.codes
        residual = subtract(residual, result.projected)
        @context.guard(residual.data)
        @context.submit
      end
      codes
    end

    private def vector_quantize(input : Activation, prefix : String, vocab : Int32,
                                label : String) : VQResult
      code_dim = @codec_config.codebook_dim
      projected = conv1d(input, "#{prefix}.in_proj", 1, 1, 1, 1, true)
      trace_activation("#{label}.in_proj", projected)
      normalized_input = normalize_rows(projected)
      trace_activation("#{label}.normalized_input", Activation.new(projected.rows, code_dim, normalized_input))

      codebook = tensor("#{prefix}.codebook.weight")
      validate_shape!(codebook, [vocab, code_dim], "#{prefix}.codebook.weight")
      normalized_codebook = normalize_rows(Activation.new(vocab, code_dim, @context.tensor(codebook)))
      distances = @context.alloc(checked_count(projected.rows, vocab))
      @context.dispatch("dr3e_vq_distance", SOURCE,
        [normalized_input, normalized_codebook, distances],
        [projected.rows.to_u64, vocab.to_u64, code_dim.to_u64], distances.count)
      @context.guard(distances)
      trace_activation("#{label}.distance", Activation.new(projected.rows, vocab, distances))

      selected_ids = @context.alloc(projected.rows)
      top2_margin = @context.alloc(projected.rows)
      @context.dispatch("dr3e_vq_argmin", SOURCE, [distances, selected_ids, top2_margin],
        [projected.rows.to_u64, vocab.to_u64], projected.rows)
      trace_activation("#{label}.top2_margin", Activation.new(projected.rows, 1, top2_margin))
      trace_activation("#{label}.selected_ids", Activation.new(projected.rows, 1, selected_ids))

      embedded = @context.alloc(checked_count(projected.rows, code_dim))
      @context.dispatch("dr3e_vq_gather", SOURCE, [selected_ids, @context.tensor(codebook), embedded],
        [projected.rows.to_u64, vocab.to_u64, code_dim.to_u64], embedded.count)
      quantized = @context.alloc(embedded.count)
      @context.dispatch("dr3e_vq_identity", SOURCE, [projected.data, embedded, quantized],
        [embedded.count.to_u64], embedded.count)
      output = conv1d(Activation.new(projected.rows, code_dim, quantized), "#{prefix}.out_proj", 1, 1, 1, 1, true)
      VQResult.new(selected_ids, output)
    end

    private def transformer_layer(input : Activation, prefix : String, heads : Int32,
                                  head_dim : Int32, ff : Int32, window : Int32,
                                  eps : Float32, rope : Tensor) : Activation
      time = input.rows
      channels = input.channels
      raise ArgumentError.new("transformer channel/head mismatch") unless channels == heads * head_dim
      validate_rope!(rope, time, head_dim, "#{prefix}.freqs_cis")
      attention_norm_weight = tensor("#{prefix}.attention_norm.weight")
      validate_shape!(attention_norm_weight, [channels], "#{prefix}.attention_norm.weight")
      normalized = encoder_rms(input.data, @context.tensor(attention_norm_weight), eps, time)

      qkv_weight = tensor("#{prefix}.attention.wqkv.weight")
      validate_shape!(qkv_weight, [channels * 3, channels], "#{prefix}.attention.wqkv.weight")
      qkv = linear(normalized, time, channels, "#{prefix}.attention.wqkv.weight")
      query = @context.alloc(checked_count(time, channels))
      key = @context.alloc(query.count)
      value = @context.alloc(query.count)
      @context.dispatch("dr3c_qkv_split", SOURCE, [qkv, query, key, value],
        [time.to_u64, channels.to_u64], query.count)
      rope_buffer = @context.tensor(rope)
      rope_params = [time.to_u64, heads.to_u64, head_dim.to_u64]
      @context.dispatch("dr3e_rope_loaded", SOURCE, [query, rope_buffer], rope_params,
        time * heads * (head_dim // 2))
      @context.dispatch("dr3e_rope_loaded", SOURCE, [key, rope_buffer], rope_params,
        time * heads * (head_dim // 2))

      scale = (1.0_f32 / Math.sqrt(head_dim.to_f32)).to_f32
      attended = @context.alloc(checked_count(time, channels))
      if head_dim == 64
        # Preserve the pinned MLX F32 Steel SDPA block and reduction order.
        tiles = checked_count((time + 7) // 8, heads)
        @context.dispatch("dr3e_attention_32", SOURCE, [query, key, value, attended],
          [time.to_u64, heads.to_u64, window.to_u64, scale.unsafe_as(UInt32).to_u64], checked_count(tiles, 32), 32)
      else
        # Keep the existing generic configuration contract for other head sizes.
        score_count = checked_count(time, checked_count(heads, window))
        scores = @context.alloc(score_count)
        @context.dispatch("dr3c_attention_scores", SOURCE, [query, key, scores],
          [time.to_u64, heads.to_u64, head_dim.to_u64, window.to_u64, scale.unsafe_as(UInt32).to_u64], score_count)
        probabilities = @context.alloc(score_count)
        @context.dispatch("dr3c_attention_probs", SOURCE, [scores, probabilities],
          [time.to_u64, heads.to_u64, window.to_u64], checked_count(time, heads))
        @context.dispatch("dr3c_attention_values", SOURCE, [probabilities, value, attended],
          [time.to_u64, heads.to_u64, head_dim.to_u64, window.to_u64], attended.count)
      end

      output_weight = tensor("#{prefix}.attention.wo.weight")
      validate_shape!(output_weight, [channels, channels], "#{prefix}.attention.wo.weight")
      attention_projection = linear(attended, time, channels, "#{prefix}.attention.wo.weight")
      attention_scale = tensor("#{prefix}.attention_layer_scale.gamma")
      validate_shape!(attention_scale, [channels], "#{prefix}.attention_layer_scale.gamma")
      attention_projection = channel_scale(attention_projection, time, channels, @context.tensor(attention_scale))
      after_attention = Activation.new(time, channels, @context.add(input.data, attention_projection))

      ffn_norm_weight = tensor("#{prefix}.ffn_norm.weight")
      validate_shape!(ffn_norm_weight, [channels], "#{prefix}.ffn_norm.weight")
      normalized_ffn = encoder_rms(after_attention.data, @context.tensor(ffn_norm_weight), eps, time)
      w1_weight = tensor("#{prefix}.feed_forward.w1.weight")
      w3_weight = tensor("#{prefix}.feed_forward.w3.weight")
      validate_shape!(w1_weight, [ff, channels], "#{prefix}.feed_forward.w1.weight")
      validate_shape!(w3_weight, [ff, channels], "#{prefix}.feed_forward.w3.weight")
      w1 = linear(normalized_ffn, time, channels, "#{prefix}.feed_forward.w1.weight")
      w3 = linear(normalized_ffn, time, channels, "#{prefix}.feed_forward.w3.weight")
      gated = @context.alloc(checked_count(time, ff))
      @context.dispatch("dr3e_sigmoid_gate", SOURCE, [w1, w3, gated], [gated.count.to_u64], gated.count)
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
      @context.dispatch("dr3e_layer_norm", SOURCE,
        [x.data, @context.tensor(gamma), @context.tensor(beta), normalized],
        [x.rows.to_u64, channels.to_u64, 1.0e-6_f32.unsafe_as(UInt32).to_u64], x.rows)
      hidden_channels = checked_count(channels, 4)
      first_weight = tensor("#{prefix}.pwconv1.weight")
      validate_shape!(first_weight, [hidden_channels, channels], "#{prefix}.pwconv1.weight")
      projected = linear(normalized, x.rows, channels, "#{prefix}.pwconv1.weight", "#{prefix}.pwconv1.bias")
      activated = @context.alloc(projected.count)
      @context.dispatch("dr3e_gelu", SOURCE, [projected, activated], [projected.count.to_u64], projected.count)
      second_weight = tensor("#{prefix}.pwconv2.weight")
      validate_shape!(second_weight, [channels, hidden_channels], "#{prefix}.pwconv2.weight")
      residual = linear(activated, x.rows, hidden_channels, "#{prefix}.pwconv2.weight", "#{prefix}.pwconv2.bias")
      gamma_scale = tensor("#{prefix}.gamma")
      validate_shape!(gamma_scale, [channels], "#{prefix}.gamma")
      residual = channel_scale(residual, x.rows, channels, @context.tensor(gamma_scale))
      Activation.new(input.rows, channels, @context.add(input.data, residual))
    end

    private def conv1d(input : Activation, prefix : String, kernel : Int32, stride : Int32,
                       groups : Int32, dilation : Int32, weight_normalized : Bool) : Activation
      raise ArgumentError.new("invalid encoder Conv1d parameters") unless kernel > 0 && stride > 0 && dilation > 0 && groups > 0 && input.channels % groups == 0
      weight_name = weight_normalized ? "#{prefix}.weight_v" : "#{prefix}.weight"
      weight = tensor(weight_name)
      raise ArgumentError.new("invalid Conv1d weight rank for #{weight_name}") unless weight.shape.size == 3
      output_channels = weight.shape[0]
      in_per_group = input.channels // groups
      validate_shape!(weight, [output_channels, in_per_group, kernel], weight_name)
      raise ArgumentError.new("Conv1d output channels not divisible by groups") unless output_channels % groups == 0
      bias = tensor("#{prefix}.bias")
      validate_shape!(bias, [output_channels], "#{prefix}.bias")
      resident_weight = @context.tensor(weight)
      if weight_normalized
        gain = tensor("#{prefix}.weight_g")
        validate_element_count!(gain, output_channels, "#{prefix}.weight_g")
        per_channel = checked_count(in_per_group, kernel)
        norms = @context.alloc(output_channels)
        norm_threads = if per_channel <= 64
                         1
                       elsif per_channel <= 512
                         32
                       elsif per_channel <= 1024
                         128
                       else
                         Math.min(1024, (((per_channel + 3) // 4 + 31) // 32) * 32)
                       end
        @context.dispatch("dr3e_wn_norms", SOURCE, [resident_weight, norms],
          [output_channels.to_u64, per_channel.to_u64, 0x40000000_u64],
          checked_count(output_channels, norm_threads), norm_threads)
        effective = @context.alloc(resident_weight.count)
        @context.dispatch("dr3e_wn_apply", SOURCE,
          [resident_weight, @context.tensor(gain), norms, effective],
          [resident_weight.count.to_u64, per_channel.to_u64], resident_weight.count)
        resident_weight = effective
      end
      rows = ((input.rows.to_i64 + stride - 1) // stride).to_i32
      output = @context.alloc(checked_count(rows, output_channels))
      if groups == 1 && (input.channels % 16 == 0 || input.channels == 1 || input.channels == 8)
        # The quantizer downsampling convolutions use MLX Steel's BK16
        # channel-block/filter traversal and F32 simdgroup accumulators.
        tiles = checked_count((rows + 7) // 8, (output_channels + 7) // 8)
        @context.dispatch("dr3e_conv1d_mma_bk16", SOURCE,
          [input.data, resident_weight, @context.tensor(bias), output],
          [input.rows.to_u64, rows.to_u64, input.channels.to_u64, output_channels.to_u64,
           kernel.to_u64, stride.to_u64, dilation.to_u64], checked_count(tiles, 32), 32)
      else
        @context.dispatch("dr3e_conv1d", SOURCE,
          [input.data, resident_weight, @context.tensor(bias), output],
          [input.rows.to_u64, rows.to_u64, input.channels.to_u64, output_channels.to_u64,
           kernel.to_u64, stride.to_u64, groups.to_u64, dilation.to_u64], output.count)
      end
      Activation.new(rows, output_channels, output)
    end

    private def snake(input : Activation, alpha_name : String) : Activation
      alpha = tensor(alpha_name)
      validate_element_count!(alpha, input.channels, alpha_name)
      output = @context.alloc(input.data.count)
      @context.dispatch("dr3e_snake", SOURCE, [input.data, @context.tensor(alpha), output],
        [input.rows.to_u64, input.channels.to_u64, 0x40000000_u64], output.count)
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
      bias_buffer = input
      if name = bias_name
        bias = tensor(name)
        validate_shape!(bias, [out_channels], name)
        bias_buffer = @context.tensor(bias)
      end
      # Match upstream split-K arithmetic inside one dispatch. Empty leading
      # partitions (when the stride is zero) are intentional upstream behavior.
      partitions = 1
      if ((rows.to_i64 + 15) // 16) * ((out_channels.to_i64 + 15) // 16) <= @split_k_threshold && in_channels // 16 >= 8 && in_channels >= Math.max(rows, out_channels)
        ratio = (in_channels // 16) // (((rows + 31) // 32) * ((out_channels + 31) // 32))
        partitions = 2
        while partitions < ratio && partitions < 32
          partitions *= 2
        end
      end
      partition_size = partitions == 1 ? in_channels : ((in_channels // 16) // partitions) * 16
      tiles = checked_count((rows + 7) // 8, (out_channels + 7) // 8)
      @context.dispatch("dr3e_linear_mma", SOURCE,
        [input, resident_weight, bias_buffer, output],
        [rows.to_u64, in_channels.to_u64, out_channels.to_u64, bias_name ? 1_u64 : 0_u64, partitions.to_u64, partition_size.to_u64],
        checked_count(tiles, 32), 32)
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

    private def normalize_rows(input : Activation) : Resident::Buffer
      output = @context.alloc(input.data.count)
      @context.dispatch("dr3e_l2_normalize", SOURCE, [input.data, output],
        [input.rows.to_u64, input.channels.to_u64], output.count)
      output
    end

    private def encoder_rms(input : Resident::Buffer, weight : Resident::Buffer,
                            eps : Float64, rows : Int32) : Resident::Buffer
      raise ArgumentError.new("encoder RMS shape mismatch") unless rows > 0 && input.count % rows == 0 && weight.count == input.count // rows
      raise ArgumentError.new("encoder RMS epsilon must be positive and finite") unless eps.finite? && eps > 0
      output = @context.alloc(input.count)
      threads = weight.count <= 64 ? 1 : weight.count <= 512 ? 32 : weight.count <= 1024 ? 128 : Math.min(1024, (((weight.count + 3) // 4 + 31) // 32) * 32)
      @context.dispatch("dr3e_rms", SOURCE, [input, weight, output],
        [input.count.to_u64, weight.count.to_u64, eps.to_f32.unsafe_as(UInt32).to_u64], checked_count(rows, threads), threads)
      output
    end

    private def add(left : Activation, right : Activation) : Activation
      raise ArgumentError.new("encoder residual shape mismatch") unless left.rows == right.rows && left.channels == right.channels
      Activation.new(left.rows, left.channels, @context.add(left.data, right.data))
    end

    private def subtract(left : Activation, right : Activation) : Activation
      raise ArgumentError.new("encoder residual shape mismatch") unless left.rows == right.rows && left.channels == right.channels
      output = @context.alloc(left.data.count)
      @context.dispatch("dr3e_subtract", SOURCE, [left.data, right.data, output],
        [left.data.count.to_u64], output.count)
      Activation.new(left.rows, left.channels, output)
    end

    private def validate_rope!(rope : Tensor, rows : Int32, head_dim : Int32, name : String) : Nil
      shape = rope.shape
      unless rope.dtype == "F32" && shape.size == 3 && shape[0] >= rows && shape[1] == head_dim // 2 && shape[2] == 2
        raise ArgumentError.new("#{name} must be loaded F32 [positions, #{head_dim // 2}, 2] RoPE coefficients")
      end
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
        value = begin
          @store.tensor(candidate)
        rescue ex : ArgumentError
          # Try the remaining checkpoint spellings used by PyTorch/MLX.
          next
        end
        raise ArgumentError.new("#{candidate} must be F32") unless value.dtype == "F32"
        return value
      end
      raise ArgumentError.new("missing codec tensor #{@prefix}#{name}; tried #{candidates.join(", ")}")
    end

    private def validate_shape!(value : Tensor, expected : Array(Int32), name : String) : Nil
      if expected.size == 1
        return validate_element_count!(value, expected[0], name)
      end
      raise ArgumentError.new("#{name} shape #{value.shape} does not match #{expected}") unless value.shape == expected
    end

    private def validate_element_count!(value : Tensor, expected : Int32, name : String) : Nil
      raise ArgumentError.new("#{name} contains #{value.count} elements, expected #{expected}") unless value.count == expected
    end

    private def checked_count(a : Int32, b : Int32) : Int32
      count = a.to_i64 * b.to_i64
      raise ArgumentError.new("resident codec encoder allocation exceeds Int32 capacity") unless a > 0 && b > 0 && count <= Int32::MAX
      count.to_i32
    end

    private def trace_activation(name : String, activation : Activation) : Nil
      @device_trace.try(&.call(name, activation.data, activation.rows, activation.channels))
    end
  end
end
{% end %}
