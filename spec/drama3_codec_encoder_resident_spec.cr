require "spec"
require "../src/ml/audio/drama3/codec"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/codec_encoder_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_encoder_put(tensors : Hash(String, Tensor), name : String,
                                 shape : Array(Int32), values : Vector) : Nil
    tensors["codec.#{name}"] = Tensor.new(shape, values)
  end

  private def drama3_encoder_fill(tensors : Hash(String, Tensor), name : String,
                                  shape : Array(Int32), value : Float32) : Nil
    count = shape.reduce(1) { |total, dimension| total * dimension }
    drama3_encoder_put(tensors, name, shape, Array(Float32).new(count, value))
  end

  private def drama3_encoder_wn(tensors : Hash(String, Tensor), prefix : String,
                                out_channels : Int32, in_channels : Int32,
                                kernel : Int32) : Nil
    weight_count = out_channels * in_channels * kernel
    values = Array(Float32).new(weight_count) do |i|
      (((i * 7 + prefix.bytes.sum(&.to_i)) % 19) - 9).to_f32 * 0.002_f32
    end
    drama3_encoder_put(tensors, "#{prefix}.weight_v", [out_channels, in_channels, kernel], values)
    drama3_encoder_fill(tensors, "#{prefix}.weight_g", [out_channels, 1, 1], 0.05_f32)
    drama3_encoder_fill(tensors, "#{prefix}.bias", [out_channels], 0.0005_f32)
  end

  private def drama3_encoder_convnext(tensors : Hash(String, Tensor), prefix : String,
                                      channels : Int32) : Nil
    drama3_encoder_put(tensors, "#{prefix}.dwconv.conv.weight", [channels, 1, 7],
      Array(Float32).new(channels * 7) { |i| (((i * 5 + 3) % 13) - 6).to_f32 * 0.003_f32 })
    drama3_encoder_fill(tensors, "#{prefix}.dwconv.conv.bias", [channels], 0.0002_f32)
    drama3_encoder_fill(tensors, "#{prefix}.norm.weight", [channels], 1.0_f32)
    drama3_encoder_fill(tensors, "#{prefix}.norm.bias", [channels], 0.0001_f32)
    hidden = channels * 4
    drama3_encoder_put(tensors, "#{prefix}.pwconv1.weight", [hidden, channels],
      Array(Float32).new(hidden * channels) { |i| (((i * 3 + 1) % 17) - 8).to_f32 * 0.002_f32 })
    drama3_encoder_fill(tensors, "#{prefix}.pwconv1.bias", [hidden], 0.0001_f32)
    drama3_encoder_put(tensors, "#{prefix}.pwconv2.weight", [channels, hidden],
      Array(Float32).new(channels * hidden) { |i| (((i * 11 + 2) % 23) - 11).to_f32 * 0.001_f32 })
    drama3_encoder_fill(tensors, "#{prefix}.pwconv2.bias", [channels], 0.0001_f32)
    drama3_encoder_fill(tensors, "#{prefix}.gamma", [channels], 0.1_f32)
  end

  private def drama3_encoder_transformer(tensors : Hash(String, Tensor), prefix : String,
                                         channels : Int32, ff : Int32) : Nil
    layer = "#{prefix}.layers.0"
    drama3_encoder_fill(tensors, "#{prefix}.norm.weight", [channels], 0.375_f32)
    rope = Array(Float32).new(8 * (channels // 2) * 2, 0_f32)
    (8 * (channels // 2)).times { |i| rope[i * 2] = 1_f32 }
    drama3_encoder_put(tensors, "#{prefix}.freqs_cis", [8, channels // 2, 2], rope)
    drama3_encoder_fill(tensors, "#{layer}.attention_norm.weight", [channels], 1_f32)
    drama3_encoder_put(tensors, "#{layer}.attention.wqkv.weight", [channels * 3, channels],
      Array(Float32).new(channels * channels * 3) { |i| (((i * 7 + 2) % 19) - 9).to_f32 * 0.001_f32 })
    drama3_encoder_put(tensors, "#{layer}.attention.wo.weight", [channels, channels],
      Array(Float32).new(channels * channels) { |i| (((i * 5 + 1) % 17) - 8).to_f32 * 0.001_f32 })
    # Keep the residual layer identity while still executing QKV, RoPE, and
    # attention. This lets the outer test independently calculate the final
    # RMSNorm from the captured pre-transformer activation.
    drama3_encoder_fill(tensors, "#{layer}.attention_layer_scale.gamma", [channels], 0_f32)
    drama3_encoder_fill(tensors, "#{layer}.ffn_norm.weight", [channels], 1_f32)
    drama3_encoder_put(tensors, "#{layer}.feed_forward.w1.weight", [ff, channels],
      Array(Float32).new(ff * channels) { |i| (((i * 3 + 1) % 13) - 6).to_f32 * 0.001_f32 })
    drama3_encoder_put(tensors, "#{layer}.feed_forward.w3.weight", [ff, channels],
      Array(Float32).new(ff * channels) { |i| (((i * 11 + 2) % 17) - 8).to_f32 * 0.001_f32 })
    drama3_encoder_put(tensors, "#{layer}.feed_forward.w2.weight", [channels, ff],
      Array(Float32).new(channels * ff) { |i| (((i * 5 + 4) % 19) - 9).to_f32 * 0.001_f32 })
    drama3_encoder_fill(tensors, "#{layer}.ffn_layer_scale.gamma", [channels], 0_f32)
  end

  private def drama3_encoder_fixture(codec : CodecConfig,
                                     encoder : CodecEncoderConfig) : WeightStore
    tensors = {} of String => Tensor
    drama3_encoder_wn(tensors, "encoder.block.0", encoder.encoder_dim, 1, 7)

    channels = encoder.encoder_dim
    encoder.encoder_rates.each_with_index do |stride, stage|
      channels *= 2
      residual_channels = channels // 2
      prefix = "encoder.block.#{stage + 1}.block"
      [1, 3, 9].each_with_index do |_dilation, unit|
        unit_prefix = "#{prefix}.#{unit}.block"
        drama3_encoder_fill(tensors, "#{unit_prefix}.0.alpha", [1, residual_channels, 1], 0.7_f32)
        drama3_encoder_wn(tensors, "#{unit_prefix}.1", residual_channels, residual_channels, 7)
        drama3_encoder_fill(tensors, "#{unit_prefix}.2.alpha", [1, residual_channels, 1], 0.6_f32)
        drama3_encoder_wn(tensors, "#{unit_prefix}.3", residual_channels, residual_channels, 1)
      end
      drama3_encoder_fill(tensors, "#{prefix}.3.alpha", [1, residual_channels, 1], 0.8_f32)
      drama3_encoder_wn(tensors, "#{prefix}.4", channels, residual_channels, stride * 2)
    end

    # Exercise the final transformer normalization and its loaded-RoPE path.
    # In the pinned model this block's output projection is same-width Identity.
    drama3_encoder_transformer(tensors, "encoder.block.4.block.5", channels, channels * 3)

    final_channels = channels
    drama3_encoder_fill(tensors, "encoder.block.5.alpha", [1, final_channels, 1], 0.9_f32)
    drama3_encoder_wn(tensors, "encoder.block.6", codec.latent_channels, final_channels, 3)

    encoder.quantizer_downsample_rates.each_with_index do |rate, stage|
      prefix = "quantizer.downsample.#{stage}"
      drama3_encoder_put(tensors, "#{prefix}.0.conv.weight", [codec.latent_channels, codec.latent_channels, rate],
        Array(Float32).new(codec.latent_channels * codec.latent_channels * rate) { |i| (((i * 5 + 1) % 17) - 8).to_f32 * 0.001_f32 })
      drama3_encoder_fill(tensors, "#{prefix}.0.conv.bias", [codec.latent_channels], 0.0001_f32)
      drama3_encoder_convnext(tensors, "#{prefix}.1", codec.latent_channels)
    end
    drama3_encoder_fill(tensors, "quantizer.pre_module.norm.weight", [codec.latent_channels], 1.0_f32)

    (codec.residual_codebooks + 1).times do |book|
      prefix = if book == 0
                 "quantizer.semantic_quantizer.quantizers.0"
               else
                 "quantizer.quantizer.quantizers.#{book - 1}"
               end
      vocab = book == 0 ? codec.semantic_codebook_size : codec.residual_codebook_size
      drama3_encoder_wn(tensors, "#{prefix}.in_proj", codec.codebook_dim, codec.latent_channels, 1)
      drama3_encoder_wn(tensors, "#{prefix}.out_proj", codec.latent_channels, codec.codebook_dim, 1)
      # Equal rows produce an exact tie for every input. Upstream's (-dist)
      # argmax must retain the lowest index; the production guard never
      # broadens that rule to tolerate a near tie.
      drama3_encoder_put(tensors, "#{prefix}.codebook.weight", [vocab, codec.codebook_dim],
        Array(Float32).new(vocab * codec.codebook_dim) do |i|
          i % codec.codebook_dim == 0 ? 0.25_f32 : -0.125_f32
        end)
    end
    WeightStore.new(tensors)
  end

  describe "DRAMA-3 resident codec encoder" do
    it "pads normalized input, preserves causal stride output, and takes the first exact VQ tie" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        codec = CodecConfig.new(
          latent_channels: 64,
          codebook_dim: 2,
          semantic_codebook_size: 3,
          residual_codebooks: 9,
          residual_codebook_size: 3,
          post_layers: 0,
          transformer_heads: 1,
          transformer_head_dim: 64,
          transformer_ff_dim: 16,
          transformer_window: 2,
          quantizer_upsample_rates: [2, 2],
          decoder_channels: 32,
          decoder_rates: [1],
        )
        encoder = CodecEncoderConfig.new(
          encoder_dim: 4,
          encoder_rates: [2, 4, 8, 8],
          encoder_transformer_layers: [0, 0, 0, 1],
          quantizer_downsample_rates: [2, 2],
          pre_layers: 0,
        )
        store = drama3_encoder_fixture(codec, encoder)
        context = Resident::Context.new
        samples = [0.5_f32, -0.25_f32, 0.03125_f32] + Array(Float32).new(2_046, 0_f32)
        audio = context.upload(samples)
        resident = CodecEncoderResident.new(store, context, codec, encoder)
        captures = {} of String => Tuple(Resident::Buffer, Int32, Int32)
        resident.device_trace = ->(name : String, buffer : Resident::Buffer, rows : Int32, channels : Int32) {
          context.readbacks.empty?.should be_true
          buffer.count.should eq(rows * channels)
          captures[name] = {buffer, rows, channels}
          nil
        }

        encoded = resident.encode(audio, samples.size.to_i32)
        encoded.frames.should eq(2)
        encoded.codes.count.should eq(20)
        context.readbacks.empty?.should be_true

        expected_stages = ["audio.padded"]
        (encoder.encoder_rates.size + 3).times do |block|
          if block.in?(1..encoder.encoder_rates.size)
            6.times { |child| expected_stages << "encoder.block.#{block}.block.#{child}" }
          end
          expected_stages << "encoder.block.#{block}"
        end
        expected_stages << "encoder.output"
        encoder.quantizer_downsample_rates.size.times do |stage|
          expected_stages << "quantizer.downsample.#{stage}.conv"
          expected_stages << "quantizer.downsample.#{stage}.convnext"
        end
        expected_stages.concat([
          "quantizer.pre_module", "quantizer.semantic.input",
          "quantizer.pre_module.input_proj", "quantizer.pre_module.look_ahead_conv",
          "quantizer.pre_module.norm", "quantizer.pre_module.output_proj",
        ])
        10.times do |book|
          label = book == 0 ? "quantizer.semantic.0" : "quantizer.residual.#{book - 1}"
          expected_stages.concat(["#{label}.in_proj", "#{label}.normalized_input", "#{label}.distance",
                                  "#{label}.top2_margin", "#{label}.selected_ids"])
        end
        expected_stages << "final.codes"
        captures.keys.sort.should eq(expected_stages.sort)

        padded = context.read(captures["audio.padded"][0], "diagnostic")
        padded[0, 3].should eq([0.5_f32, -0.25_f32, 0.03125_f32])
        padded.size.should eq(4_096)
        padded[3, padded.size - 3].all?(&.== 0_f32).should be_true
        captures["encoder.block.1.block.4"][1..2].should eq({padded.size // encoder.encoder_rates[0], encoder.encoder_dim * 2})
        encoder_rows = padded.size // encoder.encoder_rates.product
        encoder_channels = encoder.encoder_dim * (1 << encoder.encoder_rates.size)
        final_transformer = "encoder.block.#{encoder.encoder_rates.size}.block.5"
        transformed = context.read(captures[final_transformer][0], "diagnostic")
        captures[final_transformer][1..2].should eq({encoder_rows, encoder_channels})
        captures["encoder.output"][1..2].should eq({encoder_rows, encoder_channels})
        transformed_channels = captures[final_transformer][2]
        pre_norm = context.read(captures["encoder.block.#{encoder.encoder_rates.size}.block.4"][0], "diagnostic")
        pre_norm.size.should eq(transformed.size)
        transformed.each_slice(transformed_channels).with_index do |row, row_index|
          input_base = row_index * transformed_channels
          mean_square = 0_f64
          transformed_channels.times do |channel|
            value = pre_norm[input_base + channel].to_f64
            mean_square += value * value
          end
          mean_square /= transformed_channels
          inverse = 1.0 / Math.sqrt(mean_square + 1.0e-5_f64)
          expected_rms = 0.375_f64 * Math.sqrt(mean_square / (mean_square + 1.0e-5_f64))
          actual_rms = Math.sqrt(row.sum { |value| value.to_f64 * value.to_f64 } / transformed_channels)
          (actual_rms - expected_rms).abs.should be < 2.0e-6_f64
          transformed_channels.times do |channel|
            expected = (pre_norm[input_base + channel].to_f64 * inverse * 0.375_f64).to_f32
            (row[channel] - expected).abs.should be < 2.0e-6_f32
          end
        end
        encoder.quantizer_downsample_rates.each_with_index do |rate, stage|
          encoder_rows //= rate
          captures["quantizer.downsample.#{stage}.conv"][1..2].should eq({encoder_rows, codec.latent_channels})
          captures["quantizer.downsample.#{stage}.convnext"][1..2].should eq({encoder_rows, codec.latent_channels})
        end
        captures["quantizer.pre_module"][1..2].should eq({encoded.frames, codec.latent_channels})
        captures["quantizer.semantic.input"][1..2].should eq({encoded.frames, codec.latent_channels})
        10.times do |book|
          label = book == 0 ? "quantizer.semantic.0" : "quantizer.residual.#{book - 1}"
          vocab = book == 0 ? codec.semantic_codebook_size : codec.residual_codebook_size
          {"in_proj"          => {encoded.frames, codec.codebook_dim},
           "normalized_input" => {encoded.frames, codec.codebook_dim},
           "distance"         => {encoded.frames, vocab},
           "top2_margin"      => {encoded.frames, 1},
           "selected_ids"     => {encoded.frames, 1}}.each do |stage, expected_shape|
            captures["#{label}.#{stage}"][1..2].should eq(expected_shape)
          end
        end

        # The same source-order CPU calculation checks Fish's WN materialize
        # order and post-convolution bias at the stride-one causal stem.
        stem = context.read(captures["encoder.block.0"][0], "diagnostic")
        stem_channels = encoder.encoder_dim
        weight = store.tensor("codec.encoder.block.0.weight_v").values
        gain = store.tensor("codec.encoder.block.0.weight_g").values
        bias = store.tensor("codec.encoder.block.0.bias").values
        input = padded
        expected = Array(Float32).new(stem.size, 0_f32)
        (stem.size // stem_channels).times do |row|
          stem_channels.times do |out_channel|
            norm = 0_f32
            7.times do |tap|
              value = weight[out_channel * 7 + tap]
              norm += value * value
            end
            denominator = Math.sqrt(norm.to_f64).to_f32
            sum = 0_f32
            7.times do |tap|
              input_row = row - 6 + tap
              next if input_row < 0 || input_row >= input.size
              effective_weight = (gain[out_channel] * weight[out_channel * 7 + tap]) / denominator
              sum += input[input_row] * effective_weight
            end
            expected[row * stem_channels + out_channel] = sum + bias[out_channel]
          end
        end
        stem.zip(expected).each { |actual, reference| (actual - reference).abs.should be < 2.0e-6_f32 }

        # codes are codebook-major [10, frames]; every book must preserve the
        # first row under an exact distance tie.
        context.read(encoded.codes, "terminal").should eq(Array(Float32).new(20, 0_f32))
        context.readbacks["terminal"].should eq(80_i64)
        captures["quantizer.semantic.0.top2_margin"][1..2].should eq({2, 1})
        captures["final.codes"][1..2].should eq({2, 10})
      ensure
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "preserves scalar fallback reductions and encoder nonlinearities" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        context = Resident::Context.new
        input = context.upload([100_000_000_f32, 1_f32, -100_000_000_f32, 1_f32])
        weights = context.upload([1_f32, 1_f32, 1_f32, 1_f32])
        bias = context.upload([0.25_f32])

        convolution = context.alloc(1)
        context.dispatch("dr3e_conv1d", CodecEncoderResident::SOURCE,
          [input, weights, bias, convolution], [1_u64, 1_u64, 4_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64], 1)

        # A serial F32 sum drops both unit terms here. Encoder LayerNorm must
        # preserve them while remaining F32, and must use the MLX rsqrt path.
        norm_input = context.upload([100_000_000_f32, 1_f32, -100_000_000_f32, 1_f32])
        norm_gamma = context.upload([100_000_000_f32] * 4)
        norm_beta = context.upload([0_f32] * 4)
        layer_norm = context.alloc(4)
        context.dispatch("dr3e_layer_norm", CodecEncoderResident::SOURCE,
          [norm_input, norm_gamma, norm_beta, layer_norm], [1_u64, 4_u64, 1.0e-6_f32.unsafe_as(UInt32).to_u64], 1)

        snake_input_values = [0.125_f32, -0.75_f32, 0.5_f32, 1.25_f32]
        snake_alpha_values = [1.0_f32, 0.7_f32, 0.5_f32, 2.0_f32]
        snake_input = context.upload(snake_input_values)
        snake_alpha = context.upload(snake_alpha_values)
        snake_output = context.alloc(4)
        context.dispatch("dr3e_snake", CodecEncoderResident::SOURCE,
          [snake_input, snake_alpha, snake_output], [1_u64, 4_u64, 0x40000000_u64], 4)

        gelu_values = [-4.0_f32, -1.0_f32, 0.0_f32, 0.5_f32, 2.0_f32, 10.0_f32]
        gelu_input = context.upload(gelu_values)
        gelu_output = context.alloc(gelu_values.size.to_i32)
        context.dispatch("dr3e_gelu", CodecEncoderResident::SOURCE,
          [gelu_input, gelu_output], [gelu_values.size.to_u64], gelu_values.size.to_i32)

        # RMSNorm scales each normalized input by its per-channel weight and
        # includes epsilon inside the mean-square square root.
        rms_values = [3.0_f32, 4.0_f32, -3.0_f32, -4.0_f32]
        rms_weights = [2.0_f32, -1.0_f32]
        rms_input = context.upload(rms_values)
        rms_weight = context.upload(rms_weights)
        rms_output = context.alloc(rms_values.size)
        rms_epsilon = 0.5_f32
        context.dispatch("dr3e_rms", CodecEncoderResident::SOURCE,
          [rms_input, rms_weight, rms_output],
          [rms_values.size.to_u64, rms_weights.size.to_u64, rms_epsilon.unsafe_as(UInt32).to_u64], 2, 1)

        context.read(convolution, "terminal").should eq([2.25_f32])
        normalized = context.read(layer_norm, "diagnostic")
        (normalized[1] - 0.70710677_f32).abs.should be < 2.0e-5_f32
        (normalized[3] - 0.70710677_f32).abs.should be < 2.0e-5_f32
        snake_expected = snake_input_values.zip(snake_alpha_values).map do |value, alpha|
          sine = Math.sin((alpha * value).to_f64).to_f32
          value + (1.0_f32 / (alpha + 1.0e-9_f32)) * (sine.to_f64 ** 2.0).to_f32
        end
        context.read(snake_output, "diagnostic").zip(snake_expected).each do |actual, expected|
          (actual - expected).abs.should be < 2.0e-6_f32
        end
        gelu_expected = gelu_values.map do |value|
          erf_argument = (value / 1.4142135623730951_f32).to_f64
          (value.to_f64 * (1.0 + Math.erf(erf_argument)) / 2.0).to_f32
        end
        context.read(gelu_output, "terminal").zip(gelu_expected).each do |actual, expected|
          (actual - expected).abs.should be < 2.0e-6_f32
        end
        rms_expected = Array(Float32).new(rms_values.size) do |i|
          row = i // rms_weights.size
          channel = i % rms_weights.size
          row_base = row * rms_weights.size
          sum_squares = 0_f64
          rms_weights.size.times do |c|
            sum_squares += rms_values[row_base + c].to_f64 ** 2
          end
          mean_square = sum_squares / rms_weights.size
          (rms_values[i].to_f64 / Math.sqrt(mean_square + rms_epsilon.to_f64) * rms_weights[channel]).to_f32
        end
        context.read(rms_output, "terminal").zip(rms_expected).each do |actual, expected|
          (actual - expected).abs.should be < 2.0e-6_f32
        end
      ensure
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "handles partial BK16 row and output-channel tiles with odd-time right padding" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        # Nine rows leave one live row in the second 8-row tile. Odd input
        # lengths also force the final stride-sized window to read right zeros.
        # The C=1/C=8 cases exercise the small-channel reduction mapping and
        # causal dilation; the larger cases cover one/two BK16 blocks and
        # partial 8-channel output tiles.
        cases = [
          {input_rows: 17, rows: 9, channels: 1, out_channels: 5, kernel: 3, stride: 2, dilation: 3},
          {input_rows: 17, rows: 9, channels: 8, out_channels: 9, kernel: 3, stride: 2, dilation: 3},
          {input_rows: 17, rows: 9, channels: 16, out_channels: 5, kernel: 2, stride: 2, dilation: 1},
          {input_rows: 33, rows: 9, channels: 32, out_channels: 9, kernel: 4, stride: 4, dilation: 1},
        ]
        cases.each do |shape|
          input_values = Array(Float32).new(shape[:input_rows] * shape[:channels]) do |i|
            row = i // shape[:channels]
            channel = i % shape[:channels]
            (((row * 17 + channel * 11 + row * channel * 3) % 53) - 26).to_f32 / 128_f32
          end
          weight_values = Array(Float32).new(shape[:out_channels] * shape[:channels] * shape[:kernel]) do |i|
            tap = i % shape[:kernel]
            channel = (i // shape[:kernel]) % shape[:channels]
            out_channel = i // (shape[:kernel] * shape[:channels])
            (((out_channel * 13 + channel * 7 + tap * 5 + out_channel * tap * 3) % 37) - 18).to_f32 / 256_f32
          end
          bias_values = Array(Float32).new(shape[:out_channels]) do |out_channel|
            (((out_channel * 5 + 3) % 13) - 6).to_f32 / 128_f32
          end

          context = Resident::Context.new
          input = context.upload(input_values)
          weights = context.upload(weight_values)
          bias = context.upload(bias_values)
          output = context.alloc(shape[:rows] * shape[:out_channels])
          tiles = ((shape[:rows] + 7) // 8) * ((shape[:out_channels] + 7) // 8)
          context.dispatch("dr3e_conv1d_mma_bk16", CodecEncoderResident::SOURCE,
            [input, weights, bias, output],
            [shape[:input_rows].to_u64, shape[:rows].to_u64, shape[:channels].to_u64,
             shape[:out_channels].to_u64, shape[:kernel].to_u64, shape[:stride].to_u64,
             shape[:dilation].to_u64],
            tiles * 32, 32)

          # Independent scalar reference in PyTorch [out, in, tap] layout.
          # Float64 accumulation reduces oracle rounding while inputs and
          # weights remain the exact Float32 values uploaded to the device.
          expected = Array(Float32).new(shape[:rows] * shape[:out_channels]) do |i|
            row = i // shape[:out_channels]
            out_channel = i % shape[:out_channels]
            sum = 0_f64
            left_pad = (shape[:kernel] - 1) * shape[:dilation] + 1 - shape[:stride]
            shape[:kernel].times do |tap|
              input_row = row * shape[:stride] - left_pad + tap * shape[:dilation]
              next if input_row >= shape[:input_rows]
              next if input_row < 0
              shape[:channels].times do |channel|
                x = input_values[input_row * shape[:channels] + channel].to_f64
                w = weight_values[(out_channel * shape[:channels] + channel) * shape[:kernel] + tap].to_f64
                sum += x * w
              end
            end
            (sum + bias_values[out_channel].to_f64).to_f32
          end

          actual = context.read(output, "terminal")
          actual.size.should eq(expected.size)
          actual.zip(expected).each do |got, want|
            (got - want).abs.should be < 2.0e-5_f32
          end
          context.readbacks["terminal"].should eq((shape[:rows] * shape[:out_channels] * 4).to_i64)
        end
      ensure
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "covers split-K tails, partial output tiles, and upstream empty partitions" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        # The first shape has a two-column tail beyond eight 16-wide
        # partitions. The second deliberately mirrors the upstream heuristic:
        # P=16 with S=0 leaves the first fifteen partitions empty and assigns
        # the whole reduction to the final partition.
        cases = [
          {rows: 9, channels: 130, out_channels: 9, partitions: 8, partition_size: 16},
          {rows: 9, channels: 144, out_channels: 9, partitions: 16, partition_size: 0},
        ]
        cases.each do |shape|
          input_values = Array(Float32).new(shape[:rows] * shape[:channels]) do |i|
            row = i // shape[:channels]
            channel = i % shape[:channels]
            (((row * 29 + channel * 17 + row * channel * 3) % 67) - 33).to_f32 / 256_f32
          end
          weight_values = Array(Float32).new(shape[:out_channels] * shape[:channels]) do |i|
            out_channel = i // shape[:channels]
            channel = i % shape[:channels]
            (((out_channel * 23 + channel * 11 + out_channel * channel * 5) % 71) - 35).to_f32 / 512_f32
          end
          bias_values = Array(Float32).new(shape[:out_channels]) do |out_channel|
            (((out_channel * 7 + 2) % 17) - 8).to_f32 / 128_f32
          end

          context = Resident::Context.new
          input = context.upload(input_values)
          weights = context.upload(weight_values)
          bias = context.upload(bias_values)
          output = context.alloc(shape[:rows] * shape[:out_channels])
          tiles = ((shape[:rows] + 7) // 8) * ((shape[:out_channels] + 7) // 8)
          context.dispatch("dr3e_linear_mma", CodecEncoderResident::SOURCE,
            [input, weights, bias, output],
            [shape[:rows].to_u64, shape[:channels].to_u64, shape[:out_channels].to_u64,
             1_u64, shape[:partitions].to_u64, shape[:partition_size].to_u64],
            tiles * 32, 32)

          # Float64 scalar oracle in [M,K] x [N,K] layout independently
          # checks all reduction indices, partition boundaries, and bias.
          expected = Array(Float32).new(shape[:rows] * shape[:out_channels]) do |i|
            row = i // shape[:out_channels]
            out_channel = i % shape[:out_channels]
            sum = 0_f64
            shape[:channels].times do |channel|
              sum += input_values[row * shape[:channels] + channel].to_f64 *
                     weight_values[out_channel * shape[:channels] + channel].to_f64
            end
            (sum + bias_values[out_channel].to_f64).to_f32
          end

          actual = context.read(output, "terminal")
          actual.size.should eq(expected.size)
          actual.all?(&.finite?).should be_true
          actual.zip(expected).each do |got, want|
            (got - want).abs.should be < 2.0e-5_f32
          end
          context.readbacks["terminal"].should eq((shape[:rows] * shape[:out_channels] * 4).to_i64)
        end
      ensure
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "rejects empty and mismatched audio before any device dispatch" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        context = Resident::Context.new
        store = WeightStore.new({} of String => Tensor)
        resident = CodecEncoderResident.new(store, context)
        audio = context.upload([0_f32])
        expect_raises(ArgumentError, /non-empty/) { resident.encode(audio, 0) }
        expect_raises(ArgumentError, /sample count/) { resident.encode(audio, 2) }
        context.submissions.should eq(0_i64)
        context.readbacks.empty?.should be_true
      ensure
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end
  end
{% end %}
