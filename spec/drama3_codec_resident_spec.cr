require "spec"
require "../src/ml/audio/drama3/codec"
require "../src/ml/audio/drama3/codec_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_resident_tensor(tensors : Hash(String, Tensor), name : String,
                                     shape : Array(Int32), values : Vector) : Nil
    tensors["codec.#{name}"] = Tensor.new(shape, values)
  end

  private def drama3_resident_values(name : String, count : Int32, scale : Float32 = 0.001_f32) : Vector
    seed = name.bytes.sum { |byte| byte.to_i64 }.to_i32
    Array(Float32).new(count) do |i|
      (((i * 13 + seed * 7) % 23) - 11).to_f32 * scale
    end
  end

  private def drama3_resident_fixture(config : CodecConfig) : WeightStore
    tensors = {} of String => Tensor
    put = ->(name : String, shape : Array(Int32), scale : Float32) do
      count = shape.reduce(1) { |product, dimension| product * dimension }
      values = drama3_resident_values(name, count, scale)
      drama3_resident_tensor(tensors, name, shape, values)
    end
    put_special = ->(name : String, shape : Array(Int32), value : Float32) do
      count = shape.reduce(1) { |product, dimension| product * dimension }
      drama3_resident_tensor(tensors, name, shape, Array(Float32).new(count, value))
    end

    # Semantic book plus the nine residual books used by Generation frames.
    10.times do |book|
      prefix = book == 0 ? "quantizer.semantic_quantizer.quantizers.0" : "quantizer.quantizer.quantizers.#{book - 1}"
      vocab = book == 0 ? config.semantic_codebook_size : config.residual_codebook_size
      put.call("#{prefix}.codebook.weight", [vocab, config.codebook_dim], 0.025_f32)
      put.call("#{prefix}.out_proj.weight_v", [config.latent_channels, config.codebook_dim, 1], 0.015_f32)
      # Checkpoints retain PyTorch weight-normalization singleton dimensions;
      # Resident must consume these as flattened per-channel gains.
      put_special.call("#{prefix}.out_proj.weight_g", [config.latent_channels, 1, 1], 0.08_f32)
      put.call("#{prefix}.out_proj.bias", [config.latent_channels], 0.002_f32)
    end

    config.post_layers.times do |layer|
      prefix = "quantizer.post_module.layers.#{layer}"
      channels = config.latent_channels
      ff = config.transformer_ff_dim
      put_special.call("#{prefix}.attention_norm.weight", [channels], 1.0_f32)
      put.call("#{prefix}.attention.wqkv.weight", [channels * 3, channels], 0.003_f32)
      put.call("#{prefix}.attention.wo.weight", [channels, channels], 0.003_f32)
      put_special.call("#{prefix}.attention_layer_scale.gamma", [channels], 0.1_f32)
      put_special.call("#{prefix}.ffn_norm.weight", [channels], 1.0_f32)
      put.call("#{prefix}.feed_forward.w1.weight", [ff, channels], 0.003_f32)
      put.call("#{prefix}.feed_forward.w3.weight", [ff, channels], 0.003_f32)
      put.call("#{prefix}.feed_forward.w2.weight", [channels, ff], 0.003_f32)
      put_special.call("#{prefix}.ffn_layer_scale.gamma", [channels], 0.1_f32)
    end
    put_special.call("quantizer.post_module.norm.weight", [config.latent_channels], 1.0_f32)

    config.quantizer_upsample_rates.reverse.each_with_index do |_rate, stage|
      prefix = "quantizer.upsample.#{stage}"
      channels = config.latent_channels
      hidden = channels * 4
      put.call("#{prefix}.0.conv.weight", [channels, channels, config.quantizer_upsample_rates.reverse[stage]], 0.003_f32)
      put.call("#{prefix}.0.conv.bias", [channels], 0.002_f32)
      put.call("#{prefix}.1.dwconv.conv.weight", [channels, 1, 7], 0.003_f32)
      put.call("#{prefix}.1.dwconv.conv.bias", [channels], 0.002_f32)
      put_special.call("#{prefix}.1.norm.weight", [channels], 1.0_f32)
      put_special.call("#{prefix}.1.norm.bias", [channels], 0.001_f32)
      put.call("#{prefix}.1.pwconv1.weight", [hidden, channels], 0.003_f32)
      put.call("#{prefix}.1.pwconv1.bias", [hidden], 0.002_f32)
      put.call("#{prefix}.1.pwconv2.weight", [channels, hidden], 0.003_f32)
      put.call("#{prefix}.1.pwconv2.bias", [channels], 0.002_f32)
      put_special.call("#{prefix}.1.gamma", [channels], 0.1_f32)
    end

    decoder_channels = config.decoder_channels
    put.call("decoder.model.0.weight_v", [decoder_channels, config.latent_channels, 7], 0.003_f32)
    put_special.call("decoder.model.0.weight_g", [decoder_channels, 1, 1], 0.08_f32)
    put.call("decoder.model.0.bias", [decoder_channels], 0.002_f32)

    config.decoder_rates.each_with_index do |rate, stage|
      input_channels = decoder_channels >> stage
      channels = decoder_channels >> (stage + 1)
      prefix = "decoder.model.#{stage + 1}.block"
      put_special.call("#{prefix}.0.alpha", [1, input_channels, 1], 0.2_f32)
      put.call("#{prefix}.1.weight_v", [input_channels, channels, rate * 2], 0.003_f32)
      put_special.call("#{prefix}.1.weight_g", [input_channels, 1, 1], 0.08_f32)
      put.call("#{prefix}.1.bias", [channels], 0.002_f32)
      3.times do |unit|
        unit_prefix = "#{prefix}.#{unit + 2}.block"
        put_special.call("#{unit_prefix}.0.alpha", [1, channels, 1], 0.2_f32)
        put.call("#{unit_prefix}.1.weight_v", [channels, channels, 7], 0.003_f32)
        put_special.call("#{unit_prefix}.1.weight_g", [channels, 1, 1], 0.08_f32)
        put.call("#{unit_prefix}.1.bias", [channels], 0.002_f32)
        put_special.call("#{unit_prefix}.2.alpha", [1, channels, 1], 0.2_f32)
        put.call("#{unit_prefix}.3.weight_v", [channels, channels, 1], 0.003_f32)
        put_special.call("#{unit_prefix}.3.weight_g", [channels, 1, 1], 0.08_f32)
        put.call("#{unit_prefix}.3.bias", [channels], 0.002_f32)
      end
    end

    final_index = config.decoder_rates.size + 1
    final_channels = decoder_channels >> config.decoder_rates.size
    put_special.call("decoder.model.#{final_index}.alpha", [1, final_channels, 1], 0.2_f32)
    put.call("decoder.model.#{final_index + 1}.weight_v", [1, final_channels, 7], 0.003_f32)
    put_special.call("decoder.model.#{final_index + 1}.weight_g", [1, 1, 1], 0.08_f32)
    put.call("decoder.model.#{final_index + 1}.bias", [1], 0.002_f32)
    WeightStore.new(tensors)
  end

  describe "DRAMA-3 resident codec" do
    it "matches the native codec in both code layouts without activation readbacks" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        config = CodecConfig.new(
          latent_channels: 8,
          codebook_dim: 2,
          semantic_codebook_size: 5,
          residual_codebooks: 9,
          residual_codebook_size: 4,
          post_layers: 1,
          transformer_heads: 2,
          transformer_head_dim: 4,
          transformer_ff_dim: 12,
          transformer_window: 2,
          quantizer_upsample_rates: [2],
          decoder_channels: 8,
          decoder_rates: [2],
        )
        store = drama3_resident_fixture(config)
        frames = 3
        codes = Array(Float32).new(frames * 11, 0.0_f32)
        semantic = [-1, 2, 99]
        cpu_books = Array(Array(Int32)).new(10) { Array(Int32).new(frames, 0) }
        frames.times do |frame|
          codes[frame * 11] = (10_000 + frame).to_f32 # Main token is deliberately ignored.
          codes[frame * 11 + 1] = semantic[frame].to_f32
          cpu_books[0][frame] = semantic[frame]
          9.times do |residual|
            token = (frame * 3 + residual * 2) % 4
            column = residual + 2 # c1..c9 follow semantic c0.
            codes[frame * 11 + column] = token.to_f32
            cpu_books[residual + 1][frame] = token
          end
        end

        expected = Codec.new(store, config).decode(cpu_books)
        context = Resident::Context.new
        input = context.upload(codes)
        decoder = CodecResident.new(store, context, config)
        output = decoder.decode(input, frames)
        output.count.should eq(expected.size)
        actual = context.read(output, "terminal")
        actual.zip(expected).each { |a, b| (a - b).abs.should be < 5.0e-4_f32 }

        codebook_major = Array(Float32).new(frames * 10, 0.0_f32)
        frames.times do |frame|
          codebook_major[frame] = semantic[frame].to_f32
          9.times do |residual|
            codebook_major[(residual + 1) * frames + frame] = cpu_books[residual + 1][frame].to_f32
          end
        end
        second = decoder.decode(context.upload(codebook_major), frames, true)
        actual_major = context.read(second, "terminal")
        actual_major.zip(expected).each { |a, b| (a - b).abs.should be < 5.0e-4_f32 }
        context.readbacks.should eq({"control" => 8_i64, "terminal" => expected.size.to_i64 * 8})

        # A device observer retains the actual buffers, rather than invoking
        # the legacy CPU trace callback. Enabling it must preserve all output
        # bits and leave the readback ledger empty until the terminal read.
        traced_context = Resident::Context.new
        traced_decoder = CodecResident.new(store, traced_context, config)
        captures = {} of String => Tuple(Resident::Buffer, Int32, Int32)
        traced_decoder.device_trace = ->(name : String, buffer : Resident::Buffer, rows : Int32, channels : Int32) {
          traced_context.readbacks.empty?.should be_true
          buffer.count.should eq(rows * channels)
          captures[name] = {buffer, rows, channels}
          nil
        }
        traced_output = traced_decoder.decode(traced_context.upload(codebook_major), frames, true)
        traced_context.readbacks.empty?.should be_true
        traced_context.read(traced_output, "terminal").should eq(actual_major)
        captures["quantizer.book.0.embedding"][1..2].should eq({frames, config.codebook_dim})
        captures["quantizer.book.9.projection"][1..2].should eq({frames, config.latent_channels})
        captures.has_key?("quantizer.post_module.layers.0").should be_true
        captures["waveform"][0].should be(traced_output)
        traced_context.readbacks.should eq({"control" => 4_i64, "terminal" => expected.size.to_i64 * 4})
      ensure
        if previous
          ENV["DRAMA3_BACKEND"] = previous
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end

    it "rejects a malformed frame-major token buffer before dispatch" do
      previous = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        config = CodecConfig.new(latent_channels: 8, transformer_heads: 2, transformer_head_dim: 4,
          decoder_channels: 8, decoder_rates: [1])
        store = WeightStore.new({} of String => Tensor)
        context = Resident::Context.new
        short = context.upload([0_f32])
        expect_raises(ArgumentError, /frame-major|token buffer/) { CodecResident.new(store, context, config).decode(short, 1) }
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
