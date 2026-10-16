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

  private def drama3_resident_fixture(config : CodecConfig, omit_name : String? = nil) : WeightStore
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
    if name = omit_name
      tensors.delete("codec.#{name}")
    end
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

    it "reuses decoder scratch bitwise at synchronous and graph depth two" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_scratch = ENV["DRAMA3_DAC_SCRATCH"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_DAC_SCRATCH"] = "0"
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
        codes = Array(Float32).new(frames * 11, 0_f32)

        {"0", "2"}.each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
          context = Resident::Context.new
          input = context.upload(codes)
          before_reference_graph = context.graph_stats
          before_reference_readbacks = context.readbacks.dup
          reference = CodecResident.new(store, context, config).decode(input, frames)
          reference_graph = context.graph_stats
          context.readbacks.should eq(before_reference_readbacks)
          reference_graph["pending"].should eq(0_i64)
          reference_graph["completed"].should eq(reference_graph["submitted"])
          reference_submissions = reference_graph["submitted"] - before_reference_graph["submitted"]
          reference_waits = reference_graph["completion_waits"] - before_reference_graph["completion_waits"]
          expected_bits = context.read(reference, "terminal").map(&.unsafe_as(UInt32))

          ENV["DRAMA3_DAC_SCRATCH"] = "1"
          decoder = CodecResident.new(store, context, config)
          2.times do
            before_graph = context.graph_stats
            before_readbacks = context.readbacks.dup
            output = decoder.decode(input, frames)
            after_graph = context.graph_stats
            context.readbacks.should eq(before_readbacks)
            after_graph["pending"].should eq(0_i64)
            after_graph["completed"].should eq(after_graph["submitted"])
            (after_graph["submitted"] - before_graph["submitted"]).should eq(reference_submissions)
            (after_graph["completion_waits"] - before_graph["completion_waits"]).should eq(reference_waits)
            context.read(output, "terminal").map(&.unsafe_as(UInt32)).should eq(expected_bits)
            context.readbacks["terminal"].should eq(before_readbacks["terminal"] + output.count.to_i64 * 4)
            context.readbacks["control"].should eq(before_readbacks["control"] + 4_i64)
            context.readbacks["diagnostic"].should eq(before_readbacks["diagnostic"])
          end
          stats = decoder.dac_scratch_stats
          stats[:enabled].should be_true
          stats[:bytes].should eq(3_i64 * 48 * sizeof(Float32))
          stats[:reuses].should be > 0
          stats[:released].should be_true
          stats[:quarantined].should be_false
          ENV["DRAMA3_DAC_SCRATCH"] = "0"
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
        if previous_scratch
          ENV["DRAMA3_DAC_SCRATCH"] = previous_scratch
        else
          ENV.delete("DRAMA3_DAC_SCRATCH")
        end
      end
    end

    it "drains after a late decoder failure and leaves the aborted context closed" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_scratch = ENV["DRAMA3_DAC_SCRATCH"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ENV["DRAMA3_DAC_SCRATCH"] = "1"
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
        frames = 3
        codes = Array(Float32).new(frames * 11, 0_f32)
        incomplete = drama3_resident_fixture(config, "decoder.model.3.weight_v")
        context = Resident::Context.new
        input = context.upload(codes)
        decoder = CodecResident.new(incomplete, context, config)

        expect_raises(ArgumentError, /missing codec tensor codec\.decoder\.model\.3\.weight_v/) do
          decoder.decode(input, frames)
        end

        scratch = decoder.dac_scratch_stats
        scratch[:enabled].should be_true
        scratch[:bytes].should eq(3_i64 * 48 * sizeof(Float32))
        scratch[:released].should be_true
        scratch[:quarantined].should be_false
        failed_graph = context.graph_stats
        failed_graph["submitted"].should be > 3_i64
        failed_graph["pending"].should eq(0_i64)
        failed_graph["completed"].should eq(failed_graph["submitted"])
        expect_raises(ArgumentError, /resident context aborted/) { decoder.decode(input, frames) }

        # Context abort is terminal. A fresh context remains usable after this
        # late decoder failure and its queue drain.
        next_context = Resident::Context.new
        next_input = next_context.upload(codes)
        next_decoder = CodecResident.new(drama3_resident_fixture(config), next_context, config)
        next_output = next_decoder.decode(next_input, frames)
        next_decoder.dac_scratch_stats[:enabled].should be_true
        next_context.read(next_output, "terminal").size.should eq(12)
        next_context.graph_stats["pending"].should eq(0_i64)

        # Depth zero reports completed=submitted synthetically. An error there
        # must retain scratch because Context#abort cannot certify its wait.
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        synchronous_context = Resident::Context.new
        synchronous_input = synchronous_context.upload(codes)
        synchronous_decoder = CodecResident.new(incomplete, synchronous_context, config)
        expect_raises(ArgumentError, /missing codec tensor codec\.decoder\.model\.3\.weight_v/) do
          synchronous_decoder.decode(synchronous_input, frames)
        end
        synchronous_decoder.dac_scratch_stats[:released].should be_false
        synchronous_decoder.dac_scratch_stats[:quarantined].should be_true
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
        if previous_scratch
          ENV["DRAMA3_DAC_SCRATCH"] = previous_scratch
        else
          ENV.delete("DRAMA3_DAC_SCRATCH")
        end
      end
    end

    it "keeps captured device activations immutable by disabling scratch" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_scratch = ENV["DRAMA3_DAC_SCRATCH"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ENV["DRAMA3_DAC_SCRATCH"] = "1"
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
        codes = Array(Float32).new(frames * 11, 0_f32)
        context = Resident::Context.new
        decoder = CodecResident.new(store, context, config)
        captures = {} of String => Resident::Buffer
        decoder.device_trace = ->(name : String, buffer : Resident::Buffer, _rows : Int32, _channels : Int32) {
          captures[name] = buffer if name == "decoder.model.1.block.2"
          nil
        }

        decoder.decode(context.upload(codes), frames)
        first_capture = captures["decoder.model.1.block.2"]
        first_bits = context.read(first_capture, "diagnostic").map(&.unsafe_as(UInt32))
        decoder.dac_scratch_stats[:enabled].should be_false
        decoder.dac_scratch_stats[:bytes].should eq(0_i64)

        changed_codes = codes.dup
        changed_codes[1] = 1_f32
        decoder.decode(context.upload(changed_codes), frames)
        context.read(first_capture, "diagnostic").map(&.unsafe_as(UInt32)).should eq(first_bits)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
        if previous_scratch
          ENV["DRAMA3_DAC_SCRATCH"] = previous_scratch
        else
          ENV.delete("DRAMA3_DAC_SCRATCH")
        end
      end
    end
  end
{% end %}
