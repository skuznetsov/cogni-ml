require "spec"
require "../src/ml/audio/drama3/codec"
require "../src/ml/audio/drama3/codec_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  DR3C_LAYOUT_RECEIPT_SOURCE = CodecResident::SOURCE + <<-METAL
    inline bool dr3c_layout_bits_differ(uint actual, uint reference) {
        return actual != reference;
    }

    kernel void dr3c_layout_receipt(
        device const float* native_effective [[buffer(0)]],
        device const float* packed_effective [[buffer(1)]],
        device const float* output_native [[buffer(2)]],
        device const float* output_candidate [[buffer(3)]],
        device atomic_uint* receipt [[buffer(4)]],
        constant ulong* params [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
        uint weight_count = uint(params[0]);
        uint per_channel = uint(params[1]);
        uint out_channels = uint(params[2]);
        uint output_count = uint(params[3]);

        if (index < weight_count) {
            uint out_channel = index % out_channels;
            uint inner = index / out_channels;
            uint native_index = out_channel * per_channel + inner;
            float actual = native_effective[native_index];
            float packed = packed_effective[index];
            if (!isfinite(actual) || !isfinite(packed)) {
                atomic_fetch_add_explicit(&receipt[1], 1u, memory_order_relaxed);
            }
            if (dr3c_layout_bits_differ(as_type<uint>(actual), as_type<uint>(packed))) {
                atomic_fetch_add_explicit(&receipt[0], 1u, memory_order_relaxed);
            }
        }

        if (index < output_count) {
            float actual = output_native[index];
            float candidate = output_candidate[index];
            uint actual_bits = as_type<uint>(actual);
            uint candidate_bits = as_type<uint>(candidate);
            if (!isfinite(actual) || !isfinite(candidate)) {
                atomic_fetch_add_explicit(&receipt[3], 1u, memory_order_relaxed);
            }
            if (dr3c_layout_bits_differ(actual_bits, candidate_bits)) {
                atomic_fetch_add_explicit(&receipt[2], 1u, memory_order_relaxed);
            }

            // Exercise the same bit comparator with a deliberately perturbed
            // candidate, without materializing or reading an activation array.
            if (ulong(index) == params[4]) {
                uint negative_bits = candidate_bits ^ 1u;
                if (dr3c_layout_bits_differ(actual_bits, negative_bits)) {
                    atomic_fetch_add_explicit(&receipt[4], 1u, memory_order_relaxed);
                }
            }
        }
    }
    METAL

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
      previous_layout = ENV["DRAMA3_DAC_CONV_LAYOUT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "native"
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

        # The small real-geometry codec fixture includes ungrouped normalized
        # Conv1d, plus non-normalized depthwise Conv1d in ConvNeXt. The opt-in
        # path must preserve every terminal bit and leave the depthwise path on
        # its original grouped kernel.
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "coalesced"
        coalesced_context = Resident::Context.new
        coalesced_decoder = CodecResident.new(store, coalesced_context, config)
        coalesced = coalesced_decoder.decode(coalesced_context.upload(codebook_major), frames, true)
        coalesced_context.read(coalesced, "terminal").map(&.unsafe_as(UInt32)).should eq(actual_major.map(&.unsafe_as(UInt32)))
        coalesced_context.readbacks.should eq({"control" => 4_i64, "terminal" => expected.size.to_i64 * 4})
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "native"

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
        if previous_layout
          ENV["DRAMA3_DAC_CONV_LAYOUT"] = previous_layout
        else
          ENV.delete("DRAMA3_DAC_CONV_LAYOUT")
        end
      end
    end

    it "keeps the selected Conv1d layout frozen and rejects invalid values before dispatch" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_layout = ENV["DRAMA3_DAC_CONV_LAYOUT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        config = CodecConfig.new(latent_channels: 8, transformer_heads: 2, transformer_head_dim: 4,
          decoder_channels: 8, decoder_rates: [1])
        context = Resident::Context.new
        store = WeightStore.new({} of String => Tensor)

        ENV.delete("DRAMA3_DAC_CONV_LAYOUT")
        CodecResident.new(store, context, config).dac_conv_layout.should eq("coalesced")
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "native"
        decoder = CodecResident.new(store, context, config)
        decoder.dac_conv_layout.should eq("native")
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "coalesced"
        decoder.dac_conv_layout.should eq("native")

        before = context.submissions
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "invalid"
        expect_raises(ArgumentError, /DRAMA3_DAC_CONV_LAYOUT/) do
          CodecResident.new(store, context, config)
        end
        context.submissions.should eq(before)
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_layout
          ENV["DRAMA3_DAC_CONV_LAYOUT"] = previous_layout
        else
          ENV.delete("DRAMA3_DAC_CONV_LAYOUT")
        end
      end
    end

    it "packs normalized ungrouped Conv1d weights without changing F32 results" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_layout = ENV["DRAMA3_DAC_CONV_LAYOUT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_DAC_CONV_LAYOUT"] = "coalesced"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        # The odd shape exercises effective-weight and output tails, dilation,
        # and bias placement. The second matches the checkpoint's final
        # residual-unit Conv1d geometry (96 channels, 7 taps, dilation 9).
        [
          {rows: 23, in_channels: 7, out_channels: 11, kernel: 7, stride: 1, dilation: 3},
          {rows: 23, in_channels: 96, out_channels: 96, kernel: 7, stride: 1, dilation: 9},
        ].each do |geometry|
          rows = geometry[:rows]
          in_channels = geometry[:in_channels]
          out_channels = geometry[:out_channels]
          kernel_size = geometry[:kernel]
          stride = geometry[:stride]
          dilation = geometry[:dilation]
          per_channel = in_channels * kernel_size
          weight_count = out_channels * per_channel
          input_values = Array(Float32).new(rows * in_channels) do |i|
            (((i * 17) % 43) - 21).to_f32 * 0.0005_f32
          end
          weight_values = Array(Float32).new(weight_count) do |i|
            (((i * 7) % 23) - 11).to_f32 * 0.001_f32
          end
          gains = Array(Float32).new(out_channels) do |i|
            (0.04_f32 + i.to_f32 * 0.003_f32)
          end
          biases = Array(Float32).new(out_channels) do |i|
            (i - 5).to_f32 * 0.002_f32
          end

          context = Resident::Context.new
          input = context.upload(input_values)
          weight = context.upload(weight_values)
          gain = context.upload(gains)
          bias = context.upload(biases)
          scales = context.alloc(out_channels)
          effective_native = context.alloc(weight_count)
          effective_coalesced = context.alloc(weight_count)
          output_native = context.alloc(rows * out_channels)
          output_coalesced = context.alloc(rows * out_channels)

          context.dispatch("dr3c_wn_scales", CodecResident::SOURCE, [weight, gain, scales],
            [out_channels.to_u64, per_channel.to_u64], out_channels)
          context.dispatch("dr3c_wn_apply", CodecResident::SOURCE, [weight, scales, effective_native],
            [weight_count.to_u64, per_channel.to_u64], weight_count)
          context.dispatch("dr3c_wn_apply_coalesced", CodecResident::SOURCE, [weight, scales, effective_coalesced],
            [weight_count.to_u64, per_channel.to_u64], weight_count)
          context.dispatch("dr3c_conv1d", CodecResident::SOURCE,
            [input, effective_native, bias, output_native],
            [rows.to_u64, in_channels.to_u64, out_channels.to_u64, kernel_size.to_u64,
             stride.to_u64, 1_u64, dilation.to_u64], rows * out_channels)
          context.dispatch("dr3c_conv1d_coalesced", CodecResident::SOURCE,
            [input, effective_coalesced, bias, output_coalesced],
            [rows.to_u64, in_channels.to_u64, out_channels.to_u64, kernel_size.to_u64,
             stride.to_u64, dilation.to_u64], rows * out_channels)

          output_count = rows * out_channels
          negative_index = output_count - 1
          receipt = context.upload(Array(Float32).new(5, 0_f32))
          context.dispatch("dr3c_layout_receipt", DR3C_LAYOUT_RECEIPT_SOURCE,
            [effective_native, effective_coalesced, output_native, output_coalesced, receipt],
            [weight_count.to_u64, per_channel.to_u64, out_channels.to_u64,
             output_count.to_u64, negative_index.to_u64], Math.max(weight_count, output_count))
          context.readbacks.should be_empty
          counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
          counts.should eq([0_u32, 0_u32, 0_u32, 0_u32, 1_u32])
          context.readbacks.should eq({"terminal" => 20_i64})
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_layout
          ENV["DRAMA3_DAC_CONV_LAYOUT"] = previous_layout
        else
          ENV.delete("DRAMA3_DAC_CONV_LAYOUT")
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
