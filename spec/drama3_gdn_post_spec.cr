require "spec"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private alias Drama3GdnPostBuffer = ML::Audio::Drama3::Resident::Buffer
  private alias Drama3GdnPostPair = Tuple(Drama3GdnPostBuffer, Drama3GdnPostBuffer)

  private def drama3_gdn_post_with_device(depth : String = "0", &block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_DENSE_REDUCTION", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
            "DRAMA3_METAL_TIMELINE", "DRAMA3_METAL_RESOURCES", "DRAMA3_GDN_POST_FUSION",
            "DRAMA3_PROFILE"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_METAL_RESOURCES"] = "0"
      ENV.delete("DRAMA3_GDN_POST_FUSION")
      ENV.delete("DRAMA3_PROFILE")
      ML::Audio::Drama3::Profile.configure_from_env!
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
      yield
    ensure
      ML::Audio::Drama3::MetalBackend.reset!
      keys.each do |key|
        if value = previous[key]
          ENV[key] = value.not_nil!
        else
          ENV.delete(key)
        end
      end
      ML::Audio::Drama3::Profile.configure_from_env!
    end
  end

  private DRAMA3_GDN_POST_RECEIPT_SOURCE = ML::Audio::Drama3::Resident::SOURCE + <<-METAL
    inline uint dr3_gdn_post_class(float value) {
        if (isnan(value)) return 1u;
        if (isinf(value)) return (as_type<uint>(value) >> 31) ? 3u : 2u;
        return 0u;
    }

    inline void dr3_gdn_post_record(device atomic_uint* receipt, uint base,
                                    float expected, float actual) {
        uint expected_class = dr3_gdn_post_class(expected);
        uint actual_class = dr3_gdn_post_class(actual);
        if (expected_class == 0u && actual_class == 0u) {
            if (as_type<uint>(expected) != as_type<uint>(actual))
                atomic_fetch_add_explicit(&receipt[base], 1u, memory_order_relaxed);
        }
        if (expected_class != actual_class)
            atomic_fetch_add_explicit(&receipt[base + 1], 1u, memory_order_relaxed);
        if (expected_class == 0u) atomic_fetch_add_explicit(&receipt[base + 2], 1u, memory_order_relaxed);
        if (actual_class == 0u) atomic_fetch_add_explicit(&receipt[base + 3], 1u, memory_order_relaxed);
        if (expected_class == 1u) atomic_fetch_add_explicit(&receipt[base + 4], 1u, memory_order_relaxed);
        if (actual_class == 1u) atomic_fetch_add_explicit(&receipt[base + 5], 1u, memory_order_relaxed);
        if (expected_class == 2u) atomic_fetch_add_explicit(&receipt[base + 6], 1u, memory_order_relaxed);
        if (actual_class == 2u) atomic_fetch_add_explicit(&receipt[base + 7], 1u, memory_order_relaxed);
        if (expected_class == 3u) atomic_fetch_add_explicit(&receipt[base + 8], 1u, memory_order_relaxed);
        if (actual_class == 3u) atomic_fetch_add_explicit(&receipt[base + 9], 1u, memory_order_relaxed);
    }

    kernel void dr3_gdn_post_receipt(
        device const float* expected [[buffer(0)]], device const float* actual [[buffer(1)]],
        device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
        if (ulong(index) >= params[0]) return;
        dr3_gdn_post_record(receipt, uint(params[1]), expected[index], actual[index]);
    }
    METAL

  private def drama3_gdn_post_upload_view(context : ML::Audio::Drama3::Resident::Context,
                                          values : Array(Float32)) : Drama3GdnPostBuffer
    context.upload([123.25_f32] + values + [-321.5_f32]).slice(1, values.size)
  end

  private def drama3_gdn_post_receipt_pair(context : ML::Audio::Drama3::Resident::Context,
                                           expected : Drama3GdnPostBuffer,
                                           actual : Drama3GdnPostBuffer,
                                           receipt : Drama3GdnPostBuffer,
                                           base : UInt64) : Nil
    expected.count.should eq(actual.count)
    context.dispatch("dr3_gdn_post_receipt", DRAMA3_GDN_POST_RECEIPT_SOURCE,
      [expected, actual, receipt], [expected.count.to_u64, base], expected.count)
  end

  private def drama3_gdn_post_accepts?(counts : Array(UInt32), base : Int32) : Bool
    counts[base] == 0_u32 && counts[base + 1] == 0_u32 &&
      counts[base + 2] == counts[base + 3] &&
      counts[base + 4] == counts[base + 5] &&
      counts[base + 6] == counts[base + 7] &&
      counts[base + 8] == counts[base + 9]
  end

  private def drama3_gdn_post_pending_unchanged(context : ML::Audio::Drama3::Resident::Context,
                                                graph : Hash(String, Int64),
                                                readbacks : Hash(String, Int64)) : Nil
    current = context.graph_stats
    current["submitted"].should eq(graph["submitted"])
    current["completion_waits"].should eq(graph["completion_waits"])
    current["pending"].should eq(graph["pending"])
    context.readbacks.should eq(readbacks)
  end

  private def drama3_gdn_post_compare_and_read(context : ML::Audio::Drama3::Resident::Context,
                                               pairs : Array(Drama3GdnPostPair),
                                               before_graph : Hash(String, Int64),
                                               before_readbacks : Hash(String, Int64)) : Array(UInt32)
    # Ten counters per pair: finite bit mismatches, class mismatches, and the
    # expected/actual counts for finite, NaN, +Inf, and -Inf values. NaN payload
    # bits are deliberately not compared; class parity is the established contract.
    receipt = context.upload(Array(Float32).new((pairs.size + 1) * 10, 0_f32))
    pairs.each_with_index do |pair, index|
      drama3_gdn_post_receipt_pair(context, pair[0], pair[1], receipt, (index * 10).to_u64)
    end

    # Seed one finite ULP mismatch, finite-vs-NaN, and opposite infinity signs.
    one = 1_f32
    one_ulp_up = (one.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
    negative_expected = context.upload([one, Float32::NAN, Float32::INFINITY, -Float32::INFINITY])
    negative_actual = context.upload([one_ulp_up, 0_f32, -Float32::INFINITY, Float32::INFINITY])
    control_base = (pairs.size * 10).to_u64
    drama3_gdn_post_receipt_pair(context, negative_expected, negative_actual, receipt, control_base)
    drama3_gdn_post_pending_unchanged(context, before_graph, before_readbacks)

    counts = context.read(receipt, "terminal").map(&.unsafe_as(UInt32))
    pairs.each_index do |index|
      drama3_gdn_post_accepts?(counts, index * 10).should be_true
    end
    control = pairs.size * 10
    counts[control, 10].should eq([1_u32, 3_u32, 1_u32, 2_u32, 1_u32, 0_u32,
                                   1_u32, 1_u32, 1_u32, 1_u32])
    drama3_gdn_post_accepts?(counts, control).should be_false
    context.readbacks.keys.should eq(["terminal"])
    context.readbacks["terminal"].should eq(receipt.count.to_i64 * 4_i64)
    context.graph_stats["submitted"].should eq(1_i64)
    context.graph_stats["completed"].should eq(1_i64)
    context.graph_stats["completion_waits"].should eq(1_i64)
    context.graph_stats["pending"].should eq(0_i64)
    counts
  end

  private def drama3_gdn_post_finite_values(width : Int32, heads : Int32)
    count = width * heads
    tiny = 0x00000001_u32.unsafe_as(Float32)
    negative_zero = 0x80000000_u32.unsafe_as(Float32)
    x = Array(Float32).new(count) do |index|
      lane = (index + (index // width) * 3) % 13
      case lane
      when 0 then 1_048_576_f32
      when 1 then 280_f32
      when 2 then -280_f32
      when 3 then tiny
      when 4 then -tiny
      when 5 then negative_zero
      when 6 then 1e20_f32
      when 7 then -1e20_f32
      when 8 then 4096_f32
      when 9 then -4096_f32
      else        (((index * 17 + index // 5) % 37) - 18).to_f32 / 32_f32
      end
    end
    z = Array(Float32).new(count) do |index|
      case index % 11
      when 0 then -80_f32
      when 1 then 80_f32
      when 2 then negative_zero
      when 3 then tiny
      when 4 then -tiny
      when 5 then 20_f32
      when 6 then -20_f32
      else        (((index * 13 + 7) % 29) - 14).to_f32 / 8_f32
      end
    end
    w = Array(Float32).new(width) do |index|
      case index % 9
      when 0 then 0.75_f32
      when 1 then -1.25_f32
      when 2 then 4096_f32
      when 3 then 1e-20_f32
      when 4 then tiny
      when 5 then negative_zero
      else        (((index * 7 + 3) % 17) - 8).to_f32 / 4_f32
      end
    end
    {x: x, z: z, w: w}
  end

  private def drama3_gdn_post_well_scaled_values(width : Int32, heads : Int32)
    count = width * heads
    x = Array(Float32).new(count) do |index|
      head = index // width
      (((index * 17 + width * 3 + head * 11) % 41) - 20).to_f32 / 16_f32
    end
    z = Array(Float32).new(count) do |index|
      head = index // width
      (((index * 13 + width * 5 + head * 7) % 49) - 24).to_f32 / 8_f32
    end
    w = Array(Float32).new(width) do |index|
      (((index * 11 + width * 3) % 25) - 12).to_f32 / 16_f32
    end
    {x: x, z: z, w: w}
  end

  describe "DRAMA-3 resident GDN post fusion" do
    it "defaults off, freezes the selector, and keeps the explicit allocation path" do
      drama3_gdn_post_with_device do
        ENV.delete("DRAMA3_GDN_POST_FUSION")
        context = ML::Audio::Drama3::Resident::Context.new
        begin
          context.gdn_post_fusion_enabled.should be_false
          values = drama3_gdn_post_finite_values(33, 2)
          x = drama3_gdn_post_upload_view(context, values[:x])
          z = drama3_gdn_post_upload_view(context, values[:z])
          w = drama3_gdn_post_upload_view(context, values[:w])
          [x.offset, z.offset, w.offset].should eq([4_i64, 4_i64, 4_i64])
          before_graph = context.graph_stats.dup
          before_readbacks = context.readbacks.dup

          reference = context.mul(context.rms(x, w, 1.0e-6, heads: 2), context.silu(z))
          before_api_allocations = context.allocation_stats[:allocations]
          before_fused = context.gdn_post_fused_dispatches
          ENV["DRAMA3_GDN_POST_FUSION"] = "1"
          context.gdn_post_fusion_enabled.should be_false
          actual = context.gdn_post(x, z, w, 1.0e-6, heads: 2)
          actual.count.should eq(x.count)
          context.allocation_stats[:allocations].should eq(before_api_allocations + 3_i64)
          context.gdn_post_fused_dispatches.should eq(before_fused)

          drama3_gdn_post_compare_and_read(context, [{reference, actual}], before_graph, before_readbacks)
        ensure
          context.abort
        end
      end
    end

    it "strictly validates the frozen 0-or-1 selector without GPU work" do
      drama3_gdn_post_with_device do
        ENV.delete("DRAMA3_GDN_POST_FUSION")
        default_context = ML::Audio::Drama3::Resident::Context.new
        begin
          default_context.gdn_post_fusion_enabled.should be_false
          ENV["DRAMA3_GDN_POST_FUSION"] = "1"
          default_context.gdn_post_fusion_enabled.should be_false
          default_context.allocation_stats[:allocations].should eq(0_i64)
          default_context.gdn_post_fused_dispatches.should eq(0_i64)
          default_context.graph_stats["submitted"].should eq(0_i64)
          default_context.readbacks.should be_empty

          enabled_context = ML::Audio::Drama3::Resident::Context.new
          begin
            enabled_context.gdn_post_fusion_enabled.should be_true
            ENV["DRAMA3_GDN_POST_FUSION"] = "0"
            enabled_context.gdn_post_fusion_enabled.should be_true
            enabled_context.allocation_stats[:allocations].should eq(0_i64)
            enabled_context.gdn_post_fused_dispatches.should eq(0_i64)
            enabled_context.graph_stats["submitted"].should eq(0_i64)
            enabled_context.readbacks.should be_empty
          ensure
            enabled_context.abort
          end

          ["", "true", "2"].each do |invalid|
            ENV["DRAMA3_GDN_POST_FUSION"] = invalid
            expect_raises(ArgumentError, /DRAMA3_GDN_POST_FUSION/) do
              ML::Audio::Drama3::Resident::Context.new
            end
          end
        ensure
          default_context.abort
        end
      end
    end

    it "matches the explicit RMS then SiLU then multiply path bitwise across head widths" do
      ["0", "2"].each do |depth|
        drama3_gdn_post_with_device(depth) do
          ENV["DRAMA3_GDN_POST_FUSION"] = "1"
          context = ML::Audio::Drama3::Resident::Context.new
          begin
            context.graph_stats["depth"].should eq(depth.to_i64)
            context.gdn_post_fusion_enabled.should be_true
            ENV["DRAMA3_GDN_POST_FUSION"] = "0"
            context.gdn_post_fusion_enabled.should be_true
            before_graph = context.graph_stats.dup
            before_readbacks = context.readbacks.dup
            pairs = [] of Drama3GdnPostPair
            widths = [1, 31, 32, 33, 64, 127, 128, 129, 256]

            widths.each do |width|
              heads = 3
              {drama3_gdn_post_finite_values(width, heads),
               drama3_gdn_post_well_scaled_values(width, heads)}.each do |values|
                x = drama3_gdn_post_upload_view(context, values[:x])
                z = drama3_gdn_post_upload_view(context, values[:z])
                w = drama3_gdn_post_upload_view(context, values[:w])
                [x.offset, z.offset, w.offset].should eq([4_i64, 4_i64, 4_i64])

                reference = context.mul(context.rms(x, w, 1.0e-6, heads: heads), context.silu(z))
                before_allocations = context.allocation_stats[:allocations]
                before_fused = context.gdn_post_fused_dispatches
                actual = context.gdn_post(x, z, w, 1.0e-6, heads: heads)
                actual.count.should eq(width * heads)
                context.allocation_stats[:allocations].should eq(before_allocations + 1_i64)
                context.gdn_post_fused_dispatches.should eq(before_fused + 1_i64)
                pairs << {reference, actual}
              end
            end

            drama3_gdn_post_compare_and_read(context, pairs, before_graph, before_readbacks)
          ensure
            context.abort
          end
        end
      end
    end

    it "checks non-finite output classes without comparing NaN payloads" do
      drama3_gdn_post_with_device do
        ENV["DRAMA3_GDN_POST_FUSION"] = "1"
        context = ML::Audio::Drama3::Resident::Context.new
        begin
          x_values = [1_f32, 2_f32, 3_f32, 4_f32,
                      Float32::INFINITY, 1_f32, 2_f32, 3_f32,
                      Float32::NAN, 1_f32, 2_f32, 3_f32,
                      1_f32, 2_f32, 3_f32, 4_f32]
          z_values = [Float32::NAN, Float32::INFINITY, -Float32::INFINITY, 0_f32] +
                     Array(Float32).new(12, 1_f32)
          w_values = [1_f32, Float32::INFINITY, -Float32::INFINITY, 1_f32]
          x = drama3_gdn_post_upload_view(context, x_values)
          z = drama3_gdn_post_upload_view(context, z_values)
          w = drama3_gdn_post_upload_view(context, w_values)
          before_graph = context.graph_stats.dup
          before_readbacks = context.readbacks.dup

          reference = context.mul(context.rms(x, w, 1.0e-6, heads: 4), context.silu(z))
          actual = context.gdn_post(x, z, w, 1.0e-6, heads: 4)
          context.gdn_post_fused_dispatches.should eq(1_i64)
          counts = drama3_gdn_post_compare_and_read(context, [{reference, actual}], before_graph, before_readbacks)
          counts[4].should be > 0_u32
          (counts[6] + counts[8]).should be > 0_u32
        ensure
          context.abort
        end
      end
    end

    it "rejects foreign buffers and invalid shapes before allocation or encoding" do
      drama3_gdn_post_with_device do
        ENV["DRAMA3_GDN_POST_FUSION"] = "1"
        context = ML::Audio::Drama3::Resident::Context.new
        foreign = ML::Audio::Drama3::Resident::Context.new
        begin
          x = context.upload(Array(Float32).new(6, 0.25_f32))
          z = context.upload(Array(Float32).new(6, -0.5_f32))
          w = context.upload(Array(Float32).new(3, 1_f32))
          short_z = context.upload(Array(Float32).new(5, 0_f32))
          short_w = context.upload(Array(Float32).new(2, 1_f32))
          foreign_x = foreign.upload(Array(Float32).new(6, 0.25_f32))
          foreign_z = foreign.upload(Array(Float32).new(6, -0.5_f32))
          foreign_w = foreign.upload(Array(Float32).new(3, 1_f32))
          before_allocations = context.allocation_stats[:allocations]
          before_graph = context.graph_stats.dup
          before_readbacks = context.readbacks.dup
          before_fused = context.gdn_post_fused_dispatches
          before_pipelines = ML::Metal::PipelineCache.entry_count

          expect_raises(ArgumentError) { context.gdn_post(foreign_x, z, w, 1.0e-6, heads: 2) }
          expect_raises(ArgumentError) { context.gdn_post(x, foreign_z, w, 1.0e-6, heads: 2) }
          expect_raises(ArgumentError) { context.gdn_post(x, z, foreign_w, 1.0e-6, heads: 2) }
          expect_raises(ArgumentError) { context.gdn_post(x, short_z, w, 1.0e-6, heads: 2) }
          expect_raises(ArgumentError) { context.gdn_post(x, z, short_w, 1.0e-6, heads: 2) }
          expect_raises(ArgumentError) { context.gdn_post(x, z, w, 1.0e-6, heads: 0) }
          expect_raises(ArgumentError) { context.gdn_post(x, z, w, 1.0e-6, heads: 4) }
          expect_raises(ArgumentError) { context.gdn_post(x, z, w, 0.0, heads: 2) }
          expect_raises(ArgumentError) { context.gdn_post(x, z, w, Float64::NAN, heads: 2) }

          context.allocation_stats[:allocations].should eq(before_allocations)
          context.graph_stats.should eq(before_graph)
          context.readbacks.should eq(before_readbacks)
          context.gdn_post_fused_dispatches.should eq(before_fused)
          ML::Metal::PipelineCache.entry_count.should eq(before_pipelines)
        ensure
          context.abort
          foreign.abort
        end
      end
    end
  end
{% end %}
