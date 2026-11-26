require "spec"
require "../examples/support/drama3_gdn_scan"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private alias Drama3GdnScanSpecContext = ML::Audio::Drama3::Resident::Context
  private alias Drama3GdnScanSpecBuffer = ML::Audio::Drama3::Resident::Buffer
  private alias Drama3GdnScanSpecInputs = Drama3GDNScan::Inputs

  private def drama3_gdn_scan_spec_with_device(depth : String = "0", &block : -> T) : T forall T
    keys = ["DRAMA3_BACKEND", "DRAMA3_DENSE_REDUCTION", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
            "DRAMA3_METAL_TIMELINE", "DRAMA3_METAL_RESOURCES", "DRAMA3_PROFILE"]
    previous = keys.to_h { |key| {key, ENV[key]?} }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
      ENV["DRAMA3_METAL_TIMELINE"] = "0"
      ENV["DRAMA3_METAL_RESOURCES"] = "0"
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

  private def drama3_gdn_scan_spec_upload_view(context : Drama3GdnScanSpecContext,
                                               values : Array(Float32), pad : Int32) : Drama3GdnScanSpecBuffer
    context.upload(Array(Float32).new(pad, -173.25_f32) + values + [291.5_f32]).slice(pad, values.size)
  end

  private def drama3_gdn_scan_spec_values(count : Int32, scale : Float32, salt : Int32) : Array(Float32)
    Array(Float32).new(count) do |index|
      (((index * 17 + salt * 11) % 37) - 18).to_f32 * scale
    end
  end

  private def drama3_gdn_scan_spec_inputs(context : Drama3GdnScanSpecContext,
                                          rows : Int32, value_heads : Int32,
                                          value_dim : Int32, key_heads : Int32,
                                          key_dim : Int32, zero_start : Bool = false)
    q = drama3_gdn_scan_spec_upload_view(context,
      drama3_gdn_scan_spec_values(rows * key_heads * key_dim, 0.03125_f32, rows + 1), 1)
    k = drama3_gdn_scan_spec_upload_view(context,
      drama3_gdn_scan_spec_values(rows * key_heads * key_dim, 0.015625_f32, key_dim + 2), 2)
    v = drama3_gdn_scan_spec_upload_view(context,
      drama3_gdn_scan_spec_values(rows * value_heads * value_dim, 0.0625_f32, value_dim + 3), 3)
    a = drama3_gdn_scan_spec_upload_view(context,
      drama3_gdn_scan_spec_values(rows * value_heads, 0.03125_f32, value_heads + 4), 1)
    b = drama3_gdn_scan_spec_upload_view(context,
      drama3_gdn_scan_spec_values(rows * value_heads, 0.0625_f32, rows + 5), 2)
    alog_values = drama3_gdn_scan_spec_values(value_heads, 0.015625_f32, key_heads + 6)
      .map { |value| value - 1.25_f32 }
    dt_values = drama3_gdn_scan_spec_values(value_heads, 0.015625_f32, value_dim + 7)
      .map { |value| value + 0.1875_f32 }
    alog = drama3_gdn_scan_spec_upload_view(context, alog_values, 3)
    dt = drama3_gdn_scan_spec_upload_view(context, dt_values, 1)
    initial_values = if zero_start
                       Array(Float32).new(value_heads * key_dim * value_dim, 0_f32)
                     else
                       drama3_gdn_scan_spec_values(value_heads * key_dim * value_dim, 0.0078125_f32, rows + key_dim + 8)
                     end
    initial = drama3_gdn_scan_spec_upload_view(context, initial_values, 2)
    buffers = [q, k, v, a, b, alog, dt, initial]
    inputs = Drama3GdnScanSpecInputs.new(q, k, v, a, b, alog, dt, initial,
      rows, value_heads, value_dim, key_heads, key_dim)
    {inputs: inputs, buffers: buffers}
  end

  private def drama3_gdn_scan_spec_read_metrics(context : Drama3GdnScanSpecContext,
                                                comparisons : Array(Tuple(String, Drama3GdnScanSpecBuffer, Drama3GdnScanSpecBuffer)))
    comparisons.map do |comparison|
      Drama3GDNScan.compare(context, comparison[1], comparison[2])
    end
  end

  private def drama3_gdn_scan_spec_assert_exact_metrics(metrics : Array(Array(Float32))) : Nil
    metrics.each do |values|
      values.size.should eq(5)
      values[0].should eq(0_f32) # finite bit mismatches
      values[1].should eq(0_f32) # tolerance failures
      values[2].should eq(0_f32) # maximum absolute error
      values[4].should eq(0_f32) # non-finite values
    end
  end

  private def drama3_gdn_scan_spec_run_case(depth : String, rows : Int32, key_dim : Int32,
                                            value_dim : Int32, block_size : Int32,
                                            zero_start : Bool) : Nil
    drama3_gdn_scan_spec_with_device(depth) do
      context = Drama3GdnScanSpecContext.new
      begin
        value_heads = 2
        key_heads = 1
        prepared = drama3_gdn_scan_spec_inputs(context, rows, value_heads, value_dim,
          key_heads, key_dim, zero_start)
        inputs = prepared[:inputs]
        prepared[:buffers].each { |buffer| buffer.offset.should be > 0_i64 }
        inputs.key_width.should eq(key_heads * key_dim)
        inputs.value_width.should eq(value_heads * value_dim)
        inputs.state_count.should eq(value_heads * key_dim * value_dim)
        plan = Drama3GDNScan::Plan.new(context, inputs, block_size)

        before_encoding_graph = context.graph_stats.dup
        before_encoding_readbacks = context.readbacks.dup
        before_invalid_mode_allocations = context.allocation_stats[:allocations]
        before_invalid_mode_graph = context.graph_stats.dup
        expect_raises(ArgumentError) { plan.encode("not-a-scan-mode") }
        context.allocation_stats[:allocations].should eq(before_invalid_mode_allocations)
        context.graph_stats["completion_waits"].should eq(before_invalid_mode_graph["completion_waits"])
        context.graph_stats["submitted"].should eq(before_invalid_mode_graph["submitted"])

        ["sequential", "serial_tile", "affine_scan"].each { |mode| plan.encode(mode) }
        comparisons = [] of Tuple(String, Drama3GdnScanSpecBuffer, Drama3GdnScanSpecBuffer)
        comparisons << {"sequential_serial_output", plan.output("sequential"), plan.output("serial_tile")}
        comparisons << {"sequential_serial_state", plan.state("sequential"), plan.state("serial_tile")}
        comparisons << {"sequential_affine_output", plan.output("sequential"), plan.output("affine_scan")}
        comparisons << {"sequential_affine_state", plan.state("sequential"), plan.state("affine_scan")}
        receipts = drama3_gdn_scan_spec_read_metrics(context, comparisons)
        context.graph_stats["completion_waits"].should eq(before_encoding_graph["completion_waits"])
        context.readbacks.should eq(before_encoding_readbacks)

        metrics = receipts.map { |receipt| context.read(receipt, "diagnostic").to_a }
        drama3_gdn_scan_spec_assert_exact_metrics([metrics[0], metrics[1]])

        # Affine composition changes floating-point association. Keep this a
        # diagnostic receipt; the native-bitwise promotion gate remains closed.
        [metrics[2], metrics[3]].each do |values|
          values.size.should eq(5)
          values.each { |value| value.finite?.should be_true }
          values[0].should be >= 0_f32
          values[1].should be >= 0_f32
          values[2].should be >= 0_f32
          values[4].should be >= 0_f32
        end
      ensure
        context.abort
      end
    end
  end

  describe "DRAMA-3 resident GDN scan diagnostics" do
    it "matches sequential output and final state bitwise in tiny and production-width cases" do
      # rows=9, block_size=4 exercises a partial tail block.
      drama3_gdn_scan_spec_run_case("0", 1, 3, 2, 1, false)
      drama3_gdn_scan_spec_run_case("2", 3, 5, 4, 2, true)
      drama3_gdn_scan_spec_run_case("0", 9, 7, 3, 4, false)
      drama3_gdn_scan_spec_run_case("2", 9, 128, 128, 4, false)
    end

    it "keeps the all-zero k=0 recurrence exactly degenerate through the affine scan" do
      drama3_gdn_scan_spec_with_device("2") do
        context = Drama3GdnScanSpecContext.new
        begin
          rows = 3
          value_heads = 2
          value_dim = 4
          key_heads = 1
          key_dim = 5
          q = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(rows * key_heads * key_dim, 0_f32), 1)
          k = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(rows * key_heads * key_dim, 0_f32), 2)
          v = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(rows * value_heads * value_dim, 0_f32), 3)
          a = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(rows * value_heads, 0_f32), 1)
          b = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(rows * value_heads, 0_f32), 2)
          alog = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(value_heads, 0_f32), 3)
          dt = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(value_heads, 0_f32), 1)
          initial = drama3_gdn_scan_spec_upload_view(context, Array(Float32).new(value_heads * key_dim * value_dim, 0_f32), 2)
          inputs = Drama3GdnScanSpecInputs.new(q, k, v, a, b, alog, dt, initial,
            rows, value_heads, value_dim, key_heads, key_dim)
          plan = Drama3GDNScan::Plan.new(context, inputs, 2)
          ["sequential", "affine_scan"].each { |mode| plan.encode(mode) }
          comparisons = [
            {"zero_output", plan.output("sequential"), plan.output("affine_scan")},
            {"zero_state", plan.state("sequential"), plan.state("affine_scan")},
          ] of Tuple(String, Drama3GdnScanSpecBuffer, Drama3GdnScanSpecBuffer)
          receipts = drama3_gdn_scan_spec_read_metrics(context, comparisons)
          metrics = receipts.map { |receipt| context.read(receipt, "diagnostic").to_a }
          drama3_gdn_scan_spec_assert_exact_metrics(metrics)
        ensure
          context.abort
        end
      end
    end

    it "rejects bad shapes, owners, block sizes, and modes before encoding work" do
      drama3_gdn_scan_spec_with_device("0") do
        context = Drama3GdnScanSpecContext.new
        other = Drama3GdnScanSpecContext.new
        begin
          prepared = drama3_gdn_scan_spec_inputs(context, 3, 2, 4, 1, 5)
          inputs = prepared[:inputs]
          buffers = prepared[:buffers]
          before_invalid_allocations = context.allocation_stats[:allocations]
          before_invalid_graph = context.graph_stats.dup

          expect_raises(ArgumentError) do
            short_q = drama3_gdn_scan_spec_upload_view(context,
              drama3_gdn_scan_spec_values(3 * 1 * 5 - 1, 0.03125_f32, 99), 1)
            malformed = Drama3GdnScanSpecInputs.new(short_q, buffers[1], buffers[2], buffers[3], buffers[4],
              buffers[5], buffers[6], buffers[7], 3, 2, 4, 1, 5)
            Drama3GDNScan::Plan.new(context, malformed, 2)
          end
          # Uploading the malformed view itself is the only permitted allocation.
          context.allocation_stats[:allocations].should eq(before_invalid_allocations + 1_i64)
          after_bad_shape = context.allocation_stats[:allocations]

          expect_raises(ArgumentError) { Drama3GDNScan::Plan.new(context, inputs, 0) }
          expect_raises(ArgumentError) { Drama3GDNScan::Plan.new(context, inputs, 4) }
          context.allocation_stats[:allocations].should eq(after_bad_shape)

          expect_raises(ArgumentError) do
            foreign_q = drama3_gdn_scan_spec_upload_view(other,
              drama3_gdn_scan_spec_values(3 * 1 * 5, 0.03125_f32, 101), 1)
            mixed = Drama3GdnScanSpecInputs.new(foreign_q, buffers[1], buffers[2], buffers[3], buffers[4],
              buffers[5], buffers[6], buffers[7], 3, 2, 4, 1, 5)
            Drama3GDNScan::Plan.new(context, mixed, 2)
          end
          context.allocation_stats[:allocations].should eq(after_bad_shape)
          context.graph_stats["completion_waits"].should eq(before_invalid_graph["completion_waits"])
          context.graph_stats["submitted"].should eq(before_invalid_graph["submitted"])
          context.readbacks.should be_empty
        ensure
          context.abort
          other.abort
        end
      end
    end

    it "detects one-ULP, large finite, and non-finite comparator defects" do
      drama3_gdn_scan_spec_with_device("0") do
        context = Drama3GdnScanSpecContext.new
        begin
          one = 1_f32
          one_ulp_up = (one.unsafe_as(UInt32) + 1_u32).unsafe_as(Float32)
          one_ref = context.upload([one])
          one_actual = context.upload([one_ulp_up])
          large_ref = context.upload([1_f32])
          large_actual = context.upload([2_f32])
          finite_ref = context.upload([0_f32, 1_f32, -1_f32])
          nonfinite_actual = context.upload([Float32::NAN, Float32::INFINITY, -Float32::INFINITY])
          checks = [
            Drama3GDNScan.compare(context, one_actual, one_ref),
            Drama3GDNScan.compare(context, large_actual, large_ref),
            Drama3GDNScan.compare(context, nonfinite_actual, finite_ref),
          ]
          results = checks.map { |check| context.read(check, "diagnostic").to_a }
          results.each { |values| values.size.should eq(5) }
          results[0][0].should be > 0_f32
          results[0][1].should eq(0_f32) # one ULP is within 6e-5 + 3e-4*|reference|
          results[0][4].should eq(0_f32)
          results[1][0].should be > 0_f32
          results[1][1].should be > 0_f32
          results[1][2].should be >= 1_f32
          results[2][4].should be > 0_f32
        ensure
          context.abort
        end
      end
    end
  end
{% end %}
