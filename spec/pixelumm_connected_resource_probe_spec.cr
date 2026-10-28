require "spec"
require "../examples/support/pixelumm_connected_resource_probe"

describe PixelUMMConnectedResourceProbe do
  it "admits only the seven pinned BF16 stage records and derives the production Plan byte budget" do
    PixelUMMConnectedResourceProbe::CAPTURE_NAMES.should eq([
      "attention", "post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output",
    ])
    PixelUMMConnectedResourceProbe.capture_shape("attention", 21120).should eq([1, 21120, 32, 128])
    PixelUMMConnectedResourceProbe.capture_shape("post_attn_norm", 21120).should eq([1, 21120, 4096])
    PixelUMMConnectedResourceProbe.capture_shape("gate_proj", 21120).should eq([1, 21120, 12288])
    PixelUMMConnectedResourceProbe.capture_shape("output", 21120).should eq([1, 21120, 4096])
    expect_raises(ArgumentError) { PixelUMMConnectedResourceProbe.capture_shape("silu_gate", 21120) }

    budget = PixelUMMConnectedResourceProbe.budget(21120)
    budget[:plan_bytes].should eq(4_545_944_548_i64)
    budget[:borrowed_bytes].should eq(580_534_784_i64)
    budget[:expected_bytes].should eq(2_249_195_520_i64)
    budget[:metric_bytes].should eq(128_i64)
    budget[:total_bytes].should eq(7_375_674_980_i64)
    admitted = PixelUMMConnectedResourceProbe.validate_budget!(21120, 8_i64 * PixelUMMConnectedResourceProbe::GIB)
    admitted[:total_bytes].should eq(budget[:total_bytes])
    expect_raises(ArgumentError) do
      PixelUMMConnectedResourceProbe.validate_budget!(21120, budget[:total_bytes] - 1)
    end
    PixelUMMConnectedResourceProbe.budget(2048)[:total_bytes].should be < budget[:total_bytes]
  end

  it "runs two pool-mode warmups before four balanced measured requests" do
    schedule = PixelUMMConnectedResourceProbe.schedule(shared: false,
      reverse_warmups: false, reverse_first: false)
    schedule.size.should eq(6)
    schedule.map { |entry| entry[:pool_mode] }.should eq([false, true, false, true, true, false])
    schedule.map { |entry| entry[:kind] }.should eq(["warmup", "warmup", "timed", "timed", "timed", "timed"])
    schedule.all? { |entry| !entry[:shared_gate_up] }.should be_true

    reverse_warmups = PixelUMMConnectedResourceProbe.schedule(shared: true,
      reverse_warmups: true, reverse_first: false)
    reverse_warmups.map { |entry| entry[:pool_mode] }.should eq([true, false, false, true, true, false])
    reverse_warmups.all? { |entry| entry[:shared_gate_up] }.should be_true

    reverse_first = PixelUMMConnectedResourceProbe.schedule(shared: true,
      reverse_warmups: false, reverse_first: true)
    reverse_first.map { |entry| entry[:pool_mode] }.should eq([false, true, true, false, false, true])
  end

  it "keeps plan and buffer release behind a successful sole terminal wait after partial submission" do
    lifecycle = PixelUMMConnectedResourceProbe::SubmissionLifecycle.new
    lifecycle.release_allowed?.should be_true
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 1) }
    lifecycle.begin_request!
    lifecycle.mark_attempt! # Covers the first graph encoder call, even if it partially enqueues.
    lifecycle.release_allowed?.should be_false
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: false, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(completed: true, wait_count: 2) }
    lifecycle.release_allowed?.should be_false

    lifecycle.mark_terminal_wait!(completed: true, wait_count: 1)
    lifecycle.require_readable!(completed: true, wait_count: 1).should be_nil
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: false, wait_count: 1) }
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 2) }
    lifecycle.release_allowed?.should be_true
    lifecycle.begin_request!
    expect_raises(ArgumentError) { lifecycle.require_readable!(completed: true, wait_count: 1) }
    lifecycle.mark_attempt! # Comparator/next request shares the same resident resources.
    lifecycle.release_allowed?.should be_false
    lifecycle.mark_terminal_wait!(completed: true, wait_count: 1)
    lifecycle.release_allowed?.should be_true
  end

  it "fails closed on snapshot overflow, missing graph names, and malformed pool phases" do
    expect_raises(ArgumentError) do
      PixelUMMConnectedResourceProbe.validate_snapshot_inventory!(
        event_count: 4097, overflow: false, sample_failed: false, unknown_graphs: 0)
    end
    expect_raises(ArgumentError) do
      PixelUMMConnectedResourceProbe.validate_snapshot_inventory!(
        event_count: 1, overflow: true, sample_failed: false, unknown_graphs: 0)
    end
    expect_raises(ArgumentError) do
      PixelUMMConnectedResourceProbe.validate_snapshot_inventory!(
        event_count: 2, overflow: false, sample_failed: true, unknown_graphs: 0)
    end
    expect_raises(ArgumentError) do
      PixelUMMConnectedResourceProbe.validate_snapshot_inventory!(
        event_count: 2, overflow: false, sample_failed: false, unknown_graphs: 1)
    end
    PixelUMMConnectedResourceProbe.validate_snapshot_inventory!(
      event_count: 585, overflow: false, sample_failed: false, unknown_graphs: 0).should be_nil
  end

  it "checks native graph IDs, call totals, ABI sizes, and equal three-sample cadence" do
    sizeof(PixelUMMConnectedResourceProbe::NativeEvent).should eq(48)
    sizeof(PixelUMMConnectedResourceProbe::NativeInfo).should eq(20)
    PixelUMMConnectedResourceProbe.sample_phases(false).should eq([0_u32, 1_u32, 3_u32])
    PixelUMMConnectedResourceProbe.sample_phases(true).should eq([0_u32, 1_u32, 2_u32])

    [false, true].each do |shared_gate_up|
      [false, true].each do |pool_mode|
        ids = PixelUMMConnectedResourceProbe.expected_graph_ids(16, shared_gate_up)
        ids.size.should eq(shared_gate_up ? 62 : 63)
        phases = PixelUMMConnectedResourceProbe.sample_phases(pool_mode)
        events = [] of PixelUMMConnectedResourceProbe::NativeEvent
        ids.each_with_index do |graph_id, graph_index|
          phases.each do |phase|
            event = PixelUMMConnectedResourceProbe::NativeEvent.new
            event.monotonic_ns = (events.size + 1).to_u64
            event.device_allocated_bytes = 100_u64
            event.process_phys_footprint_bytes = 200_u64
            event.process_rss_bytes = 300_u64
            event.graph_id = graph_id
            event.graph_index = graph_index.to_u32
            event.phase = phase
            event.pool_mode = pool_mode ? 1_u32 : 0_u32
            events << event
          end
        end
        info = PixelUMMConnectedResourceProbe::NativeInfo.new
        info.event_count = events.size.to_u32
        info.graph_calls = ids.size.to_u32
        PixelUMMConnectedResourceProbe.validate_native_events!(events, info,
          tile_count: 16, shared_gate_up: shared_gate_up, pool_mode: pool_mode).should be_nil

        bad_events = events.dup
        bad_last = bad_events[2]
        bad_last.phase = pool_mode ? 3_u32 : 2_u32
        bad_events[2] = bad_last
        expect_raises(ArgumentError) do
          PixelUMMConnectedResourceProbe.validate_native_events!(bad_events, info,
            tile_count: 16, shared_gate_up: shared_gate_up, pool_mode: pool_mode)
        end
      end
    end

    PixelUMMConnectedResourceProbe.expected_graph_ids(60, true).size.should eq(194)
    PixelUMMConnectedResourceProbe.expected_graph_ids(60, false).size.should eq(195)
  end

  it "requires the injected nonfinite comparator control to produce a positive GPU metric" do
    baseline = {failures: 0_u32, different_words: 3_u32, nonfinite: 0_u32, elements: 86_507_520_u32}
    detected = {failures: 1_u32, different_words: 3_u32, nonfinite: 1_u32, elements: 86_507_520_u32}
    PixelUMMConnectedResourceProbe.validate_positive_control!(baseline, detected).should be_nil
    expect_raises(ArgumentError) do
      PixelUMMConnectedResourceProbe.validate_positive_control!(baseline,
        {failures: 0_u32, different_words: 3_u32, nonfinite: 0_u32, elements: 86_507_520_u32})
    end
  end
end
