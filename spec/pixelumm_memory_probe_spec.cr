require "spec"
require "json"
require "../examples/support/pixelumm_two_block_probe"

private class FailingPixelUMMMemoryProbeIO < IO
  def read(slice : Bytes) : Int32
    0
  end

  def write(slice : Bytes) : Nil
    raise IO::Error.new("injected telemetry output failure")
  end
end

describe PixelUMMMemoryProbe do
  it "matches the macOS rusage_info_v2 size and sampled field offsets" do
    sizeof(PixelUMMMemoryProbe::RUsageInfoV2).should eq(160)
    offsetof(PixelUMMMemoryProbe::RUsageInfoV2, @ri_resident_size).should eq(64)
    offsetof(PixelUMMMemoryProbe::RUsageInfoV2, @ri_phys_footprint).should eq(72)
  end

  it "emits monotonic JSONL events with explicitly non-additive memory categories" do
    samples = [
      PixelUMMMemoryProbe::Readings.new(
        process_phys_footprint_bytes: 1_000,
        process_rss_bytes: 900,
        metal_device_allocated_bytes: 500,
        named_metal_buffer_live_count: 2,
        named_metal_buffer_live_bytes: 400,
        named_metal_buffer_peak_bytes: 450,
        crystal_gc_heap_bytes: 300,
        crystal_gc_free_bytes: 100,
        crystal_gc_unmapped_bytes: 20,
        crystal_gc_total_bytes: 700,
      ),
      PixelUMMMemoryProbe::Readings.new(
        process_phys_footprint_bytes: 1_200,
        process_rss_bytes: 1_100,
        metal_device_allocated_bytes: 650,
        named_metal_buffer_live_count: 3,
        named_metal_buffer_live_bytes: 550,
        named_metal_buffer_peak_bytes: 550,
        crystal_gc_heap_bytes: 350,
        crystal_gc_free_bytes: 110,
        crystal_gc_unmapped_bytes: 20,
        crystal_gc_total_bytes: 900,
      ),
    ]
    timestamps = [10_i64, 20_i64]
    expected_guard_timestamps = [7_346_455_510_i64, 7_346_455_520_i64]
    guard_timestamps = expected_guard_timestamps.dup
    stderr = IO::Memory.new
    recorder = PixelUMMMemoryProbe::Recorder.new(stderr,
      -> { samples.shift }, -> { timestamps.shift }, ->{ guard_timestamps.shift? })

    recorder.emit("before_metadata")
    recorder.emit("after_weight_slot_upload", slot: 1, request: 2,
      phase_bytes_kind: "weight_slot_uploaded_bytes", phase_bytes: 385_892_864_i64)

    lines = stderr.to_s.lines
    lines.size.should eq(2)
    lines.all?(&.starts_with?(PixelUMMMemoryProbe::PREFIX)).should be_true
    output = lines.map { |line| JSON.parse(line[PixelUMMMemoryProbe::PREFIX.size..]) }
    output.map { |event| event["monotonic_ns"].as_i64 }.should eq([10_i64, 20_i64])
    output.map { |event| event["guard_monotonic_ns"].as_i64 }.should eq(expected_guard_timestamps)
    output[0]["timestamp_clock"].as_s.should eq("CLOCK_MONOTONIC")
    output[0]["guard_timestamp_clock"].as_s.should eq("mach_absolute_time")
    output[0]["monotonic_ns"].as_i64.should_not eq(output[0]["guard_monotonic_ns"].as_i64)
    recorder.events.size.should eq(2)
    event = recorder.events[1]
    event["phase"].as_s.should eq("after_weight_slot_upload")
    event["slot"].as_i.should eq(1)
    event["request"].as_i.should eq(2)
    event["phase_bytes_kind"].as_s.should eq("weight_slot_uploaded_bytes")
    event["phase_bytes"].as_i64.should eq(385_892_864_i64)
    event["process_phys_footprint_bytes"].as_i64.should eq(1_200_i64)
    event["process_rss_bytes"].as_i64.should eq(1_100_i64)
    event["metal_device_allocated_bytes"].as_i64.should eq(650_i64)
    event["named_metal_buffer_live_bytes"].as_i64.should eq(550_i64)
    event["crystal_gc_heap_bytes"].as_i64.should eq(350_i64)
    event["memory_categories_additive"].as_bool.should be_false
    event["memory_category_notes"].as_s.should contain("overlap")
    event.as_h.has_key?("opaque_mps_bytes").should be_false
  end

  it "renders invalid scalar readings as null instead of inventing byte counts" do
    samples = [PixelUMMMemoryProbe::Readings.new(
      process_phys_footprint_bytes: -1_i64,
      process_rss_bytes: 0_i64,
      metal_device_allocated_bytes: -4_i64,
      named_metal_buffer_live_count: -1_i32,
      named_metal_buffer_live_bytes: -10_i64,
      named_metal_buffer_peak_bytes: -20_i64,
      crystal_gc_heap_bytes: -30_i64,
      crystal_gc_free_bytes: -40_i64,
      crystal_gc_unmapped_bytes: -50_i64,
      crystal_gc_total_bytes: -60_i64,
    )]
    stderr = IO::Memory.new
    recorder = PixelUMMMemoryProbe::Recorder.new(stderr,
      -> { samples.shift }, -> { 1_i64 }, ->{ nil.as(Int64?) })

    event = recorder.emit("invalid_reading")
    {"process_phys_footprint_bytes", "process_rss_bytes", "metal_device_allocated_bytes",
     "named_metal_buffer_live_count", "named_metal_buffer_live_bytes", "named_metal_buffer_peak_bytes",
     "crystal_gc_heap_bytes", "crystal_gc_free_bytes", "crystal_gc_unmapped_bytes", "crystal_gc_total_bytes"}.each do |key|
      event[key].raw.should be_nil
    end
  end

  it "rejects a backwards monotonic clock without emitting a duplicate or reordered event" do
    samples = [PixelUMMMemoryProbe::Readings.new, PixelUMMMemoryProbe::Readings.new]
    timestamps = [100_i64, 99_i64]
    guard_timestamps = [200_i64, 199_i64]
    stderr = IO::Memory.new
    recorder = PixelUMMMemoryProbe::Recorder.new(stderr,
      -> { samples.shift }, -> { timestamps.shift }, ->{ guard_timestamps.shift? })

    recorder.emit("first")
    expect_raises(ArgumentError, /monotonic/) { recorder.emit("backwards") }
    recorder.events.size.should eq(1)
    stderr.to_s.lines.size.should eq(1)
  end

  it "rejects a backwards guard clock independently of CLOCK_MONOTONIC" do
    samples = [PixelUMMMemoryProbe::Readings.new, PixelUMMMemoryProbe::Readings.new]
    timestamps = [100_i64, 101_i64]
    guard_timestamps = [200_i64, 199_i64]
    stderr = IO::Memory.new
    recorder = PixelUMMMemoryProbe::Recorder.new(stderr,
      -> { samples.shift }, -> { timestamps.shift }, ->{ guard_timestamps.shift? })

    recorder.emit("first")
    expect_raises(ArgumentError, /guard clock.*monotonic/) { recorder.emit("guard_backwards") }
    recorder.events.size.should eq(1)
    stderr.to_s.lines.size.should eq(1)
  end
end

describe PixelUMMMemoryProbe::Cleanup do
  it "attempts every release across sampler and IO failures, then rethrows the first error" do
    released = [] of String
    failing_sample = PixelUMMMemoryProbe::Recorder.new(IO::Memory.new,
      -> { raise IO::Error.new("injected telemetry sample failure"); PixelUMMMemoryProbe::Readings.new },
      -> { 10_i64 }, ->{ 20_i64.as(Int64?) })
    failing_output = PixelUMMMemoryProbe::Recorder.new(FailingPixelUMMMemoryProbeIO.new,
      -> { PixelUMMMemoryProbe::Readings.new }, -> { 11_i64 }, ->{ 21_i64.as(Int64?) })
    cleanup = PixelUMMMemoryProbe::Cleanup.new

    cleanup.phase(failing_sample, "before_plan_release")
    cleanup.attempt { released << "plan" }
    cleanup.phase(failing_output, "after_plan_release")
    cleanup.attempt { released << "workspace" }
    cleanup.attempt { released << "bindings" }

    released.should eq(["plan", "workspace", "bindings"])
    cleanup.errors.size.should eq(2)
    expect_raises(IO::Error, /sample failure/) { cleanup.raise_first! }
  end
end

describe PixelUMMTwoBlockProbe do
  it "selects the exact 402 common ring captures from a full sixteen block inventory" do
    keys = (0...16).flat_map do |slot|
      PixelUMMTwoBlockProbe::Decoder::Plan::CAPTURE_NAMES.to_a.map do |stage|
        "block#{slot}.#{stage}"
      end
    end

    selected = PixelUMMTwoBlockProbe.common_ring_comparison_capture_order(keys, [2, 32, 32], 16)
    selected.size.should eq(402)
    selected.should eq(keys.select do |key|
      slot = PixelUMMTwoBlockProbe.capture_slot(key, 16)
      stage = PixelUMMTwoBlockProbe.capture_stage(key, 16)
      PixelUMMTwoBlockProbe::Decoder::Workspace.capture_survives?(stage, slot, 16, reuse_outputs: true)
    end)
  end

  it "selects the exact 18 common ring captures from the small sixteen block inventory" do
    keys = (0...16).flat_map do |slot|
      ["attention", "output"].map { |stage| "block#{slot}.#{stage}" }
    end

    selected = PixelUMMTwoBlockProbe.common_ring_comparison_capture_order(keys, [6, 44, 80], 16)
    selected.size.should eq(18)
    selected.should eq(keys.select do |key|
      slot = PixelUMMTwoBlockProbe.capture_slot(key, 16)
      stage = PixelUMMTwoBlockProbe.capture_stage(key, 16)
      PixelUMMTwoBlockProbe::Decoder::Workspace.capture_survives?(stage, slot, 16, reuse_outputs: true)
    end)
  end
end
