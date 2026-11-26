require "spec"
require "../src/ml/video/mps_continuation_trace"

private def continuation_info(count = 1_u64, roots = 2_u64,
                              denominator = 1_u64, failed = 0_u64)
  ML::Video::MPSContinuationTraceInfo.new(42_u64, 1_u64, denominator.to_u64,
    1_000_000_000_u64, ML::Video::MPSContinuationTrace::CAPACITY, count.to_u64, roots.to_u64, failed.to_u64)
end

private def continuation_event(begin_ticks = 1_100_000_000_u64,
                               end_ticks = 1_300_000_000_u64,
                               before = 0_u64, after = 1_u64,
                               flags = 14_u32, parent = 0_u64,
                               depth = 0_u32)
  ML::Video::MPSContinuationTraceEvent.new(1_u64, begin_ticks, end_ticks,
    100_u64, 110_u64, 2_u64, parent, before, after,
    4_u32, depth, 0_u32, 2_u32, 0_u32, flags)
end

private def continuation_roots(first_end = 1.4, source = 1_u32)
  [
    ML::Video::MPSContinuationTraceRoot.new(0_u64, 1_000_000_000_u64,
      1_410_000_000_u64, 1.2.unsafe_as(UInt64), first_end.unsafe_as(UInt64),
      0_u32, 4_u32, 0_u32, source),
    ML::Video::MPSContinuationTraceRoot.new(1_u64, 1_300_000_000_u64,
      1_510_000_000_u64, 1.4.unsafe_as(UInt64), 1.5.unsafe_as(UInt64),
      0_u32, 4_u32, 0_u32, 1_u32),
  ]
end

private def continuation_profile(wall_ns : UInt64)
  zero = ML::Video::MPSAPITimingStats.new(0_u64, 0_u64, 0_u64, 0_u64, 0_u64,
    0_u64, 0_u64, 0_u64, 0_u64, 0_u64)
  commit = ML::Video::MPSAPITimingStats.new(1_u64, wall_ns, wall_ns, 10_u64, 10_u64,
    0_u64, 0_u64, 0_u64, wall_ns, 10_u64)
  samples = [] of ML::Video::MPSAPIProfileSample
  ML::Video::MPSAPIProfile::PHASE_NAMES.each_key do |phase|
    ML::Video::MPSAPIProfile::OPERATION_NAMES.each_key do |operation|
      samples << ML::Video::MPSAPIProfileSample.new(phase, operation,
        phase == 4 && operation == 1 ? commit : zero)
    end
  end
  ML::Video::MPSAPIProfile.build(samples)
end

describe ML::Video::MPSContinuationTrace do
  it "retains a full PixelUMM chain above the former 4096-event limit" do
    count = 4578
    events = Array(ML::Video::MPSContinuationTraceEvent).new(count) do |index|
      start = 1_100_000_000_u64 + index.to_u64 * 1_000_000_u64
      cpu = 1_000_000_u64 + index.to_u64 * 1000_u64
      ML::Video::MPSContinuationTraceEvent.new(index.to_u64 + 1, start, start + 100_000,
        cpu, cpu + 200, index.to_u64 + 1, 0_u64, index.to_u64, index.to_u64 + 1,
        4_u32, 0_u32, 0_u32, 2_u32, 0_u32, 14_u32)
    end
    roots = Array(ML::Video::MPSContinuationTraceRoot).new(count + 1) do |index|
      start = 1_100_200_000_u64 + index.to_u64 * 1_000_000_u64
      tracked = index == 0 ? 1_000_000_000_u64 : events[index - 1].end_mach_ticks
      ML::Video::MPSContinuationTraceRoot.new(index.to_u64, tracked, start + 600_000,
        (start.to_f64 / 1e9).unsafe_as(UInt64), ((start + 500_000).to_f64 / 1e9).unsafe_as(UInt64),
        0_u32, 4_u32, 0_u32, 1_u32)
    end
    trace = ML::Video::MPSContinuationTrace.build(continuation_info(count, count + 1), events, roots)
    trace.events.size.should eq(count)
    stale = ML::Video::MPSContinuationTraceInfo.new(42_u64, 1_u64, 1_u64,
      1_000_000_000_u64, 4096_u64, count.to_u64, (count + 1).to_u64, 0_u64)
    expect_raises(ArgumentError) { ML::Video::MPSContinuationTrace.build(stale, events, roots) }
  end

  it "accepts a complete trace and preserves separate root and callback timestamps" do
    trace = ML::Video::MPSContinuationTrace.build(continuation_info,
      [continuation_event], continuation_roots)
    trace.events.size.should eq(1)
    trace.roots[0].gpu_end_seconds.should eq(1.4)
    trace.host_seconds(1_300_000_000_u64).should eq(1.3)
  end

  it "rejects missing, overflowed, failed, or invalid-clock traces" do
    [continuation_info(2), continuation_info(ML::Video::MPSContinuationTrace::CAPACITY + 1),
     continuation_info(1, 3), continuation_info(1, 2, 0),
     continuation_info(1, 2, 1, 1)].each do |info|
      expect_raises(ArgumentError) do
        ML::Video::MPSContinuationTrace.build(info,
          [continuation_event], continuation_roots)
      end
    end
  end

  it "rejects backwards, overlapping, or orphaned continuation events" do
    [continuation_event(1_300_000_000_u64, 1_100_000_000_u64),
     continuation_event(1_100_000_000_u64, 1_100_000_000_u64),
     continuation_event(before: 1_u64), continuation_event(after: 2_u64)].each do |event|
      expect_raises(ArgumentError) do
        ML::Video::MPSContinuationTrace.build(continuation_info,
          [event], continuation_roots)
      end
    end
  end

  it "rejects missing validity flags, unknown bits, and inconsistent graph parents" do
    [continuation_event(flags: 0_u32), continuation_event(flags: 30_u32),
     continuation_event(parent: 1_u64), continuation_event(depth: 1_u32),
     continuation_event(flags: 15_u32)].each do |event|
      expect_raises(ArgumentError) do
        ML::Video::MPSContinuationTrace.build(continuation_info,
          [event], continuation_roots)
      end
    end
  end

  it "rejects a next-root observation before continuation returned" do
    roots = continuation_roots
    root = roots[1]
    roots[1] = ML::Video::MPSContinuationTraceRoot.new(root.sequence,
      1_200_000_000_u64, root.completion_observed_mach_ticks,
      root.gpu_start_time_bits, root.gpu_end_time_bits,
      root.status_at_track_or_enable, root.final_status,
      root.has_error, root.completion_source)
    expect_raises(ArgumentError) do
      ML::Video::MPSContinuationTrace.build(continuation_info,
        [continuation_event], roots)
    end
  end

  it "rejects incompatible GPU clocks, nonfinite times, and absent completion evidence" do
    [continuation_roots(2.0), continuation_roots(Float64::NAN),
     continuation_roots(1.4, 0_u32)].each do |roots|
      expect_raises(ArgumentError) do
        ML::Video::MPSContinuationTrace.build(continuation_info,
          [continuation_event], roots)
      end
    end
  end

  it "bounds independent aggregate clock rounding without accepting a larger timer mismatch" do
    trace = ML::Video::MPSContinuationTrace.build(continuation_info,
      [continuation_event], continuation_roots)
    trace.validate_api_profile!(continuation_profile(199_999_000_u64))
    expect_raises(ArgumentError) do
      trace.validate_api_profile!(continuation_profile(199_997_000_u64))
    end
  end
end
