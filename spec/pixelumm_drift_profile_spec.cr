require "spec"
require "../examples/support/pixelumm_drift_profile"

private def drift_profile_markers(start_root : Int32) : Array(ML::Video::PixelUMMBlockPhaseMarker)
  names = ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES
  names.map_with_index do |name, index|
    root_count = index == names.size - 1 ? start_root + 1 : start_root
    ML::Video::PixelUMMBlockPhaseMarker.new(name, index.to_f64 * 0.01,
      index.to_f64 * 0.02, root_count)
  end.to_a
end

private def drift_profile_roots(count : Int32) : Array(ML::Video::MPSCommandRootTiming)
  Array(ML::Video::MPSCommandRootTiming).new(count) do |index|
    start_time = 2.1 + index.to_f64 * 0.8
    ML::Video::MPSCommandRootTiming.new(index, true, false, start_time,
      start_time + 0.9)
  end
end

private def drift_profile_trace(count : Int32,
                                event_phase : UInt32 = 1_u32,
                                nested : Bool = true,
                                gpu_offset : Float64 = 0.0) : ML::Video::MPSContinuationTrace
  trace_start = 2_000_000_000_u64
  events = Array(ML::Video::MPSContinuationTraceEvent).new(count - 1) do |index|
    start = trace_start + index.to_u64 * 800_000_000_u64
    ML::Video::MPSContinuationTraceEvent.new(index.to_u64 + 1,
      start + 50_000_000_u64, start + 60_000_000_u64,
      1_000_000_u64 + index.to_u64 * 1_000_u64,
      1_000_500_u64 + index.to_u64 * 1_000_u64,
      index.to_u64 + 2, nested ? index.to_u64 + 1 : 0_u64,
      index.to_u64, index.to_u64 + 1, event_phase, nested ? 1_u32 : 0_u32,
      0_u32, 2_u32, 0_u32, nested ? 15_u32 : 14_u32)
  end
  roots = Array(ML::Video::MPSContinuationTraceRoot).new(count) do |index|
    tracked = trace_start + index.to_u64 * 800_000_000_u64
    start_time = tracked.to_f64 / 1e9 + 0.1 + gpu_offset
    end_time = tracked.to_f64 / 1e9 + 1.0 + gpu_offset
    ML::Video::MPSContinuationTraceRoot.new(index.to_u64, tracked,
      tracked + 1_100_000_000_u64, start_time.unsafe_as(UInt64),
      end_time.unsafe_as(UInt64), 0_u32, 4_u32, 0_u32, 1_u32)
  end
  info = ML::Video::MPSContinuationTraceInfo.new(42_u64, 1_u64, 1_u64,
    trace_start, ML::Video::MPSContinuationTrace::CAPACITY,
    events.size.to_u64, roots.size.to_u64, 0_u64)
  ML::Video::MPSContinuationTrace.build(info, events, roots)
end

private def drift_profile_api(commit_phase : Int32 = 1,
                              attention_graph_count : UInt64 = 12_u64) : ML::Video::MPSAPIProfile
  graph_counts = {0 => 0_u64, 1 => 36_u64, 2 => attention_graph_count,
                  3 => 12_u64, 4 => 0_u64, 5 => 12_u64}
  commit_counts = {} of Int32 => UInt64
  commit_counts[commit_phase] = 5_u64
  samples = [] of ML::Video::MPSAPIProfileSample
  ML::Video::MPSAPIProfile::PHASE_NAMES.keys.sort.each do |phase_id|
    graph_count = graph_counts[phase_id]
    commit_count = commit_counts[phase_id]? || 0_u64
    graph_wall = graph_count * 30_000_000_u64
    graph_cpu = graph_count * 10_000_u64
    commit_wall = commit_count * 20_000_000_u64
    commit_cpu = commit_count * 500_u64
    graph = if graph_count == 0
              drift_profile_stats(0_u64, 0_u64, 0)
            else
              ML::Video::MPSAPITimingStats.new(graph_count, graph_wall,
                30_000_000_u64, graph_cpu, 10_000_u64, commit_count,
                commit_wall, commit_cpu, graph_wall - commit_wall,
                graph_cpu - commit_cpu)
            end
    commit = if commit_count == 0
               drift_profile_stats(0_u64, 0_u64, 1)
             else
               ML::Video::MPSAPITimingStats.new(commit_count, commit_wall,
                 20_000_000_u64, commit_cpu, 500_u64, commit_count,
                 commit_wall, commit_cpu, commit_wall, commit_cpu)
             end
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 0, graph)
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 1, commit)
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 2,
      drift_profile_stats(0_u64, 0_u64, 2))
  end
  ML::Video::MPSAPIProfile.build(samples)
end

private def drift_profile_stats(count : UInt64, nested_count : UInt64,
                                operation : Int32) : ML::Video::MPSAPITimingStats
  if count == 0
    return ML::Video::MPSAPITimingStats.new(0_u64, 0_u64, 0_u64,
      0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64)
  end
  wall = count * 20_000_000_u64
  cpu = count * 500_u64
  ML::Video::MPSAPITimingStats.new(count, wall, 20_000_000_u64,
    cpu, 500_u64, nested_count,
    operation == 0 ? nested_count * 20_000_000_u64 : wall,
    operation == 0 ? nested_count * 500_u64 : cpu,
    operation == 0 ? wall - nested_count * 20_000_000_u64 : wall,
    operation == 0 ? cpu - nested_count * 500_u64 : cpu)
end

private def drift_profile_inputs(commit_phase : Int32 = 1)
  roots = drift_profile_roots(6)
  markers = [1, 2, 3, 4].map { |start| drift_profile_markers(start) }
  api = drift_profile_api(commit_phase)
  trace = drift_profile_trace(roots.size)
  {markers, roots, api, trace}
end

private def build_drift_profile(markers, roots, api, trace)
  PixelUMMDriftProfile::Profile.new(markers, roots,
    roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds },
    api, trace)
end

describe PixelUMMDriftProfile::Profile do
  it "preserves adjacent-plan root overlap and labels final comparator roots" do
    markers, roots, api, trace = drift_profile_inputs
    profile = build_drift_profile(markers, roots, api, trace)

    profile.phase_wall_seconds.size.should eq(40)
    profile.phase_thread_cpu_seconds.size.should eq(40)
    profile.phase_wall_seconds["window1/block1/final_residual"].should be_close(0.01, 1e-12)
    profile.phase_definite_root_duration_sum_seconds["window1/block1/final_residual"]
      .should be_close(0.9, 1e-12)
    profile.root_candidates[1].should contain("window0/block0/final_residual")
    profile.root_candidates[1].should contain("window0/block1/input_norm")
    profile.root_candidates[0].should contain("request_prelude")
    profile.root_candidates[4].should contain("request_final_comparators")
    profile.root_candidates[5].should eq(["request_final_comparators"])
    profile.root_candidates.all?(&.any?).should be_true
    profile.root_profile.root_count.should eq(6)
    profile.root_profile.phase_union_seconds.keys.should eq(["whole_chain"])

    json = JSON.parse(profile.to_json)
    json["markers_by_plan"].as_a.size.should eq(4)
    json["markers_by_plan"][2]["plan"].as_s.should eq("window1/block0")
    json["phase_thread_cpu_seconds"].as_h.size.should eq(40)
    json["roots"][1]["candidates"].as_a.map(&.as_s)
      .should contain("window0/block1/input_norm")
    json["continuation_trace"]["events"].as_a.size.should eq(5)
    json["api_profile"]["phases"].as_a.size.should eq(6)
    json["gpu_phase_interpretation"].as_s.should contain("exclusive GPU work")
  end

  it "rejects malformed plan counts, marker order, clocks, and global root cursors" do
    markers, roots, api, trace = drift_profile_inputs
    duration = roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }

    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(markers[0...3], roots, duration, api, trace)
    end

    invalid = markers.map(&.dup)
    invalid[0][1] = ML::Video::PixelUMMBlockPhaseMarker.new("qk_norm_rope",
      0.01, 0.02, 1)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(invalid, roots, duration, api, trace)
    end

    invalid = markers.map(&.dup)
    invalid[0][0] = ML::Video::PixelUMMBlockPhaseMarker.new("input_norm",
      0.01, 0.0, 1)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(invalid, roots, duration, api, trace)
    end

    invalid = markers.map(&.dup)
    invalid[1][0] = ML::Video::PixelUMMBlockPhaseMarker.new("input_norm",
      0.0, 0.0, 3)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(invalid, roots, duration, api, trace)
    end

    invalid = markers.map(&.dup)
    invalid[3][10] = ML::Video::PixelUMMBlockPhaseMarker.new("complete",
      0.1, 0.2, roots.size + 1)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(invalid, roots, duration, api, trace)
    end
  end

  it "rejects API or continuation nesting that does not describe all four plans" do
    markers, roots, api, trace = drift_profile_inputs(commit_phase: 2)
    duration = roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(markers, roots, duration, api, trace)
    end

    markers, roots, api, trace = drift_profile_inputs
    invalid_api = drift_profile_api(attention_graph_count: 11_u64)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(markers, roots, duration, invalid_api, trace)
    end

    markers, roots, api, trace = drift_profile_inputs
    non_nested_trace = drift_profile_trace(roots.size, nested: false)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(markers, roots, duration, api, non_nested_trace)
    end

    shifted_trace = drift_profile_trace(roots.size, gpu_offset: 0.001)
    expect_raises(ArgumentError) do
      PixelUMMDriftProfile::Profile.new(markers, roots, duration, api, shifted_trace)
    end
  end
end
