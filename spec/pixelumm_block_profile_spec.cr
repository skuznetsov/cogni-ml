require "spec"
require "../src/ml/video/pixelumm_block_profile"

private def pixelumm_profile_markers(root_counts : Array(Int32),
                                     wall_step : Float64 = 0.1,
                                     cpu_step : Float64 = 0.2) : Array(ML::Video::PixelUMMBlockPhaseMarker)
  ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES.map_with_index do |name, index|
    ML::Video::PixelUMMBlockPhaseMarker.new(name, index.to_f64 * wall_step,
      index.to_f64 * cpu_step, root_counts[index])
  end.to_a
end

private def pixelumm_profile_roots(count : Int32) : Array(ML::Video::MPSCommandRootTiming)
  Array(ML::Video::MPSCommandRootTiming).new(count) do |index|
    start_time = index.to_f64 * 0.5
    ML::Video::MPSCommandRootTiming.new(index, true, false,
      start_time, start_time + 1.0)
  end
end

private def pixelumm_api_stats(count : UInt64 = 0_u64,
                               nested_count : UInt64 = 0_u64,
                               operation : Int32 = 0) : ML::Video::MPSAPITimingStats
  return ML::Video::MPSAPITimingStats.new(0_u64, 0_u64, 0_u64, 0_u64, 0_u64,
    0_u64, 0_u64, 0_u64, 0_u64, 0_u64) if count == 0

  wall_total = count * (operation == 1 ? 2_u64 : 10_u64)
  wall_max = operation == 1 ? 2_u64 : 10_u64
  cpu_total = count * (operation == 1 ? 1_u64 : 7_u64)
  cpu_max = operation == 1 ? 1_u64 : 7_u64
  nested_wall = operation == 0 ? nested_count * 2_u64 : (operation == 1 ? wall_total : 0_u64)
  nested_cpu = operation == 0 ? nested_count : (operation == 1 ? cpu_total : 0_u64)
  exclusive_wall = operation == 0 ? wall_total - nested_wall : wall_total
  exclusive_cpu = operation == 0 ? cpu_total - nested_cpu : cpu_total
  ML::Video::MPSAPITimingStats.new(count, wall_total, wall_max,
    cpu_total, cpu_max, nested_count, nested_wall, nested_cpu,
    exclusive_wall, exclusive_cpu)
end

private def pixelumm_api_profile(tile_count : Int32,
                                 root_count : Int32,
                                 graph_attention_override : UInt64? = nil,
                                 continuation_override : UInt64? = nil,
                                 linear_calls : UInt64 = 0_u64) : ML::Video::MPSAPIProfile
  graph_counts = {
    0 => 0_u64,
    1 => 9_u64,
    2 => graph_attention_override || (3_u64 * tile_count.to_u64),
    3 => 3_u64,
    4 => 0_u64,
    5 => 3_u64,
  }
  continuation_total = continuation_override || (root_count - 1).to_u64
  continuation_counts = {} of Int32 => UInt64
  remaining = continuation_total
  {1, 2, 3, 5}.each do |phase_id|
    count = {remaining, graph_counts[phase_id]}.min
    continuation_counts[phase_id] = count
    remaining -= count
  end
  raise ArgumentError.new("spec helper cannot nest more commits than graph calls") unless remaining == 0

  samples = [] of ML::Video::MPSAPIProfileSample
  ML::Video::MPSAPIProfile::PHASE_NAMES.keys.sort.each do |phase_id|
    graph_count = graph_counts[phase_id]
    continuation_count = continuation_counts[phase_id]? || 0_u64
    linear_count = phase_id == 1 ? linear_calls : 0_u64
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 0,
      pixelumm_api_stats(graph_count, continuation_count, 0))
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 1,
      pixelumm_api_stats(continuation_count, continuation_count, 1))
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 2,
      pixelumm_api_stats(linear_count, 0_u64, 2))
  end
  ML::Video::MPSAPIProfile.build(samples)
end

private def pixelumm_root_duration_sum(roots : Array(ML::Video::MPSCommandRootTiming)) : Float64
  roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }
end

describe ML::Video::PixelUMMBlockPhaseProfile do
  it "interprets ordered CPU markers and preserves inclusive shared-root candidates" do
    markers = pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5])
    roots = pixelumm_profile_roots(5)
    api = pixelumm_api_profile(1, roots.size)

    profile = ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
      pixelumm_root_duration_sum(roots), api, 1)

    profile.markers.map(&.name).should eq(ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES.to_a)
    profile.phase_wall_seconds.size.should eq(10)
    profile.phase_thread_cpu_seconds.size.should eq(10)
    profile.phase_wall_seconds.values.all? { |value| (value - 0.1).abs < 1e-12 }.should be_true
    profile.phase_thread_cpu_seconds.values.all? { |value| (value - 0.2).abs < 1e-12 }.should be_true
    # Synthetic clocks are deliberately independent here. The validator checks
    # each clock's order; it does not infer GPU waits from their difference.
    profile.phase_thread_cpu_seconds.values.sum.should be_close(2.0, 1e-12)

    profile.root_candidates.should eq([
      ["input_norm"],
      ["input_norm", "qkv_projection", "qk_norm_rope"],
      ["qk_norm_rope", "attention"],
      ["attention", "output_projection", "residual_post_norm"],
      ["residual_post_norm", "ffn_gate_up", "ffn_activation", "ffn_down", "final_residual"],
    ])
    profile.phase_candidate_root_duration_sum_seconds["input_norm"].should eq(2.0)
    profile.phase_definite_root_duration_sum_seconds["input_norm"].should eq(1.0)
    profile.phase_definite_root_duration_sum_seconds["attention"].should eq(0.0)
    # Candidate sums can count a root in more than one phase. The complete
    # profile reports the non-additive union, envelope, and overlap separately.
    profile.phase_candidate_root_duration_sum_seconds.values.sum.should be >
                                                                        profile.root_profile.root_duration_seconds
    profile.root_profile.root_count.should eq(5)
    profile.root_profile.root_duration_seconds.should eq(5.0)
    profile.root_profile.union_seconds.should eq(3.0)
    profile.root_profile.envelope_seconds.should eq(3.0)
    profile.root_profile.overlap_seconds.should eq(2.0)
    profile.tile_count.should eq(1)

    profile.api_profile.phases.find { |phase| phase.phase_id == 2 }.not_nil!
      .operations["graph_encode"].count.should eq(3_u64)
    profile.api_profile.phases.find { |phase| phase.phase_id == 4 }.not_nil!
      .operations["graph_encode"].count.should eq(0_u64)
  end

  it "allows every phase to share a single root without manufacturing a split" do
    markers = pixelumm_profile_markers(Array(Int32).new(11, 1))
    roots = pixelumm_profile_roots(1)
    profile = ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
      pixelumm_root_duration_sum(roots), pixelumm_api_profile(1, 1), 1)

    profile.root_candidates.should eq([
      ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES.to_a[0...-1],
    ])
    profile.phase_definite_root_duration_sum_seconds.values.sum.should eq(0.0)
    profile.phase_candidate_root_duration_sum_seconds.values.sum.should eq(10.0)
  end

  it "serializes PixelUMM API labels, source IDs, timing stats, and root timestamps" do
    markers = pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5])
    roots = pixelumm_profile_roots(5)
    profile = ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
      pixelumm_root_duration_sum(roots), pixelumm_api_profile(1, roots.size), 1)

    json = JSON.parse(profile.to_json)
    json["api_profile"]["source_schema"].as_s.should eq("MPSAPIProfile WAN phase IDs")
    api_phases = json["api_profile"]["phases"].as_a
    attention = api_phases.find { |phase| phase["source_phase_id"].as_i == 2 }.not_nil!
    attention["pixelumm_phase"].as_s.should eq("local_attention")
    attention["source_phase_name"].as_s.should eq("self_attention")
    attention["operations"]["graph_encode"]["count"].as_i.should eq(3)
    attention["operations"]["graph_encode"]["exclusive_wall_total_ns"].as_i.should be > 0

    json["roots"].as_a.size.should eq(5)
    json["roots"][1]["gpu_start_seconds"].as_f.should eq(0.5)
    json["roots"][1]["candidate_phases"].as_a.map(&.as_s)
      .should contain("qkv_projection")
    json["root_profile"]["union_seconds"].as_f.should eq(3.0)
  end

  it "rejects malformed marker inventory, time, and root cursors" do
    roots = pixelumm_profile_roots(5)
    api = pixelumm_api_profile(1, roots.size)
    valid = pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5])

    expect_raises(ArgumentError) do
      ML::Video::PixelUMMBlockPhaseProfile.build(valid[0...-1], roots,
        pixelumm_root_duration_sum(roots), api, 1)
    end

    reordered = valid.dup
    reordered[1] = ML::Video::PixelUMMBlockPhaseMarker.new("qk_norm_rope",
      reordered[1].elapsed_seconds, reordered[1].thread_cpu_seconds,
      reordered[1].root_count)
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMBlockPhaseProfile.build(reordered, roots,
        pixelumm_root_duration_sum(roots), api, 1)
    end

    invalid_markers = [
      pixelumm_profile_markers([2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5]),
      pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 4]),
      pixelumm_profile_markers([1, 3, 2, 3, 4, 4, 5, 5, 5, 5, 5]),
      pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6]),
    ]
    invalid_markers.each do |markers|
      expect_raises(ArgumentError) do
        ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
          pixelumm_root_duration_sum(roots), api, 1)
      end
    end

    first_time = valid.dup
    first_time[0] = ML::Video::PixelUMMBlockPhaseMarker.new("input_norm",
      0.01, 0.0, 1)
    backwards_wall = valid.dup
    backwards_wall[4] = ML::Video::PixelUMMBlockPhaseMarker.new("output_projection",
      0.2, 0.8, 4)
    backwards_cpu = valid.dup
    backwards_cpu[4] = ML::Video::PixelUMMBlockPhaseMarker.new("output_projection",
      0.4, 0.3, 4)
    nonfinite = valid.dup
    nonfinite[3] = ML::Video::PixelUMMBlockPhaseMarker.new("attention",
      Float64::NAN, 0.6, 3)
    invalid_times = [first_time, backwards_wall, backwards_cpu, nonfinite]
    invalid_times.each do |markers|
      expect_raises(ArgumentError) do
        ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
          pixelumm_root_duration_sum(roots), api, 1)
      end
    end
  end

  it "rejects source graph-count, T5-linear, continuation-count, and tile-count mismatches" do
    markers = pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5])
    roots = pixelumm_profile_roots(5)
    duration = pixelumm_root_duration_sum(roots)
    invalid_profiles = [
      pixelumm_api_profile(1, roots.size, graph_attention_override: 4_u64),
      pixelumm_api_profile(1, roots.size, linear_calls: 1_u64),
      pixelumm_api_profile(1, roots.size, continuation_override: 3_u64),
    ]
    invalid_profiles.each do |api|
      expect_raises(ArgumentError) do
        ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots, duration, api, 1)
      end
    end

    expect_raises(ArgumentError) do
      ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots, duration,
        pixelumm_api_profile(1, roots.size), 0)
    end
  end

  it "rejects invalid roots, native duration, and zero attention geometry" do
    markers = pixelumm_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5])
    roots = pixelumm_profile_roots(5)
    api = pixelumm_api_profile(1, roots.size)

    expect_raises(ArgumentError) do
      ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
        pixelumm_root_duration_sum(roots) + 0.5, api, 1)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMBlockPhaseProfile.build(markers, roots,
        pixelumm_root_duration_sum(roots), api, -1)
    end

    bad_roots = roots.dup
    bad_roots[2] = ML::Video::MPSCommandRootTiming.new(2, false, false, 1.0, 2.0)
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMBlockPhaseProfile.build(markers, bad_roots,
        pixelumm_root_duration_sum(bad_roots), api, 1)
    end
  end
end
