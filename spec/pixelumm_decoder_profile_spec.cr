require "spec"
require "../src/ml/video/pixelumm_decoder_profile"

private def decoder_profile_markers(cursors : Array(Int32),
                                    wall_step : Float64 = 0.1,
                                    cpu_step : Float64 = 0.2) : Array(ML::Video::PixelUMMBlockPhaseMarker)
  ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES.map_with_index do |name, index|
    ML::Video::PixelUMMBlockPhaseMarker.new(name, index.to_f64 * wall_step,
      index.to_f64 * cpu_step, cursors[index])
  end.to_a
end

private def decoder_profile_roots(count : Int32,
                                  duration : Float64 = 1.0) : Array(ML::Video::MPSCommandRootTiming)
  Array(ML::Video::MPSCommandRootTiming).new(count) do |index|
    start_time = index.to_f64 * 0.5
    ML::Video::MPSCommandRootTiming.new(index, true, false,
      start_time, start_time + duration)
  end
end

private def decoder_profile_api_stats(count : UInt64 = 0_u64,
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

private def decoder_profile_api(block_count : Int32, tile_counts : Array(Int32),
                                root_count : Int32,
                                graph_attention_override : UInt64? = nil,
                                graph_head_override : UInt64? = nil,
                                continuation_override : UInt64? = nil,
                                linear_calls : UInt64 = 0_u64) : ML::Video::MPSAPIProfile
  graph_counts = {
    0 => graph_head_override || 5_u64,
    1 => 9_u64 * block_count.to_u64,
    2 => graph_attention_override || (3_u64 * tile_counts.sum(&.to_u64)),
    3 => 3_u64 * block_count.to_u64,
    4 => 0_u64,
    5 => 3_u64 * block_count.to_u64,
  }
  continuation_total = continuation_override || (root_count - 1).to_u64
  continuation_counts = {} of Int32 => UInt64
  remaining = continuation_total
  {0, 1, 2, 3, 5}.each do |phase_id|
    count = {remaining, graph_counts[phase_id]}.min
    continuation_counts[phase_id] = count
    remaining -= count
  end
  raise ArgumentError.new("spec helper cannot nest more commits than graph calls") unless remaining == 0

  samples = [] of ML::Video::MPSAPIProfileSample
  ML::Video::MPSAPIProfile::PHASE_NAMES.keys.sort.each do |phase_id|
    graph_count = graph_counts[phase_id]
    continuation_count = continuation_counts[phase_id]? || 0_u64
    linear_count = phase_id == 0 ? linear_calls : 0_u64
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 0,
      decoder_profile_api_stats(graph_count, continuation_count, 0))
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 1,
      decoder_profile_api_stats(continuation_count, continuation_count, 1))
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 2,
      decoder_profile_api_stats(linear_count, 0_u64, 2))
  end
  ML::Video::MPSAPIProfile.build(samples)
end

private def decoder_profile_valid_inputs
  block_markers = [
    decoder_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5]),
    decoder_profile_markers([5, 6, 6, 7, 8, 8, 9, 9, 9, 9, 9]),
  ]
  tile_counts = [1, 2]
  roots = decoder_profile_roots(11)
  api = decoder_profile_api(block_markers.size, tile_counts, roots.size)
  {block_markers, tile_counts, roots, api}
end

private def decoder_profile_build(block_markers, tile_counts, roots, api)
  ML::Video::PixelUMMDecoderPhaseProfile.build(block_markers, tile_counts,
    9, roots.size, roots, roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }, api)
end

describe ML::Video::PixelUMMDecoderPhaseProfile do
  it "aggregates per-block phases and preserves shared-root candidates across block boundaries" do
    block_markers, tile_counts, roots, api = decoder_profile_valid_inputs
    profile = decoder_profile_build(block_markers, tile_counts, roots, api)

    profile.blocks.size.should eq(2)
    profile.blocks.map(&.phase_wall_seconds).size.should eq(2)
    profile.blocks.each do |block|
      phase_times = block.phase_wall_seconds
      phase_times.size.should eq(10)
      phase_times.values.all? { |value| (value - 0.1).abs < 1e-12 }.should be_true
    end
    profile.blocks.each do |block|
      phase_times = block.phase_thread_cpu_seconds
      phase_times.values.all? { |value| (value - 0.2).abs < 1e-12 }.should be_true
    end
    profile.aggregate_phase_wall_seconds.values.all? { |value| (value - 0.2).abs < 1e-12 }.should be_true
    profile.aggregate_phase_thread_cpu_seconds.values.all? { |value| (value - 0.4).abs < 1e-12 }.should be_true

    # Root sequence 4 is the open root shared at the block boundary and must
    # retain both candidates; assigning it exclusively would overstate phase precision.
    profile.root_candidates[4].should contain("block_0/final_residual")
    profile.root_candidates[4].should contain("block_1/input_norm")
    profile.root_candidates[8].should contain("head")
    profile.root_candidates.size.should eq(roots.size)
    profile.root_candidates.all?(&.any?).should be_true
    profile.phase_candidate_root_duration_sum_seconds.values.sum.should be >
                                                                        profile.root_profile.root_duration_seconds
    profile.root_profile.root_count.should eq(11)
    profile.root_profile.root_duration_seconds.should eq(11.0)
    profile.root_profile.union_seconds.should eq(6.0)
    profile.root_profile.envelope_seconds.should eq(6.0)
    profile.root_profile.overlap_seconds.should eq(5.0)

    profile.api_profile.phases.find { |phase| phase.phase_id == 0 }.not_nil!
      .operations["graph_encode"].count.should eq(5_u64)
    profile.api_profile.phases.find { |phase| phase.phase_id == 2 }.not_nil!
      .operations["graph_encode"].count.should eq(9_u64)
  end

  it "serializes block clocks, global root candidates, aggregate API counters and overlap semantics" do
    block_markers, tile_counts, roots, api = decoder_profile_valid_inputs
    json = JSON.parse(decoder_profile_build(block_markers, tile_counts, roots, api).to_json)

    json["schema"].as_s.should eq("cogni_ml.pixelumm_decoder_phase_profile/v1")
    json["blocks"].as_a.size.should eq(2)
    json["blocks"][1]["slot"].as_i.should eq(1)
    json["blocks"][1]["markers"][0]["root_count"].as_i.should eq(5)
    (json["aggregate_phase_wall_seconds"]["attention"].as_f - 0.2).abs
      .should be < 1e-12
    json["root_candidates"][4].as_a.map(&.as_s)
      .should contain("block_1/input_norm")
    json["roots"][8]["candidate_phases"].as_a.map(&.as_s).should contain("head")
    json["root_profile"]["union_seconds"].as_f.should eq(6.0)
    json["phase_root_duration_sum_semantics"].as_s.should contain("non-additive")
    json["api_profile"]["phases"].as_a.size.should eq(6)
    json["api_profile"]["phases"][0]["operations"]["graph_encode"]["count"].as_i.should eq(5)
  end

  it "rejects malformed block counts, marker order, local clocks, and cross-block cursors" do
    block_markers, tile_counts, roots, api = decoder_profile_valid_inputs

    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderPhaseProfile.build(block_markers[0...1], tile_counts,
        9, roots.size, roots, roots.sum { |r| r.gpu_end_seconds - r.gpu_start_seconds }, api)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderPhaseProfile.build(block_markers, [1, 0],
        9, roots.size, roots, roots.sum { |r| r.gpu_end_seconds - r.gpu_start_seconds }, api)
    end

    invalid = block_markers.map(&.dup)
    invalid[0][1] = ML::Video::PixelUMMBlockPhaseMarker.new("qk_norm_rope", 0.1, 0.2, 2)
    expect_raises(ArgumentError) { decoder_profile_build(invalid, tile_counts, roots, api) }

    invalid = block_markers.map(&.dup)
    invalid[0][0] = ML::Video::PixelUMMBlockPhaseMarker.new("input_norm", 0.01, 0.0, 1)
    expect_raises(ArgumentError) { decoder_profile_build(invalid, tile_counts, roots, api) }

    invalid = block_markers.map(&.dup)
    invalid[1][0] = ML::Video::PixelUMMBlockPhaseMarker.new("input_norm", 0.0, 0.0, 6)
    expect_raises(ArgumentError) { decoder_profile_build(invalid, tile_counts, roots, api) }

    invalid = block_markers.map(&.dup)
    invalid[0][4] = ML::Video::PixelUMMBlockPhaseMarker.new("output_projection", Float64::NAN, 0.8, 4)
    expect_raises(ArgumentError) { decoder_profile_build(invalid, tile_counts, roots, api) }
  end

  it "rejects bad head cursors, incomplete root coverage, and nonmonotone marker cursors" do
    block_markers, tile_counts, roots, api = decoder_profile_valid_inputs
    duration = roots.sum { |r| r.gpu_end_seconds - r.gpu_start_seconds }

    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderPhaseProfile.build(block_markers, tile_counts,
        8, roots.size, roots, duration, api)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderPhaseProfile.build(block_markers, tile_counts,
        9, roots.size - 1, roots, duration, api)
    end

    invalid = block_markers.map(&.dup)
    invalid[1][3] = ML::Video::PixelUMMBlockPhaseMarker.new("attention", 0.3, 0.6, 5)
    expect_raises(ArgumentError) { decoder_profile_build(invalid, tile_counts, roots, api) }
  end

  it "accepts the smallest one-block integrated profile" do
    markers = [decoder_profile_markers([1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5])]
    tile_counts = [1]
    roots = decoder_profile_roots(6)
    api = decoder_profile_api(1, tile_counts, roots.size)
    profile = ML::Video::PixelUMMDecoderPhaseProfile.build(markers, tile_counts,
      5, roots.size, roots, roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }, api)

    profile.blocks.size.should eq(1)
    profile.root_candidates.size.should eq(6)
    profile.root_candidates.all?(&.any?).should be_true
    profile.root_candidates.last.should contain("head")
  end

  it "rejects aggregate Graph, T5 linear, and continuation count mismatches" do
    block_markers, tile_counts, roots, _api = decoder_profile_valid_inputs
    duration = roots.sum { |r| r.gpu_end_seconds - r.gpu_start_seconds }
    bad_profiles = [
      decoder_profile_api(2, tile_counts, roots.size, graph_attention_override: 10_u64),
      decoder_profile_api(2, tile_counts, roots.size, graph_head_override: 4_u64),
      decoder_profile_api(2, tile_counts, roots.size, linear_calls: 1_u64),
      decoder_profile_api(2, tile_counts, roots.size, continuation_override: 9_u64),
    ]
    bad_profiles.each do |profile|
      expect_raises(ArgumentError) do
        ML::Video::PixelUMMDecoderPhaseProfile.build(block_markers, tile_counts,
          9, roots.size, roots, duration, profile)
      end
    end
  end
end
