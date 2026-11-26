require "spec"
require "../src/ml/video/mps_api_profile"

private def api_timing(count : UInt64 = 0_u64,
                       wall_total : UInt64 = 0_u64,
                       wall_max : UInt64 = 0_u64,
                       cpu_total : UInt64 = 0_u64,
                       cpu_max : UInt64 = 0_u64,
                       nested_count : UInt64 = 0_u64,
                       nested_wall : UInt64 = 0_u64,
                       nested_cpu : UInt64 = 0_u64,
                       exclusive_wall : UInt64 = 0_u64,
                       exclusive_cpu : UInt64 = 0_u64) : ML::Video::MPSAPITimingStats
  ML::Video::MPSAPITimingStats.new(count, wall_total, wall_max,
    cpu_total, cpu_max, nested_count, nested_wall, nested_cpu,
    exclusive_wall, exclusive_cpu)
end

private def profile_samples(graph_override : ML::Video::MPSAPITimingStats? = nil,
                            commit_override : ML::Video::MPSAPITimingStats? = nil,
                            linear_override : ML::Video::MPSAPITimingStats? = nil) : Array(ML::Video::MPSAPIProfileSample)
  samples = [] of ML::Video::MPSAPIProfileSample
  ML::Video::MPSAPIProfile::PHASE_NAMES.each_key do |phase_id|
    graph = phase_id == 2 ? (graph_override || api_timing) : api_timing
    commit = phase_id == 2 ? (commit_override || api_timing) : api_timing
    linear = phase_id == 3 ? (linear_override || api_timing) : api_timing
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 0, graph)
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 1, commit)
    samples << ML::Video::MPSAPIProfileSample.new(phase_id, 2, linear)
  end
  samples
end

describe ML::Video::MPSAPIProfile do
  it "splits immediately before the currently open root and rejects a one-root split" do
    ML::Video::MPSAPIPhaseBoundary.precurrent_root_end_count(2, 0).should eq(1)
    ML::Video::MPSAPIPhaseBoundary.precurrent_root_end_count(5, 2).should eq(4)

    expect_raises(ArgumentError) do
      ML::Video::MPSAPIPhaseBoundary.precurrent_root_end_count(1, 0)
    end
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIPhaseBoundary.precurrent_root_end_count(3, 2)
    end
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIPhaseBoundary.precurrent_root_end_count(0, 0)
    end
  end

  it "requires a complete phase-operation matrix and preserves nested inclusive/exclusive metrics" do
    graph = api_timing(2, 100, 60, 50, 30, 1, 20, 10, 80, 40)
    commit = api_timing(1, 20, 20, 10, 10, 1, 20, 10, 20, 10)
    samples = profile_samples(graph, commit)
    profile = ML::Video::MPSAPIProfile.build(samples)

    profile.phases.map(&.name).should eq([
      "unassigned", "self_prepare", "self_attention",
      "self_projection_residual", "cross_attention_projection_residual", "ffn",
    ])
    attention = profile.phases.find { |phase| phase.phase_id == 2 }.not_nil!
    attention.operations["graph_encode"].wall_total_ns.should eq(100_u64)
    attention.operations["graph_encode"].exclusive_wall_total_ns.should eq(80_u64)
    attention.operations["commit_and_continue"].nested_count.should eq(1_u64)

    samples.pop
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIProfile.build(samples)
    end
  end

  it "allows multiple nested commits within one graph span" do
    graph = api_timing(1, 100, 100, 50, 50, 2, 20, 10, 80, 40)
    commit = api_timing(2, 20, 10, 10, 5, 2, 20, 10, 20, 10)
    profile = ML::Video::MPSAPIProfile.build(profile_samples(graph, commit))
    attention = profile.phases.find { |phase| phase.phase_id == 2 }.not_nil!
    attention.operations["graph_encode"].count.should eq(1_u64)
    attention.operations["graph_encode"].nested_count.should eq(2_u64)
  end

  it "rejects an empty matrix and checks fixed full-block API call counts" do
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIProfile.build(profile_samples)
    end

    graph = api_timing(2, 100, 60, 50, 30, 1, 10, 5, 90, 45)
    commit = api_timing(2, 20, 10, 10, 5, 1, 10, 5, 20, 10)
    linear = api_timing(1, 10, 10, 5, 5, 0, 0, 0, 10, 5)
    profile = ML::Video::MPSAPIProfile.build(profile_samples(graph, commit, linear))
    profile.validate_full_block_counts!(2_u64, 1_u64, 2_u64)

    expect_raises(ArgumentError) do
      profile.validate_full_block_counts!(3_u64, 1_u64, 2_u64)
    end
    expect_raises(ArgumentError) do
      profile.validate_full_block_counts!(2_u64, 2_u64, 2_u64)
    end
  end

  it "rejects duplicate and unknown phase-operation snapshots" do
    samples = profile_samples
    samples[-1] = samples[0]
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIProfile.build(samples)
    end

    samples = profile_samples
    samples[0] = ML::Video::MPSAPIProfileSample.new(16, 0, api_timing)
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIProfile.build(samples)
    end
  end

  it "rejects invalid maxima, nested accounting, and parent-child disagreement" do
    invalid = [
      api_timing(1, 10, 11, 5, 5, 0, 0, 0, 10, 5),
      api_timing(1, 10, 10, 5, 6, 0, 0, 0, 10, 5),
      api_timing(1, 10, 10, 5, 5, 1, 11, 1, 0, 4),
      api_timing(1, 10, 10, 5, 5, 0, 0, 0, 9, 5),
    ]
    invalid.each do |stats|
      samples = profile_samples(stats, api_timing(1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
      expect_raises(ArgumentError) do
        ML::Video::MPSAPIProfile.build(samples)
      end
    end

    expect_raises(ArgumentError) do
      api_timing(1, 10, 10, 5, 5, 1, 0, 0, 10, 5).validate!(2)
    end

    expect_raises(ArgumentError) do
      api_timing(1, 10, 10, 5, 5, 0, 1, 0, 9, 5).validate!(0)
    end
    expect_raises(ArgumentError) do
      api_timing(1, 10, 10, 5, 5, 1, 0, 0, 10, 5).validate!(0)
    end

    graph = api_timing(1, 10, 10, 5, 5, 1, 2, 1, 8, 4)
    commit = api_timing(1, 2, 2, 1, 1, 1, 1, 1, 2, 1)
    expect_raises(ArgumentError) do
      ML::Video::MPSAPIProfile.build(profile_samples(graph, commit))
    end
  end
end
