require "spec"
require "../src/ml/video/mps_command_phase_profile"

private def root_timing(sequence : Int32, start_time : Float64, end_time : Float64,
                        completed : Bool = true, has_error : Bool = false) : ML::Video::MPSCommandRootTiming
  ML::Video::MPSCommandRootTiming.new(sequence, completed, has_error,
    start_time, end_time)
end

private def phase_range(name : String, first_root : Int32,
                        end_root : Int32) : ML::Video::MPSCommandPhaseRange
  ML::Video::MPSCommandPhaseRange.new(name, first_root, end_root)
end

describe ML::Video::MPSCommandPhaseProfile do
  it "reports root sum, envelope, interval union, temporal overlap, and per-phase unions separately" do
    # Phase ranges are half-open sequence intervals. The second phase starts at
    # the first phase's end count, matching count_after_continue - 1.
    roots = [
      root_timing(0, 0.0, 2.0),
      root_timing(1, 1.0, 3.0),
      root_timing(2, 4.0, 5.0),
    ]
    phases = [phase_range("self_prepare", 0, 1), phase_range("self_branch", 1, 3)]

    profile = ML::Video::MPSCommandPhaseProfile.build(roots, phases, 5.0)

    profile.root_duration_seconds.should eq(5.0)
    profile.envelope_seconds.should eq(5.0)
    profile.union_seconds.should eq(4.0)
    profile.overlap_seconds.should eq(1.0)
    profile.phase_union_seconds["self_prepare"].should eq(2.0)
    profile.phase_union_seconds["self_branch"].should eq(3.0)
    # Per-phase unions can overlap in time, so their sum is not the global union.
    profile.phase_union_seconds.values.sum.should eq(5.0)
  end

  it "does not lose interval precision when timestamps cross a large exponent boundary" do
    roots = [
      root_timing(0, 1_048_574.3803184337, 1_048_575.4040455254),
      root_timing(1, 1_048_576.16187885, 1_048_576.363356434),
    ]
    native_sum = 1.225204675574787

    profile = ML::Video::MPSCommandPhaseProfile.build(roots,
      [phase_range("all", 0, 2)], native_sum)

    profile.union_seconds.should eq(profile.root_duration_seconds)
    profile.overlap_seconds.should eq(0.0)
  end

  it "rejects an empty root set or missing phase ranges" do
    expect_raises(ArgumentError) do
      ML::Video::MPSCommandPhaseProfile.build(
        [] of ML::Video::MPSCommandRootTiming,
        [phase_range("all", 0, 0)], 0.0)
    end

    roots = [root_timing(0, 0.0, 1.0)]
    expect_raises(ArgumentError) do
      ML::Video::MPSCommandPhaseProfile.build(roots,
        [] of ML::Video::MPSCommandPhaseRange, 1.0)
    end

    roots = [root_timing(0, 0.0, 1.0), root_timing(1, 1.0, 2.0),
             root_timing(2, 2.0, 3.0)]
    expect_raises(ArgumentError) do
      ML::Video::MPSCommandPhaseProfile.build(roots,
        [phase_range("left", 0, 1), phase_range("right", 2, 3)], 3.0)
    end
  end

  it "rejects duplicate or overlapping phase assignments" do
    roots = [root_timing(0, 0.0, 1.0), root_timing(1, 1.0, 2.0),
             root_timing(2, 2.0, 3.0)]
    [
      [phase_range("all", 0, 3), phase_range("duplicate", 0, 3)],
      [phase_range("left", 0, 2), phase_range("right", 1, 3)],
    ].each do |phases|
      expect_raises(ArgumentError) do
        ML::Video::MPSCommandPhaseProfile.build(roots, phases, 3.0)
      end
    end
  end

  it "rejects missing or duplicate root sequence numbers" do
    phases = [phase_range("all", 0, 2)]
    [
      [root_timing(0, 0.0, 1.0), root_timing(2, 1.0, 2.0)],
      [root_timing(0, 0.0, 1.0), root_timing(0, 1.0, 2.0)],
      [root_timing(1, 0.0, 1.0), root_timing(0, 1.0, 2.0)],
    ].each do |roots|
      expect_raises(ArgumentError) do
        ML::Video::MPSCommandPhaseProfile.build(roots, phases, 2.0)
      end
    end
  end

  it "rejects incomplete roots, errors, and invalid GPU timestamps" do
    phases = [phase_range("all", 0, 1)]
    [
      root_timing(0, 0.0, 1.0, completed: false),
      root_timing(0, 0.0, 1.0, has_error: true),
      root_timing(0, Float64::NAN, 1.0),
      root_timing(0, 0.0, 0.0),
      root_timing(0, 2.0, 1.0),
      root_timing(0, -1.0, 1.0),
    ].each do |root|
      expect_raises(ArgumentError) do
        ML::Video::MPSCommandPhaseProfile.build([root], phases, 1.0)
      end
    end
  end

  it "rejects phase ranges that do not cover roots from zero through the terminal count" do
    roots = [root_timing(0, 0.0, 1.0), root_timing(1, 1.0, 2.0)]
    [
      [phase_range("starts_late", 1, 2)],
      [phase_range("ends_early", 0, 1)],
      [phase_range("empty", 0, 0), phase_range("rest", 0, 2)],
    ].each do |phases|
      expect_raises(ArgumentError) do
        ML::Video::MPSCommandPhaseProfile.build(roots, phases, 2.0)
      end
    end
  end

  it "rejects duplicate phase names even when root ranges otherwise cover exactly" do
    roots = [root_timing(0, 0.0, 1.0), root_timing(1, 1.0, 2.0)]
    expect_raises(ArgumentError) do
      ML::Video::MPSCommandPhaseProfile.build(roots,
        [phase_range("same", 0, 1), phase_range("same", 1, 2)], 2.0)
    end
  end

  it "rejects a root-duration sum inconsistent with the native aggregate" do
    roots = [root_timing(0, 0.0, 1.0), root_timing(1, 1.0, 2.0)]
    expect_raises(ArgumentError) do
      ML::Video::MPSCommandPhaseProfile.build(roots,
        [phase_range("all", 0, 2)], 2.5)
    end
  end
end
