require "spec"
require "../src/ml/video/hy_nomas_block_profile"

private def hy_profile_markers(counts : Array(Int32))
  ML::Video::HyNomasBlockPhaseProfile::MARKER_NAMES.map_with_index do |name, index|
    ML::Video::HyNomasBlockPhaseMarker.new(name, index.to_f64 * 0.001, counts[index])
  end.to_a
end

private def hy_profile_roots(count : Int32)
  Array(ML::Video::MPSCommandRootTiming).new(count) do |index|
    ML::Video::MPSCommandRootTiming.new(index, true, false,
      index.to_f64, index.to_f64 + 0.5)
  end
end

describe ML::Video::HyNomasBlockPhaseProfile do
  it "keeps one shared root ambiguous and validation CPU-only" do
    profile = ML::Video::HyNomasBlockPhaseProfile.build(hy_profile_markers(Array(Int32).new(9, 1)),
      hy_profile_roots(1), 0.5)
    profile.root_candidates.should eq([ML::Video::HyNomasBlockPhaseProfile::MARKER_NAMES.to_a[1...-1]])
    profile.root_candidates.flatten.should_not contain("validate")
    profile.root_candidates.flatten.should_not contain("complete")
    profile.phase_cpu_seconds.keys.should eq(ML::Video::HyNomasBlockPhaseProfile::MARKER_NAMES.to_a[0...-1])
    profile.phase_cpu_seconds.values.sum.should be_close(0.008, 1e-12)
    profile.root_profile.union_seconds.should eq(0.5)
  end

  it "preserves root jumps and both names on a boundary root" do
    counts = [1, 1, 3, 3, 4, 7, 7, 8, 9]
    profile = ML::Video::HyNomasBlockPhaseProfile.build(hy_profile_markers(counts), hy_profile_roots(9), 4.5)
    profile.root_candidates.should eq([
      ["modulation"], ["modulation"], ["modulation", "qkv", "joint_pack"],
      ["joint_pack", "attention"], ["attention"], ["attention"],
      ["attention", "attention_projection_residual", "image_ffn"],
      ["image_ffn", "text_ffn"], ["text_ffn"],
    ])
    profile.root_profile.root_count.should eq(9)
    profile.root_profile.union_seconds.should eq(4.5)
    profile.root_profile.envelope_seconds.should eq(8.5)
  end

  it "rejects incomplete, stale, decreasing or nonfinite observations" do
    roots = hy_profile_roots(9)
    valid = hy_profile_markers([1, 1, 3, 3, 4, 7, 7, 8, 9])
    expect_raises(ArgumentError) { ML::Video::HyNomasBlockPhaseProfile.build(valid[0...-1], roots, 4.5) }
    [
      hy_profile_markers([2, 2, 3, 3, 4, 7, 7, 8, 9]),
      hy_profile_markers([1, 2, 3, 3, 4, 7, 7, 8, 9]),
      hy_profile_markers([1, 1, 3, 2, 4, 7, 7, 8, 9]),
      hy_profile_markers([1, 1, 3, 3, 4, 7, 7, 8, 8]),
    ].each do |markers|
      expect_raises(ArgumentError) { ML::Video::HyNomasBlockPhaseProfile.build(markers, roots, 4.5) }
    end
    invalid_time = valid.dup
    invalid_time[3] = ML::Video::HyNomasBlockPhaseMarker.new("joint_pack", Float64::NAN, 3)
    expect_raises(ArgumentError) { ML::Video::HyNomasBlockPhaseProfile.build(invalid_time, roots, 4.5) }
    expect_raises(ArgumentError) { ML::Video::HyNomasBlockPhaseProfile.build(valid, roots, 4.0) }
    bad_roots = roots.dup
    bad_roots[4] = ML::Video::MPSCommandRootTiming.new(4, false, false, 4.0, 4.5)
    expect_raises(ArgumentError) { ML::Video::HyNomasBlockPhaseProfile.build(valid, bad_roots, 4.5) }
  end
end
