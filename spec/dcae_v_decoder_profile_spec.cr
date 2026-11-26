require "spec"
require "../src/ml/video/dcae_v_decoder_profile"

describe ML::Video::DCAEVDecoderProfile do
  it "preserves overlapping root candidates and reports group duration bounds" do
    operations = [
      ML::Video::DCAEVDecoderOperationTiming.new(
        "tile0/decoder.stages.5.op_list.0.main.conv1.conv3d",
        "residual_conv3d", "stage_5", 0.01, 1, 2),
      ML::Video::DCAEVDecoderOperationTiming.new(
        "tile0/decoder.stages.5.op_list.0.main.conv1.norm.rmsnorm",
        "norm", "stage_5", 0.002, 2, 2),
    ]
    roots = [
      ML::Video::MPSCommandRootTiming.new(0, true, false, 0.0, 2.0),
      ML::Video::MPSCommandRootTiming.new(1, true, false, 1.0, 3.0),
    ]

    profile = ML::Video::DCAEVDecoderProfile.build(operations, roots, 4.0)

    profile.root_candidates.should eq([
      ["tile0/decoder.stages.5.op_list.0.main.conv1.conv3d"],
      ["tile0/decoder.stages.5.op_list.0.main.conv1.conv3d",
       "tile0/decoder.stages.5.op_list.0.main.conv1.norm.rmsnorm"],
    ])
    profile.operation_candidate_root_duration_sum_seconds[
      "tile0/decoder.stages.5.op_list.0.main.conv1.conv3d"].should eq(4.0)
    profile.operation_definite_root_duration_sum_seconds[
      "tile0/decoder.stages.5.op_list.0.main.conv1.conv3d"].should eq(2.0)
    profile.operation_candidate_root_duration_sum_seconds[
      "tile0/decoder.stages.5.op_list.0.main.conv1.norm.rmsnorm"].should eq(2.0)
    profile.operation_definite_root_duration_sum_seconds[
      "tile0/decoder.stages.5.op_list.0.main.conv1.norm.rmsnorm"].should eq(0.0)
    profile.group_candidate_root_duration_sum_seconds["stage_5/residual_conv3d"].should eq(4.0)
    profile.group_definite_root_duration_sum_seconds["stage_5/residual_conv3d"].should eq(2.0)
    profile.group_candidate_root_duration_sum_seconds["stage_5/norm"].should eq(2.0)
    profile.group_definite_root_duration_sum_seconds["stage_5/norm"].should eq(0.0)
    profile.root_profile.union_seconds.should eq(3.0)
    profile.root_profile.overlap_seconds.should eq(1.0)
  end

  it "rejects gaps or uncovered roots in operation attribution" do
    roots = [
      ML::Video::MPSCommandRootTiming.new(0, true, false, 0.0, 1.0),
      ML::Video::MPSCommandRootTiming.new(1, true, false, 1.0, 2.0),
    ]
    gap = [
      ML::Video::DCAEVDecoderOperationTiming.new("first", "norm", "stage_5", 0.0, 1, 1),
      ML::Video::DCAEVDecoderOperationTiming.new("second", "shuffle", "stage_5", 0.0, 2, 2),
    ]
    expect_raises(ArgumentError, /root cursors/) do
      ML::Video::DCAEVDecoderProfile.build(gap, roots, 2.0)
    end

    uncovered = [
      ML::Video::DCAEVDecoderOperationTiming.new("first", "norm", "stage_5", 0.0, 1, 1),
    ]
    expect_raises(ArgumentError, /cover the fresh chain/) do
      ML::Video::DCAEVDecoderProfile.build(uncovered, roots, 2.0)
    end
  end

  it "rejects reused operation labels across temporal tiles" do
    roots = [ML::Video::MPSCommandRootTiming.new(0, true, false, 0.0, 1.0)]
    repeated = Array.new(2) do
      ML::Video::DCAEVDecoderOperationTiming.new("norm", "norm", "stage_2", 0.0, 1, 1)
    end
    expect_raises(ArgumentError, /unique/) do
      ML::Video::DCAEVDecoderProfile.build(repeated, roots, 1.0)
    end
    distinct = [
      ML::Video::DCAEVDecoderOperationTiming.new("tile0/norm", "norm", "stage_2", 0.0, 1, 1),
      ML::Video::DCAEVDecoderOperationTiming.new("tile1/norm", "norm", "stage_2", 0.0, 1, 1),
    ]
    profile = ML::Video::DCAEVDecoderProfile.build(distinct, roots, 1.0)
    profile.group_definite_root_duration_sum_seconds["stage_2/norm"].should eq(1.0)
  end
end
