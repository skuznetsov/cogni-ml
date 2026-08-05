require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_subdivision_mask"
require "../../../src/ml/three_d/trellis2/shape_slat_subdivision_projection"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# SparseResBlockC2S3d first returns raw SparseLinear(C, 8) logits and then
# derives a separate strict `subdiv.feats > 0` mask for C2S expansion.

describe "TRELLIS.2 shape SLat subdivision projection and mask" do
  it "keeps raw logits and composes the strict mask in source order" do
    features = [
      1.0_f32, 2.0_f32, 3.0_f32,
      4.0_f32, 5.0_f32, 6.0_f32,
    ]
    weights = [
      1.0_f32, 2.0_f32, 3.0_f32,
      -1.0_f32, 0.0_f32, 2.0_f32,
      2.0_f32, -1.0_f32, 0.0_f32,
      0.0_f32, 1.0_f32, -1.0_f32,
      1.0_f32, -1.0_f32, 1.0_f32,
      -2.0_f32, 1.0_f32, 0.0_f32,
      0.0_f32, 0.0_f32, 1.0_f32,
      3.0_f32, -2.0_f32, -1.0_f32,
    ]
    bias = [
      -6.0_f32, 0.5_f32, -1.0_f32, 2.0_f32,
      0.0_f32, -3.0_f32, -4.0_f32, 1.0_f32,
    ]
    features_before = features.dup
    weights_before = weights.dup
    bias_before = bias.dup

    result = ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project_and_binarize(
      features,
      weights,
      bias,
      input_point_count: 2_i32,
      input_channels: 3_i32
    )

    result[0].should eq([
      8.0_f32, 5.5_f32, -1.0_f32, 1.0_f32,
      2.0_f32, -3.0_f32, -1.0_f32, -3.0_f32,
      26.0_f32, 8.5_f32, 2.0_f32, 1.0_f32,
      5.0_f32, -6.0_f32, 2.0_f32, -3.0_f32,
    ])
    result[1].should eq([
      [true, true, false, true, true, false, false, false],
      [true, true, true, true, true, false, true, false],
    ])

    direct_masks = ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
      result[0],
      input_point_count: 2_i32
    )
    result[1].should eq(direct_masks)
    features.should eq(features_before)
    weights.should eq(weights_before)
    bias.should eq(bias_before)
  end

  it "returns both empty raw logits and an empty mask for zero rows" do
    result = ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project_and_binarize(
      [] of Float32,
      [0.0_f32] * 16,
      [0.0_f32] * 8,
      input_point_count: 0_i32,
      input_channels: 2_i32
    )

    result[0].should be_empty
    result[1].should be_empty
  end

  it "preflights the combined resident budget and delegates finite guards" do
    features = [1.0_f32, 2.0_f32, 3.0_f32]
    weights = [1.0_f32] * 24
    bias = [0.0_f32] * 8

    # 12 feature + 96 weight + 32 bias + 32 logits + 8 bool-mask bytes.
    expect_raises(ArgumentError, /projection.*mask.*resident byte budget/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project_and_binarize(
        features,
        weights,
        bias,
        input_point_count: 1_i32,
        input_channels: 3_i32,
        max_resident_bytes: 179_i64
      )
    end

    accepted = ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project_and_binarize(
      features,
      weights,
      bias,
      input_point_count: 1_i32,
      input_channels: 3_i32,
      max_resident_bytes: 180_i64
    )
    accepted[0].size.should eq(8)
    accepted[1].should eq([[true, true, true, true, true, true, true, true]])

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project_and_binarize(
        [Float32::NAN, 0.0_f32, 0.0_f32],
        weights,
        bias,
        input_point_count: 1_i32,
        input_channels: 3_i32
      )
    end
  end
end
