require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_subdivision_mask"
require "../../../src/ml/three_d/trellis2/shape_slat_subdivision_projection"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/modules/sparse/linear.py::SparseLinear is nn.Linear and therefore
# computes raw logits as x @ W.T + b; the decoder thresholds those logits only
# at the subsequent `subdiv.feats > 0` boundary.

describe "TRELLIS.2 shape SLat subdivision projection" do
  it "preserves row-major affine order and raw logits" do
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

    logits = ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
      features,
      weights,
      bias,
      input_point_count: 2_i32,
      input_channels: 3_i32
    )

    logits.should eq([
      8.0_f32, 5.5_f32, -1.0_f32, 1.0_f32,
      2.0_f32, -3.0_f32, -1.0_f32, -3.0_f32,
      26.0_f32, 8.5_f32, 2.0_f32, 1.0_f32,
      5.0_f32, -6.0_f32, 2.0_f32, -3.0_f32,
    ])
    features.should eq(features_before)
    weights.should eq(weights_before)
    bias.should eq(bias_before)

    masks = ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
      logits,
      input_point_count: 2_i32
    )
    masks.should eq([
      [true, true, false, true, true, false, false, false],
      [true, true, true, true, true, false, true, false],
    ])
  end

  it "returns an empty raw-logit payload for zero parent rows" do
    logits = ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
      [] of Float32,
      [0.0_f32] * 16,
      [0.0_f32] * 8,
      input_point_count: 0_i32,
      input_channels: 2_i32
    )

    logits.should be_empty
  end

  it "rejects malformed, non-finite, over-cap, and over-budget inputs" do
    expect_raises(ArgumentError, /payload size/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [1.0_f32] * 3,
        [1.0_f32] * 8,
        [0.0_f32] * 8,
        input_point_count: 2_i32,
        input_channels: 3_i32
      )
    end

    expect_raises(ArgumentError, /weight payload/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [1.0_f32] * 3,
        [1.0_f32] * 7,
        [0.0_f32] * 8,
        input_point_count: 1_i32,
        input_channels: 3_i32
      )
    end

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [Float32::NAN, 0.0_f32, 0.0_f32],
        [1.0_f32] * 24,
        [0.0_f32] * 8,
        input_point_count: 1_i32,
        input_channels: 3_i32
      )
    end

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [0.0_f32, 0.0_f32, 0.0_f32],
        [Float32::INFINITY] + ([0.0_f32] * 23),
        [0.0_f32] * 8,
        input_point_count: 1_i32,
        input_channels: 3_i32
      )
    end

    expect_raises(ArgumentError, /non-negative/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [] of Float32,
        [] of Float32,
        [] of Float32,
        input_point_count: -1_i32,
        input_channels: 3_i32
      )
    end

    expect_raises(ArgumentError, /exceeds/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [] of Float32,
        [] of Float32,
        [] of Float32,
        input_point_count: 1_i32,
        input_channels: ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU::MAX_INPUT_CHANNELS + 1_i32
      )
    end

    expect_raises(ArgumentError, /resident byte budget/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionProjectionCPU.project(
        [1.0_f32] * 3,
        [1.0_f32] * 24,
        [0.0_f32] * 8,
        input_point_count: 1_i32,
        input_channels: 3_i32,
        max_resident_bytes: 4_i64
      )
    end
  end
end
