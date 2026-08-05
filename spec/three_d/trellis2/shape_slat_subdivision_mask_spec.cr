require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_subdivision_mask"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/models/sc_vaes/sparse_unet_vae.py::SparseResBlockC2S3d (the
# up-block named by the shipped shape/texture SC-VAE configs) and the
# alternate SparseResBlockUpsample3d definition binarize predicted
# subdivision logits with the strict comparison `subdiv.feats > 0` before
# passing an 8-column mask to their spatial upsample operators.

describe "TRELLIS.2 shape SLat subdivision mask" do
  it "uses strict positive logits in parent and slot order" do
    logits = [
      1.0_f32, 0.0_f32, -1.0_f32, -0.0_f32,
      0.000001_f32, -0.000001_f32, 2.0_f32, -2.0_f32,
      -3.0_f32, 4.0_f32, 0.0_f32, 5.0_f32,
      -6.0_f32, 7.0_f32, -8.0_f32, 9.0_f32,
    ]
    before = logits.dup

    masks = ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
      logits,
      input_point_count: 2_i32
    )

    masks.should eq([
      [true, false, false, false, true, false, true, false],
      [false, true, false, true, false, true, false, true],
    ])
    logits.should eq(before)
  end

  it "returns an empty mask list for zero parent rows" do
    masks = ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
      [] of Float32,
      input_point_count: 0_i32
    )

    masks.should be_empty
  end

  it "rejects malformed, non-finite, and over-budget logits" do
    expect_raises(ArgumentError, /point count/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
        [] of Float32,
        input_point_count: -1_i32
      )
    end

    expect_raises(ArgumentError, /payload size/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
        [1.0_f32] * 8,
        input_point_count: 2_i32
      )
    end

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
        [Float32::NAN] + ([0.0_f32] * 7),
        input_point_count: 1_i32
      )
    end

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
        [Float32::INFINITY] + ([0.0_f32] * 7),
        input_point_count: 1_i32
      )
    end

    expect_raises(ArgumentError, /exceeds/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
        [] of Float32,
        input_point_count: ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU::MAX_INPUT_POINTS + 1_i32
      )
    end

    expect_raises(ArgumentError, /resident byte budget/) do
      ML::ThreeD::Trellis2::ShapeSlatSubdivisionMaskCPU.binarize(
        [1.0_f32] * 8,
        input_point_count: 1_i32,
        max_resident_bytes: 4_i64
      )
    end
  end
end
