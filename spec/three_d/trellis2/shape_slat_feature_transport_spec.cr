require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_feature_transport"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/modules/sparse/spatial/basic.py::SparseUpsample uses
# `new_feats = x.feats[idx]` after the subdivision-derived parent index list.

describe "TRELLIS.2 shape SLat feature transport" do
  it "gathers parent feature rows in the upsample layout order" do
    coordinates = [
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 1, 2, 3),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 4, 5, 6),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 7, 8, 9),
    ]
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      coordinates,
      [
        [true, false, false, false, false, true, false, false],
        [false, false, true, false, false, false, false, false],
        Array(Bool).new(8, false),
      ]
    )

    input = [
      1.0_f32, 10.0_f32,
      2.0_f32, 20.0_f32,
      3.0_f32, 30.0_f32,
    ]
    output = ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
      layout,
      input,
      input_point_count: 3_i32,
      channels: 2_i32
    )

    output.should eq([
      1.0_f32, 10.0_f32,
      1.0_f32, 10.0_f32,
      2.0_f32, 20.0_f32,
    ])
    input.should eq([
      1.0_f32, 10.0_f32,
      2.0_f32, 20.0_f32,
      3.0_f32, 30.0_f32,
    ])
  end

  it "returns an independent empty payload when no child slot is active" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [Array(Bool).new(8, false)]
    )

    output = ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
      layout,
      [4.0_f32, 5.0_f32],
      input_point_count: 1_i32,
      channels: 2_i32
    )

    output.should be_empty
  end

  it "rejects malformed payloads, non-finite values, mappings, and budgets" do
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayout.new(
      2_i32,
      [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)],
      [1_i32],
      [0_i32]
    )

    expect_raises(ArgumentError, /point count/) do
      ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
        layout,
        [1.0_f32, 2.0_f32],
        input_point_count: 0_i32,
        channels: 2_i32
      )
    end

    expect_raises(ArgumentError, /channel count/) do
      ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
        layout,
        [] of Float32,
        input_point_count: 0_i32,
        channels: 0_i32
      )
    end

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
        ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
          [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)],
          [[true, false, false, false, false, false, false, false]]
        ),
        [Float32::NAN],
        input_point_count: 1_i32,
        channels: 1_i32
      )
    end

    expect_raises(ArgumentError, /parent index/) do
      ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
        layout,
        [1.0_f32, 2.0_f32],
        input_point_count: 1_i32,
        channels: 2_i32
      )
    end

    valid_layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)],
      [[true, false, false, false, false, false, false, false]]
    )
    expect_raises(ArgumentError, /feature byte budget/) do
      ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
        valid_layout,
        [1.0_f32, 2.0_f32],
        input_point_count: 1_i32,
        channels: 2_i32,
        max_feature_bytes: 4_i64
      )
    end

    expect_raises(ArgumentError, /resident bytes/) do
      ML::ThreeD::Trellis2::ShapeSlatFeatureTransportCPU.gather(
        valid_layout,
        [1.0_f32, 2.0_f32],
        input_point_count: 1_i32,
        channels: 2_i32,
        max_feature_bytes: 12_i64
      )
    end
  end
end
