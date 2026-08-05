require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_c2s_index_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_c2s_feature_transport"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/modules/sparse/spatial/spatial2channel.py::SparseChannel2Spatial
# selects `x.feats.reshape(N * 8, -1)[idx * 8 + subidx]`.

describe "TRELLIS.2 shape SLat C2S feature transport" do
  it "gathers distinct packed channel blocks in parent/slot order" do
    coordinates = [
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 1, 2, 3),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 4, 5, 6),
    ]
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      coordinates,
      [
        [false, true, false, false, false, false, false, false],
        [false, false, false, false, false, false, true, false],
      ]
    )
    plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      layout,
      input_point_count: 2_i32,
      packed_channels: 24_i32
    )

    packed_features = Array(Float32).new
    2.times do |parent|
      8.times do |slot|
        3.times do |channel|
          packed_features << (parent * 100 + slot * 10 + channel).to_f32
        end
      end
    end
    snapshot = packed_features.dup

    output = ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
      layout,
      plan,
      packed_features,
      input_point_count: 2_i32,
      packed_channels: 24_i32
    )

    plan.packed_source_indices.should eq([1_i32, 14_i32])
    output.should eq([
      10.0_f32, 11.0_f32, 12.0_f32,
      160.0_f32, 161.0_f32, 162.0_f32,
    ])
    packed_features.should eq(snapshot)
  end

  it "preserves duplicate source blocks and returns an empty payload" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 7, 8, 9)
    duplicate_layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayout.new(
      2_i32,
      [coordinate, coordinate],
      [0_i32, 0_i32],
      [1_i32, 1_i32]
    )
    duplicate_plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      duplicate_layout,
      input_point_count: 2_i32,
      packed_channels: 16_i32
    )
    packed_features = Array(Float32).new
    2.times do |parent|
      8.times do |slot|
        2.times do |channel|
          packed_features << (parent * 100 + slot * 10 + channel).to_f32
        end
      end
    end

    output = ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
      duplicate_layout,
      duplicate_plan,
      packed_features,
      input_point_count: 2_i32,
      packed_channels: 16_i32
    )
    output.should eq([10.0_f32, 11.0_f32, 10.0_f32, 11.0_f32])

    empty_layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [Array(Bool).new(8, false)]
    )
    empty_plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      empty_layout,
      input_point_count: 1_i32,
      packed_channels: 8_i32
    )
    empty_output = ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
      empty_layout,
      empty_plan,
      Array(Float32).new(8, 4.0_f32),
      input_point_count: 1_i32,
      packed_channels: 8_i32
    )
    empty_output.should be_empty
  end

  it "rejects payload, shape, mapping, and resident-budget violations" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [[true, false, false, false, false, false, false, false]]
    )
    plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      layout,
      input_point_count: 1_i32,
      packed_channels: 16_i32
    )
    valid_features = Array(Float32).new(16, 1.0_f32)

    expect_raises(ArgumentError, /payload size/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
        layout,
        plan,
        valid_features[0, 15],
        input_point_count: 1_i32,
        packed_channels: 16_i32
      )
    end

    expect_raises(ArgumentError, /output channel/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
        layout,
        plan,
        valid_features,
        input_point_count: 1_i32,
        packed_channels: 8_i32
      )
    end

    expect_raises(ArgumentError, /finite/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
        layout,
        plan,
        [Float32::NAN] + Array(Float32).new(15, 1.0_f32),
        input_point_count: 1_i32,
        packed_channels: 16_i32
      )
    end

    forged_plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayout.new(
      2_i32,
      8_i32,
      2_i32,
      [1_i32]
    )
    expect_raises(ArgumentError, /does not match source layout/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
        layout,
        forged_plan,
        valid_features,
        input_point_count: 1_i32,
        packed_channels: 16_i32
      )
    end

    mutable_layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [[true, false, false, false, false, false, false, false]]
    )
    mutable_plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      mutable_layout,
      input_point_count: 1_i32,
      packed_channels: 8_i32
    )
    mutable_layout.parent_indices << 0_i32
    expect_raises(ArgumentError, /mappings must match child count/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
        mutable_layout,
        mutable_plan,
        Array(Float32).new(8, 1.0_f32),
        input_point_count: 1_i32,
        packed_channels: 8_i32
      )
    end

    expect_raises(ArgumentError, /resident bytes/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
        layout,
        plan,
        valid_features,
        input_point_count: 1_i32,
        packed_channels: 16_i32,
        max_feature_bytes: 71_i64
      )
    end

    accepted = ML::ThreeD::Trellis2::ShapeSlatC2SFeatureTransportCPU.gather(
      layout,
      plan,
      valid_features,
      input_point_count: 1_i32,
      packed_channels: 16_i32,
      max_feature_bytes: 72_i64
    )
    accepted.should eq([1.0_f32, 1.0_f32])
  end
end
