require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_c2s_index_layout"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/modules/sparse/spatial/spatial2channel.py::SparseChannel2Spatial
# flattens x.feats to [N*8, C] and selects `idx * 8 + subidx`.

describe "TRELLIS.2 shape SLat C2S packed index layout" do
  it "derives source rows in parent/slot order and preserves output width" do
    coordinates = [
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 1, 2, 3),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 4, 5, 6),
    ]
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      coordinates,
      [
        [true, false, false, false, false, false, false, true],
        [false, true, false, false, false, false, true, false],
      ]
    )

    plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      layout,
      input_point_count: 2_i32,
      packed_channels: 16_i32
    )

    plan.factor.should eq(2_i32)
    plan.slots_per_parent.should eq(8_i32)
    plan.output_channels.should eq(2_i32)
    plan.packed_source_indices.should eq([0_i32, 7_i32, 9_i32, 14_i32])
    plan.child_count.should eq(4_i32)
  end

  it "preserves duplicates and accepts an empty child layout" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 7, 8, 9)
    duplicate_layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate, coordinate],
      [
        [false, true, false, false, false, false, false, false],
        [false, true, false, false, false, false, false, false],
      ]
    )
    duplicate_plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      duplicate_layout,
      input_point_count: 2_i32,
      packed_channels: 8_i32
    )
    duplicate_plan.packed_source_indices.should eq([1_i32, 9_i32])

    empty_layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [Array(Bool).new(8, false)]
    )
    empty_plan = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      empty_layout,
      input_point_count: 1_i32,
      packed_channels: 8_i32
    )
    empty_plan.packed_source_indices.should be_empty
    empty_plan.child_count.should eq(0_i32)
  end

  it "rejects invalid factors, packed widths, mappings, and metadata budgets" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [[true, false, false, false, false, false, false, false]]
    )

    expect_raises(ArgumentError, /factor/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
        ML::ThreeD::Trellis2::ShapeSlatUpsampleLayout.new(
          4_i32,
          layout.coordinates,
          layout.parent_indices,
          layout.subindices
        ),
        input_point_count: 1_i32,
        packed_channels: 8_i32
      )
    end

    expect_raises(ArgumentError, /multiple of 8/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
        layout,
        input_point_count: 1_i32,
        packed_channels: 10_i32
      )
    end

    expect_raises(ArgumentError, /parent index/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
        ML::ThreeD::Trellis2::ShapeSlatUpsampleLayout.new(
          2_i32,
          layout.coordinates,
          [1_i32],
          layout.subindices
        ),
        input_point_count: 1_i32,
        packed_channels: 8_i32
      )
    end

    expect_raises(ArgumentError, /subdivision slot/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
        ML::ThreeD::Trellis2::ShapeSlatUpsampleLayout.new(
          2_i32,
          layout.coordinates,
          layout.parent_indices,
          [8_i32]
        ),
        input_point_count: 1_i32,
        packed_channels: 8_i32
      )
    end

    expect_raises(ArgumentError, /coordinates must be non-negative/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
        ML::ThreeD::Trellis2::ShapeSlatUpsampleLayout.new(
          2_i32,
          [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, -1, 0, 0)],
          layout.parent_indices,
          layout.subindices
        ),
        input_point_count: 1_i32,
        packed_channels: 8_i32
      )
    end

    expect_raises(ArgumentError, /metadata byte budget/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
        layout,
        input_point_count: 1_i32,
        packed_channels: 8_i32,
        max_metadata_bytes: 3_i64
      )
    end

    accepted = ML::ThreeD::Trellis2::ShapeSlatC2SIndexLayoutCPU.derive(
      layout,
      input_point_count: 1_i32,
      packed_channels: 8_i32,
      max_metadata_bytes: 4_i64
    )
    accepted.packed_source_indices.should eq([0_i32])
  end
end
