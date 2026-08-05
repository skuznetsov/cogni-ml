require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"
require "../../../src/ml/three_d/trellis2/shape_slat_upsample_layout"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/modules/sparse/spatial/basic.py::SparseUpsample expands each
# active 2x2x2 subdivision slot in parent order and preserves that order.

describe "TRELLIS.2 shape SLat upsample layout" do
  it "expands active 2x2x2 slots in source order with parent mappings" do
    coordinates = [
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 1, 2, 3),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(1, 4, 5, 6),
    ]
    subdivisions = [
      [true, false, false, true, false, true, false, false],
      [false, false, true, false, false, false, false, true],
    ]

    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      coordinates,
      subdivisions
    )

    layout.factor.should eq(2_i32)
    layout.coordinates.should eq([
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 2, 4, 6),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 3, 5, 6),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 3, 4, 7),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(1, 8, 11, 12),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(1, 9, 11, 13),
    ])
    layout.parent_indices.should eq([0_i32, 0_i32, 0_i32, 1_i32, 1_i32])
    layout.subindices.should eq([0_i32, 3_i32, 5_i32, 2_i32, 7_i32])
    layout.child_count.should eq(5_i32)
  end

  it "preserves duplicate children and permits an empty subdivision" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 7, 8, 9)
    subdivisions = [
      [false, true, false, false, false, false, false, false],
      [false, true, false, false, false, false, false, false],
    ]

    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate, coordinate],
      subdivisions
    )
    layout.coordinates.should eq([
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 15, 16, 18),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 15, 16, 18),
    ])
    layout.parent_indices.should eq([0_i32, 1_i32])
    layout.subindices.should eq([1_i32, 1_i32])

    empty = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [coordinate],
      [Array(Bool).new(8, false)]
    )
    empty.coordinates.should be_empty
    empty.parent_indices.should be_empty
    empty.subindices.should be_empty
  end

  it "uses little-endian slot bits for all eight child offsets" do
    layout = ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
      [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 10, 20, 30)],
      [Array(Bool).new(8, true)]
    )

    layout.coordinates.should eq([
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 20, 40, 60),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 21, 40, 60),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 20, 41, 60),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 21, 41, 60),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 20, 40, 61),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 21, 40, 61),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 20, 41, 61),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(2, 21, 41, 61),
    ])
    layout.parent_indices.should eq(Array(Int32).new(8, 0_i32))
    layout.subindices.should eq((0_i32..7_i32).to_a)
  end

  it "rejects malformed subdivisions, coordinates, and output budgets" do
    coordinate = ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0)

    expect_raises(ArgumentError, /one mask per coordinate/) do
      ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
        [coordinate],
        [] of Array(Bool)
      )
    end

    expect_raises(ArgumentError, /exactly 8/) do
      ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
        [coordinate],
        [[true, false]]
      )
    end

    expect_raises(ArgumentError, /non-negative/) do
      ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
        [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, -1, 0, 0)],
        [Array(Bool).new(8, false)]
      )
    end

    expect_raises(ArgumentError, /output coordinate budget/) do
      ML::ThreeD::Trellis2::ShapeSlatUpsampleLayoutCPU.expand(
        [coordinate],
        [[true, false, false, false, false, false, false, false]],
        max_output_coordinates: 0_i32
      )
    end
  end
end
