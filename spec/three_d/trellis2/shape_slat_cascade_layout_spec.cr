require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_layout"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/pipelines/trellis2_image_to_3d.py quantizes decoder coordinates
# before the 1024/1536 cascade token-budget fallback.

describe "TRELLIS.2 shape SLat cascade layout" do
  it "quantizes, deduplicates, and sorts decoder coordinates" do
    coordinates = [
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(1, 511, 511, 511),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 16, 0, 0),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 16, 0, 0),
    ]

    quantized = ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.quantize(
      coordinates,
      lr_resolution: 512_i32,
      target_resolution: 1024_i32
    )

    quantized.should eq([
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 2, 0, 0),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(1, 63, 63, 63),
    ])
  end

  it "uses the source strict token threshold and stops at 1024" do
    coordinates = [
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 16, 0, 0),
    ]

    reduced = ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.select(
      coordinates,
      lr_resolution: 512_i32,
      requested_resolution: 1536_i32,
      max_num_tokens: 2_i32
    )
    reduced.requested_resolution.should eq(1536_i32)
    reduced.actual_resolution.should eq(1024_i32)
    reduced.token_count.should eq(2_i32)

    retained = ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.select(
      coordinates,
      lr_resolution: 512_i32,
      requested_resolution: 1536_i32,
      max_num_tokens: 3_i32
    )
    retained.actual_resolution.should eq(1536_i32)
    retained.token_count.should eq(2_i32)
    retained.coordinates.should eq([
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 0, 0, 0),
      ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 3, 0, 0),
    ])
    ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU::DEFAULT_MAX_NUM_TOKENS.should eq(49_152_i32)
  end

  it "rejects coordinates and resolutions outside the admitted cascade domain" do
    expect_raises(ArgumentError) do
      ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.quantize(
        [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, -1, 0, 0)],
        lr_resolution: 512_i32,
        target_resolution: 1024_i32
      )
    end

    expect_raises(ArgumentError) do
      ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.quantize(
        [ML::ThreeD::Trellis2::ShapeSlatCoordinate.new(0, 512, 0, 0)],
        lr_resolution: 512_i32,
        target_resolution: 1024_i32
      )
    end

    expect_raises(ArgumentError) do
      ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.select(
        [] of ML::ThreeD::Trellis2::ShapeSlatCoordinate,
        lr_resolution: 512_i32,
        requested_resolution: 960_i32,
        max_num_tokens: 2_i32
      )
    end

    expect_raises(ArgumentError) do
      ML::ThreeD::Trellis2::ShapeSlatCascadeLayoutCPU.select(
        [] of ML::ThreeD::Trellis2::ShapeSlatCoordinate,
        lr_resolution: 512_i32,
        requested_resolution: 1664_i32,
        max_num_tokens: 2_i32
      )
    end
  end
end
