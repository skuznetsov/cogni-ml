require "spec"
require "../../../src/ml/three_d/trellis2/shape_slat_c2s_scale"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# trellis2/modules/sparse/spatial/spatial2channel.py::SparseChannel2Spatial
# assigns every uncached output scale as `s / self.factor`.

describe "TRELLIS.2 shape SLat C2S scale metadata" do
  it "halves each exact source scale without narrowing to floating point" do
    input = ML::ThreeD::Trellis2::ShapeSlatScale.new(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(3_i64, 2_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(5_i64, 4_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(7_i64, 8_i64)
    )

    output = ML::ThreeD::Trellis2::ShapeSlatC2SScaleCPU.transition(input)

    output.x.should eq(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(3_i64, 4_i64)
    )
    output.y.should eq(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(5_i64, 8_i64)
    )
    output.z.should eq(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(7_i64, 16_i64)
    )
  end

  it "normalizes equivalent source fractions and halves unit scale" do
    normalized = ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(6_i64, 4_i64)
    normalized.numerator.should eq(3_i64)
    normalized.denominator.should eq(2_i64)

    unit = ML::ThreeD::Trellis2::ShapeSlatScale.new(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64)
    )
    output = ML::ThreeD::Trellis2::ShapeSlatC2SScaleCPU.transition(unit)

    output.x.should eq(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 2_i64)
    )
    output.y.should eq(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 2_i64)
    )
    output.z.should eq(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 2_i64)
    )
  end

  it "rejects a non-C2S factor and invalid exact scales" do
    scale = ML::ThreeD::Trellis2::ShapeSlatScale.new(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64)
    )

    expect_raises(ArgumentError, /factor/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SScaleCPU.transition(
        scale,
        factor: 4_i32
      )
    end
    expect_raises(ArgumentError, /numerator/) do
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(0_i64, 1_i64)
    end
    expect_raises(ArgumentError, /denominator/) do
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 0_i64)
    end
  end

  it "rejects denominator overflow before returning a transition" do
    near_limit = ML::ThreeD::Trellis2::ShapeSlatScale.new(
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, Int64::MAX),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64),
      ML::ThreeD::Trellis2::ShapeSlatScaleValue.new(1_i64, 1_i64)
    )

    expect_raises(ArgumentError, /overflow/) do
      ML::ThreeD::Trellis2::ShapeSlatC2SScaleCPU.transition(near_limit)
    end
  end
end
