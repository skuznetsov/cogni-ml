require "../../spec_helper"
require "../../../src/ml/three_d/trellis2/layout"

private def f16_words(values : Enumerable(UInt16)) : Bytes
  values = values.to_a
  bytes = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 2, 2])
  end
  bytes
end

private def raw_tensor(
  name : String,
  shape : Array(Int64),
  values : Enumerable(UInt16),
) : ML::ThreeD::Trellis2::RawTensor
  ML::ThreeD::Trellis2::RawTensor.new(
    name,
    "F16",
    shape,
    f16_words(values)
  )
end

describe ML::ThreeD::Trellis2::Layout do
  it "preserves exact bytes for identity" do
    source = raw_tensor("source", [2_i64, 2_i64], 0_u16..3_u16)
    converted = ML::ThreeD::Trellis2::Layout.identity(source, "destination")

    converted.name.should eq("destination")
    converted.shape.should eq([2_i64, 2_i64])
    converted.bytes.should eq(source.bytes)
  end

  it "transposes two axes in row-major element order" do
    source = raw_tensor("matrix", [2_i64, 3_i64], 0_u16..5_u16)
    converted = ML::ThreeD::Trellis2::Layout.transpose(
      source,
      "matrix_t",
      0,
      1
    )

    converted.shape.should eq([3_i64, 2_i64])
    converted.bytes.should eq(f16_words([0_u16, 3_u16, 1_u16, 4_u16, 2_u16, 5_u16]))
  end

  it "round-trips a general permutation exactly" do
    source = raw_tensor("cube", [2_i64, 2_i64, 3_i64], 0_u16..11_u16)
    permuted = ML::ThreeD::Trellis2::Layout.permute(
      source,
      "cube_p",
      [2, 0, 1]
    )
    restored = ML::ThreeD::Trellis2::Layout.permute(
      permuted,
      "cube_restored",
      [1, 2, 0]
    )

    permuted.shape.should eq([3_i64, 2_i64, 2_i64])
    restored.shape.should eq(source.shape)
    restored.bytes.should eq(source.bytes)
  end

  it "round-trips split and concat along a non-leading axis" do
    source = raw_tensor("wide", [2_i64, 4_i64], 0_u16..7_u16)
    parts = ML::ThreeD::Trellis2::Layout.split(
      source,
      ["left", "right"],
      1,
      [1_i64, 3_i64]
    )

    parts.map(&.shape).should eq([
      [2_i64, 1_i64],
      [2_i64, 3_i64],
    ])
    parts[0].bytes.should eq(f16_words([0_u16, 4_u16]))
    parts[1].bytes.should eq(f16_words([1_u16, 2_u16, 3_u16, 5_u16, 6_u16, 7_u16]))

    restored = ML::ThreeD::Trellis2::Layout.concat(
      parts,
      "wide_restored",
      1
    )
    restored.shape.should eq(source.shape)
    restored.bytes.should eq(source.bytes)
  end

  it "preserves zero-element tensors without inventing payload" do
    source = raw_tensor("empty", [0_i64, 3_i64], [] of UInt16)
    converted = ML::ThreeD::Trellis2::Layout.permute(
      source,
      "empty_p",
      [1, 0]
    )

    converted.shape.should eq([3_i64, 0_i64])
    converted.bytes.should be_empty
  end

  it "rejects invalid permutations and split sizes" do
    source = raw_tensor("matrix", [2_i64, 3_i64], 0_u16..5_u16)

    expect_raises(
      ML::ThreeD::Trellis2::LayoutError,
      /permutation/
    ) do
      ML::ThreeD::Trellis2::Layout.permute(
        source,
        "bad",
        [0, 0]
      )
    end

    expect_raises(
      ML::ThreeD::Trellis2::LayoutError,
      /split sizes/
    ) do
      ML::ThreeD::Trellis2::Layout.split(
        source,
        ["a", "b"],
        1,
        [1_i64, 1_i64]
      )
    end
  end

  it "rejects incompatible concat inputs" do
    left = raw_tensor("left", [2_i64, 1_i64], 0_u16..1_u16)
    wrong_shape = raw_tensor("right", [3_i64, 1_i64], 2_u16..4_u16)

    expect_raises(
      ML::ThreeD::Trellis2::LayoutError,
      /concat.*shape/
    ) do
      ML::ThreeD::Trellis2::Layout.concat(
        [left, wrong_shape],
        "bad",
        1
      )
    end
  end

  it "rejects shape arithmetic overflow before allocation" do
    expect_raises(
      ML::ThreeD::Trellis2::LayoutError,
      /shape overflow/
    ) do
      ML::ThreeD::Trellis2::RawTensor.new(
        "overflow",
        "F16",
        [Int64::MAX, 2_i64],
        Bytes.empty
      )
    end
  end
end
