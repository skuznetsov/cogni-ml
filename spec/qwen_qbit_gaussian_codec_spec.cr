require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"

describe ML::GGUF::QwenQBitGaussianCodec do
  codec = ML::GGUF::QwenQBitGaussianCodec

  it "packs the declared number of Gaussian code planes" do
    values = Array(Float32).new(1030) { |i| ((i % 41) - 20).to_f32 / 7.0_f32 }

    p8 = codec.encode(values, block_size: 1024, precision: 8)
    p7 = codec.encode(values, block_size: 1024, precision: 7)
    p6 = codec.encode(values, block_size: 1024, precision: 6)

    p8.payload.size.should eq(codec.payload_size(values.size, 1024, 8))
    p7.payload.size.should eq(codec.payload_size(values.size, 1024, 7))
    p6.payload.size.should eq(codec.payload_size(values.size, 1024, 6))
    p6.payload.size.should be < p7.payload.size
    p7.payload.size.should be < p8.payload.size
  end

  it "uses one symmetric conditional mean for every retained prefix" do
    positive = codec.reconstruct_raw_code(0x2a_u8, 6)
    same_prefix = codec.reconstruct_raw_code(0x2b_u8, 6)
    mirrored = codec.reconstruct_raw_code(0xd5_u8, 6)

    same_prefix.should eq(positive)
    mirrored.should eq(-positive)
  end

  it "matches ClickHouse's generated p6 and p7 centroid bits" do
    codec.reconstruct_raw_code(0_u8, 6).unsafe_as(UInt32).should eq(0x3ca13bf4_u32)
    codec.reconstruct_raw_code(0_u8, 7).unsafe_as(UInt32).should eq(0x3c2137cb_u32)
  end

  it "packs p8 bytes in ClickHouse QBit subcolumn order" do
    values = Array(Float32).new(8) { |i| i.even? ? 1.0_f32 : -1.0_f32 }
    payload = codec.encode(values, block_size: 8, precision: 8).payload

    payload[8, 8].should eq(Bytes[0xaa, 0x55, 0xaa, 0x55, 0xaa, 0x55, 0x55, 0xaa])
  end

  it "reconstructs deterministic Gaussian-like data monotonically from p6 to p8" do
    values = Array(Float32).new(4096) do |i|
      # Deterministic Box-Muller coverage without making the test depend on a
      # random generator.
      u1 = (i + 0.5) / 4096.0
      u2 = (((i * 1543) % 4096) + 0.5) / 4096.0
      (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)).to_f32
    end

    errors = [6, 7, 8].map do |precision|
      decoded = codec.decode(codec.encode(values, block_size: 1024, precision: precision))
      values.each_with_index.sum(0.0_f64) do |value, i|
        delta = value.to_f64 - decoded[i].to_f64
        delta * delta
      end / values.size
    end

    errors[1].should be < errors[0]
    errors[2].should be < errors[1]
  end

  it "fails closed on unsupported precision and non-finite state" do
    expect_raises(ArgumentError, /precision/) do
      codec.encode([0.0_f32], block_size: 8, precision: 5)
    end
    expect_raises(ArgumentError, /finite/) do
      codec.encode([Float32::NAN], block_size: 8, precision: 7)
    end
    expect_raises(ArgumentError, /multiple of 8/) do
      codec.encode([0.0_f32], block_size: 7, precision: 7)
    end
  end
end
