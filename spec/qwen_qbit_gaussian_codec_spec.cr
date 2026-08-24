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

  it "supports ClickHouse-compatible p4 and p5 prefix centroids" do
    values = Array(Float32).new(256) { |i| ((i % 37) - 18).to_f32 / 5.0_f32 }
    p4 = codec.encode(values, block_size: 256, precision: 4)
    p5 = codec.encode(values, block_size: 256, precision: 5)

    p4.payload.size.should eq(codec.payload_size(values.size, 256, 4))
    p5.payload.size.should eq(codec.payload_size(values.size, 256, 5))
    p4.payload.size.should be < p5.payload.size
    p4_centroids = [
      0x3da18fb8_u32, 0x3e747262_u32, 0x3ecf6cea_u32, 0x3f158a3a_u32,
      0x3f491a06_u32, 0x3f8408fb_u32, 0x3fb45dcf_u32, 0x4007469a_u32,
    ]
    p5_centroids = [
      0x3d214c9e_u32, 0x3df27670_u32, 0x3e4aeb9c_u32, 0x3e8efa3f_u32,
      0x3eb97bb4_u32, 0x3ee558d1_u32, 0x3f0982c6_u32, 0x3f218b69_u32,
      0x3f3b2b35_u32, 0x3f56f4e8_u32, 0x3f75d8c1_u32, 0x3f8cda65_u32,
      0x3fa379d0_u32, 0x3fc3f223_u32, 0x3ff8a44f_u32, 0x4028a4fe_u32,
    ]
    p4_centroids.each_with_index do |expected, prefix|
      raw_code = (prefix << 4).to_u8
      codec.reconstruct_raw_code(raw_code, 4).unsafe_as(UInt32).should eq(expected)
    end
    p5_centroids.each_with_index do |expected, prefix|
      raw_code = (prefix << 3).to_u8
      codec.reconstruct_raw_code(raw_code, 5).unsafe_as(UInt32).should eq(expected)
    end
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

  it "reconstructs deterministic Gaussian-like data monotonically from p4 to p8" do
    values = Array(Float32).new(4096) do |i|
      # Deterministic Box-Muller coverage without making the test depend on a
      # random generator.
      u1 = (i + 0.5) / 4096.0
      u2 = (((i * 1543) % 4096) + 0.5) / 4096.0
      (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)).to_f32
    end

    errors = [4, 5, 6, 7, 8].map do |precision|
      decoded = codec.decode(codec.encode(values, block_size: 1024, precision: precision))
      values.each_with_index.sum(0.0_f64) do |value, i|
        delta = value.to_f64 - decoded[i].to_f64
        delta * delta
      end / values.size
    end

    errors.each_cons_pair do |lower, higher|
      higher.should be < lower
    end
  end

  it "fails closed on unsupported precision and non-finite state" do
    expect_raises(ArgumentError, /precision/) do
      codec.encode([0.0_f32], block_size: 8, precision: 3)
    end
    expect_raises(ArgumentError, /finite/) do
      codec.encode([Float32::NAN], block_size: 8, precision: 7)
    end
    expect_raises(ArgumentError, /multiple of 8/) do
      codec.encode([0.0_f32], block_size: 7, precision: 7)
    end
  end
end
