require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_metal_restore"

describe ML::GGUF::QwenQBitMetalRestore do
  codec = ML::GGUF::QwenQBitGaussianCodec
  restore = ML::GGUF::QwenQBitMetalRestore

  it "fuses p7 plane decode and affine restore into a Metal buffer" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    values = Array(Float32).new(1031) do |i|
      case i % 7
      when 0 then 0.0_f32
      when 1 then -0.0_f32
      when 2 then 12.0_f32
      when 3 then -12.0_f32
      else ((i % 97) - 48).to_f32 / 11.0_f32
      end
    end
    encoded = codec.encode(values, block_size: 1024, precision: 7)
    expected = codec.decode(encoded)

    live_before = ML::MetalBuffer.stats[:live_bytes]
    buffer = restore.decode(encoded)
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before + buffer.size)
    actual = buffer.read(encoded.value_count)
    actual.each_with_index do |value, i|
      value.should be_close(expected[i], 2e-6_f32)
    end
    buffer.release
  end

  it "rejects malformed or non-p7 payloads before dispatch" do
    malformed = ML::GGUF::QwenQBitGaussianCodec::Encoded.new(8, 8, 7, Bytes.new(1, 0_u8))
    expect_raises(ArgumentError, /payload/) { restore.decode(malformed) }

    p8 = codec.encode([0.0_f32] * 8, block_size: 8, precision: 8)
    expect_raises(ArgumentError, /precision/) { restore.decode(p8) }
  end

  it "decodes multiple records in one Metal command buffer" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    a = codec.encode(Array(Float32).new(17) { |i| i.to_f32 / 7.0_f32 }, block_size: 8, precision: 7)
    b = codec.encode(Array(Float32).new(9) { |i| -i.to_f32 / 5.0_f32 }, block_size: 8, precision: 7)
    a_dst = ML::MetalBuffer.new(a.value_count.to_i64 * sizeof(Float32))
    b_dst = ML::MetalBuffer.new(b.value_count.to_i64 * sizeof(Float32))
    live_before = ML::MetalBuffer.stats[:live_bytes]
    restore.decode_into([
      ML::GGUF::QwenQBitMetalRestore::Job.new(a, a_dst),
      ML::GGUF::QwenQBitMetalRestore::Job.new(b, b_dst),
    ])
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

    a_dst.read(a.value_count).each_with_index { |value, i| value.should be_close(codec.decode(a)[i], 2e-6_f32) }
    b_dst.read(b.value_count).each_with_index { |value, i| value.should be_close(codec.decode(b)[i], 2e-6_f32) }
    a_dst.release
    b_dst.release
  end
end
