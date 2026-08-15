require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_native_block"
require "../src/ml/gguf/qwen_qbit_native_writer"
require "../src/ml/gguf/qwen_qbit_metal_restore"

describe "QBit Native restore" do
  codec = ML::GGUF::QwenQBitGaussianCodec
  writer = ML::GGUF::QwenQBitNativeWriter
  parser = ML::GGUF::QwenQBitNativeBlock
  restore = ML::GGUF::QwenQBitMetalRestore

  it "parses and validates the exact revision-zero column layout" do
    first = codec.encode(Array(Float32).new(17) { |i| (i - 8).to_f32 / 3.0_f32 }, block_size: 8, precision: 7)
    second = codec.encode(Array(Float32).new(9) { |i| -(i.to_f32 / 5.0_f32) }, block_size: 8, precision: 7)
    bytes = writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 0_u8, first),
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 1_u8, second),
    ])

    block = parser.parse(bytes)
    block.block_size.should eq(8)
    block.row_count.should eq(5)
    block.record_spans.map { |span| {span.cache_id, span.layer, span.kind, span.row_start, span.tile_count, span.value_count} }.should eq([
      {42_u64, 3_i32, 0_u8, 0_i32, 3_i32, 17_i32},
      {42_u64, 3_i32, 1_u8, 3_i32, 2_i32, 9_i32},
    ])

    trailing = Bytes.new(bytes.size + 1)
    trailing[0, bytes.size].copy_from(bytes)
    expect_raises(ArgumentError, /trailing/) { parser.parse(trailing) }

    nonzero_lsb = bytes.dup
    nonzero_lsb[block.codes_offset + 7 * block.row_count * (block.block_size // 8)] = 1_u8
    expect_raises(ArgumentError, /LSB/) { parser.parse(nonzero_lsb) }

    nonfinite_moment = bytes.dup
    nan_bits = Float32::NAN.unsafe_as(UInt32)
    4.times { |i| nonfinite_moment[block.mean_offset + i] = ((nan_bits >> (i * 8)) & 0xff).to_u8 }
    expect_raises(ArgumentError, /moments/) { parser.parse(nonfinite_moment) }
  end

  it "restores columnar Native p7 streams into multiple Metal state buffers" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    first_values = Array(Float32).new(1031) { |i| ((i % 71) - 35).to_f32 / 9.0_f32 }
    second_values = Array(Float32).new(257) { |i| i.even? ? -6.25_f32 : 4.75_f32 }
    first = codec.encode(first_values, block_size: 1024, precision: 7)
    second = codec.encode(second_values, block_size: 1024, precision: 7)
    block = parser.parse(writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(7_u64, 2_i32, 0_u8, first),
      ML::GGUF::QwenQBitNativeWriter::Record.new(7_u64, 2_i32, 1_u8, second),
    ]))

    first_dst = ML::MetalBuffer.new(first.value_count.to_i64 * sizeof(Float32))
    second_dst = ML::MetalBuffer.new(second.value_count.to_i64 * sizeof(Float32))
    live_before = ML::MetalBuffer.stats[:live_bytes]
    restore.decode_native_into(block, [
      ML::GGUF::QwenQBitMetalRestore::NativeJob.new(block.record_spans[0], first_dst),
      ML::GGUF::QwenQBitMetalRestore::NativeJob.new(block.record_spans[1], second_dst),
    ])
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

    first_dst.read(first.value_count).each_with_index { |value, i| value.should be_close(codec.decode(first)[i], 2e-6_f32) }
    second_dst.read(second.value_count).each_with_index { |value, i| value.should be_close(codec.decode(second)[i], 2e-6_f32) }
    first_dst.release
    second_dst.release
  end
end
