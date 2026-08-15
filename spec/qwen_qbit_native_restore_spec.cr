require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_native_block"
require "../src/ml/gguf/qwen_qbit_native_writer"
require "../src/ml/gguf/qwen_qbit_metal_restore"

private def qbit_native_concat(parts : Array(Bytes)) : Bytes
  bytes = Bytes.new(parts.sum(&.size))
  offset = 0
  parts.each do |part|
    bytes[offset, part.size].copy_from(part)
    offset += part.size
  end
  bytes
end

private def qbit_native_rebase_tiles(bytes : Bytes, start_tile : UInt32, tile_count : Int32) : Bytes
  rebased = bytes.dup
  marker = "\u0004tile\u0006UInt32".to_slice
  marker_offset = nil.as(Int32?)
  (0..(rebased.size - marker.size)).each do |offset|
    if rebased[offset, marker.size] == marker
      marker_offset = offset
      break
    end
  end
  raise "Native tile marker not found" unless marker_offset
  tile_offset = marker_offset.not_nil! + marker.size
  tile_count.times do |i|
    value = start_tile + i.to_u32
    4.times { |byte| rebased[tile_offset + i * 4 + byte] = ((value >> (byte * 8)) & 0xff).to_u8 }
  end
  rebased
end

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

  it "validates a record split across ordered Native response blocks" do
    first = codec.encode(Array(Float32).new(16) { |i| i.to_f32 / 7.0_f32 }, block_size: 8, precision: 7)
    second = codec.encode(Array(Float32).new(9) { |i| -(i.to_f32 / 5.0_f32) }, block_size: 8, precision: 7)
    first_block = writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 0_u8, first),
    ])
    second_block = qbit_native_rebase_tiles(writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 0_u8, second),
    ]), 2_u32, 2)

    stream = parser.parse_stream(qbit_native_concat([first_block, second_block]))
    stream.block_size.should eq(8)
    stream.row_count.should eq(4)
    stream.blocks.map(&.row_count).should eq([2, 2])
    stream.record_spans.map { |span| {span.cache_id, span.layer, span.kind, span.tile_count, span.value_count, span.chunks.size} }.should eq([
      {42_u64, 3_i32, 0_u8, 4_i32, 25_i32, 2},
    ])
    stream.record_spans[0].chunks.map { |chunk| {chunk.block_index, chunk.row_start, chunk.tile_start, chunk.tile_count, chunk.value_start, chunk.value_count} }.should eq([
      {0_i32, 0_i32, 0_i32, 2_i32, 0_i32, 16_i32},
      {1_i32, 0_i32, 2_i32, 2_i32, 16_i32, 9_i32},
    ])

    skipped_tile = qbit_native_rebase_tiles(second_block, 3_u32, 2)
    expect_raises(ArgumentError, /tile sequence/) do
      parser.parse_stream(qbit_native_concat([first_block, skipped_tile]))
    end

    partial_first = writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 0_u8, second),
    ])
    expect_raises(ArgumentError, /non-final tile/) do
      parser.parse_stream(qbit_native_concat([partial_first, second_block]))
    end

    different_width = writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(
        43_u64,
        3_i32,
        0_u8,
        codec.encode(Array(Float32).new(16, 1.0_f32), block_size: 16, precision: 7),
      ),
    ])
    expect_raises(ArgumentError, /block size mismatch/) do
      parser.parse_stream(qbit_native_concat([first_block, different_width]))
    end
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

  it "restores a record split across Native response blocks into one Metal buffer" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    first = codec.encode(Array(Float32).new(16) { |i| i.to_f32 / 7.0_f32 }, block_size: 8, precision: 7)
    second = codec.encode(Array(Float32).new(9) { |i| -(i.to_f32 / 5.0_f32) }, block_size: 8, precision: 7)
    first_block = writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 0_u8, first),
    ])
    second_block = qbit_native_rebase_tiles(writer.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 0_u8, second),
    ]), 2_u32, 2)
    stream = parser.parse_stream(qbit_native_concat([first_block, second_block]))

    destination = ML::MetalBuffer.new(25_i64 * sizeof(Float32))
    live_before = ML::MetalBuffer.stats[:live_bytes]
    restore.decode_native_stream_into(stream, [
      ML::GGUF::QwenQBitMetalRestore::NativeStreamJob.new(stream.record_spans[0], destination),
    ])
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

    expected = codec.decode(first) + codec.decode(second)
    destination.read(expected.size).each_with_index do |value, i|
      value.should be_close(expected[i], 2e-6_f32)
    end
    destination.release
  end
end
