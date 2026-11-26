require "./spec_helper"
require "../src/ml/gguf/gemma4_weights"

private alias Gemma4LifecycleMetadata = String | Int32 | Array(Bool)

private def gemma4_lifecycle_write_string(io : IO, value : String) : Nil
  io.write_bytes(value.bytesize.to_u64, IO::ByteFormat::LittleEndian)
  io.write(value.to_slice)
end

private def gemma4_lifecycle_write_metadata(io : IO, value : Gemma4LifecycleMetadata) : Nil
  case value
  when String
    io.write_bytes(8_u32, IO::ByteFormat::LittleEndian)
    gemma4_lifecycle_write_string(io, value)
  when Int32
    io.write_bytes(5_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(value, IO::ByteFormat::LittleEndian)
  when Array(Bool)
    io.write_bytes(9_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(7_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(value.size.to_u64, IO::ByteFormat::LittleEndian)
    value.each { |item| io.write_byte(item ? 1_u8 : 0_u8) }
  end
end

private def gemma4_lifecycle_pad_to(io : IO, alignment : Int32) : Nil
  padding = ((alignment - io.pos % alignment) % alignment).to_i
  io.write(Bytes.new(padding, 0_u8)) if padding > 0
end

private def write_gemma4_lifecycle_fixture(path : String, *, include_block_count : Bool = true) : Nil
  metadata = [
    {"general.architecture", "gemma4".as(Gemma4LifecycleMetadata)},
    {"gemma4.block_count", 0.as(Gemma4LifecycleMetadata)},
    {"gemma4.embedding_length", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.feed_forward_length", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.context_length", 16.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.head_count", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.head_count_kv", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.key_length", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.value_length", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.key_length_swa", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.value_length_swa", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.sliding_window", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.attention.sliding_window_pattern", [] of Bool},
    {"gemma4.rope.dimension_count", 1.as(Gemma4LifecycleMetadata)},
    {"gemma4.rope.dimension_count_swa", 1.as(Gemma4LifecycleMetadata)},
  ]
  metadata.reject! { |key, _| key == "gemma4.block_count" } unless include_block_count

  io = IO::Memory.new
  io.write("GGUF".to_slice)
  io.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(2_i64, IO::ByteFormat::LittleEndian)
  io.write_bytes(metadata.size.to_i64, IO::ByteFormat::LittleEndian)
  metadata.each do |key, value|
    gemma4_lifecycle_write_string(io, key)
    gemma4_lifecycle_write_metadata(io, value)
  end

  [
    {"token_embd.weight", [1_i64, 1_i64], 0_u64},
    {"output_norm.weight", [1_i64], 4_u64},
  ].each do |name, dims, offset|
    gemma4_lifecycle_write_string(io, name)
    io.write_bytes(dims.size.to_u32, IO::ByteFormat::LittleEndian)
    dims.each { |dimension| io.write_bytes(dimension, IO::ByteFormat::LittleEndian) }
    io.write_bytes(0_u32, IO::ByteFormat::LittleEndian) # F32
    io.write_bytes(offset, IO::ByteFormat::LittleEndian)
  end

  gemma4_lifecycle_pad_to(io, 32)
  io.write_bytes(1.0_f32, IO::ByteFormat::LittleEndian)
  io.write_bytes(1.0_f32, IO::ByteFormat::LittleEndian)
  io.write(Bytes.new(16_384 - io.size, 0_u8)) if io.size < 16_384
  File.open(path, "wb") { |file| file.write(io.to_slice) }
end

private def gemma4_lifecycle_mapped_bytes(path : String) : UInt64
  if File.exists?("/proc/self/maps")
    return File.read("/proc/self/maps").lines
      .select(&.includes?(path))
      .reduce(0_u64) { |total, line| total + gemma4_lifecycle_range_bytes(line.split[0]) }
  end

  output = IO::Memory.new
  error = IO::Memory.new
  status = Process.run("/usr/bin/vmmap", [Process.pid.to_s], output: output, error: error)
  raise "vmmap failed: #{error}" unless status.success?
  output.to_s.lines
    .select(&.includes?(path))
    .reduce(0_u64) { |total, line| total + gemma4_lifecycle_range_bytes(line.split[2]) }
end

private def gemma4_lifecycle_range_bytes(range : String) : UInt64
  dash = range.index('-').not_nil!
  start_address = range[0...dash].to_u64(16)
  end_address = range[(dash + 1)..].to_u64(16)
  end_address - start_address
end

describe "Gemma4 weight mmap lifetime" do
  {% unless flag?(:cpu_only) %}
    it "runs the synthetic mmap checks only in CPU-only mode" do
      pending!("run with -Dcpu_only to avoid Metal initialization")
    end
  {% else %}
    it "releases its GGUF mapping on explicit close" do
      path = File.tempname("gemma4-lifecycle", ".gguf")
      weights : ML::GGUF::Gemma4Weights? = nil

      begin
        write_gemma4_lifecycle_fixture(path)
        File.size(path).should be <= 1_048_576

        weights = ML::GGUF::Gemma4Weights.from_gguf(path)
        weights.not_nil!.token_embd.raw.size.should eq(4)
        gemma4_lifecycle_mapped_bytes(path).should be > 0_u64

        weights.not_nil!.close
        weights.not_nil!.close
        gemma4_lifecycle_mapped_bytes(path).should eq(0_u64)
      ensure
        weights.try(&.close)
        File.delete(path) if File.exists?(path)
      end
    end

    it "releases its GGUF mapping when weight initialization fails" do
      path = File.tempname("gemma4-lifecycle-failure", ".gguf")

      begin
        write_gemma4_lifecycle_fixture(path, include_block_count: false)
        File.size(path).should be <= 1_048_576

        expect_raises(Exception, /gemma4.block_count/) do
          ML::GGUF::Gemma4Weights.from_gguf(path)
        end
        gemma4_lifecycle_mapped_bytes(path).should eq(0_u64)
      ensure
        File.delete(path) if File.exists?(path)
      end
    end
  {% end %}
end
