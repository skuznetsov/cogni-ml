require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_state_snapshot"

private def put_gguf_string(io : IO, value : String) : Nil
  io.write_bytes(value.bytesize.to_u64, IO::ByteFormat::LittleEndian)
  io << value
end

private def portable_qwen35_fixture : String
  path = File.tempname("recent-qwen35-meta", ".gguf")
  io = IO::Memory.new
  io << "GGUF"
  io.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(0_i64, IO::ByteFormat::LittleEndian)
  entries = [
    {"general.architecture", 8_u32, "qwen35"},
  ]
  ints = [
    {"qwen35.block_count", 1},
    {"qwen35.embedding_length", 4},
    {"qwen35.feed_forward_length", 8},
    {"qwen35.context_length", 8},
    {"qwen35.attention.head_count", 1},
    {"qwen35.attention.head_count_kv", 1},
    {"qwen35.attention.key_length", 4},
    {"qwen35.attention.value_length", 4},
    {"qwen35.full_attention_interval", 1},
    {"qwen35.rope.dimension_count", 4},
    {"qwen35.ssm.conv_kernel", 2},
    {"qwen35.ssm.state_size", 4},
    {"qwen35.ssm.group_count", 1},
    {"qwen35.ssm.time_step_rank", 1},
    {"qwen35.ssm.inner_size", 4},
  ]
  floats = [
    {"qwen35.attention.layer_norm_rms_epsilon", 1.0e-6_f32},
    {"qwen35.rope.freq_base", 10000.0_f32},
  ]
  io.write_bytes((entries.size + ints.size + floats.size + 1).to_i64, IO::ByteFormat::LittleEndian)
  entries.each do |entry|
    put_gguf_string(io, entry[0])
    io.write_bytes(entry[1], IO::ByteFormat::LittleEndian)
    put_gguf_string(io, entry[2])
  end
  ints.each do |entry|
    put_gguf_string(io, entry[0])
    io.write_bytes(5_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(entry[1].to_i32, IO::ByteFormat::LittleEndian)
  end
  floats.each do |entry|
    put_gguf_string(io, entry[0])
    io.write_bytes(6_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(entry[1], IO::ByteFormat::LittleEndian)
  end
  put_gguf_string(io, "qwen35.rope.dimension_sections")
  io.write_bytes(9_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(5_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(4_u64, IO::ByteFormat::LittleEndian)
  [1_i32, 1_i32, 1_i32, 1_i32].each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
  File.open(path, "wb") { |file| file.write(io.to_slice) }
  path
end

describe "QBit F16 owner admission" do
  it "rejects before the direct Metal restore route" do
    path = portable_qwen35_fixture
    gguf = ML::GGUF::GGUFFile.new(path, mmap_tensors: false)
    begin
      hp = ML::GGUF::Qwen35Hparams.new(gguf)
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 8, kv_cache_f16: true)
      snapshot = ML::GGUF::QwenQBitStateSnapshot::Snapshot.new(
        8, 1, [0_i32], [] of ML::GGUF::QwenQBitStateSnapshot::EncodedRecord, 8, 7,
      )
      ML::GGUF::Qwen35Metal.available?.should be_true
      expect_raises(ArgumentError, /RawF32 QBit snapshot cannot be restored into an F16 KV owner/) do
        ML::GGUF::QwenQBitStateSnapshot.restore_into(snapshot, hp, state, prefer_metal: true)
      end
    ensure
      gguf.close
      File.delete(path) if File.exists?(path)
    end
  end
end
