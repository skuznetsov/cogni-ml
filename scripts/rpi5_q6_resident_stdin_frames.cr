#!/usr/bin/env crystal

require "../src/ml/gguf/qwen35_rpi5_allowed_head_client"

def usage
  STDERR.puts "usage: crystal scripts/rpi5_q6_resident_stdin_frames.cr X.f32 HIDDEN_DIM IDS_GROUPS ROWS [VOCAB_ROWS]"
  STDERR.puts
  STDERR.puts "Writes resident stdin binary frames to stdout. IDS_GROUPS is the"
  STDERR.puts "colon-separated id-list metadata emitted by export_allowed_head_capture_replay.cr."
  exit 2
end

x_f32_path = ARGV[0]? || usage
hidden_dim = (ARGV[1]? || usage).to_i
ids_groups = ARGV[2]? || usage
rows = (ARGV[3]? || usage).to_i
vocab_rows = ARGV[4]?.try(&.to_i)

raise "hidden_dim must be positive" unless hidden_dim > 0
raise "rows must be positive" unless rows > 0
raise "missing Float32 hidden batch: #{x_f32_path}" unless File.exists?(x_f32_path)

groups = ids_groups.split(':')
raise "ids_groups count #{groups.size} does not match rows #{rows}" unless groups.size == rows

row_bytes = hidden_dim * sizeof(Float32)
expected_bytes = row_bytes.to_i64 * rows
actual_bytes = File.size(x_f32_path)
raise "hidden batch byte size #{actual_bytes} does not match expected #{expected_bytes}" unless actual_bytes == expected_bytes

File.open(x_f32_path, "rb") do |file|
  rows.times do |row|
    raw = Bytes.new(row_bytes)
    read = file.read(raw)
    raise "short read on row #{row}: #{read} of #{row_bytes}" unless read == row_bytes

    hidden = Array(Float32).new(hidden_dim, 0.0_f32)
    io = IO::Memory.new(raw, writeable: false)
    hidden_dim.times do |i|
      hidden[i] = io.read_bytes(Float32, IO::ByteFormat::LittleEndian)
    end

    allowed_ids = groups[row].split(',').map(&.to_i32)
    ML::GGUF::Qwen35Rpi5AllowedHeadClient.write_binary_frame(STDOUT, hidden, allowed_ids, vocab_rows)
  end
end
