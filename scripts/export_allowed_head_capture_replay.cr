#!/usr/bin/env crystal

require "json"

def usage
  STDERR.puts "usage: crystal scripts/export_allowed_head_capture_replay.cr CAPTURE.jsonl OUT.f32 [MAX_ROWS]"
  STDERR.puts
  STDERR.puts "Writes concatenated little-endian Float32 hidden rows for RPi5 replay and"
  STDERR.puts "prints tab-separated replay metadata, including ids_groups."
  STDERR.puts
  STDERR.puts "Environment:"
  STDERR.puts "  MIN_ALLOWED=0  Skip capture rows with fewer allowed ids."
  exit 2
end

capture_path = ARGV[0]? || usage
out_path = ARGV[1]? || usage
max_rows = (ARGV[2]? || ENV["MAX_ROWS"]? || "0").to_i
min_allowed = (ENV["MIN_ALLOWED"]? || "0").to_i
raise "MAX_ROWS must be non-negative" if max_rows < 0
raise "MIN_ALLOWED must be non-negative" if min_allowed < 0

rows = [] of NamedTuple(pos: Int32, token: Int32, ids: Array(Int32), hidden: Array(Float32))
hidden_dim : Int32? = nil

File.each_line(capture_path) do |line|
  next if line.strip.empty?
  obj = JSON.parse(line)
  next unless obj["kind"].as_s == "qwen35_allowed_head_hidden"

  ids = obj["allowed_ids"].as_a.map(&.as_i.to_i32)
  next if ids.size < min_allowed
  hidden = obj["hidden"].as_a.map(&.as_f.to_f32)
  dim = obj["hidden_dim"].as_i.to_i32
  raise "hidden_dim field mismatch: #{dim} vs #{hidden.size}" unless dim == hidden.size
  if existing = hidden_dim
    raise "mixed hidden_dim rows: #{existing} vs #{dim}" unless existing == dim
  else
    hidden_dim = dim
  end

  rows << {
    pos: obj["pos"].as_i.to_i32,
    token: obj["input_token_id"].as_i.to_i32,
    ids: ids,
    hidden: hidden,
  }
  break if max_rows > 0 && rows.size >= max_rows
end

raise "no capture rows found" if rows.empty?

File.open(out_path, "wb") do |io|
  rows.each do |row|
    row[:hidden].each do |value|
      io.write_bytes(value, IO::ByteFormat::LittleEndian)
    end
  end
end

max_allowed = rows.max_of { |row| row[:ids].size }
ids_groups = rows.map { |row| row[:ids].join(",") }.join(":")
labels = rows.map { |row| "pos#{row[:pos]}" }.join(",")

puts "replay_rows=#{rows.size}"
puts "hidden_dim=#{hidden_dim.not_nil!}"
puts "max_allowed=#{max_allowed}"
puts "x_f32_path=#{out_path}"
puts "ids_groups=#{ids_groups}"
puts "labels=#{labels}"
