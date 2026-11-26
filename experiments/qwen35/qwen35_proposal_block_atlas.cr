#!/usr/bin/env crystal

# Summarize speculative cycle JSONL dumps as a proposal-source atlas.
#
# This is intentionally offline/read-only. It turns existing --dump-cycles
# records into accepted-block distributions so we can compare neural, n-gram,
# target-only, and future self-draft proposal sources under one metric frame.

require "json"
require "option_parser"

class SourceStats
  property cycles : Int32 = 0
  property proposal_cycles : Int32 = 0
  property generated : Int32 = 0
  property proposed : Int32 = 0
  property accepted : Int32 = 0
  property rejects : Int32 = 0
  property positive_gain_cycles : Int32 = 0
  property gain_ms : Float64 = 0.0
  property wall_ms : Float64 = 0.0
  property proposal_ms : Float64 = 0.0
  property verify_ms : Float64 = 0.0
  property draft_ms : Float64 = 0.0
  property backup_ms : Float64 = 0.0
  property accepted_blocks = [] of Int32
  property proposed_blocks = [] of Int32
  property generated_blocks = [] of Int32

  def add(rec : JSON::Any)
    proposed_count = json_i(rec, "proposed_count")
    accepted_count = json_i(rec, "accepted_count")
    generated_count = json_i(rec, "generated_count")
    expected_gain = json_f(rec, "expected_gain_ms")

    @cycles += 1
    @proposal_cycles += 1 if proposed_count > 0
    @generated += generated_count
    @proposed += proposed_count
    @accepted += accepted_count
    @rejects += 1 if json_i(rec, "reject_index") >= 0
    @positive_gain_cycles += 1 if expected_gain > 0.0
    @gain_ms += expected_gain
    @wall_ms += json_f(rec, "wall_ms")
    @proposal_ms += json_f(rec, "proposal_ms")
    @verify_ms += json_f(rec, "target_verify_ms")
    @draft_ms += json_f(rec, "draft_ms")
    @backup_ms += json_f(rec, "target_backup_ms") + json_f(rec, "draft_backup_ms") + json_f(rec, "draft_resync_ms")
    @accepted_blocks << accepted_count if proposed_count > 0
    @proposed_blocks << proposed_count if proposed_count > 0
    @generated_blocks << generated_count
  end
end

def json_i(rec : JSON::Any, key : String) : Int32
  rec[key]?.try(&.as_i.to_i32) || 0
end

def json_f(rec : JSON::Any, key : String) : Float64
  rec[key]?.try(&.as_f?) || 0.0
end

def percentile(values : Array(Int32), q : Float64) : Float64
  return 0.0 if values.empty?

  sorted = values.sort
  idx = ((sorted.size - 1) * q).round.to_i.clamp(0, sorted.size - 1)
  sorted[idx].to_f64
end

def source_key(rec : JSON::Any, category : Bool) : String
  policy = rec["policy"]?.try(&.as_s?) || "unknown"
  kind = rec["kind"]?.try(&.as_s?) || "unknown"
  base = "#{policy}/#{kind}"
  if category
    cat = rec["prompt_category"]?.try(&.as_s?) || "unknown"
    "#{cat}/#{base}"
  else
    base
  end
end

paths = [] of String
by_category = false
min_cycles = 1

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_proposal_block_atlas [PATH ...] [--by-category] [--min-cycles N]"
  p.on("--by-category", "Split rows by prompt_category/source instead of source only") { by_category = true }
  p.on("--min-cycles N", "Hide rows with fewer than N cycles (default: 1)") { |v| min_cycles = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
  p.unknown_args do |args|
    paths.concat(args)
  end
end

abort "provide at least one dump file or directory" if paths.empty?

files = [] of String
paths.each do |path|
  if File.directory?(path)
    files.concat(Dir.glob(File.join(path, "**", "*.jsonl")))
  elsif File.file?(path)
    files << path
  else
    abort "not found: #{path}"
  end
end
files.reject! { |path| File.basename(path) == "prompt_manifest.jsonl" }
abort "no cycle JSONL files found" if files.empty?

stats = Hash(String, SourceStats).new { |hash, key| hash[key] = SourceStats.new }
files.sort.each do |path|
  File.each_line(path) do |line|
    line = line.strip
    next if line.empty?

    rec = JSON.parse(line)
    stats[source_key(rec, by_category)].add(rec)
  end
end

puts "Qwen35 proposal block atlas"
puts "files=#{files.size} split=#{by_category ? "category/source" : "source"}"
puts
printf "%-34s %7s %7s %9s %9s %8s %8s %8s %9s %9s %9s %9s %9s %9s %9s\n",
  "source", "cycles", "prop", "accepted", "proposed", "acc%", "rej%", "pos%",
  "p50_acc", "p90_acc", "p50_prop", "gain_ms", "wall_ms", "draft", "verify"

stats.keys.sort.each do |key|
  stat = stats[key]
  next if stat.cycles < min_cycles

  acc_rate = stat.proposed > 0 ? stat.accepted.to_f64 * 100.0 / stat.proposed : 100.0
  reject_rate = stat.cycles > 0 ? stat.rejects.to_f64 * 100.0 / stat.cycles : 0.0
  positive_rate = stat.cycles > 0 ? stat.positive_gain_cycles.to_f64 * 100.0 / stat.cycles : 0.0
  accepted = "#{stat.accepted}/#{stat.proposed}"
  printf "%-34s %7d %7d %9s %9d %7.1f%% %7.1f%% %7.1f%% %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n",
    key,
    stat.cycles,
    stat.proposal_cycles,
    accepted,
    stat.proposed,
    acc_rate,
    reject_rate,
    positive_rate,
    percentile(stat.accepted_blocks, 0.50),
    percentile(stat.accepted_blocks, 0.90),
    percentile(stat.proposed_blocks, 0.50),
    stat.gain_ms,
    stat.wall_ms,
    stat.draft_ms,
    stat.verify_ms
end
