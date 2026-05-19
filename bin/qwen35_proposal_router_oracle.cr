#!/usr/bin/env crystal

# Offline proposal-source router oracle for speculative cycle JSONL dumps.
#
# This intentionally does not change runtime policy. It summarizes measured
# cycle economics and asks which proposal source would be selected per prompt
# category if the router were allowed to fail closed to direct exact decode.

require "json"
require "option_parser"

class SourceStats
  property cycles : Int32 = 0
  property proposal_cycles : Int32 = 0
  property generated : Int32 = 0
  property proposed : Int32 = 0
  property accepted : Int32 = 0
  property rejects : Int32 = 0
  property gain_ms : Float64 = 0.0
  property wall_ms : Float64 = 0.0
  property draft_ms : Float64 = 0.0
  property verify_ms : Float64 = 0.0
  property prompt_hashes = Set(String).new

  def add(rec : JSON::Any)
    proposed_count = json_i(rec, "proposed_count")
    accepted_count = json_i(rec, "accepted_count")

    @cycles += 1
    @proposal_cycles += 1 if proposed_count > 0
    @generated += json_i(rec, "generated_count")
    @proposed += proposed_count
    @accepted += accepted_count
    @rejects += 1 if json_i(rec, "reject_index") >= 0
    @gain_ms += json_f(rec, "expected_gain_ms")
    @wall_ms += json_f(rec, "wall_ms")
    @draft_ms += json_f(rec, "draft_ms")
    @verify_ms += json_f(rec, "target_verify_ms")
    if prompt_hash = rec["prompt_hash"]?.try(&.as_s?)
      @prompt_hashes << prompt_hash
    end
  end

  def accept_rate : Float64
    @proposed > 0 ? @accepted.to_f64 * 100.0 / @proposed : 100.0
  end

  def reject_rate : Float64
    @cycles > 0 ? @rejects.to_f64 * 100.0 / @cycles : 0.0
  end
end

def json_i(rec : JSON::Any, key : String) : Int32
  rec[key]?.try(&.as_i.to_i32) || 0
end

def json_f(rec : JSON::Any, key : String) : Float64
  rec[key]?.try(&.as_f?) || 0.0
end

def any_f(value : JSON::Any?) : Float64
  return 0.0 unless value

  value.as_f? || value.as_i?.try(&.to_f64) || 0.0
end

def feature_f(rec : JSON::Any, group : String, key : String) : Float64
  obj = rec[group]?
  return 0.0 unless obj

  any_f(obj[key]?)
end

def category_key(rec : JSON::Any) : String
  rec["prompt_category"]?.try(&.as_s?) || "unknown"
end

def source_key(rec : JSON::Any) : String
  policy = rec["policy"]?.try(&.as_s?) || "unknown"
  kind = rec["kind"]?.try(&.as_s?) || "unknown"
  "#{policy}/#{kind}"
end

paths = [] of String
min_cycles = 1
min_gain_ms = 0.0
include_target_only = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_proposal_router_oracle [PATH ...] [--min-cycles N] [--min-gain-ms X] [--include-target-only]"
  p.on("--min-cycles N", "Require at least N cycles before a source can be selected (default: 1)") { |v| min_cycles = v.to_i }
  p.on("--min-gain-ms X", "Fail closed unless selected source total gain exceeds X ms (default: 0)") { |v| min_gain_ms = v.to_f64 }
  p.on("--include-target-only", "Allow target_only/plain_fallback rows to compete as measured sources; default treats direct exact decode as zero-gain fallback") { include_target_only = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
  p.unknown_args do |args|
    paths.concat(args)
  end
end

abort "provide at least one dump file or directory" if paths.empty?
abort "--min-cycles must be positive" unless min_cycles > 0

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

stats = Hash(Tuple(String, String), SourceStats).new { |hash, key| hash[key] = SourceStats.new }
global = Hash(String, SourceStats).new { |hash, key| hash[key] = SourceStats.new }
categories = Set(String).new
records = [] of JSON::Any

files.sort.each do |path|
  File.each_line(path) do |line|
    line = line.strip
    next if line.empty?

    rec = JSON.parse(line)
    category = category_key(rec)
    source = source_key(rec)
    next if !include_target_only && source.includes?("target_only")

    categories << category
    records << rec
    stats[{category, source}].add(rec)
    global[source].add(rec)
  end
end

puts "Qwen35 proposal router oracle"
puts "files=#{files.size} min_cycles=#{min_cycles} min_gain_ms=#{min_gain_ms} include_target_only=#{include_target_only}"
puts
puts "Source economics"
printf "%-36s %7s %7s %8s %8s %8s %9s %9s %9s %8s\n",
  "source", "cycles", "prompts", "acc%", "rej%", "prop", "accepted", "gain_ms", "wall_ms", "draft"

global.keys.sort_by { |source| {-global[source].gain_ms, source} }.each do |source|
  stat = global[source]
  printf "%-36s %7d %7d %7.1f%% %7.1f%% %8d %9d %9.1f %9.1f %8.1f\n",
    source, stat.cycles, stat.prompt_hashes.size, stat.accept_rate, stat.reject_rate,
    stat.proposed, stat.accepted, stat.gain_ms, stat.wall_ms, stat.draft_ms
end

puts
puts "Category fail-closed oracle"
printf "%-14s %-36s %7s %7s %8s %8s %9s %9s %s\n",
  "category", "selected_source", "cycles", "prompts", "acc%", "rej%", "gain_ms", "wall_ms", "decision"

selected_gain = 0.0
selected_wall = 0.0
selected_cycles = 0
selected_categories = 0
failed_closed = 0

categories.to_a.sort.each do |category|
  candidates = stats.select do |(cat, _source), stat|
    cat == category && stat.cycles >= min_cycles && stat.proposal_cycles > 0
  end
  best = candidates.max_by? { |(_key, stat)| stat.gain_ms }

  if best && best[1].gain_ms > min_gain_ms
    source = best[0][1]
    stat = best[1]
    selected_gain += stat.gain_ms
    selected_wall += stat.wall_ms
    selected_cycles += stat.cycles
    selected_categories += 1
    decision = "select"
    printf "%-14s %-36s %7d %7d %7.1f%% %7.1f%% %9.1f %9.1f %s\n",
      category, source, stat.cycles, stat.prompt_hashes.size, stat.accept_rate, stat.reject_rate,
      stat.gain_ms, stat.wall_ms, decision
  else
    failed_closed += 1
    source = best ? best[0][1] : "none"
    gain = best ? best[1].gain_ms : 0.0
    wall = best ? best[1].wall_ms : 0.0
    cycles = best ? best[1].cycles : 0
    prompts = best ? best[1].prompt_hashes.size : 0
    acc = best ? best[1].accept_rate : 100.0
    rej = best ? best[1].reject_rate : 0.0
    printf "%-14s %-36s %7d %7d %7.1f%% %7.1f%% %9.1f %9.1f fail_closed\n",
      category, source, cycles, prompts, acc, rej, gain, wall
  end
end

puts
puts "Oracle summary selected_categories=#{selected_categories} fail_closed=#{failed_closed} selected_cycles=#{selected_cycles} selected_gain_ms=#{selected_gain.round(1)} selected_wall_ms=#{selected_wall.round(1)}"

ngram_rows = records.select { |rec| source_key(rec) == "ngram/ngram" && json_i(rec, "proposed_count") > 0 }
unless ngram_rows.empty?
  puts
  puts "N-gram feature gate sweep"
  printf "%-34s %7s %8s %8s %8s %9s %9s %s\n",
    "gate", "cycles", "prop", "accepted", "rej%", "gain_ms", "wall_ms", "decision"

  gate_specs = [
    {"match_len>=6", ->(rec : JSON::Any) { json_i(rec, "ngram_match_len") >= 6 }},
    {"match_len>=7", ->(rec : JSON::Any) { json_i(rec, "ngram_match_len") >= 7 }},
    {"match_len>=8", ->(rec : JSON::Any) { json_i(rec, "ngram_match_len") >= 8 }},
    {"match_ratio>=0.875", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "ngram_match_ratio") >= 0.875 }},
    {"match_ratio>=1.0", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "ngram_match_ratio") >= 1.0 }},
    {"lag2>=0.5", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "candidate_lag2_ratio") >= 0.5 }},
    {"lag4>=0.5", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "candidate_lag4_ratio") >= 0.5 }},
    {"lag8>=0.5", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "candidate_lag8_ratio") >= 0.5 }},
    {"unique_ratio<=0.5", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "candidate_unique_ratio") <= 0.5 }},
    {"entropy<=0.6", ->(rec : JSON::Any) { feature_f(rec, "candidate_features", "candidate_entropy_norm") <= 0.6 }},
  ]

  gate_specs.each do |name, gate|
    stat = SourceStats.new
    ngram_rows.each { |rec| stat.add(rec) if gate.call(rec) }
    decision = stat.cycles > 0 && stat.gain_ms > min_gain_ms ? "select" : "fail_closed"
    printf "%-34s %7d %8d %8d %7.1f%% %9.1f %9.1f %s\n",
      name, stat.cycles, stat.proposed, stat.accepted, stat.reject_rate,
      stat.gain_ms, stat.wall_ms, decision
  end
end
