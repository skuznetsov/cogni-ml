#!/usr/bin/env crystal

# Build compact router-training rows from qwen35_speculative_accept cycle JSONL dumps.
# The output is JSONL by default so downstream notebooks/tools can add features freely.

require "json"
require "option_parser"
require "set"

inputs = [] of String
out_path = nil.as(String?)
summary_only = false

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_spec_router_dataset --input DIR_OR_JSONL [--input ...] [--out PATH] [--summary-only]"
  p.on("--input PATH", "Cycle dump directory or JSONL file; can be repeated") { |v| inputs << v }
  p.on("--out PATH", "Write derived JSONL rows to PATH instead of stdout") { |v| out_path = v }
  p.on("--summary-only", "Do not emit rows; only print aggregate summary to stderr") { summary_only = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "at least one --input is required" if inputs.empty?

def safe_name(name : String) : String
  cleaned = name.gsub(/[^A-Za-z0-9_.-]/, "_")
  cleaned.empty? ? "prompt" : cleaned
end

record PromptInfo, index : Int32, name : String, hash : String, safe_name : String

def load_manifest(dir : String) : Hash(Int32, PromptInfo)
  manifest = File.join(dir, "prompt_manifest.jsonl")
  prompts = {} of Int32 => PromptInfo
  return prompts unless File.exists?(manifest)

  File.each_line(manifest) do |line|
    next if line.empty?
    rec = JSON.parse(line)
    index = rec["prompt_index"].as_i
    name = rec["prompt_name"].as_s
    hash = rec["prompt_hash"].as_s
    prompts[index] = PromptInfo.new(index, name, hash, safe_name(name))
  end
  prompts
end

def json_f(rec : JSON::Any, key : String) : Float64
  value = rec[key]?
  value ? value.as_f : 0.0
end

def json_i(rec : JSON::Any, key : String) : Int32
  value = rec[key]?
  value ? value.as_i : 0
end

def json_s(rec : JSON::Any, key : String) : String
  value = rec[key]?
  value ? value.as_s : ""
end

def json_b(rec : JSON::Any, key : String) : Bool
  value = rec[key]?
  value ? value.as_bool : false
end

def json_i_array(rec : JSON::Any, key : String) : Array(Int32)
  value = rec[key]?
  return [] of Int32 unless value
  value.as_a.map(&.as_i.to_i32)
rescue TypeCastError
  [] of Int32
end

record CandidateStats,
  present : Bool,
  unique_ratio : Float64,
  pair_unique_ratio : Float64,
  entropy_norm : Float64,
  longest_run_ratio : Float64,
  exact_period_over_8 : Float64,
  lag1_ratio : Float64,
  lag2_ratio : Float64,
  lag4_ratio : Float64,
  lag8_ratio : Float64

def lag_ratio(ids : Array(Int32), lag : Int32) : Float64
  return 0.0 if ids.size <= lag
  matches = 0
  lag.upto(ids.size - 1) do |i|
    matches += 1 if ids[i] == ids[i - lag]
  end
  matches.to_f / (ids.size - lag)
end

def exact_period(ids : Array(Int32), max_period : Int32) : Int32
  return 0 if ids.empty?
  1.upto(Math.min(max_period, ids.size)) do |period|
    exact = true
    period.upto(ids.size - 1) do |i|
      if ids[i] != ids[i % period]
        exact = false
        break
      end
    end
    return period if exact
  end
  0
end

def candidate_stats(rec : JSON::Any) : CandidateStats
  ids = json_i_array(rec, "candidates")
  n = ids.size
  return CandidateStats.new(false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if n == 0

  counts = Hash(Int32, Int32).new(0)
  ids.each { |id| counts[id] += 1 }
  unique_ratio = counts.size.to_f / n

  pair_unique_ratio = 0.0
  if n > 1
    pairs = Set(Tuple(Int32, Int32)).new
    0.upto(n - 2) { |i| pairs << {ids[i], ids[i + 1]} }
    pair_unique_ratio = pairs.size.to_f / (n - 1)
  end

  entropy = 0.0
  counts.each_value do |count|
    p = count.to_f / n
    entropy -= p * (Math.log(p) / Math.log(2.0))
  end
  max_entropy = n > 1 ? Math.log(n.to_f) / Math.log(2.0) : 1.0
  entropy_norm = max_entropy > 0.0 ? entropy / max_entropy : 0.0

  longest = 1
  run = 1
  1.upto(n - 1) do |i|
    if ids[i] == ids[i - 1]
      run += 1
    else
      longest = Math.max(longest, run)
      run = 1
    end
  end
  longest = Math.max(longest, run)

  period = exact_period(ids, 8)
  CandidateStats.new(
    true,
    unique_ratio,
    pair_unique_ratio,
    entropy_norm,
    longest.to_f / n,
    period > 0 ? period.to_f / 8.0 : 0.0,
    lag_ratio(ids, 1),
    lag_ratio(ids, 2),
    lag_ratio(ids, 4),
    lag_ratio(ids, 8)
  )
end

def infer_run_meta(path : String, prompts : Hash(Int32, PromptInfo)) : {Int32, Int32, String, String}
  base = File.basename(path, ".jsonl")
  match = base.match(/^rep(\d+)_prompt(\d+)_(.+)$/)
  return {-1, -1, "", base} unless match

  rep = match[1].to_i
  prompt_index = match[2].to_i
  tail = match[3]
  prompt = prompts[prompt_index]?
  prompt_name = prompt ? prompt.name : ""
  sweep_policy = tail
  if prompt
    prefix = "#{prompt.safe_name}_"
    sweep_policy = tail.starts_with?(prefix) ? tail[prefix.size..] : tail
  end
  {rep, prompt_index, prompt_name, sweep_policy}
end

def prompt_category(prompt_name : String) : String
  return "unknown" if prompt_name.empty?
  head = prompt_name.split('_', 2)[0]
  case head
  when ""
    "unknown"
  when "templ"
    "template"
  else
    head
  end
end

record SummaryKey, sweep_policy : String, kind : String

class Summary
  property count = 0
  property generated = 0
  property proposed = 0
  property accepted = 0
  property full_accepts = 0
  property positive_gain = 0
  property wall_ms = 0.0
  property expected_gain_ms = 0.0

  def add(generated : Int32, proposed : Int32, accepted : Int32, full_accept : Bool, positive_gain : Bool, wall : Float64, gain : Float64)
    @count += 1
    @generated += generated
    @proposed += proposed
    @accepted += accepted
    @full_accepts += 1 if full_accept
    @positive_gain += 1 if positive_gain
    @wall_ms += wall
    @expected_gain_ms += gain
  end
end

files = [] of String
manifest_by_dir = {} of String => Hash(Int32, PromptInfo)

inputs.each do |input|
  if Dir.exists?(input)
    manifest_by_dir[input] = load_manifest(input)
    Dir.glob(File.join(input, "*.jsonl")).sort.each do |path|
      next if File.basename(path) == "prompt_manifest.jsonl"
      files << path
    end
  elsif File.file?(input)
    files << input
  else
    abort "input not found: #{input}"
  end
end

abort "no cycle JSONL files found" if files.empty?

out_io = out_path ? File.open(out_path.not_nil!, "w") : STDOUT
summary = Hash(SummaryKey, Summary).new { |hash, key| hash[key] = Summary.new }
rows = 0

begin
  files.each do |path|
    dir = File.dirname(path)
    prompts = manifest_by_dir[dir]? || (manifest_by_dir[dir] = load_manifest(dir))
    rep, prompt_index, prompt_name, sweep_policy = infer_run_meta(path, prompts)

    File.each_line(path) do |line|
      next if line.empty?
      rec = JSON.parse(line)

      proposed = json_i(rec, "proposed_count")
      accepted = json_i(rec, "accepted_count")
      generated = json_i(rec, "generated_count")
      wall = json_f(rec, "wall_ms")
      gain = rec["expected_gain_ms"]?.try(&.as_f) || 0.0
      accepted_ratio = proposed > 0 ? accepted.to_f / proposed : 0.0
      full_accept = proposed > 0 && accepted == proposed
      positive_gain = gain > 0.0
      kind = json_s(rec, "kind")
      candidates = candidate_stats(rec)

      summary[SummaryKey.new(sweep_policy, kind)].add(generated, proposed, accepted, full_accept, positive_gain, wall, gain)

      unless summary_only
        out_io.puts({
          "source_file"                   => File.basename(path),
          "rep"                           => rep,
          "prompt_index"                  => prompt_index,
          "prompt_name"                   => prompt_name,
          "prompt_category"               => prompt_category(prompt_name),
          "prompt_hash"                   => json_s(rec, "prompt_hash"),
          "sweep_policy"                  => sweep_policy,
          "kind"                          => kind,
          "policy"                        => json_s(rec, "policy"),
          "verify_mode"                   => json_s(rec, "verify_mode"),
          "target_model"                  => json_s(rec, "target_model"),
          "draft_model"                   => json_s(rec, "draft_model"),
          "position"                      => json_i(rec, "position"),
          "generated_before"              => json_i(rec, "generated_before"),
          "generated_count"               => generated,
          "gamma"                         => json_i(rec, "gamma"),
          "proposed_count"                => proposed,
          "accepted_count"                => accepted,
          "accepted_ratio"                => accepted_ratio,
          "full_accept"                   => full_accept,
          "accept_ge_75pct"               => proposed > 0 && accepted_ratio >= 0.75,
          "positive_gain"                 => positive_gain,
          "expected_gain_ms"              => gain,
          "reject_index"                  => json_i(rec, "reject_index"),
          "ngram_match_len"               => json_i(rec, "ngram_match_len"),
          "ngram_min"                     => json_i(rec, "ngram_min"),
          "ngram_max"                     => json_i(rec, "ngram_max"),
          "ngram_recursive"               => json_b(rec, "ngram_recursive"),
          "ngram_disabled_before"         => json_b(rec, "ngram_disabled_before"),
          "ngram_disabled_after"          => json_b(rec, "ngram_disabled_after"),
          "candidate_features_present"    => candidates.present,
          "candidate_unique_ratio"        => candidates.unique_ratio,
          "candidate_pair_unique_ratio"   => candidates.pair_unique_ratio,
          "candidate_entropy_norm"        => candidates.entropy_norm,
          "candidate_longest_run_ratio"   => candidates.longest_run_ratio,
          "candidate_exact_period_over_8" => candidates.exact_period_over_8,
          "candidate_lag1_ratio"          => candidates.lag1_ratio,
          "candidate_lag2_ratio"          => candidates.lag2_ratio,
          "candidate_lag4_ratio"          => candidates.lag4_ratio,
          "candidate_lag8_ratio"          => candidates.lag8_ratio,
          "draft_ms"                      => json_f(rec, "draft_ms"),
          "target_verify_ms"              => json_f(rec, "target_verify_ms"),
          "target_backup_ms"              => json_f(rec, "target_backup_ms"),
          "draft_backup_ms"               => json_f(rec, "draft_backup_ms"),
          "draft_resync_ms"               => json_f(rec, "draft_resync_ms"),
          "wall_ms"                       => wall,
          "candidate_hash"                => json_s(rec, "candidate_hash"),
        }.to_json)
      end
      rows += 1
    end
  end
ensure
  out_io.close if out_path
end

STDERR.puts "Router dataset rows=#{rows} files=#{files.size}"
STDERR.printf "%-28s %-20s %7s %9s %9s %9s %9s %10s %10s\n",
  "sweep_policy", "kind", "rows", "accepted", "proposed", "full", "pos_gain", "wall_ms", "gain_ms"
summary.keys.sort_by { |key| {key.sweep_policy, key.kind} }.each do |key|
  stat = summary[key]
  STDERR.printf "%-28s %-20s %7d %9d %9d %9d %9d %10.1f %10.1f\n",
    key.sweep_policy, key.kind, stat.count, stat.accepted, stat.proposed,
    stat.full_accepts, stat.positive_gain, stat.wall_ms, stat.expected_gain_ms
end
