#!/usr/bin/env crystal

# Offline scorer for qwen35_deltanet_fixed_basis_probe self-spec router traces.
# It never changes inference semantics; it estimates whether a branch guard
# warning would be worth a guarded-prefix snapshot under different thresholds.

require "json"
require "option_parser"

input_paths = [] of String
thresholds = [0.5, 1.0, 2.0]
min_prefixes = [1, 2, 3, 4]
suffix_thresholds = [Float64::INFINITY]
no_snapshot_thresholds = [] of Float64
snapshot_cost_ms = 0.0
replay_token_ms = 0.0

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_self_spec_router_trace_score --input trace.jsonl [--input ...] [--branch-thresholds LIST] [--min-prefixes LIST] [--suffix-thresholds LIST] [--no-snapshot-thresholds LIST]"
  p.on("--input PATH", "Router trace JSONL from --simulate-self-spec-gpu-pipeline-router-trace; can be repeated") { |v| input_paths << v }
  p.on("--branch-thresholds LIST", "Draft top1/top2 margin thresholds to score (default: 0.5,1.0,2.0)") { |v| thresholds = v.split(',').map(&.strip).reject(&.empty?).map(&.to_f64) }
  p.on("--min-prefixes LIST", "Snapshot min-prefix candidates to score (default: 1,2,3,4)") { |v| min_prefixes = v.split(',').map(&.strip).reject(&.empty?).map(&.to_i).uniq.sort }
  p.on("--suffix-thresholds LIST", "Only snapshot if the post-guard suffix has a margin <= threshold; use inf for no gate (default: inf)") do |v|
    suffix_thresholds = v.split(',').map(&.strip).reject(&.empty?).map { |raw| raw.downcase == "inf" ? Float64::INFINITY : raw.to_f64 }.uniq.sort
  end
  p.on("--no-snapshot-thresholds LIST", "Score replayless no-snapshot guard value for thresholds; use inf for always-on (default: disabled)") do |v|
    no_snapshot_thresholds = v.split(',').map(&.strip).reject(&.empty?).map { |raw| raw.downcase == "inf" ? Float64::INFINITY : raw.to_f64 }.uniq.sort
  end
  p.on("--snapshot-cost-ms F", "Optional per-snapshot cost for net_ms scoring") { |v| snapshot_cost_ms = v.to_f64 }
  p.on("--replay-token-ms F", "Optional saved-prefix replay cost per token for net_ms scoring") { |v| replay_token_ms = v.to_f64 }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "at least one --input is required" if input_paths.empty?
abort "--branch-thresholds must not be empty" if thresholds.empty?
abort "--min-prefixes must contain positive integers" if min_prefixes.empty? || min_prefixes.any? { |v| v <= 0 }
abort "--suffix-thresholds must not be empty" if suffix_thresholds.empty?
abort "--no-snapshot-thresholds must be non-negative" if no_snapshot_thresholds.any? { |v| v < 0.0 }
abort "--snapshot-cost-ms must be non-negative" if snapshot_cost_ms < 0.0
abort "--replay-token-ms must be non-negative" if replay_token_ms < 0.0

record TraceRow,
  source : String,
  label : String,
  chunk : Int32,
  index : Int32,
  generated_offset : Int32,
  chunk_size : Int32,
  verifier_size : Int32,
  draft_margin : Float64?,
  reject : Bool,
  top1_hit : Bool,
  top2_hit : Bool,
  rejections_before : Int32

class Score
  property chunks = 0
  property candidates = 0
  property prefix_rejects = 0
  property guard_rejects = 0
  property guard_passes = 0
  property pass_clean = 0
  property pass_suffix_reject = 0
  property snapshots = 0
  property useful_snapshots = 0
  property wasted_snapshots = 0
  property suffix_gate_skips = 0
  property saved_prefix_tokens = 0
  property max_saved_prefix = 0
  property post_reject_candidates = 0

  def add_snapshot(prefix_len : Int32, suffix_reject : Bool)
    @snapshots += 1
    if suffix_reject
      @useful_snapshots += 1
      @saved_prefix_tokens += prefix_len
      @max_saved_prefix = prefix_len if prefix_len > @max_saved_prefix
    else
      @wasted_snapshots += 1
    end
  end
end

class NoSnapshotScore
  property chunks = 0
  property candidates = 0
  property prefix_rejects = 0
  property guard_rejects = 0
  property guard_passes = 0
  property pass_clean = 0
  property pass_suffix_reject = 0
  property post_reject_candidates = 0
  property useful_suffix_tokens = 0
  property pass_suffix_tokens = 0
  property max_useful_suffix = 0

  def add_guard_reject(suffix_tokens : Int32)
    @guard_rejects += 1
    @useful_suffix_tokens += suffix_tokens
    @max_useful_suffix = suffix_tokens if suffix_tokens > @max_useful_suffix
  end

  def add_guard_pass(suffix_tokens : Int32, suffix_reject : Bool)
    @guard_passes += 1
    @pass_suffix_tokens += suffix_tokens
    if suffix_reject
      @pass_suffix_reject += 1
    else
      @pass_clean += 1
    end
  end
end

class PrefixSuffixScore
  property passes = 0
  property pass_clean = 0
  property pass_suffix_reject = 0
  property snapshots = 0
  property useful_snapshots = 0
  property wasted_snapshots = 0
  property saved_prefix_tokens = 0

  def add_pass(prefix_len : Int32, suffix_reject : Bool, snapshot : Bool)
    @passes += 1
    if suffix_reject
      @pass_suffix_reject += 1
    else
      @pass_clean += 1
    end

    return unless snapshot

    @snapshots += 1
    if suffix_reject
      @useful_snapshots += 1
      @saved_prefix_tokens += prefix_len
    else
      @wasted_snapshots += 1
    end
  end
end

def j_s(obj : JSON::Any, key : String, fallback : String = "") : String
  value = obj[key]?
  value ? value.as_s : fallback
end

def j_i(obj : JSON::Any, key : String, fallback : Int32 = 0) : Int32
  value = obj[key]?
  value ? value.as_i.to_i32 : fallback
end

def j_b(obj : JSON::Any, key : String, fallback : Bool = false) : Bool
  value = obj[key]?
  value ? value.as_bool : fallback
end

def j_f?(obj : JSON::Any, key : String) : Float64?
  value = obj[key]?
  return nil unless value
  return nil if value.raw.nil?
  value.as_f
end

groups = Hash(String, Array(TraceRow)).new { |h, k| h[k] = [] of TraceRow }
input_paths.each_with_index do |path, path_index|
  abort "input not found: #{path}" unless File.file?(path)
  File.each_line(path) do |line|
    stripped = line.strip
    next if stripped.empty?
    obj = JSON.parse(stripped)
    label = j_s(obj, "label", "main")
    chunk = j_i(obj, "chunk")
    source = "#{path_index}:#{File.basename(path)}"
    groups["#{source}|#{label}|#{chunk}"] << TraceRow.new(
      source: source,
      label: label,
      chunk: chunk,
      index: j_i(obj, "index"),
      generated_offset: j_i(obj, "generated_offset"),
      chunk_size: j_i(obj, "chunk_size"),
      verifier_size: j_i(obj, "verifier_size"),
      draft_margin: j_f?(obj, "draft_margin"),
      reject: j_b(obj, "reject"),
      top1_hit: j_b(obj, "top1_hit"),
      top2_hit: j_b(obj, "top2_hit"),
      rejections_before: j_i(obj, "rejections_before")
    )
  end
end

scores = {} of {Float64, Int32, Float64} => Score
thresholds.each do |threshold|
  min_prefixes.each do |min_prefix|
    suffix_thresholds.each do |suffix_threshold|
      scores[{threshold, min_prefix, suffix_threshold}] = Score.new
    end
  end
end
prefix_suffix_scores = Hash({Float64, Int32, Float64}, PrefixSuffixScore).new { |h, k| h[k] = PrefixSuffixScore.new }
no_snapshot_scores = Hash({String, Float64}, NoSnapshotScore).new { |h, k| h[k] = NoSnapshotScore.new }

total_chunks = 0
chunks_with_reject = 0
rows_total = 0

groups.values.each do |rows|
  rows.sort_by!(&.index)
  next if rows.empty?
  total_chunks += 1
  rows_total += rows.size

  reject_index = rows.find(&.reject).try(&.index)
  chunks_with_reject += 1 if reject_index
  verifier_size = rows.first.verifier_size
  label = rows.first.label

  thresholds.each do |threshold|
    guard_index = rows.find { |row| row.index < verifier_size && (m = row.draft_margin) && m <= threshold }.try(&.index)
    if bgi = guard_index
      suffix_size = verifier_size - bgi - 1
      if suffix_size > 0
        suffix_rows = rows.select { |row| row.index > bgi && row.index < verifier_size }
        suffix_min_margin = suffix_rows.compact_map(&.draft_margin).min?
        prefix_len = bgi + 1
        suffix_reject = false
        guard_pass = false
        if r = reject_index
          if r > bgi
            suffix_reject = true
            guard_pass = true
          end
        else
          guard_pass = true
        end

        if guard_pass
          suffix_thresholds.each do |suffix_threshold|
            suffix_gate_ok = !!suffix_threshold.infinite? || (!!suffix_min_margin && suffix_min_margin.not_nil! <= suffix_threshold)
            prefix_suffix_scores[{threshold, prefix_len, suffix_threshold}].add_pass(prefix_len, suffix_reject, suffix_gate_ok)
          end
        end
      end
    end
    min_prefixes.each do |min_prefix|
      suffix_thresholds.each do |suffix_threshold|
        score = scores[{threshold, min_prefix, suffix_threshold}]
        score.chunks += 1
        next unless bgi = guard_index

        score.candidates += 1
        score.post_reject_candidates += 1 if rows.first.rejections_before > 0

        suffix_rows = rows.select { |row| row.index > bgi && row.index < verifier_size }
        suffix_min_margin = suffix_rows.compact_map(&.draft_margin).min?
        suffix_gate_ok = !!suffix_threshold.infinite? || (!!suffix_min_margin && suffix_min_margin.not_nil! <= suffix_threshold)

        if r = reject_index
          if r < bgi
            score.prefix_rejects += 1
          elsif r == bgi
            score.guard_rejects += 1
          else
            score.guard_passes += 1
            score.pass_suffix_reject += 1
            prefix_len = bgi + 1
            suffix_size = verifier_size - bgi - 1
            if suffix_size > 0 && prefix_len >= min_prefix
              if suffix_gate_ok
                score.add_snapshot(prefix_len, true)
              else
                score.suffix_gate_skips += 1
              end
            end
          end
        else
          score.guard_passes += 1
          score.pass_clean += 1
          prefix_len = bgi + 1
          suffix_size = verifier_size - bgi - 1
          if suffix_size > 0 && prefix_len >= min_prefix
            if suffix_gate_ok
              score.add_snapshot(prefix_len, false)
            else
              score.suffix_gate_skips += 1
            end
          end
        end
      end
    end
  end

  no_snapshot_thresholds.each do |threshold|
    guard_index = rows.find { |row| row.index < verifier_size && (m = row.draft_margin) && m <= threshold }.try(&.index)
    ["all", label].each do |score_label|
      score = no_snapshot_scores[{score_label, threshold}]
      score.chunks += 1
      next unless bgi = guard_index

      score.candidates += 1
      score.post_reject_candidates += 1 if rows.first.rejections_before > 0
      suffix_tokens = verifier_size - bgi - 1

      if r = reject_index
        if r < bgi
          score.prefix_rejects += 1
        elsif r == bgi
          score.add_guard_reject(suffix_tokens)
        else
          score.add_guard_pass(suffix_tokens, true)
        end
      else
        score.add_guard_pass(suffix_tokens, false)
      end
    end
  end
end

suffix_threshold_labels = suffix_thresholds.map { |v| v.infinite? ? "inf" : v.to_s }
no_snapshot_threshold_labels = no_snapshot_thresholds.map { |v| v.infinite? ? "inf" : v.to_s }
puts "self_spec_router_trace_score inputs=#{input_paths.size} chunks=#{total_chunks} rows=#{rows_total} chunks_with_reject=#{chunks_with_reject} thresholds=#{thresholds.join(',')} min_prefixes=#{min_prefixes.join(',')} suffix_thresholds=#{suffix_threshold_labels.join(',')} no_snapshot_thresholds=#{no_snapshot_threshold_labels.join(',')} snapshot_cost_ms=#{snapshot_cost_ms} replay_token_ms=#{replay_token_ms}"

thresholds.each do |threshold|
  min_prefixes.each do |min_prefix|
    suffix_thresholds.each do |suffix_threshold|
      score = scores[{threshold, min_prefix, suffix_threshold}]
      next if score.chunks == 0

      candidate_rate = score.chunks > 0 ? 100.0 * score.candidates / score.chunks : 0.0
      useful_rate = score.snapshots > 0 ? 100.0 * score.useful_snapshots / score.snapshots : 0.0
      avg_saved = score.useful_snapshots > 0 ? score.saved_prefix_tokens.to_f64 / score.useful_snapshots : 0.0
      net_ms = score.saved_prefix_tokens * replay_token_ms - score.snapshots * snapshot_cost_ms
      suffix_label = suffix_threshold.infinite? ? "inf" : suffix_threshold.to_s
      puts "branch_guard_value threshold=#{threshold} min_prefix=#{min_prefix} suffix_threshold=#{suffix_label} chunks=#{score.chunks} candidates=#{score.candidates} candidate_rate=#{candidate_rate.round(2)}% prefix_rejects=#{score.prefix_rejects} guard_rejects=#{score.guard_rejects} guard_passes=#{score.guard_passes} pass_clean=#{score.pass_clean} pass_suffix_reject=#{score.pass_suffix_reject} post_reject_candidates=#{score.post_reject_candidates} snapshots=#{score.snapshots} useful_snapshots=#{score.useful_snapshots} wasted_snapshots=#{score.wasted_snapshots} suffix_gate_skips=#{score.suffix_gate_skips} useful_snapshot_rate=#{useful_rate.round(2)}% saved_prefix_tokens=#{score.saved_prefix_tokens} avg_saved_prefix=#{avg_saved.round(3)} max_saved_prefix=#{score.max_saved_prefix} net_ms=#{net_ms.round(3)}"
    end
  end
end

prefix_suffix_scores.keys.sort_by { |key| {key[0], key[1], key[2].infinite? ? Float64::INFINITY : key[2]} }.each do |key|
  threshold, prefix_len, suffix_threshold = key
  score = prefix_suffix_scores[key]
  next if score.passes == 0

  useful_rate = score.snapshots > 0 ? 100.0 * score.useful_snapshots / score.snapshots : 0.0
  net_ms = score.saved_prefix_tokens * replay_token_ms - score.snapshots * snapshot_cost_ms
  suffix_label = suffix_threshold.infinite? ? "inf" : suffix_threshold.to_s
  puts "branch_guard_prefix_suffix_value threshold=#{threshold} prefix_len=#{prefix_len} suffix_threshold=#{suffix_label} passes=#{score.passes} pass_clean=#{score.pass_clean} pass_suffix_reject=#{score.pass_suffix_reject} snapshots=#{score.snapshots} useful_snapshots=#{score.useful_snapshots} wasted_snapshots=#{score.wasted_snapshots} useful_snapshot_rate=#{useful_rate.round(2)}% saved_prefix_tokens=#{score.saved_prefix_tokens} net_ms=#{net_ms.round(3)}"
end

no_snapshot_scores.keys.sort_by { |key| {key[0] == "all" ? "" : key[0], key[1].infinite? ? Float64::INFINITY : key[1]} }.each do |key|
  label, threshold = key
  score = no_snapshot_scores[key]
  next if score.chunks == 0

  candidate_rate = score.chunks > 0 ? 100.0 * score.candidates / score.chunks : 0.0
  useful_rate = score.candidates > 0 ? 100.0 * score.guard_rejects / score.candidates : 0.0
  pass_rate = score.candidates > 0 ? 100.0 * score.guard_passes / score.candidates : 0.0
  avg_useful_suffix = score.guard_rejects > 0 ? score.useful_suffix_tokens.to_f64 / score.guard_rejects : 0.0
  net_ms = score.useful_suffix_tokens * replay_token_ms
  threshold_label = threshold.infinite? ? "inf" : threshold.to_s
  puts "branch_guard_no_snapshot_value label=#{label} threshold=#{threshold_label} chunks=#{score.chunks} candidates=#{score.candidates} candidate_rate=#{candidate_rate.round(2)}% prefix_rejects=#{score.prefix_rejects} guard_rejects=#{score.guard_rejects} guard_passes=#{score.guard_passes} pass_clean=#{score.pass_clean} pass_suffix_reject=#{score.pass_suffix_reject} useful_guard_rate=#{useful_rate.round(2)}% guard_pass_rate=#{pass_rate.round(2)}% post_reject_candidates=#{score.post_reject_candidates} useful_suffix_tokens=#{score.useful_suffix_tokens} pass_suffix_tokens=#{score.pass_suffix_tokens} avg_useful_suffix=#{avg_useful_suffix.round(3)} max_useful_suffix=#{score.max_useful_suffix} net_ms=#{net_ms.round(3)}"
end
