#!/usr/bin/env crystal
# Score simple MTP entry and verifier-submit policies over qwen35_mtp_sidecar_probe
# router traces. Entry policies use only prompt/session features available before
# MTP proposal. Verifier-submit policies use MTP features available after proposal
# but before exact verifier submission. Target-margin policies are oracle-only.

require "json"
require "option_parser"

paths = [] of String
thresholds = [0.1_f64, 0.25_f64, 0.5_f64, 1.0_f64, 2.0_f64, 4.0_f64, 8.0_f64]
entry_rate_thresholds = [0.1_f64, 0.25_f64, 0.5_f64, 0.75_f64, 0.9_f64]
entry_token_thresholds = [16.0_f64, 32.0_f64, 64.0_f64, 128.0_f64]
include_target_oracle = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen36_mtp_wall_trace_score TRACE.jsonl [TRACE2.jsonl ...] [--thresholds LIST] [--entry-rate-thresholds LIST] [--entry-token-thresholds LIST] [--include-target-oracle]"
  p.on("--thresholds LIST", "Comma-separated thresholds; default #{thresholds.join(",")}") do |v|
    thresholds = v.split(',').map(&.strip).reject(&.empty?).map(&.to_f64)
  end
  p.on("--entry-rate-thresholds LIST", "Thresholds for prompt rate entry features") do |v|
    entry_rate_thresholds = v.split(',').map(&.strip).reject(&.empty?).map(&.to_f64)
  end
  p.on("--entry-token-thresholds LIST", "Thresholds for prompt token count entry feature") do |v|
    entry_token_thresholds = v.split(',').map(&.strip).reject(&.empty?).map(&.to_f64)
  end
  p.on("--include-target-oracle", "Also score target_margin, which is oracle/offline-only for exact-first runtime") { include_target_oracle = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
  p.unknown_args do |args|
    paths.concat(args)
  end
end

abort "expected at least one trace path" if paths.empty?

struct TraceRecord
  getter label : String
  getter gamma : Int32
  getter pass : Int32
  getter start_i : Int32
  getter end_i : Int32
  getter wall_before_ms : Float64
  getter wall_after_ms : Float64
  getter plain_suffix_ms : Float64
  getter entry_prev_token : Int32?
  getter prompt_tokens : Float64?
  getter prompt_unique_rate : Float64?
  getter prompt_repeat_rate : Float64?
  getter prompt_bigram_repeat_rate : Float64?
  getter prompt_adjacent_repeat_rate : Float64?
  getter mtp_first_margin : Float64?
  getter mtp_min_margin : Float64?
  getter target_margin : Float64?
  getter accepted_delta : Int32
  getter rejections_delta : Int32
  getter fallback_delta : Int32

  def initialize(@label, @gamma, @pass, @start_i, @end_i, @wall_before_ms, @wall_after_ms, @plain_suffix_ms,
                 @entry_prev_token, @prompt_tokens, @prompt_unique_rate, @prompt_repeat_rate,
                 @prompt_bigram_repeat_rate, @prompt_adjacent_repeat_rate,
                 @mtp_first_margin, @mtp_min_margin, @target_margin,
                 @accepted_delta, @rejections_delta, @fallback_delta)
  end
end

private def f64(obj : JSON::Any, key : String) : Float64?
  v = obj[key]?
  return nil unless v
  return nil if v.raw.nil?
  v.as_f?
end

private def i32(obj : JSON::Any, key : String) : Int32
  obj[key].as_i.to_i32
end

private def i32_opt(obj : JSON::Any, key : String) : Int32?
  obj[key]?.try(&.as_i?.try(&.to_i32))
end

private def str(obj : JSON::Any, key : String) : String
  obj[key].as_s
end

records = [] of TraceRecord
paths.each do |path|
  File.each_line(path) do |line|
    next if line.strip.empty?
    obj = JSON.parse(line)
    next unless obj["kind"]?.try(&.as_s?) == "mtp_wall_router_pass"
    records << TraceRecord.new(
      str(obj, "label"),
      i32(obj, "gamma"),
      i32(obj, "pass"),
      i32(obj, "start_i"),
      i32(obj, "end_i"),
      f64(obj, "wall_before_ms").not_nil!,
      f64(obj, "wall_after_ms").not_nil!,
      f64(obj, "plain_suffix_ms").not_nil!,
      i32_opt(obj, "entry_prev_token"),
      f64(obj, "prompt_tokens"),
      f64(obj, "prompt_unique_rate"),
      f64(obj, "prompt_repeat_rate"),
      f64(obj, "prompt_bigram_repeat_rate"),
      f64(obj, "prompt_adjacent_repeat_rate"),
      f64(obj, "mtp_first_margin"),
      f64(obj, "mtp_min_margin"),
      f64(obj, "target_margin"),
      i32(obj, "accepted_delta"),
      i32(obj, "rejections_delta"),
      i32(obj, "fallback_delta"))
  end
end

abort "no mtp_wall_router_pass records found" if records.empty?

groups = records.group_by { |r| {r.label, r.gamma} }
features = [{"mtp_first_margin", false}, {"mtp_min_margin", false}]
features << {"target_margin", true} if include_target_oracle

puts "mtp_wall_trace_score records=#{records.size} groups=#{groups.size} thresholds=#{thresholds.join(",")}"

entry_features = {
  "prompt_tokens"               => entry_token_thresholds,
  "prompt_unique_rate"          => entry_rate_thresholds,
  "prompt_repeat_rate"          => entry_rate_thresholds,
  "prompt_bigram_repeat_rate"   => entry_rate_thresholds,
  "prompt_adjacent_repeat_rate" => entry_rate_thresholds,
}

entry_features.each do |feature, feature_thresholds|
  feature_thresholds.each do |threshold|
    {"lt", "gte"}.each do |op|
      actual_sum = 0.0
      modeled_sum = 0.0
      skipped_groups = 0
      skipped_tokens = 0

      groups.each_value do |rows|
        sorted = rows.sort_by(&.pass)
        first = sorted.first
        actual_wall = sorted.map(&.wall_after_ms).max
        actual_sum += actual_wall
        value = case feature
                when "prompt_tokens" then first.prompt_tokens
                when "prompt_unique_rate" then first.prompt_unique_rate
                when "prompt_repeat_rate" then first.prompt_repeat_rate
                when "prompt_bigram_repeat_rate" then first.prompt_bigram_repeat_rate
                when "prompt_adjacent_repeat_rate" then first.prompt_adjacent_repeat_rate
                else nil
                end
        should_skip = if value
                        op == "lt" ? value < threshold : value >= threshold
                      else
                        false
                      end

        if should_skip
          skipped_groups += 1
          skipped_tokens += Math.max(0, sorted.last.end_i - first.start_i)
          # Entry/no-entry model: pay work already done before this MTP pass
          # (usually exact-first) and then finish exact from the same boundary.
          modeled_sum += first.wall_before_ms + first.plain_suffix_ms
        else
          modeled_sum += actual_wall
        end
      end

      delta = modeled_sum - actual_sum
      ratio = actual_sum > 0 ? modeled_sum / actual_sum : 0.0
      puts "entry_policy feature=#{feature} kind=runtime_legal_pre_mtp_entry op=#{op} threshold=#{threshold} groups=#{groups.size} skipped_groups=#{skipped_groups} actual_wall_ms=#{actual_sum.round(3)} modeled_wall_ms=#{modeled_sum.round(3)} delta_ms=#{delta.round(3)} ratio=#{ratio.round(4)} skipped_tokens=#{skipped_tokens}"
    end
  end
end

features.each do |feature, oracle|
  thresholds.each do |threshold|
    actual_sum = 0.0
    modeled_sum = 0.0
    skipped_groups = 0
    skipped_tokens = 0
    accepted_before_skip = 0
    rejected_before_skip = 0
    fallback_before_skip = 0

    groups.each_value do |rows|
      sorted = rows.sort_by(&.pass)
      actual_wall = sorted.map(&.wall_after_ms).max
      actual_sum += actual_wall
      skip = sorted.find do |r|
        value = case feature
                when "mtp_first_margin" then r.mtp_first_margin
                when "mtp_min_margin" then r.mtp_min_margin
                when "target_margin" then r.target_margin
                else nil
                end
        value && value < threshold
      end

      if skip
        skipped_groups += 1
        skipped_tokens += Math.max(0, sorted.last.end_i - skip.start_i)
        # The trace stores wall_before at the pass boundary and exact suffix cost
        # from the same start index. That is the legal fail-closed model for a
        # pre-verifier skip after paying MTP features for this pass.
        modeled_sum += skip.wall_before_ms + skip.plain_suffix_ms
        sorted.each do |r|
          break if r.pass >= skip.pass
          accepted_before_skip += r.accepted_delta
          rejected_before_skip += r.rejections_delta
          fallback_before_skip += r.fallback_delta
        end
      else
        modeled_sum += actual_wall
      end
    end

    delta = modeled_sum - actual_sum
    ratio = actual_sum > 0 ? modeled_sum / actual_sum : 0.0
    kind = oracle ? "oracle" : "runtime_legal_pre_verifier"
    puts "policy feature=#{feature} kind=#{kind} threshold=#{threshold} groups=#{groups.size} skipped_groups=#{skipped_groups} actual_wall_ms=#{actual_sum.round(3)} modeled_wall_ms=#{modeled_sum.round(3)} delta_ms=#{delta.round(3)} ratio=#{ratio.round(4)} accepted_before_skip=#{accepted_before_skip} rejections_before_skip=#{rejected_before_skip} fallback_before_skip=#{fallback_before_skip} skipped_tokens=#{skipped_tokens}"
  end
end
