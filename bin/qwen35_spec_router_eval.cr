#!/usr/bin/env crystal

# Evaluate an offline qwen35_spec_router_train JSON model against router rows.
# This is intentionally offline: runtime decode still verifies every proposed token.

require "json"
require "option_parser"

model_path = nil.as(String?)
input_paths = [] of String
threshold_override = nil.as(Float64?)
apply_model_filters = true

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_spec_router_eval --model model.json --input rows.jsonl [--input ...] [--threshold X]"
  p.on("--model PATH", "Model JSON emitted by qwen35_spec_router_train") { |v| model_path = v }
  p.on("--input PATH", "Router rows JSONL; can be repeated") { |v| input_paths << v }
  p.on("--threshold X", "Override model threshold") { |v| threshold_override = v.to_f }
  p.on("--all-rows", "Ignore model row filters and evaluate every input row") { apply_model_filters = false }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "--model is required" unless model_path
abort "at least one --input is required" if input_paths.empty?
abort "model not found: #{model_path}" unless File.file?(model_path.not_nil!)

model = JSON.parse(File.read(model_path.not_nil!))
feature_names = model["feature_names"].as_a.map(&.as_s)
weights = model["weights"].as_a.map(&.as_f)
threshold = threshold_override || model["threshold"].as_f
threshold = threshold.not_nil!
label_name = model["label"]?.try(&.as_s) || "positive_gain"
min_gain_ms = model["min_gain_ms"]?.try(&.as_f) || 0.0
filters = model["filters"]?
include_kinds = filters ? filters["include_kind"].as_a.map(&.as_s) : [] of String
exclude_kinds = filters ? filters["exclude_kind"].as_a.map(&.as_s) : [] of String

abort "feature/weight size mismatch" unless feature_names.size == weights.size

def s(rec : JSON::Any, key : String) : String
  value = rec[key]?
  value ? value.as_s : ""
end

def i(rec : JSON::Any, key : String) : Int32
  value = rec[key]?
  value ? value.as_i : 0
end

def b(rec : JSON::Any, key : String) : Bool
  value = rec[key]?
  value ? value.as_bool : false
end

def label_for(rec : JSON::Any, label_name : String, min_gain_ms : Float64) : Bool
  case label_name
  when "positive_gain"
    f(rec, "expected_gain_ms") > min_gain_ms
  else
    b(rec, label_name)
  end
end

def f(rec : JSON::Any, key : String) : Float64
  value = rec[key]?
  value ? value.as_f : 0.0
end

def sigmoid(z : Float64) : Float64
  if z >= 0
    1.0 / (1.0 + Math.exp(-z))
  else
    ez = Math.exp(z)
    ez / (1.0 + ez)
  end
end

def feature_value(name : String, rec : JSON::Any) : Float64
  case name
  when "bias"
    1.0
  when "gamma_over_32"
    i(rec, "gamma").clamp(0, 64).to_f / 32.0
  when "proposed_over_32"
    i(rec, "proposed_count").clamp(0, 64).to_f / 32.0
  when "generated_before_over_128"
    i(rec, "generated_before").clamp(0, 512).to_f / 128.0
  when "ngram_match_ratio"
    ngram_max = i(rec, "ngram_max")
    ngram_max > 0 ? i(rec, "ngram_match_len").clamp(0, ngram_max).to_f / ngram_max : 0.0
  when "ngram_disabled_before"
    b(rec, "ngram_disabled_before") ? 1.0 : 0.0
  else
    if name.starts_with?("category=")
      s(rec, "prompt_category") == name["category=".size..] ? 1.0 : 0.0
    elsif name.starts_with?("kind=")
      s(rec, "kind") == name["kind=".size..] ? 1.0 : 0.0
    elsif name.starts_with?("verify=")
      s(rec, "verify_mode") == name["verify=".size..] ? 1.0 : 0.0
    elsif name.starts_with?("sweep=")
      s(rec, "sweep_policy") == name["sweep=".size..] ? 1.0 : 0.0
    elsif name.starts_with?("draft=")
      s(rec, "draft_model") == name["draft=".size..] ? 1.0 : 0.0
    else
      0.0
    end
  end
end

class Metrics
  property count = 0
  property positives = 0
  property predicted = 0
  property true_positive = 0
  property gain_all = 0.0
  property gain_selected = 0.0
  property wall_all = 0.0
  property wall_selected = 0.0

  def add(label : Bool, pred : Bool, gain : Float64, wall : Float64)
    @count += 1
    @positives += 1 if label
    @predicted += 1 if pred
    @true_positive += 1 if label && pred
    @gain_all += gain
    @wall_all += wall
    if pred
      @gain_selected += gain
      @wall_selected += wall
    end
  end

  def accuracy(correct : Int32) : Float64
    @count > 0 ? correct.to_f / @count : 0.0
  end

  def precision : Float64
    @predicted > 0 ? @true_positive.to_f / @predicted : 0.0
  end

  def recall : Float64
    @positives > 0 ? @true_positive.to_f / @positives : 0.0
  end
end

rows = [] of JSON::Any
input_paths.each do |path|
  abort "input not found: #{path}" unless File.file?(path)
  File.each_line(path) do |line|
    next if line.empty?
    rec = JSON.parse(line)
    if apply_model_filters
      kind = s(rec, "kind")
      next if include_kinds.any? && !include_kinds.includes?(kind)
      next if exclude_kinds.includes?(kind)
    end
    rows << rec
  end
end

metrics = Metrics.new
correct = 0
by_category = Hash(String, {Metrics, Int32}).new { |h, k| h[k] = {Metrics.new, 0} }
by_kind = Hash(String, {Metrics, Int32}).new { |h, k| h[k] = {Metrics.new, 0} }

rows.each do |rec|
  score = 0.0
  feature_names.each_with_index do |name, idx|
    score += weights[idx] * feature_value(name, rec)
  end
  prob = sigmoid(score)
  pred = prob >= threshold
  label = label_for(rec, label_name, min_gain_ms)
  gain = f(rec, "expected_gain_ms")
  wall = f(rec, "wall_ms")
  correct += 1 if pred == label
  metrics.add(label, pred, gain, wall)

  cat = s(rec, "prompt_category")
  cat_metrics, cat_correct = by_category[cat]
  cat_correct += 1 if pred == label
  cat_metrics.add(label, pred, gain, wall)
  by_category[cat] = {cat_metrics, cat_correct}

  kind = s(rec, "kind")
  kind_metrics, kind_correct = by_kind[kind]
  kind_correct += 1 if pred == label
  kind_metrics.add(label, pred, gain, wall)
  by_kind[kind] = {kind_metrics, kind_correct}
end

puts "Router model eval"
puts "model=#{model_path} rows=#{rows.size} threshold=#{threshold} label=#{label_name} min_gain_ms=#{min_gain_ms} apply_model_filters=#{apply_model_filters}"
printf "%-18s %7s %7s %7s %7s %9s %10s %10s\n",
  "scope", "rows", "acc", "prec", "recall", "selected", "gain_sel", "gain_all"
printf "%-18s %7d %7.3f %7.3f %7.3f %9d %10.1f %10.1f\n",
  "all", metrics.count, metrics.accuracy(correct), metrics.precision, metrics.recall,
  metrics.predicted, metrics.gain_selected, metrics.gain_all

puts
puts "By category"
by_category.keys.sort.each do |key|
  stat, stat_correct = by_category[key]
  printf "%-18s %7d %7.3f %7.3f %7.3f %9d %10.1f %10.1f\n",
    key, stat.count, stat.accuracy(stat_correct), stat.precision, stat.recall,
    stat.predicted, stat.gain_selected, stat.gain_all
end

puts
puts "By kind"
by_kind.keys.sort.each do |key|
  stat, stat_correct = by_kind[key]
  printf "%-18s %7d %7.3f %7.3f %7.3f %9d %10.1f %10.1f\n",
    key, stat.count, stat.accuracy(stat_correct), stat.precision, stat.recall,
    stat.predicted, stat.gain_selected, stat.gain_all
end
