#!/usr/bin/env crystal

# Train a tiny offline logistic baseline on qwen35_spec_router_dataset JSONL rows.
# This is a research tool: it emits a compact JSON model, but does not affect runtime decode.

require "json"
require "option_parser"
require "set"

inputs = [] of String
out_path = nil.as(String?)
label_name = "positive_gain"
epochs = 200
lr = 0.2
l2 = 1.0e-4
threshold = 0.5
holdout_every = 5

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_spec_router_train --input ROWS.jsonl [--input ...] [--out model.json]"
  p.on("--input PATH", "Router dataset JSONL from qwen35_spec_router_dataset; can be repeated") { |v| inputs << v }
  p.on("--out PATH", "Write model JSON to PATH; default prints to stdout") { |v| out_path = v }
  p.on("--label NAME", "Boolean label: positive_gain, full_accept, accept_ge_75pct (default: positive_gain)") { |v| label_name = v }
  p.on("--epochs N", "Training epochs (default: 200)") { |v| epochs = v.to_i }
  p.on("--lr X", "Learning rate (default: 0.2)") { |v| lr = v.to_f }
  p.on("--l2 X", "L2 regularization (default: 1e-4)") { |v| l2 = v.to_f }
  p.on("--threshold X", "Classification threshold for metrics/model (default: 0.5)") { |v| threshold = v.to_f }
  p.on("--holdout-every N", "Every Nth row is deterministic holdout; <=0 disables (default: 5)") { |v| holdout_every = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "at least one --input is required" if inputs.empty?
abort "--epochs must be positive" unless epochs > 0
abort "--lr must be positive" unless lr > 0
abort "--threshold must be in [0,1]" unless threshold >= 0.0 && threshold <= 1.0
unless {"positive_gain", "full_accept", "accept_ge_75pct"}.includes?(label_name)
  abort "unknown --label #{label_name.inspect}; expected positive_gain, full_accept, or accept_ge_75pct"
end

class Row
  getter rec : JSON::Any
  getter label : Float64

  def initialize(@rec : JSON::Any, label_name : String)
    @label = bool_field(label_name) ? 1.0 : 0.0
  end

  def s(key : String) : String
    value = @rec[key]?
    value ? value.as_s : ""
  end

  def i(key : String) : Int32
    value = @rec[key]?
    value ? value.as_i : 0
  end

  def b(key : String) : Bool
    value = @rec[key]?
    value ? value.as_bool : false
  end

  private def bool_field(key : String) : Bool
    value = @rec[key]?
    value ? value.as_bool : false
  end
end

rows = [] of Row
inputs.each do |path|
  abort "input not found: #{path}" unless File.file?(path)
  File.each_line(path) do |line|
    next if line.empty?
    rows << Row.new(JSON.parse(line), label_name)
  end
end

abort "no rows found" if rows.empty?

kind_values = Set(String).new
verify_values = Set(String).new
sweep_values = Set(String).new
draft_values = Set(String).new

rows.each do |row|
  kind_values << row.s("kind")
  verify_values << row.s("verify_mode")
  sweep_values << row.s("sweep_policy")
  draft_values << row.s("draft_model")
end

feature_names = [
  "bias",
  "gamma_over_32",
  "proposed_over_32",
  "generated_before_over_128",
  "ngram_match_ratio",
  "ngram_disabled_before",
]

kind_values.to_a.sort.each { |v| feature_names << "kind=#{v}" }
verify_values.to_a.sort.each { |v| feature_names << "verify=#{v}" }
sweep_values.to_a.sort.each { |v| feature_names << "sweep=#{v}" }
draft_values.to_a.sort.each { |v| feature_names << "draft=#{v}" }

kind_offset = 6
verify_offset = kind_offset + kind_values.size
sweep_offset = verify_offset + verify_values.size
draft_offset = sweep_offset + sweep_values.size

kind_index = kind_values.to_a.sort.each_with_index.to_h
verify_index = verify_values.to_a.sort.each_with_index.to_h
sweep_index = sweep_values.to_a.sort.each_with_index.to_h
draft_index = draft_values.to_a.sort.each_with_index.to_h

def feature_vector(row : Row,
                   feature_count : Int32,
                   kind_offset : Int32,
                   verify_offset : Int32,
                   sweep_offset : Int32,
                   draft_offset : Int32,
                   kind_index : Hash(String, Int32),
                   verify_index : Hash(String, Int32),
                   sweep_index : Hash(String, Int32),
                   draft_index : Hash(String, Int32)) : Array(Float64)
  x = Array.new(feature_count, 0.0)
  x[0] = 1.0
  x[1] = row.i("gamma").clamp(0, 64).to_f / 32.0
  x[2] = row.i("proposed_count").clamp(0, 64).to_f / 32.0
  x[3] = row.i("generated_before").clamp(0, 512).to_f / 128.0
  ngram_max = row.i("ngram_max")
  x[4] = ngram_max > 0 ? row.i("ngram_match_len").clamp(0, ngram_max).to_f / ngram_max : 0.0
  x[5] = row.b("ngram_disabled_before") ? 1.0 : 0.0

  if idx = kind_index[row.s("kind")]?
    x[kind_offset + idx] = 1.0
  end
  if idx = verify_index[row.s("verify_mode")]?
    x[verify_offset + idx] = 1.0
  end
  if idx = sweep_index[row.s("sweep_policy")]?
    x[sweep_offset + idx] = 1.0
  end
  if idx = draft_index[row.s("draft_model")]?
    x[draft_offset + idx] = 1.0
  end
  x
end

xs = rows.map do |row|
  feature_vector(row, feature_names.size, kind_offset, verify_offset, sweep_offset, draft_offset,
    kind_index, verify_index, sweep_index, draft_index)
end

train_idx = [] of Int32
holdout_idx = [] of Int32
rows.each_index do |i|
  if holdout_every > 0 && rows.size >= holdout_every * 2 && (i % holdout_every == 0)
    holdout_idx << i
  else
    train_idx << i
  end
end
train_idx = rows.each_index.to_a if train_idx.empty?

weights = Array.new(feature_names.size, 0.0)

def sigmoid(z : Float64) : Float64
  if z >= 0
    1.0 / (1.0 + Math.exp(-z))
  else
    ez = Math.exp(z)
    ez / (1.0 + ez)
  end
end

def dot(weights : Array(Float64), x : Array(Float64)) : Float64
  sum = 0.0
  weights.each_index { |i| sum += weights[i] * x[i] }
  sum
end

epochs.times do
  train_idx.each do |i|
    x = xs[i]
    y = rows[i].label
    pred = sigmoid(dot(weights, x))
    err = pred - y
    weights.each_index do |j|
      reg = j == 0 ? 0.0 : l2 * weights[j]
      weights[j] -= lr * (err * x[j] + reg)
    end
  end
end

record Metrics, count : Int32, positives : Int32, predicted_positive : Int32, true_positive : Int32, accuracy : Float64, precision : Float64, recall : Float64, logloss : Float64

def metrics(indices : Array(Int32), rows : Array(Row), xs : Array(Array(Float64)), weights : Array(Float64), threshold : Float64) : Metrics
  return Metrics.new(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0) if indices.empty?

  correct = 0
  positives = 0
  predicted_positive = 0
  true_positive = 0
  logloss = 0.0
  eps = 1.0e-12

  indices.each do |i|
    y = rows[i].label
    pred = sigmoid(dot(weights, xs[i]))
    pred_label = pred >= threshold ? 1.0 : 0.0
    correct += 1 if pred_label == y
    positives += 1 if y > 0.5
    predicted_positive += 1 if pred_label > 0.5
    true_positive += 1 if pred_label > 0.5 && y > 0.5
    p = pred.clamp(eps, 1.0 - eps)
    logloss += -(y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p))
  end

  precision = predicted_positive > 0 ? true_positive.to_f / predicted_positive : 0.0
  recall = positives > 0 ? true_positive.to_f / positives : 0.0
  Metrics.new(
    indices.size,
    positives,
    predicted_positive,
    true_positive,
    correct.to_f / indices.size,
    precision,
    recall,
    logloss / indices.size
  )
end

train_metrics = metrics(train_idx, rows, xs, weights, threshold)
holdout_metrics = metrics(holdout_idx, rows, xs, weights, threshold)

def metrics_json(metrics : Metrics)
  {
    "count"              => metrics.count,
    "positives"          => metrics.positives,
    "predicted_positive" => metrics.predicted_positive,
    "true_positive"      => metrics.true_positive,
    "accuracy"           => metrics.accuracy,
    "precision"          => metrics.precision,
    "recall"             => metrics.recall,
    "logloss"            => metrics.logloss,
  }
end

STDERR.puts "Router logistic train rows=#{train_idx.size} holdout=#{holdout_idx.size} features=#{feature_names.size} label=#{label_name}"
STDERR.printf "train   acc=%.3f precision=%.3f recall=%.3f logloss=%.4f positives=%d/%d predicted=%d\n",
  train_metrics.accuracy, train_metrics.precision, train_metrics.recall, train_metrics.logloss,
  train_metrics.positives, train_metrics.count, train_metrics.predicted_positive
STDERR.printf "holdout acc=%.3f precision=%.3f recall=%.3f logloss=%.4f positives=%d/%d predicted=%d\n",
  holdout_metrics.accuracy, holdout_metrics.precision, holdout_metrics.recall, holdout_metrics.logloss,
  holdout_metrics.positives, holdout_metrics.count, holdout_metrics.predicted_positive

model = {
  "version"       => 1,
  "kind"          => "qwen35_spec_router_logistic",
  "label"         => label_name,
  "threshold"     => threshold,
  "feature_names" => feature_names,
  "weights"       => weights,
  "train"         => metrics_json(train_metrics),
  "holdout"       => metrics_json(holdout_metrics),
  "notes"         => "Offline research baseline only; target verification remains mandatory for every proposed token.",
}

if path = out_path
  File.write(path, model.to_json)
else
  puts model.to_json
end
