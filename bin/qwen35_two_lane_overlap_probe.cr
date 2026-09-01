#!/usr/bin/env crystal

require "option_parser"
require "digest/sha256"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

RELEASE_BUILD     = {{ flag?(:release) }}
DEFAULT_MODEL     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = "The capital of France is"
pairs = 10

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_two_lane_overlap_probe [--model PATH] [--tokenizer PATH] [--prompt TEXT] [--pairs N]"
  p.on("--model PATH", "Qwen GGUF model path") { |v| model = v }
  p.on("--tokenizer PATH", "llama-tokenize binary path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Prompt to prefill before the probe") { |v| prompt = v }
  p.on("--pairs N", "Number of paired decode positions to measure (default: 10)") { |v| pairs = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--pairs must be at least 10" unless pairs >= 10
raise "model not found: #{model}" unless File.exists?(model)

puts "Qwen35 two-lane overlap probe"
puts "model=#{File.basename(model)} pairs=#{pairs}"

g = ML::GGUF::GGUFFile.new(model)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, model, tokenizer_bin)
g.close
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
hp = weights.hparams
ids = tok.encode(prompt)
raise "prompt must tokenize to at least one token" if ids.empty?

max_seq = ids.size + pairs + 1
base = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
ML::GGUF::Qwen35CPU.prepare_state_metal!(base, hp)
top, _ = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, ids, 0, base)
pos = ids.size
token = top.to_i32
puts "prefill_tokens=#{ids.size} branch_token=#{token} branch_piece=#{tok.decode_single(token).inspect}"
raise "probe capacity invariant failed" unless pos + pairs <= max_seq

states = Array(ML::GGUF::Qwen35CPU::State).new(6) { base.fork }
ML::GGUF::Qwen35CPU.release_state_metal!(base)
seq_a, seq_b, lane_a, lane_b, async_a, async_b = states

seq_tokens = {token, (token + 1) % weights.output.out_dim}
lane_tokens = seq_tokens
async_tokens = seq_tokens
seq_ms = [] of Float64
lane_ms = [] of Float64
async_ms = [] of Float64
results = [] of Int32

run_seq = ->(sample_pos : Int32) do
  started = Time.instant
  a, al = ML::GGUF::Qwen35CPU.forward_top1(weights, seq_tokens[0], sample_pos, seq_a)
  b, bl = ML::GGUF::Qwen35CPU.forward_top1(weights, seq_tokens[1], sample_pos, seq_b)
  seq_ms << (Time.instant - started).total_milliseconds
  seq_tokens = {a, b}
  {a, al, b, bl}
end

run_lane = ->(sample_pos : Int32) do
  started = Time.instant
  sub_a = ML::GGUF::Qwen35CPU.forward_top1_async(weights, lane_tokens[0], sample_pos, lane_a,
    fresh_scratch: false, scratch_namespace: "probe_a").not_nil!
  a, al = ML::GGUF::Qwen35CPU.wait_forward_top1(sub_a)
  sub_b = ML::GGUF::Qwen35CPU.forward_top1_async(weights, lane_tokens[1], sample_pos, lane_b,
    fresh_scratch: false, scratch_namespace: "probe_b").not_nil!
  b, bl = ML::GGUF::Qwen35CPU.wait_forward_top1(sub_b)
  lane_ms << (Time.instant - started).total_milliseconds
  lane_tokens = {a, b}
  {a, al, b, bl}
end

run_async = ->(sample_pos : Int32, reverse : Bool) do
  started = Time.instant
  if reverse
    sub_b = ML::GGUF::Qwen35CPU.forward_top1_async(weights, async_tokens[1], sample_pos, async_b,
      fresh_scratch: false, scratch_namespace: "probe_b").not_nil!
    sub_a = ML::GGUF::Qwen35CPU.forward_top1_async(weights, async_tokens[0], sample_pos, async_a,
      fresh_scratch: false, scratch_namespace: "probe_a").not_nil!
    b, bl = ML::GGUF::Qwen35CPU.wait_forward_top1(sub_b)
    a, al = ML::GGUF::Qwen35CPU.wait_forward_top1(sub_a)
  else
    sub_a = ML::GGUF::Qwen35CPU.forward_top1_async(weights, async_tokens[0], sample_pos, async_a,
      fresh_scratch: false, scratch_namespace: "probe_a").not_nil!
    sub_b = ML::GGUF::Qwen35CPU.forward_top1_async(weights, async_tokens[1], sample_pos, async_b,
      fresh_scratch: false, scratch_namespace: "probe_b").not_nil!
    a, al = ML::GGUF::Qwen35CPU.wait_forward_top1(sub_a)
    b, bl = ML::GGUF::Qwen35CPU.wait_forward_top1(sub_b)
  end
  async_ms << (Time.instant - started).total_milliseconds
  async_tokens = {a, b}
  {a, al, b, bl}
end

pairs.times do |i|
  sample_pos = pos + i
  raise "input trajectory diverged before sample #{i}" unless seq_tokens == lane_tokens && seq_tokens == async_tokens
  rows = case i % 3
         when 0 then {run_seq.call(sample_pos), run_lane.call(sample_pos), run_async.call(sample_pos, i.odd?)}
         when 1 then {run_lane.call(sample_pos), run_async.call(sample_pos, i.odd?), run_seq.call(sample_pos)}
         else        {run_async.call(sample_pos, i.odd?), run_seq.call(sample_pos), run_lane.call(sample_pos)}
         end
  canonical = rows[0]
  rows.each do |row|
    unless row[0] == canonical[0] && row[2] == canonical[2] &&
           (row[1] - canonical[1]).abs <= 1e-4_f32 && (row[3] - canonical[3]).abs <= 1e-4_f32
      raise "result mismatch at sample #{i}: #{rows}"
    end
  end
  results << canonical[0] << canonical[2]
end

def median(values : Array(Float64)) : Float64
  sorted = values.sort
  mid = sorted.size // 2
  sorted.size.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0
end

def state_sha256(state : ML::GGUF::Qwen35CPU::State) : String
  snapshot = ML::GGUF::Qwen35StateSnapshot.capture(state)
  digest = Digest::SHA256.new
  snapshot.positions.each { |position| digest.update(position.to_s) }
  snapshot.records.each { |record| digest.update(record.bytes) }
  digest.final.hexstring
end

seq_hashes = {state_sha256(seq_a), state_sha256(seq_b)}
raise "serial lane state mismatch" unless seq_hashes == {state_sha256(lane_a), state_sha256(lane_b)}
raise "async lane state mismatch" unless seq_hashes == {state_sha256(async_a), state_sha256(async_b)}

seq_mean = seq_ms.sum / seq_ms.size
lane_mean = lane_ms.sum / lane_ms.size
async_mean = async_ms.sum / async_ms.size
speedup_pct = (lane_mean - async_mean) / lane_mean * 100.0
wins = lane_ms.zip(async_ms).count { |lane, overlapped| overlapped < lane }
gate = RELEASE_BUILD && speedup_pct >= 15.0 && wins >= 8
puts "release_build=#{RELEASE_BUILD} pooled_sequential_mean_ms=#{seq_mean.round(3)} serial_lane_mean_ms=#{lane_mean.round(3)} async_lane_mean_ms=#{async_mean.round(3)} speedup_pct=#{speedup_pct.round(3)} wins=#{wins}/#{pairs} gate=#{gate ? "PASS" : "FAIL"}"
puts "pooled_sequential_median_ms=#{median(seq_ms).round(3)} serial_lane_median_ms=#{median(lane_ms).round(3)} async_lane_median_ms=#{median(async_ms).round(3)}"
puts "state_sha256=#{seq_hashes.join(',')} next_ids=#{results.join(',')}"

states.each { |state| ML::GGUF::Qwen35CPU.release_state_metal!(state) }
weights.close
