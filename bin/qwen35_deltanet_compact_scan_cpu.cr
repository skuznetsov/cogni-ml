#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_deltanet_block_scan"

alias BlockScan = ML::GGUF::Qwen35DeltaNetBlockScan
alias DeltaInputs = ML::GGUF::Qwen35DeltaNetBlockScan::DeltaInputs

private def elapsed_ms(&)
  t0 = Time.instant
  value = yield
  {(Time.instant - t0).total_milliseconds, value}
end

s = 16
n_tokens = 64
block_size = 16
runs = 3
seed = 0xC0C07_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_compact_scan_cpu [--s N] [--tokens N] [--block N] [--runs N]"
  p.on("--s=N", "State size (default: 16)") { |v| s = v.to_i }
  p.on("--tokens=N", "Token count (default: 64)") { |v| n_tokens = v.to_i }
  p.on("--block=N", "Block size (default: 16)") { |v| block_size = v.to_i }
  p.on("--runs=N", "Timed runs (default: 3)") { |v| runs = v.to_i }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "tokens must be positive" unless n_tokens > 0
raise "block must be positive" unless block_size > 0
raise "runs must be positive" unless runs > 0

rng = Random.new(seed)
initial = Array.new(s) { Array.new(s) { ((rng.next_float - 0.5) * 0.2) } }
inputs = Array.new(n_tokens) do
  DeltaInputs.new(
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    0.88 + 0.10 * rng.next_float,
    rng.next_float
  )
end
scale = 1.0 / Math.sqrt(s.to_f64)
blocks = inputs.each_slice(block_size).to_a

serial_ms = [] of Float64
compact_ms = [] of Float64
summary_factor_count = 0
max_delta = 0.0

runs.times do
  sm, serial_state = elapsed_ms { BlockScan.replay_final_state(initial, inputs, scale) }
  cm, compact_state = elapsed_ms do
    summaries = blocks.map { |block| BlockScan.compact_summary_for_block(block) }
    summary_factor_count = summaries.sum(&.b_lefts.size)

    prefix_state = initial
    blocks.each_with_index do |block, i|
      replayed = BlockScan.replay_final_state(prefix_state, block, scale)
      prefix_state = BlockScan.apply_compact(prefix_state, summaries[i])
      d = BlockScan.max_abs_delta(replayed, prefix_state)
      raise "block replay mismatch #{d}" if d > 1.0e-8
    end
    prefix_state
  end

  serial_ms << sm
  compact_ms << cm
  d = BlockScan.max_abs_delta(serial_state, compact_state)
  max_delta = d if d > max_delta
end

serial_avg = serial_ms.sum / serial_ms.size
compact_avg = compact_ms.sum / compact_ms.size
compact_storage_f64 = blocks.size * s * s + 2 * summary_factor_count * s
dense_storage_f64 = blocks.size * 2 * s * s

puts "Qwen35 DeltaNet compact block-scan CPU baseline"
puts "s=#{s} tokens=#{n_tokens} block=#{block_size} blocks=#{blocks.size} runs=#{runs}"
puts "serial_avg_ms=#{serial_avg.round(3)}"
puts "compact_seq_avg_ms=#{compact_avg.round(3)}"
puts "sequential_ratio=#{(serial_avg / compact_avg).round(4)}"
puts "max_state_delta=#{max_delta}"
puts "compact_storage_f64=#{compact_storage_f64}"
puts "dense_storage_f64=#{dense_storage_f64}"
puts "storage_ratio=#{(compact_storage_f64.to_f64 / dense_storage_f64).round(4)}"
puts "note=compact_seq stores dense A plus low-rank B factors; it avoids dense B materialization but still runs sequentially on CPU."
