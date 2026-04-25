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
seed = 0xD35E_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_dense_scan_cpu [--s N] [--tokens N] [--block N] [--runs N]"
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
dense_ms = [] of Float64
max_delta = 0.0

runs.times do
  sm, serial_state = elapsed_ms { BlockScan.replay_final_state(initial, inputs, scale) }
  dm, dense_state = elapsed_ms do
    summaries = blocks.map { |block| BlockScan.compose_all(block) }

    prefix_state = initial
    blocks.each_with_index do |block, i|
      replayed = BlockScan.replay_final_state(prefix_state, block, scale)
      prefix_state = BlockScan.apply_affine(prefix_state, summaries[i])
      d = BlockScan.max_abs_delta(replayed, prefix_state)
      raise "block replay mismatch #{d}" if d > 1.0e-8
    end
    prefix_state
  end

  serial_ms << sm
  dense_ms << dm
  d = BlockScan.max_abs_delta(serial_state, dense_state)
  max_delta = d if d > max_delta
end

serial_avg = serial_ms.sum / serial_ms.size
dense_avg = dense_ms.sum / dense_ms.size

puts "Qwen35 DeltaNet dense block-scan CPU baseline"
puts "s=#{s} tokens=#{n_tokens} block=#{block_size} blocks=#{blocks.size} runs=#{runs}"
puts "serial_avg_ms=#{serial_avg.round(3)}"
puts "dense_seq_avg_ms=#{dense_avg.round(3)}"
puts "sequential_ratio=#{(serial_avg / dense_avg).round(4)}"
puts "max_state_delta=#{max_delta}"
puts "note=dense_seq includes summary build + sequential block replay on one CPU thread; use it as exactness and overhead evidence, not as GPU speedup evidence."
