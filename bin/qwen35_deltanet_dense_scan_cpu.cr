#!/usr/bin/env crystal

require "option_parser"

alias Matrix = Array(Array(Float64))

struct DeltaInputs
  getter k : Array(Float64)
  getter v : Array(Float64)
  getter q : Array(Float64)
  getter g : Float64
  getter beta : Float64

  def initialize(@k : Array(Float64), @v : Array(Float64), @q : Array(Float64),
                 @g : Float64, @beta : Float64)
  end
end

struct AffineDelta
  getter a : Matrix
  getter b : Matrix

  def initialize(@a : Matrix, @b : Matrix)
  end
end

private def zeros(rows : Int32, cols : Int32) : Matrix
  Array.new(rows) { Array.new(cols, 0.0) }
end

private def identity(n : Int32) : Matrix
  m = zeros(n, n)
  n.times { |i| m[i][i] = 1.0 }
  m
end

private def matmul(a : Matrix, b : Matrix) : Matrix
  rows = a.size
  mid = b.size
  cols = b[0].size
  out = zeros(rows, cols)
  rows.times do |i|
    cols.times do |j|
      acc = 0.0
      mid.times { |k| acc += a[i][k] * b[k][j] }
      out[i][j] = acc
    end
  end
  out
end

private def matadd(a : Matrix, b : Matrix) : Matrix
  rows = a.size
  cols = a[0].size
  out = zeros(rows, cols)
  rows.times do |i|
    cols.times { |j| out[i][j] = a[i][j] + b[i][j] }
  end
  out
end

private def dot(a : Array(Float64), b : Array(Float64)) : Float64
  acc = 0.0
  a.size.times { |i| acc += a[i] * b[i] }
  acc
end

private def max_abs_delta(a : Matrix, b : Matrix) : Float64
  max = 0.0
  a.size.times do |i|
    a[i].size.times do |j|
      d = (a[i][j] - b[i][j]).abs
      max = d if d > max
    end
  end
  max
end

private def serial_delta_step(state : Matrix, inp : DeltaInputs, scale : Float64) : Tuple(Matrix, Array(Float64))
  s = state.size
  next_state = zeros(s, s)
  y = Array.new(s, 0.0)

  s.times do |d2|
    decayed = Array.new(s, 0.0)
    s.times { |d1| decayed[d1] = state[d2][d1] * inp.g }

    sk = dot(decayed, inp.k)
    delt = inp.beta * (inp.v[d2] - sk)

    s.times { |d1| next_state[d2][d1] = decayed[d1] + inp.k[d1] * delt }
    y[d2] = dot(next_state[d2], inp.q) * scale
  end

  {next_state, y}
end

private def affine_for(inp : DeltaInputs) : AffineDelta
  s = inp.k.size
  a = identity(s)
  s.times do |i|
    s.times do |j|
      a[i][j] = inp.g * (a[i][j] - inp.beta * inp.k[i] * inp.k[j])
    end
  end

  b = zeros(s, s)
  s.times do |d2|
    s.times { |d1| b[d2][d1] = inp.beta * inp.v[d2] * inp.k[d1] }
  end

  AffineDelta.new(a, b)
end

private def compose(first : AffineDelta, second : AffineDelta) : AffineDelta
  AffineDelta.new(
    matmul(first.a, second.a),
    matadd(matmul(first.b, second.a), second.b)
  )
end

private def apply_affine(state : Matrix, tr : AffineDelta) : Matrix
  matadd(matmul(state, tr.a), tr.b)
end

private def replay_block(state : Matrix, inputs : Array(DeltaInputs), scale : Float64) : Matrix
  cur = state
  inputs.each { |inp| cur, _ = serial_delta_step(cur, inp, scale) }
  cur
end

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
  sm, serial_state = elapsed_ms { replay_block(initial, inputs, scale) }
  dm, dense_state = elapsed_ms do
    summaries = blocks.map do |block|
      block.map { |inp| affine_for(inp) }.reduce { |acc, tr| compose(acc, tr) }
    end

    prefix_state = initial
    blocks.each_with_index do |block, i|
      replayed = replay_block(prefix_state, block, scale)
      prefix_state = apply_affine(prefix_state, summaries[i])
      d = max_abs_delta(replayed, prefix_state)
      raise "block replay mismatch #{d}" if d > 1.0e-8
    end
    prefix_state
  end

  serial_ms << sm
  dense_ms << dm
  d = max_abs_delta(serial_state, dense_state)
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
