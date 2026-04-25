#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/core/buffer"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen35_deltanet_block_scan"

alias BlockScan = ML::GGUF::Qwen35DeltaNetBlockScan
alias DeltaInputs = ML::GGUF::Qwen35DeltaNetBlockScan::DeltaInputs

SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;

#define MAX_S 128

kernel void adjoint_replay_outputs(
    device const float* state [[buffer(0)]],
    device const float* tq    [[buffer(1)]],
    device const float* add   [[buffer(2)]],
    device       float* out   [[buffer(3)]],
    constant     uint&  s     [[buffer(4)]],
    constant     uint&  ntok  [[buffer(5)]],
    constant     float& scale [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint t = gid.y;
    if (row >= s || t >= ntok || s > MAX_S) return;

    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += state[row * s + d] * tq[t * s + d];
    }
    out[t * s + row] = (acc + add[t * s + row]) * scale;
}
METAL

private def elapsed_ms(&)
  t0 = Time.instant
  value = yield
  {(Time.instant - t0).total_milliseconds, value}
end

private def flatten_f32(m : BlockScan::Matrix) : Array(Float32)
  out = Array(Float32).new(m.size * m[0].size)
  m.each { |row| row.each { |v| out << v.to_f32 } }
  out
end

private def flatten_vecs_f32(vs : Array(Array(Float64))) : Array(Float32)
  out = Array(Float32).new(vs.sum(&.size))
  vs.each { |row| row.each { |v| out << v.to_f32 } }
  out
end

private def max_output_delta(cpu : Array(Array(Float64)), gpu : Array(Float32)) : Float64
  max = 0.0
  s = cpu[0].size
  cpu.each_with_index do |row, t|
    row.each_with_index do |v, d|
      delta = (v - gpu[t * s + d].to_f64).abs
      max = delta if delta > max
    end
  end
  max
end

s = 128
tokens = 16
runs = 20
warmup = 3
seed = 0xAD501_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_adjoint_replay_micro [--s N] [--tokens N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default/max: 128)") { |v| s = v.to_i }
  p.on("--tokens=N", "Block token count (default: 16)") { |v| tokens = v.to_i }
  p.on("--runs=N", "Timed runs (default: 20)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs (default: 3)") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "s must be <= 128" if s > 128
raise "tokens must be positive" unless tokens > 0
raise "runs must be positive" unless runs > 0
abort "Metal unavailable" unless ML::Metal::Device.init!

rng = Random.new(seed)
initial = Array.new(s) { Array.new(s) { ((rng.next_float - 0.5) * 0.2) } }
inputs = Array.new(tokens) do
  DeltaInputs.new(
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    0.88 + 0.10 * rng.next_float,
    rng.next_float
  )
end
scale = 1.0_f32

terms = BlockScan.adjoint_output_terms_for_block(inputs)
_, serial_y = BlockScan.replay_block(initial, inputs, scale.to_f64)
tq = terms.map(&.transformed_q)
add = terms.map(&.additive)

state_buf = ML::MetalBuffer.from_array(flatten_f32(initial))
tq_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(tq))
add_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(add))
out_buf = ML::MetalBuffer.new((tokens * s).to_i64 * sizeof(Float32))
pipe = ML::Metal::ComputePipeline.new("adjoint_replay_outputs", SOURCE)
s_u = s.to_u32
tokens_u = tokens.to_u32

run_adjoint = -> do
  ML::Metal::Dispatch.execute(pipe) do |enc|
    enc.set_buffer(state_buf, 0)
    enc.set_buffer(tq_buf, 1)
    enc.set_buffer(add_buf, 2)
    enc.set_buffer(out_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(tokens_u, 5)
    enc.set_value(scale, 6)
    enc.dispatch({s, tokens, 1}, {16, 16, 1})
  end
end

rowwise_state_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))
q_flat = flatten_vecs_f32(inputs.map(&.q))
k_flat = flatten_vecs_f32(inputs.map(&.k))
v_flat = flatten_vecs_f32(inputs.map(&.v))
g_flat = inputs.map(&.g.to_f32)
beta_flat = inputs.map(&.beta.to_f32)
run_rowwise = -> do
  rowwise_state_buf.write(flatten_f32(initial))
  ML::GGUF::Qwen35Metal.delta_net_chunk(
    rowwise_state_buf,
    q_flat,
    k_flat,
    v_flat,
    g_flat,
    beta_flat,
    1, 1, s, tokens, scale
  )
end

warmup.times { run_adjoint.call; run_rowwise.call }
gpu_y = out_buf.read(tokens * s)
delta = max_output_delta(serial_y, gpu_y)

adjoint_ms = [] of Float64
rowwise_ms = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_adjoint.call }
  adjoint_ms << ms
  ms2, _ = elapsed_ms { run_rowwise.call }
  rowwise_ms << ms2
end
adj_p50 = adjoint_ms.sort[adjoint_ms.size // 2]
row_p50 = rowwise_ms.sort[rowwise_ms.size // 2]

puts "Qwen35 DeltaNet adjoint replay output microbench"
puts "s=#{s} tokens=#{tokens} runs=#{runs} warmup=#{warmup}"
puts "max_output_delta=#{delta}"
puts "adjoint_output_p50_ms=#{adj_p50.round(4)}"
puts "rowwise_chunk_p50_ms=#{row_p50.round(4)}"
puts "adjoint_vs_rowwise=#{(row_p50 / adj_p50).round(4)}"
puts "note=lower-bound output replay only; excludes transformed-query/additive-term construction and final-state prefix scan."
