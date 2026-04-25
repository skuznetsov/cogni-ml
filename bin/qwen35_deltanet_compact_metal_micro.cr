#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/core/buffer"
require "../src/ml/gguf/qwen35_deltanet_block_scan"

alias BlockScan = ML::GGUF::Qwen35DeltaNetBlockScan
alias DeltaInputs = ML::GGUF::Qwen35DeltaNetBlockScan::DeltaInputs

SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;

kernel void compact_transition_coeffs(
    device const float* state  [[buffer(0)]],
    device const float* u_cols [[buffer(1)]],
    device       float* coeffs [[buffer(2)]],
    constant     uint&  s      [[buffer(3)]],
    constant     uint&  rank   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint r = gid.y;
    if (row >= s || r >= rank) return;

    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += state[row * s + d] * u_cols[r * s + d];
    }
    coeffs[row * rank + r] = acc;
}

kernel void compact_summary_apply(
    device const float* state    [[buffer(0)]],
    device const float* coeffs   [[buffer(1)]],
    device const float* v_cols   [[buffer(2)]],
    device const float* b_lefts  [[buffer(3)]],
    device const float* b_rights [[buffer(4)]],
    device       float* out      [[buffer(5)]],
    constant     float& gamma    [[buffer(6)]],
    constant     uint&  s        [[buffer(7)]],
    constant     uint&  rank     [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint col = gid.y;
    if (row >= s || col >= s) return;

    float acc = state[row * s + col];
    for (uint r = 0; r < rank; ++r) {
        acc += coeffs[row * rank + r] * v_cols[r * s + col];
    }
    acc *= gamma;

    for (uint r = 0; r < rank; ++r) {
        acc += b_lefts[r * s + row] * b_rights[r * s + col];
    }

    out[row * s + col] = acc;
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

private def max_abs_delta_f32(cpu : BlockScan::Matrix, gpu : Array(Float32)) : Float64
  max = 0.0
  s = cpu.size
  s.times do |i|
    s.times do |j|
      d = (cpu[i][j] - gpu[i * s + j].to_f64).abs
      max = d if d > max
    end
  end
  max
end

s = 128
block = 16
runs = 20
warmup = 3
seed = 0xC0DA_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_compact_metal_micro [--s N] [--block N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default: 128)") { |v| s = v.to_i }
  p.on("--block=N", "Summary rank/block size (default: 16)") { |v| block = v.to_i }
  p.on("--runs=N", "Timed Metal runs (default: 20)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup Metal runs (default: 3)") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "block must be positive" unless block > 0
raise "runs must be positive" unless runs > 0

unless ML::Metal::Device.init!
  abort "Metal unavailable"
end

rng = Random.new(seed)
state = Array.new(s) { Array.new(s) { ((rng.next_float - 0.5) * 0.2) } }
inputs = Array.new(block) do
  DeltaInputs.new(
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    0.88 + 0.10 * rng.next_float,
    rng.next_float
  )
end
summary = BlockScan.fully_compact_summary_for_block(inputs)
cpu_out = BlockScan.apply_fully_compact(state, summary)
rank = summary.b_lefts.size
raise "rank mismatch" unless rank == block

state_buf = ML::MetalBuffer.from_array(flatten_f32(state))
u_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(summary.transition.u_cols))
v_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(summary.transition.v_cols))
bl_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(summary.b_lefts))
br_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(summary.b_rights))
coeff_buf = ML::MetalBuffer.new((s * rank).to_i64 * sizeof(Float32))
out_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))

coeff_pipe = ML::Metal::ComputePipeline.new("compact_transition_coeffs", SOURCE)
apply_pipe = ML::Metal::ComputePipeline.new("compact_summary_apply", SOURCE)
s_u = s.to_u32
rank_u = rank.to_u32
gamma = summary.transition.gamma.to_f32

run_once = -> do
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc1 = ML::Metal::ComputeEncoder.new(cmd)
    enc1.set_pipeline(coeff_pipe)
    enc1.set_buffer(state_buf, 0)
    enc1.set_buffer(u_buf, 1)
    enc1.set_buffer(coeff_buf, 2)
    enc1.set_value(s_u, 3)
    enc1.set_value(rank_u, 4)
    enc1.dispatch({s, rank, 1}, {16, 16, 1})
    enc1.end_encoding

    enc2 = ML::Metal::ComputeEncoder.new(cmd)
    enc2.set_pipeline(apply_pipe)
    enc2.set_buffer(state_buf, 0)
    enc2.set_buffer(coeff_buf, 1)
    enc2.set_buffer(v_buf, 2)
    enc2.set_buffer(bl_buf, 3)
    enc2.set_buffer(br_buf, 4)
    enc2.set_buffer(out_buf, 5)
    enc2.set_value(gamma, 6)
    enc2.set_value(s_u, 7)
    enc2.set_value(rank_u, 8)
    enc2.dispatch({s, s, 1}, {16, 16, 1})
    enc2.end_encoding
  end
end

warmup.times { run_once.call }
gpu_out = out_buf.read(s * s)
max_delta = max_abs_delta_f32(cpu_out, gpu_out)

metal_ms = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_once.call }
  metal_ms << ms
end
metal_avg = metal_ms.sum / metal_ms.size
metal_sorted = metal_ms.sort
metal_p50 = metal_sorted[metal_sorted.size // 2]

cpu_ms, _ = elapsed_ms { BlockScan.apply_fully_compact(state, summary) }

puts "Qwen35 DeltaNet fully compact Metal microbench"
puts "s=#{s} block=#{block} rank=#{rank} runs=#{runs} warmup=#{warmup}"
puts "max_delta_vs_cpu_f64=#{max_delta}"
puts "metal_avg_ms=#{metal_avg.round(4)}"
puts "metal_p50_ms=#{metal_p50.round(4)}"
puts "cpu_apply_ms=#{cpu_ms.round(4)}"
puts "speedup_vs_cpu_apply=#{(cpu_ms / metal_p50).round(3)}"
puts "note=synthetic summary apply only; excludes summary construction and prefill integration."
