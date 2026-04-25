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

kernel void compact_a_dot_coeffs(
    device const float* k     [[buffer(0)]],
    device const float* beta  [[buffer(1)]],
    device       float* coeff [[buffer(2)]],
    constant     uint&  s     [[buffer(3)]],
    constant     uint&  rank  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint r = gid.x;
    const uint j = gid.y;
    if (r >= rank || j >= rank) return;
    if (j >= r) {
        coeff[r * rank + j] = 0.0f;
        return;
    }

    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += k[j * s + d] * (-beta[r] * k[r * s + d]);
    }
    coeff[r * rank + j] = acc;
}

kernel void compact_a_build_factors(
    device const float* k      [[buffer(0)]],
    device const float* beta   [[buffer(1)]],
    device const float* g      [[buffer(2)]],
    device const float* coeff  [[buffer(3)]],
    device       float* u_out  [[buffer(4)]],
    device       float* v_out  [[buffer(5)]],
    device       float* gamma  [[buffer(6)]],
    constant     uint&  s      [[buffer(7)]],
    constant     uint&  rank   [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    const uint d = gid;
    if (d >= s) return;

    for (uint r = 0; r < rank; ++r) {
        float val = -beta[r] * k[r * s + d];
        for (uint j = 0; j < r; ++j) {
            val += u_out[j * s + d] * coeff[r * rank + j];
        }
        u_out[r * s + d] = val;
        v_out[r * s + d] = k[r * s + d];
    }

    if (d == 0) {
        float prod = 1.0f;
        for (uint r = 0; r < rank; ++r) prod *= g[r];
        gamma[0] = prod;
    }
}
METAL

private def elapsed_ms(&)
  t0 = Time.instant
  value = yield
  {(Time.instant - t0).total_milliseconds, value}
end

private def flatten_vecs_f32(vs : Array(Array(Float64))) : Array(Float32)
  out = Array(Float32).new(vs.sum(&.size))
  vs.each { |row| row.each { |v| out << v.to_f32 } }
  out
end

private def max_abs_delta_vec(cpu : Array(Array(Float64)), gpu : Array(Float32)) : Float64
  max = 0.0
  s = cpu[0].size
  cpu.each_with_index do |row, r|
    row.each_with_index do |v, d|
      delta = (v - gpu[r * s + d].to_f64).abs
      max = delta if delta > max
    end
  end
  max
end

s = 128
block = 16
runs = 20
warmup = 3
seed = 0xB011D_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_compact_build_metal_micro [--s N] [--block N] [--runs N] [--warmup N]"
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
inputs = Array.new(block) do
  DeltaInputs.new(
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    0.88 + 0.10 * rng.next_float,
    rng.next_float
  )
end
cpu_transition = BlockScan.compact_transition_for_block(inputs)
k = inputs.map(&.k)
beta = inputs.map(&.beta.to_f32)
g = inputs.map(&.g.to_f32)

k_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(k))
beta_buf = ML::MetalBuffer.from_array(beta)
g_buf = ML::MetalBuffer.from_array(g)
coeff_buf = ML::MetalBuffer.new((block * block).to_i64 * sizeof(Float32))
u_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
v_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
gamma_buf = ML::MetalBuffer.new(sizeof(Float32).to_i64)

dot_pipe = ML::Metal::ComputePipeline.new("compact_a_dot_coeffs", SOURCE)
build_pipe = ML::Metal::ComputePipeline.new("compact_a_build_factors", SOURCE)
s_u = s.to_u32
rank_u = block.to_u32

run_once = -> do
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc1 = ML::Metal::ComputeEncoder.new(cmd)
    enc1.set_pipeline(dot_pipe)
    enc1.set_buffer(k_buf, 0)
    enc1.set_buffer(beta_buf, 1)
    enc1.set_buffer(coeff_buf, 2)
    enc1.set_value(s_u, 3)
    enc1.set_value(rank_u, 4)
    enc1.dispatch({block, block, 1}, {16, 16, 1})
    enc1.end_encoding

    enc2 = ML::Metal::ComputeEncoder.new(cmd)
    enc2.set_pipeline(build_pipe)
    enc2.set_buffer(k_buf, 0)
    enc2.set_buffer(beta_buf, 1)
    enc2.set_buffer(g_buf, 2)
    enc2.set_buffer(coeff_buf, 3)
    enc2.set_buffer(u_buf, 4)
    enc2.set_buffer(v_buf, 5)
    enc2.set_buffer(gamma_buf, 6)
    enc2.set_value(s_u, 7)
    enc2.set_value(rank_u, 8)
    enc2.dispatch({s, 1, 1}, {256, 1, 1})
    enc2.end_encoding
  end
end

warmup.times { run_once.call }
gpu_u = u_buf.read(block * s)
gpu_v = v_buf.read(block * s)
gpu_gamma = gamma_buf.read(1)[0]
max_u = max_abs_delta_vec(cpu_transition.u_cols, gpu_u)
max_v = max_abs_delta_vec(cpu_transition.v_cols, gpu_v)
gamma_delta = (cpu_transition.gamma - gpu_gamma.to_f64).abs

metal_ms = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_once.call }
  metal_ms << ms
end
metal_avg = metal_ms.sum / metal_ms.size
metal_sorted = metal_ms.sort
metal_p50 = metal_sorted[metal_sorted.size // 2]

cpu_ms, _ = elapsed_ms { BlockScan.compact_transition_for_block(inputs) }

puts "Qwen35 DeltaNet compact-A Metal build microbench"
puts "s=#{s} block=#{block} rank=#{block} runs=#{runs} warmup=#{warmup}"
puts "max_u_delta=#{max_u}"
puts "max_v_delta=#{max_v}"
puts "gamma_delta=#{gamma_delta}"
puts "metal_avg_ms=#{metal_avg.round(4)}"
puts "metal_p50_ms=#{metal_p50.round(4)}"
puts "cpu_build_ms=#{cpu_ms.round(4)}"
puts "speedup_vs_cpu_build=#{(cpu_ms / metal_p50).round(3)}"
puts "note=synthetic compact-A construction only; excludes compact-B right factors, prefix scan, replay, and prefill integration."
