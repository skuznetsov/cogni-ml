#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/core/buffer"

SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;

#define MAX_S 128

kernel void row_basis_compose_pairs(
    device const float* d1    [[buffer(0)]],
    device const float* d2    [[buffer(1)]],
    device const float* b1    [[buffer(2)]],
    device const float* b2    [[buffer(3)]],
    device const float* gamma2 [[buffer(4)]],
    device       float* d_out [[buffer(5)]],
    device       float* b_out [[buffer(6)]],
    constant     uint&  s     [[buffer(7)]],
    constant     uint&  pairs [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint row = gid.y;
    const uint p = gid.z;
    if (row >= s || col >= s || p >= pairs || s > MAX_S) return;

    const uint base = p * s * s;
    const uint idx = base + row * s + col;

    float d_prod = 0.0f;
    float b_prod = 0.0f;
    for (uint k = 0; k < s; ++k) {
        d_prod += d1[base + row * s + k] * d2[base + k * s + col];
        b_prod += b1[base + row * s + k] * d2[base + k * s + col];
    }

    const float g2 = gamma2[p];
    d_out[idx] = d1[idx] + d2[idx] + d_prod;
    b_out[idx] = b2[idx] + g2 * (b1[idx] + b_prod);
}

kernel void row_basis_compose_pairs_tiled(
    device const float* d1    [[buffer(0)]],
    device const float* d2    [[buffer(1)]],
    device const float* b1    [[buffer(2)]],
    device const float* b2    [[buffer(3)]],
    device const float* gamma2 [[buffer(4)]],
    device       float* d_out [[buffer(5)]],
    device       float* b_out [[buffer(6)]],
    constant     uint&  s     [[buffer(7)]],
    constant     uint&  pairs [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint2 tid = uint2(tid3.x, tid3.y);
    const uint col = gid.x;
    const uint row = gid.y;
    const uint p = gid.z;
    if (p >= pairs || s > MAX_S) return;

    threadgroup float d1_tile[16][16];
    threadgroup float b1_tile[16][16];
    threadgroup float d2_tile[16][16];

    const uint base = p * s * s;
    float d_prod = 0.0f;
    float b_prod = 0.0f;

    for (uint k0 = 0; k0 < s; k0 += 16) {
        const uint k_col = k0 + tid.x;
        const uint k_row = k0 + tid.y;

        d1_tile[tid.y][tid.x] = (row < s && k_col < s) ? d1[base + row * s + k_col] : 0.0f;
        b1_tile[tid.y][tid.x] = (row < s && k_col < s) ? b1[base + row * s + k_col] : 0.0f;
        d2_tile[tid.y][tid.x] = (k_row < s && col < s) ? d2[base + k_row * s + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 16; ++kk) {
            d_prod += d1_tile[tid.y][kk] * d2_tile[kk][tid.x];
            b_prod += b1_tile[tid.y][kk] * d2_tile[kk][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row >= s || col >= s) return;

    const uint idx = base + row * s + col;
    const float g2 = gamma2[p];
    d_out[idx] = d1[idx] + d2[idx] + d_prod;
    b_out[idx] = b2[idx] + g2 * (b1[idx] + b_prod);
}
METAL

private def elapsed_ms(&)
  t0 = Time.instant
  value = yield
  {(Time.instant - t0).total_milliseconds, value}
end

private def max_abs_delta(cpu : Array(Float32), gpu : Array(Float32), count : Int32) : Float64
  max = 0.0
  count.times do |i|
    delta = (cpu[i].to_f64 - gpu[i].to_f64).abs
    max = delta if delta > max
  end
  max
end

s = 128
pairs = 32
runs = 20
warmup = 3
seed = 0xC0DE_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_row_basis_compose_micro [--s N] [--pairs N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default/max: 128)") { |v| s = v.to_i }
  p.on("--pairs=N", "Independent summary pairs to compose in one dispatch (default: 32)") { |v| pairs = v.to_i }
  p.on("--runs=N", "Timed Metal runs (default: 20)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs (default: 3)") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "s must be <= 128" if s > 128
raise "pairs must be positive" unless pairs > 0
raise "runs must be positive" unless runs > 0
abort "Metal unavailable" unless ML::Metal::Device.init!

rng = Random.new(seed)
count = pairs * s * s
scale = 0.015_f32
d1 = Array.new(count) { ((rng.next_float - 0.5) * scale).to_f32 }
d2 = Array.new(count) { ((rng.next_float - 0.5) * scale).to_f32 }
b1 = Array.new(count) { ((rng.next_float - 0.5) * scale).to_f32 }
b2 = Array.new(count) { ((rng.next_float - 0.5) * scale).to_f32 }
gamma2 = Array.new(pairs) { (0.80 + 0.19 * rng.next_float).to_f32 }

cpu_d0 = Array.new(s * s, 0.0_f32)
cpu_b0 = Array.new(s * s, 0.0_f32)
s.times do |row|
  s.times do |col|
    d_prod = 0.0_f32
    b_prod = 0.0_f32
    s.times do |k|
      d_prod += d1[row * s + k] * d2[k * s + col]
      b_prod += b1[row * s + k] * d2[k * s + col]
    end
    idx = row * s + col
    cpu_d0[idx] = d1[idx] + d2[idx] + d_prod
    cpu_b0[idx] = b2[idx] + gamma2[0] * (b1[idx] + b_prod)
  end
end

d1_buf = ML::MetalBuffer.from_array(d1)
d2_buf = ML::MetalBuffer.from_array(d2)
b1_buf = ML::MetalBuffer.from_array(b1)
b2_buf = ML::MetalBuffer.from_array(b2)
gamma2_buf = ML::MetalBuffer.from_array(gamma2)
d_out_buf = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32))
b_out_buf = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32))
d_tiled_buf = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32))
b_tiled_buf = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32))
pipe = ML::Metal::ComputePipeline.new("row_basis_compose_pairs", SOURCE)
tiled_pipe = ML::Metal::ComputePipeline.new("row_basis_compose_pairs_tiled", SOURCE)
s_u = s.to_u32
pairs_u = pairs.to_u32

run_scalar = -> do
  ML::Metal::Dispatch.execute(pipe) do |enc|
    enc.set_buffer(d1_buf, 0)
    enc.set_buffer(d2_buf, 1)
    enc.set_buffer(b1_buf, 2)
    enc.set_buffer(b2_buf, 3)
    enc.set_buffer(gamma2_buf, 4)
    enc.set_buffer(d_out_buf, 5)
    enc.set_buffer(b_out_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(pairs_u, 8)
    enc.dispatch({s, s, pairs}, {16, 16, 1})
  end
end

run_tiled = -> do
  ML::Metal::Dispatch.execute(tiled_pipe) do |enc|
    enc.set_buffer(d1_buf, 0)
    enc.set_buffer(d2_buf, 1)
    enc.set_buffer(b1_buf, 2)
    enc.set_buffer(b2_buf, 3)
    enc.set_buffer(gamma2_buf, 4)
    enc.set_buffer(d_tiled_buf, 5)
    enc.set_buffer(b_tiled_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(pairs_u, 8)
    enc.dispatch({s, s, pairs}, {16, 16, 1})
  end
end

warmup.times { run_scalar.call; run_tiled.call }
gpu_d = d_out_buf.read(s * s)
gpu_b = b_out_buf.read(s * s)
gpu_td = d_tiled_buf.read(s * s)
gpu_tb = b_tiled_buf.read(s * s)
max_d = max_abs_delta(cpu_d0, gpu_d, s * s)
max_b = max_abs_delta(cpu_b0, gpu_b, s * s)
max_td = max_abs_delta(cpu_d0, gpu_td, s * s)
max_tb = max_abs_delta(cpu_b0, gpu_tb, s * s)

scalar_times = [] of Float64
tiled_times = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_scalar.call }
  scalar_times << ms
  ms2, _ = elapsed_ms { run_tiled.call }
  tiled_times << ms2
end
scalar_sorted = scalar_times.sort
tiled_sorted = tiled_times.sort
scalar_p50 = scalar_sorted[scalar_sorted.size // 2]
tiled_p50 = tiled_sorted[tiled_sorted.size // 2]
scalar_avg = scalar_times.sum / scalar_times.size
tiled_avg = tiled_times.sum / tiled_times.size

puts "Qwen35 DeltaNet row-basis summary compose microbench"
puts "s=#{s} pairs=#{pairs} runs=#{runs} warmup=#{warmup}"
puts "max_d_delta=#{max_d}"
puts "max_b_delta=#{max_b}"
puts "max_tiled_d_delta=#{max_td}"
puts "max_tiled_b_delta=#{max_tb}"
puts "scalar_compose_avg_ms=#{scalar_avg.round(4)}"
puts "scalar_compose_p50_ms=#{scalar_p50.round(4)}"
puts "scalar_per_pair_p50_ms=#{(scalar_p50 / pairs).round(6)}"
puts "tiled_compose_avg_ms=#{tiled_avg.round(4)}"
puts "tiled_compose_p50_ms=#{tiled_p50.round(4)}"
puts "tiled_per_pair_p50_ms=#{(tiled_p50 / pairs).round(6)}"
puts "tiled_vs_scalar=#{(scalar_p50 / tiled_p50).round(4)}"
puts "note=one prefix-scan level over row-basis summaries; computes D_out=D1+D2+D1*D2 and B_out=B2+gamma2*B1*(I+D2)."
