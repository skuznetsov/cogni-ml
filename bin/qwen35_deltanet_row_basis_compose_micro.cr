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
pipe = ML::Metal::ComputePipeline.new("row_basis_compose_pairs", SOURCE)
s_u = s.to_u32
pairs_u = pairs.to_u32

run_once = -> do
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

warmup.times { run_once.call }
gpu_d = d_out_buf.read(s * s)
gpu_b = b_out_buf.read(s * s)
max_d = max_abs_delta(cpu_d0, gpu_d, s * s)
max_b = max_abs_delta(cpu_b0, gpu_b, s * s)

times = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_once.call }
  times << ms
end
sorted = times.sort
p50 = sorted[sorted.size // 2]
avg = times.sum / times.size

puts "Qwen35 DeltaNet row-basis summary compose microbench"
puts "s=#{s} pairs=#{pairs} runs=#{runs} warmup=#{warmup}"
puts "max_d_delta=#{max_d}"
puts "max_b_delta=#{max_b}"
puts "compose_avg_ms=#{avg.round(4)}"
puts "compose_p50_ms=#{p50.round(4)}"
puts "compose_per_pair_p50_ms=#{(p50 / pairs).round(6)}"
puts "note=one prefix-scan level over row-basis summaries; computes D_out=D1+D2+D1*D2 and B_out=B2+gamma2*B1*(I+D2)."
