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

#define MAX_S 128

kernel void compact_b_build_rights(
    device const float* k      [[buffer(0)]],
    device const float* beta   [[buffer(1)]],
    device const float* g      [[buffer(2)]],
    device       float* rights [[buffer(3)]],
    constant     uint&  s      [[buffer(4)]],
    constant     uint&  rank   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= rank || s > MAX_S) return;

    float r[MAX_S];
    for (uint d = 0; d < s; ++d) {
        r[d] = beta[i] * k[i * s + d];
    }

    for (uint t = i + 1; t < rank; ++t) {
        float dot = 0.0f;
        for (uint d = 0; d < s; ++d) {
            dot += r[d] * k[t * s + d];
        }
        const float scale = -beta[t] * dot;
        const float gt = g[t];
        for (uint d = 0; d < s; ++d) {
            r[d] = gt * (r[d] + scale * k[t * s + d]);
        }
    }

    for (uint d = 0; d < s; ++d) {
        rights[i * s + d] = r[d];
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
seed = 0xBEEF_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_compact_b_metal_micro [--s N] [--block N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default/max: 128)") { |v| s = v.to_i }
  p.on("--block=N", "Summary rank/block size (default: 16)") { |v| block = v.to_i }
  p.on("--runs=N", "Timed Metal runs (default: 20)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup Metal runs (default: 3)") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "s must be <= 128 for this microbench" if s > 128
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
cpu_summary = BlockScan.compact_summary_for_block(inputs)
k = inputs.map(&.k)
beta = inputs.map(&.beta.to_f32)
g = inputs.map(&.g.to_f32)

k_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(k))
beta_buf = ML::MetalBuffer.from_array(beta)
g_buf = ML::MetalBuffer.from_array(g)
rights_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))

pipe = ML::Metal::ComputePipeline.new("compact_b_build_rights", SOURCE)
s_u = s.to_u32
rank_u = block.to_u32

run_once = -> do
  ML::Metal::Dispatch.execute(pipe) do |enc|
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(rights_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(rank_u, 5)
    enc.dispatch({block, 1, 1}, {Math.min(block, 64), 1, 1})
  end
end

warmup.times { run_once.call }
gpu_rights = rights_buf.read(block * s)
max_right = max_abs_delta_vec(cpu_summary.b_rights, gpu_rights)

metal_ms = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_once.call }
  metal_ms << ms
end
metal_avg = metal_ms.sum / metal_ms.size
metal_sorted = metal_ms.sort
metal_p50 = metal_sorted[metal_sorted.size // 2]

cpu_ms, _ = elapsed_ms { BlockScan.compact_summary_for_block(inputs) }

puts "Qwen35 DeltaNet compact-B Metal right-factor microbench"
puts "s=#{s} block=#{block} rank=#{block} runs=#{runs} warmup=#{warmup}"
puts "max_right_delta=#{max_right}"
puts "metal_avg_ms=#{metal_avg.round(4)}"
puts "metal_p50_ms=#{metal_p50.round(4)}"
puts "cpu_compact_summary_ms=#{cpu_ms.round(4)}"
puts "speedup_vs_cpu_summary=#{(cpu_ms / metal_p50).round(3)}"
puts "note=synthetic compact-B right-factor construction only; excludes compact-A build, prefix scan, replay, and prefill integration."
