#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/core/buffer"
require "../src/ml/gguf/qwen35_deltanet_block_scan"

alias BlockScan = ML::GGUF::Qwen35DeltaNetBlockScan

SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;

#define MAX_S 128
#define MAX_FACTORS 256

kernel void factor_basis_compress_serial(
    device const float* lefts      [[buffer(0)]],
    device const float* rights     [[buffer(1)]],
    device       float* out_lefts  [[buffer(2)]],
    device       float* out_rights [[buffer(3)]],
    device       float* out_rank   [[buffer(4)]],
    device       float* reduced    [[buffer(5)]],
    device       uint*  pivots     [[buffer(6)]],
    constant     uint&  s          [[buffer(7)]],
    constant     uint&  n_factors  [[buffer(8)]],
    constant     float& eps        [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0 || s > MAX_S || n_factors > MAX_FACTORS) return;

    uint rank = 0;

    for (uint i = 0; i < n_factors; ++i) {
        float residual[MAX_S];
        for (uint d = 0; d < s; ++d) {
            residual[d] = lefts[i * s + d];
        }

        for (uint r = 0; r < rank; ++r) {
            const float factor = residual[pivots[r]];
            if (factor != 0.0f) {
                for (uint d = 0; d < s; ++d) {
                    residual[d] -= factor * reduced[r * MAX_S + d];
                }
            }
        }

        uint pivot = 0;
        float pivot_abs = 0.0f;
        for (uint d = 0; d < s; ++d) {
            const float av = fabs(residual[d]);
            if (av > pivot_abs) {
                pivot_abs = av;
                pivot = d;
            }
        }
        if (pivot_abs <= eps || rank >= s) continue;

        const float inv_pivot = 1.0f / residual[pivot];
        for (uint d = 0; d < s; ++d) {
            reduced[rank * MAX_S + d] = residual[d] * inv_pivot;
            out_lefts[rank * s + d] = residual[d] * inv_pivot;
            out_rights[rank * s + d] = 0.0f;
        }
        pivots[rank] = pivot;
        rank += 1;
    }

    for (uint i = 0; i < n_factors; ++i) {
        float residual[MAX_S];
        float coeff[MAX_S];
        for (uint d = 0; d < s; ++d) {
            residual[d] = lefts[i * s + d];
            coeff[d] = 0.0f;
        }

        for (uint r = 0; r < rank; ++r) {
            const float factor = residual[pivots[r]];
            coeff[r] = factor;
            if (factor != 0.0f) {
                for (uint d = 0; d < s; ++d) {
                    residual[d] -= factor * reduced[r * MAX_S + d];
                }
            }
        }

        for (uint r = 0; r < rank; ++r) {
            const float c = coeff[r];
            if (c != 0.0f) {
                for (uint d = 0; d < s; ++d) {
                    out_rights[r * s + d] += c * rights[i * s + d];
                }
            }
        }
    }

    out_rank[0] = float(rank);
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

private def dense_from_factors(lefts : Array(Float32), rights : Array(Float32), rank : Int32, s : Int32) : Array(Float32)
  dense = Array.new(s * s, 0.0_f32)
  rank.times do |r|
    s.times do |i|
      l = lefts[r * s + i]
      next if l == 0.0_f32
      s.times do |j|
        dense[i * s + j] += l * rights[r * s + j]
      end
    end
  end
  dense
end

private def max_abs_delta(a : Array(Float32), b : Array(Float32)) : Float64
  max = 0.0
  a.size.times do |i|
    d = (a[i].to_f64 - b[i].to_f64).abs
    max = d if d > max
  end
  max
end

s = 128
n_factors = 256
runs = 10
warmup = 2
seed = 0xFA_C70B_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_factor_basis_compress_micro [--s N] [--factors N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default/max: 128)") { |v| s = v.to_i }
  p.on("--factors=N", "Input factor count (default/max: 256)") { |v| n_factors = v.to_i }
  p.on("--runs=N", "Timed Metal runs (default: 10)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs (default: 2)") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "s must be <= 128" if s > 128
raise "factors must be positive" unless n_factors > 0
raise "factors must be <= 256" if n_factors > 256
raise "runs must be positive" unless runs > 0
abort "Metal unavailable" unless ML::Metal::Device.init!

rng = Random.new(seed)
lefts = Array.new(n_factors) do
  Array.new(s) { ((rng.next_float - 0.5) * 0.5) }
end
rights = Array.new(n_factors) do
  Array.new(s) { ((rng.next_float - 0.5) * 0.5) }
end
cpu_lefts, cpu_rights = BlockScan.compress_outer_factors_left_basis(lefts, rights, 1.0e-7)
cpu_dense = dense_from_factors(flatten_vecs_f32(lefts), flatten_vecs_f32(rights), n_factors, s)

lefts_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(lefts))
rights_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(rights))
out_lefts_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))
out_rights_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))
rank_buf = ML::MetalBuffer.new(sizeof(Float32).to_i64)
reduced_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))
pivots_buf = ML::MetalBuffer.new(s.to_i64 * sizeof(UInt32))
pipe = ML::Metal::ComputePipeline.new("factor_basis_compress_serial", SOURCE)
s_u = s.to_u32
n_factors_u = n_factors.to_u32
eps = 1.0e-7_f32

run_once = -> do
  ML::Metal::Dispatch.execute(pipe) do |enc|
    enc.set_buffer(lefts_buf, 0)
    enc.set_buffer(rights_buf, 1)
    enc.set_buffer(out_lefts_buf, 2)
    enc.set_buffer(out_rights_buf, 3)
    enc.set_buffer(rank_buf, 4)
    enc.set_buffer(reduced_buf, 5)
    enc.set_buffer(pivots_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(n_factors_u, 8)
    enc.set_value(eps, 9)
    enc.dispatch({1, 1, 1}, {1, 1, 1})
  end
end

warmup.times { run_once.call }
rank = rank_buf.read(1)[0].round.to_i
gpu_lefts = out_lefts_buf.read(s * s)
gpu_rights = out_rights_buf.read(s * s)
gpu_dense = dense_from_factors(gpu_lefts, gpu_rights, rank, s)
cpu_compressed_dense = dense_from_factors(flatten_vecs_f32(cpu_lefts), flatten_vecs_f32(cpu_rights), cpu_lefts.size, s)

times = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_once.call }
  times << ms
end
p50 = times.sort[times.size // 2]
avg = times.sum / times.size

puts "Qwen35 DeltaNet factor-basis compression Metal microbench"
puts "s=#{s} factors=#{n_factors} runs=#{runs} warmup=#{warmup}"
puts "cpu_rank=#{cpu_lefts.size}"
puts "gpu_rank=#{rank}"
puts "max_cpu_compressed_delta=#{max_abs_delta(cpu_dense, cpu_compressed_dense)}"
puts "max_gpu_delta=#{max_abs_delta(cpu_dense, gpu_dense)}"
puts "metal_avg_ms=#{avg.round(4)}"
puts "metal_p50_ms=#{p50.round(4)}"
puts "note=single-thread serial basis-selection cost gate; proves whether naive GPU factor compression is viable, not a tuned production kernel."
