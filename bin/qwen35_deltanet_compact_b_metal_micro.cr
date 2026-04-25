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

kernel void compact_a_dot_coeffs_blocks(
    device const float* k     [[buffer(0)]],
    device const float* beta  [[buffer(1)]],
    device       float* coeff [[buffer(2)]],
    constant     uint&  s     [[buffer(3)]],
    constant     uint&  rank  [[buffer(4)]],
    constant     uint&  nblk  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint r = gid.x;
    const uint j = gid.y;
    const uint b = gid.z;
    if (r >= rank || j >= rank || b >= nblk) return;
    const uint coeff_base = (b * rank + r) * rank;
    if (j >= r) { coeff[coeff_base + j] = 0.0f; return; }

    const uint rtok = b * rank + r;
    const uint jtok = b * rank + j;
    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += k[jtok * s + d] * (-beta[rtok] * k[rtok * s + d]);
    }
    coeff[coeff_base + j] = acc;
}

kernel void compact_a_build_factors_blocks(
    device const float* k      [[buffer(0)]],
    device const float* beta   [[buffer(1)]],
    device const float* g      [[buffer(2)]],
    device const float* coeff  [[buffer(3)]],
    device       float* u_out  [[buffer(4)]],
    device       float* v_out  [[buffer(5)]],
    device       float* gamma  [[buffer(6)]],
    constant     uint&  s      [[buffer(7)]],
    constant     uint&  rank   [[buffer(8)]],
    constant     uint&  nblk   [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= s || b >= nblk) return;

    for (uint r = 0; r < rank; ++r) {
        const uint tok = b * rank + r;
        const uint out_base = (b * rank + r) * s;
        const uint coeff_base = (b * rank + r) * rank;
        float val = -beta[tok] * k[tok * s + d];
        for (uint j = 0; j < r; ++j) {
            val += u_out[(b * rank + j) * s + d] * coeff[coeff_base + j];
        }
        u_out[out_base + d] = val;
        v_out[out_base + d] = k[tok * s + d];
    }

    if (d == 0) {
        float prod = 1.0f;
        for (uint r = 0; r < rank; ++r) {
            prod *= g[b * rank + r];
        }
        gamma[b] = prod;
    }
}

kernel void compact_b_build_rights_tg(
    device const float* k      [[buffer(0)]],
    device const float* beta   [[buffer(1)]],
    device const float* g      [[buffer(2)]],
    device       float* rights [[buffer(3)]],
    constant     uint&  s      [[buffer(4)]],
    constant     uint&  rank   [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    const uint i = tg.x;
    if (i >= rank || s > MAX_S) return;

    threadgroup float r[MAX_S];
    threadgroup float partial[MAX_S];

    if (tid < s) {
        r[tid] = beta[i] * k[i * s + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = i + 1; t < rank; ++t) {
        partial[tid] = (tid < s) ? (r[tid] * k[t * s + tid]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = MAX_S >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float scale = -beta[t] * partial[0];
        const float gt = g[t];
        if (tid < s) {
            r[tid] = gt * (r[tid] + scale * k[t * s + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < s) {
        rights[i * s + tid] = r[tid];
    }
}

kernel void compact_b_build_rights_blocks_tg(
    device const float* k      [[buffer(0)]],
    device const float* beta   [[buffer(1)]],
    device const float* g      [[buffer(2)]],
    device       float* rights [[buffer(3)]],
    constant     uint&  s      [[buffer(4)]],
    constant     uint&  rank   [[buffer(5)]],
    constant     uint&  nblk   [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    const uint i = tg.x;
    const uint b = tg.y;
    if (i >= rank || b >= nblk || s > MAX_S) return;

    const uint base = b * rank;
    const uint tok = base + i;
    threadgroup float r[MAX_S];
    threadgroup float partial[MAX_S];

    if (tid < s) {
        r[tid] = beta[tok] * k[tok * s + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_t = i + 1; local_t < rank; ++local_t) {
        const uint t = base + local_t;
        partial[tid] = (tid < s) ? (r[tid] * k[t * s + tid]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = MAX_S >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float scale = -beta[t] * partial[0];
        const float gt = g[t];
        if (tid < s) {
            r[tid] = gt * (r[tid] + scale * k[t * s + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < s) {
        rights[(b * rank + i) * s + tid] = r[tid];
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

private def max_abs_delta_vec32(cpu : Array(Array(Float64)), gpu : Array(Float32), offset : Int32 = 0) : Float64
  max = 0.0
  s = cpu[0].size
  cpu.each_with_index do |row, r|
    row.each_with_index do |v, d|
      delta = (v - gpu[offset + r * s + d].to_f64).abs
      max = delta if delta > max
    end
  end
  max
end

s = 128
block = 16
blocks = 1
runs = 20
warmup = 3
seed = 0xBEEF_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_compact_b_metal_micro [--s N] [--block N] [--blocks N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default/max: 128)") { |v| s = v.to_i }
  p.on("--block=N", "Summary rank/block size (default: 16)") { |v| block = v.to_i }
  p.on("--blocks=N", "Independent blocks to build in one dispatch (default: 1)") { |v| blocks = v.to_i }
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
raise "blocks must be positive" unless blocks > 0
raise "runs must be positive" unless runs > 0

unless ML::Metal::Device.init!
  abort "Metal unavailable"
end

rng = Random.new(seed)
all_inputs = Array.new(block * blocks) do
  DeltaInputs.new(
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    0.88 + 0.10 * rng.next_float,
    rng.next_float
  )
end
inputs = all_inputs[0, block]
cpu_summary = BlockScan.compact_summary_for_block(inputs)
cpu_block_summaries = Array.new(blocks) do |b|
  BlockScan.compact_summary_for_block(all_inputs[b * block, block])
end
k = all_inputs.map(&.k)
q = all_inputs.map(&.q)
v = all_inputs.map(&.v)
beta = all_inputs.map(&.beta.to_f32)
g = all_inputs.map(&.g.to_f32)

k_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(k))
beta_buf = ML::MetalBuffer.from_array(beta)
g_buf = ML::MetalBuffer.from_array(g)
rights_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
rights_tg_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
rights_blocks_tg_buf = ML::MetalBuffer.new((blocks * block * s).to_i64 * sizeof(Float32))
coeff_a_blocks_buf = ML::MetalBuffer.new((blocks * block * block).to_i64 * sizeof(Float32))
u_blocks_buf = ML::MetalBuffer.new((blocks * block * s).to_i64 * sizeof(Float32))
v_blocks_buf = ML::MetalBuffer.new((blocks * block * s).to_i64 * sizeof(Float32))
gamma_blocks_buf = ML::MetalBuffer.new(blocks.to_i64 * sizeof(Float32))

pipe = ML::Metal::ComputePipeline.new("compact_b_build_rights", SOURCE)
pa_dot_blocks = ML::Metal::ComputePipeline.new("compact_a_dot_coeffs_blocks", SOURCE)
pa_build_blocks = ML::Metal::ComputePipeline.new("compact_a_build_factors_blocks", SOURCE)
tg_pipe = ML::Metal::ComputePipeline.new("compact_b_build_rights_tg", SOURCE)
blocks_tg_pipe = ML::Metal::ComputePipeline.new("compact_b_build_rights_blocks_tg", SOURCE)
s_u = s.to_u32
rank_u = block.to_u32
nblk_u = blocks.to_u32

run_once = ->(out_buf : ML::MetalBuffer) do
  ML::Metal::Dispatch.execute(pipe) do |enc|
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(out_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(rank_u, 5)
    enc.dispatch({block, 1, 1}, {Math.min(block, 64), 1, 1})
  end
end

run_tg_once = -> do
  ML::Metal::Dispatch.execute(tg_pipe) do |enc|
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(rights_tg_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(rank_u, 5)
    enc.dispatch_threadgroups({block, 1, 1}, {128, 1, 1})
  end
end

run_blocks_tg_once = -> do
  ML::Metal::Dispatch.execute(blocks_tg_pipe) do |enc|
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(rights_blocks_tg_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(rank_u, 5)
    enc.set_value(nblk_u, 6)
    enc.dispatch_threadgroups({block, blocks, 1}, {128, 1, 1})
  end
end

run_a_blocks_once = -> do
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pa_dot_blocks)
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(coeff_a_blocks_buf, 2)
    enc.set_value(s_u, 3)
    enc.set_value(rank_u, 4)
    enc.set_value(nblk_u, 5)
    enc.dispatch({block, block, blocks}, {16, 16, 1})
    enc.end_encoding

    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pa_build_blocks)
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(coeff_a_blocks_buf, 3)
    enc.set_buffer(u_blocks_buf, 4)
    enc.set_buffer(v_blocks_buf, 5)
    enc.set_buffer(gamma_blocks_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(rank_u, 8)
    enc.set_value(nblk_u, 9)
    enc.dispatch({s, blocks, 1}, {128, 1, 1})
    enc.end_encoding
  end
end

run_summary_blocks_once = -> do
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pa_dot_blocks)
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(coeff_a_blocks_buf, 2)
    enc.set_value(s_u, 3)
    enc.set_value(rank_u, 4)
    enc.set_value(nblk_u, 5)
    enc.dispatch({block, block, blocks}, {16, 16, 1})
    enc.end_encoding

    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pa_build_blocks)
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(coeff_a_blocks_buf, 3)
    enc.set_buffer(u_blocks_buf, 4)
    enc.set_buffer(v_blocks_buf, 5)
    enc.set_buffer(gamma_blocks_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(rank_u, 8)
    enc.set_value(nblk_u, 9)
    enc.dispatch({s, blocks, 1}, {128, 1, 1})
    enc.end_encoding

    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(blocks_tg_pipe)
    enc.set_buffer(k_buf, 0)
    enc.set_buffer(beta_buf, 1)
    enc.set_buffer(g_buf, 2)
    enc.set_buffer(rights_blocks_tg_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(rank_u, 5)
    enc.set_value(nblk_u, 6)
    enc.dispatch_threadgroups({block, blocks, 1}, {128, 1, 1})
    enc.end_encoding
  end
end

rowwise_state = Array.new(s) { Array.new(s) { ((rng.next_float - 0.5) * 0.2) } }
rowwise_state_buf = ML::MetalBuffer.from_array(rowwise_state.flat_map { |row| row.map(&.to_f32) })
q_flat = flatten_vecs_f32(q)
k_flat = flatten_vecs_f32(k)
v_flat = flatten_vecs_f32(v)
run_rowwise_once = -> do
  rowwise_state_buf.write(rowwise_state.flat_map { |row| row.map(&.to_f32) })
  ML::GGUF::Qwen35Metal.delta_net_chunk(
    rowwise_state_buf,
    q_flat,
    k_flat,
    v_flat,
    g,
    beta,
    1, 1, s, block * blocks, 1.0_f32
  )
end

warmup.times { run_once.call(rights_buf); run_tg_once.call; run_blocks_tg_once.call; run_a_blocks_once.call; run_summary_blocks_once.call; run_rowwise_once.call }
gpu_rights = rights_buf.read(block * s)
gpu_rights_tg = rights_tg_buf.read(block * s)
gpu_rights_blocks_tg = rights_blocks_tg_buf.read(blocks * block * s)
gpu_u_blocks = u_blocks_buf.read(blocks * block * s)
gpu_v_blocks = v_blocks_buf.read(blocks * block * s)
gpu_gamma_blocks = gamma_blocks_buf.read(blocks)
max_right = max_abs_delta_vec(cpu_summary.b_rights, gpu_rights)
max_right_tg = max_abs_delta_vec(cpu_summary.b_rights, gpu_rights_tg)
max_right_blocks_tg = cpu_block_summaries.each_with_index.reduce(0.0) do |max, (summary, b)|
  offset = b * block * s
  block_delta = max_abs_delta_vec(summary.b_rights, gpu_rights_blocks_tg[offset, block * s])
  block_delta > max ? block_delta : max
end
cpu_block_transitions = Array.new(blocks) do |b|
  BlockScan.compact_transition_for_block(all_inputs[b * block, block])
end
max_a_u_blocks = cpu_block_transitions.each_with_index.reduce(0.0) do |max, (tr, b)|
  delta = max_abs_delta_vec32(tr.u_cols, gpu_u_blocks, b * block * s)
  delta > max ? delta : max
end
max_a_v_blocks = cpu_block_transitions.each_with_index.reduce(0.0) do |max, (tr, b)|
  delta = max_abs_delta_vec32(tr.v_cols, gpu_v_blocks, b * block * s)
  delta > max ? delta : max
end
max_a_gamma_blocks = cpu_block_transitions.each_with_index.reduce(0.0) do |max, (tr, b)|
  delta = (tr.gamma - gpu_gamma_blocks[b].to_f64).abs
  delta > max ? delta : max
end

metal_ms = [] of Float64
metal_tg_ms = [] of Float64
metal_blocks_tg_ms = [] of Float64
a_blocks_ms = [] of Float64
summary_blocks_ms = [] of Float64
rowwise_ms = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_once.call(rights_buf) }
  metal_ms << ms
  tg_ms, _ = elapsed_ms { run_tg_once.call }
  metal_tg_ms << tg_ms
  blocks_tg_ms, _ = elapsed_ms { run_blocks_tg_once.call }
  metal_blocks_tg_ms << blocks_tg_ms
  a_ms, _ = elapsed_ms { run_a_blocks_once.call }
  a_blocks_ms << a_ms
  summary_ms, _ = elapsed_ms { run_summary_blocks_once.call }
  summary_blocks_ms << summary_ms
  rowwise_t, _ = elapsed_ms { run_rowwise_once.call }
  rowwise_ms << rowwise_t
end
metal_avg = metal_ms.sum / metal_ms.size
metal_sorted = metal_ms.sort
metal_p50 = metal_sorted[metal_sorted.size // 2]
metal_tg_avg = metal_tg_ms.sum / metal_tg_ms.size
metal_tg_sorted = metal_tg_ms.sort
metal_tg_p50 = metal_tg_sorted[metal_tg_sorted.size // 2]
metal_blocks_tg_avg = metal_blocks_tg_ms.sum / metal_blocks_tg_ms.size
metal_blocks_tg_sorted = metal_blocks_tg_ms.sort
metal_blocks_tg_p50 = metal_blocks_tg_sorted[metal_blocks_tg_sorted.size // 2]
a_blocks_avg = a_blocks_ms.sum / a_blocks_ms.size
a_blocks_sorted = a_blocks_ms.sort
a_blocks_p50 = a_blocks_sorted[a_blocks_sorted.size // 2]
summary_blocks_avg = summary_blocks_ms.sum / summary_blocks_ms.size
summary_blocks_sorted = summary_blocks_ms.sort
summary_blocks_p50 = summary_blocks_sorted[summary_blocks_sorted.size // 2]
rowwise_avg = rowwise_ms.sum / rowwise_ms.size
rowwise_sorted = rowwise_ms.sort
rowwise_p50 = rowwise_sorted[rowwise_sorted.size // 2]

cpu_ms, _ = elapsed_ms { BlockScan.compact_summary_for_block(inputs) }
cpu_blocks_ms, _ = elapsed_ms do
  blocks.times { |b| BlockScan.compact_summary_for_block(all_inputs[b * block, block]) }
end

puts "Qwen35 DeltaNet compact-B Metal right-factor microbench"
puts "s=#{s} block=#{block} blocks=#{blocks} tokens=#{block * blocks} rank=#{block} runs=#{runs} warmup=#{warmup}"
puts "max_right_delta=#{max_right}"
puts "max_right_tg_delta=#{max_right_tg}"
puts "max_right_blocks_tg_delta=#{max_right_blocks_tg}"
puts "max_a_u_blocks_delta=#{max_a_u_blocks}"
puts "max_a_v_blocks_delta=#{max_a_v_blocks}"
puts "max_a_gamma_blocks_delta=#{max_a_gamma_blocks}"
puts "metal_avg_ms=#{metal_avg.round(4)}"
puts "metal_p50_ms=#{metal_p50.round(4)}"
puts "metal_tg_avg_ms=#{metal_tg_avg.round(4)}"
puts "metal_tg_p50_ms=#{metal_tg_p50.round(4)}"
puts "metal_blocks_tg_avg_ms=#{metal_blocks_tg_avg.round(4)}"
puts "metal_blocks_tg_p50_ms=#{metal_blocks_tg_p50.round(4)}"
puts "metal_a_blocks_avg_ms=#{a_blocks_avg.round(4)}"
puts "metal_a_blocks_p50_ms=#{a_blocks_p50.round(4)}"
puts "metal_summary_blocks_avg_ms=#{summary_blocks_avg.round(4)}"
puts "metal_summary_blocks_p50_ms=#{summary_blocks_p50.round(4)}"
puts "rowwise_chunk_avg_ms=#{rowwise_avg.round(4)}"
puts "rowwise_chunk_p50_ms=#{rowwise_p50.round(4)}"
puts "tg_vs_scalar=#{(metal_p50 / metal_tg_p50).round(3)}"
puts "blocks_tg_per_block_ms=#{(metal_blocks_tg_p50 / blocks).round(4)}"
puts "summary_blocks_per_block_ms=#{(summary_blocks_p50 / blocks).round(4)}"
puts "blocks_tg_vs_rowwise=#{(rowwise_p50 / metal_blocks_tg_p50).round(3)}"
puts "summary_blocks_vs_rowwise=#{(rowwise_p50 / summary_blocks_p50).round(3)}"
puts "cpu_compact_summary_ms=#{cpu_ms.round(4)}"
puts "cpu_blocks_summary_ms=#{cpu_blocks_ms.round(4)}"
puts "speedup_vs_cpu_summary=#{(cpu_ms / metal_p50).round(3)}"
puts "tg_speedup_vs_cpu_summary=#{(cpu_ms / metal_tg_p50).round(3)}"
puts "blocks_tg_speedup_vs_cpu_blocks=#{(cpu_blocks_ms / metal_blocks_tg_p50).round(3)}"
puts "note=synthetic compact-B right-factor construction only; excludes compact-A build, prefix scan, replay, and prefill integration. Rowwise baseline computes full DeltaNet outputs and final state."
