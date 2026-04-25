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
#define MAX_RANK 64

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
    if (j >= r) { coeff[r * rank + j] = 0.0f; return; }
    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) acc += k[j * s + d] * (-beta[r] * k[r * s + d]);
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
        for (uint j = 0; j < r; ++j) val += u_out[j * s + d] * coeff[r * rank + j];
        u_out[r * s + d] = val;
        v_out[r * s + d] = k[r * s + d];
    }
    if (d == 0) {
        float prod = 1.0f;
        for (uint r = 0; r < rank; ++r) prod *= g[r];
        gamma[0] = prod;
    }
}

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
    for (uint d = 0; d < s; ++d) r[d] = beta[i] * k[i * s + d];
    for (uint t = i + 1; t < rank; ++t) {
        float dot = 0.0f;
        for (uint d = 0; d < s; ++d) dot += r[d] * k[t * s + d];
        const float scale = -beta[t] * dot;
        const float gt = g[t];
        for (uint d = 0; d < s; ++d) r[d] = gt * (r[d] + scale * k[t * s + d]);
    }
    for (uint d = 0; d < s; ++d) rights[i * s + d] = r[d];
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

    if (tid < s) r[tid] = beta[i] * k[i * s + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = i + 1; t < rank; ++t) {
        partial[tid] = (tid < s) ? (r[tid] * k[t * s + tid]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = MAX_S >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float scale = -beta[t] * partial[0];
        const float gt = g[t];
        if (tid < s) r[tid] = gt * (r[tid] + scale * k[t * s + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < s) rights[i * s + tid] = r[tid];
}

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
    for (uint d = 0; d < s; ++d) acc += state[row * s + d] * u_cols[r * s + d];
    coeffs[row * rank + r] = acc;
}

kernel void compact_summary_apply(
    device const float* state    [[buffer(0)]],
    device const float* coeffs   [[buffer(1)]],
    device const float* v_cols   [[buffer(2)]],
    device const float* b_lefts  [[buffer(3)]],
    device const float* b_rights [[buffer(4)]],
    device       float* out      [[buffer(5)]],
    device const float* gamma_p  [[buffer(6)]],
    constant     uint&  s        [[buffer(7)]],
    constant     uint&  rank     [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint col = gid.y;
    if (row >= s || col >= s) return;
    float acc = state[row * s + col];
    for (uint r = 0; r < rank; ++r) acc += coeffs[row * rank + r] * v_cols[r * s + col];
    acc *= gamma_p[0];
    for (uint r = 0; r < rank; ++r) acc += b_lefts[r * s + row] * b_rights[r * s + col];
    out[row * s + col] = acc;
}

kernel void compact_summary_apply_row_tg(
    device const float* state    [[buffer(0)]],
    device const float* u_cols   [[buffer(1)]],
    device const float* v_cols   [[buffer(2)]],
    device const float* b_lefts  [[buffer(3)]],
    device const float* b_rights [[buffer(4)]],
    device       float* out      [[buffer(5)]],
    device const float* gamma_p  [[buffer(6)]],
    constant     uint&  s        [[buffer(7)]],
    constant     uint&  rank     [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    const uint row = tg.x;
    const uint col = tid;
    if (row >= s || s > MAX_S || rank > MAX_RANK) return;

    threadgroup float partial[MAX_S];
    threadgroup float coeff[MAX_RANK];

    for (uint r = 0; r < rank; ++r) {
        partial[tid] = (tid < s) ? (state[row * s + tid] * u_cols[r * s + tid]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = MAX_S >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) coeff[r] = partial[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (col < s) {
        float acc = state[row * s + col];
        for (uint r = 0; r < rank; ++r) acc += coeff[r] * v_cols[r * s + col];
        acc *= gamma_p[0];
        for (uint r = 0; r < rank; ++r) acc += b_lefts[r * s + row] * b_rights[r * s + col];
        out[row * s + col] = acc;
    }
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

private def max_abs_delta_flat(cpu : BlockScan::Matrix, gpu : Array(Float32)) : Float64
  max = 0.0
  s = cpu.size
  s.times do |i|
    s.times do |j|
      delta = (cpu[i][j] - gpu[i * s + j].to_f64).abs
      max = delta if delta > max
    end
  end
  max
end

s = 128
block = 16
runs = 10
warmup = 2
seed = 0xF011_u64

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_compact_vs_rowwise_micro [--s N] [--block N] [--runs N] [--warmup N]"
  p.on("--s=N", "State size (default/max: 128)") { |v| s = v.to_i }
  p.on("--block=N", "Block/rank/tokens (default: 16)") { |v| block = v.to_i }
  p.on("--runs=N", "Timed runs (default: 10)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs (default: 2)") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "s must be positive" unless s > 0
raise "s must be <= 128" if s > 128
raise "block must be positive" unless block > 0
raise "runs must be positive" unless runs > 0
abort "Metal unavailable" unless ML::Metal::Device.init!

rng = Random.new(seed)
initial = Array.new(s) { Array.new(s) { ((rng.next_float - 0.5) * 0.2) } }
inputs = Array.new(block) do
  DeltaInputs.new(
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    Array.new(s) { ((rng.next_float - 0.5) * 0.5) },
    0.88 + 0.10 * rng.next_float,
    rng.next_float
  )
end
cpu_final = BlockScan.replay_final_state(initial, inputs, 1.0)

k = inputs.map(&.k)
v = inputs.map(&.v)
q = inputs.map(&.q)
g = inputs.map(&.g)
beta = inputs.map(&.beta)

state_buf = ML::MetalBuffer.from_array(flatten_f32(initial))
k_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(k))
v_in_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(v))
q_in_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(q))
g_buf = ML::MetalBuffer.from_array(g.map(&.to_f32))
beta_buf = ML::MetalBuffer.from_array(beta.map(&.to_f32))
coeff_a_buf = ML::MetalBuffer.new((block * block).to_i64 * sizeof(Float32))
u_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
v_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
br_buf = ML::MetalBuffer.new((block * s).to_i64 * sizeof(Float32))
coeff_apply_buf = ML::MetalBuffer.new((s * block).to_i64 * sizeof(Float32))
gamma_buf = ML::MetalBuffer.new(sizeof(Float32).to_i64)
out_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))

pa_dot = ML::Metal::ComputePipeline.new("compact_a_dot_coeffs", SOURCE)
pa_build = ML::Metal::ComputePipeline.new("compact_a_build_factors", SOURCE)
pb_build = ML::Metal::ComputePipeline.new("compact_b_build_rights", SOURCE)
pb_build_tg = ML::Metal::ComputePipeline.new("compact_b_build_rights_tg", SOURCE)
pcoeff = ML::Metal::ComputePipeline.new("compact_transition_coeffs", SOURCE)
papply = ML::Metal::ComputePipeline.new("compact_summary_apply", SOURCE)
papply_row_tg = ML::Metal::ComputePipeline.new("compact_summary_apply_row_tg", SOURCE)
s_u = s.to_u32
rank_u = block.to_u32

use_tg_b = ENV["QWEN35_COMPACT_B_SCALAR"]? != "1"
use_tg_apply = ENV["QWEN35_COMPACT_APPLY_ROW_TG"]? == "1"

run_compact = -> do
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pa_dot)
    enc.set_buffer(k_buf, 0); enc.set_buffer(beta_buf, 1); enc.set_buffer(coeff_a_buf, 2)
    enc.set_value(s_u, 3); enc.set_value(rank_u, 4)
    enc.dispatch({block, block, 1}, {16, 16, 1})
    enc.end_encoding

    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pa_build)
    enc.set_buffer(k_buf, 0); enc.set_buffer(beta_buf, 1); enc.set_buffer(g_buf, 2); enc.set_buffer(coeff_a_buf, 3)
    enc.set_buffer(u_buf, 4); enc.set_buffer(v_buf, 5); enc.set_buffer(gamma_buf, 6)
    enc.set_value(s_u, 7); enc.set_value(rank_u, 8)
    enc.dispatch({s, 1, 1}, {256, 1, 1})
    enc.end_encoding

    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(use_tg_b ? pb_build_tg : pb_build)
    enc.set_buffer(k_buf, 0); enc.set_buffer(beta_buf, 1); enc.set_buffer(g_buf, 2); enc.set_buffer(br_buf, 3)
    enc.set_value(s_u, 4); enc.set_value(rank_u, 5)
    if use_tg_b
      enc.dispatch_threadgroups({block, 1, 1}, {128, 1, 1})
    else
      enc.dispatch({block, 1, 1}, {Math.min(block, 64), 1, 1})
    end
    enc.end_encoding

    if use_tg_apply
      enc = ML::Metal::ComputeEncoder.new(cmd)
      enc.set_pipeline(papply_row_tg)
      enc.set_buffer(state_buf, 0); enc.set_buffer(u_buf, 1); enc.set_buffer(v_buf, 2)
      enc.set_buffer(v_in_buf, 3); enc.set_buffer(br_buf, 4); enc.set_buffer(out_buf, 5); enc.set_buffer(gamma_buf, 6)
      enc.set_value(s_u, 7); enc.set_value(rank_u, 8)
      enc.dispatch_threadgroups({s, 1, 1}, {128, 1, 1})
      enc.end_encoding
    else
      enc = ML::Metal::ComputeEncoder.new(cmd)
      enc.set_pipeline(pcoeff)
      enc.set_buffer(state_buf, 0); enc.set_buffer(u_buf, 1); enc.set_buffer(coeff_apply_buf, 2)
      enc.set_value(s_u, 3); enc.set_value(rank_u, 4)
      enc.dispatch({s, block, 1}, {16, 16, 1})
      enc.end_encoding

      enc = ML::Metal::ComputeEncoder.new(cmd)
      enc.set_pipeline(papply)
      enc.set_buffer(state_buf, 0); enc.set_buffer(coeff_apply_buf, 1); enc.set_buffer(v_buf, 2)
      enc.set_buffer(v_in_buf, 3); enc.set_buffer(br_buf, 4); enc.set_buffer(out_buf, 5); enc.set_buffer(gamma_buf, 6)
      enc.set_value(s_u, 7); enc.set_value(rank_u, 8)
      enc.dispatch({s, s, 1}, {16, 16, 1})
      enc.end_encoding
    end
  end
end

# Current production-style rowwise chunk path computes all outputs too; for this
# lower-bound comparison we only read final state.
rowwise_state_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))
q_flat = flatten_vecs_f32(q)
k_flat = flatten_vecs_f32(k)
v_flat = flatten_vecs_f32(v)
g_flat = g.map(&.to_f32)
beta_flat = beta.map(&.to_f32)
run_rowwise = -> do
  rowwise_state_buf.write(flatten_f32(initial))
  ML::GGUF::Qwen35Metal.delta_net_chunk(
    rowwise_state_buf,
    q_flat,
    k_flat,
    v_flat,
    g_flat,
    beta_flat,
    1, 1, s, block, 1.0_f32
  )
end

warmup.times { run_compact.call; run_rowwise.call }
compact_out = out_buf.read(s * s)
rowwise_state = rowwise_state_buf.read(s * s)
compact_delta = max_abs_delta_flat(cpu_final, compact_out)
rowwise_delta = max_abs_delta_flat(cpu_final, rowwise_state)

compact_ms = [] of Float64
rowwise_ms = [] of Float64
runs.times do
  ms, _ = elapsed_ms { run_compact.call }
  compact_ms << ms
  ms2, _ = elapsed_ms { run_rowwise.call }
  rowwise_ms << ms2
end
compact_p50 = compact_ms.sort[compact_ms.size // 2]
rowwise_p50 = rowwise_ms.sort[rowwise_ms.size // 2]

puts "Qwen35 DeltaNet compact full lower-bound vs rowwise microbench"
puts "s=#{s} block=#{block} runs=#{runs} warmup=#{warmup}"
puts "compact_b_kernel=#{use_tg_b ? "threadgroup" : "scalar"}"
puts "compact_apply_kernel=#{use_tg_apply ? "row_threadgroup" : "two_dispatch_2d"}"
puts "compact_delta_vs_cpu=#{compact_delta}"
puts "rowwise_delta_vs_cpu=#{rowwise_delta}"
puts "compact_p50_ms=#{compact_p50.round(4)}"
puts "rowwise_p50_ms=#{rowwise_p50.round(4)}"
puts "compact_vs_rowwise=#{(rowwise_p50 / compact_p50).round(4)}"
puts "note=compact path computes final state only; rowwise path computes per-token outputs too. This is a lower-bound gate, not an integration result."
