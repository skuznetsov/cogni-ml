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

kernel void adjoint_terms_reverse_tg(
    device const float* q     [[buffer(0)]],
    device const float* k     [[buffer(1)]],
    device const float* v     [[buffer(2)]],
    device const float* g     [[buffer(3)]],
    device const float* beta  [[buffer(4)]],
    device       float* tq    [[buffer(5)]],
    device       float* add   [[buffer(6)]],
    constant     uint&  s     [[buffer(7)]],
    constant     uint&  ntok  [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (tg >= ntok || s > MAX_S) return;

    threadgroup float x[MAX_S];
    threadgroup float addv[MAX_S];
    threadgroup float partial[MAX_S];

    if (lid < s) {
        x[lid] = q[tg * s + lid];
        addv[lid] = 0.0f;
    }
    partial[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int step = int(tg); step >= 0; --step) {
        const uint off = uint(step) * s;
        partial[lid] = (lid < s) ? k[off + lid] * x[lid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = MAX_S >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                partial[lid] += partial[lid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float dot = partial[0];
        const float coeff = beta[step] * dot;
        const float gate = g[step];
        if (lid < s) {
            addv[lid] += v[off + lid] * coeff;
            x[lid] = gate * (x[lid] - k[off + lid] * coeff);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < s) {
        tq[tg * s + lid] = x[lid];
        add[tg * s + lid] = addv[lid];
    }
}

kernel void adjoint_terms_output_fused_tg(
    device const float* q     [[buffer(0)]],
    device const float* k     [[buffer(1)]],
    device const float* v     [[buffer(2)]],
    device const float* g     [[buffer(3)]],
    device const float* beta  [[buffer(4)]],
    device const float* state [[buffer(5)]],
    device       float* out   [[buffer(6)]],
    constant     uint&  s     [[buffer(7)]],
    constant     uint&  ntok  [[buffer(8)]],
    constant     float& scale [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (tg >= ntok || s > MAX_S) return;

    threadgroup float x[MAX_S];
    threadgroup float addv[MAX_S];
    threadgroup float partial[MAX_S];

    if (lid < s) {
        x[lid] = q[tg * s + lid];
        addv[lid] = 0.0f;
    }
    partial[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int step = int(tg); step >= 0; --step) {
        const uint off = uint(step) * s;
        partial[lid] = (lid < s) ? k[off + lid] * x[lid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = MAX_S >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                partial[lid] += partial[lid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float dot = partial[0];
        const float coeff = beta[step] * dot;
        const float gate = g[step];
        if (lid < s) {
            addv[lid] += v[off + lid] * coeff;
            x[lid] = gate * (x[lid] - k[off + lid] * coeff);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < s) {
        float acc = 0.0f;
        for (uint d = 0; d < s; ++d) {
            acc += state[lid * s + d] * x[d];
        }
        out[tg * s + lid] = (acc + addv[lid]) * scale;
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

private def max_flat_delta(cpu : Array(Array(Float64)), gpu : Array(Float32)) : Float64
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
initial_flat = flatten_f32(initial)

state_buf = ML::MetalBuffer.from_array(initial_flat)
tq_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(tq))
add_buf = ML::MetalBuffer.from_array(flatten_vecs_f32(add))
tq_metal_buf = ML::MetalBuffer.new((tokens * s).to_i64 * sizeof(Float32))
add_metal_buf = ML::MetalBuffer.new((tokens * s).to_i64 * sizeof(Float32))
out_buf = ML::MetalBuffer.new((tokens * s).to_i64 * sizeof(Float32))
out_full_buf = ML::MetalBuffer.new((tokens * s).to_i64 * sizeof(Float32))
out_fused_buf = ML::MetalBuffer.new((tokens * s).to_i64 * sizeof(Float32))
pipe = ML::Metal::ComputePipeline.new("adjoint_replay_outputs", SOURCE)
terms_pipe = ML::Metal::ComputePipeline.new("adjoint_terms_reverse_tg", SOURCE)
fused_pipe = ML::Metal::ComputePipeline.new("adjoint_terms_output_fused_tg", SOURCE)
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

q_flat = flatten_vecs_f32(inputs.map(&.q))
k_flat = flatten_vecs_f32(inputs.map(&.k))
v_flat = flatten_vecs_f32(inputs.map(&.v))
g_flat = inputs.map(&.g.to_f32)
beta_flat = inputs.map(&.beta.to_f32)
q_buf = ML::MetalBuffer.from_array(q_flat)
k_buf = ML::MetalBuffer.from_array(k_flat)
v_buf = ML::MetalBuffer.from_array(v_flat)
g_buf = ML::MetalBuffer.from_array(g_flat)
beta_buf = ML::MetalBuffer.from_array(beta_flat)

run_terms = -> do
  ML::Metal::Dispatch.execute(terms_pipe) do |enc|
    enc.set_buffer(q_buf, 0)
    enc.set_buffer(k_buf, 1)
    enc.set_buffer(v_buf, 2)
    enc.set_buffer(g_buf, 3)
    enc.set_buffer(beta_buf, 4)
    enc.set_buffer(tq_metal_buf, 5)
    enc.set_buffer(add_metal_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(tokens_u, 8)
    enc.dispatch_threadgroups({tokens, 1, 1}, {128, 1, 1})
  end
end

run_adjoint_full = -> do
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(terms_pipe)
    enc.set_buffer(q_buf, 0)
    enc.set_buffer(k_buf, 1)
    enc.set_buffer(v_buf, 2)
    enc.set_buffer(g_buf, 3)
    enc.set_buffer(beta_buf, 4)
    enc.set_buffer(tq_metal_buf, 5)
    enc.set_buffer(add_metal_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(tokens_u, 8)
    enc.dispatch_threadgroups({tokens, 1, 1}, {128, 1, 1})
    enc.end_encoding

    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pipe)
    enc.set_buffer(state_buf, 0)
    enc.set_buffer(tq_metal_buf, 1)
    enc.set_buffer(add_metal_buf, 2)
    enc.set_buffer(out_full_buf, 3)
    enc.set_value(s_u, 4)
    enc.set_value(tokens_u, 5)
    enc.set_value(scale, 6)
    enc.dispatch({s, tokens, 1}, {16, 16, 1})
    enc.end_encoding
  end
end

run_adjoint_fused = -> do
  ML::Metal::Dispatch.execute(fused_pipe) do |enc|
    enc.set_buffer(q_buf, 0)
    enc.set_buffer(k_buf, 1)
    enc.set_buffer(v_buf, 2)
    enc.set_buffer(g_buf, 3)
    enc.set_buffer(beta_buf, 4)
    enc.set_buffer(state_buf, 5)
    enc.set_buffer(out_fused_buf, 6)
    enc.set_value(s_u, 7)
    enc.set_value(tokens_u, 8)
    enc.set_value(scale, 9)
    enc.dispatch_threadgroups({tokens, 1, 1}, {128, 1, 1})
  end
end

rowwise_state_buf = ML::MetalBuffer.new((s * s).to_i64 * sizeof(Float32))
run_rowwise = -> do
  rowwise_state_buf.write(initial_flat)
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

warmup.times { run_terms.call; run_adjoint.call; run_adjoint_full.call; run_adjoint_fused.call; run_rowwise.call }
gpu_y = out_buf.read(tokens * s)
gpu_full_y = out_full_buf.read(tokens * s)
gpu_fused_y = out_fused_buf.read(tokens * s)
gpu_tq = tq_metal_buf.read(tokens * s)
gpu_add = add_metal_buf.read(tokens * s)
delta = max_output_delta(serial_y, gpu_y)
full_delta = max_output_delta(serial_y, gpu_full_y)
fused_delta = max_output_delta(serial_y, gpu_fused_y)
tq_delta = max_flat_delta(tq, gpu_tq)
add_delta = max_flat_delta(add, gpu_add)

terms_ms = [] of Float64
adjoint_ms = [] of Float64
full_ms = [] of Float64
fused_ms = [] of Float64
rowwise_ms = [] of Float64
runs.times do
  ms0, _ = elapsed_ms { run_terms.call }
  terms_ms << ms0
  ms, _ = elapsed_ms { run_adjoint.call }
  adjoint_ms << ms
  ms1, _ = elapsed_ms { run_adjoint_full.call }
  full_ms << ms1
  ms3, _ = elapsed_ms { run_adjoint_fused.call }
  fused_ms << ms3
  ms2, _ = elapsed_ms { run_rowwise.call }
  rowwise_ms << ms2
end
terms_p50 = terms_ms.sort[terms_ms.size // 2]
adj_p50 = adjoint_ms.sort[adjoint_ms.size // 2]
full_p50 = full_ms.sort[full_ms.size // 2]
fused_p50 = fused_ms.sort[fused_ms.size // 2]
row_p50 = rowwise_ms.sort[rowwise_ms.size // 2]

puts "Qwen35 DeltaNet adjoint replay output microbench"
puts "s=#{s} tokens=#{tokens} runs=#{runs} warmup=#{warmup}"
puts "max_output_delta=#{delta}"
puts "max_full_output_delta=#{full_delta}"
puts "max_fused_output_delta=#{fused_delta}"
puts "max_tq_delta=#{tq_delta}"
puts "max_add_delta=#{add_delta}"
puts "terms_reverse_tg_p50_ms=#{terms_p50.round(4)}"
puts "adjoint_output_p50_ms=#{adj_p50.round(4)}"
puts "adjoint_full_p50_ms=#{full_p50.round(4)}"
puts "adjoint_fused_p50_ms=#{fused_p50.round(4)}"
puts "rowwise_chunk_p50_ms=#{row_p50.round(4)}"
puts "adjoint_vs_rowwise=#{(row_p50 / adj_p50).round(4)}"
puts "adjoint_full_vs_rowwise=#{(row_p50 / full_p50).round(4)}"
puts "adjoint_fused_vs_rowwise=#{(row_p50 / fused_p50).round(4)}"
puts "note=fused path constructs adjoint terms in threadgroup memory and immediately emits outputs; all adjoint paths still exclude final-state prefix scan."
