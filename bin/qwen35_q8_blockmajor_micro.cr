#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/core/buffer"
require "../src/ml/gguf/reader"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;

constant uint Q8_0_QK = 32;
constant short MV8_NSG = 4;
constant short MV8_NR0 = 1;

struct block_q8_0 {
    half   d;
    int8_t qs[32];
};

kernel void q8_rowmajor(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint row_id = tgpig.x * MV8_NSG + sgitg;
    if (row_id >= out_dim) return;

    const uint row_bytes = nb * 34;
    float sumf = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q8_0 * blk =
            (device const block_q8_0 *)(w_raw + row_id * row_bytes) + ib;
        sumf += (float)blk->d * x[ib * Q8_0_QK + tiisg] * (float)blk->qs[tiisg];
    }
    const float total = simd_sum(sumf);
    if (tiisg == 0) output[row_id] = total;
}

kernel void q8_blockmajor(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint row_id = tgpig.x * MV8_NSG + sgitg;
    if (row_id >= out_dim) return;

    float sumf = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q8_0 * blk =
            (device const block_q8_0 *)(w_raw + (ib * out_dim + row_id) * 34);
        sumf += (float)blk->d * x[ib * Q8_0_QK + tiisg] * (float)blk->qs[tiisg];
    }
    const float total = simd_sum(sumf);
    if (tiisg == 0) output[row_id] = total;
}
METAL

private def quant_tensor_bytes(model_path : String, name : String) : {Bytes, Int32, Int32}
  g = ML::GGUF::GGUFFile.new(model_path)
  info = g.tensor(name) || raise "missing tensor #{name.inspect}"
  raise "expected Q8_0 tensor, got #{info.type}" unless info.type.q8_0?
  raw = g.read_tensor_raw(info).dup
  {raw, info.dims[0].to_i32, info.dims[1].to_i32}
ensure
  g.try(&.close)
end

private def pack_blockmajor(rowmajor : Bytes, in_dim : Int32, out_dim : Int32) : Bytes
  nb = in_dim // 32
  out = Bytes.new(rowmajor.size)
  out_dim.times do |row|
    nb.times do |ib|
      src = row * nb * 34 + ib * 34
      dst = (ib * out_dim + row) * 34
      out[dst, 34].copy_from(rowmajor[src, 34])
    end
  end
  out
end

private def write_bytes(buf : ML::MetalBuffer, bytes : Bytes) : Nil
  buf.write_bytes(bytes.to_unsafe, bytes.size)
end

private def run_kernel(pipe : ML::Metal::ComputePipeline,
                       w_buf : ML::MetalBuffer,
                       x_buf : ML::MetalBuffer,
                       out_buf : ML::MetalBuffer,
                       in_dim : Int32,
                       out_dim : Int32) : Nil
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pipe)
    enc.set_buffer(w_buf, 0)
    enc.set_buffer(x_buf, 1)
    enc.set_buffer(out_buf, 2)
    enc.set_value(in_dim.to_u32, 3)
    enc.set_value(out_dim.to_u32, 4)
    enc.dispatch_threadgroups({(out_dim + 3) // 4, 1, 1}, {128, 1, 1})
    enc.end_encoding
  end
end

private def max_abs_delta(a : Array(Float32), b : Array(Float32)) : Float64
  max = 0.0
  a.each_with_index do |av, i|
    d = (av - b[i]).to_f64.abs
    max = d if d > max
  end
  max
end

model = ENV["QWEN35_DRAFT"]? || DEFAULT_MODEL
tensor = "blk.0.ffn_up.weight"
runs = 200
warmup = 20

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_q8_blockmajor_micro [--model PATH] [--tensor NAME] [--runs N] [--warmup N]"
  p.on("--model=PATH", "Q8_0 GGUF model path") { |v| model = v }
  p.on("--tensor=NAME", "Q8_0 tensor name") { |v| tensor = v }
  p.on("--runs=N", "Timed runs") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "runs must be positive" unless runs > 0
raise "warmup must be non-negative" unless warmup >= 0

raw, in_dim, out_dim = quant_tensor_bytes(model, tensor)
packed = pack_blockmajor(raw, in_dim, out_dim)
rng = Random.new(42)
x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }

row_pipe = ML::Metal::ComputePipeline.new("q8_rowmajor", SOURCE)
blk_pipe = ML::Metal::ComputePipeline.new("q8_blockmajor", SOURCE)
row_w = ML::MetalBuffer.new(raw.size.to_i64)
blk_w = ML::MetalBuffer.new(packed.size.to_i64)
x_buf = ML::MetalBuffer.from_array(x)
row_out = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32))
blk_out = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32))
write_bytes(row_w, raw)
write_bytes(blk_w, packed)

run_kernel(row_pipe, row_w, x_buf, row_out, in_dim, out_dim)
run_kernel(blk_pipe, blk_w, x_buf, blk_out, in_dim, out_dim)
row_res = row_out.read(out_dim)
blk_res = blk_out.read(out_dim)
delta = max_abs_delta(row_res, blk_res)

row_times = [] of Float64
blk_times = [] of Float64
(warmup + runs).times do |i|
  t0 = Time.instant
  run_kernel(row_pipe, row_w, x_buf, row_out, in_dim, out_dim)
  dt = (Time.instant - t0).total_milliseconds
  row_times << dt if i >= warmup

  t1 = Time.instant
  run_kernel(blk_pipe, blk_w, x_buf, blk_out, in_dim, out_dim)
  dt2 = (Time.instant - t1).total_milliseconds
  blk_times << dt2 if i >= warmup
end

row_sorted = row_times.sort
blk_sorted = blk_times.sort
row_p50 = row_sorted[row_sorted.size // 2]
blk_p50 = blk_sorted[blk_sorted.size // 2]
row_p10 = row_sorted[row_sorted.size // 10]
blk_p10 = blk_sorted[blk_sorted.size // 10]
row_p90 = row_sorted[(row_sorted.size * 9 // 10).clamp(0, row_sorted.size - 1)]
blk_p90 = blk_sorted[(blk_sorted.size * 9 // 10).clamp(0, blk_sorted.size - 1)]

puts "Q8 block-major GEMV lower-bound"
puts "model=#{File.basename(model)} tensor=#{tensor} shape=#{in_dim}x#{out_dim} runs=#{runs} warmup=#{warmup}"
puts "pack_bytes=#{packed.size} raw_bytes=#{raw.size} max_delta=#{delta}"
puts "rowmajor_ms p10=#{row_p10.round(6)} p50=#{row_p50.round(6)} p90=#{row_p90.round(6)}"
puts "blockmajor_ms p10=#{blk_p10.round(6)} p50=#{blk_p50.round(6)} p90=#{blk_p90.round(6)} speedup=#{(row_p50 / blk_p50).round(4)}x"
