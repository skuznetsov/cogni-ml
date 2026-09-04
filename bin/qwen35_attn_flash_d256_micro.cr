#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/core/buffer"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"

BASELINE_SOURCE = "#define QWEN35_KV_CACHE_F16 1\n" + {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/fullattn_qwen35.metal") }}
FLASH_SOURCE    = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/qwen35_attn_flash_d256.metal") }}

private def f32_to_f16_bits(value : Float32) : UInt16
  bits = value.unsafe_as(UInt32)
  sign = ((bits >> 16) & 0x8000).to_u16
  exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
  mantissa = bits & 0x7fffff

  return sign if exponent < -10
  if exponent <= 0
    mantissa = (mantissa | 0x800000) >> (1 - exponent)
    return (sign | ((mantissa + 0x1000) >> 13).to_u16)
  end
  return (sign | 0x7c00_u16 | (mantissa == 0 ? 0_u16 : 0x0200_u16)) if exponent >= 31

  sign | (exponent.to_u16 << 10) | ((mantissa + 0x1000) >> 13).to_u16
end

private def buffer_from_u16(values : Array(UInt16)) : ML::MetalBuffer
  buffer = ML::MetalBuffer.new(values.size.to_i64 * sizeof(UInt16))
  buffer.write_bytes(values.to_unsafe.as(Pointer(UInt8)), values.size * sizeof(UInt16))
  buffer
end

private def run_baseline(pipe : ML::Metal::ComputePipeline,
                         q : ML::MetalBuffer,
                         gate : ML::MetalBuffer,
                         k : ML::MetalBuffer,
                         v : ML::MetalBuffer,
                         output : ML::MetalBuffer,
                         n_tokens : Int32,
                         n_head : Int32,
                         n_head_kv : Int32) : Nil
  head_dim = 256
  heads_per_group = n_head // n_head_kv
  scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pipe)
    enc.set_buffer(q, 0)
    enc.set_buffer(gate, 1)
    enc.set_buffer(k, 2)
    enc.set_buffer(v, 3)
    enc.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
    enc.set_value(0_u32, 5)
    enc.set_value(n_tokens.to_u32, 6)
    enc.set_value(n_head.to_u32, 7)
    enc.set_value(n_head_kv.to_u32, 8)
    enc.set_value(head_dim.to_u32, 9)
    enc.set_value(heads_per_group.to_u32, 10)
    enc.set_value(scale, 11)
    enc.dispatch_threadgroups({n_head, (n_tokens + 3) // 4, 1}, {128, 1, 1})
    enc.end_encoding
  end
end

private def run_flash(pipe : ML::Metal::ComputePipeline,
                      q : ML::MetalBuffer,
                      gate : ML::MetalBuffer,
                      k : ML::MetalBuffer,
                      v : ML::MetalBuffer,
                      output : ML::MetalBuffer,
                      n_tokens : Int32,
                      n_head : Int32,
                      n_head_kv : Int32) : Nil
  head_dim = 256
  heads_per_group = n_head // n_head_kv
  scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pipe)
    enc.set_buffer(q, 0)
    enc.set_buffer(gate, 1)
    enc.set_buffer(k, 2)
    enc.set_buffer(v, 3)
    enc.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
    enc.set_value(0_u32, 5)
    enc.set_value(n_tokens.to_u32, 6)
    enc.set_value(n_head.to_u32, 7)
    enc.set_value(n_head_kv.to_u32, 8)
    enc.set_value(head_dim.to_u32, 9)
    enc.set_value(heads_per_group.to_u32, 10)
    enc.set_value(scale, 11)
    enc.set_threadgroup_memory(16 * 1024, 0)
    enc.dispatch_threadgroups({(n_tokens + 7) // 8, n_head, 1}, {32, 4, 1})
    enc.end_encoding
  end
end

private def percentile(values : Array(Float64), fraction : Float64) : Float64
  sorted = values.sort
  sorted[((sorted.size - 1) * fraction).round.to_i]
end

private def compare(reference : Array(Float32), candidate : Array(Float32))
  raise "output size mismatch" unless reference.size == candidate.size
  dot = 0.0
  ref_norm = 0.0
  candidate_norm = 0.0
  squared = 0.0
  max_abs = 0.0
  finite = true
  reference.each_with_index do |expected, i|
    actual = candidate[i]
    finite &&= expected.finite? && actual.finite?
    delta = (actual - expected).to_f64
    max_abs = delta.abs if delta.abs > max_abs
    squared += delta * delta
    dot += expected.to_f64 * actual
    ref_norm += expected.to_f64 * expected
    candidate_norm += actual.to_f64 * actual
  end
  cosine = dot / Math.sqrt(ref_norm * candidate_norm)
  {finite: finite, cosine: cosine, rmse: Math.sqrt(squared / reference.size), max_abs: max_abs}
end

tokens = [1024, 2048]
n_head = 16
n_head_kv = 4
warmup = 4
reps = 12

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_attn_flash_d256_micro [--tokens 1024,2048] [--heads 16] [--kv-heads 4] [--warmup 4] [--reps 12]"
  parser.on("--tokens=LIST", "Comma-separated multiples of 64") { |value| tokens = value.split(',').map(&.to_i) }
  parser.on("--heads=N", "Query heads") { |value| n_head = value.to_i }
  parser.on("--kv-heads=N", "KV heads") { |value| n_head_kv = value.to_i }
  parser.on("--warmup=N", "Warmup pairs") { |value| warmup = value.to_i }
  parser.on("--reps=N", "Timed samples per path") { |value| reps = value.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "Metal not available" unless ML::Metal::Device.available?
raise "heads must be divisible by kv-heads" unless n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
raise "warmup must be non-negative" unless warmup >= 0
raise "reps must be at least 4" unless reps >= 4
tokens.each { |count| raise "token count must be a positive multiple of 64" unless count > 0 && count % 64 == 0 }

baseline_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", BASELINE_SOURCE)
flash_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_flash_d256", FLASH_SOURCE)

tokens.each do |n_tokens|
  rng = Random.new(0x35_38 + n_tokens)
  q_values = Array(Float32).new(n_tokens * n_head * 256) { rng.rand(-0.25_f32..0.25_f32) }
  gate_values = Array(Float32).new(n_tokens * n_head * 256) { rng.rand(-1.0_f32..1.0_f32) }
  k_values = Array(UInt16).new(n_tokens * n_head_kv * 256) { f32_to_f16_bits(rng.rand(-0.25_f32..0.25_f32)) }
  v_values = Array(UInt16).new(n_tokens * n_head_kv * 256) { f32_to_f16_bits(rng.rand(-0.25_f32..0.25_f32)) }

  q_buf = ML::MetalBuffer.from_array(q_values)
  gate_buf = ML::MetalBuffer.from_array(gate_values)
  k_buf = buffer_from_u16(k_values)
  v_buf = buffer_from_u16(v_values)
  output_values = n_tokens.to_i64 * n_head * 256
  output_bytes = output_values * sizeof(Float32)
  baseline_out = ML::MetalBuffer.new(output_bytes)
  flash_out = ML::MetalBuffer.new(output_bytes)

  warmup.times do
    run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out, n_tokens, n_head, n_head_kv)
    run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out, n_tokens, n_head, n_head_kv)
  end

  run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out, n_tokens, n_head, n_head_kv)
  run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out, n_tokens, n_head, n_head_kv)
  quality = compare(baseline_out.read(output_values.to_i32), flash_out.read(output_values.to_i32))

  baseline_ms = [] of Float64
  flash_ms = [] of Float64
  while baseline_ms.size < reps || flash_ms.size < reps
    order = ((baseline_ms.size + flash_ms.size) // 4).even? ? [:baseline, :flash, :flash, :baseline] : [:flash, :baseline, :baseline, :flash]
    order.each do |variant|
      break if baseline_ms.size >= reps && flash_ms.size >= reps
      next if variant == :baseline && baseline_ms.size >= reps
      next if variant == :flash && flash_ms.size >= reps
      started = Time.instant
      if variant == :baseline
        run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out, n_tokens, n_head, n_head_kv)
        baseline_ms << (Time.instant - started).total_milliseconds
      else
        run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out, n_tokens, n_head, n_head_kv)
        flash_ms << (Time.instant - started).total_milliseconds
      end
    end
  end

  baseline_p50 = percentile(baseline_ms, 0.5)
  flash_p50 = percentile(flash_ms, 0.5)
  speedup = baseline_p50 / flash_p50
  quality_pass = quality[:finite] && quality[:cosine] >= 0.9999
  timing_pass = speedup >= 2.8

  puts "tokens=#{n_tokens} heads=#{n_head}/#{n_head_kv} reps=#{reps}"
  puts "quality finite=#{quality[:finite]} cosine=#{quality[:cosine].round(9)} rmse=#{quality[:rmse].round(9)} max_abs=#{quality[:max_abs].round(9)} pass=#{quality_pass}"
  puts "baseline_ms p10=#{percentile(baseline_ms, 0.1).round(6)} p50=#{baseline_p50.round(6)} p90=#{percentile(baseline_ms, 0.9).round(6)}"
  puts "flash_ms p10=#{percentile(flash_ms, 0.1).round(6)} p50=#{flash_p50.round(6)} p90=#{percentile(flash_ms, 0.9).round(6)} speedup=#{speedup.round(4)}x pass=#{timing_pass}"
  puts "admission=#{quality_pass && timing_pass ? "PASS" : "FAIL"}"
end
