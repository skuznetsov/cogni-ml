#!/usr/bin/env crystal
# Isolated staging falsifier. Never changes the production shader or dispatch.
require "option_parser"
require "json"
require "digest/sha256"
require "../src/ml/gguf/qwen35_metal"

IN_DIM  =  5120
OUT_DIM = 17408
NAMES   = {"simd_mm_q4k_h16_b128_sg8", "simd_mm_q4k_h16_b128_sg8_swiglu_h16"}

private def candidate_source(base : String) : String
  result = base
  NAMES.each_with_index do |name, index|
    start = base.byte_index("kernel void #{name}(") || raise "missing kernel #{name}"
    stop_name = index == 0 ? NAMES[1] : "simd_mm_q4k_h16_b64_swiglu"
    stop = base.byte_index("kernel void #{stop_name}(", start + 1) || raise "missing kernel boundary"
    kernel = base.byte_slice(start, stop - start)
    raise "invalid byte boundary" unless kernel.starts_with?("kernel void #{name}(")
    replacements = {
      "kernel void #{name}("                            => "kernel void #{name}_single(",
      "shmem + MM128_SG8_TILE_SIZE + MM128_SG8_SA_SIZE" => "shmem + MM128_SG8_SA_SIZE",
      "shmem + MM128_SG8_TILE_SIZE"                     => "shmem",
      "if (iter + 1 < n_iter) {"                        => "if (iter + 1 < n_iter) {\n            threadgroup_barrier(mem_flags::mem_threadgroup); // All readers finish before alias overwrite.",
    }
    replacements.each do |before, after|
      raise "source drift: #{name}: #{before}" unless kernel.scan(before).size == 1
      kernel = kernel.sub(before, after)
    end
    result += "\n" + kernel
  end
  result
end

private def exact_f32!(a : Slice(Float32), b : Slice(Float32)) : Nil
  raise "F32 size mismatch" unless a.size == b.size
  a.each_with_index do |value, i|
    raise "F32 mismatch/nonfinite at #{i}" unless value.finite? && b[i].finite? && value.unsafe_as(UInt32) == b[i].unsafe_as(UInt32)
  end
  raise "empty F32 output" unless a.any? { |v| v != 0 }
end

private def exact_h16!(a : Slice(UInt16), b : Slice(UInt16)) : Nil
  raise "H16 size mismatch" unless a.size == b.size
  a.each_with_index do |value, i|
    raise "H16 mismatch/nonfinite at #{i}" unless value == b[i] && (value & 0x7c00_u16) != 0x7c00_u16
  end
  raise "empty H16 output" unless a.any? { |v| (v & 0x7fff_u16) != 0 }
end

private def rejects!(&block : ->) : Nil
  rejected = false
  begin
    yield
  rescue
    rejected = true
  end
  raise "negative control was accepted" unless rejected
end

private def input_values(batch : Int32) : Array(UInt16)
  # Use high LCG bits: low bits repeat every 1024 elements and would make all
  # 5120-wide rows identical, weakening the cross-row correctness falsifier.
  Array(UInt16).new(batch * IN_DIM) do |i|
    hash = (i.to_i64 * 1103515245 + 12345) & 0xffffffff_i64
    (((hash >> 16) & 0x83ff) | 0x2800).to_u16
  end
end

private def self_test : Nil
  base = ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE
  source = candidate_source(base)
  NAMES.each { |name| raise "missing candidate" unless source.scan("kernel void #{name}_single(").size == 1 }
  rejects! { candidate_source(base.gsub("if (iter + 1 < n_iter) {", "if (false) {")) }
  f = Slice[1.0_f32, -0.5_f32]
  h = Slice[0x3c00_u16, 0xb800_u16]
  exact_f32!(f, f)
  exact_h16!(h, h)
  rejects! { exact_f32!(f, Slice[1.0_f32, -0.25_f32]) }
  rejects! { exact_h16!(h, Slice[0x3c01_u16, 0xb800_u16]) }
  rejects! { exact_f32!(Slice[Float32::NAN], Slice[Float32::NAN]) }
  rejects! { exact_h16!(Slice[0x7e00_u16], Slice[0x7e00_u16]) }
  rejects! { exact_f32!(Slice[0.0_f32], Slice[0.0_f32]) }
  rejects! { exact_h16!(Slice[0_u16], Slice[0_u16]) }
  input = input_values(2)
  raise "fixture rows repeat" if input[0, IN_DIM] == input[IN_DIM, IN_DIM]
  puts "self_test=true source_mutation_rejected=true corruption_nan_empty_rejected=true"
end

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2
end

private def run_pair(pipelines : Array(ML::Metal::ComputePipeline), weights : Array(ML::MetalBuffer),
                     input : ML::MetalBuffer, gate : ML::MetalBuffer, output : ML::MetalBuffer,
                     batch : Int32, single : Bool) : Float64
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  enc.set_pipeline(pipelines[0])
  enc.set_buffer(weights[0], 0)
  enc.set_buffer(input, 1)
  enc.set_buffer(gate, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(IN_DIM.to_u32, 3)
  enc.set_value(OUT_DIM.to_u32, 4)
  enc.set_value(batch.to_u32, 5)
  enc.set_threadgroup_memory(single ? 12288 : 24576, 0)
  enc.dispatch_threadgroups({batch // 128, OUT_DIM // 64, 1}, {256, 1, 1})
  enc.set_pipeline(pipelines[1])
  enc.set_buffer(weights[1], 0)
  enc.set_buffer(input, 1)
  enc.set_buffer(gate, 2)
  enc.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
  enc.set_value(IN_DIM.to_u32, 4)
  enc.set_value(OUT_DIM.to_u32, 5)
  enc.set_value(batch.to_u32, 6)
  enc.set_threadgroup_memory(single ? 16384 : 24576, 0)
  enc.dispatch_threadgroups({batch // 128, OUT_DIM // 64, 1}, {256, 1, 1})
  enc.end_encoding
  ms = cmd.commit_and_wait_gpu_elapsed_seconds * 1000
  raise "invalid GPU interval" unless ms.finite? && ms > 0
  ms
end

mode = ""
batch = 512
model = "/Users/sergey/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
OptionParser.parse do |p|
  {"self-test", "compile-only", "run"}.each do |option|
    p.on("--#{option}", option) { raise "choose one mode" unless mode.empty?; mode = option }
  end
  p.on("--batch=N", "256, 512, 1024 or 2048") { |v| batch = v.to_i32 }
  p.on("--model=PATH", "Qwen3.8 GGUF") { |v| model = v }
  p.on("--help", "Help") { puts p; exit }
end
raise "choose --self-test, --compile-only or --run" if mode.empty?
raise "unsupported batch" unless {256, 512, 1024, 2048}.includes?(batch)
self_test
exit if mode == "self-test"
# Advisory acknowledgement only: the actual process/memory guards are external.
raise "set COGNI_RUN_SAFE_ACTIVE=1 and run through scripts/run_safe.sh" if mode == "run" && ENV["COGNI_RUN_SAFE_ACTIVE"]? != "1"

base = ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE
source = candidate_source(base)
baseline = NAMES.map { |name| ML::Metal::ComputePipeline.new(name, source) }.to_a
candidate = NAMES.map { |name| ML::Metal::ComputePipeline.new("#{name}_single", source) }.to_a
2.times do |i|
  a, b = baseline[i].max_total_threads_per_threadgroup, candidate[i].max_total_threads_per_threadgroup
  puts({event: "pipeline", name: NAMES[i], baseline_max_threads: a, candidate_max_threads: b}.to_json)
  raise "pipeline launch-resource regression" unless b >= a && a >= 256
end
puts({event: "source", device: ML::Metal::Device.instance.name, baseline_sha256: Digest::SHA256.hexdigest(base), candidate_sha256: Digest::SHA256.hexdigest(source), model_loaded: false}.to_json)
exit if mode == "compile-only"

raw = [] of Bytes
gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
begin
  tensors = {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"}.map do |name|
    info = gguf.tensor(name) || raise "missing #{name}"
    raise "unsupported tensor shape/type" unless info.type.q4_k? && info.dims == [IN_DIM.to_i64, OUT_DIM.to_i64]
    {name, info}
  end
  tensors.each do |name, info|
    bytes = gguf.read_tensor_raw(info)
    raise "invalid Q4 byte count" unless bytes.size == OUT_DIM * (IN_DIM // 256) * 144
    puts({event: "tensor", name: name, bytes: bytes.size, sha256: Digest::SHA256.hexdigest(bytes), mmap_tensors: false}.to_json)
    raw << bytes
  end
ensure
  gguf.close
end
weights = raw.map do |bytes|
  buffer = ML::MetalBuffer.new(bytes.size.to_i64)
  buffer.write_bytes(bytes.to_unsafe, bytes.size)
  buffer
end
# Direct finite half bits give deterministic signed magnitudes in [1/32, 1/16).
values = input_values(batch)
puts({event: "input", batch: batch, sha256: Digest::SHA256.hexdigest(Slice.new(values.to_unsafe.as(Pointer(UInt8)), values.size * 2)), first_two_rows_distinct: true}.to_json)
input = ML::MetalBuffer.new(values.size.to_i64 * 2)
input.write_bytes(values.to_unsafe.as(Pointer(UInt8)), values.size * 2)
count = batch * OUT_DIM
gates = Array.new(2) { ML::MetalBuffer.new(count.to_i64 * 4) }
outputs = Array.new(2) { ML::MetalBuffer.new(count.to_i64 * 2) }
arms = [baseline, candidate]
run = ->(i : Int32) { run_pair(arms[i], weights, input, gates[i], outputs[i], batch, i == 1) }
check = -> {
  exact_f32!(Slice.new(gates[0].contents.as(Pointer(Float32)), count), Slice.new(gates[1].contents.as(Pointer(Float32)), count))
  exact_h16!(Slice.new(outputs[0].contents.as(Pointer(UInt16)), count), Slice.new(outputs[1].contents.as(Pointer(UInt16)), count))
}
poison = -> {
  gates.each { |buf| Slice.new(buf.contents.as(Pointer(Float32)), count).fill(Float32::NAN) }
  outputs.each { |buf| Slice.new(buf.contents.as(Pointer(UInt16)), count).fill(0x7e00_u16) }
}
poison.call
run.call(0)
run.call(1)
check.call
5.times { {0, 1, 1, 0}.each { |i| run.call(i) } }
a_times, b_times = [] of Float64, [] of Float64
wins = 0
10.times do |cycle|
  a0, b0, b1, a1 = run.call(0), run.call(1), run.call(1), run.call(0)
  a_times.concat([a0, a1])
  b_times.concat([b0, b1])
  wins += 1 if b0 + b1 < a0 + a1
  puts({event: "cycle", batch: batch, cycle: cycle, baseline_first_ms: a0, candidate_first_ms: b0, candidate_second_ms: b1, baseline_second_ms: a1}.to_json)
end
check.call
poison.call
run.call(1)
run.call(0)
check.call
puts({event: "summary", batch: batch, warmups: 5, cycles: 10, elements: count, finite_bitwise_equal: true, baseline_pair_median_ms: median(a_times), candidate_pair_median_ms: median(b_times), time_reduction_pct: 100 * (1 - median(b_times) / median(a_times)), candidate_wins: wins}.to_json)
