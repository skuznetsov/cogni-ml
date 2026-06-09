#!/usr/bin/env crystal

require "../src/ml/gguf/reader"

def usage
  STDERR.puts "usage: crystal scripts/estimate_v3d_prepack_memory.cr MODEL.gguf"
  exit 2
end

path = ARGV[0]? || usage
g = ML::GGUF::GGUFFile.new(path)

total_raw = 0_u64
q4_raw = 0_u64
q4_pre = 0_u64
q8_raw = 0_u64
q8_pre = 0_u64
q4_tensors = 0
q8_tensors = 0
q4_params = 0_u64
q8_params = 0_u64

g.tensors.each do |t|
  total_raw += t.data_bytes.to_u64
  next unless t.dims.size == 2
  in_dim = t.dims[0].to_u64
  out_dim = t.dims[1].to_u64
  params = in_dim * out_dim
  if t.type.q4_k?
    q4_tensors += 1
    q4_params += params
    q4_raw += t.data_bytes.to_u64
    q4_pre += out_dim * (in_dim // 32_u64) * 24_u64
  elsif t.type.q8_0?
    q8_tensors += 1
    q8_params += params
    q8_raw += t.data_bytes.to_u64
    q8_pre += out_dim * (in_dim // 32_u64) * 36_u64
  end
end

def mib(bytes : UInt64)
  bytes.to_f / 1024.0 / 1024.0
end

def gib(bytes : UInt64)
  bytes.to_f / 1024.0 / 1024.0 / 1024.0
end

pre_total = total_raw - q4_raw - q8_raw + q4_pre + q8_pre
delta = pre_total.to_i64 - total_raw.to_i64

puts "model=#{path}"
puts "tensors=#{g.tensors.size}"
puts "gguf_tensor_raw_gib=#{gib(total_raw).round(3)}"
puts "q4_k_tensors=#{q4_tensors} q4_k_params=#{q4_params} q4_k_raw_gib=#{gib(q4_raw).round(3)} q4_k_pre_gib=#{gib(q4_pre).round(3)} q4_k_delta_mib=#{mib(q4_pre - q4_raw).round(1)}"
puts "q8_0_tensors=#{q8_tensors} q8_0_params=#{q8_params} q8_0_raw_gib=#{gib(q8_raw).round(3)} q8_0_pre_gib=#{gib(q8_pre).round(3)} q8_0_delta_mib=#{mib(q8_pre - q8_raw).round(1)}"
puts "estimated_prepacked_tensor_gib=#{gib(pre_total.to_u64).round(3)} delta_mib=#{mib(delta.abs.to_u64).round(1)} delta_sign=#{delta >= 0 ? "+" : "-"}"
