#!/usr/bin/env crystal

require "../src/ml/gguf/reader"

def usage
  STDERR.puts "usage: crystal scripts/estimate_v3d_layer_plan.cr MODEL.gguf [budget_gib]"
  exit 2
end

path = ARGV[0]? || usage
budget_gib = (ARGV[1]? || "3.5").to_f64
budget = (budget_gib * 1024.0 * 1024.0 * 1024.0).to_u64
g = ML::GGUF::GGUFFile.new(path)

def prepack_bytes(t) : UInt64
  raw = t.data_bytes.to_u64
  return raw unless t.dims.size == 2
  in_dim = t.dims[0].to_u64
  out_dim = t.dims[1].to_u64
  if t.type.q4_k?
    out_dim * ((in_dim + 31_u64) // 32_u64) * 24_u64
  elsif t.type.q8_0?
    out_dim * ((in_dim + 31_u64) // 32_u64) * 36_u64
  else
    raw
  end
end

def mib(bytes : UInt64)
  bytes.to_f / 1024.0 / 1024.0
end

def gib(bytes : UInt64)
  bytes.to_f / 1024.0 / 1024.0 / 1024.0
end

blocks = Hash(Int32, UInt64).new(0_u64)
blocks_raw = Hash(Int32, UInt64).new(0_u64)
nonblock_pre = 0_u64
nonblock_raw = 0_u64
total_pre = 0_u64
total_raw = 0_u64

g.tensors.each do |t|
  raw = t.data_bytes.to_u64
  pre = prepack_bytes(t)
  total_raw += raw
  total_pre += pre
  if m = t.name.match(/^blk\.(\d+)\./)
    idx = m[1].to_i
    blocks[idx] += pre
    blocks_raw[idx] += raw
  else
    nonblock_pre += pre
    nonblock_raw += raw
  end
end

sorted = blocks.keys.sort
max_block = sorted.max_of? { |i| blocks[i] } || 0_u64
min_block = sorted.min_of? { |i| blocks[i] } || 0_u64
avg_block = sorted.empty? ? 0.0 : sorted.sum { |i| blocks[i].to_f } / sorted.size

puts "model=#{path}"
puts "budget_gib=#{budget_gib}"
puts "blocks=#{sorted.size}"
puts "nonblock_raw_mib=#{mib(nonblock_raw).round(1)} nonblock_pre_mib=#{mib(nonblock_pre).round(1)}"
puts "total_raw_gib=#{gib(total_raw).round(3)} total_pre_gib=#{gib(total_pre).round(3)}"
puts "block_pre_mib min=#{mib(min_block).round(1)} avg=#{(avg_block / 1024.0 / 1024.0).round(1)} max=#{mib(max_block).round(1)}"

if max_block > budget
  puts "single_block_fits=false max_block_mib=#{mib(max_block).round(1)}"
else
  puts "single_block_fits=true"
end

best = 0
best_bytes = 0_u64
sorted.each_with_index do |start, si|
  sum = 0_u64
  count = 0
  sorted[si..].each do |idx|
    next if idx != start + count
    b = blocks[idx]
    break if sum + b > budget
    sum += b
    count += 1
  end
  if count > best
    best = count
    best_bytes = sum
  end
end

puts "max_contiguous_blocks_in_budget=#{best} bytes_mib=#{mib(best_bytes).round(1)}"
puts "resident_all_blocks_fits=#{blocks.values.sum <= budget}"
puts "resident_all_with_nonblock_fits=#{total_pre <= budget}"

puts "top_blocks_by_pre_mib:"
sorted.sort { |a, b| blocks[b] <=> blocks[a] }.first(12).each do |i|
  puts "blk.#{i}\traw_mib=#{mib(blocks_raw[i]).round(1)}\tpre_mib=#{mib(blocks[i]).round(1)}"
end
