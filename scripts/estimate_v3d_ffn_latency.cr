#!/usr/bin/env crystal

require "../src/ml/gguf/reader"

def usage
  STDERR.puts "usage: crystal scripts/estimate_v3d_ffn_latency.cr MODEL.gguf [gops]"
  exit 2
end

path = ARGV[0]? || usage
gops = (ARGV[1]? || "2.75").to_f64
g = ML::GGUF::GGUFFile.new(path)

ffn_tensors = g.tensors.select do |t|
  t.dims.size == 2 && t.name.matches?(/^blk\.\d+\.ffn_(gate|up|down)\.weight$/)
end

ops_by_block = Hash(Int32, Float64).new(0.0)
names_by_block = Hash(Int32, Array(String)).new { |h, k| h[k] = [] of String }
ffn_tensors.each do |t|
  next unless m = t.name.match(/^blk\.(\d+)\./)
  idx = m[1].to_i
  in_dim = t.dims[0].to_f64
  out_dim = t.dims[1].to_f64
  ops_by_block[idx] += 2.0 * in_dim * out_dim
  names_by_block[idx] << t.name
end

total_ops = ops_by_block.values.sum
total_ms = total_ops / (gops * 1e9) * 1000.0
blocks = ops_by_block.keys.sort
missing_down = blocks.count { |i| names_by_block[i].none?(&.includes?(".ffn_down.")) }

puts "model=#{path}"
puts "assumed_v3d_gops=#{gops}"
puts "ffn_weight_tensors=#{ffn_tensors.size} blocks_with_ffn=#{blocks.size} blocks_missing_q4_ffn_down=#{missing_down}"
puts "total_ffn_ops_gop=#{(total_ops / 1e9).round(3)} estimated_ffn_ms_per_token=#{total_ms.round(1)} estimated_ffn_tok_s_upper_bound=#{(1000.0 / total_ms).round(3)}"
puts "top_ffn_blocks_by_estimated_ms:"
blocks.sort { |a, b| ops_by_block[b] <=> ops_by_block[a] }.first(12).each do |i|
  ms = ops_by_block[i] / (gops * 1e9) * 1000.0
  puts "blk.#{i}\tops_gop=#{(ops_by_block[i] / 1e9).round(3)}\test_ms=#{ms.round(2)}\ttensors=#{names_by_block[i].sort.join(",")}"
end
