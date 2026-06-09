#!/usr/bin/env crystal

require "../src/ml/gguf/reader"

def usage
  STDERR.puts "usage: crystal scripts/estimate_v3d_weight_ops.cr MODEL.gguf [gops]"
  exit 2
end

path = ARGV[0]? || usage
gops = (ARGV[1]? || "12.0").to_f64
g = ML::GGUF::GGUFFile.new(path)

has_output_weight = g.tensors.any? { |t| t.name == "output.weight" }

record Row,
  family : String,
  name : String,
  type : String,
  in_dim : Int64,
  out_dim : Int64,
  ops : Float64,
  dense_per_token : Bool

def classify(name : String, has_output_weight : Bool) : {String, Bool}
  case name
  when /^blk\.\d+\.ffn_(gate|up|down)\.weight$/
    {"ffn", true}
  when /^blk\.\d+\.attn_(qkv|q|k|v|output|gate)\.weight$/
    {"attention", true}
  when /^blk\.\d+\.ssm_(out|alpha|beta)\.weight$/
    {"ssm_weight", true}
  when /^blk\.\d+\.ssm_conv1d\.weight$/
    {"ssm_conv1d", true}
  when /^token_embd\.weight$/
    has_output_weight ? {"embedding_lookup", false} : {"tied_lm_head", true}
  when /^output\.weight$/
    {"lm_head", true}
  else
    {"other", true}
  end
end

rows = [] of Row
g.tensors.each do |t|
  next unless t.dims.size == 2
  in_dim = t.dims[0]
  out_dim = t.dims[1]
  ops = 2.0 * in_dim.to_f64 * out_dim.to_f64
  family, dense_per_token = classify(t.name, has_output_weight)
  rows << Row.new(family, t.name, t.type.name, in_dim, out_dim, ops, dense_per_token)
end

family_ops = Hash(String, Float64).new(0.0)
family_count = Hash(String, Int32).new(0)
family_types = Hash(String, Hash(String, Int32)).new { |h, k| h[k] = Hash(String, Int32).new(0) }
rows.each do |r|
  family_ops[r.family] += r.ops
  family_count[r.family] += 1
  family_types[r.family][r.type] += 1
end

dense_rows = rows.select(&.dense_per_token)
dense_ops = dense_rows.sum(&.ops)
dense_ms = dense_ops / (gops * 1e9) * 1000.0
total_ops = rows.sum(&.ops)

puts "model=#{path}"
puts "assumed_dense_weight_gops=#{gops}"
puts "has_output_weight=#{has_output_weight}"
puts "two_d_weight_tensors=#{rows.size}"
puts "dense_per_token_weight_tensors=#{dense_rows.size}"
puts "dense_per_token_ops_gop=#{(dense_ops / 1e9).round(3)} estimated_dense_ms_per_token=#{dense_ms.round(1)} estimated_dense_tok_s_upper_bound=#{(1000.0 / dense_ms).round(3)}"
puts "all_2d_tensor_ops_gop=#{(total_ops / 1e9).round(3)}"
puts
puts "by_family:"
family_ops.to_a.sort_by { |(_, ops)| -ops }.each do |family, ops|
  pct = dense_ops > 0 ? (ops / dense_ops * 100.0) : 0.0
  ms = ops / (gops * 1e9) * 1000.0
  dense = rows.any? { |r| r.family == family && r.dense_per_token }
  types = family_types[family].to_a.sort_by { |(type, _)| type }.map { |type, count| "#{type}:#{count}" }.join(",")
  dense_note = dense ? "dense" : "lookup/non-dense"
  puts "#{family}\t#{dense_note}\ttensors=#{family_count[family]}\tops_gop=#{(ops / 1e9).round(3)}\tpct_of_dense=#{pct.round(2)}\test_ms_if_dense=#{ms.round(1)}\ttypes=#{types}"
end

puts
puts "top_tensors:"
rows.sort_by { |r| -r.ops }.first(24).each do |r|
  pct = dense_ops > 0 ? (r.ops / dense_ops * 100.0) : 0.0
  dense_note = r.dense_per_token ? "dense" : "lookup/non-dense"
  puts "#{r.name}\t#{dense_note}\tfamily=#{r.family}\ttype=#{r.type}\tshape=#{r.in_dim}x#{r.out_dim}\tops_gop=#{(r.ops / 1e9).round(3)}\tpct_of_dense=#{pct.round(2)}"
end
