#!/usr/bin/env crystal

require "../src/ml/gguf/reader"

def usage
  STDERR.puts "usage: crystal scripts/estimate_rpi5_hybrid_routes.cr MODEL.gguf CPU_MS_PER_TOKEN"
  exit 2
end

path = ARGV[0]? || usage
cpu_ms = (ARGV[1]? || usage).to_f64
g = ML::GGUF::GGUFFile.new(path)
has_output_weight = g.tensors.any? { |t| t.name == "output.weight" }

record Row,
  family : String,
  type : String,
  ops : Float64,
  dense : Bool

def family_for(name : String, has_output_weight : Bool) : {String, Bool}
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
    has_output_weight ? {"embedding_lookup", false} : {"lm_head", true}
  when /^output\.weight$/
    {"lm_head", true}
  else
    {"other", true}
  end
end

rows = [] of Row
g.tensors.each do |t|
  next unless t.dims.size == 2
  family, dense = family_for(t.name, has_output_weight)
  ops = 2.0 * t.dims[0].to_f64 * t.dims[1].to_f64
  rows << Row.new(family, t.type.name, ops, dense)
end
dense_rows = rows.select(&.dense)
dense_ops = dense_rows.sum(&.ops)

# Current measured Pi 5 V3D rates from LANDMARKS. These are deliberately
# simple route-planning constants, not promoted universal kernel laws.
def measured_gops(row : Row) : Float64
  case row.type
  when "Q4_K"
    row.family == "ffn" ? 12.0 : 9.5
  when "Q6_K"
    3.0
  when "Q8_0"
    3.2
  when "Q5_K"
    3.0
  when "F32"
    10.0
  else
    3.0
  end
end

def gpu_ms(rows : Array(Row)) : Float64
  rows.sum { |r| r.ops / (measured_gops(r) * 1e9) * 1000.0 }
end

def fmt_ms(v : Float64) : String
  "#{v.round(1)}ms"
end

routes = {
  "ffn_only" => ->(r : Row) { r.family == "ffn" },
  "q4_ffn_only" => ->(r : Row) { r.family == "ffn" && r.type == "Q4_K" },
  "lm_head_only" => ->(r : Row) { r.family == "lm_head" },
  "attention_only" => ->(r : Row) { r.family == "attention" },
  "q4_dense" => ->(r : Row) { r.type == "Q4_K" },
  "all_quant_dense" => ->(r : Row) { r.type != "F32" },
}

puts "model=#{path}"
puts "cpu_ms_per_token=#{cpu_ms}"
puts "dense_ops_gop=#{(dense_ops / 1e9).round(3)}"
puts "assumed_v3d_gops=Q4_FFN:12.0,Q4_other:9.5,Q6:3.0,Q8:3.2,Q5:3.0,F32:10.0"
puts "hybrid_bound=cpu_ms - cpu_ms*selected_dense_op_fraction + gpu_selected_ms"
puts
puts "route\tselected_gop\tselected_pct\tgpu_ms\toptimistic_cpu_saved\toptimistic_hybrid_ms\tvs_cpu"
routes.each do |name, pred|
  selected = dense_rows.select { |r| pred.call(r) }
  ops = selected.sum(&.ops)
  pct = dense_ops > 0 ? ops / dense_ops : 0.0
  gms = gpu_ms(selected)
  saved = cpu_ms * pct
  hybrid = cpu_ms - saved + gms
  vs = hybrid / cpu_ms
  puts "#{name}\t#{(ops / 1e9).round(3)}\t#{(pct * 100.0).round(2)}%\t#{fmt_ms(gms)}\t#{fmt_ms(saved)}\t#{fmt_ms(hybrid)}\t#{vs.round(3)}x"
end
