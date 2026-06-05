require "option_parser"
require "../src/ml/gguf/gemma4_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

record OpRef, name : String, qw : ML::GGUF::QuantWeight
record ShapeStats,
  type : String,
  in_dim : Int32,
  out_dim : Int32,
  count : Int32,
  sample : ML::GGUF::QuantWeight,
  names : Array(String)

record BenchRow,
  type : String,
  in_dim : Int32,
  out_dim : Int32,
  count : Int32,
  batch : Int32,
  weight_bytes : Int64,
  p50_ms : Float64,
  wait_ms : Float64,
  per_row_ms : Float64,
  wait_per_row_ms : Float64,
  weighted_ms : Float64,
  weighted_wait_ms : Float64,
  names : Array(String)

private def row_category(row : BenchRow) : String
  return "head" if row.names.any? { |name| name == "output.full_logits_equiv" }
  return "ffn_upgate" if row.names.any? { |name| name.includes?(".ffn.g") || name.includes?(".ffn.u") }
  return "ffn_down" if row.names.any? { |name| name.includes?(".ffn.d") }
  return "attn_proj" if row.names.any? { |name| name.includes?(".attn.") }
  "other"
end

record PairRef,
  gate_name : String,
  up_name : String,
  gate_qw : ML::GGUF::QuantWeight,
  up_qw : ML::GGUF::QuantWeight

record Q4Layout, nsg : Int32, nr0 : Int32
record Q6Layout, nsg : Int32, nr0 : Int32

private def add_op(ops : Array(OpRef), name : String, qw : ML::GGUF::QuantWeight) : Nil
  ops << OpRef.new(name, qw)
end

private def collect_ops(w : ML::GGUF::Gemma4Weights) : Array(OpRef)
  ops = [] of OpRef
  w.layers.each_with_index do |lw, il|
    add_op(ops, "L#{il}.attn.q", lw.attn_q_qw)
    add_op(ops, "L#{il}.attn.k", lw.attn_k_qw)
    if v_qw = lw.attn_v_qw
      add_op(ops, "L#{il}.attn.v", v_qw)
    else
      # Gemma4 full-attention layers reuse K as V in the implemented route.
      add_op(ops, "L#{il}.attn.v_reuse_k", lw.attn_k_qw)
    end
    add_op(ops, "L#{il}.attn.o", lw.attn_output_qw)
    add_op(ops, "L#{il}.ffn.g", lw.ffn_gate_qw)
    add_op(ops, "L#{il}.ffn.u", lw.ffn_up_qw)
    add_op(ops, "L#{il}.ffn.d", lw.ffn_down_qw)
  end
  add_op(ops, "output.full_logits_equiv", w.token_embd)
  ops
end

private def collect_ffn_gate_up_pairs(w : ML::GGUF::Gemma4Weights) : Array(PairRef)
  pairs = [] of PairRef
  w.layers.each_with_index do |lw, il|
    pairs << PairRef.new("L#{il}.ffn.g", "L#{il}.ffn.u", lw.ffn_gate_qw, lw.ffn_up_qw)
  end
  pairs
end

private def shape_stats(ops : Array(OpRef)) : Array(ShapeStats)
  by_shape = Hash({String, Int32, Int32}, ShapeStats).new
  ops.each do |op|
    qw = op.qw
    key = {qw.type.to_s, qw.in_dim, qw.out_dim}
    if stats = by_shape[key]?
      stats.names << op.name
      by_shape[key] = ShapeStats.new(stats.type, stats.in_dim, stats.out_dim,
        stats.count + 1, stats.sample, stats.names)
    else
      by_shape[key] = ShapeStats.new(qw.type.to_s, qw.in_dim, qw.out_dim, 1, qw, [op.name])
    end
  end
  by_shape.values
end

private def input_for(in_dim : Int32, batch : Int32) : Array(Float32)
  Array(Float32).new(in_dim * batch) do |i|
    ((((i.to_i64 * 1103515245_i64 + 12345_i64) & 0xffff_i64) / 32768.0) - 1.0).to_f32
  end
end

private def percentile(sorted : Array(Float64), pct : Int32) : Float64
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

private def mib(bytes : Int64 | Float64) : Float64
  bytes.to_f64 / 1_048_576.0
end

private def gbps(bytes : Int64 | Float64, ms : Float64) : Float64
  return 0.0 unless ms > 0.0
  bytes.to_f64 / (ms / 1000.0) / 1_000_000_000.0
end

private def bench_q4_h16_pair(pair : PairRef, warmup : Int32, runs : Int32, batch : Int32) : Float64
  x = input_for(pair.gate_qw.in_dim, batch)
  ML::GGUF::Qwen35Metal.bench_q4_h16_pair_wait_ms(pair.gate_qw, pair.up_qw, x, batch, validate: true)
  warmup.times { ML::GGUF::Qwen35Metal.bench_q4_h16_pair_wait_ms(pair.gate_qw, pair.up_qw, x, batch) }
  times = Array(Float64).new(runs) do
    ML::GGUF::Qwen35Metal.bench_q4_h16_pair_wait_ms(pair.gate_qw, pair.up_qw, x, batch)
  end
  percentile(times.sort, 50)
end

private def parse_q4_layouts(spec : String) : Array(Q4Layout)
  spec.split(",", remove_empty: true).map do |part|
    fields = part.strip.split("x", remove_empty: true)
    raise "invalid Q4 layout '#{part}', expected NSGxNR0" unless fields.size == 2
    Q4Layout.new(fields[0].to_i, fields[1].to_i)
  end
end

private def parse_q6_layouts(spec : String) : Array(Q6Layout)
  spec.split(",", remove_empty: true).map do |part|
    fields = part.strip.split("x", remove_empty: true)
    raise "invalid Q6 layout '#{part}', expected NSGxNR0" unless fields.size == 2
    Q6Layout.new(fields[0].to_i, fields[1].to_i)
  end
end

private def parse_int_list(spec : String) : Array(Int32)
  spec.split(",", remove_empty: true).map { |part| part.strip.to_i }
end

private def bench_q4_layout(stats : ShapeStats,
                            layout : Q4Layout,
                            warmup : Int32,
                            runs : Int32,
                            batch : Int32,
                            validate : Bool) : Float64
  x = input_for(stats.in_dim, batch)
  ML::GGUF::Qwen35Metal.bench_q4_layout_wait_ms(stats.sample, x, batch, layout.nsg, layout.nr0, validate: validate)
  warmup.times { ML::GGUF::Qwen35Metal.bench_q4_layout_wait_ms(stats.sample, x, batch, layout.nsg, layout.nr0) }
  times = Array(Float64).new(runs) do
    ML::GGUF::Qwen35Metal.bench_q4_layout_wait_ms(stats.sample, x, batch, layout.nsg, layout.nr0)
  end
  percentile(times.sort, 50)
end

private def bench_q6_layout(stats : ShapeStats,
                            layout : Q6Layout,
                            warmup : Int32,
                            runs : Int32,
                            batch : Int32,
                            validate : Bool) : Float64
  x = input_for(stats.in_dim, batch)
  ML::GGUF::Qwen35Metal.bench_q6_layout_wait_ms(stats.sample, x, batch, layout.nsg, layout.nr0, validate: validate)
  warmup.times { ML::GGUF::Qwen35Metal.bench_q6_layout_wait_ms(stats.sample, x, batch, layout.nsg, layout.nr0) }
  times = Array(Float64).new(runs) do
    ML::GGUF::Qwen35Metal.bench_q6_layout_wait_ms(stats.sample, x, batch, layout.nsg, layout.nr0)
  end
  percentile(times.sort, 50)
end

private def bench_shape(stats : ShapeStats, warmup : Int32, runs : Int32, batch : Int32, profile_wait : Bool) : BenchRow
  x = input_for(stats.in_dim, batch)
  warmup.times { ML::GGUF::Qwen35Metal.matmul(stats.sample, x, batch) }

  times = [] of Float64
  waits = [] of Float64
  runs.times do
    ML::GGUF::Qwen35Metal::Profile.reset if profile_wait
    ML::GGUF::Qwen35Metal::Profile.enable! if profile_wait
    t0 = Time.instant
    begin
      out = ML::GGUF::Qwen35Metal.matmul(stats.sample, x, batch)
      raise "matmul returned nil for #{stats.type} #{stats.in_dim}->#{stats.out_dim}" if out.nil?
      times << (Time.instant - t0).total_milliseconds
      waits << ML::GGUF::Qwen35Metal::Profile.matmul_wait_ms if profile_wait
    ensure
      ML::GGUF::Qwen35Metal::Profile.disable! if profile_wait
    end
  end

  p50 = percentile(times.sort, 50)
  wait_p50 = profile_wait ? percentile(waits.sort, 50) : 0.0
  weight_bytes = stats.sample.raw.size.to_i64
  BenchRow.new(
    type: stats.type,
    in_dim: stats.in_dim,
    out_dim: stats.out_dim,
    count: stats.count,
    batch: batch,
    weight_bytes: weight_bytes,
    p50_ms: p50,
    wait_ms: wait_p50,
    per_row_ms: p50 / batch,
    wait_per_row_ms: wait_p50 / batch,
    weighted_ms: (p50 / batch) * stats.count,
    weighted_wait_ms: (wait_p50 / batch) * stats.count,
    names: stats.names,
  )
end

private def print_prefill_q4_pair_table(w : ML::GGUF::Gemma4Weights, warmup : Int32, runs : Int32, batch : Int32) : Nil
  pairs = collect_ffn_gate_up_pairs(w).select do |p|
    p.gate_qw.type.q4_k? && p.up_qw.type.q4_k? &&
      p.gate_qw.in_dim == p.up_qw.in_dim &&
      p.gate_qw.out_dim == p.up_qw.out_dim
  end
  by_shape = pairs.group_by { |p| {p.gate_qw.in_dim, p.gate_qw.out_dim} }

  puts
  puts "Gemma4 Q4_H16 FFN gate+up pair route"
  puts "note: pair route is the actual fused gate/up corridor used by row prefill when enabled."
  printf "%7s %8s %5s %5s %10s %10s %12s %14s  %s\n",
    "in", "out", "pairs", "batch", "pair_wait", "batch_wait", "per_row", "weighted_pair", "examples"
  by_shape.to_a.sort_by { |(shape, shape_pairs)| -(shape[0].to_i64 * shape[1] * shape_pairs.size) }.each do |shape, shape_pairs|
    p50 = bench_q4_h16_pair(shape_pairs.first, warmup, runs, batch)
    weighted = (p50 / batch) * shape_pairs.size
    examples = shape_pairs.first(3).map { |p| "#{p.gate_name}+#{p.up_name}" }.join(",")
    examples += ",..." if shape_pairs.size > 3
    printf "%7d %8d %5d %5d %10.3f %10.3f %12.3f %14.3f  %s\n",
      shape[0], shape[1], shape_pairs.size, batch, p50, p50 * shape_pairs.size, p50 / batch, weighted, examples
  end
end

private def print_q4_layout_sweep(stats : Array(ShapeStats),
                                  layouts : Array(Q4Layout),
                                  warmup : Int32,
                                  runs : Int32,
                                  batch : Int32) : Nil
  q4_stats = stats.select(&.sample.type.q4_k?)
  return if q4_stats.empty? || layouts.empty?

  puts
  puts "Gemma4 Q4_K GEMV layout sweep"
  puts "note: layout is NSGxNR0; default is 2x2. p50_wait is command wait only."
  printf "%7s %8s %5s %5s %8s %10s %10s %9s  %s\n",
    "in", "out", "calls", "batch", "layout", "p50_wait", "weighted", "eff_gbps", "examples"
  q4_stats.each do |shape|
    layouts.each do |layout|
      p50 = bench_q4_layout(shape, layout, warmup, runs, batch, validate: true)
      weighted = (p50 / batch) * shape.count
      examples = shape.names.first(3).join(",")
      examples += ",..." if shape.names.size > 3
      printf "%7d %8d %5d %5d %8s %10.3f %10.3f %9.1f  %s\n",
        shape.in_dim, shape.out_dim, shape.count, batch, "#{layout.nsg}x#{layout.nr0}",
        p50, weighted, gbps(shape.sample.raw.size.to_i64, p50), examples
    end
  end
end

private def print_q6_layout_sweep(stats : Array(ShapeStats),
                                  layouts : Array(Q6Layout),
                                  warmup : Int32,
                                  runs : Int32,
                                  batch : Int32) : Nil
  q6_stats = stats.select(&.sample.type.q6_k?)
  return if q6_stats.empty? || layouts.empty?

  puts
  puts "Gemma4 Q6_K GEMV layout sweep"
  puts "note: layout is NSGxNR0; default is 2x1. p50_wait is command wait only."
  printf "%7s %8s %5s %5s %8s %10s %10s %9s  %s\n",
    "in", "out", "calls", "batch", "layout", "p50_wait", "weighted", "eff_gbps", "examples"
  q6_stats.each do |shape|
    layouts.each do |layout|
      p50 = bench_q6_layout(shape, layout, warmup, runs, batch, validate: true)
      weighted = (p50 / batch) * shape.count
      examples = shape.names.first(3).join(",")
      examples += ",..." if shape.names.size > 3
      printf "%7d %8d %5d %5d %8s %10.3f %10.3f %9.1f  %s\n",
        shape.in_dim, shape.out_dim, shape.count, batch, "#{layout.nsg}x#{layout.nr0}",
        p50, weighted, gbps(shape.sample.raw.size.to_i64, p50), examples
    end
  end
end

private def print_head_top1_rows_sweep(w : ML::GGUF::Gemma4Weights,
                                       rows_values : Array(Int32),
                                       warmup : Int32,
                                       runs : Int32) : Nil
  return if rows_values.empty?
  x = input_for(w.token_embd.in_dim, 1)
  qw = w.token_embd

  puts
  puts "Gemma4 Q6_K head top1 rows-per-threadgroup sweep"
  puts "note: default is 12 rows/tg; p50_wait includes RMSNorm, tile top1, and tile reduce."
  printf "%10s %10s %10s %9s\n", "rows_tg", "p50_wait", "tile_count", "eff_gbps"
  rows_values.each do |rows_per_tg|
    ML::GGUF::Qwen35Metal.bench_head_top1_rows_wait_ms(qw, x, w.output_norm, w.hparams.rms_eps, rows_per_tg, validate: true)
    warmup.times { ML::GGUF::Qwen35Metal.bench_head_top1_rows_wait_ms(qw, x, w.output_norm, w.hparams.rms_eps, rows_per_tg) }
    times = Array(Float64).new(runs) do
      ML::GGUF::Qwen35Metal.bench_head_top1_rows_wait_ms(qw, x, w.output_norm, w.hparams.rms_eps, rows_per_tg)
    end
    p50 = percentile(times.sort, 50)
    tile_count = (qw.out_dim + rows_per_tg - 1) // rows_per_tg
    printf "%10d %10.3f %10d %9.1f\n", rows_per_tg, p50, tile_count, gbps(qw.raw.size.to_i64, p50)
  end
end

model = MODEL_PATH
warmup = 2
runs = 5
limit = 0
batch = 256
profile_wait = false
prefill_q4_pair_wait = false
prefill_q4_pair_only = false
exclude_output_head = false
q4_layout_sweep = [] of Q4Layout
q6_layout_sweep = [] of Q6Layout
head_top1_rows_sweep = [] of Int32

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_op_attribution [--model PATH] [--warmup N] [--runs N] [--limit N] [--batch N] [--profile-wait] [--exclude-output-head] [--prefill-q4-pair-wait] [--prefill-q4-pair-only] [--q4-layout-sweep=NSGxNR0,...] [--q6-layout-sweep=NSGxNR0,...] [--head-top1-rows-sweep=ROWS,...]"
  p.on("--model=PATH", "Gemma4 GGUF model path") { |v| model = v }
  p.on("--warmup=N", "Warmup runs per shape (default: 2)") { |v| warmup = v.to_i }
  p.on("--runs=N", "Measured runs per shape (default: 5)") { |v| runs = v.to_i }
  p.on("--limit=N", "Only benchmark top-N dense-MAC shapes (default: all)") { |v| limit = v.to_i }
  p.on("--batch=N", "Rows per matmul call (default: 256)") { |v| batch = v.to_i }
  p.on("--profile-wait", "Also report Metal command wait time, excluding host-side input write/readback") { profile_wait = true }
  p.on("--exclude-output-head", "Exclude tied full-logits head from the benchmark rows and totals") { exclude_output_head = true }
  p.on("--prefill-q4-pair-wait", "Also benchmark the actual Q4_H16 FFN gate+up pair route used by prefill") { prefill_q4_pair_wait = true }
  p.on("--prefill-q4-pair-only", "Only benchmark the actual Q4_H16 FFN gate+up pair route used by prefill") { prefill_q4_pair_wait = true; prefill_q4_pair_only = true }
  p.on("--q4-layout-sweep=LIST", "Benchmark alternate Q4_K GEMV layouts, e.g. 1x1,1x2,2x1,2x2,2x3,2x4") { |v| q4_layout_sweep = parse_q4_layouts(v) }
  p.on("--q6-layout-sweep=LIST", "Benchmark alternate Q6_K GEMV layouts, e.g. 1x1,1x2,2x1,2x2,2x3,2x4") { |v| q6_layout_sweep = parse_q6_layouts(v) }
  p.on("--head-top1-rows-sweep=LIST", "Benchmark Q6_K head top1 rows/tg values, e.g. 8,10,12,16,20,24") { |v| head_top1_rows_sweep = parse_int_list(v) }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
raise "--batch must be positive" unless batch > 0
raise "model not found: #{model}" unless File.exists?(model)

w = ML::GGUF::Gemma4Weights.from_gguf(model)
ops = collect_ops(w)
ops = ops.reject { |op| op.name == "output.full_logits_equiv" } if exclude_output_head
stats = shape_stats(ops)
stats = stats.sort_by { |s| -(s.in_dim.to_i64 * s.out_dim * s.count) }
stats = stats.first(limit) if limit > 0

puts "Gemma4 op attribution"
puts "model=#{model}"
puts "warmup=#{warmup} runs=#{runs} batch=#{batch}"
puts "note: output.full_logits_equiv uses token_embd as tied full Q6_K lm-head matmul." unless exclude_output_head
puts "note: output.full_logits_equiv excluded from rows/totals." if exclude_output_head
puts "note: v_reuse_k entries count the actual full-attention K-as-V projection route."
puts "note: p50_ms is standalone matmul latency for the whole batch; weighted_ms=(p50/batch)*calls."
puts "note: batch_total_ms=(p50_ms*calls) estimates full prompt-prefill wall contribution for this shape."
puts "note: weight_mib is quantized weight traffic per standalone call; weighted_mib=(weight_mib/batch)*calls."
puts "note: wait_ms is Metal command wait time only, excluding host-side input write/readback." if profile_wait
puts

if prefill_q4_pair_wait
  print_prefill_q4_pair_table(w, warmup, runs, batch)
  exit if prefill_q4_pair_only
  puts
end

print_q4_layout_sweep(stats, q4_layout_sweep, warmup, runs, batch) unless q4_layout_sweep.empty?
print_q6_layout_sweep(stats, q6_layout_sweep, warmup, runs, batch) unless q6_layout_sweep.empty?
print_head_top1_rows_sweep(w, head_top1_rows_sweep, warmup, runs) unless head_top1_rows_sweep.empty?

rows = stats.map { |s| bench_shape(s, warmup, runs, batch, profile_wait) }
rows.sort_by! { |r| profile_wait ? -r.weighted_wait_ms : -r.weighted_ms }

if profile_wait
  printf "%-8s %7s %8s %5s %5s %10s %10s %10s %10s %10s %11s %10s %12s %12s %9s  %s\n",
    "type", "in", "out", "calls", "batch", "p50_ms", "wait_ms", "batch_ms", "batch_wait", "per_row", "weight_mib", "eff_gbps", "weighted_ms", "weighted_wait", "w_mib", "examples"
else
  printf "%-8s %7s %8s %5s %5s %10s %10s %10s %11s %10s %12s %9s  %s\n",
    "type", "in", "out", "calls", "batch", "p50_ms", "batch_ms", "per_row", "weight_mib", "eff_gbps", "weighted_ms", "w_mib", "examples"
end
rows.each do |r|
  examples = r.names.first(3).join(",")
  examples += ",..." if r.names.size > 3
  weighted_bytes = r.weight_bytes.to_f64 * r.count / r.batch
  batch_total_ms = r.p50_ms * r.count
  batch_wait_ms = r.wait_ms * r.count
  if profile_wait
    printf "%-8s %7d %8d %5d %5d %10.3f %10.3f %10.3f %10.3f %10.3f %11.2f %10.1f %12.3f %12.3f %9.1f  %s\n",
      r.type, r.in_dim, r.out_dim, r.count, r.batch, r.p50_ms, r.wait_ms,
      batch_total_ms, batch_wait_ms, r.per_row_ms, mib(r.weight_bytes), gbps(r.weight_bytes, r.wait_ms),
      r.weighted_ms, r.weighted_wait_ms, mib(weighted_bytes), examples
  else
    printf "%-8s %7d %8d %5d %5d %10.3f %10.3f %10.3f %11.2f %10.1f %12.3f %9.1f  %s\n",
      r.type, r.in_dim, r.out_dim, r.count, r.batch, r.p50_ms,
      batch_total_ms, r.per_row_ms, mib(r.weight_bytes), gbps(r.weight_bytes, r.p50_ms),
      r.weighted_ms, mib(weighted_bytes), examples
  end
end

puts
printf "total_weighted_measured_ms=%.3f\n", rows.sum(&.weighted_ms)
printf "total_weighted_wait_ms=%.3f\n", rows.sum(&.weighted_wait_ms) if profile_wait
printf "total_batch_measured_ms=%.3f\n", rows.sum { |row| row.p50_ms * row.count }
printf "total_batch_wait_ms=%.3f\n", rows.sum { |row| row.wait_ms * row.count } if profile_wait
total_weighted_bytes = rows.sum { |row| row.weight_bytes.to_f64 * row.count / row.batch }
total_batch_bytes = rows.sum { |row| row.weight_bytes.to_f64 * row.count }
printf "total_weighted_weight_mib=%.3f\n", mib(total_weighted_bytes)
printf "total_batch_weight_mib=%.3f\n", mib(total_batch_bytes)
if profile_wait
  total_wait = rows.sum(&.weighted_wait_ms)
  printf "total_weighted_effective_gbps=%.3f\n", gbps(total_weighted_bytes, total_wait) if total_wait > 0.0
  total_batch_wait = rows.sum { |row| row.wait_ms * row.count }
  printf "total_batch_effective_gbps=%.3f\n", gbps(total_batch_bytes, total_batch_wait) if total_batch_wait > 0.0
else
  total_ms = rows.sum(&.weighted_ms)
  printf "total_weighted_effective_gbps=%.3f\n", gbps(total_weighted_bytes, total_ms) if total_ms > 0.0
  total_batch_ms = rows.sum { |row| row.p50_ms * row.count }
  printf "total_batch_effective_gbps=%.3f\n", gbps(total_batch_bytes, total_batch_ms) if total_batch_ms > 0.0
end

puts
puts "Category summary"
category_order = ["ffn_upgate", "ffn_down", "attn_proj", "head", "other"]
by_category = rows.group_by { |row| row_category(row) }
category_order.each do |category|
  category_rows = by_category[category]? || next
  measured = category_rows.sum(&.weighted_ms)
  wait = category_rows.sum(&.weighted_wait_ms)
  category_bytes = category_rows.sum { |row| row.weight_bytes.to_f64 * row.count / row.batch }
  category_batch_ms = category_rows.sum { |row| row.p50_ms * row.count }
  category_batch_wait = category_rows.sum { |row| row.wait_ms * row.count }
  category_batch_bytes = category_rows.sum { |row| row.weight_bytes.to_f64 * row.count }
  if profile_wait
    printf "  %-10s rows=%2d weighted_ms=%8.3f weighted_wait_ms=%8.3f weighted_mib=%8.1f batch_ms=%9.3f batch_wait_ms=%9.3f batch_mib=%9.1f eff_gbps=%7.1f\n",
      category, category_rows.size, measured, wait, mib(category_bytes),
      category_batch_ms, category_batch_wait, mib(category_batch_bytes), gbps(category_batch_bytes, category_batch_wait)
  else
    printf "  %-10s rows=%2d weighted_ms=%8.3f weighted_mib=%8.1f batch_ms=%9.3f batch_mib=%9.1f eff_gbps=%7.1f\n",
      category, category_rows.size, measured, mib(category_bytes),
      category_batch_ms, mib(category_batch_bytes), gbps(category_batch_bytes, category_batch_ms)
  end
end

body_rows = rows.reject { |row| row_category(row) == "head" }
body_weighted_bytes = body_rows.sum { |row| row.weight_bytes.to_f64 * row.count / row.batch }
body_batch_bytes = body_rows.sum { |row| row.weight_bytes.to_f64 * row.count }
printf "body_no_head_weighted_measured_ms=%.3f\n", body_rows.sum(&.weighted_ms)
printf "body_no_head_weighted_wait_ms=%.3f\n", body_rows.sum(&.weighted_wait_ms) if profile_wait
printf "body_no_head_weighted_weight_mib=%.3f\n", mib(body_weighted_bytes)
printf "body_no_head_batch_measured_ms=%.3f\n", body_rows.sum { |row| row.p50_ms * row.count }
printf "body_no_head_batch_wait_ms=%.3f\n", body_rows.sum { |row| row.wait_ms * row.count } if profile_wait
printf "body_no_head_batch_weight_mib=%.3f\n", mib(body_batch_bytes)
