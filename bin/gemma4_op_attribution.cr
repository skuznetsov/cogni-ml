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
  p50_ms : Float64,
  wait_ms : Float64,
  per_row_ms : Float64,
  wait_per_row_ms : Float64,
  weighted_ms : Float64,
  weighted_wait_ms : Float64,
  names : Array(String)

record PairRef,
  gate_name : String,
  up_name : String,
  gate_qw : ML::GGUF::QuantWeight,
  up_qw : ML::GGUF::QuantWeight

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

private def bench_q4_h16_pair(pair : PairRef, warmup : Int32, runs : Int32, batch : Int32) : Float64
  x = input_for(pair.gate_qw.in_dim, batch)
  ML::GGUF::Qwen35Metal.bench_q4_h16_pair_wait_ms(pair.gate_qw, pair.up_qw, x, batch, validate: true)
  warmup.times { ML::GGUF::Qwen35Metal.bench_q4_h16_pair_wait_ms(pair.gate_qw, pair.up_qw, x, batch) }
  times = Array(Float64).new(runs) do
    ML::GGUF::Qwen35Metal.bench_q4_h16_pair_wait_ms(pair.gate_qw, pair.up_qw, x, batch)
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
  BenchRow.new(
    type: stats.type,
    in_dim: stats.in_dim,
    out_dim: stats.out_dim,
    count: stats.count,
    batch: batch,
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
  printf "%7s %8s %5s %5s %10s %10s %14s  %s\n",
    "in", "out", "pairs", "batch", "pair_wait", "per_row", "weighted_pair", "examples"
  by_shape.to_a.sort_by { |(shape, shape_pairs)| -(shape[0].to_i64 * shape[1] * shape_pairs.size) }.each do |shape, shape_pairs|
    p50 = bench_q4_h16_pair(shape_pairs.first, warmup, runs, batch)
    weighted = (p50 / batch) * shape_pairs.size
    examples = shape_pairs.first(3).map { |p| "#{p.gate_name}+#{p.up_name}" }.join(",")
    examples += ",..." if shape_pairs.size > 3
    printf "%7d %8d %5d %5d %10.3f %10.3f %14.3f  %s\n",
      shape[0], shape[1], shape_pairs.size, batch, p50, p50 / batch, weighted, examples
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

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_op_attribution [--model PATH] [--warmup N] [--runs N] [--limit N] [--batch N] [--profile-wait] [--prefill-q4-pair-wait] [--prefill-q4-pair-only]"
  p.on("--model=PATH", "Gemma4 GGUF model path") { |v| model = v }
  p.on("--warmup=N", "Warmup runs per shape (default: 2)") { |v| warmup = v.to_i }
  p.on("--runs=N", "Measured runs per shape (default: 5)") { |v| runs = v.to_i }
  p.on("--limit=N", "Only benchmark top-N dense-MAC shapes (default: all)") { |v| limit = v.to_i }
  p.on("--batch=N", "Rows per matmul call (default: 256)") { |v| batch = v.to_i }
  p.on("--profile-wait", "Also report Metal command wait time, excluding host-side input write/readback") { profile_wait = true }
  p.on("--prefill-q4-pair-wait", "Also benchmark the actual Q4_H16 FFN gate+up pair route used by prefill") { prefill_q4_pair_wait = true }
  p.on("--prefill-q4-pair-only", "Only benchmark the actual Q4_H16 FFN gate+up pair route used by prefill") { prefill_q4_pair_wait = true; prefill_q4_pair_only = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
raise "--batch must be positive" unless batch > 0
raise "model not found: #{model}" unless File.exists?(model)

w = ML::GGUF::Gemma4Weights.from_gguf(model)
stats = shape_stats(collect_ops(w))
stats = stats.sort_by { |s| -(s.in_dim.to_i64 * s.out_dim * s.count) }
stats = stats.first(limit) if limit > 0

puts "Gemma4 op attribution"
puts "model=#{model}"
puts "warmup=#{warmup} runs=#{runs} batch=#{batch}"
puts "note: output.full_logits_equiv uses token_embd as tied full Q6_K lm-head matmul."
puts "note: v_reuse_k entries count the actual full-attention K-as-V projection route."
puts "note: p50_ms is standalone matmul latency for the whole batch; weighted_ms=(p50/batch)*calls."
puts "note: wait_ms is Metal command wait time only, excluding host-side input write/readback." if profile_wait
puts

if prefill_q4_pair_wait
  print_prefill_q4_pair_table(w, warmup, runs, batch)
  exit if prefill_q4_pair_only
  puts
end

rows = stats.map { |s| bench_shape(s, warmup, runs, batch, profile_wait) }
rows.sort_by! { |r| profile_wait ? -r.weighted_wait_ms : -r.weighted_ms }

if profile_wait
  printf "%-8s %7s %8s %5s %5s %10s %10s %10s %12s %12s  %s\n",
    "type", "in", "out", "calls", "batch", "p50_ms", "wait_ms", "per_row", "weighted_ms", "weighted_wait", "examples"
else
  printf "%-8s %7s %8s %5s %5s %10s %10s %12s  %s\n",
    "type", "in", "out", "calls", "batch", "p50_ms", "per_row", "weighted_ms", "examples"
end
rows.each do |r|
  examples = r.names.first(3).join(",")
  examples += ",..." if r.names.size > 3
  if profile_wait
    printf "%-8s %7d %8d %5d %5d %10.3f %10.3f %10.3f %12.3f %12.3f  %s\n",
      r.type, r.in_dim, r.out_dim, r.count, r.batch, r.p50_ms, r.wait_ms,
      r.per_row_ms, r.weighted_ms, r.weighted_wait_ms, examples
  else
    printf "%-8s %7d %8d %5d %5d %10.3f %10.3f %12.3f  %s\n",
      r.type, r.in_dim, r.out_dim, r.count, r.batch, r.p50_ms,
      r.per_row_ms, r.weighted_ms, examples
  end
end

puts
printf "total_weighted_measured_ms=%.3f\n", rows.sum(&.weighted_ms)
printf "total_weighted_wait_ms=%.3f\n", rows.sum(&.weighted_wait_ms) if profile_wait
