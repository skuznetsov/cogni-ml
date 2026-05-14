require "option_parser"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

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
  avg_ms : Float64,
  wait_ms : Float64,
  per_row_ms : Float64,
  wait_per_row_ms : Float64,
  weighted_ms : Float64,
  weighted_wait_ms : Float64,
  names : Array(String)

def add_op(ops : Array(OpRef), name : String, qw : ML::GGUF::QuantWeight) : Nil
  ops << OpRef.new(name, qw)
end

def collect_ops(w : ML::GGUF::Qwen35Weights) : Array(OpRef)
  ops = [] of OpRef
  w.layers.each_with_index do |lw, il|
    case lw
    in ML::GGUF::Qwen35FullAttnWeights
      add_op(ops, "L#{il}.fa.q", lw.attn_q_qw)
      add_op(ops, "L#{il}.fa.k", lw.attn_k_qw)
      add_op(ops, "L#{il}.fa.v", lw.attn_v_qw)
      add_op(ops, "L#{il}.fa.o", lw.attn_output_qw)
      add_op(ops, "L#{il}.ffn.g", lw.ffn_gate_qw)
      add_op(ops, "L#{il}.ffn.u", lw.ffn_up_qw)
      add_op(ops, "L#{il}.ffn.d", lw.ffn_down_qw)
    in ML::GGUF::Qwen35RecurrentWeights
      add_op(ops, "L#{il}.rec.qkv", lw.attn_qkv_qw)
      add_op(ops, "L#{il}.rec.gate", lw.attn_gate_qw)
      add_op(ops, "L#{il}.rec.alpha", lw.ssm_alpha_qw)
      add_op(ops, "L#{il}.rec.beta", lw.ssm_beta_qw)
      add_op(ops, "L#{il}.rec.out", lw.ssm_out_qw)
      add_op(ops, "L#{il}.ffn.g", lw.ffn_gate_qw)
      add_op(ops, "L#{il}.ffn.u", lw.ffn_up_qw)
      add_op(ops, "L#{il}.ffn.d", lw.ffn_down_qw)
    end
  end
  add_op(ops, "output.full_logits_equiv", w.output)
  ops
end

def shape_stats(ops : Array(OpRef)) : Array(ShapeStats)
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

def input_for(in_dim : Int32, batch : Int32) : Array(Float32)
  Array(Float32).new(in_dim * batch) do |i|
    # Deterministic bounded values; avoids RNG setup in measured loops.
    ((((i.to_i64 * 1103515245_i64 + 12345_i64) & 0xffff_i64) / 32768.0) - 1.0).to_f32
  end
end

def percentile(sorted : Array(Float64), pct : Int32) : Float64
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

def bench_shape(stats : ShapeStats, warmup : Int32, runs : Int32, batch : Int32, profile_wait : Bool) : BenchRow
  x = input_for(stats.in_dim, batch)
  warmup.times { ML::GGUF::Qwen35Metal.matmul(stats.sample, x, batch) }

  times = Array(Float64).new(runs)
  wait_times = Array(Float64).new(runs)
  runs.times do
    ML::GGUF::Qwen35Metal::Profile.reset if profile_wait
    ML::GGUF::Qwen35Metal::Profile.enable! if profile_wait
    t0 = Time.instant
    begin
      out = ML::GGUF::Qwen35Metal.matmul(stats.sample, x, batch)
      raise "matmul returned nil for #{stats.type} #{stats.in_dim}->#{stats.out_dim}" if out.nil?
      times << (Time.instant - t0).total_milliseconds
      wait_times << ML::GGUF::Qwen35Metal::Profile.matmul_wait_ms if profile_wait
    ensure
      ML::GGUF::Qwen35Metal::Profile.disable! if profile_wait
    end
  end

  sorted = times.sort
  sorted_waits = wait_times.sort
  p50 = percentile(sorted, 50)
  wait_p50 = profile_wait ? percentile(sorted_waits, 50) : 0.0
  BenchRow.new(
    type: stats.type,
    in_dim: stats.in_dim,
    out_dim: stats.out_dim,
    count: stats.count,
    batch: batch,
    avg_ms: p50,
    wait_ms: wait_p50,
    per_row_ms: p50 / batch,
    wait_per_row_ms: wait_p50 / batch,
    weighted_ms: (p50 / batch) * stats.count,
    weighted_wait_ms: (wait_p50 / batch) * stats.count,
    names: stats.names,
  )
end

model = MODEL_PATH
warmup = 3
runs = 9
limit = 0
batch = 1
profile_wait = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_op_attribution [--model PATH] [--warmup N] [--runs N] [--limit N] [--batch N] [--profile-wait]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--warmup=N", "Warmup runs per shape (default: 3)") { |v| warmup = v.to_i }
  p.on("--runs=N", "Measured runs per shape (default: 9)") { |v| runs = v.to_i }
  p.on("--limit=N", "Only benchmark top-N dense-MAC shapes (default: all)") { |v| limit = v.to_i }
  p.on("--batch=N", "Rows per matmul call (default: 1)") { |v| batch = v.to_i }
  p.on("--profile-wait", "Also report Metal command wait time, excluding host-side input write/readback") { profile_wait = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
raise "--batch must be positive" unless batch > 0

w = ML::GGUF::Qwen35Weights.from_gguf(model)
stats = shape_stats(collect_ops(w))
stats = stats.sort_by { |s| -(s.in_dim.to_i64 * s.out_dim * s.count) }
stats = stats.first(limit) if limit > 0

puts "Qwen35 op attribution"
puts "model=#{model}"
puts "warmup=#{warmup} runs=#{runs} batch=#{batch}"
puts "note: output.full_logits_equiv measures full Q6_K lm-head matmul, not fused greedy top1."
puts "note: p50_ms is standalone matmul latency for the whole batch; per_row_ms and weighted_ms divide by batch."
puts "note: wait_ms is Metal command wait time only, excluding host-side input write/readback." if profile_wait
puts "note: weighted_ms ranks shapes, not total token latency."
puts

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
      r.type, r.in_dim, r.out_dim, r.count, r.batch, r.avg_ms, r.wait_ms,
      r.per_row_ms, r.weighted_ms, r.weighted_wait_ms, examples
  else
    printf "%-8s %7d %8d %5d %5d %10.3f %10.3f %12.3f  %s\n",
      r.type, r.in_dim, r.out_dim, r.count, r.batch, r.avg_ms, r.per_row_ms, r.weighted_ms, examples
  end
end

total = rows.sum(&.weighted_ms)
total_wait = rows.sum(&.weighted_wait_ms)
puts
printf "total_weighted_measured_ms=%.3f\n", total
printf "total_weighted_wait_ms=%.3f\n", total_wait if profile_wait
