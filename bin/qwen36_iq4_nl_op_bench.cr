require "option_parser"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_NL.gguf"

record ShapeRef,
  name : String,
  in_dim : Int32,
  out_dim : Int32,
  full_raw : Bytes,
  bench_raw : Bytes,
  bench_out_dim : Int32,
  tensor_count : Int32,
  examples : Array(String)

record BenchRow,
  name : String,
  in_dim : Int32,
  out_dim : Int32,
  bench_out_dim : Int32,
  tensor_count : Int32,
  batch : Int32,
  metal_p50_ms : Float64,
  metal_min_ms : Float64,
  metal_max_ms : Float64,
  cpu_subset_ms : Float64?,
  cos : Float64?,
  max_abs_diff : Float32?,
  examples : Array(String)

def percentile(sorted : Array(Float64), pct : Int32) : Float64
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

def input_for(in_dim : Int32, batch : Int32) : Array(Float32)
  Array(Float32).new(in_dim * batch) do |i|
    ((((i.to_i64 * 1103515245_i64 + 12345_i64) & 0xffff_i64) / 32768.0) - 1.0).to_f32
  end
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  na = 0.0
  nb = 0.0
  a.each_with_index do |av, i|
    bv = b[i]
    dot += av.to_f64 * bv.to_f64
    na += av.to_f64 * av.to_f64
    nb += bv.to_f64 * bv.to_f64
  end
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  m = 0.0_f32
  a.each_with_index do |av, i|
    d = (av - b[i]).abs
    m = d if d > m
  end
  m
end

def collect_iq4_shapes(g : ML::GGUF::GGUFFile, max_rows : Int32) : Array(ShapeRef)
  grouped = g.tensors
    .select { |t| t.type.iq4_nl? && t.dims.size == 2 }
    .group_by { |t| {t.dims[0].to_i32, t.dims[1].to_i32} }

  grouped.map do |(dims, tensors)|
    in_dim, out_dim = dims
    sample = tensors.first
    full_raw = g.read_tensor_raw(sample)
    row_bytes = ((in_dim + 31) // 32) * 18
    bench_out_dim = Math.min(out_dim, max_rows)
    bench_raw = full_raw[0, row_bytes * bench_out_dim]
    ShapeRef.new(
      name: "#{in_dim}x#{out_dim}",
      in_dim: in_dim,
      out_dim: out_dim,
      full_raw: full_raw,
      bench_raw: bench_raw,
      bench_out_dim: bench_out_dim,
      tensor_count: tensors.size,
      examples: tensors.first(4).map(&.name),
    )
  end.sort_by { |s| -(s.in_dim.to_i64 * s.out_dim.to_i64 * s.tensor_count.to_i64) }
end

def bench_shape(shape : ShapeRef, batch : Int32, warmup : Int32, runs : Int32, cpu_check_rows : Int32) : BenchRow
  x = input_for(shape.in_dim, batch)
  warmup.times do
    ML::GGUF::Qwen35Metal.matmul_iq4_nl(x, shape.bench_raw, shape.in_dim, shape.bench_out_dim, batch)
  end

  times = Array(Float64).new(runs)
  runs.times do
    t0 = Time.instant
    ML::GGUF::Qwen35Metal.matmul_iq4_nl(x, shape.bench_raw, shape.in_dim, shape.bench_out_dim, batch)
    times << (Time.instant - t0).total_milliseconds
  end
  sorted = times.sort

  cpu_subset_ms = nil
  cos = nil
  maxd = nil
  if cpu_check_rows > 0
    check_out_dim = Math.min(shape.bench_out_dim, cpu_check_rows)
    row_bytes = ((shape.in_dim + 31) // 32) * 18
    raw = shape.bench_raw[0, row_bytes * check_out_dim]
    t0 = Time.instant
    cpu = ML::GGUF::QuantMatmul.matmul_add(
      x, batch, shape.in_dim, raw, ML::GGUF::TensorType::IQ4_NL, check_out_dim,
      Array(Float32).new(check_out_dim, 0.0_f32)
    )
    cpu_subset_ms = (Time.instant - t0).total_milliseconds
    gpu = ML::GGUF::Qwen35Metal.matmul_iq4_nl(x, raw, shape.in_dim, check_out_dim, batch)
    cos = cosine(gpu, cpu)
    maxd = max_abs_diff(gpu, cpu)
  end

  BenchRow.new(
    name: shape.name,
    in_dim: shape.in_dim,
    out_dim: shape.out_dim,
    bench_out_dim: shape.bench_out_dim,
    tensor_count: shape.tensor_count,
    batch: batch,
    metal_p50_ms: percentile(sorted, 50),
    metal_min_ms: sorted.first,
    metal_max_ms: sorted.last,
    cpu_subset_ms: cpu_subset_ms,
    cos: cos,
    max_abs_diff: maxd,
    examples: shape.examples,
  )
end

model = DEFAULT_MODEL
batch = 1
warmup = 2
runs = 5
limit = 8
max_rows = Int32::MAX
cpu_check_rows = 8

OptionParser.parse do |p|
  p.banner = "Usage: qwen36_iq4_nl_op_bench [--model PATH] [--batch N] [--warmup N] [--runs N] [--limit N] [--max-rows N] [--cpu-check-rows N]"
  p.on("--model=PATH", "Qwen3.6 IQ4_NL GGUF path") { |v| model = v }
  p.on("--batch=N", "Input rows per GEMV call (default: 1)") { |v| batch = v.to_i }
  p.on("--warmup=N", "Warmup runs per shape (default: 2)") { |v| warmup = v.to_i }
  p.on("--runs=N", "Measured runs per shape (default: 5)") { |v| runs = v.to_i }
  p.on("--limit=N", "Benchmark top-N IQ4_NL shapes by dense MAC weight (default: 8, 0=all)") { |v| limit = v.to_i }
  p.on("--max-rows=N", "Benchmark only first N output rows per shape (default: full shape)") { |v| max_rows = v.to_i }
  p.on("--cpu-check-rows=N", "CPU correctness subset rows per shape (default: 8, 0=off)") { |v| cpu_check_rows = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "--batch must be positive" unless batch > 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "--runs must be positive" unless runs > 0
raise "--max-rows must be positive" unless max_rows > 0
raise "--cpu-check-rows must be non-negative" unless cpu_check_rows >= 0
raise "model not found: #{model}" unless File.exists?(model)
raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?

g = ML::GGUF::GGUFFile.new(model)
if region = g.mmap_region
  base, size = region
  ML::GGUF::Qwen35Metal.register_mmap(base, size)
end

shapes = collect_iq4_shapes(g, max_rows)
raise "no IQ4_NL 2D tensors found in #{model}" if shapes.empty?
shapes = shapes.first(limit) if limit > 0

puts "Qwen3.6 IQ4_NL operator bench"
puts "model=#{model}"
puts "batch=#{batch} warmup=#{warmup} runs=#{runs} max_rows=#{max_rows == Int32::MAX ? "full" : max_rows} cpu_check_rows=#{cpu_check_rows}"
puts "note: Metal timings include host call/readback overhead; use them for route comparison, not pure kernel wait."
puts "note: CPU check is a small correctness subset by default; full CPU reference is intentionally avoided."
puts

rows = shapes.map { |shape| bench_shape(shape, batch, warmup, runs, cpu_check_rows) }

printf "%11s %7s %8s %8s %6s %5s %10s %10s %10s %12s %12s %12s  %s\n",
  "shape", "in", "out", "bench_o", "calls", "batch", "metal_p50", "metal_min", "metal_max",
  "cpu_subset", "cos", "max_abs", "examples"
rows.each do |r|
  examples = r.examples.join(",")
  examples += ",..." if r.examples.size >= 4
  cpu = r.cpu_subset_ms ? "%.3f" % r.cpu_subset_ms.not_nil! : "-"
  cos_s = r.cos ? "%.8f" % r.cos.not_nil! : "-"
  maxd_s = r.max_abs_diff ? "%.8g" % r.max_abs_diff.not_nil! : "-"
  printf "%11s %7d %8d %8d %6d %5d %10.3f %10.3f %10.3f %12s %12s %12s  %s\n",
    r.name, r.in_dim, r.out_dim, r.bench_out_dim, r.tensor_count, r.batch,
    r.metal_p50_ms, r.metal_min_ms, r.metal_max_ms, cpu, cos_s, maxd_s, examples
end

g.close
