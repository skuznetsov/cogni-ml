require "option_parser"
require "../src/ml/gguf/gemma4_weights"
require "../src/ml/gguf/qwen35_metal"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
layer = 0
batch = 16
runs = 7
warmups = 1

OptionParser.parse(ARGV) do |p|
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layer N", "Layer index") { |v| layer = v.to_i }
  p.on("--batch N", "Batch size") { |v| batch = v.to_i }
  p.on("--runs N", "Measured runs") { |v| runs = v.to_i }
  p.on("--warmups N", "Warmup runs") { |v| warmups = v.to_i }
end

def percentile(xs : Array(Float64), p : Float64) : Float64
  sorted = xs.sort
  sorted[((sorted.size - 1).to_f64 * p).round.to_i]
end

def max_abs(a : Array(Float32), b : Array(Float32)) : Float64
  m = 0.0
  a.each_with_index do |x, i|
    d = (x - b[i]).abs.to_f64
    m = d if d > m
  end
  m
end

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
lw = weights.layers[layer]
hidden = weights.hparams.n_embd
ffn = lw.ffn_gate_qw.out_dim
raise "batch must be positive" unless batch > 0

x = Array(Float32).new(batch * hidden) { |i| Math.sin(i.to_f64 * 0.013).to_f32 * 0.1_f32 }
x_buf = ML::MetalBuffer.from_array(x)

def run_serial(lw, x, batch, hidden, ffn)
  gate = Array(Float32).new(batch * ffn, 0.0_f32)
  up = Array(Float32).new(batch * ffn, 0.0_f32)
  batch.times do |r|
    row = x[r * hidden, hidden]
    outs = ML::GGUF::Qwen35Metal.matmul_many([lw.ffn_gate_qw, lw.ffn_up_qw], row).not_nil!
    gate[(r * ffn), ffn] = outs[0]
    up[(r * ffn), ffn] = outs[1]
  end
  {gate, up}
end

def run_batch(lw, x_buf, batch, ffn)
  gate_buf = ML::MetalBuffer.new(batch.to_i64 * ffn * sizeof(Float32))
  up_buf = ML::MetalBuffer.new(batch.to_i64 * ffn * sizeof(Float32))
  ok = ML::GGUF::Qwen35Metal.matmul_many_to_buffers([lw.ffn_gate_qw, lw.ffn_up_qw], x_buf, [gate_buf, up_buf], batch)
  raise "batch matmul failed" unless ok
  {gate_buf.read(batch * ffn), up_buf.read(batch * ffn)}
end

serial_ref = run_serial(lw, x, batch, hidden, ffn)
batch_ref = run_batch(lw, x_buf, batch, ffn)
puts "model=#{File.basename(model)} layer=#{layer} batch=#{batch} hidden=#{hidden} ffn=#{ffn}"
puts "gate_max_abs=#{max_abs(serial_ref[0], batch_ref[0])} up_max_abs=#{max_abs(serial_ref[1], batch_ref[1])}"

serial_times = [] of Float64
batch_times = [] of Float64
warmups.times do
  run_serial(lw, x, batch, hidden, ffn)
  run_batch(lw, x_buf, batch, ffn)
end
runs.times do
  t0 = Time.instant
  run_serial(lw, x, batch, hidden, ffn)
  serial_times << (Time.instant - t0).total_milliseconds
  t1 = Time.instant
  run_batch(lw, x_buf, batch, ffn)
  batch_times << (Time.instant - t1).total_milliseconds
end
sp = percentile(serial_times, 0.5)
bp = percentile(batch_times, 0.5)
puts "serial_runs=#{serial_times.map { |v| v.round(3) }.join(',')}"
puts "batch_runs=#{batch_times.map { |v| v.round(3) }.join(',')}"
puts "serial_p50_ms=#{sp.round(3)} batch_p50_ms=#{bp.round(3)} speedup=#{(sp / bp).round(4)}"
