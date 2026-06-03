require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/gguf/gemma4_metal"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
layer = 0
warmups = 1
runs = 5
mode = "both"

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_metal_tail_profile [--layer 0] [--runs 5] [--mode host|resident|both]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layer N", "Layer index") { |v| layer = v.to_i }
  p.on("--warmups N", "Warmup iterations") { |v| warmups = v.to_i }
  p.on("--runs N", "Measured iterations") { |v| runs = v.to_i }
  p.on("--mode MODE", "host, resident, or both") { |v| mode = v }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "runs must be positive" unless runs > 0
raise "mode must be host, resident, or both" unless {"host", "resident", "both"}.includes?(mode)

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
raise "layer out of range" if layer < 0 || layer >= weights.layers.size
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

lw = weights.layers[layer]
x = ML::GGUF::Gemma4CPU.embedding_lookup(weights.token_embd, 42)
scale = Math.sqrt(weights.hparams.n_embd.to_f64).to_f32
x.size.times { |i| x[i] *= scale }
attn_projected = Array(Float32).new(weights.hparams.n_embd) { |i| Math.sin(i.to_f32 * 0.017_f32).to_f32 * 0.125_f32 }

def percentile(sorted : Array(Float64), p : Float64) : Float64
  return 0.0 if sorted.empty?
  sorted[((sorted.size - 1).to_f64 * p).round.to_i]
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  max = 0.0_f32
  a.each_with_index do |av, i|
    diff = (av - b[i]).abs
    max = diff if diff > max
  end
  max
end

def summarize(label : String, samples : Array(Float64)) : Float64
  sorted = samples.sort
  mean = samples.sum / samples.size
  p50 = percentile(sorted, 0.50)
  p90 = percentile(sorted, 0.90)
  puts "#{label}_runs=#{samples.map { |v| v.round(3) }.join(',')}"
  puts "#{label}_mean_ms=#{mean.round(3)} #{label}_p50_ms=#{p50.round(3)} #{label}_p90_ms=#{p90.round(3)}"
  p50
end

expected = ML::GGUF::Gemma4Metal.layer_tail(x, attn_projected, lw, weights.hparams).not_nil!
actual = ML::GGUF::Gemma4Metal.layer_tail_resident_buffers(x, attn_projected, lw, weights.hparams).not_nil!
puts "model=#{File.basename(model)} layer=#{layer} warmups=#{warmups} runs=#{runs} mode=#{mode} parity_max_abs=#{max_abs_diff(expected, actual)}"

host_samples = [] of Float64
resident_samples = [] of Float64

if mode == "host" || mode == "both"
  warmups.times { ML::GGUF::Gemma4Metal.layer_tail(x, attn_projected, lw, weights.hparams).not_nil! }
  runs.times do
    t0 = Time.instant
    ML::GGUF::Gemma4Metal.layer_tail(x, attn_projected, lw, weights.hparams).not_nil!
    host_samples << (Time.instant - t0).total_milliseconds
  end
  summarize("host", host_samples)
end

if mode == "resident" || mode == "both"
  warmups.times { ML::GGUF::Gemma4Metal.layer_tail_resident_buffers(x, attn_projected, lw, weights.hparams).not_nil! }
  runs.times do
    t0 = Time.instant
    ML::GGUF::Gemma4Metal.layer_tail_resident_buffers(x, attn_projected, lw, weights.hparams).not_nil!
    resident_samples << (Time.instant - t0).total_milliseconds
  end
  summarize("resident", resident_samples)
end

if mode == "both"
  puts "resident_vs_host_p50_speedup=#{(percentile(host_samples.sort, 0.50) / percentile(resident_samples.sort, 0.50)).round(4)}"
end
