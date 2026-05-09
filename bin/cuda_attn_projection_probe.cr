# GPU-resident CUDA full-attention projection probe for Qwen GGUF weights.
#
# Runs: Q4_K attn_q GEMV + Q4_K attn_k GEMV + Q4_K/Q6_K attn_v GEMV from
# the same hidden vector sequence. Outputs are copied back only after all
# projections finish, then compared against CPU QuantMatmul references.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/cuda/qwen_full_attn_projection_runner"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise ArgumentError.new("size mismatch") unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |v, i|
    diff = (v - b[i]).abs
    max = diff if diff > max
  end
  max
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.each_with_index do |v, i|
    av = v.to_f64
    bv = b[i].to_f64
    dot += av * bv
    na += av * av
    nb += bv * bv
  end
  dot / Math.sqrt(na * nb)
end

def append_stats(name : String, gpu : Array(Float32), cpu : Array(Float32), lines : Array(String)) : Bool
  cos = cosine(gpu, cpu)
  max_diff = max_abs_diff(gpu, cpu)
  ok = cos >= 0.99999 && max_diff <= 1.0e-3_f32
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layer = 3
seed = 23_u64
reps = 1
warmup = 0
tokens = 1

OptionParser.parse do |p|
  p.banner = "Usage: cuda_attn_projection_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N] [--tokens N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Full-attention layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed projection-bundle launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup projection-bundle launches") { |v| warmup = v.to_i }
  p.on("--tokens N", "GPU-resident sequence length") { |v| tokens = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "tokens must be positive" unless tokens > 0

gguf = ML::GGUF::GGUFFile.new(model)
weights = ML::CUDA::QwenFullAttnProjectionRunner::Weights.load(gguf, layer)
hidden = weights.hidden
q_dim = weights.q_dim
k_dim = weights.k_dim
v_dim = weights.v_dim

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }
q_cpu_all = Array(Float32).new(tokens * q_dim, 0.0_f32)
k_cpu_all = Array(Float32).new(tokens * k_dim, 0.0_f32)
v_cpu_all = Array(Float32).new(tokens * v_dim, 0.0_f32)

cpu_t0 = Time.instant
tokens.times do |tok|
  x = xs[tok * hidden, hidden]
  q_cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, weights.q_raw, ML::GGUF::TensorType::Q4_K, q_dim, Array(Float32).new(q_dim, 0.0_f32))
  k_cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, weights.k_raw, ML::GGUF::TensorType::Q4_K, k_dim, Array(Float32).new(k_dim, 0.0_f32))
  v_cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, weights.v_raw, weights.v_type, v_dim, Array(Float32).new(v_dim, 0.0_f32))
  q_dim.times { |i| q_cpu_all[tok * q_dim + i] = q_cpu[i] }
  k_dim.times { |i| k_cpu_all[tok * k_dim + i] = k_cpu[i] }
  v_dim.times { |i| v_cpu_all[tok * v_dim + i] = v_cpu[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda_ctx = nil.as(ML::CUDA::Context?)
runner = nil.as(ML::CUDA::QwenFullAttnProjectionRunner?)

begin
  cuda_ctx = ML::CUDA::Context.create
  runner = ML::CUDA::QwenFullAttnProjectionRunner.from_weights(weights, tokens, xs)

  upload_t0 = Time.instant
  runner.upload_weights
  ML::CUDA.synchronize!("cuCtxSynchronize(upload_weights)")
  weight_upload_ms = (Time.instant - upload_t0).total_milliseconds
  runner.reset_sequence

  warmup.times { runner.run_sequence }
  ML::CUDA.synchronize!("cuCtxSynchronize(warmup)") if warmup > 0
  runner.reset_sequence

  gpu_t0 = Time.instant
  reps.times { runner.run_sequence }
  ML::CUDA.synchronize!("cuCtxSynchronize")
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps

  runner.read_outputs

  lines = [] of String
  q_ok = append_stats("q", runner.q_gpu_all, q_cpu_all, lines)
  k_ok = append_stats("k", runner.k_gpu_all, k_cpu_all, lines)
  v_ok = append_stats("v", runner.v_gpu_all, v_cpu_all, lines)
  ok = q_ok && k_ok && v_ok

  puts "device=#{cuda_ctx.device_name}"
  puts "compute_capability=#{cuda_ctx.compute_capability_major}.#{cuda_ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "hidden=#{hidden}"
  puts "q_dim=#{q_dim}"
  puts "k_dim=#{k_dim}"
  puts "v_dim=#{v_dim}"
  puts "tokens=#{tokens}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "weight_upload_ms=#{weight_upload_ms.round(3)}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cuda_ms_per_token=#{(gpu_ms / tokens).round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cpu_ms_per_token=#{(cpu_ms / tokens).round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  runner.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
