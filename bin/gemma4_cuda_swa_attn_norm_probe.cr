# CogniGemma CUDA SWA attention projection+norm smoke.
#
# This combines input RMSNorm, Q/K/V GEMV projections, and per-head Q/K/V
# normalization for a Gemma4 SWA layer with explicit V. It intentionally stops
# before RoPE and KV-cache writes.

require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/cuda/qwen_full_attn_projection_runner"

NORM_PTX = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}

DEFAULT_MODEL = ENV["GEMMA4_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

def bytesize_f32(elements : Int32) : LibC::SizeT
  (elements * sizeof(Float32)).to_u64
end

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
  ok = cos >= 0.999999 && max_diff <= 1.0e-4_f32
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

def launch_norm(fn : ML::CUDA::KernelFunction,
                x_ptr : ML::CUDA::DevicePtr,
                norm_ptr : ML::CUDA::DevicePtr,
                out_ptr : ML::CUDA::DevicePtr,
                rows : Int32,
                dim : Int32,
                eps : Float32,
                label : String) : Nil
  d_x = x_ptr
  d_norm = norm_ptr
  d_out = out_ptr
  dim_u32 = dim.to_u32
  eps_f32 = eps
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_x).as(Void*)
  params[1] = pointerof(d_norm).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(dim_u32).as(Void*)
  params[4] = pointerof(eps_f32).as(Void*)
  ML::CUDA.launch!(fn, rows.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
end

model = DEFAULT_MODEL
layer = 0
seed = 23_u64
tokens = 4
reps = 20
warmup = 3

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_swa_attn_norm_probe [--model PATH] [--layer N] [--tokens N] [--reps N] [--warmup N] [--seed N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layer N", "Gemma4 SWA layer index with explicit V") { |v| layer = v.to_i }
  p.on("--tokens N", "Rows in the known-span projection bundle") { |v| tokens = v.to_i }
  p.on("--reps N", "Timed projection+norm launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokens must be positive" unless tokens > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
hp = weights.hparams
raise "layer #{layer} is not a Gemma4 SWA layer" unless hp.sliding_window?(layer)
lw = weights.layers[layer]
v_qw = lw.attn_v_qw || raise "layer #{layer} has no explicit V; use a full-layer K-as-V probe"
head_dim = hp.head_dim_for_layer(layer)
n_head = hp.n_head
n_head_kv = hp.n_head_kv(layer)
hidden = hp.n_embd
q_dim = n_head * head_dim
k_dim = n_head_kv * head_dim
v_dim = n_head_kv * head_dim

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }
q_cpu = Array(Float32).new(tokens * q_dim, 0.0_f32)
k_cpu = Array(Float32).new(tokens * k_dim, 0.0_f32)
v_cpu = Array(Float32).new(tokens * v_dim, 0.0_f32)

cpu_t0 = Time.instant
tokens.times do |tok|
  row = xs[tok * hidden, hidden]
  proj = ML::GGUF::Gemma4CPU.attention_project_normed(lw, row, hp, layer)
  q_dim.times { |i| q_cpu[tok * q_dim + i] = proj.q[i] }
  k_dim.times { |i| k_cpu[tok * k_dim + i] = proj.k[i] }
  v_dim.times { |i| v_cpu[tok * v_dim + i] = proj.v[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

proj_weights = ML::CUDA::QwenFullAttnProjectionRunner::Weights.new(
  hidden, q_dim, k_dim, v_dim,
  lw.attn_q_qw.raw, lw.attn_k_qw.raw, v_qw.raw, v_qw.type)

ctx = nil.as(ML::CUDA::Context?)
runner = nil.as(ML::CUDA::QwenFullAttnProjectionRunner?)
norm_mod = nil.as(ML::CUDA::CUDAModule?)
buffers = [] of ML::CUDA::DeviceBuffer

begin
  ctx = ML::CUDA::Context.create
  runner = ML::CUDA::QwenFullAttnProjectionRunner.from_weights_with_input_norm(proj_weights, tokens, xs, lw.attn_norm, hp.rms_eps)
  norm_mod = ML::CUDA::CUDAModule.load(NORM_PTX, "gemma4_swa_attn_norm")
  norm_fn = norm_mod.function("rmsnorm_vec_parallel_batched_probe")

  q_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_q_norm.size)); buffers << q_norm_buf
  k_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_k_norm.size)); buffers << k_norm_buf
  ones_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(head_dim)); buffers << ones_buf
  q_out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * q_dim)); buffers << q_out_buf
  k_out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * k_dim)); buffers << k_out_buf
  v_out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * v_dim)); buffers << v_out_buf
  ones = Array(Float32).new(head_dim, 1.0_f32)

  ML::CUDA.copy_htod!(q_norm_buf.ptr, lw.attn_q_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_q_norm.size), "q_norm")
  ML::CUDA.copy_htod!(k_norm_buf.ptr, lw.attn_k_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_k_norm.size), "k_norm")
  ML::CUDA.copy_htod!(ones_buf.ptr, ones.to_unsafe.as(Void*), bytesize_f32(ones.size), "plain_v_norm")
  runner.upload_weights
  runner.reset_sequence

  run_once = -> {
    runner.not_nil!.run_sequence
    launch_norm(norm_fn, runner.not_nil!.q_device_ptr, q_norm_buf.ptr, q_out_buf.ptr, tokens * n_head, head_dim, hp.rms_eps, "q_head_norm")
    launch_norm(norm_fn, runner.not_nil!.k_device_ptr, k_norm_buf.ptr, k_out_buf.ptr, tokens * n_head_kv, head_dim, hp.rms_eps, "k_head_norm")
    launch_norm(norm_fn, runner.not_nil!.v_device_ptr, ones_buf.ptr, v_out_buf.ptr, tokens * n_head_kv, head_dim, hp.rms_eps, "v_plain_norm")
  }

  warmup.times { run_once.call }
  ML::CUDA.synchronize!("warmup")

  t0 = Time.instant
  reps.times { run_once.call }
  ML::CUDA.synchronize!("timed")
  cuda_ms = (Time.instant - t0).total_milliseconds / reps

  q_gpu = Array(Float32).new(tokens * q_dim, 0.0_f32)
  k_gpu = Array(Float32).new(tokens * k_dim, 0.0_f32)
  v_gpu = Array(Float32).new(tokens * v_dim, 0.0_f32)
  ML::CUDA.copy_dtoh!(q_gpu.to_unsafe.as(Void*), q_out_buf.ptr, bytesize_f32(q_gpu.size), "q_normed")
  ML::CUDA.copy_dtoh!(k_gpu.to_unsafe.as(Void*), k_out_buf.ptr, bytesize_f32(k_gpu.size), "k_normed")
  ML::CUDA.copy_dtoh!(v_gpu.to_unsafe.as(Void*), v_out_buf.ptr, bytesize_f32(v_gpu.size), "v_normed")

  lines = [] of String
  q_ok = append_stats("q", q_gpu, q_cpu, lines)
  k_ok = append_stats("k", k_gpu, k_cpu, lines)
  v_ok = append_stats("v", v_gpu, v_cpu, lines)
  ok = q_ok && k_ok && v_ok

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "tokens=#{tokens}"
  puts "hidden=#{hidden}"
  puts "head_dim=#{head_dim}"
  puts "n_head=#{n_head}"
  puts "n_head_kv=#{n_head_kv}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{cuda_ms.round(4)}"
  puts "cuda_ms_per_token=#{(cuda_ms / tokens).round(4)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
  exit(ok ? 0 : 1)
ensure
  buffers.each(&.close)
  norm_mod.try(&.close)
  runner.try(&.close)
  ctx.try(&.close)
end
