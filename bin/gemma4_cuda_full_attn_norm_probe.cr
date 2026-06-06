# CogniGemma CUDA full-attention projection+norm smoke.
#
# Gemma4 full-attention layers omit attn_v.weight: V reuses the pre-normalized
# K projection, then K receives learned RMSNorm while V receives plain RMSNorm.
# This probe validates that CUDA corridor before RoPE/KV-cache writes.

require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/cuda/driver"

Q4K_PTX  = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
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

def launch_q4_batched(fn : ML::CUDA::KernelFunction,
                      w_ptr : ML::CUDA::DevicePtr,
                      x_ptr : ML::CUDA::DevicePtr,
                      out_ptr : ML::CUDA::DevicePtr,
                      tokens : Int32,
                      in_dim : Int32,
                      out_dim : Int32,
                      label : String) : Nil
  d_w = w_ptr
  d_x = x_ptr
  d_out = out_ptr
  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(in_dim_u32).as(Void*)
  params[4] = pointerof(out_dim_u32).as(Void*)
  grid = (((out_dim + 3) // 4) * tokens).to_u32
  ML::CUDA.launch!(fn, grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, label)
end

model = DEFAULT_MODEL
layer = 5
seed = 23_u64
tokens = 4
reps = 20
warmup = 3

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_full_attn_norm_probe [--model PATH] [--layer N] [--tokens N] [--reps N] [--warmup N] [--seed N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layer N", "Gemma4 full-attention layer index without explicit V") { |v| layer = v.to_i }
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
raise "layer #{layer} is not a Gemma4 full-attention layer" unless hp.full_attention?(layer)
lw = weights.layers[layer]
raise "layer #{layer} unexpectedly has explicit V" if lw.attn_v_qw
head_dim = hp.head_dim_for_layer(layer)
n_head = hp.n_head
n_head_kv = hp.n_head_kv(layer)
hidden = hp.n_embd
q_dim = n_head * head_dim
k_dim = n_head_kv * head_dim
v_dim = k_dim

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

ctx = nil.as(ML::CUDA::Context?)
q4_mod = nil.as(ML::CUDA::CUDAModule?)
norm_mod = nil.as(ML::CUDA::CUDAModule?)
buffers = [] of ML::CUDA::DeviceBuffer

begin
  ctx = ML::CUDA::Context.create
  q4_mod = ML::CUDA::CUDAModule.load(Q4K_PTX, "gemma4_full_q4")
  norm_mod = ML::CUDA::CUDAModule.load(NORM_PTX, "gemma4_full_norm")
  q4_fn = q4_mod.function("q4_k_gemv_warp4_f32_batched")
  norm_fn = norm_mod.function("rmsnorm_vec_parallel_batched_probe")

  x_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(xs.size)); buffers << x_buf
  input_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_norm.size)); buffers << input_norm_buf
  x_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(xs.size)); buffers << x_norm_buf
  q_w_buf = ML::CUDA::DeviceBuffer.new(lw.attn_q_qw.raw.size.to_u64); buffers << q_w_buf
  k_w_buf = ML::CUDA::DeviceBuffer.new(lw.attn_k_qw.raw.size.to_u64); buffers << k_w_buf
  q_raw_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * q_dim)); buffers << q_raw_buf
  k_raw_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * k_dim)); buffers << k_raw_buf
  q_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_q_norm.size)); buffers << q_norm_buf
  k_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_k_norm.size)); buffers << k_norm_buf
  ones_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(head_dim)); buffers << ones_buf
  q_out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * q_dim)); buffers << q_out_buf
  k_out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * k_dim)); buffers << k_out_buf
  v_out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * v_dim)); buffers << v_out_buf
  ones = Array(Float32).new(head_dim, 1.0_f32)

  ML::CUDA.copy_htod!(x_buf.ptr, xs.to_unsafe.as(Void*), bytesize_f32(xs.size), "xs")
  ML::CUDA.copy_htod!(input_norm_buf.ptr, lw.attn_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_norm.size), "input_norm")
  ML::CUDA.copy_htod!(q_w_buf.ptr, lw.attn_q_qw.raw.to_unsafe.as(Void*), lw.attn_q_qw.raw.size.to_u64, "q_w")
  ML::CUDA.copy_htod!(k_w_buf.ptr, lw.attn_k_qw.raw.to_unsafe.as(Void*), lw.attn_k_qw.raw.size.to_u64, "k_w")
  ML::CUDA.copy_htod!(q_norm_buf.ptr, lw.attn_q_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_q_norm.size), "q_norm")
  ML::CUDA.copy_htod!(k_norm_buf.ptr, lw.attn_k_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_k_norm.size), "k_norm")
  ML::CUDA.copy_htod!(ones_buf.ptr, ones.to_unsafe.as(Void*), bytesize_f32(ones.size), "plain_v_norm")

  run_once = -> {
    launch_norm(norm_fn, x_buf.ptr, input_norm_buf.ptr, x_norm_buf.ptr, tokens, hidden, hp.rms_eps, "input_norm")
    launch_q4_batched(q4_fn, q_w_buf.ptr, x_norm_buf.ptr, q_raw_buf.ptr, tokens, hidden, q_dim, "q_proj")
    launch_q4_batched(q4_fn, k_w_buf.ptr, x_norm_buf.ptr, k_raw_buf.ptr, tokens, hidden, k_dim, "k_proj")
    launch_norm(norm_fn, q_raw_buf.ptr, q_norm_buf.ptr, q_out_buf.ptr, tokens * n_head, head_dim, hp.rms_eps, "q_head_norm")
    launch_norm(norm_fn, k_raw_buf.ptr, k_norm_buf.ptr, k_out_buf.ptr, tokens * n_head_kv, head_dim, hp.rms_eps, "k_head_norm")
    launch_norm(norm_fn, k_raw_buf.ptr, ones_buf.ptr, v_out_buf.ptr, tokens * n_head_kv, head_dim, hp.rms_eps, "v_plain_from_k")
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
  q4_mod.try(&.close)
  ctx.try(&.close)
end
