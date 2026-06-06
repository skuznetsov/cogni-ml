# CogniGemma CUDA RMSNorm primitive smoke.
#
# Proves the existing CUDA RMSNorm kernel can cover Gemma4 full-vector norms,
# per-head Q/K norms, and plain V norms represented with an all-ones weight.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/gemma4_meta"
require "../src/ml/cuda/driver"

NORM_PTX = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}

DEFAULT_MODEL = ENV["GEMMA4_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

record NormCase, name : String, dim : Int32, norm_tensor : String?, tokens : Int32

def bytesize_f32(elements : Int32) : LibC::SizeT
  (elements * sizeof(Float32)).to_u64
end

def rms_norm_cpu(x : Array(Float32), norm : Array(Float32), tokens : Int32, dim : Int32, eps : Float32) : Array(Float32)
  out = Array(Float32).new(tokens * dim, 0.0_f32)
  tokens.times do |tok|
    base = tok * dim
    ss = 0.0_f64
    dim.times do |i|
      v = x[base + i]
      ss += v.to_f64 * v.to_f64
    end
    inv = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
    dim.times { |i| out[base + i] = x[base + i] * inv * norm[i] }
  end
  out
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

def run_case(gguf : ML::GGUF::GGUFFile,
             tc : NormCase,
             fn : ML::CUDA::KernelFunction,
             rng : Random,
             eps : Float32,
             reps : Int32,
             warmup : Int32) : Bool
  norm = if tensor = tc.norm_tensor
           info = gguf.tensor(tensor) || raise "missing norm tensor #{tensor.inspect}"
           values = gguf.read_tensor_f32(info)
           raise "#{tc.name}: norm size mismatch #{values.size} != #{tc.dim}" unless values.size == tc.dim
           values
         else
           Array(Float32).new(tc.dim, 1.0_f32)
         end
  x = Array(Float32).new(tc.tokens * tc.dim) { rng.rand(-2.0_f32..2.0_f32) }
  cpu = rms_norm_cpu(x, norm, tc.tokens, tc.dim, eps)
  gpu = Array(Float32).new(cpu.size, 0.0_f32)

  x_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(x.size))
  norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(norm.size))
  out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(gpu.size))

  begin
    ML::CUDA.copy_htod!(x_buf.ptr, x.to_unsafe.as(Void*), bytesize_f32(x.size), "#{tc.name} x")
    ML::CUDA.copy_htod!(norm_buf.ptr, norm.to_unsafe.as(Void*), bytesize_f32(norm.size), "#{tc.name} norm")

    dim_u32 = tc.dim.to_u32
    eps_f32 = eps
    d_x = x_buf.ptr
    d_norm = norm_buf.ptr
    d_out = out_buf.ptr
    params = Pointer(Void*).malloc(5)
    params[0] = pointerof(d_x).as(Void*)
    params[1] = pointerof(d_norm).as(Void*)
    params[2] = pointerof(d_out).as(Void*)
    params[3] = pointerof(dim_u32).as(Void*)
    params[4] = pointerof(eps_f32).as(Void*)

    warmup.times do
      ML::CUDA.launch!(fn, tc.tokens.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, "#{tc.name} warmup")
    end
    ML::CUDA.synchronize!("warmup #{tc.name}") if warmup > 0

    t0 = Time.instant
    reps.times do
      ML::CUDA.launch!(fn, tc.tokens.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, tc.name)
    end
    ML::CUDA.synchronize!("timed #{tc.name}")
    cuda_ms = (Time.instant - t0).total_milliseconds / reps

    ML::CUDA.copy_dtoh!(gpu.to_unsafe.as(Void*), out_buf.ptr, bytesize_f32(gpu.size), "#{tc.name} out")

    cos = cosine(gpu, cpu)
    diff = max_abs_diff(gpu, cpu)
    ok = cos >= 0.999999 && diff <= 1.0e-4_f32
    puts "case=#{tc.name}"
    puts "dim=#{tc.dim}"
    puts "tokens=#{tc.tokens}"
    puts "norm_tensor=#{tc.norm_tensor || "plain_ones"}"
    puts "cuda_ms=#{cuda_ms.round(4)}"
    puts "cuda_ms_per_token=#{(cuda_ms / tc.tokens).round(4)}"
    puts "cos=#{cos.round(8)}"
    puts "max_diff=#{diff}"
    puts "ok=#{ok}"
    puts
    ok
  ensure
    out_buf.close
    norm_buf.close
    x_buf.close
  end
end

model = DEFAULT_MODEL
seed = 23_u64
reps = 20
warmup = 3
tokens = 4

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_norm_probe [--model PATH] [--seed N] [--reps N] [--warmup N] [--tokens N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed launches per case") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches per case") { |v| warmup = v.to_i }
  p.on("--tokens N", "Rows per batched norm case") { |v| tokens = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "tokens must be positive" unless tokens > 0

gguf = ML::GGUF::GGUFFile.new(model)
ctx = nil.as(ML::CUDA::Context?)
mod = nil.as(ML::CUDA::CUDAModule?)

begin
  hp = ML::GGUF::Gemma4Hparams.new(gguf)
  raise "expected gemma4 arch, got #{hp.arch.inspect}" unless hp.arch == "gemma4"

  ctx = ML::CUDA::Context.create
  mod = ML::CUDA::CUDAModule.load(NORM_PTX, "gemma4_norm")
  fn = mod.function("rmsnorm_vec_parallel_batched_probe")
  rng = Random.new(seed)

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "eps=#{hp.rms_eps}"
  puts

  cases = [
    NormCase.new("full_attn_input_norm", hp.n_embd, "blk.0.attn_norm.weight", tokens),
    NormCase.new("swa_q_head_norm", hp.head_dim_swa, "blk.0.attn_q_norm.weight", tokens),
    NormCase.new("swa_v_plain_norm", hp.head_dim_swa, nil, tokens),
    NormCase.new("full_q_head_norm", hp.head_dim, "blk.5.attn_q_norm.weight", tokens),
    NormCase.new("full_v_plain_norm", hp.head_dim, nil, tokens),
    NormCase.new("output_norm", hp.n_embd, "output_norm.weight", tokens),
  ]
  ok_all = true
  cases.each do |tc|
    ok = run_case(gguf, tc, fn, rng, hp.rms_eps, reps, warmup)
    ok_all &&= ok
  end
  puts "summary_ok=#{ok_all}"
  exit(ok_all ? 0 : 1)
ensure
  mod.try(&.close)
  ctx.try(&.close)
  gguf.close
end
