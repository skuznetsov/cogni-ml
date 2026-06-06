# Dedicated CogniGemma CUDA primitive smoke.
#
# This intentionally stays below the full-layer boundary: it proves that Gemma4
# GGUF Q4_K/Q6_K matrices can use the existing CUDA quant GEMV kernels with CPU
# reference parity. Full CogniGemma CUDA scheduling should build on this only
# after Gemma-specific attention/FFN semantics are wired.

require "option_parser"
require "../src/ml/gguf/gemma4_meta"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/cuda/driver"

Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q6K_PTX = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

DEFAULT_MODEL = ENV["GEMMA4_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

record TensorCase, name : String, type : ML::GGUF::TensorType, reps : Int32, warmup : Int32

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

def launch_gemv(fn : ML::CUDA::KernelFunction,
                d_w : ML::CUDA::DevicePtr,
                d_x : ML::CUDA::DevicePtr,
                d_out : ML::CUDA::DevicePtr,
                in_dim : Int32,
                out_dim : Int32) : Nil
  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(in_dim_u32).as(Void*)
  params[4] = pointerof(out_dim_u32).as(Void*)

  grid = ((out_dim + 3) // 4).to_u32
  ML::CUDA.launch!(fn, grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, fn.name)
end

def run_case(gguf : ML::GGUF::GGUFFile,
             tc : TensorCase,
             q4_fn : ML::CUDA::KernelFunction,
             q6_fn : ML::CUDA::KernelFunction,
             seed : UInt64) : Bool
  info = gguf.tensor(tc.name) || raise "missing tensor #{tc.name.inspect}"
  raise "#{tc.name}: expected #{tc.type.name}, got #{info.type.name}" unless info.type == tc.type
  raise "#{tc.name}: expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2

  in_dim = info.dims[0].to_i32
  out_dim = info.dims[1].to_i32
  raise "#{tc.name}: GEMV requires in_dim multiple of 256, got #{in_dim}" unless in_dim % 256 == 0

  w_raw = gguf.read_tensor_raw(info)
  rng = Random.new(seed)
  x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
  zero_bias = Array(Float32).new(out_dim, 0.0_f32)

  cpu_t0 = Time.instant
  cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, tc.type, out_dim, zero_bias)
  cpu_ms = (Time.instant - cpu_t0).total_milliseconds

  gpu_out = Array(Float32).new(out_dim, 0.0_f32)
  w_buf = ML::CUDA::DeviceBuffer.new(w_raw.size.to_u64)
  x_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(in_dim))
  out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))

  begin
    ML::CUDA.copy_htod!(w_buf.ptr, w_raw.to_unsafe.as(Void*), w_raw.size.to_u64, "#{tc.name} weights")
    ML::CUDA.copy_htod!(x_buf.ptr, x.to_unsafe.as(Void*), bytesize_f32(in_dim), "#{tc.name} input")

    fn = tc.type.q4_k? ? q4_fn : q6_fn
    tc.warmup.times { launch_gemv(fn, w_buf.ptr, x_buf.ptr, out_buf.ptr, in_dim, out_dim) }
    ML::CUDA.synchronize!("warmup #{tc.name}") if tc.warmup > 0

    gpu_t0 = Time.instant
    tc.reps.times { launch_gemv(fn, w_buf.ptr, x_buf.ptr, out_buf.ptr, in_dim, out_dim) }
    ML::CUDA.synchronize!("timed #{tc.name}")
    cuda_ms = (Time.instant - gpu_t0).total_milliseconds / tc.reps

    ML::CUDA.copy_dtoh!(gpu_out.to_unsafe.as(Void*), out_buf.ptr, bytesize_f32(out_dim), "#{tc.name} output")

    max_diff = max_abs_diff(gpu_out, cpu)
    cos = cosine(gpu_out, cpu)
    ok = cos >= 0.99999 && max_diff <= 1.0e-3_f32

    puts "case=#{tc.name}"
    puts "type=#{tc.type.name}"
    puts "shape=#{in_dim}x#{out_dim}"
    puts "raw_bytes=#{w_raw.size}"
    puts "reps=#{tc.reps}"
    puts "warmup=#{tc.warmup}"
    puts "cuda_ms=#{cuda_ms.round(3)}"
    puts "cpu_ms=#{cpu_ms.round(3)}"
    puts "cos=#{cos.round(8)}"
    puts "max_diff=#{max_diff}"
    puts "ok=#{ok}"
    puts
    ok
  ensure
    out_buf.close
    x_buf.close
    w_buf.close
  end
end

model = DEFAULT_MODEL
seed = 23_u64
reps = 20
warmup = 3
include_head = false
include_attn = false

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_primitive_probe [--model PATH] [--seed N] [--reps N] [--warmup N] [--include-attn] [--include-head]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed kernel launches per tensor") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches per tensor") { |v| warmup = v.to_i }
  p.on("--include-attn", "Also run representative SWA/full attention projection tensors") { include_attn = true }
  p.on("--include-head", "Also run the large Q6_K tied embedding/head tensor") { include_head = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

gguf = ML::GGUF::GGUFFile.new(model)
ctx = nil.as(ML::CUDA::Context?)
q4_mod = nil.as(ML::CUDA::CUDAModule?)
q6_mod = nil.as(ML::CUDA::CUDAModule?)

begin
  hp = ML::GGUF::Gemma4Hparams.new(gguf)
  raise "expected Gemma4 arch, got #{hp.arch.inspect}" unless hp.arch == "gemma4"

  ctx = ML::CUDA::Context.create
  q4_mod = ML::CUDA::CUDAModule.load(Q4K_PTX, "gemma4_q4k")
  q6_mod = ML::CUDA::CUDAModule.load(Q6K_PTX, "gemma4_q6k")
  q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
  q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "arch=#{hp.arch}"
  puts "layers=#{hp.n_layer} embd=#{hp.n_embd} ffn=#{hp.n_ff} vocab=#{hp.vocab_size}"
  puts

  cases = [
    TensorCase.new("blk.0.ffn_gate.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
    TensorCase.new("blk.0.ffn_down.weight", ML::GGUF::TensorType::Q6_K, reps, warmup),
  ]
  if include_attn
    # SWA layer 0 has explicit V. Full-attention layer 5 has no V tensor and
    # reuses K before divergent normalization, so only Q/K/output are checked.
    cases.concat([
      TensorCase.new("blk.0.attn_q.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
      TensorCase.new("blk.0.attn_k.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
      TensorCase.new("blk.0.attn_v.weight", ML::GGUF::TensorType::Q6_K, reps, warmup),
      TensorCase.new("blk.0.attn_output.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
      TensorCase.new("blk.5.attn_q.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
      TensorCase.new("blk.5.attn_k.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
      TensorCase.new("blk.5.attn_output.weight", ML::GGUF::TensorType::Q4_K, reps, warmup),
    ])
  end
  if include_head
    # This is the real tied LM-head corridor for Gemma4 Q4_K_M; keep it opt-in
    # because it uploads ~826MB of raw weights.
    cases << TensorCase.new("token_embd.weight", ML::GGUF::TensorType::Q6_K, {reps, 5}.min, {warmup, 1}.min)
  end

  ok_all = true
  cases.each_with_index do |tc, i|
    ok_all &&= run_case(gguf, tc, q4_fn, q6_fn, seed + i.to_u64)
  end
  puts "summary_ok=#{ok_all}"
  exit(ok_all ? 0 : 1)
ensure
  q6_mod.try(&.close)
  q4_mod.try(&.close)
  ctx.try(&.close)
  gguf.close
end
