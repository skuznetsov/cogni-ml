# GPU-resident CUDA recurrent-layer projection bundle probe for Qwen GGUF weights.
#
# Runs Q5_K attn_qkv plus Q4_K attn_gate/ssm_alpha/ssm_beta from the same
# hidden vector. Outputs are copied back only after all projections finish.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDARecProj
  alias CUdevice = Int32
  alias CUcontext = Void*
  alias CUmodule = Void*
  alias CUfunction = Void*
  alias CUdeviceptr = UInt64

  fun cuInit(flags : UInt32) : Int32
  fun cuDeviceGet(device : CUdevice*, ordinal : Int32) : Int32
  fun cuDeviceGetName(name : UInt8*, len : Int32, dev : CUdevice) : Int32
  fun cuDeviceComputeCapability(major : Int32*, minor : Int32*, dev : CUdevice) : Int32
  fun cuCtxCreate_v2(ctx : CUcontext*, flags : UInt32, dev : CUdevice) : Int32
  fun cuCtxDestroy_v2(ctx : CUcontext) : Int32
  fun cuModuleLoadData(mod : CUmodule*, image : Void*) : Int32
  fun cuModuleUnload(mod : CUmodule) : Int32
  fun cuModuleGetFunction(fn : CUfunction*, mod : CUmodule, name : UInt8*) : Int32
  fun cuMemAlloc_v2(dptr : CUdeviceptr*, bytesize : LibC::SizeT) : Int32
  fun cuMemFree_v2(dptr : CUdeviceptr) : Int32
  fun cuMemcpyHtoD_v2(dst : CUdeviceptr, src : Void*, bytesize : LibC::SizeT) : Int32
  fun cuMemcpyDtoH_v2(dst : Void*, src : CUdeviceptr, bytesize : LibC::SizeT) : Int32
  fun cuLaunchKernel(fn : CUfunction, grid_x : UInt32, grid_y : UInt32, grid_z : UInt32,
                     block_x : UInt32, block_y : UInt32, block_z : UInt32,
                     shared_mem_bytes : UInt32, stream : Void*,
                     kernel_params : Void**, extra : Void**) : Int32
  fun cuCtxSynchronize : Int32
end

Q4K_PTX       = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q4K_DUAL_PTX  = {{ read_file("src/ml/cuda/kernels/q4k_dual_gemv_probe.ptx") }}
Q5K_PTX       = {{ read_file("src/ml/cuda/kernels/q5k_gemv_probe.ptx") }}
DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def cuda!(code : Int32, what : String) : Nil
  raise "#{what} failed with CUDA error #{code}" unless code == 0
end

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
  ok = cos >= 0.99999 && max_diff <= 1.0e-3_f32
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layer = 0
seed = 23_u64
reps = 1
warmup = 0
tokens = 1
batched = false
batched_dual_alpha_beta = false

OptionParser.parse do |p|
  p.banner = "Usage: cuda_recurrent_projection_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N] [--tokens N] [--batched] [--batched-dual-alpha-beta]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Recurrent layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed projection-bundle launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup projection-bundle launches") { |v| warmup = v.to_i }
  p.on("--tokens N", "Number of independent projection rows to run") { |v| tokens = v.to_i }
  p.on("--batched", "Use known-span batched Q4/Q5 projection kernels") { batched = true }
  p.on("--batched-dual-alpha-beta", "Use one batched Q4 dual kernel for ssm_alpha and ssm_beta projections") { batched_dual_alpha_beta = true; batched = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "tokens must be positive" unless tokens > 0

gguf = ML::GGUF::GGUFFile.new(model)
prefix = "blk.#{layer}"
qkv_info = gguf.tensor("#{prefix}.attn_qkv.weight") || raise "missing #{prefix}.attn_qkv.weight"
gate_info = gguf.tensor("#{prefix}.attn_gate.weight") || raise "missing #{prefix}.attn_gate.weight"
alpha_info = gguf.tensor("#{prefix}.ssm_alpha.weight") || raise "missing #{prefix}.ssm_alpha.weight"
beta_info = gguf.tensor("#{prefix}.ssm_beta.weight") || raise "missing #{prefix}.ssm_beta.weight"
raise "expected Q5_K attn_qkv" unless qkv_info.type.q5_k?
raise "expected Q4_K gate/alpha/beta" unless gate_info.type.q4_k? && alpha_info.type.q4_k? && beta_info.type.q4_k?

hidden = qkv_info.dims[0].to_i32
qkv_dim = qkv_info.dims[1].to_i32
gate_dim = gate_info.dims[1].to_i32
alpha_dim = alpha_info.dims[1].to_i32
beta_dim = beta_info.dims[1].to_i32
raise "input shape mismatch" unless [gate_info, alpha_info, beta_info].all? { |i| i.dims[0].to_i32 == hidden }
raise "--batched-dual-alpha-beta requires equal alpha/beta output dims" if batched_dual_alpha_beta && alpha_info.dims[1] != beta_info.dims[1]

qkv_raw = gguf.read_tensor_raw(qkv_info)
gate_raw = gguf.read_tensor_raw(gate_info)
alpha_raw = gguf.read_tensor_raw(alpha_info)
beta_raw = gguf.read_tensor_raw(beta_info)
rng = Random.new(seed)
x = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }

cpu_t0 = Time.instant
qkv_cpu = Array(Float32).new(tokens * qkv_dim, 0.0_f32)
gate_cpu = Array(Float32).new(tokens * gate_dim, 0.0_f32)
alpha_cpu = Array(Float32).new(tokens * alpha_dim, 0.0_f32)
beta_cpu = Array(Float32).new(tokens * beta_dim, 0.0_f32)
qkv_zero = Array(Float32).new(qkv_dim, 0.0_f32)
gate_zero = Array(Float32).new(gate_dim, 0.0_f32)
alpha_zero = Array(Float32).new(alpha_dim, 0.0_f32)
beta_zero = Array(Float32).new(beta_dim, 0.0_f32)
tokens.times do |tok|
  row = x[tok * hidden, hidden]
  qkv_row = ML::GGUF::QuantMatmul.matmul_add(row, 1, hidden, qkv_raw, ML::GGUF::TensorType::Q5_K, qkv_dim, qkv_zero)
  gate_row = ML::GGUF::QuantMatmul.matmul_add(row, 1, hidden, gate_raw, ML::GGUF::TensorType::Q4_K, gate_dim, gate_zero)
  alpha_row = ML::GGUF::QuantMatmul.matmul_add(row, 1, hidden, alpha_raw, ML::GGUF::TensorType::Q4_K, alpha_dim, alpha_zero)
  beta_row = ML::GGUF::QuantMatmul.matmul_add(row, 1, hidden, beta_raw, ML::GGUF::TensorType::Q4_K, beta_dim, beta_zero)
  qkv_dim.times { |i| qkv_cpu[tok * qkv_dim + i] = qkv_row[i] }
  gate_dim.times { |i| gate_cpu[tok * gate_dim + i] = gate_row[i] }
  alpha_dim.times { |i| alpha_cpu[tok * alpha_dim + i] = alpha_row[i] }
  beta_dim.times { |i| beta_cpu[tok * beta_dim + i] = beta_row[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDARecProj.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDARecProj::CUdevice
cuda! LibCUDARecProj.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"
name_buf = Bytes.new(256)
cuda! LibCUDARecProj.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDARecProj.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"
ctx = Pointer(Void).null
cuda! LibCUDARecProj.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

q4_mod = Pointer(Void).null
q4_dual_mod = Pointer(Void).null
q5_mod = Pointer(Void).null
q4_fn = Pointer(Void).null
q4_dual_batched_fn = Pointer(Void).null
q5_fn = Pointer(Void).null
q4_batched_fn = Pointer(Void).null
q5_batched_fn = Pointer(Void).null
ptrs = [] of UInt64

begin
  cuda! LibCUDARecProj.cuModuleLoadData(pointerof(q4_mod), Q4K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q4)"
  cuda! LibCUDARecProj.cuModuleLoadData(pointerof(q4_dual_mod), Q4K_DUAL_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q4 dual)"
  cuda! LibCUDARecProj.cuModuleLoadData(pointerof(q5_mod), Q5K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q5)"
  cuda! LibCUDARecProj.cuModuleGetFunction(pointerof(q4_fn), q4_mod, "q4_k_gemv_warp4_f32"), "cuModuleGetFunction(q4)"
  cuda! LibCUDARecProj.cuModuleGetFunction(pointerof(q4_dual_batched_fn), q4_dual_mod, "q4_k_dual_gemv_warp4_f32_batched"), "cuModuleGetFunction(q4 dual batched)"
  cuda! LibCUDARecProj.cuModuleGetFunction(pointerof(q5_fn), q5_mod, "q5_k_gemv_warp4_f32"), "cuModuleGetFunction(q5)"
  cuda! LibCUDARecProj.cuModuleGetFunction(pointerof(q4_batched_fn), q4_mod, "q4_k_gemv_warp4_f32_batched"), "cuModuleGetFunction(q4 batched)"
  cuda! LibCUDARecProj.cuModuleGetFunction(pointerof(q5_batched_fn), q5_mod, "q5_k_gemv_warp4_f32_batched"), "cuModuleGetFunction(q5 batched)"

  sizes = [bytesize_f32(tokens * hidden), qkv_raw.size.to_u64, gate_raw.size.to_u64, alpha_raw.size.to_u64, beta_raw.size.to_u64,
           bytesize_f32(tokens * qkv_dim), bytesize_f32(tokens * gate_dim), bytesize_f32(tokens * alpha_dim), bytesize_f32(tokens * beta_dim)]
  sizes.each_with_index do |size, i|
    pdev = 0_u64
    cuda! LibCUDARecProj.cuMemAlloc_v2(pointerof(pdev), size), "cuMemAlloc(#{i})"
    ptrs << pdev
  end
  d_x, d_qkv_w, d_gate_w, d_alpha_w, d_beta_w, d_qkv, d_gate, d_alpha, d_beta = ptrs

  cuda! LibCUDARecProj.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), bytesize_f32(tokens * hidden)), "cuMemcpyHtoD(x)"
  cuda! LibCUDARecProj.cuMemcpyHtoD_v2(d_qkv_w, qkv_raw.to_unsafe.as(Void*), qkv_raw.size.to_u64), "cuMemcpyHtoD(qkv_w)"
  cuda! LibCUDARecProj.cuMemcpyHtoD_v2(d_gate_w, gate_raw.to_unsafe.as(Void*), gate_raw.size.to_u64), "cuMemcpyHtoD(gate_w)"
  cuda! LibCUDARecProj.cuMemcpyHtoD_v2(d_alpha_w, alpha_raw.to_unsafe.as(Void*), alpha_raw.size.to_u64), "cuMemcpyHtoD(alpha_w)"
  cuda! LibCUDARecProj.cuMemcpyHtoD_v2(d_beta_w, beta_raw.to_unsafe.as(Void*), beta_raw.size.to_u64), "cuMemcpyHtoD(beta_w)"

  hidden_u32 = hidden.to_u32
  qkv_dim_u32 = qkv_dim.to_u32
  gate_dim_u32 = gate_dim.to_u32
  alpha_dim_u32 = alpha_dim.to_u32
  beta_dim_u32 = beta_dim.to_u32
  qkv_grid = ((qkv_dim + 3) // 4).to_u32
  gate_grid = ((gate_dim + 3) // 4).to_u32
  alpha_grid = ((alpha_dim + 3) // 4).to_u32
  beta_grid = ((beta_dim + 3) // 4).to_u32
  alpha_beta_dual_grid = (((alpha_dim * 2) + 3) // 4).to_u32
  tokens_u32 = tokens.to_u32
  d_x_cur = d_x
  d_qkv_cur = d_qkv
  d_gate_cur = d_gate
  d_alpha_cur = d_alpha
  d_beta_cur = d_beta

  qkv_params = Pointer(Void*).malloc(5)
  qkv_params[0] = pointerof(d_qkv_w).as(Void*)
  qkv_params[1] = pointerof(d_x_cur).as(Void*)
  qkv_params[2] = pointerof(d_qkv_cur).as(Void*)
  qkv_params[3] = pointerof(hidden_u32).as(Void*)
  qkv_params[4] = pointerof(qkv_dim_u32).as(Void*)

  gate_params = Pointer(Void*).malloc(5)
  gate_params[0] = pointerof(d_gate_w).as(Void*)
  gate_params[1] = pointerof(d_x_cur).as(Void*)
  gate_params[2] = pointerof(d_gate_cur).as(Void*)
  gate_params[3] = pointerof(hidden_u32).as(Void*)
  gate_params[4] = pointerof(gate_dim_u32).as(Void*)

  alpha_params = Pointer(Void*).malloc(5)
  alpha_params[0] = pointerof(d_alpha_w).as(Void*)
  alpha_params[1] = pointerof(d_x_cur).as(Void*)
  alpha_params[2] = pointerof(d_alpha_cur).as(Void*)
  alpha_params[3] = pointerof(hidden_u32).as(Void*)
  alpha_params[4] = pointerof(alpha_dim_u32).as(Void*)

  beta_params = Pointer(Void*).malloc(5)
  beta_params[0] = pointerof(d_beta_w).as(Void*)
  beta_params[1] = pointerof(d_x_cur).as(Void*)
  beta_params[2] = pointerof(d_beta_cur).as(Void*)
  beta_params[3] = pointerof(hidden_u32).as(Void*)
  beta_params[4] = pointerof(beta_dim_u32).as(Void*)

  alpha_beta_dual_params = Pointer(Void*).malloc(7)
  alpha_beta_dual_params[0] = pointerof(d_alpha_w).as(Void*)
  alpha_beta_dual_params[1] = pointerof(d_beta_w).as(Void*)
  alpha_beta_dual_params[2] = pointerof(d_x_cur).as(Void*)
  alpha_beta_dual_params[3] = pointerof(d_alpha_cur).as(Void*)
  alpha_beta_dual_params[4] = pointerof(d_beta_cur).as(Void*)
  alpha_beta_dual_params[5] = pointerof(hidden_u32).as(Void*)
  alpha_beta_dual_params[6] = pointerof(alpha_dim_u32).as(Void*)

  run_bundle = -> {
    if batched
      d_x_cur = d_x
      d_qkv_cur = d_qkv
      d_gate_cur = d_gate
      d_alpha_cur = d_alpha
      d_beta_cur = d_beta
      cuda! LibCUDARecProj.cuLaunchKernel(q5_batched_fn, qkv_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, qkv_params, Pointer(Void*).null), "qkv proj batched"
      cuda! LibCUDARecProj.cuLaunchKernel(q4_batched_fn, gate_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, gate_params, Pointer(Void*).null), "gate proj batched"
      if batched_dual_alpha_beta
        cuda! LibCUDARecProj.cuLaunchKernel(q4_dual_batched_fn, alpha_beta_dual_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, alpha_beta_dual_params, Pointer(Void*).null), "alpha beta dual proj batched"
      else
        cuda! LibCUDARecProj.cuLaunchKernel(q4_batched_fn, alpha_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, alpha_params, Pointer(Void*).null), "alpha proj batched"
        cuda! LibCUDARecProj.cuLaunchKernel(q4_batched_fn, beta_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, beta_params, Pointer(Void*).null), "beta proj batched"
      end
    else
      tokens.times do |tok|
        d_x_cur = d_x + bytesize_f32(tok * hidden)
        d_qkv_cur = d_qkv + bytesize_f32(tok * qkv_dim)
        d_gate_cur = d_gate + bytesize_f32(tok * gate_dim)
        d_alpha_cur = d_alpha + bytesize_f32(tok * alpha_dim)
        d_beta_cur = d_beta + bytesize_f32(tok * beta_dim)
        cuda! LibCUDARecProj.cuLaunchKernel(q5_fn, qkv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, qkv_params, Pointer(Void*).null), "qkv proj"
        cuda! LibCUDARecProj.cuLaunchKernel(q4_fn, gate_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, gate_params, Pointer(Void*).null), "gate proj"
        cuda! LibCUDARecProj.cuLaunchKernel(q4_fn, alpha_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, alpha_params, Pointer(Void*).null), "alpha proj"
        cuda! LibCUDARecProj.cuLaunchKernel(q4_fn, beta_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, beta_params, Pointer(Void*).null), "beta proj"
      end
    end
  }

  warmup.times { run_bundle.call }
  cuda! LibCUDARecProj.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0
  gpu_t0 = Time.instant
  reps.times { run_bundle.call }
  cuda! LibCUDARecProj.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps

  qkv_gpu = Array(Float32).new(tokens * qkv_dim, 0.0_f32)
  gate_gpu = Array(Float32).new(tokens * gate_dim, 0.0_f32)
  alpha_gpu = Array(Float32).new(tokens * alpha_dim, 0.0_f32)
  beta_gpu = Array(Float32).new(tokens * beta_dim, 0.0_f32)
  cuda! LibCUDARecProj.cuMemcpyDtoH_v2(qkv_gpu.to_unsafe.as(Void*), d_qkv, bytesize_f32(tokens * qkv_dim)), "cuMemcpyDtoH(qkv)"
  cuda! LibCUDARecProj.cuMemcpyDtoH_v2(gate_gpu.to_unsafe.as(Void*), d_gate, bytesize_f32(tokens * gate_dim)), "cuMemcpyDtoH(gate)"
  cuda! LibCUDARecProj.cuMemcpyDtoH_v2(alpha_gpu.to_unsafe.as(Void*), d_alpha, bytesize_f32(tokens * alpha_dim)), "cuMemcpyDtoH(alpha)"
  cuda! LibCUDARecProj.cuMemcpyDtoH_v2(beta_gpu.to_unsafe.as(Void*), d_beta, bytesize_f32(tokens * beta_dim)), "cuMemcpyDtoH(beta)"

  lines = [] of String
  ok = true
  ok &&= append_stats("qkv", qkv_gpu, qkv_cpu, lines)
  ok &&= append_stats("gate", gate_gpu, gate_cpu, lines)
  ok &&= append_stats("alpha", alpha_gpu, alpha_cpu, lines)
  ok &&= append_stats("beta", beta_gpu, beta_cpu, lines)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "hidden=#{hidden}"
  puts "qkv_dim=#{qkv_dim}"
  puts "gate_dim=#{gate_dim}"
  puts "alpha_dim=#{alpha_dim}"
  puts "beta_dim=#{beta_dim}"
  puts "tokens=#{tokens}"
  puts "batched=#{batched}"
  puts "batched_dual_alpha_beta=#{batched_dual_alpha_beta}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  ptrs.each { |ptr| LibCUDARecProj.cuMemFree_v2(ptr) unless ptr == 0_u64 }
  LibCUDARecProj.cuModuleUnload(q4_mod) unless q4_mod.null?
  LibCUDARecProj.cuModuleUnload(q4_dual_mod) unless q4_dual_mod.null?
  LibCUDARecProj.cuModuleUnload(q5_mod) unless q5_mod.null?
  LibCUDARecProj.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
