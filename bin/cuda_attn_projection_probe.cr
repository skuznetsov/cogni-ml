# GPU-resident CUDA full-attention projection probe for Qwen GGUF weights.
#
# Runs: Q4_K attn_q GEMV + Q4_K attn_k GEMV + Q6_K attn_v GEMV from the
# same GPU-resident hidden vector. Outputs are copied back only after all
# projections finish, then compared against CPU QuantMatmul references.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDAAttnProj
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

Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q6K_PTX = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

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
layer = 3
seed = 23_u64
reps = 1
warmup = 0

OptionParser.parse do |p|
  p.banner = "Usage: cuda_attn_projection_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Full-attention layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed projection-bundle launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup projection-bundle launches") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

gguf = ML::GGUF::GGUFFile.new(model)
prefix = "blk.#{layer}"
q_info = gguf.tensor("#{prefix}.attn_q.weight") || raise "missing #{prefix}.attn_q.weight"
k_info = gguf.tensor("#{prefix}.attn_k.weight") || raise "missing #{prefix}.attn_k.weight"
v_info = gguf.tensor("#{prefix}.attn_v.weight") || raise "missing #{prefix}.attn_v.weight"
raise "expected Q4_K q/k" unless q_info.type.q4_k? && k_info.type.q4_k?
raise "expected Q6_K v" unless v_info.type.q6_k?

hidden = q_info.dims[0].to_i32
q_dim = q_info.dims[1].to_i32
k_dim = k_info.dims[1].to_i32
v_dim = v_info.dims[1].to_i32
raise "input shape mismatch" unless k_info.dims[0].to_i32 == hidden && v_info.dims[0].to_i32 == hidden

q_raw = gguf.read_tensor_raw(q_info)
k_raw = gguf.read_tensor_raw(k_info)
v_raw = gguf.read_tensor_raw(v_info)
rng = Random.new(seed)
x = Array(Float32).new(hidden) { rng.rand(-1.0_f32..1.0_f32) }
zero_q = Array(Float32).new(q_dim, 0.0_f32)
zero_k = Array(Float32).new(k_dim, 0.0_f32)
zero_v = Array(Float32).new(v_dim, 0.0_f32)

cpu_t0 = Time.instant
q_cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, q_raw, ML::GGUF::TensorType::Q4_K, q_dim, zero_q)
k_cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, k_raw, ML::GGUF::TensorType::Q4_K, k_dim, zero_k)
v_cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, v_raw, ML::GGUF::TensorType::Q6_K, v_dim, zero_v)
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDAAttnProj.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDAAttnProj::CUdevice
cuda! LibCUDAAttnProj.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"
name_buf = Bytes.new(256)
cuda! LibCUDAAttnProj.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDAAttnProj.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"

ctx = Pointer(Void).null
cuda! LibCUDAAttnProj.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

q4_mod = Pointer(Void).null
q6_mod = Pointer(Void).null
q4_fn = Pointer(Void).null
q6_fn = Pointer(Void).null
ptrs = [] of UInt64

begin
  cuda! LibCUDAAttnProj.cuModuleLoadData(pointerof(q4_mod), Q4K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q4)"
  cuda! LibCUDAAttnProj.cuModuleLoadData(pointerof(q6_mod), Q6K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q6)"
  cuda! LibCUDAAttnProj.cuModuleGetFunction(pointerof(q4_fn), q4_mod, "q4_k_gemv_warp4_f32"), "cuModuleGetFunction(q4)"
  cuda! LibCUDAAttnProj.cuModuleGetFunction(pointerof(q6_fn), q6_mod, "q6_k_gemv_warp4_f32"), "cuModuleGetFunction(q6)"

  sizes = [bytesize_f32(hidden), q_raw.size.to_u64, k_raw.size.to_u64, v_raw.size.to_u64,
           bytesize_f32(q_dim), bytesize_f32(k_dim), bytesize_f32(v_dim)]
  sizes.each_with_index do |size, i|
    pdev = 0_u64
    cuda! LibCUDAAttnProj.cuMemAlloc_v2(pointerof(pdev), size), "cuMemAlloc(#{i})"
    ptrs << pdev
  end
  d_x, d_q_w, d_k_w, d_v_w, d_q, d_k, d_v = ptrs

  cuda! LibCUDAAttnProj.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), bytesize_f32(hidden)), "cuMemcpyHtoD(x)"
  cuda! LibCUDAAttnProj.cuMemcpyHtoD_v2(d_q_w, q_raw.to_unsafe.as(Void*), q_raw.size.to_u64), "cuMemcpyHtoD(q_w)"
  cuda! LibCUDAAttnProj.cuMemcpyHtoD_v2(d_k_w, k_raw.to_unsafe.as(Void*), k_raw.size.to_u64), "cuMemcpyHtoD(k_w)"
  cuda! LibCUDAAttnProj.cuMemcpyHtoD_v2(d_v_w, v_raw.to_unsafe.as(Void*), v_raw.size.to_u64), "cuMemcpyHtoD(v_w)"

  hidden_u32 = hidden.to_u32
  q_dim_u32 = q_dim.to_u32
  k_dim_u32 = k_dim.to_u32
  v_dim_u32 = v_dim.to_u32
  q_grid = ((q_dim + 3) // 4).to_u32
  k_grid = ((k_dim + 3) // 4).to_u32
  v_grid = ((v_dim + 3) // 4).to_u32

  q_params = Pointer(Void*).malloc(5)
  q_params[0] = pointerof(d_q_w).as(Void*)
  q_params[1] = pointerof(d_x).as(Void*)
  q_params[2] = pointerof(d_q).as(Void*)
  q_params[3] = pointerof(hidden_u32).as(Void*)
  q_params[4] = pointerof(q_dim_u32).as(Void*)

  k_params = Pointer(Void*).malloc(5)
  k_params[0] = pointerof(d_k_w).as(Void*)
  k_params[1] = pointerof(d_x).as(Void*)
  k_params[2] = pointerof(d_k).as(Void*)
  k_params[3] = pointerof(hidden_u32).as(Void*)
  k_params[4] = pointerof(k_dim_u32).as(Void*)

  v_params = Pointer(Void*).malloc(5)
  v_params[0] = pointerof(d_v_w).as(Void*)
  v_params[1] = pointerof(d_x).as(Void*)
  v_params[2] = pointerof(d_v).as(Void*)
  v_params[3] = pointerof(hidden_u32).as(Void*)
  v_params[4] = pointerof(v_dim_u32).as(Void*)

  run_bundle = -> {
    cuda! LibCUDAAttnProj.cuLaunchKernel(q4_fn, q_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, q_params, Pointer(Void*).null), "q proj"
    cuda! LibCUDAAttnProj.cuLaunchKernel(q4_fn, k_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, k_params, Pointer(Void*).null), "k proj"
    cuda! LibCUDAAttnProj.cuLaunchKernel(q6_fn, v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, v_params, Pointer(Void*).null), "v proj"
  }

  warmup.times { run_bundle.call }
  cuda! LibCUDAAttnProj.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0

  gpu_t0 = Time.instant
  reps.times { run_bundle.call }
  cuda! LibCUDAAttnProj.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps

  q_gpu = Array(Float32).new(q_dim, 0.0_f32)
  k_gpu = Array(Float32).new(k_dim, 0.0_f32)
  v_gpu = Array(Float32).new(v_dim, 0.0_f32)
  cuda! LibCUDAAttnProj.cuMemcpyDtoH_v2(q_gpu.to_unsafe.as(Void*), d_q, bytesize_f32(q_dim)), "cuMemcpyDtoH(q)"
  cuda! LibCUDAAttnProj.cuMemcpyDtoH_v2(k_gpu.to_unsafe.as(Void*), d_k, bytesize_f32(k_dim)), "cuMemcpyDtoH(k)"
  cuda! LibCUDAAttnProj.cuMemcpyDtoH_v2(v_gpu.to_unsafe.as(Void*), d_v, bytesize_f32(v_dim)), "cuMemcpyDtoH(v)"

  lines = [] of String
  ok = true
  ok &&= append_stats("q", q_gpu, q_cpu, lines)
  ok &&= append_stats("k", k_gpu, k_cpu, lines)
  ok &&= append_stats("v", v_gpu, v_cpu, lines)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "hidden=#{hidden}"
  puts "q_dim=#{q_dim}"
  puts "k_dim=#{k_dim}"
  puts "v_dim=#{v_dim}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  ptrs.each { |ptr| LibCUDAAttnProj.cuMemFree_v2(ptr) unless ptr == 0_u64 }
  LibCUDAAttnProj.cuModuleUnload(q4_mod) unless q4_mod.null?
  LibCUDAAttnProj.cuModuleUnload(q6_mod) unless q6_mod.null?
  LibCUDAAttnProj.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
