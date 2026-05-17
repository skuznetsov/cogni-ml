# Minimal CUDA Q5_K GEMV probe for recurrent Qwen QKV tensors.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDAQ5K
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

PTX            = {{ read_file("src/ml/cuda/kernels/q5k_gemv_probe.ptx") }}
DEFAULT_MODEL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TENSOR = "blk.0.attn_qkv.weight"

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

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
tensor_name = DEFAULT_TENSOR
seed = 23_u64
reps = 1
warmup = 0
tokens = 1
batched = false

OptionParser.parse do |p|
  p.banner = "Usage: cuda_q5k_gemv_probe [--model PATH] [--tensor NAME] [--seed N] [--reps N] [--warmup N] [--tokens N] [--batched]"
  p.on("--model PATH", "Q5_K GGUF model path") { |v| model = v }
  p.on("--tensor NAME", "Q5_K tensor name") { |v| tensor_name = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed kernel launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--tokens N", "Number of independent input rows to run") { |v| tokens = v.to_i }
  p.on("--batched", "Use the batched Q5_K GEMV kernel over all rows") { batched = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "tokens must be positive" unless tokens > 0

gguf = ML::GGUF::GGUFFile.new(model)
info = gguf.tensor(tensor_name) || raise "missing tensor #{tensor_name.inspect}"
raise "expected Q5_K tensor, got #{info.type.name}" unless info.type.q5_k?
raise "expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2
in_dim = info.dims[0].to_i32
out_dim = info.dims[1].to_i32
raise "Q5_K GEMV requires in_dim multiple of 256, got #{in_dim}" unless in_dim % 256 == 0

w_raw = gguf.read_tensor_raw(info)
rng = Random.new(seed)
x = Array(Float32).new(tokens * in_dim) { rng.rand(-1.0_f32..1.0_f32) }
zero_bias = Array(Float32).new(out_dim, 0.0_f32)

cpu_t0 = Time.instant
cpu = Array(Float32).new(tokens * out_dim, 0.0_f32)
tokens.times do |tok|
  row = x[tok * in_dim, in_dim]
  out = ML::GGUF::QuantMatmul.matmul_add(row, 1, in_dim, w_raw, ML::GGUF::TensorType::Q5_K, out_dim, zero_bias)
  out_dim.times { |i| cpu[tok * out_dim + i] = out[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDAQ5K.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDAQ5K::CUdevice
cuda! LibCUDAQ5K.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"
name_buf = Bytes.new(256)
cuda! LibCUDAQ5K.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDAQ5K.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"
ctx = Pointer(Void).null
cuda! LibCUDAQ5K.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

mod = Pointer(Void).null
fn = Pointer(Void).null
batched_fn = Pointer(Void).null
d_w = d_x = d_out = 0_u64

begin
  cuda! LibCUDAQ5K.cuModuleLoadData(pointerof(mod), PTX.to_unsafe.as(Void*)), "cuModuleLoadData"
  cuda! LibCUDAQ5K.cuModuleGetFunction(pointerof(fn), mod, "q5_k_gemv_warp4_f32"), "cuModuleGetFunction"
  cuda! LibCUDAQ5K.cuModuleGetFunction(pointerof(batched_fn), mod, "q5_k_gemv_warp4_f32_batched"), "cuModuleGetFunction(batched)"
  gpu_out = Array(Float32).new(tokens * out_dim, 0.0_f32)
  cuda! LibCUDAQ5K.cuMemAlloc_v2(pointerof(d_w), w_raw.size.to_u64), "cuMemAlloc(w)"
  cuda! LibCUDAQ5K.cuMemAlloc_v2(pointerof(d_x), bytesize_f32(tokens * in_dim)), "cuMemAlloc(x)"
  cuda! LibCUDAQ5K.cuMemAlloc_v2(pointerof(d_out), bytesize_f32(tokens * out_dim)), "cuMemAlloc(out)"
  cuda! LibCUDAQ5K.cuMemcpyHtoD_v2(d_w, w_raw.to_unsafe.as(Void*), w_raw.size.to_u64), "cuMemcpyHtoD(w)"
  cuda! LibCUDAQ5K.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), bytesize_f32(tokens * in_dim)), "cuMemcpyHtoD(x)"

  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  d_x_cur = d_x
  d_out_cur = d_out
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x_cur).as(Void*)
  params[2] = pointerof(d_out_cur).as(Void*)
  params[3] = pointerof(in_dim_u32).as(Void*)
  params[4] = pointerof(out_dim_u32).as(Void*)
  grid = ((out_dim + 3) // 4).to_u32
  tokens_u32 = tokens.to_u32

  run_once = -> {
    if batched
      d_x_cur = d_x
      d_out_cur = d_out
      cuda! LibCUDAQ5K.cuLaunchKernel(batched_fn, grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel(batched)"
    else
      tokens.times do |tok|
        d_x_cur = d_x + bytesize_f32(tok * in_dim)
        d_out_cur = d_out + bytesize_f32(tok * out_dim)
        cuda! LibCUDAQ5K.cuLaunchKernel(fn, grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel"
      end
    end
  }

  warmup.times do
    run_once.call
  end
  cuda! LibCUDAQ5K.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0
  gpu_t0 = Time.instant
  reps.times do
    run_once.call
  end
  cuda! LibCUDAQ5K.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps
  cuda! LibCUDAQ5K.cuMemcpyDtoH_v2(gpu_out.to_unsafe.as(Void*), d_out, bytesize_f32(tokens * out_dim)), "cuMemcpyDtoH(out)"

  cos = cosine(gpu_out, cpu)
  max_diff = max_abs_diff(gpu_out, cpu)
  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "tensor=#{tensor_name}"
  puts "shape=#{in_dim}x#{out_dim}"
  puts "tokens=#{tokens}"
  puts "batched=#{batched}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cos=#{cos.round(8)}"
  puts "max_diff=#{max_diff}"
  puts "ok=#{cos >= 0.99999 && max_diff <= 1.0e-3_f32}"
ensure
  LibCUDAQ5K.cuMemFree_v2(d_w) unless d_w == 0_u64
  LibCUDAQ5K.cuMemFree_v2(d_x) unless d_x == 0_u64
  LibCUDAQ5K.cuMemFree_v2(d_out) unless d_out == 0_u64
  LibCUDAQ5K.cuModuleUnload(mod) unless mod.null?
  LibCUDAQ5K.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
