# Minimal CUDA Q4_K GEMV probe for the Linux/CUDA backend boundary.
#
# Loads one real Q4_K tensor from a GGUF file, runs a Crystal-driven CUDA
# Driver API kernel over the raw GGUF block layout, and compares against the
# existing CPU QuantMatmul reference.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDAQ4K
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

PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}

DEFAULT_MODEL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TENSOR = "blk.0.ssm_alpha.weight"

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
block = 128_u32
reps = 1
warmup = 0
kernel = "warp4"
tokens = 1
batched = false
xsum = false

OptionParser.parse do |p|
  p.banner = "Usage: cuda_q4k_gemv_probe [--model PATH] [--tensor NAME] [--seed N] [--kernel scalar|warp4] [--reps N] [--warmup N] [--block N] [--tokens N] [--batched] [--xsum]"
  p.on("--model PATH", "Q4_K GGUF model path") { |v| model = v }
  p.on("--tensor NAME", "Q4_K tensor name") { |v| tensor_name = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--kernel NAME", "CUDA kernel: scalar or warp4") { |v| kernel = v }
  p.on("--reps N", "Timed kernel launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--block N", "CUDA block size") { |v| block = v.to_u32 }
  p.on("--tokens N", "Number of independent input rows to run") { |v| tokens = v.to_i }
  p.on("--batched", "Use the batched Q4_K warp4 GEMV kernel over all rows") { batched = true }
  p.on("--xsum", "Use probe-only warp4 kernel with precomputed 32-float activation sums") { xsum = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "block must be positive" unless block > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "kernel must be scalar or warp4, got #{kernel.inspect}" unless {"scalar", "warp4"}.includes?(kernel)
raise "tokens must be positive" unless tokens > 0
raise "--batched requires --kernel warp4" if batched && kernel != "warp4"
raise "--xsum requires --kernel warp4" if xsum && kernel != "warp4"
raise "--xsum is not yet wired for --batched" if xsum && batched

gguf = ML::GGUF::GGUFFile.new(model)
info = gguf.tensor(tensor_name) || raise "missing tensor #{tensor_name.inspect}"
raise "expected Q4_K tensor, got #{info.type.name}" unless info.type.q4_k?
raise "expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2

in_dim = info.dims[0].to_i32
out_dim = info.dims[1].to_i32
raise "Q4_K GEMV requires in_dim multiple of 256, got #{in_dim}" unless in_dim % 256 == 0

w_raw = gguf.read_tensor_raw(info)
rng = Random.new(seed)
x = Array(Float32).new(tokens * in_dim) { rng.rand(-1.0_f32..1.0_f32) }
zero_bias = Array(Float32).new(out_dim, 0.0_f32)

blocks_per_row = in_dim // 256
xsum_host = Array(Float32).new(tokens * blocks_per_row * 8, 0.0_f32)
if xsum
  tokens.times do |tok|
    tok_x = tok * in_dim
    tok_sum = tok * blocks_per_row * 8
    blocks_per_row.times do |blk|
      8.times do |chunk|
        acc = 0.0_f32
        base = tok_x + blk * 256 + chunk * 32
        32.times { |i| acc += x[base + i] }
        xsum_host[tok_sum + blk * 8 + chunk] = acc
      end
    end
  end
end

cpu_t0 = Time.instant
cpu = Array(Float32).new(tokens * out_dim, 0.0_f32)
tokens.times do |tok|
  row = x[tok * in_dim, in_dim]
  out = ML::GGUF::QuantMatmul.matmul_add(row, 1, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, zero_bias)
  out_dim.times { |i| cpu[tok * out_dim + i] = out[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDAQ4K.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDAQ4K::CUdevice
cuda! LibCUDAQ4K.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"

name_buf = Bytes.new(256)
cuda! LibCUDAQ4K.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDAQ4K.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"

ctx = Pointer(Void).null
cuda! LibCUDAQ4K.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

mod = Pointer(Void).null
fn = Pointer(Void).null
batched_fn = Pointer(Void).null
d_w = 0_u64
d_x = 0_u64
d_xsum = 0_u64
d_out = 0_u64

begin
  cuda! LibCUDAQ4K.cuModuleLoadData(pointerof(mod), PTX.to_unsafe.as(Void*)), "cuModuleLoadData"
  kernel_fn = if xsum
                "q4_k_gemv_warp4_f32_xsum"
              elsif kernel == "warp4"
                "q4_k_gemv_warp4_f32"
              else
                "q4_k_gemv_scalar_f32"
              end
  cuda! LibCUDAQ4K.cuModuleGetFunction(pointerof(fn), mod, kernel_fn), "cuModuleGetFunction"
  cuda! LibCUDAQ4K.cuModuleGetFunction(pointerof(batched_fn), mod, "q4_k_gemv_warp4_f32_batched"), "cuModuleGetFunction(batched)" if batched

  gpu_out = Array(Float32).new(tokens * out_dim, 0.0_f32)
  raw_size = w_raw.size.to_u64
  x_size = bytesize_f32(tokens * in_dim)
  xsum_size = bytesize_f32(xsum_host.size)
  out_size = bytesize_f32(tokens * out_dim)
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_w), raw_size), "cuMemAlloc(w)"
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_x), x_size), "cuMemAlloc(x)"
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_xsum), xsum_size), "cuMemAlloc(xsum)" if xsum
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_out), out_size), "cuMemAlloc(out)"
  cuda! LibCUDAQ4K.cuMemcpyHtoD_v2(d_w, w_raw.to_unsafe.as(Void*), raw_size), "cuMemcpyHtoD(w)"
  cuda! LibCUDAQ4K.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), x_size), "cuMemcpyHtoD(x)"
  cuda! LibCUDAQ4K.cuMemcpyHtoD_v2(d_xsum, xsum_host.to_unsafe.as(Void*), xsum_size), "cuMemcpyHtoD(xsum)" if xsum

  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  d_x_cur = d_x
  d_xsum_cur = d_xsum
  d_out_cur = d_out
  params = Pointer(Void*).malloc(xsum ? 6 : 5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x_cur).as(Void*)
  params[2] = pointerof(d_out_cur).as(Void*)
  if xsum
    params[3] = pointerof(d_xsum_cur).as(Void*)
    params[4] = pointerof(in_dim_u32).as(Void*)
    params[5] = pointerof(out_dim_u32).as(Void*)
  else
    params[3] = pointerof(in_dim_u32).as(Void*)
    params[4] = pointerof(out_dim_u32).as(Void*)
  end

  launch_block = kernel == "warp4" ? 128_u32 : block
  grid = if kernel == "warp4"
           ((out_dim + 3) // 4).to_u32
         else
           ((out_dim + block.to_i - 1) // block.to_i).to_u32
         end
  tokens_u32 = tokens.to_u32
  run_once = -> {
    if batched
      d_x_cur = d_x
      d_out_cur = d_out
      cuda! LibCUDAQ4K.cuLaunchKernel(batched_fn, grid * tokens_u32, 1_u32, 1_u32, launch_block, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel(batched)"
    else
      tokens.times do |tok|
        d_x_cur = d_x + bytesize_f32(tok * in_dim)
        d_xsum_cur = d_xsum + bytesize_f32(tok * blocks_per_row * 8) if xsum
        d_out_cur = d_out + bytesize_f32(tok * out_dim)
        cuda! LibCUDAQ4K.cuLaunchKernel(fn, grid, 1_u32, 1_u32, launch_block, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel"
      end
    end
  }
  warmup.times do
    run_once.call
  end
  cuda! LibCUDAQ4K.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0

  gpu_t0 = Time.instant
  reps.times do
    run_once.call
  end
  cuda! LibCUDAQ4K.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps
  cuda! LibCUDAQ4K.cuMemcpyDtoH_v2(gpu_out.to_unsafe.as(Void*), d_out, out_size), "cuMemcpyDtoH(out)"

  max_diff = max_abs_diff(gpu_out, cpu)
  cos = cosine(gpu_out, cpu)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "tensor=#{tensor_name}"
  puts "shape=#{in_dim}x#{out_dim}"
  puts "kernel=#{kernel}"
  puts "tokens=#{tokens}"
  puts "batched=#{batched}"
  puts "xsum=#{xsum}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "raw_bytes=#{w_raw.size}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cos=#{cos.round(8)}"
  puts "max_diff=#{max_diff}"
  puts "ok=#{cos >= 0.99999 && max_diff <= 1.0e-3_f32}"
ensure
  LibCUDAQ4K.cuMemFree_v2(d_w) unless d_w == 0_u64
  LibCUDAQ4K.cuMemFree_v2(d_x) unless d_x == 0_u64
  LibCUDAQ4K.cuMemFree_v2(d_xsum) unless d_xsum == 0_u64
  LibCUDAQ4K.cuMemFree_v2(d_out) unless d_out == 0_u64
  LibCUDAQ4K.cuModuleUnload(mod) unless mod.null?
  LibCUDAQ4K.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
