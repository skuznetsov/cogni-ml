# CUDA DeltaNet state/output slice probe for Qwen GGUF weights.
#
# Runs a synthetic DeltaNet step on CUDA, keeps the y vector GPU-resident, then
# feeds it directly into the real Q4_K ssm_out projection. This intentionally
# omits recurrent post RMSNorm/SiLU gating; it tests the stateful-kernel to
# quantized-output-projection boundary before the full recurrent layer facade.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/gguf/qwen35_cpu"

@[Link(ldflags: "-lcuda")]
lib LibCUDADeltaOut
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

DN_PTX        = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
Q4K_PTX       = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
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
    d = (v - b[i]).abs
    max = d if d > max
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

def report_pair(name : String, gpu : Array(Float32), cpu : Array(Float32), lines : Array(String), max_allowed : Float32) : Bool
  cos = cosine(gpu, cpu)
  max_diff = max_abs_diff(gpu, cpu)
  ok = cos >= 0.99999 && max_diff <= max_allowed
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layer = 0
seed = 29_u64
reps = 1
warmup = 0

OptionParser.parse do |p|
  p.banner = "Usage: cuda_deltanet_output_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Recurrent layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed DeltaNet+ssm_out launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

h_k = 16
h_v = 32
s = 128
inner_dim = h_v * s
scale = (1.0 / Math.sqrt(s.to_f64)).to_f32

gguf = ML::GGUF::GGUFFile.new(model)
prefix = "blk.#{layer}"
out_info = gguf.tensor("#{prefix}.ssm_out.weight") || raise "missing #{prefix}.ssm_out.weight"
raise "expected Q4_K ssm_out" unless out_info.type.q4_k?
raise "ssm_out input mismatch: expected #{inner_dim}, got #{out_info.dims[0]}" unless out_info.dims[0].to_i32 == inner_dim
out_dim = out_info.dims[1].to_i32
out_raw = gguf.read_tensor_raw(out_info)

rng = Random.new(seed)
state_init = Array(Float32).new(h_v * s * s) { ((rng.next_float - 0.5) * 0.05).to_f32 }
q = Array(Float32).new(h_k * s) { ((rng.next_float - 0.5) * 0.2).to_f32 }
k = Array(Float32).new(h_k * s) { ((rng.next_float - 0.5) * 0.2).to_f32 }
v = Array(Float32).new(h_v * s) { ((rng.next_float - 0.5) * 0.2).to_f32 }
g = Array(Float32).new(h_v) { (0.90 + 0.09 * rng.next_float).to_f32 }
beta = Array(Float32).new(h_v) { rng.next_float.to_f32 }

cpu_t0 = Time.instant
state_cpu = state_init.dup
y_cpu = Array(Float32).new(inner_dim, 0.0_f32)
ML::GGUF::Qwen35CPU.delta_net_step!(state_cpu, q, k, v, g, beta, y_cpu, h_k, h_v, s, scale)
proj_cpu = ML::GGUF::QuantMatmul.matmul_add(y_cpu, 1, inner_dim, out_raw, ML::GGUF::TensorType::Q4_K, out_dim, Array(Float32).new(out_dim, 0.0_f32))
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDADeltaOut.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDADeltaOut::CUdevice
cuda! LibCUDADeltaOut.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"
name_buf = Bytes.new(256)
cuda! LibCUDADeltaOut.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDADeltaOut.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"
ctx = Pointer(Void).null
cuda! LibCUDADeltaOut.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

dn_mod = Pointer(Void).null
q4_mod = Pointer(Void).null
dn_fn = Pointer(Void).null
q4_fn = Pointer(Void).null
ptrs = [] of UInt64

begin
  cuda! LibCUDADeltaOut.cuModuleLoadData(pointerof(dn_mod), DN_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(delta)"
  cuda! LibCUDADeltaOut.cuModuleLoadData(pointerof(q4_mod), Q4K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q4)"
  cuda! LibCUDADeltaOut.cuModuleGetFunction(pointerof(dn_fn), dn_mod, "deltanet_step_128_probe"), "cuModuleGetFunction(delta)"
  cuda! LibCUDADeltaOut.cuModuleGetFunction(pointerof(q4_fn), q4_mod, "q4_k_gemv_warp4_f32"), "cuModuleGetFunction(q4)"

  sizes = [bytesize_f32(state_init.size), bytesize_f32(q.size), bytesize_f32(k.size), bytesize_f32(v.size),
           bytesize_f32(g.size), bytesize_f32(beta.size), bytesize_f32(y_cpu.size), out_raw.size.to_u64,
           bytesize_f32(out_dim)]
  sizes.each_with_index do |size_bytes, i|
    pdev = 0_u64
    cuda! LibCUDADeltaOut.cuMemAlloc_v2(pointerof(pdev), size_bytes), "cuMemAlloc(#{i})"
    ptrs << pdev
  end
  d_state, d_q, d_k, d_v, d_g, d_beta, d_y, d_out_w, d_proj = ptrs

  copy_inputs = -> {
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_state, state_init.to_unsafe.as(Void*), bytesize_f32(state_init.size)), "cuMemcpyHtoD(state)"
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_q, q.to_unsafe.as(Void*), bytesize_f32(q.size)), "cuMemcpyHtoD(q)"
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_k, k.to_unsafe.as(Void*), bytesize_f32(k.size)), "cuMemcpyHtoD(k)"
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_v, v.to_unsafe.as(Void*), bytesize_f32(v.size)), "cuMemcpyHtoD(v)"
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_g, g.to_unsafe.as(Void*), bytesize_f32(g.size)), "cuMemcpyHtoD(g)"
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_beta, beta.to_unsafe.as(Void*), bytesize_f32(beta.size)), "cuMemcpyHtoD(beta)"
    cuda! LibCUDADeltaOut.cuMemcpyHtoD_v2(d_out_w, out_raw.to_unsafe.as(Void*), out_raw.size.to_u64), "cuMemcpyHtoD(out_w)"
  }
  copy_inputs.call

  h_k_u32 = h_k.to_u32
  h_v_u32 = h_v.to_u32
  inner_u32 = inner_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  q4_grid = ((out_dim + 3) // 4).to_u32

  dn_params = Pointer(Void*).malloc(10)
  dn_params[0] = pointerof(d_state).as(Void*)
  dn_params[1] = pointerof(d_q).as(Void*)
  dn_params[2] = pointerof(d_k).as(Void*)
  dn_params[3] = pointerof(d_v).as(Void*)
  dn_params[4] = pointerof(d_g).as(Void*)
  dn_params[5] = pointerof(d_beta).as(Void*)
  dn_params[6] = pointerof(d_y).as(Void*)
  dn_params[7] = pointerof(h_k_u32).as(Void*)
  dn_params[8] = pointerof(h_v_u32).as(Void*)
  dn_params[9] = pointerof(scale).as(Void*)

  q4_params = Pointer(Void*).malloc(5)
  q4_params[0] = pointerof(d_out_w).as(Void*)
  q4_params[1] = pointerof(d_y).as(Void*)
  q4_params[2] = pointerof(d_proj).as(Void*)
  q4_params[3] = pointerof(inner_u32).as(Void*)
  q4_params[4] = pointerof(out_dim_u32).as(Void*)

  run_bundle = -> {
    cuda! LibCUDADeltaOut.cuLaunchKernel(dn_fn, h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, dn_params, Pointer(Void*).null), "delta step"
    cuda! LibCUDADeltaOut.cuLaunchKernel(q4_fn, q4_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, q4_params, Pointer(Void*).null), "ssm_out"
  }

  warmup.times { run_bundle.call }
  cuda! LibCUDADeltaOut.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0
  copy_inputs.call
  gpu_t0 = Time.instant
  reps.times { run_bundle.call }
  cuda! LibCUDADeltaOut.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps

  copy_inputs.call
  run_bundle.call
  cuda! LibCUDADeltaOut.cuCtxSynchronize, "cuCtxSynchronize(correctness)"

  state_gpu = Array(Float32).new(state_init.size, 0.0_f32)
  y_gpu = Array(Float32).new(inner_dim, 0.0_f32)
  proj_gpu = Array(Float32).new(out_dim, 0.0_f32)
  cuda! LibCUDADeltaOut.cuMemcpyDtoH_v2(state_gpu.to_unsafe.as(Void*), d_state, bytesize_f32(state_gpu.size)), "cuMemcpyDtoH(state)"
  cuda! LibCUDADeltaOut.cuMemcpyDtoH_v2(y_gpu.to_unsafe.as(Void*), d_y, bytesize_f32(y_gpu.size)), "cuMemcpyDtoH(y)"
  cuda! LibCUDADeltaOut.cuMemcpyDtoH_v2(proj_gpu.to_unsafe.as(Void*), d_proj, bytesize_f32(proj_gpu.size)), "cuMemcpyDtoH(proj)"

  lines = [] of String
  ok = true
  ok &&= report_pair("state", state_gpu, state_cpu, lines, 1.0e-5_f32)
  ok &&= report_pair("y", y_gpu, y_cpu, lines, 1.0e-5_f32)
  ok &&= report_pair("proj", proj_gpu, proj_cpu, lines, 1.0e-3_f32)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "h_k=#{h_k}"
  puts "h_v=#{h_v}"
  puts "state_dim=#{state_init.size}"
  puts "inner_dim=#{inner_dim}"
  puts "out_dim=#{out_dim}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  ptrs.each { |ptr| LibCUDADeltaOut.cuMemFree_v2(ptr) unless ptr == 0_u64 }
  LibCUDADeltaOut.cuModuleUnload(dn_mod) unless dn_mod.null?
  LibCUDADeltaOut.cuModuleUnload(q4_mod) unless q4_mod.null?
  LibCUDADeltaOut.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
