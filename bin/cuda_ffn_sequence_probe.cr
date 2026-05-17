# GPU-resident CUDA FFN sequence probe for Qwen GGUF weights.
#
# Runs: Q4_K gate GEMV + Q4_K up GEMV -> SwiGLU activation -> Q6_K down GEMV.
# Only the final output is copied back for CPU-reference comparison.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDAFFN
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
  fun cuMemsetD32_v2(dst : CUdeviceptr, value : UInt32, count : LibC::SizeT) : Int32
  fun cuLaunchKernel(fn : CUfunction, grid_x : UInt32, grid_y : UInt32, grid_z : UInt32,
                     block_x : UInt32, block_y : UInt32, block_z : UInt32,
                     shared_mem_bytes : UInt32, stream : Void*,
                     kernel_params : Void**, extra : Void**) : Int32
  fun cuCtxSynchronize : Int32
end

Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q6K_PTX = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}

SWIGLU_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry swiglu_f32(
    .param .u64 gate,
    .param .u64 up,
    .param .u64 out,
    .param .u32 n
)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [gate];
    ld.param.u64 %rd2, [up];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p, %r5, %r1;
    @%p bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    neg.f32 %f3, %f1;
    mov.f32 %f4, 0f3FB8AA3B; // log2(e)
    mul.rn.f32 %f5, %f3, %f4;
    ex2.approx.ftz.f32 %f6, %f5;
    add.rn.f32 %f7, %f6, 0f3F800000;
    rcp.approx.ftz.f32 %f8, %f7;
    mul.rn.f32 %f9, %f1, %f8;
    mul.rn.f32 %f10, %f9, %f2;
    st.global.f32 [%rd7], %f10;

DONE:
    ret;
}
PTX

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

def silu(x : Float32) : Float32
  x / (1.0_f32 + Math.exp(-x).to_f32)
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layer = 0
seed = 23_u64
reps = 1
warmup = 0
tokens = 1
batched = false

OptionParser.parse do |p|
  p.banner = "Usage: cuda_ffn_sequence_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N] [--tokens N] [--batched]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed FFN sequence launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup FFN sequence launches") { |v| warmup = v.to_i }
  p.on("--tokens N", "Number of independent FFN rows to run") { |v| tokens = v.to_i }
  p.on("--batched", "Use grid.y batched Q4/Q6 GEMV kernels for all rows") { batched = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "tokens must be positive" unless tokens > 0

gguf = ML::GGUF::GGUFFile.new(model)
prefix = "blk.#{layer}"
gate_info = gguf.tensor("#{prefix}.ffn_gate.weight") || raise "missing #{prefix}.ffn_gate.weight"
up_info = gguf.tensor("#{prefix}.ffn_up.weight") || raise "missing #{prefix}.ffn_up.weight"
down_info = gguf.tensor("#{prefix}.ffn_down.weight") || raise "missing #{prefix}.ffn_down.weight"
raise "expected Q4_K gate/up" unless gate_info.type.q4_k? && up_info.type.q4_k?
raise "expected Q6_K down" unless down_info.type.q6_k?

hidden = gate_info.dims[0].to_i32
ffn_dim = gate_info.dims[1].to_i32
raise "gate/up shape mismatch" unless up_info.dims[0].to_i32 == hidden && up_info.dims[1].to_i32 == ffn_dim
raise "down shape mismatch" unless down_info.dims[0].to_i32 == ffn_dim && down_info.dims[1].to_i32 == hidden

gate_raw = gguf.read_tensor_raw(gate_info)
up_raw = gguf.read_tensor_raw(up_info)
down_raw = gguf.read_tensor_raw(down_info)
rng = Random.new(seed)
x = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }
zero_ffn = Array(Float32).new(ffn_dim, 0.0_f32)
zero_hidden = Array(Float32).new(hidden, 0.0_f32)

cpu_t0 = Time.instant
out_cpu = Array(Float32).new(tokens * hidden, 0.0_f32)
tokens.times do |tok|
  x_row = x[tok * hidden, hidden]
  gate_cpu = ML::GGUF::QuantMatmul.matmul_add(x_row, 1, hidden, gate_raw, ML::GGUF::TensorType::Q4_K, ffn_dim, zero_ffn)
  up_cpu = ML::GGUF::QuantMatmul.matmul_add(x_row, 1, hidden, up_raw, ML::GGUF::TensorType::Q4_K, ffn_dim, zero_ffn)
  combined_cpu = Array(Float32).new(ffn_dim) { |i| silu(gate_cpu[i]) * up_cpu[i] }
  out_row = ML::GGUF::QuantMatmul.matmul_add(combined_cpu, 1, ffn_dim, down_raw, ML::GGUF::TensorType::Q6_K, hidden, zero_hidden)
  hidden.times { |i| out_cpu[tok * hidden + i] = out_row[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDAFFN.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDAFFN::CUdevice
cuda! LibCUDAFFN.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"
name_buf = Bytes.new(256)
cuda! LibCUDAFFN.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDAFFN.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"

ctx = Pointer(Void).null
cuda! LibCUDAFFN.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

q4_mod = Pointer(Void).null
q6_mod = Pointer(Void).null
act_mod = Pointer(Void).null
q4_fn = Pointer(Void).null
q4_batched_fn = Pointer(Void).null
q6_fn = Pointer(Void).null
q6_batched_fn = Pointer(Void).null
act_fn = Pointer(Void).null

ptrs = [] of UInt64

begin
  cuda! LibCUDAFFN.cuModuleLoadData(pointerof(q4_mod), Q4K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q4)"
  cuda! LibCUDAFFN.cuModuleLoadData(pointerof(q6_mod), Q6K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q6)"
  cuda! LibCUDAFFN.cuModuleLoadData(pointerof(act_mod), SWIGLU_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(swiglu)"
  cuda! LibCUDAFFN.cuModuleGetFunction(pointerof(q4_fn), q4_mod, "q4_k_gemv_warp4_f32"), "cuModuleGetFunction(q4)"
  cuda! LibCUDAFFN.cuModuleGetFunction(pointerof(q4_batched_fn), q4_mod, "q4_k_gemv_warp4_f32_batched"), "cuModuleGetFunction(q4 batched)"
  cuda! LibCUDAFFN.cuModuleGetFunction(pointerof(q6_fn), q6_mod, "q6_k_gemv_warp4_f32"), "cuModuleGetFunction(q6)"
  cuda! LibCUDAFFN.cuModuleGetFunction(pointerof(q6_batched_fn), q6_mod, "q6_k_gemv_warp4_f32_batched"), "cuModuleGetFunction(q6 batched)"
  cuda! LibCUDAFFN.cuModuleGetFunction(pointerof(act_fn), act_mod, "swiglu_f32"), "cuModuleGetFunction(swiglu)"

  d_x = d_gate_w = d_up_w = d_down_w = d_gate = d_up = d_combined = d_out = 0_u64
  {bytesize_f32(tokens * hidden), gate_raw.size.to_u64, up_raw.size.to_u64, down_raw.size.to_u64,
   bytesize_f32(tokens * ffn_dim), bytesize_f32(tokens * ffn_dim), bytesize_f32(tokens * ffn_dim), bytesize_f32(tokens * hidden)}.each_with_index do |size, i|
    pdev = 0_u64
    cuda! LibCUDAFFN.cuMemAlloc_v2(pointerof(pdev), size), "cuMemAlloc(#{i})"
    ptrs << pdev
  end
  d_x, d_gate_w, d_up_w, d_down_w, d_gate, d_up, d_combined, d_out = ptrs

  cuda! LibCUDAFFN.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), bytesize_f32(tokens * hidden)), "cuMemcpyHtoD(x)"
  cuda! LibCUDAFFN.cuMemcpyHtoD_v2(d_gate_w, gate_raw.to_unsafe.as(Void*), gate_raw.size.to_u64), "cuMemcpyHtoD(gate_w)"
  cuda! LibCUDAFFN.cuMemcpyHtoD_v2(d_up_w, up_raw.to_unsafe.as(Void*), up_raw.size.to_u64), "cuMemcpyHtoD(up_w)"
  cuda! LibCUDAFFN.cuMemcpyHtoD_v2(d_down_w, down_raw.to_unsafe.as(Void*), down_raw.size.to_u64), "cuMemcpyHtoD(down_w)"

  hidden_u32 = hidden.to_u32
  ffn_u32 = ffn_dim.to_u32
  ffn_all_u32 = (tokens * ffn_dim).to_u32
  q4_grid = ((ffn_dim + 3) // 4).to_u32
  q6_grid = ((hidden + 3) // 4).to_u32
  act_block = 256_u32
  act_grid = ((ffn_dim + act_block.to_i - 1) // act_block.to_i).to_u32
  act_grid_all = (((tokens * ffn_dim) + act_block.to_i - 1) // act_block.to_i).to_u32
  tokens_u32 = tokens.to_u32
  d_x_cur = d_x
  d_gate_cur = d_gate
  d_up_cur = d_up
  d_combined_cur = d_combined
  d_out_cur = d_out
  act_n_cur = ffn_u32

  q4_gate_params = Pointer(Void*).malloc(5)
  q4_gate_params[0] = pointerof(d_gate_w).as(Void*)
  q4_gate_params[1] = pointerof(d_x_cur).as(Void*)
  q4_gate_params[2] = pointerof(d_gate_cur).as(Void*)
  q4_gate_params[3] = pointerof(hidden_u32).as(Void*)
  q4_gate_params[4] = pointerof(ffn_u32).as(Void*)

  q4_up_params = Pointer(Void*).malloc(5)
  q4_up_params[0] = pointerof(d_up_w).as(Void*)
  q4_up_params[1] = pointerof(d_x_cur).as(Void*)
  q4_up_params[2] = pointerof(d_up_cur).as(Void*)
  q4_up_params[3] = pointerof(hidden_u32).as(Void*)
  q4_up_params[4] = pointerof(ffn_u32).as(Void*)

  act_params = Pointer(Void*).malloc(4)
  act_params[0] = pointerof(d_gate_cur).as(Void*)
  act_params[1] = pointerof(d_up_cur).as(Void*)
  act_params[2] = pointerof(d_combined_cur).as(Void*)
  act_params[3] = pointerof(act_n_cur).as(Void*)

  q6_params = Pointer(Void*).malloc(5)
  q6_params[0] = pointerof(d_down_w).as(Void*)
  q6_params[1] = pointerof(d_combined_cur).as(Void*)
  q6_params[2] = pointerof(d_out_cur).as(Void*)
  q6_params[3] = pointerof(ffn_u32).as(Void*)
  q6_params[4] = pointerof(hidden_u32).as(Void*)

  run_sequence = -> {
    if batched
      d_x_cur = d_x
      d_gate_cur = d_gate
      d_up_cur = d_up
      d_combined_cur = d_combined
      d_out_cur = d_out
      act_n_cur = ffn_all_u32
      cuda! LibCUDAFFN.cuLaunchKernel(q4_batched_fn, q4_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, q4_gate_params, Pointer(Void*).null), "q4 gate batched"
      cuda! LibCUDAFFN.cuLaunchKernel(q4_batched_fn, q4_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, q4_up_params, Pointer(Void*).null), "q4 up batched"
      cuda! LibCUDAFFN.cuLaunchKernel(act_fn, act_grid_all, 1_u32, 1_u32, act_block, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, act_params, Pointer(Void*).null), "swiglu batched"
      cuda! LibCUDAFFN.cuLaunchKernel(q6_batched_fn, q6_grid * tokens_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
        0_u32, Pointer(Void).null, q6_params, Pointer(Void*).null), "q6 down batched"
    else
      tokens.times do |tok|
        d_x_cur = d_x + bytesize_f32(tok * hidden)
        d_gate_cur = d_gate + bytesize_f32(tok * ffn_dim)
        d_up_cur = d_up + bytesize_f32(tok * ffn_dim)
        d_combined_cur = d_combined + bytesize_f32(tok * ffn_dim)
        d_out_cur = d_out + bytesize_f32(tok * hidden)
        act_n_cur = ffn_u32
        cuda! LibCUDAFFN.cuLaunchKernel(q4_fn, q4_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, q4_gate_params, Pointer(Void*).null), "q4 gate"
        cuda! LibCUDAFFN.cuLaunchKernel(q4_fn, q4_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, q4_up_params, Pointer(Void*).null), "q4 up"
        cuda! LibCUDAFFN.cuLaunchKernel(act_fn, act_grid, 1_u32, 1_u32, act_block, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, act_params, Pointer(Void*).null), "swiglu"
        cuda! LibCUDAFFN.cuLaunchKernel(q6_fn, q6_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32,
          0_u32, Pointer(Void).null, q6_params, Pointer(Void*).null), "q6 down"
      end
    end
  }

  warmup.times { run_sequence.call }
  cuda! LibCUDAFFN.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0

  gpu_t0 = Time.instant
  reps.times { run_sequence.call }
  cuda! LibCUDAFFN.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps

  out_gpu = Array(Float32).new(tokens * hidden, 0.0_f32)
  cuda! LibCUDAFFN.cuMemcpyDtoH_v2(out_gpu.to_unsafe.as(Void*), d_out, bytesize_f32(tokens * hidden)), "cuMemcpyDtoH(out)"
  max_diff = max_abs_diff(out_gpu, out_cpu)
  cos = cosine(out_gpu, out_cpu)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "hidden=#{hidden}"
  puts "ffn_dim=#{ffn_dim}"
  puts "tokens=#{tokens}"
  puts "batched=#{batched}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cos=#{cos.round(8)}"
  puts "max_diff=#{max_diff}"
  puts "ok=#{cos >= 0.999 && max_diff <= 0.5_f32}"
ensure
  ptrs.each { |ptr| LibCUDAFFN.cuMemFree_v2(ptr) unless ptr == 0_u64 }
  LibCUDAFFN.cuModuleUnload(q4_mod) unless q4_mod.null?
  LibCUDAFFN.cuModuleUnload(q6_mod) unless q6_mod.null?
  LibCUDAFFN.cuModuleUnload(act_mod) unless act_mod.null?
  LibCUDAFFN.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
