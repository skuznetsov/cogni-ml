# Minimal CUDA Q8_0 GEMV probe for the Linux/CUDA backend boundary.
#
# Loads one real Q8_0 tensor from a GGUF file, runs a Crystal-driven CUDA
# Driver API kernel over the raw GGUF block layout, and compares against the
# existing CPU QuantMatmul reference.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDAQ8
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

PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry q8_0_gemv_f32(
    .param .u64 w_raw,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p;
    .reg .b16 %h<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [x];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.s32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p, %r6, %r2;
    @%p bra DONE;

    shr.u32 %r7, %r1, 5;        // blocks_per_row = in_dim / 32
    mul.lo.u32 %r8, %r7, 34;    // row_bytes
    mul.wide.u32 %rd4, %r6, %r8;
    add.s64 %rd5, %rd1, %rd4;   // row base

    mov.f32 %f1, 0f00000000;
    mov.u32 %r9, 0;

BLOCK_LOOP:
    setp.ge.u32 %p, %r9, %r7;
    @%p bra STORE;

    mul.lo.u32 %r10, %r9, 34;
    cvt.u64.u32 %rd6, %r10;
    add.s64 %rd7, %rd5, %rd6;   // block base
    ld.global.u16 %h1, [%rd7];
    cvt.f32.f16 %f2, %h1;       // block scale
    add.s64 %rd8, %rd7, 2;      // qs base

    shl.b32 %r11, %r9, 5;       // ib * 32
    mov.u32 %r12, 0;

INNER_LOOP:
    setp.ge.u32 %p, %r12, 32;
    @%p bra NEXT_BLOCK;

    add.u32 %r13, %r11, %r12;
    mul.wide.u32 %rd9, %r13, 4;
    add.s64 %rd10, %rd2, %rd9;
    ld.global.f32 %f3, [%rd10];

    cvt.u64.u32 %rd11, %r12;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.s8 %r14, [%rd12];
    cvt.rn.f32.s32 %f4, %r14;

    mul.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f4, %f1;

    add.u32 %r12, %r12, 1;
    bra INNER_LOOP;

NEXT_BLOCK:
    add.u32 %r9, %r9, 1;
    bra BLOCK_LOOP;

STORE:
    mul.wide.u32 %rd13, %r6, 4;
    add.s64 %rd14, %rd3, %rd13;
    st.global.f32 [%rd14], %f1;

DONE:
    ret;
}

.visible .entry q8_0_gemv_warp4_f32(
    .param .u64 w_raw,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p;
    .reg .b16 %h<2>;
    .reg .b32 %r<36>;
    .reg .b64 %rd<28>;
    .reg .f32 %f<10>;
    .shared .align 4 .b8 smem[512];

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [x];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;       // lane inside row warp
    shr.u32 %r6, %r3, 5;        // warp id inside 128-thread block
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;      // output row
    setp.ge.u32 %p, %r8, %r2;
    @%p bra WARP4_DONE;

    shr.u32 %r9, %r1, 5;        // blocks_per_row = in_dim / 32
    mul.lo.u32 %r10, %r9, 34;   // row_bytes
    mul.wide.u32 %rd4, %r8, %r10;
    add.s64 %rd5, %rd1, %rd4;   // row base

    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;

WARP4_BLOCK_LOOP:
    setp.ge.u32 %p, %r11, %r9;
    @%p bra WARP4_REDUCE_PREP;

    mul.lo.u32 %r12, %r11, 34;
    cvt.u64.u32 %rd6, %r12;
    add.s64 %rd7, %rd5, %rd6;   // block base
    ld.global.u16 %h1, [%rd7];
    cvt.f32.f16 %f2, %h1;       // block scale
    add.s64 %rd8, %rd7, 2;      // qs base

    shl.b32 %r13, %r11, 5;
    add.u32 %r14, %r13, %r5;    // ib*32 + lane
    mul.wide.u32 %rd9, %r14, 4;
    add.s64 %rd10, %rd2, %rd9;
    ld.global.f32 %f3, [%rd10];

    cvt.u64.u32 %rd11, %r5;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.s8 %r15, [%rd12];
    cvt.rn.f32.s32 %f4, %r15;

    mul.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f4, %f1;

    add.u32 %r11, %r11, 1;
    bra WARP4_BLOCK_LOOP;

WARP4_REDUCE_PREP:
    shl.b32 %r16, %r3, 2;
    mov.u64 %rd15, smem;
    cvt.u64.u32 %rd16, %r16;
    add.s64 %rd17, %rd15, %rd16;
    st.shared.f32 [%rd17], %f1;
    bar.sync 0;

    setp.ne.u32 %p, %r5, 0;
    @%p bra WARP4_DONE;

    shl.b32 %r17, %r6, 7;       // warp shared base = warp * 32 * 4
    mov.f32 %f6, 0f00000000;
    mov.u32 %r18, 0;

WARP4_SUM_LOOP:
    setp.ge.u32 %p, %r18, 32;
    @%p bra WARP4_STORE;
    shl.b32 %r19, %r18, 2;
    add.u32 %r20, %r17, %r19;
    cvt.u64.u32 %rd18, %r20;
    add.s64 %rd19, %rd15, %rd18;
    ld.shared.f32 %f7, [%rd19];
    add.rn.f32 %f6, %f6, %f7;
    add.u32 %r18, %r18, 1;
    bra WARP4_SUM_LOOP;

WARP4_STORE:
    mul.wide.u32 %rd13, %r8, 4;
    add.s64 %rd14, %rd3, %rd13;
    st.global.f32 [%rd14], %f6;

WARP4_DONE:
    ret;
}
PTX

DEFAULT_MODEL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
DEFAULT_TENSOR = "blk.0.ffn_up.weight"

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
kernel = "warp4"
reps = 1
warmup = 0

OptionParser.parse do |p|
  p.banner = "Usage: cuda_q8_gemv_probe [--model PATH] [--tensor NAME] [--seed N] [--kernel scalar|warp4] [--reps N] [--warmup N] [--block N]"
  p.on("--model PATH", "Q8_0 GGUF model path") { |v| model = v }
  p.on("--tensor NAME", "Q8_0 tensor name") { |v| tensor_name = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--kernel NAME", "CUDA kernel: scalar or warp4") { |v| kernel = v }
  p.on("--reps N", "Timed kernel launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--block N", "CUDA block size") { |v| block = v.to_u32 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "block must be positive" unless block > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "kernel must be scalar or warp4, got #{kernel.inspect}" unless {"scalar", "warp4"}.includes?(kernel)

gguf = ML::GGUF::GGUFFile.new(model)
info = gguf.tensor(tensor_name) || raise "missing tensor #{tensor_name.inspect}"
raise "expected Q8_0 tensor, got #{info.type.name}" unless info.type.q8_0?
raise "expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2

in_dim = info.dims[0].to_i32
out_dim = info.dims[1].to_i32
raise "Q8_0 GEMV requires in_dim multiple of 32, got #{in_dim}" unless in_dim % 32 == 0

w_raw = gguf.read_tensor_raw(info)
rng = Random.new(seed)
x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
zero_bias = Array(Float32).new(out_dim, 0.0_f32)

cpu_t0 = Time.instant
cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q8_0, out_dim, zero_bias)
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDAQ8.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDAQ8::CUdevice
cuda! LibCUDAQ8.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"

name_buf = Bytes.new(256)
cuda! LibCUDAQ8.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDAQ8.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"

ctx = Pointer(Void).null
cuda! LibCUDAQ8.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

mod = Pointer(Void).null
fn = Pointer(Void).null
d_w = 0_u64
d_x = 0_u64
d_out = 0_u64

begin
  cuda! LibCUDAQ8.cuModuleLoadData(pointerof(mod), PTX.to_unsafe.as(Void*)), "cuModuleLoadData"
  kernel_fn = kernel == "warp4" ? "q8_0_gemv_warp4_f32" : "q8_0_gemv_f32"
  cuda! LibCUDAQ8.cuModuleGetFunction(pointerof(fn), mod, kernel_fn), "cuModuleGetFunction"

  gpu_out = Array(Float32).new(out_dim, 0.0_f32)
  raw_size = w_raw.size.to_u64
  x_size = bytesize_f32(in_dim)
  out_size = bytesize_f32(out_dim)
  cuda! LibCUDAQ8.cuMemAlloc_v2(pointerof(d_w), raw_size), "cuMemAlloc(w)"
  cuda! LibCUDAQ8.cuMemAlloc_v2(pointerof(d_x), x_size), "cuMemAlloc(x)"
  cuda! LibCUDAQ8.cuMemAlloc_v2(pointerof(d_out), out_size), "cuMemAlloc(out)"
  cuda! LibCUDAQ8.cuMemcpyHtoD_v2(d_w, w_raw.to_unsafe.as(Void*), raw_size), "cuMemcpyHtoD(w)"
  cuda! LibCUDAQ8.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), x_size), "cuMemcpyHtoD(x)"

  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(in_dim_u32).as(Void*)
  params[4] = pointerof(out_dim_u32).as(Void*)

  launch_block = kernel == "warp4" ? 128_u32 : block
  grid = if kernel == "warp4"
           ((out_dim + 3) // 4).to_u32
         else
           ((out_dim + block.to_i - 1) // block.to_i).to_u32
         end

  warmup.times do
    cuda! LibCUDAQ8.cuLaunchKernel(fn, grid, 1_u32, 1_u32, launch_block, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel(warmup)"
  end
  cuda! LibCUDAQ8.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0

  gpu_t0 = Time.instant
  reps.times do
    cuda! LibCUDAQ8.cuLaunchKernel(fn, grid, 1_u32, 1_u32, launch_block, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel"
  end
  cuda! LibCUDAQ8.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps
  cuda! LibCUDAQ8.cuMemcpyDtoH_v2(gpu_out.to_unsafe.as(Void*), d_out, out_size), "cuMemcpyDtoH(out)"

  max_diff = max_abs_diff(gpu_out, cpu)
  cos = cosine(gpu_out, cpu)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "tensor=#{tensor_name}"
  puts "shape=#{in_dim}x#{out_dim}"
  puts "kernel=#{kernel}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "raw_bytes=#{w_raw.size}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cos=#{cos.round(8)}"
  puts "max_diff=#{max_diff}"
  puts "ok=#{cos >= 0.999999 && max_diff <= 1.0e-4_f32}"
ensure
  LibCUDAQ8.cuMemFree_v2(d_w) unless d_w == 0_u64
  LibCUDAQ8.cuMemFree_v2(d_x) unless d_x == 0_u64
  LibCUDAQ8.cuMemFree_v2(d_out) unless d_out == 0_u64
  LibCUDAQ8.cuModuleUnload(mod) unless mod.null?
  LibCUDAQ8.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
