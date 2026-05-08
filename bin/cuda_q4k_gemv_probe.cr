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

PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry q4_k_gemv_scalar_f32(
    .param .u64 w_raw,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<6>;
    .reg .b16 %h<3>;
    .reg .b32 %r<90>;
    .reg .b64 %rd<70>;
    .reg .f32 %f<24>;

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [x];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.s32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra DONE;

    shr.u32 %r7, %r1, 8;       // blocks_per_row = in_dim / 256
    mul.lo.u32 %r8, %r7, 144;  // row_bytes
    mul.wide.u32 %rd4, %r6, %r8;
    add.s64 %rd5, %rd1, %rd4;  // row base

    mov.f32 %f1, 0f00000000;
    mov.u32 %r9, 0;

BLOCK_LOOP:
    setp.ge.u32 %p1, %r9, %r7;
    @%p1 bra STORE;

    mul.lo.u32 %r10, %r9, 144;
    cvt.u64.u32 %rd6, %r10;
    add.s64 %rd7, %rd5, %rd6;  // block base
    ld.global.u16 %h1, [%rd7];
    add.s64 %rd8, %rd7, 2;
    ld.global.u16 %h2, [%rd8];
    cvt.f32.f16 %f2, %h1;      // d
    cvt.f32.f16 %f3, %h2;      // dmin
    add.s64 %rd9, %rd7, 4;     // scales base
    add.s64 %rd10, %rd7, 16;   // qs base

    mov.u32 %r11, 0;           // group 0..3

GROUP_LOOP:
    setp.ge.u32 %p1, %r11, 4;
    @%p1 bra NEXT_BLOCK;

    shl.b32 %r12, %r11, 1;     // scale index 0,2,4,6

    // First sub-block scale/min for group*64 + 0..31.
    setp.lt.u32 %p2, %r12, 4;
    @%p2 bra SCALE1_LOW;

SCALE1_HIGH:
    add.u32 %r13, %r12, 4;
    cvt.u64.u32 %rd11, %r13;
    add.s64 %rd12, %rd9, %rd11;
    ld.global.u8 %r14, [%rd12];
    and.b32 %r20, %r14, 15;
    add.u32 %r15, %r12, -4;
    cvt.u64.u32 %rd13, %r15;
    add.s64 %rd14, %rd9, %rd13;
    ld.global.u8 %r16, [%rd14];
    shr.u32 %r17, %r16, 6;
    shl.b32 %r17, %r17, 4;
    or.b32 %r20, %r20, %r17;
    shr.u32 %r21, %r14, 4;
    cvt.u64.u32 %rd15, %r12;
    add.s64 %rd16, %rd9, %rd15;
    ld.global.u8 %r18, [%rd16];
    shr.u32 %r19, %r18, 6;
    shl.b32 %r19, %r19, 4;
    or.b32 %r21, %r21, %r19;
    bra SCALE1_DONE;

SCALE1_LOW:
    cvt.u64.u32 %rd17, %r12;
    add.s64 %rd18, %rd9, %rd17;
    ld.global.u8 %r20, [%rd18];
    and.b32 %r20, %r20, 63;
    add.u32 %r22, %r12, 4;
    cvt.u64.u32 %rd19, %r22;
    add.s64 %rd20, %rd9, %rd19;
    ld.global.u8 %r21, [%rd20];
    and.b32 %r21, %r21, 63;

SCALE1_DONE:
    // Second sub-block scale/min for group*64 + 32..63.
    add.u32 %r23, %r12, 1;
    setp.lt.u32 %p3, %r23, 4;
    @%p3 bra SCALE2_LOW;

SCALE2_HIGH:
    add.u32 %r24, %r23, 4;
    cvt.u64.u32 %rd21, %r24;
    add.s64 %rd22, %rd9, %rd21;
    ld.global.u8 %r25, [%rd22];
    and.b32 %r30, %r25, 15;
    add.u32 %r26, %r23, -4;
    cvt.u64.u32 %rd23, %r26;
    add.s64 %rd24, %rd9, %rd23;
    ld.global.u8 %r27, [%rd24];
    shr.u32 %r28, %r27, 6;
    shl.b32 %r28, %r28, 4;
    or.b32 %r30, %r30, %r28;
    shr.u32 %r31, %r25, 4;
    cvt.u64.u32 %rd25, %r23;
    add.s64 %rd26, %rd9, %rd25;
    ld.global.u8 %r29, [%rd26];
    shr.u32 %r32, %r29, 6;
    shl.b32 %r32, %r32, 4;
    or.b32 %r31, %r31, %r32;
    bra SCALE2_DONE;

SCALE2_LOW:
    cvt.u64.u32 %rd27, %r23;
    add.s64 %rd28, %rd9, %rd27;
    ld.global.u8 %r30, [%rd28];
    and.b32 %r30, %r30, 63;
    add.u32 %r33, %r23, 4;
    cvt.u64.u32 %rd29, %r33;
    add.s64 %rd30, %rd9, %rd29;
    ld.global.u8 %r31, [%rd30];
    and.b32 %r31, %r31, 63;

SCALE2_DONE:
    cvt.rn.f32.u32 %f4, %r20;
    mul.rn.f32 %f4, %f4, %f2;   // d * sc1
    cvt.rn.f32.u32 %f5, %r21;
    mul.rn.f32 %f5, %f5, %f3;   // dmin * min1
    cvt.rn.f32.u32 %f6, %r30;
    mul.rn.f32 %f6, %f6, %f2;   // d * sc2
    cvt.rn.f32.u32 %f7, %r31;
    mul.rn.f32 %f7, %f7, %f3;   // dmin * min2

    mul.lo.u32 %r34, %r11, 32;  // qs offset
    mul.lo.u32 %r35, %r11, 64;  // input group offset
    shl.b32 %r36, %r9, 8;       // block input base
    add.u32 %r37, %r36, %r35;

    mov.u32 %r38, 0;
LOW_LOOP:
    setp.ge.u32 %p4, %r38, 32;
    @%p4 bra HIGH_PREP;

    add.u32 %r39, %r34, %r38;
    cvt.u64.u32 %rd31, %r39;
    add.s64 %rd32, %rd10, %rd31;
    ld.global.u8 %r40, [%rd32];
    and.b32 %r41, %r40, 15;
    cvt.rn.f32.u32 %f8, %r41;
    mul.rn.f32 %f9, %f4, %f8;
    sub.rn.f32 %f9, %f9, %f5;

    add.u32 %r42, %r37, %r38;
    mul.wide.u32 %rd33, %r42, 4;
    add.s64 %rd34, %rd2, %rd33;
    ld.global.f32 %f10, [%rd34];
    fma.rn.f32 %f1, %f10, %f9, %f1;

    add.u32 %r38, %r38, 1;
    bra LOW_LOOP;

HIGH_PREP:
    mov.u32 %r43, 0;
HIGH_LOOP:
    setp.ge.u32 %p5, %r43, 32;
    @%p5 bra GROUP_NEXT;

    add.u32 %r44, %r34, %r43;
    cvt.u64.u32 %rd35, %r44;
    add.s64 %rd36, %rd10, %rd35;
    ld.global.u8 %r45, [%rd36];
    shr.u32 %r46, %r45, 4;
    cvt.rn.f32.u32 %f11, %r46;
    mul.rn.f32 %f12, %f6, %f11;
    sub.rn.f32 %f12, %f12, %f7;

    add.u32 %r47, %r37, 32;
    add.u32 %r47, %r47, %r43;
    mul.wide.u32 %rd37, %r47, 4;
    add.s64 %rd38, %rd2, %rd37;
    ld.global.f32 %f13, [%rd38];
    fma.rn.f32 %f1, %f13, %f12, %f1;

    add.u32 %r43, %r43, 1;
    bra HIGH_LOOP;

GROUP_NEXT:
    add.u32 %r11, %r11, 1;
    bra GROUP_LOOP;

NEXT_BLOCK:
    add.u32 %r9, %r9, 1;
    bra BLOCK_LOOP;

STORE:
    mul.wide.u32 %rd39, %r6, 4;
    add.s64 %rd40, %rd3, %rd39;
    st.global.f32 [%rd40], %f1;

DONE:
    ret;
}
PTX

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

OptionParser.parse do |p|
  p.banner = "Usage: cuda_q4k_gemv_probe [--model PATH] [--tensor NAME] [--seed N] [--reps N] [--warmup N] [--block N]"
  p.on("--model PATH", "Q4_K GGUF model path") { |v| model = v }
  p.on("--tensor NAME", "Q4_K tensor name") { |v| tensor_name = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed kernel launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--block N", "CUDA block size") { |v| block = v.to_u32 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "block must be positive" unless block > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

gguf = ML::GGUF::GGUFFile.new(model)
info = gguf.tensor(tensor_name) || raise "missing tensor #{tensor_name.inspect}"
raise "expected Q4_K tensor, got #{info.type.name}" unless info.type.q4_k?
raise "expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2

in_dim = info.dims[0].to_i32
out_dim = info.dims[1].to_i32
raise "Q4_K GEMV requires in_dim multiple of 256, got #{in_dim}" unless in_dim % 256 == 0

w_raw = gguf.read_tensor_raw(info)
rng = Random.new(seed)
x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
zero_bias = Array(Float32).new(out_dim, 0.0_f32)

cpu_t0 = Time.instant
cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, zero_bias)
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
d_w = 0_u64
d_x = 0_u64
d_out = 0_u64

begin
  cuda! LibCUDAQ4K.cuModuleLoadData(pointerof(mod), PTX.to_unsafe.as(Void*)), "cuModuleLoadData"
  cuda! LibCUDAQ4K.cuModuleGetFunction(pointerof(fn), mod, "q4_k_gemv_scalar_f32"), "cuModuleGetFunction"

  gpu_out = Array(Float32).new(out_dim, 0.0_f32)
  raw_size = w_raw.size.to_u64
  x_size = bytesize_f32(in_dim)
  out_size = bytesize_f32(out_dim)
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_w), raw_size), "cuMemAlloc(w)"
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_x), x_size), "cuMemAlloc(x)"
  cuda! LibCUDAQ4K.cuMemAlloc_v2(pointerof(d_out), out_size), "cuMemAlloc(out)"
  cuda! LibCUDAQ4K.cuMemcpyHtoD_v2(d_w, w_raw.to_unsafe.as(Void*), raw_size), "cuMemcpyHtoD(w)"
  cuda! LibCUDAQ4K.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), x_size), "cuMemcpyHtoD(x)"

  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(in_dim_u32).as(Void*)
  params[4] = pointerof(out_dim_u32).as(Void*)

  grid = ((out_dim + block.to_i - 1) // block.to_i).to_u32
  warmup.times do
    cuda! LibCUDAQ4K.cuLaunchKernel(fn, grid, 1_u32, 1_u32, block, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel(warmup)"
  end
  cuda! LibCUDAQ4K.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0

  gpu_t0 = Time.instant
  reps.times do
    cuda! LibCUDAQ4K.cuLaunchKernel(fn, grid, 1_u32, 1_u32, block, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel"
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
  LibCUDAQ4K.cuMemFree_v2(d_out) unless d_out == 0_u64
  LibCUDAQ4K.cuModuleUnload(mod) unless mod.null?
  LibCUDAQ4K.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
