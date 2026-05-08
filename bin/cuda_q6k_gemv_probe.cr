# Minimal CUDA Q6_K GEMV probe for the Linux/CUDA backend boundary.
#
# Loads one real Q6_K tensor from a GGUF file, runs a Crystal-driven CUDA
# Driver API kernel over the raw GGUF block layout, and compares against the
# existing CPU QuantMatmul reference.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"

@[Link(ldflags: "-lcuda")]
lib LibCUDAQ6K
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

.visible .entry q6_k_gemv_scalar_f32(
    .param .u64 w_raw,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<6>;
    .reg .b16 %h<2>;
    .reg .b32 %r<80>;
    .reg .b64 %rd<60>;
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
    mul.lo.u32 %r8, %r7, 210;  // row_bytes
    mul.wide.u32 %rd4, %r6, %r8;
    add.s64 %rd5, %rd1, %rd4;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r9, 0;

BLOCK_LOOP:
    setp.ge.u32 %p1, %r9, %r7;
    @%p1 bra STORE;

    mul.lo.u32 %r10, %r9, 210;
    cvt.u64.u32 %rd6, %r10;
    add.s64 %rd7, %rd5, %rd6;  // block base
    add.s64 %rd8, %rd7, 128;   // qh base
    add.s64 %rd9, %rd7, 192;   // scales base
    add.s64 %rd10, %rd7, 208;  // d base
    ld.global.u16 %h1, [%rd10];
    cvt.f32.f16 %f2, %h1;

    mov.u32 %r11, 0;           // n_iter 0..1

N_LOOP:
    setp.ge.u32 %p2, %r11, 2;
    @%p2 bra NEXT_BLOCK;

    mul.lo.u32 %r12, %r11, 64; // ql offset
    mul.lo.u32 %r13, %r11, 32; // qh offset
    shl.b32 %r14, %r11, 7;     // x offset n*128
    shl.b32 %r15, %r11, 3;     // scales offset n*8
    shl.b32 %r16, %r9, 8;      // block input base
    add.u32 %r17, %r16, %r14;
    mov.u32 %r18, 0;

L_LOOP:
    setp.ge.u32 %p3, %r18, 32;
    @%p3 bra NEXT_N;

    add.u32 %r19, %r12, %r18;
    cvt.u64.u32 %rd11, %r19;
    add.s64 %rd12, %rd7, %rd11;
    ld.global.u8 %r20, [%rd12];       // ql low/high source

    add.u32 %r21, %r19, 32;
    cvt.u64.u32 %rd13, %r21;
    add.s64 %rd14, %rd7, %rd13;
    ld.global.u8 %r22, [%rd14];       // ql second source

    add.u32 %r23, %r13, %r18;
    cvt.u64.u32 %rd15, %r23;
    add.s64 %rd16, %rd8, %rd15;
    ld.global.u8 %r24, [%rd16];       // qh

    and.b32 %r25, %r20, 15;
    and.b32 %r26, %r24, 3;
    shl.b32 %r26, %r26, 4;
    or.b32 %r25, %r25, %r26;
    add.s32 %r25, %r25, -32;

    and.b32 %r27, %r22, 15;
    shr.u32 %r28, %r24, 2;
    and.b32 %r28, %r28, 3;
    shl.b32 %r28, %r28, 4;
    or.b32 %r27, %r27, %r28;
    add.s32 %r27, %r27, -32;

    shr.u32 %r29, %r20, 4;
    shr.u32 %r30, %r24, 4;
    and.b32 %r30, %r30, 3;
    shl.b32 %r30, %r30, 4;
    or.b32 %r29, %r29, %r30;
    add.s32 %r29, %r29, -32;

    shr.u32 %r31, %r22, 4;
    shr.u32 %r32, %r24, 6;
    and.b32 %r32, %r32, 3;
    shl.b32 %r32, %r32, 4;
    or.b32 %r31, %r31, %r32;
    add.s32 %r31, %r31, -32;

    shr.u32 %r33, %r18, 4;           // is = lane / 16
    add.u32 %r34, %r15, %r33;
    cvt.u64.u32 %rd17, %r34;
    add.s64 %rd18, %rd9, %rd17;
    ld.global.s8 %r35, [%rd18];
    add.u32 %r36, %r34, 2;
    cvt.u64.u32 %rd19, %r36;
    add.s64 %rd20, %rd9, %rd19;
    ld.global.s8 %r37, [%rd20];
    add.u32 %r38, %r34, 4;
    cvt.u64.u32 %rd21, %r38;
    add.s64 %rd22, %rd9, %rd21;
    ld.global.s8 %r39, [%rd22];
    add.u32 %r40, %r34, 6;
    cvt.u64.u32 %rd23, %r40;
    add.s64 %rd24, %rd9, %rd23;
    ld.global.s8 %r41, [%rd24];

    cvt.rn.f32.s32 %f3, %r25;
    cvt.rn.f32.s32 %f4, %r27;
    cvt.rn.f32.s32 %f5, %r29;
    cvt.rn.f32.s32 %f6, %r31;
    cvt.rn.f32.s32 %f7, %r35;
    cvt.rn.f32.s32 %f8, %r37;
    cvt.rn.f32.s32 %f9, %r39;
    cvt.rn.f32.s32 %f10, %r41;

    mul.rn.f32 %f11, %f2, %f7;
    mul.rn.f32 %f11, %f11, %f3;
    mul.rn.f32 %f12, %f2, %f8;
    mul.rn.f32 %f12, %f12, %f4;
    mul.rn.f32 %f13, %f2, %f9;
    mul.rn.f32 %f13, %f13, %f5;
    mul.rn.f32 %f14, %f2, %f10;
    mul.rn.f32 %f14, %f14, %f6;

    add.u32 %r42, %r17, %r18;
    mul.wide.u32 %rd25, %r42, 4;
    add.s64 %rd26, %rd2, %rd25;
    ld.global.f32 %f15, [%rd26];
    fma.rn.f32 %f1, %f15, %f11, %f1;

    add.u32 %r43, %r42, 32;
    mul.wide.u32 %rd27, %r43, 4;
    add.s64 %rd28, %rd2, %rd27;
    ld.global.f32 %f16, [%rd28];
    fma.rn.f32 %f1, %f16, %f12, %f1;

    add.u32 %r44, %r42, 64;
    mul.wide.u32 %rd29, %r44, 4;
    add.s64 %rd30, %rd2, %rd29;
    ld.global.f32 %f17, [%rd30];
    fma.rn.f32 %f1, %f17, %f13, %f1;

    add.u32 %r45, %r42, 96;
    mul.wide.u32 %rd31, %r45, 4;
    add.s64 %rd32, %rd2, %rd31;
    ld.global.f32 %f18, [%rd32];
    fma.rn.f32 %f1, %f18, %f14, %f1;

    add.u32 %r18, %r18, 1;
    bra L_LOOP;

NEXT_N:
    add.u32 %r11, %r11, 1;
    bra N_LOOP;

NEXT_BLOCK:
    add.u32 %r9, %r9, 1;
    bra BLOCK_LOOP;

STORE:
    mul.wide.u32 %rd33, %r6, 4;
    add.s64 %rd34, %rd3, %rd33;
    st.global.f32 [%rd34], %f1;

DONE:
    ret;
}

.visible .entry q6_k_gemv_warp4_f32(
    .param .u64 w_raw,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<6>;
    .reg .b16 %h<2>;
    .reg .b32 %r<80>;
    .reg .b64 %rd<60>;
    .reg .f32 %f<24>;
    .shared .align 4 .b8 smem[512];

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [x];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;
    shr.u32 %r6, %r3, 5;
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra WARP_DONE;

    shr.u32 %r9, %r1, 8;
    mul.lo.u32 %r10, %r9, 210;
    mul.wide.u32 %rd4, %r8, %r10;
    add.s64 %rd5, %rd1, %rd4;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;

WARP_BLOCK_LOOP:
    setp.ge.u32 %p1, %r11, %r9;
    @%p1 bra WARP_REDUCE_PREP;

    mul.lo.u32 %r12, %r11, 210;
    cvt.u64.u32 %rd6, %r12;
    add.s64 %rd7, %rd5, %rd6;
    add.s64 %rd8, %rd7, 128;
    add.s64 %rd9, %rd7, 192;
    add.s64 %rd10, %rd7, 208;
    ld.global.u16 %h1, [%rd10];
    cvt.f32.f16 %f2, %h1;

    mov.u32 %r13, 0;

WARP_N_LOOP:
    setp.ge.u32 %p2, %r13, 2;
    @%p2 bra WARP_NEXT_BLOCK;

    mul.lo.u32 %r14, %r13, 64;
    mul.lo.u32 %r15, %r13, 32;
    shl.b32 %r16, %r13, 7;
    shl.b32 %r17, %r13, 3;
    shl.b32 %r18, %r11, 8;
    add.u32 %r19, %r18, %r16;

    add.u32 %r20, %r14, %r5;
    cvt.u64.u32 %rd11, %r20;
    add.s64 %rd12, %rd7, %rd11;
    ld.global.u8 %r21, [%rd12];
    add.u32 %r22, %r20, 32;
    cvt.u64.u32 %rd13, %r22;
    add.s64 %rd14, %rd7, %rd13;
    ld.global.u8 %r23, [%rd14];
    add.u32 %r24, %r15, %r5;
    cvt.u64.u32 %rd15, %r24;
    add.s64 %rd16, %rd8, %rd15;
    ld.global.u8 %r25, [%rd16];

    and.b32 %r26, %r21, 15;
    and.b32 %r27, %r25, 3;
    shl.b32 %r27, %r27, 4;
    or.b32 %r26, %r26, %r27;
    add.s32 %r26, %r26, -32;

    and.b32 %r28, %r23, 15;
    shr.u32 %r29, %r25, 2;
    and.b32 %r29, %r29, 3;
    shl.b32 %r29, %r29, 4;
    or.b32 %r28, %r28, %r29;
    add.s32 %r28, %r28, -32;

    shr.u32 %r30, %r21, 4;
    shr.u32 %r31, %r25, 4;
    and.b32 %r31, %r31, 3;
    shl.b32 %r31, %r31, 4;
    or.b32 %r30, %r30, %r31;
    add.s32 %r30, %r30, -32;

    shr.u32 %r32, %r23, 4;
    shr.u32 %r33, %r25, 6;
    and.b32 %r33, %r33, 3;
    shl.b32 %r33, %r33, 4;
    or.b32 %r32, %r32, %r33;
    add.s32 %r32, %r32, -32;

    shr.u32 %r34, %r5, 4;
    add.u32 %r35, %r17, %r34;
    cvt.u64.u32 %rd17, %r35;
    add.s64 %rd18, %rd9, %rd17;
    ld.global.s8 %r36, [%rd18];
    add.u32 %r37, %r35, 2;
    cvt.u64.u32 %rd19, %r37;
    add.s64 %rd20, %rd9, %rd19;
    ld.global.s8 %r38, [%rd20];
    add.u32 %r39, %r35, 4;
    cvt.u64.u32 %rd21, %r39;
    add.s64 %rd22, %rd9, %rd21;
    ld.global.s8 %r40, [%rd22];
    add.u32 %r41, %r35, 6;
    cvt.u64.u32 %rd23, %r41;
    add.s64 %rd24, %rd9, %rd23;
    ld.global.s8 %r42, [%rd24];

    cvt.rn.f32.s32 %f3, %r26;
    cvt.rn.f32.s32 %f4, %r28;
    cvt.rn.f32.s32 %f5, %r30;
    cvt.rn.f32.s32 %f6, %r32;
    cvt.rn.f32.s32 %f7, %r36;
    cvt.rn.f32.s32 %f8, %r38;
    cvt.rn.f32.s32 %f9, %r40;
    cvt.rn.f32.s32 %f10, %r42;

    mul.rn.f32 %f11, %f2, %f7;
    mul.rn.f32 %f11, %f11, %f3;
    mul.rn.f32 %f12, %f2, %f8;
    mul.rn.f32 %f12, %f12, %f4;
    mul.rn.f32 %f13, %f2, %f9;
    mul.rn.f32 %f13, %f13, %f5;
    mul.rn.f32 %f14, %f2, %f10;
    mul.rn.f32 %f14, %f14, %f6;

    add.u32 %r43, %r19, %r5;
    mul.wide.u32 %rd25, %r43, 4;
    add.s64 %rd26, %rd2, %rd25;
    ld.global.f32 %f15, [%rd26];
    fma.rn.f32 %f1, %f15, %f11, %f1;

    add.u32 %r44, %r43, 32;
    mul.wide.u32 %rd27, %r44, 4;
    add.s64 %rd28, %rd2, %rd27;
    ld.global.f32 %f16, [%rd28];
    fma.rn.f32 %f1, %f16, %f12, %f1;

    add.u32 %r45, %r43, 64;
    mul.wide.u32 %rd29, %r45, 4;
    add.s64 %rd30, %rd2, %rd29;
    ld.global.f32 %f17, [%rd30];
    fma.rn.f32 %f1, %f17, %f13, %f1;

    add.u32 %r46, %r43, 96;
    mul.wide.u32 %rd31, %r46, 4;
    add.s64 %rd32, %rd2, %rd31;
    ld.global.f32 %f18, [%rd32];
    fma.rn.f32 %f1, %f18, %f14, %f1;

    add.u32 %r13, %r13, 1;
    bra WARP_N_LOOP;

WARP_NEXT_BLOCK:
    add.u32 %r11, %r11, 1;
    bra WARP_BLOCK_LOOP;

WARP_REDUCE_PREP:
    shl.b32 %r47, %r3, 2;
    mov.u64 %rd33, smem;
    cvt.u64.u32 %rd34, %r47;
    add.s64 %rd35, %rd33, %rd34;
    st.shared.f32 [%rd35], %f1;
    bar.sync 0;

    setp.ne.u32 %p4, %r5, 0;
    @%p4 bra WARP_DONE;

    shl.b32 %r48, %r6, 7;
    mov.f32 %f19, 0f00000000;
    mov.u32 %r49, 0;

WARP_SUM_LOOP:
    setp.ge.u32 %p5, %r49, 32;
    @%p5 bra WARP_STORE;
    shl.b32 %r50, %r49, 2;
    add.u32 %r51, %r48, %r50;
    cvt.u64.u32 %rd36, %r51;
    add.s64 %rd37, %rd33, %rd36;
    ld.shared.f32 %f20, [%rd37];
    add.rn.f32 %f19, %f19, %f20;
    add.u32 %r49, %r49, 1;
    bra WARP_SUM_LOOP;

WARP_STORE:
    mul.wide.u32 %rd38, %r8, 4;
    add.s64 %rd39, %rd3, %rd38;
    st.global.f32 [%rd39], %f19;

WARP_DONE:
    ret;
}
PTX

DEFAULT_MODEL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TENSOR = "blk.0.ffn_down.weight"

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

OptionParser.parse do |p|
  p.banner = "Usage: cuda_q6k_gemv_probe [--model PATH] [--tensor NAME] [--seed N] [--kernel scalar|warp4] [--reps N] [--warmup N] [--block N]"
  p.on("--model PATH", "Q6_K GGUF model path") { |v| model = v }
  p.on("--tensor NAME", "Q6_K tensor name") { |v| tensor_name = v }
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
raise "expected Q6_K tensor, got #{info.type.name}" unless info.type.q6_k?
raise "expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2

in_dim = info.dims[0].to_i32
out_dim = info.dims[1].to_i32
raise "Q6_K GEMV requires in_dim multiple of 256, got #{in_dim}" unless in_dim % 256 == 0

w_raw = gguf.read_tensor_raw(info)
rng = Random.new(seed)
x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
zero_bias = Array(Float32).new(out_dim, 0.0_f32)

cpu_t0 = Time.instant
cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q6_K, out_dim, zero_bias)
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDAQ6K.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDAQ6K::CUdevice
cuda! LibCUDAQ6K.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"

name_buf = Bytes.new(256)
cuda! LibCUDAQ6K.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDAQ6K.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"

ctx = Pointer(Void).null
cuda! LibCUDAQ6K.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

mod = Pointer(Void).null
fn = Pointer(Void).null
d_w = 0_u64
d_x = 0_u64
d_out = 0_u64

begin
  cuda! LibCUDAQ6K.cuModuleLoadData(pointerof(mod), PTX.to_unsafe.as(Void*)), "cuModuleLoadData"
  kernel_fn = kernel == "warp4" ? "q6_k_gemv_warp4_f32" : "q6_k_gemv_scalar_f32"
  cuda! LibCUDAQ6K.cuModuleGetFunction(pointerof(fn), mod, kernel_fn), "cuModuleGetFunction"

  gpu_out = Array(Float32).new(out_dim, 0.0_f32)
  raw_size = w_raw.size.to_u64
  x_size = bytesize_f32(in_dim)
  out_size = bytesize_f32(out_dim)
  cuda! LibCUDAQ6K.cuMemAlloc_v2(pointerof(d_w), raw_size), "cuMemAlloc(w)"
  cuda! LibCUDAQ6K.cuMemAlloc_v2(pointerof(d_x), x_size), "cuMemAlloc(x)"
  cuda! LibCUDAQ6K.cuMemAlloc_v2(pointerof(d_out), out_size), "cuMemAlloc(out)"
  cuda! LibCUDAQ6K.cuMemcpyHtoD_v2(d_w, w_raw.to_unsafe.as(Void*), raw_size), "cuMemcpyHtoD(w)"
  cuda! LibCUDAQ6K.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), x_size), "cuMemcpyHtoD(x)"

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
    cuda! LibCUDAQ6K.cuLaunchKernel(fn, grid, 1_u32, 1_u32, launch_block, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel(warmup)"
  end
  cuda! LibCUDAQ6K.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0

  gpu_t0 = Time.instant
  reps.times do
    cuda! LibCUDAQ6K.cuLaunchKernel(fn, grid, 1_u32, 1_u32, launch_block, 1_u32, 1_u32,
      0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel"
  end
  cuda! LibCUDAQ6K.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps
  cuda! LibCUDAQ6K.cuMemcpyDtoH_v2(gpu_out.to_unsafe.as(Void*), d_out, out_size), "cuMemcpyDtoH(out)"

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
  puts "ok=#{cos >= 0.99999 && max_diff <= 1.0e-3_f32}"
ensure
  LibCUDAQ6K.cuMemFree_v2(d_w) unless d_w == 0_u64
  LibCUDAQ6K.cuMemFree_v2(d_x) unless d_x == 0_u64
  LibCUDAQ6K.cuMemFree_v2(d_out) unless d_out == 0_u64
  LibCUDAQ6K.cuModuleUnload(mod) unless mod.null?
  LibCUDAQ6K.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
