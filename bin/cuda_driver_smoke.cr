# Minimal Crystal -> CUDA Driver API smoke.
#
# This intentionally does not depend on the Qwen/Metal stack. It proves that a
# Crystal binary can link libcuda, load embedded PTX, launch a kernel, and copy
# results back on the Linux CUDA host.

@[Link(ldflags: "-lcuda")]
lib LibCUDA
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

.visible .entry vadd_f32(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .pred %p;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r4, [n];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r5, %r2, %r3, %r1;
    setp.ge.u32 %p, %r5, %r4;
    @%p bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
PTX

def cuda!(code : Int32, what : String) : Nil
  raise "#{what} failed with CUDA error #{code}" unless code == 0
end

def bytesize(elements : Int32) : LibC::SizeT
  (elements * sizeof(Float32)).to_u64
end

n = (ARGV[0]? || "1024").to_i
raise ArgumentError.new("N must be positive") unless n > 0

cuda! LibCUDA.cuInit(0_u32), "cuInit"

dev = uninitialized LibCUDA::CUdevice
cuda! LibCUDA.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"

name_buf = Bytes.new(256)
cuda! LibCUDA.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip

cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDA.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"

ctx = Pointer(Void).null
cuda! LibCUDA.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

mod = Pointer(Void).null
fn = Pointer(Void).null
d_a = 0_u64
d_b = 0_u64
d_c = 0_u64

begin
  cuda! LibCUDA.cuModuleLoadData(pointerof(mod), PTX.to_unsafe.as(Void*)), "cuModuleLoadData"
  cuda! LibCUDA.cuModuleGetFunction(pointerof(fn), mod, "vadd_f32"), "cuModuleGetFunction"

  a = Array(Float32).new(n) { |i| i.to_f32 }
  b = Array(Float32).new(n) { |i| (1000 - i).to_f32 }
  c = Array(Float32).new(n, 0.0_f32)
  n32 = n.to_u32

  size = bytesize(n)
  cuda! LibCUDA.cuMemAlloc_v2(pointerof(d_a), size), "cuMemAlloc(a)"
  cuda! LibCUDA.cuMemAlloc_v2(pointerof(d_b), size), "cuMemAlloc(b)"
  cuda! LibCUDA.cuMemAlloc_v2(pointerof(d_c), size), "cuMemAlloc(c)"
  cuda! LibCUDA.cuMemcpyHtoD_v2(d_a, a.to_unsafe.as(Void*), size), "cuMemcpyHtoD(a)"
  cuda! LibCUDA.cuMemcpyHtoD_v2(d_b, b.to_unsafe.as(Void*), size), "cuMemcpyHtoD(b)"

  params = Pointer(Void*).malloc(4)
  params[0] = pointerof(d_a).as(Void*)
  params[1] = pointerof(d_b).as(Void*)
  params[2] = pointerof(d_c).as(Void*)
  params[3] = pointerof(n32).as(Void*)

  block = 128_u32
  grid = ((n + block.to_i - 1) // block.to_i).to_u32
  cuda! LibCUDA.cuLaunchKernel(fn, grid, 1_u32, 1_u32, block, 1_u32, 1_u32,
    0_u32, Pointer(Void).null, params, Pointer(Void*).null), "cuLaunchKernel"
  cuda! LibCUDA.cuCtxSynchronize, "cuCtxSynchronize"
  cuda! LibCUDA.cuMemcpyDtoH_v2(c.to_unsafe.as(Void*), d_c, size), "cuMemcpyDtoH(c)"

  max_err = 0.0_f32
  n.times do |i|
    expected = a[i] + b[i]
    err = (c[i] - expected).abs
    max_err = err if err > max_err
  end

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "n=#{n}"
  puts "first=#{c[0]}"
  puts "last=#{c[n - 1]}"
  puts "max_err=#{max_err}"
  puts "ok=#{max_err <= 0.0_f32}"
ensure
  LibCUDA.cuMemFree_v2(d_a) unless d_a == 0_u64
  LibCUDA.cuMemFree_v2(d_b) unless d_b == 0_u64
  LibCUDA.cuMemFree_v2(d_c) unless d_c == 0_u64
  LibCUDA.cuModuleUnload(mod) unless mod.null?
  LibCUDA.cuCtxDestroy_v2(ctx) unless ctx.null?
end
