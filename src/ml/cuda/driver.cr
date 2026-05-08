@[Link(ldflags: "-lcuda")]
lib LibCUDADriver
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

module ML::CUDA
  alias DevicePtr = LibCUDADriver::CUdeviceptr

  def self.check!(code : Int32, what : String) : Nil
    raise "#{what} failed with CUDA error #{code}" unless code == 0
  end

  class Context
    getter device_name : String
    getter compute_capability_major : Int32
    getter compute_capability_minor : Int32

    def self.create(ordinal : Int32 = 0) : self
      ML::CUDA.check! LibCUDADriver.cuInit(0_u32), "cuInit"
      device = uninitialized LibCUDADriver::CUdevice
      ML::CUDA.check! LibCUDADriver.cuDeviceGet(pointerof(device), ordinal), "cuDeviceGet"

      name_buf = Bytes.new(256)
      ML::CUDA.check! LibCUDADriver.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, device), "cuDeviceGetName"
      device_name = String.new(name_buf.to_unsafe).strip

      cc_major = uninitialized Int32
      cc_minor = uninitialized Int32
      ML::CUDA.check! LibCUDADriver.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), device), "cuDeviceComputeCapability"

      handle = Pointer(Void).null
      ML::CUDA.check! LibCUDADriver.cuCtxCreate_v2(pointerof(handle), 0_u32, device), "cuCtxCreate"
      new(handle, device_name, cc_major, cc_minor)
    end

    def initialize(@handle : Void*, @device_name : String, @compute_capability_major : Int32, @compute_capability_minor : Int32)
      @closed = false
    end

    def close : Nil
      return if @closed

      LibCUDADriver.cuCtxDestroy_v2(@handle) unless @handle.null?
      @closed = true
    end
  end

  class DeviceBuffer
    getter ptr : DevicePtr
    getter bytesize : LibC::SizeT

    def initialize(@bytesize : LibC::SizeT)
      @ptr = 0_u64
      ML::CUDA.check! LibCUDADriver.cuMemAlloc_v2(pointerof(@ptr), @bytesize), "cuMemAlloc"
      @closed = false
    end

    def close : Nil
      return if @closed

      LibCUDADriver.cuMemFree_v2(@ptr) unless @ptr == 0_u64
      @closed = true
    end
  end

  class CUDAModule
    def self.load(ptx : String, label : String) : self
      handle = Pointer(Void).null
      ML::CUDA.check! LibCUDADriver.cuModuleLoadData(pointerof(handle), ptx.to_unsafe.as(Void*)), "cuModuleLoadData(#{label})"
      new(handle)
    end

    def initialize(@handle : Void*)
      @closed = false
    end

    def function(name : String) : KernelFunction
      fn = Pointer(Void).null
      ML::CUDA.check! LibCUDADriver.cuModuleGetFunction(pointerof(fn), @handle, name), "cuModuleGetFunction(#{name})"
      KernelFunction.new(fn, name)
    end

    def close : Nil
      return if @closed

      LibCUDADriver.cuModuleUnload(@handle) unless @handle.null?
      @closed = true
    end
  end

  class KernelFunction
    getter handle : LibCUDADriver::CUfunction
    getter name : String

    def initialize(@handle : LibCUDADriver::CUfunction, @name : String)
    end
  end

  def self.copy_htod!(dst : DevicePtr, src : Void*, bytesize : LibC::SizeT, what : String) : Nil
    check! LibCUDADriver.cuMemcpyHtoD_v2(dst, src, bytesize), "cuMemcpyHtoD(#{what})"
  end

  def self.copy_dtoh!(dst : Void*, src : DevicePtr, bytesize : LibC::SizeT, what : String) : Nil
    check! LibCUDADriver.cuMemcpyDtoH_v2(dst, src, bytesize), "cuMemcpyDtoH(#{what})"
  end

  def self.synchronize!(what : String = "cuCtxSynchronize") : Nil
    check! LibCUDADriver.cuCtxSynchronize, what
  end

  def self.launch!(fn : KernelFunction,
                   grid_x : UInt32, grid_y : UInt32, grid_z : UInt32,
                   block_x : UInt32, block_y : UInt32, block_z : UInt32,
                   params : Void**, label : String,
                   shared_mem_bytes : UInt32 = 0_u32) : Nil
    check! LibCUDADriver.cuLaunchKernel(fn.handle, grid_x, grid_y, grid_z,
      block_x, block_y, block_z, shared_mem_bytes, Pointer(Void).null,
      params, Pointer(Void*).null), "cuLaunchKernel(#{label})"
  end

  class ResidentSequenceRunner
    getter tokens : Int32

    def initialize(@tokens : Int32,
                   @upload_weights : Proc(Nil),
                   @reset_sequence : Proc(Nil),
                   @run_token : Proc(Int32, Nil),
                   @read_outputs : Proc(Nil)? = nil)
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
    end

    def upload_weights : Nil
      @upload_weights.call
    end

    def reset_sequence : Nil
      @reset_sequence.call
    end

    def run_sequence : Nil
      @tokens.times { |tok| @run_token.call(tok) }
    end

    def run_repeated(reps : Int32) : Int32
      raise ArgumentError.new("reps must be positive") unless reps > 0

      reps.times { run_sequence }
      reps * @tokens
    end

    def read_outputs : Nil
      @read_outputs.try(&.call)
    end
  end
end
