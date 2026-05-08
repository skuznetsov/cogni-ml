@[Link(ldflags: "-lcuda")]
lib LibCUDADriver
  alias CUdevice = Int32
  alias CUcontext = Void*
  alias CUdeviceptr = UInt64

  fun cuInit(flags : UInt32) : Int32
  fun cuDeviceGet(device : CUdevice*, ordinal : Int32) : Int32
  fun cuDeviceGetName(name : UInt8*, len : Int32, dev : CUdevice) : Int32
  fun cuDeviceComputeCapability(major : Int32*, minor : Int32*, dev : CUdevice) : Int32
  fun cuCtxCreate_v2(ctx : CUcontext*, flags : UInt32, dev : CUdevice) : Int32
  fun cuCtxDestroy_v2(ctx : CUcontext) : Int32
  fun cuMemAlloc_v2(dptr : CUdeviceptr*, bytesize : LibC::SizeT) : Int32
  fun cuMemFree_v2(dptr : CUdeviceptr) : Int32
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
end
