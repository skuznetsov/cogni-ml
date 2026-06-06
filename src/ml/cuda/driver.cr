@[Link(ldflags: "-lcuda")]
lib LibCUDADriver
  alias CUdevice = Int32
  alias CUcontext = Void*
  alias CUmodule = Void*
  alias CUfunction = Void*
  alias CUstream = Void*
  alias CUgraph = Void*
  alias CUgraphExec = Void*
  alias CUgraphNode = Void*
  alias CUdeviceptr = UInt64

  struct CUGraphInstantiateParams
    flags : UInt64
    h_upload_stream : CUstream
    h_err_node_out : CUgraphNode
    result_out : UInt32
  end

  fun cuInit(flags : UInt32) : Int32
  fun cuDeviceGet(device : CUdevice*, ordinal : Int32) : Int32
  fun cuDeviceGetName(name : UInt8*, len : Int32, dev : CUdevice) : Int32
  fun cuDeviceComputeCapability(major : Int32*, minor : Int32*, dev : CUdevice) : Int32
  fun cuCtxCreate_v2(ctx : CUcontext*, flags : UInt32, dev : CUdevice) : Int32
  fun cuCtxDestroy_v2(ctx : CUcontext) : Int32
  fun cuModuleLoadData(mod : CUmodule*, image : Void*) : Int32
  fun cuModuleUnload(mod : CUmodule) : Int32
  fun cuModuleGetFunction(fn : CUfunction*, mod : CUmodule, name : UInt8*) : Int32
  fun cuFuncGetAttribute(pi : Int32*, attrib : Int32, hfunc : CUfunction) : Int32
  fun cuStreamCreate(stream : CUstream*, flags : UInt32) : Int32
  fun cuStreamDestroy_v2(stream : CUstream) : Int32
  fun cuStreamSynchronize(stream : CUstream) : Int32
  fun cuStreamBeginCapture(stream : CUstream, mode : Int32) : Int32
  fun cuStreamEndCapture(stream : CUstream, graph : CUgraph*) : Int32
  fun cuGraphInstantiateWithFlags(exec : CUgraphExec*, graph : CUgraph, flags : UInt64) : Int32
  fun cuGraphInstantiateWithParams(exec : CUgraphExec*, graph : CUgraph, params : CUGraphInstantiateParams*) : Int32
  fun cuGraphUpload(exec : CUgraphExec, stream : CUstream) : Int32
  fun cuGraphLaunch(exec : CUgraphExec, stream : CUstream) : Int32
  fun cuGraphDestroy(graph : CUgraph) : Int32
  fun cuGraphExecDestroy(exec : CUgraphExec) : Int32
  fun cuMemAlloc_v2(dptr : CUdeviceptr*, bytesize : LibC::SizeT) : Int32
  fun cuMemFree_v2(dptr : CUdeviceptr) : Int32
  fun cuMemcpyHtoD_v2(dst : CUdeviceptr, src : Void*, bytesize : LibC::SizeT) : Int32
  fun cuMemcpyDtoH_v2(dst : Void*, src : CUdeviceptr, bytesize : LibC::SizeT) : Int32
  fun cuMemcpyDtoD_v2(dst : CUdeviceptr, src : CUdeviceptr, bytesize : LibC::SizeT) : Int32
  fun cuLaunchKernel(fn : CUfunction, grid_x : UInt32, grid_y : UInt32, grid_z : UInt32,
                     block_x : UInt32, block_y : UInt32, block_z : UInt32,
                     shared_mem_bytes : UInt32, stream : Void*,
                     kernel_params : Void**, extra : Void**) : Int32
  fun cuCtxSynchronize : Int32
end

module ML::CUDA
  alias DevicePtr = LibCUDADriver::CUdeviceptr
  @@current_stream = Pointer(Void).null

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

    def attribute(attr : FunctionAttribute) : Int32
      value = uninitialized Int32
      ML::CUDA.check! LibCUDADriver.cuFuncGetAttribute(pointerof(value), attr.value, @handle),
        "cuFuncGetAttribute(#{@name}, #{attr})"
      value
    end
  end

  enum FunctionAttribute
    MaxThreadsPerBlock            = 0
    SharedSizeBytes               = 1
    ConstSizeBytes                = 2
    LocalSizeBytes                = 3
    NumRegs                       = 4
    PtxVersion                    = 5
    BinaryVersion                 = 6
    CacheModeCa                   = 7
    MaxDynamicSharedSizeBytes     = 8
    PreferredSharedMemoryCarveout = 9
  end

  class CUDAStream
    getter handle : LibCUDADriver::CUstream

    def initialize(flags : UInt32 = 0_u32)
      @handle = Pointer(Void).null
      ML::CUDA.check! LibCUDADriver.cuStreamCreate(pointerof(@handle), flags), "cuStreamCreate"
      @closed = false
    end

    def synchronize : Nil
      ML::CUDA.check! LibCUDADriver.cuStreamSynchronize(@handle), "cuStreamSynchronize"
    end

    def begin_capture(mode : Int32 = 1) : Nil
      ML::CUDA.check! LibCUDADriver.cuStreamBeginCapture(@handle, mode), "cuStreamBeginCapture"
    end

    def end_capture : CUDAGraph
      graph = Pointer(Void).null
      ML::CUDA.check! LibCUDADriver.cuStreamEndCapture(@handle, pointerof(graph)), "cuStreamEndCapture"
      CUDAGraph.new(graph)
    end

    def close : Nil
      return if @closed

      LibCUDADriver.cuStreamDestroy_v2(@handle) unless @handle.null?
      @closed = true
    end
  end

  class CUDAGraph
    DEVICE_LAUNCH_FLAG = 4_u64

    def initialize(@handle : LibCUDADriver::CUgraph)
      @closed = false
    end

    def instantiate : CUDAGraphExec
      exec = Pointer(Void).null
      ML::CUDA.check! LibCUDADriver.cuGraphInstantiateWithFlags(pointerof(exec), @handle, 0_u64), "cuGraphInstantiateWithFlags"
      CUDAGraphExec.new(exec)
    end

    def instantiate_device_launch(upload_stream : CUDAStream? = nil) : CUDAGraphExec
      exec = Pointer(Void).null
      params = LibCUDADriver::CUGraphInstantiateParams.new
      params.flags = DEVICE_LAUNCH_FLAG
      params.h_upload_stream = upload_stream ? upload_stream.not_nil!.handle : Pointer(Void).null
      params.h_err_node_out = Pointer(Void).null
      params.result_out = 0_u32
      ML::CUDA.check! LibCUDADriver.cuGraphInstantiateWithParams(pointerof(exec), @handle, pointerof(params)),
        "cuGraphInstantiateWithParams(device_launch)"
      CUDAGraphExec.new(exec)
    end

    def close : Nil
      return if @closed

      LibCUDADriver.cuGraphDestroy(@handle) unless @handle.null?
      @closed = true
    end
  end

  class CUDAGraphExec
    def initialize(@handle : LibCUDADriver::CUgraphExec)
      @closed = false
    end

    def launch(stream : CUDAStream) : Nil
      ML::CUDA.check! LibCUDADriver.cuGraphLaunch(@handle, stream.handle), "cuGraphLaunch"
    end

    def upload(stream : CUDAStream) : Nil
      ML::CUDA.check! LibCUDADriver.cuGraphUpload(@handle, stream.handle), "cuGraphUpload"
    end

    def close : Nil
      return if @closed

      LibCUDADriver.cuGraphExecDestroy(@handle) unless @handle.null?
      @closed = true
    end
  end

  def self.with_stream(stream : CUDAStream) : Nil
    previous = @@current_stream
    @@current_stream = stream.handle
    begin
      yield
    ensure
      @@current_stream = previous
    end
  end

  def self.copy_htod!(dst : DevicePtr, src : Void*, bytesize : LibC::SizeT, what : String) : Nil
    check! LibCUDADriver.cuMemcpyHtoD_v2(dst, src, bytesize), "cuMemcpyHtoD(#{what})"
  end

  def self.copy_dtoh!(dst : Void*, src : DevicePtr, bytesize : LibC::SizeT, what : String) : Nil
    check! LibCUDADriver.cuMemcpyDtoH_v2(dst, src, bytesize), "cuMemcpyDtoH(#{what})"
  end

  def self.copy_dtod!(dst : DevicePtr, src : DevicePtr, bytesize : LibC::SizeT, what : String) : Nil
    check! LibCUDADriver.cuMemcpyDtoD_v2(dst, src, bytesize), "cuMemcpyDtoD(#{what})"
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
      block_x, block_y, block_z, shared_mem_bytes, @@current_stream,
      params, Pointer(Void*).null), "cuLaunchKernel(#{label})"
  end

  class ResidentSequenceRunner
    getter tokens : Int32

    def initialize(@tokens : Int32,
                   @upload_weights : Proc(Nil),
                   @reset_sequence : Proc(Nil),
                   @run_token : Proc(Int32, Nil),
                   @read_outputs : Proc(Nil)? = nil,
                   @run_sequence_override : Proc(Int32, Nil)? = nil)
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      @active_tokens = @tokens
    end

    def active_tokens : Int32
      @active_tokens
    end

    def active_tokens=(@active_tokens : Int32) : Int32
      raise ArgumentError.new("active_tokens must be positive") unless @active_tokens > 0
      raise ArgumentError.new("active_tokens must be <= tokens") unless @active_tokens <= @tokens

      @active_tokens
    end

    def reset_active_tokens : Nil
      @active_tokens = @tokens
    end

    def upload_weights : Nil
      @upload_weights.call
    end

    def reset_sequence : Nil
      @reset_sequence.call
    end

    def run_sequence : Nil
      if override = @run_sequence_override
        override.call(@active_tokens)
      else
        @active_tokens.times { |tok| @run_token.call(tok) }
      end
    end

    def run_repeated(reps : Int32) : Int32
      raise ArgumentError.new("reps must be positive") unless reps > 0

      reps.times { run_sequence }
      reps * @active_tokens
    end

    def read_outputs : Nil
      @read_outputs.try(&.call)
    end
  end
end
