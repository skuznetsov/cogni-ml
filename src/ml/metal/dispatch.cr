{% if flag?(:cpu_only) %}
# CPU-only stubs (Metal disabled)
module ML
  module Metal
    enum BufferAccess
      Read
      Write
      ReadWrite
    end

    struct ComputeEncoder
      def set_pipeline(pipeline : ComputePipeline) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_buffer(buffer : MetalBuffer, index : Int32, access : BufferAccess = BufferAccess::Read, offset : Int64 = 0) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_tensor(tensor : Tensor, index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_bytes(data : Pointer(Void), length : Int32, index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_value(value : Float32, index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_value(value : Int32, index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_value(value : UInt32, index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_value(value : StaticArray(Float32, 3), index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_value(value : StaticArray(Float32, 4), index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def dispatch(grid_size : {Int32, Int32, Int32}, threadgroup_size : {Int32, Int32, Int32}) : self
        raise "Metal disabled (cpu_only)"
      end

      def dispatch_1d(count : Int32, threadgroup_size : Int32 = 256) : self
        raise "Metal disabled (cpu_only)"
      end

      def dispatch_2d(width : Int32, height : Int32, threadgroup_size : {Int32, Int32} = {16, 16}) : self
        raise "Metal disabled (cpu_only)"
      end

      def dispatch_3d(width : Int32, height : Int32, depth : Int32, threadgroup_size : {Int32, Int32, Int32} = {8, 8, 8}) : self
        raise "Metal disabled (cpu_only)"
      end

      def set_threadgroup_memory(length : Int32, index : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def end_encoding : Nil
      end
    end

    struct BlitEncoder
      def fill_buffer(buffer : MetalBuffer, value : UInt8, offset : Int32, length : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def copy_buffer(src : MetalBuffer, src_offset : Int32, dst : MetalBuffer, dst_offset : Int32, size : Int32) : self
        raise "Metal disabled (cpu_only)"
      end

      def end_encoding : Nil
      end
    end

    module Dispatch
      extend self

      def execute(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : Nil
        raise "Metal disabled (cpu_only)"
      end

      def dispatch(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : Nil
        raise "Metal disabled (cpu_only)"
      end

      def execute_sequence(&block : CommandBuffer -> Nil) : Nil
        raise "Metal disabled (cpu_only)"
      end

      def execute_async(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : CommandBuffer
        raise "Metal disabled (cpu_only)"
      end

      def execute_blit(&block : BlitEncoder -> Nil) : Nil
        raise "Metal disabled (cpu_only)"
      end

      def execute_blit_async(&block : BlitEncoder -> Nil) : CommandBuffer
        raise "Metal disabled (cpu_only)"
      end

      def optimal_threadgroup_1d(count : Int32, pipeline : ComputePipeline? = nil) : Int32
        1
      end

      def optimal_threadgroup_2d(width : Int32, height : Int32, pipeline : ComputePipeline? = nil) : {Int32, Int32}
        {1, 1}
      end
    end

    module KernelParams
      struct ElementwiseParams
        property count : UInt32
        property alpha : Float32
        property beta : Float32

        def initialize(@count : UInt32, @alpha : Float32 = 1.0_f32, @beta : Float32 = 0.0_f32)
        end
      end

      struct MatmulParams
        property m : UInt32
        property n : UInt32
        property k : UInt32
        property alpha : Float32
        property beta : Float32

        def initialize(@m : UInt32, @n : UInt32, @k : UInt32, @alpha : Float32 = 1.0_f32, @beta : Float32 = 0.0_f32)
        end
      end

      struct ReductionParams
        property count : UInt32
        property stride : UInt32

        def initialize(@count : UInt32, @stride : UInt32 = 1)
        end
      end

      struct Grid2DParams
        property width : UInt32
        property height : UInt32
        property channels : UInt32

        def initialize(@width : UInt32, @height : UInt32, @channels : UInt32 = 1)
        end
      end
    end
  end
end
{% else %}
# Kernel dispatch utilities for Metal compute shaders
# Handles grid/threadgroup sizing and buffer binding

require "./device"
require "../core/tensor"

module ML
  module Metal
    enum BufferAccess
      Read
      Write
      ReadWrite
    end

    # Compute encoder for setting up kernel dispatch
    struct ComputeEncoder
      @cmd_buffer : Pointer(Void)
      @encoder : Pointer(Void)
      @pipeline : ComputePipeline?

      def initialize(command_buffer : CommandBuffer, concurrent : Bool = false)
        @cmd_buffer = command_buffer.handle
        @encoder = if concurrent
                     MetalDispatchFFI.create_concurrent_compute_encoder(@cmd_buffer)
                   else
                     MetalDispatchFFI.create_compute_encoder(@cmd_buffer)
                   end
        raise "Failed to create compute encoder" if @encoder.null?
        @pipeline = nil
      end

      # Insert memory barrier between dependent dispatches (concurrent encoder only)
      def memory_barrier : self
        MetalDispatchFFI.encoder_memory_barrier(@encoder)
        self
      end

      def set_pipeline(pipeline : ComputePipeline) : self
        @pipeline = pipeline
        MetalDispatchFFI.encoder_set_pipeline(@encoder, pipeline.handle)
        self
      end

      # Bind Metal buffer at index (access/length/partition accepted for GraphEncoder API compat)
      def set_buffer(buffer : MetalBuffer, index : Int32, access : BufferAccess = BufferAccess::Read,
                     offset : Int64 = 0, length : Int64 = -1, partition : Int32 = -1) : self
        MetalDispatchFFI.encoder_set_buffer(@encoder, buffer.handle, offset, index)
        self
      end

      # Bind tensor's underlying buffer
      def set_tensor(tensor : Tensor, index : Int32) : self
        raise "Tensor must be on GPU" unless tensor.on_gpu?
        set_buffer(tensor.buffer.not_nil!, index)
      end

      # Bind raw bytes (for small constants)
      def set_bytes(data : Pointer(Void), length : Int32, index : Int32) : self
        MetalDispatchFFI.encoder_set_bytes(@encoder, data, length, index)
        self
      end

      # Bind scalar value
      def set_value(value : Float32, index : Int32) : self
        ptr = pointerof(value).as(Pointer(Void))
        set_bytes(ptr, sizeof(Float32), index)
      end

      def set_value(value : Int32, index : Int32) : self
        ptr = pointerof(value).as(Pointer(Void))
        set_bytes(ptr, sizeof(Int32), index)
      end

      def set_value(value : UInt32, index : Int32) : self
        ptr = pointerof(value).as(Pointer(Void))
        set_bytes(ptr, sizeof(UInt32), index)
      end

      # Set float3 (as StaticArray(Float32, 3))
      def set_value(value : StaticArray(Float32, 3), index : Int32) : self
        ptr = value.to_unsafe.as(Pointer(Void))
        set_bytes(ptr, sizeof(Float32) * 3, index)
      end

      # Set float4 (as StaticArray(Float32, 4))
      def set_value(value : StaticArray(Float32, 4), index : Int32) : self
        ptr = value.to_unsafe.as(Pointer(Void))
        set_bytes(ptr, sizeof(Float32) * 4, index)
      end

      # Dispatch with explicit grid/threadgroup sizes (total threads)
      def dispatch(grid_size : {Int32, Int32, Int32}, threadgroup_size : {Int32, Int32, Int32}) : self
        MetalDispatchFFI.encoder_dispatch_threads(
          @encoder,
          grid_size[0], grid_size[1], grid_size[2],
          threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]
        )
        self
      end

      # Dispatch with threadgroup count (not total threads)
      def dispatch_threadgroups(threadgroup_count : {Int32, Int32, Int32}, threadgroup_size : {Int32, Int32, Int32}) : self
        MetalDispatchFFI.encoder_dispatch_threadgroups(
          @encoder,
          threadgroup_count[0], threadgroup_count[1], threadgroup_count[2],
          threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]
        )
        self
      end

      # Indirect dispatch — threadgroup counts come from GPU buffer
      def dispatch_threadgroups_indirect(indirect_buffer : ML::MetalBuffer, offset : Int64, threadgroup_size : {Int32, Int32, Int32}) : self
        MetalDispatchFFI.encoder_dispatch_threadgroups_indirect(
          @encoder,
          indirect_buffer.handle, offset,
          threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]
        )
        self
      end

      # Dispatch 1D workload
      def dispatch_1d(count : Int32, threadgroup_size : Int32 = 256) : self
        max_threads = @pipeline.try(&.max_total_threads_per_threadgroup) || 1024
        tg_size = Math.min(threadgroup_size, max_threads)
        dispatch({count, 1, 1}, {tg_size, 1, 1})
      end

      # Dispatch 2D workload (e.g., images)
      def dispatch_2d(width : Int32, height : Int32, threadgroup_size : {Int32, Int32} = {16, 16}) : self
        max_threads = @pipeline.try(&.max_total_threads_per_threadgroup) || 1024
        tg_w = Math.min(threadgroup_size[0], max_threads)
        tg_h = Math.min(threadgroup_size[1], max_threads // tg_w)
        dispatch({width, height, 1}, {tg_w, tg_h, 1})
      end

      # Dispatch 3D workload
      def dispatch_3d(width : Int32, height : Int32, depth : Int32, threadgroup_size : {Int32, Int32, Int32} = {8, 8, 8}) : self
        max_threads = @pipeline.try(&.max_total_threads_per_threadgroup) || 1024
        tg_w = Math.min(threadgroup_size[0], max_threads)
        tg_h = Math.min(threadgroup_size[1], max_threads // tg_w)
        tg_d = Math.min(threadgroup_size[2], max_threads // (tg_w * tg_h))
        dispatch({width, height, depth}, {tg_w, tg_h, tg_d})
      end

      # Set threadgroup memory (for kernels using threadgroup storage)
      def set_threadgroup_memory(length : Int32, index : Int32) : self
        MetalDispatchFFI.encoder_set_threadgroup_memory(@encoder, length, index)
        self
      end

      # End encoding
      def end_encoding : Nil
        MetalDispatchFFI.encoder_end_encoding(@encoder)
      end
    end

    # Blit encoder for memory operations (copy, fill)
    struct BlitEncoder
      @cmd_buffer : Pointer(Void)
      @encoder : Pointer(Void)

      def initialize(command_buffer : CommandBuffer)
        @cmd_buffer = command_buffer.handle
        @encoder = MetalDispatchFFI.create_blit_encoder(@cmd_buffer)
        raise "Failed to create blit encoder" if @encoder.null?
      end

      # Fill buffer with a byte value
      def fill_buffer(buffer : MetalBuffer, value : UInt8, offset : Int32, length : Int32) : self
        MetalDispatchFFI.blit_fill_buffer(@encoder, buffer.handle, value, offset.to_i64, length.to_i64)
        self
      end

      # Copy between buffers
      def copy_buffer(src : MetalBuffer, src_offset : Int32, dst : MetalBuffer, dst_offset : Int32, size : Int32) : self
        MetalDispatchFFI.blit_copy_buffer(@encoder, src.handle, src_offset.to_i64, dst.handle, dst_offset.to_i64, size.to_i64)
        self
      end

      # End encoding
      def end_encoding : Nil
        MetalDispatchFFI.blit_end_encoding(@encoder)
      end
    end

    # High-level dispatch helper
    module Dispatch
      extend self

      # Execute a kernel synchronously
      def execute(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : Nil
        cmd_buffer = CommandBuffer.new
        encoder = ComputeEncoder.new(cmd_buffer)
        encoder.set_pipeline(pipeline)

        yield encoder

        encoder.end_encoding
        cmd_buffer.commit_and_wait
      end

      # Backwards-compatible alias
      def dispatch(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : Nil
        execute(pipeline, &block)
      end

      # Execute multiple kernels in sequence (single command buffer)
      def execute_sequence(&block : CommandBuffer -> Nil) : Nil
        cmd_buffer = CommandBuffer.new
        yield cmd_buffer
        cmd_buffer.commit_and_wait
      end

      # Execute kernel asynchronously
      def execute_async(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : CommandBuffer
        cmd_buffer = CommandBuffer.new
        encoder = ComputeEncoder.new(cmd_buffer)
        encoder.set_pipeline(pipeline)

        yield encoder

        encoder.end_encoding
        cmd_buffer.commit
        cmd_buffer
      end

      # Execute blit (memory) operations synchronously
      def execute_blit(&block : BlitEncoder -> Nil) : Nil
        cmd_buffer = CommandBuffer.new
        encoder = BlitEncoder.new(cmd_buffer)

        yield encoder

        encoder.end_encoding
        cmd_buffer.commit_and_wait
      end

      # Execute blit operations asynchronously
      def execute_blit_async(&block : BlitEncoder -> Nil) : CommandBuffer
        cmd_buffer = CommandBuffer.new
        encoder = BlitEncoder.new(cmd_buffer)

        yield encoder

        encoder.end_encoding
        cmd_buffer.commit
        cmd_buffer
      end

      # Optimal threadgroup size for 1D workloads
      def optimal_threadgroup_1d(count : Int32, pipeline : ComputePipeline? = nil) : Int32
        max_threads = pipeline.try(&.max_total_threads_per_threadgroup) || Device.instance.max_threads_per_threadgroup
        size = 256
        while size > max_threads
          size //= 2
        end
        size
      end

      # Optimal threadgroup size for 2D workloads (images, tiles)
      def optimal_threadgroup_2d(width : Int32, height : Int32, pipeline : ComputePipeline? = nil) : {Int32, Int32}
        max_threads = pipeline.try(&.max_total_threads_per_threadgroup) || Device.instance.max_threads_per_threadgroup
        w = 16
        h = 16
        while w * h > max_threads
          if w >= h
            w //= 2
          else
            h //= 2
          end
        end
        {w, h}
      end
    end

    # Common kernel parameter structs
    module KernelParams
      struct ElementwiseParams
        property count : UInt32
        property alpha : Float32
        property beta : Float32

        def initialize(@count : UInt32, @alpha : Float32 = 1.0_f32, @beta : Float32 = 0.0_f32)
        end
      end

      struct MatmulParams
        property m : UInt32
        property n : UInt32
        property k : UInt32
        property alpha : Float32
        property beta : Float32

        def initialize(@m : UInt32, @n : UInt32, @k : UInt32, @alpha : Float32 = 1.0_f32, @beta : Float32 = 0.0_f32)
        end
      end

      struct ReductionParams
        property count : UInt32
        property stride : UInt32

        def initialize(@count : UInt32, @stride : UInt32 = 1)
        end
      end

      struct Grid2DParams
        property width : UInt32
        property height : UInt32
        property channels : UInt32

        def initialize(@width : UInt32, @height : UInt32, @channels : UInt32 = 1)
        end
      end
    end
  end
end

# Metal Dispatch FFI declarations
{% if flag?(:darwin) %}
@[Link(ldflags: "-framework Metal -framework Foundation")]
lib MetalDispatchFFI
  # Compute encoder
  fun create_compute_encoder = gs_create_compute_encoder(cmd_buffer : Pointer(Void)) : Pointer(Void)
  fun encoder_set_pipeline = gs_encoder_set_pipeline(encoder : Pointer(Void), pipeline : Pointer(Void)) : Void
  fun encoder_set_buffer = gs_encoder_set_buffer(encoder : Pointer(Void), buffer : Pointer(Void), offset : Int64, index : Int32) : Void
  fun encoder_set_bytes = gs_encoder_set_bytes(encoder : Pointer(Void), data : Pointer(Void), length : Int32, index : Int32) : Void
  fun encoder_dispatch_threads = gs_encoder_dispatch_threads(
    encoder : Pointer(Void),
    grid_x : Int32, grid_y : Int32, grid_z : Int32,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun encoder_dispatch_threadgroups = gs_encoder_dispatch_threadgroups(
    encoder : Pointer(Void),
    tg_count_x : Int32, tg_count_y : Int32, tg_count_z : Int32,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun encoder_dispatch_threadgroups_indirect = gs_encoder_dispatch_threadgroups_indirect(
    encoder : Pointer(Void),
    indirect_buffer : Pointer(Void), indirect_offset : Int64,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun create_concurrent_compute_encoder = gs_create_concurrent_compute_encoder(cmd : Pointer(Void)) : Pointer(Void)
  fun encoder_memory_barrier = gs_encoder_memory_barrier(encoder : Pointer(Void)) : Void
  fun encoder_end_encoding = gs_encoder_end_encoding(encoder : Pointer(Void)) : Void
  fun encoder_set_threadgroup_memory = gs_encoder_set_threadgroup_memory(encoder : Pointer(Void), length : Int32, index : Int32) : Void

  # Blit encoder (memory operations)
  fun create_blit_encoder = gs_create_blit_encoder(cmd_buffer : Pointer(Void)) : Pointer(Void)
  fun blit_fill_buffer = gs_blit_fill_buffer(encoder : Pointer(Void), buffer : Pointer(Void), value : UInt8, offset : Int64, length : Int64) : Void
  fun blit_copy_buffer = gs_blit_copy_buffer(encoder : Pointer(Void), src : Pointer(Void), src_offset : Int64, dst : Pointer(Void), dst_offset : Int64, size : Int64) : Void
  fun blit_end_encoding = gs_blit_end_encoding(encoder : Pointer(Void)) : Void
end
{% else %}
lib MetalDispatchFFI
  fun create_compute_encoder = gs_create_compute_encoder(cmd_buffer : Pointer(Void)) : Pointer(Void)
  fun encoder_set_pipeline = gs_encoder_set_pipeline(encoder : Pointer(Void), pipeline : Pointer(Void)) : Void
  fun encoder_set_buffer = gs_encoder_set_buffer(encoder : Pointer(Void), buffer : Pointer(Void), offset : Int64, index : Int32) : Void
  fun encoder_set_bytes = gs_encoder_set_bytes(encoder : Pointer(Void), data : Pointer(Void), length : Int32, index : Int32) : Void
  fun encoder_dispatch_threads = gs_encoder_dispatch_threads(
    encoder : Pointer(Void),
    grid_x : Int32, grid_y : Int32, grid_z : Int32,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun encoder_dispatch_threadgroups = gs_encoder_dispatch_threadgroups(
    encoder : Pointer(Void),
    tg_count_x : Int32, tg_count_y : Int32, tg_count_z : Int32,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun encoder_dispatch_threadgroups_indirect = gs_encoder_dispatch_threadgroups_indirect(
    encoder : Pointer(Void),
    indirect_buffer : Pointer(Void), indirect_offset : Int64,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun create_concurrent_compute_encoder = gs_create_concurrent_compute_encoder(cmd : Pointer(Void)) : Pointer(Void)
  fun encoder_memory_barrier = gs_encoder_memory_barrier(encoder : Pointer(Void)) : Void
  fun encoder_end_encoding = gs_encoder_end_encoding(encoder : Pointer(Void)) : Void
  fun encoder_set_threadgroup_memory = gs_encoder_set_threadgroup_memory(encoder : Pointer(Void), length : Int32, index : Int32) : Void

  # Blit encoder (memory operations)
  fun create_blit_encoder = gs_create_blit_encoder(cmd_buffer : Pointer(Void)) : Pointer(Void)
  fun blit_fill_buffer = gs_blit_fill_buffer(encoder : Pointer(Void), buffer : Pointer(Void), value : UInt8, offset : Int64, length : Int64) : Void
  fun blit_copy_buffer = gs_blit_copy_buffer(encoder : Pointer(Void), src : Pointer(Void), src_offset : Int64, dst : Pointer(Void), dst_offset : Int64, size : Int64) : Void
  fun blit_end_encoding = gs_blit_end_encoding(encoder : Pointer(Void)) : Void
end
{% end %}
{% end %}
