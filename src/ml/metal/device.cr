{% if flag?(:cpu_only) %}
  # CPU-only stubs (Metal disabled)
  module ML
    module Metal
      class Device
        @@instance : Device?

        def self.instance : Device
          @@instance ||= new
        end

        def self.available? : Bool
          false
        end

        def self.current_allocated_size_if_initialized : Int64?
          nil
        end

        def self.init! : Bool
          false
        end

        def available? : Bool
          false
        end

        def name : String
          "CPU"
        end

        def max_threads_per_threadgroup : Int32
          1
        end

        def recommended_working_set_size : Int64
          0_i64
        end

        def current_allocated_size : Int64
          0_i64
        end

        def has_unified_memory? : Bool
          false
        end

        def device_handle : Pointer(Void)
          Pointer(Void).null
        end

        def queue_handle : Pointer(Void)
          Pointer(Void).null
        end

        def synchronize : Nil
        end

        def self.synchronize : Nil
        end
      end

      class CommandBuffer
        def transport_identity : UInt64
          0_u64
        end

        def initialize
          raise "Metal disabled (cpu_only)"
        end

        def handle : Pointer(Void)
          Pointer(Void).null
        end

        def commit_and_wait : Nil
          raise "Metal disabled (cpu_only)"
        end

        def commit_and_wait_gpu_elapsed_seconds : Float64
          raise "Metal disabled (cpu_only)"
        end

        def wait_gpu_elapsed_seconds? : Float64?
          raise "Metal disabled (cpu_only)"
        end

        def completed? : Bool
          false
        end

        def completed_successfully? : Bool
          false
        end

        def commit : Nil
          raise "Metal disabled (cpu_only)"
        end

        def committed? : Bool
          false
        end

        def discard : Nil
        end
      end

      class ComputePipeline
        getter name : String

        def initialize(@name : String, source : String, function_name : String? = nil)
          raise "Metal disabled (cpu_only)"
        end

        def self.from_library(name : String, library_path : String? = nil) : ComputePipeline
          raise "Metal disabled (cpu_only)"
        end

        def handle : Pointer(Void)
          Pointer(Void).null
        end

        def max_total_threads_per_threadgroup : Int32
          1
        end
      end

      class PipelineCache
        def self.get(name : String, &block : -> ComputePipeline) : ComputePipeline
          raise "Metal disabled (cpu_only)"
        end

        def self.get_or_compile(name : String, source : String) : ComputePipeline
          raise "Metal disabled (cpu_only)"
        end

        def self.get_from_library(name : String) : ComputePipeline
          raise "Metal disabled (cpu_only)"
        end

        def self.clear : Nil
        end
      end
    end
  end
{% else %}
# Metal device and command queue management
# Singleton pattern for global GPU access

require "../core/buffer"

module ML
  module Metal
    # Singleton Metal device manager
    class Device
      @@instance : Device?
      @@initialized : Bool = false

      getter? available : Bool
      @device_handle : Pointer(Void)
      @queue_handle : Pointer(Void)

      private def initialize
        @device_handle = Pointer(Void).null
        @queue_handle = Pointer(Void).null
        @available = false

        {% if flag?(:cpu_only) %}
          @available = false
        {% else %}
          {% if flag?(:darwin) %}
            result = MetalDeviceFFI.init_device
            if result == 0
              @device_handle = MetalDeviceFFI.get_device
              @queue_handle = MetalDeviceFFI.get_command_queue
              @available = !@device_handle.null? && !@queue_handle.null?
            end
          {% end %}
        {% end %}
      end

      def self.instance : Device
        @@instance ||= new
      end

      # Observation must not create a device or compile its first pipeline.
      def self.current_allocated_size_if_initialized : Int64?
        @@instance.try { |device| device.available? ? device.current_allocated_size : nil }
      end

      def self.available? : Bool
        instance.available?
      end

      def self.init! : Bool
        return true if @@initialized
        @@initialized = instance.available?
        @@initialized
      end

      # Device properties
      def name : String
        return "CPU (Metal unavailable)" unless @available
        {% if flag?(:darwin) %}
          ptr = MetalDeviceFFI.device_name
          String.new(ptr)
        {% else %}
          "CPU"
        {% end %}
      end

      def max_threads_per_threadgroup : Int32
        return 1 unless @available
        {% if flag?(:darwin) %}
          MetalDeviceFFI.max_threads_per_threadgroup
        {% else %}
          1
        {% end %}
      end

      def recommended_working_set_size : Int64
        return 0_i64 unless @available
        {% if flag?(:darwin) %}
          MetalDeviceFFI.recommended_working_set_size
        {% else %}
          0_i64
        {% end %}
      end

      # Metal-visible resource bytes for this device. On Apple Silicon these
      # consume the same unified-memory pool as CPU allocations and therefore
      # complement, rather than duplicate, process RSS telemetry.
      def current_allocated_size : Int64
        return 0_i64 unless @available
        {% if flag?(:darwin) %}
          MetalDeviceFFI.current_allocated_size
        {% else %}
          0_i64
        {% end %}
      end

      def has_unified_memory? : Bool
        {% if flag?(:darwin) %}
          @available && MetalDeviceFFI.has_unified_memory != 0
        {% else %}
          false
        {% end %}
      end

      # Internal handles for kernel dispatch
      def device_handle : Pointer(Void)
        @device_handle
      end

      def queue_handle : Pointer(Void)
        @queue_handle
      end

      # Synchronize all pending GPU work
      def synchronize : Nil
        return unless @available
        {% if flag?(:darwin) %}
          MetalDeviceFFI.synchronize
        {% end %}
      end

      def self.synchronize : Nil
        instance.synchronize
      end
    end

    # Command buffer for batching operations
    class CommandQueue
      getter handle : Pointer(Void)

      def initialize
        raise "Metal not available" unless Device.available?
        @handle = MetalDeviceFFI.create_command_queue
        raise "Failed to create command queue" if @handle.null?
      end

      def finalize
        MetalDeviceFFI.release_command_queue(@handle) unless @handle.null?
      end
    end

    class CommandBuffer
      @handle : Pointer(Void)
      @queue_owner : CommandQueue?
      getter transport_identity : UInt64
      @committed : Bool = false
      @completed : Bool = false
      @completion_status : Int32? = nil
      @gpu_elapsed_seconds : Float64? = nil

      def initialize(fast : Bool = false, queue : CommandQueue? = nil)
        raise "Metal not available" unless Device.available?
        # Retain an explicit queue for the command lifetime. The address is a
        # host-side ordering certificate; it is never passed to a kernel.
        @queue_owner = queue
        @transport_identity = if q = queue
                                q.handle.address.to_u64
                              else
                                Device.instance.queue_handle.address.to_u64
                              end
        @handle = if q = queue
                    fast ? MetalDeviceFFI.create_command_buffer_fast_on_queue(q.handle) : MetalDeviceFFI.create_command_buffer_on_queue(q.handle)
                  else
                    fast ? MetalDeviceFFI.create_command_buffer_fast : MetalDeviceFFI.create_command_buffer
                  end
        raise "Failed to create command buffer" if @handle.null?
      end

      def handle : Pointer(Void)
        @handle
      end

      # Enqueue: tells Metal execution order WITHOUT committing
      def enqueue : Nil
        MetalDeviceFFI.enqueue_command_buffer(@handle)
      end

      # Commit and wait for completion
      def commit_and_wait : Nil
        return verify_completion! if @completed
        return wait if @committed
        @committed = true
        @completion_status = MetalDeviceFFI.commit_and_wait_status(@handle)
        @completed = true
        verify_completion!
      end

      # Commit once and return Metal's completed-command GPU execution interval.
      # This is a profiling seam, not wall time and not a hardware-counter API.
      def commit_and_wait_gpu_elapsed_seconds : Float64
        raise "cannot GPU-time an already committed Metal command buffer" if @committed
        @committed = true
        elapsed_seconds = 0.0_f64
        @completion_status = MetalDeviceFFI.commit_and_wait_status_gpu_elapsed(
          @handle, pointerof(elapsed_seconds),
        )
        @completed = true
        verify_completion!
        unless elapsed_seconds.finite? && elapsed_seconds > 0.0
          raise "Metal GPU execution timestamps unavailable"
        end
        @gpu_elapsed_seconds = elapsed_seconds
        elapsed_seconds
      end

      # Wait for an already committed command and opportunistically capture
      # Metal's GPU interval. Missing timestamps do not turn successful work
      # into an inference failure; callers can omit that profile sample.
      def wait_gpu_elapsed_seconds? : Float64?
        if @completed
          verify_completion!
          return @gpu_elapsed_seconds
        end
        raise ArgumentError.new("cannot GPU-time an uncommitted Metal command buffer") unless @committed

        elapsed_seconds = 0.0_f64
        @completion_status = MetalDeviceFFI.wait_command_buffer_status_gpu_elapsed(
          @handle, pointerof(elapsed_seconds),
        )
        @completed = true
        verify_completion!
        if elapsed_seconds.finite? && elapsed_seconds > 0.0
          @gpu_elapsed_seconds = elapsed_seconds
        end
        @gpu_elapsed_seconds
      end

      # Commit without waiting (async GPU execution)
      def commit : Nil
        return if @committed
        MetalDeviceFFI.commit_command_buffer(@handle)
        @committed = true
      end

      # Wait for already-committed buffer to complete
      def wait : Nil
        return verify_completion! if @completed
        raise "cannot wait for an uncommitted Metal command buffer" unless @committed
        @completion_status = MetalDeviceFFI.wait_command_buffer_status(@handle)
        @completed = true
        verify_completion!
      end

      def committed? : Bool
        @committed
      end

      def completed? : Bool
        @completed
      end

      def completed_successfully? : Bool
        @completed && @completion_status == 0
      end

      # Release a command that will never be submitted. Committed command
      # buffers remain owned by their wait path and cannot be discarded.
      def discard : Nil
        raise "cannot discard a committed Metal command buffer" if @committed
        return if @handle.null?
        MetalDeviceFFI.release_command_buffer(@handle)
        @handle = Pointer(Void).null
      end

      private def verify_completion! : Nil
        status = @completion_status
        unless status == 0
          raise "Metal command buffer failed (completion_status=#{status})"
        end
      end

      def finalize
        discard unless @committed || @handle.null?
      end
    end

    # Compute pipeline state for a compiled kernel
    class ComputePipeline
      getter name : String
      @handle : Pointer(Void)

      # Internal constructor for from_library
      protected def initialize(@name : String, @handle : Pointer(Void))
      end

      def initialize(@name : String, source : String, function_name : String? = nil)
        raise "Metal not available" unless Device.available?

        fn_name = function_name || @name
        @handle = MetalDeviceFFI.create_pipeline(source, fn_name)
        raise "Failed to compile kernel '#{fn_name}'" if @handle.null?
      end

      def self.from_library(name : String, library_path : String? = nil) : ComputePipeline
        raise "Metal not available" unless Device.available?

        handle = if library_path
                   MetalDeviceFFI.create_pipeline_from_library(library_path, name)
                 else
                   MetalDeviceFFI.create_pipeline_from_default_library(name)
                 end
        raise "Failed to load kernel '#{name}'" if handle.null?

        ComputePipeline.new(name, handle)
      end

      def handle : Pointer(Void)
        @handle
      end

      def max_total_threads_per_threadgroup : Int32
        MetalDeviceFFI.pipeline_max_threads(@handle)
      end
    end

    # Pipeline cache for reusing compiled kernels
    class PipelineCache
      @@cache = Hash(String, ComputePipeline).new

      # Cache keys, not driver compiler variants or pipeline allocation bytes.
      def self.entry_count : Int32
        @@cache.size
      end

      def self.get(name : String, &block : -> ComputePipeline) : ComputePipeline
        @@cache[name] ||= yield
      end

      def self.get_or_compile(name : String, source : String) : ComputePipeline
        get(name) { ComputePipeline.new(name, source) }
      end

      def self.get_from_library(name : String) : ComputePipeline
        get(name) { ComputePipeline.from_library(name) }
      end

      def self.clear : Nil
        @@cache.clear
      end
    end
  end
end
{% end %}

{% if !flag?(:cpu_only) %}
# Metal Device FFI declarations
{% if flag?(:darwin) %}
@[Link(ldflags: "-framework Metal -framework Foundation")]
lib MetalDeviceFFI
  # Device initialization
  fun init_device = gs_init_device : Int32
  fun get_device = gs_get_device : Pointer(Void)
  fun get_command_queue = gs_get_command_queue : Pointer(Void)
  fun synchronize = gs_synchronize : Void

  # Device properties
  fun device_name = gs_device_name : Pointer(UInt8)
  fun max_threads_per_threadgroup = gs_max_threads_per_threadgroup : Int32
  fun recommended_working_set_size = gs_recommended_working_set_size : Int64
  fun current_allocated_size = gs_current_allocated_size : Int64
  fun has_unified_memory = gs_has_unified_memory : Int32

  # Command buffer
  fun create_command_queue = gs_create_command_queue : Pointer(Void)
  fun release_command_queue = gs_release_command_queue(queue : Pointer(Void)) : Void
  fun release_command_buffer = gs_release_command_buffer(cmd : Pointer(Void)) : Void
  fun create_command_buffer = gs_create_command_buffer : Pointer(Void)
  fun create_command_buffer_fast = gs_create_command_buffer_fast : Pointer(Void)
  fun create_command_buffer_on_queue = gs_create_command_buffer_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun create_command_buffer_fast_on_queue = gs_create_command_buffer_fast_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun enqueue_command_buffer = gs_enqueue_command_buffer(cmd : Pointer(Void)) : Void
  fun commit_command_buffer = gs_commit_command_buffer(cmd : Pointer(Void)) : Void
  fun wait_command_buffer = gs_wait_command_buffer(cmd : Pointer(Void)) : Void
  fun wait_command_buffer_status = gs_wait_command_buffer_status(cmd : Pointer(Void)) : Int32
  fun wait_command_buffer_status_gpu_elapsed = gs_wait_command_buffer_status_gpu_elapsed(cmd : Pointer(Void), elapsed_seconds : Pointer(Float64)) : Int32
  fun commit_and_wait = gs_commit_and_wait(cmd_buffer : Pointer(Void)) : Void
  fun commit_and_wait_status = gs_commit_and_wait_status(cmd_buffer : Pointer(Void)) : Int32
  fun commit_and_wait_status_gpu_elapsed = gs_commit_and_wait_status_gpu_elapsed(cmd_buffer : Pointer(Void), elapsed_seconds : Pointer(Float64)) : Int32
  fun commit = gs_commit(cmd_buffer : Pointer(Void)) : Void

  # Pipeline compilation
  fun create_pipeline = gs_create_pipeline(source : Pointer(UInt8), function_name : Pointer(UInt8)) : Pointer(Void)
  fun create_pipeline_from_library = gs_create_pipeline_from_library(library_path : Pointer(UInt8), function_name : Pointer(UInt8)) : Pointer(Void)
  fun create_pipeline_from_default_library = gs_create_pipeline_from_default_library(function_name : Pointer(UInt8)) : Pointer(Void)
  fun pipeline_max_threads = gs_pipeline_max_threads(pipeline : Pointer(Void)) : Int32
end
{% else %}
# Stubs for non-Darwin platforms
lib MetalDeviceFFI
  fun init_device = gs_init_device : Int32
  fun get_device = gs_get_device : Pointer(Void)
  fun get_command_queue = gs_get_command_queue : Pointer(Void)
  fun synchronize = gs_synchronize : Void
  fun device_name = gs_device_name : Pointer(UInt8)
  fun max_threads_per_threadgroup = gs_max_threads_per_threadgroup : Int32
  fun recommended_working_set_size = gs_recommended_working_set_size : Int64
  fun current_allocated_size = gs_current_allocated_size : Int64
  fun has_unified_memory = gs_has_unified_memory : Int32
  fun create_command_queue = gs_create_command_queue : Pointer(Void)
  fun release_command_queue = gs_release_command_queue(queue : Pointer(Void)) : Void
  fun release_command_buffer = gs_release_command_buffer(cmd : Pointer(Void)) : Void
  fun create_command_buffer = gs_create_command_buffer : Pointer(Void)
  fun create_command_buffer_fast = gs_create_command_buffer_fast : Pointer(Void)
  fun create_command_buffer_on_queue = gs_create_command_buffer_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun create_command_buffer_fast_on_queue = gs_create_command_buffer_fast_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun enqueue_command_buffer = gs_enqueue_command_buffer(cmd : Pointer(Void)) : Void
  fun commit_command_buffer = gs_commit_command_buffer(cmd : Pointer(Void)) : Void
  fun wait_command_buffer = gs_wait_command_buffer(cmd : Pointer(Void)) : Void
  fun wait_command_buffer_status = gs_wait_command_buffer_status(cmd : Pointer(Void)) : Int32
  fun wait_command_buffer_status_gpu_elapsed = gs_wait_command_buffer_status_gpu_elapsed(cmd : Pointer(Void), elapsed_seconds : Pointer(Float64)) : Int32
  fun commit_and_wait = gs_commit_and_wait(cmd_buffer : Pointer(Void)) : Void
  fun commit_and_wait_status = gs_commit_and_wait_status(cmd_buffer : Pointer(Void)) : Int32
  fun commit_and_wait_status_gpu_elapsed = gs_commit_and_wait_status_gpu_elapsed(cmd_buffer : Pointer(Void), elapsed_seconds : Pointer(Float64)) : Int32
  fun commit = gs_commit(cmd_buffer : Pointer(Void)) : Void
  fun create_pipeline = gs_create_pipeline(source : Pointer(UInt8), function_name : Pointer(UInt8)) : Pointer(Void)
  fun create_pipeline_from_library = gs_create_pipeline_from_library(library_path : Pointer(UInt8), function_name : Pointer(UInt8)) : Pointer(Void)
  fun create_pipeline_from_default_library = gs_create_pipeline_from_default_library(function_name : Pointer(UInt8)) : Pointer(Void)
  fun pipeline_max_threads = gs_pipeline_max_threads(pipeline : Pointer(Void)) : Int32
end
{% end %}
{% end %}
