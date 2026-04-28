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
        def initialize
          raise "Metal disabled (cpu_only)"
        end

        def handle : Pointer(Void)
          Pointer(Void).null
        end

        def commit_and_wait : Nil
          raise "Metal disabled (cpu_only)"
        end

        def commit : Nil
          raise "Metal disabled (cpu_only)"
        end

        def committed? : Bool
          false
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
      @committed : Bool = false

      def initialize(fast : Bool = false, queue : CommandQueue? = nil)
        raise "Metal not available" unless Device.available?
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
        return if @committed
        MetalDeviceFFI.commit_and_wait(@handle)
        @committed = true
      end

      # Commit without waiting (async GPU execution)
      def commit : Nil
        return if @committed
        MetalDeviceFFI.commit_command_buffer(@handle)
        @committed = true
      end

      # Wait for already-committed buffer to complete
      def wait : Nil
        MetalDeviceFFI.wait_command_buffer(@handle)
      end

      def committed? : Bool
        @committed
      end

      def finalize
        # Command buffers are autoreleased after commit
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
  fun has_unified_memory = gs_has_unified_memory : Int32

  # Command buffer
  fun create_command_queue = gs_create_command_queue : Pointer(Void)
  fun release_command_queue = gs_release_command_queue(queue : Pointer(Void)) : Void
  fun create_command_buffer = gs_create_command_buffer : Pointer(Void)
  fun create_command_buffer_fast = gs_create_command_buffer_fast : Pointer(Void)
  fun create_command_buffer_on_queue = gs_create_command_buffer_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun create_command_buffer_fast_on_queue = gs_create_command_buffer_fast_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun enqueue_command_buffer = gs_enqueue_command_buffer(cmd : Pointer(Void)) : Void
  fun commit_command_buffer = gs_commit_command_buffer(cmd : Pointer(Void)) : Void
  fun wait_command_buffer = gs_wait_command_buffer(cmd : Pointer(Void)) : Void
  fun commit_and_wait = gs_commit_and_wait(cmd_buffer : Pointer(Void)) : Void
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
  fun has_unified_memory = gs_has_unified_memory : Int32
  fun create_command_queue = gs_create_command_queue : Pointer(Void)
  fun release_command_queue = gs_release_command_queue(queue : Pointer(Void)) : Void
  fun create_command_buffer = gs_create_command_buffer : Pointer(Void)
  fun create_command_buffer_fast = gs_create_command_buffer_fast : Pointer(Void)
  fun create_command_buffer_on_queue = gs_create_command_buffer_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun create_command_buffer_fast_on_queue = gs_create_command_buffer_fast_on_queue(queue : Pointer(Void)) : Pointer(Void)
  fun enqueue_command_buffer = gs_enqueue_command_buffer(cmd : Pointer(Void)) : Void
  fun commit_command_buffer = gs_commit_command_buffer(cmd : Pointer(Void)) : Void
  fun wait_command_buffer = gs_wait_command_buffer(cmd : Pointer(Void)) : Void
  fun commit_and_wait = gs_commit_and_wait(cmd_buffer : Pointer(Void)) : Void
  fun commit = gs_commit(cmd_buffer : Pointer(Void)) : Void
  fun create_pipeline = gs_create_pipeline(source : Pointer(UInt8), function_name : Pointer(UInt8)) : Pointer(Void)
  fun create_pipeline_from_library = gs_create_pipeline_from_library(library_path : Pointer(UInt8), function_name : Pointer(UInt8)) : Pointer(Void)
  fun create_pipeline_from_default_library = gs_create_pipeline_from_default_library(function_name : Pointer(UInt8)) : Pointer(Void)
  fun pipeline_max_threads = gs_pipeline_max_threads(pipeline : Pointer(Void)) : Int32
end
{% end %}
{% end %}
