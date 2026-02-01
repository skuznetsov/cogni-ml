# Metal Buffer wrapper with RAII semantics
# Manages GPU memory lifecycle, supports zero-copy on Apple Silicon

module ML
  # Storage mode for Metal buffers
  enum StorageMode
    Shared  # CPU + GPU accessible, zero-copy on Apple Silicon (default)
    Private # GPU only, fastest for GPU-exclusive data
    Managed # macOS only, explicit sync required
  end

  enum PurgeableState
    NonVolatile
    Volatile
    Empty
  end

  # Wraps a Metal buffer with automatic resource management
  # Uses MTLResourceStorageModeShared for unified memory on M-series chips
  class MetalBuffer
    getter size : Int64
    getter storage_mode : StorageMode
    getter? valid : Bool
    @@live_buffers : Int32 = 0
    @@live_bytes : Int64 = 0
    @@peak_bytes : Int64 = 0

    # Pointer to underlying MTLBuffer (void* in Crystal FFI)
    @handle : Pointer(Void)

    # Track if we own the buffer (vs wrapping external)
    @owned : Bool
    @counted : Bool = false

    def initialize(@size : Int64, @storage_mode : StorageMode = StorageMode::Shared)
      raise ArgumentError.new("Buffer size must be positive") if @size <= 0
      @handle = MetalFFI.gs_create_buffer(@size, @storage_mode.value)
      @owned = true
      @valid = !@handle.null?
      raise "Failed to allocate Metal buffer of size #{@size}" unless @valid
      track_alloc(@size)
    end

    # Wrap existing Array without copy (zero-copy)
    def self.from_array(data : Array(Float32), mode : StorageMode = StorageMode::Shared) : MetalBuffer
      byte_size = data.size.to_i64 * sizeof(Float32)
      buffer = new(byte_size, mode)
      buffer.write(data)
      buffer
    end

    # Wrap existing Slice
    def self.from_slice(data : Slice(Float32), mode : StorageMode = StorageMode::Shared) : MetalBuffer
      byte_size = data.size.to_i64 * sizeof(Float32)
      buffer = new(byte_size, mode)
      buffer.write_slice(data)
      buffer
    end

    # Write data to buffer
    def write(data : Array(Float32)) : Nil
      raise "Buffer invalid" unless @valid
      byte_size = data.size.to_i64 * sizeof(Float32)
      raise "Data size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_write(@handle, data.to_unsafe.as(Pointer(Void)), byte_size)
    end

    def write_slice(data : Slice(Float32)) : Nil
      raise "Buffer invalid" unless @valid
      byte_size = data.size.to_i64 * sizeof(Float32)
      raise "Data size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_write(@handle, data.to_unsafe.as(Pointer(Void)), byte_size)
    end

    # Read data from buffer
    def read(count : Int32) : Array(Float32)
      raise "Buffer invalid" unless @valid
      byte_size = count.to_i64 * sizeof(Float32)
      raise "Read size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      result = Array(Float32).new(count, 0.0_f32)
      actual = MetalFFI.gs_buffer_read(@handle, result.to_unsafe.as(Pointer(Void)), byte_size)
      raise "Buffer read truncated: requested #{byte_size}, got #{actual}" if actual != byte_size
      result
    end

    def read_to_slice(dest : Slice(Float32)) : Nil
      raise "Buffer invalid" unless @valid
      byte_size = dest.size.to_i64 * sizeof(Float32)
      raise "Read size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      actual = MetalFFI.gs_buffer_read(@handle, dest.to_unsafe.as(Pointer(Void)), byte_size)
      raise "Buffer read truncated: requested #{byte_size}, got #{actual}" if actual != byte_size
    end

    # Write raw bytes to buffer
    def write_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
      raise "Buffer invalid" unless @valid
      raise "Data size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_write(@handle, ptr.as(Pointer(Void)), byte_size.to_i64)
    end

    # Read raw bytes from buffer (validates actual bytes read)
    def read_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
      raise "Buffer invalid" unless @valid
      raise "Read size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      actual = MetalFFI.gs_buffer_read(@handle, ptr.as(Pointer(Void)), byte_size.to_i64)
      raise "Buffer read truncated: requested #{byte_size}, got #{actual}" if actual != byte_size
    end

    # Get raw pointer for Metal kernel bindings
    def contents : Pointer(Void)
      raise "Buffer invalid" unless @valid
      MetalFFI.gs_buffer_contents(@handle)
    end

    # Get underlying handle for FFI
    def handle : Pointer(Void)
      @handle
    end

    # Element count (assuming Float32)
    def element_count : Int32
      (@size // sizeof(Float32)).to_i32
    end

    # Copy from another buffer
    def copy_from(src : MetalBuffer, byte_size : Int64) : Nil
      raise "Destination buffer invalid" unless @valid
      raise "Source buffer invalid" unless src.valid?
      raise "Copy size #{byte_size} exceeds destination buffer size #{@size}" if byte_size > @size
      raise "Copy size #{byte_size} exceeds source buffer size #{src.size}" if byte_size > src.size

      # On unified memory (Apple Silicon), we can directly copy via contents pointers
      MetalFFI.gs_buffer_copy(src.handle, self.handle, byte_size)
    end

    # Synchronize (for Managed mode on macOS)
    def sync : Nil
      return unless @valid && @storage_mode == StorageMode::Managed
      MetalFFI.gs_buffer_sync(@handle)
    end

    def set_purgeable(state : PurgeableState) : Nil
      return unless @valid
      {% if flag?(:darwin) %}
        MetalFFI.gs_buffer_set_purgeable(@handle, state.value)
      {% end %}
    end

    # RAII cleanup
    def finalize
      if @valid && @owned
        release
      end
    end

    # Manual release (for explicit control)
    def release : Nil
      if @valid && @owned
        set_purgeable(PurgeableState::Empty)
        MetalFFI.gs_release_buffer(@handle)
        @valid = false
        track_free(@size)
      end
    end

    def self.stats : NamedTuple(live_buffers: Int32, live_bytes: Int64, peak_bytes: Int64)
      {live_buffers: @@live_buffers, live_bytes: @@live_bytes, peak_bytes: @@peak_bytes}
    end

    private def track_alloc(bytes : Int64) : Nil
      return if @counted
      @counted = true
      @@live_buffers += 1
      @@live_bytes += bytes
      @@peak_bytes = @@live_bytes if @@live_bytes > @@peak_bytes
    end

    private def track_free(bytes : Int64) : Nil
      return unless @counted
      @counted = false
      @@live_buffers -= 1
      @@live_bytes -= bytes
    end
  end

  # Buffer pool for reusing allocations (reduces Metal allocation overhead)
  class BufferPool
    DEFAULT_MAX_CACHED = 32
    DEFAULT_MAX_BUFFER_BYTES = 256_i64 * 1024 * 1024
    DEFAULT_MAX_CACHED_BYTES = 2_i64 * 1024 * 1024 * 1024
    ALIGNMENT = 256_i64 # Metal buffer alignment

    @pools : Hash(Int64, Array(MetalBuffer))
    @max_cached : Int32
    @max_buffer_bytes : Int64
    @max_cached_bytes : Int64
    @cached_bytes : Int64

    def initialize(
      @max_cached : Int32 = DEFAULT_MAX_CACHED,
      @max_buffer_bytes : Int64 = DEFAULT_MAX_BUFFER_BYTES,
      @max_cached_bytes : Int64 = DEFAULT_MAX_CACHED_BYTES
    )
      @pools = Hash(Int64, Array(MetalBuffer)).new
      @cached_bytes = 0_i64
    end

    # Get buffer of at least `size` bytes, reusing if possible
    def acquire(size : Int64, mode : StorageMode = StorageMode::Shared) : MetalBuffer
      aligned_size = align_size(size)

      if pool = @pools[aligned_size]?
        if buffer = pool.pop?
          @cached_bytes -= aligned_size
          if buffer.valid?
            buffer.set_purgeable(PurgeableState::NonVolatile)
            return buffer
          end
        end
      end

      MetalBuffer.new(aligned_size, mode)
    end

    # Return buffer to pool for reuse
    def release(buffer : MetalBuffer) : Nil
      return unless buffer.valid?

      aligned_size = align_size(buffer.size)
      if aligned_size > @max_buffer_bytes
        buffer.release
        return
      end

      if (@cached_bytes + aligned_size) > @max_cached_bytes
        buffer.release
        return
      end

      pool = @pools[aligned_size] ||= Array(MetalBuffer).new

      if pool.size < @max_cached
        if state = pool_purgeable_state
          buffer.set_purgeable(state)
        end
        pool << buffer
        @cached_bytes += aligned_size
      else
        buffer.release
      end
    end

    # Clear all cached buffers
    def clear : Nil
      @pools.each_value do |pool|
        pool.each(&.release)
        pool.clear
      end
      @cached_bytes = 0_i64
    end

    def stats : NamedTuple(cached_buffers: Int32, cached_bytes: Int64)
      cached_buffers = 0
      @pools.each_value { |pool| cached_buffers += pool.size }
      {cached_buffers: cached_buffers, cached_bytes: @cached_bytes}
    end

    private def align_size(size : Int64) : Int64
      aligned = ((size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
      next_power_of_two(aligned)
    end

    private def next_power_of_two(value : Int64) : Int64
      return ALIGNMENT if value <= 0

      v = value - 1
      v |= v >> 1
      v |= v >> 2
      v |= v >> 4
      v |= v >> 8
      v |= v >> 16
      v |= v >> 32
      v + 1
    end

    private def pool_purgeable_state : PurgeableState?
      raw = ENV["GS_PURGEABLE_POOL"]?
      return nil if raw.nil? || raw.empty?
      value = raw.downcase
      return nil if value == "0" || value == "off"
      return PurgeableState::Empty if value == "empty"
      PurgeableState::Volatile
    end

    def self.env_i64(name : String, default : Int64) : Int64
      value = ENV[name]?
      return default if value.nil? || value.empty?
      parsed = value.to_i64?
      parsed && parsed > 0 ? parsed : default
    end
  end

  # Global buffer pool instance
  class_getter buffer_pool : BufferPool = BufferPool.new(
    max_cached: BufferPool.env_i64("GS_BUFFER_POOL_MAX_CACHED", BufferPool::DEFAULT_MAX_CACHED).to_i32,
    max_buffer_bytes: BufferPool.env_i64("GS_BUFFER_POOL_MAX_BUFFER_BYTES", BufferPool::DEFAULT_MAX_BUFFER_BYTES),
    max_cached_bytes: BufferPool.env_i64("GS_BUFFER_POOL_MAX_BYTES", BufferPool::DEFAULT_MAX_CACHED_BYTES)
  )
end

# Metal FFI declarations (implemented in metal_bridge.mm)
{% if flag?(:darwin) %}
@[Link(ldflags: "-framework Metal -framework Foundation")]
lib MetalFFI
  fun gs_create_buffer = gs_create_buffer(size : Int64, storage_mode : Int32) : Pointer(Void)
  fun gs_release_buffer = gs_release_buffer(handle : Pointer(Void)) : Void
  fun gs_buffer_contents = gs_buffer_contents(handle : Pointer(Void)) : Pointer(Void)
  fun gs_buffer_size = gs_buffer_size(handle : Pointer(Void)) : Int64
  fun gs_buffer_write = gs_buffer_write(handle : Pointer(Void), data : Pointer(Void), size : Int64) : Void
  fun gs_buffer_read = gs_buffer_read(handle : Pointer(Void), dest : Pointer(Void), size : Int64) : Int64
  fun gs_buffer_sync = gs_buffer_sync(handle : Pointer(Void)) : Void
  fun gs_buffer_copy = gs_buffer_copy(src : Pointer(Void), dst : Pointer(Void), size : Int64) : Void
  fun gs_buffer_set_purgeable = gs_buffer_set_purgeable(handle : Pointer(Void), state : Int32) : Int32
end
{% else %}
# Stub for non-Darwin platforms (CPU fallback)
lib MetalFFI
  fun gs_create_buffer = gs_create_buffer(size : Int64, storage_mode : Int32) : Pointer(Void)
  fun gs_release_buffer = gs_release_buffer(handle : Pointer(Void)) : Void
  fun gs_buffer_contents = gs_buffer_contents(handle : Pointer(Void)) : Pointer(Void)
  fun gs_buffer_size = gs_buffer_size(handle : Pointer(Void)) : Int64
  fun gs_buffer_write = gs_buffer_write(handle : Pointer(Void), data : Pointer(Void), size : Int64) : Void
  fun gs_buffer_read = gs_buffer_read(handle : Pointer(Void), dest : Pointer(Void), size : Int64) : Int64
  fun gs_buffer_sync = gs_buffer_sync(handle : Pointer(Void)) : Void
  fun gs_buffer_copy = gs_buffer_copy(src : Pointer(Void), dst : Pointer(Void), size : Int64) : Void
  fun gs_buffer_set_purgeable = gs_buffer_set_purgeable(handle : Pointer(Void), state : Int32) : Int32
end
{% end %}
