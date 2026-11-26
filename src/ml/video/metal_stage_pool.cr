require "c/sys/mman"
require "c/unistd"
require "../metal/dispatch"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  lib MetalStagePoolPlatform
    fun gs_max_buffer_length : Int64
  end
{% end %}

module ML::Video
  class MetalStagePool
  end

  # A no-copy view into MetalStagePool's mmap. The pool retains both this view
  # and its backing mapping until the owning Arena releases the view.
  class MetalStageBufferView
    getter buffer : ML::MetalBuffer
    getter phase : String
    getter lane : String
    getter offset : Int64
    getter requested_bytes : Int64
    getter mapped_bytes : Int64
    getter lease_id : UInt64

    @arena_owner : UInt64?
    @released : Bool

    def initialize(@pool : MetalStagePool, @buffer : ML::MetalBuffer,
                   @phase : String, @lane : String, @offset : Int64,
                   @requested_bytes : Int64, @mapped_bytes : Int64,
                   @lease_id : UInt64)
      @arena_owner = nil
      @released = false
    end

    def released? : Bool
      @released
    end

    # Ownership transfer is explicit; one view belongs to one Arena.
    def claim_for_arena!(arena_id : UInt64) : Nil
      raise ArgumentError.new("Released stage buffer view") if @released
      raise ArgumentError.new("Stage buffer view already belongs to an Arena") if @arena_owner
      @arena_owner = arena_id
    end

    def release! : Nil
      raise ArgumentError.new("Arena-owned stage view must be released by its Arena") if @arena_owner
      @pool.release_view!(self, nil)
    end

    def release_from_arena!(arena_id : UInt64) : Nil
      unless @arena_owner == arena_id
        raise ArgumentError.new("Stage buffer view Arena owner mismatch")
      end
      @pool.release_view!(self, arena_id)
    end

    def mark_released! : Nil
      @released = true
      @arena_owner = nil
    end

    def owner : MetalStagePool
      @pool
    end

    def arena_owned? : Bool
      !@arena_owner.nil?
    end

    def arena_owner_matches?(arena_id : UInt64) : Bool
      @arena_owner == arena_id
    end
  end

  # Page-aligned anonymous backing for phase-overlapped Metal views. Each
  # phase has an independent monotonic cursor starting at zero; allocations
  # within a phase are disjoint across lanes. Different phases intentionally
  # reuse the same physical ranges. The caller must serialize phase use on its
  # existing GPU event/command chain and retain every view through completion.
  # This class never writes the backing bytes from the CPU.
  class MetalStagePool
    getter size : Int64 = 0_i64
    getter page_size : Int64 = 0_i64
    getter max_buffer_bytes : Int64 = 0_i64
    getter? released : Bool = true

    # Defaults keep non-Darwin/cpu_only builds well-typed even though their
    # initializer raises before creating native storage.
    @base : Pointer(UInt8) = Pointer(UInt8).null
    @phase_cursors = {} of String => Int64
    @lane_cursors = {} of {String, String} => Int64
    @live_views = {} of UInt64 => MetalStageBufferView
    @next_lease_id : UInt64 = 1_u64

    def initialize(size_bytes : Int64, max_buffer_bytes : Int64? = nil)
      raise ArgumentError.new("Stage pool size must be positive") unless size_bytes > 0
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        raise "Native stage pool requires Metal" unless ML::Metal::Device.init!
        device_limit = MetalStagePoolPlatform.gs_max_buffer_length
        raise "Metal reported an invalid maxBufferLength" unless device_limit > 0
        @page_size = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
        raise "Unable to read VM page size" unless @page_size > 0
        requested_limit = max_buffer_bytes || device_limit
        raise ArgumentError.new("Stage view limit must be positive") unless requested_limit > 0
        @max_buffer_bytes = Math.min(requested_limit, device_limit)
      {% else %}
        raise "Metal stage pool requires a Darwin build with Metal enabled"
      {% end %}

      raise ArgumentError.new("Stage pool size overflows page rounding") if size_bytes > Int64::MAX - (@page_size - 1)
      @size = align(size_bytes)
      @base = LibC.mmap(Pointer(Void).null, @size.to_u64,
        LibC::PROT_READ | LibC::PROT_WRITE,
        LibC::MAP_PRIVATE | LibC::MAP_ANON, -1, 0).as(UInt8*)
      raise "Unable to reserve stage pool backing memory" if @base.as(Void*) == LibC::MAP_FAILED
      @released = false
    end

    # Produces a page-aligned no-copy MTLBuffer view. A wrapper never exceeds
    # maxBufferLength even when the backing mmap is much larger.
    def allocate!(phase : String, lane : String, bytes : Int64) : MetalStageBufferView
      ensure_live!
      validate_name!(phase, "phase")
      validate_name!(lane, "lane")
      raise ArgumentError.new("Stage view size must be positive") unless bytes > 0
      raise ArgumentError.new("Stage view size overflows page rounding") if bytes > Int64::MAX - (@page_size - 1)
      mapped = align(bytes)
      raise ArgumentError.new("Stage view exceeds device maxBufferLength") if mapped > @max_buffer_bytes

      offset = @phase_cursors[phase]? || 0_i64
      unless offset <= @size && mapped <= @size - offset
        raise ArgumentError.new("Stage pool capacity exceeded for phase #{phase}")
      end
      pointer = @base + offset
      raise ArgumentError.new("Stage view pointer is not page aligned") unless pointer.address % @page_size.to_u64 == 0

      buffer = ML::MetalBuffer.wrap_no_copy(pointer.as(Void*), mapped)
      unless buffer.valid? && buffer.size == mapped && buffer.contents == pointer.as(Void*)
        buffer.release if buffer.valid?
        raise "Metal stage view did not preserve its no-copy mapping"
      end
      lease_id = @next_lease_id
      raise "Stage view lease id exhausted" if lease_id == UInt64::MAX
      @next_lease_id += 1_u64
      view = MetalStageBufferView.new(self, buffer, phase, lane, offset, bytes, mapped, lease_id)
      @live_views[lease_id] = view
      next_offset = offset + mapped
      @phase_cursors[phase] = next_offset
      @lane_cursors[{phase, lane}] = next_offset
      view
    end

    # A phase can rewind only after every view from that phase has been
    # explicitly released. Other phases may still retain overlapping views.
    def reset_phase!(phase : String) : Nil
      ensure_live!
      validate_name!(phase, "phase")
      if @live_views.values.any? { |view| view.phase == phase }
        raise ArgumentError.new("Cannot reset a phase with live stage views")
      end
      @phase_cursors.delete(phase)
      @lane_cursors.reject! { |key, _| key[0] == phase }
    end

    def phase_cursor(phase : String) : Int64
      @phase_cursors[phase]? || 0_i64
    end

    def lane_cursor(phase : String, lane : String) : Int64
      @lane_cursors[{phase, lane}]? || 0_i64
    end

    def live_view_count : Int32
      @live_views.size.to_i32
    end

    # Release is deliberately explicit and fails while any wrapper remains
    # live. No finalizer can unmap memory beneath a queued GPU command.
    def release : Nil
      return if @released
      raise ArgumentError.new("Cannot release stage pool with live views") unless @live_views.empty?
      status = LibC.munmap(@base.as(Void*), @size.to_u64)
      raise "Unable to unmap stage pool backing memory" unless status == 0
      @released = true
      @phase_cursors.clear
      @lane_cursors.clear
    end

    def release_view!(view : MetalStageBufferView, arena_id : UInt64?) : Nil
      ensure_live!
      registered = @live_views[view.lease_id]?
      unless registered && registered.same?(view) && view.owner.same?(self)
        raise ArgumentError.new("Unknown or already released stage buffer view")
      end
      if arena_id
        unless view.arena_owned? && view.arena_owner_matches?(arena_id)
          raise ArgumentError.new("Stage buffer view Arena owner mismatch")
        end
      elsif view.arena_owned?
        raise ArgumentError.new("Arena-owned stage view must be released by its Arena")
      end
      view.buffer.release
      view.mark_released!
      @live_views.delete(view.lease_id)
    end

    private def align(bytes : Int64) : Int64
      ((bytes + @page_size - 1) // @page_size) * @page_size
    end

    private def validate_name!(value : String, kind : String) : Nil
      raise ArgumentError.new("Stage #{kind} name must not be empty") if value.empty?
    end

    private def ensure_live! : Nil
      raise ArgumentError.new("Stage pool has been released") if @released
    end
  end
end
