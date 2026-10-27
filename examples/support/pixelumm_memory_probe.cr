require "json"
require "c/time"
require "../../src/ml/metal/device"

module PixelUMMMemoryProbe
  PREFIX = "PIXELUMM_MEMORY "
  SCHEMA = "cogni_ml.pixelumm_memory_phase/v1"
  RUSAGE_INFO_V2 = 2

  # Exact 64-bit macOS rusage_info_v2 field order from sys/resource.h, matching
  # examples/support/wan_memory_guard.py. The full shape keeps the two sampled
  # counters at their ABI offsets instead of relying on a truncated struct.
  @[Extern]
  struct RUsageInfoV2
    property ri_uuid : StaticArray(UInt8, 16)
    property ri_user_time : UInt64
    property ri_system_time : UInt64
    property ri_pkg_idle_wkups : UInt64
    property ri_interrupt_wkups : UInt64
    property ri_pageins : UInt64
    property ri_wired_size : UInt64
    property ri_resident_size : UInt64
    property ri_phys_footprint : UInt64
    property ri_proc_start_abstime : UInt64
    property ri_proc_exit_abstime : UInt64
    property ri_child_user_time : UInt64
    property ri_child_system_time : UInt64
    property ri_child_pkg_idle_wkups : UInt64
    property ri_child_interrupt_wkups : UInt64
    property ri_child_pageins : UInt64
    property ri_child_elapsed_abstime : UInt64
    property ri_diskio_bytesread : UInt64
    property ri_diskio_byteswritten : UInt64

    def initialize
      @ri_uuid = StaticArray(UInt8, 16).new(0_u8)
      @ri_user_time = 0_u64
      @ri_system_time = 0_u64
      @ri_pkg_idle_wkups = 0_u64
      @ri_interrupt_wkups = 0_u64
      @ri_pageins = 0_u64
      @ri_wired_size = 0_u64
      @ri_resident_size = 0_u64
      @ri_phys_footprint = 0_u64
      @ri_proc_start_abstime = 0_u64
      @ri_proc_exit_abstime = 0_u64
      @ri_child_user_time = 0_u64
      @ri_child_system_time = 0_u64
      @ri_child_pkg_idle_wkups = 0_u64
      @ri_child_interrupt_wkups = 0_u64
      @ri_child_pageins = 0_u64
      @ri_child_elapsed_abstime = 0_u64
      @ri_diskio_bytesread = 0_u64
      @ri_diskio_byteswritten = 0_u64
    end

    def resident_bytes : UInt64
      ri_resident_size
    end

    def physical_footprint_bytes : UInt64
      ri_phys_footprint
    end
  end

  @[Extern]
  struct MachTimebaseInfo
    property numer : UInt32
    property denom : UInt32

    def initialize
      @numer = 0_u32
      @denom = 0_u32
    end
  end

  {% if flag?(:darwin) %}
    @[Link("proc")]
    lib LibProc
      fun proc_pid_rusage(pid : Int32, flavor : Int32, buffer : RUsageInfoV2*) : Int32
    end

    @[Link("System")]
    lib LibMachClock
      fun mach_absolute_time : UInt64
      fun mach_timebase_info(info : MachTimebaseInfo*) : Int32
    end
  {% end %}

  struct Readings
    getter process_phys_footprint_bytes : Int64?
    getter process_rss_bytes : Int64?
    getter metal_device_allocated_bytes : Int64?
    getter named_metal_buffer_live_count : Int32?
    getter named_metal_buffer_live_bytes : Int64?
    getter named_metal_buffer_peak_bytes : Int64?
    getter crystal_gc_heap_bytes : Int64?
    getter crystal_gc_free_bytes : Int64?
    getter crystal_gc_unmapped_bytes : Int64?
    getter crystal_gc_total_bytes : Int64?

    def initialize(
      @process_phys_footprint_bytes : Int64? = nil,
      @process_rss_bytes : Int64? = nil,
      @metal_device_allocated_bytes : Int64? = nil,
      @named_metal_buffer_live_count : Int32? = nil,
      @named_metal_buffer_live_bytes : Int64? = nil,
      @named_metal_buffer_peak_bytes : Int64? = nil,
      @crystal_gc_heap_bytes : Int64? = nil,
      @crystal_gc_free_bytes : Int64? = nil,
      @crystal_gc_unmapped_bytes : Int64? = nil,
      @crystal_gc_total_bytes : Int64? = nil,
    )
    end
  end

  # Injectable scalar sampler and clock keep event formatting testable without
  # initializing Metal or depending on host memory pressure.
  class Recorder
    getter events : Array(JSON::Any)

    @sequence = 0_i64
    @last_timestamp_ns = -1_i64
    @last_guard_timestamp_ns : Int64? = nil

    def self.system(output : IO = STDERR) : Recorder
      new(output, ->{ PixelUMMMemoryProbe.capture_readings },
        ->{ PixelUMMMemoryProbe.clock_monotonic_ns },
        ->{ PixelUMMMemoryProbe.guard_monotonic_ns })
    end

    def initialize(@output : IO, @sample : Proc(Readings), @clock : Proc(Int64),
                   @guard_clock : Proc(Int64?))
      @events = [] of JSON::Any
    end

    def emit(phase : String, *, slot : Int32? = nil, request : Int32? = nil,
             phase_bytes_kind : String? = nil, phase_bytes : Int64? = nil) : JSON::Any
      timestamp_ns = @clock.call
      if timestamp_ns < 0 || timestamp_ns < @last_timestamp_ns
        raise ArgumentError.new("memory probe clock must be non-negative and monotonic")
      end
      guard_timestamp_ns = @guard_clock.call
      if guard_timestamp_ns && (guard_timestamp_ns < 0 || (@last_guard_timestamp_ns && guard_timestamp_ns < @last_guard_timestamp_ns.not_nil!))
        raise ArgumentError.new("memory probe guard clock must be non-negative and monotonic")
      end
      readings = @sample.call
      @sequence += 1
      event = JSON.parse({
        schema: SCHEMA,
        sequence: @sequence,
        timestamp_clock: "CLOCK_MONOTONIC",
        monotonic_ns: timestamp_ns,
        guard_timestamp_clock: "mach_absolute_time",
        guard_monotonic_ns: guard_timestamp_ns,
        phase: phase,
        slot: slot,
        request: request,
        phase_bytes_kind: phase_bytes_kind,
        phase_bytes: nonnegative(phase_bytes),
        process_phys_footprint_bytes: positive(readings.process_phys_footprint_bytes),
        process_rss_bytes: positive(readings.process_rss_bytes),
        metal_device_allocated_bytes: nonnegative(readings.metal_device_allocated_bytes),
        named_metal_buffer_live_count: nonnegative(readings.named_metal_buffer_live_count),
        named_metal_buffer_live_bytes: nonnegative(readings.named_metal_buffer_live_bytes),
        named_metal_buffer_peak_bytes: nonnegative(readings.named_metal_buffer_peak_bytes),
        crystal_gc_heap_bytes: nonnegative(readings.crystal_gc_heap_bytes),
        crystal_gc_free_bytes: nonnegative(readings.crystal_gc_free_bytes),
        crystal_gc_unmapped_bytes: nonnegative(readings.crystal_gc_unmapped_bytes),
        crystal_gc_total_bytes: nonnegative(readings.crystal_gc_total_bytes),
        memory_categories_additive: false,
        memory_category_notes: "Process footprint, RSS, Metal allocation, named MetalBuffer counters, and Crystal GC counters overlap; do not sum them. Opaque MPS bytes are not estimated.",
      }.to_json)
      @last_timestamp_ns = timestamp_ns
      @last_guard_timestamp_ns = guard_timestamp_ns if guard_timestamp_ns
      @events << event
      @output.puts("#{PREFIX}#{event.to_json}")
      @output.flush
      event
    end

    private def nonnegative(value : Int64?) : Int64?
      return unless value && value >= 0
      value
    end

    private def nonnegative(value : Int32?) : Int32?
      return unless value && value >= 0
      value
    end

    private def positive(value : Int64?) : Int64?
      return unless value && value > 0
      value
    end
  end

  # Collect teardown errors without letting any one observer or resource
  # failure prevent the remaining releases from being attempted.
  class Cleanup
    getter errors : Array(Exception)

    def initialize
      @errors = [] of Exception
    end

    def attempt(&block) : Nil
      begin
        yield
      rescue error
        @errors << error
      end
      nil
    end

    def phase(observer : Recorder?, name : String, *, slot : Int32? = nil,
              request : Int32? = nil, phase_bytes_kind : String? = nil,
              phase_bytes : Int64? = nil) : Nil
      return unless active = observer
      attempt do
        active.emit(name, slot: slot, request: request,
          phase_bytes_kind: phase_bytes_kind, phase_bytes: phase_bytes)
      end
    end

    def raise_first! : Nil
      if error = @errors.first?
        raise error
      end
      nil
    end
  end

  def self.clock_monotonic_ns : Int64
    timespec = uninitialized LibC::Timespec
    result = LibC.clock_gettime(LibC::CLOCK_MONOTONIC, pointerof(timespec))
    raise RuntimeError.new("clock_gettime(CLOCK_MONOTONIC) failed") unless result == 0
    seconds, nanoseconds = timespec.tv_sec.to_i64, timespec.tv_nsec.to_i64
    raise RuntimeError.new("CLOCK_MONOTONIC returned a negative timestamp") if seconds < 0 || nanoseconds < 0
    seconds &* 1_000_000_000_i64 &+ nanoseconds
  end

  # Python's time.monotonic_ns() on macOS is based on mach_absolute_time.
  # Keep this reading separate from clock_gettime(CLOCK_MONOTONIC): their
  # epochs differ on the supported host, so consumers must join on this field.
  def self.guard_monotonic_ns : Int64?
    {% if flag?(:darwin) %}
      timebase = uninitialized MachTimebaseInfo
      return unless LibMachClock.mach_timebase_info(pointerof(timebase)) == 0 && timebase.denom > 0
      nanoseconds = LibMachClock.mach_absolute_time.to_u128 &* timebase.numer.to_u128
      nanoseconds //= timebase.denom.to_u128
      return if nanoseconds > Int64::MAX.to_u128
      nanoseconds.to_i64
    {% else %}
      nil
    {% end %}
  end

  def self.capture_readings : Readings
    footprint_bytes, rss_bytes = process_readings
    metal_bytes = ML::Metal::Device.current_allocated_size_if_initialized
    named = ML::MetalBuffer.stats
    gc = GC.stats
    Readings.new(
      process_phys_footprint_bytes: footprint_bytes,
      process_rss_bytes: rss_bytes,
      metal_device_allocated_bytes: metal_bytes,
      named_metal_buffer_live_count: named[:live_buffers],
      named_metal_buffer_live_bytes: named[:live_bytes],
      named_metal_buffer_peak_bytes: named[:peak_bytes],
      crystal_gc_heap_bytes: int64_or_nil(gc.heap_size),
      crystal_gc_free_bytes: int64_or_nil(gc.free_bytes),
      crystal_gc_unmapped_bytes: int64_or_nil(gc.unmapped_bytes),
      crystal_gc_total_bytes: int64_or_nil(gc.total_bytes),
    )
  end

  private def self.process_readings : {Int64?, Int64?}
    {% if flag?(:darwin) %}
      usage = uninitialized RUsageInfoV2
      if LibProc.proc_pid_rusage(Process.pid.to_i32, RUSAGE_INFO_V2, pointerof(usage)) == 0
        return {int64_or_nil(usage.physical_footprint_bytes), int64_or_nil(usage.resident_bytes)}
      end
    {% end %}
    {nil, nil}
  end

  private def self.int64_or_nil(value : UInt64) : Int64?
    return if value > Int64::MAX.to_u64
    value.to_i64
  end
end
