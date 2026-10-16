require "json"

{% if flag?(:darwin) && !flag?(:gc_none) && !flag?(:wasm32) %}
  # Crystal's GC library already links libgc on this target. These two optional
  # symbols are available when Boehm is built with clock support.
  lib LibGC
    fun start_performance_measurement = GC_start_performance_measurement
    fun get_full_gc_total_time = GC_get_full_gc_total_time : Word
  end
{% end %}

module ML::Audio::Drama3
  # Keep the enum order stable: it is also the index order for the fixed-size
  # profile arrays below.
  enum HostEvent
    Backing
    BufferView
    Workspace
    Pipeline
    EncoderCreate
    EncoderAPI
    Retention
    CommandCreate
    CommandSubmit
    CompletionWait
    EncoderAcquire
  end

  enum HostCounter
    PendingEntries
    UniqueRetained
    RetainCallbacks
    GenericDispatches
    BindingEntries
    ParameterElements
    PipelineBindings
    BufferBindings
    ByteBindings
    KernelDispatches
    EncoderEnds
  end

  # These lengths must match the enum members above. Fixed arrays keep the hot
  # measurement path free of per-scope frame and collection allocations.
  HOST_EVENT_COUNT         = 11
  HOST_COUNTER_COUNT       = 11
  HOST_PROFILE_MAX_NESTING = 64

  # Process-wide Boehm timer. It reports cumulative milliseconds spent in full
  # collections only. The API is unsynchronized, so callers sample it at coarse
  # phase boundaries and never force a collection from this utility.
  module HostGCClock
    {% if flag?(:darwin) && !flag?(:gc_none) && !flag?(:wasm32) %}
      @@started = false
      @@mutex = Mutex.new
    {% end %}

    def self.start_once : Nil
      {% if flag?(:darwin) && !flag?(:gc_none) && !flag?(:wasm32) %}
        @@mutex.synchronize do
          unless @@started
            LibGC.start_performance_measurement
            @@started = true
          end
        end
      {% end %}
    end

    def self.full_gc_ms : UInt64?
      {% if flag?(:darwin) && !flag?(:gc_none) && !flag?(:wasm32) %}
        @@started ? LibGC.get_full_gc_total_time.to_u64 : nil
      {% else %}
        nil
      {% end %}
    end
  end

  # Cumulative process GC readings at a phase boundary. `allocated_bytes` is
  # GC.stats.total_bytes; `collection_count` is Boehm's cumulative gc_no and
  # includes startup/empty collections. Both counters may wrap.
  struct HostGCSnapshot
    getter allocated_bytes : UInt64
    getter collection_count : UInt64
    getter full_gc_ms : UInt64?

    def initialize(@allocated_bytes : UInt64, @collection_count : UInt64, @full_gc_ms : UInt64?)
    end

    def self.capture : HostGCSnapshot
      HostGCSnapshot.new(GC.stats.total_bytes, GC.prof_stats.gc_no, HostGCClock.full_gc_ms)
    end
  end

  struct HostProfileSnapshot
    getter calls : StaticArray(Int64, HOST_EVENT_COUNT)
    getter inclusive_ns : StaticArray(Int64, HOST_EVENT_COUNT)
    getter exclusive_ns : StaticArray(Int64, HOST_EVENT_COUNT)
    getter counters : StaticArray(Int64, HOST_COUNTER_COUNT)
    getter gc : HostGCSnapshot

    def initialize(
      @calls : StaticArray(Int64, HOST_EVENT_COUNT),
      @inclusive_ns : StaticArray(Int64, HOST_EVENT_COUNT),
      @exclusive_ns : StaticArray(Int64, HOST_EVENT_COUNT),
      @counters : StaticArray(Int64, HOST_COUNTER_COUNT),
      @gc : HostGCSnapshot,
    )
    end
  end

  # `measure` result type for JSON output. Allocations in this structure happen
  # only when converting a coarse snapshot delta, never in the measured scope.
  struct HostProfileDeltaEvent
    include JSON::Serializable

    getter name : String
    getter calls : Int64
    getter inclusive_ns : Int64
    getter exclusive_ns : Int64

    def initialize(@name : String, @calls : Int64, @inclusive_ns : Int64, @exclusive_ns : Int64)
    end
  end

  struct HostProfileDeltaGC
    include JSON::Serializable

    getter allocated_bytes : UInt64
    getter collection_count : UInt64
    getter full_gc_ms : UInt64?

    def initialize(@allocated_bytes : UInt64, @collection_count : UInt64, @full_gc_ms : UInt64?)
    end
  end

  struct HostProfileDelta
    include JSON::Serializable

    getter events : Array(HostProfileDeltaEvent)
    getter counters : Hash(String, Int64)
    getter gc : HostProfileDeltaGC

    def initialize(@events : Array(HostProfileDeltaEvent), @counters : Hash(String, Int64), @gc : HostProfileDeltaGC)
    end
  end

  # A per-context, single-owner accumulator. The hot path uses fixed arrays,
  # monotonic nanosecond ticks, and no mutex, Hash, or heap-allocated frame.
  class HostProfile
    getter enabled : Bool

    def initialize(@enabled : Bool)
      HostGCClock.start_once if @enabled
      @calls = StaticArray(Int64, HOST_EVENT_COUNT).new(0_i64)
      @inclusive_ns = StaticArray(Int64, HOST_EVENT_COUNT).new(0_i64)
      @exclusive_ns = StaticArray(Int64, HOST_EVENT_COUNT).new(0_i64)
      @counters = StaticArray(Int64, HOST_COUNTER_COUNT).new(0_i64)
      @stack_children_ns = StaticArray(Int64, HOST_PROFILE_MAX_NESTING).new(0_i64)
      @depth = 0
    end

    def enabled? : Bool
      @enabled
    end

    # On the extremely unlikely path beyond the fixed nesting capacity, the
    # operation remains transparent but is not included in the profile.
    def measure(event : HostEvent, &block : -> T) : T forall T
      return yield unless @enabled

      depth = @depth
      return yield if depth >= HOST_PROFILE_MAX_NESTING

      index = event.value
      @stack_children_ns[depth] = 0_i64
      started = Crystal::System::Time.ticks
      @depth = depth + 1

      begin
        yield
      ensure
        elapsed = (Crystal::System::Time.ticks &- started).to_i64!
        child_elapsed = @stack_children_ns[depth]
        @depth = depth

        @calls[index] = @calls[index] &+ 1_i64
        @inclusive_ns[index] = @inclusive_ns[index] &+ elapsed
        @exclusive_ns[index] = @exclusive_ns[index] &+ (elapsed &- child_elapsed)

        if depth > 0
          @stack_children_ns[depth - 1] = @stack_children_ns[depth - 1] &+ elapsed
        end
      end
    end

    def add(counter : HostCounter, amount : Int64 = 1_i64) : Nil
      return unless @enabled

      index = counter.value
      @counters[index] = @counters[index] &+ amount
    end

    # GC values are sampled even when fine profiling is disabled so callers can
    # compare coarse request phases without turning on event timers/counters.
    def snapshot : HostProfileSnapshot
      HostProfileSnapshot.new(@calls, @inclusive_ns, @exclusive_ns, @counters, HostProfile.gc_snapshot)
    end

    def self.gc_snapshot : HostGCSnapshot
      HostGCSnapshot.capture
    end

    def self.gc_delta(before : HostGCSnapshot, after : HostGCSnapshot) : HostProfileDeltaGC
      full_gc_ms = if before_ms = before.full_gc_ms
                     if after_ms = after.full_gc_ms
                       after_ms &- before_ms
                     end
                   end

      HostProfileDeltaGC.new(
        after.allocated_bytes &- before.allocated_bytes,
        after.collection_count &- before.collection_count,
        full_gc_ms
      )
    end

    def self.delta(before : HostProfileSnapshot, after : HostProfileSnapshot) : HostProfileDelta
      events = [] of HostProfileDeltaEvent
      HostEvent.each do |event|
        index = event.value
        calls = after.calls[index] &- before.calls[index]
        inclusive = after.inclusive_ns[index] &- before.inclusive_ns[index]
        exclusive = after.exclusive_ns[index] &- before.exclusive_ns[index]
        next if calls.zero? && inclusive.zero? && exclusive.zero?

        events << HostProfileDeltaEvent.new(event.to_s, calls, inclusive, exclusive)
      end

      counters = {} of String => Int64
      HostCounter.each do |counter|
        index = counter.value
        counters[counter.to_s] = after.counters[index] &- before.counters[index]
      end

      HostProfileDelta.new(events, counters, gc_delta(before.gc, after.gc))
    end
  end
end
