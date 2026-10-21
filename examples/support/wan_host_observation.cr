# Benchmark-only host observations for bounded Wan measurements.
# These snapshots read process/GC counters only: they do not touch Metal
# buffers, submit work, or wait for a GPU command.
module WanHostObservation
  {% if flag?(:darwin) && !flag?(:wasm32) %}
    lib WanHostObservationFFI
      fun capture_process_resources = gs_drama3_process_resources(
        values : UInt64*, sample_begin : Float64*, sample_end : Float64*,
      ) : Int32
    end
  {% end %}

  @@origin : Time::Instant? = nil

  # A process-relative monotonic reading suitable for ordering arm boundaries.
  def self.monotonic_seconds : Float64
    origin = @@origin
    unless origin
      origin = Time.instant
      @@origin = origin
    end
    (Time.instant - origin).total_seconds
  end

  struct Snapshot
    getter monotonic_seconds : Float64
    getter valid_mask : Int32
    getter native_sample_begin_seconds : Float64
    getter native_sample_end_seconds : Float64
    getter user_ns : UInt64?
    getter system_ns : UInt64?
    getter pageins : UInt64?
    getter faults : UInt64?
    getter cow_faults : UInt64?
    getter disk_read_bytes : UInt64?
    getter resident_bytes : UInt64?
    getter footprint_bytes : UInt64?
    getter gc_allocated_bytes : UInt64
    getter gc_collection_count : UInt64

    def initialize(@monotonic_seconds : Float64, @valid_mask : Int32,
                   @native_sample_begin_seconds : Float64,
                   @native_sample_end_seconds : Float64,
                   values : StaticArray(UInt64, 8),
                   @gc_allocated_bytes : UInt64,
                   @gc_collection_count : UInt64)
      if @valid_mask & 1 != 0
        @user_ns = values[0]
        @system_ns = values[1]
      end
      if @valid_mask & 2 != 0
        @pageins = values[2]
        @faults = values[3]
        @cow_faults = values[4]
      end
      if @valid_mask & 4 != 0
        @disk_read_bytes = values[5]
        @resident_bytes = values[6]
        @footprint_bytes = values[7]
      end
    end

    def self.capture : Snapshot
      monotonic = WanHostObservation.monotonic_seconds
      values = StaticArray(UInt64, 8).new(0_u64)
      sample_begin = 0.0_f64
      sample_end = 0.0_f64
      valid = 0
      {% if flag?(:darwin) && !flag?(:wasm32) %}
        valid = WanHostObservationFFI.capture_process_resources(
          values.to_unsafe, pointerof(sample_begin), pointerof(sample_end))
      {% end %}
      Snapshot.new(monotonic, valid, sample_begin, sample_end, values,
        GC.stats.total_bytes.to_u64, GC.prof_stats.gc_no.to_u64)
    end
  end

  private def self.counter_delta(before : UInt64?, after : UInt64?) : UInt64?
    return unless before && after
    return if after < before
    after - before
  end

  private def self.gauge_delta(before : UInt64?, after : UInt64?) : Int64?
    return unless before && after
    if after >= before
      delta = after - before
      return if delta > Int64::MAX.to_u64
      delta.to_i64
    else
      delta = before - after
      return if delta > Int64::MAX.to_u64
      -delta.to_i64
    end
  end

  # Counter decreases and unavailable sensor groups are represented as null.
  # Resident/footprint values are gauges, so signed decreases remain visible.
  def self.between(before : Snapshot, after : Snapshot)
    {
      valid_mask_before:                  before.valid_mask,
      valid_mask_after:                   after.valid_mask,
      snapshot_monotonic_seconds_before:  before.monotonic_seconds,
      snapshot_monotonic_seconds_after:   after.monotonic_seconds,
      native_sample_begin_seconds_before: before.native_sample_begin_seconds,
      native_sample_end_seconds_before:   before.native_sample_end_seconds,
      native_sample_begin_seconds_after:  after.native_sample_begin_seconds,
      native_sample_end_seconds_after:    after.native_sample_end_seconds,
      user_ns_before:                     before.user_ns,
      user_ns_after:                      after.user_ns,
      user_ns_delta:                      counter_delta(before.user_ns, after.user_ns),
      system_ns_before:                   before.system_ns,
      system_ns_after:                    after.system_ns,
      system_ns_delta:                    counter_delta(before.system_ns, after.system_ns),
      pageins_before:                     before.pageins,
      pageins_after:                      after.pageins,
      pageins_delta:                      counter_delta(before.pageins, after.pageins),
      faults_before:                      before.faults,
      faults_after:                       after.faults,
      faults_delta:                       counter_delta(before.faults, after.faults),
      cow_faults_before:                  before.cow_faults,
      cow_faults_after:                   after.cow_faults,
      cow_faults_delta:                   counter_delta(before.cow_faults, after.cow_faults),
      disk_read_bytes_before:             before.disk_read_bytes,
      disk_read_bytes_after:              after.disk_read_bytes,
      disk_read_bytes_delta:              counter_delta(before.disk_read_bytes, after.disk_read_bytes),
      resident_bytes_before:              before.resident_bytes,
      resident_bytes_after:               after.resident_bytes,
      resident_bytes_delta:               gauge_delta(before.resident_bytes, after.resident_bytes),
      footprint_bytes_before:             before.footprint_bytes,
      footprint_bytes_after:              after.footprint_bytes,
      footprint_bytes_delta:              gauge_delta(before.footprint_bytes, after.footprint_bytes),
      gc_allocated_bytes_before:          before.gc_allocated_bytes,
      gc_allocated_bytes_after:           after.gc_allocated_bytes,
      gc_allocated_bytes_delta:           counter_delta(before.gc_allocated_bytes, after.gc_allocated_bytes),
      gc_collection_count_before:         before.gc_collection_count,
      gc_collection_count_after:          after.gc_collection_count,
      gc_collection_count_delta:          counter_delta(before.gc_collection_count, after.gc_collection_count),
    }
  end
end
