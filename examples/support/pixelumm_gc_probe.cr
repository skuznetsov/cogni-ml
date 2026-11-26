require "json"
require "./pixelumm_ffn_weight_reuse_probe"

# A standalone diagnostic: no inference default or production GC policy changes.
module PixelUMMGCProbe
  extend self

  NORMAL       = "normal"
  DISABLED     = "scoped_disabled"
  TIMED_TRIALS = 64
  CLOCK_ORIGIN = Time.instant
  alias Trial = NamedTuple(index: Int32, policy: String, kind: String)

  def schedule : Array(Trial)
    policies = [NORMAL, DISABLED, NORMAL, DISABLED, DISABLED, NORMAL]
    kinds = ["correctness", "correctness", "warmup", "warmup", "warmup", "warmup"]
    balanced = [NORMAL, DISABLED, DISABLED, NORMAL, DISABLED, NORMAL, NORMAL, DISABLED]
    TIMED_TRIALS.times do |index|
      policies << balanced[index % balanced.size]
      kinds << "timed"
    end
    policies.map_with_index { |policy, index| {index: index.to_i32, policy: policy, kind: kinds[index]} }
  end

  def entry(trial : Trial, tail : Bool = false) : PixelUMMFFNWeightReuseProbe::ScheduleEntry
    {index: trial[:index] * 2 + (tail ? 1 : 0), mode: "baseline",
     kind: tail ? "tail" : trial[:kind], capture_mode: !tail && trial[:kind] == "correctness",
     baseline_schedule: "window-major", candidate_schedule: "block-major",
     timing_eligible: !tail && trial[:kind] == "timed"}
  end

  # No allocation or collector API is allowed in the native event callback.
  # Field order is shared with pixelumm_gc_observer.h and checked by its CPU test.
  lib Observer
    struct Counters
      collection_count : UInt64
      collection_total_ns : UInt64
      collection_max_ns : UInt64
      rendezvous_pause_count : UInt64
      rendezvous_pause_total_ns : UInt64
      rendezvous_pause_max_ns : UInt64
      stopped_interval_count : UInt64
      stopped_interval_total_ns : UInt64
      stopped_interval_max_ns : UInt64
      unmatched_event_count : UInt64
      active : UInt32
      callback_installed : UInt32
    end

    fun start = pixelumm_gc_observer_start : Int32
    fun stop = pixelumm_gc_observer_stop : Int32
    fun snapshot = pixelumm_gc_observer_snapshot(out : Counters*) : Int32
  end

  def require_boehm! : Nil
    {% if flag?(:gc_none) || flag?(:wasm32) || flag?(:tracing) || !flag?(:darwin) %}
      raise ArgumentError.new("GC probe requires macOS Boehm without tracing")
    {% else %}
      raise ArgumentError.new("GC probe must start with GC enabled") if LibGC.is_disabled != 0
    {% end %}
  end

  def with_policy(policy : String, &)
    unless policy == NORMAL || policy == DISABLED
      raise ArgumentError.new("unknown bounded GC policy")
    end
    require_boehm!
    if policy == DISABLED
      GC.disable
      begin
        yield
      ensure
        GC.enable
      end
    else
      yield
    end
  end

  def start_observer! : Nil
    require_boehm!
    check_observer!(Observer.start, "start")
  end

  def stop_observer! : Nil
    check_observer!(Observer.stop, "stop")
  end

  def check_observer!(status : Int32, operation : String) : Nil
    unless status == 0
      raise ArgumentError.new("GC observer #{operation} failed with status #{status}")
    end
  end

  struct Point
    include JSON::Serializable
    getter time_ns : Int64
    getter allocated_bytes : UInt64
    getter gc_no : UInt64
    getter heap_bytes : UInt64
    getter free_bytes : UInt64
    getter bytes_since_gc : UInt64
    getter bytes_before_gc : UInt64
    getter collection_count : UInt64
    getter collection_total_ns : UInt64
    getter collection_max_ns : UInt64
    getter rendezvous_pause_count : UInt64
    getter rendezvous_pause_total_ns : UInt64
    getter rendezvous_pause_max_ns : UInt64
    getter stopped_interval_count : UInt64
    getter stopped_interval_total_ns : UInt64
    getter stopped_interval_max_ns : UInt64
    getter unmatched_event_count : UInt64

    def initialize
      stats = GC.stats
      prof = GC.prof_stats
      PixelUMMGCProbe.check_observer!(Observer.snapshot(out native), "snapshot")
      unless native.active == 1 && native.callback_installed == 1 && native.unmatched_event_count == 0
        raise ArgumentError.new("GC observer lost callback ownership or event pairing")
      end
      @time_ns = (Time.instant - CLOCK_ORIGIN).total_nanoseconds.to_i64
      @allocated_bytes = stats.total_bytes
      @gc_no = prof.gc_no
      @heap_bytes = stats.heap_size
      @free_bytes = stats.free_bytes
      @bytes_since_gc = prof.bytes_since_gc
      @bytes_before_gc = prof.bytes_before_gc
      @collection_count = native.collection_count
      @collection_total_ns = native.collection_total_ns
      @collection_max_ns = native.collection_max_ns
      @rendezvous_pause_count = native.rendezvous_pause_count
      @rendezvous_pause_total_ns = native.rendezvous_pause_total_ns
      @rendezvous_pause_max_ns = native.rendezvous_pause_max_ns
      @stopped_interval_count = native.stopped_interval_count
      @stopped_interval_total_ns = native.stopped_interval_total_ns
      @stopped_interval_max_ns = native.stopped_interval_max_ns
      @unmatched_event_count = native.unmatched_event_count
    end
  end

  # Maxima are cumulative values in Point, not subtractable phase maxima.
  # Collection envelope and pause windows overlap and must never be summed.
  def delta(first : Point, last : Point)
    {seconds:                  (last.time_ns - first.time_ns) / 1e9,
     allocated_bytes:          last.allocated_bytes - first.allocated_bytes,
     collections:              last.collection_count - first.collection_count,
     gc_no_delta:              last.gc_no - first.gc_no,
     collection_seconds:       (last.collection_total_ns - first.collection_total_ns) / 1e9,
     rendezvous_pauses:        last.rendezvous_pause_count - first.rendezvous_pause_count,
     rendezvous_pause_seconds: (last.rendezvous_pause_total_ns - first.rendezvous_pause_total_ns) / 1e9,
     stopped_intervals:        last.stopped_interval_count - first.stopped_interval_count,
     stopped_seconds:          (last.stopped_interval_total_ns - first.stopped_interval_total_ns) / 1e9}
  end

  {% unless flag?(:cpu_only) %}
    # The arrays are created before the timing loop. Serialization is deferred
    # until every trial and its mandatory ordinary-GC tail have completed.
    alias Request = NamedTuple(index: Int32, policy: String, kind: String,
      points: StaticArray(Point, 7), metrics: Array(PixelUMMFFNWeightReuseProbe::Metric),
      body_encode_seconds: Float64, comparator_encode_seconds: Float64,
      gpu_elapsed_seconds: Float64, named_allocations: Int32, readbacks: Int32)

    def run_request(owner : PixelUMMFFNWeightReuseProbe::Owner,
                    request : PixelUMMFFNWeightReuseProbe::ScheduleEntry,
                    policy : String) : Request
      chain = nil.as(ML::Metal::MPSCommandChain?)
      wait_attempted = false
      wait_succeeded = false
      initial = Point.new
      points = StaticArray(Point, 7).new(initial)
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      body_seconds = comparator_seconds = gpu_seconds = 0.0
      begin
        with_policy(policy) do
          begin
            points[1] = Point.new
            chain = ML::Metal::MPSCommandChain.new
            owner.begin_request!(request, owner.expected_input_ids)
            owner.track_chain!(chain.not_nil!)
            owner.poison_gate_buffers!(chain.not_nil!)
            owner.clear_metrics!(chain.not_nil!)
            points[2] = Point.new
            encoded = owner.encode_request!(chain.not_nil!, request)
            body_seconds = encoded[:body_encode_seconds]
            comparator_seconds = encoded[:comparator_encode_seconds]
            unless encoded[:stage_conversion_count] == 0 && encoded[:stage_identity_check_count] == 0 &&
                   ML::MetalBuffer.hy_text_probe_allocations == allocations_before &&
                   ML::MetalBuffer.hy_text_probe_readbacks == readbacks_before && chain.not_nil!.wait_count == 0
              raise ArgumentError.new("GC probe changed the fixed native baseline command/readback/allocation boundary")
            end
            points[3] = Point.new
            wait_attempted = true
            chain.not_nil!.commit_and_wait
            wait_succeeded = true
            points[4] = Point.new
          rescue error
            # Drain before with_policy's ensure restores GC. Even exceptional
            # partial submissions retain the bounded disable-through-wait scope.
            begin
              abort_and_drain!(owner, chain, wait_attempted, wait_succeeded)
            ensure
              chain = nil
            end
            raise error
          end
        end
        points[5] = Point.new
        require_boehm!
        gpu_seconds = chain.not_nil!.gpu_elapsed_seconds
        metrics = owner.finish_request!(chain.not_nil!, request)
        points[6] = Point.new
        allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
        readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
        unless allocations == 0 && readbacks == 1 && chain.not_nil!.wait_count == 1
          raise ArgumentError.new("GC probe requires one terminal wait/readback and no named allocations")
        end
        if policy == DISABLED && points[4].collection_count != points[1].collection_count
          raise ArgumentError.new("a collection occurred while GC was disabled")
        end
        {index: request[:index], policy: policy, kind: request[:kind], points: points,
         metrics: metrics, body_encode_seconds: body_seconds, comparator_encode_seconds: comparator_seconds,
         gpu_elapsed_seconds: gpu_seconds, named_allocations: allocations.to_i32, readbacks: readbacks.to_i32}
      rescue error
        abort_and_drain!(owner, chain, wait_attempted, wait_succeeded) if chain
        raise error
      end
    end

    private def abort_and_drain!(owner : PixelUMMFFNWeightReuseProbe::Owner,
                                 chain : ML::Metal::MPSCommandChain?,
                                 wait_attempted : Bool, wait_succeeded : Bool) : Nil
      owner.abort_request!
      # Identical drain policy to the existing probe: at most the sole wait,
      # and a failed wait never authorizes a retry or in-flight buffer release.
      if chain
        if !wait_attempted && !chain.completed? && chain.wait_count == 0
          begin
            chain.commit_and_wait
            wait_succeeded = true
          rescue
          end
        end
        if wait_succeeded && chain.completed? && chain.wait_count == 1
          if owner.lifecycle.attempted && !owner.lifecycle.terminal_wait_succeeded
            owner.lifecycle.mark_cleanup_wait!(completed: true, wait_count: 1)
          end
          chain.release
        end
      end
    end

    def request_json(request : Request)
      points = request[:points]
      {index: request[:index], policy: request[:policy], kind: request[:kind],
       phases: {policy_enter: delta(points[0], points[1]),
                admission_poison: delta(points[1], points[2]),
                encode_body_and_comparators: delta(points[2], points[3]),
                terminal_wait: delta(points[3], points[4]),
                policy_exit: delta(points[4], points[5]),
                metrics_release: delta(points[5], points[6]),
                critical: delta(points[1], points[4]),
                total: delta(points[0], points[6])},
       body_encode_seconds: request[:body_encode_seconds],
       comparator_encode_seconds: request[:comparator_encode_seconds],
       gpu_elapsed_seconds: request[:gpu_elapsed_seconds],
       named_buffer_allocations: request[:named_allocations], terminal_wait_count: 1,
       preterminal_readbacks: 0, comparator_readbacks_after_wait: request[:readbacks],
       comparator_readback_bytes: PixelUMMFFNWeightReuseProbe::METRIC_BYTES,
       raw_points: points,
       capture_metrics: request[:kind] == "correctness" ? request[:metrics] : nil}
    end
  {% end %}
end
