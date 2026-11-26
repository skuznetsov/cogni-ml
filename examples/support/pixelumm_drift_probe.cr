require "./pixelumm_gc_probe"
require "./pixelumm_drift_profile"

# Diagnostic only. Profiling reads CPU metadata around existing submissions;
# numerical checks and the sole terminal wait are inherited from the BF16 probe.
module PixelUMMDriftProbe
  extend self
  alias B = PixelUMMFFNWeightReuseProbe
  alias G = PixelUMMGCProbe
  TIMED_REQUESTS = 96
  alias Trial = NamedTuple(index: Int32, profile: Bool, kind: String)

  def schedule : Array(Trial)
    controls = [false, true, false, true, true, false]
    kinds = ["correctness", "correctness", "warmup", "warmup", "warmup", "warmup"]
    balanced = [false, true, true, false, true, false, false, true]
    TIMED_REQUESTS.times do |index|
      controls << balanced[index % balanced.size]
      kinds << "timed"
    end
    controls.map_with_index { |enabled, index| {index: index.to_i32, profile: enabled, kind: kinds[index]} }
  end

  def entry(trial : Trial) : B::ScheduleEntry
    {index: trial[:index], mode: "baseline", kind: trial[:kind],
     capture_mode: trial[:kind] == "correctness", baseline_schedule: "window-major",
     candidate_schedule: "block-major", timing_eligible: trial[:kind] == "timed"}
  end

  lib Host
    # IDs verified against the local Darwin SDK _time.h.
    fun clock_ns = clock_gettime_nsec_np(clock_id : UInt32) : UInt64
    fun thermal_state = gs_pixelumm_probe_thermal_state : Int32
  end

  struct HostPoint
    include JSON::Serializable
    getter wall_ns : UInt64
    getter thread_cpu_ns : UInt64
    getter thermal_state : Int32

    def initialize
      @wall_ns = Host.clock_ns(6_u32)        # CLOCK_MONOTONIC
      @thread_cpu_ns = Host.clock_ns(16_u32) # CLOCK_THREAD_CPUTIME_ID
      @thermal_state = Host.thermal_state
      unless @wall_ns > 0 && @thread_cpu_ns > 0 && (0..3).includes?(@thermal_state)
        raise ArgumentError.new("invalid host clock or advisory thermal-state sample")
      end
    end
  end

  def host_delta(first : HostPoint, last : HostPoint)
    {wall_seconds:       (last.wall_ns - first.wall_ns) / 1e9,
     thread_cpu_seconds: (last.thread_cpu_ns - first.thread_cpu_ns) / 1e9}
  end

  {% unless flag?(:cpu_only) %}
    alias Request = NamedTuple(index: Int32, profile_enabled: Bool, kind: String,
      points: StaticArray(G::Point, 7), host_points: StaticArray(HostPoint, 7),
      metrics: Array(B::Metric), body_encode_seconds: Float64,
      comparator_encode_seconds: Float64, gpu_elapsed_seconds: Float64,
      root_count: Int32, profile: PixelUMMDriftProfile::Profile?, named_allocations: Int32, readbacks: Int32)

    def run_request(owner : B::Owner, trial : Trial,
                    markers : Array(Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker))) : Request
      request = entry(trial)
      chain = nil.as(ML::Metal::MPSCommandChain?)
      wait_attempted = wait_succeeded = false
      points = StaticArray(G::Point, 7).new(G::Point.new)
      host = StaticArray(HostPoint, 7).new(HostPoint.new)
      profile = nil.as(PixelUMMDriftProfile::Profile?)
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      begin
        G.require_boehm!
        points[1] = G::Point.new
        host[1] = HostPoint.new
        chain = ML::Metal::MPSCommandChain.new
        if trial[:profile]
          chain.enable_api_profile!(phase_roots: false)
          chain.enable_continuation_trace!
        end
        owner.begin_request!(request, owner.expected_input_ids)
        owner.track_chain!(chain)
        owner.poison_gate_buffers!(chain)
        owner.clear_metrics!(chain)
        points[2] = G::Point.new
        host[2] = HostPoint.new
        encoded = owner.encode_request!(chain, request, trial[:profile] ? markers : nil)
        unless encoded[:stage_conversion_count] == 0 && encoded[:stage_identity_check_count] == 0 &&
               ML::MetalBuffer.hy_text_probe_allocations == allocations_before &&
               ML::MetalBuffer.hy_text_probe_readbacks == readbacks_before && chain.wait_count == 0
          raise ArgumentError.new("drift probe changed the fixed native command/readback/allocation boundary")
        end
        points[3] = G::Point.new
        host[3] = HostPoint.new
        wait_attempted = true
        chain.commit_and_wait
        wait_succeeded = true
        points[4] = G::Point.new
        host[4] = HostPoint.new
        gpu_seconds = chain.gpu_elapsed_seconds
        roots = chain.segment_count
        if trial[:profile]
          raw = markers.map do |plan|
            plan.map_with_index do |value, index|
              raise ArgumentError.new("native phase marker order changed") unless value.phase == index
              ML::Video::PixelUMMBlockPhaseMarker.new(
                ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES[index],
                value.wall_ns / 1e9, value.thread_cpu_ns / 1e9, value.root_count)
            end.to_a
          end
          api = chain.api_profile
          trace = chain.continuation_trace
          profile = PixelUMMDriftProfile::Profile.new(raw, chain.completed_root_timings,
            gpu_seconds, api, trace)
        end
        points[5] = G::Point.new
        host[5] = HostPoint.new
        metrics = owner.finish_request!(chain, request)
        points[6] = G::Point.new
        host[6] = HostPoint.new
        allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
        readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
        unless allocations == 0 && readbacks == 1 && chain.wait_count == 1
          raise ArgumentError.new("drift probe requires one terminal wait/readback and no named allocations")
        end
        {index: trial[:index], profile_enabled: trial[:profile], kind: trial[:kind],
         points: points, host_points: host, metrics: metrics,
         body_encode_seconds: encoded[:body_encode_seconds],
         comparator_encode_seconds: encoded[:comparator_encode_seconds],
         gpu_elapsed_seconds: gpu_seconds, root_count: roots, profile: profile,
         named_allocations: allocations.to_i32, readbacks: readbacks.to_i32}
      rescue error
        owner.abort_request!
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
        raise error
      end
    end

    def request_json(request : Request)
      points = request[:points]
      host = request[:host_points]
      {index: request[:index], profile_enabled: request[:profile_enabled], kind: request[:kind],
       body_encode_seconds: request[:body_encode_seconds],
       comparator_encode_seconds: request[:comparator_encode_seconds],
       gpu_elapsed_seconds: request[:gpu_elapsed_seconds],
       root_count: request[:root_count],
       phases: {admission_poison: host_delta(host[1], host[2]),
                encode: host_delta(host[2], host[3]), terminal_wait: host_delta(host[3], host[4]),
                profile_extraction: host_delta(host[4], host[5]),
                metrics_release: host_delta(host[5], host[6]),
                critical: host_delta(host[1], host[4]), total: host_delta(host[0], host[6])},
       gc: G.delta(points[0], points[6]), raw_points: points, host_points: host,
       named_buffer_allocations: request[:named_allocations], terminal_wait_count: 1,
       preterminal_readbacks: 0, comparator_readbacks_after_wait: request[:readbacks],
       comparator_readback_bytes: B::METRIC_BYTES,
       numerical_metrics: request[:metrics],
       capture_metrics: request[:kind] == "correctness" ? request[:metrics] : nil,
       profile: request[:profile]}
    end
  {% end %}
end
