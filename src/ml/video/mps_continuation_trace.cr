require "json"
require "./mps_api_profile"

# Post-wait validation of the opt-in fixed-capacity native continuation trace.
# Root creation, submission, GPU completion and callback delivery are distinct
# events. A continuation bounds submission; it does not timestamp it exactly.
module ML::Video
  record MPSContinuationTraceInfo,
    owner_thread_id : UInt64, mach_timebase_numer : UInt64,
    mach_timebase_denom : UInt64, trace_start_mach_ticks : UInt64,
    event_capacity : UInt64, event_count : UInt64, root_count : UInt64,
    failed : UInt64 do
    include JSON::Serializable
  end

  record MPSContinuationTraceEvent,
    serial : UInt64, begin_mach_ticks : UInt64, end_mach_ticks : UInt64,
    begin_thread_cpu_ns : UInt64, end_thread_cpu_ns : UInt64,
    api_scope_serial : UInt64, parent_graph_serial : UInt64,
    before_root_sequence : UInt64, after_root_sequence : UInt64,
    phase : UInt32, graph_depth : UInt32, before_root_status : UInt32,
    before_root_status_after_call : UInt32, after_root_status : UInt32,
    flags : UInt32 do
    include JSON::Serializable
  end

  record MPSContinuationTraceRoot,
    sequence : UInt64, tracked_or_observed_mach_ticks : UInt64,
    completion_observed_mach_ticks : UInt64,
    gpu_start_time_bits : UInt64, gpu_end_time_bits : UInt64,
    status_at_track_or_enable : UInt32, final_status : UInt32,
    has_error : UInt32, completion_source : UInt32 do
    def gpu_start_seconds : Float64
      gpu_start_time_bits.unsafe_as(Float64)
    end

    def gpu_end_seconds : Float64
      gpu_end_time_bits.unsafe_as(Float64)
    end

    def to_json(json : JSON::Builder) : Nil
      {
        sequence: sequence,
        tracked_or_observed_mach_ticks: tracked_or_observed_mach_ticks,
        completion_observed_mach_ticks: completion_observed_mach_ticks,
        gpu_start_time_bits: gpu_start_time_bits, gpu_end_time_bits: gpu_end_time_bits,
        status_at_track_or_enable: status_at_track_or_enable,
        final_status: final_status, has_error: has_error,
        completion_source: completion_source,
        gpu_start_seconds: gpu_start_seconds, gpu_end_seconds: gpu_end_seconds,
      }.to_json(json)
    end
  end

  struct MPSContinuationTrace
    CAPACITY = 4096_u64
    # This bound covers timestamp reporting/conversion only. It is unrelated
    # to and does not widen any tensor comparison tolerance.
    CLOCK_ROUNDING_SECONDS = 2.0e-6
    # Native enable rejects CLOCK_MONOTONIC resolutions above 1 microsecond.
    # Its independently rounded endpoints cannot support a nanosecond-level
    # enclosure assertion against mach ticks. This is a clock bound only.
    AGGREGATE_WALL_ROUNDING_NS_PER_CALL = 2000.0

    getter info : MPSContinuationTraceInfo
    getter events : Array(MPSContinuationTraceEvent)
    getter roots : Array(MPSContinuationTraceRoot)

    private def initialize(@info : MPSContinuationTraceInfo,
                           @events : Array(MPSContinuationTraceEvent),
                           @roots : Array(MPSContinuationTraceRoot))
    end

    def host_seconds(ticks : UInt64) : Float64
      ticks.to_f64 * @info.mach_timebase_numer.to_f64 /
        @info.mach_timebase_denom.to_f64 / 1.0e9
    end

    def self.build(info : MPSContinuationTraceInfo,
                   events : Array(MPSContinuationTraceEvent),
                   roots : Array(MPSContinuationTraceRoot)) : self
      unless info.failed == 0 && info.owner_thread_id > 0 &&
             info.mach_timebase_numer > 0 && info.mach_timebase_denom > 0 &&
             info.trace_start_mach_ticks > 0 && info.event_capacity == CAPACITY &&
             info.event_count > 0 && info.event_count <= info.event_capacity &&
             info.event_count == events.size && info.root_count == roots.size &&
             info.root_count == info.event_count + 1
        raise ArgumentError.new("MPS continuation trace has invalid counts, clock, owner or failure state")
      end
      result = new(info, events, roots)
      roots.each_with_index do |root, index|
        unless root.sequence == index && root.final_status == 4 &&
               root.has_error == 0 && {1_u32, 2_u32}.includes?(root.completion_source) &&
               root.status_at_track_or_enable <= 1 &&
               root.tracked_or_observed_mach_ticks >= info.trace_start_mach_ticks &&
               root.completion_observed_mach_ticks >= root.tracked_or_observed_mach_ticks
          raise ArgumentError.new("MPS continuation trace has incomplete or invalid root #{index}")
        end
        start_time = root.gpu_start_seconds
        end_time = root.gpu_end_seconds
        unless start_time.finite? && end_time.finite? && start_time > 0.0 && end_time > start_time &&
               result.host_seconds(root.tracked_or_observed_mach_ticks) <= start_time + CLOCK_ROUNDING_SECONDS &&
               end_time <= result.host_seconds(root.completion_observed_mach_ticks) + CLOCK_ROUNDING_SECONDS
          raise ArgumentError.new("MPS continuation trace GPU/host clock bounds disagree for root #{index}")
        end
      end

      previous_end = info.trace_start_mach_ticks
      previous_api_serial = 0_u64
      events.each_with_index do |event, index|
        nested = event.graph_depth > 0
        unless event.serial == index + 1 && event.before_root_sequence == index &&
               event.after_root_sequence == index + 1 &&
               event.api_scope_serial > previous_api_serial &&
               event.begin_mach_ticks >= previous_end && event.end_mach_ticks > event.begin_mach_ticks &&
               event.begin_thread_cpu_ns > 0 &&
               event.end_thread_cpu_ns >= event.begin_thread_cpu_ns &&
               MPSAPIProfileSchema::PHASE_NAMES.has_key?(event.phase.to_i32) &&
               event.graph_depth <= 16 && event.flags == (nested ? 15_u32 : 14_u32) &&
               (nested ? event.parent_graph_serial > 0 && event.parent_graph_serial < event.api_scope_serial : event.parent_graph_serial == 0) &&
               event.before_root_status <= 1 &&
               (2_u32..4_u32).includes?(event.before_root_status_after_call) &&
               event.after_root_status <= 1 &&
               roots[index].tracked_or_observed_mach_ticks <= event.begin_mach_ticks &&
               roots[index + 1].tracked_or_observed_mach_ticks >= event.end_mach_ticks
          raise ArgumentError.new("MPS continuation trace has invalid event order, root, status or nesting at #{index}")
        end
        # The submitted current root cannot start on the GPU before entry into
        # its continuation call. No assertion equates callback time to GPU end.
        if roots[index].gpu_start_seconds + CLOCK_ROUNDING_SECONDS < result.host_seconds(event.begin_mach_ticks)
          raise ArgumentError.new("MPS continuation trace root started before its submission bound")
        end
        previous_end = event.end_mach_ticks
        previous_api_serial = event.api_scope_serial
      end
      result
    end

    def validate_api_profile!(profile : MPSAPIProfile) : Nil
      profile.phases.each do |phase|
        stats = phase.operations["commit_and_continue"]
        calls = @events.select { |event| event.phase == phase.phase_id }
        unless stats.count == calls.size &&
               stats.nested_count == calls.count { |event| event.graph_depth > 0 }
          raise ArgumentError.new("MPS continuation trace disagrees with aggregate API counts")
        end
        wall_ns = calls.sum do |event|
          (event.end_mach_ticks - event.begin_mach_ticks).to_f64 *
            @info.mach_timebase_numer.to_f64 / @info.mach_timebase_denom.to_f64
        end
        cpu_ns = calls.sum { |event| event.end_thread_cpu_ns - event.begin_thread_cpu_ns }
        # The aggregate timer encloses these narrower per-call samples.
        if wall_ns > stats.wall_total_ns.to_f64 + calls.size * AGGREGATE_WALL_ROUNDING_NS_PER_CALL ||
           cpu_ns > stats.thread_cpu_total_ns
          raise ArgumentError.new("MPS continuation trace exceeds its enclosing API timer in phase #{phase.phase_id}: trace wall/CPU #{wall_ns}/#{cpu_ns}, aggregate #{stats.wall_total_ns}/#{stats.thread_cpu_total_ns}")
        end
      end
    end
  end
end
