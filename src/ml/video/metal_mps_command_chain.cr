require "../metal/dispatch"
require "./mps_command_phase_profile"
require "./mps_api_profile"
require "./mps_continuation_trace"

{% unless flag?(:cpu_only) %}
  lib MPSCommandChainFFI
    fun create = gs_mps_chain_create(queue : Void*) : Void*
    fun wrapper = gs_mps_chain_wrapper(chain : Void*) : Void*
    fun current_root = gs_mps_chain_current_root(chain : Void*) : Void*
    fun continue = gs_mps_chain_continue(chain : Void*) : Int32
    fun finish_wait = gs_mps_chain_finish_wait(chain : Void*, seconds : Float64*, segments : Int32*) : Int32
    fun root_count = gs_mps_chain_root_count(chain : Void*, count : Int32*) : Int32
    fun root_timing = gs_mps_chain_root_timing(chain : Void*, sequence : Int32,
                                               completed : Int32*, has_error : Int32*, gpu_start : Float64*, gpu_end : Float64*) : Int32
    fun release = gs_mps_chain_release(chain : Void*) : Int32
  end

  lib MPSAPIProfileFFI
    struct Stats
      count : UInt64
      wall_total_ns : UInt64
      wall_max_ns : UInt64
      thread_cpu_total_ns : UInt64
      thread_cpu_max_ns : UInt64
      nested_count : UInt64
      nested_wall_total_ns : UInt64
      nested_thread_cpu_total_ns : UInt64
      exclusive_wall_total_ns : UInt64
      exclusive_thread_cpu_total_ns : UInt64
    end

    fun enable = gs_mps_chain_api_profile_enable(chain : Void*) : Int32
    fun set_phase = gs_mps_chain_api_profile_set_phase(chain : Void*, phase : Int32) : Int32
    fun snapshot = gs_mps_chain_api_profile_snapshot(chain : Void*, phase : Int32,
                                                     operation : Int32, stats : Stats*) : Int32
  end

  lib MPSContinuationTraceFFI
    struct Info
      owner_thread_id : UInt64
      mach_timebase_numer : UInt64
      mach_timebase_denom : UInt64
      trace_start_mach_ticks : UInt64
      event_capacity : UInt64
      event_count : UInt64
      root_count : UInt64
      failed : UInt64
    end

    struct Event
      serial : UInt64
      begin_mach_ticks : UInt64
      end_mach_ticks : UInt64
      begin_thread_cpu_ns : UInt64
      end_thread_cpu_ns : UInt64
      api_scope_serial : UInt64
      parent_graph_serial : UInt64
      before_root_sequence : UInt64
      after_root_sequence : UInt64
      phase : UInt32
      graph_depth : UInt32
      before_root_status : UInt32
      before_root_status_after_call : UInt32
      after_root_status : UInt32
      flags : UInt32
    end

    struct Root
      sequence : UInt64
      tracked_or_observed_mach_ticks : UInt64
      completion_observed_mach_ticks : UInt64
      gpu_start_time_bits : UInt64
      gpu_end_time_bits : UInt64
      status_at_track_or_enable : UInt32
      final_status : UInt32
      has_error : UInt32
      completion_source : UInt32
    end

    fun enable = gs_mps_chain_api_trace_enable(chain : Void*) : Int32
    fun info = gs_mps_chain_api_trace_info(chain : Void*, info : Info*) : Int32
    fun event = gs_mps_chain_api_trace_event(chain : Void*, index : UInt64, event : Event*) : Int32
    fun root = gs_mps_chain_api_trace_root(chain : Void*, index : UInt64, root : Root*) : Int32
  end
{% end %}

module ML::Metal
  # Exclusive serial chain. Encoder values are borrowed until the next pause;
  # callers must not retain or use an old copy after pause!/resume!. Graph plans
  # and buffers must remain resident through the one terminal wait.
  class MPSCommandChain
    @native : Pointer(Void) = Pointer(Void).null
    @encoder : ComputeEncoder? = nil
    @completed = false
    @status = 0
    @phase_root_profile_enabled = false
    @phase_root_profile_finished = false
    @api_profile_enabled = false
    @continuation_trace_enabled = false
    @phase_root_cursor = 0_i32
    @phase_ranges : Array(ML::Video::MPSCommandPhaseRange)? = nil
    getter gpu_elapsed_seconds : Float64 = 0.0
    getter segment_count : Int32 = 0
    getter wait_count : Int32 = 0

    def initialize
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless Device.available?
        @native = MPSCommandChainFFI.create(Device.instance.queue_handle)
        raise "Failed to create MPS command chain" if @native.null?
        resume!
      {% end %}
    end

    def encoder : ComputeEncoder
      check_live!
      @encoder || raise ArgumentError.new("MPS chain encoder is paused")
    end

    def pause! : Nil
      e = encoder
      e.end_encoding
      @encoder = nil
    end

    def handle : Pointer(Void)
      check_live!
      raise ArgumentError.new("Pause the native encoder before borrowing MPS command chain") if @encoder
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        MPSCommandChainFFI.wrapper(@native)
      {% end %}
    end

    # Borrow the current underlying root only while no encoder is open. I/O
    # event dependencies must follow MPSGraph's latest commitAndContinue root.
    def root_handle : Pointer(Void)
      handle
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        root = MPSCommandChainFFI.current_root(@native)
        raise "MPS chain has no current root" if root.null?
        root
      {% end %}
    end

    def resume! : Nil
      check_live!
      raise ArgumentError.new("MPS chain encoder is already open") if @encoder
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        root = MPSCommandChainFFI.current_root(@native)
        raise "MPS chain has no current root" if root.null?
        @encoder = ComputeEncoder.new(root)
      {% end %}
    end

    # Optional asynchronous continuation, also used to exercise root switching
    # without depending on MPSGraph's private scheduling heuristic.
    def continue_async! : Nil
      handle
      {% unless flag?(:cpu_only) %}
        raise "MPS continuation rejected" unless MPSCommandChainFFI.continue(@native) == 0
      {% end %}
    end

    def commit_and_wait : Nil
      unless @completed
        check_live!
        pause! if @encoder
        {% unless flag?(:cpu_only) %}
          @wait_count += 1
          @status = MPSCommandChainFFI.finish_wait(@native, pointerof(@gpu_elapsed_seconds), pointerof(@segment_count))
        {% end %}
        @completed = true
      end
      raise "MPS command chain completion failed (#{@status})" unless @status == 0
    end

    # Enable source-bound block phase profiling on a fresh chain. The caller
    # must create the chain immediately before WanS2VBlock.encode, with no
    # prelude in its first root, and must terminal-wait immediately after the
    # block returns, with no suffix or audio work in the final root. Root
    # timestamps cannot separate such same-root work from the named phases.
    def enable_phase_root_profile! : Nil
      initialize_phase_root_profile!
    end

    # Enables per-chain MPS API wall/thread-CPU timing. Root ranges are optional
    # because short diagnostic fixtures may not produce a completed root that
    # isolates the attention interval from the following projection tail.
    def enable_api_profile!(phase_roots : Bool = true) : Nil
      check_live!
      raise ArgumentError.new("MPS API profiling is already enabled") if @api_profile_enabled
      raise ArgumentError.new("MPS phase-root profiling is already enabled") if @phase_root_profile_enabled
      roots = root_count
      raise ArgumentError.new("MPS API profiling requires one fresh command root") unless roots == 1
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        status = MPSAPIProfileFFI.enable(@native)
        raise ArgumentError.new("MPS API profiling could not attach to the command chain (#{status})") unless status == 0
      {% end %}
      initialize_phase_root_profile! if phase_roots
      @api_profile_enabled = true
    end

    private def initialize_phase_root_profile! : Nil
      check_live!
      raise ArgumentError.new("MPS phase-root profiling is already enabled") if @phase_root_profile_enabled
      roots = root_count
      raise ArgumentError.new("MPS phase-root profiling requires one fresh command root") unless roots == 1
      @phase_root_profile_enabled = true
      @phase_root_cursor = 0
      @phase_ranges = [] of ML::Video::MPSCommandPhaseRange
    end

    def api_profile_enabled? : Bool
      @api_profile_enabled
    end

    # The fixed native trace buffer is allocated once before block encoding.
    # Enabling it does not enqueue work or change continuation boundaries.
    def enable_continuation_trace! : Nil
      check_live!
      require_api_profile!
      raise ArgumentError.new("MPS continuation tracing is already enabled") if @continuation_trace_enabled
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        status = MPSContinuationTraceFFI.enable(@native)
        raise ArgumentError.new("MPS continuation trace could not attach (#{status})") unless status == 0
      {% end %}
      @continuation_trace_enabled = true
    end

    def continuation_trace : ML::Video::MPSContinuationTrace
      unless @continuation_trace_enabled && @completed && @wait_count == 1 && @status == 0
        raise ArgumentError.new("MPS continuation trace requires enable and the successful existing terminal wait")
      end
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raw_info = MPSContinuationTraceFFI::Info.new
        status = MPSContinuationTraceFFI.info(@native, pointerof(raw_info))
        raise ArgumentError.new("MPS continuation trace info failed (#{status})") unless status == 0
        unless raw_info.failed == 0 && raw_info.event_capacity == ML::Video::MPSContinuationTrace::CAPACITY &&
               raw_info.event_count > 0 && raw_info.event_count <= raw_info.event_capacity &&
               raw_info.root_count == raw_info.event_count + 1 && raw_info.root_count == @segment_count
          raise ArgumentError.new("MPS continuation trace has invalid native counts")
        end
        info = ML::Video::MPSContinuationTraceInfo.new(raw_info.owner_thread_id,
          raw_info.mach_timebase_numer, raw_info.mach_timebase_denom,
          raw_info.trace_start_mach_ticks, raw_info.event_capacity,
          raw_info.event_count, raw_info.root_count, raw_info.failed)
        events = Array(ML::Video::MPSContinuationTraceEvent).new(raw_info.event_count.to_i)
        raw_info.event_count.times do |index|
          raw = MPSContinuationTraceFFI::Event.new
          status = MPSContinuationTraceFFI.event(@native, index, pointerof(raw))
          raise ArgumentError.new("MPS continuation trace event #{index} failed (#{status})") unless status == 0
          events << ML::Video::MPSContinuationTraceEvent.new(raw.serial,
            raw.begin_mach_ticks, raw.end_mach_ticks,
            raw.begin_thread_cpu_ns, raw.end_thread_cpu_ns,
            raw.api_scope_serial, raw.parent_graph_serial,
            raw.before_root_sequence, raw.after_root_sequence, raw.phase,
            raw.graph_depth, raw.before_root_status, raw.before_root_status_after_call,
            raw.after_root_status, raw.flags)
        end
        roots = Array(ML::Video::MPSContinuationTraceRoot).new(raw_info.root_count.to_i)
        raw_info.root_count.times do |index|
          raw = MPSContinuationTraceFFI::Root.new
          status = MPSContinuationTraceFFI.root(@native, index, pointerof(raw))
          raise ArgumentError.new("MPS continuation trace root #{index} failed (#{status})") unless status == 0
          roots << ML::Video::MPSContinuationTraceRoot.new(raw.sequence,
            raw.tracked_or_observed_mach_ticks, raw.completion_observed_mach_ticks,
            raw.gpu_start_time_bits, raw.gpu_end_time_bits,
            raw.status_at_track_or_enable, raw.final_status, raw.has_error,
            raw.completion_source)
        end
        result = ML::Video::MPSContinuationTrace.build(info, events, roots)
        result.validate_api_profile!(api_profile)
        result
      {% end %}
    end

    # Tags subsequent wrapped MPS API calls. Phase 0 is the unassigned/outside
    # bucket and is set immediately before the existing terminal wait.
    def set_api_profile_phase!(phase_id : Int32) : Nil
      check_live!
      require_api_profile!
      raise ArgumentError.new("MPS API profile phase is finished") if @phase_root_profile_finished
      raise ArgumentError.new("MPS API profile phase id is unknown") unless ML::Video::MPSAPIProfile::PHASE_NAMES.has_key?(phase_id)
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        status = MPSAPIProfileFFI.set_phase(@native, phase_id)
        raise ArgumentError.new("MPS API profile phase update failed (#{status})") unless status == 0
      {% end %}
    end

    def phase_root_profile_enabled? : Bool
      @phase_root_profile_enabled
    end

    # Called only at existing continuation boundaries, before pause/continue.
    # The returned count is the exclusive end of this phase's root range.
    def record_phase_root_boundary!(name : String) : Int32
      check_live!
      require_phase_root_profile!
      raise ArgumentError.new("MPS phase-root profile is already finished") if @phase_root_profile_finished
      ranges = @phase_ranges || raise ArgumentError.new("MPS phase-root profile ranges are unavailable")
      boundary_limit = @api_profile_enabled ? 4 : 3
      raise ArgumentError.new("MPS phase-root profile has too many phase boundaries") if ranges.size >= boundary_limit
      end_root = root_count
      raise ArgumentError.new("MPS phase-root boundary contains no roots") if end_root <= @phase_root_cursor
      ranges << ML::Video::MPSCommandPhaseRange.new(name, @phase_root_cursor, end_root)
      @phase_root_cursor = end_root
      end_root
    end

    # Split before the currently open root without ending or continuing it.
    # The open root remains in the following phase, whose name must disclose
    # any attention tail that shares that root with projection/residual work.
    def record_precurrent_root_phase_boundary!(name : String) : Int32
      check_live!
      require_phase_root_profile!
      require_api_profile!
      raise ArgumentError.new("MPS phase-root profile is already finished") if @phase_root_profile_finished
      ranges = @phase_ranges || raise ArgumentError.new("MPS phase-root profile ranges are unavailable")
      raise ArgumentError.new("MPS phase-root profile has too many phase boundaries") if ranges.size >= 4
      end_root = ML::Video::MPSAPIPhaseBoundary.precurrent_root_end_count(root_count, @phase_root_cursor)
      ranges << ML::Video::MPSCommandPhaseRange.new(name, @phase_root_cursor, end_root)
      @phase_root_cursor = end_root
      end_root
    end

    # Validate the one root created by the existing continuation. No root is
    # added for profiling; this catches count-vs-index mistakes at the source.
    def verify_phase_root_continuation!(prior_end_count : Int32) : Nil
      check_live!
      require_phase_root_profile!
      after_count = root_count
      unless after_count == prior_end_count + 1 && after_count - 1 == prior_end_count
        raise ArgumentError.new("MPS phase-root continuation did not create exactly one next root")
      end
    end

    # Called inside WanS2VBlock.encode after the final output has been encoded.
    # Callers must append no suffix or audio work before the terminal wait;
    # command-root timing cannot separate work sharing the final root.
    def finish_phase_root_profile! : Nil
      check_live!
      require_phase_root_profile!
      raise ArgumentError.new("MPS phase-root profile is already finished") if @phase_root_profile_finished
      ranges = @phase_ranges || raise ArgumentError.new("MPS phase-root profile ranges are unavailable")
      expected_boundaries = @api_profile_enabled ? 4 : 3
      raise ArgumentError.new("MPS phase-root profile has an incomplete phase boundary set") unless ranges.size == expected_boundaries
      end_root = root_count
      raise ArgumentError.new("MPS final phase contains no roots") if end_root <= @phase_root_cursor
      ranges << ML::Video::MPSCommandPhaseRange.new("ffn", @phase_root_cursor, end_root)
      @phase_root_cursor = end_root
      @phase_root_profile_finished = true
    end

    # Root timestamps are unavailable as a valid profile until the existing
    # terminal wait has completed and all root callbacks/fallbacks are recorded.
    def phase_root_profile : ML::Video::MPSCommandPhaseProfile
      require_phase_root_profile!
      unless @completed && @wait_count == 1
        raise ArgumentError.new("MPS phase-root profile is available only after the terminal wait")
      end
      raise ArgumentError.new("MPS command chain completion failed (#{@status})") unless @status == 0
      raise ArgumentError.new("MPS phase-root profile was not finished by the block") unless @phase_root_profile_finished

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        roots = Array(ML::Video::MPSCommandRootTiming).new(@segment_count)
        @segment_count.times do |sequence|
          completed = 0_i32
          has_error = 0_i32
          gpu_start = 0.0
          gpu_end = 0.0
          status = MPSCommandChainFFI.root_timing(@native, sequence.to_i32,
            pointerof(completed), pointerof(has_error), pointerof(gpu_start), pointerof(gpu_end))
          raise ArgumentError.new("Failed to read MPS root timing summary #{sequence} (#{status})") unless status == 0
          roots << ML::Video::MPSCommandRootTiming.new(sequence.to_i32,
            completed == 1, has_error != 0, gpu_start, gpu_end)
        end
        ranges = @phase_ranges || raise ArgumentError.new("MPS phase-root profile ranges are unavailable")
        ML::Video::MPSCommandPhaseProfile.build(roots, ranges, @gpu_elapsed_seconds)
      {% end %}
    end

    # Native API stats become valid after the existing terminal wait has
    # completed. The fixed phase/operation matrix makes missing snapshots a
    # hard diagnostic failure rather than silently reporting partial totals.
    def api_profile : ML::Video::MPSAPIProfile
      require_api_profile!
      unless @completed && @wait_count == 1
        raise ArgumentError.new("MPS API profile is available only after the terminal wait")
      end
      raise ArgumentError.new("MPS command chain completion failed (#{@status})") unless @status == 0
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        samples = [] of ML::Video::MPSAPIProfileSample
        ML::Video::MPSAPIProfile::PHASE_NAMES.keys.sort.each do |phase_id|
          ML::Video::MPSAPIProfile::OPERATION_NAMES.keys.sort.each do |operation_id|
            raw = MPSAPIProfileFFI::Stats.new
            status = MPSAPIProfileFFI.snapshot(@native, phase_id, operation_id, pointerof(raw))
            unless status == 0
              raise ArgumentError.new("MPS API profile snapshot failed for phase #{phase_id}, operation #{operation_id} (#{status})")
            end
            stats = ML::Video::MPSAPITimingStats.new(
              raw.count, raw.wall_total_ns, raw.wall_max_ns,
              raw.thread_cpu_total_ns, raw.thread_cpu_max_ns,
              raw.nested_count, raw.nested_wall_total_ns,
              raw.nested_thread_cpu_total_ns, raw.exclusive_wall_total_ns,
              raw.exclusive_thread_cpu_total_ns)
            samples << ML::Video::MPSAPIProfileSample.new(phase_id, operation_id, stats)
          end
        end
        ML::Video::MPSAPIProfile.build(samples)
      {% end %}
    end

    def completed? : Bool
      @completed
    end

    # No implicit wait or finalizer: ownership must end explicitly after the
    # terminal wait, including on an exception after a graph encoded work.
    def release : Nil
      return if @native.null?
      raise ArgumentError.new("Complete MPS command chain before release") unless @completed
      {% unless flag?(:cpu_only) %}
        raise "MPS command chain release rejected" unless MPSCommandChainFFI.release(@native) == 0
      {% end %}
      @native = Pointer(Void).null
    end

    private def check_live! : Nil
      raise ArgumentError.new("MPS command chain is completed or released") if @completed || @native.null?
    end

    private def root_count : Int32
      check_live!
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        count = 0_i32
        status = MPSCommandChainFFI.root_count(@native, pointerof(count))
        raise ArgumentError.new("Failed to query MPS command-root count (#{status})") unless status == 0
        count
      {% end %}
    end

    private def require_phase_root_profile! : Nil
      raise ArgumentError.new("MPS phase-root profiling is not enabled") unless @phase_root_profile_enabled
    end

    private def require_api_profile! : Nil
      raise ArgumentError.new("MPS API profiling is not enabled") unless @api_profile_enabled
    end
  end
end
