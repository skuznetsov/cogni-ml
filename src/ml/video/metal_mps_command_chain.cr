require "../metal/dispatch"
require "./mps_command_phase_profile"

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
      check_live!
      raise ArgumentError.new("MPS phase-root profiling is already enabled") if @phase_root_profile_enabled
      roots = root_count
      raise ArgumentError.new("MPS phase-root profiling requires one fresh command root") unless roots == 1
      @phase_root_profile_enabled = true
      @phase_root_cursor = 0
      @phase_ranges = [] of ML::Video::MPSCommandPhaseRange
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
      raise ArgumentError.new("MPS phase-root profile has too many phase boundaries") if ranges.size >= 3
      end_root = root_count
      raise ArgumentError.new("MPS phase-root boundary contains no roots") if end_root <= @phase_root_cursor
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
      raise ArgumentError.new("MPS phase-root profile requires three continuation boundaries") unless ranges.size == 3
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
  end
end
