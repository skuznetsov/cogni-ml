require "../metal/dispatch"

{% unless flag?(:cpu_only) %}
  lib MPSCommandChainFFI
    fun create = gs_mps_chain_create(queue : Void*) : Void*
    fun wrapper = gs_mps_chain_wrapper(chain : Void*) : Void*
    fun current_root = gs_mps_chain_current_root(chain : Void*) : Void*
    fun continue = gs_mps_chain_continue(chain : Void*) : Int32
    fun finish_wait = gs_mps_chain_finish_wait(chain : Void*, seconds : Float64*, segments : Int32*) : Int32
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
  end
end
