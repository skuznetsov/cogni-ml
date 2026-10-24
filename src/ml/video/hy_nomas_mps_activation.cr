# Prepared MPSGraph activations matching the pinned PyTorch 2.8 MPS operation
# sequences. Used by the opt-in block backend after same-input bitwise controls;
# the default Metal backend is unchanged.
require "../metal/dispatch"
require "../core/buffer"
require "./metal_ops"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasMPSActivationFFI
    fun plan_create = gs_hy_nomas_mps_activation_plan_create(
      device : Pointer(Void), activation : Int32, shape : Pointer(Int64), rank : Int32,
      input : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_mps_activation_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_hy_nomas_mps_activation_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video::HyNomasMPSActivation
  # A fixed-shape unary graph bound to caller-owned shared F32 buffers. Keep
  # the plan and those buffers alive until the chain's terminal wait.
  class UnaryPlan
    SILU = 0
    GELU = 1

    @shape : Array(Int32)
    @input : ML::MetalBuffer
    @output : ML::MetalBuffer
    @expected_bytes : Int64
    @activation : Int32
    @handle : Pointer(Void) = Pointer(Void).null
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def shape : Array(Int32)
      @shape.dup
    end

    protected def initialize(shape : Array(Int32), input : ML::MetalBuffer,
                             output : ML::MetalBuffer, @activation : Int32)
      raise ArgumentError.new("MPSGraph activation kind is unsupported") unless {SILU, GELU}.includes?(@activation)
      @shape = shape.dup
      @input = input
      @output = output
      raise ArgumentError.new("activation shape rank must be in 1..8") if @shape.size < 1 || @shape.size > 8
      count = 1_i64
      @shape.each do |dimension|
        raise ArgumentError.new("activation shape dimensions must be positive") unless dimension > 0
        raise ArgumentError.new("activation tensor exceeds the native count limit") if count > Int32::MAX // dimension
        count *= dimension
      end
      @expected_bytes = count * sizeof(Float32)
      validate_bindings!
      MetalOps.check(@input, count)
      MetalOps.check(@output, count)
      MetalOps.distinct(@input, @output)

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        shape64 = @shape.map(&.to_i64)
        @handle = HyNomasMPSActivationFFI.plan_create(
          ML::Metal::Device.instance.device_handle, @activation,
          shape64.to_unsafe, @shape.size.to_i32, @input.handle, @output.handle,
        )
        raise "Failed to prepare pinned PyTorch MPSGraph activation plan" if @handle.null?
      {% else %}
        raise ArgumentError.new("MPSGraph activation requires macOS Metal Performance Shaders")
      {% end %}
    end

    # Encode serially on the caller's chain. Repeated use on the same chain is
    # safe because the chain preserves command order; switching chains requires
    # the prior chain's existing terminal wait to have completed.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("MPSGraph activation plan has been released") if @released || @handle.null?
      validate_bindings!
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("MPSGraph activation plan cannot span in-flight command chains")
        end
      end
      @last_chain = chain

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = HyNomasMPSActivationFFI.encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "MPSGraph activation encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("MPSGraph activation requires macOS Metal Performance Shaders")
      {% end %}
    end

    # A plan with encoded work is released only after its actual bound chain
    # drains. The optional argument is checked as an identity witness.
    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the MPS command chain before releasing activation plan") unless previous.completed?
        if supplied = chain
          raise ArgumentError.new("Activation release chain does not match the bound chain") unless previous.same?(supplied)
        end
      elsif chain && !chain.not_nil!.completed?
        raise ArgumentError.new("Complete the supplied MPS chain before releasing activation plan")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasMPSActivationFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @last_chain = nil
      @released = true
    end

    # Preserve native plan and fixed buffers if finalization occurs before the
    # encoded work drains. Explicit release remains the normal ownership path.
    def finalize
      return if @released
      return if @last_chain && !@last_chain.not_nil!.completed?
      release
    end

    private def validate_bindings! : Nil
      raise ArgumentError.new("MPSGraph activation plan has been released") if @released
      unless @input.valid? && @output.valid? &&
             @input.storage_mode == ML::StorageMode::Shared &&
             @output.storage_mode == ML::StorageMode::Shared &&
             @input.size == @expected_bytes && @output.size == @expected_bytes
        raise ArgumentError.new("MPSGraph activation fixed F32 buffer binding changed")
      end
      MetalOps.distinct(@input, @output)
    end
  end

  class SiLUPlan < UnaryPlan
    def initialize(shape : Array(Int32), input : ML::MetalBuffer,
                   output : ML::MetalBuffer)
      super(shape, input, output, UnaryPlan::SILU)
    end
  end

  class GELUPlan < UnaryPlan
    def initialize(shape : Array(Int32), input : ML::MetalBuffer,
                   output : ML::MetalBuffer)
      super(shape, input, output, UnaryPlan::GELU)
    end
  end
end
