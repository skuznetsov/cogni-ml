# Fixed-binding MPSGraph execution for a BF16 Hy-Nomas linear layer.
require "../metal/dispatch"
require "../core/buffer"
require "./metal_ops"
require "./checkpoint"
require "./hy_nomas_block"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasMPSBF16LinearFFI
    fun plan_create = gs_hy_nomas_mps_bf16_linear_plan_create(
      device : Pointer(Void), rows : Int32, input_width : Int32, output_width : Int32,
      variant : Int32, input : Pointer(Void), weight : Pointer(Void), bias : Pointer(Void),
      output_bf16 : Pointer(Void), output_f32 : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_mps_bf16_linear_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_hy_nomas_mps_bf16_linear_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video::HyNomasMPSBF16Linear
  enum Variant
    # The graph's matmul and bias-add tensors are both BF16. MPSGraph may fuse
    # the two nodes internally; this names graph-visible dtype boundaries.
    BF16GraphOps
    # BF16 operands are widened for F32 matmul and bias addition, then the
    # completed result is rounded to BF16 once.
    F32Accumulation

    def self.parse(value : String) : Variant
      case value
      when "bf16-graph" then BF16GraphOps
      when "f32-accum"  then F32Accumulation
      else
        raise ArgumentError.new("BF16 linear variant must be bf16-graph or f32-accum")
      end
    end

    def label : String
      self == BF16GraphOps ? "bf16_graph_ops" : "f32_accumulation_bf16_output"
    end
  end

  # One fixed-binding projection with caller-selected finite geometry. Caller
  # owns all input, checkpoint, and output buffers through the terminal wait.
  class Plan
    # Keep plans and their Crystal-side buffer owners strongly reachable until
    # callers explicitly release them after draining the bound command chain.
    # An abandoned in-flight plan therefore leaks safely until process exit.
    @@lifetime_mutex = Mutex.new
    @@live_plans = [] of Plan

    getter rows : Int32
    getter input_width : Int32
    getter output_width : Int32
    @input : ML::MetalBuffer
    @weights : ML::Video::HyNomasDualStreamBlock::LinearWeights
    @output_bf16 : ML::MetalBuffer
    @output : ML::MetalBuffer
    @variant : Variant
    @input_elements : Int64
    @weight_elements : Int64
    @bias_elements : Int64
    @output_elements : Int64
    @handle : Pointer(Void) = Pointer(Void).null
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@input : ML::MetalBuffer,
                   @weights : ML::Video::HyNomasDualStreamBlock::LinearWeights,
                   @output_bf16 : ML::MetalBuffer,
                   @output : ML::MetalBuffer,
                   @variant : Variant,
                   @rows : Int32 = 1,
                   @input_width : Int32 = 2048,
                   @output_width : Int32 = 12_288)
      @input_elements = checked_count(@rows, @input_width)
      @weight_elements = checked_count(@output_width, @input_width)
      @bias_elements = @output_width.to_i64
      @output_elements = checked_count(@rows, @output_width)
      validate_bindings!
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @handle = HyNomasMPSBF16LinearFFI.plan_create(
          ML::Metal::Device.instance.device_handle,
          @rows, @input_width, @output_width, @variant.value,
          @input.handle, @weights.weight.handle, @weights.bias.handle,
          @output_bf16.handle, @output.handle,
        )
        raise "Failed to prepare fixed-binding Hy-Nomas BF16 linear graph" if @handle.null?
        retain_lifetime_root!
      {% else %}
        raise ArgumentError.new("Hy-Nomas BF16 linear MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def variant : Variant
      @variant
    end

    # Encode one MPSGraph execution on the caller-owned chain. This method
    # creates no Metal buffers and performs no wait or readback.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas BF16 linear plan has been released") if @released || @handle.null?
      validate_bindings!
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas BF16 linear plan cannot span in-flight command chains")
        end
      end
      @last_chain = chain

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = HyNomasMPSBF16LinearFFI.encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "Hy-Nomas BF16 linear encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Hy-Nomas BF16 linear MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the MPS command chain before releasing BF16 linear plan") unless previous.completed?
        if supplied = chain
          raise ArgumentError.new("BF16 linear release chain does not match the bound chain") unless previous.same?(supplied)
        end
      elsif chain && !chain.not_nil!.completed?
        raise ArgumentError.new("Complete the supplied MPS chain before releasing BF16 linear plan")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasMPSBF16LinearFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @last_chain = nil
      @released = true
      release_lifetime_root!
    end

    def finalize
      return if @released
      return if @last_chain && !@last_chain.not_nil!.completed?
      release
    end

    private def validate_bindings! : Nil
      raise ArgumentError.new("Hy-Nomas BF16 linear plan has been released") if @released
      unless @weights.weight_dtype.bf16? && @weights.bias_dtype.bf16?
        raise ArgumentError.new("Hy-Nomas BF16 linear requires BF16 weights and bias")
      end
      buffers = {@input, @weights.weight, @weights.bias, @output_bf16, @output}
      buffers.each do |buffer|
        unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("Hy-Nomas BF16 linear requires valid shared buffers")
        end
      end
      unless @input.size == @input_elements * sizeof(Float32).to_i64 &&
             @weights.weight.size == @weight_elements * 2 &&
             @weights.bias.size == @bias_elements * 2 &&
             @output_bf16.size == @output_elements * 2 &&
             @output.size == @output_elements * sizeof(Float32).to_i64
        raise ArgumentError.new("Hy-Nomas BF16 linear fixed buffer extent mismatch")
      end
      ML::Video::MetalOps.check(@input, @input_elements)
      ML::Video::MetalOps.check_storage(@weights.weight, @weight_elements, ML::Video::WeightDType::BF16)
      ML::Video::MetalOps.check_storage(@weights.bias, @bias_elements, ML::Video::WeightDType::BF16)
      ML::Video::MetalOps.check_storage(@output_bf16, @output_elements, ML::Video::WeightDType::BF16)
      ML::Video::MetalOps.check(@output, @output_elements)
      buffers.each_with_index do |left, index|
        buffers.each_with_index do |right, other_index|
          next if other_index <= index
          raise ArgumentError.new("Hy-Nomas BF16 linear buffers must not alias") if ML::Video::MetalOps.overlap?(left, right)
        end
      end
    end

    private def checked_count(rows : Int32, width : Int32) : Int64
      raise ArgumentError.new("Hy-Nomas BF16 linear dimensions must be positive") unless rows > 0 && width > 0
      count = rows.to_i64 * width.to_i64
      if count > Int32::MAX.to_i64
        raise ArgumentError.new("Hy-Nomas BF16 linear tensor exceeds the native element-count limit")
      end
      count
    end

    private def retain_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_plans << self unless @@live_plans.any? { |current| current.same?(self) }
      end
    end

    private def release_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_plans.reject! { |current| current.same?(self) }
      end
    end
  end
end
