# Prepared BF16 MPSGraph plans for DCAE-V normalization and SiLU. All graph
# tensor bindings are fixed during setup and remain resident until the caller's
# MPS command chain has completed its terminal wait.
require "../metal/dispatch"
require "../core/buffer"
require "./metal_ops"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib DCAEVNormMPSFFI
    fun rms_plan_create = gs_dcae_v_rms_norm_mps_plan_create(
      device : Pointer(Void), shape : Pointer(Int64), rank : Int32,
      input : Pointer(Void), full_scratch : Pointer(Void),
      mean_a : Pointer(Void), mean_b : Pointer(Void),
      weight : Pointer(Void), bias : Pointer(Void), epsilon : Float32,
    ) : Pointer(Void)
    fun rms_encode = gs_dcae_v_rms_norm_mps_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun rms_release = gs_dcae_v_rms_norm_mps_plan_release(plan : Pointer(Void)) : Void
    fun silu_plan_create = gs_dcae_v_silu_mps_plan_create(
      device : Pointer(Void), shape : Pointer(Int64), rank : Int32,
      input : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun silu_encode = gs_dcae_v_silu_mps_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun silu_release = gs_dcae_v_silu_mps_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video::DCAEVNormMPS
  def self.checked_count(label : String, shape : Array(Int32)) : Int32
    raise ArgumentError.new("DCAE-V #{label} shape must have rank five") unless shape.size == 5
    count = 1_i64
    shape.each do |dimension|
      unless dimension > 0 && count <= Int32::MAX.to_i64 // dimension
        raise ArgumentError.new("DCAE-V #{label} exceeds the supported element count")
      end
      count *= dimension
    end
    count.to_i32
  end

  def self.checked_rows(label : String, shape : Array(Int32)) : Int32
    checked_count(label, [shape[0], shape[2], shape[3], shape[4], 1])
  end

  def self.bytes(count : Int32) : Int64
    count.to_i64 * 2_i64
  end

  def self.validate_buffer!(buffer : ML::MetalBuffer, count : Int32,
                            label : String, exact : Bool = true) : Nil
    expected_bytes = bytes(count)
    size_matches = exact ? buffer.size == expected_bytes : buffer.size >= expected_bytes
    unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared &&
           size_matches
      size_requirement = exact ? "exact-size" : "at-least-size"
      raise ArgumentError.new("DCAE-V #{label} must be a #{size_requirement} shared BF16 buffer")
    end
    MetalOps.check_storage(buffer, count.to_i64, WeightDType::BF16)
  end

  def self.validate_disjoint!(buffers : Array(ML::MetalBuffer)) : Nil
    buffers.each_with_index do |buffer, index|
      buffers[0...index].each do |other|
        raise ArgumentError.new("DCAE-V MPSGraph buffers must be disjoint") if MetalOps.overlap?(buffer, other)
      end
    end
  end

  # One eager PyTorch RMSNorm3d invocation. Every operation writes a BF16
  # result to caller-owned storage: square, F32-accumulating channel mean,
  # epsilon add, sqrt, divide, affine multiply and affine bias add.
  class RMSNormPlan
    getter shape_bcthw : Array(Int32)
    getter output : ML::MetalBuffer

    @input : ML::MetalBuffer
    @full_scratch : ML::MetalBuffer
    @mean_a : ML::MetalBuffer
    @mean_b : ML::MetalBuffer
    @weight : ML::MetalBuffer
    @bias : ML::MetalBuffer
    @handle : Pointer(Void) = Pointer(Void).null
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(shape_bcthw : Array(Int32), @input : ML::MetalBuffer,
                   @full_scratch : ML::MetalBuffer, @mean_a : ML::MetalBuffer,
                   @mean_b : ML::MetalBuffer, @weight : ML::MetalBuffer,
                   @bias : ML::MetalBuffer,
                   epsilon : Float32 = 1.0e-5_f32)
      @shape_bcthw = shape_bcthw.dup
      raise ArgumentError.new("DCAE-V RMSNorm shape must have rank five") unless @shape_bcthw.size == 5
      raise ArgumentError.new("DCAE-V RMSNorm epsilon must be finite and positive") unless epsilon.finite? && epsilon > 0.0_f32
      count = DCAEVNormMPS.checked_count("RMSNorm activation", @shape_bcthw)
      row_count = DCAEVNormMPS.checked_rows("RMSNorm row", @shape_bcthw)
      channels = @shape_bcthw[1]
      DCAEVNormMPS.validate_buffer!(@input, count, "RMSNorm input", exact: false)
      DCAEVNormMPS.validate_buffer!(@full_scratch, count, "RMSNorm full scratch", exact: false)
      DCAEVNormMPS.validate_buffer!(@mean_a, row_count, "RMSNorm mean A", exact: false)
      DCAEVNormMPS.validate_buffer!(@mean_b, row_count, "RMSNorm mean B", exact: false)
      DCAEVNormMPS.validate_buffer!(@weight, channels, "RMSNorm weight")
      DCAEVNormMPS.validate_buffer!(@bias, channels, "RMSNorm bias")
      DCAEVNormMPS.validate_disjoint!([@input, @full_scratch, @mean_a, @mean_b, @weight, @bias])

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        shape64 = @shape_bcthw.map(&.to_i64)
        @handle = DCAEVNormMPSFFI.rms_plan_create(
          ML::Metal::Device.instance.device_handle, shape64.to_unsafe,
          @shape_bcthw.size.to_i32, @input.handle, @full_scratch.handle,
          @mean_a.handle, @mean_b.handle, @weight.handle, @bias.handle, epsilon,
        )
        raise "Failed to prepare DCAE-V BF16 RMSNorm MPSGraph plan" if @handle.null?
      {% else %}
        raise ArgumentError.new("DCAE-V RMSNorm MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
      @output = @full_scratch
    end

    # Append the seven upstream eager BF16 boundaries on a caller-owned chain.
    # MPSGraph's private allocations are opaque; this bridge itself creates no
    # MTLBuffer/TensorData/plan wrappers, waits, or readbacks during encode.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("DCAE-V RMSNorm plan has been released") if @released || @handle.null?
      validate_bindings!
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("DCAE-V RMSNorm plan cannot span in-flight command chains")
        end
      end
      @last_chain = chain

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = DCAEVNormMPSFFI.rms_encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "DCAE-V RMSNorm MPSGraph encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("DCAE-V RMSNorm MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the MPS command chain before releasing DCAE-V RMSNorm plan") unless previous.completed?
        if supplied = chain
          raise ArgumentError.new("DCAE-V RMSNorm release chain does not match the bound chain") unless previous.same?(supplied)
        end
      elsif chain && !chain.not_nil!.completed?
        raise ArgumentError.new("Complete the supplied MPS chain before releasing DCAE-V RMSNorm plan")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        DCAEVNormMPSFFI.rms_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @last_chain = nil
      @released = true
    end

    def finalize
      return if @released
      return if @last_chain && !@last_chain.not_nil!.completed?
      release
    end

    private def validate_bindings! : Nil
      raise ArgumentError.new("DCAE-V RMSNorm plan has been released") if @released
      count = DCAEVNormMPS.checked_count("RMSNorm activation", @shape_bcthw)
      row_count = DCAEVNormMPS.checked_rows("RMSNorm row", @shape_bcthw)
      unless @input.valid? && @full_scratch.valid? && @mean_a.valid? &&
             @mean_b.valid? && @weight.valid? && @bias.valid? &&
             @input.size >= DCAEVNormMPS.bytes(count) &&
             @full_scratch.size >= DCAEVNormMPS.bytes(count) &&
             @mean_a.size >= DCAEVNormMPS.bytes(row_count) &&
             @mean_b.size >= DCAEVNormMPS.bytes(row_count) &&
             @weight.size == DCAEVNormMPS.bytes(@shape_bcthw[1]) &&
             @bias.size == DCAEVNormMPS.bytes(@shape_bcthw[1])
        raise ArgumentError.new("DCAE-V RMSNorm fixed BF16 binding changed")
      end
      DCAEVNormMPS.validate_disjoint!([@input, @full_scratch, @mean_a, @mean_b, @weight, @bias])
    end
  end

  # One eager PyTorch F.silu invocation. Its graph preserves the pinned MPS
  # sequence: negative, exp, add one, divide; it writes to a distinct BF16
  # caller buffer and shares the same chain/lifetime rules as RMSNormPlan.
  class SiLUPlan
    getter shape_bcthw : Array(Int32)

    @input : ML::MetalBuffer
    @output : ML::MetalBuffer
    @handle : Pointer(Void) = Pointer(Void).null
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(shape_bcthw : Array(Int32), @input : ML::MetalBuffer,
                   @output : ML::MetalBuffer)
      @shape_bcthw = shape_bcthw.dup
      count = DCAEVNormMPS.checked_count("SiLU activation", @shape_bcthw)
      DCAEVNormMPS.validate_buffer!(@input, count, "SiLU input", exact: false)
      DCAEVNormMPS.validate_buffer!(@output, count, "SiLU output", exact: false)
      DCAEVNormMPS.validate_disjoint!([@input, @output])

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        shape64 = @shape_bcthw.map(&.to_i64)
        @handle = DCAEVNormMPSFFI.silu_plan_create(
          ML::Metal::Device.instance.device_handle, shape64.to_unsafe,
          @shape_bcthw.size.to_i32, @input.handle, @output.handle,
        )
        raise "Failed to prepare DCAE-V BF16 SiLU MPSGraph plan" if @handle.null?
      {% else %}
        raise ArgumentError.new("DCAE-V SiLU MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("DCAE-V SiLU plan has been released") if @released || @handle.null?
      validate_bindings!
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("DCAE-V SiLU plan cannot span in-flight command chains")
        end
      end
      @last_chain = chain

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = DCAEVNormMPSFFI.silu_encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "DCAE-V SiLU MPSGraph encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("DCAE-V SiLU MPSGraph requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the MPS command chain before releasing DCAE-V SiLU plan") unless previous.completed?
        if supplied = chain
          raise ArgumentError.new("DCAE-V SiLU release chain does not match the bound chain") unless previous.same?(supplied)
        end
      elsif chain && !chain.not_nil!.completed?
        raise ArgumentError.new("Complete the supplied MPS chain before releasing DCAE-V SiLU plan")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        DCAEVNormMPSFFI.silu_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @last_chain = nil
      @released = true
    end

    def finalize
      return if @released
      return if @last_chain && !@last_chain.not_nil!.completed?
      release
    end

    private def validate_bindings! : Nil
      raise ArgumentError.new("DCAE-V SiLU plan has been released") if @released
      count = DCAEVNormMPS.checked_count("SiLU activation", @shape_bcthw)
      unless @input.valid? && @output.valid? &&
             @input.size >= DCAEVNormMPS.bytes(count) &&
             @output.size >= DCAEVNormMPS.bytes(count)
        raise ArgumentError.new("DCAE-V SiLU fixed BF16 binding changed")
      end
      DCAEVNormMPS.validate_disjoint!([@input, @output])
    end
  end
end
