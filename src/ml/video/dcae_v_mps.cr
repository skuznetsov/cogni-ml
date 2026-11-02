require "../metal/dispatch"
require "../core/buffer"
require "./metal_ops"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib DCAEVConvMPSFFI
    fun template_create = gs_dcae_v_mps_conv_template_create(
      device : Pointer(Void), batches : Int32, in_channels : Int32,
      conv_out_channels : Int32, output_channels : Int32,
      frames : Int32, source_frames : Int32, height : Int32, width : Int32,
      input_frame_offset : Int32, full_clip : Int32, chunk_causal : Int32,
      cache_input : Int32, use_bias : Int32, spatial_factor : Int32,
      temporal_factor : Int32, shortcut : Int32,
    ) : Pointer(Void)
    fun plan_create = gs_dcae_v_mps_conv_plan_create(
      template : Pointer(Void), input : Pointer(Void), cache : Pointer(Void),
      weight : Pointer(Void), bias : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_dcae_v_mps_conv_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_dcae_v_mps_conv_plan_release(plan : Pointer(Void)) : Void
    fun template_release = gs_dcae_v_mps_conv_template_release(template : Pointer(Void)) : Void
  end
{% end %}

module ML::Video::DCAEVConvMPS
  # Reusable graph geometry. Distinct operation plans bind the same compiled
  # graph to different caller-owned buffers and weights.
  class Template
    @handle = Pointer(Void).null
    @released = false

    def initialize(batches : Int32, in_channels : Int32,
                   conv_out_channels : Int32, output_channels : Int32,
                   frames : Int32, source_frames : Int32,
                   height : Int32, width : Int32, input_frame_offset : Int32,
                   full_clip : Bool, chunk_causal : Bool, cache_input : Bool,
                   use_bias : Bool, spatial_factor : Int32,
                   temporal_factor : Int32, shortcut : Bool)
      unless {batches, in_channels, conv_out_channels, output_channels,
              frames, source_frames, height, width, spatial_factor,
              temporal_factor}.all?(&.positive?) &&
             0 <= input_frame_offset <= source_frames - frames
        raise ArgumentError.new("Invalid DCAE-V MPSGraph convolution shape")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @handle = DCAEVConvMPSFFI.template_create(
          ML::Metal::Device.instance.device_handle,
          batches, in_channels, conv_out_channels, output_channels,
          frames, source_frames, height, width, input_frame_offset,
          full_clip ? 1 : 0, chunk_causal ? 1 : 0,
          cache_input ? 1 : 0, use_bias ? 1 : 0,
          spatial_factor, temporal_factor, shortcut ? 1 : 0,
        )
        if @handle.null?
          raise "Failed to prepare DCAE-V MPSGraph Conv3d template (B=#{batches}, Cin=#{in_channels}, Cconv=#{conv_out_channels}, Cout=#{output_channels}, T=#{frames}/#{source_frames}, HW=#{height}x#{width}, offset=#{input_frame_offset}, full=#{full_clip}, causal=#{chunk_causal}, cache=#{cache_input}, bias=#{use_bias}, unshuffle=#{temporal_factor}x#{spatial_factor}, shortcut=#{shortcut})"
        end
      {% else %}
        raise ArgumentError.new("DCAE-V MPSGraph convolution requires macOS Metal Performance Shaders")
      {% end %}
    end

    def bind(input : ML::MetalBuffer, cache : ML::MetalBuffer?,
             weight : ML::MetalBuffer, bias : ML::MetalBuffer?,
             output : ML::MetalBuffer) : Plan
      raise "DCAE-V MPSGraph template has been released" if @released || @handle.null?
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        handle = DCAEVConvMPSFFI.plan_create(@handle, input.handle,
          cache.try(&.handle) || Pointer(Void).null, weight.handle,
          bias.try(&.handle) || Pointer(Void).null, output.handle)
        raise "Failed to bind DCAE-V MPSGraph Conv3d buffers" if handle.null?
        buffers = [input, weight, output] of ML::MetalBuffer
        buffers << cache if cache
        buffers << bias if bias
        Plan.new(handle, buffers)
      {% else %}
        raise ArgumentError.new("DCAE-V MPSGraph convolution requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        DCAEVConvMPSFFI.template_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    def finalize
      release
    end
  end

  # Fixed TensorData binding for one operation. Keep the buffers and template
  # alive until the command chain that used this plan has drained.
  class Plan
    @handle : Pointer(Void)
    @buffers : Array(ML::MetalBuffer)
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@handle : Pointer(Void), buffers : Array(ML::MetalBuffer))
      @buffers = buffers.uniq { |buffer| buffer.object_id }
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("DCAE-V MPSGraph plan has been released") if @released || @handle.null?
      raise ArgumentError.new("DCAE-V MPSGraph chain encoder is unavailable") unless chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("DCAE-V MPSGraph plan cannot span in-flight command chains")
        end
      end
      @last_chain = chain
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = DCAEVConvMPSFFI.encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "DCAE-V MPSGraph Conv3d encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("DCAE-V MPSGraph convolution requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if previous = @last_chain
        raise ArgumentError.new("Drain the DCAE-V MPSGraph chain before releasing its plan") unless previous.completed?
        if supplied = chain
          raise ArgumentError.new("DCAE-V plan release chain identity mismatch") unless previous.same?(supplied)
        end
      elsif chain && !chain.not_nil!.completed?
        raise ArgumentError.new("Complete the supplied DCAE-V chain before releasing its plan")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        DCAEVConvMPSFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @last_chain = nil
      @buffers.clear
      @released = true
    end

    def finalize
      return if @released
      return if @last_chain && !@last_chain.not_nil!.completed?
      release
    end
  end
end
