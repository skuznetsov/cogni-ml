# Canonical Hy-Nomas attention through the pinned rank-4 MPSGraph SDPA path.
# The block inputs and output stay caller-owned; fixed contiguous head-major
# staging buffers live in this plan's Arena until after the terminal wait.
require "./hy_nomas_block"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasAttentionMPSFFI
    fun plan_create = gs_hy_nomas_mps_reduction_plan_create(
      device : Pointer(Void), mode : Int32, batches : Int32, query_rows : Int32,
      key_rows : Int32, heads : Int32, head_dim : Int32, scale : Float32,
      input_a : Pointer(Void), input_b : Pointer(Void), input_c : Pointer(Void),
      packed_a : Pointer(Void), packed_b : Pointer(Void), packed_c : Pointer(Void),
      packed_output : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_mps_reduction_encode(plan : Pointer(Void),
                                                  command : Pointer(Void)) : Int32
    fun plan_release = gs_hy_nomas_mps_reduction_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video
  # Prepared, fixed-buffer canonical SDPA. It owns four setup-time F32 staging
  # buffers and the retained MPSGraph plan, but it does not own Q/K/V/output.
  # Keep this plan and its caller-owned bindings alive and immutable through the
  # command chain's terminal wait. Reuse is serialized across completed chains.
  class HyNomasAttentionMPSPlan
    Q_STAGE_KEY      = "hy_nomas_attention_mps.q_head_major"
    K_STAGE_KEY      = "hy_nomas_attention_mps.k_head_major"
    V_STAGE_KEY      = "hy_nomas_attention_mps.v_head_major"
    OUTPUT_STAGE_KEY = "hy_nomas_attention_mps.output_head_major"

    getter batches : Int32
    getter tokens : Int32
    getter heads : Int32
    getter head_dim : Int32

    @q : ML::MetalBuffer
    @k : ML::MetalBuffer
    @v : ML::MetalBuffer
    @output : ML::MetalBuffer
    @arena : Arena
    @q_stage : ML::MetalBuffer? = nil
    @k_stage : ML::MetalBuffer? = nil
    @v_stage : ML::MetalBuffer? = nil
    @output_stage : ML::MetalBuffer? = nil
    @handle = Pointer(Void).null
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def resident_bytes : Int64
      @arena.resident_bytes
    end

    def initialize(@batches : Int32, @tokens : Int32, @heads : Int32,
                   @head_dim : Int32, @q : ML::MetalBuffer,
                   @k : ML::MetalBuffer, @v : ML::MetalBuffer,
                   @output : ML::MetalBuffer)
      count = HyNomasDualStreamBlock.checked_count(@batches, @tokens,
        @heads, @head_dim)
      # Reject unsupported Graph score geometry before allocating staging.
      HyNomasDualStreamBlock.checked_count(@batches, @heads, @tokens, @tokens)
      caller_buffers = {@q, @k, @v, @output}
      caller_buffers.each do |buffer|
        MetalOps.check(buffer, count.to_i64)
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("Hy-Nomas MPS attention requires shared F32 buffers")
        end
      end
      validate_distinct!(caller_buffers)

      @arena = Arena.new
      begin
        @q_stage = @arena.alloc(Q_STAGE_KEY, count)
        @k_stage = @arena.alloc(K_STAGE_KEY, count)
        @v_stage = @arena.alloc(V_STAGE_KEY, count)
        @output_stage = @arena.alloc(OUTPUT_STAGE_KEY, count)
        validate_stage_bindings!(count)

        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          scale = 1.0_f32 / Math.sqrt(@head_dim.to_f32)
          @handle = HyNomasAttentionMPSFFI.plan_create(
            ML::Metal::Device.instance.device_handle, 3, @batches, @tokens,
            @tokens, @heads, @head_dim, scale, @q.handle, @k.handle,
            @v.handle, q_stage.handle, k_stage.handle, v_stage.handle,
            output_stage.handle, @output.handle,
          )
          raise "Failed to prepare canonical Hy-Nomas MPS attention plan" if @handle.null?
        {% else %}
          raise ArgumentError.new("Hy-Nomas MPS attention requires macOS Metal Performance Shaders")
        {% end %}
      rescue ex
        release_native_plan
        @arena.release
        @released = true
        raise ex
      end
    end

    # Reject rebinding to a different caller buffer before any command is
    # appended. The plan's graph and staging buffers remain fixed for life.
    def validate_bindings!(q : ML::MetalBuffer, k : ML::MetalBuffer,
                           v : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      raise "Hy-Nomas MPS attention plan has been released" if @released
      unless @q.same?(q) && @k.same?(k) && @v.same?(v) && @output.same?(output)
        raise ArgumentError.new("Hy-Nomas MPS attention binding changed")
      end
      count = HyNomasDualStreamBlock.checked_count(@batches, @tokens,
        @heads, @head_dim)
      {@q, @k, @v, @output}.each do |buffer|
        MetalOps.check(buffer, count.to_i64)
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("Hy-Nomas MPS attention requires shared F32 buffers")
        end
      end
      validate_distinct!({@q, @k, @v, @output})
      validate_stage_bindings!(count)
    end

    # Transposes fixed BQHD bindings into contiguous BHQD staging buffers,
    # encodes the rank-4 canonical Graph, then restores BQHD output. No buffers
    # are allocated and no command is waited on in this method.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise "Hy-Nomas MPS attention plan has been released" if @released || @handle.null?
      validate_bindings!(@q, @k, @v, @output)
      chain.encoder
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas MPS attention plan cannot span in-flight command chains")
        end
      end
      @chain = chain

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = HyNomasAttentionMPSFFI.encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "Hy-Nomas MPS attention encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Hy-Nomas MPS attention requires macOS Metal Performance Shaders")
      {% end %}
    end

    # Release the graph first, then its owned staging buffers. The caller must
    # perform the terminal wait before releasing a plan that encoded work.
    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("Complete Hy-Nomas MPS attention chain before release") unless chain.completed?
      end
      release_native_plan
      @arena.release
      @q_stage = nil
      @k_stage = nil
      @v_stage = nil
      @output_stage = nil
      @released = true
    end

    # Preserve the graph and staging buffers if a plan becomes unreachable
    # while its command chain is still in flight.
    def finalize
      return if @released
      return if @chain && !@chain.not_nil!.completed?
      release
    end

    private def validate_stage_bindings!(count : Int32) : Nil
      stages = {q_stage, k_stage, v_stage, output_stage}
      stages.each do |buffer|
        MetalOps.check(buffer, count.to_i64)
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("Hy-Nomas MPS staging buffers must be shared F32")
        end
      end
      all_buffers = [@q, @k, @v, @output]
      stages.each { |buffer| all_buffers << buffer }
      validate_distinct!(all_buffers)
      unless @arena.allocated?(Q_STAGE_KEY) && @arena[Q_STAGE_KEY].same?(q_stage) &&
             @arena.allocated?(K_STAGE_KEY) && @arena[K_STAGE_KEY].same?(k_stage) &&
             @arena.allocated?(V_STAGE_KEY) && @arena[V_STAGE_KEY].same?(v_stage) &&
             @arena.allocated?(OUTPUT_STAGE_KEY) && @arena[OUTPUT_STAGE_KEY].same?(output_stage)
        raise ArgumentError.new("Hy-Nomas MPS attention staging Arena binding changed")
      end
      unless @arena.dtype(Q_STAGE_KEY).f32? && @arena.dtype(K_STAGE_KEY).f32? &&
             @arena.dtype(V_STAGE_KEY).f32? && @arena.dtype(OUTPUT_STAGE_KEY).f32?
        raise ArgumentError.new("Hy-Nomas MPS attention staging dtype changed")
      end
    end

    private def validate_distinct!(buffers : Tuple(ML::MetalBuffer, ML::MetalBuffer,
                                                   ML::MetalBuffer, ML::MetalBuffer)) : Nil
      buffers.each_with_index do |left, i|
        buffers.each_with_index do |right, j|
          next if j <= i
          raise ArgumentError.new("Hy-Nomas MPS attention buffers must not alias") if MetalOps.overlap?(left, right)
        end
      end
    end

    private def validate_distinct!(buffers : Array(ML::MetalBuffer)) : Nil
      buffers.each_with_index do |left, i|
        buffers.each_with_index do |right, j|
          next if j <= i
          raise ArgumentError.new("Hy-Nomas MPS attention buffers must not alias") if MetalOps.overlap?(left, right)
        end
      end
    end

    private def q_stage : ML::MetalBuffer
      @q_stage || raise ArgumentError.new("Hy-Nomas MPS Q staging buffer is unavailable")
    end

    private def k_stage : ML::MetalBuffer
      @k_stage || raise ArgumentError.new("Hy-Nomas MPS K staging buffer is unavailable")
    end

    private def v_stage : ML::MetalBuffer
      @v_stage || raise ArgumentError.new("Hy-Nomas MPS V staging buffer is unavailable")
    end

    private def output_stage : ML::MetalBuffer
      @output_stage || raise ArgumentError.new("Hy-Nomas MPS output staging buffer is unavailable")
    end

    private def release_native_plan : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasAttentionMPSFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
    end
  end
end
