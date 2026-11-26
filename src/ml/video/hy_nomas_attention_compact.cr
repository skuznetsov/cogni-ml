# Opt-in query-chunked/head-grouped canonical F32 MPS attention. K/V are
# packed once per encode; fixed smaller Q/output stages are reused serially.
require "./hy_nomas_block"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasCompactAttentionMPSFFI
    fun plan_create = gs_hy_nomas_mps_attention_compact_plan_create(
      device : Pointer(Void), batches : Int32, tokens : Int32, heads : Int32,
      head_dim : Int32, query_chunk : Int32, heads_per_group : Int32, scale : Float32,
      q : Pointer(Void), k : Pointer(Void), v : Pointer(Void),
      q_stage : Pointer(Void), k_stage : Pointer(Void), v_stage : Pointer(Void),
      output_stage : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_mps_attention_compact_encode(
      plan : Pointer(Void), command : Pointer(Void),
    ) : Int32
    fun plan_release = gs_hy_nomas_mps_attention_compact_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video
  class HyNomasCompactAttentionMPSPlan
    record Geometry, query_elements : Int32, kv_elements : Int32,
      max_score_elements : Int64, graph_calls : Int32, resident_bytes : Int64

    Q_STAGE_KEY      = "hy_nomas_compact_attention.q"
    K_STAGE_KEY      = "hy_nomas_compact_attention.k"
    V_STAGE_KEY      = "hy_nomas_compact_attention.v"
    OUTPUT_STAGE_KEY = "hy_nomas_compact_attention.output"

    # Explicit release is the ownership boundary. Keep buffers/plans reachable
    # even if a caller drops its last reference before the terminal GPU wait.
    # Abandoned plans deliberately remain retained until process exit; callers
    # must release after draining the latest chain, including error cleanup.
    @@lifetime_mutex = Mutex.new
    @@live_plans = [] of HyNomasCompactAttentionMPSPlan

    getter batches : Int32
    getter tokens : Int32
    getter heads : Int32
    getter head_dim : Int32
    getter query_chunk : Int32
    getter heads_per_group : Int32
    getter shape : Geometry

    @q : ML::MetalBuffer
    @k : ML::MetalBuffer
    @v : ML::MetalBuffer
    @output : ML::MetalBuffer
    @arena : Arena
    @handle = Pointer(Void).null
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    # This is a named-buffer/score-shape bound, not a bound on opaque Graph
    # temporaries or queued executions. Admission still needs the runtime guard.
    def self.geometry(batches : Int32, tokens : Int32, heads : Int32,
                      head_dim : Int32, query_chunk : Int32,
                      heads_per_group : Int32) : Geometry
      unless batches > 0 && tokens > 0 && heads > 0 && head_dim > 0 &&
             query_chunk > 0 && query_chunk <= tokens &&
             heads_per_group > 0 && heads_per_group <= heads
        raise ArgumentError.new("Invalid compact Hy-Nomas attention geometry")
      end
      query_elements = HyNomasDualStreamBlock.checked_count(batches, heads, query_chunk, head_dim)
      kv_elements = HyNomasDualStreamBlock.checked_count(batches, heads, tokens, head_dim)
      score_elements = HyNomasDualStreamBlock.checked_count(heads_per_group, query_chunk, tokens)
      query_chunks = ((tokens.to_i64 + query_chunk - 1) // query_chunk).to_i32
      head_groups = ((heads.to_i64 + heads_per_group - 1) // heads_per_group).to_i32
      graph_calls = HyNomasDualStreamBlock.checked_count(batches, query_chunks, head_groups)
      Geometry.new(query_elements, kv_elements, score_elements.to_i64,
        graph_calls, 2_i64 * (query_elements.to_i64 + kv_elements) * sizeof(Float32))
    end

    def self.retained_plan_count : Int32
      @@lifetime_mutex.synchronize { @@live_plans.size.to_i32 }
    end

    # These guards reject known MPS reduction counterexamples. They do not
    # qualify every remaining shape or input for BF16 equivalence. Workspace
    # query chunking preserves all heads and admits only one batch; the general
    # standalone plan remains available for explicit diagnostics.
    def self.workspace_geometry(batches : Int32, tokens : Int32, heads : Int32,
                                head_dim : Int32, query_chunk : Int32,
                                heads_per_group : Int32) : Geometry
      shape = geometry(batches, tokens, heads, head_dim, query_chunk, heads_per_group)
      unless heads_per_group == heads
        raise ArgumentError.new("Grouped compact Hy-Nomas attention is diagnostic only; preserve all heads")
      end
      if query_chunk < tokens && batches != 1
        raise ArgumentError.new("Multi-batch compact Hy-Nomas chunks are diagnostic only; preserve the full sequence or use one batch")
      end
      if query_chunk < tokens && query_chunk < 32
        raise ArgumentError.new("Short compact Hy-Nomas chunks are diagnostic only; use at least 32 rows or the full sequence")
      end
      shape
    end

    def initialize(@batches : Int32, @tokens : Int32, @heads : Int32,
                   @head_dim : Int32, @q : ML::MetalBuffer,
                   @k : ML::MetalBuffer, @v : ML::MetalBuffer,
                   @output : ML::MetalBuffer, @query_chunk : Int32,
                   @heads_per_group : Int32)
      @shape = self.class.geometry(@batches, @tokens, @heads, @head_dim,
        @query_chunk, @heads_per_group)
      validate_caller_bindings!
      @arena = Arena.new
      begin
        @arena.alloc(Q_STAGE_KEY, @shape.query_elements)
        @arena.alloc(K_STAGE_KEY, @shape.kv_elements)
        @arena.alloc(V_STAGE_KEY, @shape.kv_elements)
        @arena.alloc(OUTPUT_STAGE_KEY, @shape.query_elements)
        validate_stage_bindings!
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          @handle = HyNomasCompactAttentionMPSFFI.plan_create(
            ML::Metal::Device.instance.device_handle, @batches, @tokens,
            @heads, @head_dim, @query_chunk, @heads_per_group,
            1.0_f32 / Math.sqrt(@head_dim.to_f32), @q.handle, @k.handle,
            @v.handle, @arena[Q_STAGE_KEY].handle, @arena[K_STAGE_KEY].handle,
            @arena[V_STAGE_KEY].handle, @arena[OUTPUT_STAGE_KEY].handle, @output.handle)
          raise "Failed to prepare compact Hy-Nomas MPS attention plan" if @handle.null?
        {% else %}
          raise ArgumentError.new("Compact Hy-Nomas MPS attention requires macOS Metal Performance Shaders")
        {% end %}
        @@lifetime_mutex.synchronize { @@live_plans << self }
      rescue ex
        release_native_plan
        @arena.release
        @released = true
        raise ex
      end
    end

    def resident_bytes : Int64
      @arena.resident_bytes
    end

    def validate_bindings!(q : ML::MetalBuffer, k : ML::MetalBuffer,
                           v : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Compact Hy-Nomas MPS attention plan has been released") if @released
      unless @q.same?(q) && @k.same?(k) && @v.same?(v) && @output.same?(output)
        raise ArgumentError.new("Compact Hy-Nomas MPS attention binding changed")
      end
      validate_caller_bindings!
      validate_stage_bindings!
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Compact Hy-Nomas MPS attention plan has been released") if @released || @handle.null?
      validate_bindings!(@q, @k, @v, @output)
      chain.encoder
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Compact Hy-Nomas MPS attention cannot span in-flight command chains")
        end
      end
      @chain = chain
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = HyNomasCompactAttentionMPSFFI.encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "Compact Hy-Nomas MPS attention encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Compact Hy-Nomas MPS attention requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release : Nil
      return if @released
      if chain = @chain
        unless chain.completed?
          raise ArgumentError.new("Complete compact Hy-Nomas MPS attention chain before release")
        end
      end
      release_native_plan
      @arena.release
      @released = true
      @@lifetime_mutex.synchronize { @@live_plans.reject! { |current| current.same?(self) } }
    end

    private def validate_caller_bindings! : Nil
      caller_buffers = [@q, @k, @v, @output]
      caller_buffers.each do |buffer|
        MetalOps.check(buffer, @shape.kv_elements.to_i64)
        unless buffer.storage_mode.shared?
          raise ArgumentError.new("Compact Hy-Nomas MPS attention requires shared F32 buffers")
        end
      end
      validate_distinct!(caller_buffers)
    end

    private def validate_stage_bindings! : Nil
      buffers = [@q, @k, @v, @output]
      {Q_STAGE_KEY, K_STAGE_KEY, V_STAGE_KEY, OUTPUT_STAGE_KEY}.each do |key|
        stage = @arena[key]
        count = {Q_STAGE_KEY, OUTPUT_STAGE_KEY}.includes?(key) ? @shape.query_elements : @shape.kv_elements
        MetalOps.check(stage, count.to_i64)
        unless stage.storage_mode.shared? && @arena.dtype(key).f32?
          raise ArgumentError.new("Compact Hy-Nomas MPS staging requires shared F32 buffers")
        end
        buffers << stage
      end
      validate_distinct!(buffers)
    end

    private def validate_distinct!(buffers : Array(ML::MetalBuffer)) : Nil
      buffers.each_with_index do |left, i|
        buffers.each_with_index do |right, j|
          next if j <= i
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Compact Hy-Nomas MPS attention buffers must not alias")
          end
        end
      end
    end

    private def release_native_plan : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasCompactAttentionMPSFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
    end
  end
end
