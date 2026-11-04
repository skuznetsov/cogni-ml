# Default-off compatibility probe for the PyTorch 2.8.0 MPS one-pass vector
# attention path. Fixed F32 BQHD bindings represent BF16 source tensors.
require "./hy_nomas_block"
require "./metal_mps_command_chain"

module ML::Video
  class HyNomasVectorAttentionPlan
    KERNEL = "hy_nomas_vector_attention_bf16"
    SOURCE = {{ read_file("#{__DIR__}/kernels/hy_nomas_vector_attention.metal") }}

    @pipeline : ML::Metal::ComputePipeline
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    # The source wrapper chooses its one-pass vector path only below 1024
    # keys, with at most eight queries. This adaptation admits D=V=128 and
    # equal Q/K/V head counts, no mask, no causality and no dropout.
    def self.validate_geometry!(batches : Int32, input_query_rows : Int32,
                                query_start : Int32, query_rows : Int32,
                                key_rows : Int32, heads : Int32,
                                head_dim : Int32) : Nil
      MetalOps.dims(batches, input_query_rows, query_rows, key_rows, heads, head_dim)
      unless 1 <= query_rows <= 8 && query_rows <= key_rows < 1024 &&
             head_dim == 128 && query_start >= 0 &&
             query_start.to_i64 + query_rows <= input_query_rows
        raise ArgumentError.new("Unsupported Hy-Nomas one-pass vector attention geometry")
      end
      HyNomasDualStreamBlock.checked_count(batches, input_query_rows, heads, head_dim)
      HyNomasDualStreamBlock.checked_count(batches, key_rows, heads, head_dim)
      HyNomasDualStreamBlock.checked_count(batches, query_rows, heads, head_dim)
      unless batches.to_i64 * heads * 1024 <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas vector dispatch exceeds the Metal grid limit")
      end
    end

    def initialize(@batches : Int32, @input_query_rows : Int32,
                   @query_start : Int32, @query_rows : Int32,
                   @key_rows : Int32, @heads : Int32, @head_dim : Int32,
                   @q : ML::MetalBuffer, @k : ML::MetalBuffer,
                   @v : ML::MetalBuffer, @output : ML::MetalBuffer)
      self.class.validate_geometry!(@batches, @input_query_rows, @query_start,
        @query_rows, @key_rows, @heads, @head_dim)
      validate_buffers!
      @pipeline = ML::Metal::PipelineCache.get(KERNEL) do
        ML::Metal::ComputePipeline.new(KERNEL, SOURCE)
      end
      unless @pipeline.thread_execution_width == 32 &&
             @pipeline.max_total_threads_per_threadgroup >= 1024
        raise ArgumentError.new("Hy-Nomas vector attention requires 32 SIMD lanes and 1024 threads")
      end
    end

    # No owned GPU stages: read caller Q/K/V directly, preserving the source
    # scalar/SIMD reduction order, then write only the compact query interval.
    def resident_bytes : Int64
      0_i64
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas vector attention plan has been released") if @released
      validate_buffers!
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas vector attention cannot span in-flight command chains")
        end
      end
      @chain = chain
      scale = (1.0_f64 / Math.sqrt(@head_dim.to_f64)).to_f32
      chain.encoder.set_pipeline(@pipeline)
        .set_buffer(@q, 0)
        .set_buffer(@k, 1)
        .set_buffer(@v, 2)
        .set_buffer(@output, 3, ML::Metal::BufferAccess::Write)
        .set_value(@input_query_rows.to_u32, 4)
        .set_value(@query_start.to_u32, 5)
        .set_value(@query_rows.to_u32, 6)
        .set_value(@key_rows.to_u32, 7)
        .set_value(@heads.to_u32, 8)
        .set_value(scale, 9)
        .dispatch({@batches * @heads * 1024, @query_rows, 1}, {1024, 1, 1})
    end

    # The plan retains its caller bindings and pipeline through completion.
    # It neither submits nor waits for GPU work, including during release.
    def release : Nil
      return if @released
      if chain = @chain
        unless chain.completed?
          raise ArgumentError.new("Complete Hy-Nomas vector attention chain before release")
        end
      end
      @released = true
    end

    private def validate_buffers! : Nil
      q_count = @batches.to_i64 * @input_query_rows * @heads * @head_dim
      kv_count = @batches.to_i64 * @key_rows * @heads * @head_dim
      output_count = @batches.to_i64 * @query_rows * @heads * @head_dim
      validate_f32!(@q, q_count)
      validate_f32!(@k, kv_count)
      validate_f32!(@v, kv_count)
      validate_f32!(@output, output_count)
      MetalOps.distinct(@output, @q, @k, @v)
    end

    private def validate_f32!(buffer : ML::MetalBuffer, count : Int64) : Nil
      MetalOps.check(buffer, count)
      unless buffer.storage_mode.shared? && buffer.size == count * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas vector attention requires exact-size shared F32 bindings")
      end
    end
  end
end
