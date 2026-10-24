# Opt-in per-element F32 arithmetic controls for Hy-Nomas modulation and
# gated residuals. The kernels preserve separate multiply/add rounding to
# match the measured upstream pointwise operation boundaries.
require "./metal_ops"

module ML::Video::HyNomasMPSPointwise
  MODULATE_KERNEL = "hy_nomas_modulate_uncontracted_f32"
  RESIDUAL_KERNEL = "hy_nomas_residual_uncontracted_f32"
  ROPE_KERNEL     = "hy_nomas_rope_uncontracted_f32"
  SOURCE          = {{ read_file("#{__DIR__}/kernels/hy_nomas_mps_pointwise.metal") }}

  # Setup-prepared fixed feature width. Rows remain dynamic within the block's
  # configured image/text geometry; encode only validates and dispatches.
  class Plan
    getter dim : Int32

    @modulate_pipeline : ML::Metal::ComputePipeline
    @residual_pipeline : ML::Metal::ComputePipeline
    @rope_pipeline : ML::Metal::ComputePipeline

    def initialize(@dim : Int32)
      unless @dim > 0
        raise ArgumentError.new("Hy-Nomas MPS pointwise feature width must be positive")
      end
      @modulate_pipeline = ML::Metal::PipelineCache.get(MODULATE_KERNEL) do
        ML::Metal::ComputePipeline.new(MODULATE_KERNEL, SOURCE)
      end
      @residual_pipeline = ML::Metal::PipelineCache.get(RESIDUAL_KERNEL) do
        ML::Metal::ComputePipeline.new(RESIDUAL_KERNEL, SOURCE)
      end
      @rope_pipeline = ML::Metal::PipelineCache.get(ROPE_KERNEL) do
        ML::Metal::ComputePipeline.new(ROPE_KERNEL, SOURCE)
      end
    end

    def modulate(e : ML::Metal::ComputeEncoder,
                 input : ML::MetalBuffer, shift : ML::MetalBuffer,
                 scale : ML::MetalBuffer, output : ML::MetalBuffer,
                 rows : Int32, dim : Int32) : Nil
      count = checked_count(rows, dim)
      validate_tensor!(input, count)
      validate_tensor!(shift, count)
      validate_tensor!(scale, count)
      validate_tensor!(output, count)
      MetalOps.inplace_or_distinct(input, output)
      MetalOps.distinct(output, shift, scale)

      e.set_pipeline(@modulate_pipeline)
        .set_buffer(input, 0)
        .set_buffer(shift, 1)
        .set_buffer(scale, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 4)
        .dispatch_1d(count)
    end

    def gated_residual(e : ML::Metal::ComputeEncoder,
                       input : ML::MetalBuffer, delta : ML::MetalBuffer,
                       gate : ML::MetalBuffer, output : ML::MetalBuffer,
                       rows : Int32, dim : Int32) : Nil
      count = checked_count(rows, dim)
      validate_tensor!(input, count)
      validate_tensor!(delta, count)
      validate_tensor!(gate, count)
      validate_tensor!(output, count)
      MetalOps.inplace_or_distinct(input, output)
      MetalOps.inplace_or_distinct(delta, output)
      MetalOps.distinct(output, gate)

      e.set_pipeline(@residual_pipeline)
        .set_buffer(input, 0)
        .set_buffer(delta, 1)
        .set_buffer(gate, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 4)
        .dispatch_1d(count)
    end

    # Encode the pinned tuple-frequency RoPE equation over contiguous
    # [B,T,H,Dh] data. Products and their sum remain separate F32 operations.
    # The frequency table is [T,Dh] and broadcasts over batch and heads.
    def rope(e : ML::Metal::ComputeEncoder,
             input : ML::MetalBuffer, cosine : ML::MetalBuffer,
             sine : ML::MetalBuffer, output : ML::MetalBuffer,
             batches : Int32, tokens : Int32, heads : Int32,
             head_dim : Int32) : Nil
      count, table_count = checked_rope_counts(batches, tokens, heads, head_dim)
      validate_tensor!(input, count)
      validate_tensor!(output, count)
      validate_tensor!(cosine, table_count)
      validate_tensor!(sine, table_count)
      validate_pairwise_distinct!(input, cosine, sine, output)

      e.set_pipeline(@rope_pipeline)
        .set_buffer(input, 0)
        .set_buffer(cosine, 1)
        .set_buffer(sine, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        .set_value(tokens.to_u32, 4)
        .set_value(heads.to_u32, 5)
        .set_value(head_dim.to_u32, 6)
        .set_value(batches.to_u32, 7)
        .dispatch_1d(count)
    end

    private def checked_count(rows : Int32, dim : Int32) : Int32
      MetalOps.dims(rows, dim)
      unless dim == @dim
        raise ArgumentError.new("Hy-Nomas MPS pointwise feature width changed")
      end
      count = rows.to_i64 * dim
      unless count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas MPS pointwise tensor exceeds the Metal count limit")
      end
      count.to_i32
    end

    private def checked_rope_counts(batches : Int32, tokens : Int32,
                                    heads : Int32, head_dim : Int32) : Tuple(Int32, Int32)
      MetalOps.dims(batches, tokens, heads, head_dim)
      unless head_dim % 2 == 0 && heads.to_i64 * head_dim == @dim
        raise ArgumentError.new("Hy-Nomas MPS RoPE geometry does not match the prepared feature width")
      end
      rows = batches.to_i64 * tokens
      unless rows <= Int32::MAX.to_i64 // @dim
        raise ArgumentError.new("Hy-Nomas MPS RoPE tensor exceeds the Metal count limit")
      end
      count = rows * @dim
      table_count = tokens.to_i64 * head_dim
      unless count <= Int32::MAX && table_count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas MPS RoPE tensor exceeds the Metal count limit")
      end
      {count.to_i32, table_count.to_i32}
    end

    private def validate_pairwise_distinct!(input : ML::MetalBuffer,
                                            cosine : ML::MetalBuffer,
                                            sine : ML::MetalBuffer,
                                            output : ML::MetalBuffer) : Nil
      buffers = [input, cosine, sine, output]
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Hy-Nomas MPS RoPE buffers must not overlap")
          end
        end
      end
    end

    private def validate_tensor!(buffer : ML::MetalBuffer, count : Int32) : Nil
      MetalOps.check(buffer, count.to_i64)
      expected_bytes = count.to_i64 * sizeof(Float32)
      unless buffer.size == expected_bytes &&
             buffer.storage_mode == ML::StorageMode::Shared
        raise ArgumentError.new("Hy-Nomas MPS pointwise ops require exact-size shared F32 buffers")
      end
    end
  end
end
