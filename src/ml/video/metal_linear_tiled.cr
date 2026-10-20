require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::MetalLinearTiled
  SOURCE = {{ read_file("#{__DIR__}/kernels/linear_tiled.metal") }}

  ROW_TILE =  8
  THREADS  = 32

  # The scalar MetalOps.linear path stays the default. This opt-in path adapts
  # the repository's DR3 F32 simdgroup GEMM, with runtime F32/BF16 storage
  # selectors and F32 operands/accumulators/output. Call prepare! during setup
  # before encoding; retain every input, weight, bias, and output buffer through
  # the caller's terminal command completion.
  def self.prepare! : Nil
    pipeline
  end

  def self.pipeline : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("video_linear_tiled_f32_accum") do
      ML::Metal::ComputePipeline.new("video_linear_tiled_f32_accum", SOURCE)
    end
  end

  def self.linear(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                  w : ML::MetalBuffer, bias : ML::MetalBuffer, y : ML::MetalBuffer,
                  rows : Int32, in_dim : Int32, out_dim : Int32,
                  input_dtype : WeightDType = WeightDType::F32,
                  weight_dtype : WeightDType = WeightDType::F32,
                  bias_dtype : WeightDType = WeightDType::F32) : Nil
    linear_range(e, x, w, bias, y, rows, in_dim, out_dim, 0, rows,
      input_dtype, weight_dtype, bias_dtype)
  end

  # Bounded dispatches for long projections. Each non-final row range ends on
  # an 8-row tile and is continued asynchronously on the same command chain;
  # only the caller performs the one terminal wait. Retain all buffers through
  # that wait, then release the chain.
  def self.linear_bounded(chain : ML::Metal::MPSCommandChain,
                          x : ML::MetalBuffer, w : ML::MetalBuffer,
                          bias : ML::MetalBuffer, y : ML::MetalBuffer,
                          rows : Int32, in_dim : Int32, out_dim : Int32,
                          input_dtype : WeightDType = WeightDType::F32,
                          weight_dtype : WeightDType = WeightDType::F32,
                          bias_dtype : WeightDType = WeightDType::F32,
                          rows_per_segment : Int32 = 64) : Nil
    MetalOps.dims(rows, in_dim, out_dim)
    unless rows_per_segment >= ROW_TILE && rows_per_segment % ROW_TILE == 0
      raise ArgumentError.new("Tiled video linear segment rows must be a positive multiple of 8")
    end

    first_row = 0
    while first_row < rows
      count = Math.min(rows_per_segment, rows - first_row)
      linear_range(chain.encoder, x, w, bias, y, rows, in_dim, out_dim,
        first_row, count, input_dtype, weight_dtype, bias_dtype)
      first_row += count
      unless first_row == rows
        chain.pause!
        chain.continue_async!
        chain.resume!
      end
    end
  end

  # Encodes a contiguous, 8-row-aligned range while preserving absolute input
  # and output row indices. Non-final ranges must contain a whole number of
  # row tiles; only the final range may end on a partial tile. This lets callers
  # bound one GPU dispatch and chain ranges on the same serial encoder without
  # changing any row's K traversal or adding a wait.
  def self.linear_range(e : ML::Metal::ComputeEncoder,
                        x : ML::MetalBuffer, w : ML::MetalBuffer,
                        bias : ML::MetalBuffer, y : ML::MetalBuffer,
                        rows : Int32, in_dim : Int32, out_dim : Int32,
                        first_row : Int32, row_count : Int32,
                        input_dtype : WeightDType = WeightDType::F32,
                        weight_dtype : WeightDType = WeightDType::F32,
                        bias_dtype : WeightDType = WeightDType::F32) : Nil
    MetalOps.dims(rows, in_dim, out_dim)
    range_end = first_row.to_i64 + row_count
    unless first_row >= 0 && row_count > 0 && range_end <= rows
      raise ArgumentError.new("Tiled video linear row range is outside the input")
    end
    unless first_row % ROW_TILE == 0 && (range_end == rows || row_count % ROW_TILE == 0)
      raise ArgumentError.new("Tiled video linear ranges must preserve 8-row tile boundaries")
    end

    input_count = checked_count(rows, in_dim)
    weight_count = checked_count(out_dim, in_dim)
    output_count = checked_count(rows, out_dim)
    MetalOps.check_storage(x, input_count, input_dtype)
    MetalOps.check_storage(w, weight_count, weight_dtype)
    MetalOps.check_storage(bias, out_dim.to_i64, bias_dtype)
    MetalOps.check(y, output_count)
    MetalOps.distinct(y, x, w, bias)

    row_tiles = (row_count.to_i64 + ROW_TILE - 1) // ROW_TILE
    output_tiles = (out_dim.to_i64 + ROW_TILE - 1) // ROW_TILE
    groups = row_tiles * output_tiles
    unless groups > 0 && groups <= Int32::MAX
      raise ArgumentError.new("Tiled video linear dispatch is too large")
    end

    params = StaticArray[rows.to_u32, in_dim.to_u32, out_dim.to_u32,
      first_row.to_u32, row_count.to_u32,
      input_dtype.bf16? ? 1_u32 : 0_u32,
      weight_dtype.bf16? ? 1_u32 : 0_u32,
      bias_dtype.bf16? ? 1_u32 : 0_u32]
    e.set_pipeline(pipeline)
    # The kernel chooses one typed pointer for each duplicated binding. Binding
    # the same resident buffer twice avoids temporary conversion buffers.
    e.set_buffer(x, 0).set_buffer(x, 1)
    e.set_buffer(w, 2).set_buffer(w, 3)
    e.set_buffer(bias, 4).set_buffer(bias, 5)
    e.set_buffer(y, 6, ML::Metal::BufferAccess::Write)
    e.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 7)
    e.dispatch_threadgroups({groups.to_i32, 1, 1}, {THREADS, 1, 1})
  end

  private def self.checked_count(a : Int32, b : Int32) : Int64
    count = a.to_i64 * b
    unless count > 0 && count <= Int32::MAX
      raise ArgumentError.new("Tiled video linear tensor exceeds supported element count")
    end
    count
  end
end
