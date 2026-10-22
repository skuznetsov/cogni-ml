require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::QBitCodebookLinear
  SOURCE       = {{ read_file("#{__DIR__}/kernels/qbit_codebook_linear.metal") }}
  ROW_TILE     = 16
  OUTPUT_TILE  =  8
  K_TILE       =  8
  THREADS      = 32
  STATUS_WORDS =  8

  @@linear : ML::Metal::ComputePipeline?
  @@dense : ML::Metal::ComputePipeline?

  # Compile during setup, before a caller opens or chains an encoder.
  def self.prepare! : Nil
    @@linear ||= pipeline("video_qbit_codebook_linear_f32")
    @@dense ||= pipeline("video_qbit_codebook_dense_linear_f32")
  end

  def self.pipeline : ML::Metal::ComputePipeline
    @@linear || raise ArgumentError.new("Call QBitCodebookLinear.prepare! before encode")
  end

  def self.dense_pipeline : ML::Metal::ComputePipeline
    @@dense || raise ArgumentError.new("Call QBitCodebookLinear.prepare! before encode")
  end

  # Encodes the full F32 projection without allocating or waiting. The packed
  # positional status must be initialized by a successful pack operation (or
  # explicitly zeroed by the caller). Status words 5/6 reject failed packing;
  # word 7 accumulates consumer-side invalid-record errors. Do not inspect or
  # reuse status until the caller's terminal completion, and clear it before a
  # later run if word 7 became nonzero.
  def self.linear(e : ML::Metal::ComputeEncoder,
                  x : ML::MetalBuffer, offsets : ML::MetalBuffer,
                  blob : ML::MetalBuffer, lut4 : ML::MetalBuffer,
                  lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                  y : ML::MetalBuffer, status : ML::MetalBuffer,
                  rows : Int32, in_dim : Int32, out_dim : Int32,
                  tile_values : Int32 = 1024) : Nil
    linear_range(e, x, offsets, blob, lut4, lut5, bias, y, status,
      rows, in_dim, out_dim, 0, rows, tile_values)
  end

  # Encodes one row-aligned range on the caller's serial encoder. The final
  # range may be partial; every earlier range must end on ROW_TILE. The kernel
  # retains K traversal order across ranges and shares each decoded B fragment
  # between two 8-row accumulators.
  def self.linear_range(e : ML::Metal::ComputeEncoder,
                        x : ML::MetalBuffer, offsets : ML::MetalBuffer,
                        blob : ML::MetalBuffer, lut4 : ML::MetalBuffer,
                        lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                        y : ML::MetalBuffer, status : ML::MetalBuffer,
                        rows : Int32, in_dim : Int32, out_dim : Int32,
                        first_row : Int32, row_count : Int32,
                        tile_values : Int32 = 1024) : Nil
    validate_range!(rows, in_dim, out_dim, first_row, row_count)
    tiles = positional_tiles!(in_dim, out_dim, tile_values)
    input_count = checked_count(rows, in_dim)
    weight_count = checked_count(out_dim, in_dim)
    output_count = checked_count(rows, out_dim)
    MetalOps.check_storage(x, input_count, WeightDType::F32)
    MetalOps.check_storage(bias, out_dim.to_i64, WeightDType::F32)
    MetalOps.check(y, output_count)
    bytes!(offsets, (tiles.to_i64 + 1) * 4)
    capacity!(blob)
    MetalOps.check_storage(lut4, 256_i64, WeightDType::F32)
    MetalOps.check_storage(lut5, 256_i64, WeightDType::F32)
    bytes!(status, STATUS_WORDS.to_i64 * 4)
    MetalOps.distinct(y, x, offsets, blob, lut4, lut5, bias, status)
    MetalOps.distinct(status, x, offsets, blob, lut4, lut5, bias, y)

    row_tiles = (row_count.to_i64 + ROW_TILE - 1) // ROW_TILE
    output_tiles = (out_dim.to_i64 + OUTPUT_TILE - 1) // OUTPUT_TILE
    groups = row_tiles * output_tiles
    unless groups > 0 && groups <= Int32::MAX
      raise ArgumentError.new("Codebook linear dispatch is too large")
    end
    params = StaticArray[rows.to_u32, in_dim.to_u32, out_dim.to_u32,
      first_row.to_u32, row_count.to_u32, tile_values.to_u32, tiles.to_u32,
      blob.size.to_u32]
    e.set_pipeline(pipeline)
      .set_buffer(x, 0).set_buffer(offsets, 1).set_buffer(blob, 2)
      .set_buffer(lut4, 3).set_buffer(lut5, 4).set_buffer(bias, 5)
      .set_buffer(y, 6, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 7, ML::Metal::BufferAccess::ReadWrite)
      .set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 8)
      .dispatch_threadgroups({groups.to_i32, 1, 1}, {THREADS, 1, 1})
  end

  # Bounded asynchronous row dispatches on one existing command chain. This
  # only pauses/continues the chain between ranges; the caller owns its single
  # terminal wait and all buffer lifetimes.
  def self.linear_bounded(chain : ML::Metal::MPSCommandChain,
                          x : ML::MetalBuffer, offsets : ML::MetalBuffer,
                          blob : ML::MetalBuffer, lut4 : ML::MetalBuffer,
                          lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                          y : ML::MetalBuffer, status : ML::MetalBuffer,
                          rows : Int32, in_dim : Int32, out_dim : Int32,
                          rows_per_segment : Int32 = 64,
                          tile_values : Int32 = 1024) : Nil
    unless rows_per_segment >= ROW_TILE && rows_per_segment % ROW_TILE == 0
      raise ArgumentError.new("Codebook linear segments must be a positive multiple of 16 rows")
    end
    first = 0
    while first < rows
      count = Math.min(rows_per_segment, rows - first)
      linear_range(chain.encoder, x, offsets, blob, lut4, lut5, bias, y, status,
        rows, in_dim, out_dim, first, count, tile_values)
      first += count
      unless first == rows
        chain.pause!
        chain.continue_async!
        chain.resume!
      end
    end
  end

  # Same-geometry dense F32 control. It uses the same row/output tiling, K
  # fragment width, two accumulators, matrix instruction order, and bias add;
  # only the right-hand fragment source is a resident dense matrix.
  def self.dense_linear(e : ML::Metal::ComputeEncoder,
                        x : ML::MetalBuffer, weight : ML::MetalBuffer,
                        bias : ML::MetalBuffer, y : ML::MetalBuffer,
                        rows : Int32, in_dim : Int32, out_dim : Int32) : Nil
    dense_linear_range(e, x, weight, bias, y, rows, in_dim, out_dim, 0, rows)
  end

  def self.dense_linear_range(e : ML::Metal::ComputeEncoder,
                              x : ML::MetalBuffer, weight : ML::MetalBuffer,
                              bias : ML::MetalBuffer, y : ML::MetalBuffer,
                              rows : Int32, in_dim : Int32, out_dim : Int32,
                              first_row : Int32, row_count : Int32) : Nil
    validate_range!(rows, in_dim, out_dim, first_row, row_count)
    input_count = checked_count(rows, in_dim)
    weight_count = checked_count(out_dim, in_dim)
    output_count = checked_count(rows, out_dim)
    MetalOps.check_storage(x, input_count, WeightDType::F32)
    MetalOps.check_storage(weight, weight_count, WeightDType::F32)
    MetalOps.check_storage(bias, out_dim.to_i64, WeightDType::F32)
    MetalOps.check(y, output_count)
    MetalOps.distinct(y, x, weight, bias)

    row_tiles = (row_count.to_i64 + ROW_TILE - 1) // ROW_TILE
    output_tiles = (out_dim.to_i64 + OUTPUT_TILE - 1) // OUTPUT_TILE
    groups = row_tiles * output_tiles
    unless groups > 0 && groups <= Int32::MAX
      raise ArgumentError.new("Dense codebook control dispatch is too large")
    end
    params = StaticArray[rows.to_u32, in_dim.to_u32, out_dim.to_u32,
      first_row.to_u32, row_count.to_u32]
    e.set_pipeline(dense_pipeline)
      .set_buffer(x, 0).set_buffer(weight, 1).set_buffer(bias, 2)
      .set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      .set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 4)
      .dispatch_threadgroups({groups.to_i32, 1, 1}, {THREADS, 1, 1})
  end

  def self.dense_linear_bounded(chain : ML::Metal::MPSCommandChain,
                                x : ML::MetalBuffer, weight : ML::MetalBuffer,
                                bias : ML::MetalBuffer, y : ML::MetalBuffer,
                                rows : Int32, in_dim : Int32, out_dim : Int32,
                                rows_per_segment : Int32 = 64) : Nil
    unless rows_per_segment >= ROW_TILE && rows_per_segment % ROW_TILE == 0
      raise ArgumentError.new("Dense codebook segments must be a positive multiple of 16 rows")
    end
    first = 0
    while first < rows
      count = Math.min(rows_per_segment, rows - first)
      dense_linear_range(chain.encoder, x, weight, bias, y,
        rows, in_dim, out_dim, first, count)
      first += count
      unless first == rows
        chain.pause!
        chain.continue_async!
        chain.resume!
      end
    end
  end

  private def self.validate_range!(rows : Int32, in_dim : Int32, out_dim : Int32,
                                   first_row : Int32, row_count : Int32) : Nil
    MetalOps.dims(rows, in_dim, out_dim)
    range_end = first_row.to_i64 + row_count
    unless first_row >= 0 && row_count > 0 && range_end <= rows
      raise ArgumentError.new("Codebook linear row range is outside the input")
    end
    unless first_row % ROW_TILE == 0 &&
           (range_end == rows || row_count % ROW_TILE == 0)
      raise ArgumentError.new("Codebook linear ranges must preserve 16-row tile boundaries")
    end
  end

  private def self.positional_tiles!(in_dim : Int32, out_dim : Int32,
                                     tile_values : Int32) : Int32
    unless (tile_values == 256 || tile_values == 1024) && in_dim % tile_values == 0
      raise ArgumentError.new("Codebook consumer requires 256- or 1024-value records and aligned K")
    end
    weights = checked_count(out_dim, in_dim)
    tiles = weights // tile_values
    unless tiles > 0 && tiles <= UInt32::MAX
      raise ArgumentError.new("Codebook positional offset count exceeds UInt32")
    end
    tiles.to_i32
  end

  private def self.checked_count(a : Int32, b : Int32) : Int64
    count = a.to_i64 * b
    unless count > 0 && count <= Int32::MAX
      raise ArgumentError.new("Codebook linear tensor exceeds supported element count")
    end
    count
  end

  private def self.bytes!(buffer : ML::MetalBuffer, minimum : Int64) : Nil
    unless buffer.valid? && minimum > 0 && buffer.size >= minimum
      raise ArgumentError.new("Invalid or undersized codebook linear buffer")
    end
  end

  private def self.capacity!(buffer : ML::MetalBuffer) : Nil
    bytes!(buffer, 4_i64)
    if buffer.size > UInt32::MAX
      raise ArgumentError.new("Codebook positional blob capacity exceeds UInt32")
    end
  end

  private def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
  end
end
