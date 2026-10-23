require "./wan_tensorops_kernel"

# Opt-in 4-SIMDgroup probe. These adapters compile a runtime shader during
# preparation; encode only appends one dispatch to the caller's command chain.
class WanSharedPanelKernel < WanTensorOpsKernel
  SHARED_PANEL_SOURCE = {{ read_file("#{__DIR__}/wan_shared_panel_gemm.metal") }}

  def initialize(tile_m : Int32 = 32, tile_n : Int32 = 32,
                 shader : String = SHARED_PANEL_SOURCE,
                 function : String = "wan_shared_panel_f32")
    unless [{32, 32}, {64, 32}].includes?({tile_m, tile_n})
      raise ArgumentError.new("Unadmitted shared-panel tile")
    end
    super(tile_m, tile_n, shader, function)
    unless @pipeline.thread_execution_width == 32 &&
           @pipeline.max_total_threads_per_threadgroup >= 128 &&
           ML::Metal::Device.instance.max_threads_per_threadgroup >= 128
      raise "Shared-panel probe requires four 32-lane SIMD groups"
    end
    if @pipeline.static_threadgroup_memory_length > 16_i64 * 1024
      raise "Shared-panel probe exceeds its 16 KiB static-memory admission"
    end
  end

  def route : String
    "shared_panel_f32"
  end

  def encode(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
             weight : ML::MetalBuffer, bias : ML::MetalBuffer,
             output : ML::MetalBuffer, rows : Int32, din : Int32, dout : Int32) : Nil
    ML::Video::MetalOps.dims(rows, din, dout)
    {rows.to_i64 * din, din.to_i64 * dout, rows.to_i64 * dout}.each do |count|
      raise ArgumentError.new("Shared-panel probe extent exceeds Int32") if count > Int32::MAX
    end
    ML::Video::MetalOps.check(input, rows.to_i64 * din)
    ML::Video::MetalOps.check(weight, din.to_i64 * dout)
    ML::Video::MetalOps.check(bias, dout.to_i64)
    ML::Video::MetalOps.check(output, rows.to_i64 * dout)
    ML::Video::MetalOps.distinct(output, input, weight, bias)
    shape = StaticArray[rows.to_u32, din.to_u32, dout.to_u32, 0_u32]
    e.set_pipeline(@pipeline).set_buffer(input, 0).set_buffer(weight, 1)
      .set_buffer(bias, 2).set_buffer(output, 3, ML::Metal::BufferAccess::Write)
      .set_bytes(shape.to_unsafe.as(Void*), 16, 5)
      .dispatch_threadgroups({((dout.to_i64 + @tile_n - 1) // @tile_n).to_i32,
                              ((rows.to_i64 + @tile_m - 1) // @tile_m).to_i32, 1}, {32, 4, 1})
  end
end

# Immutable positional-QBit adapter. It mirrors the existing TensorOps probe
# API and fail-closed record/status contract while decoding B into shared
# threadgroup memory instead of materializing a global F32 weight matrix.
class WanSharedPanelQBitKernel < WanSharedPanelKernel
  QBIT_SHARED_PANEL_SOURCE = {{ read_file("#{__DIR__}/wan_shared_panel_gemm.metal") }}

  def initialize(tile_m : Int32, tile_n : Int32, @blob : ML::MetalBuffer,
                 @offsets : ML::MetalBuffer, @lut4 : ML::MetalBuffer,
                 @lut5 : ML::MetalBuffer, @status : ML::MetalBuffer,
                 @blob_bytes : Int32, @offset_count : Int32)
    unless @blob.valid? && @blob_bytes > 0 && @blob.size >= @blob_bytes
      raise ArgumentError.new("Invalid QBit blob byte capacity")
    end
    ML::Video::MetalOps.check(@offsets, @offset_count.to_i64)
    ML::Video::MetalOps.check(@lut4, 256_i64)
    ML::Video::MetalOps.check(@lut5, 256_i64)
    ML::Video::MetalOps.check(@status, 8_i64)
    super(tile_m, tile_n, QBIT_SHARED_PANEL_SOURCE, "wan_shared_panel_qbit_positional")
  end

  def route : String
    "shared_panel_qbit_positional"
  end

  def status_buffer : ML::MetalBuffer?
    @status
  end

  def encode(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
             weight : ML::MetalBuffer, bias : ML::MetalBuffer,
             output : ML::MetalBuffer, rows : Int32, din : Int32, dout : Int32) : Nil
    ML::Video::MetalOps.dims(rows, din, dout)
    raise ArgumentError.new("QBit probe requires K divisible by 1024") unless din % 1024 == 0
    {rows.to_i64 * din, din.to_i64 * dout, rows.to_i64 * dout}.each do |count|
      raise ArgumentError.new("QBit probe extent exceeds Int32") if count > Int32::MAX
    end
    expected_offsets = dout.to_i64 * (din // 1024) + 1
    raise ArgumentError.new("QBit offset capacity mismatch") unless @offset_count.to_i64 == expected_offsets
    ML::Video::MetalOps.check(input, rows.to_i64 * din)
    ML::Video::MetalOps.check(bias, dout.to_i64)
    ML::Video::MetalOps.check(output, rows.to_i64 * dout)
    ML::Video::MetalOps.distinct(output, input, weight, bias, @blob, @offsets, @lut4, @lut5, @status)
    ML::Video::MetalOps.distinct(@status, input, weight, bias, @blob, @offsets, @lut4, @lut5)
    shape = StaticArray[rows.to_u32, din.to_u32, dout.to_u32, 1024_u32]
    limits = StaticArray[@blob_bytes.to_u32, @offset_count.to_u32, 256_u32, 256_u32]
    e.set_pipeline(@pipeline).set_buffer(input, 0).set_buffer(@blob, 1)
      .set_buffer(@offsets, 2).set_buffer(@lut4, 3).set_buffer(@lut5, 4)
      .set_buffer(bias, 5).set_buffer(output, 6, ML::Metal::BufferAccess::Write)
      .set_buffer(@status, 7, ML::Metal::BufferAccess::ReadWrite)
      .set_bytes(shape.to_unsafe.as(Void*), 16, 8)
      .set_bytes(limits.to_unsafe.as(Void*), 16, 9)
      .dispatch_threadgroups({((dout.to_i64 + @tile_n - 1) // @tile_n).to_i32,
                              ((rows.to_i64 + @tile_m - 1) // @tile_m).to_i32, 1}, {32, 4, 1})
  end
end
