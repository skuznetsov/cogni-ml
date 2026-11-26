require "../metal/dispatch"
require "./metal_ops"

module ML::Video::Q4KRangeWeights
  BLOCK_ELEMENTS   =   256
  Q4_K_BLOCK_BYTES =   144
  STATUS_WORDS     =     1
  STATUS_ERROR     = 1_u32

  SOURCE = {{ read_file("#{__DIR__}/kernels/q4k_range_weights.metal") }}

  @@pack_pipeline : ML::Metal::ComputePipeline?

  # Compile the packer during setup. Encoding binds only caller-owned buffers
  # and dispatches into the caller's already-open serial encoder.
  def self.prepare! : Nil
    @@pack_pipeline ||= ML::Metal::PipelineCache.get("wan_q4k_range_pack") do
      ML::Metal::ComputePipeline.new("wan_q4k_range_pack", SOURCE)
    end
  end

  def self.payload_bytes(count : Int32) : Int64
    validate_count!(count)
    (count.to_i64 // BLOCK_ELEMENTS) * Q4_K_BLOCK_BYTES
  end

  # Pack BF16 values into canonical 144-byte Q4_K blocks. The caller owns and
  # initializes the one-word sticky status buffer and all command lifetime.
  def self.pack(e : ML::Metal::ComputeEncoder, original_bf16 : ML::MetalBuffer,
                payload : ML::MetalBuffer, status : ML::MetalBuffer,
                count : Int32, ranges : Int32) : Nil
    validate_count!(count)
    unless ranges == 1 || ranges == 8
      raise ArgumentError.new("Q4_K range count must be 1 or 8")
    end

    source_bytes = count.to_i64 * 2
    output_bytes = payload_bytes(count)
    check_buffer!(original_bf16, source_bytes)
    check_buffer!(payload, output_bytes)
    check_buffer!(status, STATUS_WORDS.to_i64 * sizeof(UInt32))

    if ML::Video::MetalOps.overlap?(original_bf16, payload) ||
       ML::Video::MetalOps.overlap?(original_bf16, status) ||
       ML::Video::MetalOps.overlap?(payload, status)
      raise ArgumentError.new("Q4_K source, payload, and status buffers must be disjoint")
    end

    selected_pipeline = @@pack_pipeline || raise ArgumentError.new("Call Q4KRangeWeights.prepare! before encode")

    e.set_pipeline(selected_pipeline)
    e.set_buffer(original_bf16, 0)
      .set_buffer(payload, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(count.to_u32, 3).set_value(ranges.to_u32, 4)
    e.dispatch_1d(count // BLOCK_ELEMENTS)
  end

  private def self.validate_count!(count : Int32) : Nil
    unless count > 0 && count % BLOCK_ELEMENTS == 0
      raise ArgumentError.new("Q4_K value count must be a positive multiple of 256")
    end
  end

  private def self.check_buffer!(buffer : ML::MetalBuffer, required_bytes : Int64) : Nil
    unless buffer.valid? && required_bytes > 0 && buffer.size >= required_bytes
      raise ArgumentError.new("Invalid or undersized Q4_K range buffer")
    end
  end
end
