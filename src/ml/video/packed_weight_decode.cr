require "../metal/dispatch"
require "./metal_ops"

module ML::Video::PackedWeightDecode
  # Stateless decode helpers for a bounded Wan projection probe. The caller
  # owns packed weights, LUTs, destination storage, command lifetime, and timing.
  BLOCK_ELEMENTS      = 256
  Q4_K_BLOCK_BYTES    = 144
  QBIT_P4_BLOCK_BYTES = 136
  QBIT_P5_BLOCK_BYTES = 168
  QBIT_LUT_ELEMENTS   = 256

  SOURCE = {{ read_file("#{__DIR__}/kernels/packed_weight_decode.metal") }}

  @@q4_k_pipeline : ML::Metal::ComputePipeline?
  @@qbit_pipeline : ML::Metal::ComputePipeline?

  # Compile the fixed kernels during setup so encode calls only bind and
  # dispatch into the caller's already-open serial encoder.
  def self.prepare! : Nil
    @@q4_k_pipeline ||= ML::Metal::PipelineCache.get("wan_packed_q4k_to_f32") do
      ML::Metal::ComputePipeline.new("wan_packed_q4k_to_f32", SOURCE)
    end
    @@qbit_pipeline ||= ML::Metal::PipelineCache.get("wan_packed_qbit_to_f32") do
      ML::Metal::ComputePipeline.new("wan_packed_qbit_to_f32", SOURCE)
    end
  end

  # Decode canonical ggml Q4_K blocks into caller-provided F32 storage.
  # `count` must describe whole 256-value Q4_K blocks.
  def self.q4_k(e : ML::Metal::ComputeEncoder, packed : ML::MetalBuffer,
                destination : ML::MetalBuffer, count : Int32) : Nil
    unless count > 0 && count % BLOCK_ELEMENTS == 0
      raise ArgumentError.new("Q4_K value count must be a positive multiple of 256")
    end
    ML::Video::MetalOps.check(destination, count.to_i64)
    packed_bytes = (count // BLOCK_ELEMENTS).to_i64 * Q4_K_BLOCK_BYTES
    check_packed!(packed, packed_bytes)
    ML::Video::MetalOps.distinct(destination, packed)
    selected_pipeline = @@q4_k_pipeline || raise ArgumentError.new("Call PackedWeightDecode.prepare! before encode")

    e.set_pipeline(selected_pipeline)
    e.set_buffer(packed, 0).set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
    e.set_value(count.to_u32, 2)
    e.dispatch_1d(count)
  end

  # Decode 256-element QBit Gaussian blocks. The caller uploads the codec's
  # 256-entry reconstruction LUT for precision 4 or 5 during setup.
  def self.qbit(e : ML::Metal::ComputeEncoder, packed : ML::MetalBuffer,
                destination : ML::MetalBuffer, lut : ML::MetalBuffer,
                count : Int32, precision : Int32) : Nil
    unless precision == 4 || precision == 5
      raise ArgumentError.new("QBit precision must be 4 or 5")
    end
    unless count > 0
      raise ArgumentError.new("QBit value count must be positive")
    end

    ML::Video::MetalOps.check(destination, count.to_i64)
    ML::Video::MetalOps.check(lut, QBIT_LUT_ELEMENTS.to_i64)
    blocks = (count.to_i64 + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
    block_bytes = precision == 4 ? QBIT_P4_BLOCK_BYTES : QBIT_P5_BLOCK_BYTES
    check_packed!(packed, blocks * block_bytes)
    ML::Video::MetalOps.distinct(destination, packed)
    ML::Video::MetalOps.distinct(destination, lut)
    if ML::Video::MetalOps.overlap?(packed, lut)
      raise ArgumentError.new("QBit packed payload and LUT overlap")
    end
    selected_pipeline = @@qbit_pipeline || raise ArgumentError.new("Call PackedWeightDecode.prepare! before encode")

    e.set_pipeline(selected_pipeline)
    e.set_buffer(packed, 0).set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
    e.set_buffer(lut, 2)
    e.set_value(count.to_u32, 3).set_value(precision.to_u32, 4)
    e.dispatch_1d(count)
  end

  private def self.check_packed!(buffer : ML::MetalBuffer, required_bytes : Int64) : Nil
    unless buffer.valid? && required_bytes > 0 && buffer.size >= required_bytes
      raise ArgumentError.new("Invalid or undersized packed video weight buffer")
    end
  end
end
