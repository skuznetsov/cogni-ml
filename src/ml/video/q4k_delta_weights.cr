require "../metal/dispatch"
require "./metal_ops"

module ML::Video::Q4KDeltaWeights
  BLOCK_ELEMENTS   =   256
  Q4_K_BLOCK_BYTES =   144
  STATUS_WORDS     =     1
  STATUS_ERROR     = 1_u32

  SOURCE = {{ read_file("#{__DIR__}/kernels/q4k_delta_weights.metal") }}

  @@pack_pipeline : ML::Metal::ComputePipeline?
  @@decode_pipeline : ML::Metal::ComputePipeline?

  # Compile the opt-in pack and fused decode kernels during setup. Encode calls
  # only validate, bind caller-owned storage, and dispatch on the open encoder.
  def self.prepare! : Nil
    @@pack_pipeline ||= ML::Metal::PipelineCache.get("wan_q4k_delta_pack") do
      ML::Metal::ComputePipeline.new("wan_q4k_delta_pack", SOURCE)
    end
    @@decode_pipeline ||= ML::Metal::PipelineCache.get("wan_q4k_delta_decode") do
      ML::Metal::ComputePipeline.new("wan_q4k_delta_decode", SOURCE)
    end
  end

  def self.sidecar_bytes(count : Int32, patches : Int32) : Int64
    validate_count!(count)
    validate_patches!(patches)
    (count.to_i64 // BLOCK_ELEMENTS) * patches.to_i64 * 3_i64
  end

  # Select the K largest block-local absolute reconstruction errors and write
  # fixed-stride {local U8 index, little-endian F16 delta} records. The caller
  # supplies canonical Q4_K bytes and owns all buffers and command lifetime.
  def self.pack(e : ML::Metal::ComputeEncoder, original_bf16 : ML::MetalBuffer,
                canonical_q4k_payload : ML::MetalBuffer, sidecar : ML::MetalBuffer,
                status : ML::MetalBuffer, count : Int32, patches : Int32) : Nil
    validate_count!(count)
    validate_patches!(patches)

    source_bytes = count.to_i64 * 2_i64
    payload_bytes = (count.to_i64 // BLOCK_ELEMENTS) * Q4_K_BLOCK_BYTES
    deltas_bytes = sidecar_bytes(count, patches)
    check_buffer!(original_bf16, source_bytes)
    check_buffer!(canonical_q4k_payload, payload_bytes)
    check_buffer!(sidecar, deltas_bytes)
    check_status!(status)
    unless disjoint?(original_bf16, canonical_q4k_payload, sidecar, status)
      raise ArgumentError.new("Q4_K delta source, payload, sidecar, and status buffers must be disjoint")
    end

    selected_pipeline = @@pack_pipeline || raise ArgumentError.new("Call Q4KDeltaWeights.prepare! before encode")
    e.set_pipeline(selected_pipeline)
      .set_buffer(original_bf16, 0)
      .set_buffer(canonical_q4k_payload, 1)
      .set_buffer(sidecar, 2, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(count.to_u32, 4).set_value(patches.to_u32, 5)
    e.dispatch_1d(count // BLOCK_ELEMENTS)
  end

  # Decode canonical Q4_K values and apply sparse half deltas in one kernel
  # dispatch. A malformed record set makes every output in that block zero.
  def self.decode(e : ML::Metal::ComputeEncoder, canonical_q4k_payload : ML::MetalBuffer,
                  sidecar : ML::MetalBuffer, destination : ML::MetalBuffer,
                  status : ML::MetalBuffer, count : Int32, patches : Int32) : Nil
    validate_count!(count)
    validate_patches!(patches)

    payload_bytes = (count.to_i64 // BLOCK_ELEMENTS) * Q4_K_BLOCK_BYTES
    deltas_bytes = sidecar_bytes(count, patches)
    check_buffer!(canonical_q4k_payload, payload_bytes)
    check_buffer!(sidecar, deltas_bytes)
    ML::Video::MetalOps.check(destination, count.to_i64)
    check_status!(status)
    unless disjoint?(canonical_q4k_payload, sidecar, destination, status)
      raise ArgumentError.new("Q4_K delta payload, sidecar, destination, and status buffers must be disjoint")
    end

    selected_pipeline = @@decode_pipeline || raise ArgumentError.new("Call Q4KDeltaWeights.prepare! before decode")
    e.set_pipeline(selected_pipeline)
      .set_buffer(canonical_q4k_payload, 0)
      .set_buffer(sidecar, 1)
      .set_buffer(destination, 2, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(count.to_u32, 4).set_value(patches.to_u32, 5)
    e.dispatch_1d(count)
  end

  private def self.validate_count!(count : Int32) : Nil
    unless count > 0 && count % BLOCK_ELEMENTS == 0
      raise ArgumentError.new("Q4_K delta value count must be a positive multiple of 256")
    end
  end

  private def self.validate_patches!(patches : Int32) : Nil
    unless patches == 1 || patches == 4 || patches == 8
      raise ArgumentError.new("Q4_K delta patch count must be 1, 4, or 8")
    end
  end

  private def self.check_buffer!(buffer : ML::MetalBuffer, required_bytes : Int64) : Nil
    unless buffer.valid? && required_bytes > 0 && buffer.size >= required_bytes
      raise ArgumentError.new("Invalid or undersized Q4_K delta buffer")
    end
  end

  private def self.check_status!(status : ML::MetalBuffer) : Nil
    check_buffer!(status, STATUS_WORDS.to_i64 * sizeof(UInt32))
  end

  private def self.disjoint?(a : ML::MetalBuffer, b : ML::MetalBuffer,
                             c : ML::MetalBuffer, d : ML::MetalBuffer) : Bool
    !ML::Video::MetalOps.overlap?(a, b) &&
      !ML::Video::MetalOps.overlap?(a, c) &&
      !ML::Video::MetalOps.overlap?(a, d) &&
      !ML::Video::MetalOps.overlap?(b, c) &&
      !ML::Video::MetalOps.overlap?(b, d) &&
      !ML::Video::MetalOps.overlap?(c, d)
  end
end
