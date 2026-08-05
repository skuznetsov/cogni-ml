# Bounded CPU/F32 subdivision-mask projection for the TRELLIS.2 shape-SLat
# upsample seam.
#
# This leaf mirrors the finite-input portion of the decoder callsite
# `subdiv.feats > 0`, shared by the C2S3d up-block used in the shipped SC-VAE
# configs and by the alternate Upsample3d block definition. It owns no learned
# SparseLinear projection, sparse tensor, coordinate map, cache, device
# storage, or decoder/mesh execution.

module ML::ThreeD::Trellis2
  module ShapeSlatSubdivisionMaskCPU
    extend self

    SLOT_COUNT         = ShapeSlatUpsampleLayoutCPU::SUBDIVISION_SLOTS
    MAX_INPUT_POINTS   = ShapeSlatUpsampleLayoutCPU::MAX_INPUT_COORDINATES
    MAX_RESIDENT_BYTES = 64_i64 * 1024_i64 * 1024_i64

    def binarize(
      logits : Array(Float32),
      *,
      input_point_count : Int32,
      max_resident_bytes : Int64 = MAX_RESIDENT_BYTES,
    ) : Array(Array(Bool))
      validate_budget!(max_resident_bytes)
      unless input_point_count >= 0
        raise ArgumentError.new("input point count must be non-negative")
      end
      if input_point_count > MAX_INPUT_POINTS
        raise ArgumentError.new(
          "input point count exceeds #{MAX_INPUT_POINTS}"
        )
      end

      input_elements = input_point_count.to_i64 * SLOT_COUNT.to_i64
      unless logits.size.to_i64 == input_elements
        raise ArgumentError.new(
          "subdivision logits payload size must match [input point count, 8 slots]"
        )
      end

      logit_bytes = checked_bytes(input_elements, "subdivision logits")
      mask_bytes = input_elements
      resident_bytes = logit_bytes + mask_bytes
      unless resident_bytes >= logit_bytes
        raise ArgumentError.new("subdivision mask resident byte budget overflow")
      end
      if resident_bytes > max_resident_bytes
        raise ArgumentError.new(
          "subdivision mask resident byte budget requires #{resident_bytes} bytes, " \
          "limit is #{max_resident_bytes}"
        )
      end

      # Validate all logits before allocating nested mask rows. The local
      # contract is finite-only; source comparison behavior for NaN/Inf is
      # not admitted as a decoder safety policy.
      logits.each_with_index do |value, index|
        unless value.finite?
          raise ArgumentError.new(
            "subdivision logit[#{index}] must be finite"
          )
        end
      end

      Array(Array(Bool)).new(input_point_count.to_i) do |parent_index|
        row_offset = parent_index.to_i64 * SLOT_COUNT.to_i64
        Array(Bool).new(SLOT_COUNT.to_i) do |slot_index|
          logits[(row_offset + slot_index.to_i64).to_i] > 0.0_f32
        end
      end
    end

    private def validate_budget!(max_resident_bytes : Int64) : Nil
      unless 0_i64 < max_resident_bytes <= MAX_RESIDENT_BYTES
        raise ArgumentError.new(
          "subdivision mask resident byte budget must be in 1..#{MAX_RESIDENT_BYTES}"
        )
      end
    end

    private def checked_bytes(elements : Int64, label : String) : Int64
      bytes = elements * 4_i64
      unless bytes >= elements
        raise ArgumentError.new("#{label} byte count overflow")
      end
      bytes
    end
  end
end
