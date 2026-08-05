# Bounded CPU/F32 feature transport for the TRELLIS.2 shape-SLat C2S seam.
#
# This leaf mirrors only the uncached SparseChannel2Spatial gather:
# `x.feats.reshape(N * 8, -1)[idx * 8 + subidx]`. The caller supplies the exact
# ShapeSlatUpsampleLayout together with its derived index plan and a contiguous
# row-major [N, 8*C] payload. It does not allocate a sparse carrier, inspect
# caches, transport `_scale`, execute a convolution, or claim decoder/mesh
# parity.

module ML::ThreeD::Trellis2
  module ShapeSlatC2SFeatureTransportCPU
    extend self

    MAX_INPUT_POINTS    = ShapeSlatC2SIndexLayoutCPU::MAX_INPUT_COORDINATES
    MAX_OUTPUT_POINTS   = ShapeSlatC2SIndexLayoutCPU::MAX_OUTPUT_COORDINATES
    MAX_OUTPUT_CHANNELS = ShapeSlatC2SIndexLayoutCPU::MAX_OUTPUT_CHANNELS
    MAX_PACKED_CHANNELS = ShapeSlatC2SIndexLayoutCPU::MAX_PACKED_CHANNELS
    MAX_FEATURE_BYTES   = 64_i64 * 1024_i64 * 1024_i64

    # Gather one C-wide output row for each packed source index in `plan`.
    # `layout` and `plan` are checked together so a public plan constructor
    # cannot silently pair a valid map with a different source layout.
    # `packed_channels` is checked against the caller payload and plan width;
    # the payload itself is the authority for actual storage size.
    def gather(
      layout : ShapeSlatUpsampleLayout,
      plan : ShapeSlatC2SIndexLayout,
      packed_features : Array(Float32),
      *,
      input_point_count : Int32,
      packed_channels : Int32,
      max_feature_bytes : Int64 = MAX_FEATURE_BYTES,
    ) : Array(Float32)
      validate_budget!(max_feature_bytes)
      validate_input_points!(input_point_count)
      validate_packed_channels!(packed_channels)
      validate_plan!(layout, plan, packed_channels, input_point_count)
      if layout.child_count > MAX_OUTPUT_POINTS
        raise ArgumentError.new(
          "output point count exceeds #{MAX_OUTPUT_POINTS}"
        )
      end

      input_elements = checked_elements(
        input_point_count,
        packed_channels,
        "packed input feature"
      )
      unless packed_features.size.to_i64 == input_elements
        raise ArgumentError.new(
          "packed feature payload size must match [input point count, packed channel count]"
        )
      end
      input_bytes = checked_bytes(input_elements, "packed input feature")
      output_elements = checked_elements(
        layout.child_count,
        plan.output_channels,
        "C2S output feature"
      )
      output_bytes = checked_bytes(output_elements, "C2S output feature")
      resident_bytes = input_bytes + output_bytes
      unless resident_bytes >= input_bytes
        raise ArgumentError.new("C2S feature byte budget overflow")
      end
      if resident_bytes > max_feature_bytes
        raise ArgumentError.new(
          "C2S feature resident bytes require #{resident_bytes} bytes, " \
          "limit is #{max_feature_bytes}"
        )
      end

      packed_features.each_with_index do |value, index|
        unless value.finite?
          raise ArgumentError.new("packed feature[#{index}] must be finite")
        end
      end

      output = Array(Float32).new(output_elements.to_i, 0.0_f32)
      plan.packed_source_indices.each_with_index do |packed_index, child_index|
        source_offset = packed_index.to_i64 * plan.output_channels.to_i64
        output_offset = child_index.to_i64 * plan.output_channels.to_i64
        plan.output_channels.times do |channel|
          output[(output_offset + channel.to_i64).to_i] =
            packed_features[(source_offset + channel.to_i64).to_i]
        end
      end
      output
    end

    private def validate_budget!(max_feature_bytes : Int64) : Nil
      unless 0_i64 < max_feature_bytes <= MAX_FEATURE_BYTES
        raise ArgumentError.new(
          "feature byte budget must be in 1..#{MAX_FEATURE_BYTES}"
        )
      end
    end

    private def validate_input_points!(input_point_count : Int32) : Nil
      unless 0 <= input_point_count <= MAX_INPUT_POINTS
        raise ArgumentError.new(
          "input point count must be in 0..#{MAX_INPUT_POINTS}"
        )
      end
    end

    private def validate_packed_channels!(packed_channels : Int32) : Nil
      unless 1 <= packed_channels <= MAX_PACKED_CHANNELS
        raise ArgumentError.new(
          "packed channel count must be in 1..#{MAX_PACKED_CHANNELS}"
        )
      end
      unless packed_channels % ShapeSlatC2SIndexLayoutCPU::SUBDIVISION_SLOTS == 0
        raise ArgumentError.new(
          "packed channel count must be a multiple of " \
          "#{ShapeSlatC2SIndexLayoutCPU::SUBDIVISION_SLOTS}"
        )
      end
    end

    private def validate_plan!(
      layout : ShapeSlatUpsampleLayout,
      plan : ShapeSlatC2SIndexLayout,
      packed_channels : Int32,
      input_point_count : Int32,
    ) : Nil
      unless layout.factor == ShapeSlatC2SIndexLayoutCPU::FACTOR
        raise ArgumentError.new(
          "C2S source layout requires factor #{ShapeSlatC2SIndexLayoutCPU::FACTOR}"
        )
      end
      unless plan.factor == ShapeSlatC2SIndexLayoutCPU::FACTOR
        raise ArgumentError.new(
          "C2S plan requires factor #{ShapeSlatC2SIndexLayoutCPU::FACTOR}"
        )
      end
      unless layout.child_count == plan.child_count
        raise ArgumentError.new("C2S plan child count must match source layout")
      end
      unless layout.parent_indices.size == layout.child_count &&
             layout.subindices.size == layout.child_count
        raise ArgumentError.new("C2S source layout mappings must match child count")
      end
      unless plan.slots_per_parent == ShapeSlatC2SIndexLayoutCPU::SUBDIVISION_SLOTS
        raise ArgumentError.new("C2S plan requires 8 slots per parent")
      end
      output_channels = packed_channels // ShapeSlatC2SIndexLayoutCPU::SUBDIVISION_SLOTS
      unless 1 <= plan.output_channels <= MAX_OUTPUT_CHANNELS
        raise ArgumentError.new(
          "C2S plan output channel count must be in 1..#{MAX_OUTPUT_CHANNELS}"
        )
      end
      unless plan.output_channels == output_channels
        raise ArgumentError.new(
          "C2S plan output channel count must match packed channel count"
        )
      end

      layout.coordinates.each_with_index do |coordinate, child_index|
        unless coordinate.batch >= 0 && coordinate.x >= 0 &&
               coordinate.y >= 0 && coordinate.z >= 0
          raise ArgumentError.new(
            "C2S source layout coordinate #{child_index} must be non-negative"
          )
        end
        parent_index = layout.parent_indices[child_index]?
        subindex = layout.subindices[child_index]?
        unless parent_index && subindex
          raise ArgumentError.new("C2S source layout mappings must match child count")
        end
        unless 0 <= parent_index < input_point_count
          raise ArgumentError.new(
            "parent index #{parent_index} for child #{child_index} is out of range"
          )
        end
        unless 0 <= subindex < ShapeSlatC2SIndexLayoutCPU::SUBDIVISION_SLOTS
          raise ArgumentError.new(
            "subdivision slot #{subindex} for child #{child_index} is out of range"
          )
        end
        expected_index = parent_index.to_i64 *
                         ShapeSlatC2SIndexLayoutCPU::SUBDIVISION_SLOTS.to_i64 +
                         subindex.to_i64
        actual_index = plan.packed_source_indices[child_index]?
        unless actual_index && actual_index.to_i64 == expected_index
          raise ArgumentError.new(
            "packed source index for child #{child_index} does not match source layout"
          )
        end
      end
    end

    private def checked_elements(
      point_count : Int32,
      channels : Int32,
      label : String,
    ) : Int64
      elements = point_count.to_i64 * channels.to_i64
      unless elements <= Int32::MAX.to_i64
        raise ArgumentError.new("#{label} element count exceeds Int32 range")
      end
      elements
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
