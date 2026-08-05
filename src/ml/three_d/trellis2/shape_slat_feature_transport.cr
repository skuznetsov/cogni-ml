# Bounded CPU/F32 feature transport for the TRELLIS.2 shape-SLat upsample seam.
#
# This leaf mirrors SparseUpsample's nearest-neighbor gather
# `new_feats = x.feats[idx]` after ShapeSlatUpsampleLayout has established the
# source-order parent mapping. It owns one fresh flat [child_count, C] payload;
# it does not run a sparse convolution, predict subdivisions, decode weights,
# allocate device storage, or claim mesh parity.

module ML::ThreeD::Trellis2
  module ShapeSlatFeatureTransportCPU
    extend self

    MAX_CHANNELS      = 1_536_i32
    MAX_INPUT_POINTS  = ShapeSlatUpsampleLayoutCPU::MAX_INPUT_COORDINATES
    MAX_OUTPUT_POINTS = ShapeSlatUpsampleLayoutCPU::MAX_OUTPUT_COORDINATES
    MAX_FEATURE_BYTES = 64_i64 * 1024_i64 * 1024_i64

    def gather(
      layout : ShapeSlatUpsampleLayout,
      input_features : Array(Float32),
      *,
      input_point_count : Int32,
      channels : Int32,
      max_feature_bytes : Int64 = MAX_FEATURE_BYTES,
    ) : Array(Float32)
      validate_budget!(max_feature_bytes)
      unless input_point_count >= 0
        raise ArgumentError.new("input point count must be non-negative")
      end
      if input_point_count > MAX_INPUT_POINTS
        raise ArgumentError.new(
          "input point count exceeds #{MAX_INPUT_POINTS}"
        )
      end
      unless 1 <= channels <= MAX_CHANNELS
        raise ArgumentError.new("feature channel count must be in 1..#{MAX_CHANNELS}")
      end

      input_elements = checked_elements(
        input_point_count,
        channels,
        "input feature"
      )
      unless input_features.size.to_i64 == input_elements
        raise ArgumentError.new(
          "input feature payload size must match [input point count, channel count]"
        )
      end
      input_bytes = checked_bytes(input_elements, "input feature")
      if input_bytes > max_feature_bytes
        raise ArgumentError.new(
          "input feature byte budget requires #{input_bytes} bytes, limit is #{max_feature_bytes}"
        )
      end

      # Validate all source values and mappings before allocating the output.
      input_features.each_with_index do |value, index|
        unless value.finite?
          raise ArgumentError.new("input feature[#{index}] must be finite")
        end
      end
      layout.parent_indices.each_with_index do |parent_index, child_index|
        unless 0 <= parent_index < input_point_count
          raise ArgumentError.new(
            "parent index #{parent_index} for child #{child_index} is out of range"
          )
        end
      end
      if layout.child_count > MAX_OUTPUT_POINTS
        raise ArgumentError.new(
          "output point count exceeds #{MAX_OUTPUT_POINTS}"
        )
      end

      output_elements = checked_elements(
        layout.child_count,
        channels,
        "output feature"
      )
      output_bytes = checked_bytes(output_elements, "output feature")
      if output_bytes > max_feature_bytes
        raise ArgumentError.new(
          "output feature byte budget requires #{output_bytes} bytes, limit is #{max_feature_bytes}"
        )
      end
      resident_bytes = input_bytes + output_bytes
      unless resident_bytes >= input_bytes
        raise ArgumentError.new("feature byte budget overflow")
      end
      if resident_bytes > max_feature_bytes
        raise ArgumentError.new(
          "feature byte budget requires #{resident_bytes} resident bytes, limit is #{max_feature_bytes}"
        )
      end

      output = Array(Float32).new(output_elements.to_i, 0.0_f32)
      layout.parent_indices.each_with_index do |parent_index, child_index|
        source_offset = parent_index.to_i64 * channels.to_i64
        output_offset = child_index.to_i64 * channels.to_i64
        channels.times do |channel|
          output[(output_offset + channel.to_i64).to_i] =
            input_features[(source_offset + channel.to_i64).to_i]
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
