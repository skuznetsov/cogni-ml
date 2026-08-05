# Bounded CPU/F32 subdivision-logit projection for the TRELLIS.2 shape-SLat
# decoder seam.
#
# This leaf mirrors only SparseLinear's affine operation
# `features @ weights.T + bias` over flat caller-owned rows. It returns raw
# logits for the separate strict `> 0` mask boundary; it owns no sparse tensor,
# coordinate map, checkpoint loader, convolution, cache, device storage, or
# decoder/mesh execution. Finiteness, overflow, and resident-budget checks are
# local safety admission and are not claimed as upstream validation behavior.

module ML::ThreeD::Trellis2
  module ShapeSlatSubdivisionProjectionCPU
    extend self

    OUTPUT_CHANNELS    = ShapeSlatUpsampleLayoutCPU::SUBDIVISION_SLOTS
    MAX_INPUT_POINTS   = ShapeSlatUpsampleLayoutCPU::MAX_INPUT_COORDINATES
    MAX_INPUT_CHANNELS = 1_536_i32
    MAX_RESIDENT_BYTES = 64_i64 * 1024_i64 * 1024_i64

    def project(
      features : Array(Float32),
      weights : Array(Float32),
      bias : Array(Float32),
      *,
      input_point_count : Int32,
      input_channels : Int32,
      max_resident_bytes : Int64 = MAX_RESIDENT_BYTES,
    ) : Array(Float32)
      validate_budget!(max_resident_bytes)
      input_elements, output_elements, weight_elements = projection_sizes(
        features,
        weights,
        bias,
        input_point_count,
        input_channels
      )

      input_bytes = checked_bytes(input_elements, "feature")
      weight_bytes = checked_bytes(weight_elements, "subdivision weight")
      bias_bytes = checked_bytes(OUTPUT_CHANNELS.to_i64, "subdivision bias")
      output_bytes = checked_bytes(output_elements, "subdivision logits")
      resident_bytes = checked_sum(input_bytes, weight_bytes, "resident")
      resident_bytes = checked_sum(resident_bytes, bias_bytes, "resident")
      resident_bytes = checked_sum(resident_bytes, output_bytes, "resident")
      if resident_bytes > max_resident_bytes
        raise ArgumentError.new(
          "subdivision projection resident byte budget requires #{resident_bytes} bytes, " \
          "limit is #{max_resident_bytes}"
        )
      end

      features.each_with_index do |value, index|
        unless value.finite?
          raise ArgumentError.new("feature[#{index}] must be finite")
        end
      end
      weights.each_with_index do |value, index|
        unless value.finite?
          raise ArgumentError.new("subdivision weight[#{index}] must be finite")
        end
      end
      bias.each_with_index do |value, index|
        unless value.finite?
          raise ArgumentError.new("subdivision bias[#{index}] must be finite")
        end
      end

      logits = Array(Float32).new(output_elements.to_i)
      input_point_count.times do |row|
        input_offset = row * input_channels
        OUTPUT_CHANNELS.times do |output_channel|
          weight_offset = output_channel * input_channels
          sum = 0.0_f32
          input_channels.times do |input_channel|
            sum += features[input_offset + input_channel] *
                   weights[weight_offset + input_channel]
          end
          value = sum + bias[output_channel]
          unless value.finite?
            raise ArgumentError.new(
              "subdivision logit[#{logits.size}] must be finite"
            )
          end
          logits << value
        end
      end
      logits
    end

    # Compose the admitted raw-logit projection with the separate strict mask
    # boundary. The preflight counts both outputs together so a caller cannot
    # pass two individually valid stages while exceeding the shared logical
    # resident budget.
    def project_and_binarize(
      features : Array(Float32),
      weights : Array(Float32),
      bias : Array(Float32),
      *,
      input_point_count : Int32,
      input_channels : Int32,
      max_resident_bytes : Int64 = MAX_RESIDENT_BYTES,
    ) : Tuple(Array(Float32), Array(Array(Bool)))
      validate_budget!(max_resident_bytes)
      input_elements, output_elements, weight_elements = projection_sizes(
        features,
        weights,
        bias,
        input_point_count,
        input_channels
      )

      input_bytes = checked_bytes(input_elements, "feature")
      weight_bytes = checked_bytes(weight_elements, "subdivision weight")
      bias_bytes = checked_bytes(OUTPUT_CHANNELS.to_i64, "subdivision bias")
      logits_bytes = checked_bytes(output_elements, "subdivision logits")
      resident_bytes = checked_sum(input_bytes, weight_bytes, "resident")
      resident_bytes = checked_sum(resident_bytes, bias_bytes, "resident")
      resident_bytes = checked_sum(resident_bytes, logits_bytes, "resident")
      # The mask is a logical one-byte-per-slot payload in this CPU boundary.
      resident_bytes = checked_sum(
        resident_bytes,
        output_elements,
        "projection and mask resident"
      )
      if resident_bytes > max_resident_bytes
        raise ArgumentError.new(
          "subdivision projection and mask resident byte budget requires #{resident_bytes} bytes, " \
          "limit is #{max_resident_bytes}"
        )
      end

      logits = project(
        features,
        weights,
        bias,
        input_point_count: input_point_count,
        input_channels: input_channels,
        max_resident_bytes: max_resident_bytes
      )
      masks = ShapeSlatSubdivisionMaskCPU.binarize(
        logits,
        input_point_count: input_point_count,
        max_resident_bytes: max_resident_bytes
      )
      {logits, masks}
    end

    private def projection_sizes(
      features : Array(Float32),
      weights : Array(Float32),
      bias : Array(Float32),
      input_point_count : Int32,
      input_channels : Int32,
    ) : Tuple(Int64, Int64, Int64)
      unless input_point_count >= 0
        raise ArgumentError.new("input point count must be non-negative")
      end
      if input_point_count > MAX_INPUT_POINTS
        raise ArgumentError.new(
          "input point count exceeds #{MAX_INPUT_POINTS}"
        )
      end
      if input_channels > MAX_INPUT_CHANNELS
        raise ArgumentError.new(
          "input channel count exceeds #{MAX_INPUT_CHANNELS}"
        )
      end
      unless input_channels > 0
        raise ArgumentError.new(
          "input channel count must be in 1..#{MAX_INPUT_CHANNELS}"
        )
      end

      input_elements = checked_elements(
        input_point_count,
        input_channels,
        "feature"
      )
      output_elements = checked_elements(
        input_point_count,
        OUTPUT_CHANNELS,
        "subdivision logits"
      )
      weight_elements = checked_elements(
        OUTPUT_CHANNELS,
        input_channels,
        "subdivision weight"
      )

      unless features.size.to_i64 == input_elements
        raise ArgumentError.new(
          "feature payload size must match [input point count, input channel count]"
        )
      end
      unless weights.size.to_i64 == weight_elements
        raise ArgumentError.new(
          "subdivision weight payload size must match [8, input channel count]"
        )
      end
      unless bias.size == OUTPUT_CHANNELS
        raise ArgumentError.new(
          "subdivision bias payload size must match [8]"
        )
      end

      {input_elements, output_elements, weight_elements}
    end

    private def validate_budget!(max_resident_bytes : Int64) : Nil
      unless 0_i64 < max_resident_bytes <= MAX_RESIDENT_BYTES
        raise ArgumentError.new(
          "subdivision projection resident byte budget must be in 1..#{MAX_RESIDENT_BYTES}"
        )
      end
    end

    private def checked_elements(
      first : Int32,
      second : Int32,
      label : String,
    ) : Int64
      elements = first.to_i64 * second.to_i64
      if first > 0 && elements // first.to_i64 != second.to_i64
        raise ArgumentError.new("#{label} element count overflow")
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

    private def checked_sum(left : Int64, right : Int64, label : String) : Int64
      sum = left + right
      unless sum >= left && sum >= right
        raise ArgumentError.new("subdivision projection #{label} byte budget overflow")
      end
      sum
    end
  end
end
