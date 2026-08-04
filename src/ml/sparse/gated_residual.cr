require "./tensor"

module ML::Sparse
  class TensorCPU
    # Applies the post-attention gated residual used by TRELLIS.2 modulated
    # sparse transformer blocks: x + h * gate_msa[batch]. This graphless CPU/F32
    # leaf borrows canonical inputs and allocates exactly one output payload.
    def self.apply_gated_residual(
      residual : TensorCPU,
      attention_output : TensorCPU,
      gate_msa : ML::Tensor,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse gated residual requires the base TensorCPU receiver"
        )
      end
      unless residual.@initialized && attention_output.@initialized
        raise SparseTensorError.new(
          "sparse gated residual requires initialized sparse values"
        )
      end
      residual_limit = standard_carrier_channel_limit(
        residual,
        "sparse gated residual"
      )
      attention_limit = standard_carrier_channel_limit(
        attention_output,
        "sparse gated residual"
      )
      unless residual.@carrier_role == attention_output.@carrier_role &&
             residual_limit == attention_limit
        raise SparseTensorError.new(
          "sparse gated residual requires the same carrier role"
        )
      end

      residual_map = residual.@coordinate_map
      attention_map = attention_output.@coordinate_map
      unless Box(CoordinateMap3D).box(residual_map) ==
               Box(CoordinateMap3D).box(attention_map)
        raise SparseTensorError.new(
          "sparse gated residual requires the same immutable coordinate map object"
        )
      end

      batch_size, map_point_count = CoordinateMap3D.kernel_layout(residual_map)
      unless 1 <= batch_size <= CoordinateMap3D::MAX_BATCH_SIZE
        raise SparseTensorError.new(
          "sparse gated residual batch size must be in 1..#{CoordinateMap3D::MAX_BATCH_SIZE}"
        )
      end
      unless 0 <= map_point_count <= CoordinateMap3D::MAX_POINTS
        raise SparseTensorError.new(
          "sparse gated residual point count must be in 0..#{CoordinateMap3D::MAX_POINTS}"
        )
      end
      unless residual.@point_count == map_point_count &&
             attention_output.@point_count == map_point_count
        raise SparseTensorError.new(
          "sparse gated residual requires coordinate and feature point counts to match"
        )
      end

      point_count = residual.@point_count
      channels = residual.@channels
      unless point_count == attention_output.@point_count &&
             channels == attention_output.@channels
        raise SparseTensorError.new(
          "sparse gated residual requires the same row and channel shape"
        )
      end
      unless 1 <= channels <= residual_limit
        raise SparseTensorError.new(
          "sparse gated residual channel count must be in 1..#{residual_limit}"
        )
      end
      unless 0_i64 < residual.@max_feature_bytes <= MAX_FEATURE_BYTES &&
             0_i64 < attention_output.@max_feature_bytes <= MAX_FEATURE_BYTES
        raise SparseTensorError.new(
          "sparse gated residual feature byte budgets must be in 1..#{MAX_FEATURE_BYTES}"
        )
      end

      expected_elements = point_count.to_i64 * channels.to_i64
      unless residual.@features.size.to_i64 == expected_elements &&
             attention_output.@features.size.to_i64 == expected_elements
        raise SparseTensorError.new(
          "sparse gated residual feature storage size must match [N, C]"
        )
      end

      # Gate storage is borrowed under Tensor's existing no-concurrent-mutation
      # precondition. Read base-owned ivars directly so a Tensor subclass cannot
      # spoof metadata or substitute a virtual CPUReadView.
      gate_device = gate_msa.@device
      gate_dtype = gate_msa.@dtype
      gate_data = gate_msa.@cpu_data
      gate_buffer = gate_msa.@buffer
      unless gate_device.cpu? && gate_buffer.nil? && gate_data
        raise SparseTensorError.new(
          "sparse gated residual gate_msa must be on CPU"
        )
      end
      unless gate_dtype.f32?
        raise SparseTensorError.new(
          "sparse gated residual gate_msa must use F32"
        )
      end
      gate_shape = gate_msa.@shape
      gate_strides = gate_msa.@strides
      unless gate_strides.contiguous?(gate_shape)
        raise SparseTensorError.new(
          "sparse gated residual gate_msa must be contiguous"
        )
      end
      unless gate_shape.ndim == 2 &&
             gate_shape[0] == batch_size &&
             gate_shape[1] == channels
        raise SparseTensorError.new(
          "sparse gated residual gate_msa shape must be [#{batch_size}, #{channels}]"
        )
      end
      gate_values = gate_data.not_nil!
      unless gate_values.size == gate_shape.numel
        raise SparseTensorError.new(
          "sparse gated residual gate_msa storage size must match [B, C]"
        )
      end

      output_budget = Math.min(
        residual.@max_feature_bytes,
        attention_output.@max_feature_bytes
      )
      output_bytes = expected_elements * 4_i64
      if output_bytes > output_budget
        raise SparseTensorBudgetError.new(
          "sparse gated residual output features require #{output_bytes} bytes, limit is #{output_budget}"
        )
      end

      gate_values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse gated residual gate_msa[#{index}] must be finite"
          )
        end
      end

      output_features = Array(Float32).new(expected_elements.to_i)
      point_count.times do |row|
        batch = CoordinateMap3D.kernel_batch_index(residual_map, row)
        unless 0 <= batch < batch_size
          raise SparseTensorError.new(
            "sparse gated residual batch map entry #{batch} is outside 0...#{batch_size}"
          )
        end
        feature_offset = row * channels
        gate_offset = batch * channels
        channels.times do |channel|
          scaled = attention_output.@features[feature_offset + channel] *
                   gate_values[gate_offset + channel]
          value = residual.@features[feature_offset + channel] + scaled
          unless value.finite?
            raise SparseTensorError.new(
              "sparse gated residual output[#{output_features.size}] must be finite"
            )
          end
          output_features << value
        end
      end

      TensorCPU.from_owned_features(
        output_features,
        residual_map,
        point_count,
        channels,
        output_budget,
        residual.@carrier_role
      )
    end
  end
end
