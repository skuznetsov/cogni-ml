require "./tensor"

module ML::Sparse
  class TensorCPU
    # Applies the plain sparse residual used after TRELLIS.2 cross-attention:
    # x + h. Inputs are borrowed, and one owned output payload is allocated only
    # after carrier, map, shape, storage, and byte-ceiling checks pass.
    def self.apply_residual(
      residual : TensorCPU,
      update : TensorCPU,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse residual requires the base TensorCPU receiver"
        )
      end
      unless residual.@initialized && update.@initialized
        raise SparseTensorError.new(
          "sparse residual requires initialized sparse values"
        )
      end

      residual_limit = standard_carrier_channel_limit(
        residual,
        "sparse residual"
      )
      update_limit = standard_carrier_channel_limit(update, "sparse residual")
      unless residual.@carrier_role == update.@carrier_role &&
             residual_limit == update_limit
        raise SparseTensorError.new(
          "sparse residual requires the same carrier role"
        )
      end

      residual_map = residual.@coordinate_map
      update_map = update.@coordinate_map
      unless Box(CoordinateMap3D).box(residual_map) ==
               Box(CoordinateMap3D).box(update_map)
        raise SparseTensorError.new(
          "sparse residual requires the same immutable coordinate map object"
        )
      end

      _, map_point_count = CoordinateMap3D.kernel_layout(residual_map)
      unless residual.@point_count == map_point_count &&
             update.@point_count == map_point_count
        raise SparseTensorError.new(
          "sparse residual requires coordinate and feature point counts to match"
        )
      end

      point_count = residual.@point_count
      channels = residual.@channels
      unless point_count == update.@point_count && channels == update.@channels
        raise SparseTensorError.new(
          "sparse residual requires the same row and channel shape"
        )
      end
      unless 1 <= channels <= residual_limit
        raise SparseTensorError.new(
          "sparse residual channel count must be in 1..#{residual_limit}"
        )
      end
      unless 0_i64 < residual.@max_feature_bytes <= MAX_FEATURE_BYTES &&
             0_i64 < update.@max_feature_bytes <= MAX_FEATURE_BYTES
        raise SparseTensorError.new(
          "sparse residual feature byte budgets must be in 1..#{MAX_FEATURE_BYTES}"
        )
      end

      expected_elements = point_count.to_i64 * channels.to_i64
      unless residual.@features.size.to_i64 == expected_elements &&
             update.@features.size.to_i64 == expected_elements
        raise SparseTensorError.new(
          "sparse residual feature storage size must match [N, C]"
        )
      end

      output_budget = Math.min(
        residual.@max_feature_bytes,
        update.@max_feature_bytes
      )
      output_bytes = expected_elements * 4_i64
      if output_bytes > output_budget
        raise SparseTensorBudgetError.new(
          "sparse residual output features require #{output_bytes} bytes, limit is #{output_budget}"
        )
      end

      output_features = Array(Float32).new(expected_elements.to_i)
      expected_elements.to_i.times do |index|
        value = residual.@features[index] + update.@features[index]
        unless value.finite?
          raise SparseTensorError.new(
            "sparse residual output[#{index}] must be finite"
          )
        end
        output_features << value
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
