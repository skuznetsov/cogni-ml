require "./tensor"

module ML::Sparse
  class TensorCPU
    # Concatenates exactly two aligned sparse values along their feature axis.
    # The immutable coordinate map is reused; only one new feature buffer is
    # allocated after all output bounds have passed.
    def self.concat_features(left : TensorCPU, right : TensorCPU) : TensorCPU
      # `new` below must resolve only to TensorCPU's validated ownership
      # constructor, never to an inherited subclass overload.
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse feature concat requires the base TensorCPU receiver"
        )
      end

      # Read the base value's owned state directly. Crystal permits subclasses
      # to override even `class` and ordinary getters, so those are not an
      # authority boundary for invariants established by TensorCPU itself.
      unless left.@initialized && right.@initialized
        raise SparseTensorError.new(
          "sparse feature concat requires initialized sparse values"
        )
      end
      left_limit = standard_carrier_channel_limit(
        left,
        "sparse feature concat"
      )
      right_limit = standard_carrier_channel_limit(
        right,
        "sparse feature concat"
      )
      unless left.@carrier_role == right.@carrier_role
        raise SparseTensorError.new(
          "sparse feature concat requires the same carrier role"
        )
      end
      unless left_limit == right_limit
        raise SparseTensorError.new(
          "sparse feature concat carrier limits must match"
        )
      end
      left_map = left.@coordinate_map
      right_map = right.@coordinate_map
      # Box is a non-allocating pointer cast for references; pointer equality
      # preserves `same?` identity without virtual dispatch through subclasses.
      unless Box(CoordinateMap3D).box(left_map) == Box(CoordinateMap3D).box(right_map)
        raise SparseTensorError.new(
          "sparse feature concat requires the same immutable coordinate map object"
        )
      end

      left_channels = left.@channels
      right_channels = right.@channels
      output_channels_i64 = left_channels.to_i64 + right_channels.to_i64
      if output_channels_i64 > left_limit
        raise SparseTensorBudgetError.new(
          "sparse feature concat output channel count #{output_channels_i64} exceeds #{left_limit}"
        )
      end

      # A derived value must not silently widen either caller-selected budget.
      output_budget = Math.min(left.@max_feature_bytes, right.@max_feature_bytes)
      point_count = left.@point_count
      output_bytes = point_count.to_i64 * output_channels_i64 * 4_i64
      if output_bytes > output_budget
        raise SparseTensorBudgetError.new(
          "sparse feature concat output features require #{output_bytes} bytes, limit is #{output_budget}"
        )
      end

      output_channels = output_channels_i64.to_i32
      output_elements = point_count.to_i64 * output_channels_i64
      output_features = Array(Float32).new(output_elements.to_i, 0.0_f32)
      point_count.times do |row|
        output_offset = row * output_channels
        left_channels.times do |channel|
          output_features[output_offset + channel] =
            left.@features[row * left_channels + channel]
        end
        right_channels.times do |channel|
          output_features[output_offset + left_channels + channel] =
            right.@features[row * right_channels + channel]
        end
      end

      TensorCPU.from_owned_features(
        output_features,
        left_map,
        point_count,
        output_channels,
        output_budget,
        left.@carrier_role
      )
    end
  end
end
