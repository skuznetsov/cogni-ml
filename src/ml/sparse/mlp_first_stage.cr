require "./tensor"
require "./linear"

module ML::Sparse
  class TensorCPU
    # TRELLIS.2 SparseFeedForwardNet uses this exact ratio in its production
    # SLat blocks: hidden = int(channels * 5.3334).  This leaf stops after the
    # first SparseLinear and tanh-approximate GELU, before the second projection.
    TRELLIS2_MLP_RATIO               = 5.3334_f64
    TRELLIS2_MLP_MAX_INPUT_CHANNELS  =     48_i32
    TRELLIS2_MLP_MAX_HIDDEN_CHANNELS =    256_i32
    TRELLIS2_MLP_MAX_WORK_ELEMENTS   = 128_i64 * 1024_i64 * 1024_i64
    TRELLIS2_GELU_TANH_COEFFICIENT   =           0.044715_f32
    TRELLIS2_GELU_SQRT_2_OVER_PI     = 0.7978845608028654_f32

    # Applies the first production SparseFeedForwardNet projection followed by
    # SparseGELU(approximate="tanh").  The operation is intentionally a
    # bounded, fused CPU/F32 reference: it borrows the input and frozen linear
    # parameters, allocates one owned hidden feature buffer, and preserves the
    # coordinate map and feature-byte budget.
    def self.apply_mlp_first_projection_gelu(
      input : TensorCPU,
      linear : ML::NN::Linear,
      max_work_elements : Int64 = TRELLIS2_MLP_MAX_WORK_ELEMENTS,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires an initialized sparse value"
        )
      end
      unless input.@carrier_role == CarrierRole::Bounded
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires a bounded carrier"
        )
      end
      unless linear.class == ML::NN::Linear
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires a base ML::NN::Linear"
        )
      end
      unless 1_i64 <= max_work_elements <= TRELLIS2_MLP_MAX_WORK_ELEMENTS
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU work budget must be in 1..#{TRELLIS2_MLP_MAX_WORK_ELEMENTS}"
        )
      end

      coordinate_map = input.@coordinate_map
      unless coordinate_map.class == CoordinateMap3D
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires a base CoordinateMap3D"
        )
      end
      batch_size, map_point_count = CoordinateMap3D.kernel_layout(coordinate_map)
      unless 1 <= batch_size <= CoordinateMap3D::MAX_BATCH_SIZE
        raise SparseTensorError.new(
          "sparse MLP first projection GELU batch size must be in 1..#{CoordinateMap3D::MAX_BATCH_SIZE}"
        )
      end
      unless 0 <= map_point_count <= CoordinateMap3D::MAX_POINTS
        raise SparseTensorError.new(
          "sparse MLP first projection GELU point count must be in 0..#{CoordinateMap3D::MAX_POINTS}"
        )
      end
      unless input.@point_count == map_point_count
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires coordinate and feature point counts to match"
        )
      end

      point_count = input.@point_count
      input_channels = input.@channels
      unless 1 <= input_channels <= TRELLIS2_MLP_MAX_INPUT_CHANNELS
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU input channel count must be in 1..#{TRELLIS2_MLP_MAX_INPUT_CHANNELS}"
        )
      end
      hidden_channels = (input_channels.to_f64 * TRELLIS2_MLP_RATIO).to_i32
      unless 1 <= hidden_channels <= TRELLIS2_MLP_MAX_HIDDEN_CHANNELS
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU hidden channel count must be in 1..#{TRELLIS2_MLP_MAX_HIDDEN_CHANNELS}"
        )
      end
      unless linear.in_features == input_channels
        raise SparseTensorError.new(
          "sparse MLP first projection GELU input channels #{input_channels} do not match Linear in_features #{linear.in_features}"
        )
      end
      unless linear.out_features == hidden_channels
        raise SparseTensorError.new(
          "sparse MLP first projection GELU output channels #{linear.out_features} do not match int(C * 5.3334) #{hidden_channels}"
        )
      end

      unless 0_i64 < input.@max_feature_bytes <= MAX_FEATURE_BYTES
        raise SparseTensorError.new(
          "sparse MLP first projection GELU feature byte budget must be in 1..#{MAX_FEATURE_BYTES}"
        )
      end

      input_elements = checked_multiply(
        point_count.to_i64,
        input_channels.to_i64,
        "input elements"
      )
      hidden_elements = checked_multiply(
        point_count.to_i64,
        hidden_channels.to_i64,
        "hidden elements"
      )
      input_bytes = checked_multiply(input_elements, 4_i64, "input bytes")
      hidden_bytes = checked_multiply(hidden_elements, 4_i64, "hidden bytes")
      work_elements = checked_multiply(
        hidden_elements,
        input_channels.to_i64,
        "projection work"
      )
      unless input.@features.size.to_i64 == input_elements
        raise SparseTensorError.new(
          "sparse MLP first projection GELU input feature storage size must match [N, C]"
        )
      end
      if input_bytes > input.@max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU input features require #{input_bytes} bytes, limit is #{input.@max_feature_bytes}"
        )
      end
      if hidden_bytes > input.@max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU output features require #{hidden_bytes} bytes, limit is #{input.@max_feature_bytes}"
        )
      end
      if work_elements > max_work_elements
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU work would require #{work_elements} MAC elements, limit is #{max_work_elements}"
        )
      end

      # Validate the owned input payload only after all shape, byte, and work
      # bounds. A poisoned payload therefore cannot force parameter reads.
      input.@features.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse MLP first projection GELU input[#{index}] must be finite"
          )
        end
      end

      unless linear.bias
        raise SparseTensorError.new(
          "sparse MLP first projection GELU requires a bias"
        )
      end
      # The existing bounded linear leaf performs the complete frozen
      # CPU/F32/contiguous/shape/borrowed-read validation and owns exactly one
      # fresh output buffer. GELU is then applied in place to that same buffer;
      # no second hidden tensor or generic activation output is allocated.
      output = apply_linear_bounded(
        input,
        linear,
        TRELLIS2_MLP_MAX_HIDDEN_CHANNELS,
        CarrierRole::Bounded
      )
      output.@features.each_with_index do |projected, index|
        cubic = projected * projected * projected
        tanh_argument = TRELLIS2_GELU_SQRT_2_OVER_PI *
                        (projected + TRELLIS2_GELU_TANH_COEFFICIENT * cubic)
        gelu = 0.5_f32 * projected *
               (1.0_f32 + Math.tanh(tanh_argument.to_f64).to_f32)
        unless gelu.finite?
          raise SparseTensorError.new(
            "sparse MLP first projection GELU output[#{index}] must be finite"
          )
        end
        output.@features[index] = gelu
      end
      output
    end

    private def self.checked_multiply(
      left : Int64,
      right : Int64,
      label : String,
    ) : Int64
      if left < 0_i64 || right < 0_i64 ||
         (right > 0_i64 && left > Int64::MAX // right)
        raise SparseTensorBudgetError.new(
          "sparse MLP first projection GELU #{label} overflow Int64"
        )
      end
      left * right
    end
  end
end
