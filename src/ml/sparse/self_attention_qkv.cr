require "./linear"

module ML::Sparse
  # Immutable logical [N, 3, H, D] view over one flat [N, 3C]
  # self-attention projection. The view retains the flat TensorCPU and does not
  # copy or expose its owned feature buffer.
  class SelfAttentionQKVCPU
    getter flat_projection : TensorCPU
    getter num_heads : Int32
    getter head_dim : Int32
    getter channels : Int32

    def initialize(@flat_projection : TensorCPU, @num_heads : Int32)
      unless @flat_projection.class == TensorCPU
        raise SparseTensorError.new(
          "sparse self-attention QKV view requires a base TensorCPU projection"
        )
      end
      TensorCPU.validate_self_attention_qkv_carrier!(
        @flat_projection,
        "sparse self-attention QKV view"
      )
      unless @num_heads > 0
        raise SparseTensorError.new(
          "sparse self-attention QKV head count must be positive"
        )
      end

      projected_channels = @flat_projection.channels
      unless projected_channels >= 3 && projected_channels % 3 == 0
        raise SparseTensorError.new(
          "sparse self-attention QKV projected channels #{projected_channels} must be divisible by three"
        )
      end
      @channels = projected_channels // 3
      unless @channels % @num_heads == 0
        raise SparseTensorError.new(
          "sparse self-attention QKV channels #{@channels} must be divisible by heads #{@num_heads}"
        )
      end
      @head_dim = @channels // @num_heads
    end

    def coordinate_map : CoordinateMap3D
      @flat_projection.coordinate_map
    end

    def point_count : Int32
      @flat_projection.point_count
    end

    def max_feature_bytes : Int64
      @flat_projection.max_feature_bytes
    end

    def shape : Tuple(Int32, Int32, Int32, Int32)
      {@flat_projection.shape[0], 3_i32, @num_heads, @head_dim}
    end

    def feature_shape : Tuple(Int32, Int32, Int32, Int32)
      {point_count, 3_i32, @num_heads, @head_dim}
    end

    def feature(
      row : Int32,
      component : Int32,
      head : Int32,
      channel : Int32,
    ) : Float32
      unless 0 <= row < point_count
        raise IndexError.new("sparse self-attention QKV row #{row} is out of bounds")
      end
      unless 0 <= component < 3
        raise IndexError.new(
          "sparse self-attention QKV component #{component} is out of bounds"
        )
      end
      unless 0 <= head < @num_heads
        raise IndexError.new("sparse self-attention QKV head #{head} is out of bounds")
      end
      unless 0 <= channel < @head_dim
        raise IndexError.new(
          "sparse self-attention QKV channel #{channel} is out of bounds"
        )
      end

      flat_channel = component * @channels + head * @head_dim + channel
      @flat_projection.feature(row, flat_channel)
    end

    def features_copy : Array(Float32)
      @flat_projection.features_copy
    end
  end

  class TensorCPU
    # Executes only the frozen self-attention to_qkv projection and wraps its
    # flat output in a zero-copy logical [N, 3, H, D] view. Reusing the bounded
    # sparse linear leaf keeps the standard input ceiling while admitting only
    # this role-specific packed output up to 3C <= 4608.
    def self.apply_self_attention_qkv(
      input : TensorCPU,
      linear : ML::NN::Linear,
      num_heads : Int32,
    ) : SelfAttentionQKVCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse self-attention QKV requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "sparse self-attention QKV requires an initialized sparse value"
        )
      end
      unless num_heads > 0
        raise SparseTensorError.new(
          "sparse self-attention QKV head count must be positive"
        )
      end

      input_channels = input.@channels
      standard_limit = standard_carrier_channel_limit(
        input,
        "sparse self-attention QKV"
      )
      linear_input_channels = linear.in_features
      unless input_channels == linear_input_channels
        raise SparseTensorError.new(
          "sparse self-attention QKV input channels #{input_channels} do not match Linear in_features #{linear_input_channels}"
        )
      end
      unless input_channels % num_heads == 0
        raise SparseTensorError.new(
          "sparse self-attention QKV channels #{input_channels} must be divisible by heads #{num_heads}"
        )
      end

      expected_output_channels = input_channels * 3
      packed_role = packed_qkv_role(input.@carrier_role)
      packed_limit = if standard_limit == MAX_CHANNELS
                       MAX_CHANNELS
                     else
                       MAX_SELF_ATTENTION_QKV_CHANNELS
                     end
      output_channels = linear.out_features
      unless output_channels == expected_output_channels
        raise SparseTensorError.new(
          "sparse self-attention QKV Linear out_features #{output_channels} do not match expected #{expected_output_channels}"
        )
      end
      if expected_output_channels > packed_limit
        raise SparseTensorBudgetError.new(
          "bounded sparse self-attention QKV requires 3C <= #{packed_limit}, got #{expected_output_channels}"
        )
      end

      flat_projection = apply_linear_bounded(
        input,
        linear,
        packed_limit,
        packed_role
      )
      SelfAttentionQKVCPU.new(flat_projection, num_heads)
    end
  end
end
