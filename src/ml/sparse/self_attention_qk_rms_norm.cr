require "./self_attention_qkv"

module ML::Sparse
  class TensorCPU
    SELF_ATTENTION_QK_RMS_NORM_EPSILON = 1e-12_f32

    # Normalizes packed Q and K independently per row/head and leaves V exact.
    # This is the graphless CPU/F32 seam used before sparse self-attention, not
    # the conventional mean-square RMSNorm operation.
    def self.apply_self_attention_qk_rms_norm(
      qkv : SelfAttentionQKVCPU,
      q_gamma : ML::Tensor,
      k_gamma : ML::Tensor,
    ) : SelfAttentionQKVCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization requires the base TensorCPU receiver"
        )
      end
      unless qkv.class == SelfAttentionQKVCPU
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization requires a base SelfAttentionQKVCPU"
        )
      end

      flat = qkv.flat_projection
      unless flat.class == TensorCPU && flat.@initialized
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization requires an initialized base TensorCPU projection"
        )
      end
      point_count = qkv.point_count
      num_heads = qkv.num_heads
      head_dim = qkv.head_dim
      channels = qkv.channels
      projected_channels = channels * 3
      unless num_heads > 0 && head_dim > 0 &&
             num_heads * head_dim == channels &&
             flat.@point_count == point_count &&
             flat.@channels == projected_channels
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization requires exact packed [N, 3, H, D] dimensions"
        )
      end

      output_elements = point_count.to_i64 * projected_channels.to_i64
      unless flat.@features.size.to_i64 == output_elements
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization storage size must match [N, 3C]"
        )
      end
      output_bytes = output_elements * 4_i64
      if output_bytes > flat.@max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse self-attention Q/K RMS normalization output requires #{output_bytes} bytes, limit is #{flat.@max_feature_bytes}"
        )
      end

      q_values = self_attention_qk_gamma_values(
        "q_gamma",
        q_gamma,
        num_heads,
        head_dim
      )
      k_values = self_attention_qk_gamma_values(
        "k_gamma",
        k_gamma,
        num_heads,
        head_dim
      )
      scale = Math.sqrt(head_dim.to_f32)

      # Prove every result finite before allocating the fresh packed payload.
      normalize_self_attention_qk_component!(
        flat.@features,
        nil,
        point_count,
        projected_channels,
        num_heads,
        head_dim,
        0_i32,
        q_values,
        scale,
        "Q"
      )
      normalize_self_attention_qk_component!(
        flat.@features,
        nil,
        point_count,
        projected_channels,
        num_heads,
        head_dim,
        channels,
        k_values,
        scale,
        "K"
      )

      output = flat.@features.dup
      normalize_self_attention_qk_component!(
        flat.@features,
        output,
        point_count,
        projected_channels,
        num_heads,
        head_dim,
        0_i32,
        q_values,
        scale,
        "Q"
      )
      normalize_self_attention_qk_component!(
        flat.@features,
        output,
        point_count,
        projected_channels,
        num_heads,
        head_dim,
        channels,
        k_values,
        scale,
        "K"
      )

      SelfAttentionQKVCPU.new(
        TensorCPU.from_owned_features(
          output,
          flat.@coordinate_map,
          point_count,
          projected_channels,
          flat.@max_feature_bytes,
          flat.@carrier_role
        ),
        num_heads
      )
    end

    private def self.self_attention_qk_gamma_values(
      name : String,
      parameter : ML::Tensor,
      num_heads : Int32,
      head_dim : Int32,
    ) : ML::Tensor::CPUReadView
      unless parameter.on_cpu?
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization #{name} must be on CPU"
        )
      end
      unless parameter.dtype.f32?
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization #{name} must use F32"
        )
      end
      unless parameter.contiguous?
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization #{name} must be contiguous"
        )
      end
      unless parameter.ndim == 2 &&
             parameter.shape[0] == num_heads &&
             parameter.shape[1] == head_dim
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization #{name} shape must be [#{num_heads}, #{head_dim}]"
        )
      end

      values = parameter.cpu_read
      unless values.borrowed? && values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse self-attention Q/K RMS normalization #{name} read must not materialize storage"
        )
      end
      values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse self-attention Q/K RMS normalization #{name}[#{index}] must be finite"
          )
        end
      end
      values
    end

    private def self.normalize_self_attention_qk_component!(
      source : Array(Float32),
      output : Array(Float32)?,
      point_count : Int32,
      projected_channels : Int32,
      num_heads : Int32,
      head_dim : Int32,
      component_offset : Int32,
      gamma : ML::Tensor::CPUReadView,
      scale : Float32,
      label : String,
    ) : Nil
      point_count.times do |row|
        row_offset = row * projected_channels + component_offset
        num_heads.times do |head|
          head_offset = row_offset + head * head_dim
          gamma_offset = head * head_dim
          sum_sq = 0.0_f32
          head_dim.times do |channel|
            value = source[head_offset + channel]
            sum_sq += value * value
          end
          norm = Math.sqrt(sum_sq)
          norm = SELF_ATTENTION_QK_RMS_NORM_EPSILON if norm < SELF_ATTENTION_QK_RMS_NORM_EPSILON
          head_dim.times do |channel|
            index = head_offset + channel
            value = source[index] / norm * gamma[gamma_offset + channel] * scale
            unless value.finite?
              raise SparseTensorError.new(
                "sparse self-attention normalized #{label} output at row #{row}, head #{head}, channel #{channel} must be finite"
              )
            end
            output.try { |destination| destination[index] = value }
          end
        end
      end
    end
  end
end
