require "./self_attention_qkv"

module ML::Sparse
  class TensorCPU
    SELF_ATTENTION_ROPE_SPATIAL_DIM = 3_i32

    # Applies the pinned TRELLIS.2 adjacent-real-pair 3D RoPE to packed Q and K.
    # V, row order, the coordinate-map object, and the inherited byte budget
    # remain unchanged. This graphless CPU/F32 leaf intentionally has no phase
    # cache or phase-plane allocation.
    def self.apply_self_attention_rope(
      qkv : SelfAttentionQKVCPU,
      rope_low : Float32 = 1.0_f32,
      rope_high : Float32 = 10000.0_f32,
    ) : SelfAttentionQKVCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse self-attention RoPE requires the base TensorCPU receiver"
        )
      end
      unless qkv.class == SelfAttentionQKVCPU
        raise SparseTensorError.new(
          "sparse self-attention RoPE requires a base SelfAttentionQKVCPU"
        )
      end

      flat = qkv.flat_projection
      unless flat.class == TensorCPU && flat.@initialized
        raise SparseTensorError.new(
          "sparse self-attention RoPE requires an initialized base TensorCPU projection"
        )
      end
      unless rope_low.finite? && rope_low > 0.0_f32 &&
             rope_high.finite? && rope_high > 0.0_f32
        raise SparseTensorError.new(
          "sparse self-attention RoPE frequencies must be finite and positive"
        )
      end

      point_count = qkv.point_count
      num_heads = qkv.num_heads
      head_dim = qkv.head_dim
      channels = qkv.channels
      projected_channels = channels * 3
      unless head_dim > 0 && head_dim.even?
        raise SparseTensorError.new(
          "sparse self-attention RoPE head dimension must be positive and even"
        )
      end
      unless num_heads > 0 && num_heads * head_dim == channels &&
             flat.@point_count == point_count &&
             flat.@channels == projected_channels
        raise SparseTensorError.new(
          "sparse self-attention RoPE requires exact packed [N, 3, H, D] dimensions"
        )
      end

      output_elements = point_count.to_i64 * projected_channels.to_i64
      unless flat.@features.size.to_i64 == output_elements
        raise SparseTensorError.new(
          "sparse self-attention RoPE storage size must match [N, 3C]"
        )
      end
      output_bytes = output_elements * 4_i64
      if output_bytes > flat.@max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse self-attention RoPE output requires #{output_bytes} bytes, limit is #{flat.@max_feature_bytes}"
        )
      end

      frequency_dim = head_dim // 2 // SELF_ATTENTION_ROPE_SPATIAL_DIM
      frequencies = Array(Float32).new(frequency_dim, 0.0_f32)
      frequency_dim.times do |frequency_index|
        exponent = frequency_index.to_f64 / frequency_dim.to_f64
        frequency = rope_low.to_f64 / (rope_high.to_f64 ** exponent)
        unless frequency.finite? && frequency > 0.0 && frequency <= Float32::MAX.to_f64
          raise SparseTensorError.new(
            "sparse self-attention RoPE frequency #{frequency_index} must be representable in F32"
          )
        end
        frequencies[frequency_index] = frequency.to_f32
      end

      # D=2 and D=4 deliberately follow upstream's all-identity floor behavior
      # because frequency_dim is zero. The output is private until every active
      # rotation passes its finite checks, so failure cannot mutate the input or
      # expose a partial result.
      output = flat.@features.dup
      rotate_self_attention_qk!(
        flat.@features,
        output,
        flat.@coordinate_map,
        point_count,
        projected_channels,
        channels,
        num_heads,
        head_dim,
        frequencies
      )

      SelfAttentionQKVCPU.new(
        new(
          output,
          flat.@coordinate_map,
          point_count,
          projected_channels,
          flat.@max_feature_bytes
        ),
        num_heads
      )
    end

    private def self.rotate_self_attention_qk!(
      source : Array(Float32),
      output : Array(Float32),
      coordinate_map : CoordinateMap3D,
      point_count : Int32,
      projected_channels : Int32,
      channels : Int32,
      num_heads : Int32,
      head_dim : Int32,
      frequencies : Array(Float32),
    ) : Nil
      frequency_dim = frequencies.size.to_i32
      point_count.times do |row|
        SELF_ATTENTION_ROPE_SPATIAL_DIM.times do |axis|
          coordinate = CoordinateMap3D.kernel_spatial_coordinate(
            coordinate_map,
            row,
            axis
          ).to_f32
          frequency_dim.times do |frequency_index|
            pair = axis * frequency_dim + frequency_index
            angle = coordinate * frequencies[frequency_index]
            unless angle.finite?
              raise SparseTensorError.new(
                "sparse self-attention RoPE angle at row #{row}, axis #{axis}, frequency #{frequency_index} must be finite"
              )
            end
            cosine = Math.cos(angle.to_f64).to_f32
            sine = Math.sin(angle.to_f64).to_f32
            unless cosine.finite? && sine.finite?
              raise SparseTensorError.new(
                "sparse self-attention RoPE phase at row #{row}, axis #{axis}, frequency #{frequency_index} must be finite"
              )
            end

            2.times do |component|
              component_offset = row * projected_channels + component * channels
              num_heads.times do |head|
                pair_offset = component_offset + head * head_dim + pair * 2
                real = source[pair_offset]
                imaginary = source[pair_offset + 1]
                rotated_real = real * cosine - imaginary * sine
                rotated_imaginary = imaginary * cosine + real * sine
                unless rotated_real.finite? && rotated_imaginary.finite?
                  raise SparseTensorError.new(
                    "sparse self-attention RoPE output at row #{row}, component #{component}, head #{head}, pair #{pair} must be finite"
                  )
                end
                output[pair_offset] = rotated_real
                output[pair_offset + 1] = rotated_imaginary
              end
            end
          end
        end
      end
    end
  end
end
