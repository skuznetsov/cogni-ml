require "./full_self_attention_plan"

module ML::Sparse
  class TensorCPU
    # Executes bounded full self-attention independently inside each canonical
    # sparse batch slice. This is a graphless CPU/F32 mathematical reference,
    # not a parity claim for the upstream xFormers/FlashAttention backends.
    #
    # The planner admits the logical H * sum(L_b^2) score volume, while the
    # executor materializes only one reusable score row of max(L_b) elements.
    def self.apply_full_self_attention(
      qkv : SelfAttentionQKVCPU,
      max_score_bytes : Int64 = FullSelfAttentionPlanCPU::MAX_SCORE_BYTES,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse full-attention requires the base TensorCPU receiver"
        )
      end

      # Admission must happen before numerical work or output allocation.
      plan = FullSelfAttentionPlanCPU.build(qkv, max_score_bytes)
      flat = qkv.flat_projection
      unless flat.@initialized
        raise SparseTensorError.new(
          "sparse full-attention requires an initialized QKV projection"
        )
      end

      channels = plan.num_heads * plan.head_dim
      projected_channels = channels * 3
      unless flat.@point_count == plan.point_count &&
             flat.@channels == projected_channels
        raise SparseTensorError.new(
          "sparse full-attention QKV storage shape must match [N, 3C]"
        )
      end
      expected_storage = plan.point_count.to_i64 * projected_channels.to_i64
      unless flat.@features.size.to_i64 == expected_storage
        raise SparseTensorError.new(
          "sparse full-attention QKV storage size must match [N, 3C]"
        )
      end

      output = Array(Float32).new(plan.output_elements.to_i, 0.0_f32)
      score_row = Array(Float32).new(plan.max_batch_length, 0.0_f32)
      scale = 1.0_f32 / Math.sqrt(plan.head_dim.to_f32)

      plan.batch_size.times do |batch|
        batch_slice = CoordinateMap3D.kernel_batch_slice(
          flat.@coordinate_map,
          batch
        )
        length = batch_slice.size
        next if length == 0

        plan.num_heads.times do |head|
          head_offset = head * plan.head_dim
          length.times do |query_index|
            query_row = batch_slice.start + query_index
            query_base = query_row * projected_channels + head_offset

            row_max = -Float32::INFINITY
            length.times do |key_index|
              key_row = batch_slice.start + key_index
              key_base = key_row * projected_channels + channels + head_offset
              score = 0.0_f32
              plan.head_dim.times do |channel|
                score += flat.@features[query_base + channel] *
                         flat.@features[key_base + channel]
              end
              score *= scale
              unless score.finite?
                raise SparseTensorError.new(
                  "sparse full-attention score must be finite"
                )
              end
              score_row[key_index] = score
              row_max = score if score > row_max
            end

            normalizer = 0.0_f32
            length.times do |key_index|
              weight = Math.exp(score_row[key_index] - row_max)
              unless weight.finite?
                raise SparseTensorError.new(
                  "sparse full-attention exponent must be finite"
                )
              end
              score_row[key_index] = weight
              normalizer += weight
            end
            unless normalizer.finite? && normalizer > 0.0_f32
              raise SparseTensorError.new(
                "sparse full-attention normalizer must be finite and positive"
              )
            end

            length.times do |key_index|
              weight = score_row[key_index] / normalizer
              unless weight.finite?
                raise SparseTensorError.new(
                  "sparse full-attention weight must be finite"
                )
              end
              score_row[key_index] = weight
            end

            output_base = query_row * channels + head_offset
            plan.head_dim.times do |channel|
              value = 0.0_f32
              length.times do |key_index|
                value_row = batch_slice.start + key_index
                value_base = value_row * projected_channels +
                             2 * channels + head_offset
                value += score_row[key_index] *
                         flat.@features[value_base + channel]
              end
              unless value.finite?
                raise SparseTensorError.new(
                  "sparse full-attention output must be finite"
                )
              end
              output[output_base + channel] = value
            end
          end
        end
      end

      TensorCPU.from_owned_features(
        output,
        flat.@coordinate_map,
        plan.point_count,
        channels,
        flat.@max_feature_bytes,
        attention_output_role(flat.@carrier_role)
      )
    end
  end
end
