require "./cross_attention_qk_rms_norm"

module ML::Sparse
  class TensorCPU
    # Executes only the pinned full cross-attention score/softmax/value
    # reduction. Q is consumed in flat sparse row order as [N,H,D], K/V are
    # consumed dense and batch-major as [B,L,H,D], and the returned standard
    # sparse carrier stores the same result flat as [N,C]. No padded query or
    # persistent score plane is created; one context-length score row is
    # reused for every query/head. The following to_out projection remains a
    # separate boundary.
    #
    # This is a graphless synchronous CPU/F32 reference, not parity with an
    # xFormers/FlashAttention backend. The normalized carrier is immutable and
    # has already validated source finiteness; arithmetic still fails closed
    # on any score/softmax/output value that becomes non-finite.
    def self.apply_cross_attention_pre_output(
      normalized : CrossAttentionQKNormalizedCPU,
      max_score_bytes : Int64 = CrossAttentionPlanCPU::MAX_SCORE_BYTES,
      max_work_elements : Int64 = CrossAttentionPlanCPU::MAX_WORK_ELEMENTS,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse cross-attention requires the base TensorCPU receiver"
        )
      end
      unless normalized.class == CrossAttentionQKNormalizedCPU
        raise SparseTensorError.new(
          "sparse cross-attention pre-output requires a base CrossAttentionQKNormalizedCPU"
        )
      end

      plan = normalized.plan
      unless plan.class == CrossAttentionPlanCPU
        raise SparseTensorError.new(
          "sparse cross-attention pre-output requires a base CrossAttentionPlanCPU"
        )
      end
      validate_cross_attention_pre_output_budget!(
        plan,
        max_score_bytes,
        max_work_elements
      )
      validate_cross_attention_pre_output_geometry!(normalized, plan)

      channels = plan.num_heads * plan.head_dim
      if plan.point_count > 0 && plan.score_row_bytes > max_score_bytes
        raise SparseTensorBudgetError.new(
          "sparse cross-attention score row would require #{plan.score_row_bytes} bytes, limit is #{max_score_bytes}"
        )
      end

      output_role = attention_output_role(
        plan.query_production_width ? CarrierRole::Production : CarrierRole::Bounded
      )
      output = Array(Float32).new(plan.output_elements.to_i, 0.0_f32)
      # An all-empty sparse query has no score work at all. Avoid even the
      # bounded scratch allocation in that case; this also makes the caller's
      # score-row budget meaningful only when a row can actually be emitted.
      return TensorCPU.from_owned_features(
        output,
        plan.coordinate_map,
        plan.point_count,
        channels,
        plan.query_max_feature_bytes,
        output_role
      ) if plan.point_count == 0

      score_row = Array(Float32).new(plan.score_row_elements.to_i, 0.0_f32)
      scale = 1.0_f32 / Math.sqrt(plan.head_dim.to_f32)

      plan.batch_size.times do |batch|
        batch_slice = plan.query_batch_slice(batch)
        query_length = batch_slice.size
        next if query_length == 0

        plan.num_heads.times do |head|
          head_offset = head * plan.head_dim
          query_length.times do |query_index|
            query_row = batch_slice.start + query_index

            row_max = -Float32::INFINITY
            plan.context_length.times do |token|
              score = 0.0_f32
              plan.head_dim.times do |channel|
                score += normalized.query_feature(query_row, head, channel) *
                         normalized.key_feature(batch, token, head, channel)
              end
              score *= scale
              unless score.finite?
                raise SparseTensorError.new(
                  "sparse cross-attention score must be finite"
                )
              end
              score_row[token] = score
              row_max = score if score > row_max
            end

            normalizer = 0.0_f32
            plan.context_length.times do |token|
              weight = Math.exp(score_row[token] - row_max).to_f32
              unless weight.finite?
                raise SparseTensorError.new(
                  "sparse cross-attention exponent must be finite"
                )
              end
              score_row[token] = weight
              normalizer += weight
            end
            unless normalizer.finite? && normalizer > 0.0_f32
              raise SparseTensorError.new(
                "sparse cross-attention normalizer must be finite and positive"
              )
            end

            plan.context_length.times do |token|
              weight = score_row[token] / normalizer
              unless weight.finite?
                raise SparseTensorError.new(
                  "sparse cross-attention weight must be finite"
                )
              end
              score_row[token] = weight
            end

            output_base = query_row * channels + head_offset
            plan.head_dim.times do |channel|
              value = 0.0_f32
              plan.context_length.times do |token|
                value += score_row[token] *
                         normalized.value_feature(batch, token, head, channel)
              end
              unless value.finite?
                raise SparseTensorError.new(
                  "sparse cross-attention output must be finite"
                )
              end
              output[output_base + channel] = value
            end
          end
        end
      end

      TensorCPU.from_owned_features(
        output,
        plan.coordinate_map,
        plan.point_count,
        channels,
        plan.query_max_feature_bytes,
        attention_output_role(output_role)
      )
    end

    private def self.validate_cross_attention_pre_output_budget!(
      plan : CrossAttentionPlanCPU,
      max_score_bytes : Int64,
      max_work_elements : Int64,
    ) : Nil
      unless 1_i64 <= max_score_bytes <= CrossAttentionPlanCPU::MAX_SCORE_BYTES
        raise SparseTensorBudgetError.new(
          "sparse cross-attention score byte budget must be in 1..#{CrossAttentionPlanCPU::MAX_SCORE_BYTES}"
        )
      end
      unless 1_i64 <= max_work_elements <= CrossAttentionPlanCPU::MAX_WORK_ELEMENTS
        raise SparseTensorBudgetError.new(
          "sparse cross-attention work budget must be in 1..#{CrossAttentionPlanCPU::MAX_WORK_ELEMENTS}"
        )
      end
      if plan.score_bytes > max_score_bytes
        raise SparseTensorBudgetError.new(
          "bounded sparse cross-attention score budget would require #{plan.score_bytes} bytes, limit is #{max_score_bytes}"
        )
      end
      if plan.attention_mac_elements > max_work_elements
        raise SparseTensorBudgetError.new(
          "bounded sparse cross-attention attention work would require #{plan.attention_mac_elements} MAC elements, limit is #{max_work_elements}"
        )
      end
      if plan.output_bytes > plan.query_max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse cross-attention output would require #{plan.output_bytes} bytes, limit is #{plan.query_max_feature_bytes}"
        )
      end
    end

    private def self.validate_cross_attention_pre_output_geometry!(
      normalized : CrossAttentionQKNormalizedCPU,
      plan : CrossAttentionPlanCPU,
    ) : Nil
      unless normalized.coordinate_map.same?(plan.coordinate_map) &&
             normalized.point_count == plan.point_count &&
             normalized.channels == plan.query_channels &&
             normalized.num_heads == plan.num_heads &&
             normalized.head_dim == plan.head_dim &&
             normalized.context_length == plan.context_length &&
             normalized.query_feature_shape == {
               plan.point_count,
               plan.num_heads,
               plan.head_dim,
             } &&
             normalized.key_value_feature_shape == {
               plan.batch_size,
               plan.context_length,
               plan.num_heads,
               plan.head_dim,
             }
        raise SparseTensorError.new(
          "sparse cross-attention normalized carrier geometry does not match its plan"
        )
      end
      unless plan.output_elements == plan.point_count.to_i64 *
                                     plan.num_heads.to_i64 * plan.head_dim.to_i64
        raise SparseTensorError.new(
          "sparse cross-attention output geometry must be flat [N,H,D]"
        )
      end
      unless plan.score_row_elements == plan.context_length.to_i64 &&
             plan.score_row_bytes == plan.score_row_elements * 4_i64
        raise SparseTensorError.new(
          "sparse cross-attention score scratch geometry must be one context-length row"
        )
      end
    end
  end
end
