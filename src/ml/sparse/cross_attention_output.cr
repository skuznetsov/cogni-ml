require "./cross_attention_pre_output"
require "./linear"

module ML::Sparse
  class TensorCPU
    # Applies only the source-owned frozen biased C-to-C output projection to
    # an already admitted cross-attention pre-output. The retained plan checks
    # that otherwise plain TensorCPU carrier against its map, geometry, role,
    # byte ceiling, and total cross-attention work admission.
    #
    # This remains a synchronous graphless CPU/F32 reference. It does not
    # rerun score arithmetic, assemble a cross-attention block, or claim an
    # aggregate live-memory bound or authenticate how the carrier was produced.
    def self.apply_cross_attention_output(
      pre_output : TensorCPU,
      plan : CrossAttentionPlanCPU,
      to_out : ML::NN::Linear,
      max_output_bytes : Int64? = nil,
      max_total_work_elements : Int64? = nil,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse cross-attention output requires the base TensorCPU receiver"
        )
      end
      unless pre_output.class == TensorCPU
        raise SparseTensorError.new(
          "sparse cross-attention output requires a base TensorCPU pre-output"
        )
      end
      unless plan.class == CrossAttentionPlanCPU
        raise SparseTensorError.new(
          "sparse cross-attention output requires a base CrossAttentionPlanCPU"
        )
      end

      standard_carrier_channel_limit(
        pre_output,
        "sparse cross-attention output"
      )
      validate_cross_attention_output_contract!(pre_output, plan)

      output_budget = max_output_bytes || plan.query_max_feature_bytes
      unless 1_i64 <= output_budget <= plan.query_max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse cross-attention output byte budget must be in 1..#{plan.query_max_feature_bytes}"
        )
      end
      if plan.output_bytes > output_budget
        raise SparseTensorBudgetError.new(
          "sparse cross-attention output would require #{plan.output_bytes} bytes, limit is #{output_budget}"
        )
      end

      work_budget = max_total_work_elements || plan.max_work_elements
      unless 1_i64 <= work_budget <= plan.max_work_elements
        raise SparseTensorBudgetError.new(
          "sparse cross-attention total-work budget must be in 1..#{plan.max_work_elements}"
        )
      end
      if plan.work_elements > work_budget
        raise SparseTensorBudgetError.new(
          "sparse cross-attention total work would require #{plan.work_elements} MAC elements, limit is #{work_budget}"
        )
      end

      channels = pre_output.@channels
      unless to_out.in_features == channels && to_out.out_features == channels
        raise SparseTensorError.new(
          "sparse cross-attention output requires to_out Linear(#{channels}, #{channels})"
        )
      end
      bias = to_out.bias
      unless bias
        raise SparseTensorError.new(
          "sparse cross-attention output requires a biased to_out"
        )
      end
      if to_out.weight.requires_grad? || bias.requires_grad?
        raise SparseTensorError.new(
          "sparse cross-attention output requires frozen graphless to_out parameters"
        )
      end

      # SparseLinear owns CPU/F32/contiguous/finiteness validation and performs
      # the exact scalar-F32 affine loop before returning a fresh owned carrier.
      apply_linear(pre_output, to_out)
    end

    private def self.validate_cross_attention_output_contract!(
      pre_output : TensorCPU,
      plan : CrossAttentionPlanCPU,
    ) : Nil
      output_elements = pre_output.@point_count.to_i64 *
                        pre_output.@channels.to_i64
      output_bytes = output_elements * 4_i64
      output_work = output_elements * pre_output.@channels.to_i64
      unless pre_output.@coordinate_map.same?(plan.coordinate_map) &&
             pre_output.@point_count == plan.point_count &&
             pre_output.@channels == plan.query_channels &&
             pre_output.@max_feature_bytes == plan.query_max_feature_bytes &&
             pre_output.production_width? == plan.query_production_width &&
             plan.output_elements == output_elements &&
             plan.output_bytes == output_bytes &&
             plan.output_projection_mac_elements == output_work
        raise SparseTensorError.new(
          "sparse cross-attention pre-output does not match its plan"
        )
      end
      if plan.output_bytes > pre_output.@max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse cross-attention output would require #{plan.output_bytes} bytes, limit is #{pre_output.@max_feature_bytes}"
        )
      end
    end
  end
end
