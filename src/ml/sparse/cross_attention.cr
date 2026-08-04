require "./cross_attention_output"

module ML::Sparse
  class TensorCPU
    # Executes the source-pinned normalized full cross-attention path by
    # composing the admitted graphless CPU/F32 leaves. The projection leaf
    # owns preflight, so this wrapper neither duplicates attention arithmetic
    # nor constructs a second plan or padded query.
    #
    # The three public limits retain their plan-wide meanings. Intermediate
    # owners may overlap until this synchronous call returns, so the limits do
    # not claim an aggregate live-memory, RSS, native-allocation, or peak bound.
    def self.apply_cross_attention(
      query : TensorCPU,
      context : ML::Tensor,
      num_heads : Int32,
      to_q : ML::NN::Linear,
      to_kv : ML::NN::Linear,
      q_gamma : ML::Tensor,
      k_gamma : ML::Tensor,
      to_out : ML::NN::Linear,
      max_score_bytes : Int64 = CrossAttentionPlanCPU::MAX_SCORE_BYTES,
      max_projection_bytes : Int64 = CrossAttentionPlanCPU::MAX_PROJECTION_BYTES,
      max_total_work_elements : Int64 = CrossAttentionPlanCPU::MAX_WORK_ELEMENTS,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse cross-attention requires the base TensorCPU receiver"
        )
      end

      projection = CrossAttentionProjectionCPU.project(
        query,
        context,
        num_heads,
        to_q,
        to_kv,
        max_score_bytes: max_score_bytes,
        max_projection_bytes: max_projection_bytes,
        max_work_elements: max_total_work_elements
      )
      normalized = CrossAttentionQKNormalizedCPU.normalize(
        projection,
        q_gamma,
        k_gamma,
        max_normalized_bytes: max_projection_bytes
      )
      pre_output = TensorCPU.apply_cross_attention_pre_output(
        normalized,
        max_score_bytes: max_score_bytes,
        max_work_elements: max_total_work_elements
      )
      TensorCPU.apply_cross_attention_output(
        pre_output,
        projection.plan,
        to_out,
        max_total_work_elements: max_total_work_elements
      )
    end
  end
end
