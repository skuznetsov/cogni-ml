require "./self_attention_qkv"
require "./self_attention_qk_rms_norm"
require "./self_attention_rope"
require "./full_self_attention"

module ML::Sparse
  class TensorCPU
    # Executes the admitted SparseMultiHeadAttention self/full path through the
    # context immediately before its output projection. This deliberately is a
    # thin composition of bounded CPU/F32 leaves, not a generic attention graph
    # and not parity evidence for upstream sparse GPU backends.
    def self.apply_full_self_attention_pre_output(
      input : TensorCPU,
      to_qkv : ML::NN::Linear,
      num_heads : Int32,
      q_gamma : ML::Tensor? = nil,
      k_gamma : ML::Tensor? = nil,
      use_rope : Bool = false,
      rope_low : Float32 = 1.0_f32,
      rope_high : Float32 = 10000.0_f32,
      max_score_bytes : Int64 = FullSelfAttentionPlanCPU::MAX_SCORE_BYTES,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse full self-attention pre-output requires the base TensorCPU receiver"
        )
      end
      unless q_gamma.nil? == k_gamma.nil?
        raise SparseTensorError.new(
          "sparse full self-attention pre-output q_gamma and k_gamma must be provided together"
        )
      end

      # Reject the quadratic score/output budget before any composed leaf
      # allocates a projected or transformed feature payload.
      FullSelfAttentionPlanCPU.preflight(input, num_heads, max_score_bytes)
      qkv = apply_self_attention_qkv(input, to_qkv, num_heads)
      if q = q_gamma
        qkv = apply_self_attention_qk_rms_norm(qkv, q, k_gamma.not_nil!)
      end
      if use_rope
        qkv = apply_self_attention_rope(qkv, rope_low, rope_high)
      end
      apply_full_self_attention(qkv, max_score_bytes)
    end
  end
end
