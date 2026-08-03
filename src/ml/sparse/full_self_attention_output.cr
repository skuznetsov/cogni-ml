require "./full_self_attention_pre_output"
require "./linear"

module ML::Sparse
  class TensorCPU
    # Executes the admitted self/full attention path through the exact upstream
    # frozen biased C-to-C output projection. This remains a bounded graphless
    # CPU/F32 reference, not a generic transformer block or GPU parity claim.
    def self.apply_full_self_attention_output(
      input : TensorCPU,
      to_qkv : ML::NN::Linear,
      to_out : ML::NN::Linear,
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
          "sparse full self-attention output requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "sparse full self-attention output requires an initialized sparse value"
        )
      end

      channels = input.@channels
      unless to_out.in_features == channels && to_out.out_features == channels
        raise SparseTensorError.new(
          "sparse full self-attention output requires to_out Linear(#{channels}, #{channels})"
        )
      end
      bias = to_out.bias
      unless bias
        raise SparseTensorError.new(
          "sparse full self-attention output requires a biased to_out"
        )
      end
      if to_out.weight.requires_grad? || bias.requires_grad?
        raise SparseTensorError.new(
          "sparse full self-attention output requires frozen graphless to_out parameters"
        )
      end

      # Score admission and final structural metadata are checked early.
      # Storage and finiteness remain owned by apply_linear after the bounded
      # context exists, but before that leaf allocates the final feature payload.
      context = apply_full_self_attention_pre_output(
        input,
        to_qkv,
        num_heads,
        q_gamma: q_gamma,
        k_gamma: k_gamma,
        use_rope: use_rope,
        rope_low: rope_low,
        rope_high: rope_high,
        max_score_bytes: max_score_bytes
      )
      apply_linear(context, to_out)
    end
  end
end
