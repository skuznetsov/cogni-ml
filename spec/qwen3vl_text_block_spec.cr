require "./spec_helper"
require "../src/ml/gguf/qwen3vl_text_block"

private def qwen3vl_text_block_spec_bf16(value : Float32) : Float32
  bits = value.unsafe_as(UInt32)
  rounded = bits + 0x7fff_u32 + ((bits >> 16) & 1_u32)
  (rounded & 0xffff0000_u32).unsafe_as(Float32)
end

private def qwen3vl_text_block_spec_weight(
  out_dim : Int32, in_dim : Int32, phase : Int32,
) : Array(Float32)
  Array(Float32).new(out_dim * in_dim) do |index|
    value = (((index * 7 + phase * 5) % 23) - 11).to_f32 / 13.0_f32
    qwen3vl_text_block_spec_bf16(value)
  end
end

private def qwen3vl_text_block_spec_weights : ML::GGUF::Qwen3VLTextBlockWeights
  ML::GGUF::Qwen3VLTextBlockWeights.new(
    input_layernorm: [1.0_f32, 0.5_f32, -0.25_f32, 1.5_f32],
    q_proj: qwen3vl_text_block_spec_weight(4, 4, 1),
    k_proj: qwen3vl_text_block_spec_weight(2, 4, 2),
    v_proj: qwen3vl_text_block_spec_weight(2, 4, 3),
    q_norm: [1.0_f32, 0.75_f32],
    k_norm: [0.8_f32, 1.2_f32],
    o_proj: qwen3vl_text_block_spec_weight(4, 4, 4),
    post_attention_layernorm: [0.9_f32, 1.1_f32, 0.75_f32, 1.2_f32],
    gate_proj: qwen3vl_text_block_spec_weight(3, 4, 5),
    up_proj: qwen3vl_text_block_spec_weight(3, 4, 6),
    down_proj: qwen3vl_text_block_spec_weight(4, 3, 7),
  )
end

private def qwen3vl_text_block_spec_hidden : Array(Float32)
  [
    0.5_f32, -0.75_f32, 1.25_f32, 0.25_f32,
    1.5_f32, 0.5_f32, -0.5_f32, 1.0_f32,
    -1.0_f32, 0.25_f32, 0.75_f32, -0.25_f32,
  ]
end

describe ML::GGUF::Qwen3VLTextBlock do
  config = ML::GGUF::Qwen3VLTextBlockConfig.new(
    hidden_dim: 4,
    heads: 2,
    kv_heads: 1,
    head_dim: 2,
    intermediate_dim: 3,
    eps: 1e-6_f32,
    rope_theta: 10_000.0_f32,
  )

  it "matches an independent PyTorch BF16 tiny GQA decoder-layer oracle" do
    # Golden values were generated independently with PyTorch 2.6.0 CPU BF16
    # (the pinned reference venv) using F.linear, Qwen3-VL RMSNorm/RoPE/eager
    # GQA, F.silu, and residual ops. Reproduce from this spec's H/weight helper:
    # create every tensor with dtype=torch.bfloat16; RMSNorm is
    # `xf=x.float(); (xf*rsqrt((xf*xf).mean(-1,keepdim=True)+eps)).to(bf16)*w`;
    # use F.linear(x, W), reshape Q/K/V to [1,T,heads,D], per-head RMSNorm,
    # text arange RoPE with repeated half-frequencies and rotate_half; for each
    # query/head compute `(q @ k.T) * D**-0.5`, softmax visible causal keys in
    # float32 then cast to BF16, and multiply by V. Finish with O projection,
    # BF16 residual, post RMSNorm, `F.linear` gate/up, F.silu(gate)*up,
    # down projection, and BF16 residual. The fixture has 2 query heads sharing
    # 1 KV head and masks token 1.
    expected = [
      0.92578125_f32, -1.28125_f32, 0.72265625_f32, 0.46484375_f32,
      1.5546875_f32, -0.478515625_f32, -0.796875_f32, 1.375_f32,
      -1.40625_f32, 1.0_f32, 0.7734375_f32, -0.44140625_f32,
    ]

    actual = ML::GGUF::Qwen3VLTextBlock.forward(
      qwen3vl_text_block_spec_hidden,
      [true, false, true],
      qwen3vl_text_block_spec_weights,
      config,
    )

    actual.should eq(expected)
  end

  it "distinguishes the PyTorch SDPA F32-intermediate arithmetic from eager BF16" do
    # Oracle provenance: PyTorch 2.6.0 CPU BF16 for the block operations, using
    # the existing block's BF16 Q/K/V projection, normalization, and RoPE
    # materialization boundaries, followed by F.scaled_dot_product_attention
    # under SDPBackend.MATH with repeated GQA KV and a boolean causal/key mask.
    # SDPA keeps scores, softmax probabilities, and the weighted V sum in F32
    # before returning BF16. The existing RoPE boundary is intentionally held
    # fixed here; this isolates the attention arithmetic mode.
    expected = [
      0.92578125_f32, -1.28125_f32, 0.72265625_f32, 0.46484375_f32,
      1.5546875_f32, -0.478515625_f32, -0.796875_f32, 1.375_f32,
      -1.40625_f32, 0.99609375_f32, 0.7734375_f32, -0.447265625_f32,
    ]

    sdpa_config = ML::GGUF::Qwen3VLTextBlockConfig.new(
      hidden_dim: 4,
      heads: 2,
      kv_heads: 1,
      head_dim: 2,
      intermediate_dim: 3,
      eps: 1e-6_f32,
      rope_theta: 10_000.0_f32,
      attention_arithmetic: ML::GGUF::Qwen3VLTextBlockConfig::AttentionArithmetic::SdpaF32,
    )
    eager = ML::GGUF::Qwen3VLTextBlock.forward(
      qwen3vl_text_block_spec_hidden,
      [true, false, true],
      qwen3vl_text_block_spec_weights,
      config,
    )
    actual = ML::GGUF::Qwen3VLTextBlock.forward(
      qwen3vl_text_block_spec_hidden,
      [true, false, true],
      qwen3vl_text_block_spec_weights,
      sdpa_config,
    )

    actual.should eq(expected)
    actual.should_not eq(eager)
  end

  it "captures copied BF16 module boundaries when a trace sink is supplied" do
    expected_sizes = {
      "layer0_input"                      => 12,
      "layers.0.input_layernorm"          => 12,
      "layers.0.self_attn.q_proj"         => 12,
      "layers.0.self_attn.k_proj"         => 6,
      "layers.0.self_attn.v_proj"         => 6,
      "layers.0.self_attn.q_norm"         => 12,
      "layers.0.self_attn.k_norm"         => 6,
      "post_rope_q"                       => 12,
      "post_rope_k"                       => 6,
      "attended"                          => 12,
      "layers.0.self_attn.o_proj"         => 12,
      "layers.0.post_attention_layernorm" => 12,
      "layers.0.mlp.gate_proj"            => 9,
      "layers.0.mlp.up_proj"              => 9,
      "layers.0.mlp.down_proj"            => 12,
      "layers.0"                          => 12,
    }
    trace = Hash(String, Array(Float32)).new
    untraced = ML::GGUF::Qwen3VLTextBlock.forward(
      qwen3vl_text_block_spec_hidden,
      [true, false, true],
      qwen3vl_text_block_spec_weights,
      config,
    )
    traced = ML::GGUF::Qwen3VLTextBlock.forward(
      qwen3vl_text_block_spec_hidden,
      [true, false, true],
      qwen3vl_text_block_spec_weights,
      config,
      trace: trace,
    )

    traced.should eq(untraced)
    trace.keys.sort.should eq(expected_sizes.keys.sort)
    expected_sizes.each do |name, size|
      trace[name].size.should eq(size)
    end
    trace["layers.0"].should eq(traced)
    trace["layers.0"][0] = -99.0_f32
    traced.should eq(untraced)
  end

  it "does not let a masked key row affect a later visible query" do
    hidden = qwen3vl_text_block_spec_hidden
    baseline = ML::GGUF::Qwen3VLTextBlock.forward(
      hidden, [true, false, true], qwen3vl_text_block_spec_weights, config,
    )
    changed_masked_key = hidden.dup
    changed_masked_key[4] = 6.0_f32
    changed_masked_key[5] = -7.0_f32
    changed_masked_key[6] = 8.0_f32
    changed_masked_key[7] = -9.0_f32
    changed = ML::GGUF::Qwen3VLTextBlock.forward(
      changed_masked_key, [true, false, true], qwen3vl_text_block_spec_weights, config,
    )

    changed[8, 4].should eq(baseline[8, 4])
  end

  it "uses raw text arange positions even where the attention mask is zero" do
    ML::GGUF::Qwen3VLTextBlock.text_only_position_ids(4).should eq([0, 1, 2, 3])
  end

  it "rejects input and mask shapes that are not one-batch token rows" do
    expect_raises(ArgumentError, /hidden_states size mismatch/) do
      ML::GGUF::Qwen3VLTextBlock.forward(
        [0.0_f32] * 11, [true, true, true], qwen3vl_text_block_spec_weights, config,
      )
    end

    expect_raises(ArgumentError, /attention_mask size mismatch/) do
      ML::GGUF::Qwen3VLTextBlock.forward(
        qwen3vl_text_block_spec_hidden,
        [true, true],
        qwen3vl_text_block_spec_weights,
        config,
      )
    end
  end
end
