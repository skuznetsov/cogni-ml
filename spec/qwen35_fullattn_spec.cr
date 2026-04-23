require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_FULLATTN = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35CPU, "full-attention layer forward" do
  pending!("9B model not present") unless File.exists?(QWEN_9B_FULLATTN)

  # KV cache lives on whichever backend the routing helper picked up:
  # Array on CPU, MetalBuffer on GPU. Cache fields are mutually exclusive.
  read_kv = ->(ls : ML::GGUF::Qwen35CPU::LayerState, expected : Int32, pick_k : Bool) do
    buf = pick_k ? ls.k_cache_buf : ls.v_cache_buf
    if buf
      buf.read(expected)
    else
      (pick_k ? ls.k_cache : ls.v_cache).not_nil!
    end
  end

  it "runs blk.3 (first full-attn layer) at pos=0 on embedding of token 0" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FULLATTN)
    hp = w.hparams
    hp.full_attention?(3).should be_true

    lw = w.layers[3].as(ML::GGUF::Qwen35FullAttnWeights)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)

    inp = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 0)
    inp.size.should eq(hp.n_embd)

    out = ML::GGUF::Qwen35CPU.forward_full_attn_layer(inp, 0, lw, state.layers[3], hp, 64)

    out.size.should eq(hp.n_embd)
    out.all? { |v| v.finite? }.should be_true

    # KV cache should have been written at pos=0
    state.layers[3].position = 1  # bump for next call
    kv_dim = hp.head_dim * hp.n_head_kv
    expected_size = 64 * kv_dim
    kc = read_kv.call(state.layers[3], expected_size, true)
    vc = read_kv.call(state.layers[3], expected_size, false)
    kc.size.should eq(expected_size)
    vc.size.should eq(expected_size)

    # Cache at pos=0 should be non-zero (values were written)
    nonzero_k = (0...kv_dim).count { |i| kc[i] != 0.0_f32 }
    nonzero_k.should be > kv_dim // 10
  end

  it "produces different output at pos=1 vs pos=0 (M-RoPE effect)" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FULLATTN)
    hp = w.hparams
    lw = w.layers[3].as(ML::GGUF::Qwen35FullAttnWeights)

    state_a = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)
    state_b = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)

    inp = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 42)

    out_a = ML::GGUF::Qwen35CPU.forward_full_attn_layer(inp, 0, lw, state_a.layers[3], hp, 64)
    out_b = ML::GGUF::Qwen35CPU.forward_full_attn_layer(inp, 1, lw, state_b.layers[3], hp, 64)

    # Different positions → different rotated Q/K → different output
    diff = out_a.zip(out_b).count { |a, b| (a - b).abs > 1.0e-6_f32 }
    diff.should be > hp.n_embd // 4
  end
end
