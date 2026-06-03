require "./spec_helper"
require "../src/ml/gguf/gemma4_cpu"

GEMMA4_CPU_12B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

describe ML::GGUF::Gemma4CPU do
  it "applies weighted and plain RMSNorm with Gemma epsilon" do
    x = [3.0_f32, 4.0_f32]
    w = [2.0_f32, 0.5_f32]
    eps = 0.0_f32

    plain = ML::GGUF::Gemma4CPU.rms_norm_plain(x, eps)
    plain[0].should be_close(0.84852815_f32, 1.0e-6_f32)
    plain[1].should be_close(1.1313708_f32, 1.0e-6_f32)

    weighted = ML::GGUF::Gemma4CPU.rms_norm(x, w, eps)
    weighted[0].should be_close(plain[0] * 2.0_f32, 1.0e-6_f32)
    weighted[1].should be_close(plain[1] * 0.5_f32, 1.0e-6_f32)
  end

  it "applies per-slice plain RMSNorm without touching neighboring heads" do
    x = [9.0_f32, 3.0_f32, 4.0_f32, 7.0_f32]

    ML::GGUF::Gemma4CPU.rms_norm_plain_slice!(x, 1, 2, 0.0_f32)

    x[0].should eq(9.0_f32)
    x[1].should be_close(0.84852815_f32, 1.0e-6_f32)
    x[2].should be_close(1.1313708_f32, 1.0e-6_f32)
    x[3].should eq(7.0_f32)
  end

  it "uses llama.cpp tanh-GELU approximation for Gemma FFN" do
    ML::GGUF::Gemma4CPU.gelu(0.0_f32).should be_close(0.0_f32, 1.0e-7_f32)
    ML::GGUF::Gemma4CPU.gelu(1.0_f32).should be_close(0.84119199_f32, 1.0e-6_f32)
    ML::GGUF::Gemma4CPU.gelu(-1.0_f32).should be_close(-0.158808_f32, 1.0e-6_f32)
  end

  it "applies final logit softcapping and leaves cap<=0 unchanged" do
    x = [-60.0_f32, 0.0_f32, 60.0_f32]
    capped = ML::GGUF::Gemma4CPU.logit_softcap(x, 30.0_f32)

    capped[0].should be_close(-28.920826_f32, 1.0e-5_f32)
    capped[1].should eq(0.0_f32)
    capped[2].should be_close(28.920826_f32, 1.0e-5_f32)

    unchanged = ML::GGUF::Gemma4CPU.logit_softcap(x, 0.0_f32)
    unchanged.should eq(x)
  end

  it "rejects RMSNorm weight size mismatches" do
    expect_raises(ArgumentError, /weight size mismatch/) do
      ML::GGUF::Gemma4CPU.rms_norm([1.0_f32, 2.0_f32], [1.0_f32], 1.0e-6_f32)
    end
  end

  it "dequantizes one Gemma4 token embedding row without vocab-sized allocation" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    emb0 = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 0)
    emb42 = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)

    emb0.size.should eq(w.hparams.n_embd)
    emb0.count { |v| v != 0.0_f32 }.should be > 100
    emb0.zip(emb42).any? { |a, b| a != b }.should be_true
    ML::GGUF::Gemma4CPU.quant_row_bytes(w.token_embd.type, w.token_embd.in_dim).should eq(3150)

    expect_raises(ArgumentError, /out of range/) do
      ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, w.token_embd.out_dim)
    end
  end

  it "projects and normalizes SWA layer Q/K/V with explicit V" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    proj = ML::GGUF::Gemma4CPU.attention_project_normed(w.layers[0], x, w.hparams, 0)

    proj.reused_k_as_v.should be_false
    proj.q.size.should eq(16 * 256)
    proj.k.size.should eq(8 * 256)
    proj.v.size.should eq(8 * 256)
    proj.q.count { |v| v != 0.0_f32 }.should be > 100
    proj.k.count { |v| v != 0.0_f32 }.should be > 100
    proj.v.count { |v| v != 0.0_f32 }.should be > 100
  end

  it "projects full-attention layer Q/K and reuses pre-norm K as V" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, w.layers[5].attn_norm, w.hparams.rms_eps)
    pre = ML::GGUF::Gemma4CPU.attention_project_pre_norm(w.layers[5], x_norm)

    pre.reused_k_as_v.should be_true
    pre.q.size.should eq(16 * 512)
    pre.k.size.should eq(1 * 512)
    pre.v.should eq(pre.k)

    ML::GGUF::Gemma4CPU.normalize_attention_projection!(pre, w.layers[5], w.hparams, 5)
    pre.q.size.should eq(16 * 512)
    pre.k.size.should eq(1 * 512)
    pre.v.size.should eq(1 * 512)
    pre.k.should_not eq(pre.v)
  end

  it "applies text-mode RoPE to Q/K but not V" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    proj = ML::GGUF::Gemma4CPU.attention_project_normed(w.layers[0], x, w.hparams, 0)
    q0 = proj.q.dup
    k0 = proj.k.dup
    v0 = proj.v.dup

    ML::GGUF::Gemma4CPU.apply_rope_to_qk!(proj, w.hparams, 0, 7)

    proj.q.should_not eq(q0)
    proj.k.should_not eq(k0)
    proj.v.should eq(v0)
  end

  it "accepts Gemma4 full-layer proportional RoPE factors" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    proj = ML::GGUF::Gemma4CPU.attention_project_normed(w.layers[5], x, w.hparams, 5)
    q0 = proj.q.dup
    k0 = proj.k.dup
    v0 = proj.v.dup

    ML::GGUF::Gemma4CPU.apply_rope_to_qk!(proj, w.hparams, 5, 7, w.rope_freqs)

    proj.q.should_not eq(q0)
    proj.k.should_not eq(k0)
    proj.v.should eq(v0)
  end

  it "computes a one-token SWA attention context and projected output" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    state = ML::GGUF::Gemma4CPU::State.new(w.hparams, 8)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)

    ctx = ML::GGUF::Gemma4CPU.attention_context(w, 0, x, 0, state)
    ctx.size.should eq(16 * 256)
    ctx.all? { |v| v.finite? }.should be_true
    ctx.count { |v| v != 0.0_f32 }.should be > 100
    state.layers[0].position.should eq(1)
    state.layers[0].k_cache.not_nil!.size.should eq(8 * 8 * 256)
    state.layers[0].v_cache.not_nil!.size.should eq(8 * 8 * 256)

    out = ML::GGUF::Gemma4CPU.attention_projected_output(w, 0, x, 1, state)
    out.size.should eq(w.hparams.n_embd)
    out.all? { |v| v.finite? }.should be_true
    state.layers[0].position.should eq(2)
  end

  it "computes full-layer attention context with normalized K-as-V cache split" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    state = ML::GGUF::Gemma4CPU::State.new(w.hparams, 8)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)

    ctx = ML::GGUF::Gemma4CPU.attention_context(w, 5, x, 0, state)
    ctx.size.should eq(16 * 512)
    ctx.all? { |v| v.finite? }.should be_true
    state.layers[5].position.should eq(1)
    state.layers[5].k_cache.not_nil!.size.should eq(8 * 1 * 512)
    state.layers[5].v_cache.not_nil!.size.should eq(8 * 1 * 512)
    state.layers[5].k_cache.not_nil![0, 512].should_not eq(state.layers[5].v_cache.not_nil![0, 512])
  end

  it "runs one Gemma4 layer through attention, FFN, residuals, and layer scale" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_CPU_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_CPU_12B_Q4KM)
    state = ML::GGUF::Gemma4CPU::State.new(w.hparams, 8)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    y = ML::GGUF::Gemma4CPU.forward_layer(w, 0, x, 0, state)

    y.size.should eq(w.hparams.n_embd)
    y.all? { |v| v.finite? }.should be_true
    y.should_not eq(x)
    state.layers[0].position.should eq(1)
  end
end
