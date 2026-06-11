require "./spec_helper"
require "../src/ml/gguf/diffusion_gemma_meta"
require "../src/ml/gguf/diffusion_gemma_weights"
require "../src/ml/gguf/diffusion_gemma_runtime"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/gemma4_metal"

DIFFUSION_GEMMA_26B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

def fake_diffusion_gemma_projection(hp : ML::GGUF::DiffusionGemmaHparams,
                                    il : Int32,
                                    value : Float32) : ML::GGUF::DiffusionGemmaCPU::AttentionProjection
  head_dim = hp.head_dim_for_layer(il)
  q_dim = hp.n_head * head_dim
  kv_dim = hp.n_head_kv(il) * head_dim
  ML::GGUF::DiffusionGemmaCPU::AttentionProjection.new(
    q: Array(Float32).new(q_dim, 0.0_f32),
    k: Array(Float32).new(kv_dim, 0.0_f32),
    v: Array(Float32).new(kv_dim, value),
    reused_k_as_v: false,
  )
end

def deterministic_diffusion_gemma_projection(hp : ML::GGUF::DiffusionGemmaHparams,
                                             il : Int32) : ML::GGUF::DiffusionGemmaCPU::AttentionProjection
  head_dim = hp.head_dim_for_layer(il)
  q_dim = hp.n_head * head_dim
  kv_dim = hp.n_head_kv(il) * head_dim
  ML::GGUF::DiffusionGemmaCPU::AttentionProjection.new(
    q: Array(Float32).new(q_dim) { |i| ((i % 97) - 48).to_f32 / 37.0_f32 },
    k: Array(Float32).new(kv_dim) { |i| ((i % 53) - 26).to_f32 / 29.0_f32 },
    v: Array(Float32).new(kv_dim) { |i| ((i % 31) - 15).to_f32 / 23.0_f32 },
    reused_k_as_v: false,
  )
end

def diffusion_gemma_attention_context_rows_reference(q_rows : Array(Float32),
                                                     k_cache : Array(Float32),
                                                     v_cache : Array(Float32),
                                                     base_pos : Int32,
                                                     rows : Int32,
                                                     n_head : Int32,
                                                     n_head_kv : Int32,
                                                     head_dim : Int32,
                                                     sliding_window : Int32) : Array(Float32)
  q_dim = n_head * head_dim
  kv_dim = n_head_kv * head_dim
  heads_per_group = n_head // n_head_kv
  out = Array(Float32).new(rows * q_dim, 0.0_f32)
  scores = Array(Float32).new(base_pos + rows, 0.0_f32)

  rows.times do |row|
    row_pos = base_pos + row
    start_pos = sliding_window == 0 || row_pos + 1 <= sliding_window ? 0 : row_pos + 1 - sliding_window
    len = row_pos - start_pos + 1

    n_head.times do |h|
      kvh = h // heads_per_group
      q_off = (row * n_head + h) * head_dim

      max_score = -Float32::INFINITY
      len.times do |i|
        pos = start_pos + i
        k_off = pos * kv_dim + kvh * head_dim
        score = 0.0_f32
        head_dim.times { |d| score += q_rows[q_off + d] * k_cache[k_off + d] }
        scores[i] = score
        max_score = score if score > max_score
      end

      sum = 0.0_f32
      len.times do |i|
        weight = Math.exp((scores[i] - max_score).to_f64).to_f32
        scores[i] = weight
        sum += weight
      end

      out_off = (row * n_head + h) * head_dim
      len.times do |i|
        pos = start_pos + i
        v_off = pos * kv_dim + kvh * head_dim
        weight = scores[i] / sum
        head_dim.times { |d| out[out_off + d] += weight * v_cache[v_off + d] }
      end
    end
  end

  out
end

def diffusion_gemma_rope_pair_energy(x : Array(Float32), offset : Int32, half : Int32, i : Int32) : Float64
  x0 = x[offset + i].to_f64
  x1 = x[offset + i + half].to_f64
  x0 * x0 + x1 * x1
end

describe ML::GGUF::DiffusionGemmaHparams do
  it "parses the local DiffusionGemma 26B GGUF metadata" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    g = ML::GGUF::GGUFFile.new(DIFFUSION_GEMMA_26B_Q4KM)
    begin
      hp = ML::GGUF::DiffusionGemmaHparams.new(g)

      hp.arch.should eq("diffusion-gemma")
      hp.n_layer.should eq(30)
      hp.n_embd.should eq(2816)
      hp.n_ff.should eq(2112)
      hp.expert_ff.should eq(704)
      hp.expert_count.should eq(128)
      hp.expert_used_count.should eq(8)
      hp.context_length.should eq(262_144)
      hp.canvas_length.should eq(256)
      hp.vocab_size.should eq(262_144)
      hp.causal_attention.should be_false
      hp.eb_max_steps.should eq(48)
      hp.eb_t_min.should eq(0.4_f32)
      hp.eb_t_max.should eq(0.8_f32)
      hp.eb_entropy_bound.should eq(0.1_f32)
      hp.eb_stability_threshold.should eq(1)
      hp.eb_confidence_threshold.should eq(0.005_f32)

      hp.n_head.should eq(16)
      hp.head_dim.should eq(512)
      hp.head_dim_swa.should eq(256)
      hp.rope_dim_count.should eq(512)
      hp.rope_dim_count_swa.should eq(256)
      hp.sliding_window.should eq(1024)

      hp.full_attention_layers.should eq([5, 11, 17, 23, 29])
      hp.sliding_window_layers.size.should eq(25)
      hp.n_head_kv_by_layer.count(8).should eq(25)
      hp.n_head_kv_by_layer.count(2).should eq(5)
      hp.has_kv?(0).should be_true
      hp.has_kv?(29).should be_true
      hp.attention_start_pos(0, 1500).should eq(477)
      hp.attention_start_pos(5, 1500).should eq(0)
    ensure
      g.close
    end
  end
end

describe ML::GGUF::DiffusionGemmaDenoiseParams do
  it "uses oracle-compatible entropy-bound defaults from hparams" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    g = ML::GGUF::GGUFFile.new(DIFFUSION_GEMMA_26B_Q4KM)
    begin
      hp = ML::GGUF::DiffusionGemmaHparams.new(g)
      params = ML::GGUF::DiffusionGemmaDenoiseParams.from_hparams(hp, max_steps: 10, seed: 7)

      params.max_steps.should eq(10)
      params.t_min.should eq(0.4_f32)
      params.t_max.should eq(0.8_f32)
      params.entropy_bound.should eq(0.1_f32)
      params.stability_threshold.should eq(1)
      params.confidence_threshold.should eq(0.005_f32)
      params.kv_cache.should be_true
      params.seed.should eq(7)

      sample_a = ML::GGUF::DiffusionGemmaCPU.sample_u_steps(params.seed, 2, 3)
      sample_b = ML::GGUF::DiffusionGemmaCPU.sample_u_steps(params.seed, 2, 3)
      sample_c = ML::GGUF::DiffusionGemmaCPU.sample_u_steps(params.seed + 1, 2, 3)
      sample_a.should eq(sample_b)
      sample_a.should_not eq(sample_c)
      sample_a.size.should eq(2)
      sample_a.each do |row|
        row.size.should eq(3)
        row.each do |v|
          v.should be >= 0.0_f32
          v.should be < 1.0_f32
        end
      end
      ML::GGUF::DiffusionGemmaCPU.sample_u_rows(params.seed, 3, 1).should eq(sample_a[1])
    ensure
      g.close
    end
  end

  it "rejects invalid denoise params before runtime execution" do
    expect_raises(ArgumentError, /max_steps/) do
      ML::GGUF::DiffusionGemmaDenoiseParams.new(
        max_steps: 0,
        t_min: 0.4_f32,
        t_max: 0.8_f32,
        entropy_bound: 0.1_f32,
        stability_threshold: 1,
        confidence_threshold: 0.005_f32,
      )
    end
    expect_raises(ArgumentError, /t_max/) do
      ML::GGUF::DiffusionGemmaDenoiseParams.new(
        max_steps: 1,
        t_min: 0.9_f32,
        t_max: 0.8_f32,
        entropy_bound: 0.1_f32,
        stability_threshold: 1,
        confidence_threshold: 0.005_f32,
      )
    end
    expect_raises(ArgumentError, /sample_u steps/) do
      ML::GGUF::DiffusionGemmaCPU.sample_u_steps(7, 0, 1)
    end
    expect_raises(ArgumentError, /canvas_len/) do
      ML::GGUF::DiffusionGemmaCPU.sample_u_rows(7, 0)
    end
    expect_raises(ArgumentError, /step/) do
      ML::GGUF::DiffusionGemmaCPU.sample_u_rows(7, 1, -1)
    end
  end
end

describe ML::GGUF::DiffusionGemmaRequest do
  it "validates prompt canvas split and builds packed tokens" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    g = ML::GGUF::GGUFFile.new(DIFFUSION_GEMMA_26B_Q4KM)
    begin
      hp = ML::GGUF::DiffusionGemmaHparams.new(g)
      req = ML::GGUF::DiffusionGemmaRequest.with_blank_canvas([11, 22, 33], hp, mask_token_id: 0)

      req.prompt_len.should eq(3)
      req.canvas_len.should eq(256)
      req.total_tokens.should eq(259)
      req.canvas_start.should eq(3)
      req.canvas_position?(2).should be_false
      req.canvas_position?(3).should be_true
      req.packed_tokens[0, 3].should eq([11, 22, 33])
      req.packed_tokens[3, 3].should eq([0, 0, 0])
    ensure
      g.close
    end
  end

  it "rejects malformed requests before forward" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    g = ML::GGUF::GGUFFile.new(DIFFUSION_GEMMA_26B_Q4KM)
    begin
      hp = ML::GGUF::DiffusionGemmaHparams.new(g)
      expect_raises(ArgumentError, /prompt/) do
        ML::GGUF::DiffusionGemmaRequest.with_blank_canvas([] of Int32, hp, mask_token_id: 0)
      end
      expect_raises(ArgumentError, /canvas size/) do
        ML::GGUF::DiffusionGemmaRequest.new([1], [0, 0], hp)
      end
      expect_raises(ArgumentError, /mask_token_id/) do
        ML::GGUF::DiffusionGemmaRequest.with_blank_canvas([1], hp, mask_token_id: -1)
      end
    ensure
      g.close
    end
  end
end

describe ML::GGUF::DiffusionGemmaAttentionMask do
  it "matches the oracle region-aware unified attention mask" do
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 4, canvas_len: 3, sliding_window: 3)

    mask.allow_unified?(2, 1, sliding: false).should be_true
    mask.allow_unified?(2, 3, sliding: false).should be_false
    mask.allow_unified?(2, 4, sliding: false).should be_false
    mask.allow_unified?(5, 0, sliding: false).should be_true
    mask.allow_unified?(5, 6, sliding: false).should be_true

    mask.canvas_prompt_low.should eq(2)
    mask.allow_unified?(5, 1, sliding: true).should be_false
    mask.allow_unified?(5, 2, sliding: true).should be_true
    mask.allow_unified?(5, 6, sliding: true).should be_true
    mask.allow_unified?(3, 0, sliding: true).should be_false
    mask.allow_unified?(3, 1, sliding: true).should be_true
  end

  it "matches the oracle prompt-KV decode mask" do
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 4, canvas_len: 3, sliding_window: 3)

    mask.allow_decode?(0, 0, sliding: false).should be_true
    mask.allow_decode?(2, 6, sliding: false).should be_true
    mask.allow_decode?(1, 1, sliding: true).should be_false
    mask.allow_decode?(1, 2, sliding: true).should be_true
    mask.allow_decode?(1, 5, sliding: true).should be_true
  end
end

describe ML::GGUF::DiffusionGemmaCPU do
  it "builds oracle-compatible zero-SC region embeddings" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    req = ML::GGUF::DiffusionGemmaRequest.with_blank_canvas([1, 2], hp, mask_token_id: 0)
    rows = ML::GGUF::DiffusionGemmaCPU.region_embeddings(w, req)

    rows.size.should eq(req.total_tokens * hp.n_embd)
    prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    rows[0, hp.n_embd].should eq(prompt_row)
    rows[req.canvas_start * hp.n_embd, hp.n_embd].should eq(canvas_row)

    prompt_rms = ML::GGUF::DiffusionGemmaCPU.row_rms(rows, 0, hp.n_embd)
    canvas_rms = ML::GGUF::DiffusionGemmaCPU.row_rms(rows, req.canvas_start * hp.n_embd, hp.n_embd)
    prompt_rms.should be > 1.0_f32
    canvas_rms.should be_close(1.0_f32, 0.001_f32)
  end

  it "rejects invalid embedding token ids before forward" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    expect_raises(ArgumentError, /token_id/) do
      ML::GGUF::DiffusionGemmaCPU.embedding_lookup(w, -1)
    end
    expect_raises(ArgumentError, /token_id/) do
      ML::GGUF::DiffusionGemmaCPU.embedding_lookup(w, w.hparams.vocab_size)
    end
  end

  it "computes self-conditioning soft embeddings and canvas SC injection" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    token_id = 0

    soft = ML::GGUF::DiffusionGemmaCPU.self_conditioning_soft_embedding(w, [token_id], [0.0_f32])
    soft.should eq(ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, token_id))

    signal = ML::GGUF::DiffusionGemmaCPU.self_conditioning_signal(w, soft)
    signal.size.should eq(hp.n_embd)
    signal.all? { |v| v.finite? }.should be_true
    signal.any? { |v| v.abs > 0.000001_f32 }.should be_true

    zero_sc = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, token_id)
    ML::GGUF::DiffusionGemmaCPU.canvas_embedding_with_self_conditioning(
      w,
      token_id,
      [] of Int32,
      [] of Float32,
      sc_use: 0.0_f32,
    ).should eq(zero_sc)

    with_sc = ML::GGUF::DiffusionGemmaCPU.canvas_embedding_with_self_conditioning(w, token_id, [token_id], [0.0_f32])
    with_sc.size.should eq(hp.n_embd)
    with_sc.all? { |v| v.finite? }.should be_true
    with_sc.should_not eq(zero_sc)
    ML::GGUF::DiffusionGemmaCPU.row_rms(with_sc, 0, hp.n_embd).should be_close(1.0_f32, 0.001_f32)

    zero_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(w, [token_id])
    zero_rows.should eq(zero_sc)
    sc_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(w, [token_id], [[token_id]], [[0.0_f32]])
    sc_rows.should eq(with_sc)
    sc_pred = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([token_id], [0.0_f32])
    ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_prediction_self_conditioning(w, [token_id], [sc_pred]).should eq(with_sc)

    expect_raises(ArgumentError, /logits size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.self_conditioning_soft_embedding(w, [token_id], [] of Float32)
    end
    expect_raises(ArgumentError, /temp_inv/) do
      ML::GGUF::DiffusionGemmaCPU.self_conditioning_soft_embedding(w, [token_id], [0.0_f32], temp_inv: 0.0_f32)
    end
    expect_raises(ArgumentError, /supplied together/) do
      ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(w, [token_id], [[token_id]])
    end
    expect_raises(ArgumentError, /prediction self-conditioning/) do
      ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_prediction_self_conditioning(w, [token_id], [] of ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction)
    end
  end

  it "projects and normalizes QKV for SWA and full-attention layers" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    x = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)

    swa = ML::GGUF::DiffusionGemmaCPU.attention_project_normed(w, 0, x, pos: 3)
    swa.reused_k_as_v.should be_false
    swa.q.size.should eq(hp.n_head * hp.head_dim_swa)
    swa.k.size.should eq(hp.n_head_kv(0) * hp.head_dim_swa)
    swa.v.size.should eq(hp.n_head_kv(0) * hp.head_dim_swa)
    swa_q_norm_rms = ML::GGUF::DiffusionGemmaCPU.row_rms(w.layers[0].attn_q_norm, 0, hp.head_dim_swa)
    swa_k_norm_rms = ML::GGUF::DiffusionGemmaCPU.row_rms(w.layers[0].attn_k_norm, 0, hp.head_dim_swa)
    ML::GGUF::DiffusionGemmaCPU.row_rms(swa.q, 0, hp.head_dim_swa).should be_close(swa_q_norm_rms, 0.01_f32)
    ML::GGUF::DiffusionGemmaCPU.row_rms(swa.k, 0, hp.head_dim_swa).should be_close(swa_k_norm_rms, 0.01_f32)
    ML::GGUF::DiffusionGemmaCPU.row_rms(swa.v, 0, hp.head_dim_swa).should be_close(1.0_f32, 0.01_f32)

    full = ML::GGUF::DiffusionGemmaCPU.attention_project_normed(w, 5, x, pos: 3)
    full.reused_k_as_v.should be_true
    full.q.size.should eq(hp.n_head * hp.head_dim)
    full.k.size.should eq(hp.n_head_kv(5) * hp.head_dim)
    full.v.size.should eq(hp.n_head_kv(5) * hp.head_dim)
    full_q_norm_rms = ML::GGUF::DiffusionGemmaCPU.row_rms(w.layers[5].attn_q_norm, 0, hp.head_dim)
    full_k_norm_rms = ML::GGUF::DiffusionGemmaCPU.row_rms(w.layers[5].attn_k_norm, 0, hp.head_dim)
    ML::GGUF::DiffusionGemmaCPU.row_rms(full.q, 0, hp.head_dim).should be_close(full_q_norm_rms, 0.01_f32)
    ML::GGUF::DiffusionGemmaCPU.row_rms(full.k, 0, hp.head_dim).should be_close(full_k_norm_rms, 0.01_f32)
    ML::GGUF::DiffusionGemmaCPU.row_rms(full.v, 0, hp.head_dim).should be_close(1.0_f32, 0.01_f32)
  end

  it "applies RoPE only to Q/K while preserving per-head pair energy" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams

    [0, 5].each do |il|
      head_dim = hp.head_dim_for_layer(il)
      n_rot = hp.rope_dim_for_layer(il)
      half = n_rot // 2

      no_op = deterministic_diffusion_gemma_projection(hp, il)
      no_op_q = no_op.q.dup
      no_op_k = no_op.k.dup
      no_op_v = no_op.v.dup
      ML::GGUF::DiffusionGemmaCPU.apply_rope_to_qk!(no_op, hp, il, pos: 0, rope_freqs: w.rope_freqs)
      no_op.q.should eq(no_op_q)
      no_op.k.should eq(no_op_k)
      no_op.v.should eq(no_op_v)

      proj = deterministic_diffusion_gemma_projection(hp, il)
      q_before = proj.q.dup
      k_before = proj.k.dup
      v_before = proj.v.dup
      ML::GGUF::DiffusionGemmaCPU.apply_rope_to_qk!(proj, hp, il, pos: 7, rope_freqs: w.rope_freqs)

      proj.q.should_not eq(q_before)
      proj.k.should_not eq(k_before)
      proj.v.should eq(v_before)

      hp.n_head.times do |h|
        offset = h * head_dim
        half.times do |i|
          before = diffusion_gemma_rope_pair_energy(q_before, offset, half, i)
          after = diffusion_gemma_rope_pair_energy(proj.q, offset, half, i)
          after.should be_close(before, 0.00001)
        end
      end
      hp.n_head_kv(il).times do |h|
        offset = h * head_dim
        half.times do |i|
          before = diffusion_gemma_rope_pair_energy(k_before, offset, half, i)
          after = diffusion_gemma_rope_pair_energy(proj.k, offset, half, i)
          after.should be_close(before, 0.00001)
        end
      end
    end
  end

  it "computes region-aware unified and decode attention context" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    g = ML::GGUF::GGUFFile.new(DIFFUSION_GEMMA_26B_Q4KM)
    begin
      hp = ML::GGUF::DiffusionGemmaHparams.new(g)
      il = 0
      head_dim = hp.head_dim_for_layer(il)
      mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 4, canvas_len: 2, sliding_window: 3)
      projections = (0...6).map do |i|
        fake_diffusion_gemma_projection(hp, il, (i + 1).to_f32)
      end

      prompt0 = ML::GGUF::DiffusionGemmaCPU.attention_context_unified(projections, hp, il, query_pos: 0, mask: mask)
      prompt0[0].should be_close(1.0_f32, 0.0001_f32)
      prompt0[(hp.n_head - 1) * head_dim].should be_close(1.0_f32, 0.0001_f32)

      prompt1 = ML::GGUF::DiffusionGemmaCPU.attention_context_unified(projections, hp, il, query_pos: 1, mask: mask)
      prompt1[0].should be_close(1.5_f32, 0.0001_f32)

      canvas0 = ML::GGUF::DiffusionGemmaCPU.attention_context_unified(projections, hp, il, query_pos: 4, mask: mask)
      canvas0[0].should be_close(4.5_f32, 0.0001_f32)

      decoded = ML::GGUF::DiffusionGemmaCPU.attention_context_decode(
        prompt_projections: projections[0, 4],
        canvas_projections: projections[4, 2],
        hp: hp,
        il: il,
        canvas_query_index: 0,
        mask: mask,
      )
      decoded[0].should be_close(4.5_f32, 0.0001_f32)
    ensure
      g.close
    end
  end

  it "projects attention context, applies post-attention norm, and adds residual" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    x = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)

    swa_context = Array(Float32).new(hp.n_head * hp.head_dim_swa, 0.0_f32)
    swa_projected = ML::GGUF::DiffusionGemmaCPU.attention_output_project(w, 0, swa_context)
    swa_projected.size.should eq(hp.n_embd)
    swa_projected.all?(&.zero?).should be_true
    ML::GGUF::DiffusionGemmaCPU.attention_residual_from_context(w, 0, x, swa_context).should eq(x)

    full_context = Array(Float32).new(hp.n_head * hp.head_dim, 0.0_f32)
    full_projected = ML::GGUF::DiffusionGemmaCPU.attention_output_project(w, 5, full_context)
    full_projected.size.should eq(hp.n_embd)
    full_projected.all?(&.zero?).should be_true
    ML::GGUF::DiffusionGemmaCPU.attention_residual_from_context(w, 5, x, full_context).should eq(x)

    expect_raises(ArgumentError, /attention context size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.attention_output_project(w, 0, [0.0_f32])
    end
  end

  it "computes the shared dense FFN branch and residual combiner boundary" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    zero = Array(Float32).new(hp.n_embd, 0.0_f32)

    dense = ML::GGUF::DiffusionGemmaCPU.shared_dense_ffn(w, 0, zero)
    dense.size.should eq(hp.n_embd)
    dense.all?(&.zero?).should be_true
    ML::GGUF::DiffusionGemmaCPU.ffn_residual_from_parts(w, 0, zero, dense).should eq(zero)

    expect_raises(ArgumentError, /shared_dense_ffn input size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.shared_dense_ffn(w, 0, [0.0_f32])
    end
    expect_raises(ArgumentError, /shared_dense size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.ffn_residual_from_parts(w, 0, zero, [0.0_f32])
    end
  end

  it "computes softmax router probabilities and deterministic top-k expert routes" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    x = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)

    router_x = ML::GGUF::DiffusionGemmaCPU.router_input(w, 0, x)
    router_x.size.should eq(hp.n_embd)
    logits = ML::GGUF::DiffusionGemmaCPU.router_logits(w, 0, x)
    logits.size.should eq(hp.expert_count)
    probs = ML::GGUF::DiffusionGemmaCPU.softmax(logits)
    probs.sum.should be_close(1.0_f32, 0.0001_f32)

    routes = ML::GGUF::DiffusionGemmaCPU.route_experts(w, 0, x)
    routes.size.should eq(hp.expert_used_count)
    routes.each { |route| route.expert.should be >= 0; route.expert.should be < hp.expert_count }
    (0...(routes.size - 1)).each do |i|
      routes[i].weight.should be >= routes[i + 1].weight
    end
    routes.sum(&.weight).should be < 1.0_f32

    zero = Array(Float32).new(hp.n_embd, 0.0_f32)
    zero_routes = ML::GGUF::DiffusionGemmaCPU.route_experts(w, 0, zero)
    zero_routes.map(&.expert).should eq((0...hp.expert_used_count).to_a)
    zero_routes.each { |route| route.weight.should be_close(1.0_f32 / hp.expert_count.to_f32, 0.000001_f32) }

    expect_raises(ArgumentError, /router input size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.router_logits(w, 0, [0.0_f32])
    end
  end

  it "computes the routed MoE expert branch with quantized expert slices" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    zero = Array(Float32).new(hp.n_embd, 0.0_f32)

    zero_moe = ML::GGUF::DiffusionGemmaCPU.moe_ffn(w, 0, zero)
    zero_moe.size.should eq(hp.n_embd)
    zero_moe.all?(&.zero?).should be_true

    x = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    routes = ML::GGUF::DiffusionGemmaCPU.route_experts(w, 0, x)
    one_route_moe = ML::GGUF::DiffusionGemmaCPU.moe_ffn(w, 0, x, routes[0, 1])
    one_route_moe.size.should eq(hp.n_embd)
    one_route_moe.all? { |v| v.finite? }.should be_true
    one_route_moe.any? { |v| v.abs > 0.000001_f32 }.should be_true

    expect_raises(ArgumentError, /expert id out of range/) do
      ML::GGUF::DiffusionGemmaCPU.moe_expert_output(w, 0, hp.expert_count, zero)
    end
    expect_raises(ArgumentError, /routes must not be empty/) do
      ML::GGUF::DiffusionGemmaCPU.moe_ffn(w, 0, zero, [] of ML::GGUF::DiffusionGemmaCPU::ExpertRoute)
    end
  end

  it "combines dense and routed MoE FFN branches into the residual output" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    zero = Array(Float32).new(hp.n_embd, 0.0_f32)

    ML::GGUF::DiffusionGemmaCPU.ffn_residual(w, 0, zero).should eq(zero)

    x = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, 0, x)[0, 1]
    result = ML::GGUF::DiffusionGemmaCPU.ffn_residual(w, 0, x, route)
    result.size.should eq(hp.n_embd)
    result.all? { |v| v.finite? }.should be_true
    result.should_not eq(x)
  end

  it "applies the one-row layer tail and region-aware output scale" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    zero = Array(Float32).new(hp.n_embd, 0.0_f32)
    zero_context = Array(Float32).new(hp.n_head * hp.head_dim_swa, 0.0_f32)

    ML::GGUF::DiffusionGemmaCPU.layer_output_from_context(w, 0, zero, zero_context, canvas: true).should eq(zero)

    x = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, 0, x)[0, 1]
    unscaled = ML::GGUF::DiffusionGemmaCPU.ffn_residual(w, 0, x, route)
    canvas_scaled = ML::GGUF::DiffusionGemmaCPU.scale_layer_output(w, 0, unscaled, canvas: true)
    prompt_scaled = ML::GGUF::DiffusionGemmaCPU.scale_layer_output(w, 0, unscaled, canvas: false)
    canvas_scale = w.layers[0].layer_output_scale[0]
    prompt_scale = w.layers[0].encoder_layer_output_scale[0]
    canvas_scaled[0].should eq(unscaled[0] * canvas_scale)
    prompt_scaled[0].should eq(unscaled[0] * prompt_scale)

    layer_out = ML::GGUF::DiffusionGemmaCPU.layer_output_from_context(w, 0, x, zero_context, canvas: true, routes: route)
    layer_out.size.should eq(hp.n_embd)
    layer_out.all? { |v| v.finite? }.should be_true
    layer_out.should eq(canvas_scaled)
  end

  it "matches prompt-only layer rows to the unified prompt path" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    il = 0
    prompt0 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    prompt1 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 2)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    prompt_rows = prompt0 + prompt1
    rows = prompt_rows + canvas_row
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 2, canvas_len: 1, sliding_window: 3)
    prompt_routes = [
      ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt0)[0, 1],
      ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt1)[0, 1],
    ]
    canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, canvas_row)[0, 1]

    prompt_only = ML::GGUF::DiffusionGemmaCPU.layer_forward_prompt_rows(w, il, prompt_rows, mask, prompt_routes)
    unified = ML::GGUF::DiffusionGemmaCPU.layer_forward_unified_rows(w, il, rows, mask, prompt_routes + [canvas_route])
    prompt_only.size.should eq(2 * hp.n_embd)
    prompt_only.all? { |v| v.finite? }.should be_true
    prompt_only.should eq(unified[0, 2 * hp.n_embd])

    expect_raises(ArgumentError, /prompt rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_prompt_rows(w, il, [0.0_f32], mask)
    end
    expect_raises(ArgumentError, /routes_by_prompt_row/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_prompt_rows(w, il, prompt_rows, mask, [prompt_routes[0]])
    end
  end

  it "builds a bounded prompt layer cache from prompt-only rows" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    il = 0
    prompt0 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    prompt1 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 2)
    prompt_rows = prompt0 + prompt1
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 2, canvas_len: 1, sliding_window: 3)
    prompt_routes = [
      ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt0)[0, 1],
      ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt1)[0, 1],
    ]

    cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      w,
      prompt_rows,
      mask,
      max_layers: 1,
      routes_by_layer_by_prompt_row: [prompt_routes],
    )
    expected_projections = ML::GGUF::DiffusionGemmaCPU.prompt_attention_projections(w, il, prompt_rows, mask)
    scalar_projection0 = ML::GGUF::DiffusionGemmaCPU.attention_project_normed(w, il, prompt0, pos: 0)
    scalar_projection1 = ML::GGUF::DiffusionGemmaCPU.attention_project_normed(w, il, prompt1, pos: 1)
    expected_rows = ML::GGUF::DiffusionGemmaCPU.layer_forward_prompt_rows(w, il, prompt_rows, mask, prompt_routes)
    expected_projections[0].q.should eq(scalar_projection0.q)
    expected_projections[0].k.should eq(scalar_projection0.k)
    expected_projections[1].q.should eq(scalar_projection1.q)
    expected_projections[1].k.should eq(scalar_projection1.k)
    cache.layers.should eq(1)
    cache.projections_by_layer[0].size.should eq(2)
    cache.projections_by_layer[0][0].q.should eq(expected_projections[0].q)
    cache.projections_by_layer[0][1].k.should eq(expected_projections[1].k)
    cache.final_rows.should eq(expected_rows)
    cache.final_rows.size.should eq(2 * hp.n_embd)

    expect_raises(ArgumentError, /max_layers/) do
      ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(w, prompt_rows, mask, max_layers: 0)
    end
    expect_raises(ArgumentError, /routes_by_layer_by_prompt_row/) do
      ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
        w,
        prompt_rows,
        mask,
        max_layers: 1,
        routes_by_layer_by_prompt_row: [] of Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)),
      )
    end
  end

  it "keeps fused prompt QK norm and RoPE projection parity" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    prompt0 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    prompt1 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 2)
    prompt_rows = prompt0 + prompt1
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 2, canvas_len: 1, sliding_window: 3)
    old_fused = ENV["DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE"]?
    old_metal_off = ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF"]?

    begin
      ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF"] = "1"
      ENV.delete("DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE")
      base = [0, 5].map do |il|
        ML::GGUF::DiffusionGemmaCPU.prompt_attention_projections(w, il, prompt_rows, mask)
      end

      ENV["DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE"] = "1"
      fused = [0, 5].map do |il|
        ML::GGUF::DiffusionGemmaCPU.prompt_attention_projections(w, il, prompt_rows, mask)
      end

      base.each_with_index do |base_layer, layer_index|
        fused_layer = fused[layer_index]
        base_layer.each_with_index do |base_projection, pos|
          fused_projection = fused_layer[pos]
          fused_projection.q.should eq(base_projection.q)
          fused_projection.k.should eq(base_projection.k)
          fused_projection.v.should eq(base_projection.v)
          fused_projection.reused_k_as_v.should eq(base_projection.reused_k_as_v)
        end
      end
      hp.full_attention?(5).should be_true
      hp.sliding_window?(0).should be_true
    ensure
      if old_fused
        ENV["DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE"] = old_fused
      else
        ENV.delete("DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE")
      end
      if old_metal_off
        ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF"] = old_metal_off
      else
        ENV.delete("DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF")
      end
    end
  end

  it "reuses Gemma4 Metal attention context rows on DiffusionGemma GQA shapes" do
    next unless ML::GGUF::Gemma4Metal.available?

    n_head = 16
    n_head_kv = 8
    head_dim = 256
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    base_pos = 64
    sliding_window = 64

    [1, 2].each do |rows|
      q_rows = Array(Float32).new(rows * q_dim) do |i|
        ((((i * 17 + rows * 23) % 257) - 128).to_f32 / 512.0_f32)
      end
      k_cache = Array(Float32).new((base_pos + rows) * kv_dim) do |i|
        ((((i * 13 + rows * 11) % 251) - 125).to_f32 / 512.0_f32)
      end
      v_cache = Array(Float32).new((base_pos + rows) * kv_dim) do |i|
        ((((i * 19 + rows * 7) % 241) - 120).to_f32 / 512.0_f32)
      end

      expected = diffusion_gemma_attention_context_rows_reference(
        q_rows, k_cache, v_cache, base_pos, rows, n_head, n_head_kv, head_dim, sliding_window)
      actual = ML::GGUF::Gemma4Metal.attention_context_rows(
        q_rows, k_cache, v_cache, base_pos, rows, n_head, n_head_kv, head_dim, sliding_window).not_nil!

      max_diff = 0.0_f32
      expected.size.times do |i|
        diff = (expected[i] - actual[i]).abs
        max_diff = diff if diff > max_diff
      end
      max_diff.should be < 1.0e-4_f32
    end
  end

  it "keeps gated Metal decode attention context parity" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)
    next unless ML::GGUF::Gemma4Metal.available?

    g = ML::GGUF::GGUFFile.new(DIFFUSION_GEMMA_26B_Q4KM)
    old_context_metal = ENV["DIFFUSION_GEMMA_CONTEXT_METAL"]?
    old_context_metal_off = ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"]?
    begin
      hp = ML::GGUF::DiffusionGemmaHparams.new(g)
      il = 0
      head_dim = hp.head_dim_for_layer(il)
      q_dim = hp.n_head * head_dim
      kv_dim = hp.n_head_kv(il) * head_dim
      make_proj = ->(seed : Int32) {
        ML::GGUF::DiffusionGemmaCPU::AttentionProjection.new(
          q: Array(Float32).new(q_dim) { |i| ((((i * 17 + seed) % 257) - 128).to_f32 / 512.0_f32) },
          k: Array(Float32).new(kv_dim) { |i| ((((i * 13 + seed) % 251) - 125).to_f32 / 512.0_f32) },
          v: Array(Float32).new(kv_dim) { |i| ((((i * 19 + seed) % 241) - 120).to_f32 / 512.0_f32) },
          reused_k_as_v: false,
        )
      }
      prompt_projections = (0...4).map { |i| make_proj.call(i + 1) }
      canvas_projections = (0...2).map { |i| make_proj.call(i + 101) }
      mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 4, canvas_len: 2, sliding_window: 3)

      ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL")
      ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"] = "1"
      expected = ML::GGUF::DiffusionGemmaCPU.attention_context_decode_timed(
        prompt_projections, canvas_projections, hp, il, canvas_query_index: 0, mask: mask).context

      ENV["DIFFUSION_GEMMA_CONTEXT_METAL"] = "1"
      ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL_OFF")
      actual = ML::GGUF::DiffusionGemmaCPU.attention_context_decode_timed(
        prompt_projections, canvas_projections, hp, il, canvas_query_index: 0, mask: mask).context
      resident_cache = ML::GGUF::DiffusionGemmaCPU::PromptLayerMetalCache.new(
        prompt_projections, mask.prompt_len, mask.canvas_len, q_dim, kv_dim)
      resident_cache.write_canvas!(canvas_projections)
      resident = ML::GGUF::DiffusionGemmaCPU.attention_context_decode_timed(
        prompt_projections, canvas_projections, hp, il, canvas_query_index: 0, mask: mask, prompt_metal_cache: resident_cache).context

      max_diff = 0.0_f32
      expected.size.times do |i|
        diff = (expected[i] - actual[i]).abs
        max_diff = diff if diff > max_diff
      end
      max_diff.should be < 1.0e-4_f32
      max_resident_diff = 0.0_f32
      expected.size.times do |i|
        diff = (expected[i] - resident[i]).abs
        max_resident_diff = diff if diff > max_resident_diff
      end
      max_resident_diff.should be < 1.0e-4_f32
    ensure
      g.close
      if old_context_metal
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL"] = old_context_metal
      else
        ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL")
      end
      if old_context_metal_off
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"] = old_context_metal_off
      else
        ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL_OFF")
      end
    end
  end

  it "keeps gated batched Metal decode layer parity for multiple canvas rows" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)
    next unless ML::GGUF::Gemma4Metal.available?

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    old_context_metal = ENV["DIFFUSION_GEMMA_CONTEXT_METAL"]?
    old_context_metal_off = ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"]?
    old_batch_rows_off = ENV["DIFFUSION_GEMMA_CONTEXT_METAL_BATCH_ROWS_OFF"]?
    begin
      il = 0
      prompt0 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
      prompt1 = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 2)
      canvas0 = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
      canvas1 = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 1)
      prompt_rows = prompt0 + prompt1
      canvas_rows = canvas0 + canvas1
      mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 2, canvas_len: 2, sliding_window: 3)

      ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL")
      ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"] = "1"
      base_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(w, prompt_rows, mask, max_layers: 1, materialize_final_rows: false)
      base = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections_timed(
        w, il, base_cache.projections_by_layer[0], canvas_rows, mask).rows

      ENV["DIFFUSION_GEMMA_CONTEXT_METAL"] = "1"
      ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL_OFF")
      metal_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(w, prompt_rows, mask, max_layers: 1, materialize_final_rows: false)
      metal = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections_timed(
        w, il, metal_cache.projections_by_layer[0], canvas_rows, mask, metal_cache.metal_cache_by_layer[0]?).rows
      ENV["DIFFUSION_GEMMA_CONTEXT_METAL_BATCH_ROWS_OFF"] = "1"
      metal_unbatched_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(w, prompt_rows, mask, max_layers: 1, materialize_final_rows: false)
      metal_unbatched = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections_timed(
        w, il, metal_unbatched_cache.projections_by_layer[0], canvas_rows, mask, metal_unbatched_cache.metal_cache_by_layer[0]?).rows

      max_diff = 0.0_f32
      base.size.times do |i|
        diff = (base[i] - metal[i]).abs
        max_diff = diff if diff > max_diff
      end
      max_diff.should be < 1.0e-3_f32
      max_unbatched_diff = 0.0_f32
      base.size.times do |i|
        diff = (base[i] - metal_unbatched[i]).abs
        max_unbatched_diff = diff if diff > max_unbatched_diff
      end
      max_unbatched_diff.should be < 1.0e-3_f32
    ensure
      if old_context_metal
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL"] = old_context_metal
      else
        ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL")
      end
      if old_context_metal_off
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"] = old_context_metal_off
      else
        ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL_OFF")
      end
      if old_batch_rows_off
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL_BATCH_ROWS_OFF"] = old_batch_rows_off
      else
        ENV.delete("DIFFUSION_GEMMA_CONTEXT_METAL_BATCH_ROWS_OFF")
      end
    end
  end

  it "decodes canvas rows through a bounded prompt layer cache" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    il = 0
    prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 1, canvas_len: 1, sliding_window: 3)
    prompt_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt_row)[0, 1]
    canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, canvas_row)[0, 1]
    prompt_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      w,
      prompt_row,
      mask,
      max_layers: 1,
      routes_by_layer_by_prompt_row: [[prompt_route]],
    )
    decode_only_prompt_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      w,
      prompt_row,
      mask,
      max_layers: 1,
      routes_by_layer_by_prompt_row: [[prompt_route]],
      materialize_final_rows: false,
    )

    direct = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections(
      w,
      il,
      prompt_cache.projections_by_layer[0],
      canvas_row,
      mask,
      [canvas_route],
    )
    cached_stack = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache(
      w,
      canvas_row,
      mask,
      prompt_cache,
      max_layers: 1,
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    cached_stack.should eq(direct)
    decode_only_stack = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache(
      w,
      canvas_row,
      mask,
      decode_only_prompt_cache,
      max_layers: 1,
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    decode_only_stack.should eq(direct)
    decode_only_prompt_cache.final_rows.should eq(prompt_row)

    expect_raises(ArgumentError, /canvas rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache(w, [0.0_f32], mask, prompt_cache, max_layers: 1)
    end
    expect_raises(ArgumentError, /fewer layers/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache(w, canvas_row, mask, prompt_cache, max_layers: 2)
    end
    expect_raises(ArgumentError, /routes_by_layer_by_canvas_row/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache(
        w,
        canvas_row,
        mask,
        prompt_cache,
        max_layers: 1,
        routes_by_layer_by_canvas_row: [] of Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)),
      )
    end
  end

  it "computes bounded canvas predictions after cached decode" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    il = 0
    prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 1, canvas_len: 1, sliding_window: 3)
    prompt_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt_row)[0, 1]
    canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, canvas_row)[0, 1]
    prompt_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      w,
      prompt_row,
      mask,
      max_layers: 1,
      routes_by_layer_by_prompt_row: [[prompt_route]],
    )

    candidate_rows = [[0, 1, 2]]
    predictions = ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_predictions(
      w,
      canvas_row,
      mask,
      prompt_cache,
      candidate_rows,
      max_layers: 1,
      sample_us: [0.5_f32],
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    hidden_rows = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache(
      w,
      canvas_row,
      mask,
      prompt_cache,
      max_layers: 1,
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    expected = ML::GGUF::DiffusionGemmaCPU.bounded_denoise_prediction(w, hidden_rows[0, hp.n_embd], candidate_rows[0], sample_u: 0.5_f32)
    predictions.size.should eq(1)
    predictions[0].candidate_token_ids.should eq(expected.candidate_token_ids)
    predictions[0].logits.should eq(expected.logits)
    predictions[0].probabilities.should eq(expected.probabilities)
    predictions[0].sampled_token_id.should eq(expected.sampled_token_id)

    step = ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_step(
      w,
      [0],
      canvas_row,
      mask,
      prompt_cache,
      candidate_rows,
      entropy_bound: 0.0_f32,
      max_layers: 1,
      sample_us: [0.5_f32],
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    step.predictions[0].logits.should eq(expected.logits)
    step.accepted.should eq([true])
    step.updated_canvas_tokens.should eq([expected.sampled_token_id])
    step.updated_canvas_rows.not_nil![0, hp.n_embd].should eq(ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, expected.sampled_token_id))

    loop = ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_loop(
      w,
      [0],
      canvas_row,
      mask,
      prompt_cache,
      ML::GGUF::DiffusionGemmaCPU.current_token_candidate_steps([0], hp.vocab_size, 1),
      entropy_bound: 0.0_f32,
      stability_threshold: 1,
      max_layers: 1,
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    loop.steps_run.should eq(1)
    loop.converged.should be_true
    loop.stop_reason.should eq("converged")
    loop.accepted_token_count.should eq(1)
    loop.step_traces.size.should eq(1)
    loop.step_traces[0].step.should eq(0)
    loop.step_traces[0].prediction_count.should eq(1)
    loop.step_traces[0].accepted_count.should eq(1)
    loop.step_traces[0].total_candidate_tokens.should eq(1)
    loop.step_traces[0].max_candidate_tokens.should eq(1)
    loop.step_traces[0].mean_candidate_tokens.should eq(1.0_f32)
    loop.step_traces[0].prediction_ms.should be > 0.0
    loop.step_traces[0].decode_stack_ms.should be > 0.0
    loop.step_traces[0].decode_qkv_ms.should be >= 0.0
    loop.step_traces[0].decode_context_ms.should be >= 0.0
    loop.step_traces[0].decode_context_score_ms.should be >= 0.0
    loop.step_traces[0].decode_context_softmax_ms.should be >= 0.0
    loop.step_traces[0].decode_context_value_ms.should be >= 0.0
    (
      loop.step_traces[0].decode_context_score_ms +
        loop.step_traces[0].decode_context_softmax_ms +
        loop.step_traces[0].decode_context_value_ms
    ).should be <= loop.step_traces[0].decode_context_ms
    loop.step_traces[0].decode_attention_out_ms.should be >= 0.0
    loop.step_traces[0].decode_shared_ffn_ms.should be >= 0.0
    loop.step_traces[0].decode_moe_ffn_ms.should be >= 0.0
    loop.step_traces[0].decode_combine_scale_ms.should be >= 0.0
    loop.step_traces[0].output_head_ms.should be >= 0.0
    loop.step_traces[0].prediction_ms.should be >= loop.step_traces[0].decode_stack_ms + loop.step_traces[0].output_head_ms
    loop.step_traces[0].update_ms.should be >= 0.0
    loop.step_traces[0].regenerate_ms.should be >= 0.0
    loop.step_traces[0].proposal_ms.should eq(0.0)
    loop.summary.steps_run.should eq(1)
    loop.summary.converged.should be_true
    loop.summary.stop_reason.should eq("converged")
    loop.summary.prediction_count.should eq(1)
    loop.summary.accepted_count.should eq(1)
    loop.summary.total_candidate_tokens.should eq(1)
    loop.summary.max_candidate_tokens.should eq(1)
    loop.summary.mean_candidate_tokens.should eq(1.0_f32)
    loop.summary.acceptance_rate.should eq(1.0_f32)
    loop.final_canvas_tokens.should eq([0])
    loop.final_canvas_rows.not_nil![0, hp.n_embd].should eq(ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0))

    adaptive = ML::GGUF::DiffusionGemmaCPU.decode_canvas_adaptive_bounded_loop(
      w,
      [0],
      canvas_row,
      mask,
      prompt_cache,
      ML::GGUF::DiffusionGemmaCPU.current_token_candidate_rows([0], hp.vocab_size),
      entropy_bound: 0.0_f32,
      stability_threshold: 1,
      max_steps: 1,
      proposal_top_k: 1,
      max_layers: 1,
      sample_us_by_step_by_canvas_row: ML::GGUF::DiffusionGemmaCPU.sample_u_steps(7, 1, 1),
      routes_by_layer_by_canvas_row: [[canvas_route]],
    )
    adaptive.steps_run.should eq(1)
    adaptive.converged.should be_true
    adaptive.stop_reason.should eq("converged")
    adaptive.step_traces[0].accepted_count.should eq(loop.step_traces[0].accepted_count)
    adaptive.step_traces[0].total_candidate_tokens.should eq(loop.step_traces[0].total_candidate_tokens)
    adaptive.step_traces[0].prediction_ms.should be > 0.0
    adaptive.step_traces[0].decode_stack_ms.should be > 0.0
    adaptive.final_canvas_tokens.should eq(loop.final_canvas_tokens)
    adaptive.final_canvas_rows.not_nil!.should eq(loop.final_canvas_rows.not_nil!)

    sc_loop = ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_loop(
      w,
      [0],
      canvas_row,
      mask,
      prompt_cache,
      ML::GGUF::DiffusionGemmaCPU.current_token_candidate_steps([0], hp.vocab_size, 1),
      entropy_bound: 0.0_f32,
      stability_threshold: 1,
      max_layers: 1,
      routes_by_layer_by_canvas_row: [[canvas_route]],
      use_sparse_self_conditioning: true,
    )
    sc_pred = sc_loop.updates[0].predictions[0]
    expected_sc_row = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_prediction_self_conditioning(w, [0], [sc_pred])
    sc_loop.final_canvas_rows.not_nil!.should eq(expected_sc_row)
    sc_loop.final_canvas_rows.not_nil!.should_not eq(loop.final_canvas_rows.not_nil!)

    expect_raises(ArgumentError, /candidate rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_predictions(w, canvas_row, mask, prompt_cache, [] of Array(Int32), max_layers: 1)
    end
    expect_raises(ArgumentError, /sample_us/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_predictions(w, canvas_row, mask, prompt_cache, candidate_rows, max_layers: 1, sample_us: [] of Float32)
    end
    expect_raises(ArgumentError, /canvas token count/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_step(w, [] of Int32, canvas_row, mask, prompt_cache, candidate_rows, entropy_bound: 0.0_f32, max_layers: 1)
    end
    expect_raises(ArgumentError, /max_steps/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_adaptive_bounded_loop(w, [0], canvas_row, mask, prompt_cache, candidate_rows, entropy_bound: 0.0_f32, stability_threshold: 1, max_steps: 0, proposal_top_k: 1, max_layers: 1)
    end
    expect_raises(ArgumentError, /proposal_top_k/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_adaptive_bounded_loop(w, [0], canvas_row, mask, prompt_cache, candidate_rows, entropy_bound: 0.0_f32, stability_threshold: 1, max_steps: 1, proposal_top_k: 0, max_layers: 1)
    end
    expect_raises(ArgumentError, /sample_us step/) do
      ML::GGUF::DiffusionGemmaCPU.decode_canvas_adaptive_bounded_loop(w, [0], canvas_row, mask, prompt_cache, candidate_rows, entropy_bound: 0.0_f32, stability_threshold: 1, max_steps: 2, proposal_top_k: 1, max_layers: 1, sample_us_by_step_by_canvas_row: [[0.0_f32]])
    end
  end

  it "runs a small unified layer over prompt and canvas rows" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    il = 0
    prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    rows = prompt_row + canvas_row
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 1, canvas_len: 1, sliding_window: 3)
    prompt_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt_row)[0, 1]
    canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, canvas_row)[0, 1]

    result_rows = ML::GGUF::DiffusionGemmaCPU.layer_forward_unified_rows(w, il, rows, mask, [prompt_route, canvas_route])
    result_rows.size.should eq(2 * hp.n_embd)
    result_rows.all? { |v| v.finite? }.should be_true
    result_rows[0, hp.n_embd].should_not eq(prompt_row)
    result_rows[hp.n_embd, hp.n_embd].should_not eq(canvas_row)

    projections = [
      ML::GGUF::DiffusionGemmaCPU.attention_project_normed(w, il, prompt_row, 0),
      ML::GGUF::DiffusionGemmaCPU.attention_project_normed(w, il, canvas_row, 1),
    ]
    prompt_context = ML::GGUF::DiffusionGemmaCPU.attention_context_unified(projections, hp, il, query_pos: 0, mask: mask)
    canvas_context = ML::GGUF::DiffusionGemmaCPU.attention_context_unified(projections, hp, il, query_pos: 1, mask: mask)
    expected_prompt = ML::GGUF::DiffusionGemmaCPU.layer_output_from_context(w, il, prompt_row, prompt_context, canvas: false, routes: prompt_route)
    expected_canvas = ML::GGUF::DiffusionGemmaCPU.layer_output_from_context(w, il, canvas_row, canvas_context, canvas: true, routes: canvas_route)
    result_rows[0, hp.n_embd].should eq(expected_prompt)
    result_rows[hp.n_embd, hp.n_embd].should eq(expected_canvas)

    expect_raises(ArgumentError, /layer rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_unified_rows(w, il, [0.0_f32], mask)
    end
    expect_raises(ArgumentError, /routes_by_row/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_unified_rows(w, il, rows, mask, [prompt_route])
    end
  end

  it "matches prompt-KV decode canvas rows to the unified canvas path" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    il = 0
    prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    rows = prompt_row + canvas_row
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 1, canvas_len: 1, sliding_window: 3)
    prompt_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, prompt_row)[0, 1]
    canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, canvas_row)[0, 1]

    unified = ML::GGUF::DiffusionGemmaCPU.layer_forward_unified_rows(w, il, rows, mask, [prompt_route, canvas_route])
    decoded = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows(w, il, prompt_row, canvas_row, mask, [canvas_route])
    decoded.size.should eq(hp.n_embd)
    decoded.all? { |v| v.finite? }.should be_true
    decoded.should eq(unified[hp.n_embd, hp.n_embd])

    expect_raises(ArgumentError, /prompt rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows(w, il, [0.0_f32], canvas_row, mask)
    end
    expect_raises(ArgumentError, /canvas rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows(w, il, prompt_row, [0.0_f32], mask)
    end
    expect_raises(ArgumentError, /routes_by_canvas_row/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows(w, il, prompt_row, canvas_row, mask, [] of Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute))
    end
  end

  it "reuses cached prompt projections for decode canvas rows" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    il = 0
    prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(w, 1)
    canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 1, canvas_len: 1, sliding_window: 3)
    canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(w, il, canvas_row)[0, 1]

    prompt_projections = ML::GGUF::DiffusionGemmaCPU.prompt_attention_projections(w, il, prompt_row, mask)
    uncached = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows(w, il, prompt_row, canvas_row, mask, [canvas_route])
    cached = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections(w, il, prompt_projections, canvas_row, mask, [canvas_route])
    cached.should eq(uncached)

    expect_raises(ArgumentError, /prompt rows size mismatch/) do
      ML::GGUF::DiffusionGemmaCPU.prompt_attention_projections(w, il, [0.0_f32], mask)
    end
    expect_raises(ArgumentError, /prompt projection count/) do
      ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections(w, il, [] of ML::GGUF::DiffusionGemmaCPU::AttentionProjection, canvas_row, mask, [canvas_route])
    end
  end

  it "computes bounded output logits from selected token rows" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    zero = Array(Float32).new(hp.n_embd, 0.0_f32)

    ML::GGUF::DiffusionGemmaCPU.output_logits_for_tokens(w, zero, [0, 1, 2]).should eq([0.0_f32, 0.0_f32, 0.0_f32])

    hidden = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    logits = ML::GGUF::DiffusionGemmaCPU.output_logits_for_tokens(w, hidden, [0, 1, 2])
    logits.size.should eq(3)
    logits.all? { |v| v.finite? }.should be_true
    logits.any? { |v| v.abs > 0.000001_f32 }.should be_true

    expect_raises(ArgumentError, /token_ids/) do
      ML::GGUF::DiffusionGemmaCPU.output_logits_for_tokens(w, hidden, [] of Int32)
    end
    expect_raises(ArgumentError, /token id/) do
      ML::GGUF::DiffusionGemmaCPU.output_logits_for_tokens(w, hidden, [hp.vocab_size])
    end
  end

  it "computes bounded denoise candidate distributions and entropy-bound acceptance" do
    pred = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction(
      candidate_token_ids: [10, 20],
      raw_logits: [0.0_f32, 2.0_f32],
      sample_u: 0.5_f32,
    )
    pred.argmax_token_id.should eq(20)
    pred.sampled_token_id.should eq(20)
    pred.probabilities.sum.should be_close(1.0_f32, 0.000001_f32)
    pred.entropy.should be < Math.log(2.0).to_f32

    uniform = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction(
      candidate_token_ids: [0, 1, 2],
      raw_logits: [0.0_f32, 0.0_f32, 0.0_f32],
      sample_u: 0.5_f32,
    )
    uniform.argmax_token_id.should eq(0)
    uniform.sampled_token_id.should eq(1)
    uniform.entropy.should be_close(Math.log(3.0).to_f32, 0.000001_f32)

    ML::GGUF::DiffusionGemmaCPU.entropy_bound_accept([0.2_f32, 0.05_f32, 0.3_f32], 0.1_f32).should eq([true, true, false])
    ML::GGUF::DiffusionGemmaCPU.update_canvas_token([7, 8, 9], 1, 20).should eq([7, 20, 9])
    ML::GGUF::DiffusionGemmaCPU.current_token_candidate_rows([3, 1], 10).should eq([[3], [1]])
    ML::GGUF::DiffusionGemmaCPU.generated_candidate_rows([8, 9], 3, 10).should eq([[0, 8, 9], [0, 1, 9]])
    ML::GGUF::DiffusionGemmaCPU.merge_candidate_rows([3, 1], [[4, 3, 2], [1, 9]], 10).should eq([[2, 3, 4], [1, 9]])
    candidate_steps = ML::GGUF::DiffusionGemmaCPU.current_token_candidate_steps([5], 10, 2)
    candidate_steps.should eq([[[5]], [[5]]])
    candidate_steps[0][0] << 6
    candidate_steps[1][0].should eq([5])
    ML::GGUF::DiffusionGemmaCPU.merge_candidate_steps([1], [[[3]], [[2, 1]]], 10).should eq([[[1, 3]], [[1, 2]]])

    proposal_pred = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([2, 4, 6], [0.0_f32, 2.0_f32, 2.0_f32])
    ML::GGUF::DiffusionGemmaCPU.top_k_prediction_tokens(proposal_pred, 2).should eq([4, 6])
    ML::GGUF::DiffusionGemmaCPU.top_k_prediction_tokens(proposal_pred, 10).should eq([4, 6, 2])
    ML::GGUF::DiffusionGemmaCPU.prediction_proposal_rows([proposal_pred], 2).should eq([[4, 6]])
    ML::GGUF::DiffusionGemmaCPU.next_candidate_rows_from_predictions([5], [proposal_pred], 10, 2).should eq([[4, 5, 6]])
    repeated = ML::GGUF::DiffusionGemmaCPU.repeated_candidate_steps_from_predictions([5], [proposal_pred], 10, 2, 2)
    repeated.should eq([[[4, 5, 6]], [[4, 5, 6]]])
    repeated[0][0] << 7
    repeated[1][0].should eq([4, 5, 6])

    first = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([10, 20], [0.0_f32, 0.0_f32], sample_u: 0.75_f32)
    second = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([30, 40], [0.0_f32, 0.0_f32], sample_u: 0.75_f32)
    sampled_update = ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_predictions([0, 0], [first, second], 0.0_f32)
    sampled_update.accepted.should eq([true, false])
    sampled_update.updated_canvas_tokens.should eq([20, 0])
    argmax_update = ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_predictions([0], [first], 0.0_f32, use_sampled_token: false)
    argmax_update.updated_canvas_tokens.should eq([10])

    first_step = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([10, 20], [0.0_f32, 0.0_f32], sample_u: 0.75_f32)
    stable_step = ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([20, 30], [2.0_f32, 0.0_f32], sample_u: 0.0_f32)
    loop = ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_prediction_steps([0], [[first_step], [stable_step]], 10.0_f32, 1)
    loop.final_canvas_tokens.should eq([20])
    loop.stable_counts.should eq([1])
    loop.steps_run.should eq(2)
    loop.converged.should be_true
    loop.stop_reason.should eq("converged")
    loop.accepted_token_count.should eq(2)
    loop.step_traces.map(&.step).should eq([0, 1])
    loop.step_traces.map(&.prediction_count).should eq([1, 1])
    loop.step_traces.map(&.accepted_count).should eq([1, 1])
    loop.step_traces.map(&.total_candidate_tokens).should eq([2, 2])
    loop.step_traces.map(&.max_candidate_tokens).should eq([2, 2])
    loop.step_traces[0].mean_candidate_tokens.should eq(2.0_f32)
    loop.step_traces[0].mean_entropy.should be_close(first_step.entropy, 1e-6)
    loop.summary.prediction_count.should eq(2)
    loop.summary.accepted_count.should eq(2)
    loop.summary.total_candidate_tokens.should eq(4)
    loop.summary.max_candidate_tokens.should eq(2)
    loop.summary.mean_candidate_tokens.should eq(2.0_f32)
    loop.summary.mean_entropy.should be_close((first_step.entropy + stable_step.entropy) / 2.0_f32, 1e-6)
    loop.summary.acceptance_rate.should eq(1.0_f32)

    exhausted_loop = ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_prediction_steps([0], [[first_step]], 0.0_f32, 2)
    exhausted_loop.converged.should be_false
    exhausted_loop.stop_reason.should eq("prediction_budget")
    exhausted_loop.step_traces[0].accepted_count.should eq(1)
    exhausted_loop.summary.stop_reason.should eq("prediction_budget")
    exhausted_loop.summary.acceptance_rate.should eq(1.0_f32)

    weighted_loop = ML::GGUF::DiffusionGemmaCPU::BoundedDenoiseLoopResult.new(
      [0],
      nil,
      [] of ML::GGUF::DiffusionGemmaCPU::BoundedCanvasUpdate,
      [] of Int32,
      2,
      false,
      [
        ML::GGUF::DiffusionGemmaCPU::BoundedDenoiseStepTrace.new(0, 1, 1, 2, 2, 2.0_f32, 1.0_f32),
        ML::GGUF::DiffusionGemmaCPU::BoundedDenoiseStepTrace.new(1, 3, 1, 12, 5, 4.0_f32, 3.0_f32),
      ],
      "step_budget",
    )
    weighted_loop.summary.prediction_count.should eq(4)
    weighted_loop.summary.accepted_count.should eq(2)
    weighted_loop.summary.total_candidate_tokens.should eq(14)
    weighted_loop.summary.max_candidate_tokens.should eq(5)
    weighted_loop.summary.mean_candidate_tokens.should eq(3.5_f32)
    weighted_loop.summary.mean_entropy.should eq(2.5_f32)
    weighted_loop.summary.acceptance_rate.should eq(0.5_f32)

    ML::GGUF::DiffusionGemmaCPU.advance_stability_counts([1], [2], [true], [7]).should eq([0])
    ML::GGUF::DiffusionGemmaCPU.advance_stability_counts([1], [1], [false], [7]).should eq([0])
    ML::GGUF::DiffusionGemmaCPU.advance_stability_counts([1], [1], [true], [7]).should eq([8])

    expect_raises(ArgumentError, /strictly increasing/) do
      ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([2, 1], [0.0_f32, 0.0_f32])
    end
    expect_raises(ArgumentError, /sample_u/) do
      ML::GGUF::DiffusionGemmaCPU.bounded_candidate_prediction([1], [0.0_f32], sample_u: 1.0_f32)
    end
    expect_raises(ArgumentError, /trace step/) do
      ML::GGUF::DiffusionGemmaCPU.bounded_denoise_step_trace(-1, sampled_update)
    end
    expect_raises(ArgumentError, /trace accepted/) do
      bad_trace_update = ML::GGUF::DiffusionGemmaCPU::BoundedCanvasUpdate.new([0], [] of Bool, [first])
      ML::GGUF::DiffusionGemmaCPU.bounded_denoise_step_trace(0, bad_trace_update)
    end
    expect_raises(ArgumentError, /entropy_bound/) do
      ML::GGUF::DiffusionGemmaCPU.entropy_bound_accept([0.1_f32], -0.1_f32)
    end
    expect_raises(ArgumentError, /canvas prediction count/) do
      ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_predictions([0], [first, second], 0.0_f32)
    end
    expect_raises(ArgumentError, /candidate token id/) do
      ML::GGUF::DiffusionGemmaCPU.current_token_candidate_rows([10], 10)
    end
    expect_raises(ArgumentError, /proposal rows/) do
      ML::GGUF::DiffusionGemmaCPU.merge_candidate_rows([1], [] of Array(Int32), 10)
    end
    expect_raises(ArgumentError, /candidate steps/) do
      ML::GGUF::DiffusionGemmaCPU.current_token_candidate_steps([1], 10, 0)
    end
    expect_raises(ArgumentError, /proposal steps/) do
      ML::GGUF::DiffusionGemmaCPU.merge_candidate_steps([1], [] of Array(Array(Int32)), 10)
    end
    expect_raises(ArgumentError, /top-k/) do
      ML::GGUF::DiffusionGemmaCPU.top_k_prediction_tokens(proposal_pred, 0)
    end
    expect_raises(ArgumentError, /proposal rows/) do
      ML::GGUF::DiffusionGemmaCPU.prediction_proposal_rows([] of ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction, 1)
    end
    expect_raises(ArgumentError, /prediction rows/) do
      ML::GGUF::DiffusionGemmaCPU.next_candidate_rows_from_predictions([1, 2], [proposal_pred], 10, 1)
    end
    expect_raises(ArgumentError, /prediction steps/) do
      ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_prediction_steps([0], [] of Array(ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction), 0.0_f32, 1)
    end
    expect_raises(ArgumentError, /stability_threshold/) do
      ML::GGUF::DiffusionGemmaCPU.apply_entropy_bound_prediction_steps([0], [[first]], 0.0_f32, 0)
    end
  end

  it "computes a model-backed bounded denoise prediction" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams
    zero = Array(Float32).new(hp.n_embd, 0.0_f32)

    pred = ML::GGUF::DiffusionGemmaCPU.bounded_denoise_prediction(w, zero, [0, 1, 2], sample_u: 0.5_f32)
    pred.candidate_token_ids.should eq([0, 1, 2])
    pred.logits.should eq([0.0_f32, 0.0_f32, 0.0_f32])
    pred.probabilities.each { |p| p.should be_close(1.0_f32 / 3.0_f32, 0.000001_f32) }
    pred.argmax_token_id.should eq(0)
    pred.sampled_token_id.should eq(1)

    hidden = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(w, 0)
    nonzero = ML::GGUF::DiffusionGemmaCPU.bounded_denoise_prediction(w, hidden, [0, 1, 2])
    nonzero.logits.all? { |v| v.finite? }.should be_true
    nonzero.entropy.should be >= 0.0_f32
  end
end

describe ML::GGUF::DiffusionGemmaWeights do
  it "maps local DiffusionGemma tensors structurally without dequantizing large weights" do
    pending!("DiffusionGemma 26B GGUF not found") unless File.exists?(DIFFUSION_GEMMA_26B_Q4KM)

    w = ML::GGUF::DiffusionGemmaWeights.from_gguf(DIFFUSION_GEMMA_26B_Q4KM)
    hp = w.hparams

    w.token_embd.in_dim.should eq(2816)
    w.token_embd.out_dim.should eq(262_144)
    w.token_embd.type.name.should eq("Q6_K")
    w.self_conditioning.pre_norm.size.should eq(2816)
    w.self_conditioning.gate_qw.type.name.should eq("Q4_K")
    w.self_conditioning.up_qw.type.name.should eq("Q4_K")
    w.self_conditioning.down_qw.in_dim.should eq(2112)
    w.self_conditioning.down_qw.out_dim.should eq(2816)
    w.self_conditioning.down_qw.type.name.should eq("Q5_0")
    w.layers.size.should eq(30)

    swa = w.layers[0]
    hp.sliding_window?(0).should be_true
    swa.attn_q_qw.in_dim.should eq(2816)
    swa.attn_q_qw.out_dim.should eq(4096)
    swa.attn_k_qw.out_dim.should eq(2048)
    swa.attn_v_qw.should_not be_nil
    swa.explicit_v?.should be_true
    swa.reuse_k_as_v?.should be_false
    swa.attn_output_qw.in_dim.should eq(4096)
    swa.attn_output_qw.out_dim.should eq(2816)
    swa.encoder_layer_output_scale.size.should eq(1)
    swa.layer_output_scale.size.should eq(1)
    swa.ffn_gate_inp_qw.in_dim.should eq(2816)
    swa.ffn_gate_inp_qw.out_dim.should eq(128)
    swa.ffn_gate_inp_scale.size.should eq(2816)
    swa.ffn_gate_up_exps_qw.not_nil!.in_dim.should eq(2816)
    swa.ffn_gate_up_exps_qw.not_nil!.out_dim.should eq(1408 * 128)
    swa.ffn_down_exps_qw.in_dim.should eq(704)
    swa.ffn_down_exps_qw.out_dim.should eq(2816 * 128)
    swa.ffn_down_exps_scale.size.should eq(128)
    swa.pre_ffw_norm_2.size.should eq(2816)
    swa.post_ffw_norm_1.size.should eq(2816)
    swa.post_ffw_norm_2.size.should eq(2816)
    swa.combined_gate_up_experts?.should be_true

    full = w.layers[5]
    hp.full_attention?(5).should be_true
    full.attn_q_qw.out_dim.should eq(8192)
    full.attn_k_qw.out_dim.should eq(1024)
    full.attn_v_qw.should be_nil
    full.explicit_v?.should be_false
    full.reuse_k_as_v?.should be_true
    full.attn_output_qw.in_dim.should eq(8192)
    full.attn_output_qw.out_dim.should eq(2816)

    req = ML::GGUF::DiffusionGemmaRequest.with_blank_canvas([1, 2, 3], hp, mask_token_id: 0)
    plan = ML::GGUF::DiffusionGemmaRuntimePlan.new(hp, req)
    plan.max_length.should eq(259)
    plan.self_conditioning_logits_bytes.should eq(256_i64 * 262_144_i64 * 4_i64)
    plan.kv_dim_for_layer(hp, 0).should eq(8 * 256)
    plan.kv_dim_for_layer(hp, 5).should eq(2 * 512)
    plan.layer_scale_for_pos(w, 0, 0).should eq(swa.encoder_layer_output_scale[0])
    plan.layer_scale_for_pos(w, 0, req.canvas_start).should eq(swa.layer_output_scale[0])
  end
end
