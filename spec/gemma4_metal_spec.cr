require "./spec_helper"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/qwen35_metal"

GEMMA4_METAL_12B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

def gemma4_metal_cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  na = 0.0
  nb = 0.0
  a.each_with_index do |av, i|
    bv = b[i]
    dot += av.to_f64 * bv.to_f64
    na += av.to_f64 * av.to_f64
    nb += bv.to_f64 * bv.to_f64
  end
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

def gemma4_metal_max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  max = 0.0_f32
  a.each_with_index do |av, i|
    diff = (av - b[i]).abs
    max = diff if diff > max
  end
  max
end

def gemma4_metal_expect_close(label : String, cpu : Array(Float32), gpu : Array(Float32)) : Nil
  gpu.size.should eq(cpu.size)
  cos = gemma4_metal_cosine(cpu, gpu)
  diff = gemma4_metal_max_abs_diff(cpu, gpu)
  scale = cpu.map(&.abs).max
  puts "  [#{label}] cos=#{cos.round(8)}, max|d|=#{diff}"
  cos.should be >= 0.9999
  diff.should be < 0.02_f32 * scale
end

def gemma4_cpu_norm_rope_projection(w : ML::GGUF::Gemma4Weights,
                                    il : Int32,
                                    token_id : Int32,
                                    pos : Int32) : ML::GGUF::Gemma4CPU::AttentionProjection
  x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, token_id)
  lw = w.layers[il]
  proj = ML::GGUF::Gemma4CPU.attention_project_normed(lw, x, w.hparams, il)
  ML::GGUF::Gemma4CPU.apply_rope_to_qk!(proj, w.hparams, il, pos, w.hparams.full_attention?(il) ? w.rope_freqs : nil)
  proj
end

def gemma4_expect_context_parity(w : ML::GGUF::Gemma4Weights,
                                 il : Int32,
                                 token_ids : Array(Int32),
                                 label : String) : Nil
  max_seq = 8
  hp = w.hparams
  kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
  cpu_state = ML::GGUF::Gemma4CPU::LayerState.new
  metal_k_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
  metal_v_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)

  token_ids.each_with_index do |token_id, pos|
    proj = gemma4_cpu_norm_rope_projection(w, il, token_id, pos)
    cpu_ctx = ML::GGUF::Gemma4CPU.attention_context_from_projection!(proj, hp, il, pos, cpu_state, max_seq)
    metal = ML::GGUF::Gemma4Metal.attention_context_from_projection(
      proj, hp, il, pos, max_seq, metal_k_cache, metal_v_cache).not_nil!
    metal_k_cache = metal[:k_cache]
    metal_v_cache = metal[:v_cache]

    if pos == token_ids.size - 1
      gemma4_metal_expect_close("#{label}_ctx", cpu_ctx, metal[:context])
      cpu_k = cpu_state.k_cache.not_nil![0, metal_k_cache.size]
      cpu_v = cpu_state.v_cache.not_nil![0, metal_v_cache.size]
      gemma4_metal_expect_close("#{label}_k_cache", cpu_k, metal_k_cache)
      gemma4_metal_expect_close("#{label}_v_cache", cpu_v, metal_v_cache)
    end
  end
end

def gemma4_cpu_context_for_tokens(w : ML::GGUF::Gemma4Weights,
                                  il : Int32,
                                  token_ids : Array(Int32)) : Array(Float32)
  max_seq = 8
  state = ML::GGUF::Gemma4CPU::LayerState.new
  ctx = [] of Float32
  token_ids.each_with_index do |token_id, pos|
    proj = gemma4_cpu_norm_rope_projection(w, il, token_id, pos)
    ctx = ML::GGUF::Gemma4CPU.attention_context_from_projection!(proj, w.hparams, il, pos, state, max_seq)
  end
  ctx
end

def gemma4_expect_attn_output_projection_parity(w : ML::GGUF::Gemma4Weights,
                                                il : Int32,
                                                label : String) : Nil
  ctx = gemma4_cpu_context_for_tokens(w, il, [42, 43, 44])
  lw = w.layers[il]
  cpu = ML::GGUF::Gemma4CPU.matmul(lw.attn_output_qw, ctx)
  gpu = ML::GGUF::Qwen35Metal.matmul(lw.attn_output_qw, ctx, 1).not_nil!

  gemma4_metal_expect_close(label, cpu, gpu)
end

def gemma4_cpu_layer_tail(w : ML::GGUF::Gemma4Weights,
                          il : Int32,
                          x : Array(Float32),
                          attn_projected : Array(Float32)) : Array(Float32)
  lw = w.layers[il]
  hp = w.hparams
  attn_normed = ML::GGUF::Gemma4CPU.rms_norm(attn_projected, lw.post_attention_norm, hp.rms_eps)
  attn_out = Array(Float32).new(hp.n_embd) { |i| x[i] + attn_normed[i] }
  ffn_in = ML::GGUF::Gemma4CPU.rms_norm(attn_out, lw.ffn_norm, hp.rms_eps)
  up = ML::GGUF::Gemma4CPU.matmul(lw.ffn_up_qw, ffn_in)
  gate = ML::GGUF::Gemma4CPU.matmul(lw.ffn_gate_qw, ffn_in)
  gate.size.times { |i| gate[i] = ML::GGUF::Gemma4CPU.gelu(gate[i]) * up[i] }
  ffn = ML::GGUF::Gemma4CPU.matmul(lw.ffn_down_qw, gate)
  ffn = ML::GGUF::Gemma4CPU.rms_norm(ffn, lw.post_ffw_norm, hp.rms_eps)
  out = Array(Float32).new(hp.n_embd) { |i| attn_out[i] + ffn[i] }
  if scale = lw.layer_output_scale.first?
    out.size.times { |i| out[i] *= scale }
  end
  out
end

def gemma4_expect_layer_tail_parity(w : ML::GGUF::Gemma4Weights,
                                    il : Int32,
                                    label : String) : Nil
  x = ML::GGUF::Gemma4CPU.scaled_embedding_lookup(w, 42)
  state = ML::GGUF::Gemma4CPU::State.new(w.hparams, 8)
  attn_projected = ML::GGUF::Gemma4CPU.attention_projected_output(w, il, x, 0, state)
  cpu = gemma4_cpu_layer_tail(w, il, x, attn_projected)
  gpu = ML::GGUF::Gemma4Metal.layer_tail(x, attn_projected, w.layers[il], w.hparams).not_nil!

  gemma4_metal_expect_close(label, cpu, gpu)
end

def gemma4_expect_forward_layer_parity(w : ML::GGUF::Gemma4Weights,
                                       il : Int32,
                                       label : String) : Nil
  max_seq = 8
  hp = w.hparams
  kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
  cpu_state = ML::GGUF::Gemma4CPU::State.new(hp, max_seq)
  metal_k_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
  metal_v_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
  cpu_out = [] of Float32
  metal_out = [] of Float32

  [42, 43, 44].each_with_index do |token_id, pos|
    x = ML::GGUF::Gemma4CPU.scaled_embedding_lookup(w, token_id)
    cpu_out = ML::GGUF::Gemma4CPU.forward_layer(w, il, x, pos, cpu_state)
    metal = ML::GGUF::Gemma4Metal.forward_layer(w, il, x, pos, max_seq, metal_k_cache, metal_v_cache).not_nil!
    metal_out = metal[:out]
    metal_k_cache = metal[:k_cache]
    metal_v_cache = metal[:v_cache]
  end

  gemma4_metal_expect_close("#{label}_out", cpu_out, metal_out)
  cpu_k = cpu_state.layers[il].k_cache.not_nil![0, metal_k_cache.size]
  cpu_v = cpu_state.layers[il].v_cache.not_nil![0, metal_v_cache.size]
  gemma4_metal_expect_close("#{label}_k_cache", cpu_k, metal_k_cache)
  gemma4_metal_expect_close("#{label}_v_cache", cpu_v, metal_v_cache)
end

describe "Gemma4 Metal primitives" do
  pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_METAL_12B_Q4KM)
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "dequantizes a Q6_K token embedding row on Metal like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    w.token_embd.type.q6_k?.should be_true

    token_id = 42
    cpu = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, token_id)
    gpu = ML::GGUF::Qwen35Metal.embedding_q6k_from_token_id(w.token_embd, token_id).not_nil!

    gpu.size.should eq(w.hparams.n_embd)
    cos = gemma4_metal_cosine(cpu, gpu)
    diff = gemma4_metal_max_abs_diff(cpu, gpu)
    puts "  [gemma4_q6k_embed] cos=#{cos.round(8)}, max|d|=#{diff}"
    cos.should be >= 0.999999
    diff.should be < 1.0e-6_f32
  end

  it "zeros an out-of-range Q6_K token embedding request" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    gpu = ML::GGUF::Qwen35Metal.embedding_q6k_from_token_id(w.token_embd, w.token_embd.out_dim).not_nil!

    gpu.size.should eq(w.hparams.n_embd)
    gpu.all? { |v| v == 0.0_f32 }.should be_true
  end

  it "projects SWA-layer Q/K/V with matmul_many like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    lw = w.layers[0]
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, lw.attn_norm, w.hparams.rms_eps)
    v_qw = lw.attn_v_qw.not_nil!

    cpu = [
      ML::GGUF::Gemma4CPU.matmul(lw.attn_q_qw, x_norm),
      ML::GGUF::Gemma4CPU.matmul(lw.attn_k_qw, x_norm),
      ML::GGUF::Gemma4CPU.matmul(v_qw, x_norm),
    ]
    gpu = ML::GGUF::Qwen35Metal.matmul_many([lw.attn_q_qw, lw.attn_k_qw, v_qw], x_norm).not_nil!

    gemma4_metal_expect_close("gemma4_swa_q_proj", cpu[0], gpu[0])
    gemma4_metal_expect_close("gemma4_swa_k_proj", cpu[1], gpu[1])
    gemma4_metal_expect_close("gemma4_swa_v_proj", cpu[2], gpu[2])
  end

  it "projects full-attention Q/K and preserves the K-as-V structural boundary" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    lw = w.layers[5]
    lw.attn_v_qw.should be_nil
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, lw.attn_norm, w.hparams.rms_eps)

    cpu = [
      ML::GGUF::Gemma4CPU.matmul(lw.attn_q_qw, x_norm),
      ML::GGUF::Gemma4CPU.matmul(lw.attn_k_qw, x_norm),
    ]
    gpu = ML::GGUF::Qwen35Metal.matmul_many([lw.attn_q_qw, lw.attn_k_qw], x_norm).not_nil!

    gemma4_metal_expect_close("gemma4_full_q_proj", cpu[0], gpu[0])
    gemma4_metal_expect_close("gemma4_full_k_proj", cpu[1], gpu[1])
  end

  it "normalizes and RoPEs a SWA projection like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    lw = w.layers[0]
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, lw.attn_norm, w.hparams.rms_eps)
    pre = ML::GGUF::Gemma4CPU.attention_project_pre_norm(lw, x_norm)

    cpu = ML::GGUF::Gemma4CPU::AttentionProjection.new(pre.q.dup, pre.k.dup, pre.v.dup, pre.reused_k_as_v)
    ML::GGUF::Gemma4CPU.normalize_attention_projection!(cpu, lw, w.hparams, 0)
    ML::GGUF::Gemma4CPU.apply_rope_to_qk!(cpu, w.hparams, 0, 7)
    gpu = ML::GGUF::Gemma4Metal.normalize_and_rope_projection(pre, lw, w.hparams, 0, 7).not_nil!

    gemma4_metal_expect_close("gemma4_swa_norm_rope_q", cpu.q, gpu.q)
    gemma4_metal_expect_close("gemma4_swa_norm_rope_k", cpu.k, gpu.k)
    gemma4_metal_expect_close("gemma4_swa_norm_rope_v", cpu.v, gpu.v)
  end

  it "normalizes and RoPEs a full-attention projection with Gemma4 frequency factors" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    lw = w.layers[5]
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, lw.attn_norm, w.hparams.rms_eps)
    pre = ML::GGUF::Gemma4CPU.attention_project_pre_norm(lw, x_norm)

    cpu = ML::GGUF::Gemma4CPU::AttentionProjection.new(pre.q.dup, pre.k.dup, pre.v.dup, pre.reused_k_as_v)
    ML::GGUF::Gemma4CPU.normalize_attention_projection!(cpu, lw, w.hparams, 5)
    ML::GGUF::Gemma4CPU.apply_rope_to_qk!(cpu, w.hparams, 5, 7, w.rope_freqs)
    gpu = ML::GGUF::Gemma4Metal.normalize_and_rope_projection(pre, lw, w.hparams, 5, 7, w.rope_freqs).not_nil!

    gemma4_metal_expect_close("gemma4_full_norm_rope_q", cpu.q, gpu.q)
    gemma4_metal_expect_close("gemma4_full_norm_rope_k", cpu.k, gpu.k)
    gemma4_metal_expect_close("gemma4_full_norm_rope_v", cpu.v, gpu.v)
  end

  it "computes ungated SWA GQA attention context like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_context_parity(w, 0, [42, 43, 44], "gemma4_swa_attn")
  end

  it "computes ungated full-layer GQA attention context like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_context_parity(w, 5, [42, 43, 44], "gemma4_full_attn")
  end

  it "projects SWA attention context through the output weight like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_attn_output_projection_parity(w, 0, "gemma4_swa_attn_out_proj")
  end

  it "projects full-attention context through the output weight like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_attn_output_projection_parity(w, 5, "gemma4_full_attn_out_proj")
  end

  it "runs the SWA post-attention and GELU FFN tail like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_layer_tail_parity(w, 0, "gemma4_swa_layer_tail")
  end

  it "runs the full-attention post-attention and GELU FFN tail like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_layer_tail_parity(w, 5, "gemma4_full_layer_tail")
  end

  it "runs one composed SWA layer like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_forward_layer_parity(w, 0, "gemma4_swa_forward_layer")
  end

  it "runs one composed full-attention layer like the CPU reference" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_12B_Q4KM)

    gemma4_expect_forward_layer_parity(w, 5, "gemma4_full_forward_layer")
  end
end
