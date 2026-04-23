require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_meta"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_WPATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35Weights do
  it "loads Qwen 3.5 9B weights and dispatches layer types correctly" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_WPATH)

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_WPATH)
    h = w.hparams

    w.layers.size.should eq(h.n_layer)
    w.layers.size.should eq(32)

    # Count types
    full_count = w.layers.count(&.is_a?(ML::GGUF::Qwen35FullAttnWeights))
    recurrent_count = w.layers.count(&.is_a?(ML::GGUF::Qwen35RecurrentWeights))
    full_count.should eq(8)       # il ∈ {3,7,11,15,19,23,27,31}
    recurrent_count.should eq(24)

    # Spot-check layer 3 (first full-attn layer)
    l3 = w.layers[3].as(ML::GGUF::Qwen35FullAttnWeights)
    l3.attn_norm.size.should eq(h.n_embd)                                     # 4096
    l3.attn_q_qw.in_dim.should eq(h.n_embd)                                   # 4096
    l3.attn_q_qw.out_dim.should eq(2 * h.head_dim * h.n_head)                 # 2*256*16 = 8192
    l3.attn_q_norm.size.should eq(h.head_dim)                                 # 256
    l3.attn_k_qw.in_dim.should eq(h.n_embd)
    l3.attn_k_qw.out_dim.should eq(h.head_dim * h.n_head_kv)                  # 256*4 = 1024
    l3.attn_k_norm.size.should eq(h.head_dim)
    l3.attn_v_qw.out_dim.should eq(h.head_dim * h.n_head_kv)                  # 1024
    l3.attn_output_qw.in_dim.should eq(h.head_dim * h.n_head)                 # 4096 (n_head*head_dim != n_embd accidentally here, but fully-proj 4096)
    l3.attn_output_qw.out_dim.should eq(h.n_embd)
    l3.post_attention_norm.size.should eq(h.n_embd)
    l3.ffn_gate_qw.in_dim.should eq(h.n_embd)
    l3.ffn_gate_qw.out_dim.should eq(h.n_ff)                                  # 12288
    l3.ffn_up_qw.out_dim.should eq(h.n_ff)
    l3.ffn_down_qw.in_dim.should eq(h.n_ff)
    l3.ffn_down_qw.out_dim.should eq(h.n_embd)

    # Spot-check layer 0 (first recurrent layer)
    l0 = w.layers[0].as(ML::GGUF::Qwen35RecurrentWeights)
    l0.attn_norm.size.should eq(h.n_embd)
    l0.attn_qkv_qw.in_dim.should eq(h.n_embd)
    # qkv_dim = 2*num_k_heads*state + num_v_heads*state = 2*16*128 + 32*128 = 4096 + 4096 = 8192
    l0.attn_qkv_qw.out_dim.should eq(2 * h.ssm_group_count * h.ssm_state_size + h.ssm_time_step_rank * h.ssm_state_size)
    l0.attn_qkv_qw.out_dim.should eq(8192)
    l0.attn_gate_qw.in_dim.should eq(h.n_embd)
    l0.attn_gate_qw.out_dim.should eq(h.n_embd)
    l0.ssm_a.size.should eq(h.ssm_time_step_rank)                             # 32
    l0.ssm_alpha_qw.in_dim.should eq(h.n_embd)
    l0.ssm_alpha_qw.out_dim.should eq(h.ssm_time_step_rank)                   # 32
    l0.ssm_beta_qw.in_dim.should eq(h.n_embd)
    l0.ssm_beta_qw.out_dim.should eq(h.ssm_time_step_rank)
    # ssm_conv1d has shape [conv_kernel, 2*num_k_heads*state] = [4, 2*16*128] = [4, 4096]
    # But we saw shape [4, 8192] — includes a different channel count. Verify.
    # (Double-check: conv is applied to Q, K, V together per delta-net? 2*num_k*state + num_v*state = 8192)
    l0.ssm_conv1d.size.should eq(h.ssm_conv_kernel * 8192)                    # 4*8192 = 32768
    l0.ssm_dt_bias.size.should eq(h.ssm_time_step_rank)                       # 32
    l0.ssm_norm.size.should eq(h.ssm_state_size)                              # 128
    l0.ssm_out_qw.in_dim.should eq(h.n_embd)
    l0.ssm_out_qw.out_dim.should eq(h.n_embd)
    l0.post_attention_norm.size.should eq(h.n_embd)
    l0.ffn_up_qw.out_dim.should eq(h.n_ff)

    # Global
    w.token_embd.in_dim.should eq(h.n_embd)    # [n_embd, vocab]
    w.output_norm.size.should eq(h.n_embd)
    w.output.in_dim.should eq(h.n_embd)
    w.token_embd.out_dim.should eq(w.output.out_dim)   # both vocab_size
  end
end
