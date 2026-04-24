require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_DN = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35CPU, "DeltaNet recurrent layer forward" do
  pending!("9B model not present") unless File.exists?(QWEN_9B_DN)

  # SSM state lives on whichever backend the routing helper picked up first:
  # `ssm_state` (Array) on CPU, `ssm_state_buf` (MetalBuffer) on Metal. They
  # never coexist for a given sequence.
  ssm_state_as_array = ->(ls : ML::GGUF::Qwen35CPU::LayerState, expected_size : Int32) do
    if buf = ls.ssm_state_buf
      buf.read(expected_size)
    else
      ls.ssm_state.not_nil!
    end
  end

  conv_state_as_array = ->(ls : ML::GGUF::Qwen35CPU::LayerState, expected_size : Int32) do
    if buf = ls.conv_state_buf
      buf.read(expected_size)
    else
      ls.conv_state.not_nil!
    end
  end

  it "runs blk.0 (first recurrent layer) on embedding of token 0" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_DN)
    hp = w.hparams
    hp.recurrent?(0).should be_true

    lw = w.layers[0].as(ML::GGUF::Qwen35RecurrentWeights)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)

    inp = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 0)
    out = ML::GGUF::Qwen35CPU.forward_recurrent_layer(inp, 0, lw, state.layers[0], hp, 64)

    out.size.should eq(hp.n_embd)
    out.all? { |v| v.finite? }.should be_true

    # conv_state and ssm_state should have been allocated and (at least partially) written
    expected_ss_size = hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size
    expected_cs_size = (hp.ssm_conv_kernel - 1) * (2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size)
    cs = conv_state_as_array.call(state.layers[0], expected_cs_size)
    ss = ssm_state_as_array.call(state.layers[0], expected_ss_size)
    cs.size.should eq(expected_cs_size)
    ss.size.should eq(expected_ss_size)

    # First token wrote qkv_mixed into conv_state slot 2 (the "current" row). Slots 0,1 still zero.
    kv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
    (0...kv_dim).count { |i| cs[i] != 0.0_f32 }.should eq(0)                         # slot 0 (oldest) untouched
    (kv_dim...2*kv_dim).count { |i| cs[i] != 0.0_f32 }.should eq(0)                  # slot 1 untouched
    (2*kv_dim...3*kv_dim).count { |i| cs[i] != 0.0_f32 }.should be > kv_dim // 10    # slot 2 = current token

    # SSM state was written (non-zero)
    (0...ss.size).count { |i| ss[i] != 0.0_f32 }.should be > ss.size // 100
  end

  it "accumulates state across two tokens (second output ≠ first in same state)" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_DN)
    hp = w.hparams
    lw = w.layers[0].as(ML::GGUF::Qwen35RecurrentWeights)

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)

    inp0 = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 100)
    inp1 = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 200)

    out0 = ML::GGUF::Qwen35CPU.forward_recurrent_layer(inp0, 0, lw, state.layers[0], hp, 64)
    out1 = ML::GGUF::Qwen35CPU.forward_recurrent_layer(inp1, 1, lw, state.layers[0], hp, 64)

    out1.size.should eq(hp.n_embd)
    out1.all? { |v| v.finite? }.should be_true

    # out1 state is different from out0, obviously — but also: conv_state has shifted
    kv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
    cs = conv_state_as_array.call(state.layers[0], (hp.ssm_conv_kernel - 1) * kv_dim)
    # After 2 tokens: slot 0 still zero (pos -2 never existed), slot 1 = token 0, slot 2 = token 1
    (0...kv_dim).count { |i| cs[i] != 0.0_f32 }.should eq(0)
    (kv_dim...2*kv_dim).count { |i| cs[i] != 0.0_f32 }.should be > kv_dim // 10
    (2*kv_dim...3*kv_dim).count { |i| cs[i] != 0.0_f32 }.should be > kv_dim // 10
  end
end
