require "./spec_helper"
require "../src/ml/gguf/gemma4_meta"
require "../src/ml/gguf/gemma4_weights"

GEMMA4_12B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

describe ML::GGUF::Gemma4Hparams do
  it "parses the local Gemma4 12B GGUF metadata" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_12B_Q4KM)

    g = ML::GGUF::GGUFFile.new(GEMMA4_12B_Q4KM)
    begin
      hp = ML::GGUF::Gemma4Hparams.new(g)

      hp.arch.should eq("gemma4")
      hp.n_layer.should eq(48)
      hp.n_embd.should eq(3840)
      hp.n_ff.should eq(15360)
      hp.context_length.should eq(131_072)
      hp.vocab_size.should eq(262_144)

      hp.n_head.should eq(16)
      hp.head_dim.should eq(512)
      hp.head_dim_swa.should eq(256)
      hp.rope_dim_count.should eq(512)
      hp.rope_dim_count_swa.should eq(256)
      hp.sliding_window.should eq(1024)

      hp.full_attention_layers.should eq([5, 11, 17, 23, 29, 35, 41, 47])
      hp.sliding_window_layers.size.should eq(40)
      hp.n_head_kv_by_layer.count(8).should eq(40)
      hp.n_head_kv_by_layer.count(1).should eq(8)
    ensure
      g.close
    end
  end
end

describe ML::GGUF::Gemma4Weights do
  it "maps local Gemma4 tensors structurally without dequantizing large weights" do
    pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_12B_Q4KM)

    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_12B_Q4KM)
    hp = w.hparams

    w.token_embd.in_dim.should eq(3840)
    w.token_embd.out_dim.should eq(262_144)
    w.token_embd.type.name.should eq("Q6_K")
    w.layers.size.should eq(48)

    swa = w.layers[0]
    swa.attn_q_qw.in_dim.should eq(3840)
    swa.attn_q_qw.out_dim.should eq(4096)
    swa.attn_k_qw.out_dim.should eq(2048)
    swa.attn_v_qw.should_not be_nil
    swa.explicit_v?.should be_true
    swa.reuse_k_as_v?.should be_false
    swa.attn_output_qw.in_dim.should eq(4096)
    swa.attn_output_qw.out_dim.should eq(3840)

    full = w.layers[5]
    hp.full_attention?(5).should be_true
    full.attn_q_qw.out_dim.should eq(8192)
    full.attn_k_qw.out_dim.should eq(512)
    full.attn_v_qw.should be_nil
    full.explicit_v?.should be_false
    full.reuse_k_as_v?.should be_true
    full.attn_output_qw.in_dim.should eq(8192)
    full.attn_output_qw.out_dim.should eq(3840)
  end
end
