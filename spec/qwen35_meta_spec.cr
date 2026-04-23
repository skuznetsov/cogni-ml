require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_meta"

# Regression tests for Qwen 3.5/3.6 hparam parsing.
# Skipped when the model files are not present locally.

QWEN_9B_PATH  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN_27B_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35Hparams do
  describe "Qwen 3.5 9B" do
    it "parses hparams matching GGUF metadata" do
      pending!("9B model not present") unless File.exists?(QWEN_9B_PATH)

      g = ML::GGUF::GGUFFile.new(QWEN_9B_PATH)
      h = ML::GGUF::Qwen35Hparams.new(g)
      g.close

      h.arch.should eq("qwen35")
      h.n_layer.should eq(32)
      h.n_embd.should eq(4096)
      h.n_ff.should eq(12288)
      h.context_length.should eq(262144)
      h.n_head.should eq(16)
      h.n_head_kv.should eq(4)
      h.head_dim.should eq(256)
      h.full_attention_interval.should eq(4)
      h.rope_dim_count.should eq(64)
      h.rope_freq_base.should be_close(10_000_000.0_f32, 1.0)
      h.rope_sections.should eq([11, 11, 10, 0])
      h.ssm_conv_kernel.should eq(4)
      h.ssm_state_size.should eq(128)
      h.ssm_group_count.should eq(16)
      h.ssm_time_step_rank.should eq(32)
      h.ssm_inner_size.should eq(4096)
      h.ssm_head_v_dim.should eq(128)
    end

    it "computes correct recurrent/full-attn cadence (layers 3,7,...,31 are full-attn)" do
      pending!("9B model not present") unless File.exists?(QWEN_9B_PATH)
      g = ML::GGUF::GGUFFile.new(QWEN_9B_PATH)
      h = ML::GGUF::Qwen35Hparams.new(g)
      g.close

      h.full_attention_layers.should eq([3, 7, 11, 15, 19, 23, 27, 31])
      h.recurrent_layers.size.should eq(24)
      h.full_attention?(3).should be_true
      h.full_attention?(4).should be_false
      h.recurrent?(0).should be_true
      h.recurrent?(31).should be_false
    end
  end

  describe "Qwen 3.6 27B" do
    it "parses hparams matching GGUF metadata (same arch, scaled dims)" do
      pending!("27B model not present") unless File.exists?(QWEN_27B_PATH)

      g = ML::GGUF::GGUFFile.new(QWEN_27B_PATH)
      h = ML::GGUF::Qwen35Hparams.new(g)
      g.close

      h.arch.should eq("qwen35")
      h.n_layer.should eq(64)
      h.n_embd.should eq(5120)
      h.head_dim.should eq(256)
      h.full_attention_interval.should eq(4)
      h.rope_sections.should eq([11, 11, 10, 0])
      h.full_attention_layers.size.should eq(16)
      h.recurrent_layers.size.should eq(48)
    end
  end
end

describe ML::GGUF::GGUFFile do
  it "get_int_array handles Int32 arrays (rope.dimension_sections)" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_PATH)
    g = ML::GGUF::GGUFFile.new(QWEN_9B_PATH)
    arr = g.get_int_array("qwen35.rope.dimension_sections")
    g.close

    arr.should_not be_nil
    arr.not_nil!.should eq([11_i64, 11_i64, 10_i64, 0_i64])
  end

  it "get_int_array returns nil for missing keys" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_PATH)
    g = ML::GGUF::GGUFFile.new(QWEN_9B_PATH)
    arr = g.get_int_array("does.not.exist")
    g.close
    arr.should be_nil
  end
end
