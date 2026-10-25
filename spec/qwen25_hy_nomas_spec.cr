require "spec"
require "../src/ml/gguf/qwen25_hy_nomas"

describe ML::GGUF::Qwen25VLConfig do
  it "selects the Hy-Nomas hidden state without the final norm" do
    c = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    c.dim.should eq(3584)
    c.ffn_dim.should eq(18944)
    c.heads.should eq(28)
    c.kv_heads.should eq(4)
    c.head_dim.should eq(128)
    c.layers.should eq(28)
    c.rms_norm_eps.should eq(1e-6_f32)
    c.rope_theta.should eq(1_000_000_f32)
    c.max_positions.should eq(128000)
    c.selected_block_count(2).should eq(26)
    c.selected_block_count(0).should eq(28)
    expect_raises(ArgumentError) { c.selected_block_count(28) }
    expect_raises(ArgumentError) { c.selected_block_count(-1) }
  end

  it "rejects unsupported geometry before GPU setup" do
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLConfig.new(dim: 63, heads: 2, kv_heads: 1, head_dim: 32) }
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLConfig.new(dim: 96, heads: 3, kv_heads: 2, head_dim: 32) }
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLConfig.new(dim: 62, heads: 2, kv_heads: 1, head_dim: 31) }
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLConfig.new(rms_norm_eps: Float32::NAN) }
  end
end
