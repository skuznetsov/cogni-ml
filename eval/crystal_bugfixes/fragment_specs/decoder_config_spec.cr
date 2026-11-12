require "spec"
require "../src/ml/gguf/decoder"

describe ML::GGUF::DecoderConfig do
  it "rejects a model width inconsistent with the attention head geometry" do
    expect_raises(Exception) do
      ML::GGUF::DecoderConfig.new(dim: 10, n_heads: 2, n_kv_heads: 1, head_dim: 4)
    end
  end

  it "rejects an incompatible GQA grouping" do
    expect_raises(Exception) do
      ML::GGUF::DecoderConfig.new(dim: 12, n_heads: 3, n_kv_heads: 2, head_dim: 4)
    end
  end

  it "rejects an odd rotary head dimension" do
    expect_raises(Exception) do
      ML::GGUF::DecoderConfig.new(dim: 6, n_heads: 2, n_kv_heads: 1, head_dim: 3)
    end
  end

  it "retains valid GQA geometry" do
    config = ML::GGUF::DecoderConfig.new(dim: 16, n_heads: 4, n_kv_heads: 2, head_dim: 4)
    config.n_kv_heads.should eq(2)
  end
end
