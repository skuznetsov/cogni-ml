require "spec"
require "../src/ml/gguf/qwen25_vl_embedding"

private def tiny_qwen25_embedding_config : ML::GGUF::Qwen25VLConfig
  ML::GGUF::Qwen25VLConfig.new(
    dim: 4,
    ffn_dim: 8,
    heads: 2,
    kv_heads: 1,
    head_dim: 2,
    layers: 3,
    max_positions: 16,
  )
end

describe ML::GGUF::Qwen25VLEmbeddingPlan do
  it "accounts for compact rows, remapped IDs, and BF16 output by name" do
    bytes = ML::GGUF::Qwen25VLEmbeddingPlan.resident_bytes(
      tiny_qwen25_embedding_config, 3, 2, 5)

    bytes.compact_rows.should eq(16_i64)
    bytes.remapped_ids.should eq(12_i64)
    bytes.output.should eq(24_i64)
    bytes.total.should eq(52_i64)
    ML::GGUF::Qwen25VLEmbeddingPlan.validate_budget!(bytes, 52_i64)
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen25VLEmbeddingPlan.validate_budget!(bytes, 51_i64)
    end
  end

  it "matches the 7B BF16 bytes for a 1536-token all-unique prefill" do
    bytes = ML::GGUF::Qwen25VLEmbeddingPlan.resident_bytes(
      ML::GGUF::Qwen25VLConfig.hy_nomas_7b, 1536, 1536)

    bytes.compact_rows.should eq(11_010_048_i64)
    bytes.remapped_ids.should eq(6_144_i64)
    bytes.output.should eq(11_010_048_i64)
    bytes.total.should eq(22_026_240_i64)
  end

  it "validates exact row-table geometry and remapped ID bounds without Metal" do
    config = tiny_qwen25_embedding_config
    checked = ML::GGUF::Qwen25VLEmbeddingPlan.validate_inputs(
      config, 3, 16_i64, [1, 0, 1], 5)

    checked[:unique_rows].should eq(2)
    checked[:named_bytes].total.should eq(52_i64)

    expect_raises(ArgumentError) do
      ML::GGUF::Qwen25VLEmbeddingPlan.validate_inputs(config, 3, 16_i64, [0, 2, 1], 5)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen25VLEmbeddingPlan.validate_inputs(config, 3, 16_i64, [0, 0, 0], 5)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen25VLEmbeddingPlan.validate_inputs(config, 3, 15_i64, [0, 1, 0], 5)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen25VLEmbeddingPlan.validate_inputs(config, 17, 16_i64, [0] of Int32, 5)
    end
  end
end
