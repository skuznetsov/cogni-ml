require "spec"
require "../src/ml/video/hy_nomas_text_refiner"
require "../src/ml/video/hy_nomas_block_mps"

describe ML::Video::HyNomasTextRefinerMPSPlan do
  it "keeps valid-prefix compaction by default and retains padded activations with valid-prefix attention" do
    mask = Array(Int32).new(1280) { |index| index < 23 ? 1 : 0 }

    ML::Video::HyNomasTextRefinerMPSPlan.processed_tokens(mask).should eq(23)
    ML::Video::HyNomasTextRefinerMPSPlan.processed_tokens(mask, retain_padding: true).should eq(1280)

    compact = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask)
    compact.activations.should eq(8_610_820_i64)
    compact.linear_stage.should eq(67_108_864_i64)
    compact.attention_stage.should eq(1_507_328_i64)
    compact.total.should eq(77_227_012_i64)

    padded = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true)
    padded.activations.should eq(469_416_964_i64)
    padded.linear_stage.should eq(67_108_864_i64)
    # Full padded activations are retained, but attention MPS staging remains
    # sized to the 23 valid rows because padded-query attention is zero-filled.
    padded.attention_stage.should eq(compact.attention_stage)
    padded.attention_stage.should eq(1_507_328_i64)
    padded.total.should eq(538_033_156_i64)
    ML::Video::HyNomasTextRefinerMPSPlan.admit_named_bytes!(mask, padded.total,
      retain_padding: true).should eq(padded)
    expect_raises(ArgumentError, /budget/) do
      ML::Video::HyNomasTextRefinerMPSPlan.admit_named_bytes!(mask, padded.total - 1,
        retain_padding: true)
    end
  end
end

describe ML::Video::HyNomasMPSWorkspace do
  it "admits only a batch-one contiguous text-prefix attention extent" do
    config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 6, text_tokens: 1280,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)

    ML::Video::HyNomasMPSWorkspace.attention_tokens_for(config, nil).should eq(1286)
    ML::Video::HyNomasMPSWorkspace.attention_tokens_for(config, 23).should eq(29)
    ML::Video::HyNomasMPSWorkspace.attention_tokens_for(config, 1280).should eq(1286)

    expect_raises(ArgumentError, /positive and no greater/) do
      ML::Video::HyNomasMPSWorkspace.attention_tokens_for(config, 0)
    end
    expect_raises(ArgumentError, /positive and no greater/) do
      ML::Video::HyNomasMPSWorkspace.attention_tokens_for(config, 1281)
    end

    batched = ML::Video::HyNomasConfig.new(batches: 2, video_tokens: 6, text_tokens: 1280,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)
    expect_raises(ArgumentError, /batch-one/) do
      ML::Video::HyNomasMPSWorkspace.attention_tokens_for(batched, 23)
    end
  end
end
