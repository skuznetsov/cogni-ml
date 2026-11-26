require "../../src/ml/video/hy_nomas_reference_block_mps"
require "./hy_nomas_reference_block0_probe"

module HyNomasReferenceBlock0Setup
  extend self
  include HyNomasReferenceBlock0Probe

  def config(precision : ML::Video::HyNomasAttentionPrecision,
             text_tokens : Int32 = TEXT_STORAGE_TOKENS) : ML::Video::HyNomasConfig
    ML::Video::HyNomasConfig.new(
      batches: 1,
      video_tokens: IMAGE_TOKENS,
      text_tokens: text_tokens,
      dim: DIM,
      heads: HEADS,
      head_dim: HEAD_DIM,
      ffn_dim: 4 * DIM,
      attention_precision: precision,
    ).tap(&.validate!)
  end

  def bindings : Array(ML::Video::WeightBinding)
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config(
      ML::Video::HyNomasAttentionPrecision::F32Control),
      ML::GGUF::SafeTensorDType::BF16)
  end

  def rms_named_bytes(config : ML::Video::HyNomasConfig) : Int64
    per_stream = ->(tokens : Int32) do
      rows = config.batches.to_i64 * tokens * config.heads
      square = rows * config.head_dim * sizeof(Float32)
      mean = rows * sizeof(Float32)
      2_i64 * (square + mean) # Q and K each own square and row-mean buffers.
    end
    per_stream.call(config.video_tokens) + per_stream.call(config.text_tokens)
  end

  def workspace_named_bytes(config : ML::Video::HyNomasConfig, attention_tokens : Int32) : Hash(String, Int64)
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum(0_i64) do |(_name, count)|
      count.to_i64 * sizeof(Float32)
    end
    attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
      1, attention_tokens, HEADS, HEAD_DIM, 256, HEADS).resident_bytes
    {
      "block_scratch"                  => scratch,
      "mps_weight_stage"               => ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * sizeof(Float32),
      "compact_attention_buffers"      => attention,
      "qk_rms_square_and_mean_buffers" => rms_named_bytes(config),
    }
  end
end
