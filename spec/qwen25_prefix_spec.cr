require "spec"
require "../src/ml/gguf/qwen25_hy_nomas"

describe ML::GGUF::Qwen25VLPrefixPlan do
  it "admits shared workspace and all retained hidden outputs before allocation" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    # The one-block baseline gains 1,280 F32 RMS-statistics values (5,120 B).
    # All stages/scratch except one hidden_out are shared; each of the 26
    # hidden outputs remains distinct.
    tokens = 1280
    one_block = 320_471_040_i64 + 5_120_i64
    hidden = 1_i64 * tokens * 3584 * 2
    expected = one_block - hidden + 26 * hidden
    ML::GGUF::Qwen25VLPrefixPlan.workspace_bytes(config, tokens, 1).should eq(one_block)
    ML::GGUF::Qwen25VLPrefixPlan.workspace_bytes(config, tokens, 26).should eq(expected)
    ML::GGUF::Qwen25VLPrefixPlan.weight_bytes(config, 26).should eq(12_119_005_184_i64)
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLPrefixPlan.workspace_bytes(config, 0, 26) }
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLPrefixPlan.workspace_bytes(config, tokens, 29) }
  end

  it "accounts for the RMS statistics scratch as F32 while retaining BF16 scratch bytes" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    tokens = 1280
    rms_statistics_shape = [1, tokens, 1]
    input_normalized_shape = [1, tokens, config.dim]

    ML::GGUF::Qwen25VLBlockPlan.scratch_buffer_bytes("rms_statistics", rms_statistics_shape).should eq(5_120_i64)
    ML::GGUF::Qwen25VLBlockPlan.scratch_buffer_bytes("input_normalized", input_normalized_shape).should eq(9_175_040_i64)
  end

  it "admits one private exponent shared by both block RMS sites before allocation" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    legacy = 320_476_160_i64
    candidate = legacy + 4
    ML::GGUF::Qwen25VLBlockPlan.workspace_bytes(config, 1280).should eq(legacy)
    ML::GGUF::Qwen25VLBlockPlan.workspace_bytes(config, 1280, rms_power_fed: true).should eq(candidate)
    ML::GGUF::Qwen25VLBlockPlan.admit_workspace_bytes!(config, 1280, candidate, rms_power_fed: true).should eq(candidate)
    expect_raises(ArgumentError, /workspace exceeds/) do
      ML::GGUF::Qwen25VLBlockPlan.admit_workspace_bytes!(config, 1280, candidate - 1, rms_power_fed: true)
    end
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLBlockPlan.workspace_bytes(config, 0, rms_power_fed: true) }
    expect_raises(ArgumentError) { ML::GGUF::Qwen25VLBlockPlan.workspace_bytes(config, config.max_positions + 1) }
  end

  it "admits per-block RMS exponent bytes without charging the shared workspace" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    {1, 26}.each do |count|
      legacy = ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, 1280, count)
      fed = ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, 1280, count,
        rms_power_fed: true)
      expected_legacy = count == 1 ? 799_048_960_i64 : 12_681_314_560_i64
      expected_workspace = count == 1 ? 320_476_160_i64 : 549_852_160_i64

      legacy.should eq(expected_legacy)
      fed.should eq(legacy + count.to_i64 * 4)
      ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, 1280, count,
        rms_power_fed: false).should eq(legacy)
      ML::GGUF::Qwen25VLPrefixPlan.admit_named_bytes!(config, 1280, count, fed,
        rms_power_fed: true).should eq(fed)
      expect_raises(ArgumentError, /prefix exceeds/) do
        ML::GGUF::Qwen25VLPrefixPlan.admit_named_bytes!(config, 1280, count,
          fed - 1, rms_power_fed: true)
      end

      shared = ML::GGUF::Qwen25VLPrefixPlan.workspace_bytes(config, 1280, count)
      shared.should eq(expected_workspace)
      (fed - legacy).should eq(count.to_i64 * 4)
    end
  end
end
