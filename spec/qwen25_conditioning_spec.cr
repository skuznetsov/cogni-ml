require "spec"
require "../src/ml/gguf/qwen25_hy_nomas_conditioning"

describe ML::GGUF::Qwen25VLConditioningPlan do
  it "admits the raw mask and both cropped GPU destinations before allocation" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    bytes = ML::GGUF::Qwen25VLConditioningPlan.resident_bytes(config, 1400, 120)
    bytes.raw_mask.should eq(5600_i64)
    bytes.hidden.should eq(9_175_040_i64)
    bytes.mask.should eq(5120_i64)
    bytes.total.should eq(9_185_760_i64)
    ML::GGUF::Qwen25VLConditioningPlan.validate_budget!(bytes, bytes.total)
    expect_raises(ArgumentError, "conditioning exceeds admitted named byte budget") do
      ML::GGUF::Qwen25VLConditioningPlan.validate_budget!(bytes, bytes.total - 1)
    end
  end

  it "keeps the full raw context and a strictly positive cropped suffix" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    [0, -1, 1400, 1401].each do |crop|
      expect_raises(ArgumentError) do
        ML::GGUF::Qwen25VLConditioningPlan.resident_bytes(config, 1400, crop)
      end
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen25VLConditioningPlan.resident_bytes(config, config.max_positions + 1, 120)
    end
  end

  it "checks borrowed buffer extents without allocating or reading activations" do
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    raw_hidden_bytes = 1400_i64 * config.dim * 2
    ML::GGUF::Qwen25VLConditioningPlan.validate_extents!(config, 1400, 120, raw_hidden_bytes, 5600_i64)
    expect_raises(ArgumentError, "raw hidden extent mismatch") do
      ML::GGUF::Qwen25VLConditioningPlan.validate_extents!(config, 1400, 120, raw_hidden_bytes - 2, 5600_i64)
    end
    expect_raises(ArgumentError, "raw mask extent mismatch") do
      ML::GGUF::Qwen25VLConditioningPlan.validate_extents!(config, 1400, 120, raw_hidden_bytes, 5596_i64)
    end
  end

  it "rejects released sources and destinations before touching a command chain" do
    ML::GGUF::Qwen25VLConditioningPlan.validate_liveness!(true, true, true, true)
    4.times do |invalid|
      flags = Array(Bool).new(4) { |index| index != invalid }
      expect_raises(ArgumentError, "source or destination was released") do
        ML::GGUF::Qwen25VLConditioningPlan.validate_liveness!(flags[0], flags[1], flags[2], flags[3])
      end
    end
  end
end
