require "spec"
require "../src/ml/gguf/qwen_qbit_adaptive_metal_policy"

describe ML::GGUF::QwenQBitAdaptiveMetalPolicy do
  it "admits tile 15 only on the measured M2 Max corridor" do
    ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Apple M2 Max").should eq(15)
    ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Apple M2 Pro").should eq(16)
    ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Apple M5 Max").should eq(16)
    ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Unknown Metal Device").should eq(16)
  end

  it "allows an explicit benchmark override" do
    ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Apple M2 Max", "16").should eq(16)
    ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Unknown Metal Device", "15").should eq(15)
  end

  it "rejects unknown overrides" do
    expect_raises(ArgumentError, "QWEN35_ADAPTIVE_GQA6_TILE must be auto, 15, or 16") do
      ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile("Apple M2 Max", "24")
    end
  end

  it "admits split-K only for a long one-token uniform decode" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.decode_splitk?(255, 1, true, "1").should be_true
    policy.decode_splitk?(254, 1, true, "1").should be_false
    policy.decode_splitk?(255, 2, true, "1").should be_false
    policy.decode_splitk?(255, 1, false, "1").should be_false
    policy.decode_splitk?(255, 1, true, "0").should be_false
  end

  it "fails closed on invalid split-K policy values" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    expect_raises(ArgumentError, /must be 0 or 1/) do
      policy.decode_splitk?(127, 1, true, "maybe", "128")
    end
    expect_raises(ArgumentError, /positive integer/) do
      policy.decode_splitk?(127, 1, true, "1", "zero")
    end
  end

  it "keeps register-local t4 dequantization behind an exact boolean override" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.dequant_t4?(nil).should be_false
    policy.dequant_t4?("").should be_false
    policy.dequant_t4?("0").should be_false
    policy.dequant_t4?("1").should be_true
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_DEQUANT_T4/) do
      policy.dequant_t4?("true")
    end
  end
end
