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
end
