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

  it "scopes fused split-K stage2 to the measured M2 Max corridor" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.splitk_stage2_fused?("Apple M2 Max", false, true, 255).should be_true
    policy.splitk_stage2_fused?("Apple M2 Max", true, false, 6_143).should be_false
    policy.splitk_stage2_fused?("Apple M2 Max", true, false, 6_144).should be_true
    policy.splitk_stage2_fused?("Apple M2 Max", true, false, 8_192, "0").should be_false
    policy.splitk_stage2_fused?("Apple M2 Max", true, false, 6_143, "1").should be_false
    policy.splitk_stage2_fused?("Apple M2 Pro", true, false, 8_192).should be_false
    policy.splitk_stage2_fused?("Unknown Metal Device", false, true, 8_192).should be_false
    policy.splitk_stage2_fused?("Apple M2 Max", true, false, 8_192, nil, "0").should be_false
    policy.splitk_stage2_fused?("Apple M2 Pro", false, false, 1, nil, "1").should be_true
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED/) do
      policy.splitk_stage2_fused?("Apple M2 Max", true, false, 8_192, nil, "")
    end
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED/) do
      policy.splitk_stage2_fused?("Apple M2 Max", true, false, 8_192, nil, "true")
    end
  end

  it "scopes prefix-only adaptive pack quantization to the measured M2 Max corridor" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.pack_prefix_quant?("Apple M2 Max", true).should be_true
    policy.pack_prefix_quant?("Apple M2 Max", false).should be_false
    policy.pack_prefix_quant?("Apple M2 Pro", true).should be_false
    policy.pack_prefix_quant?("Unknown Metal Device", true).should be_false
    policy.pack_prefix_quant?("Apple M2 Max", true, "0").should be_false
    policy.pack_prefix_quant?("Apple M2 Pro", false, "1").should be_true
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_PACK_PREFIX_QUANT/) do
      policy.pack_prefix_quant?("Apple M2 Max", true, "")
    end
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_PACK_PREFIX_QUANT/) do
      policy.pack_prefix_quant?("Apple M2 Max", true, "true")
    end
  end

  it "auto-admits register-local t4 only on measured M2 Max corridors" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.automatic_dequant_t4?(1, false, true, false).should be_false
    policy.automatic_dequant_t4?(1, false, false, true).should be_false
    policy.automatic_dequant_t4?(64, false, true, false).should be_true
    policy.automatic_dequant_t4?(64, false, false, true).should be_true
    policy.automatic_dequant_t4?(64, false, false, false).should be_false
    policy.automatic_dequant_t4?(1, true, true, false).should be_false
    policy.automatic_dequant_t4?(1, true, false, true).should be_true

    policy.dequant_t4?("Apple M2 Max", true).should be_true
    policy.dequant_t4?("Apple M2 Max", false).should be_false
    policy.dequant_t4?("Apple M2 Pro", true).should be_false
    policy.dequant_t4?("Unknown Metal Device", true).should be_false

    policy.dequant_t4?("Apple M2 Max", true, "0").should be_false
    policy.dequant_t4?("Apple M2 Pro", false, "1").should be_true
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_DEQUANT_T4/) do
      policy.dequant_t4?("Apple M2 Max", true, "")
    end
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_DEQUANT_T4/) do
      policy.dequant_t4?("Apple M2 Max", true, "true")
    end
  end

  it "auto-admits the P4 split-K t8 loader only in the measured long-context corridor" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.p4_splitk_t8?("Apple M2 Max", 6_143).should be_false
    policy.p4_splitk_t8?("Apple M2 Max", 6_144).should be_true
    policy.p4_splitk_t8?("Apple M2 Max", 6_145).should be_true
    policy.p4_splitk_t8?("Apple M2 Pro", 6_144).should be_false
    policy.p4_splitk_t8?("Apple M5 Max", 6_144).should be_false
    policy.p4_splitk_t8?("Unknown Metal Device", 6_144).should be_false
    policy.p4_splitk_t8?("Apple M2 Max", 6_144, "0").should be_false
    policy.p4_splitk_t8?("Apple M2 Pro", 1, "1").should be_true
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_P4_SPLITK_T8/) do
      policy.p4_splitk_t8?("Apple M2 Max", 6_144, "")
    end
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_P4_SPLITK_T8/) do
      policy.p4_splitk_t8?("Apple M2 Max", 6_144, "true")
    end
  end

  it "auto-admits the BF16 split-K t8 loader only in the measured long-context corridor" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.bf16_splitk_t8?("Apple M2 Max", 6_143).should be_false
    policy.bf16_splitk_t8?("Apple M2 Max", 6_144).should be_true
    policy.bf16_splitk_t8?("Apple M2 Max", 6_145).should be_true
    policy.bf16_splitk_t8?("Apple M2 Pro", 6_144).should be_false
    policy.bf16_splitk_t8?("Unknown Metal Device", 6_144).should be_false
    policy.bf16_splitk_t8?("Apple M2 Max", 6_144, "0").should be_false
    policy.bf16_splitk_t8?("Apple M2 Pro", 1, "1").should be_true
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_BF16_SPLITK_T8/) do
      policy.bf16_splitk_t8?("Apple M2 Max", 6_144, "")
    end
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_BF16_SPLITK_T8/) do
      policy.bf16_splitk_t8?("Apple M2 Max", 6_144, "true")
    end
  end

  it "keeps the experimental P4 direct-QK stage explicitly off unless enabled with 1" do
    policy = ML::GGUF::QwenQBitAdaptiveMetalPolicy

    policy.p4_splitk_direct_qk?(true, true).should be_false
    policy.p4_splitk_direct_qk?(true, true, "0").should be_false
    policy.p4_splitk_direct_qk?(true, true, "1").should be_true
    policy.p4_splitk_direct_qk?(false, true, "1").should be_false
    policy.p4_splitk_direct_qk?(true, false, "1").should be_false
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_P4_SPLITK_DIRECT_QK/) do
      policy.p4_splitk_direct_qk?(true, true, "true")
    end
    expect_raises(ArgumentError, /QWEN35_ADAPTIVE_P4_SPLITK_DIRECT_QK/) do
      policy.p4_splitk_direct_qk?(true, true, "")
    end
  end
end
