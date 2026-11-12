require "spec"
require "../src/ml/gguf/nomic_metal_policy"

describe ML::GGUF::NomicMetalPolicy do
  it "enables the repaired GEMM and MoE corridor on M5 Max" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M5 Max").should be_true
  end

  it "fails closed for matrix GEMM on unrecognized devices" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Intel GPU").should be_false
  end

  it "allows explicit matrix GEMM overrides" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Intel GPU", "on").should be_true
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M5 Max", "off").should be_false
  end

  it "controls matrix attention independently of GEMM" do
    policy = ML::GGUF::NomicMetalPolicy
    if policy.responds_to?(:matrix_attention_enabled?)
      policy.matrix_attention_enabled?("Apple M5 Max").should be_false
      policy.matrix_attention_enabled?("Apple M2 Max").should be_true
      policy.matrix_attention_enabled?("Apple M5 Max", "on").should be_true
      policy.matrix_attention_enabled?("Apple M2 Max", "off").should be_false
    else
      fail "Missing independent matrix attention policy"
    end
  end
end
