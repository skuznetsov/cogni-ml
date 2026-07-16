require "spec"
require "../src/ml/gguf/nomic_metal_policy"

describe ML::GGUF::NomicMetalPolicy do
  it "keeps the matrix corridor enabled on verified M2 hardware" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M2 Max").should be_true
  end

  it "enables the repaired matrix GEMM and MoE corridor on M5 Max" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M5 Max").should be_true
  end

  it "fails closed on unknown Metal devices" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Intel GPU").should be_false
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Unknown Metal Device").should be_false
  end

  it "allows an explicit probe override" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M5 Max", "on").should be_true
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M2 Max", "off").should be_false
  end

  it "rejects unknown override values" do
    expect_raises(ArgumentError, "NOMIC_SIMDGROUP_MATRIX must be auto, on, or off") do
      ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M2 Max", "maybe")
    end
  end

  it "keeps matrix attention fail closed on M5 Max independently" do
    ML::GGUF::NomicMetalPolicy.matrix_attention_enabled?("Apple M5 Max").should be_false
    ML::GGUF::NomicMetalPolicy.matrix_attention_enabled?("Apple M2 Max").should be_true
    ML::GGUF::NomicMetalPolicy.matrix_attention_enabled?("Apple M5 Max", "on").should be_true
  end

  it "rejects unknown matrix attention override values" do
    expect_raises(ArgumentError, "NOMIC_MATRIX_ATTENTION must be auto, on, or off") do
      ML::GGUF::NomicMetalPolicy.matrix_attention_enabled?("Apple M5 Max", "maybe")
    end
  end
end
