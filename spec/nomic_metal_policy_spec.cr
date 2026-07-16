require "spec"
require "../src/ml/gguf/nomic_metal_policy"

describe ML::GGUF::NomicMetalPolicy do
  it "keeps the matrix corridor enabled on verified M2 hardware" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M2 Max").should be_true
  end

  it "fails closed on the measured-red M5 Max hardware" do
    ML::GGUF::NomicMetalPolicy.simdgroup_matrix_enabled?("Apple M5 Max").should be_false
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
end
