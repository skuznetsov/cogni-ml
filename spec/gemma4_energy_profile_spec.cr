require "spec"
require "../src/ml/gguf/gemma4_energy_profile"

describe ML::GGUF::Gemma4EnergyProfile do
  it "provides the three bounded defaults" do
    eco = ML::GGUF::Gemma4EnergyProfile.for_name("eco")
    balanced = ML::GGUF::Gemma4EnergyProfile.for_name("Balanced")
    quality = ML::GGUF::Gemma4EnergyProfile.for_name("quality")

    {eco.name, eco.reasoning_tokens, eco.answer_tokens}.should eq({"Eco", 0, 256})
    {balanced.name, balanced.reasoning_tokens, balanced.answer_tokens}.should eq({"Balanced", 256, 512})
    {quality.name, quality.reasoning_tokens, quality.answer_tokens}.should eq({"Quality", 1024, 1024})
  end

  it "allows independent nonnegative reasoning and positive answer overrides" do
    profile = ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 0, answer_tokens: 17)

    profile.reasoning_tokens.should eq(0)
    profile.answer_tokens.should eq(17)
  end

  it "rejects unknown names, negative reasoning, zero answers, and totals over the safe cap" do
    expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyProfile.for_name("fast") }
    expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyProfile.for_name("eco", reasoning_tokens: -1) }
    expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyProfile.for_name("eco", answer_tokens: 0) }
    expect_raises(ArgumentError) do
      ML::GGUF::Gemma4EnergyProfile.for_name(
        "quality",
        reasoning_tokens: ML::GGUF::Gemma4EnergyProfile::MAX_TOTAL_TOKENS,
        answer_tokens: 1,
      )
    end
  end
end
