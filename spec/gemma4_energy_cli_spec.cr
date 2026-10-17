require "spec"
require "../src/ml/gguf/gemma4_energy_cli"

private def energy_import_fixture : String
  {
    schema: "cogni.gemma4.energy.v1", backend: "llama.cpp", full_depth: true,
    prompt_sha256: "a" * 64, model_identity: "/models/gemma-4.gguf",
    runs: [{status: "completed", usage_complete: true, comparison_eligible: false,
            model_identity: "/models/gemma-4.gguf", answer: "partial", started_at_epoch_ms: 1, ended_at_epoch_ms: 2,
            profile: {name: "Eco", reasoning_tokens: 0, answer_tokens: 1},
            phases: {reasoning: {budget_tokens: 0, prompt_tokens: 0, predicted_tokens: 0, returned_token_ids: 0,
                                 control_overhead: 0, outcome: "skipped_zero_budget", stop_type: "skipped", stopping_word: "", forced_close: false, truncated: false},
                     answer: {budget_tokens: 1, prompt_tokens: 12, predicted_tokens: 1, returned_token_ids: 1,
                              control_overhead: 0, outcome: "answer_limit", stop_type: "limit", stopping_word: "", forced_close: false, truncated: true}}}],
  }.to_json
end

describe ML::GGUF::Gemma4EnergyCLI do
  it "keeps truncated runs explicit while rejecting imported quality approval and other models" do
    source = energy_import_fixture
    ML::GGUF::Gemma4EnergyCLI.validate_report!(JSON.parse(source))
    expect_raises(ArgumentError) do
      ML::GGUF::Gemma4EnergyCLI.validate_report!(JSON.parse(source.gsub("gemma-4", "mistral")))
    end
    forged = JSON.parse(source)
    forged["runs"][0].as_h["comparison_eligible"] = JSON::Any.new(true)
    expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyCLI.validate_report!(forged) }
  end

  it "rejects impossible or incomplete imported phase accounting" do
    mutations = [
      {"predicted_tokens", JSON::Any.new(0)},
      {"returned_token_ids", JSON::Any.new(2)},
      {"control_overhead", JSON::Any.new(1)},
      {"outcome", JSON::Any.new("eos")},
      {"truncated", JSON::Any.new(false)},
      {"forced_close", JSON::Any.new(true)},
    ]
    mutations.each do |key, value|
      report = JSON.parse(energy_import_fixture)
      report["runs"][0]["phases"]["answer"].as_h[key] = value
      expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyCLI.validate_report!(report) }
    end
    incomplete = JSON.parse(energy_import_fixture)
    incomplete["runs"][0]["phases"]["answer"].as_h.delete("returned_token_ids")
    expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyCLI.validate_report!(incomplete) }
  end
  it "rejects empty and incomplete imported reports before contacting VictoriaMetrics" do
    expect_raises(ArgumentError) do
      ML::GGUF::Gemma4EnergyCLI.validate_report!(JSON.parse(%({"schema":"cogni.gemma4.energy.v1","runs":[]})))
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Gemma4EnergyCLI.validate_report!(JSON.parse(%({"schema":"cogni.gemma4.energy.v1","runs":[{"started_at_epoch_ms":1,"ended_at_epoch_ms":2}]})))
    end
  end

  it "rejects conflicting profile and prompt sources and canonical profile duplicates" do
    [
      ["--profile", "eco", "--profiles", "quality", "--dry-run"],
      ["--profiles", "eco,ECO", "--dry-run"],
      ["--prompt", "one", "two", "--dry-run"],
      ["--reasoning-tokens", "1", "--reasoning-tokens", "2", "--dry-run"],
    ].each do |args|
      expect_raises(ArgumentError) { ML::GGUF::Gemma4EnergyCLI.run(args) }
    end
  end
end
