require "spec"
require "../src/ml/gguf/gemma4_layer_experiment"

describe ML::GGUF::Gemma4LayerExperiment do
  it "rejects malformed measurement URLs before contacting the inference server" do
    expect_raises(ArgumentError, /VictoriaMetrics URL/) do
      ML::GGUF::Gemma4LayerExperiment.run([
        "--backend-url", "http://127.0.0.1:1", "--vm-url", "invalid",
        "--layers", "16", "--label", "guard", "--campaign-id", "guard",
        "--model-file", "/nonexistent/gemma4.gguf",
        "--output", "/nonexistent/gemma4-layer-report.json",
      ])
    end
  end

  it "admits only exact work with consistent token IDs" do
    completed = JSON.parse(%({"tokens_predicted":3,"tokens_evaluated":12,"stop_type":"limit","tokens":[1,2,3],"truncated":false}))
    ML::GGUF::Gemma4LayerExperiment.validate_completion!(completed, 3).should eq(3)
    [
      {"tokens_predicted", JSON::Any.new(2_i64)},
      {"stop_type", JSON::Any.new("eos")},
      {"truncated", JSON::Any.new(true)},
      {"tokens", JSON.parse("[1,2]")},
      {"tokens_evaluated", JSON::Any.new(0_i64)},
    ].each do |key, value|
      malformed = JSON.parse(completed.to_json)
      malformed.as_h[key] = value
      expect_raises(ArgumentError) { ML::GGUF::Gemma4LayerExperiment.validate_completion!(malformed, 3) }
    end
  end
end
