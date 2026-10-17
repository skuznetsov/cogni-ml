require "spec"
require "http/server"
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

  it "keeps nonstream settings by default and enables token-timed streaming only by opt-in" do
    rendered = "rendered prompt"
    baseline = JSON.parse(ML::GGUF::Gemma4LayerExperiment.request_payload(rendered, 64))
    baseline["stream"].as_bool.should be_false
    baseline["return_tokens"].as_bool.should be_true
    baseline["timings_per_token"].as_bool.should be_false

    progress = JSON.parse(ML::GGUF::Gemma4LayerExperiment.request_payload(rendered, 64, true))
    progress["stream"].as_bool.should be_true
    progress["return_tokens"].as_bool.should be_true
    progress["timings_per_token"].as_bool.should be_true
  end

  it "aggregates token IDs from SSE frames while preserving empty text chunks" do
    server = HTTP::Server.new do |context|
      context.response.headers["Content-Type"] = "text/event-stream"
      context.response.print("data: {\"content\":\"\",\"tokens\":[41,42],\"tokens_predicted\":2,\"stop\":false}\n\n")
      sleep 2.milliseconds
      context.response.print("data: {\"content\":\"answer\",\"tokens\":[],\"tokens_predicted\":2,\"stop\":true,\"stop_type\":\"limit\",\"tokens_evaluated\":10,\"truncated\":false,\"timings\":{\"predicted_n\":2}}\n\n")
    end
    address = server.bind_unused_port
    spawn { server.listen }
    begin
      start_epoch = Time.utc.to_unix_ms
      monotonic = Time.instant
      collector = ML::GGUF::Gemma4TokenProgress::Collector.new(start_epoch, 2)
      response, streamed_content = ML::GGUF::Gemma4LayerExperiment.stream_completion(
        "http://127.0.0.1:#{address.port}", "{}", collector, monotonic)
      ML::GGUF::Gemma4LayerExperiment.validate_completion!(response, 2).should eq(2)
      response["tokens"].as_a.map(&.as_i64).should eq([41_i64, 42_i64])
      streamed_content.should eq("answer")
      collector.finish!(Time.utc.to_unix_ms, monotonic.elapsed.total_milliseconds.to_i64)
      collector.to_json["status"].as_s.should eq("available")
    ensure
      server.close unless server.closed?
    end
  end
end
