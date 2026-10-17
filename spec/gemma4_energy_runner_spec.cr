require "spec"
require "../src/ml/gguf/gemma4_energy_runner"

private def gemma4_energy_json(source : String) : JSON::Any
  JSON.parse(source)
end

private def gemma4_energy_server_props(model_path : String = "/models/gemma-4-12b-it-Q4_K_M.gguf", n_ctx : Int32 = 8192) : JSON::Any
  gemma4_energy_json(%({
    "model_path": #{model_path.to_json},
    "default_generation_settings": {"n_ctx": #{n_ctx}}
  }))
end

private def gemma4_energy_marker_tokens : JSON::Any
  gemma4_energy_json(%({"tokens":[
    {"id":1,"piece":"<|think|>"},
    {"id":2,"piece":"<|channel>"},
    {"id":3,"piece":"<channel|>"},
    {"id":4,"piece":"<|turn>"},
    {"id":5,"piece":"<turn|>"}
  ]}))
end

private def gemma4_energy_completion(content : String, predicted_n : Int32,
                                     stop_type : String, stopping_word : String = "",
                                     tokens : String = "[7,8,9]", prompt_n : Int32 = 12,
                                     truncated : Bool = false) : JSON::Any
  gemma4_energy_json(%({
    "content": #{content.to_json},
    "tokens": #{tokens},
    "stop_type": #{stop_type.to_json},
    "stopping_word": #{stopping_word.to_json},
    "timings": {"predicted_n": #{predicted_n}},
    "tokens_evaluated": #{prompt_n},
    "truncated": #{truncated}
  }))
end

describe ML::GGUF::Gemma4EnergyRunner do
  it "runs bounded reasoning then a separately bounded answer and keeps reasoning private" do
    requests = [] of Tuple(String, JSON::Any?)
    completions = [
      gemma4_energy_completion("private chain", 3, "limit", tokens: "[7,8,9]"),
      gemma4_energy_completion("42 amps", 2, "word", "<turn|>", tokens: "[10,11]", prompt_n: 21),
    ]
    transport = ->(path : String, body : JSON::Any?) do
      requests << {path, body}
      case path
      when "/props"
        gemma4_energy_server_props
      when "/tokenize"
        gemma4_energy_marker_tokens
      when "/completion"
        completions.shift
      else
        raise "unexpected endpoint"
      end
    end

    runner = ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: transport)
    report = runner.run(
      "What is the reading?",
      ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 3),
    )

    report["answer"].as_s.should eq("42 amps")
    report["model_identity"].as_s.should eq("/models/gemma-4-12b-it-Q4_K_M.gguf")
    report["phases"]["reasoning"]["outcome"].as_s.should eq("forced_close")
    report["phases"]["reasoning"]["predicted_tokens"].as_i.should eq(3)
    report["phases"]["answer"]["outcome"].as_s.should eq("turn_end")
    report["phases"]["answer"]["budget_tokens"].as_i.should eq(512)
    report.to_json.should_not contain("private chain")
    runner.model_identity.should eq("/models/gemma-4-12b-it-Q4_K_M.gguf")

    completion_bodies = requests.select { |path, _| path == "/completion" }.map(&.[1].not_nil!)
    completion_bodies.size.should eq(2)
    reason = completion_bodies[0]
    reason["n_predict"].as_i.should eq(3)
    reason["temperature"].as_f.should eq(0.0)
    reason["seed"].as_i.should eq(1)
    reason["cache_prompt"].as_bool.should be_false
    reason["return_tokens"].as_bool.should be_true
    reason["prompt"].as_s.should contain("<|channel>thought\n")
    reason["stop"].as_a.map(&.as_s).should eq(["<channel|>", "<turn|>"])

    answer = completion_bodies[1]
    answer["n_predict"].as_i.should eq(512)
    answer_prompt = answer["prompt"].as_s
    answer_prompt.should contain("private chain<channel|>")
    answer_prompt.should contain("<|turn>model\n<|channel>thought\n")
  end

  it "distinguishes a natural thought delimiter from forced closure and preserves answer budget" do
    requests = [] of Tuple(String, JSON::Any?)
    completions = [
      gemma4_energy_completion("reasoned", 4, "word", "<channel|>", tokens: "[1,2,3]"),
      gemma4_energy_completion("done", 5, "eos", tokens: "[5,6,7,8,9]"),
    ]
    transport = ->(path : String, body : JSON::Any?) do
      requests << {path, body}
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then completions.shift
      else                    raise "unexpected endpoint"
      end
    end

    report = ML::GGUF::Gemma4EnergyRunner.new(
      "http://127.0.0.1:8080",
      request_transport: transport,
    ).run(
      "prompt",
      ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 8, answer_tokens: 19),
    )

    report["phases"]["reasoning"]["outcome"].as_s.should eq("natural_channel_close")
    report["phases"]["reasoning"]["predicted_tokens"].as_i.should eq(4)
    report["phases"]["reasoning"]["control_overhead"].as_i.should eq(1)
    report["phases"]["answer"]["budget_tokens"].as_i.should eq(19)
    report["started_at_epoch_ms"].as_i64.should be > 0
    report["ended_at_epoch_ms"].as_i64.should be >= report["started_at_epoch_ms"].as_i64
    report["elapsed_ms"].as_i64.should be >= 0
    answer_prompt = requests.select { |path, _| path == "/completion" }.last[1].not_nil!["prompt"].as_s
    answer_prompt.should contain("reasoned<channel|>")
  end

  it "skips a zero-token reasoning request while closing the empty thought channel for the answer" do
    requests = [] of Tuple(String, JSON::Any?)
    transport = ->(path : String, body : JSON::Any?) do
      requests << {path, body}
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then gemma4_energy_completion("answer", 1, "eos", tokens: "[9]")
      else                    raise "unexpected endpoint"
      end
    end

    report = ML::GGUF::Gemma4EnergyRunner.new(
      "http://127.0.0.1:8080",
      request_transport: transport,
    ).run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("eco"))

    report["phases"]["reasoning"]["outcome"].as_s.should eq("skipped_zero_budget")
    requests.count { |path, _| path == "/completion" }.should eq(1)
    answer_prompt = requests.select { |path, _| path == "/completion" }.first[1].not_nil!["prompt"].as_s
    answer_prompt.should_not contain("<|think|>")
    answer_prompt.ends_with?("<|channel>thought\n<channel|>").should be_true
    answer_prompt.scan(/<channel\|>/).size.should eq(1)
  end

  it "fails closed on unexpected reasoning EOS, context truncation, invalid usage, and non-Gemma models" do
    eos_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then gemma4_energy_completion("unfinished", 1, "eos", tokens: "[1]")
      else                    raise "unexpected endpoint"
      end
    end
    runner = ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: eos_transport)
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      runner.run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("balanced"))
    end

    truncated_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then gemma4_energy_completion("partial", 3, "limit", tokens: "[1,2,3]", truncated: true)
      else                    raise "unexpected endpoint"
      end
    end
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: truncated_transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("balanced"))
    end

    bad_usage_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then gemma4_energy_completion("too much", 257, "limit", tokens: "[1,2]")
      else                    raise "unexpected endpoint"
      end
    end
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: bad_usage_transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("balanced"))
    end

    bad_ids_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then gemma4_energy_completion("bad count", 3, "limit", tokens: "[1,2,3,4]")
      else                    raise "unexpected endpoint"
      end
    end
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: bad_ids_transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 3))
    end

    zero_usage_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then gemma4_energy_completion("fabricated answer", 0, "eos", tokens: "[]")
      else                    raise "unexpected endpoint"
      end
    end
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: zero_usage_transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("eco"))
    end

    non_gemma_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props" then gemma4_energy_server_props("/models/not-gemma.gguf")
      else               raise "unexpected endpoint"
      end
    end
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: non_gemma_transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("eco"))
    end

    marker_transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"    then gemma4_energy_server_props
      when "/tokenize" then gemma4_energy_json(%({"tokens":[{"id":1,"piece":"<|think|>"}]}))
      else                  raise "unexpected endpoint"
      end
    end
    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: marker_transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("eco"))
    end
  end

  it "marks answer budget exhaustion as truncation instead of claiming a complete answer" do
    completions = [
      gemma4_energy_completion("reason", 1, "word", "<channel|>", tokens: "[1]"),
      gemma4_energy_completion("partial answer", 7, "limit", tokens: "[2,3,4,5,6,7,8]"),
    ]
    transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then completions.shift
      else                    raise "unexpected endpoint"
      end
    end

    report = ML::GGUF::Gemma4EnergyRunner.new(
      "http://127.0.0.1:8080",
      request_transport: transport,
    ).run(
      "prompt",
      ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 2, answer_tokens: 7),
    )

    report["answer"].as_s.should eq("partial answer")
    report["phases"]["answer"]["outcome"].as_s.should eq("answer_limit")
    report["phases"]["answer"]["truncated"].as_bool.should be_true
  end

  it "rejects answer output that reopens a private thought channel" do
    ["<|channel>thought\nprivate", "<|think|>private"].each do |answer_content|
      completions = [
        gemma4_energy_completion("reason", 1, "word", "<channel|>", tokens: "[1]"),
        gemma4_energy_completion(answer_content, 2, "eos", tokens: "[2,3]"),
      ]
      transport = ->(path : String, _body : JSON::Any?) do
        case path
        when "/props"      then gemma4_energy_server_props
        when "/tokenize"   then gemma4_energy_marker_tokens
        when "/completion" then completions.shift
        else                    raise "unexpected endpoint"
        end
      end

      expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
        ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: transport)
          .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 2))
      end
    end
  end

  it "rejects an empty answer even when the backend reports normal EOS" do
    completions = [
      gemma4_energy_completion("reason", 1, "word", "<channel|>", tokens: "[1]"),
      gemma4_energy_completion("", 0, "eos", tokens: "[]"),
    ]
    transport = ->(path : String, _body : JSON::Any?) do
      case path
      when "/props"      then gemma4_energy_server_props
      when "/tokenize"   then gemma4_energy_marker_tokens
      when "/completion" then completions.shift
      else                    raise "unexpected endpoint"
      end
    end

    expect_raises(ML::GGUF::Gemma4EnergyRunner::Error) do
      ML::GGUF::Gemma4EnergyRunner.new("http://127.0.0.1:8080", request_transport: transport)
        .run("prompt", ML::GGUF::Gemma4EnergyProfile.for_name("balanced", reasoning_tokens: 2))
    end
  end
end
