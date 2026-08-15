require "./spec_helper"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_tokenizer"

private alias QwenEngine = ML::GGUF::Qwen35Engine

QWEN_9B_CHAT     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN_38_27B_CHAT = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35Chat do
  it "renders Qwen XML tool instructions and user prompt" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]))
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Weather in Paris?",
      system: "You are terse.",
      tools: tools,
    )

    rendered.should contain("<|im_start|>system\n# Tools")
    rendered.should contain("<tools>\n{\"type\":\"function\"")
    rendered.should contain("<tool_call>\n<function=example_function_name>")
    rendered.should contain("You are terse.<|im_end|>\n")
    rendered.should contain("<|im_start|>user\nWeather in Paris?<|im_end|>\n")
    rendered.ends_with?("<|im_start|>assistant\n").should be_true
  end

  it "renders Qwen no-thinking generation prompt when requested" do
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Write Crystal code.",
      enable_thinking: false,
    )

    rendered.should end_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
  end

  it "renders Qwen thinking generation prompt when explicitly enabled" do
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Think before answering.",
      enable_thinking: true,
    )

    rendered.should end_with("<|im_start|>assistant\n<think>\n")
  end

  it "renders exact Qwen3.8 low reasoning instructions and opens thinking" do
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Answer briefly.",
      reasoning_effort: QwenEngine::ReasoningEffort::Low,
    )

    rendered.should eq(
      "<|im_start|>system\n#{ML::GGUF::Qwen35Chat::LOW_REASONING_INSTRUCTION}<|im_end|>\n" \
      "<|im_start|>user\nAnswer briefly.<|im_end|>\n" \
      "<|im_start|>assistant\n<think>\n"
    )
  end

  it "renders medium reasoning without inventing a system instruction" do
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Solve it.",
      system: "Be precise.",
      reasoning_effort: QwenEngine::ReasoningEffort::Medium,
    )

    rendered.should eq(
      "<|im_start|>system\nBe precise.<|im_end|>\n" \
      "<|im_start|>user\nSolve it.<|im_end|>\n" \
      "<|im_start|>assistant\n<think>\n"
    )
  end

  it "shares the pre-generation prefix between none and medium only" do
    messages = [ML::GGUF::Qwen35Chat::Message.new("user", "Solve it.")]
    none_prefix = ML::GGUF::Qwen35Chat.render(
      messages,
      add_generation_prompt: false,
      reasoning_effort: QwenEngine::ReasoningEffort::None,
    )
    medium_prefix = ML::GGUF::Qwen35Chat.render(
      messages,
      add_generation_prompt: false,
      reasoning_effort: QwenEngine::ReasoningEffort::Medium,
    )
    none_prompt = ML::GGUF::Qwen35Chat.render(
      messages,
      reasoning_effort: QwenEngine::ReasoningEffort::None,
    )
    medium_prompt = ML::GGUF::Qwen35Chat.render(
      messages,
      reasoning_effort: QwenEngine::ReasoningEffort::Medium,
    )

    medium_prefix.should eq(none_prefix)
    medium_prompt.should_not eq(none_prompt)
  end

  it "places xhigh instructions before Qwen tool instructions" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]))
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Find it.",
      system: "Use trusted sources.",
      tools: tools,
      reasoning_effort: QwenEngine::ReasoningEffort::XHigh,
    )

    system_prefix = "<|im_start|>system\n#{ML::GGUF::Qwen35Chat::XHIGH_REASONING_INSTRUCTION}\n\n# Tools"
    rendered.should start_with(system_prefix)
    rendered.should contain("</IMPORTANT>\n\nUse trusted sources.<|im_end|>\n")
    rendered.should end_with("<|im_start|>assistant\n<think>\n")
  end

  it "maps engine none to the existing no-thinking suffix and rejects conflicts" do
    rendered = ML::GGUF::Qwen35Chat.render_user_prompt(
      "Answer directly.",
      reasoning_effort: QwenEngine::ReasoningEffort::None,
    )
    rendered.should end_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    expect_raises(ArgumentError, /conflicts/) do
      ML::GGUF::Qwen35Chat.render_user_prompt(
        "Answer directly.",
        enable_thinking: true,
        reasoning_effort: QwenEngine::ReasoningEffort::None,
      )
    end
  end

  it "detects only the exact Qwen3.8 effort template contract" do
    exact = "enable_thinking reasoning_effort #{ML::GGUF::Qwen35Chat::LOW_REASONING_INSTRUCTION} #{ML::GGUF::Qwen35Chat::XHIGH_REASONING_INSTRUCTION}"
    ML::GGUF::Qwen35Chat.supports_reasoning_effort?(exact).should be_true
    ML::GGUF::Qwen35Chat.supports_reasoning_effort?("enable_thinking reasoning_effort").should be_false
    ML::GGUF::Qwen35Chat.supports_reasoning_effort?(nil).should be_false
  end

  it "parses Qwen XML tool calls with multiline parameters" do
    text = "I will call it.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=notes>\nline one\nline two\n</parameter>\n</function>\n</tool_call>"
    calls = ML::GGUF::Qwen35Chat.parse_tool_calls(text)

    calls.size.should eq(1)
    calls[0].name.should eq("get_weather")
    calls[0].arguments["city"].should eq("Paris")
    calls[0].arguments["notes"].should eq("line one\nline two")
    ML::GGUF::Qwen35Chat.tool_calls_to_json(calls).should contain("get_weather")
  end

  it "normalizes Qwen XML tool calls to CrystalBall-style JSON" do
    text = "Checking.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/foo.cr\n</parameter>\n<parameter=limit>\n3\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n</function>\n</tool_call>"
    calls = ML::GGUF::Qwen35Chat.parse_tool_calls(text)
    payload = JSON.parse(ML::GGUF::Qwen35Chat.tool_response_to_json(
      calls,
      ML::GGUF::Qwen35Chat.content_without_tool_calls(text),
    ))

    payload["content"].as_s.should eq("Checking.")
    payload["tool_calls"].as_a.size.should eq(1)
    call = payload["tool_calls"][0]
    call["name"].as_s.should eq("read_file")
    call["arguments"]["path"].as_s.should eq("src/foo.cr")
    call["arguments"]["limit"].as_i.should eq(3)
    call["arguments"]["exact"].as_bool.should be_true
  end

  it "uses tool schema when normalizing typed arguments" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
        "path":{"type":"string"},
        "limit":{"type":"integer"},
        "exact":{"type":"boolean"}
      },"required":["path"]}}}
    ]))
    text = "<tool_call>\n<function=read_file>\n<parameter=path>\n3\n</parameter>\n<parameter=limit>\n3\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n</function>\n</tool_call>"
    calls = ML::GGUF::Qwen35Chat.parse_tool_calls(text)
    payload = JSON.parse(ML::GGUF::Qwen35Chat.tool_response_to_json(calls, nil, tools))
    args = payload["tool_calls"][0]["arguments"]

    args["path"].as_s.should eq("3")
    args["limit"].as_i.should eq(3)
    args["exact"].as_bool.should be_true
  end

  it "uses tool schema for OpenAI-style function arguments" do
    tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"set_limit","parameters":{"type":"object","properties":{
        "limit":{"type":"integer"}
      },"required":["limit"]}}}
    ]))
    calls = [ML::GGUF::Qwen35Chat::ToolCall.new("set_limit", {"limit" => "3"})]
    payload = JSON.parse(ML::GGUF::Qwen35Chat.tool_response_to_openai_json(calls, nil, tools))
    args = JSON.parse(payload["tool_calls"][0]["function"]["arguments"].as_s)

    args["limit"].as_i.should eq(3)
  end

  it "can emit OpenAI-style tool call wrappers" do
    calls = [ML::GGUF::Qwen35Chat::ToolCall.new("grep", {"pattern" => "class Foo"})]
    payload = JSON.parse(ML::GGUF::Qwen35Chat.tool_response_to_openai_json(calls))
    tool_call = payload["tool_calls"][0]

    tool_call["id"].as_s.should eq("call_0")
    tool_call["type"].as_s.should eq("function")
    tool_call["function"]["name"].as_s.should eq("grep")
    JSON.parse(tool_call["function"]["arguments"].as_s)["pattern"].as_s.should eq("class Foo")
  end

  it "converts OpenAI messages with assistant tool calls into Qwen chat messages" do
    request_messages = [
      JSON::Any.new({
        "role"    => JSON::Any.new("user"),
        "content" => JSON::Any.new("Read src/foo.cr"),
      }),
      JSON::Any.new({
        "role"       => JSON::Any.new("assistant"),
        "content"    => JSON::Any.new(nil),
        "tool_calls" => JSON::Any.new([
          JSON::Any.new({
            "id"       => JSON::Any.new("call_0"),
            "type"     => JSON::Any.new("function"),
            "function" => JSON::Any.new({
              "name"      => JSON::Any.new("read_file"),
              "arguments" => JSON::Any.new({"path" => "src/foo.cr"}.to_json),
            }),
          }),
        ]),
      }),
      JSON::Any.new({
        "role"         => JSON::Any.new("tool"),
        "tool_call_id" => JSON::Any.new("call_0"),
        "content"      => JSON::Any.new("class Foo\nend"),
      }),
    ]
    messages = ML::GGUF::Qwen35Chat.messages_from_openai_json(JSON::Any.new(request_messages).to_json)
    rendered = ML::GGUF::Qwen35Chat.render(messages, add_generation_prompt: true)

    rendered.should contain("<|im_start|>user\nRead src/foo.cr<|im_end|>")
    rendered.should_not contain("<|im_start|>assistant\nnull")
    rendered.should contain("<tool_call>\n<function=read_file>")
    rendered.should contain("<parameter=path>\nsrc/foo.cr\n</parameter>")
    rendered.should contain("<|im_start|>tool\nclass Foo\nend<|im_end|>")
  end

  it "loads chat_template metadata from local Qwen GGUF when present" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_CHAT)
    g = ML::GGUF::GGUFFile.new(QWEN_9B_CHAT)
    tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, QWEN_9B_CHAT)
    g.close

    tok.chat_template.should_not be_nil
    tok.chat_template.not_nil!.should contain("<tool_call>")
    tok.chat_template.not_nil!.should contain("tools")
  end

  it "detects reasoning effort from real Qwen templates when present" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_27B_CHAT)

    qwen38_gguf = ML::GGUF::GGUFFile.new(QWEN_38_27B_CHAT)
    qwen38 = ML::GGUF::Qwen35Tokenizer.from_gguf(qwen38_gguf, QWEN_38_27B_CHAT)
    qwen38_gguf.close
    ML::GGUF::Qwen35Chat.supports_reasoning_effort?(qwen38.chat_template).should be_true

    if File.exists?(QWEN_9B_CHAT)
      qwen35_gguf = ML::GGUF::GGUFFile.new(QWEN_9B_CHAT)
      qwen35 = ML::GGUF::Qwen35Tokenizer.from_gguf(qwen35_gguf, QWEN_9B_CHAT)
      qwen35_gguf.close
      ML::GGUF::Qwen35Chat.supports_reasoning_effort?(qwen35.chat_template).should be_false
    end
  end
end
