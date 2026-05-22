require "./spec_helper"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_tokenizer"

QWEN_9B_CHAT = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

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

  it "parses Qwen XML tool calls with multiline parameters" do
    text = "I will call it.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=notes>\nline one\nline two\n</parameter>\n</function>\n</tool_call>"
    calls = ML::GGUF::Qwen35Chat.parse_tool_calls(text)

    calls.size.should eq(1)
    calls[0].name.should eq("get_weather")
    calls[0].arguments["city"].should eq("Paris")
    calls[0].arguments["notes"].should eq("line one\nline two")
    ML::GGUF::Qwen35Chat.tool_calls_to_json(calls).should contain("get_weather")
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
end
