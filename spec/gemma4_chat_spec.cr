require "./spec_helper"
require "../src/ml/gguf/gemma4_chat"

GEMMA4_TOOL_JSON = %([{
  "type":"function",
  "function":{
    "name":"set_flag",
    "description":"Set a boolean flag",
    "parameters":{
      "type":"object",
      "properties":{
        "enabled":{"type":"boolean","description":"Whether to enable it"},
        "mode":{"type":"string","enum":["fast","safe"]}
      },
      "required":["enabled"]
    }
  }
}])

describe ML::GGUF::Gemma4Chat do
  it "renders Gemma4 native tool declarations and generation prompt" do
    tools = ML::GGUF::Gemma4Chat.parse_tools_json(GEMMA4_TOOL_JSON)
    rendered = ML::GGUF::Gemma4Chat.render_user_prompt("Set safe mode", system: "Be precise.", tools: tools)

    rendered.should start_with("<|turn>system\nBe precise.<|tool>declaration:set_flag")
    rendered.should contain("description:<|\"|>Set a boolean flag<|\"|>")
    rendered.should contain("enabled:{description:<|\"|>Whether to enable it<|\"|>,type:<|\"|>BOOLEAN<|\"|>}")
    rendered.should contain("required:[<|\"|>enabled<|\"|>]")
    rendered.should contain("<|turn>user\nSet safe mode<turn|>\n")
    rendered.should end_with("<|turn>model\n<|channel>thought\n<channel|>")
  end

  it "parses Gemma4 native tool calls and normalizes typed JSON" do
    tools = ML::GGUF::Gemma4Chat.parse_tools_json(GEMMA4_TOOL_JSON)
    text = "Calling.\n<|tool_call>call:set_flag{enabled:true,mode:<|\"|>safe<|\"|>}<tool_call|>"
    calls = ML::GGUF::Gemma4Chat.parse_tool_calls(text)
    payload = JSON.parse(ML::GGUF::Gemma4Chat.tool_response_to_json(calls, ML::GGUF::Gemma4Chat.content_without_tool_calls(text), tools))

    payload["content"].as_s.should eq("Calling.")
    call = payload["tool_calls"].as_a[0]
    call["name"].as_s.should eq("set_flag")
    call["arguments"]["enabled"].as_bool.should be_true
    call["arguments"]["mode"].as_s.should eq("safe")
  end

  it "does not split Gemma4 string arguments on commas" do
    text = %(<|tool_call>call:note{body:<|"|>alpha,beta<|"|>,enabled:true}<tool_call|>)
    calls = ML::GGUF::Gemma4Chat.parse_tool_calls(text)

    calls[0].arguments["body"].should eq("alpha,beta")
    calls[0].arguments["enabled"].should eq("true")
  end

  it "renders assistant tool calls using Gemma4 native syntax" do
    message = ML::GGUF::Gemma4Chat::Message.new(
      "assistant",
      "",
      [ML::GGUF::Gemma4Chat::ToolCall.new("set_flag", {"enabled" => "true", "mode" => "safe"})],
    )

    rendered = ML::GGUF::Gemma4Chat.render([message], add_generation_prompt: false)
    rendered.should eq("<|turn>model\n<|tool_call>call:set_flag{enabled:<|\"|>true<|\"|>,mode:<|\"|>safe<|\"|>}<tool_call|><turn|>\n")
  end

  it "builds Gemma4 native finite tool-call options from schemas" do
    tools = ML::GGUF::Gemma4Chat.parse_tools_json(%([
      {"type":"function","function":{"name":"set_mode","parameters":{"type":"object","properties":{
        "mode":{"type":"string","enum":["fast","safe"]},
        "enabled":{"type":"boolean"},
        "limit":{"type":"integer","minimum":1,"maximum":2}
      }}}}
    ]))

    options = ML::GGUF::Gemma4Chat.native_tool_finite_call_options(tools)
    options.should contain("<|tool_call>call:set_mode{mode:<|\"|>fast<|\"|>}<tool_call|>")
    options.should contain("<|tool_call>call:set_mode{enabled:true}<tool_call|>")
    options.should contain("<|tool_call>call:set_mode{limit:2}<tool_call|>")
  end

  it "tracks token-option corridors through singleton spans and branch points" do
    corridor = ML::GGUF::Gemma4Chat::TokenOptionCorridor.from_options([
      [10, 20, 30],
      [10, 20, 40],
      [10, 50],
    ])

    corridor.next_ids.should eq([10])
    corridor = corridor.advance(10)
    corridor.next_ids.should eq([20, 50])
    corridor = corridor.advance(20)
    corridor.next_ids.should eq([30, 40])
    corridor = corridor.advance(30)
    corridor.complete?.should be_true
  end

  it "maps an emitted token trace back to the selected finite literal option" do
    options = [
      [1, 2, 3],
      [1, 2, 4],
      [1, 2, 4, 5],
    ]

    ML::GGUF::Gemma4Chat::TokenOptionCorridor.selected_literal_index?(options, [1, 2, 4]).should eq(1)
    ML::GGUF::Gemma4Chat::TokenOptionCorridor.selected_literal_index?(options, [1, 2, 4, 5]).should eq(2)
    ML::GGUF::Gemma4Chat::TokenOptionCorridor.selected_literal_index?(options, [1, 9]).should be_nil
  end
end
