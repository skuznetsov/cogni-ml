require "json"
require "option_parser"
require "../src/ml/gguf/gemma4_chat"

mode = "parse-output"
format = "simple"
content_override = nil.as(String?)

OptionParser.parse do |parser|
  parser.banner = "Usage: gemma4_tool_json_adapter [--parse-output|--render-request] [--format=simple|openai]"
  parser.on("--parse-output", "Parse generated Gemma4 native tool calls from stdin and emit JSON tool response") { mode = "parse-output" }
  parser.on("--render-request", "Render an OpenAI/CrystalBall-style JSON request from stdin into a Gemma4 chat prompt") { mode = "render-request" }
  parser.on("--format=FORMAT", "Tool response format: simple or openai") { |value| format = value.downcase }
  parser.on("--content=TEXT", "Override response content when parsing generated output") { |value| content_override = value }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "format must be simple or openai" unless format == "simple" || format == "openai"

input = STDIN.gets_to_end
case mode
when "parse-output"
  calls = ML::GGUF::Gemma4Chat.parse_tool_calls(input)
  content = content_override || ML::GGUF::Gemma4Chat.content_without_tool_calls(input)
  if format == "openai"
    puts ML::GGUF::Gemma4Chat.tool_response_to_openai_json(calls, content)
  else
    puts ML::GGUF::Gemma4Chat.tool_response_to_json(calls, content)
  end
when "render-request"
  request = JSON.parse(input).as_h
  tools = request["tools"]?.try(&.as_a) || [] of JSON::Any
  messages = if raw_messages = request["messages"]?
               ML::GGUF::Gemma4Chat.messages_from_openai_json(raw_messages.to_json)
             elsif prompt = request["prompt"]?.try(&.as_s?)
               [ML::GGUF::Gemma4Chat::Message.new("user", prompt)]
             else
               raise "request must contain messages or prompt"
             end
  puts ML::GGUF::Gemma4Chat.render(messages, tools: tools, add_generation_prompt: true)
else
  raise "unknown mode: #{mode}"
end
