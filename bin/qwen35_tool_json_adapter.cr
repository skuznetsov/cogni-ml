require "json"
require "option_parser"
require "../src/ml/gguf/qwen35_chat"

mode = "parse-output"
format = "simple"
content_override = nil.as(String?)

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_tool_json_adapter [--parse-output|--render-request] [--format=simple|openai]"
  parser.on("--parse-output", "Parse generated Qwen XML tool calls from stdin and emit JSON tool response") do
    mode = "parse-output"
  end
  parser.on("--render-request", "Render an OpenAI/CrystalBall-style JSON request from stdin into Qwen chat prompt") do
    mode = "render-request"
  end
  parser.on("--format=FORMAT", "Tool response format: simple or openai") do |value|
    format = value.downcase
  end
  parser.on("--content=TEXT", "Override response content when parsing generated output") do |value|
    content_override = value
  end
  parser.on("-h", "--help", "Show help") do
    puts parser
    exit
  end
end

unless format == "simple" || format == "openai"
  raise "format must be simple or openai"
end

input = STDIN.gets_to_end

case mode
when "parse-output"
  calls = ML::GGUF::Qwen35Chat.parse_tool_calls(input)
  content = content_override || ML::GGUF::Qwen35Chat.content_without_tool_calls(input)
  if format == "openai"
    puts ML::GGUF::Qwen35Chat.tool_response_to_openai_json(calls, content)
  else
    puts ML::GGUF::Qwen35Chat.tool_response_to_json(calls, content)
  end
when "render-request"
  request = JSON.parse(input).as_h
  tools = if raw_tools = request["tools"]?
            raw_tools.as_a
          else
            [] of JSON::Any
          end
  messages = if raw_messages = request["messages"]?
               ML::GGUF::Qwen35Chat.messages_from_openai_json(raw_messages.to_json)
             elsif prompt = request["prompt"]?.try(&.as_s?)
               [ML::GGUF::Qwen35Chat::Message.new("user", prompt)]
             else
               raise "request must contain messages or prompt"
             end
  puts ML::GGUF::Qwen35Chat.render(messages, tools: tools, add_generation_prompt: true)
else
  raise "unknown mode: #{mode}"
end
