require "json"
require "../../src/ml/gguf/qwen35_chat"
require "../../src/ml/gguf/qwen35_tokenizer"

include ML::GGUF

abort "usage: qlora_token_counts MODEL" unless ARGV.size == 1
model = ARGV[0]
file = GGUFFile.new(model, mmap_tensors: false)
tokenizer = begin
  Qwen35Tokenizer.from_gguf(file, model)
ensure
  file.close
end

STDIN.each_line.with_index(1) do |line, number|
  record = JSON.parse(line)
  messages = Qwen35Chat.messages_from_openai_json(record["messages"].to_json)
  completion = messages.pop
  raise ArgumentError.new("missing assistant completion in row #{number}") unless completion.role == "assistant"
  prompt = tokenizer.encode(Qwen35Chat.render(messages, enable_thinking: false), add_bos_override: false)
  target = tokenizer.encode(completion.content + "<|im_end|>", add_bos_override: false)
  puts({row: number, id: record["id"]?.try(&.as_s), prefix_tokens: prompt.size,
        target_tokens: target.size, input_tokens: prompt.size + target.size - 1}.to_json)
end
