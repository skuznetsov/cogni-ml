require "ml/llm/llama"

if ARGV.empty?
  STDERR.puts "Usage: crystal run examples/llm_inference.cr -- <model.gguf> [prompt]"
  exit 1
end

model_path = ARGV[0]
prompt = ARGV[1]? || "Hello!"

ML::LLM.init
begin
  model = ML::LLM::Model.new(model_path)
  generator = ML::LLM::Generator.new(model, prompt_mode: ML::LLM::PromptMode::Raw)

  puts "Model: #{model.description}"
  puts "Prompt: #{prompt}"
  puts "---"
  puts generator.ask(prompt, max_tokens: 128)
ensure
  generator.try(&.free)
  model.try(&.free)
  ML::LLM.cleanup
end
