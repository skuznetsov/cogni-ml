#!/usr/bin/env crystal

# Synthetic, non-private prompt suite for speculative-router data collection.
# Default output is a pipe-separated NAME::TEXT list accepted by qwen35_speculative_sweep --only-prompts.

require "json"
require "option_parser"
require "set"

record PromptCase, name : String, category : String, text : String

limit = 0
offset = 0
format = "sweep"
categories_filter = [] of String

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_spec_router_prompt_suite [--limit N] [--offset N] [--format sweep|jsonl|plain] [--categories LIST]"
  p.on("--limit N", "Maximum prompts to emit; 0 means all (default: 0)") { |v| limit = v.to_i }
  p.on("--offset N", "Skip the first N prompts after category filtering (default: 0)") { |v| offset = v.to_i }
  p.on("--format FORMAT", "Output format: sweep, jsonl, or plain (default: sweep)") { |v| format = v }
  p.on("--categories LIST", "Comma-separated categories to keep") { |v| categories_filter = v.split(',').map(&.strip).reject(&.empty?) }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "--limit must be non-negative" if limit < 0
abort "--offset must be non-negative" if offset < 0
abort "--format must be sweep, jsonl, or plain" unless {"sweep", "jsonl", "plain"}.includes?(format)

prompts = [] of PromptCase

facts = {
  "france_capital" => "The capital of France is",
  "japan_capital"  => "The capital of Japan is",
  "mars_planet"    => "Mars is the fourth planet from",
  "water_freezes"  => "Water freezes at",
  "einstein"       => "Albert Einstein developed the theory of",
  "photosynthesis" => "Photosynthesis converts sunlight into",
  "cpu_gpu"        => "A CPU differs from a GPU because",
  "postgres_index" => "A PostgreSQL index helps queries by",
}
facts.each { |name, text| prompts << PromptCase.new("fact_#{name}", "fact", text) }

code = {
  "fib_py"      => "def fibonacci(n):",
  "fib_crystal" => "def fibonacci(n : Int32)",
  "json_parse"  => "import json\n\ndef load_config(path):",
  "http_server" => "require \"http/server\"\nserver = HTTP::Server.new do |context|",
  "sql_index"   => "CREATE INDEX idx_events_created_at ON events",
  "rust_vec"    => "fn sum(values: &[i32]) -> i32 {",
  "go_handler"  => "func handler(w http.ResponseWriter, r *http.Request) {",
  "js_map"      => "const doubled = values.map((value) =>",
  "cpp_loop"    => "for (int i = 0; i < n; ++i) {",
  "bash_find"   => "find . -name '*.cr' -print0 |",
}
code.each { |name, text| prompts << PromptCase.new("code_#{name}", "code", text) }

repeats = {
  "alpha4"       => "alpha beta gamma delta alpha beta gamma delta",
  "log_level"    => "INFO start INFO start INFO start",
  "csv_header"   => "id,name,value\n1,alice,10\n2,bob,20\nid,name,value",
  "markdown_tbl" => "| name | score |\n| cat | 3 |\n| dog | 5 |\n| name | score |",
  "yaml_hosts"   => "hosts:\n  - name: web\n    port: 443\n  - name: web",
  "json_pairs"   => "{\"a\":1,\"b\":2,\"a\":1,\"b\":2",
  "abab"         => "red blue red blue red blue",
  "sql_values"   => "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (1, 'a')",
}
repeats.each { |name, text| prompts << PromptCase.new("repeat_#{name}", "repeat", text) }

structured = {
  "json_shards" => "Return only JSON for three database shards with host, port, and role fields.",
  "yaml_plan"   => "Write a YAML deployment plan with services, ports, and health checks.",
  "markdown"    => "Create a markdown checklist for debugging a slow Metal kernel.",
  "sql_query"   => "Write a SQL query that groups orders by customer and month.",
  "logs"        => "Summarize these logs and identify the most likely failure mode:",
  "csv"         => "Convert the following records into CSV with columns name, role, and score:",
  "toml"        => "Generate a TOML config for a local inference server.",
  "dockerfile"  => "Write a minimal Dockerfile for a Crystal command-line application.",
}
structured.each { |name, text| prompts << PromptCase.new("structured_#{name}", "structured", text) }

chat = {
  "story"     => "Once upon a time",
  "fox"       => "The quick brown fox",
  "assistant" => "User: Can you explain why the benchmark changed?\nAssistant:",
  "recipe"    => "A simple recipe for tomato soup starts with",
  "travel"    => "For a two day trip to Kyoto, I would",
  "email"     => "Draft a concise email asking for a benchmark rerun because",
  "review"    => "Review this code for correctness and performance risks:",
  "compare"   => "Compare linear attention and flash attention in practical inference:",
}
chat.each { |name, text| prompts << PromptCase.new("chat_#{name}", "chat", text) }

math = {
  "sequence"   => "The sequence 2, 4, 8, 16 continues with",
  "derivative" => "The derivative of x^2 + 3x is",
  "matrix"     => "Given a 2 by 2 matrix [[1, 2], [3, 4]], the determinant is",
  "prob"       => "If a fair coin is tossed three times, the probability of exactly two heads is",
  "units"      => "Convert 42 kilometers per hour to meters per second:",
  "prime"      => "The first ten prime numbers are",
}
math.each { |name, text| prompts << PromptCase.new("math_#{name}", "math", text) }

templates = [
  {"explain", "Explain %{topic} in one paragraph for an engineer."},
  {"steps", "List the first three steps to debug %{topic}."},
  {"tradeoffs", "Describe the trade-offs of %{topic} for local inference."},
  {"risk", "What can go wrong when optimizing %{topic}?"},
  {"example", "Give a compact example of %{topic}."},
  {"benchmark", "Design a small benchmark for %{topic}."},
  {"counterexample", "Give a counterexample where %{topic} is the wrong approach."},
  {"checklist", "Create a short checklist for validating %{topic}."},
  {"pseudocode", "Write pseudocode for a minimal %{topic} experiment."},
  {"failure", "Diagnose a failure caused by %{topic} in a local ML runtime."},
  {"compare_llama", "Compare %{topic} with the corresponding llama.cpp path."},
  {"memory", "Estimate the memory traffic of %{topic}."},
  {"cache", "How would caching change the cost of %{topic}?"},
  {"macos", "What is special about %{topic} on Apple Silicon?"},
  {"rollback", "What rollback plan is needed before changing %{topic}?"},
  {"metric", "Which metric best detects a regression in %{topic}?"},
  {"short_answer", "Answer briefly: when does %{topic} help?"},
]
topics = [
  {"metal_gemv", "Metal GEMV kernels"},
  {"kv_cache", "KV cache reuse"},
  {"spec_decode", "speculative decoding"},
  {"prompt_cache", "prompt cache lookup by hash"},
  {"q4_quant", "Q4 quantization"},
  {"deltanet", "DeltaNet recurrence"},
  {"prefill", "LLM prefill"},
  {"routing", "a small acceptance router"},
  {"postgres_heap", "clustered PostgreSQL heaps"},
  {"pca_ffn", "PCA-compressed FFN sidecars"},
]
topics.each do |topic_name, topic|
  templates.each do |template_name, template|
    prompts << PromptCase.new("templ_#{template_name}_#{topic_name}", "template", template.gsub("%{topic}", topic))
  end
end

filtered = if categories_filter.empty?
             prompts
           else
             allowed = categories_filter.to_set
             prompts.select { |p| allowed.includes?(p.category) }
           end

selected = filtered.skip(offset)
selected = selected.first(limit) if limit > 0

case format
when "sweep"
  puts selected.map { |p| "#{p.name}::#{p.text.gsub('|', '/')}" }.join('|')
when "jsonl"
  selected.each do |p|
    puts({"name" => p.name, "category" => p.category, "text" => p.text}.to_json)
  end
when "plain"
  selected.each { |p| puts "#{p.name}\t#{p.category}\t#{p.text.inspect}" }
end
