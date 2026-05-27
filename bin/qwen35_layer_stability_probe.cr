require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = ENV["QWEN35_MODEL_PATH"]? || DEFAULT_MODEL_PATH
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER_BIN
prompt = "The capital of France is"
gen = 4
max_seq = 1024
emit_layers = false
add_bos_override = nil.as(Bool?)

struct PromptEntryFeatures
  getter token_count : Int32
  getter unique_rate : Float64
  getter repeat_rate : Float64
  getter bigram_repeat_rate : Float64
  getter adjacent_repeat_rate : Float64

  def initialize(@token_count : Int32, @unique_rate : Float64, @repeat_rate : Float64,
                 @bigram_repeat_rate : Float64, @adjacent_repeat_rate : Float64)
  end
end

def prompt_entry_features(token_ids : Array(Int32)) : PromptEntryFeatures
  token_count = token_ids.size
  counts = Hash(Int32, Int32).new(0)
  token_ids.each { |id| counts[id] += 1 }
  unique = counts.size
  repeat_rate = token_count > 0 ? (token_count - unique).to_f64 / token_count : 0.0
  unique_rate = token_count > 0 ? unique.to_f64 / token_count : 0.0

  bigram_counts = Hash(Tuple(Int32, Int32), Int32).new(0)
  if token_count > 1
    (0...(token_count - 1)).each do |i|
      bigram_counts[{token_ids[i], token_ids[i + 1]}] += 1
    end
  end
  repeated_bigrams = 0
  bigram_counts.each_value do |count|
    repeated_bigrams += count - 1 if count > 1
  end

  adjacent_repeats = 0
  if token_count > 1
    (0...(token_count - 1)).each do |i|
      adjacent_repeats += 1 if token_ids[i] == token_ids[i + 1]
    end
  end

  bigram_total = token_count > 1 ? token_count - 1 : 0
  bigram_repeat_rate = bigram_total > 0 ? repeated_bigrams.to_f64 / bigram_total : 0.0
  adjacent_repeat_rate = bigram_total > 0 ? adjacent_repeats.to_f64 / bigram_total : 0.0
  PromptEntryFeatures.new(token_count, unique_rate, repeat_rate, bigram_repeat_rate, adjacent_repeat_rate)
end

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_layer_stability_probe [options]"
  p.on("--model PATH", "GGUF model path") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Prompt text") { |v| prompt = v }
  p.on("--gen N", "Generated tokens to trace, default 4") { |v| gen = v.to_i }
  p.on("--max-seq N", "State max sequence, default 1024") { |v| max_seq = v.to_i }
  p.on("--layer-rows", "Emit one TSV row per layer as well as summaries") { emit_layers = true }
  p.on("--add-bos", "Force tokenizer BOS") { add_bos_override = true }
  p.on("--no-bos", "Disable tokenizer BOS") { add_bos_override = false }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--gen must be positive" unless gen > 0
raise "--max-seq must be positive" unless max_seq > 0
raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "tokenizer binary not found: #{tokenizer_bin}" unless File.exists?(tokenizer_bin)

gguf = ML::GGUF::GGUFFile.new(model_path)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path, tokenizer_bin)
ids = tok.encode(prompt, add_bos_override: add_bos_override)
raise "prompt tokenized to zero tokens" if ids.empty?
raise "prompt+gen exceeds max_seq" if ids.size + gen > max_seq

weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq)
ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)

if ids.size > 1
  ML::GGUF::Qwen35CPU.prefill_tokens(weights, ids[0...-1], 0, state)
end

current_token = ids[-1]
pos = ids.size - 1
history_ids = ids.dup

puts "kind\tstep\tpos\tinput_token\tfinal_top1\tstable_from_layer\ttop1_changes\tfinal_logit\tprompt_tokens\tunique_rate\trepeat_rate\tbigram_repeat_rate\tadjacent_repeat_rate"
if emit_layers
  puts "layer\tstep\tpos\tlayer_index\tlayer_top1\tmatch_final\tlogit"
end

gen.times do |step|
  features = prompt_entry_features(history_ids)
  trace = ML::GGUF::Qwen35CPU.forward_layer_top1_trace(weights, current_token, pos, state)
  raise "empty layer trace" if trace.empty?

  final = trace[-1]
  stable_from = trace.size - 1
  trace.each_with_index do |row, i|
    if trace[i..].all? { |candidate| candidate.top1 == final.top1 }
      stable_from = i
      break
    end
  end
  changes = 0
  prev = trace[0].top1
  trace[1..].each do |row|
    if row.top1 != prev
      changes += 1
      prev = row.top1
    end
  end

  puts ["summary", step, pos, current_token, final.top1, stable_from, changes, final.logit,
        features.token_count, features.unique_rate, features.repeat_rate,
        features.bigram_repeat_rate, features.adjacent_repeat_rate].join('\t')
  if emit_layers
    trace.each do |row|
      puts ["layer", step, pos, row.layer, row.top1, row.top1 == final.top1, row.logit].join('\t')
    end
  end

  current_token = final.top1
  history_ids << current_token
  pos += 1
end
