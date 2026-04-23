# Greedy generation demo for Qwen 3.5 9B (CPU reference).
#
# Usage:
#   crystal run bin/qwen35_generate.cr -- "Your prompt here" [n_tokens]
#
# This is the slow reference path. Each token takes minutes on CPU.
# Phase 2 will bring Metal acceleration.

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_tokenizer"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

prompt = ARGV[0]? || "The capital of France is"
n_gen  = (ARGV[1]? || "8").to_i

puts "Loading model and weights..."
t0 = Time.instant
g = ML::GGUF::GGUFFile.new(MODEL_PATH)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, MODEL_PATH, LLAMA_TOKENIZE_BIN)
g.close
w = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
hp = w.hparams
puts "Loaded in #{(Time.instant - t0).total_seconds.round(1)}s. n_layer=#{hp.n_layer} n_embd=#{hp.n_embd} n_ff=#{hp.n_ff} vocab=#{w.output.out_dim}"

# Encode prompt
ids = tok.encode(prompt)
puts "Prompt tokens (#{ids.size}): #{ids.inspect}"
puts "Prompt decoded: #{tok.decode(ids).inspect}"

max_seq = ids.size + n_gen + 8
state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)

output_ids = [] of Int32
pos = 0

# Prefill: run forward on each prompt token sequentially
puts "\nPrefilling #{ids.size} tokens..."
ids.each_with_index do |tid, i|
  tstart = Time.instant
  logits = ML::GGUF::Qwen35CPU.forward(w, tid, pos, state)
  dt = (Time.instant - tstart).total_seconds
  STDOUT << "  token #{i+1}/#{ids.size} id=#{tid} took #{dt.round(2)}s\n"
  STDOUT.flush
  if i == ids.size - 1
    # Sample next token from last prefill logit
    top = logits.index(logits.max).not_nil!
    output_ids << top.to_i32
  end
  pos += 1
end

# Decode loop
puts "\nGenerating #{n_gen} tokens greedily..."
(n_gen - 1).times do |g_i|
  prev = output_ids.last
  tstart = Time.instant
  logits = ML::GGUF::Qwen35CPU.forward(w, prev, pos, state)
  dt = (Time.instant - tstart).total_seconds
  top = logits.index(logits.max).not_nil!.to_i32
  piece = tok.decode_single(top)
  STDOUT << "  gen #{g_i+1}/#{n_gen} pos=#{pos} id=#{top} piece=#{piece.inspect} took #{dt.round(2)}s\n"
  STDOUT.flush
  output_ids << top
  pos += 1
  break if top == tok.eos_id
end

puts "\n=== Generated token ids ==="
puts output_ids.inspect
puts "\n=== Generated text ==="
puts tok.decode(output_ids)
puts "\n=== Full output ==="
puts prompt + tok.decode(output_ids)
