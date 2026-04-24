# Greedy generation demo for Qwen 3.5 9B.
#
# Usage:
#   crystal run bin/qwen35_generate.cr -- "Your prompt here" [n_tokens]
#
# Uses the native Metal wave path when available; set
# `QWEN35_DECODE_WAVE_OFF=1` for the slow CPU reference path.

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_tokenizer"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

prompt = ARGV[0]? || "The capital of France is"
n_gen  = (ARGV[1]? || "8").to_i
prompt_cache_enabled = ENV["QWEN35_PROMPT_CACHE"]? == "1"

def cache_model_id(path : String) : String
  info = File.info(path)
  ML::GGUF::Qwen35PromptCache.short_hash("model\0#{path}\0#{info.size}\0#{info.modification_time.to_unix}")
end

def cache_tokenizer_id(model_id : String, tok : ML::GGUF::Qwen35Tokenizer) : String
  ML::GGUF::Qwen35PromptCache.short_hash("tokenizer\0#{model_id}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}")
end

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
cache_store = nil.as(ML::GGUF::Qwen35PromptCache::Store?)
cache_model = ""
cache_tokenizer = ""

output_ids = [] of Int32
pos = 0

if prompt_cache_enabled
  cache_root = ENV["QWEN35_PROMPT_CACHE_ROOT"]? || ML::GGUF::Qwen35PromptCache.default_root
  cache_store = ML::GGUF::Qwen35PromptCache::Store.new(cache_root)
  cache_model = cache_model_id(MODEL_PATH)
  cache_tokenizer = cache_tokenizer_id(cache_model, tok)
  max_prefix_len = ids.size > 0 ? ids.size - 1 : 0

  if max_prefix_len > 0 && (hit = cache_store.not_nil!.lookup_longest_prefix(cache_model, cache_tokenizer, ids, max_prefix_len: max_prefix_len))
    tstart = Time.instant
    replay = cache_store.not_nil!.restore_and_replay_suffix(hit, w, ids)
    dt = (Time.instant - tstart).total_seconds
    state = replay.state
    pos = ids.size
    if top = replay.next_token_id
      output_ids << top
      STDOUT << "\nPrompt cache hit: reused #{replay.reused_prefix_len}/#{ids.size} prompt tokens, replayed #{replay.replayed_tokens}, restore+replay took #{dt.round(3)}s\n"
    else
      STDOUT << "\nPrompt cache hit had no suffix logits; falling back to normal prefill\n"
      pos = 0
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
    end
  else
    STDOUT << "\nPrompt cache miss (root=#{cache_root})\n"
  end
end

# Prefill: run forward on each prompt token sequentially
if output_ids.empty?
  puts "\nPrefilling #{ids.size} tokens..."
  ids.each_with_index do |tid, i|
    if prompt_cache_enabled && i == ids.size - 1 && i > 0
      prefix_ids = ids.first(i)
      preview = ENV["QWEN35_PROMPT_CACHE_PREVIEW"]? == "1" ? tok.decode(prefix_ids) : nil
      saved = cache_store.not_nil!.save(
        session_id: ENV["QWEN35_SESSION_ID"]? || "default",
        turn_id: ENV["QWEN35_TURN_ID"]?,
        model_id: cache_model,
        tokenizer_id: cache_tokenizer,
        prompt_text: "",
        token_ids: prefix_ids,
        state: state,
        prompt_preview: preview,
      )
      STDOUT << "  saved prompt-cache prefix #{prefix_ids.size} tokens sha=#{saved.artifact_sha256[0, 12]}\n"
    end

    tstart = Time.instant
    if i == ids.size - 1
      top, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, tid, pos, state)
      output_ids << top.to_i32
    else
      ML::GGUF::Qwen35CPU.prefill_token(w, tid, pos, state)
    end
    dt = (Time.instant - tstart).total_seconds
    STDOUT << "  token #{i+1}/#{ids.size} id=#{tid} took #{dt.round(2)}s\n"
    STDOUT.flush
    pos += 1
  end
end

# Decode loop
puts "\nGenerating #{n_gen} tokens greedily..."
(n_gen - 1).times do |g_i|
  prev = output_ids.last
  tstart = Time.instant
  top, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prev, pos, state)
  dt = (Time.instant - tstart).total_seconds
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
