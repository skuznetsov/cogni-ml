# Greedy generation demo for Qwen 3.5 9B.
#
# Usage:
#   crystal run bin/qwen35_generate.cr -- "Your prompt here" [n_tokens]
#
# Uses the native Metal wave path when available; set
# `QWEN35_DECODE_WAVE_OFF=1` for the slow CPU reference path.

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/ngram_draft"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_tokenizer"

MODEL_PATH         = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

prompt = ARGV[0]? || "The capital of France is"
n_gen = (ARGV[1]? || "8").to_i
prompt_cache_enabled = ENV["QWEN35_PROMPT_CACHE"]? == "1"
ngram_decode_enabled = ENV["QWEN35_NGRAM_DECODE"]? == "1"
ngram_gamma = (ENV["QWEN35_NGRAM_GAMMA"]? || "32").to_i
ngram_min = (ENV["QWEN35_NGRAM_MIN"]? || "6").to_i
ngram_max = (ENV["QWEN35_NGRAM_MAX"]? || "8").to_i
ngram_recursive = ENV["QWEN35_NGRAM_RECURSIVE_OFF"]? != "1"
ngram_disable_after_reject = ENV["QWEN35_NGRAM_DISABLE_AFTER_REJECT_OFF"]? != "1"

raise "QWEN35_NGRAM_GAMMA must be positive" unless ngram_gamma > 0
raise "QWEN35_NGRAM_MIN must be positive" unless ngram_min > 0
raise "QWEN35_NGRAM_MAX must be >= QWEN35_NGRAM_MIN" unless ngram_max >= ngram_min

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

# Prefill known non-final prompt tokens through the shared helper, then run the
# final token for next-token logits. The recurrent chunk path is default-on;
# set `QWEN35_PREFILL_CHUNK_OFF=1` to force the older whole-token prefill loop.
if output_ids.empty?
  puts "\nPrefilling #{ids.size} tokens..."
  if !prompt_cache_enabled && ids.size > 1
    tstart = Time.instant
    top, top_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, ids, pos, state)
    output_ids << top.to_i32
    dt = (Time.instant - tstart).total_seconds
    STDOUT << "  chunked #{ids.size}/#{ids.size} tokens with final top1 took #{dt.round(2)}s\n"
    STDOUT.flush
    pos += ids.size
  elsif ids.size > 1
    prefix_ids = ids[0...-1]
    tstart = Time.instant
    ML::GGUF::Qwen35CPU.prefill_tokens(w, prefix_ids, pos, state)
    dt = (Time.instant - tstart).total_seconds
    pos += prefix_ids.size
    STDOUT << "  prefix #{prefix_ids.size}/#{ids.size} tokens took #{dt.round(2)}s\n"
    STDOUT.flush

    if prompt_cache_enabled
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
  end

  if output_ids.empty? && (final_id = ids.last?)
    tstart = Time.instant
    top, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, final_id, pos, state)
    output_ids << top.to_i32
    dt = (Time.instant - tstart).total_seconds
    STDOUT << "  final token #{ids.size}/#{ids.size} id=#{final_id} took #{dt.round(2)}s\n"
    STDOUT.flush
    pos += 1
  end
end

# Decode loop
if ngram_decode_enabled && !output_ids.empty?
  puts "\nGenerating #{n_gen} tokens with exact n-gram speculative decode..."
  puts "  ngram gamma=#{ngram_gamma} min=#{ngram_min} max=#{ngram_max} recursive=#{ngram_recursive} disable_after_reject=#{ngram_disable_after_reject}"
  next_id = output_ids.pop
  history = ids.dup
  ngram_disabled = false
  ngram_cycles = 0
  ngram_accepted = 0
  ngram_proposed = 0
  plain_steps = 0
  backup = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)

  while output_ids.size < n_gen
    remaining = n_gen - output_ids.size
    candidates = ngram_disabled ? [] of Int32 : ML::GGUF::NgramDraft.candidates(
      history, Math.min(ngram_gamma, remaining), ngram_max, ngram_min, recursive: ngram_recursive)

    if candidates.empty?
      tstart = Time.instant
      emitted = next_id
      top, _top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, emitted, pos, state)
      dt = (Time.instant - tstart).total_seconds
      output_ids << emitted
      history << emitted
      next_id = top
      piece = tok.decode_single(emitted)
      STDOUT << "  gen #{output_ids.size}/#{n_gen} pos=#{pos} id=#{emitted} piece=#{piece.inspect} mode=plain took #{dt.round(2)}s\n"
      STDOUT.flush
      pos += 1
      plain_steps += 1
      break if emitted == tok.eos_id
      next
    end

    ngram_cycles += 1
    ngram_proposed += candidates.size
    backup.copy_from!(state)
    tstart = Time.instant
    target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, pos, state)
    dt = (Time.instant - tstart).total_seconds

    expected = next_id
    accepted_or_corrected = [] of Int32
    rejected = false
    candidates.each_with_index do |cand, i|
      break if output_ids.size >= n_gen
      if cand == expected
        output_ids << cand
        history << cand
        accepted_or_corrected << cand
        ngram_accepted += 1
        expected = target_nexts[i][0]
        break if cand == tok.eos_id
      else
        output_ids << expected
        history << expected
        accepted_or_corrected << expected
        rejected = true
        break
      end
    end

    if rejected
      ngram_disabled = true if ngram_disable_after_reject
      state.copy_from!(backup)
      corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, accepted_or_corrected, pos, state)
      next_id = corrected[-1][0]
      pos += accepted_or_corrected.size
    else
      next_id = target_nexts[accepted_or_corrected.size - 1][0]
      pos += accepted_or_corrected.size
    end

    STDOUT << "  ngram cycle=#{ngram_cycles} accepted=#{ngram_accepted}/#{ngram_proposed} emitted=#{accepted_or_corrected.size} pos=#{pos} rejected=#{rejected} took=#{dt.round(2)}s\n"
    STDOUT.flush
    break if output_ids.last? == tok.eos_id
  end

  rate = ngram_proposed > 0 ? (ngram_accepted.to_f64 * 100.0 / ngram_proposed.to_f64) : 0.0
  STDOUT << "  ngram summary: accepted=#{ngram_accepted}/#{ngram_proposed} rate=#{rate.round(2)}% cycles=#{ngram_cycles} plain_steps=#{plain_steps} disabled=#{ngram_disabled}\n"
else
  puts "\nGenerating #{n_gen} tokens greedily..."
  (n_gen - 1).times do |g_i|
    prev = output_ids.last
    tstart = Time.instant
    top, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prev, pos, state)
    dt = (Time.instant - tstart).total_seconds
    piece = tok.decode_single(top)
    STDOUT << "  gen #{g_i + 1}/#{n_gen} pos=#{pos} id=#{top} piece=#{piece.inspect} took #{dt.round(2)}s\n"
    STDOUT.flush
    output_ids << top
    pos += 1
    break if top == tok.eos_id
  end
end

puts "\n=== Generated token ids ==="
puts output_ids.inspect
puts "\n=== Generated text ==="
puts tok.decode(output_ids)
puts "\n=== Full output ==="
puts prompt + tok.decode(output_ids)
