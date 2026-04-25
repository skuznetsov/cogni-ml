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
DRAFT_MODEL_PATH   = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

prompt = ARGV[0]? || "The capital of France is"
n_gen = (ARGV[1]? || "8").to_i
prompt_cache_enabled = ENV["QWEN35_PROMPT_CACHE"]? == "1"
decode_policy = (ENV["QWEN35_DECODE_POLICY"]? || "").downcase
unless decode_policy.empty? || decode_policy == "greedy" || decode_policy == "ngram" || decode_policy == "speculative" || decode_policy == "auto"
  raise "QWEN35_DECODE_POLICY must be greedy, ngram, speculative, or auto"
end
legacy_speculative_decode_enabled = ENV["QWEN35_SPECULATIVE_DECODE"]? == "1" || ENV.has_key?("QWEN35_DRAFT_MODEL")
legacy_ngram_decode_enabled = ENV["QWEN35_NGRAM_DECODE"]? == "1"
if decode_policy.empty? && legacy_speculative_decode_enabled && legacy_ngram_decode_enabled
  raise "QWEN35_SPECULATIVE_DECODE and QWEN35_NGRAM_DECODE are mutually exclusive; set QWEN35_DECODE_POLICY=ngram or speculative"
end
speculative_decode_enabled = false
ngram_decode_enabled = false
case decode_policy
when "greedy"
  # Explicit policy overrides legacy env toggles.
when "ngram", "auto"
  ngram_decode_enabled = true
when "speculative"
  speculative_decode_enabled = true
else
  speculative_decode_enabled = legacy_speculative_decode_enabled
  ngram_decode_enabled = legacy_ngram_decode_enabled
end
draft_model_path = ENV["QWEN35_DRAFT_MODEL"]? || DRAFT_MODEL_PATH
spec_gamma = (ENV["QWEN35_SPEC_GAMMA"]? || "4").to_i
spec_max_gamma = (ENV["QWEN35_SPEC_MAX_GAMMA"]? || "32").to_i
spec_plain_fallback_gamma = (ENV["QWEN35_SPEC_PLAIN_FALLBACK_GAMMA"]? || "2").to_i
spec_full_accept_streak = (ENV["QWEN35_SPEC_FULL_ACCEPT_STREAK"]? || "2").to_i
spec_fast_regrow_min_gamma = (ENV["QWEN35_SPEC_FAST_REGROW_MIN_GAMMA"]? || "8").to_i
spec_bootstrap_gamma = (ENV["QWEN35_SPEC_BOOTSTRAP_GAMMA"]? || "0").to_i
ngram_gamma = (ENV["QWEN35_NGRAM_GAMMA"]? || "32").to_i
ngram_min = (ENV["QWEN35_NGRAM_MIN"]? || "6").to_i
ngram_max = (ENV["QWEN35_NGRAM_MAX"]? || "8").to_i
ngram_recursive = ENV["QWEN35_NGRAM_RECURSIVE_OFF"]? != "1"
ngram_disable_after_reject = ENV["QWEN35_NGRAM_DISABLE_AFTER_REJECT_OFF"]? != "1"

raise "QWEN35_NGRAM_GAMMA must be positive" unless ngram_gamma > 0
raise "QWEN35_NGRAM_MIN must be positive" unless ngram_min > 0
raise "QWEN35_NGRAM_MAX must be >= QWEN35_NGRAM_MIN" unless ngram_max >= ngram_min
raise "QWEN35_SPEC_GAMMA must be positive" unless spec_gamma > 0
raise "QWEN35_SPEC_MAX_GAMMA must be positive" unless spec_max_gamma > 0
raise "QWEN35_SPEC_PLAIN_FALLBACK_GAMMA must be positive" unless spec_plain_fallback_gamma > 0
raise "QWEN35_SPEC_FULL_ACCEPT_STREAK must be positive" unless spec_full_accept_streak > 0
raise "QWEN35_SPEC_FAST_REGROW_MIN_GAMMA must be non-negative" unless spec_fast_regrow_min_gamma >= 0
raise "QWEN35_SPEC_BOOTSTRAP_GAMMA must be non-negative" unless spec_bootstrap_gamma >= 0
spec_max_gamma = Math.max(spec_max_gamma, spec_gamma)

def cache_model_id(path : String) : String
  info = File.info(path)
  ML::GGUF::Qwen35PromptCache.short_hash("model\0#{path}\0#{info.size}\0#{info.modification_time.to_unix}")
end

def cache_tokenizer_id(model_id : String, tok : ML::GGUF::Qwen35Tokenizer) : String
  ML::GGUF::Qwen35PromptCache.short_hash("tokenizer\0#{model_id}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}")
end

def prefill_next(weights : ML::GGUF::Qwen35Weights,
                 token_ids : Array(Int32),
                 state : ML::GGUF::Qwen35CPU::State) : Int32
  top, _logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, token_ids, 0, state)
  top.to_i32
end

def advance_next(weights : ML::GGUF::Qwen35Weights,
                 token_id : Int32,
                 pos : Int32,
                 state : ML::GGUF::Qwen35CPU::State) : Int32
  top, _logit = ML::GGUF::Qwen35CPU.forward_top1(weights, token_id, pos, state)
  top.to_i32
end

def resync_draft!(weights : ML::GGUF::Qwen35Weights,
                  state : ML::GGUF::Qwen35CPU::State,
                  base : ML::GGUF::Qwen35CPU::State,
                  accepted_or_corrected : Array(Int32),
                  start_pos : Int32) : Int32
  state.copy_from!(base)
  next_id = -1
  accepted_or_corrected.each_with_index do |tok, i|
    next_id = advance_next(weights, tok, start_pos + i, state)
  end
  next_id
end

def with_guarded_full_rows_disabled
  old_guard = ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"]?
  ENV.delete("QWEN35_HEAD_FULL_ROWS_GUARDED")
  yield
ensure
  if old_guard
    ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"] = old_guard
  else
    ENV.delete("QWEN35_HEAD_FULL_ROWS_GUARDED")
  end
end

puts "Loading model and weights..."
t0 = Time.instant
g = ML::GGUF::GGUFFile.new(MODEL_PATH)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, MODEL_PATH, LLAMA_TOKENIZE_BIN)
g.close
w = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
hp = w.hparams
puts "Loaded in #{(Time.instant - t0).total_seconds.round(1)}s. n_layer=#{hp.n_layer} n_embd=#{hp.n_embd} n_ff=#{hp.n_ff} vocab=#{w.output.out_dim}"

draft = nil.as(ML::GGUF::Qwen35Weights?)
if speculative_decode_enabled
  raise "draft model not found: #{draft_model_path}" unless File.exists?(draft_model_path)
  tstart = Time.instant
  draft = ML::GGUF::Qwen35Weights.from_gguf(draft_model_path)
  raise "target/draft vocab mismatch: #{w.output.out_dim} != #{draft.not_nil!.output.out_dim}" unless w.output.out_dim == draft.not_nil!.output.out_dim
  puts "Loaded draft in #{(Time.instant - tstart).total_seconds.round(1)}s. n_layer=#{draft.not_nil!.hparams.n_layer} n_embd=#{draft.not_nil!.hparams.n_embd}"
end

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
if speculative_decode_enabled && !output_ids.empty?
  puts "\nGenerating #{n_gen} tokens with exact neural speculative decode..."
  puts "  draft=#{draft_model_path}"
  puts "  gamma=#{spec_gamma} max_gamma=#{spec_max_gamma} bootstrap_gamma=#{spec_bootstrap_gamma} fallback_gamma=#{spec_plain_fallback_gamma} full_accept_streak=#{spec_full_accept_streak} fast_regrow_min_gamma=#{spec_fast_regrow_min_gamma}"

  decode_t0 = Time.instant
  target_next = output_ids.pop
  draft_weights = draft.not_nil!
  draft_state = ML::GGUF::Qwen35CPU::State.new(draft_weights.hparams, max_seq: max_seq)
  draft_next = prefill_next(draft_weights, ids, draft_state)
  target_backup_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  draft_cycle_base = ML::GGUF::Qwen35CPU::State.new(draft_weights.hparams, max_seq: max_seq)

  current_gamma = spec_gamma
  full_accept_streak = 0
  adaptive_growth_allowed = true
  accepted = 0
  proposed = 0
  cycles = 0
  plain_fallback_steps = 0
  early_rejects = 0
  target_verify_ms = 0.0
  draft_ms = 0.0

  while output_ids.size < n_gen
    if !adaptive_growth_allowed && current_gamma <= spec_plain_fallback_gamma
      emitted = target_next
      tstart = Time.instant
      target_next = advance_next(w, emitted, pos, state)
      target_verify_ms += (Time.instant - tstart).total_milliseconds
      output_ids << emitted
      piece = tok.decode_single(emitted)
      STDOUT << "  gen #{output_ids.size}/#{n_gen} pos=#{pos} id=#{emitted} piece=#{piece.inspect} mode=target-fallback\n"
      STDOUT.flush
      pos += 1
      plain_fallback_steps += 1
      break if emitted == tok.eos_id
      next
    end

    cycles += 1
    cycle_start_pos = pos
    cycle_gamma = Math.min(current_gamma, n_gen - output_ids.size)
    correction_or_accepted = [] of Int32
    candidates = [] of Int32
    rejected = false

    if draft_next != target_next
      emitted = target_next
      tstart = Time.instant
      target_next = advance_next(w, emitted, pos, state)
      target_verify_ms += (Time.instant - tstart).total_milliseconds
      output_ids << emitted
      correction_or_accepted << emitted
      proposed += 1
      pos += 1
      rejected = true
      early_rejects += 1
      STDOUT << "  spec cycle=#{cycles} early_reject emitted=1 gamma=#{current_gamma}\n"
      STDOUT.flush
    else
      tstart = Time.instant
      draft_cycle_base.copy_from!(draft_state)
      cycle_gamma.times do |i|
        candidates << draft_next
        draft_next = advance_next(draft_weights, draft_next, pos + i, draft_state)
      end
      draft_ms += (Time.instant - tstart).total_milliseconds
      proposed += candidates.size

      target_backup_state.copy_from!(state)
      tstart = Time.instant
      target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, cycle_start_pos, state)
      target_verify_ms += (Time.instant - tstart).total_milliseconds

      expected = target_next
      candidates.each_with_index do |cand, i|
        if cand == expected
          output_ids << cand
          correction_or_accepted << cand
          accepted += 1
          expected = target_nexts[i][0]
          break if cand == tok.eos_id
        else
          output_ids << expected
          correction_or_accepted << expected
          rejected = true
          break
        end
      end

      if rejected
        state.copy_from!(target_backup_state)
        tstart = Time.instant
        corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, correction_or_accepted, cycle_start_pos, state)
        target_verify_ms += (Time.instant - tstart).total_milliseconds
        target_next = corrected[-1][0]
        draft_next = resync_draft!(draft_weights, draft_state, draft_cycle_base, correction_or_accepted, cycle_start_pos)
      else
        target_next = target_nexts[correction_or_accepted.size - 1][0]
      end
      pos += correction_or_accepted.size

      STDOUT << "  spec cycle=#{cycles} accepted=#{accepted}/#{proposed} emitted=#{correction_or_accepted.size} gamma=#{current_gamma} rejected=#{rejected}\n"
      STDOUT.flush
    end

    if rejected
      full_accept_streak = 0
      adaptive_growth_allowed = false
      current_gamma = Math.max(1, current_gamma // 2)
    elsif adaptive_growth_allowed && candidates.size == cycle_gamma && current_gamma < spec_max_gamma
      full_accept_streak += 1
      if spec_bootstrap_gamma > current_gamma && current_gamma == spec_gamma
        current_gamma = Math.min(spec_max_gamma, spec_bootstrap_gamma)
        full_accept_streak = 0
      else
        required = if spec_fast_regrow_min_gamma > 0 && current_gamma >= spec_fast_regrow_min_gamma
                     1
                   else
                     spec_full_accept_streak
                   end
        if full_accept_streak >= required
          current_gamma = Math.min(spec_max_gamma, current_gamma * 2)
          full_accept_streak = 0
        end
      end
    end

    break if output_ids.last? == tok.eos_id
  end

  decode_ms = (Time.instant - decode_t0).total_milliseconds
  rate = proposed > 0 ? (accepted.to_f64 * 100.0 / proposed.to_f64) : 0.0
  STDOUT << "  speculative summary: accepted=#{accepted}/#{proposed} rate=#{rate.round(2)}% cycles=#{cycles} fallback_steps=#{plain_fallback_steps} early_rejects=#{early_rejects} wall_ms=#{decode_ms.round(1)} ms_per_tok=#{(decode_ms / output_ids.size).round(2)} draft_ms=#{draft_ms.round(1)} target_ms=#{target_verify_ms.round(1)}\n"
elsif ngram_decode_enabled && !output_ids.empty?
  puts "\nGenerating #{n_gen} tokens with exact n-gram speculative decode..."
  puts "  ngram gamma=#{ngram_gamma} min=#{ngram_min} max=#{ngram_max} recursive=#{ngram_recursive} disable_after_reject=#{ngram_disable_after_reject}"
  decode_t0 = Time.instant
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
    target_nexts = with_guarded_full_rows_disabled do
      ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, pos, state)
    end
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
      corrected = with_guarded_full_rows_disabled do
        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, accepted_or_corrected, pos, state)
      end
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

  decode_ms = (Time.instant - decode_t0).total_milliseconds
  rate = ngram_proposed > 0 ? (ngram_accepted.to_f64 * 100.0 / ngram_proposed.to_f64) : 0.0
  STDOUT << "  ngram summary: accepted=#{ngram_accepted}/#{ngram_proposed} rate=#{rate.round(2)}% cycles=#{ngram_cycles} plain_steps=#{plain_steps} disabled=#{ngram_disabled} wall_ms=#{decode_ms.round(1)} ms_per_tok=#{(decode_ms / output_ids.size).round(2)}\n"
else
  puts "\nGenerating #{n_gen} tokens greedily..."
  decode_t0 = Time.instant
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
  decode_ms = (Time.instant - decode_t0).total_milliseconds
  STDOUT << "  greedy summary: wall_ms=#{decode_ms.round(1)} ms_per_tok=#{(decode_ms / output_ids.size).round(2)}\n"
end

puts "\n=== Generated token ids ==="
puts output_ids.inspect
puts "\n=== Generated text ==="
puts tok.decode(output_ids)
puts "\n=== Full output ==="
puts prompt + tok.decode(output_ids)
