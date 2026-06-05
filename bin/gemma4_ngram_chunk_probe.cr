require "option_parser"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/gemma4_state_snapshot"
require "../src/ml/gguf/gemma4_tokenizer"
require "../src/ml/gguf/ngram_draft"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER_BIN
prompt = "alpha beta gamma alpha beta gamma alpha beta"
tokens_arg = nil.as(String?)
gen = 32
max_seq = 256
prefill_chunk = 128
gamma = 8
min_ngram = 2
max_ngram = 8
min_candidates = 4
recursive = true
risk_gate = true
batch_verify = false
oracle_exact_proposals = false
trust_batch_accepts = false
adaptive_prefixes = [] of Int32

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_ngram_chunk_probe [options]"
  p.on("--model PATH", "Gemma4 GGUF model path") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Raw prompt text") { |v| prompt = v }
  p.on("--tokens IDS", "Comma-separated token ids; bypasses tokenizer") { |v| tokens_arg = v }
  p.on("--gen N", "Generated token count, default 32") { |v| gen = v.to_i }
  p.on("--gamma N", "Max proposal span, default 8") { |v| gamma = v.to_i }
  p.on("--min-ngram N", "Minimum n-gram, default 2") { |v| min_ngram = v.to_i }
  p.on("--max-ngram N", "Maximum n-gram, default 8") { |v| max_ngram = v.to_i }
  p.on("--min-candidates N", "Minimum proposed span, default 4") { |v| min_candidates = v.to_i }
  p.on("--no-recursive", "Disable recursive n-gram expansion") { recursive = false }
  p.on("--no-risk-gate", "Disable NgramDraft.risky_candidate_shape? gate") { risk_gate = false }
  p.on("--batch-verify", "Verify proposed spans with row-prefill + row-top1 instead of serial top1") { batch_verify = true }
  p.on("--oracle-exact-proposals", "Diagnostic: propose exact greedy spans to isolate verifier economics") { oracle_exact_proposals = true }
  p.on("--trust-batch-accepts", "Diagnostic: do not serial-confirm batch accepts for real proposals") { trust_batch_accepts = true }
  p.on("--adaptive-prefix-verify LIST", "Diagnostic: batch-verify prefixes, e.g. 4,8,16") do |v|
    adaptive_prefixes = v.split(',', remove_empty: true).map { |part| part.strip.to_i32 }.select { |n| n > 0 }
    batch_verify = true
  end
  p.on("--max-seq N", "Resident state sequence capacity, default 256") { |v| max_seq = v.to_i }
  p.on("--prefill-chunk N", "Prefill chunk size, default 128") { |v| prefill_chunk = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "--gen must be positive" unless gen > 0
raise "--gamma must be positive" unless gamma > 0
raise "--min-candidates must be non-negative" unless min_candidates >= 0
raise "--min-ngram must be positive" unless min_ngram > 0
raise "--max-ngram must be >= --min-ngram" unless max_ngram >= min_ngram
raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "tokenizer binary not found: #{tokenizer_bin}" unless tokens_arg || File.exists?(tokenizer_bin)

weights = ML::GGUF::Gemma4Weights.from_gguf(model_path)
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

ids = if raw = tokens_arg
        raw.split(',').reject(&.empty?).map(&.to_i32)
      else
        g = ML::GGUF::GGUFFile.new(model_path)
        tokenizer = ML::GGUF::Gemma4Tokenizer.from_gguf(g, model_path, tokenizer_bin)
        g.close
        tokenizer.encode(prompt)
      end
raise "prompt tokenized to zero tokens" if ids.empty?
raise "prompt+gen exceeds max_seq" if ids.size + gen + gamma + 2 > max_seq

def prefill_prefix!(weights, ids : Array(Int32), state, prefill_chunk : Int32) : Nil
  return if ids.size <= 1
  ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(
    weights, ids[0...-1], 0, state,
    chunk_size: prefill_chunk,
    stop_layer: weights.hparams.n_layer,
    read_last_hidden: false
  ).not_nil!
end

def generate_exact(weights, ids : Array(Int32), gen : Int32, max_seq : Int32, prefill_chunk : Int32) : {Array(Int32), Float64}
  state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  prefill_prefix!(weights, ids, state, prefill_chunk)
  generated = [] of Int32
  current = ids[-1]
  pos = ids.size - 1
  t0 = Time.instant
  gen.times do
    nxt = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(weights, current, pos.to_i32, state, weights.hparams.n_layer).not_nil!
    generated << nxt
    current = nxt
    pos += 1
  end
  {generated, (Time.instant - t0).total_milliseconds}
end

record NgramRun,
  ids : Array(Int32),
  ms : Float64,
  proposal_ms : Float64,
  verify_ms : Float64,
  commit_ms : Float64,
  cycles : Int32,
  proposed : Int32,
  accepted : Int32,
  full_accept_chunks : Int32,
  rejected_chunks : Int32,
  exact_fallback_tokens : Int32,
  skipped_risk : Int32,
  skipped_empty : Int32

def generate_ngram(weights, ids : Array(Int32), gen : Int32, max_seq : Int32, prefill_chunk : Int32,
                   gamma : Int32, min_ngram : Int32, max_ngram : Int32,
                   min_candidates : Int32, recursive : Bool, risk_gate : Bool,
                   batch_verify : Bool,
                   oracle_ids : Array(Int32)? = nil,
                   trust_batch_accepts : Bool = false,
                   adaptive_prefixes : Array(Int32) = [] of Int32) : NgramRun
  main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  side_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  prefill_prefix!(weights, ids, main_state, prefill_chunk)

  history = ids.dup
  index = ML::GGUF::NgramDraft::IndexedHistory.new(history, max_ngram: max_ngram, min_ngram: min_ngram)
  generated = [] of Int32
  current = ids[-1]
  pos = ids.size - 1

  proposal_ms = 0.0
  verify_ms = 0.0
  commit_ms = 0.0
  cycles = 0
  proposed = 0
  accepted = 0
  full_accept_chunks = 0
  rejected_chunks = 0
  exact_fallback_tokens = 0
  skipped_risk = 0
  skipped_empty = 0

  t0 = Time.instant
  while generated.size < gen
    cycles += 1
    p0 = Time.instant
    remaining = gen - generated.size
    oracle_candidates = oracle_ids.try { |exact| exact[generated.size, Math.min(gamma, remaining)] }
    span = oracle_candidates ? nil : index.candidate_span(gamma: Math.min(gamma, remaining), recursive: recursive, min_candidates: min_candidates)
    proposal_ms += (Time.instant - p0).total_milliseconds

    if oracle_candidates.nil? && span.nil?
      skipped_empty += 1
      nxt = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(weights, current, pos.to_i32, main_state, weights.hparams.n_layer).not_nil!
      generated << nxt
      index.append(nxt)
      history << nxt
      current = nxt
      pos += 1
      exact_fallback_tokens += 1
      next
    end

    candidates = oracle_candidates || span.not_nil!.ids
    if oracle_candidates.nil? && risk_gate && ML::GGUF::NgramDraft.risky_candidate_shape?(candidates, min_size: min_candidates, match_len: span.not_nil!.match_len)
      skipped_risk += 1
      nxt = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(weights, current, pos.to_i32, main_state, weights.hparams.n_layer).not_nil!
      generated << nxt
      index.append(nxt)
      history << nxt
      current = nxt
      pos += 1
      exact_fallback_tokens += 1
      next
    end

    proposed += candidates.size
    c0 = Time.instant
    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state)
    commit_ms += (Time.instant - c0).total_milliseconds

    v0 = Time.instant
    verify_current = current
    verify_pos = pos
    local_accept = 0
    reject_expected = nil.as(Int32?)
    if batch_verify
      prefix_sizes = if adaptive_prefixes.empty?
                       [candidates.size]
                     else
                       adaptive_prefixes.map { |n| Math.min(n, candidates.size) }
                         .push(candidates.size)
                         .uniq
                         .sort
                     end
      prefix_sizes.each do |prefix_size|
        ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state) unless prefix_size == prefix_sizes[0]
        verify_inputs = [current]
        verify_inputs.concat(candidates[0, prefix_size - 1]) if prefix_size > 1
        hidden_rows = ML::GGUF::Gemma4Metal.prefill_tokens_hidden_resident_rows(
          weights, verify_inputs, pos.to_i32, side_state,
          chunk_size: Math.max(prefill_chunk, verify_inputs.size),
          stop_layer: weights.hparams.n_layer
        ).not_nil!
        top1_rows = ML::GGUF::Qwen35Metal.rmsnorm_project_top1_rows(
          hidden_rows,
          prefix_size.to_i32,
          weights.output_norm,
          weights.token_embd,
          weights.hparams.rms_eps
        ).not_nil!
        prefix_accept = 0
        prefix_reject = nil.as(Int32?)
        candidates[0, prefix_size].each_with_index do |cand, i|
          expected = top1_rows[i][0]
          if cand == expected
            prefix_accept += 1
          else
            prefix_reject = expected
            break
          end
        end
        local_accept = prefix_accept
        reject_expected = prefix_reject
        break if reject_expected || local_accept < prefix_size || prefix_size == candidates.size
      end

      # High-batch row-prefill can be a fast filter, but it is not an exact
      # verifier for every route. Fall back to serial exact verification on
      # batch rejection, and always confirm real (non-oracle) proposals.
      if reject_expected || (oracle_ids.nil? && !trust_batch_accepts)
        ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state)
        verify_current = current
        verify_pos = pos
        local_accept = 0
        reject_expected = nil.as(Int32?)
        candidates.each do |cand|
          expected = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(weights, verify_current, verify_pos.to_i32, side_state, weights.hparams.n_layer).not_nil!
          if cand == expected
            local_accept += 1
            verify_current = cand
            verify_pos += 1
          else
            reject_expected = expected
            verify_pos += 1
            break
          end
        end
      end
    else
      candidates.each do |cand|
        expected = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(weights, verify_current, verify_pos.to_i32, side_state, weights.hparams.n_layer).not_nil!
        if cand == expected
          local_accept += 1
          verify_current = cand
          verify_pos += 1
        else
          reject_expected = expected
          verify_pos += 1
          break
        end
      end
    end
    verify_ms += (Time.instant - v0).total_milliseconds

    emitted = [] of Int32
    if local_accept == candidates.size
      emitted.concat(candidates)
      full_accept_chunks += 1
      accepted += local_accept
      commit_prefix = pos + local_accept
    else
      emitted.concat(candidates[0, local_accept]) if local_accept > 0
      emitted << reject_expected.not_nil!
      accepted += local_accept
      rejected_chunks += 1
      exact_fallback_tokens += 1
      commit_prefix = pos + local_accept + 1
    end
    emitted = emitted[0, gen - generated.size]

    c1 = Time.instant
    commit_snapshot = ML::GGUF::Gemma4StateSnapshot.capture(side_state, prefix_len: commit_prefix.to_i32)
    ML::GGUF::Gemma4StateSnapshot.restore_into(commit_snapshot, main_state)
    commit_ms += (Time.instant - c1).total_milliseconds

    generated.concat(emitted)
    index.append(emitted)
    history.concat(emitted)
    current = emitted[-1]
    pos += emitted.size
  end

  NgramRun.new(generated, (Time.instant - t0).total_milliseconds, proposal_ms, verify_ms, commit_ms,
    cycles, proposed, accepted, full_accept_chunks, rejected_chunks, exact_fallback_tokens, skipped_risk, skipped_empty)
end

puts "model=#{File.basename(model_path)} prompt_len=#{ids.size} gen=#{gen} gamma=#{gamma} min_ngram=#{min_ngram} max_ngram=#{max_ngram} min_candidates=#{min_candidates} recursive=#{recursive} risk_gate=#{risk_gate} batch_verify=#{batch_verify} oracle_exact_proposals=#{oracle_exact_proposals} trust_batch_accepts=#{trust_batch_accepts} adaptive_prefixes=#{adaptive_prefixes.join(',')} max_seq=#{max_seq}"
exact_ids, exact_ms = generate_exact(weights, ids, gen, max_seq, prefill_chunk)
oracle_ids = oracle_exact_proposals ? exact_ids : nil
ngram = generate_ngram(weights, ids, gen, max_seq, prefill_chunk, gamma, min_ngram, max_ngram, min_candidates, recursive, risk_gate, batch_verify, oracle_ids, trust_batch_accepts, adaptive_prefixes)

matches = 0
exact_ids.each_with_index { |id, i| matches += 1 if ngram.ids[i]? == id }
puts "exact_ms=#{exact_ms.round(3)} exact_ms_per_token=#{(exact_ms / gen).round(3)}"
puts "ngram_ms=#{ngram.ms.round(3)} ngram_ms_per_token=#{(ngram.ms / gen).round(3)} speedup=#{(exact_ms / ngram.ms).round(4)}"
puts "token_match_count=#{matches}/#{gen} parity=#{matches == gen}"
puts "ngram_accounting cycles=#{ngram.cycles} proposed=#{ngram.proposed} accepted=#{ngram.accepted} accept_rate=#{(ngram.proposed > 0 ? 100.0 * ngram.accepted / ngram.proposed : 0.0).round(2)} full_accept_chunks=#{ngram.full_accept_chunks} rejected_chunks=#{ngram.rejected_chunks} exact_fallback_tokens=#{ngram.exact_fallback_tokens} skipped_risk=#{ngram.skipped_risk} skipped_empty=#{ngram.skipped_empty} proposal_ms=#{ngram.proposal_ms.round(3)} verify_ms=#{ngram.verify_ms.round(3)} commit_ms=#{ngram.commit_ms.round(3)}"
puts "exact_ids=#{exact_ids.join(',')}"
puts "ngram_ids=#{ngram.ids.join(',')}"
