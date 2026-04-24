# Greedy speculative-decode acceptance probe for Qwen35 target/draft pairs.
#
# The chunk verifiers process each gamma-sized target candidate span through the
# chunked prefill body and then emit one top1 per row. This is still not the
# final fully batched verifier, but it measures exact speed steps after a purely
# serial target verifier.

require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "option_parser"

DEFAULT_TARGET    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_DRAFT     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

target_path = ENV["QWEN35_TARGET"]? || DEFAULT_TARGET
draft_path = ENV["QWEN35_DRAFT"]? || DEFAULT_DRAFT
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = "The capital of France is"
n_gen = 32
gamma = 4
adaptive_gamma = ENV["QWEN35_SPEC_ADAPTIVE"]? != "0"
adaptive_regrow = ENV["QWEN35_SPEC_ADAPTIVE_REGROW"]? == "1"
adaptive_full_accept_streak = (ENV["QWEN35_SPEC_FULL_ACCEPT_STREAK"]? || "2").to_i
adaptive_fast_regrow_min_gamma = (ENV["QWEN35_SPEC_FAST_REGROW_MIN_GAMMA"]? || "8").to_i
adaptive_bootstrap_gamma = (ENV["QWEN35_SPEC_BOOTSTRAP_GAMMA"]? || "0").to_i
max_gamma = (ENV["QWEN35_SPEC_MAX_GAMMA"]? || "32").to_i
verify_mode = ENV["QWEN35_SPEC_VERIFY"]? || "chunk-inplace"
stage_gate = (ENV["QWEN35_SPEC_STAGE_GATE"]? || gamma.to_s).to_i
trace = ENV["QWEN35_SPEC_TRACE"]? == "1"
early_reject_enabled = ENV["QWEN35_SPEC_EARLY_REJECT_OFF"]? != "1"
single_accept_fast_enabled = ENV["QWEN35_SPEC_SINGLE_FAST_OFF"]? != "1"
plain_fallback_enabled = ENV["QWEN35_SPEC_PLAIN_FALLBACK_OFF"]? != "1"
plain_fallback_gamma = (ENV["QWEN35_SPEC_PLAIN_FALLBACK_GAMMA"]? || "2").to_i
skip_draft_before_fallback_enabled = ENV["QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF"]? != "1"
skip_draft_backup_before_fallback_enabled = ENV["QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF"]? != "1"

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_speculative_accept [--target PATH] [--draft PATH] [--gamma N] [--max-gamma N] [--bootstrap-gamma N] [--adaptive|--no-adaptive] [--tokens N] [--verify serial|chunk|chunk-inplace|hybrid|staged] [prompt]"
  parser.on("--target PATH", "Target GGUF path (default: Qwen3.5 9B Q4_K_M)") { |path| target_path = path }
  parser.on("--draft PATH", "Draft GGUF path (default: Qwen3.5 0.8B Q8_0)") { |path| draft_path = path }
  parser.on("--tokenizer-bin PATH", "llama.cpp tokenizer helper path") { |path| tokenizer_bin = path }
  parser.on("--gamma N", "Draft candidates per cycle") { |value| gamma = value.to_i }
  parser.on("--max-gamma N", "Maximum adaptive draft candidates per cycle (default: 32)") { |value| max_gamma = value.to_i }
  parser.on("--bootstrap-gamma N", "After one fully accepted initial chunk, jump to this gamma (default: env QWEN35_SPEC_BOOTSTRAP_GAMMA or 0/off)") { |value| adaptive_bootstrap_gamma = value.to_i }
  parser.on("--adaptive", "Adapt gamma: double after fully accepted cycles, halve after rejection (default)") { adaptive_gamma = true }
  parser.on("--no-adaptive", "Use fixed --gamma for every speculative cycle") { adaptive_gamma = false }
  parser.on("--tokens N", "Generated tokens to compare") { |value| n_gen = value.to_i }
  parser.on("--verify MODE", "Target verifier: serial, chunk, chunk-inplace, hybrid, or staged (default: chunk-inplace)") { |value| verify_mode = value }
  parser.on("--stage-gate N", "For --verify staged, verify this many candidates before drafting/verifying the rest") { |value| stage_gate = value.to_i }
  parser.on("--trace", "Print per-cycle verifier decisions") { trace = true }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
  parser.unknown_args do |before_dash, _after_dash|
    prompt = before_dash.join(" ") unless before_dash.empty?
  end
end

raise ArgumentError.new("--gamma must be positive") unless gamma > 0
stage_gate = gamma if stage_gate <= 0
raise ArgumentError.new("QWEN35_SPEC_FULL_ACCEPT_STREAK must be positive") unless adaptive_full_accept_streak > 0
raise ArgumentError.new("QWEN35_SPEC_FAST_REGROW_MIN_GAMMA must be non-negative") unless adaptive_fast_regrow_min_gamma >= 0
raise ArgumentError.new("QWEN35_SPEC_BOOTSTRAP_GAMMA must be non-negative") unless adaptive_bootstrap_gamma >= 0
raise ArgumentError.new("--max-gamma must be positive") unless max_gamma > 0
max_gamma = Math.max(max_gamma, gamma)
raise ArgumentError.new("QWEN35_SPEC_PLAIN_FALLBACK_GAMMA must be positive") unless plain_fallback_gamma > 0
raise ArgumentError.new("--tokens must be positive") unless n_gen > 0
raise ArgumentError.new("--verify must be serial, chunk, chunk-inplace, hybrid, or staged") unless {"serial", "chunk", "chunk-inplace", "hybrid", "staged"}.includes?(verify_mode)

def load_tokenizer(model_path : String, tokenizer_bin : String) : ML::GGUF::Qwen35Tokenizer
  g = ML::GGUF::GGUFFile.new(model_path)
  ML::GGUF::Qwen35Tokenizer.from_gguf(g, model_path, tokenizer_bin)
ensure
  g.try(&.close)
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

def greedy_sequence(weights : ML::GGUF::Qwen35Weights,
                    prompt_ids : Array(Int32),
                    n_gen : Int32) : Array(Int32)
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + n_gen + 8)
  next_id = prefill_next(weights, prompt_ids, state)
  pos = prompt_ids.size
  ids = [] of Int32
  n_gen.times do
    ids << next_id
    next_id = advance_next(weights, next_id, pos, state)
    pos += 1
  end
  ids
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

puts "Loading tokenizer and models..."
t0 = Time.instant
tok = load_tokenizer(target_path, tokenizer_bin)
target = ML::GGUF::Qwen35Weights.from_gguf(target_path)
draft = ML::GGUF::Qwen35Weights.from_gguf(draft_path)
load_s = (Time.instant - t0).total_seconds

unless target.output.out_dim == draft.output.out_dim
  raise ArgumentError.new("target/draft vocab mismatch: #{target.output.out_dim} != #{draft.output.out_dim}")
end

prompt_ids = tok.encode(prompt)
raise ArgumentError.new("prompt encoded to no tokens") if prompt_ids.empty?

puts "Loaded in #{load_s.round(2)}s"
puts "target: layers=#{target.hparams.n_layer} dim=#{target.hparams.n_embd} vocab=#{target.output.out_dim}"
puts "draft:  layers=#{draft.hparams.n_layer} dim=#{draft.hparams.n_embd} vocab=#{draft.output.out_dim}"
puts "prompt tokens=#{prompt_ids.size} gamma=#{gamma} max_gamma=#{max_gamma} adaptive=#{adaptive_gamma} adaptive_regrow=#{adaptive_regrow} full_accept_streak=#{adaptive_full_accept_streak} fast_regrow_min_gamma=#{adaptive_fast_regrow_min_gamma} bootstrap_gamma=#{adaptive_bootstrap_gamma} early_reject=#{early_reject_enabled} single_fast=#{single_accept_fast_enabled} plain_fallback=#{plain_fallback_enabled} fallback_gamma=#{plain_fallback_gamma} skip_draft_before_fallback=#{skip_draft_before_fallback_enabled} skip_draft_backup_before_fallback=#{skip_draft_backup_before_fallback_enabled} stage_gate=#{stage_gate} n_gen=#{n_gen} verify=#{verify_mode}"

max_seq = prompt_ids.size + n_gen + gamma + 8
target_state = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: max_seq)
draft_state = ML::GGUF::Qwen35CPU::State.new(draft.hparams, max_seq: max_seq)
target_backup_state = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: max_seq)
draft_cycle_base = ML::GGUF::Qwen35CPU::State.new(draft.hparams, max_seq: max_seq)

target_next = prefill_next(target, prompt_ids, target_state)
draft_next = prefill_next(draft, prompt_ids, draft_state)

generated_ids = [] of Int32
pos = prompt_ids.size
accepted = 0
proposed = 0
cycles = 0
target_verify_ms = 0.0
draft_ms = 0.0
target_backup_ms = 0.0
draft_backup_ms = 0.0
draft_resync_ms = 0.0
current_gamma = gamma
full_accept_streak = 0
adaptive_growth_allowed = true
gamma_sum = 0
gamma_max_seen = 0
early_rejects = 0
single_accept_fast = 0
plain_fallback_tokens = 0
draft_skips_before_fallback = 0
draft_backup_skips = 0

wall0 = Time.instant
while generated_ids.size < n_gen
  if plain_fallback_enabled && adaptive_gamma && !adaptive_growth_allowed && current_gamma <= plain_fallback_gamma
    generated_ids << target_next
    tv0 = Time.instant
    target_next = advance_next(target, target_next, pos, target_state)
    target_verify_ms += (Time.instant - tv0).total_milliseconds
    pos += 1
    plain_fallback_tokens += 1
    next
  end

  cycles += 1
  cycle_start_pos = pos
  cycle_gamma = adaptive_gamma ? current_gamma : gamma
  gamma_sum += cycle_gamma
  gamma_max_seen = Math.max(gamma_max_seen, cycle_gamma)
  correction_or_accepted = [] of Int32
  candidates = [] of Int32
  rejected = false
  cycle_done = false

  if early_reject_enabled && draft_next != target_next
    will_plain_fallback_after_reject = plain_fallback_enabled &&
                                       skip_draft_before_fallback_enabled &&
                                       adaptive_gamma &&
                                       !adaptive_regrow &&
                                       Math.max(1, current_gamma // 2) <= plain_fallback_gamma
    generated_ids << target_next
    correction_or_accepted << target_next
    proposed += 1
    tv0 = Time.instant
    target_next = advance_next(target, target_next, pos, target_state)
    target_verify_ms += (Time.instant - tv0).total_milliseconds
    if will_plain_fallback_after_reject || generated_ids.size >= n_gen
      draft_skips_before_fallback += 1
    else
      td0 = Time.instant
      draft_next = advance_next(draft, correction_or_accepted[0], pos, draft_state)
      draft_ms += (Time.instant - td0).total_milliseconds
    end
    pos += 1
    rejected = true
    early_rejects += 1
    cycle_done = true
  elsif single_accept_fast_enabled && cycle_gamma == 1 && draft_next == target_next
    accepted_token = draft_next
    generated_ids << accepted_token
    correction_or_accepted << accepted_token
    accepted += 1
    proposed += 1
    td0 = Time.instant
    draft_next = advance_next(draft, accepted_token, pos, draft_state)
    draft_ms += (Time.instant - td0).total_milliseconds
    tv0 = Time.instant
    target_next = advance_next(target, accepted_token, pos, target_state)
    target_verify_ms += (Time.instant - tv0).total_milliseconds
    pos += 1
    single_accept_fast += 1
    rejected = false
    cycle_done = true
  end

  unless cycle_done
    skip_draft_backup_for_fallback = plain_fallback_enabled &&
                                     skip_draft_before_fallback_enabled &&
                                     skip_draft_backup_before_fallback_enabled &&
                                     adaptive_gamma &&
                                     !adaptive_regrow &&
                                     Math.max(1, current_gamma // 2) <= plain_fallback_gamma
    unless skip_draft_backup_for_fallback
      tdb0 = Time.instant
      draft_cycle_base.copy_from!(draft_state)
      draft_backup_ms += (Time.instant - tdb0).total_milliseconds
    else
      draft_backup_skips += 1
    end

    if verify_mode == "staged"
      remaining_gamma = cycle_gamma
      stage_index = 0
      while remaining_gamma > 0 && generated_ids.size < n_gen
        if early_reject_enabled && draft_next != target_next
          generated_ids << target_next
          correction_or_accepted << target_next
          proposed += 1
          tv0 = Time.instant
          target_next = advance_next(target, target_next, pos, target_state)
          target_verify_ms += (Time.instant - tv0).total_milliseconds
          pos += 1
          rejected = true
          early_rejects += 1
          break
        end

        stage_len = if stage_index == 0 && remaining_gamma > stage_gate
                      stage_gate
                    else
                      remaining_gamma
                    end
        stage_len = Math.min(stage_len, n_gen - generated_ids.size)
        break if stage_len <= 0

        stage_start_pos = pos
        stage_candidates = [] of Int32
        td0 = Time.instant
        stage_len.times do |i|
          break if generated_ids.size + stage_candidates.size >= n_gen
          stage_candidates << draft_next
          draft_next = advance_next(draft, draft_next, stage_start_pos + i, draft_state)
        end
        draft_ms += (Time.instant - td0).total_milliseconds
        proposed += stage_candidates.size
        candidates.concat(stage_candidates)
        break if stage_candidates.empty?

        tb0 = Time.instant
        target_backup_state.copy_from!(target_state)
        target_backup_ms += (Time.instant - tb0).total_milliseconds
        tv0 = Time.instant
        target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, stage_candidates, stage_start_pos, target_state)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "cycle=#{cycles} stage=#{stage_index} pos=#{stage_start_pos} expected0=#{target_next} candidates=#{stage_candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        stage_correction_or_accepted = [] of Int32
        stage_candidates.each_with_index do |cand, i|
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            stage_correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0]
          else
            generated_ids << expected
            correction_or_accepted << expected
            stage_correction_or_accepted << expected
            rejected = true
            break
          end
        end

        if rejected
          target_state.copy_from!(target_backup_state)
          tv1 = Time.instant
          corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, stage_correction_or_accepted, stage_start_pos, target_state)
          target_verify_ms += (Time.instant - tv1).total_milliseconds
          target_next = corrected[-1][0]
          pos += stage_correction_or_accepted.size
          break
        else
          target_next = target_nexts[-1][0]
          pos += stage_candidates.size
          remaining_gamma -= stage_candidates.size
          stage_index += 1
        end
      end
    else
      td0 = Time.instant
      cycle_gamma.times do |i|
        break if generated_ids.size + candidates.size >= n_gen
        candidates << draft_next
        draft_next = advance_next(draft, draft_next, pos + i, draft_state)
      end
      draft_ms += (Time.instant - td0).total_milliseconds
      proposed += candidates.size

      if verify_mode == "serial" || (verify_mode == "hybrid" && cycles == 1)
        candidates.each do |cand|
          if cand == target_next
            generated_ids << cand
            correction_or_accepted << cand
            accepted += 1
            tv0 = Time.instant
            target_next = advance_next(target, cand, pos, target_state)
            target_verify_ms += (Time.instant - tv0).total_milliseconds
            pos += 1
          else
            generated_ids << target_next
            correction_or_accepted << target_next
            tv0 = Time.instant
            target_next = advance_next(target, target_next, pos, target_state)
            target_verify_ms += (Time.instant - tv0).total_milliseconds
            pos += 1
            rejected = true
            break
          end
        end
      elsif verify_mode == "chunk"
        target_base = target_state
        verify_state = target_base.fork
        tv0 = Time.instant
        target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, candidates, cycle_start_pos, verify_state)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "cycle=#{cycles} pos=#{cycle_start_pos} expected0=#{target_next} candidates=#{candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        reject_at = nil.as(Int32?)
        candidates.each_with_index do |cand, i|
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0]
          else
            generated_ids << expected
            correction_or_accepted << expected
            reject_at = i
            rejected = true
            break
          end
        end

        if rejected
          tv1 = Time.instant
          corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, correction_or_accepted, cycle_start_pos, target_state)
          target_verify_ms += (Time.instant - tv1).total_milliseconds
          target_next = corrected[-1][0]
          pos += correction_or_accepted.size
        else
          target_state.copy_from!(verify_state)
          target_next = target_nexts[-1][0]
          pos += candidates.size
        end
      else
        tb0 = Time.instant
        target_backup_state.copy_from!(target_state)
        target_backup_ms += (Time.instant - tb0).total_milliseconds
        tv0 = Time.instant
        target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, candidates, cycle_start_pos, target_state)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "cycle=#{cycles} pos=#{cycle_start_pos} expected0=#{target_next} candidates=#{candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        candidates.each_with_index do |cand, i|
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0]
          else
            generated_ids << expected
            correction_or_accepted << expected
            rejected = true
            break
          end
        end

        if rejected
          target_state.copy_from!(target_backup_state)
          tv1 = Time.instant
          corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, correction_or_accepted, cycle_start_pos, target_state)
          target_verify_ms += (Time.instant - tv1).total_milliseconds
          target_next = corrected[-1][0]
          pos += correction_or_accepted.size
        else
          target_next = target_nexts[-1][0]
          pos += candidates.size
        end
      end
    end

    if rejected
      will_plain_fallback_after_reject = plain_fallback_enabled &&
                                         skip_draft_before_fallback_enabled &&
                                         adaptive_gamma &&
                                         !adaptive_regrow &&
                                         Math.max(1, current_gamma // 2) <= plain_fallback_gamma
      if will_plain_fallback_after_reject || generated_ids.size >= n_gen
        draft_skips_before_fallback += 1
      else
        raise "draft backup missing before required resync" if skip_draft_backup_for_fallback
        tr0 = Time.instant
        draft_next = resync_draft!(draft, draft_state, draft_cycle_base, correction_or_accepted, cycle_start_pos)
        draft_resync_ms += (Time.instant - tr0).total_milliseconds
      end
    end
  end

  if adaptive_gamma
    if rejected
      full_accept_streak = 0
      adaptive_growth_allowed = false unless adaptive_regrow
      current_gamma = Math.max(1, current_gamma // 2)
    elsif adaptive_growth_allowed && candidates.size == cycle_gamma && current_gamma < max_gamma
      full_accept_streak += 1
      if adaptive_bootstrap_gamma > current_gamma && current_gamma == gamma
        current_gamma = Math.min(max_gamma, adaptive_bootstrap_gamma)
        full_accept_streak = 0
      else
        required_full_accept_streak = if adaptive_fast_regrow_min_gamma > 0 && current_gamma >= adaptive_fast_regrow_min_gamma
                                        1
                                      else
                                        adaptive_full_accept_streak
                                      end
        if full_accept_streak >= required_full_accept_streak
          current_gamma = Math.min(max_gamma, current_gamma * 2)
          full_accept_streak = 0
        end
      end
    end
  end
end
wall_ms = (Time.instant - wall0).total_milliseconds

plain0 = Time.instant
plain = greedy_sequence(target, prompt_ids, n_gen)
plain_ms = (Time.instant - plain0).total_milliseconds
unless plain == generated_ids
  first_diff = plain.zip(generated_ids).index { |(a, b)| a != b } || Math.min(plain.size, generated_ids.size)
  raise "speculative output diverged from target greedy at #{first_diff}: plain=#{plain.inspect} speculative=#{generated_ids.inspect}"
end

accept_rate = accepted.to_f64 / proposed.to_f64
tokens_s = n_gen.to_f64 / (wall_ms / 1000.0)
plain_tokens_s = n_gen.to_f64 / (plain_ms / 1000.0)

puts
puts "accept_rate=#{(accept_rate * 100.0).round(2)}% accepted=#{accepted}/#{proposed} cycles=#{cycles}"
avg_gamma = cycles > 0 ? (gamma_sum.to_f64 / cycles.to_f64).round(2) : 0.0
puts "gamma_stats avg=#{avg_gamma} max_seen=#{gamma_max_seen} final=#{current_gamma} early_rejects=#{early_rejects} single_fast=#{single_accept_fast} plain_fallback=#{plain_fallback_tokens} draft_skip=#{draft_skips_before_fallback} draft_backup_skip=#{draft_backup_skips}"
puts "spec_wall=#{wall_ms.round(1)} ms (#{(wall_ms / n_gen).round(2)} ms/tok, #{tokens_s.round(2)} tok/s, verify=#{verify_mode})"
puts "plain_target_wall=#{plain_ms.round(1)} ms (#{(plain_ms / n_gen).round(2)} ms/tok, #{plain_tokens_s.round(2)} tok/s)"
puts "time_breakdown draft=#{draft_ms.round(1)} ms target_verify=#{target_verify_ms.round(1)} ms target_backup=#{target_backup_ms.round(1)} ms draft_backup=#{draft_backup_ms.round(1)} ms draft_resync=#{draft_resync_ms.round(1)} ms"
puts "note=exact speculative probe; speedup still needs lower draft cost and/or verifier rollback overhead removal"
puts "generated=#{tok.decode(generated_ids).inspect}"
