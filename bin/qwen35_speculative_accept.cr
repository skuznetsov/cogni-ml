# Greedy speculative-decode acceptance probe for Qwen35 target/draft pairs.
#
# This intentionally verifies candidate tokens one-by-one with the target model.
# It is a correctness and acceptance-rate harness, not the final batched verifier
# speed path. The final speedup requires replacing serial verification with a
# multi-token target verify kernel/path.

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

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_speculative_accept [--target PATH] [--draft PATH] [--gamma N] [--tokens N] [prompt]"
  parser.on("--target PATH", "Target GGUF path (default: Qwen3.5 9B Q4_K_M)") { |path| target_path = path }
  parser.on("--draft PATH", "Draft GGUF path (default: Qwen3.5 0.8B Q8_0)") { |path| draft_path = path }
  parser.on("--tokenizer-bin PATH", "llama.cpp tokenizer helper path") { |path| tokenizer_bin = path }
  parser.on("--gamma N", "Draft candidates per cycle") { |value| gamma = value.to_i }
  parser.on("--tokens N", "Generated tokens to compare") { |value| n_gen = value.to_i }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
  parser.unknown_args do |before_dash, _after_dash|
    prompt = before_dash.join(" ") unless before_dash.empty?
  end
end

raise ArgumentError.new("--gamma must be positive") unless gamma > 0
raise ArgumentError.new("--tokens must be positive") unless n_gen > 0

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

def resync_draft(weights : ML::GGUF::Qwen35Weights,
                 base : ML::GGUF::Qwen35CPU::State,
                 accepted_or_corrected : Array(Int32),
                 start_pos : Int32) : {ML::GGUF::Qwen35CPU::State, Int32}
  state = base.fork
  next_id = -1
  accepted_or_corrected.each_with_index do |tok, i|
    next_id = advance_next(weights, tok, start_pos + i, state)
  end
  {state, next_id}
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
puts "prompt tokens=#{prompt_ids.size} gamma=#{gamma} n_gen=#{n_gen}"

max_seq = prompt_ids.size + n_gen + gamma + 8
target_state = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: max_seq)
draft_state = ML::GGUF::Qwen35CPU::State.new(draft.hparams, max_seq: max_seq)

target_next = prefill_next(target, prompt_ids, target_state)
draft_next = prefill_next(draft, prompt_ids, draft_state)

generated_ids = [] of Int32
pos = prompt_ids.size
accepted = 0
proposed = 0
cycles = 0
target_verify_ms = 0.0
draft_ms = 0.0

wall0 = Time.instant
while generated_ids.size < n_gen
  cycles += 1
  cycle_start_pos = pos
  cycle_draft_base = draft_state.fork
  candidates = [] of Int32

  td0 = Time.instant
  gamma.times do |i|
    break if generated_ids.size + candidates.size >= n_gen
    candidates << draft_next
    draft_next = advance_next(draft, draft_next, pos + i, draft_state)
  end
  draft_ms += (Time.instant - td0).total_milliseconds
  proposed += candidates.size

  correction_or_accepted = [] of Int32
  rejected = false
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

  if rejected
    draft_state, draft_next = resync_draft(draft, cycle_draft_base, correction_or_accepted, cycle_start_pos)
  end
end
wall_ms = (Time.instant - wall0).total_milliseconds

plain0 = Time.instant
plain = greedy_sequence(target, prompt_ids, n_gen)
plain_ms = (Time.instant - plain0).total_milliseconds
raise "speculative output diverged from target greedy" unless plain == generated_ids

accept_rate = accepted.to_f64 / proposed.to_f64
tokens_s = n_gen.to_f64 / (wall_ms / 1000.0)
plain_tokens_s = n_gen.to_f64 / (plain_ms / 1000.0)

puts
puts "accept_rate=#{(accept_rate * 100.0).round(2)}% accepted=#{accepted}/#{proposed} cycles=#{cycles}"
puts "serial_spec_wall=#{wall_ms.round(1)} ms (#{(wall_ms / n_gen).round(2)} ms/tok, #{tokens_s.round(2)} tok/s)"
puts "plain_target_wall=#{plain_ms.round(1)} ms (#{(plain_ms / n_gen).round(2)} ms/tok, #{plain_tokens_s.round(2)} tok/s)"
puts "time_breakdown draft=#{draft_ms.round(1)} ms target_verify=#{target_verify_ms.round(1)} ms"
puts "note=serial verifier is correctness-only; expected speedup needs batched target verify"
puts "generated=#{tok.decode(generated_ids).inspect}"
