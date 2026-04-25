require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/ngram_draft"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "option_parser"

DEFAULT_TARGET    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

target_path = ENV["QWEN35_TARGET"]? || DEFAULT_TARGET
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = "The capital of France is"
n_gen = 64
gamma = (ENV["QWEN35_NGRAM_GAMMA"]? || "16").to_i
max_ngram = (ENV["QWEN35_NGRAM_MAX"]? || "8").to_i
min_ngram = (ENV["QWEN35_NGRAM_MIN"]? || "6").to_i
check_plain = ENV["QWEN35_NGRAM_CHECK_PLAIN"]? != "0"

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_ngram_speculative [--target PATH] [--tokenizer-bin PATH] [--tokens N] [--gamma N] [--min-ngram N] [--max-ngram N] [--no-check] [prompt]"
  parser.on("--target PATH", "Target GGUF path (default: Qwen3.5 9B Q4_K_M)") { |path| target_path = path }
  parser.on("--tokenizer-bin PATH", "llama.cpp tokenizer helper path") { |path| tokenizer_bin = path }
  parser.on("--tokens N", "Generated tokens to compare (default: 64)") { |value| n_gen = value.to_i }
  parser.on("--gamma N", "Maximum n-gram draft candidates per verifier chunk (default: env QWEN35_NGRAM_GAMMA or 16)") { |value| gamma = value.to_i }
  parser.on("--min-ngram N", "Minimum suffix match length before drafting (default: env QWEN35_NGRAM_MIN or 6)") { |value| min_ngram = value.to_i }
  parser.on("--max-ngram N", "Maximum suffix match length to search (default: env QWEN35_NGRAM_MAX or 8)") { |value| max_ngram = value.to_i }
  parser.on("--no-check", "Skip plain greedy replay/equality check") { check_plain = false }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
  parser.unknown_args do |before_dash, _after_dash|
    prompt = before_dash.join(" ") unless before_dash.empty?
  end
end

raise ArgumentError.new("--tokens must be positive") unless n_gen > 0
raise ArgumentError.new("--gamma must be positive") unless gamma > 0
raise ArgumentError.new("--min-ngram must be positive") unless min_ngram > 0
raise ArgumentError.new("--max-ngram must be >= --min-ngram") unless max_ngram >= min_ngram

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
                    n_gen : Int32,
                    max_seq : Int32) : {Array(Int32), Float64}
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  next_id = prefill_next(weights, prompt_ids, state)
  ids = [] of Int32
  t0 = Time.instant
  n_gen.times do |i|
    ids << next_id
    next_id = advance_next(weights, next_id, prompt_ids.size + i, state)
  end
  {ids, (Time.instant - t0).total_milliseconds}
end

puts "Loading tokenizer and target model..."
t0 = Time.instant
tok = load_tokenizer(target_path, tokenizer_bin)
weights = ML::GGUF::Qwen35Weights.from_gguf(target_path)
load_s = (Time.instant - t0).total_seconds

prompt_ids = tok.encode(prompt)
raise ArgumentError.new("prompt encoded to no tokens") if prompt_ids.empty?

puts "Loaded in #{load_s.round(2)}s"
puts "target: layers=#{weights.hparams.n_layer} dim=#{weights.hparams.n_embd} vocab=#{weights.output.out_dim}"
puts "prompt tokens=#{prompt_ids.size} gamma=#{gamma} min_ngram=#{min_ngram} max_ngram=#{max_ngram} n_gen=#{n_gen}"

max_seq = prompt_ids.size + n_gen + gamma + 8
state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
backup = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
target_next = prefill_next(weights, prompt_ids, state)
pos = prompt_ids.size
history = prompt_ids.dup
generated_ids = [] of Int32

accepted = 0
proposed = 0
cycles = 0
plain_steps = 0
target_verify_ms = 0.0
target_backup_ms = 0.0

wall0 = Time.instant
while generated_ids.size < n_gen
  candidates = ML::GGUF::NgramDraft.candidates(history, Math.min(gamma, n_gen - generated_ids.size), max_ngram, min_ngram)

  if candidates.empty?
    generated_ids << target_next
    history << target_next
    tv0 = Time.instant
    target_next = advance_next(weights, target_next, pos, state)
    target_verify_ms += (Time.instant - tv0).total_milliseconds
    pos += 1
    plain_steps += 1
    next
  end

  cycles += 1
  proposed += candidates.size
  tb0 = Time.instant
  backup.copy_from!(state)
  target_backup_ms += (Time.instant - tb0).total_milliseconds
  tv0 = Time.instant
  target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, pos, state)
  target_verify_ms += (Time.instant - tv0).total_milliseconds

  expected = target_next
  accepted_or_corrected = [] of Int32
  rejected = false
  candidates.each_with_index do |cand, i|
    if cand == expected
      generated_ids << cand
      history << cand
      accepted_or_corrected << cand
      accepted += 1
      expected = target_nexts[i][0]
    else
      generated_ids << expected
      history << expected
      accepted_or_corrected << expected
      rejected = true
      break
    end
  end

  if rejected
    state.copy_from!(backup)
    tv1 = Time.instant
    corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, accepted_or_corrected, pos, state)
    target_verify_ms += (Time.instant - tv1).total_milliseconds
    target_next = corrected[-1][0]
    pos += accepted_or_corrected.size
  else
    target_next = target_nexts[-1][0]
    pos += candidates.size
  end
end
wall_ms = (Time.instant - wall0).total_milliseconds

plain_ms = nil.as(Float64?)
if check_plain
  plain_ids, measured_plain_ms = greedy_sequence(weights, prompt_ids, n_gen, max_seq)
  raise "ngram speculative output mismatch" unless plain_ids == generated_ids
  plain_ms = measured_plain_ms
end

accept_rate = proposed == 0 ? 0.0 : accepted * 100.0 / proposed
tokens_s = n_gen.to_f64 / (wall_ms / 1000.0)
puts "accept_rate=#{accept_rate.round(2)}% accepted=#{accepted}/#{proposed} cycles=#{cycles} plain_steps=#{plain_steps}"
puts "ngram_wall=#{wall_ms.round(1)} ms (#{(wall_ms / n_gen).round(2)} ms/tok, #{tokens_s.round(2)} tok/s)"
if measured = plain_ms
  puts "plain_target_wall=#{measured.round(1)} ms (#{(measured / n_gen).round(2)} ms/tok)"
end
puts "time_breakdown target_verify=#{target_verify_ms.round(1)} ms target_backup=#{target_backup_ms.round(1)} ms"
puts "generated=#{tok.decode(generated_ids).inspect}"
