require "option_parser"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/gemma4_state_snapshot"
require "../src/ml/gguf/gemma4_tokenizer"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER_BIN
chat_user = "Write a small Crystal function `fib(n : Int32) : Int32` using iteration. Return only code."
prompt_text = nil.as(String?)
prompt_file = nil.as(String?)
token_ids_arg = nil.as(String?)
gen = 16
max_seq = 256
prefill_chunk = 128
proposal_start = 3
proposal_layer = 34
print_text = false

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_phase_proposal_probe [options]"
  p.on("--model PATH", "Gemma4 GGUF model path") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Raw prompt text") { |v| prompt_text = v; chat_user = nil }
  p.on("--prompt-file PATH", "Read raw prompt text from file") { |v| prompt_file = v; chat_user = nil }
  p.on("--chat-user TEXT", "Format one Gemma4 user turn") { |v| chat_user = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids; bypasses tokenizer") { |v| token_ids_arg = v }
  p.on("--gen N", "Generated tokens to verify, default 16") { |v| gen = v.to_i }
  p.on("--max-seq N", "Resident state sequence capacity, default 256") { |v| max_seq = v.to_i }
  p.on("--prefill-chunk N", "Row prefill chunk size, default 128") { |v| prefill_chunk = v.to_i }
  p.on("--proposal-start N", "First decode loop step to score truncated proposal, default 3") { |v| proposal_start = v.to_i }
  p.on("--proposal-layer N", "Proposal stop layer, default 34") { |v| proposal_layer = v.to_i }
  p.on("--print-text", "Print conservative detokenized exact output") { print_text = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--gen must be positive" unless gen > 0
raise "--max-seq must be positive" unless max_seq > 0
raise "--prefill-chunk must be positive" unless prefill_chunk > 0
raise "--proposal-start must be non-negative" unless proposal_start >= 0
raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "tokenizer binary not found: #{tokenizer_bin}" unless token_ids_arg || File.exists?(tokenizer_bin)

if file = prompt_file
  prompt_text = File.read(file)
elsif user = chat_user
  prompt_text = "<|turn>user\n#{user}<turn|>\n<|turn>model\n"
end

weights = ML::GGUF::Gemma4Weights.from_gguf(model_path)
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?
raise "--proposal-layer must be positive" unless proposal_layer > 0
raise "--proposal-layer exceeds model layer count" if proposal_layer > weights.hparams.n_layer

tokenizer = nil.as(ML::GGUF::Gemma4Tokenizer?)
ids = if raw = token_ids_arg
        raw.split(',').reject(&.empty?).map(&.to_i32)
      else
        g = ML::GGUF::GGUFFile.new(model_path)
        tokenizer = ML::GGUF::Gemma4Tokenizer.from_gguf(g, model_path, tokenizer_bin)
        g.close
        text = prompt_text
        raise "prompt text is empty" unless text
        tokenizer.not_nil!.encode(text)
      end
raise "prompt tokenized to zero tokens" if ids.empty?
raise "prompt+gen exceeds max_seq" if ids.size + gen > max_seq

main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
proposal_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)

if ids.size > 1
  ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(
    weights, ids[0...-1], 0, main_state,
    chunk_size: prefill_chunk,
    stop_layer: weights.hparams.n_layer,
    read_last_hidden: false
  ).not_nil!
end

current_token = ids[-1]
pos = ids.size - 1
exact_trace = [] of Int32
proposal_hits = 0
proposal_scored = 0
proposal_misses = 0
proposal_ms = 0.0
exact_ms = 0.0

puts "model=#{File.basename(model_path)} prompt_len=#{ids.size} gen=#{gen} proposal_start=#{proposal_start} proposal_layer=#{proposal_layer} max_seq=#{max_seq}"
puts "step\tpos\tinput_token\tproposal_top1\texact_top1\thit\tproposal_ms\texact_ms"

gen.times do |step|
  proposal = -1_i32
  step_proposal_ms = 0.0
  if step >= proposal_start
    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, proposal_state)
    t0 = Time.instant
    proposal = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(
      weights, current_token, pos.to_i32, proposal_state, proposal_layer
    ).not_nil!
    step_proposal_ms = (Time.instant - t0).total_milliseconds
    proposal_ms += step_proposal_ms
    proposal_scored += 1
  end

  t1 = Time.instant
  exact = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(
    weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
  ).not_nil!
  step_exact_ms = (Time.instant - t1).total_milliseconds
  exact_ms += step_exact_ms

  hit = step < proposal_start ? true : proposal == exact
  if step >= proposal_start
    if hit
      proposal_hits += 1
    else
      proposal_misses += 1
    end
  end
  puts [step, pos, current_token, proposal, exact, hit, step_proposal_ms.round(3), step_exact_ms.round(3)].join('\t')

  exact_trace << exact
  current_token = exact
  pos += 1
end

rate = proposal_scored > 0 ? proposal_hits.to_f64 / proposal_scored : 0.0
puts "summary proposal_scored=#{proposal_scored} proposal_hits=#{proposal_hits} proposal_misses=#{proposal_misses} proposal_hit_rate=#{rate.round(4)} proposal_ms=#{proposal_ms.round(3)} exact_ms=#{exact_ms.round(3)} exact_ms_per_token=#{(exact_ms / gen).round(3)} proposal_ms_per_scored=#{proposal_scored > 0 ? (proposal_ms / proposal_scored).round(3) : 0.0}"
if print_text && tokenizer
  puts "exact_generated_text=#{tokenizer.not_nil!.decode(exact_trace).inspect}"
end
