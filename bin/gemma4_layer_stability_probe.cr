require "option_parser"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/gemma4_state_snapshot"
require "../src/ml/gguf/gemma4_tokenizer"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER_BIN
prompt_text = "Write a small Crystal function `fib(n : Int32) : Int32` using iteration. Return only code."
prompt_file = nil.as(String?)
chat_user = nil.as(String?)
token_ids_arg = nil.as(String?)
gen = 2
max_seq = 256
emit_layers = false
prefill_chunk = 128

def top1_id(logits : Array(Float32)) : Int32
  best_id = 0
  best = logits[0]
  logits.each_with_index do |v, i|
    if v > best
      best = v
      best_id = i
    end
  end
  best_id.to_i32
end

def top2_margin(logits : Array(Float32), best_id : Int32) : Float32
  best = logits[best_id]
  second = -Float32::INFINITY
  logits.each_with_index do |v, i|
    next if i == best_id
    second = v if v > second
  end
  best - second
end

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
  bigram_counts.each_value { |count| repeated_bigrams += count - 1 if count > 1 }

  adjacent_repeats = 0
  if token_count > 1
    (0...(token_count - 1)).each { |i| adjacent_repeats += 1 if token_ids[i] == token_ids[i + 1] }
  end

  bigram_total = token_count > 1 ? token_count - 1 : 0
  bigram_repeat_rate = bigram_total > 0 ? repeated_bigrams.to_f64 / bigram_total : 0.0
  adjacent_repeat_rate = bigram_total > 0 ? adjacent_repeats.to_f64 / bigram_total : 0.0
  PromptEntryFeatures.new(token_count, unique_rate, repeat_rate, bigram_repeat_rate, adjacent_repeat_rate)
end

record LayerTraceRow, layer : Int32, top1 : Int32, logit : Float32, margin : Float32

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_layer_stability_probe [options]"
  p.on("--model PATH", "Gemma4 GGUF model path") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Raw prompt text") { |v| prompt_text = v; chat_user = nil }
  p.on("--prompt-file PATH", "Read raw prompt text from file") { |v| prompt_file = v; chat_user = nil }
  p.on("--chat-user TEXT", "Format one Gemma4 user turn") { |v| chat_user = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids; bypasses tokenizer") { |v| token_ids_arg = v }
  p.on("--gen N", "Generated tokens to trace, default 2") { |v| gen = v.to_i }
  p.on("--max-seq N", "Resident state sequence capacity, default 256") { |v| max_seq = v.to_i }
  p.on("--prefill-chunk N", "Row prefill chunk size, default 128") { |v| prefill_chunk = v.to_i }
  p.on("--layer-rows", "Emit one TSV row per layer as well as summaries") { emit_layers = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--gen must be positive" unless gen > 0
raise "--max-seq must be positive" unless max_seq > 0
raise "--prefill-chunk must be positive" unless prefill_chunk > 0
raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "tokenizer binary not found: #{tokenizer_bin}" unless token_ids_arg || File.exists?(tokenizer_bin)

if file = prompt_file
  prompt_text = File.read(file)
end
if user = chat_user
  prompt_text = "<|turn>user\n#{user}<turn|>\n<|turn>model\n"
end

weights = ML::GGUF::Gemma4Weights.from_gguf(model_path)
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

ids = if raw = token_ids_arg
        raw.split(',').reject(&.empty?).map(&.to_i32)
      else
        g = ML::GGUF::GGUFFile.new(model_path)
        tok = ML::GGUF::Gemma4Tokenizer.from_gguf(g, model_path, tokenizer_bin)
        g.close
        tok.encode(prompt_text)
      end
raise "prompt tokenized to zero tokens" if ids.empty?
raise "prompt+gen exceeds max_seq" if ids.size + gen > max_seq

main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
trace_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)

if ids.size > 1
  prefix = ids[0...-1]
  ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(
    weights, prefix, 0, main_state,
    chunk_size: prefill_chunk,
    stop_layer: weights.hparams.n_layer,
    read_last_hidden: false
  ).not_nil!
end

current_token = ids[-1]
pos = ids.size - 1
history_ids = ids.dup

puts "kind\tstep\tpos\tinput_token\tfinal_top1\tstable_from_layer\ttop1_changes\tfinal_logit\tfinal_margin\tprompt_tokens\tunique_rate\trepeat_rate\tbigram_repeat_rate\tadjacent_repeat_rate"
puts "layer\tstep\tpos\tlayer_index\tlayer_top1\tmatch_final\tlogit\tmargin" if emit_layers

gen.times do |step|
  features = prompt_entry_features(history_ids)
  snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
  trace = [] of LayerTraceRow
  final_state_snapshot = nil.as(ML::GGUF::Gemma4StateSnapshot::Snapshot?)

  (1..weights.hparams.n_layer).each do |layer_count|
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, trace_state)
    hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
      weights, current_token, pos.to_i32, trace_state, layer_count
    ).not_nil!
    logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
    top1 = top1_id(logits)
    trace << LayerTraceRow.new(layer_count, top1, logits[top1], top2_margin(logits, top1))
    if layer_count == weights.hparams.n_layer
      final_state_snapshot = ML::GGUF::Gemma4StateSnapshot.capture(trace_state, prefix_len: (pos + 1).to_i32)
    end
  end

  final = trace[-1]
  stable_from = trace.size
  trace.each_with_index do |row, i|
    if trace[i..].all? { |candidate| candidate.top1 == final.top1 }
      stable_from = row.layer
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

  puts ["summary", step, pos, current_token, final.top1, stable_from, changes, final.logit, final.margin,
        features.token_count, features.unique_rate, features.repeat_rate,
        features.bigram_repeat_rate, features.adjacent_repeat_rate].join('\t')
  if emit_layers
    trace.each do |row|
      puts ["layer", step, pos, row.layer, row.top1, row.top1 == final.top1, row.logit, row.margin].join('\t')
    end
  end

  ML::GGUF::Gemma4StateSnapshot.restore_into(final_state_snapshot.not_nil!, main_state)
  current_token = final.top1
  history_ids << current_token
  pos += 1
end
