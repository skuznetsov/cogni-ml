require "../src/ml/gguf/gemma4_cpu"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
tokens = [] of Int32
top_k = 10
max_seq = 128
stop_layer : Int32? = nil
emit_head = true

i = 0
while i < ARGV.size
  case ARGV[i]
  when "--model"
    i += 1; model = ARGV[i]
  when "--tokens"
    i += 1; tokens = ARGV[i].split(',').reject(&.empty?).map(&.to_i)
  when "--top-k"
    i += 1; top_k = ARGV[i].to_i
  when "--max-seq"
    i += 1; max_seq = ARGV[i].to_i
  when "--stop-layer"
    i += 1; stop_layer = ARGV[i].to_i
  when "--no-head"
    emit_head = false
  else
    raise "unknown arg #{ARGV[i]}"
  end
  i += 1
end

raise "usage: gemma4_cpu_probe --tokens 2,9259 [--stop-layer N] [--no-head]" if tokens.empty?
raise "model not found: #{model}" unless File.exists?(model)

started = Time.instant
weights = ML::GGUF::Gemma4Weights.from_gguf(model)
state = ML::GGUF::Gemma4CPU::State.new(weights.hparams, max_seq)
load_ms = (Time.instant - started).total_milliseconds

hidden = [] of Float32
tokens.each_with_index do |tid, pos|
  t0 = Time.instant
  hidden = ML::GGUF::Gemma4CPU.forward_hidden(weights, tid, pos, state, stop_layer: stop_layer)
  dt = (Time.instant - t0).total_milliseconds
  STDERR.puts "token_index=#{pos} token_id=#{tid} hidden_size=#{hidden.size} layer_stop=#{stop_layer || weights.layers.size} ms=#{dt.round(3)}"
end

puts "model=#{File.basename(model)} tokens=#{tokens.join(',')} load_ms=#{load_ms.round(3)} hidden_size=#{hidden.size}"
if emit_head && stop_layer.nil?
  t0 = Time.instant
  logits = ML::GGUF::Gemma4CPU.forward_logits_from_hidden(weights, hidden)
  head_ms = (Time.instant - t0).total_milliseconds
  puts "head_ms=#{head_ms.round(3)} vocab=#{logits.size}"
  ML::GGUF::Gemma4CPU.top_k(logits, top_k).each_with_index do |pair, rank|
    id, logit = pair
    puts "top#{rank + 1}=#{id}:#{logit}"
  end
else
  puts "head=skipped"
end
