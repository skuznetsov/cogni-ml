require "option_parser"
require "../src/ml/gguf/gemma4_metal"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
prompt = [42, 43, 44, 45, 46, 47, 48, 49]
generate = 8
max_seq = 1024
warmups = 1
runs = 3

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_metal_decode_profile [--tokens 42,43] [--generate 8] [--max-seq 1024] [--runs 3]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids") { |v| prompt = v.split(',').reject(&.empty?).map(&.to_i) }
  p.on("--generate N", "Measured generated tokens per run") { |v| generate = v.to_i }
  p.on("--max-seq N", "KV cache sequence capacity") { |v| max_seq = v.to_i }
  p.on("--warmups N", "Warmup runs") { |v| warmups = v.to_i }
  p.on("--runs N", "Measured runs") { |v| runs = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "prompt tokens must not be empty" if prompt.empty?
raise "generate must be positive" unless generate > 0
raise "runs must be positive" unless runs > 0
raise "max-seq too small" if max_seq < prompt.size + generate

def percentile(sorted : Array(Float64), p : Float64) : Float64
  return 0.0 if sorted.empty?
  idx = ((sorted.size - 1).to_f64 * p).round.to_i
  sorted[idx]
end

def summarize(label : String, samples : Array(Float64), tokens : Int32) : Nil
  sorted = samples.sort
  mean = samples.sum / samples.size
  p50 = percentile(sorted, 0.50)
  p90 = percentile(sorted, 0.90)
  tok_s = tokens.to_f64 / (p50 / 1000.0)
  puts "#{label}_runs=#{samples.map { |v| v.round(3) }.join(',')}"
  puts "#{label}_mean_ms=#{mean.round(3)} #{label}_p50_ms=#{p50.round(3)} #{label}_p90_ms=#{p90.round(3)} #{label}_p50_tok_s=#{tok_s.round(3)}"
end

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

def forward_top1(weights : ML::GGUF::Gemma4Weights, token_id : Int32, pos : Int32,
                 state : ML::GGUF::Gemma4Metal::ResidentState) : Int32
  hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos, state, weights.hparams.n_layer).not_nil!
  logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
  top1_id(logits)
end

def run_once(weights : ML::GGUF::Gemma4Weights, prompt : Array(Int32), generate : Int32,
             max_seq : Int32) : NamedTuple(prefill_ms: Float64, decode_ms: Float64, first_id: Int32, last_id: Int32)
  state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)

  prefill_t0 = Time.instant
  next_id = 0_i32
  prompt.each_with_index do |token_id, pos|
    next_id = forward_top1(weights, token_id, pos.to_i32, state)
  end
  prefill_ms = (Time.instant - prefill_t0).total_milliseconds

  decode_t0 = Time.instant
  cur = next_id
  generate.times do |i|
    cur = forward_top1(weights, cur, (prompt.size + i).to_i32, state)
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  {prefill_ms: prefill_ms, decode_ms: decode_ms, first_id: next_id, last_id: cur}
end

started = Time.instant
weights = ML::GGUF::Gemma4Weights.from_gguf(model)
load_ms = (Time.instant - started).total_milliseconds
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

puts "model=#{File.basename(model)} prompt_tokens=#{prompt.join(',')} prompt_len=#{prompt.size} generate=#{generate} max_seq=#{max_seq} warmups=#{warmups} runs=#{runs} load_ms=#{load_ms.round(3)}"

warmups.times { run_once(weights, prompt, generate, max_seq) }

prefill_samples = [] of Float64
decode_samples = [] of Float64
first_id = 0_i32
last_id = 0_i32
runs.times do
  result = run_once(weights, prompt, generate, max_seq)
  prefill_samples << result[:prefill_ms]
  decode_samples << result[:decode_ms]
  first_id = result[:first_id]
  last_id = result[:last_id]
end

summarize("prefill", prefill_samples, prompt.size)
summarize("decode", decode_samples, generate)
decode_p50 = percentile(decode_samples.sort, 0.50)
puts "decode_ms_per_token_p50=#{(decode_p50 / generate).round(3)} first_id=#{first_id} last_id=#{last_id}"
