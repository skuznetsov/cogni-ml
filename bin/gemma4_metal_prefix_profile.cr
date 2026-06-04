require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/gguf/gemma4_metal"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
tokens = [42, 43]
stop_layer = 2
max_seq = 1024
warmups = 1
runs = 3
mode = "both"
include_state_init = false
with_head = false

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_metal_prefix_profile [--tokens 42,43] [--stop-layer 2] [--max-seq 1024] [--runs 3] [--mode host|resident|both] [--with-head]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--tokens IDS", "Comma-separated token ids") { |v| tokens = v.split(',').reject(&.empty?).map(&.to_i) }
  p.on("--stop-layer N", "Layer prefix length") { |v| stop_layer = v.to_i }
  p.on("--max-seq N", "KV cache sequence capacity") { |v| max_seq = v.to_i }
  p.on("--warmups N", "Warmup iterations") { |v| warmups = v.to_i }
  p.on("--runs N", "Measured iterations") { |v| runs = v.to_i }
  p.on("--mode MODE", "host, resident, or both") { |v| mode = v }
  p.on("--include-state-init", "Include state allocation in timed samples") { include_state_init = true }
  p.on("--with-head", "Run final output RMSNorm/lm-head on the last hidden state") { with_head = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokens must not be empty" if tokens.empty?
raise "stop-layer must be positive" unless stop_layer > 0
raise "max-seq must cover tokens" if max_seq < tokens.size
raise "runs must be positive" unless runs > 0
raise "mode must be host, resident, or both" unless {"host", "resident", "both"}.includes?(mode)

def percentile(sorted : Array(Float64), p : Float64) : Float64
  return 0.0 if sorted.empty?
  idx = ((sorted.size - 1).to_f64 * p).round.to_i
  sorted[idx]
end

def summarize(label : String, samples : Array(Float64), token_count : Int32) : Nil
  sorted = samples.sort
  mean = samples.sum / samples.size
  p50 = percentile(sorted, 0.50)
  p90 = percentile(sorted, 0.90)
  tok_s = token_count.to_f64 / (p50 / 1000.0)
  puts "#{label}_runs=#{samples.map { |v| v.round(3) }.join(',')}"
  puts "#{label}_mean_ms=#{mean.round(3)} #{label}_p50_ms=#{p50.round(3)} #{label}_p90_ms=#{p90.round(3)} #{label}_p50_tok_s=#{tok_s.round(3)}"
end

def run_host_with_state(weights : ML::GGUF::Gemma4Weights, tokens : Array(Int32), stop_layer : Int32,
                        state : ML::GGUF::Gemma4Metal::State, with_head : Bool) : Array(Float32)
  hidden = [] of Float32
  tokens.each_with_index do |token_id, pos|
    hidden = ML::GGUF::Gemma4Metal.forward_hidden(weights, token_id, pos, state, stop_layer).not_nil!
  end
  ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil! if with_head
  hidden
end

def run_resident_with_state(weights : ML::GGUF::Gemma4Weights, tokens : Array(Int32), stop_layer : Int32,
                            state : ML::GGUF::Gemma4Metal::ResidentState, with_head : Bool) : Array(Float32)
  hidden = [] of Float32
  tokens.each_with_index do |token_id, pos|
    hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos, state, stop_layer).not_nil!
  end
  ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil! if with_head
  hidden
end

started = Time.instant
weights = ML::GGUF::Gemma4Weights.from_gguf(model)
load_ms = (Time.instant - started).total_milliseconds
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

puts "model=#{File.basename(model)} tokens=#{tokens.join(',')} stop_layer=#{stop_layer} max_seq=#{max_seq} warmups=#{warmups} runs=#{runs} include_state_init=#{include_state_init} with_head=#{with_head} load_ms=#{load_ms.round(3)}"

host_samples = [] of Float64
resident_samples = [] of Float64

if mode == "host" || mode == "both"
  warmups.times do
    state = ML::GGUF::Gemma4Metal::State.new(weights.hparams, max_seq)
    run_host_with_state(weights, tokens, stop_layer, state, with_head)
  end
  runs.times do
    state = ML::GGUF::Gemma4Metal::State.new(weights.hparams, max_seq) unless include_state_init
    t0 = Time.instant
    if include_state_init
      state = ML::GGUF::Gemma4Metal::State.new(weights.hparams, max_seq)
      run_host_with_state(weights, tokens, stop_layer, state, with_head)
    else
      run_host_with_state(weights, tokens, stop_layer, state.not_nil!, with_head)
    end
    host_samples << (Time.instant - t0).total_milliseconds
  end
  summarize("host", host_samples, tokens.size)
end

if mode == "resident" || mode == "both"
  warmups.times do
    state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
    run_resident_with_state(weights, tokens, stop_layer, state, with_head)
  end
  runs.times do
    state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq) unless include_state_init
    t0 = Time.instant
    if include_state_init
      state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
      run_resident_with_state(weights, tokens, stop_layer, state, with_head)
    else
      run_resident_with_state(weights, tokens, stop_layer, state.not_nil!, with_head)
    end
    resident_samples << (Time.instant - t0).total_milliseconds
  end
  summarize("resident", resident_samples, tokens.size)
end

if mode == "both"
  host_p50 = percentile(host_samples.sort, 0.50)
  resident_p50 = percentile(resident_samples.sort, 0.50)
  speedup = host_p50 / resident_p50
  puts "resident_vs_host_p50_speedup=#{speedup.round(4)}"
end
