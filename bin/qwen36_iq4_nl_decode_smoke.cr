require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_NL.gguf"

model = DEFAULT_MODEL
token_id = 0
pos = 0
max_seq = 8
prepare_state = true
warmup = 0
runs = 1

OptionParser.parse do |p|
  p.banner = "Usage: qwen36_iq4_nl_decode_smoke [--model PATH] [--token-id N] [--pos N] [--max-seq N] [--warmup N] [--runs N] [--no-prepare-state]"
  p.on("--model=PATH", "Qwen3.6 IQ4_NL GGUF path") { |v| model = File.expand_path(v) }
  p.on("--token-id=N", "Input token id (default: 0)") { |v| token_id = v.to_i }
  p.on("--pos=N", "Decode position (default: 0)") { |v| pos = v.to_i }
  p.on("--max-seq=N", "State cache length (default: 8)") { |v| max_seq = v.to_i }
  p.on("--warmup=N", "Warmup decode steps before measuring (default: 0)") { |v| warmup = v.to_i }
  p.on("--runs=N", "Measured decode steps (default: 1)") { |v| runs = v.to_i }
  p.on("--no-prepare-state", "Skip Metal state preallocation") { prepare_state = false }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

model = File.expand_path(model)
raise "model not found: #{model}" unless File.exists?(model)
raise "--token-id must be non-negative" if token_id < 0
raise "--pos must be non-negative" if pos < 0
raise "--max-seq must be positive" unless max_seq > 0
raise "--warmup must be non-negative" if warmup < 0
raise "--runs must be positive" unless runs > 0
raise "--pos + --warmup + --runs must fit in --max-seq" unless pos + warmup + runs <= max_seq

puts "Qwen3.6 IQ4_NL decode smoke"
puts "model=#{model}"
puts "token_id=#{token_id} pos=#{pos} max_seq=#{max_seq} warmup=#{warmup} runs=#{runs} prepare_state=#{prepare_state}"
puts "metal_available=#{ML::GGUF::Qwen35Metal.available?}"

load_t0 = Time.instant
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--token-id #{token_id} out of range 0...#{weights.token_embd.out_dim}" if token_id >= weights.token_embd.out_dim

puts "hparams raw_layers=#{hp.raw_block_count} target_layers=#{hp.n_layer} nextn_layers=#{hp.nextn_predict_layers} hidden=#{hp.n_embd} ff=#{hp.n_ff}"
puts "weight_types token_embd=#{weights.token_embd.type} output=#{weights.output.type}"
puts "load_ms=%.3f" % load_ms

state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq)
prep_ms = 0.0
if prepare_state
  prep_t0 = Time.instant
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  prep_ms = (Time.instant - prep_t0).total_milliseconds
end

cur_token = token_id
cur_pos = pos
warmup.times do
  cur_token, _ = ML::GGUF::Qwen35CPU.forward_top1(weights, cur_token, cur_pos, state)
  cur_pos += 1
end

if ENV["QWEN35_METAL_PROFILE"]? == "1"
  ML::GGUF::Qwen35Metal::Profile.reset
  ML::GGUF::Qwen35Metal::Profile.enable!
end

times = Array(Float64).new(runs)
next_id = cur_token
logit = 0.0_f32
runs.times do
  decode_t0 = Time.instant
  next_id, logit = ML::GGUF::Qwen35CPU.forward_top1(weights, cur_token, cur_pos, state)
  times << (Time.instant - decode_t0).total_milliseconds
  cur_token = next_id
  cur_pos += 1
end
if ENV["QWEN35_METAL_PROFILE"]? == "1"
  ML::GGUF::Qwen35Metal::Profile.disable!
end

sorted = times.sort
p50 = sorted[sorted.size // 2]
avg = times.sum / times.size
puts "prepare_state_ms=%.3f" % prep_ms
puts "decode_ms_p50=%.3f avg=%.3f min=%.3f max=%.3f tok_s_p50=%.3f" % [p50, avg, sorted.first, sorted.last, 1000.0 / p50]
puts "decode_ms_samples=#{times.map { |t| "%.3f" % t }.join(",")}"
puts "top1_id=#{next_id} top1_logit=#{logit}"
print ML::GGUF::Qwen35Metal::Profile.report_io if ENV["QWEN35_METAL_PROFILE"]? == "1"
