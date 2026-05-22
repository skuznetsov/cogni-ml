require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_mtp"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_NL.gguf"

model = DEFAULT_MODEL
token_id = 0
pos = 0
max_seq = 8
warmup = 0
runs = 1
prepare_state = true
stateful_mtp = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen36_gguf_mtp_probe [--model PATH] [--token-id N] [--pos N] [--warmup N] [--runs N]"
  p.on("--model=PATH", "Qwen3.6 MTP GGUF path") { |v| model = File.expand_path(v) }
  p.on("--token-id=N", "Seed token id for target hidden and MTP token embedding (default: 0)") { |v| token_id = v.to_i }
  p.on("--pos=N", "Target decode position (default: 0)") { |v| pos = v.to_i }
  p.on("--max-seq=N", "Target/MTP state cache length (default: 8)") { |v| max_seq = v.to_i }
  p.on("--warmup=N", "Untimed MTP calls (default: 0)") { |v| warmup = v.to_i }
  p.on("--runs=N", "Measured MTP calls (default: 1)") { |v| runs = v.to_i }
  p.on("--stateful-mtp", "Use the full MTP attention cache path instead of the one-token shortcut") { stateful_mtp = true }
  p.on("--no-prepare-state", "Skip target Metal state preallocation") { prepare_state = false }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

model = File.expand_path(model)
raise "model not found: #{model}" unless File.exists?(model)
raise "--token-id must be non-negative" if token_id < 0
raise "--pos must be non-negative" if pos < 0
raise "--max-seq must be positive" unless max_seq > 0
raise "--warmup must be non-negative" if warmup < 0
raise "--runs must be positive" unless runs > 0
raise "--pos + --warmup + --runs must fit in --max-seq for --stateful-mtp" if stateful_mtp && pos + warmup + runs > max_seq

puts "Qwen3.6 GGUF MTP probe"
puts "model=#{model}"
puts "token_id=#{token_id} pos=#{pos} max_seq=#{max_seq} warmup=#{warmup} runs=#{runs} stateful_mtp=#{stateful_mtp} prepare_state=#{prepare_state}"
puts "metal_available=#{ML::GGUF::Qwen35Metal.available?}"

load_t0 = Time.instant
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
hp = weights.hparams
mtp = ML::GGUF::Qwen35GGUFMTPWeights.from_gguf(model, hp)
load_ms = (Time.instant - load_t0).total_milliseconds
raise "GGUF has no MTP block" unless hp.nextn_predict_layers > 0
raise "--token-id #{token_id} out of range 0...#{weights.token_embd.out_dim}" if token_id >= weights.token_embd.out_dim

puts "hparams raw_layers=#{hp.raw_block_count} target_layers=#{hp.n_layer} nextn_layers=#{hp.nextn_predict_layers} hidden=#{hp.n_embd} ff=#{hp.n_ff} mtp_block=#{mtp.block_index}"
puts "weight_types token_embd=#{weights.token_embd.type} output=#{weights.output.type} eh_proj=#{mtp.nextn_eh_proj_qw.type} attn_q=#{mtp.attn_q_qw.type} ffn_gate=#{mtp.ffn_gate_qw.type}"
puts "mtp_bytes=#{(mtp.total_raw_bytes / 1_048_576.0).round(2)} MiB load_ms=#{"%.3f" % load_ms}"

state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq)
prep_ms = 0.0
if prepare_state
  prep_t0 = Time.instant
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  prep_ms = (Time.instant - prep_t0).total_milliseconds
end

hidden_t0 = Time.instant
prev_hidden = ML::GGUF::Qwen35CPU.forward_hidden(weights, token_id, pos, state)
hidden_ms = (Time.instant - hidden_t0).total_milliseconds

if ENV["QWEN35_METAL_PROFILE"]? == "1"
  ML::GGUF::Qwen35Metal::Profile.reset
  ML::GGUF::Qwen35Metal::Profile.enable!
end

mtp_state = stateful_mtp ? ML::GGUF::Qwen35MTP::State.new(max_seq, hp.head_dim * hp.n_head_kv) : nil
cur_pos = pos
warmup.times do
  ML::GGUF::Qwen35MTP.forward_one_top1_gguf(weights, mtp, prev_hidden, token_id, cur_pos, mtp_state)
  cur_pos += 1 if stateful_mtp
end

times = Array(Float64).new(runs)
top1_id = -1
top1_logit = 0.0_f32
runs.times do
  mtp_t0 = Time.instant
  top1_id, top1_logit = ML::GGUF::Qwen35MTP.forward_one_top1_gguf(weights, mtp, prev_hidden, token_id, cur_pos, mtp_state)
  times << (Time.instant - mtp_t0).total_milliseconds
  cur_pos += 1 if stateful_mtp
end

if ENV["QWEN35_METAL_PROFILE"]? == "1"
  ML::GGUF::Qwen35Metal::Profile.disable!
end

sorted = times.sort
p50 = sorted[sorted.size // 2]
avg = times.sum / times.size
puts "prepare_state_ms=#{"%.3f" % prep_ms} target_hidden_ms=#{"%.3f" % hidden_ms}"
puts "mtp_ms_p50=#{"%.3f" % p50} avg=#{"%.3f" % avg} min=#{"%.3f" % sorted.first} max=#{"%.3f" % sorted.last} tok_s_p50=#{"%.3f" % (1000.0 / p50)}"
puts "mtp_ms_samples=#{times.map { |t| "%.3f" % t }.join(",")}"
puts "mtp_top1_id=#{top1_id} mtp_top1_logit=#{top1_logit}"
print ML::GGUF::Qwen35Metal::Profile.report_io if ENV["QWEN35_METAL_PROFILE"]? == "1"
