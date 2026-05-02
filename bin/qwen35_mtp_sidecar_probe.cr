#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_meta"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_mtp"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL          = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_MTP            = "#{ENV["HOME"]}/.cache/cogni-ml/qwen36_mtp/Qwen3.6-27B-mtp.safetensors"
DEFAULT_LLAMA_TOKENIZE = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = DEFAULT_MODEL
mtp_path = DEFAULT_MTP
llama_tokenize = DEFAULT_LLAMA_TOKENIZE
prompt = "The capital of France is"
suite_prompts = [] of {String, String}
run_forward = false
top1_only = false
max_seq = 128_i32
mtp_warmup = 0_i32
mtp_repeats = 1_i32

private def elapsed_ms(start : Time::Instant) : Float64
  (Time.instant - start).total_milliseconds
end

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_mtp_sidecar_probe [--model GGUF] [--mtp SIDE_SAFETENSORS] [--run-forward]"
  p.on("--model PATH", "Qwen3.6 GGUF target model path") { |v| model_path = v }
  p.on("--mtp PATH", "MTP-only safetensors sidecar path") { |v| mtp_path = v }
  p.on("--llama-tokenize PATH", "llama.cpp llama-tokenize path") { |v| llama_tokenize = v }
  p.on("--prompt TEXT", "Prompt for the MTP acceptance smoke") { |v| prompt = v }
  p.on("--suite-prompt NAME::TEXT", "Add a prompt row to the first-step MTP acceptance suite") do |v|
    name, text = v.split("::", 2)
    abort "--suite-prompt must use NAME::TEXT" if text.nil? || name.empty?
    suite_prompts << {name, text}
  end
  p.on("--max-seq N", "State max sequence length for forward smoke") { |v| max_seq = v.to_i32 }
  p.on("--run-forward", "Run a first-token MTP formula/acceptance smoke") { run_forward = true }
  p.on("--top1-only", "Run the MTP greedy top1 path without full-logits/top5 readback") { top1_only = true }
  p.on("--mtp-warmup N", "Untimed MTP warmup calls after prompt prefill") { |v| mtp_warmup = v.to_i32 }
  p.on("--mtp-repeats N", "Timed MTP repeats after warmup") { |v| mtp_repeats = v.to_i32 }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "model not found: #{model_path}" unless File.exists?(model_path)
abort "MTP sidecar not found: #{mtp_path}" unless File.exists?(mtp_path)

gguf = ML::GGUF::GGUFFile.new(model_path)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
mtp = ML::GGUF::Qwen35MTPWeights.from_safetensors(mtp_path)
mtp.validate_for_qwen35!(hparams)

puts "qwen35_mtp_sidecar_probe: ok"
puts "model=#{model_path}"
puts "mtp=#{mtp_path}"
puts "hparams hidden=#{hparams.n_embd} layers=#{hparams.n_layer} heads=#{hparams.n_head} kv_heads=#{hparams.n_head_kv} head_dim=#{hparams.head_dim} ffn=#{hparams.n_ff}"
puts "mtp_bytes=#{(mtp.total_raw_bytes / 1_048_576.0).round(2)} MiB"
puts "fc=#{mtp.fc.out_dim}x#{mtp.fc.in_dim}"
puts "attn q=#{mtp.q_proj.out_dim}x#{mtp.q_proj.in_dim} k=#{mtp.k_proj.out_dim}x#{mtp.k_proj.in_dim} v=#{mtp.v_proj.out_dim}x#{mtp.v_proj.in_dim} o=#{mtp.o_proj.out_dim}x#{mtp.o_proj.in_dim}"
puts "ffn gate=#{mtp.ffn_gate.out_dim}x#{mtp.ffn_gate.in_dim} up=#{mtp.ffn_up.out_dim}x#{mtp.ffn_up.in_dim} down=#{mtp.ffn_down.out_dim}x#{mtp.ffn_down.in_dim}"

exit unless run_forward
abort "llama-tokenize not found: #{llama_tokenize}" unless File.exists?(llama_tokenize)
abort "--mtp-warmup must be non-negative" if mtp_warmup < 0
abort "--mtp-repeats must be positive" if mtp_repeats <= 0

load_start = Time.instant
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path, llama_tokenize)
puts "load_weights_ms=#{elapsed_ms(load_start).round(3)}"

mtp_backend = ML::GGUF::Qwen35MTP.bf16_backend_label
rows = suite_prompts.empty? ? [{"main", prompt}] : suite_prompts
total_rows = 0
top1_hits = 0
top5_hits = 0
timed_calls = 0
timed_ms = 0.0_f64

rows.each do |label, prompt_text|
  token_ids = tokenizer.encode(prompt_text)
  abort "prompt #{label.inspect} encoded to no tokens" if token_ids.empty?
  abort "prompt #{label.inspect} length #{token_ids.size} exceeds max_seq #{max_seq}" if token_ids.size + 2 > max_seq

  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)

  prefill_start = Time.instant
  hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(weights, token_ids, 0, state)
  y1, y1_logit = ML::GGUF::Qwen35CPU.hidden_top1(weights, hidden)
  prefill_ms = elapsed_ms(prefill_start)

  verify_state = state.fork
  verify_start = Time.instant
  exact_y2, exact_y2_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, y1, token_ids.size, verify_state)
  verify_ms = elapsed_ms(verify_start)

  mtp_warmup.times do
    if top1_only
      ML::GGUF::Qwen35MTP.forward_one_top1(weights, mtp, hidden, y1, token_ids.size)
    else
      logits = ML::GGUF::Qwen35MTP.forward_one_logits(weights, mtp, hidden, y1, token_ids.size)
      ML::GGUF::Qwen35MTP.top_k(logits, 5)
    end
  end

  first_mtp_y2 = 0_i32
  first_mtp_logit = 0.0_f32
  first_top5 = [] of {Int32, Float32}
  row_ms = [] of Float64

  mtp_repeats.times do |repeat_i|
    mtp_start = Time.instant
    if top1_only
      mtp_y2, mtp_y2_logit = ML::GGUF::Qwen35MTP.forward_one_top1(weights, mtp, hidden, y1, token_ids.size)
      mtp_top5 = [] of {Int32, Float32}
    else
      mtp_logits = ML::GGUF::Qwen35MTP.forward_one_logits(weights, mtp, hidden, y1, token_ids.size)
      mtp_top5 = ML::GGUF::Qwen35MTP.top_k(mtp_logits, 5)
      mtp_y2, mtp_y2_logit = mtp_top5[0]
    end
    mtp_ms = elapsed_ms(mtp_start)
    row_ms << mtp_ms
    timed_calls += 1
    timed_ms += mtp_ms

    if repeat_i == 0
      first_mtp_y2 = mtp_y2
      first_mtp_logit = mtp_y2_logit
      first_top5 = mtp_top5
    end

    puts "mtp_repeat label=#{label.inspect} repeat=#{repeat_i} ms=#{mtp_ms.round(3)} y2=#{mtp_y2} text=#{tokenizer.decode_single(mtp_y2).inspect} accepted=#{mtp_y2 == exact_y2}"
  end

  accepted = first_mtp_y2 == exact_y2
  exact_in_top5 = top1_only ? false : first_top5.any? { |id, _| id == exact_y2 }
  total_rows += 1
  top1_hits += 1 if accepted
  top5_hits += 1 if accepted || exact_in_top5

  puts "forward_smoke label=#{label.inspect} prompt_tokens=#{token_ids.size} max_seq=#{max_seq}"
  puts "prompt=#{prompt_text.inspect}"
  puts "token_ids=#{token_ids.join(",")}"
  puts "exact_y1=#{y1} text=#{tokenizer.decode_single(y1).inspect} logit=#{y1_logit}"
  puts "exact_y2=#{exact_y2} text=#{tokenizer.decode_single(exact_y2).inspect} logit=#{exact_y2_logit}"
  puts "mtp_y2=#{first_mtp_y2} text=#{tokenizer.decode_single(first_mtp_y2).inspect} logit=#{first_mtp_logit}"
  puts "accepted=#{accepted}"
  unless top1_only
    puts "exact_in_mtp_top5=#{exact_in_top5}"
    puts "mtp_top5=#{first_top5.map { |id, logit| "#{id}:#{tokenizer.decode_single(id).inspect}:#{logit}" }.join(" | ")}"
  end
  avg_ms = row_ms.sum / row_ms.size
  sorted_ms = row_ms.sort
  p50_ms = sorted_ms[sorted_ms.size // 2]
  puts "timing_ms label=#{label.inspect} prefill=#{prefill_ms.round(3)} exact_verify=#{verify_ms.round(3)} mtp_#{mtp_backend}#{top1_only ? "_top1" : ""}_avg=#{avg_ms.round(3)} min=#{sorted_ms.first.round(3)} p50=#{p50_ms.round(3)} max=#{sorted_ms.last.round(3)} repeats=#{mtp_repeats} warmup=#{mtp_warmup}"
end

puts "mtp_suite_summary rows=#{total_rows} top1_hits=#{top1_hits} top1_rate=#{(top1_hits * 100.0 / total_rows).round(2)} top5_or_top1_hits=#{top5_hits} top5_or_top1_rate=#{(top5_hits * 100.0 / total_rows).round(2)} timed_calls=#{timed_calls} avg_mtp_ms=#{(timed_ms / timed_calls).round(3)} backend=#{mtp_backend} top1_only=#{top1_only}"
