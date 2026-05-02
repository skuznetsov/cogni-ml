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
mtp_chain_tokens = 0_i32
mtp_chain_topk = 1_i32
mtp_chain_mode = "teacher"
mtp_chain_trace = false
mtp_chain_raw_blends = [] of Float32
mtp_chain_diag_bridge = false
mtp_chain_diag_clamp = 4.0_f32
mtp_chain_diag_ridge = 1.0e-5_f32

struct DiagBridge
  getter scale : Array(Float32)
  getter bias : Array(Float32)
  getter pairs : Int32

  def initialize(@scale, @bias, @pairs)
  end

  def apply(x : Array(Float32)) : Array(Float32)
    raise ArgumentError.new("diag bridge input size #{x.size} != #{@scale.size}") unless x.size == @scale.size
    Array(Float32).new(x.size) { |i| @scale[i] * x[i] + @bias[i] }
  end
end

private def elapsed_ms(start : Time::Instant) : Float64
  (Time.instant - start).total_milliseconds
end

private def mtp_hidden_topk(weights, mtp, prev_hidden, token_id, pos, k, next_hidden_raw, mtp_state)
  start = Time.instant
  next_hidden = ML::GGUF::Qwen35MTP.forward_one_hidden(weights, mtp, prev_hidden, token_id, pos, normalized: !next_hidden_raw, mtp_state: mtp_state)
  project_hidden = if next_hidden_raw
                     ML::GGUF::Qwen35MTP.rms_norm_sidecar(next_hidden, mtp.norm, weights.hparams.rms_eps)
                   else
                     next_hidden
                   end
  topk = if k == 1
           [ML::GGUF::Qwen35MTP.hidden_top1(weights, project_hidden)]
         else
           logits = ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, project_hidden)
           ML::GGUF::Qwen35MTP.top_k(logits, k)
         end
  {next_hidden, topk, elapsed_ms(start)}
end

private def rank_in_topk(topk : Array({Int32, Float32}), target : Int32) : Int32
  topk.each_with_index do |(id, _), i|
    return i + 1 if id == target
  end
  0
end

private def blend_hidden(a : Array(Float32), b : Array(Float32), alpha : Float32) : Array(Float32)
  raise ArgumentError.new("blend hidden size mismatch #{a.size} != #{b.size}") unless a.size == b.size
  Array(Float32).new(a.size) { |i| a[i] + alpha * (b[i] - a[i]) }
end

private def target_hidden_topk(weights, hidden : Array(Float32), k : Int32) : Array({Int32, Float32})
  if k == 1
    [ML::GGUF::Qwen35CPU.hidden_top1(weights, hidden)]
  else
    x = hidden.dup
    ML::GGUF::Qwen35CPU.rms_norm!(x, weights.output_norm, weights.hparams.rms_eps)
    logits = ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
    ML::GGUF::Qwen35MTP.top_k(logits, k)
  end
end

private def prompt_hidden_rows(weights, token_ids : Array(Int32), max_seq : Int32) : Array(Array(Float32))
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)
  Array(Array(Float32)).new(token_ids.size) do |i|
    ML::GGUF::Qwen35CPU.forward_hidden(weights, token_ids[i], i, state)
  end
end

private def train_diag_bridge(weights, mtp, token_ids : Array(Int32), max_seq : Int32,
                              ridge : Float32, clamp : Float32) : DiagBridge?
  return nil if token_ids.size < 2

  hiddens = prompt_hidden_rows(weights, token_ids, max_seq)
  dim = weights.hparams.n_embd
  sum_x = Array(Float64).new(dim, 0.0)
  sum_y = Array(Float64).new(dim, 0.0)
  sum_xx = Array(Float64).new(dim, 0.0)
  sum_xy = Array(Float64).new(dim, 0.0)
  pairs = token_ids.size - 1

  pairs.times do |i|
    raw = ML::GGUF::Qwen35MTP.forward_one_hidden(weights, mtp, hiddens[i], token_ids[i + 1], i + 1, normalized: false)
    y = hiddens[i + 1]
    dim.times do |j|
      xj = raw[j].to_f64
      yj = y[j].to_f64
      sum_x[j] += xj
      sum_y[j] += yj
      sum_xx[j] += xj * xj
      sum_xy[j] += xj * yj
    end
  end

  n = pairs.to_f64
  scale = Array(Float32).new(dim, 1.0_f32)
  bias = Array(Float32).new(dim, 0.0_f32)
  dim.times do |j|
    mean_x = sum_x[j] / n
    mean_y = sum_y[j] / n
    if pairs == 1
      scale[j] = 1.0_f32
      bias[j] = (mean_y - mean_x).to_f32
      next
    end

    var_x = sum_xx[j] / n - mean_x * mean_x
    cov_xy = sum_xy[j] / n - mean_x * mean_y
    s = (cov_xy / (var_x + ridge.to_f64)).clamp(-clamp.to_f64, clamp.to_f64)
    scale[j] = s.to_f32
    bias[j] = (mean_y - s * mean_x).to_f32
  end

  DiagBridge.new(scale, bias, pairs.to_i32)
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
  p.on("--mtp-chain-tokens N", "Run an exact-sequence MTP chain quality probe for N future tokens") { |v| mtp_chain_tokens = v.to_i32 }
  p.on("--mtp-chain-topk K", "MTP chain top-K coverage to measure; K=1 uses fast top1 head") { |v| mtp_chain_topk = v.to_i32 }
  p.on("--mtp-chain-mode MODE", "MTP chain mode: teacher, recursive, recursive_raw, recursive_cached, both, or all") { |v| mtp_chain_mode = v }
  p.on("--mtp-chain-trace", "Print every MTP chain step") { mtp_chain_trace = true }
  p.on("--mtp-chain-raw-blends LIST", "Comma-separated alpha values for recursive pre-norm hidden blend probes") do |v|
    mtp_chain_raw_blends = v.split(",").reject(&.empty?).map(&.to_f32)
  end
  p.on("--mtp-chain-diag-bridge", "Train prompt-local diagonal raw-MTP-hidden -> target-hidden bridge and test recursive_diag") { mtp_chain_diag_bridge = true }
  p.on("--mtp-chain-diag-clamp X", "Clamp diagonal bridge scale to +/-X") { |v| mtp_chain_diag_clamp = v.to_f32 }
  p.on("--mtp-chain-diag-ridge X", "Diagonal bridge ridge term") { |v| mtp_chain_diag_ridge = v.to_f32 }
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
abort "--mtp-chain-tokens must be non-negative" if mtp_chain_tokens < 0
abort "--mtp-chain-topk must be positive" if mtp_chain_topk <= 0
abort "--mtp-chain-diag-clamp must be positive" if mtp_chain_diag_clamp <= 0.0_f32
abort "--mtp-chain-diag-ridge must be non-negative" if mtp_chain_diag_ridge < 0.0_f32
chain_modes = case mtp_chain_mode
              when "teacher"
                ["teacher"]
              when "recursive"
                ["recursive"]
              when "recursive_raw"
                ["recursive_raw"]
              when "recursive_cached"
                ["recursive_cached"]
              when "both"
                ["teacher", "recursive"]
              when "all"
                ["teacher", "recursive", "recursive_raw", "recursive_cached"]
              else
                abort "--mtp-chain-mode must be teacher, recursive, recursive_raw, recursive_cached, both, or all"
              end
chain_runs = [] of {String, Float32?}
chain_modes.each { |mode| chain_runs << {mode, nil} }
mtp_chain_raw_blends.each { |alpha| chain_runs << {"recursive_raw_blend#{alpha}", alpha} }
chain_runs << {"recursive_diag", nil} if mtp_chain_diag_bridge

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
  needed_seq = token_ids.size + (mtp_chain_tokens > 0 ? mtp_chain_tokens + 1 : 2)
  abort "prompt #{label.inspect} needs max_seq >= #{needed_seq}, got #{max_seq}" if needed_seq > max_seq

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

  next unless mtp_chain_tokens > 0

  exact_state = state.fork
  exact_hiddens = [] of Array(Float32)
  exact_nexts = [] of {Int32, Float32}
  exact_prev = y1
  exact_pos = token_ids.size
  exact_chain_start = Time.instant
  mtp_chain_tokens.times do
    exact_hidden = ML::GGUF::Qwen35CPU.forward_hidden(weights, exact_prev, exact_pos, exact_state)
    exact_next, exact_logit = ML::GGUF::Qwen35CPU.hidden_top1(weights, exact_hidden)
    exact_hiddens << exact_hidden
    exact_nexts << {exact_next, exact_logit}
    exact_prev = exact_next
    exact_pos += 1
  end
  exact_hidden_oracle_ms = elapsed_ms(exact_chain_start)
  exact_text = exact_nexts.map { |id, _| tokenizer.decode_single(id) }.join
  puts "mtp_chain_exact label=#{label.inspect} tokens=#{mtp_chain_tokens} start_y1=#{y1} exact_ids=#{exact_nexts.map(&.[0]).join(",")} exact_text=#{exact_text.inspect} exact_hidden_oracle_ms=#{exact_hidden_oracle_ms.round(3)}"

  diag_bridge = nil.as(DiagBridge?)
  if mtp_chain_diag_bridge
    bridge_start = Time.instant
    diag_bridge = train_diag_bridge(weights, mtp, token_ids, max_seq, mtp_chain_diag_ridge, mtp_chain_diag_clamp)
    if bridge = diag_bridge
      puts "mtp_chain_diag_bridge label=#{label.inspect} pairs=#{bridge.pairs} train_ms=#{elapsed_ms(bridge_start).round(3)} ridge=#{mtp_chain_diag_ridge} clamp=#{mtp_chain_diag_clamp}"
    else
      puts "mtp_chain_diag_bridge label=#{label.inspect} pairs=0 skipped=true"
    end
  end

  chain_runs.each do |(mode, blend_alpha)|
    next if mode == "recursive_diag" && diag_bridge.nil?

    chain_prev_hidden = hidden
    chain_prev_token = y1
    chain_pos = token_ids.size
    chain_cache = mode == "recursive_cached" ? ML::GGUF::Qwen35MTP::State.new(mtp_chain_tokens, weights.hparams.head_dim * weights.hparams.n_head_kv) : nil
    chain_ms = [] of Float64
    chain_top1_hits = 0
    chain_topk_hits = 0
    chain_first_miss = -1
    chain_candidates = [] of Int32

    mtp_chain_tokens.times do |i|
      if mode == "recursive_diag"
        step_start = Time.instant
        raw = ML::GGUF::Qwen35MTP.forward_one_hidden(weights, mtp, chain_prev_hidden, chain_prev_token, chain_pos, normalized: false)
        mtp_hidden = diag_bridge.not_nil!.apply(raw)
        mtp_topk = target_hidden_topk(weights, mtp_hidden, mtp_chain_topk)
        step_ms = elapsed_ms(step_start)
      elsif cache = chain_cache
        mtp_hidden, mtp_topk, step_ms = mtp_hidden_topk(weights, mtp, chain_prev_hidden, chain_prev_token, chain_pos, mtp_chain_topk, false, cache)
      else
        raw_next_hidden = mode == "recursive_raw" || !blend_alpha.nil?
        mtp_hidden, mtp_topk, step_ms = mtp_hidden_topk(weights, mtp, chain_prev_hidden, chain_prev_token, chain_pos, mtp_chain_topk, raw_next_hidden, nil)
      end
      chain_ms << step_ms
      candidate = mtp_topk[0][0]
      exact_id = exact_nexts[i][0]
      rank = rank_in_topk(mtp_topk, exact_id)
      top1_hit = candidate == exact_id
      topk_hit = rank > 0
      chain_top1_hits += 1 if top1_hit
      chain_topk_hits += 1 if topk_hit
      chain_first_miss = i if chain_first_miss < 0 && !top1_hit
      chain_candidates << candidate

      if mtp_chain_trace
        puts "mtp_chain_step label=#{label.inspect} mode=#{mode} i=#{i} pos=#{chain_pos} ms=#{step_ms.round(3)} exact=#{exact_id}:#{tokenizer.decode_single(exact_id).inspect} mtp=#{candidate}:#{tokenizer.decode_single(candidate).inspect} top1_hit=#{top1_hit} topk_rank=#{rank}"
      end

      if mode == "teacher"
        chain_prev_hidden = exact_hiddens[i]
        chain_prev_token = exact_id
      elsif alpha = blend_alpha
        chain_prev_hidden = blend_hidden(chain_prev_hidden, mtp_hidden, alpha)
        chain_prev_token = candidate
      else
        chain_prev_hidden = mtp_hidden
        chain_prev_token = candidate
      end
      chain_pos += 1
    end

    sorted_chain_ms = chain_ms.sort
    chain_p50_ms = sorted_chain_ms[sorted_chain_ms.size // 2]
    chain_text = chain_candidates.map { |id| tokenizer.decode_single(id) }.join
    puts "mtp_chain_summary label=#{label.inspect} mode=#{mode} tokens=#{mtp_chain_tokens} topk=#{mtp_chain_topk} top1_hits=#{chain_top1_hits} top1_rate=#{(chain_top1_hits * 100.0 / mtp_chain_tokens).round(2)} topk_hits=#{chain_topk_hits} topk_rate=#{(chain_topk_hits * 100.0 / mtp_chain_tokens).round(2)} first_miss=#{chain_first_miss} mtp_avg_ms=#{(chain_ms.sum / chain_ms.size).round(3)} mtp_min=#{sorted_chain_ms.first.round(3)} mtp_p50=#{chain_p50_ms.round(3)} mtp_max=#{sorted_chain_ms.last.round(3)} exact_hidden_oracle_ms=#{exact_hidden_oracle_ms.round(3)} candidate_ids=#{chain_candidates.join(",")} candidate_text=#{chain_text.inspect}"
  end
end

puts "mtp_suite_summary rows=#{total_rows} top1_hits=#{top1_hits} top1_rate=#{(top1_hits * 100.0 / total_rows).round(2)} top5_or_top1_hits=#{top5_hits} top5_or_top1_rate=#{(top5_hits * 100.0 / total_rows).round(2)} timed_calls=#{timed_calls} avg_mtp_ms=#{(timed_ms / timed_calls).round(3)} backend=#{mtp_backend} top1_only=#{top1_only}"
