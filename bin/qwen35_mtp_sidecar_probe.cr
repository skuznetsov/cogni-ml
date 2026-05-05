#!/usr/bin/env crystal

require "json"
require "option_parser"
require "../src/ml/gguf/qwen35_meta"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_metal"
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
mtp_chain_margin_thresholds = [] of Float64
mtp_spec_gammas = [] of Int32
mtp_spec_wall_gammas = [] of Int32
mtp_spec_wall_stage = 0_i32
mtp_spec_wall_lazy_draft = false
mtp_spec_wall_reject_offramp = 0_i32
mtp_spec_wall_stage_once = false
mtp_spec_wall_stage_bonus = false
mtp_spec_wall_top2_accounting = false
mtp_spec_wall_top2_miss_offramp = false
mtp_spec_wall_promote_top2_margin = nil.as(Float64?)
mtp_spec_wall_router_trace_path = nil.as(String?)
mtp_spec_wall_router_thresholds = [] of Float64
mtp_spec_wall_profile = false
mtp_spec_wall_serial_early_verify = false
mtp_spec_wall_snapshot_cost_probe = false
mtp_draft_state_off = false

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

class ChainAggregate
  property rows : Int32
  property tokens : Int32
  property top1_hits : Int32
  property topk_hits : Int32
  property rank_order_attempts : Int32
  property full_topk_attempts : Int32
  property mtp_ms_sum : Float64
  property exact_hidden_oracle_ms_sum : Float64
  property top1_margin_sum : Float64
  property top1_margin_count : Int32
  property top1_hit_margin_sum : Float64
  property top1_hit_margin_count : Int32
  property top1_miss_margin_sum : Float64
  property top1_miss_margin_count : Int32
  property top1_hit_margin_min : Float64?
  property top1_miss_margin_max : Float64?
  getter rank_hist : Hash(Int32, Int32)

  def initialize
    @rows = 0
    @tokens = 0
    @top1_hits = 0
    @topk_hits = 0
    @rank_order_attempts = 0
    @full_topk_attempts = 0
    @mtp_ms_sum = 0.0
    @exact_hidden_oracle_ms_sum = 0.0
    @top1_margin_sum = 0.0
    @top1_margin_count = 0
    @top1_hit_margin_sum = 0.0
    @top1_hit_margin_count = 0
    @top1_miss_margin_sum = 0.0
    @top1_miss_margin_count = 0
    @top1_hit_margin_min = nil
    @top1_miss_margin_max = nil
    @rank_hist = Hash(Int32, Int32).new(0)
  end
end

class MarginRouterAggregate
  property rows : Int32
  property tokens : Int32
  property top1_hits : Int32
  property selected : Int32
  property selected_hits : Int32
  property false_accepts : Int32

  def initialize
    @rows = 0
    @tokens = 0
    @top1_hits = 0
    @selected = 0
    @selected_hits = 0
    @false_accepts = 0
  end
end

class MtpSpecAggregate
  property rows : Int32
  property tokens : Int32
  property passes : Int32
  property emitted : Int32
  property draft_tokens : Int32
  property accepted : Int32
  property mtp_ms_sum : Float64
  property exact_ms_sum : Float64
  property target_pass_model_ms_sum : Float64

  def initialize
    @rows = 0
    @tokens = 0
    @passes = 0
    @emitted = 0
    @draft_tokens = 0
    @accepted = 0
    @mtp_ms_sum = 0.0
    @exact_ms_sum = 0.0
    @target_pass_model_ms_sum = 0.0
  end
end

class MtpSpecWallAggregate
  property rows : Int32
  property tokens : Int32
  property passes : Int32
  property emitted : Int32
  property draft_tokens : Int32
  property accepted : Int32
  property rejections : Int32
  property parity_ok : Int32
  property verifier_calls : Int32
  property verifier_tokens : Int32
  property replay_tokens : Int32
  property fallback_tokens : Int32
  property snapshot_tokens : Int32
  property top2_checks : Int32
  property top2_rescues : Int32
  property top2_wrong_tail_tokens : Int32
  property top2_replay_tokens : Int32
  property top2_replay_ms_sum : Float64
  property top2_offramp_hits : Int32
  property top2_promotions : Int32
  property top2_promoted_accepted : Int32
  property mtp_ms_sum : Float64
  property verifier_ms_sum : Float64
  property replay_ms_sum : Float64
  property backup_ms_sum : Float64
  property fallback_ms_sum : Float64
  property snapshot_ms_sum : Float64
  property snapshot_modeled_wall_ms_sum : Float64
  property wall_ms_sum : Float64
  property plain_exact_ms_sum : Float64

  def initialize
    @rows = 0
    @tokens = 0
    @passes = 0
    @emitted = 0
    @draft_tokens = 0
    @accepted = 0
    @rejections = 0
    @parity_ok = 0
    @verifier_calls = 0
    @verifier_tokens = 0
    @replay_tokens = 0
    @fallback_tokens = 0
    @snapshot_tokens = 0
    @top2_checks = 0
    @top2_rescues = 0
    @top2_wrong_tail_tokens = 0
    @top2_replay_tokens = 0
    @top2_replay_ms_sum = 0.0
    @top2_offramp_hits = 0
    @top2_promotions = 0
    @top2_promoted_accepted = 0
    @mtp_ms_sum = 0.0
    @verifier_ms_sum = 0.0
    @replay_ms_sum = 0.0
    @backup_ms_sum = 0.0
    @fallback_ms_sum = 0.0
    @snapshot_ms_sum = 0.0
    @snapshot_modeled_wall_ms_sum = 0.0
    @wall_ms_sum = 0.0
    @plain_exact_ms_sum = 0.0
  end
end

class MtpSpecWallRouterPass
  getter pass_index : Int32
  getter start_i : Int32
  getter end_i : Int32
  getter wall_before_ms : Float64
  getter wall_after_ms : Float64
  getter accepted_delta : Int32
  getter rejections_delta : Int32
  getter fallback_delta : Int32
  getter top2_rescue_delta : Int32
  getter top2_offramp_delta : Int32
  getter mtp_delta_ms : Float64
  getter verifier_delta_ms : Float64
  getter replay_delta_ms : Float64
  getter fallback_delta_ms : Float64
  getter mtp_first_top1 : Int32
  getter mtp_first_top2 : Int32
  getter mtp_first_margin : Float64?
  getter mtp_min_margin : Float64?

  def initialize(@pass_index, @start_i, @end_i, @wall_before_ms, @wall_after_ms,
                 @accepted_delta, @rejections_delta, @fallback_delta,
                 @top2_rescue_delta, @top2_offramp_delta,
                 @mtp_delta_ms, @verifier_delta_ms, @replay_delta_ms, @fallback_delta_ms,
                 @mtp_first_top1, @mtp_first_top2, @mtp_first_margin, @mtp_min_margin)
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
  topk = case k
         when 1
           [ML::GGUF::Qwen35MTP.hidden_top1(weights, project_hidden)]
         when 2
           ML::GGUF::Qwen35MTP.hidden_top2(weights, project_hidden)
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

private def rank_hist_string(hist : Hash(Int32, Int32)) : String
  parts = [] of String
  hist.keys.sort.each do |rank|
    label = rank == 0 ? "miss" : rank.to_s
    parts << "#{label}:#{hist[rank]}"
  end
  parts.empty? ? "none" : parts.join(",")
end

private def choose_mtp_wall_candidate(topk : Array({Int32, Float32}), promote_margin : Float64?) : {Int32, Int32, Bool}
  top1_id, top1_logit = topk[0]
  return {top1_id, -1, false} if topk.size < 2

  top2_id, top2_logit = topk[1]
  if threshold = promote_margin
    margin = top1_logit.to_f64 - top2_logit.to_f64
    return {top2_id, top1_id, true} if margin <= threshold
  end

  {top1_id, top2_id, false}
end

private def fmt3(v : Float64?) : String
  v.nil? ? "na" : v.round(3).to_s
end

private def pct(num : Int32, den : Int32) : Float64
  return 0.0 if den == 0
  num * 100.0 / den
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

private def target_hidden_top2(weights, hidden : Array(Float32)) : {Int32, Float32, Int32, Float32}
  x = hidden.dup
  ML::GGUF::Qwen35CPU.rms_norm!(x, weights.output_norm, weights.hparams.rms_eps)
  if top2 = ML::GGUF::Qwen35Metal.project_top2_no_norm(weights.output, x)
    return {top2[0].to_i32, top2[1], top2[2].to_i32, top2[3]}
  end

  logits = ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
  topk = ML::GGUF::Qwen35MTP.top_k(logits, 2)
  {topk[0][0], topk[0][1], topk[1][0], topk[1][1]}
end

private def write_router_trace(io : IO, label : String, gamma : Int32, pass : MtpSpecWallRouterPass,
                               target_top2 : {Int32, Float32, Int32, Float32},
                               plain_suffix_ms : Float64)
  top1_id, top1_logit, top2_id, top2_logit = target_top2
  mtp_target_rank = if pass.mtp_first_top1 == top1_id
                      1
                    elsif pass.mtp_first_top2 == top1_id
                      2
                    else
                      0
                    end
  JSON.build(io) do |json|
    json.object do
      json.field "kind", "mtp_wall_router_pass"
      json.field "label", label
      json.field "gamma", gamma
      json.field "pass", pass.pass_index
      json.field "start_i", pass.start_i
      json.field "end_i", pass.end_i
      json.field "target_top1", top1_id
      json.field "target_top2", top2_id
      json.field "target_margin", (top1_logit - top2_logit).to_f64
      json.field "mtp_first_top1", pass.mtp_first_top1
      json.field "mtp_first_top2", pass.mtp_first_top2
      json.field "mtp_first_target_rank", mtp_target_rank
      json.field "mtp_first_margin", pass.mtp_first_margin
      json.field "mtp_min_margin", pass.mtp_min_margin
      json.field "wall_before_ms", pass.wall_before_ms
      json.field "wall_after_ms", pass.wall_after_ms
      json.field "plain_suffix_ms", plain_suffix_ms
      json.field "accepted_delta", pass.accepted_delta
      json.field "rejections_delta", pass.rejections_delta
      json.field "fallback_delta", pass.fallback_delta
      json.field "top2_rescue_delta", pass.top2_rescue_delta
      json.field "top2_offramp_delta", pass.top2_offramp_delta
      json.field "mtp_delta_ms", pass.mtp_delta_ms
      json.field "verifier_delta_ms", pass.verifier_delta_ms
      json.field "replay_delta_ms", pass.replay_delta_ms
      json.field "fallback_delta_ms", pass.fallback_delta_ms
    end
  end
  io << '\n'
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
  p.on("--mtp-chain-margin-thresholds LIST", "Comma-separated MTP top1/top2 margin thresholds for top1 confidence-router attribution") do |v|
    mtp_chain_margin_thresholds = v.split(",").reject(&.empty?).map(&.to_f64)
  end
  p.on("--mtp-spec-gammas LIST", "Comma-separated vLLM-style MTP speculative accounting gammas; requires --mtp-chain-tokens") do |v|
    mtp_spec_gammas = v.split(",").reject(&.empty?).map(&.to_i32)
  end
  p.on("--mtp-spec-wall-gammas LIST", "Comma-separated real wall-clock exact-resync MTP verifier gammas; requires --mtp-chain-tokens") do |v|
    mtp_spec_wall_gammas = v.split(",").reject(&.empty?).map(&.to_i32)
  end
  p.on("--mtp-spec-wall-stage N", "Verify MTP wall proposals in N-candidate stages; 0 verifies the full proposal in one chunk") { |v| mtp_spec_wall_stage = v.to_i32 }
  p.on("--mtp-spec-wall-stage-once", "Use --mtp-spec-wall-stage as a first guard, then verify the remaining proposal tail in one chunk") { mtp_spec_wall_stage_once = true }
  p.on("--mtp-spec-wall-stage-bonus", "After an accepted staged MTP verifier chunk, emit one exact verifier bonus token and restart from that exact boundary") { mtp_spec_wall_stage_bonus = true }
  p.on("--mtp-spec-wall-top2-accounting", "Track whether MTP top2 would cover exact reject corrections inside the wall loop") { mtp_spec_wall_top2_accounting = true }
  p.on("--mtp-spec-wall-top2-miss-offramp", "After a reject whose correction is not MTP top2, finish remaining tokens with exact greedy target decode") { mtp_spec_wall_top2_miss_offramp = true }
  p.on("--mtp-spec-wall-promote-top2-margin F", "Use MTP top2 as the verifier candidate when top1-top2 margin is <= F") { |v| mtp_spec_wall_promote_top2_margin = v.to_f64 }
  p.on("--mtp-spec-wall-router-trace PATH", "Write JSONL pass records with target top2 margin and submit/skip oracle inputs") { |v| mtp_spec_wall_router_trace_path = v }
  p.on("--mtp-spec-wall-router-thresholds LIST", "Comma-separated target-margin thresholds for an offline skip-MTP oracle") do |v|
    mtp_spec_wall_router_thresholds = v.split(",").reject(&.empty?).map(&.to_f64)
  end
  p.on("--mtp-spec-wall-lazy-draft", "Only draft the next staged MTP verifier chunk; avoids generating wrong tails after early rejection") { mtp_spec_wall_lazy_draft = true }
  p.on("--mtp-spec-wall-reject-offramp N", "After N consecutive MTP wall rejects, finish the remaining tokens with exact greedy target decode; 0 disables") { |v| mtp_spec_wall_reject_offramp = v.to_i32 }
  p.on("--mtp-spec-wall-profile", "Profile only target verifier calls inside the exact-resync MTP wall loop") { mtp_spec_wall_profile = true }
  p.on("--mtp-spec-wall-serial-early-verify", "Verify MTP wall stages token-by-token and stop at the first candidate mismatch") { mtp_spec_wall_serial_early_verify = true }
  p.on("--mtp-spec-wall-snapshot-cost-probe", "After MTP wall timing, simulate recurrent branch-state snapshot blit cost for every verified row") { mtp_spec_wall_snapshot_cost_probe = true }
  p.on("--mtp-draft-state-off", "Use stateless one-token MTP proposals inside wall/spec probes (proposal-only; target verifier remains exact)") { mtp_draft_state_off = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

mtp_spec_wall_top2_accounting = true if mtp_spec_wall_top2_miss_offramp || mtp_spec_wall_promote_top2_margin

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
mtp_chain_margin_thresholds = mtp_chain_margin_thresholds.sort.uniq
abort "--mtp-chain-margin-thresholds requires --mtp-chain-topk >= 2" if !mtp_chain_margin_thresholds.empty? && mtp_chain_topk < 2
mtp_spec_gammas = mtp_spec_gammas.sort.uniq
abort "--mtp-spec-gammas requires --mtp-chain-tokens > 0" if !mtp_spec_gammas.empty? && mtp_chain_tokens <= 0
abort "--mtp-spec-gammas values must be positive" if mtp_spec_gammas.any? { |gamma| gamma <= 0 }
mtp_spec_wall_gammas = mtp_spec_wall_gammas.sort.uniq
abort "--mtp-spec-wall-gammas requires --mtp-chain-tokens > 0" if !mtp_spec_wall_gammas.empty? && mtp_chain_tokens <= 0
abort "--mtp-spec-wall-gammas values must be positive" if mtp_spec_wall_gammas.any? { |gamma| gamma <= 0 }
abort "--mtp-spec-wall-stage must be non-negative" if mtp_spec_wall_stage < 0
abort "--mtp-spec-wall-lazy-draft requires --mtp-spec-wall-stage > 0" if mtp_spec_wall_lazy_draft && mtp_spec_wall_stage <= 0
abort "--mtp-spec-wall-stage-once requires --mtp-spec-wall-stage > 0" if mtp_spec_wall_stage_once && mtp_spec_wall_stage <= 0
abort "--mtp-spec-wall-reject-offramp must be non-negative" if mtp_spec_wall_reject_offramp < 0
mtp_spec_wall_router_thresholds = mtp_spec_wall_router_thresholds.sort.uniq
abort "--mtp-spec-wall-router-thresholds requires --mtp-spec-wall-gammas" if !mtp_spec_wall_router_thresholds.empty? && mtp_spec_wall_gammas.empty?
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
chain_aggregates = Hash(String, ChainAggregate).new { |hash, key| hash[key] = ChainAggregate.new }
margin_router_aggregates = Hash(String, Array(MarginRouterAggregate)).new do |hash, key|
  hash[key] = Array(MarginRouterAggregate).new(mtp_chain_margin_thresholds.size) { MarginRouterAggregate.new }
end
mtp_spec_aggregates = Hash(Int32, MtpSpecAggregate).new { |hash, key| hash[key] = MtpSpecAggregate.new }
mtp_spec_wall_aggregates = Hash(Int32, MtpSpecWallAggregate).new { |hash, key| hash[key] = MtpSpecWallAggregate.new }
router_trace_io = mtp_spec_wall_router_trace_path.try { |path| File.open(path, "w") }
target_margin_cache = {} of Int32 => {Int32, Float32, Int32, Float32}

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

  plain_state = state.fork
  plain_prev = y1
  plain_pos = token_ids.size
  plain_ids = [] of Int32
  plain_token_ms = [] of Float64
  plain_start = Time.instant
  mtp_chain_tokens.times do
    token_start = Time.instant
    id, _ = ML::GGUF::Qwen35CPU.forward_top1(weights, plain_prev, plain_pos, plain_state)
    plain_token_ms << elapsed_ms(token_start)
    plain_ids << id
    plain_prev = id
    plain_pos += 1
  end
  plain_exact_ms = elapsed_ms(plain_start)
  plain_suffix_ms = Array(Float64).new(mtp_chain_tokens + 1, 0.0_f64)
  (mtp_chain_tokens - 1).downto(0) do |i|
    plain_suffix_ms[i] = plain_suffix_ms[i + 1] + plain_token_ms[i]
  end
  exact_ids = exact_nexts.map(&.[0])
  raise "plain exact ids mismatch" unless plain_ids == exact_ids
  puts "mtp_plain_exact label=#{label.inspect} tokens=#{mtp_chain_tokens} plain_exact_ms=#{plain_exact_ms.round(3)} ids=#{plain_ids.join(",")}"

  mtp_spec_gammas.each do |gamma|
    spec_i = 0
    spec_passes = 0
    spec_emitted = 0
    spec_draft_tokens = 0
    spec_accepted = 0
    spec_mtp_ms = 0.0_f64

    while spec_i < mtp_chain_tokens
      spec_passes += 1
      prev_hidden = spec_i == 0 ? hidden : exact_hiddens[spec_i - 1]
      prev_token = spec_i == 0 ? y1 : exact_nexts[spec_i - 1][0]
      segment_pos = token_ids.size + spec_i
      segment_state = gamma > 1 ? ML::GGUF::Qwen35MTP::State.new(gamma, weights.hparams.head_dim * weights.hparams.n_head_kv) : nil
      accepted_this_pass = 0

      gamma.times do |j|
        exact_i = spec_i + j
        break if exact_i >= mtp_chain_tokens

        mtp_hidden, mtp_top1, step_ms = mtp_hidden_topk(weights, mtp, prev_hidden, prev_token, segment_pos + j, 1, false, segment_state)
        spec_draft_tokens += 1
        spec_mtp_ms += step_ms

        candidate = mtp_top1[0][0]
        break unless candidate == exact_nexts[exact_i][0]

        spec_accepted += 1
        accepted_this_pass += 1
        prev_hidden = mtp_hidden
        prev_token = candidate
      end

      # Greedy exact speculative decoding emits the accepted prefix plus either
      # the corrected mismatch token or the target bonus token.
      emitted_this_pass = accepted_this_pass + 1
      remaining = mtp_chain_tokens - spec_i
      emitted_this_pass = remaining if emitted_this_pass > remaining
      spec_emitted += emitted_this_pass
      spec_i += emitted_this_pass
    end

    exact_avg_ms = exact_hidden_oracle_ms / mtp_chain_tokens
    target_pass_model_ms = exact_avg_ms * spec_passes
    additive_wall_ms = target_pass_model_ms + spec_mtp_ms
    overlap_wall_ms = target_pass_model_ms > spec_mtp_ms ? target_pass_model_ms : spec_mtp_ms
    target_pass_speedup_bound = mtp_chain_tokens.to_f64 / spec_passes

    agg = mtp_spec_aggregates[gamma]
    agg.rows += 1
    agg.tokens += mtp_chain_tokens
    agg.passes += spec_passes
    agg.emitted += spec_emitted
    agg.draft_tokens += spec_draft_tokens
    agg.accepted += spec_accepted
    agg.mtp_ms_sum += spec_mtp_ms
    agg.exact_ms_sum += exact_hidden_oracle_ms
    agg.target_pass_model_ms_sum += target_pass_model_ms

    puts "mtp_spec_summary label=#{label.inspect} mode=exact_resync gamma=#{gamma} tokens=#{mtp_chain_tokens} passes=#{spec_passes} emitted=#{spec_emitted} draft_tokens=#{spec_draft_tokens} accepted=#{spec_accepted} accept_rate=#{pct(spec_accepted, spec_draft_tokens).round(2)} tokens_per_pass=#{(spec_emitted.to_f64 / spec_passes).round(3)} target_pass_speedup_bound=#{target_pass_speedup_bound.round(3)} mtp_ms=#{spec_mtp_ms.round(3)} exact_hidden_oracle_ms=#{exact_hidden_oracle_ms.round(3)} target_pass_model_ms=#{target_pass_model_ms.round(3)} additive_wall_model_ms=#{additive_wall_ms.round(3)} additive_speedup_model=#{(exact_hidden_oracle_ms / additive_wall_ms).round(3)} ideal_overlap_wall_model_ms=#{overlap_wall_ms.round(3)} ideal_overlap_speedup_model=#{(exact_hidden_oracle_ms / overlap_wall_ms).round(3)}"
  end

  mtp_spec_wall_gammas.each do |gamma|
    wall_state = state.fork
    wall_hidden = hidden
    wall_token = y1
    wall_pos = token_ids.size
    wall_ids = [] of Int32
    wall_passes = 0
    wall_draft_tokens = 0
    wall_accepted = 0
    wall_rejections = 0
    wall_verifier_calls = 0
    wall_verifier_tokens = 0
    wall_replay_tokens = 0
    wall_fallback_tokens = 0
    wall_mtp_ms = 0.0_f64
    wall_verifier_ms = 0.0_f64
    wall_replay_ms = 0.0_f64
    wall_backup_ms = 0.0_f64
    wall_fallback_ms = 0.0_f64
    wall_snapshot_ms = 0.0_f64
    wall_snapshot_tokens = 0
    wall_top2_checks = 0
    wall_top2_rescues = 0
    wall_top2_wrong_tail_tokens = 0
    wall_top2_replay_tokens = 0
    wall_top2_replay_ms = 0.0_f64
    wall_top2_offramp_hits = 0
    wall_top2_promotions = 0
    wall_top2_promoted_accepted = 0
    snapshot_cost_states = [] of ML::GGUF::Qwen35CPU::State
    if mtp_spec_wall_snapshot_cost_probe
      (gamma + 1).times do
        snapshot_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq)
        ML::GGUF::Qwen35CPU.prepare_state_metal!(snapshot_state, weights.hparams)
        snapshot_cost_states << snapshot_state
      end
    end
    wall_start = Time.instant
    if mtp_spec_wall_profile
      ML::GGUF::Qwen35Metal::Profile.reset
    end
    backup_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(backup_state, weights.hparams)
    consecutive_rejections = 0
    router_records = [] of MtpSpecWallRouterPass

    while wall_ids.size < mtp_chain_tokens
      pass_wall_before_ms = elapsed_ms(wall_start)
      pass_start_i = wall_ids.size
      pass_accepted_before = wall_accepted
      pass_rejections_before = wall_rejections
      pass_fallback_before = wall_fallback_tokens
      pass_top2_rescues_before = wall_top2_rescues
      pass_top2_offramp_before = wall_top2_offramp_hits
      pass_mtp_before = wall_mtp_ms
      pass_verifier_before = wall_verifier_ms
      pass_replay_before = wall_replay_ms
      pass_fallback_ms_before = wall_fallback_ms
      pass_mtp_first_top1 = -1
      pass_mtp_first_top2 = -1
      pass_mtp_first_margin = nil.as(Float64?)
      pass_mtp_min_margin = nil.as(Float64?)
      wall_passes += 1
      remaining = mtp_chain_tokens - wall_ids.size
      draft_steps = Math.min(gamma, remaining)
      candidates = [] of Int32
      candidate_seconds = [] of Int32
      candidate_promoted = [] of Bool
      draft_hidden = wall_hidden
      draft_token = wall_token
      draft_pos = wall_pos
      stateless_mtp_draft = mtp_draft_state_off || ENV["QWEN35_MTP_DRAFT_STATE_OFF"]? == "1"
      draft_state = (!stateless_mtp_draft && draft_steps > 1) ? ML::GGUF::Qwen35MTP::State.new(draft_steps, weights.hparams.head_dim * weights.hparams.n_head_kv) : nil
      draft_generated = 0

      unless mtp_spec_wall_lazy_draft
        draft_steps.times do
          mtp_hidden, mtp_topk, step_ms = mtp_hidden_topk(weights, mtp, draft_hidden, draft_token, draft_pos, mtp_spec_wall_top2_accounting ? 2 : 1, false, draft_state)
          if mtp_topk.size > 0 && pass_mtp_first_top1 < 0
            pass_mtp_first_top1 = mtp_topk[0][0]
          end
          if mtp_topk.size > 1
            margin = (mtp_topk[0][1] - mtp_topk[1][1]).to_f64
            if pass_mtp_first_margin.nil?
              pass_mtp_first_top1 = mtp_topk[0][0]
              pass_mtp_first_top2 = mtp_topk[1][0]
              pass_mtp_first_margin = margin
            end
            pass_mtp_min_margin = pass_mtp_min_margin ? Math.min(pass_mtp_min_margin.not_nil!, margin) : margin
          end
          candidate, second_candidate, promoted = choose_mtp_wall_candidate(mtp_topk, mtp_spec_wall_promote_top2_margin)
          candidates << candidate
          candidate_seconds << second_candidate
          candidate_promoted << promoted
          wall_top2_promotions += 1 if promoted
          wall_draft_tokens += 1
          wall_mtp_ms += step_ms
          draft_hidden = mtp_hidden
          draft_token = candidate
          draft_pos += 1
        end
      end

      candidate_offset = 0
      pass_rejected = false
      pass_finished = false
      stage_token = wall_token
      stage_hidden = wall_hidden
      stage_pos = wall_pos
      stage_size = mtp_spec_wall_stage > 0 ? mtp_spec_wall_stage : draft_steps

      while !pass_rejected && !pass_finished
        if mtp_spec_wall_lazy_draft
          break if draft_generated >= draft_steps

          current_stage = if mtp_spec_wall_stage_once && draft_generated > 0
                            draft_steps - draft_generated
                          else
                            Math.min(stage_size, draft_steps - draft_generated)
                          end
          stage_candidates = [] of Int32
          stage_second_candidates = [] of Int32
          stage_promoted_candidates = [] of Bool
          current_stage.times do
            mtp_hidden, mtp_topk, step_ms = mtp_hidden_topk(weights, mtp, draft_hidden, draft_token, draft_pos, mtp_spec_wall_top2_accounting ? 2 : 1, false, draft_state)
            if mtp_topk.size > 0 && pass_mtp_first_top1 < 0
              pass_mtp_first_top1 = mtp_topk[0][0]
            end
            if mtp_topk.size > 1
              margin = (mtp_topk[0][1] - mtp_topk[1][1]).to_f64
              if pass_mtp_first_margin.nil?
                pass_mtp_first_top1 = mtp_topk[0][0]
                pass_mtp_first_top2 = mtp_topk[1][0]
                pass_mtp_first_margin = margin
              end
              pass_mtp_min_margin = pass_mtp_min_margin ? Math.min(pass_mtp_min_margin.not_nil!, margin) : margin
            end
            candidate, second_candidate, promoted = choose_mtp_wall_candidate(mtp_topk, mtp_spec_wall_promote_top2_margin)
            stage_candidates << candidate
            stage_second_candidates << second_candidate
            stage_promoted_candidates << promoted
            wall_top2_promotions += 1 if promoted
            wall_draft_tokens += 1
            wall_mtp_ms += step_ms
            draft_hidden = mtp_hidden
            draft_token = candidate
            draft_pos += 1
            draft_generated += 1
          end
          final_stage = draft_generated >= draft_steps
        else
          break if candidate_offset >= candidates.size

          current_stage = if mtp_spec_wall_stage_once && candidate_offset > 0
                            candidates.size - candidate_offset
                          else
                            Math.min(stage_size, candidates.size - candidate_offset)
                          end
          stage_candidates = candidates[candidate_offset, current_stage]
          stage_second_candidates = candidate_seconds[candidate_offset, current_stage]
          stage_promoted_candidates = candidate_promoted[candidate_offset, current_stage]
          final_stage = candidate_offset + current_stage >= candidates.size
        end
        need_bonus = (final_stage || mtp_spec_wall_stage_bonus) && wall_ids.size + current_stage < mtp_chain_tokens
        verify_tokens = [stage_token] + (need_bonus ? stage_candidates : stage_candidates[0, Math.max(current_stage - 1, 0)])
        backup_start = Time.instant
        ML::GGUF::Qwen35CPU.copy_state_metal_used!(backup_state, wall_state, weights.hparams, used_tokens: stage_pos)
        wall_backup_ms += elapsed_ms(backup_start)

        verifier_start = Time.instant
        if mtp_spec_wall_profile
          ML::GGUF::Qwen35Metal::Profile.enable!
        end
        verified = begin
          if mtp_spec_wall_serial_early_verify
            serial_hidden_rows = [] of Float32
            serial_top1s = [] of {Int32, Float32}
            rows_to_verify = stage_candidates.size + (need_bonus ? 1 : 0)
            rows_to_verify.times do |i|
              token_id = i == 0 ? stage_token : stage_candidates[i - 1]
              row = ML::GGUF::Qwen35CPU.forward_hidden(weights, token_id, stage_pos + i, wall_state)
              serial_hidden_rows.concat(row)
              top1 = ML::GGUF::Qwen35CPU.hidden_top1(weights, row)
              serial_top1s << top1
              break if i < stage_candidates.size && top1[0] != stage_candidates[i]
            end
            {hidden: serial_hidden_rows, top1s: serial_top1s, count: serial_top1s.size}
          else
            result = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(weights, verify_tokens, stage_pos, wall_state)
            {hidden: result[:hidden], top1s: result[:top1s], count: verify_tokens.size}
          end
        ensure
          ML::GGUF::Qwen35Metal::Profile.disable! if mtp_spec_wall_profile
        end
        wall_verifier_ms += elapsed_ms(verifier_start)
        wall_verifier_calls += 1
        verified_token_count = verified[:count]
        wall_verifier_tokens += verified_token_count
        top1s = verified[:top1s]
        hidden_rows = verified[:hidden]

        accepted_stage = 0
        stage_candidates.each_with_index do |candidate, i|
          break if i >= top1s.size
          break unless candidate == top1s[i][0]
          accepted_stage += 1
        end
        wall_accepted += accepted_stage
        wall_top2_promoted_accepted += stage_promoted_candidates[0, accepted_stage].count { |promoted| promoted }

        if accepted_stage == stage_candidates.size
          consecutive_rejections = 0
          stage_candidates.each do |id|
            break if wall_ids.size >= mtp_chain_tokens
            wall_ids << id
          end

          if need_bonus && wall_ids.size < mtp_chain_tokens
            bonus = top1s[accepted_stage][0]
            row_base = accepted_stage * weights.hparams.n_embd
            wall_hidden = hidden_rows[row_base, weights.hparams.n_embd]
            wall_token = bonus
            wall_pos = stage_pos + accepted_stage + 1
            wall_ids << bonus
            pass_finished = true if mtp_spec_wall_stage_bonus
          elsif final_stage
            # Final requested token accepted; no next boundary is needed.
            wall_hidden = stage_hidden
            wall_token = stage_token
            wall_pos = stage_pos
          else
            row_base = (accepted_stage - 1) * weights.hparams.n_embd
            stage_hidden = hidden_rows[row_base, weights.hparams.n_embd]
            stage_token = stage_candidates[-1]
            stage_pos += accepted_stage
            wall_hidden = stage_hidden
            wall_token = stage_token
            wall_pos = stage_pos
          end

          candidate_offset += current_stage
        else
          pass_rejected = true
          wall_rejections += 1
          consecutive_rejections += 1
          correction = top1s[accepted_stage][0]
          top2_rescued_rejection = false
          if mtp_spec_wall_top2_accounting
            second_candidate = accepted_stage < stage_second_candidates.size ? stage_second_candidates[accepted_stage] : -1
            if second_candidate >= 0
              wall_top2_checks += 1
              if second_candidate == correction
                top2_rescued_rejection = true
                wall_top2_rescues += 1
                wall_top2_wrong_tail_tokens += Math.max(verified_token_count - accepted_stage - 1, 0)
              end
            end
          end
          stage_candidates[0, accepted_stage].each do |id|
            break if wall_ids.size >= mtp_chain_tokens
            wall_ids << id
          end

          if verified_token_count == accepted_stage + 1
            # The verifier stopped exactly at the correction boundary; keep
            # the mutated state and reuse its boundary hidden instead of
            # restoring and replaying the accepted prefix.
            row_base = accepted_stage * weights.hparams.n_embd
            wall_hidden = hidden_rows[row_base, weights.hparams.n_embd]
          else
            restore_start = Time.instant
            ML::GGUF::Qwen35CPU.copy_state_metal_used!(wall_state, backup_state, weights.hparams, used_tokens: stage_pos)
            wall_backup_ms += elapsed_ms(restore_start)
            replay_tokens = verify_tokens[0, accepted_stage + 1]
            replay_start = Time.instant
            wall_hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(weights, replay_tokens, stage_pos, wall_state)
            replay_ms = elapsed_ms(replay_start)
            wall_replay_ms += replay_ms
            wall_replay_tokens += replay_tokens.size
            if top2_rescued_rejection
              wall_top2_replay_tokens += replay_tokens.size
              wall_top2_replay_ms += replay_ms
            end
          end
          wall_token = correction
          wall_pos = stage_pos + accepted_stage + 1
          wall_ids << correction if wall_ids.size < mtp_chain_tokens

          if mtp_spec_wall_top2_miss_offramp &&
             !top2_rescued_rejection &&
             wall_ids.size < mtp_chain_tokens
            wall_top2_offramp_hits += 1
            fallback_start = Time.instant
            while wall_ids.size < mtp_chain_tokens
              next_id, _next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, wall_token, wall_pos, wall_state)
              wall_ids << next_id
              wall_fallback_tokens += 1
              wall_token = next_id
              wall_pos += 1
            end
            wall_fallback_ms += elapsed_ms(fallback_start)
          elsif mtp_spec_wall_reject_offramp > 0 &&
                consecutive_rejections >= mtp_spec_wall_reject_offramp &&
                wall_ids.size < mtp_chain_tokens
            fallback_start = Time.instant
            while wall_ids.size < mtp_chain_tokens
              next_id, _next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, wall_token, wall_pos, wall_state)
              wall_ids << next_id
              wall_fallback_tokens += 1
              wall_token = next_id
              wall_pos += 1
            end
            wall_fallback_ms += elapsed_ms(fallback_start)
          end
        end
      end

      router_records << MtpSpecWallRouterPass.new(
        wall_passes,
        pass_start_i,
        wall_ids.size,
        pass_wall_before_ms,
        elapsed_ms(wall_start),
        wall_accepted - pass_accepted_before,
        wall_rejections - pass_rejections_before,
        wall_fallback_tokens - pass_fallback_before,
        wall_top2_rescues - pass_top2_rescues_before,
        wall_top2_offramp_hits - pass_top2_offramp_before,
        wall_mtp_ms - pass_mtp_before,
        wall_verifier_ms - pass_verifier_before,
        wall_replay_ms - pass_replay_before,
        wall_fallback_ms - pass_fallback_ms_before,
        pass_mtp_first_top1,
        pass_mtp_first_top2,
        pass_mtp_first_margin,
        pass_mtp_min_margin)
    end

    wall_ms = elapsed_ms(wall_start)
    if mtp_spec_wall_snapshot_cost_probe && wall_verifier_tokens > 0
      snapshot_start = Time.instant
      wall_verifier_tokens.times do |i|
        ML::GGUF::Qwen35CPU.copy_state_metal_used!(
          snapshot_cost_states[i % snapshot_cost_states.size],
          wall_state,
          weights.hparams,
          rec_only: true)
      end
      wall_snapshot_ms = elapsed_ms(snapshot_start)
      wall_snapshot_tokens = wall_verifier_tokens
    end
    parity = wall_ids == exact_ids
    unless parity
      puts "mtp_spec_wall_mismatch label=#{label.inspect} gamma=#{gamma} stage=#{mtp_spec_wall_stage} expected=#{exact_ids.join(",")} actual=#{wall_ids.join(",")}"
    end
    raise "mtp spec wall ids mismatch for #{label} gamma #{gamma}" unless parity

    if router_trace_io || !mtp_spec_wall_router_thresholds.empty?
      target_margin_cache.clear
      get_target_top2 = ->(idx : Int32) do
        cached = target_margin_cache[idx]?
        unless cached
          cached = target_hidden_top2(weights, exact_hiddens[idx])
          target_margin_cache[idx] = cached
        end
        cached
      end

      router_records.each do |record|
        next if record.start_i >= mtp_chain_tokens
        top2 = get_target_top2.call(record.start_i)
        if io = router_trace_io
          write_router_trace(io, label, gamma, record, top2, plain_suffix_ms[record.start_i])
        end
      end

      mtp_spec_wall_router_thresholds.each do |threshold|
        skip_record = router_records.find do |record|
          next false if record.start_i >= mtp_chain_tokens
          top1_id, top1_logit, top2_id, top2_logit = get_target_top2.call(record.start_i)
          (top1_logit - top2_logit).to_f64 < threshold
        end
        modeled_wall_ms = if skip_record
                            skip_record.wall_before_ms + plain_suffix_ms[skip_record.start_i]
                          else
                            wall_ms
                          end
        skip_i = skip_record ? skip_record.start_i : -1
        skip_pass = skip_record ? skip_record.pass_index : -1
        skipped_tokens = skip_record ? (mtp_chain_tokens - skip_record.start_i) : 0
        puts "mtp_spec_wall_router_oracle label=#{label.inspect} gamma=#{gamma} threshold=#{threshold} skipped=#{!!skip_record} skip_pass=#{skip_pass} skip_i=#{skip_i} skipped_tokens=#{skipped_tokens} actual_wall_ms=#{wall_ms.round(3)} modeled_wall_ms=#{modeled_wall_ms.round(3)} plain_exact_ms=#{plain_exact_ms.round(3)} actual_plain_speedup=#{(plain_exact_ms / wall_ms).round(3)} modeled_plain_speedup=#{(plain_exact_ms / modeled_wall_ms).round(3)}"
      end
    end

    agg = mtp_spec_wall_aggregates[gamma]
    agg.rows += 1
    agg.tokens += mtp_chain_tokens
    agg.passes += wall_passes
    agg.emitted += wall_ids.size
    agg.draft_tokens += wall_draft_tokens
    agg.accepted += wall_accepted
    agg.rejections += wall_rejections
    agg.parity_ok += 1 if parity
    agg.verifier_calls += wall_verifier_calls
    agg.verifier_tokens += wall_verifier_tokens
    agg.replay_tokens += wall_replay_tokens
    agg.fallback_tokens += wall_fallback_tokens
    agg.snapshot_tokens += wall_snapshot_tokens
    agg.top2_checks += wall_top2_checks
    agg.top2_rescues += wall_top2_rescues
    agg.top2_wrong_tail_tokens += wall_top2_wrong_tail_tokens
    agg.top2_replay_tokens += wall_top2_replay_tokens
    agg.top2_replay_ms_sum += wall_top2_replay_ms
    agg.top2_offramp_hits += wall_top2_offramp_hits
    agg.top2_promotions += wall_top2_promotions
    agg.top2_promoted_accepted += wall_top2_promoted_accepted
    agg.mtp_ms_sum += wall_mtp_ms
    agg.verifier_ms_sum += wall_verifier_ms
    agg.replay_ms_sum += wall_replay_ms
    agg.backup_ms_sum += wall_backup_ms
    agg.fallback_ms_sum += wall_fallback_ms
    agg.snapshot_ms_sum += wall_snapshot_ms
    snapshot_modeled_wall_ms = mtp_spec_wall_snapshot_cost_probe ? (wall_ms - wall_replay_ms + wall_snapshot_ms) : wall_ms
    agg.snapshot_modeled_wall_ms_sum += snapshot_modeled_wall_ms
    agg.wall_ms_sum += wall_ms
    agg.plain_exact_ms_sum += plain_exact_ms

    verifier_mode = mtp_spec_wall_serial_early_verify ? "serial_early" : "chunk"
    puts "mtp_spec_wall_summary label=#{label.inspect} mode=exact_resync_wall verifier=#{verifier_mode} gamma=#{gamma} stage=#{mtp_spec_wall_stage} stage_once=#{mtp_spec_wall_stage_once} stage_bonus=#{mtp_spec_wall_stage_bonus} top2_accounting=#{mtp_spec_wall_top2_accounting} top2_miss_offramp=#{mtp_spec_wall_top2_miss_offramp} promote_top2_margin=#{fmt3(mtp_spec_wall_promote_top2_margin)} lazy_draft=#{mtp_spec_wall_lazy_draft} reject_offramp=#{mtp_spec_wall_reject_offramp} snapshot_cost_probe=#{mtp_spec_wall_snapshot_cost_probe} tokens=#{mtp_chain_tokens} passes=#{wall_passes} emitted=#{wall_ids.size} draft_tokens=#{wall_draft_tokens} accepted=#{wall_accepted} rejections=#{wall_rejections} verifier_calls=#{wall_verifier_calls} verifier_tokens=#{wall_verifier_tokens} replay_tokens=#{wall_replay_tokens} fallback_tokens=#{wall_fallback_tokens} snapshot_tokens=#{wall_snapshot_tokens} top2_checks=#{wall_top2_checks} top2_rescues=#{wall_top2_rescues} top2_wrong_tail_tokens=#{wall_top2_wrong_tail_tokens} top2_replay_tokens=#{wall_top2_replay_tokens} top2_replay_ms=#{wall_top2_replay_ms.round(3)} top2_offramp_hits=#{wall_top2_offramp_hits} top2_promotions=#{wall_top2_promotions} top2_promoted_accepted=#{wall_top2_promoted_accepted} accept_rate=#{pct(wall_accepted, wall_draft_tokens).round(2)} tokens_per_pass=#{(wall_ids.size.to_f64 / wall_passes).round(3)} mtp_ms=#{wall_mtp_ms.round(3)} verifier_ms=#{wall_verifier_ms.round(3)} backup_ms=#{wall_backup_ms.round(3)} replay_ms=#{wall_replay_ms.round(3)} fallback_ms=#{wall_fallback_ms.round(3)} snapshot_sim_ms=#{wall_snapshot_ms.round(3)} snapshot_modeled_wall_ms=#{snapshot_modeled_wall_ms.round(3)} snapshot_modeled_speedup=#{(plain_exact_ms / snapshot_modeled_wall_ms).round(3)} wall_ms=#{wall_ms.round(3)} plain_exact_ms=#{plain_exact_ms.round(3)} plain_speedup=#{(plain_exact_ms / wall_ms).round(3)} parity=#{parity} ids=#{wall_ids.join(",")}"
    if mtp_spec_wall_profile
      puts "mtp_spec_wall_profile label=#{label.inspect} gamma=#{gamma} verifier_calls=#{wall_verifier_calls} verifier_tokens=#{wall_verifier_tokens}"
      puts ML::GGUF::Qwen35Metal::Profile.report_io
    end
  end

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
    chain_rank_hist = Hash(Int32, Int32).new(0)
    chain_rank_order_attempts = 0
    chain_margin_sum = 0.0_f64
    chain_margin_count = 0
    chain_hit_margin_sum = 0.0_f64
    chain_hit_margin_count = 0
    chain_miss_margin_sum = 0.0_f64
    chain_miss_margin_count = 0
    chain_hit_margin_min = nil.as(Float64?)
    chain_miss_margin_max = nil.as(Float64?)
    chain_candidates = [] of Int32
    router_tokens = Array(Int32).new(mtp_chain_margin_thresholds.size, 0)
    router_top1_hits = Array(Int32).new(mtp_chain_margin_thresholds.size, 0)
    router_selected = Array(Int32).new(mtp_chain_margin_thresholds.size, 0)
    router_selected_hits = Array(Int32).new(mtp_chain_margin_thresholds.size, 0)
    router_false_accepts = Array(Int32).new(mtp_chain_margin_thresholds.size, 0)

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
      chain_rank_hist[rank] += 1
      chain_rank_order_attempts += topk_hit ? rank : mtp_chain_topk
      chain_top1_hits += 1 if top1_hit
      chain_topk_hits += 1 if topk_hit
      chain_first_miss = i if chain_first_miss < 0 && !top1_hit
      chain_candidates << candidate

      if mtp_topk.size >= 2
        margin = (mtp_topk[0][1] - mtp_topk[1][1]).to_f64
        chain_margin_sum += margin
        chain_margin_count += 1
        if top1_hit
          chain_hit_margin_sum += margin
          chain_hit_margin_count += 1
          chain_hit_margin_min = if min = chain_hit_margin_min
                                   margin < min ? margin : min
                                 else
                                   margin
                                 end
        else
          chain_miss_margin_sum += margin
          chain_miss_margin_count += 1
          chain_miss_margin_max = if max = chain_miss_margin_max
                                    margin > max ? margin : max
                                  else
                                    margin
                                  end
        end

        mtp_chain_margin_thresholds.each_with_index do |threshold, threshold_i|
          router_tokens[threshold_i] += 1
          router_top1_hits[threshold_i] += 1 if top1_hit
          next unless margin >= threshold

          router_selected[threshold_i] += 1
          if top1_hit
            router_selected_hits[threshold_i] += 1
          else
            router_false_accepts[threshold_i] += 1
          end
        end
      end

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
    chain_mtp_ms_sum = chain_ms.sum
    chain_topk_misses = mtp_chain_tokens - chain_topk_hits
    chain_full_topk_attempts = mtp_chain_tokens * mtp_chain_topk
    chain_rank_order_wasted = chain_rank_order_attempts - chain_topk_hits
    chain_oracle_select_attempts = chain_topk_hits
    chain_avg_rank_attempts = chain_rank_order_attempts.to_f64 / mtp_chain_tokens
    chain_avg_full_attempts = chain_full_topk_attempts.to_f64 / mtp_chain_tokens
    chain_avg_oracle_attempts = chain_oracle_select_attempts.to_f64 / mtp_chain_tokens
    chain_serial_exact_plus_mtp_ms = exact_hidden_oracle_ms + chain_mtp_ms_sum
    chain_ideal_overlap_ms = exact_hidden_oracle_ms > chain_mtp_ms_sum ? exact_hidden_oracle_ms : chain_mtp_ms_sum
    chain_margin_avg = chain_margin_count > 0 ? chain_margin_sum / chain_margin_count : nil
    chain_hit_margin_avg = chain_hit_margin_count > 0 ? chain_hit_margin_sum / chain_hit_margin_count : nil
    chain_miss_margin_avg = chain_miss_margin_count > 0 ? chain_miss_margin_sum / chain_miss_margin_count : nil

    agg = chain_aggregates[mode]
    agg.rows += 1
    agg.tokens += mtp_chain_tokens
    agg.top1_hits += chain_top1_hits
    agg.topk_hits += chain_topk_hits
    agg.rank_order_attempts += chain_rank_order_attempts
    agg.full_topk_attempts += chain_full_topk_attempts
    agg.mtp_ms_sum += chain_mtp_ms_sum
    agg.exact_hidden_oracle_ms_sum += exact_hidden_oracle_ms
    chain_rank_hist.each { |rank, count| agg.rank_hist[rank] += count }
    agg.top1_margin_sum += chain_margin_sum
    agg.top1_margin_count += chain_margin_count
    agg.top1_hit_margin_sum += chain_hit_margin_sum
    agg.top1_hit_margin_count += chain_hit_margin_count
    agg.top1_miss_margin_sum += chain_miss_margin_sum
    agg.top1_miss_margin_count += chain_miss_margin_count
    if min = chain_hit_margin_min
      agg.top1_hit_margin_min = if agg_min = agg.top1_hit_margin_min
                                  min < agg_min ? min : agg_min
                                else
                                  min
                                end
    end
    if max = chain_miss_margin_max
      agg.top1_miss_margin_max = if agg_max = agg.top1_miss_margin_max
                                   max > agg_max ? max : agg_max
                                 else
                                   max
                                 end
    end

    puts "mtp_chain_summary label=#{label.inspect} mode=#{mode} tokens=#{mtp_chain_tokens} topk=#{mtp_chain_topk} top1_hits=#{chain_top1_hits} top1_rate=#{(chain_top1_hits * 100.0 / mtp_chain_tokens).round(2)} topk_hits=#{chain_topk_hits} topk_rate=#{(chain_topk_hits * 100.0 / mtp_chain_tokens).round(2)} topk_misses=#{chain_topk_misses} first_miss=#{chain_first_miss} rank_hist=#{rank_hist_string(chain_rank_hist)} rank_order_attempts=#{chain_rank_order_attempts} avg_rank_order_attempts=#{chain_avg_rank_attempts.round(3)} rank_order_wasted=#{chain_rank_order_wasted} full_topk_attempts=#{chain_full_topk_attempts} avg_full_topk_attempts=#{chain_avg_full_attempts.round(3)} oracle_select_attempts=#{chain_oracle_select_attempts} avg_oracle_select_attempts=#{chain_avg_oracle_attempts.round(3)} top1_margin_avg=#{fmt3(chain_margin_avg)} hit_margin_avg=#{fmt3(chain_hit_margin_avg)} miss_margin_avg=#{fmt3(chain_miss_margin_avg)} hit_margin_min=#{fmt3(chain_hit_margin_min)} miss_margin_max=#{fmt3(chain_miss_margin_max)} mtp_avg_ms=#{(chain_mtp_ms_sum / chain_ms.size).round(3)} mtp_min=#{sorted_chain_ms.first.round(3)} mtp_p50=#{chain_p50_ms.round(3)} mtp_max=#{sorted_chain_ms.last.round(3)} exact_hidden_oracle_ms=#{exact_hidden_oracle_ms.round(3)} exact_hidden_oracle_avg_ms=#{(exact_hidden_oracle_ms / mtp_chain_tokens).round(3)} serial_exact_plus_mtp_ms=#{chain_serial_exact_plus_mtp_ms.round(3)} serial_vs_exact=#{(chain_serial_exact_plus_mtp_ms / exact_hidden_oracle_ms).round(3)} ideal_overlap_ms=#{chain_ideal_overlap_ms.round(3)} ideal_overlap_vs_exact=#{(chain_ideal_overlap_ms / exact_hidden_oracle_ms).round(3)} candidate_ids=#{chain_candidates.join(",")} candidate_text=#{chain_text.inspect}"

    unless mtp_chain_margin_thresholds.empty?
      router_agg = margin_router_aggregates[mode]
      mtp_chain_margin_thresholds.each_with_index do |threshold, threshold_i|
        tokens = router_tokens[threshold_i]
        selected = router_selected[threshold_i]
        selected_hits = router_selected_hits[threshold_i]
        false_accepts = router_false_accepts[threshold_i]
        top1_hit_count = router_top1_hits[threshold_i]
        fallback = tokens - selected

        agg_threshold = router_agg[threshold_i]
        agg_threshold.rows += 1
        agg_threshold.tokens += tokens
        agg_threshold.top1_hits += top1_hit_count
        agg_threshold.selected += selected
        agg_threshold.selected_hits += selected_hits
        agg_threshold.false_accepts += false_accepts

        puts "mtp_chain_margin_router label=#{label.inspect} mode=#{mode} threshold=#{threshold} tokens=#{tokens} selected=#{selected} selected_rate=#{pct(selected, tokens).round(2)} selected_hits=#{selected_hits} false_accepts=#{false_accepts} precision=#{pct(selected_hits, selected).round(2)} fallback=#{fallback} hit_recall=#{pct(selected_hits, top1_hit_count).round(2)}"
      end
    end
  end
end

puts "mtp_suite_summary rows=#{total_rows} top1_hits=#{top1_hits} top1_rate=#{(top1_hits * 100.0 / total_rows).round(2)} top5_or_top1_hits=#{top5_hits} top5_or_top1_rate=#{(top5_hits * 100.0 / total_rows).round(2)} timed_calls=#{timed_calls} avg_mtp_ms=#{(timed_ms / timed_calls).round(3)} backend=#{mtp_backend} top1_only=#{top1_only}"
chain_aggregates.keys.sort.each do |mode|
  agg = chain_aggregates[mode]
  next if agg.tokens == 0

  rank_order_wasted = agg.rank_order_attempts - agg.topk_hits
  oracle_select_attempts = agg.topk_hits
  serial_exact_plus_mtp_ms = agg.exact_hidden_oracle_ms_sum + agg.mtp_ms_sum
  ideal_overlap_ms = agg.exact_hidden_oracle_ms_sum > agg.mtp_ms_sum ? agg.exact_hidden_oracle_ms_sum : agg.mtp_ms_sum
  margin_avg = agg.top1_margin_count > 0 ? agg.top1_margin_sum / agg.top1_margin_count : nil
  hit_margin_avg = agg.top1_hit_margin_count > 0 ? agg.top1_hit_margin_sum / agg.top1_hit_margin_count : nil
  miss_margin_avg = agg.top1_miss_margin_count > 0 ? agg.top1_miss_margin_sum / agg.top1_miss_margin_count : nil
  puts "mtp_chain_suite_summary mode=#{mode} rows=#{agg.rows} tokens=#{agg.tokens} topk=#{mtp_chain_topk} top1_hits=#{agg.top1_hits} top1_rate=#{(agg.top1_hits * 100.0 / agg.tokens).round(2)} topk_hits=#{agg.topk_hits} topk_rate=#{(agg.topk_hits * 100.0 / agg.tokens).round(2)} topk_misses=#{agg.tokens - agg.topk_hits} rank_hist=#{rank_hist_string(agg.rank_hist)} rank_order_attempts=#{agg.rank_order_attempts} avg_rank_order_attempts=#{(agg.rank_order_attempts.to_f64 / agg.tokens).round(3)} rank_order_wasted=#{rank_order_wasted} full_topk_attempts=#{agg.full_topk_attempts} avg_full_topk_attempts=#{(agg.full_topk_attempts.to_f64 / agg.tokens).round(3)} oracle_select_attempts=#{oracle_select_attempts} avg_oracle_select_attempts=#{(oracle_select_attempts.to_f64 / agg.tokens).round(3)} top1_margin_avg=#{fmt3(margin_avg)} hit_margin_avg=#{fmt3(hit_margin_avg)} miss_margin_avg=#{fmt3(miss_margin_avg)} hit_margin_min=#{fmt3(agg.top1_hit_margin_min)} miss_margin_max=#{fmt3(agg.top1_miss_margin_max)} mtp_avg_ms=#{(agg.mtp_ms_sum / agg.tokens).round(3)} mtp_total_ms=#{agg.mtp_ms_sum.round(3)} exact_hidden_oracle_ms=#{agg.exact_hidden_oracle_ms_sum.round(3)} exact_hidden_oracle_avg_ms=#{(agg.exact_hidden_oracle_ms_sum / agg.tokens).round(3)} serial_exact_plus_mtp_ms=#{serial_exact_plus_mtp_ms.round(3)} serial_vs_exact=#{(serial_exact_plus_mtp_ms / agg.exact_hidden_oracle_ms_sum).round(3)} ideal_overlap_ms=#{ideal_overlap_ms.round(3)} ideal_overlap_vs_exact=#{(ideal_overlap_ms / agg.exact_hidden_oracle_ms_sum).round(3)}"
end
margin_router_aggregates.keys.sort.each do |mode|
  margin_router_aggregates[mode].each_with_index do |agg, threshold_i|
    next if agg.tokens == 0

    threshold = mtp_chain_margin_thresholds[threshold_i]
    fallback = agg.tokens - agg.selected
    puts "mtp_chain_margin_router_suite mode=#{mode} threshold=#{threshold} rows=#{agg.rows} tokens=#{agg.tokens} selected=#{agg.selected} selected_rate=#{pct(agg.selected, agg.tokens).round(2)} selected_hits=#{agg.selected_hits} false_accepts=#{agg.false_accepts} precision=#{pct(agg.selected_hits, agg.selected).round(2)} fallback=#{fallback} hit_recall=#{pct(agg.selected_hits, agg.top1_hits).round(2)}"
  end
end
mtp_spec_aggregates.keys.sort.each do |gamma|
  agg = mtp_spec_aggregates[gamma]
  next if agg.tokens == 0

  additive_wall_ms = agg.target_pass_model_ms_sum + agg.mtp_ms_sum
  overlap_wall_ms = agg.target_pass_model_ms_sum > agg.mtp_ms_sum ? agg.target_pass_model_ms_sum : agg.mtp_ms_sum
  puts "mtp_spec_suite_summary mode=exact_resync gamma=#{gamma} rows=#{agg.rows} tokens=#{agg.tokens} passes=#{agg.passes} emitted=#{agg.emitted} draft_tokens=#{agg.draft_tokens} accepted=#{agg.accepted} accept_rate=#{pct(agg.accepted, agg.draft_tokens).round(2)} tokens_per_pass=#{(agg.emitted.to_f64 / agg.passes).round(3)} target_pass_speedup_bound=#{(agg.tokens.to_f64 / agg.passes).round(3)} mtp_ms=#{agg.mtp_ms_sum.round(3)} exact_hidden_oracle_ms=#{agg.exact_ms_sum.round(3)} target_pass_model_ms=#{agg.target_pass_model_ms_sum.round(3)} additive_wall_model_ms=#{additive_wall_ms.round(3)} additive_speedup_model=#{(agg.exact_ms_sum / additive_wall_ms).round(3)} ideal_overlap_wall_model_ms=#{overlap_wall_ms.round(3)} ideal_overlap_speedup_model=#{(agg.exact_ms_sum / overlap_wall_ms).round(3)}"
end
mtp_spec_wall_aggregates.keys.sort.each do |gamma|
  agg = mtp_spec_wall_aggregates[gamma]
  next if agg.tokens == 0

  verifier_mode = mtp_spec_wall_serial_early_verify ? "serial_early" : "chunk"
  puts "mtp_spec_wall_suite_summary mode=exact_resync_wall verifier=#{verifier_mode} gamma=#{gamma} stage=#{mtp_spec_wall_stage} stage_once=#{mtp_spec_wall_stage_once} stage_bonus=#{mtp_spec_wall_stage_bonus} top2_accounting=#{mtp_spec_wall_top2_accounting} top2_miss_offramp=#{mtp_spec_wall_top2_miss_offramp} promote_top2_margin=#{fmt3(mtp_spec_wall_promote_top2_margin)} lazy_draft=#{mtp_spec_wall_lazy_draft} reject_offramp=#{mtp_spec_wall_reject_offramp} snapshot_cost_probe=#{mtp_spec_wall_snapshot_cost_probe} rows=#{agg.rows} parity_ok=#{agg.parity_ok}/#{agg.rows} tokens=#{agg.tokens} passes=#{agg.passes} emitted=#{agg.emitted} draft_tokens=#{agg.draft_tokens} accepted=#{agg.accepted} rejections=#{agg.rejections} verifier_calls=#{agg.verifier_calls} verifier_tokens=#{agg.verifier_tokens} replay_tokens=#{agg.replay_tokens} fallback_tokens=#{agg.fallback_tokens} snapshot_tokens=#{agg.snapshot_tokens} top2_checks=#{agg.top2_checks} top2_rescues=#{agg.top2_rescues} top2_wrong_tail_tokens=#{agg.top2_wrong_tail_tokens} top2_replay_tokens=#{agg.top2_replay_tokens} top2_replay_ms=#{agg.top2_replay_ms_sum.round(3)} top2_offramp_hits=#{agg.top2_offramp_hits} top2_promotions=#{agg.top2_promotions} top2_promoted_accepted=#{agg.top2_promoted_accepted} accept_rate=#{pct(agg.accepted, agg.draft_tokens).round(2)} tokens_per_pass=#{(agg.emitted.to_f64 / agg.passes).round(3)} mtp_ms=#{agg.mtp_ms_sum.round(3)} verifier_ms=#{agg.verifier_ms_sum.round(3)} backup_ms=#{agg.backup_ms_sum.round(3)} replay_ms=#{agg.replay_ms_sum.round(3)} fallback_ms=#{agg.fallback_ms_sum.round(3)} snapshot_sim_ms=#{agg.snapshot_ms_sum.round(3)} snapshot_modeled_wall_ms=#{agg.snapshot_modeled_wall_ms_sum.round(3)} snapshot_modeled_speedup=#{(agg.plain_exact_ms_sum / agg.snapshot_modeled_wall_ms_sum).round(3)} wall_ms=#{agg.wall_ms_sum.round(3)} plain_exact_ms=#{agg.plain_exact_ms_sum.round(3)} plain_speedup=#{(agg.plain_exact_ms_sum / agg.wall_ms_sum).round(3)}"
end
router_trace_io.try(&.close)
