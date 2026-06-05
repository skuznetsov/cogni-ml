require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
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
train = 8
wild_gen = nil.as(Int32?)
max_seq = 256
prefill_chunk = 128
surrogate_layer = 44
rank = 8
lambda = 1.0e-3_f64
seed = 0x5eed_i64
warmup_exact = nil.as(Int32?)
diagnose_risk = false
risk_thresholds = [0.0_f32, 0.5_f32, 1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32]
oracle_topk_rescue = 0
oracle_topk_fallback_exact = false
proposal_main_state = false
verifier_continue_from_proposal = false
proposal_resident_topk = 0
proposal_body_mode = "partial"
proposal_memory_k = 4

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_late_band_wild_probe [options]"
  p.on("--model PATH", "Gemma4 GGUF model path") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Raw prompt text") { |v| prompt_text = v; chat_user = nil }
  p.on("--prompt-file PATH", "Read raw prompt text from file") { |v| prompt_file = v; chat_user = nil }
  p.on("--chat-user TEXT", "Format one Gemma4 user turn") { |v| chat_user = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids; bypasses tokenizer") { |v| token_ids_arg = v }
  p.on("--gen N", "Exact trajectory tokens to collect for fitting, default 16") { |v| gen = v.to_i }
  p.on("--train N", "Training samples from the front of the trajectory, default 8") { |v| train = v.to_i }
  p.on("--wild-gen N", "Tokens to generate in exact and surrogate wild runs, default --gen") { |v| wild_gen = v.to_i }
  p.on("--surrogate-layer N", "Stop layer used as surrogate input, default 44") { |v| surrogate_layer = v.to_i }
  p.on("--rank N", "Random-projection residual rank, default 8") { |v| rank = v.to_i }
  p.on("--lambda F", "Ridge regularization, default 1e-3") { |v| lambda = v.to_f64 }
  p.on("--warmup-exact N", "Initial exact tokens before surrogate takes over, default --train") { |v| warmup_exact = v.to_i }
  p.on("--diagnose-risk", "Compute exact-token ranks and surrogate margins for risk-gate analysis") { diagnose_risk = true }
  p.on("--oracle-topk-rescue K", "Oracle ceiling: choose exact token when it is inside surrogate top-K, default disabled") { |v| oracle_topk_rescue = v.to_i }
  p.on("--oracle-topk-fallback-exact", "Oracle ceiling: on top-K miss, also choose exact token to preserve exact path") { oracle_topk_fallback_exact = true }
  p.on("--proposal-main-state", "Run partial proposal on exact state; exact full pass overwrites same-position cache rows") { proposal_main_state = true }
  p.on("--verifier-continue-from-proposal", "Continue exact verifier from proposal hidden; requires --proposal-main-state") { verifier_continue_from_proposal = true }
  p.on("--proposal-resident-top1", "Use resident fused head top1 for surrogate proposal instead of full logits/top-k") { proposal_resident_topk = 1 }
  p.on("--proposal-resident-top2", "Use resident fused head top2 for surrogate proposal instead of full logits/top-k") { proposal_resident_topk = 2 }
  p.on("--proposal-body-mode MODE", "Proposal hidden source: partial, prev_full, temporal_full, memory_nn") { |v| proposal_body_mode = v }
  p.on("--proposal-memory-k N", "Nearest-neighbor memory shortlist size for proposal-body-mode memory_nn, default 4") { |v| proposal_memory_k = v.to_i }
  p.on("--risk-thresholds LIST", "Comma-separated surrogate top1-top2 margin thresholds, default 0,0.5,1,2,4,8") do |v|
    risk_thresholds = v.split(',').reject(&.empty?).map(&.to_f32)
  end
  p.on("--max-seq N", "Resident state sequence capacity, default 256") { |v| max_seq = v.to_i }
  p.on("--prefill-chunk N", "Row prefill chunk size, default 128") { |v| prefill_chunk = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

wild_n = wild_gen.nil? ? gen : wild_gen.not_nil!
warmup_n = warmup_exact.nil? ? train : warmup_exact.not_nil!

raise "--gen must be positive" unless gen > 0
raise "--train must be positive and smaller than --gen" unless train > 0 && train < gen
raise "--wild-gen must be positive" unless wild_n > 0
raise "--rank must be positive" unless rank > 0
raise "--warmup-exact must be non-negative" unless warmup_n >= 0
raise "--oracle-topk-rescue must be non-negative" unless oracle_topk_rescue >= 0
raise "--oracle-topk-fallback-exact requires --oracle-topk-rescue K with K > 0" if oracle_topk_fallback_exact && oracle_topk_rescue <= 0
raise "--verifier-continue-from-proposal requires --proposal-main-state" if verifier_continue_from_proposal && !proposal_main_state
raise "--proposal-body-mode must be partial, prev_full, temporal_full, or memory_nn" unless {"partial", "prev_full", "temporal_full", "memory_nn"}.includes?(proposal_body_mode)
raise "--verifier-continue-from-proposal requires --proposal-body-mode partial" if verifier_continue_from_proposal && proposal_body_mode != "partial"
raise "--proposal-body-mode #{proposal_body_mode} requires --warmup-exact > 0" if proposal_body_mode != "partial" && warmup_n <= 0
raise "--proposal-memory-k must be positive" unless proposal_memory_k > 0
raise "--max-seq must be positive" unless max_seq > 0
raise "--prefill-chunk must be positive" unless prefill_chunk > 0
raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "tokenizer binary not found: #{tokenizer_bin}" unless token_ids_arg || File.exists?(tokenizer_bin)

if file = prompt_file
  prompt_text = File.read(file)
elsif user = chat_user
  prompt_text = "<|turn>user\n#{user}<turn|>\n<|turn>model\n"
end

weights = ML::GGUF::Gemma4Weights.from_gguf(model_path)
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?
raise "--surrogate-layer must be positive" unless surrogate_layer > 0
raise "--surrogate-layer exceeds model layer count" if surrogate_layer > weights.hparams.n_layer

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
raise "prompt+max(gen,wild-gen) exceeds max_seq" if ids.size + Math.max(gen, wild_n) > max_seq

record Sample, input_token : Int32, exact_top1 : Int32, h_layer : Array(Float32), h_full : Array(Float32)
record RiskRow, step : Int32, exact_top1 : Int32, surrogate_top1 : Int32, chosen_token : Int32,
  exact_rank : Int32, margin : Float32, top5_contains_exact : Bool, oracle_rescued : Bool
record WildStats, ids : Array(Int32), ms : Float64, surrogate_steps : Int32,
  risk_rows : Array(RiskRow), proposal_ms : Float64, verifier_ms : Float64,
  snapshot_ms : Float64, restore_ms : Float64, partial_ms : Float64,
  residual_ms : Float64, head_ms : Float64, topk_ms : Float64

record MemoryProposalItem, features : Array(Float64), token : Int32

# Deterministic Rademacher projection. This keeps the probe cheap and removes
# a separate PCA pass while preserving a fixed low-rank residual basis.
def proj_sign(i : Int32, j : Int32, seed : Int64) : Float64
  x = (i.to_i64 &* 1103515245_i64) ^ (j.to_i64 &* 12345_i64) ^ seed
  (x & 1_i64) == 0_i64 ? 1.0 : -1.0
end

def project_features(x : Array(Float32), mean : Array(Float64), rank : Int32, seed : Int64) : Array(Float64)
  scale = 1.0 / Math.sqrt(x.size.to_f64)
  z = Array(Float64).new(rank, 0.0)
  rank.times do |j|
    sum = 0.0
    x.each_with_index do |v, i|
      sum += proj_sign(i, j, seed) * (v.to_f64 - mean[i])
    end
    z[j] = sum * scale
  end
  z
end

def invert_matrix(a : Array(Array(Float64))) : Array(Array(Float64))
  n = a.size
  aug = Array(Array(Float64)).new(n) do |i|
    row = Array(Float64).new(2 * n, 0.0)
    n.times { |j| row[j] = a[i][j] }
    row[n + i] = 1.0
    row
  end

  n.times do |col|
    pivot = col
    best = aug[col][col].abs
    (col + 1...n).each do |r|
      value = aug[r][col].abs
      if value > best
        best = value
        pivot = r
      end
    end
    raise "singular matrix in ridge solve" if best < 1.0e-12
    aug[col], aug[pivot] = aug[pivot], aug[col] if pivot != col

    div = aug[col][col]
    (2 * n).times { |c| aug[col][c] /= div }
    n.times do |r|
      next if r == col
      factor = aug[r][col]
      next if factor.abs < 1.0e-18
      (2 * n).times { |c| aug[r][c] -= factor * aug[col][c] }
    end
  end

  Array(Array(Float64)).new(n) do |i|
    Array(Float64).new(n) { |j| aug[i][n + j] }
  end
end

def top_k_ids(logits : Array(Float32), k : Int32) : Array(Int32)
  ids = Array(Int32).new(k, -1)
  vals = Array(Float32).new(k, -Float32::INFINITY)
  logits.each_with_index do |v, id|
    next if v <= vals[-1]
    pos = k - 1
    while pos > 0 && v > vals[pos - 1]
      vals[pos] = vals[pos - 1]
      ids[pos] = ids[pos - 1]
      pos -= 1
    end
    vals[pos] = v
    ids[pos] = id.to_i32
  end
  ids
end

def rank_of_id(logits : Array(Float32), id : Int32) : Int32
  raise "rank_of_id: token id #{id} out of range" if id < 0 || id >= logits.size
  target = logits[id]
  rank = 1
  logits.each { |v| rank += 1 if v > target }
  rank
end

def print_gate_sweep(name : String,
                     rows : Array(RiskRow),
                     proposal_ms_per_step : Float64,
                     verifier_ms_per_step : Float64,
                     exact_ms_per_token : Float64,
                     thresholds : Array(Float32)) : Nil
  return if rows.empty?
  puts "gate_sweep_BEGIN #{name}"
  puts "gate\tthreshold\tattempts\trescues\tmisses\tattempt_rate\trescue_rate_on_attempt\toptimistic_ms\toptimistic_speedup"
  thresholds.each do |threshold|
    attempted = rows.select { |r| yield r, threshold }
    rescues = attempted.count(&.oracle_rescued)
    misses = attempted.size - rescues
    # Conservative shape: non-attempted and missed rows fall back to exact;
    # attempted rows pay proposal+continuation verifier. This is an oracle
    # separability diagnostic, not a deployable scheduler estimate.
    fallback_rows = rows.size - rescues
    total_ms = attempted.size * (proposal_ms_per_step + verifier_ms_per_step) +
               fallback_rows * exact_ms_per_token
    optimistic_ms = total_ms / rows.size
    speedup = exact_ms_per_token / optimistic_ms
    attempt_rate = attempted.size.to_f64 / rows.size
    rescue_rate = attempted.empty? ? 0.0 : rescues.to_f64 / attempted.size
    puts [name, threshold, attempted.size, rescues, misses, attempt_rate.round(4), rescue_rate.round(4), optimistic_ms.round(3), speedup.round(4)].join('\t')
  end
  puts "gate_sweep_END #{name}"
end

def prefill_prefix!(weights, ids : Array(Int32), state, prefill_chunk : Int32) : Nil
  return if ids.size <= 1

  ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(
    weights, ids[0...-1], 0, state,
    chunk_size: prefill_chunk,
    stop_layer: weights.hparams.n_layer,
    read_last_hidden: false
  ).not_nil!
end

class ResidualSurrogate
  getter mean : Array(Float64)
  getter coeff : Array(Array(Float64))
  getter rank : Int32
  getter seed : Int64

  def initialize(@mean : Array(Float64), @coeff : Array(Array(Float64)), @rank : Int32, @seed : Int64)
  end

  def predict(x : Array(Float32)) : Array(Float32)
    z = project_features(x, @mean, @rank, @seed)
    out = Array(Float32).new(x.size, 0.0_f32)
    x.size.times do |i|
      delta = 0.0
      @rank.times { |j| delta += z[j] * @coeff[j][i] }
      out[i] = (x[i].to_f64 + delta).to_f32
    end
    out
  end
end

class MemoryProposalBank
  getter mean : Array(Float64)
  getter items : Array(MemoryProposalItem)
  getter rank : Int32
  getter seed : Int64
  getter k : Int32

  def initialize(@mean : Array(Float64), @items : Array(MemoryProposalItem), @rank : Int32, @seed : Int64, @k : Int32)
  end

  def propose(x : Array(Float32)) : Array(Int32)
    z = project_features(x, @mean, @rank, @seed)
    scored = @items.map do |item|
      dist = 0.0
      @rank.times do |i|
        d = z[i] - item.features[i]
        dist += d * d
      end
      {dist, item.token}
    end
    ids = [] of Int32
    scored.sort_by!(&.[0])
    scored.each do |_dist, token|
      next if ids.includes?(token)
      ids << token
      break if ids.size >= @k
    end
    ids
  end
end

def fit_surrogate(xs : Array(Array(Float32)), ys : Array(Array(Float32)), rank : Int32, lambda : Float64, seed : Int64) : ResidualSurrogate
  raise "fit_surrogate requires samples" if xs.empty?
  dim = xs[0].size
  mean = Array(Float64).new(dim, 0.0)
  xs.each do |x|
    dim.times { |i| mean[i] += x[i] }
  end
  dim.times { |i| mean[i] /= xs.size.to_f64 }

  gram = Array(Array(Float64)).new(rank) { Array(Float64).new(rank, 0.0) }
  rhs = Array(Array(Float64)).new(rank) { Array(Float64).new(dim, 0.0) }
  xs.each_with_index do |x, sidx|
    z = project_features(x, mean, rank, seed)
    rank.times do |j|
      rank.times { |k| gram[j][k] += z[j] * z[k] }
      dim.times { |i| rhs[j][i] += z[j] * (ys[sidx][i].to_f64 - x[i].to_f64) }
    end
  end
  rank.times { |j| gram[j][j] += lambda }
  inv = invert_matrix(gram)

  coeff = Array(Array(Float64)).new(rank) { Array(Float64).new(dim, 0.0) }
  rank.times do |j|
    rank.times do |k|
      factor = inv[j][k]
      next if factor.abs < 1.0e-18
      dim.times { |i| coeff[j][i] += factor * rhs[k][i] }
    end
  end
  ResidualSurrogate.new(mean, coeff, rank, seed)
end

def fit_memory_proposal_bank(samples : Array(Sample), rank : Int32, seed : Int64, k : Int32) : MemoryProposalBank
  raise "memory proposal bank requires samples" if samples.empty?
  dim = samples[0].h_full.size
  mean = Array(Float64).new(dim, 0.0)
  samples.each do |sample|
    dim.times { |i| mean[i] += sample.h_full[i] }
  end
  dim.times { |i| mean[i] /= samples.size.to_f64 }
  items = samples.map do |sample|
    MemoryProposalItem.new(project_features(sample.h_full, mean, rank, seed), sample.exact_top1)
  end
  MemoryProposalBank.new(mean, items, rank, seed, k)
end

def collect_exact_samples(weights, ids : Array(Int32), gen : Int32, surrogate_layer : Int32,
                          max_seq : Int32, prefill_chunk : Int32) : {Array(Sample), Array(Int32), Float64}
  main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  side_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  prefill_prefix!(weights, ids, main_state, prefill_chunk)

  samples = [] of Sample
  exact_ids = [] of Int32
  current_token = ids[-1]
  pos = ids.size - 1
  ms = 0.0

  gen.times do
    t0 = Time.instant
    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state)
    h_layer = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
      weights, current_token, pos.to_i32, side_state, surrogate_layer
    ).not_nil!
    h_full = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
      weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
    ).not_nil!
    logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_full).not_nil!
    exact = top_k_ids(logits, 1)[0]
    ms += (Time.instant - t0).total_milliseconds

    samples << Sample.new(current_token, exact, h_layer, h_full)
    exact_ids << exact
    current_token = exact
    pos += 1
  end

  {samples, exact_ids, ms}
end

def generate_exact(weights, ids : Array(Int32), steps : Int32, max_seq : Int32, prefill_chunk : Int32) : {Array(Int32), Float64}
  state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  prefill_prefix!(weights, ids, state, prefill_chunk)

  out = [] of Int32
  current_token = ids[-1]
  pos = ids.size - 1
  t0 = Time.instant
  steps.times do
    next_token = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(
      weights, current_token, pos.to_i32, state, weights.hparams.n_layer
    ).not_nil!
    out << next_token
    current_token = next_token
    pos += 1
  end
  {out, (Time.instant - t0).total_milliseconds}
end

def generate_surrogate_wild(weights, ids : Array(Int32), steps : Int32, warmup_n : Int32,
                            surrogate_layer : Int32, surrogate : ResidualSurrogate,
                            temporal_surrogate : ResidualSurrogate?,
                            memory_bank : MemoryProposalBank?,
                            max_seq : Int32, prefill_chunk : Int32,
                            diagnose_risk : Bool,
                            oracle_topk_rescue : Int32,
                            oracle_topk_fallback_exact : Bool,
                            proposal_main_state : Bool,
                            verifier_continue_from_proposal : Bool,
                            proposal_resident_topk : Int32,
                            proposal_body_mode : String) : WildStats
  main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  side_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  prefill_prefix!(weights, ids, main_state, prefill_chunk)

  generated_ids = [] of Int32
  current_token = ids[-1]
  pos = ids.size - 1
  surrogate_steps = 0
  risk_rows = [] of RiskRow
  proposal_ms = 0.0
  verifier_ms = 0.0
  snapshot_ms = 0.0
  restore_ms = 0.0
  partial_ms = 0.0
  residual_ms = 0.0
  head_ms = 0.0
  topk_ms = 0.0
  last_full_hidden = nil.as(Array(Float32)?)
  t0 = Time.instant

  steps.times do |step|
    if step < warmup_n
      next_token = if proposal_body_mode == "partial"
                     ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(
                       weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
                     ).not_nil!
                   else
                     h_full = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
                       weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
                     ).not_nil!
                     last_full_hidden = h_full
                     logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_full).not_nil!
                     top_k_ids(logits, 1)[0]
                   end
      generated_ids << next_token
      current_token = next_token
      pos += 1
      next
    end

    proposal_t0 = Time.instant
    proposal_state = main_state
    if proposal_body_mode == "partial"
      proposal_state = if proposal_main_state
                         main_state
                       else
                         phase_t0 = Time.instant
                         snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
                         snapshot_ms += (Time.instant - phase_t0).total_milliseconds
                         phase_t0 = Time.instant
                         ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state)
                         restore_ms += (Time.instant - phase_t0).total_milliseconds
                         side_state
                       end
    end
    h_layer = nil.as(Array(Float32)?)
    h_hat = nil.as(Array(Float32)?)
    phase_t0 = Time.instant
    if proposal_body_mode == "partial"
      h_layer = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
        weights, current_token, pos.to_i32, proposal_state, surrogate_layer
      ).not_nil!
      partial_ms += (Time.instant - phase_t0).total_milliseconds
      phase_t0 = Time.instant
      h_hat = surrogate.predict(h_layer.not_nil!)
    elsif proposal_body_mode == "prev_full"
      prior = last_full_hidden
      raise "proposal_body_mode=#{proposal_body_mode} requires --warmup-exact > 0 or prior verified hidden" if prior.nil?
      partial_ms += (Time.instant - phase_t0).total_milliseconds
      phase_t0 = Time.instant
      h_hat = prior.not_nil!
    elsif proposal_body_mode == "temporal_full"
      prior = last_full_hidden
      raise "proposal_body_mode=#{proposal_body_mode} requires --warmup-exact > 0 or prior verified hidden" if prior.nil?
      partial_ms += (Time.instant - phase_t0).total_milliseconds
      phase_t0 = Time.instant
      h_hat = temporal_surrogate.not_nil!.predict(prior.not_nil!)
    else
      prior = last_full_hidden
      raise "proposal_body_mode=#{proposal_body_mode} requires --warmup-exact > 0 or prior verified hidden" if prior.nil?
      partial_ms += (Time.instant - phase_t0).total_milliseconds
      phase_t0 = Time.instant
    end
    residual_ms += (Time.instant - phase_t0).total_milliseconds
    phase_t0 = Time.instant
    logits = nil.as(Array(Float32)?)
    resident_scores = nil.as(Array(Float32)?)
    top_ids = [] of Int32
    if proposal_body_mode == "memory_nn"
      top_ids = memory_bank.not_nil!.propose(last_full_hidden.not_nil!)
      raise "memory_nn produced no proposal tokens" if top_ids.empty?
    elsif proposal_resident_topk == 1
      resident = ML::GGUF::Qwen35Metal.rmsnorm_project_top1(h_hat.not_nil!, weights.output_norm, weights.token_embd, weights.hparams.rms_eps.to_f32).not_nil!
      top_ids << resident[0].to_i32
      resident_scores = [resident[1]]
    elsif proposal_resident_topk == 2
      resident = ML::GGUF::Qwen35Metal.rmsnorm_project_top2(h_hat.not_nil!, weights.output_norm, weights.token_embd, weights.hparams.rms_eps.to_f32).not_nil!
      top_ids << resident[0].to_i32
      top_ids << resident[2].to_i32
      resident_scores = [resident[1], resident[3]]
    else
      logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_hat.not_nil!).not_nil!
    end
    head_ms += (Time.instant - phase_t0).total_milliseconds
    phase_t0 = Time.instant
    top_ids = top_k_ids(logits.not_nil!, Math.max(5, oracle_topk_rescue)) if proposal_body_mode != "memory_nn" && proposal_resident_topk == 0
    topk_ms += (Time.instant - phase_t0).total_milliseconds
    surrogate_top1 = top_ids[0]
    next_token = surrogate_top1
    proposal_ms += (Time.instant - proposal_t0).total_milliseconds

    # Autoregressive invariant: the KV state at position `pos` belongs to the
    # consumed current token, while `next_token` is only chosen for the next
    # position. This updates the exact state boundary for the generated text but
    # intentionally ignores the exact model's preferred next token.
    if diagnose_risk || oracle_topk_rescue > 0 || oracle_topk_fallback_exact
      verifier_t0 = Time.instant
      h_full = if verifier_continue_from_proposal
                 ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave_from_hidden(
                   weights, h_layer.not_nil!, pos.to_i32, main_state, surrogate_layer, weights.hparams.n_layer
                 ).not_nil!
               else
                 ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
                   weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
                 ).not_nil!
               end
      last_full_hidden = h_full
      exact_logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_full).not_nil!
      exact_top1 = top_k_ids(exact_logits, 1)[0]
      verifier_ms += (Time.instant - verifier_t0).total_milliseconds
      exact_rank = (proposal_resident_topk > 0 || proposal_body_mode == "memory_nn") ? (top_ids.index(exact_top1).try(&.+(1)) || Int32::MAX) : rank_of_id(logits.not_nil!, exact_top1)
      oracle_rescued = if proposal_resident_topk > 0 || proposal_body_mode == "memory_nn"
                         oracle_topk_rescue > 0 && exact_rank <= Math.min(oracle_topk_rescue, top_ids.size)
                       else
                         oracle_topk_rescue > 0 && exact_rank <= oracle_topk_rescue
                       end
      next_token = exact_top1 if oracle_rescued || oracle_topk_fallback_exact
      margin = if proposal_body_mode == "memory_nn"
                 0.0_f32
               elsif proposal_resident_topk > 1
                 resident_scores.not_nil![0] - resident_scores.not_nil![1]
               elsif proposal_resident_topk == 1
                 0.0_f32
               else
                 logits.not_nil![top_ids[0]] - logits.not_nil![top_ids[1]]
               end
      top5_contains_exact = (proposal_resident_topk > 0 || proposal_body_mode == "memory_nn") ? top_ids.includes?(exact_top1) : top_ids[0, 5].includes?(exact_top1)
      risk_rows << RiskRow.new(step.to_i32, exact_top1, surrogate_top1, next_token, exact_rank, margin, top5_contains_exact, oracle_rescued)
    else
      if proposal_body_mode == "partial"
        ML::GGUF::Gemma4Metal.forward_resident_cache_wave_no_read(
          weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
        )
      else
        h_full = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
          weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
        ).not_nil!
        last_full_hidden = h_full
      end
    end
    generated_ids << next_token
    current_token = next_token
    pos += 1
    surrogate_steps += 1
  end

  WildStats.new(generated_ids, (Time.instant - t0).total_milliseconds, surrogate_steps, risk_rows, proposal_ms, verifier_ms,
    snapshot_ms, restore_ms, partial_ms, residual_ms, head_ms, topk_ms)
end

puts "model=#{File.basename(model_path)} prompt_len=#{ids.size} gen=#{gen} train=#{train} wild_gen=#{wild_n} layer=#{surrogate_layer} rank=#{rank} lambda=#{lambda} warmup_exact=#{warmup_n} diagnose_risk=#{diagnose_risk} oracle_topk_rescue=#{oracle_topk_rescue} oracle_topk_fallback_exact=#{oracle_topk_fallback_exact} proposal_main_state=#{proposal_main_state} verifier_continue_from_proposal=#{verifier_continue_from_proposal} proposal_resident_topk=#{proposal_resident_topk} proposal_body_mode=#{proposal_body_mode} proposal_memory_k=#{proposal_memory_k} max_seq=#{max_seq}"

samples, collected_exact_ids, collect_ms = collect_exact_samples(weights, ids, gen, surrogate_layer, max_seq, prefill_chunk)
xs_train = samples[0...train].map(&.h_layer)
ys_train = samples[0...train].map(&.h_full)
surrogate = fit_surrogate(xs_train, ys_train, rank, lambda, seed + surrogate_layer)

temporal_surrogate = nil.as(ResidualSurrogate?)
if proposal_body_mode == "temporal_full"
  pair_count = Math.min(train, samples.size - 1)
  raise "temporal_full proposal needs at least one temporal training pair" unless pair_count > 0
  xs_temporal = Array(Array(Float32)).new(pair_count) { |i| samples[i].h_full }
  ys_temporal = Array(Array(Float32)).new(pair_count) { |i| samples[i + 1].h_full }
  temporal_surrogate = fit_surrogate(xs_temporal, ys_temporal, rank, lambda, seed ^ 0x7130_i64)
end

memory_bank = nil.as(MemoryProposalBank?)
if proposal_body_mode == "memory_nn"
  memory_bank = fit_memory_proposal_bank(samples[0...train], rank, seed ^ 0x4d3d_i64, proposal_memory_k)
end

heldout_n = Math.max(samples.size - train, 0)
heldout_top1 = 0
heldout_top5 = 0
samples[train..].each do |sample|
  pred = surrogate.predict(sample.h_layer)
  logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, pred).not_nil!
  top5 = top_k_ids(logits, 5)
  heldout_top1 += 1 if top5[0] == sample.exact_top1
  heldout_top5 += 1 if top5.includes?(sample.exact_top1)
end

exact_ids, exact_ms = generate_exact(weights, ids, wild_n, max_seq, prefill_chunk)
wild_stats = generate_surrogate_wild(weights, ids, wild_n, warmup_n, surrogate_layer, surrogate, temporal_surrogate, memory_bank, max_seq, prefill_chunk, diagnose_risk, oracle_topk_rescue, oracle_topk_fallback_exact, proposal_main_state, verifier_continue_from_proposal, proposal_resident_topk, proposal_body_mode)
sur_ids = wild_stats.ids
sur_ms = wild_stats.ms
surrogate_steps = wild_stats.surrogate_steps
risk_rows = wild_stats.risk_rows

match_prefix = 0
limit = Math.min(exact_ids.size, sur_ids.size)
limit.times do |i|
  break unless exact_ids[i] == sur_ids[i]
  match_prefix += 1
end
same_count = 0
limit.times { |i| same_count += 1 if exact_ids[i] == sur_ids[i] }

puts "collect_ms=#{collect_ms.round(3)} collect_ms_per_token=#{(collect_ms / gen).round(3)}"
puts "heldout_top1=#{heldout_top1}/#{heldout_n} heldout_top5=#{heldout_top5}/#{heldout_n}"
puts "exact_ms=#{exact_ms.round(3)} exact_ms_per_token=#{(exact_ms / wild_n).round(3)}"
puts "surrogate_wild_ms=#{sur_ms.round(3)} surrogate_wild_ms_per_token=#{(sur_ms / wild_n).round(3)} surrogate_steps=#{surrogate_steps}"
if surrogate_steps > 0
  proposal_per_step = wild_stats.proposal_ms / surrogate_steps
  verifier_per_step = wild_stats.verifier_ms / surrogate_steps
  exact_per_token = exact_ms / wild_n
  puts "economics_current proposal_ms=#{wild_stats.proposal_ms.round(3)} proposal_ms_per_step=#{proposal_per_step.round(3)} verifier_ms=#{wild_stats.verifier_ms.round(3)} verifier_ms_per_step=#{verifier_per_step.round(3)} exact_ms_per_token=#{exact_per_token.round(3)}"
  puts "proposal_phases_per_step snapshot=#{(wild_stats.snapshot_ms / surrogate_steps).round(3)} restore=#{(wild_stats.restore_ms / surrogate_steps).round(3)} partial_layers=#{(wild_stats.partial_ms / surrogate_steps).round(3)} residual=#{(wild_stats.residual_ms / surrogate_steps).round(3)} head=#{(wild_stats.head_ms / surrogate_steps).round(3)} topk=#{(wild_stats.topk_ms / surrogate_steps).round(3)}"
end
puts "token_match_prefix=#{match_prefix}/#{limit} token_match_count=#{same_count}/#{limit}"
puts "collect_exact_ids=#{collected_exact_ids.join(',')}"
puts "exact_ids=#{exact_ids.join(',')}"
puts "surrogate_ids=#{sur_ids.join(',')}"

unless risk_rows.empty?
  correct = risk_rows.count { |r| r.exact_top1 == r.surrogate_top1 }
  top5_hits = risk_rows.count(&.top5_contains_exact)
  chosen_correct = risk_rows.count { |r| r.exact_top1 == r.chosen_token }
  rescued = risk_rows.count(&.oracle_rescued)
  fallback_rows = risk_rows.size - rescued
  correct_margins = risk_rows.select { |r| r.exact_top1 == r.surrogate_top1 }.map(&.margin)
  wrong_margins = risk_rows.select { |r| r.exact_top1 != r.surrogate_top1 }.map(&.margin)
  avg_correct_margin = correct_margins.empty? ? 0.0 : correct_margins.sum / correct_margins.size
  avg_wrong_margin = wrong_margins.empty? ? 0.0 : wrong_margins.sum / wrong_margins.size
  puts "risk_summary surrogate_correct=#{correct}/#{risk_rows.size} chosen_correct=#{chosen_correct}/#{risk_rows.size} surrogate_top5_exact=#{top5_hits}/#{risk_rows.size} oracle_rescued=#{rescued}/#{risk_rows.size} avg_correct_margin=#{avg_correct_margin.round(4)} avg_wrong_margin=#{avg_wrong_margin.round(4)}"
  if risk_rows.size > 0
    exact_per_token = exact_ms / wild_n
    proposal_per_step = wild_stats.proposal_ms / Math.max(surrogate_steps, 1)
    verifier_per_step = wild_stats.verifier_ms / Math.max(surrogate_steps, 1)
    rescue_rate = rescued.to_f64 / risk_rows.size
    fallback_rate = fallback_rows.to_f64 / risk_rows.size
    one_step_no_overlap = proposal_per_step + verifier_per_step
    # This optimistic lower bound assumes accepted candidate rows can avoid a
    # future exact-token pass and only misses pay exact fallback. It is not a
    # deployable model; it is a quick sanity check for whether this branch is
    # worth a real batched verifier.
    optimistic_candidate_ms = proposal_per_step + fallback_rate * exact_per_token
    puts "economics_oracle rescue_rate=#{rescue_rate.round(4)} fallback_rate=#{fallback_rate.round(4)} one_step_no_overlap_ms=#{one_step_no_overlap.round(3)} optimistic_candidate_ms=#{optimistic_candidate_ms.round(3)} optimistic_speedup_vs_exact=#{(exact_per_token / optimistic_candidate_ms).round(4)}"
    print_gate_sweep("margin_ge", risk_rows, proposal_per_step, verifier_per_step, exact_per_token, risk_thresholds) do |r, threshold|
      r.margin >= threshold
    end
    rank_thresholds = [1.0_f32, 2.0_f32, 3.0_f32, 5.0_f32, 8.0_f32, 16.0_f32]
    print_gate_sweep("oracle_rank_le", risk_rows, proposal_per_step, verifier_per_step, exact_per_token, rank_thresholds) do |r, threshold|
      r.exact_rank <= threshold.to_i
    end
  end
  puts "risk_threshold\taccepted\tcorrect\twrong\tprecision"
  risk_thresholds.each do |threshold|
    accepted = risk_rows.select { |r| r.margin >= threshold }
    accepted_correct = accepted.count { |r| r.exact_top1 == r.surrogate_top1 }
    accepted_wrong = accepted.size - accepted_correct
    precision = accepted.empty? ? 0.0 : accepted_correct.to_f64 / accepted.size
    puts [threshold, accepted.size, accepted_correct, accepted_wrong, precision.round(4)].join('\t')
  end
  puts "risk_rows_BEGIN"
  puts "step\texact_top1\tsurrogate_top1\tchosen_token\texact_rank_in_surrogate\tmargin\ttop5_contains_exact\toracle_rescued"
  risk_rows.each do |r|
    puts [r.step, r.exact_top1, r.surrogate_top1, r.chosen_token, r.exact_rank, r.margin.round(4), r.top5_contains_exact, r.oracle_rescued].join('\t')
  end
  puts "risk_rows_END"
end

if tok = tokenizer
  puts "exact_text_BEGIN"
  puts tok.decode(exact_ids)
  puts "exact_text_END"
  puts "surrogate_text_BEGIN"
  puts tok.decode(sur_ids)
  puts "surrogate_text_END"
end
