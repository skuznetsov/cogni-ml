require "option_parser"
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
record RiskRow, step : Int32, exact_top1 : Int32, surrogate_top1 : Int32, exact_rank : Int32, margin : Float32, top5_contains_exact : Bool

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
                            max_seq : Int32, prefill_chunk : Int32,
                            diagnose_risk : Bool) : {Array(Int32), Float64, Int32, Array(RiskRow)}
  main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  side_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  prefill_prefix!(weights, ids, main_state, prefill_chunk)

  out = [] of Int32
  current_token = ids[-1]
  pos = ids.size - 1
  surrogate_steps = 0
  risk_rows = [] of RiskRow
  t0 = Time.instant

  steps.times do |step|
    if step < warmup_n
      next_token = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(
        weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
      ).not_nil!
      out << next_token
      current_token = next_token
      pos += 1
      next
    end

    snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state)
    h_layer = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
      weights, current_token, pos.to_i32, side_state, surrogate_layer
    ).not_nil!
    h_hat = surrogate.predict(h_layer)
    logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_hat).not_nil!
    top5 = top_k_ids(logits, 5)
    next_token = top5[0]

    # Autoregressive invariant: the KV state at position `pos` belongs to the
    # consumed current token, while `next_token` is only chosen for the next
    # position. This updates the exact state boundary for the generated text but
    # intentionally ignores the exact model's preferred next token.
    if diagnose_risk
      h_full = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
        weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
      ).not_nil!
      exact_logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_full).not_nil!
      exact_top1 = top_k_ids(exact_logits, 1)[0]
      margin = logits[top5[0]] - logits[top5[1]]
      risk_rows << RiskRow.new(step.to_i32, exact_top1, next_token, rank_of_id(logits, exact_top1), margin, top5.includes?(exact_top1))
    else
      ML::GGUF::Gemma4Metal.forward_resident_cache_wave_no_read(
        weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
      )
    end
    out << next_token
    current_token = next_token
    pos += 1
    surrogate_steps += 1
  end

  {out, (Time.instant - t0).total_milliseconds, surrogate_steps, risk_rows}
end

puts "model=#{File.basename(model_path)} prompt_len=#{ids.size} gen=#{gen} train=#{train} wild_gen=#{wild_n} layer=#{surrogate_layer} rank=#{rank} lambda=#{lambda} warmup_exact=#{warmup_n} diagnose_risk=#{diagnose_risk} max_seq=#{max_seq}"

samples, collected_exact_ids, collect_ms = collect_exact_samples(weights, ids, gen, surrogate_layer, max_seq, prefill_chunk)
xs_train = samples[0...train].map(&.h_layer)
ys_train = samples[0...train].map(&.h_full)
surrogate = fit_surrogate(xs_train, ys_train, rank, lambda, seed + surrogate_layer)

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
sur_ids, sur_ms, surrogate_steps, risk_rows = generate_surrogate_wild(weights, ids, wild_n, warmup_n, surrogate_layer, surrogate, max_seq, prefill_chunk, diagnose_risk)

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
puts "token_match_prefix=#{match_prefix}/#{limit} token_match_count=#{same_count}/#{limit}"
puts "collect_exact_ids=#{collected_exact_ids.join(',')}"
puts "exact_ids=#{exact_ids.join(',')}"
puts "surrogate_ids=#{sur_ids.join(',')}"

unless risk_rows.empty?
  correct = risk_rows.count { |r| r.exact_top1 == r.surrogate_top1 }
  top5_hits = risk_rows.count(&.top5_contains_exact)
  correct_margins = risk_rows.select { |r| r.exact_top1 == r.surrogate_top1 }.map(&.margin)
  wrong_margins = risk_rows.select { |r| r.exact_top1 != r.surrogate_top1 }.map(&.margin)
  avg_correct_margin = correct_margins.empty? ? 0.0 : correct_margins.sum / correct_margins.size
  avg_wrong_margin = wrong_margins.empty? ? 0.0 : wrong_margins.sum / wrong_margins.size
  puts "risk_summary surrogate_correct=#{correct}/#{risk_rows.size} surrogate_top5_exact=#{top5_hits}/#{risk_rows.size} avg_correct_margin=#{avg_correct_margin.round(4)} avg_wrong_margin=#{avg_wrong_margin.round(4)}"
  puts "risk_threshold\taccepted\tcorrect\twrong\tprecision"
  risk_thresholds.each do |threshold|
    accepted = risk_rows.select { |r| r.margin >= threshold }
    accepted_correct = accepted.count { |r| r.exact_top1 == r.surrogate_top1 }
    accepted_wrong = accepted.size - accepted_correct
    precision = accepted.empty? ? 0.0 : accepted_correct.to_f64 / accepted.size
    puts [threshold, accepted.size, accepted_correct, accepted_wrong, precision.round(4)].join('\t')
  end
  puts "risk_rows_BEGIN"
  puts "step\texact_top1\tsurrogate_top1\texact_rank_in_surrogate\tmargin\ttop5_contains_exact"
  risk_rows.each do |r|
    puts [r.step, r.exact_top1, r.surrogate_top1, r.exact_rank, r.margin.round(4), r.top5_contains_exact].join('\t')
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
