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
gen = 12
train = 6
max_seq = 256
prefill_chunk = 128
layers = [34, 40, 44]
rank = 8
lambda = 1.0e-3_f64
seed = 0x5eed_i64
print_exact_text = false

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_late_band_surrogate_probe [options]"
  p.on("--model PATH", "Gemma4 GGUF model path") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt TEXT", "Raw prompt text") { |v| prompt_text = v; chat_user = nil }
  p.on("--prompt-file PATH", "Read raw prompt text from file") { |v| prompt_file = v; chat_user = nil }
  p.on("--chat-user TEXT", "Format one Gemma4 user turn") { |v| chat_user = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids; bypasses tokenizer") { |v| token_ids_arg = v }
  p.on("--gen N", "Generated exact trajectory tokens to collect, default 12") { |v| gen = v.to_i }
  p.on("--train N", "Training samples from the front of the trajectory, default 6") { |v| train = v.to_i }
  p.on("--layers LIST", "Comma-separated stop layers, default 34,40,44") { |v| layers = v.split(',').reject(&.empty?).map(&.to_i) }
  p.on("--rank N", "Random-projection residual rank, default 8") { |v| rank = v.to_i }
  p.on("--lambda F", "Ridge regularization, default 1e-3") { |v| lambda = v.to_f64 }
  p.on("--max-seq N", "Resident state sequence capacity, default 256") { |v| max_seq = v.to_i }
  p.on("--prefill-chunk N", "Row prefill chunk size, default 128") { |v| prefill_chunk = v.to_i }
  p.on("--print-exact-text", "Print conservative detokenized exact trajectory") { print_exact_text = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--gen must be positive" unless gen > 0
raise "--train must be positive and smaller than gen" unless train > 0 && train < gen
raise "--rank must be positive" unless rank > 0
raise "--max-seq must be positive" unless max_seq > 0
raise "--prefill-chunk must be positive" unless prefill_chunk > 0
raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "tokenizer binary not found: #{tokenizer_bin}" unless token_ids_arg || File.exists?(tokenizer_bin)
raise "--layers must not be empty" if layers.empty?

if file = prompt_file
  prompt_text = File.read(file)
elsif user = chat_user
  prompt_text = "<|turn>user\n#{user}<turn|>\n<|turn>model\n"
end

weights = ML::GGUF::Gemma4Weights.from_gguf(model_path)
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?
layers.each do |layer|
  raise "layer #{layer} must be positive" unless layer > 0
  raise "layer #{layer} exceeds model layer count" if layer > weights.hparams.n_layer
end
hidden_dim = weights.hparams.n_embd

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
raise "prompt+gen exceeds max_seq" if ids.size + gen > max_seq

record Sample, step : Int32, input_token : Int32, exact_top1 : Int32, h_full : Array(Float32), by_layer : Hash(Int32, Array(Float32))

# Deterministic Rademacher projection. It is cheap and avoids needing a full PCA
# pass before we know whether the residual-surrogate route has any signal.
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

main_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
side_state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)

if ids.size > 1
  ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(
    weights, ids[0...-1], 0, main_state,
    chunk_size: prefill_chunk,
    stop_layer: weights.hparams.n_layer,
    read_last_hidden: false
  ).not_nil!
end

samples = [] of Sample
exact_ids = [] of Int32
current_token = ids[-1]
pos = ids.size - 1
collect_ms = 0.0

puts "model=#{File.basename(model_path)} prompt_len=#{ids.size} gen=#{gen} train=#{train} layers=#{layers.join(',')} rank=#{rank} lambda=#{lambda} max_seq=#{max_seq}"

gen.times do |step|
  t0 = Time.instant
  snapshot = ML::GGUF::Gemma4StateSnapshot.capture(main_state, prefix_len: pos.to_i32)
  by_layer = {} of Int32 => Array(Float32)
  layers.each do |layer|
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, side_state)
    by_layer[layer] = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
      weights, current_token, pos.to_i32, side_state, layer
    ).not_nil!
  end

  h_full = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(
    weights, current_token, pos.to_i32, main_state, weights.hparams.n_layer
  ).not_nil!
  logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, h_full).not_nil!
  exact = top_k_ids(logits, 1)[0]
  collect_ms += (Time.instant - t0).total_milliseconds

  samples << Sample.new(step, current_token, exact, h_full, by_layer)
  exact_ids << exact
  current_token = exact
  pos += 1
end

puts "collect_ms=#{collect_ms.round(3)} collect_ms_per_token=#{(collect_ms / gen).round(3)}"
puts "layer\ttrain\ttest\trank\ttop1_hits\ttop5_hits\ttest_n\ttop1_rate\ttop5_rate\tmean_l2_rel"

layers.each do |layer|
  xs_train = samples[0...train].map { |s| s.by_layer[layer] }
  ys_train = samples[0...train].map(&.h_full)
  model = fit_surrogate(xs_train, ys_train, rank, lambda, seed + layer)

  top1_hits = 0
  top5_hits = 0
  l2_rel_sum = 0.0
  test_n = 0
  samples[train..].each do |sample|
    x = sample.by_layer[layer]
    pred = model.predict(x)
    logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, pred).not_nil!
    top5 = top_k_ids(logits, 5)
    top1_hits += 1 if top5[0] == sample.exact_top1
    top5_hits += 1 if top5.includes?(sample.exact_top1)

    num = 0.0
    den = 0.0
    hidden_dim.times do |i|
      d = pred[i].to_f64 - sample.h_full[i].to_f64
      num += d * d
      den += sample.h_full[i].to_f64 * sample.h_full[i].to_f64
    end
    l2_rel_sum += Math.sqrt(num / Math.max(den, 1.0e-12))
    test_n += 1
  end
  top1_rate = test_n > 0 ? top1_hits.to_f64 / test_n : 0.0
  top5_rate = test_n > 0 ? top5_hits.to_f64 / test_n : 0.0
  mean_l2_rel = test_n > 0 ? l2_rel_sum / test_n : 0.0
  puts [layer, train, test_n, rank, top1_hits, top5_hits, test_n, top1_rate.round(4), top5_rate.round(4), mean_l2_rel.round(4)].join('\t')
end

if print_exact_text && tokenizer
  puts "exact_generated_text=#{tokenizer.not_nil!.decode(exact_ids).inspect}"
end
