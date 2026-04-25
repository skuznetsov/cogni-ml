#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"
DEFAULT_PROMPT    = "The quick brown fox jumps over the lazy dog. Describe this scene in detail, then explain how weather, geometry, and memory interact in a compact machine learning runtime. " \
                    "Use precise technical language and include several short code-like phrases so the token stream is varied."

private struct RecurrentSample
  getter q : Array(Float32)
  getter k : Array(Float32)
  getter v : Array(Float32)
  getter ghead : Array(Float32)
  getter beta : Array(Float32)

  def initialize(@q, @k, @v, @ghead, @beta)
  end
end

private def softplus(x : Float32) : Float32
  x > 20.0_f32 ? x : Math.log(1.0_f32 + Math.exp(x)).to_f32
end

private def silu!(x : Array(Float32)) : Nil
  x.size.times do |i|
    v = x[i]
    x[i] = v / (1.0_f32 + Math.exp(-v).to_f32)
  end
end

private def l2_norm_slice!(x : Array(Float32), offset : Int32, len : Int32, eps : Float32) : Nil
  ss = 0.0_f64
  len.times { |i| ss += x[offset + i].to_f64 * x[offset + i].to_f64 }
  inv = (1.0 / Math.sqrt(ss + eps.to_f64)).to_f32
  len.times { |i| x[offset + i] *= inv }
end

private def recurrent_k_vectors_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                           token_ids : Array(Int32),
                                           layer_index : Int32) : Array(Array(Array(Float64)))
  hp = weights.hparams
  target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) ||
                 raise "layer #{layer_index} is not recurrent"
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  per_head = Array.new(h_k) { [] of Array(Float64) }

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)

    layer_index.times do |il|
      case layer = weights.layers[il]
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      end
    end

    cur = ML::GGUF::Qwen35CPU.rms_norm(x, target_layer.attn_norm, hp.rms_eps)
    qkv_mixed = ML::GGUF::Qwen35CPU.qmatvec_nobias(target_layer.attn_qkv_qw, cur)

    conv_out = Array(Float32).new(qkv_dim) do |ch|
      acc = 0.0_f32
      w_base = ch * conv_k
      (conv_k - 1).times do |t|
        acc += conv_state[t * qkv_dim + ch] * target_layer.ssm_conv1d[w_base + t]
      end
      acc + qkv_mixed[ch] * target_layer.ssm_conv1d[w_base + (conv_k - 1)]
    end

    (conv_k - 2).times do |t|
      src = (t + 1) * qkv_dim
      dst = t * qkv_dim
      qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
    end
    last = (conv_k - 2) * qkv_dim
    qkv_dim.times { |ch| conv_state[last + ch] = qkv_mixed[ch] }

    silu!(conv_out)
    k_offset = h_k * s
    h_k.times { |h| l2_norm_slice!(conv_out, k_offset + h * s, s, hp.rms_eps) }

    h_k.times do |h|
      off = k_offset + h * s
      per_head[h] << Array.new(s) { |d| conv_out[off + d].to_f64 }
    end
  end

  per_head
end

private def recurrent_samples_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                         token_ids : Array(Int32),
                                         layer_index : Int32) : Array(RecurrentSample)
  hp = weights.hparams
  target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) ||
                 raise "layer #{layer_index} is not recurrent"
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  samples = [] of RecurrentSample

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)

    layer_index.times do |il|
      case layer = weights.layers[il]
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      end
    end

    cur = ML::GGUF::Qwen35CPU.rms_norm(x, target_layer.attn_norm, hp.rms_eps)
    proj = ML::GGUF::Qwen35CPU.qmatvec_many([target_layer.attn_qkv_qw, target_layer.ssm_alpha_qw, target_layer.ssm_beta_qw], cur)
    qkv_mixed = proj[0]
    alpha = proj[1]
    beta = proj[2]
    h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
    ghead = Array(Float32).new(h_v) do |i|
      Math.exp((softplus(alpha[i] + target_layer.ssm_dt_bias[i]) * target_layer.ssm_a[i]).to_f64).to_f32
    end

    conv_out = Array(Float32).new(qkv_dim) do |ch|
      acc = 0.0_f32
      w_base = ch * conv_k
      (conv_k - 1).times do |t|
        acc += conv_state[t * qkv_dim + ch] * target_layer.ssm_conv1d[w_base + t]
      end
      acc + qkv_mixed[ch] * target_layer.ssm_conv1d[w_base + (conv_k - 1)]
    end

    (conv_k - 2).times do |t|
      src = (t + 1) * qkv_dim
      dst = t * qkv_dim
      qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
    end
    last = (conv_k - 2) * qkv_dim
    qkv_dim.times { |ch| conv_state[last + ch] = qkv_mixed[ch] }

    silu!(conv_out)
    q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
    k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
    v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
    h_k.times do |h|
      l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
      l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
    end
    samples << RecurrentSample.new(q_conv, k_conv, v_conv, ghead, beta)
  end

  samples
end

private def dot(a : Array(Float64), b : Array(Float64)) : Float64
  acc = 0.0
  a.size.times { |i| acc += a[i] * b[i] }
  acc
end

private def residual_norm(v : Array(Float64), basis : Array(Array(Float64)), rank : Int32) : Float64
  residual = v.dup
  limit = Math.min(rank, basis.size)
  limit.times do |i|
    b = basis[i]
    coeff = dot(residual, b)
    residual.size.times { |d| residual[d] -= coeff * b[d] }
  end
  Math.sqrt(dot(residual, residual))
end

private def greedy_basis(vectors : Array(Array(Float64)), max_rank : Int32, eps : Float64 = 1.0e-6) : Array(Array(Float64))
  basis = [] of Array(Float64)
  vectors.each do |v|
    break if basis.size >= max_rank
    residual = v.dup
    basis.each do |b|
      coeff = dot(residual, b)
      residual.size.times { |d| residual[d] -= coeff * b[d] }
    end
    norm = Math.sqrt(dot(residual, residual))
    next if norm <= eps
    basis << residual.map { |x| x / norm }
  end
  basis
end

private def norm(v : Array(Float64)) : Float64
  Math.sqrt(dot(v, v))
end

private def orthogonalize!(v : Array(Float64), basis : Array(Array(Float64))) : Nil
  basis.each do |b|
    coeff = dot(v, b)
    v.size.times { |i| v[i] -= coeff * b[i] }
  end
end

private def covariance_matvec(vectors : Array(Array(Float64)), x : Array(Float64)) : Array(Float64)
  out = Array.new(x.size, 0.0)
  vectors.each do |sample|
    coeff = dot(sample, x)
    x.size.times { |i| out[i] += sample[i] * coeff }
  end
  out
end

private def pca_basis(vectors : Array(Array(Float64)), max_rank : Int32,
                      iters : Int32 = 24, eps : Float64 = 1.0e-7) : Array(Array(Float64))
  return [] of Array(Float64) if vectors.empty?

  dim = vectors[0].size
  basis = [] of Array(Float64)
  max_rank.times do |rank|
    # Deterministic non-random start vector. The sinusoid avoids selecting the
    # same axis repeatedly when the covariance spectrum has near ties.
    x = Array.new(dim) { |i| Math.sin((rank + 1) * (i + 1) * 0.0137) + Math.cos((rank + 3) * (i + 1) * 0.0071) }
    orthogonalize!(x, basis)
    n = norm(x)
    break if n <= eps
    dim.times { |i| x[i] /= n }

    iters.times do
      y = covariance_matvec(vectors, x)
      orthogonalize!(y, basis)
      yn = norm(y)
      break if yn <= eps
      dim.times { |i| x[i] = y[i] / yn }
    end

    y = covariance_matvec(vectors, x)
    orthogonalize!(y, basis)
    lambda = dot(x, y)
    break if lambda.abs <= eps

    # Re-normalize after the final orthogonalization step to keep residual
    # measurements comparable to greedy MGS.
    orthogonalize!(x, basis)
    xn = norm(x)
    break if xn <= eps
    basis << x.map { |v| v / xn }
  end

  basis
end

private def build_basis(vectors : Array(Array(Float64)), max_rank : Int32,
                        mode : String, pca_iters : Int32) : Array(Array(Float64))
  case mode
  when "greedy"
    greedy_basis(vectors, max_rank)
  when "pca"
    pca_basis(vectors, max_rank, pca_iters)
  else
    raise "unsupported basis mode #{mode.inspect}; expected greedy or pca"
  end
end

private def project_with_basis(v : Array(Float32), offset : Int32,
                               basis : Array(Array(Float64)), rank : Int32) : Nil
  limit = Math.min(rank, basis.size)
  s = basis[0].size
  projected = Array.new(s, 0.0)
  limit.times do |i|
    b = basis[i]
    coeff = 0.0
    s.times { |d| coeff += v[offset + d].to_f64 * b[d] }
    s.times { |d| projected[d] += coeff * b[d] }
  end
  s.times { |d| v[offset + d] = projected[d].to_f32 }
end

private def delta_stats(exact : Array(Float32), approx : Array(Float32)) : Tuple(Float64, Float64)
  sum_sq = 0.0
  max = 0.0
  exact.size.times do |i|
    d = (exact[i] - approx[i]).to_f64.abs
    sum_sq += d * d
    max = d if d > max
  end
  {Math.sqrt(sum_sq / exact.size), max}
end

private def simulate_projected_delta(samples : Array(RecurrentSample),
                                     bases : Array(Array(Array(Float64))),
                                     rank : Int32,
                                     calib_count : Int32,
                                     h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64)
  exact_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  approx_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  y_exact = Array(Float32).new(h_v * s, 0.0_f32)
  y_approx = Array(Float32).new(h_v * s, 0.0_f32)
  y_sq = 0.0
  y_max = 0.0
  y_count = 0

  samples.each_with_index do |sample, idx|
    ML::GGUF::Qwen35CPU.delta_net_step!(
      exact_state, sample.q, sample.k, sample.v, sample.ghead, sample.beta,
      y_exact, h_k, h_v, s, scale
    )

    k_approx = sample.k.dup
    if idx >= calib_count
      h_k.times { |h| project_with_basis(k_approx, h * s, bases[h], rank) }
    end
    ML::GGUF::Qwen35CPU.delta_net_step!(
      approx_state, sample.q, k_approx, sample.v, sample.ghead, sample.beta,
      y_approx, h_k, h_v, s, scale
    )

    next if idx < calib_count

    rmse, max = delta_stats(y_exact, y_approx)
    y_sq += rmse * rmse * y_exact.size
    y_max = max if max > y_max
    y_count += y_exact.size
  end

  state_rmse, state_max = delta_stats(exact_state, approx_state)
  {
    y_rmse:     y_count > 0 ? Math.sqrt(y_sq / y_count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
  }
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = DEFAULT_PROMPT
tokens_limit = 96
calib_tokens = 32
layer_index = 0
ranks = [8, 16, 32, 64, 96, 128]
thresholds = [0.05, 0.10, 0.20, 0.35, 0.50]
basis_mode = "greedy"
pca_iters = 24
simulate_delta = false

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_fixed_basis_probe [--model PATH] [--tokenizer PATH] [--prompt TEXT] [--tokens N] [--calib-tokens N] [--layer N] [--ranks LIST] [--basis greedy|pca]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--tokenizer=PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt=TEXT", "Prompt text") { |v| prompt = v }
  p.on("--tokens=N", "Max prompt tokens to use") { |v| tokens_limit = v.to_i }
  p.on("--calib-tokens=N", "Tokens used to build the fixed basis") { |v| calib_tokens = v.to_i }
  p.on("--layer=N", "Recurrent layer index to probe (default: 0)") { |v| layer_index = v.to_i }
  p.on("--ranks=LIST", "Comma-separated ranks") { |v| ranks = v.split(',').map(&.to_i) }
  p.on("--thresholds=LIST", "Comma-separated residual thresholds for pass-rate reporting") { |v| thresholds = v.split(',').map(&.to_f) }
  p.on("--basis=MODE", "Basis builder: greedy or pca (default: greedy)") { |v| basis_mode = v }
  p.on("--pca-iters=N", "Power iterations per PCA component (default: 24)") { |v| pca_iters = v.to_i }
  p.on("--simulate-delta", "Also simulate projected-K DeltaNet output/state drift") { simulate_delta = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokenizer not found: #{tokenizer_bin}" unless File.exists?(tokenizer_bin)
raise "tokens must be positive" unless tokens_limit > 0
raise "calib-tokens must be positive" unless calib_tokens > 0
raise "ranks must not be empty" if ranks.empty?
raise "pca-iters must be positive" unless pca_iters > 0

gguf = ML::GGUF::GGUFFile.new(model)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model, tokenizer_bin)
token_ids = tok.encode(prompt, add_bos_override: false)
while token_ids.size < tokens_limit
  token_ids.concat(tok.encode(prompt, add_bos_override: false))
end
token_ids = token_ids[0, tokens_limit]

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
per_head = recurrent_k_vectors_for_prompt(weights, token_ids, layer_index)
samples = simulate_delta ? recurrent_samples_for_prompt(weights, token_ids, layer_index) : [] of RecurrentSample
max_rank = ranks.max
calib_count = Math.min(calib_tokens, token_ids.size - 1)
raise "need at least one held-out token" unless calib_count > 0 && calib_count < token_ids.size

puts "Qwen35 DeltaNet fixed-basis K residual probe"
puts "model=#{File.basename(model)}"
puts "layer=#{layer_index} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_tokens=#{token_ids.size - calib_count}"
puts "heads=#{per_head.size} state_size=#{per_head[0][0].size} ranks=#{ranks.join(',')}"
puts "basis=#{basis_mode} pca_iters=#{pca_iters}; per-head basis over first calib_tokens; reports held-out L2 residual for normalized K vectors"
puts "thresholds=#{thresholds.map { |t| t.round(4) }.join(',')}"

bases = per_head.map { |vectors| build_basis(vectors[0, calib_count], max_rank, basis_mode, pca_iters) }

ranks.each do |rank|
  all_residuals = [] of Float64
  per_head.each_with_index do |vectors, head|
    basis = bases[head]
    vectors[calib_count, vectors.size - calib_count].each do |v|
      all_residuals << residual_norm(v, basis, rank)
    end
  end

  sorted = all_residuals.sort
  mean = all_residuals.sum / all_residuals.size
  p50 = sorted[sorted.size // 2]
  p90 = sorted[(sorted.size * 90 // 100).clamp(0, sorted.size - 1)]
  p99 = sorted[(sorted.size * 99 // 100).clamp(0, sorted.size - 1)]
  max = sorted[-1]
  pass = thresholds.map do |threshold|
    passed = all_residuals.count { |r| r <= threshold }
    "#{threshold.round(4)}:#{(100.0 * passed / all_residuals.size).round(2)}%"
  end
  line = "rank=#{rank} mean_residual=#{mean.round(6)} p50=#{p50.round(6)} p90=#{p90.round(6)} p99=#{p99.round(6)} max=#{max.round(6)} pass_rates=#{pass.join(',')}"
  if simulate_delta
    hp = weights.hparams
    drift = simulate_projected_delta(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " y_rmse=#{drift[:y_rmse].round(6)} y_max=#{drift[:y_max].round(6)} state_rmse=#{drift[:state_rmse].round(6)} state_max=#{drift[:state_max].round(6)}"
  end
  puts line
end
