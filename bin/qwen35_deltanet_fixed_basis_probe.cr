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

private alias BasisSet = Array(Array(Array(Float64)))
private alias LayerBasisMap = Hash(Int32, BasisSet)

private struct RecurrentSample
  getter q : Array(Float32)
  getter k : Array(Float32)
  getter v : Array(Float32)
  getter ghead : Array(Float32)
  getter beta : Array(Float32)

  def initialize(@q, @k, @v, @ghead, @beta)
  end
end

private class LowRankState
  property initialized : Bool = false
  property full_state_current : Bool = true
  property m : Array(Float32) = [] of Float32
  property approx_steps : Int32 = 0
  property fallback_steps : Int32 = 0
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

private def residual_norm_f32(v : Array(Float32), offset : Int32,
                              basis : Array(Array(Float64)), rank : Int32) : Float64
  limit = Math.min(rank, basis.size)
  s = basis[0].size
  residual = Array.new(s) { |d| v[offset + d].to_f64 }
  limit.times do |i|
    b = basis[i]
    coeff = dot(residual, b)
    s.times { |d| residual[d] -= coeff * b[d] }
  end
  Math.sqrt(dot(residual, residual))
end

private def max_k_residual(k_conv : Array(Float32), bases : BasisSet, rank : Int32,
                           h_k : Int32, s : Int32) : Float64
  max = 0.0
  h_k.times do |h|
    residual = residual_norm_f32(k_conv, h * s, bases[h], rank)
    max = residual if residual > max
  end
  max
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

private def basis_coeffs(v : Array(Float32), offset : Int32,
                         basis : Array(Array(Float64)), rank : Int32) : Array(Float32)
  limit = Math.min(rank, basis.size)
  Array.new(limit) do |i|
    b = basis[i]
    coeff = 0.0
    b.size.times { |d| coeff += v[offset + d].to_f64 * b[d] }
    coeff.to_f32
  end
end

private def lowrank_projected_delta_step!(m_state : Array(Float32),
                                          sample : RecurrentSample,
                                          bases : Array(Array(Array(Float64))),
                                          rank : Int32,
                                          y : Array(Float32),
                                          h_k : Int32, h_v : Int32, s : Int32,
                                          scale : Float32) : Nil
  h_v.times do |h|
    k_head = h % h_k
    basis = bases[k_head]
    r = Math.min(rank, basis.size)
    q_off = k_head * s
    k_off = k_head * s
    v_off = h * s
    st_base = h * s * rank
    gh = sample.ghead[h]
    bh = sample.beta[h]
    c = basis_coeffs(sample.k, k_off, basis, r)
    qbar = basis_coeffs(sample.q, q_off, basis, r)

    s.times do |row|
      row_off = st_base + row * rank
      r.times { |j| m_state[row_off + j] *= gh }

      sk = 0.0_f32
      r.times { |j| sk += m_state[row_off + j] * c[j] }
      delt = bh * (sample.v[v_off + row] - sk)
      r.times { |j| m_state[row_off + j] += delt * c[j] }

      acc = 0.0_f32
      r.times { |j| acc += m_state[row_off + j] * qbar[j] }
      y[h * s + row] = acc * scale
    end
  end
end

private def reconstruct_lowrank_state(m_state : Array(Float32),
                                      bases : Array(Array(Array(Float64))),
                                      rank : Int32,
                                      h_k : Int32, h_v : Int32, s : Int32) : Array(Float32)
  out = Array(Float32).new(h_v * s * s, 0.0_f32)
  h_v.times do |h|
    basis = bases[h % h_k]
    r = Math.min(rank, basis.size)
    m_base = h * s * rank
    out_base = h * s * s
    s.times do |row|
      r.times do |j|
        coeff = m_state[m_base + row * rank + j].to_f64
        b = basis[j]
        s.times { |d| out[out_base + row * s + d] += (coeff * b[d]).to_f32 }
      end
    end
  end
  out
end

private def simulate_lowrank_projected_delta(samples : Array(RecurrentSample),
                                             bases : Array(Array(Array(Float64))),
                                             rank : Int32,
                                             calib_count : Int32,
                                             h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(exact_y_rmse: Float64, exact_y_max: Float64, proof_y_rmse: Float64, proof_y_max: Float64, proof_state_rmse: Float64, proof_state_max: Float64)
  exact_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  projected_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  lowrank_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  y_exact = Array(Float32).new(h_v * s, 0.0_f32)
  y_projected = Array(Float32).new(h_v * s, 0.0_f32)
  y_lowrank = Array(Float32).new(h_v * s, 0.0_f32)
  exact_sq = 0.0
  proof_sq = 0.0
  exact_max = 0.0
  proof_max = 0.0
  count = 0

  samples.each_with_index do |sample, idx|
    ML::GGUF::Qwen35CPU.delta_net_step!(
      exact_state, sample.q, sample.k, sample.v, sample.ghead, sample.beta,
      y_exact, h_k, h_v, s, scale
    )

    k_projected = sample.k.dup
    h_k.times { |h| project_with_basis(k_projected, h * s, bases[h], rank) }
    ML::GGUF::Qwen35CPU.delta_net_step!(
      projected_state, sample.q, k_projected, sample.v, sample.ghead, sample.beta,
      y_projected, h_k, h_v, s, scale
    )

    lowrank_projected_delta_step!(
      lowrank_state, sample, bases, rank, y_lowrank, h_k, h_v, s, scale
    )

    next if idx < calib_count

    exact_rmse, exact_step_max = delta_stats(y_exact, y_projected)
    proof_rmse, proof_step_max = delta_stats(y_projected, y_lowrank)
    exact_sq += exact_rmse * exact_rmse * y_exact.size
    proof_sq += proof_rmse * proof_rmse * y_exact.size
    exact_max = exact_step_max if exact_step_max > exact_max
    proof_max = proof_step_max if proof_step_max > proof_max
    count += y_exact.size
  end

  reconstructed = reconstruct_lowrank_state(lowrank_state, bases, rank, h_k, h_v, s)
  proof_state_rmse, proof_state_max = delta_stats(projected_state, reconstructed)
  {
    exact_y_rmse:     count > 0 ? Math.sqrt(exact_sq / count) : 0.0,
    exact_y_max:      exact_max,
    proof_y_rmse:     count > 0 ? Math.sqrt(proof_sq / count) : 0.0,
    proof_y_max:      proof_max,
    proof_state_rmse: proof_state_rmse,
    proof_state_max:  proof_state_max,
  }
end

private def project_full_state_to_lowrank(full_state : Array(Float32),
                                          bases : Array(Array(Array(Float64))),
                                          rank : Int32,
                                          h_k : Int32, h_v : Int32, s : Int32) : Array(Float32)
  out = Array(Float32).new(h_v * s * rank, 0.0_f32)
  h_v.times do |h|
    basis = bases[h % h_k]
    r = Math.min(rank, basis.size)
    full_base = h * s * s
    out_base = h * s * rank
    s.times do |row|
      r.times do |j|
        b = basis[j]
        acc = 0.0
        s.times { |d| acc += full_state[full_base + row * s + d].to_f64 * b[d] }
        out[out_base + row * rank + j] = acc.to_f32
      end
    end
  end
  out
end

private def recurrent_layer_cpu_exact(inpSA : Array(Float32),
                                      lw : ML::GGUF::Qwen35RecurrentWeights,
                                      lstate : ML::GGUF::Qwen35CPU::LayerState,
                                      hp : ML::GGUF::Qwen35Hparams) : Array(Float32)
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  cur = ML::GGUF::Qwen35CPU.rms_norm(inpSA, lw.attn_norm, hp.rms_eps)
  proj = ML::GGUF::Qwen35CPU.qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
  qkv_mixed = proj[0]
  z = proj[1]
  alpha = proj[2]
  beta = proj[3]
  h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
  ghead = Array(Float32).new(h_v) { |i| Math.exp((softplus(alpha[i] + lw.ssm_dt_bias[i]) * lw.ssm_a[i]).to_f64).to_f32 }

  conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  conv_out = Array(Float32).new(qkv_dim) do |ch|
    acc = 0.0_f32
    w_base = ch * conv_k
    (conv_k - 1).times { |t| acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t] }
    acc + qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
  end
  (conv_k - 2).times do |t|
    src = (t + 1) * qkv_dim
    dst = t * qkv_dim
    qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
  end
  qkv_dim.times { |ch| conv_state[(conv_k - 2) * qkv_dim + ch] = qkv_mixed[ch] }

  silu!(conv_out)
  q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
  k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
  v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
  h_k.times do |h|
    l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
    l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
  end

  y = Array(Float32).new(h_v * s, 0.0_f32)
  state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  ML::GGUF::Qwen35CPU.delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y, h_k, h_v, s, scale)
  h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y, h * s, s, lw.ssm_norm, hp.rms_eps) }
  (h_v * s).times { |i| y[i] = y[i] * ML::GGUF::Qwen35CPU.silu(z[i]) }
  attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y)

  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inpSA[i] + attn_out[i] }
  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  ffn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
  Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
end

private def recurrent_layer_cpu_lowrank(inpSA : Array(Float32),
                                        lw : ML::GGUF::Qwen35RecurrentWeights,
                                        lstate : ML::GGUF::Qwen35CPU::LayerState,
                                        hp : ML::GGUF::Qwen35Hparams,
                                        bases : BasisSet,
                                        rank : Int32,
                                        lr_state : LowRankState,
                                        fallback_threshold : Float64? = nil) : Array(Float32)
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel

  unless lr_state.initialized
    full_state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
    lr_state.m = project_full_state_to_lowrank(full_state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.initialized = true
  end

  cur = ML::GGUF::Qwen35CPU.rms_norm(inpSA, lw.attn_norm, hp.rms_eps)
  proj = ML::GGUF::Qwen35CPU.qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
  qkv_mixed = proj[0]
  z = proj[1]
  alpha = proj[2]
  beta = proj[3]
  h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
  ghead = Array(Float32).new(h_v) { |i| Math.exp((softplus(alpha[i] + lw.ssm_dt_bias[i]) * lw.ssm_a[i]).to_f64).to_f32 }

  conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  conv_out = Array(Float32).new(qkv_dim) do |ch|
    acc = 0.0_f32
    w_base = ch * conv_k
    (conv_k - 1).times { |t| acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t] }
    acc + qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
  end
  (conv_k - 2).times do |t|
    src = (t + 1) * qkv_dim
    dst = t * qkv_dim
    qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
  end
  qkv_dim.times { |ch| conv_state[(conv_k - 2) * qkv_dim + ch] = qkv_mixed[ch] }

  silu!(conv_out)
  q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
  k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
  v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
  h_k.times do |h|
    l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
    l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
  end

  y = Array(Float32).new(h_v * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  fallback = false
  if threshold = fallback_threshold
    fallback = max_k_residual(k_conv, bases, rank, h_k, s) > threshold
  end
  if fallback
    unless lr_state.full_state_current
      lstate.ssm_state = reconstruct_lowrank_state(lr_state.m, bases, rank, h_k, h_v, s)
    end
    state = lstate.ssm_state.not_nil!
    ML::GGUF::Qwen35CPU.delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y, h_k, h_v, s, scale)
    lr_state.m = project_full_state_to_lowrank(state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.fallback_steps += 1
  else
    lowrank_projected_delta_step!(lr_state.m, RecurrentSample.new(q_conv, k_conv, v_conv, ghead, beta),
      bases, rank, y, h_k, h_v, s, scale)
    lr_state.full_state_current = false
    lr_state.approx_steps += 1
  end
  h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y, h * s, s, lw.ssm_norm, hp.rms_eps) }
  (h_v * s).times { |i| y[i] = y[i] * ML::GGUF::Qwen35CPU.silu(z[i]) }
  attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y)

  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inpSA[i] + attn_out[i] }
  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  ffn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
  Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
end

private def logits_with_target_layer(weights : ML::GGUF::Qwen35Weights,
                                     token_id : Int32,
                                     pos : Int32,
                                     state : ML::GGUF::Qwen35CPU::State,
                                     target_layer : Int32,
                                     bases : Array(Array(Array(Float64))),
                                     rank : Int32,
                                     calib_count : Int32,
                                     lr_state : LowRankState?,
                                     approximate : Bool) : Array(Float32)
  hp = weights.hparams
  x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
  weights.layers.each_with_index do |layer, il|
    case layer
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
    in ML::GGUF::Qwen35RecurrentWeights
      if il == target_layer
        x = if approximate && pos >= calib_count
              recurrent_layer_cpu_lowrank(x, layer, state.layers[il], hp, bases, rank, lr_state.not_nil!)
            else
              recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
            end
      else
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
      end
    end
  end
  x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
  ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
end

private def logits_with_lowrank_policy(weights : ML::GGUF::Qwen35Weights,
                                       token_id : Int32,
                                       pos : Int32,
                                       state : ML::GGUF::Qwen35CPU::State,
                                       layer_bases : LayerBasisMap,
                                       rank : Int32,
                                       calib_count : Int32,
                                       lr_states : Hash(Int32, LowRankState),
                                       fallback_threshold : Float64?,
                                       approximate : Bool) : Array(Float32)
  hp = weights.hparams
  x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
  weights.layers.each_with_index do |layer, il|
    case layer
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
    in ML::GGUF::Qwen35RecurrentWeights
      if bases = layer_bases[il]?
        x = if approximate && pos >= calib_count
              lr_state = lr_states[il] ||= LowRankState.new
              recurrent_layer_cpu_lowrank(x, layer, state.layers[il], hp, bases, rank, lr_state, fallback_threshold)
            else
              recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
            end
      else
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
      end
    end
  end
  x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
  ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
end

private def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  aa = 0.0
  bb = 0.0
  a.size.times do |i|
    av = a[i].to_f64
    bv = b[i].to_f64
    dot += av * bv
    aa += av * av
    bb += bv * bv
  end
  dot / Math.sqrt(aa * bb)
end

private def top1(v : Array(Float32)) : Int32
  best = 0
  best_v = v[0]
  v.each_with_index do |x, i|
    if x > best_v
      best = i
      best_v = x
    end
  end
  best.to_i32
end

private def top_k_indices(v : Array(Float32), k : Int32) : Array(Int32)
  best_i = Array(Int32).new(k, -1)
  best_v = Array(Float32).new(k, -Float32::INFINITY)
  v.each_with_index do |x, i|
    next if x <= best_v[-1]

    slot = k - 1
    while slot > 0 && x > best_v[slot - 1]
      best_v[slot] = best_v[slot - 1]
      best_i[slot] = best_i[slot - 1]
      slot -= 1
    end
    best_v[slot] = x
    best_i[slot] = i.to_i32
  end
  best_i
end

private def top1_margin(v : Array(Float32)) : Float64
  best = -Float32::INFINITY
  second = -Float32::INFINITY
  v.each do |x|
    if x > best
      second = best
      best = x
    elsif x > second
      second = x
    end
  end
  (best - second).to_f64
end

private def softmax_kl(exact : Array(Float32), approx : Array(Float32)) : Float64
  max_exact = exact.max
  max_approx = approx.max
  sum_exact = 0.0
  sum_approx = 0.0
  exact.each { |x| sum_exact += Math.exp((x - max_exact).to_f64) }
  approx.each { |x| sum_approx += Math.exp((x - max_approx).to_f64) }
  log_z_exact = max_exact.to_f64 + Math.log(sum_exact)
  log_z_approx = max_approx.to_f64 + Math.log(sum_approx)
  kl = 0.0
  exact.size.times do |i|
    log_p = exact[i].to_f64 - log_z_exact
    log_q = approx[i].to_f64 - log_z_approx
    p = Math.exp(log_p)
    kl += p * (log_p - log_q)
  end
  kl
end

private def max_abs_delta(a : Array(Float32), b : Array(Float32)) : Float64
  max = 0.0
  a.size.times do |i|
    d = (a[i] - b[i]).to_f64.abs
    max = d if d > max
  end
  max
end

private def simulate_logits(weights : ML::GGUF::Qwen35Weights,
                            token_ids : Array(Int32),
                            target_layer : Int32,
                            bases : Array(Array(Array(Float64))),
                            rank : Int32,
                            calib_count : Int32) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64)
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  lr_state = LowRankState.new
  cosines = [] of Float64
  max_delta = 0.0
  top_matches = 0
  compared = 0

  token_ids.each_with_index do |token_id, pos|
    exact = logits_with_target_layer(weights, token_id, pos.to_i32, exact_state,
      target_layer, bases, rank, calib_count, nil, false)
    approx = logits_with_target_layer(weights, token_id, pos.to_i32, approx_state,
      target_layer, bases, rank, calib_count, lr_state, true)
    next if pos < calib_count

    c = cosine(exact, approx)
    cosines << c
    d = max_abs_delta(exact, approx)
    max_delta = d if d > max_delta
    top_matches += 1 if top1(exact) == top1(approx)
    compared += 1
  end

  {
    mean_cos:   cosines.sum / cosines.size,
    min_cos:    cosines.min,
    max_delta:  max_delta,
    top1_match: 100.0 * top_matches / compared,
  }
end

private def simulate_logits_policy(weights : ML::GGUF::Qwen35Weights,
                                   token_ids : Array(Int32),
                                   layer_bases : LayerBasisMap,
                                   rank : Int32,
                                   calib_count : Int32,
                                   fallback_threshold : Float64?) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_steps: Int32, fallback_steps: Int32)
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  lr_states = {} of Int32 => LowRankState
  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  compared = 0

  token_ids.each_with_index do |token_id, pos|
    exact = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, false)
    approx = logits_with_lowrank_policy(weights, token_id, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, true)
    next if pos < calib_count

    c = cosine(exact, approx)
    cosines << c
    d = max_abs_delta(exact, approx)
    max_delta = d if d > max_delta
    exact_top1 = top1(exact)
    approx_top1 = top1(approx)
    exact_margin = top1_margin(exact)
    min_margin = exact_margin if exact_margin < min_margin
    if exact_top1 == approx_top1
      top_matches += 1
    elsif exact_margin >= 0.5
      confident_mismatches += 1
    end
    top5_hits += 1 if top_k_indices(approx, 5).includes?(exact_top1)
    kls << softmax_kl(exact, approx)
    compared += 1
  end

  {
    mean_cos:             cosines.sum / cosines.size,
    min_cos:              cosines.min,
    max_delta:            max_delta,
    top1_match:           100.0 * top_matches / compared,
    top5_hit:             100.0 * top5_hits / compared,
    mean_kl:              kls.sum / kls.size,
    max_kl:               kls.max,
    min_margin:           min_margin,
    confident_mismatches: confident_mismatches,
    approx_steps:         lr_states.values.sum(&.approx_steps),
    fallback_steps:       lr_states.values.sum(&.fallback_steps),
  }
end

private def simulate_greedy_policy(weights : ML::GGUF::Qwen35Weights,
                                   prompt_ids : Array(Int32),
                                   gen_tokens : Int32,
                                   layer_bases : LayerBasisMap,
                                   rank : Int32,
                                   calib_count : Int32,
                                   fallback_threshold : Float64?) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_steps: Int32, fallback_steps: Int32, exact_ids: Array(Int32), approx_ids: Array(Int32))
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  approx_logits = [] of Float32

  prompt_ids.each_with_index do |token_id, pos|
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, false)
    approx_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, true)
  end

  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  exact_ids = [] of Int32
  approx_ids = [] of Int32

  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    approx_top1 = top1(approx_logits)
    exact_ids << exact_top1
    approx_ids << approx_top1

    c = cosine(exact_logits, approx_logits)
    cosines << c
    d = max_abs_delta(exact_logits, approx_logits)
    max_delta = d if d > max_delta
    exact_margin = top1_margin(exact_logits)
    min_margin = exact_margin if exact_margin < min_margin
    if exact_top1 == approx_top1
      top_matches += 1
    elsif exact_margin >= 0.5
      confident_mismatches += 1
    end
    top5_hits += 1 if top_k_indices(approx_logits, 5).includes?(exact_top1)
    kls << softmax_kl(exact_logits, approx_logits)

    pos = prompt_ids.size + step
    # Teacher-forced on the exact greedy trajectory. This isolates policy drift
    # from cascading different-token hidden-state divergence.
    exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, false)
    approx_logits = logits_with_lowrank_policy(weights, exact_top1, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, true)
  end

  {
    mean_cos:             cosines.sum / cosines.size,
    min_cos:              cosines.min,
    max_delta:            max_delta,
    top1_match:           100.0 * top_matches / gen_tokens,
    top5_hit:             100.0 * top5_hits / gen_tokens,
    mean_kl:              kls.sum / kls.size,
    max_kl:               kls.max,
    min_margin:           min_margin,
    confident_mismatches: confident_mismatches,
    approx_steps:         lr_states.values.sum(&.approx_steps),
    fallback_steps:       lr_states.values.sum(&.fallback_steps),
    exact_ids:            exact_ids,
    approx_ids:           approx_ids,
  }
end

private def parse_int_list(value : String) : Array(Int32)
  value.split(',').map(&.strip).reject(&.empty?).map(&.to_i)
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
simulate_lowrank = false
simulate_logit_rank : Int32? = nil
simulate_logit_layers = [] of Int32
simulate_fallback_threshold : Float64? = nil
simulate_generate_tokens = 0

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
  p.on("--simulate-lowrank", "Also prove low-rank M*B^T recurrence against full projected-K recurrence") { simulate_lowrank = true }
  p.on("--simulate-logits-rank=N", "Run full-model logit drift gate for one rank") { |v| simulate_logit_rank = v.to_i }
  p.on("--simulate-logits-layers=LIST", "Comma-separated recurrent layers to approximate together during the logit drift gate") { |v| simulate_logit_layers = parse_int_list(v) }
  p.on("--simulate-fallback-threshold=F", "Fallback to exact DeltaNet step when max per-head K residual exceeds F") { |v| simulate_fallback_threshold = v.to_f64 }
  p.on("--simulate-generate=N", "Run teacher-forced exact-greedy generation drift gate for N decode tokens") { |v| simulate_generate_tokens = v.to_i }
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
samples = (simulate_delta || simulate_lowrank) ? recurrent_samples_for_prompt(weights, token_ids, layer_index) : [] of RecurrentSample
max_rank = ranks.max
if rank = simulate_logit_rank
  max_rank = Math.max(max_rank, rank)
end
calib_count = Math.min(calib_tokens, token_ids.size - 1)
raise "need at least one held-out token" unless calib_count > 0 && calib_count < token_ids.size

puts "Qwen35 DeltaNet fixed-basis K residual probe"
puts "model=#{File.basename(model)}"
puts "layer=#{layer_index} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_tokens=#{token_ids.size - calib_count}"
puts "heads=#{per_head.size} state_size=#{per_head[0][0].size} ranks=#{ranks.join(',')}"
puts "basis=#{basis_mode} pca_iters=#{pca_iters}; per-head basis over first calib_tokens; reports held-out L2 residual for normalized K vectors"
puts "thresholds=#{thresholds.map { |t| t.round(4) }.join(',')}"

bases = per_head.map { |vectors| build_basis(vectors[0, calib_count], max_rank, basis_mode, pca_iters) }

if rank = simulate_logit_rank
  if simulate_logit_layers.empty?
    logit = simulate_logits(weights, token_ids, layer_index, bases, rank, calib_count)
    puts "logit_drift rank=#{rank} mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} max_delta=#{logit[:max_delta].round(6)} top1_match=#{logit[:top1_match].round(2)}%"
  else
    layer_bases = {} of Int32 => BasisSet
    simulate_logit_layers.uniq.each do |il|
      layer_bases[il] = if il == layer_index
                          bases
                        else
                          recurrent_k_vectors_for_prompt(weights, token_ids, il).map do |vectors|
                            build_basis(vectors[0, calib_count], max_rank, basis_mode, pca_iters)
                          end
                        end
    end
    logit = simulate_logits_policy(weights, token_ids, layer_bases, rank, calib_count, simulate_fallback_threshold)
    total_steps = logit[:approx_steps] + logit[:fallback_steps]
    approx_rate = total_steps > 0 ? (100.0 * logit[:approx_steps] / total_steps) : 0.0
    fallback_note = simulate_fallback_threshold ? " fallback_threshold=#{simulate_fallback_threshold} approx_rate=#{approx_rate.round(2)}%" : ""
    puts "logit_drift_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} max_delta=#{logit[:max_delta].round(6)} top1_match=#{logit[:top1_match].round(2)}% top5_hit=#{logit[:top5_hit].round(2)}% mean_kl=#{logit[:mean_kl].round(8)} max_kl=#{logit[:max_kl].round(8)} min_margin=#{logit[:min_margin].round(6)} confident_mismatches=#{logit[:confident_mismatches]} approx_steps=#{logit[:approx_steps]} fallback_steps=#{logit[:fallback_steps]}#{fallback_note}"

    if simulate_generate_tokens > 0
      gen = simulate_greedy_policy(weights, token_ids, simulate_generate_tokens, layer_bases, rank, calib_count, simulate_fallback_threshold)
      gen_total_steps = gen[:approx_steps] + gen[:fallback_steps]
      gen_approx_rate = gen_total_steps > 0 ? (100.0 * gen[:approx_steps] / gen_total_steps) : 0.0
      puts "greedy_drift_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} gen_tokens=#{simulate_generate_tokens} mean_cos=#{gen[:mean_cos].round(8)} min_cos=#{gen[:min_cos].round(8)} max_delta=#{gen[:max_delta].round(6)} top1_match=#{gen[:top1_match].round(2)}% top5_hit=#{gen[:top5_hit].round(2)}% mean_kl=#{gen[:mean_kl].round(8)} max_kl=#{gen[:max_kl].round(8)} min_margin=#{gen[:min_margin].round(6)} confident_mismatches=#{gen[:confident_mismatches]} approx_steps=#{gen[:approx_steps]} fallback_steps=#{gen[:fallback_steps]} approx_rate=#{gen_approx_rate.round(2)}% exact_ids=#{gen[:exact_ids].join(',')} approx_ids=#{gen[:approx_ids].join(',')}"
    end
  end
end

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
  if simulate_lowrank
    hp = weights.hparams
    lr = simulate_lowrank_projected_delta(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_exact_y_rmse=#{lr[:exact_y_rmse].round(6)} lr_exact_y_max=#{lr[:exact_y_max].round(6)} lr_proof_y_rmse=#{lr[:proof_y_rmse].round(8)} lr_proof_y_max=#{lr[:proof_y_max].round(8)} lr_proof_state_rmse=#{lr[:proof_state_rmse].round(8)} lr_proof_state_max=#{lr[:proof_state_max].round(8)}"
  end
  puts line
end
