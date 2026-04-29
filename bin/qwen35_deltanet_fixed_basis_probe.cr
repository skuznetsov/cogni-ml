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
DEFAULT_SELF_SPEC_GPU_PIPELINE_DRAFT_BLOCK_TOKENS = 1

private alias BasisSet = Array(Array(Array(Float64)))
private alias LayerBasisMap = Hash(Int32, BasisSet)
private alias FFNBasisMap = Hash(Int32, Array(Array(Float64)))
private alias HybridRoute = NamedTuple(name: String, noffn: Set(Int32)?, updown: Set(Int32)?)

private struct RecurrentSample
  getter inp : Array(Float32)
  getter q : Array(Float32)
  getter k : Array(Float32)
  getter v : Array(Float32)
  getter ghead : Array(Float32)
  getter beta : Array(Float32)
  getter z : Array(Float32)

  def initialize(@q, @k, @v, @ghead, @beta, @z = [] of Float32, @inp = [] of Float32)
  end
end

private class LowRankState
  property initialized : Bool = false
  property full_state_current : Bool = true
  property m : Array(Float32) = [] of Float32
  property m_buf : ML::MetalBuffer?
  property basis_buf : ML::MetalBuffer?
  property basis_key : String = ""
  property updown_x_mean_buf : ML::MetalBuffer?
  property updown_c_mean_buf : ML::MetalBuffer?
  property updown_coeff_w_buf : ML::MetalBuffer?
  property updown_down_buf : ML::MetalBuffer?
  property updown_key : String = ""
  property approx_steps : Int32 = 0
  property fallback_steps : Int32 = 0
end

private class GpuDraftBlock
  getter submissions : Array(ML::GGUF::Qwen35Metal::DecodeWaveSubmission)
  getter state : ML::GGUF::Qwen35CPU::State
  getter lr_bufs : Hash(Int32, ML::MetalBuffer)
  getter use_updown : Bool

  def initialize(@submissions, @state, @lr_bufs, @use_updown)
  end
end

private struct FFNAdapter
  getter basis : Array(Array(Float64))
  getter down_basis : Array(Array(Float32))

  def initialize(@basis : Array(Array(Float64)), @down_basis : Array(Array(Float32)))
  end
end

private alias FFNAdapterMap = Hash(Int32, FFNAdapter)

private struct FFNUpDownAdapter
  getter x_mean : Array(Float64)
  getter c_mean : Array(Float64)
  getter coeff_weights : Array(Array(Float64))
  getter down_basis : Array(Array(Float32))

  def initialize(@x_mean : Array(Float64),
                 @c_mean : Array(Float64),
                 @coeff_weights : Array(Array(Float64)),
                 @down_basis : Array(Array(Float32)))
  end
end

private alias FFNUpDownAdapterMap = Hash(Int32, FFNUpDownAdapter)

private class WbaTrace
  @base : Time::Instant
  @events = [] of String
  @mutex = Mutex.new

  def initialize(@label : String)
    @base = Time.instant
  end

  def self.enabled? : Bool
    ENV["QWEN35_WBA"]? == "1"
  end

  def self.maybe(label : String) : WbaTrace?
    enabled? ? new(label) : nil
  end

  def mark(lane : String, stage : String, t0 : Time::Instant, t1 : Time::Instant) : Nil
    start_ms = (t0 - @base).total_milliseconds
    end_ms = (t1 - @base).total_milliseconds
    dur_ms = (t1 - t0).total_milliseconds
    event = sprintf("wba label=%s lane=%s stage=%s start_ms=%.3f end_ms=%.3f dur_ms=%.3f",
      @label, lane, stage, start_ms, end_ms, dur_ms)
    @mutex.synchronize { @events << event }
  end

  def point(lane : String, stage : String, t : Time::Instant) : Nil
    ms = (t - @base).total_milliseconds
    event = sprintf("wba label=%s lane=%s stage=%s at_ms=%.3f",
      @label, lane, stage, ms)
    @mutex.synchronize { @events << event }
  end

  def flush : Nil
    events = @mutex.synchronize { @events.dup }
    return if events.empty?
    events.each { |event| STDERR.puts event }
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
    proj = ML::GGUF::Qwen35CPU.qmatvec_many([target_layer.attn_qkv_qw, target_layer.attn_gate_qw, target_layer.ssm_alpha_qw, target_layer.ssm_beta_qw], cur)
    qkv_mixed = proj[0]
    z = proj[1]
    alpha = proj[2]
    beta = proj[3]
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
    samples << RecurrentSample.new(q_conv, k_conv, v_conv, ghead, beta, z, x.dup)
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

private def basis_rank_range(bases : BasisSet) : NamedTuple(min: Int32, max: Int32)
  sizes = bases.map(&.size)
  {min: sizes.min, max: sizes.max}
end

private def basis_rank_note(bases : BasisSet, requested_rank : Int32) : String
  range = basis_rank_range(bases)
  note = "effective_basis_rank=#{range[:min]}..#{range[:max]}"
  if requested_rank > range[:min]
    note += " requested_rank=#{requested_rank} note=requested_rank_exceeds_some_effective_bases"
  end
  note
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

private def sync_lowrank_state_from_metal!(lr_state : LowRankState) : Nil
  return unless buf = lr_state.m_buf
  return if lr_state.m.empty?
  lr_state.m = buf.read(lr_state.m.size)
end

private def flatten_basis_for_metal(bases : BasisSet, rank : Int32, h_k : Int32, s : Int32) : Array(Float32)
  flat = Array(Float32).new(h_k * rank * s, 0.0_f32)
  h_k.times do |h|
    basis = bases[h]
    r = Math.min(rank, basis.size)
    r.times do |j|
      s.times do |d|
        flat[(h * rank + j) * s + d] = basis[j][d].to_f32
      end
    end
  end
  flat
end

private def lowrank_basis_buffer!(lr_state : LowRankState, bases : BasisSet,
                                  rank : Int32, h_k : Int32, s : Int32) : ML::MetalBuffer
  key = "#{h_k}:#{s}:#{rank}:#{bases.object_id}"
  byte_size = (h_k * rank * s).to_i64 * sizeof(Float32)
  buf = lr_state.basis_buf
  if buf.nil? || buf.size != byte_size || lr_state.basis_key != key
    flat = flatten_basis_for_metal(bases, rank, h_k, s)
    buf = ML::MetalBuffer.new(byte_size)
    buf.write(flat)
    lr_state.basis_buf = buf
    lr_state.basis_key = key
  end
  buf
end

private def updown_adapter_buffers!(lr_state : LowRankState,
                                    adapter : FFNUpDownAdapter,
                                    rank : Int32,
                                    hidden_dim : Int32) : NamedTuple(x_mean: ML::MetalBuffer, c_mean: ML::MetalBuffer, coeff_w: ML::MetalBuffer, down: ML::MetalBuffer, rank: Int32)
  limit = Math.min(rank, adapter.coeff_weights.size)
  raise "FFN up/down adapter has no coefficient weights" unless limit > 0
  raise "FFN up/down adapter output dim mismatch" unless adapter.down_basis[0].size == hidden_dim
  key = "#{adapter.coeff_weights.object_id}:#{adapter.down_basis.object_id}:#{limit}:#{hidden_dim}"
  byte_size = (limit * hidden_dim).to_i64 * sizeof(Float32)
  needs_upload = lr_state.updown_key != key ||
                 lr_state.updown_x_mean_buf.nil? ||
                 lr_state.updown_c_mean_buf.nil? ||
                 lr_state.updown_coeff_w_buf.nil? ||
                 lr_state.updown_down_buf.nil? ||
                 lr_state.updown_coeff_w_buf.not_nil!.size != byte_size ||
                 lr_state.updown_down_buf.not_nil!.size != byte_size
  if needs_upload
    coeff_weights = Array(Float32).new(limit * hidden_dim)
    down_basis = Array(Float32).new(limit * hidden_dim)
    limit.times do |j|
      hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
      hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
    end
    lr_state.updown_x_mean_buf = ML::MetalBuffer.from_array(adapter.x_mean.map(&.to_f32))
    lr_state.updown_c_mean_buf = ML::MetalBuffer.from_array(adapter.c_mean[0, limit].map(&.to_f32))
    lr_state.updown_coeff_w_buf = ML::MetalBuffer.from_array(coeff_weights)
    lr_state.updown_down_buf = ML::MetalBuffer.from_array(down_basis)
    lr_state.updown_key = key
  end
  {
    x_mean:  lr_state.updown_x_mean_buf.not_nil!,
    c_mean:  lr_state.updown_c_mean_buf.not_nil!,
    coeff_w: lr_state.updown_coeff_w_buf.not_nil!,
    down:    lr_state.updown_down_buf.not_nil!,
    rank:    limit,
  }
end

private def build_updown_adapter_buffer_maps(adapters : FFNUpDownAdapterMap,
                                             layer_ids : Enumerable(Int32),
                                             rank : Int32,
                                             hidden_dim : Int32) : NamedTuple(x_mean: Hash(Int32, ML::MetalBuffer), c_mean: Hash(Int32, ML::MetalBuffer), coeff_w: Hash(Int32, ML::MetalBuffer), down: Hash(Int32, ML::MetalBuffer), rank: Int32)
  raise "GPU pipeline pca-updown rank must be positive" unless rank > 0
  raise "GPU pipeline pca-updown rank too large for current Metal kernel" if rank > 64

  x_mean = {} of Int32 => ML::MetalBuffer
  c_mean = {} of Int32 => ML::MetalBuffer
  coeff_w = {} of Int32 => ML::MetalBuffer
  down = {} of Int32 => ML::MetalBuffer
  actual_rank = nil.as(Int32?)
  layer_ids.each do |il|
    adapter = adapters[il]? || raise "GPU pipeline pca-updown missing adapter for layer #{il}"
    bufs = updown_adapter_buffers!(LowRankState.new, adapter, rank, hidden_dim)
    if prev_rank = actual_rank
      raise "GPU pipeline pca-updown inconsistent adapter ranks: #{prev_rank} vs #{bufs[:rank]} at layer #{il}" unless prev_rank == bufs[:rank]
    else
      actual_rank = bufs[:rank]
    end
    x_mean[il] = bufs[:x_mean]
    c_mean[il] = bufs[:c_mean]
    coeff_w[il] = bufs[:coeff_w]
    down[il] = bufs[:down]
  end
  {
    x_mean:  x_mean,
    c_mean:  c_mean,
    coeff_w: coeff_w,
    down:    down,
    rank:    actual_rank || raise("GPU pipeline pca-updown has no layers"),
  }
end

private def lowrank_state_buffer!(lr_state : LowRankState) : ML::MetalBuffer
  byte_size = lr_state.m.size.to_i64 * sizeof(Float32)
  buf = lr_state.m_buf
  if buf.nil? || buf.size != byte_size
    buf = ML::MetalBuffer.new(byte_size)
    lr_state.m_buf = buf
    lr_state.full_state_current = true
  end
  buf.write(lr_state.m) if lr_state.full_state_current
  buf
end

private def lowrank_projected_delta_step_metal!(lr_state : LowRankState,
                                                sample : RecurrentSample,
                                                bases : Array(Array(Array(Float64))),
                                                rank : Int32,
                                                y : Array(Float32),
                                                h_k : Int32, h_v : Int32, s : Int32,
                                                scale : Float32,
                                                project_coeffs_on_gpu : Bool = false) : Nil
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?

  buf = lowrank_state_buffer!(lr_state)

  y_metal = if project_coeffs_on_gpu
              basis_buf = lowrank_basis_buffer!(lr_state, bases, rank, h_k, s)
              ML::GGUF::Qwen35Metal.lowrank_delta_step_projected_buf(buf, sample.q, sample.k, basis_buf, sample.v, sample.ghead, sample.beta,
                h_k, h_v, s, rank, scale)
            else
              c = Array(Float32).new(h_k * rank, 0.0_f32)
              qbar = Array(Float32).new(h_k * rank, 0.0_f32)
              h_k.times do |h|
                basis = bases[h]
                r = Math.min(rank, basis.size)
                k_coeffs = basis_coeffs(sample.k, h * s, basis, r)
                q_coeffs = basis_coeffs(sample.q, h * s, basis, r)
                r.times do |j|
                  c[h * rank + j] = k_coeffs[j]
                  qbar[h * rank + j] = q_coeffs[j]
                end
              end
              ML::GGUF::Qwen35Metal.lowrank_delta_step(buf, c, qbar, sample.v, sample.ghead, sample.beta,
                h_k, h_v, s, rank, scale)
            end
  y_metal.each_with_index { |v, i| y[i] = v }
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

private def simulate_lowrank_projected_delta_metal(samples : Array(RecurrentSample),
                                                   bases : Array(Array(Array(Float64))),
                                                   rank : Int32,
                                                   calib_count : Int32,
                                                   h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)

  y_sq = 0.0
  y_max = 0.0
  count = 0
  steps = 0
  cpu_ms = 0.0
  metal_ms = 0.0
  samples[calib_count, samples.size - calib_count].each do |sample|
    c = Array(Float32).new(h_k * rank, 0.0_f32)
    qbar = Array(Float32).new(h_k * rank, 0.0_f32)
    h_k.times do |h|
      basis = bases[h]
      r = Math.min(rank, basis.size)
      k_coeffs = basis_coeffs(sample.k, h * s, basis, r)
      q_coeffs = basis_coeffs(sample.q, h * s, basis, r)
      r.times do |j|
        c[h * rank + j] = k_coeffs[j]
        qbar[h * rank + j] = q_coeffs[j]
      end
    end

    t_cpu = Time.instant
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    cpu_ms += (Time.instant - t_cpu).total_milliseconds
    t_metal = Time.instant
    y_metal = ML::GGUF::Qwen35Metal.lowrank_delta_step(metal_buf, c, qbar, sample.v, sample.ghead, sample.beta, h_k, h_v, s, rank, scale)
    metal_ms += (Time.instant - t_metal).total_milliseconds
    y_cpu.each_with_index do |v, i|
      e = (v - y_metal[i]).abs.to_f64
      y_sq += e * e
      y_max = e if e > y_max
      count += 1
    end
    steps += 1
  end

  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  {
    y_rmse:     count > 0 ? Math.sqrt(y_sq / count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      steps,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_projected_delta_metal_project(samples : Array(RecurrentSample),
                                                           bases : BasisSet,
                                                           rank : Int32,
                                                           calib_count : Int32,
                                                           h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank projected delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)

  y_sq = 0.0
  y_max = 0.0
  count = 0
  steps = 0
  cpu_ms = 0.0
  metal_ms = 0.0
  samples[calib_count, samples.size - calib_count].each do |sample|
    t_cpu = Time.instant
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    cpu_ms += (Time.instant - t_cpu).total_milliseconds

    t_metal = Time.instant
    y_metal = ML::GGUF::Qwen35Metal.lowrank_delta_step_projected_buf(metal_buf, sample.q, sample.k, basis_buf, sample.v, sample.ghead, sample.beta,
      h_k, h_v, s, rank, scale)
    metal_ms += (Time.instant - t_metal).total_milliseconds

    y_cpu.each_with_index do |v, i|
      e = (v - y_metal[i]).abs.to_f64
      y_sq += e * e
      y_max = e if e > y_max
      count += 1
    end
    steps += 1
  end

  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  {
    y_rmse:     count > 0 ? Math.sqrt(y_sq / count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      steps,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_projected_delta_metal_chunk(samples : Array(RecurrentSample),
                                                         bases : BasisSet,
                                                         rank : Int32,
                                                         calib_count : Int32,
                                                         h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)

  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)

  heldout.each_with_index do |sample, t|
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times { |i| v_all[t * h_v * s + i] = sample.v[i] }
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    y_cpu.each_with_index { |v, i| y_cpu_all[t * h_v * s + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  y_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_buf(metal_buf, q_all, k_all, basis_buf, v_all, g_all, b_all,
    h_k, h_v, s, rank, n_tokens, scale)
  metal_ms = (Time.instant - t_metal).total_milliseconds

  y_sq = 0.0
  y_max = 0.0
  y_cpu_all.each_with_index do |v, i|
    e = (v - y_metal_all[i]).abs.to_f64
    y_sq += e * e
    y_max = e if e > y_max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = y_cpu_all.size
  {
    y_rmse:     count > 0 ? Math.sqrt(y_sq / count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_projected_delta_metal_chunk_out(samples : Array(RecurrentSample),
                                                             bases : BasisSet,
                                                             out_qw : ML::GGUF::QuantWeight,
                                                             ssm_norm : Array(Float32),
                                                             eps : Float32,
                                                             rank : Int32,
                                                             calib_count : Int32,
                                                             h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(out_rmse: Float64, out_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  out_cpu_all = Array(Float32).new(n_tokens * out_qw.out_dim, 0.0_f32)

  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample z missing for chunk_out proof" if sample.z.empty?
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, ssm_norm, eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    out = ML::GGUF::Qwen35CPU.qmatvec_nobias(out_qw, y_cpu)
    out.each_with_index { |v, i| out_cpu_all[t * out_qw.out_dim + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  out_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_out_buf(metal_buf, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    ssm_norm, out_qw, h_k, h_v, s, rank, n_tokens, eps, scale).not_nil!
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  out_cpu_all.each_with_index do |v, i|
    e = (v - out_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = out_cpu_all.size
  {
    out_rmse:   count > 0 ? Math.sqrt(sq / count) : 0.0,
    out_max:    max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def finish_recurrent_layer_cpu(inp : Array(Float32),
                                       attn_out : Array(Float32),
                                       lw : ML::GGUF::Qwen35RecurrentWeights,
                                       hp : ML::GGUF::Qwen35Hparams) : Array(Float32)
  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inp[i] + attn_out[i] }
  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  ffn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
  Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
end

private def simulate_lowrank_recurrent_layer_metal_chunk(samples : Array(RecurrentSample),
                                                         bases : BasisSet,
                                                         lw : ML::GGUF::Qwen35RecurrentWeights,
                                                         hp : ML::GGUF::Qwen35Hparams,
                                                         rank : Int32,
                                                         calib_count : Int32) : NamedTuple(layer_rmse: Float64, layer_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank recurrent layer chunk unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  layer_cpu_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)

  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample inp missing for layer chunk proof" if sample.inp.empty?
    raise "sample z missing for layer chunk proof" if sample.z.empty?
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, lw.ssm_norm, hp.rms_eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y_cpu)
    out = finish_recurrent_layer_cpu(sample.inp, attn_out, lw, hp)
    out.each_with_index { |v, i| layer_cpu_all[t * hp.n_embd + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  attn_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_out_buf(metal_buf, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    lw.ssm_norm, lw.ssm_out_qw, h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  layer_metal_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
  heldout.each_with_index do |sample, t|
    attn = attn_metal_all[t * hp.n_embd, hp.n_embd]
    out = finish_recurrent_layer_cpu(sample.inp, attn, lw, hp)
    out.each_with_index { |v, i| layer_metal_all[t * hp.n_embd + i] = v }
  end
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  layer_cpu_all.each_with_index do |v, i|
    e = (v - layer_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = layer_cpu_all.size
  {
    layer_rmse: count > 0 ? Math.sqrt(sq / count) : 0.0,
    layer_max:  max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_recurrent_layer_full_metal_chunk(samples : Array(RecurrentSample),
                                                              bases : BasisSet,
                                                              lw : ML::GGUF::Qwen35RecurrentWeights,
                                                              hp : ML::GGUF::Qwen35Hparams,
                                                              rank : Int32,
                                                              calib_count : Int32) : NamedTuple(layer_rmse: Float64, layer_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank recurrent full layer chunk unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  layer_cpu_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)

  inp_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample inp missing for full layer chunk proof" if sample.inp.empty?
    raise "sample z missing for full layer chunk proof" if sample.z.empty?
    hp.n_embd.times { |i| inp_all[t * hp.n_embd + i] = sample.inp[i] }
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, lw.ssm_norm, hp.rms_eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y_cpu)
    out = finish_recurrent_layer_cpu(sample.inp, attn_out, lw, hp)
    out.each_with_index { |v, i| layer_cpu_all[t * hp.n_embd + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  layer_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(metal_buf, inp_all, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  layer_cpu_all.each_with_index do |v, i|
    e = (v - layer_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = layer_cpu_all.size
  {
    layer_rmse: count > 0 ? Math.sqrt(sq / count) : 0.0,
    layer_max:  max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_recurrent_layer_updown_metal_chunk(samples : Array(RecurrentSample),
                                                                bases : BasisSet,
                                                                lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                hp : ML::GGUF::Qwen35Hparams,
                                                                rank : Int32,
                                                                calib_count : Int32,
                                                                adapter : FFNUpDownAdapter,
                                                                updown_rank : Int32) : NamedTuple(layer_rmse: Float64, layer_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64, updown_rank: Int32)
  raise "Metal low-rank recurrent updown layer chunk unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  hidden_dim = hp.n_embd
  bench_rank = Math.min(updown_rank, adapter.coeff_weights.size)
  raise "updown layer rank must be positive" unless bench_rank > 0
  raise "updown layer hidden mismatch" unless adapter.down_basis[0].size == hidden_dim

  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  layer_cpu_all = Array(Float32).new(n_tokens * hidden_dim, 0.0_f32)

  inp_all = Array(Float32).new(n_tokens * hidden_dim, 0.0_f32)
  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample inp missing for updown layer chunk proof" if sample.inp.empty?
    raise "sample z missing for updown layer chunk proof" if sample.z.empty?
    hidden_dim.times { |i| inp_all[t * hidden_dim + i] = sample.inp[i] }
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, lw.ssm_norm, hp.rms_eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y_cpu)
    inp_l2 = Array(Float32).new(hidden_dim) { |i| sample.inp[i] + attn_out[i] }
    ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
    ffn_out = ffn_out_from_updown_adapter(ffn_in, adapter, bench_rank)
    hidden_dim.times { |i| layer_cpu_all[t * hidden_dim + i] = inp_l2[i] + ffn_out[i] }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  coeff_weights = Array(Float32).new(bench_rank * hidden_dim)
  down_basis = Array(Float32).new(bench_rank * hidden_dim)
  bench_rank.times do |j|
    hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
    hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
  end
  x_mean_buf = ML::MetalBuffer.from_array(adapter.x_mean.map(&.to_f32))
  c_mean_buf = ML::MetalBuffer.from_array(adapter.c_mean[0, bench_rank].map(&.to_f32))
  coeff_w_buf = ML::MetalBuffer.from_array(coeff_weights)
  down_buf = ML::MetalBuffer.from_array(down_basis)

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  layer_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_updown_buf(metal_buf, inp_all, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    x_mean_buf, c_mean_buf, coeff_w_buf, down_buf,
    h_k, h_v, s, rank, n_tokens, bench_rank, hp.rms_eps.to_f32, scale).not_nil!
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  layer_cpu_all.each_with_index do |v, i|
    e = (v - layer_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = layer_cpu_all.size
  {
    layer_rmse:  count > 0 ? Math.sqrt(sq / count) : 0.0,
    layer_max:   max,
    state_rmse:  state_rmse,
    state_max:   state_max,
    steps:       n_tokens,
    cpu_ms:      cpu_ms,
    metal_ms:    metal_ms,
    updown_rank: bench_rank,
  }
end

private def lowrank_layer_chunk_inputs(samples : Array(RecurrentSample),
                                       calib_count : Int32,
                                       h_k : Int32, h_v : Int32, s : Int32,
                                       hidden_dim : Int32) : NamedTuple(n_tokens: Int32, inp: Array(Float32), q: Array(Float32), k: Array(Float32), v: Array(Float32), g: Array(Float32), beta: Array(Float32), z: Array(Float32))
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  inp_all = Array(Float32).new(n_tokens * hidden_dim, 0.0_f32)
  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  heldout.each_with_index do |sample, t|
    raise "sample inp missing for layer chunk inputs" if sample.inp.empty?
    raise "sample z missing for layer chunk inputs" if sample.z.empty?
    hidden_dim.times { |i| inp_all[t * hidden_dim + i] = sample.inp[i] }
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end
  {n_tokens: n_tokens, inp: inp_all, q: q_all, k: k_all, v: v_all, g: g_all, beta: b_all, z: z_all}
end

private def simulate_lowrank_recurrent_layer_full_async_overlap(samples : Array(RecurrentSample),
                                                                bases : BasisSet,
                                                                lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                hp : ML::GGUF::Qwen35Hparams,
                                                                rank : Int32,
                                                                calib_count : Int32) : NamedTuple(steps: Int32, serial_ms: Float64, async_ms: Float64, speedup: Float64, output_max: Float64)
  raise "Metal low-rank recurrent async overlap unavailable" unless ML::GGUF::Qwen35Metal.available?
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  state_size = h_v * s * rank
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))

  state_serial_a = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_serial_b = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_serial_a.write(Array(Float32).new(state_size, 0.0_f32))
  state_serial_b.write(Array(Float32).new(state_size, 0.0_f32))
  t_serial = Time.instant
  serial_a = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(state_serial_a, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(state_serial_b, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  serial_ms = (Time.instant - t_serial).total_milliseconds

  state_async_a = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_async_b = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_async_a.write(Array(Float32).new(state_size, 0.0_f32))
  state_async_b.write(Array(Float32).new(state_size, 0.0_f32))
  t_async = Time.instant
  sub_a = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(state_async_a, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_async_a", command_queue_name: "lr_async_a").not_nil!
  sub_b = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(state_async_b, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_async_b", command_queue_name: "lr_async_b").not_nil!
  async_a = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub_a)
  ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub_b)
  async_ms = (Time.instant - t_async).total_milliseconds

  max = 0.0
  serial_a.each_with_index do |v, i|
    e = (v - async_a[i]).abs.to_f64
    max = e if e > max
  end
  {steps: n_tokens, serial_ms: serial_ms, async_ms: async_ms, speedup: async_ms > 0.0 ? serial_ms / async_ms : 0.0, output_max: max}
end

private def verifier_state_after_prefix(weights : ML::GGUF::Qwen35Weights,
                                        prefix_ids : Array(Int32),
                                        max_seq : Int32) : ML::GGUF::Qwen35CPU::State
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  ML::GGUF::Qwen35CPU.prefill_tokens(weights, prefix_ids, 0, state) unless prefix_ids.empty?
  state
end

private def simulate_lowrank_draft_exact_verifier_overlap(samples : Array(RecurrentSample),
                                                          bases : BasisSet,
                                                          weights : ML::GGUF::Qwen35Weights,
                                                          token_ids : Array(Int32),
                                                          lw : ML::GGUF::Qwen35RecurrentWeights,
                                                          hp : ML::GGUF::Qwen35Hparams,
                                                          rank : Int32,
                                                          calib_count : Int32) : NamedTuple(steps: Int32, draft_ms: Float64, verifier_ms: Float64, serial_ms: Float64, overlap_ms: Float64, speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "Metal low-rank draft/verifier overlap unavailable" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))

  # Warm both routes outside the measured region so the comparison focuses on
  # scheduling overlap rather than one-time pipeline/constant cache setup.
  warm_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_state)
  warm_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft_state.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  serial_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  serial_draft_state.write(zero_state)
  serial_verifier_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_draft = Time.instant
  serial_draft = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(serial_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  draft_ms = (Time.instant - t_draft).total_milliseconds
  t_verify = Time.instant
  serial_verifier = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, serial_verifier_state)
  verifier_ms = (Time.instant - t_verify).total_milliseconds
  serial_ms = draft_ms + verifier_ms

  overlap_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  overlap_draft_state.write(zero_state)
  overlap_verifier_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_verifier_draft", command_queue_name: "lr_verifier_draft").not_nil!
  overlap_verifier = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, overlap_verifier_state)
  overlap_draft = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub)
  overlap_ms = (Time.instant - t_overlap).total_milliseconds

  max = 0.0
  serial_draft.each_with_index do |v, i|
    e = (v - overlap_draft[i]).abs.to_f64
    max = e if e > max
  end
  {
    steps:            n_tokens,
    draft_ms:         draft_ms,
    verifier_ms:      verifier_ms,
    serial_ms:        serial_ms,
    overlap_ms:       overlap_ms,
    speedup:          overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    hidden_ms:        serial_ms - overlap_ms,
    draft_output_max: max,
    verifier_match:   serial_verifier == overlap_verifier,
  }
end

private def simulate_lowrank_draft_exact_decode_verifier_overlap(samples : Array(RecurrentSample),
                                                                 bases : BasisSet,
                                                                 weights : ML::GGUF::Qwen35Weights,
                                                                 token_ids : Array(Int32),
                                                                 lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                 hp : ML::GGUF::Qwen35Hparams,
                                                                 rank : Int32,
                                                                 calib_count : Int32) : NamedTuple(steps: Int32, draft_ms: Float64, verifier_serial_ms: Float64, verifier_async_ms: Float64, overlap_ms: Float64, async_speedup: Float64, overlap_speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "Metal exact decode verifier overlap unavailable" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  wba = WbaTrace.maybe("decode_verifier_overlap")

  # Warm exact decode and draft lane outside measured regions.
  warm_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, candidates[0], calib_count, warm_verify)
  warm_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft_state.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  serial_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  serial_results = [] of {Int32, Float32}
  t_serial_verify = Time.instant
  candidates.each_with_index do |tok, i|
    serial_results << ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, serial_verify_state)
  end
  verifier_serial_ms = (Time.instant - t_serial_verify).total_milliseconds

  async_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  t_async_verify = Time.instant
  wba.try(&.point("verifier", "async_submit_begin", t_async_verify))
  candidates.each_with_index do |tok, i|
    submit_t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_top1_async(weights, tok, calib_count + i, async_verify_state,
      fresh_scratch: false, scratch_namespace: "exact_verify_#{i}").not_nil!
    submit_t1 = Time.instant
    wba.try(&.mark("verifier", "async_submit_#{i}", submit_t0, submit_t1))
    submissions << sub
  end
  wait_t0 = Time.instant
  async_results = submissions.map_with_index do |sub, i|
    one_wait_t0 = Time.instant
    result = ML::GGUF::Qwen35CPU.wait_forward_top1(sub)
    one_wait_t1 = Time.instant
    wba.try(&.mark("verifier", "async_wait_#{i}", one_wait_t0, one_wait_t1))
    result
  end
  wait_t1 = Time.instant
  wba.try(&.mark("verifier", "async_wait_all", wait_t0, wait_t1))
  verifier_async_ms = (Time.instant - t_async_verify).total_milliseconds

  draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  draft_state.write(zero_state)
  t_draft = Time.instant
  serial_draft = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  draft_ms = (Time.instant - t_draft).total_milliseconds

  overlap_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  overlap_draft_state.write(zero_state)
  overlap_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  wba.try(&.point("overlap", "begin", t_overlap))
  draft_submit_t0 = Time.instant
  draft_sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_decode_verify_draft", command_queue_name: "lr_decode_verify_draft").not_nil!
  draft_submit_t1 = Time.instant
  wba.try(&.mark("draft", "submit", draft_submit_t0, draft_submit_t1))
  overlap_subs = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  candidates.each_with_index do |tok, i|
    submit_t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_top1_async(weights, tok, calib_count + i, overlap_verify_state,
      fresh_scratch: false, scratch_namespace: "exact_overlap_#{i}").not_nil!
    submit_t1 = Time.instant
    wba.try(&.mark("verifier", "overlap_submit_#{i}", submit_t0, submit_t1))
    overlap_subs << sub
  end
  verifier_wait_t0 = Time.instant
  overlap_results = overlap_subs.map_with_index do |sub, i|
    one_wait_t0 = Time.instant
    result = ML::GGUF::Qwen35CPU.wait_forward_top1(sub)
    one_wait_t1 = Time.instant
    wba.try(&.mark("verifier", "overlap_wait_#{i}", one_wait_t0, one_wait_t1))
    result
  end
  verifier_wait_t1 = Time.instant
  wba.try(&.mark("verifier", "overlap_wait_all", verifier_wait_t0, verifier_wait_t1))
  draft_wait_t0 = Time.instant
  overlap_draft = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(draft_sub)
  draft_wait_t1 = Time.instant
  wba.try(&.mark("draft", "wait", draft_wait_t0, draft_wait_t1))
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.point("overlap", "end", Time.instant))
  wba.try(&.flush)

  max = 0.0
  serial_draft.each_with_index do |v, i|
    e = (v - overlap_draft[i]).abs.to_f64
    max = e if e > max
  end
  serial_overlap_ms = draft_ms + verifier_async_ms
  {
    steps:              n_tokens,
    draft_ms:           draft_ms,
    verifier_serial_ms: verifier_serial_ms,
    verifier_async_ms:  verifier_async_ms,
    overlap_ms:         overlap_ms,
    async_speedup:      verifier_async_ms > 0.0 ? verifier_serial_ms / verifier_async_ms : 0.0,
    overlap_speedup:    overlap_ms > 0.0 ? serial_overlap_ms / overlap_ms : 0.0,
    hidden_ms:          serial_overlap_ms - overlap_ms,
    draft_output_max:   max,
    verifier_match:     serial_results == async_results && serial_results == overlap_results,
  }
end

private def simulate_exact_verifier_ltp_proxy(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              calib_count : Int32) : NamedTuple(steps: Int32, decode_serial_ms: Float64, decode_queued_ms: Float64, chunk_major_ms: Float64, queued_speedup: Float64, ltp_speedup: Float64, queued_match: Bool, chunk_match: Bool)
  raise "exact verifier LTP proxy requires a non-empty held-out span" unless calib_count < token_ids.size
  hp = weights.hparams
  candidates = token_ids[calib_count, token_ids.size - calib_count]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + candidates.size + 8

  # Warm all verifier routes outside measured regions.
  warm_decode = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, candidates[0], calib_count, warm_decode)
  warm_chunk = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_chunk)

  serial_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  serial = [] of {Int32, Float32}
  t_serial = Time.instant
  candidates.each_with_index do |tok, i|
    serial << ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, serial_state)
  end
  decode_serial_ms = (Time.instant - t_serial).total_milliseconds

  queued_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  t_queued = Time.instant
  candidates.each_with_index do |tok, i|
    submissions << ML::GGUF::Qwen35CPU.forward_top1_async(weights, tok, calib_count + i, queued_state,
      fresh_scratch: false, scratch_namespace: "ltp_decode_#{i}").not_nil!
  end
  queued = submissions.map { |sub| ML::GGUF::Qwen35CPU.wait_forward_top1(sub) }
  decode_queued_ms = (Time.instant - t_queued).total_milliseconds

  chunk_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_chunk = Time.instant
  chunk = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, chunk_state)
  chunk_major_ms = (Time.instant - t_chunk).total_milliseconds

  serial_ids = serial.map(&.[0])
  queued_ids = queued.map(&.[0])
  chunk_ids = chunk.map(&.[0])
  {
    steps:            candidates.size,
    decode_serial_ms: decode_serial_ms,
    decode_queued_ms: decode_queued_ms,
    chunk_major_ms:   chunk_major_ms,
    queued_speedup:   decode_queued_ms > 0.0 ? decode_serial_ms / decode_queued_ms : 0.0,
    ltp_speedup:      chunk_major_ms > 0.0 ? decode_serial_ms / chunk_major_ms : 0.0,
    queued_match:     serial_ids == queued_ids,
    chunk_match:      serial_ids == chunk_ids,
  }
end

private def simulate_lowrank_draft_exact_chunk_verifier_thread_overlap(samples : Array(RecurrentSample),
                                                                       bases : BasisSet,
                                                                       weights : ML::GGUF::Qwen35Weights,
                                                                       token_ids : Array(Int32),
                                                                       lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                       hp : ML::GGUF::Qwen35Hparams,
                                                                       rank : Int32,
                                                                       calib_count : Int32) : NamedTuple(steps: Int32, draft_ms: Float64, chunk_verifier_ms: Float64, serial_ms: Float64, overlap_ms: Float64, speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "threaded chunk verifier overlap requires Metal" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  wba = WbaTrace.maybe("chunk_verifier_thread_overlap")

  # Warm both routes outside the measured region.
  warm_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_verify)
  warm_draft = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  serial_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  serial_draft_state.write(zero_state)
  t_draft = Time.instant
  serial_draft = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(serial_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  draft_ms = (Time.instant - t_draft).total_milliseconds

  serial_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_verify = Time.instant
  serial_verify = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, serial_verify_state)
  chunk_verifier_ms = (Time.instant - t_verify).total_milliseconds
  serial_ms = draft_ms + chunk_verifier_ms

  thread_done = Atomic(Int32).new(0)
  thread_result = nil.as(Array({Int32, Float32})?)
  thread_error = nil.as(String?)
  overlap_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  overlap_draft_state.write(zero_state)
  overlap_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  wba.try(&.point("overlap", "begin", t_overlap))
  draft_submit_t0 = Time.instant
  draft_sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "chunk_verifier_thread_draft", command_queue_name: "chunk_verifier_thread_draft").not_nil!
  draft_submit_t1 = Time.instant
  wba.try(&.mark("draft", "submit", draft_submit_t0, draft_submit_t1))
  Thread.new do
    begin
      STDERR.puts "chunk-thread: begin verifier" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
      thread_t0 = Time.instant
      result = ML::GGUF::Qwen35Metal::Scratch.with_namespace("chunk_verifier_thread") do
        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, overlap_verify_state)
      end
      thread_t1 = Time.instant
      wba.try(&.mark("verifier", "thread_prefill_top1s", thread_t0, thread_t1))
      thread_result = result
      STDERR.puts "chunk-thread: verifier done" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    rescue ex
      thread_error = "#{ex.class}: #{ex.message}\n#{ex.backtrace.join('\n')}"
      STDERR.puts "chunk-thread: error #{thread_error}" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    ensure
      thread_done.set(1)
      STDERR.puts "chunk-thread: done flag set" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    end
  end
  draft_wait_t0 = Time.instant
  overlap_draft = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(draft_sub)
  draft_wait_t1 = Time.instant
  wba.try(&.mark("draft", "wait", draft_wait_t0, draft_wait_t1))
  recv_t0 = Time.instant
  deadline = Time.instant + 120.seconds
  while thread_done.get == 0
    raise "chunk-major verifier worker did not finish within 120s" if Time.instant > deadline
    Thread.yield
  end
  if error = thread_error
    raise "chunk-major verifier worker failed: #{error}"
  end
  overlap_verify = thread_result.not_nil!
  recv_t1 = Time.instant
  wba.try(&.mark("verifier", "receive", recv_t0, recv_t1))
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.point("overlap", "end", Time.instant))
  wba.try(&.flush)

  max = 0.0
  serial_draft.each_with_index do |v, i|
    e = (v - overlap_draft[i]).abs.to_f64
    max = e if e > max
  end
  {
    steps:             n_tokens,
    draft_ms:          draft_ms,
    chunk_verifier_ms: chunk_verifier_ms,
    serial_ms:         serial_ms,
    overlap_ms:        overlap_ms,
    speedup:           overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    hidden_ms:         serial_ms - overlap_ms,
    draft_output_max:  max,
    verifier_match:    serial_verify.map(&.[0]) == overlap_verify.map(&.[0]),
  }
end

private def simulate_lowrank_multilayer_chunk_thread_overlap(samples : Array(RecurrentSample),
                                                             bases : BasisSet,
                                                             weights : ML::GGUF::Qwen35Weights,
                                                             token_ids : Array(Int32),
                                                             lw : ML::GGUF::Qwen35RecurrentWeights,
                                                             hp : ML::GGUF::Qwen35Hparams,
                                                             rank : Int32,
                                                             calib_count : Int32,
                                                             n_layers : Int32) : NamedTuple(steps: Int32, n_layers: Int32, draft_ms: Float64, draft_per_layer_ms: Float64, chunk_verifier_ms: Float64, serial_ms: Float64, overlap_ms: Float64, speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "multilayer chunk verifier overlap requires Metal" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  raise "n_layers must be positive" unless n_layers > 0
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  wba = WbaTrace.maybe("multilayer_chunk_verifier_overlap")

  # Warm both routes outside the measured region.
  warm_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_verify)
  warm_draft = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  # Serial baseline: N layer chunks back-to-back on the default queue.
  serial_states = Array(ML::MetalBuffer).new(n_layers) do
    buf = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
    buf.write(zero_state)
    buf
  end
  last_serial_output = nil.as(Array(Float32)?)
  t_draft = Time.instant
  n_layers.times do |i|
    last_serial_output = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(serial_states[i], inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
      lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
      h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  end
  draft_ms = (Time.instant - t_draft).total_milliseconds
  draft_per_layer_ms = n_layers > 0 ? draft_ms / n_layers : 0.0

  serial_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_verify = Time.instant
  serial_verify = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, serial_verify_state)
  chunk_verifier_ms = (Time.instant - t_verify).total_milliseconds
  serial_ms = draft_ms + chunk_verifier_ms

  thread_done = Atomic(Int32).new(0)
  thread_result = nil.as(Array({Int32, Float32})?)
  thread_error = nil.as(String?)
  overlap_states = Array(ML::MetalBuffer).new(n_layers) do
    buf = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
    buf.write(zero_state)
    buf
  end
  overlap_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  wba.try(&.point("overlap", "begin", t_overlap))
  draft_submit_t0 = Time.instant
  draft_subs = Array(ML::GGUF::Qwen35Metal::LowRankLayerChunkSubmission).new
  n_layers.times do |i|
    sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_states[i], inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
      lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
      h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "multi_draft_#{i}", command_queue_name: "multi_draft").not_nil!
    draft_subs << sub
  end
  draft_submit_t1 = Time.instant
  wba.try(&.mark("draft", "submit_all", draft_submit_t0, draft_submit_t1))
  Thread.new do
    begin
      STDERR.puts "multi-thread: begin verifier" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
      thread_t0 = Time.instant
      result = ML::GGUF::Qwen35Metal::Scratch.with_namespace("multi_chunk_verifier_thread") do
        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, overlap_verify_state)
      end
      thread_t1 = Time.instant
      wba.try(&.mark("verifier", "thread_prefill_top1s", thread_t0, thread_t1))
      thread_result = result
      STDERR.puts "multi-thread: verifier done" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    rescue ex
      thread_error = "#{ex.class}: #{ex.message}\n#{ex.backtrace.join('\n')}"
      STDERR.puts "multi-thread: error #{thread_error}" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    ensure
      thread_done.set(1)
      STDERR.puts "multi-thread: done flag set" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    end
  end
  draft_wait_t0 = Time.instant
  last_overlap_output = nil.as(Array(Float32)?)
  draft_subs.each do |sub|
    last_overlap_output = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub)
  end
  draft_wait_t1 = Time.instant
  wba.try(&.mark("draft", "wait_all", draft_wait_t0, draft_wait_t1))
  recv_t0 = Time.instant
  deadline = Time.instant + 120.seconds
  while thread_done.get == 0
    raise "multilayer chunk-major verifier worker did not finish within 120s" if Time.instant > deadline
    Thread.yield
  end
  if error = thread_error
    raise "multilayer chunk-major verifier worker failed: #{error}"
  end
  overlap_verify = thread_result.not_nil!
  recv_t1 = Time.instant
  wba.try(&.mark("verifier", "receive", recv_t0, recv_t1))
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.point("overlap", "end", Time.instant))
  wba.try(&.flush)

  max = 0.0
  if (ls = last_serial_output) && (lo = last_overlap_output)
    ls.each_with_index do |v, i|
      e = (v - lo[i]).abs.to_f64
      max = e if e > max
    end
  end
  {
    steps:              n_tokens,
    n_layers:           n_layers,
    draft_ms:           draft_ms,
    draft_per_layer_ms: draft_per_layer_ms,
    chunk_verifier_ms:  chunk_verifier_ms,
    serial_ms:          serial_ms,
    overlap_ms:         overlap_ms,
    speedup:            overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    hidden_ms:          serial_ms - overlap_ms,
    draft_output_max:   max,
    verifier_match:     serial_verify.map(&.[0]) == overlap_verify.map(&.[0]),
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

private def recurrent_layer_cpu_exact_with_ffn_activation(inpSA : Array(Float32),
                                                          lw : ML::GGUF::Qwen35RecurrentWeights,
                                                          lstate : ML::GGUF::Qwen35CPU::LayerState,
                                                          hp : ML::GGUF::Qwen35Hparams) : NamedTuple(out: Array(Float32), activation: Array(Float64), ffn_in: Array(Float64))
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
  {
    out:        Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] },
    activation: combined.map(&.to_f64),
    ffn_in:     ffn_in.map(&.to_f64),
  }
end

private def ffn_activation_vectors_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              layer_indices : Array(Int32),
                                              calib_count : Int32) : Hash(Int32, Array(Array(Float64)))
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  wanted = Set(Int32).new(layer_indices)
  vectors = Hash(Int32, Array(Array(Float64))).new { |h, k| h[k] = [] of Array(Float64) }

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
    weights.layers.each_with_index do |layer, il|
      case layer
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        if wanted.includes?(il)
          res = recurrent_layer_cpu_exact_with_ffn_activation(x, layer, state.layers[il], hp)
          vectors[il] << res[:activation] if pos < calib_count
          x = res[:out]
        else
          x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
        end
      end
    end
  end

  vectors
end

private def token_ids_for_prompt(tok, prompt : String, tokens_limit : Int32) : Array(Int32)
  token_ids = tok.encode(prompt, add_bos_override: false)
  while token_ids.size < tokens_limit
    token_ids.concat(tok.encode(prompt, add_bos_override: false))
  end
  token_ids[0, tokens_limit]
end

private def ffn_activation_vectors_for_token_sets(weights : ML::GGUF::Qwen35Weights,
                                                  token_sets : Array(Array(Int32)),
                                                  layer_indices : Array(Int32),
                                                  calib_tokens : Int32) : Hash(Int32, Array(Array(Float64)))
  merged = Hash(Int32, Array(Array(Float64))).new { |h, k| h[k] = [] of Array(Float64) }
  token_sets.each do |token_ids|
    prompt_calib_count = Math.min(calib_tokens, token_ids.size)
    next if prompt_calib_count <= 0

    vectors = ffn_activation_vectors_for_prompt(weights, token_ids[0, prompt_calib_count], layer_indices, prompt_calib_count)
    vectors.each do |il, layer_vectors|
      merged[il].concat(layer_vectors)
    end
  end
  merged
end

private def ffn_updown_samples_for_token_sets(weights : ML::GGUF::Qwen35Weights,
                                              token_sets : Array(Array(Int32)),
                                              layer_indices : Array(Int32),
                                              calib_tokens : Int32) : Hash(Int32, Array(NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))))
  hp = weights.hparams
  wanted = Set(Int32).new(layer_indices)
  samples = Hash(Int32, Array(NamedTuple(ffn_in: Array(Float64), activation: Array(Float64)))).new do |h, k|
    h[k] = [] of NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))
  end

  token_sets.each do |token_ids|
    prompt_calib_count = Math.min(calib_tokens, token_ids.size)
    next if prompt_calib_count <= 0

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prompt_calib_count + 2)
    token_ids[0, prompt_calib_count].each_with_index do |token_id, pos|
      x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
      weights.layers.each_with_index do |layer, il|
        case layer
        in ML::GGUF::Qwen35FullAttnWeights
          x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
        in ML::GGUF::Qwen35RecurrentWeights
          if wanted.includes?(il)
            res = recurrent_layer_cpu_exact_with_ffn_activation(x, layer, state.layers[il], hp)
            samples[il] << {ffn_in: res[:ffn_in], activation: res[:activation]}
            x = res[:out]
          else
            x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
          end
        end
      end
    end
  end

  samples
end

private def solve_linear_system(a_in : Array(Array(Float64)), b_in : Array(Float64), eps : Float64 = 1.0e-12) : Array(Float64)
  n = b_in.size
  a = a_in.map(&.dup)
  b = b_in.dup
  n.times do |col|
    pivot = col
    best = a[col][col].abs
    (col + 1).upto(n - 1) do |row|
      v = a[row][col].abs
      if v > best
        best = v
        pivot = row
      end
    end
    raise "singular ridge system" if best <= eps
    if pivot != col
      a[col], a[pivot] = a[pivot], a[col]
      b[col], b[pivot] = b[pivot], b[col]
    end
    diag = a[col][col]
    col.upto(n - 1) { |j| a[col][j] /= diag }
    b[col] /= diag
    n.times do |row|
      next if row == col
      factor = a[row][col]
      next if factor.abs <= eps
      col.upto(n - 1) { |j| a[row][j] -= factor * a[col][j] }
      b[row] -= factor * b[col]
    end
  end
  b
end

private def train_ffn_updown_adapter(samples : Array(NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))),
                                     basis : Array(Array(Float64)),
                                     down_basis : Array(Array(Float32)),
                                     rank : Int32,
                                     ridge : Float64 = 1.0e-3) : FFNUpDownAdapter
  raise "need samples for FFN up/down adapter" if samples.empty?
  limit = Math.min(rank, basis.size)
  raise "need basis vectors for FFN up/down adapter" unless limit > 0
  n = samples.size
  dim = samples[0][:ffn_in].size
  x_mean = Array(Float64).new(dim, 0.0)
  samples.each do |sample|
    dim.times { |d| x_mean[d] += sample[:ffn_in][d] }
  end
  dim.times { |d| x_mean[d] /= n }

  coeffs = Array.new(n) { Array(Float64).new(limit, 0.0) }
  c_mean = Array(Float64).new(limit, 0.0)
  samples.each_with_index do |sample, si|
    limit.times do |j|
      c = dot(sample[:activation], basis[j])
      coeffs[si][j] = c
      c_mean[j] += c
    end
  end
  limit.times { |j| c_mean[j] /= n }

  gram = Array.new(n) { Array(Float64).new(n, 0.0) }
  n.times do |i|
    xi = samples[i][:ffn_in]
    i.upto(n - 1) do |k|
      xk = samples[k][:ffn_in]
      acc = 0.0
      dim.times { |d| acc += (xi[d] - x_mean[d]) * (xk[d] - x_mean[d]) }
      gram[i][k] = acc
      gram[k][i] = acc
    end
    gram[i][i] += ridge
  end

  coeff_weights = Array.new(limit) { Array(Float64).new(dim, 0.0) }
  limit.times do |j|
    y = Array(Float64).new(n) { |i| coeffs[i][j] - c_mean[j] }
    alpha = solve_linear_system(gram, y)
    w = coeff_weights[j]
    n.times do |i|
      xi = samples[i][:ffn_in]
      dim.times { |d| w[d] += alpha[i] * (xi[d] - x_mean[d]) }
    end
  end

  FFNUpDownAdapter.new(x_mean, c_mean, coeff_weights, down_basis[0, limit])
end

private def recurrent_layer_cpu_lowrank(inpSA : Array(Float32),
                                        lw : ML::GGUF::Qwen35RecurrentWeights,
                                        lstate : ML::GGUF::Qwen35CPU::LayerState,
                                        hp : ML::GGUF::Qwen35Hparams,
                                        bases : BasisSet,
                                        rank : Int32,
                                        lr_state : LowRankState,
                                        fallback_threshold : Float64? = nil,
                                        force_fallback : Bool = false,
                                        use_metal_lowrank : Bool = false,
                                        project_coeffs_on_gpu : Bool = false,
                                        use_metal_layer_updown : Bool = false,
                                        draft_variant : String = "lowrank",
                                        ffn_basis : Array(Array(Float64))? = nil,
                                        ffn_adapter : FFNAdapter? = nil,
                                        ffn_updown_adapter : FFNUpDownAdapter? = nil) : Array(Float32)
  return inpSA if draft_variant == "skip-layer"

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
  fallback = force_fallback
  if threshold = fallback_threshold
    fallback ||= max_k_residual(k_conv, bases, rank, h_k, s) > threshold
  end
  routed_out = nil.as(Array(Float32)?)
  if fallback
    sync_lowrank_state_from_metal!(lr_state) if use_metal_lowrank
    unless lr_state.full_state_current
      lstate.ssm_state = reconstruct_lowrank_state(lr_state.m, bases, rank, h_k, h_v, s)
    end
    state = lstate.ssm_state.not_nil!
    ML::GGUF::Qwen35CPU.delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y, h_k, h_v, s, scale)
    lr_state.m = project_full_state_to_lowrank(state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.fallback_steps += 1
  else
    sample = RecurrentSample.new(q_conv, k_conv, v_conv, ghead, beta)
    pca_updown_rank = draft_variant_ffn_pca_updown_rank(draft_variant)
    if pca_updown_rank && use_metal_layer_updown && use_metal_lowrank && project_coeffs_on_gpu
      adapter = ffn_updown_adapter || raise "draft variant #{draft_variant.inspect} requires FFN up/down adapter"
      state_buf = lowrank_state_buffer!(lr_state)
      basis_buf = lowrank_basis_buffer!(lr_state, bases, rank, h_k, s)
      updown = updown_adapter_buffers!(lr_state, adapter, pca_updown_rank, hp.n_embd)
      out = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_updown_buf(
        state_buf, inpSA, q_conv, k_conv, basis_buf, v_conv, ghead, beta, z,
        lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
        updown[:x_mean], updown[:c_mean], updown[:coeff_w], updown[:down],
        h_k, h_v, s, rank, 1, updown[:rank], hp.rms_eps.to_f32, scale
      ).not_nil!
      lr_state.full_state_current = false
      lr_state.approx_steps += 1
      routed_out = out
    else
      if use_metal_lowrank
        lowrank_projected_delta_step_metal!(lr_state, sample, bases, rank, y, h_k, h_v, s, scale, project_coeffs_on_gpu)
      else
        lowrank_projected_delta_step!(lr_state.m, sample, bases, rank, y, h_k, h_v, s, scale)
      end
      lr_state.full_state_current = false
      lr_state.approx_steps += 1
    end
  end
  return routed_out.not_nil! if routed_out

  h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y, h * s, s, lw.ssm_norm, hp.rms_eps) }
  (h_v * s).times { |i| y[i] = y[i] * ML::GGUF::Qwen35CPU.silu(z[i]) }
  attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y)

  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inpSA[i] + attn_out[i] }
  return inp_l2 if draft_variant == "lowrank-no-ffn"

  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  if pca_updown_rank = draft_variant_ffn_pca_updown_rank(draft_variant)
    adapter = ffn_updown_adapter || raise "draft variant #{draft_variant.inspect} requires FFN up/down adapter"
    ffn_out = ffn_out_from_updown_adapter(ffn_in, adapter, pca_updown_rank)
    return Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
  end

  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  if percent = draft_variant_ffn_top_percent(draft_variant)
    keep_top_abs_percent!(combined, percent)
  end
  if pca_rank = draft_variant_ffn_pca_rank(draft_variant)
    basis = ffn_basis || raise "draft variant #{draft_variant.inspect} requires FFN activation basis"
    project_vector_with_basis!(combined, basis, pca_rank)
  end
  ffn_out = if pca_down_rank = draft_variant_ffn_pca_down_rank(draft_variant)
              adapter = ffn_adapter || raise "draft variant #{draft_variant.inspect} requires FFN down adapter"
              ffn_down_from_adapter(combined, adapter, pca_down_rank)
            else
              ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
            end
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
                                       refresh_interval : Int32?,
                                       approximate : Bool,
                                       use_metal_lowrank : Bool = false,
                                       project_coeffs_on_gpu : Bool = false,
                                       use_metal_layer_updown : Bool = false,
                                       draft_variant : String = "lowrank",
                                       ffn_bases : FFNBasisMap? = nil,
                                       ffn_adapters : FFNAdapterMap? = nil,
                                       ffn_updown_adapters : FFNUpDownAdapterMap? = nil) : Array(Float32)
  hp = weights.hparams
  x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
  early_exit_layers = approximate ? cheap_draft_early_exit_layers(draft_variant) : nil
  weights.layers.each_with_index do |layer, il|
    case layer
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
    in ML::GGUF::Qwen35RecurrentWeights
      if bases = layer_bases[il]?
        x = if approximate && pos >= calib_count
              lr_state = lr_states[il] ||= LowRankState.new
              force_refresh = if interval = refresh_interval
                                interval > 0 && ((pos - calib_count) % interval == 0)
                              else
                                false
                              end
              ffn_basis = ffn_bases ? ffn_bases.not_nil![il]? : nil
              ffn_adapter = ffn_adapters ? ffn_adapters.not_nil![il]? : nil
              ffn_updown_adapter = ffn_updown_adapters ? ffn_updown_adapters.not_nil![il]? : nil
              recurrent_layer_cpu_lowrank(x, layer, state.layers[il], hp, bases, rank, lr_state, fallback_threshold, force_refresh, use_metal_lowrank, project_coeffs_on_gpu, use_metal_layer_updown, draft_variant, ffn_basis, ffn_adapter, ffn_updown_adapter)
            else
              recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
            end
      else
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
      end
    end
    break if early_exit_layers && (il + 1) >= early_exit_layers
  end
  x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
  ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
end

private def refresh_due?(pos : Int32, calib_count : Int32, interval : Int32?) : Bool
  return false unless n = interval
  return false unless n > 0 && pos >= calib_count

  ((pos - calib_count + 1) % n) == 0
end

private def sync_lowrank_shadow!(approx_state : ML::GGUF::Qwen35CPU::State,
                                 exact_state : ML::GGUF::Qwen35CPU::State,
                                 layer_bases : LayerBasisMap,
                                 lr_states : Hash(Int32, LowRankState),
                                 rank : Int32,
                                 hp : ML::GGUF::Qwen35Hparams) : Nil
  approx_state.copy_from!(exact_state)
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  layer_bases.each do |il, bases|
    next unless state = approx_state.layers[il].ssm_state

    lr_state = lr_states[il] ||= LowRankState.new
    lr_state.m = project_full_state_to_lowrank(state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.initialized = true
  end
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

private def draft_variant_ffn_top_percent(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-top-")

  percent = variant["lowrank-ffn-top-".size..].to_i? || raise "invalid FFN top-percent variant #{variant.inspect}"
  raise "FFN top-percent must be in 1..100" unless percent >= 1 && percent <= 100
  percent
end

private def draft_variant_ffn_pca_rank(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-pca-")
  return nil if variant.starts_with?("lowrank-ffn-pca-down-")
  return nil if variant.starts_with?("lowrank-ffn-pca-updown-")

  rank = variant["lowrank-ffn-pca-".size..].to_i? || raise "invalid FFN PCA variant #{variant.inspect}"
  raise "FFN PCA rank must be positive" unless rank > 0
  rank
end

private def draft_variant_ffn_pca_down_rank(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-pca-down-")

  rank = variant["lowrank-ffn-pca-down-".size..].to_i? || raise "invalid FFN PCA-down variant #{variant.inspect}"
  raise "FFN PCA-down rank must be positive" unless rank > 0
  rank
end

private def draft_variant_ffn_pca_updown_rank(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-pca-updown-")

  rank = variant["lowrank-ffn-pca-updown-".size..].to_i? || raise "invalid FFN PCA-updown variant #{variant.inspect}"
  raise "FFN PCA-updown rank must be positive" unless rank > 0
  rank
end

private def keep_top_abs_percent!(values : Array(Float32), percent : Int32) : Nil
  return if percent >= 100

  keep = Math.max(1, (values.size.to_i64 * percent + 99) // 100).to_i
  return if keep >= values.size

  threshold = values.map(&.abs).sort![-keep]
  kept = 0
  values.size.times do |i|
    if values[i].abs >= threshold && kept < keep
      kept += 1
    else
      values[i] = 0.0_f32
    end
  end
end

private def project_vector_with_basis!(values : Array(Float32), basis : Array(Array(Float64)), rank : Int32) : Nil
  return if basis.empty?

  limit = Math.min(rank, basis.size)
  projected = Array(Float64).new(values.size, 0.0)
  limit.times do |i|
    b = basis[i]
    coeff = 0.0
    values.size.times { |d| coeff += values[d].to_f64 * b[d] }
    values.size.times { |d| projected[d] += coeff * b[d] }
  end
  values.size.times { |d| values[d] = projected[d].to_f32 }
end

private def ffn_down_from_adapter(combined : Array(Float32), adapter : FFNAdapter, rank : Int32) : Array(Float32)
  limit = Math.min(rank, adapter.basis.size)
  raise "FFN adapter has no basis vectors" unless limit > 0
  out_dim = adapter.down_basis[0].size
  out = Array(Float32).new(out_dim, 0.0_f32)
  limit.times do |i|
    b = adapter.basis[i]
    coeff = 0.0
    combined.size.times { |d| coeff += combined[d].to_f64 * b[d] }
    coeff_f = coeff.to_f32
    down = adapter.down_basis[i]
    out_dim.times { |d| out[d] += coeff_f * down[d] }
  end
  out
end

private def ffn_out_from_updown_adapter(ffn_in : Array(Float32), adapter : FFNUpDownAdapter, rank : Int32) : Array(Float32)
  limit = Math.min(rank, adapter.coeff_weights.size)
  raise "FFN up/down adapter has no coefficient weights" unless limit > 0
  out_dim = adapter.down_basis[0].size
  out = Array(Float32).new(out_dim, 0.0_f32)
  limit.times do |j|
    w = adapter.coeff_weights[j]
    coeff = adapter.c_mean[j]
    ffn_in.size.times { |d| coeff += (ffn_in[d].to_f64 - adapter.x_mean[d]) * w[d] }
    coeff_f = coeff.to_f32
    down = adapter.down_basis[j]
    out_dim.times { |d| out[d] += coeff_f * down[d] }
  end
  out
end

private struct TopKOracleSample
  getter ids : Array(Int32)
  getter logits : Array(Float32)
  getter exact_id : Int32

  def initialize(@ids : Array(Int32), @logits : Array(Float32), @exact_id : Int32)
  end

  def margin : Float64
    return Float64::INFINITY if @logits.size < 2

    (@logits[0] - @logits[1]).to_f64
  end
end

private def topk_oracle_sample(approx_logits : Array(Float32), exact_id : Int32, top_k : Int32) : TopKOracleSample
  ids = top_k_indices(approx_logits, top_k)
  logits = ids.map { |id| approx_logits[id] }
  TopKOracleSample.new(ids, logits, exact_id)
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
                                   fallback_threshold : Float64?,
                                   refresh_interval : Int32?,
                                   oracle_refresh_interval : Int32?,
                                   output_margin_threshold : Float64?) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_steps: Int32, fallback_steps: Int32, output_fallbacks: Int32)
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
  output_fallbacks = 0
  compared = 0

  token_ids.each_with_index do |token_id, pos|
    exact = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, false)
    approx = logits_with_lowrank_policy(weights, token_id, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, true)
    next if pos < calib_count

    approx_eval = approx
    if threshold = output_margin_threshold
      if top1_margin(approx) < threshold
        output_fallbacks += 1
        approx_eval = exact
      end
    end

    c = cosine(exact, approx_eval)
    cosines << c
    d = max_abs_delta(exact, approx_eval)
    max_delta = d if d > max_delta
    exact_top1 = top1(exact)
    approx_top1 = top1(approx_eval)
    exact_margin = top1_margin(exact)
    min_margin = exact_margin if exact_margin < min_margin
    if exact_top1 == approx_top1
      top_matches += 1
    elsif exact_margin >= 0.5
      confident_mismatches += 1
    end
    top5_hits += 1 if top_k_indices(approx_eval, 5).includes?(exact_top1)
    kls << softmax_kl(exact, approx_eval)
    compared += 1
    if refresh_due?(pos.to_i32, calib_count, oracle_refresh_interval)
      sync_lowrank_shadow!(approx_state, exact_state, layer_bases, lr_states, rank, weights.hparams)
    end
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
    output_fallbacks:     output_fallbacks,
  }
end

private def simulate_greedy_policy(weights : ML::GGUF::Qwen35Weights,
                                   prompt_ids : Array(Int32),
                                   gen_tokens : Int32,
                                   layer_bases : LayerBasisMap,
                                   rank : Int32,
                                   calib_count : Int32,
                                   fallback_threshold : Float64?,
                                   refresh_interval : Int32?,
                                   oracle_refresh_interval : Int32?,
                                   output_margin_threshold : Float64?) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_steps: Int32, fallback_steps: Int32, output_fallbacks: Int32, exact_ids: Array(Int32), approx_ids: Array(Int32))
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  approx_logits = [] of Float32

  prompt_ids.each_with_index do |token_id, pos|
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, false)
    approx_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, true)
    if refresh_due?(pos.to_i32, calib_count, oracle_refresh_interval)
      sync_lowrank_shadow!(approx_state, exact_state, layer_bases, lr_states, rank, weights.hparams)
      approx_logits = exact_logits.dup
    end
  end

  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  output_fallbacks = 0
  exact_ids = [] of Int32
  approx_ids = [] of Int32

  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    approx_eval = approx_logits
    if threshold = output_margin_threshold
      if top1_margin(approx_logits) < threshold
        output_fallbacks += 1
        approx_eval = exact_logits
      end
    end
    approx_top1 = top1(approx_eval)
    exact_ids << exact_top1
    approx_ids << approx_top1

    c = cosine(exact_logits, approx_eval)
    cosines << c
    d = max_abs_delta(exact_logits, approx_eval)
    max_delta = d if d > max_delta
    exact_margin = top1_margin(exact_logits)
    min_margin = exact_margin if exact_margin < min_margin
    if exact_top1 == approx_top1
      top_matches += 1
    elsif exact_margin >= 0.5
      confident_mismatches += 1
    end
    top5_hits += 1 if top_k_indices(approx_eval, 5).includes?(exact_top1)
    kls << softmax_kl(exact_logits, approx_eval)

    pos = prompt_ids.size + step
    # Teacher-forced on the exact greedy trajectory. This isolates policy drift
    # from cascading different-token hidden-state divergence.
    exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, false)
    approx_logits = logits_with_lowrank_policy(weights, exact_top1, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, true)
    if refresh_due?(pos.to_i32, calib_count, oracle_refresh_interval)
      sync_lowrank_shadow!(approx_state, exact_state, layer_bases, lr_states, rank, weights.hparams)
      approx_logits = exact_logits.dup
    end
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
    output_fallbacks:     output_fallbacks,
    exact_ids:            exact_ids,
    approx_ids:           approx_ids,
  }
end

private def simulate_self_spec_policy(weights : ML::GGUF::Qwen35Weights,
                                      prompt_ids : Array(Int32),
                                      gen_tokens : Int32,
                                      gamma : Int32,
                                      layer_bases : LayerBasisMap,
                                      rank : Int32,
                                      calib_count : Int32,
                                      fallback_threshold : Float64?,
                                      refresh_interval : Int32?,
                                      adaptive_min_gamma : Int32? = nil,
                                      adaptive_max_gamma : Int32? = nil,
                                      adaptive_grow_margin_threshold : Float64? = nil,
                                      draft_margin_threshold : Float64? = nil,
                                      draft_stop_margin_threshold : Float64? = nil,
                                      topk_rescue : Int32? = nil,
                                      progressive_schedule : Array(Int32)? = nil) : NamedTuple(chunks: Int32, full_accept_chunks: Int32, rejections: Int32, topk_rescues: Int32, emitted_tokens: Int32, proposed_tokens: Int32, accepted_draft_tokens: Int32, verifier_tokens: Int32, correction_steps: Int32, approx_steps: Int32, fallback_steps: Int32, draft_top2_hits: Int32, draft_top5_hits: Int32, reject_top2_hits: Int32, reject_top5_hits: Int32, accept_rate: Float64, avg_accept: Float64, break_even_draft_verify_per_proposed: Float64, draft_top2_hit_rate: Float64, draft_top5_hit_rate: Float64, gamma_history: Array(Int32), verifier_history: Array(Int32), draft_min_margin_history: Array(Float64), draft_low_margin_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "self-spec gamma must be positive" unless gamma > 0
  adaptive = !adaptive_min_gamma.nil? && !adaptive_max_gamma.nil?
  progressive = progressive_schedule && !progressive_schedule.not_nil!.empty?
  min_gamma = adaptive_min_gamma || gamma
  max_gamma = adaptive_max_gamma || (progressive ? progressive_schedule.not_nil!.max : gamma)
  raise "adaptive min gamma must be positive" if adaptive && min_gamma <= 0
  raise "adaptive max gamma must be >= min gamma" if adaptive && max_gamma < min_gamma
  raise "progressive schedule values must be positive" if progressive && progressive_schedule.not_nil!.any? { |v| v <= 0 }

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end

  chunks = 0
  full_accept_chunks = 0
  rejections = 0
  topk_rescues = 0
  emitted_tokens = 0
  proposed_tokens = 0
  accepted_draft_tokens = 0
  verifier_tokens = 0
  correction_steps = 0
  approx_steps = 0
  fallback_steps = 0
  draft_top2_hits = 0
  draft_top5_hits = 0
  reject_top2_hits = 0
  reject_top5_hits = 0
  current_gamma = gamma.clamp(min_gamma, max_gamma)
  progressive_index = 0
  gamma_history = [] of Int32
  verifier_history = [] of Int32
  draft_min_margin_history = [] of Float64
  draft_low_margin_history = [] of Int32
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    remaining = gen_tokens - emitted_tokens
    chunk_gamma = if progressive
                    progressive_schedule.not_nil![progressive_index]
                  else
                    current_gamma
                  end
    chunk_gamma = Math.min(chunk_gamma, remaining)
    gamma_history << chunk_gamma

    draft_state = state_before_last.fork
    draft_lr_states = {} of Int32 => LowRankState
    sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
    draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)

    proposal = [] of Int32
    proposal_top5 = [] of Array(Int32)
    chunk_draft_min_margin = Float64::INFINITY
    chunk_draft_low_margin = 0
    chunk_gamma.times do |j|
      draft_margin = top1_margin(draft_logits)
      chunk_draft_min_margin = draft_margin if draft_margin < chunk_draft_min_margin
      if threshold = draft_margin_threshold
        chunk_draft_low_margin += 1 if draft_margin < threshold
      end
      draft_top5 = top_k_indices(draft_logits, 5)
      proposed = top1(draft_logits)
      proposal << proposed
      proposal_top5 << draft_top5
      if threshold = draft_stop_margin_threshold
        break if proposal.size >= Math.min(min_gamma, chunk_gamma) && draft_margin < threshold
      end
      break if j == chunk_gamma - 1

      draft_logits = logits_with_lowrank_policy(weights, proposed, (pos_last + 1 + j).to_i32, draft_state,
        layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)
    end
    draft_min_margin_history << chunk_draft_min_margin
    draft_low_margin_history << chunk_draft_low_margin
    approx_steps += draft_lr_states.values.sum(&.approx_steps)
    fallback_steps += draft_lr_states.values.sum(&.fallback_steps)
    proposed_tokens += proposal.size
    verifier_tokens += proposal.size
    verifier_history << proposal.size

    accepted_this_chunk = 0
    rejected = false
    chunk_min_margin = Float64::INFINITY
    proposal.each_with_index do |draft_token, j|
      exact_top1 = top1(exact_logits)
      exact_margin = top1_margin(exact_logits)
      chunk_min_margin = exact_margin if exact_margin < chunk_min_margin
      exact_ids << exact_top1
      top5 = proposal_top5[j]
      top2_hit = top5[0, 2].includes?(exact_top1)
      top5_hit = top5.includes?(exact_top1)
      draft_top2_hits += 1 if top2_hit
      draft_top5_hits += 1 if top5_hit
      if draft_token == exact_top1
        accepted_this_chunk += 1
        accepted_draft_tokens += 1
        emitted_tokens += 1
        emitted_ids << draft_token
        state_before_last = exact_state.fork
        last_token = draft_token
        pos_last = prompt_ids.size + emitted_tokens - 1
        exact_logits = logits_with_lowrank_policy(weights, draft_token, pos_last.to_i32, exact_state,
          layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
      else
        rejections += 1
        rescue_hit = false
        if k = topk_rescue
          if k > 1
            rescue_hit = top5[0, Math.min(k, top5.size)].includes?(exact_top1)
          end
        end
        if rescue_hit
          topk_rescues += 1
        else
          correction_steps += 1
        end
        emitted_tokens += 1
        emitted_ids << exact_top1
        state_before_last = exact_state.fork
        last_token = exact_top1
        pos_last = prompt_ids.size + emitted_tokens - 1
        exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, exact_state,
          layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
        reject_top2_hits += 1 if top2_hit
        reject_top5_hits += 1 if top5_hit
        rejected = true
        break
      end
      break if emitted_tokens >= gen_tokens
    end
    full_accept_chunks += 1 unless rejected
    # If the final chunk was shorter than gamma and fully accepted, this is still
    # a full accept for the proposed chunk length.
    if adaptive
      current_gamma = if rejected
                        Math.max(min_gamma, current_gamma // 2)
                      elsif threshold = adaptive_grow_margin_threshold
                        chunk_min_margin >= threshold ? Math.min(max_gamma, current_gamma * 2) : current_gamma
                      else
                        Math.min(max_gamma, current_gamma * 2)
                      end
    elsif progressive
      progressive_index = rejected ? 0 : ((progressive_index + 1) % progressive_schedule.not_nil!.size)
    end
  end

  accept_rate = proposed_tokens > 0 ? (100.0 * accepted_draft_tokens / proposed_tokens) : 0.0
  avg_accept = chunks > 0 ? (accepted_draft_tokens.to_f64 / chunks) : 0.0
  draft_top2_hit_rate = proposed_tokens > 0 ? (100.0 * draft_top2_hits / proposed_tokens) : 0.0
  draft_top5_hit_rate = proposed_tokens > 0 ? (100.0 * draft_top5_hits / proposed_tokens) : 0.0
  # Normalized to one exact sequential target decode per emitted token. Correction
  # steps consume exact target work outside the chunk verifier, so the remaining
  # budget is what the low-rank draft plus chunk verifier may spend per proposal.
  break_even = proposed_tokens > 0 ? ((gen_tokens - correction_steps).to_f64 / proposed_tokens) : 0.0

  {
    chunks:                               chunks,
    full_accept_chunks:                   full_accept_chunks,
    rejections:                           rejections,
    topk_rescues:                         topk_rescues,
    emitted_tokens:                       emitted_tokens,
    proposed_tokens:                      proposed_tokens,
    accepted_draft_tokens:                accepted_draft_tokens,
    verifier_tokens:                      verifier_tokens,
    correction_steps:                     correction_steps,
    approx_steps:                         approx_steps,
    fallback_steps:                       fallback_steps,
    draft_top2_hits:                      draft_top2_hits,
    draft_top5_hits:                      draft_top5_hits,
    reject_top2_hits:                     reject_top2_hits,
    reject_top5_hits:                     reject_top5_hits,
    accept_rate:                          accept_rate,
    avg_accept:                           avg_accept,
    break_even_draft_verify_per_proposed: break_even,
    draft_top2_hit_rate:                  draft_top2_hit_rate,
    draft_top5_hit_rate:                  draft_top5_hit_rate,
    gamma_history:                        gamma_history,
    verifier_history:                     verifier_history,
    draft_min_margin_history:             draft_min_margin_history,
    draft_low_margin_history:             draft_low_margin_history,
    exact_ids:                            exact_ids,
    emitted_ids:                          emitted_ids,
  }
end

private def simulate_self_spec_tree_oracle(weights : ML::GGUF::Qwen35Weights,
                                           prompt_ids : Array(Int32),
                                           gen_tokens : Int32,
                                           top_k : Int32,
                                           progressive_schedule : Array(Int32),
                                           layer_bases : LayerBasisMap,
                                           rank : Int32,
                                           calib_count : Int32,
                                           fallback_threshold : Float64?,
                                           refresh_interval : Int32?) : NamedTuple(chunks: Int32, full_rescue_chunks: Int32, misses: Int32, emitted_tokens: Int32, draft_steps: Int32, top1_hits: Int32, topk_hits: Int32, branch_tokens_rank: Int32, branch_tokens_full: Int32, correction_steps: Int32, approx_steps: Int32, fallback_steps: Int32, top1_rate: Float64, topk_rate: Float64, avg_rank_branch_tokens: Float64, avg_full_branch_tokens: Float64, schedule_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "tree top_k must be >= 2" unless top_k >= 2
  raise "tree top_k must be <= 16" unless top_k <= 16
  raise "tree progressive schedule must not be empty" if progressive_schedule.empty?
  raise "tree schedule values must be positive" if progressive_schedule.any? { |v| v <= 0 }

  hp = weights.hparams
  max_gamma = progressive_schedule.max
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end

  chunks = 0
  full_rescue_chunks = 0
  misses = 0
  emitted_tokens = 0
  draft_steps = 0
  top1_hits = 0
  topk_hits = 0
  branch_tokens_rank = 0
  branch_tokens_full = 0
  correction_steps = 0
  approx_steps = 0
  fallback_steps = 0
  progressive_index = 0
  schedule_history = [] of Int32
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    chunk_gamma = Math.min(progressive_schedule[progressive_index], gen_tokens - emitted_tokens)
    schedule_history << chunk_gamma
    draft_state = state_before_last.fork
    draft_lr_states = {} of Int32 => LowRankState
    sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
    draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)

    rescued_chunk = true
    chunk_gamma.times do |j|
      exact_top1 = top1(exact_logits)
      exact_ids << exact_top1
      draft_topk = top_k_indices(draft_logits, top_k)
      draft_steps += 1
      if draft_topk[0] == exact_top1
        top1_hits += 1
        topk_hits += 1
        branch_tokens_rank += 1
        branch_tokens_full += 1
      elsif idx = draft_topk.index(exact_top1)
        topk_hits += 1
        branch_tokens_rank += idx + 1
        branch_tokens_full += top_k
      else
        misses += 1
        correction_steps += 1
        branch_tokens_rank += top_k
        branch_tokens_full += top_k
        rescued_chunk = false
      end

      emitted_tokens += 1
      emitted_ids << exact_top1
      state_before_last = exact_state.fork
      last_token = exact_top1
      pos_last = prompt_ids.size + emitted_tokens - 1
      exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, exact_state,
        layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
      break if !rescued_chunk || emitted_tokens >= gen_tokens || j == chunk_gamma - 1

      draft_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, draft_state,
        layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)
    end

    approx_steps += draft_lr_states.values.sum(&.approx_steps)
    fallback_steps += draft_lr_states.values.sum(&.fallback_steps)
    full_rescue_chunks += 1 if rescued_chunk
    progressive_index = rescued_chunk ? ((progressive_index + 1) % progressive_schedule.size) : 0
  end

  {
    chunks:                 chunks,
    full_rescue_chunks:     full_rescue_chunks,
    misses:                 misses,
    emitted_tokens:         emitted_tokens,
    draft_steps:            draft_steps,
    top1_hits:              top1_hits,
    topk_hits:              topk_hits,
    branch_tokens_rank:     branch_tokens_rank,
    branch_tokens_full:     branch_tokens_full,
    correction_steps:       correction_steps,
    approx_steps:           approx_steps,
    fallback_steps:         fallback_steps,
    top1_rate:              emitted_tokens > 0 ? 100.0 * top1_hits / emitted_tokens : 0.0,
    topk_rate:              emitted_tokens > 0 ? 100.0 * topk_hits / emitted_tokens : 0.0,
    avg_rank_branch_tokens: emitted_tokens > 0 ? branch_tokens_rank.to_f64 / emitted_tokens : 0.0,
    avg_full_branch_tokens: emitted_tokens > 0 ? branch_tokens_full.to_f64 / emitted_tokens : 0.0,
    schedule_history:       schedule_history,
    exact_ids:              exact_ids,
    emitted_ids:            emitted_ids,
  }
end

private def train_topk_oracle_biases(samples : Array(TopKOracleSample),
                                     top_k : Int32) : NamedTuple(token_bias: Hash(Int32, Float64), rank_bias: Array(Float64))
  token_seen = Hash(Int32, Int32).new(0)
  token_hit = Hash(Int32, Int32).new(0)
  rank_seen = Array(Int32).new(top_k, 0)
  rank_hit = Array(Int32).new(top_k, 0)

  samples.each do |sample|
    sample.ids.each_with_index do |id, rank|
      token_seen[id] += 1
      token_hit[id] += 1 if id == sample.exact_id
      rank_seen[rank] += 1
      rank_hit[rank] += 1 if id == sample.exact_id
    end
  end

  alpha = 0.5
  total_seen = Math.max(1, samples.size * top_k)
  total_hit = samples.count { |s| s.ids.includes?(s.exact_id) }
  global_logit = Math.log((total_hit + alpha) / (total_seen - total_hit + alpha))

  token_bias = {} of Int32 => Float64
  token_seen.each do |id, seen|
    hit = token_hit[id]
    token_bias[id] = Math.log((hit + alpha) / (seen - hit + alpha)) - global_logit
  end

  rank_bias = Array(Float64).new(top_k, 0.0)
  top_k.times do |rank|
    seen = rank_seen[rank]
    hit = rank_hit[rank]
    rank_bias[rank] = seen > 0 ? Math.log((hit + alpha) / (seen - hit + alpha)) - global_logit : 0.0
  end

  {token_bias: token_bias, rank_bias: rank_bias}
end

private def eval_topk_oracle_samples(samples : Array(TopKOracleSample),
                                     token_bias : Hash(Int32, Float64),
                                     rank_bias : Array(Float64),
                                     token_scale : Float64,
                                     rank_scale : Float64) : NamedTuple(samples: Int32, top1_hits: Int32, topk_hits: Int32, misses: Int32, branch_tokens: Int32, top1_rate: Float64, topk_rate: Float64, avg_branch_tokens: Float64)
  top1_hits = 0
  topk_hits = 0
  misses = 0
  branch_tokens = 0

  samples.each do |sample|
    order = (0...sample.ids.size).to_a
    order.sort_by! do |rank|
      id = sample.ids[rank]
      -(sample.logits[rank].to_f64 + token_scale * (token_bias[id]? || 0.0) + rank_scale * rank_bias[rank])
    end

    reranked_ids = order.map { |rank| sample.ids[rank] }
    if reranked_ids[0]? == sample.exact_id
      top1_hits += 1
      topk_hits += 1
      branch_tokens += 1
    elsif idx = reranked_ids.index(sample.exact_id)
      topk_hits += 1
      branch_tokens += idx + 1
    else
      misses += 1
      branch_tokens += sample.ids.size
    end
  end

  n = samples.size
  {
    samples:           n,
    top1_hits:         top1_hits,
    topk_hits:         topk_hits,
    misses:            misses,
    branch_tokens:     branch_tokens,
    top1_rate:         n > 0 ? 100.0 * top1_hits / n : 0.0,
    topk_rate:         n > 0 ? 100.0 * topk_hits / n : 0.0,
    avg_branch_tokens: n > 0 ? branch_tokens.to_f64 / n : 0.0,
  }
end

private def eval_topk_margin_gate(samples : Array(TopKOracleSample),
                                  margin_threshold : Float64,
                                  correction_penalty : Float64) : NamedTuple(samples: Int32, gated_steps: Int32, top1_hits: Int32, topk_hits: Int32, misses: Int32, branch_tokens: Int32, estimated_cost: Float64, gate_rate: Float64, top1_rate: Float64, topk_rate: Float64, avg_branch_tokens: Float64)
  gated_steps = 0
  top1_hits = 0
  topk_hits = 0
  misses = 0
  branch_tokens = 0

  samples.each do |sample|
    exact_rank = sample.ids.index(sample.exact_id)
    if sample.margin < margin_threshold
      gated_steps += 1
      if exact_rank
        topk_hits += 1
        top1_hits += 1 if exact_rank == 0
        branch_tokens += exact_rank + 1
      else
        misses += 1
        branch_tokens += sample.ids.size
      end
    else
      branch_tokens += 1
      if sample.ids[0]? == sample.exact_id
        top1_hits += 1
        topk_hits += 1
      elsif exact_rank
        # The token was available in topK, but the margin gate chose the cheap
        # top1 path, so the verifier would need a correction/resync.
        misses += 1
      else
        misses += 1
      end
    end
  end

  n = samples.size
  {
    samples:           n,
    gated_steps:       gated_steps,
    top1_hits:         top1_hits,
    topk_hits:         topk_hits,
    misses:            misses,
    branch_tokens:     branch_tokens,
    estimated_cost:    branch_tokens.to_f64 + correction_penalty * misses,
    gate_rate:         n > 0 ? 100.0 * gated_steps / n : 0.0,
    top1_rate:         n > 0 ? 100.0 * top1_hits / n : 0.0,
    topk_rate:         n > 0 ? 100.0 * topk_hits / n : 0.0,
    avg_branch_tokens: n > 0 ? branch_tokens.to_f64 / n : 0.0,
  }
end

private def simulate_topk_oracle_calibration(weights : ML::GGUF::Qwen35Weights,
                                             prompt_ids : Array(Int32),
                                             gen_tokens : Int32,
                                             top_k : Int32,
                                             train_tokens : Int32?,
                                             layer_bases : LayerBasisMap,
                                             rank : Int32,
                                             calib_count : Int32,
                                             fallback_threshold : Float64?,
                                             refresh_interval : Int32?) : NamedTuple(samples: Int32, train_samples: Int32, test_samples: Int32, best_token_scale: Float64, best_rank_scale: Float64, best_margin_threshold: Float64, train_top1_rate: Float64, train_topk_rate: Float64, train_avg_branch_tokens: Float64, baseline_top1_rate: Float64, baseline_topk_rate: Float64, baseline_avg_branch_tokens: Float64, calibrated_top1_rate: Float64, calibrated_topk_rate: Float64, calibrated_avg_branch_tokens: Float64, margin_gate_rate: Float64, margin_gate_topk_rate: Float64, margin_gate_avg_branch_tokens: Float64, margin_gate_misses: Int32, margin_gate_cost: Float64, baseline_misses: Int32, calibrated_misses: Int32, exact_ids: Array(Int32))
  raise "topK oracle top_k must be >= 2" unless top_k >= 2
  raise "topK oracle top_k must be <= 16" unless top_k <= 16
  raise "topK oracle gen_tokens must be >= 4" unless gen_tokens >= 4

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end

  draft_state = state_before_last.fork
  draft_lr_states = {} of Int32 => LowRankState
  sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
  draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
    layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)

  samples = [] of TopKOracleSample
  exact_ids = [] of Int32
  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    samples << topk_oracle_sample(draft_logits, exact_top1, top_k)
    exact_ids << exact_top1

    pos_last = prompt_ids.size + step
    exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
    break if step == gen_tokens - 1
    draft_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)
  end

  requested_train = train_tokens || (samples.size // 2)
  train_count = requested_train.clamp(1, samples.size - 1)
  train = samples[0, train_count]
  test = samples[train_count, samples.size - train_count]
  biases = train_topk_oracle_biases(train, top_k)
  zero_token_bias = {} of Int32 => Float64
  zero_rank_bias = Array(Float64).new(top_k, 0.0)
  baseline = eval_topk_oracle_samples(test, zero_token_bias, zero_rank_bias, 0.0, 0.0)

  token_scales = [0.0, 0.25, 0.5, 1.0, 2.0]
  rank_scales = [0.0, 0.25, 0.5, 1.0]
  best_token_scale = 0.0
  best_rank_scale = 0.0
  best_train = eval_topk_oracle_samples(train, biases[:token_bias], biases[:rank_bias], 0.0, 0.0)
  token_scales.each do |ts|
    rank_scales.each do |rs|
      cur = eval_topk_oracle_samples(train, biases[:token_bias], biases[:rank_bias], ts, rs)
      if cur[:avg_branch_tokens] < best_train[:avg_branch_tokens] ||
         (cur[:avg_branch_tokens] == best_train[:avg_branch_tokens] && cur[:top1_rate] > best_train[:top1_rate])
        best_train = cur
        best_token_scale = ts
        best_rank_scale = rs
      end
    end
  end

  calibrated = eval_topk_oracle_samples(test, biases[:token_bias], biases[:rank_bias], best_token_scale, best_rank_scale)
  correction_penalty = top_k.to_f64
  thresholds = [-1.0] + train.map(&.margin).uniq.sort + [Float64::INFINITY]
  best_margin_threshold = thresholds[0]
  best_margin_train = eval_topk_margin_gate(train, best_margin_threshold, correction_penalty)
  thresholds.each do |threshold|
    cur = eval_topk_margin_gate(train, threshold, correction_penalty)
    if cur[:estimated_cost] < best_margin_train[:estimated_cost] ||
       (cur[:estimated_cost] == best_margin_train[:estimated_cost] && cur[:gated_steps] < best_margin_train[:gated_steps])
      best_margin_train = cur
      best_margin_threshold = threshold
    end
  end
  margin_gate = eval_topk_margin_gate(test, best_margin_threshold, correction_penalty)
  {
    samples:                       samples.size,
    train_samples:                 train.size,
    test_samples:                  test.size,
    best_token_scale:              best_token_scale,
    best_rank_scale:               best_rank_scale,
    best_margin_threshold:         best_margin_threshold,
    train_top1_rate:               best_train[:top1_rate],
    train_topk_rate:               best_train[:topk_rate],
    train_avg_branch_tokens:       best_train[:avg_branch_tokens],
    baseline_top1_rate:            baseline[:top1_rate],
    baseline_topk_rate:            baseline[:topk_rate],
    baseline_avg_branch_tokens:    baseline[:avg_branch_tokens],
    calibrated_top1_rate:          calibrated[:top1_rate],
    calibrated_topk_rate:          calibrated[:topk_rate],
    calibrated_avg_branch_tokens:  calibrated[:avg_branch_tokens],
    margin_gate_rate:              margin_gate[:gate_rate],
    margin_gate_topk_rate:         margin_gate[:topk_rate],
    margin_gate_avg_branch_tokens: margin_gate[:avg_branch_tokens],
    margin_gate_misses:            margin_gate[:misses],
    margin_gate_cost:              margin_gate[:estimated_cost],
    baseline_misses:               baseline[:misses],
    calibrated_misses:             calibrated[:misses],
    exact_ids:                     exact_ids,
  }
end

private def simulate_self_spec_wall_policy(weights : ML::GGUF::Qwen35Weights,
                                           prompt_ids : Array(Int32),
                                           gen_tokens : Int32,
                                           progressive_schedule : Array(Int32),
                                           layer_bases : LayerBasisMap,
                                           rank : Int32,
                                           calib_count : Int32,
                                           fallback_threshold : Float64?,
                                           refresh_interval : Int32?,
                                           use_metal_lowrank : Bool,
                                           project_coeffs_on_gpu : Bool,
                                           use_metal_layer_updown : Bool = false,
                                           draft_variant : String = "lowrank",
                                           ffn_bases : FFNBasisMap? = nil,
                                           ffn_adapters : FFNAdapterMap? = nil,
                                           ffn_updown_adapters : FFNUpDownAdapterMap? = nil) : NamedTuple(chunks: Int32, rejections: Int32, accepted_draft_tokens: Int32, proposed_tokens: Int32, verifier_tokens: Int32, correction_steps: Int32, draft_ms: Float64, verifier_ms: Float64, replay_ms: Float64, serial_ms: Float64, overlap_est_ms: Float64, speedup_est: Float64, accept_rate: Float64, exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "wall self-spec requires a non-empty progressive schedule" if progressive_schedule.empty?
  raise "wall self-spec schedule values must be positive" if progressive_schedule.any? { |v| v <= 0 }

  hp = weights.hparams
  max_gamma = progressive_schedule.max
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4

  # CPU shadow state feeds the projected-K draft branch. The verifier state uses
  # the production chunk verifier path, which can route its exact work to Metal.
  shadow_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(verifier_state, hp)

  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = shadow_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = shadow_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, shadow_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end
  target_next, _ = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, verifier_state)
  raise "shadow/verifier prompt top1 mismatch: #{top1(exact_logits)} != #{target_next}" unless top1(exact_logits) == target_next

  emitted_tokens = 0
  chunks = 0
  rejections = 0
  proposed_tokens = 0
  accepted_draft_tokens = 0
  verifier_tokens = 0
  correction_steps = 0
  progressive_index = 0
  target_next_id = target_next.to_i32
  draft_ms = 0.0
  verifier_ms = 0.0
  replay_ms = 0.0
  chunk_draft_ms = [] of Float64
  chunk_verifier_ms = [] of Float64
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    chunk_gamma = Math.min(progressive_schedule[progressive_index], gen_tokens - emitted_tokens)

    t_draft = Time.instant
    draft_state = state_before_last.fork
    draft_lr_states = {} of Int32 => LowRankState
    sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
    draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true, use_metal_lowrank, project_coeffs_on_gpu, use_metal_layer_updown, draft_variant, ffn_bases, ffn_adapters, ffn_updown_adapters)
    proposal = [] of Int32
    chunk_gamma.times do |j|
      proposed = top1(draft_logits)
      proposal << proposed
      break if j == chunk_gamma - 1
      draft_logits = logits_with_lowrank_policy(weights, proposed, (pos_last + 1 + j).to_i32, draft_state,
        layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true, use_metal_lowrank, project_coeffs_on_gpu, use_metal_layer_updown, draft_variant, ffn_bases, ffn_adapters, ffn_updown_adapters)
    end
    dt_draft = (Time.instant - t_draft).total_milliseconds
    draft_ms += dt_draft
    chunk_draft_ms << dt_draft
    proposed_tokens += proposal.size
    verifier_tokens += proposal.size

    cycle_start_pos = prompt_ids.size + emitted_tokens
    verifier_backup = verifier_state.fork
    t_verify = Time.instant
    target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, proposal, cycle_start_pos, verifier_state)
    dt_verify = (Time.instant - t_verify).total_milliseconds
    verifier_ms += dt_verify
    chunk_verifier_ms << dt_verify

    correction_or_accepted = [] of Int32
    expected = target_next_id
    rejected = false
    proposal.each_with_index do |cand, i|
      exact_ids << expected
      emitted = if cand == expected
                  accepted_draft_tokens += 1
                  cand
                else
                  rejections += 1
                  correction_steps += 1
                  rejected = true
                  expected
                end
      correction_or_accepted << emitted
      emitted_ids << emitted
      emitted_tokens += 1

      pos = cycle_start_pos + i
      state_before_last = shadow_state.fork
      last_token = emitted
      pos_last = pos
      exact_logits = logits_with_lowrank_policy(weights, emitted, pos.to_i32, shadow_state,
        layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
      expected = target_nexts[i][0] if cand == expected
      break if rejected || emitted_tokens >= gen_tokens
    end

    if rejected
      verifier_state.copy_from!(verifier_backup)
      t_replay = Time.instant
      corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, cycle_start_pos, verifier_state)
      replay_ms += (Time.instant - t_replay).total_milliseconds
      target_next_id = corrected[-1][0]
      progressive_index = 0
    else
      target_next_id = target_nexts[correction_or_accepted.size - 1][0]
      progressive_index = (progressive_index + 1) % progressive_schedule.size
    end
  end

  serial_ms = draft_ms + verifier_ms + replay_ms
  overlap_est_ms = 0.0
  chunk_draft_ms.each_with_index do |d_ms, i|
    v_ms = chunk_verifier_ms[i]? || 0.0
    overlap_est_ms += Math.max(d_ms, v_ms)
  end
  overlap_est_ms += replay_ms
  speedup_est = overlap_est_ms > 0.0 ? serial_ms / overlap_est_ms : 0.0
  accept_rate = proposed_tokens > 0 ? (100.0 * accepted_draft_tokens / proposed_tokens) : 0.0

  {
    chunks:                chunks,
    rejections:            rejections,
    accepted_draft_tokens: accepted_draft_tokens,
    proposed_tokens:       proposed_tokens,
    verifier_tokens:       verifier_tokens,
    correction_steps:      correction_steps,
    draft_ms:              draft_ms,
    verifier_ms:           verifier_ms,
    replay_ms:             replay_ms,
    serial_ms:             serial_ms,
    overlap_est_ms:        overlap_est_ms,
    speedup_est:           speedup_est,
    accept_rate:           accept_rate,
    exact_ids:             exact_ids,
    emitted_ids:           emitted_ids,
  }
end

private def parse_int_list(value : String) : Array(Int32)
  value.split(',').map(&.strip).reject(&.empty?).map(&.to_i)
end

private def cheap_draft_variant_valid?(variant : String) : Bool
  return true if {"lowrank", "lowrank-no-ffn", "skip-layer"}.includes?(variant)
  return true if draft_variant_ffn_top_percent(variant)
  return true if draft_variant_ffn_pca_rank(variant)
  return true if draft_variant_ffn_pca_down_rank(variant)
  return true if draft_variant_ffn_pca_updown_rank(variant)
  return false unless variant.starts_with?("early-exit-")

  variant["early-exit-".size..].to_i? ? true : false
end

private def cheap_draft_early_exit_layers(variant : String) : Int32?
  return nil unless variant.starts_with?("early-exit-")

  n = variant["early-exit-".size..].to_i? || raise "invalid early-exit variant #{variant.inspect}"
  raise "early-exit layer count must be positive" unless n > 0
  n
end

private def self_spec_estimated_cost(spec,
                                     draft_cost : Float64,
                                     verifier_cost : Float64,
                                     chunk_overhead : Float64,
                                     correction_cost : Float64,
                                     overlap : Bool,
                                     overlap_efficiency : Float64) : Float64
  if overlap
    efficiency = overlap_efficiency.clamp(0.0, 1.0)
    cost = 0.0
    spec[:gamma_history].each_with_index do |draft_tokens, i|
      verifier_tokens = spec[:verifier_history][i]
      draft_segment = draft_cost * draft_tokens
      verifier_segment = verifier_cost * verifier_tokens
      hidden = Math.min(draft_segment, verifier_segment) * efficiency
      cost += draft_segment + verifier_segment - hidden + chunk_overhead
    end
    cost + correction_cost * spec[:correction_steps]
  else
    draft_cost * spec[:proposed_tokens] +
      verifier_cost * spec[:verifier_tokens] +
      chunk_overhead * spec[:chunks] +
      correction_cost * spec[:correction_steps]
  end
end

private def self_spec_tree_estimated_cost(tree,
                                          draft_cost : Float64,
                                          verifier_cost : Float64,
                                          chunk_overhead : Float64,
                                          correction_cost : Float64,
                                          branch_tokens : Int32) : Float64
  draft_cost * tree[:draft_steps] +
    verifier_cost * branch_tokens +
    chunk_overhead * tree[:chunks] +
    correction_cost * tree[:correction_steps]
end

private def simulate_self_draft_metal_baseline_run(weights : ML::GGUF::Qwen35Weights,
                                                   token_ids : Array(Int32),
                                                   calib_count : Int32,
                                                   n_draft : Int32,
                                                   layer_bases : Hash(Int32, BasisSet),
                                                   rank : Int32) : NamedTuple(steps: Int32, self_draft_ms: Float64, exact_ms: Float64, verifier_ms: Float64, self_draft_per_token_ms: Float64, exact_per_token_ms: Float64, verifier_per_token_ms: Float64, self_spec_wall_ratio: Float64, agreement: Int32, self_draft_ids: Array(Int32), exact_ids: Array(Int32), verifier_ids: Array(Int32))
  raise "Metal unavailable for self-draft baseline" unless ML::GGUF::Qwen35Metal.available?
  raise "n_draft must be positive" unless n_draft > 0
  raise "calib_count must leave a non-empty held-out span >= n_draft" unless calib_count + n_draft <= token_ids.size
  raise "layer_bases must not be empty" if layer_bases.empty?

  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  input_ids = token_ids[calib_count, n_draft]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State) {
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      ssm_buf = state.layers[il].ssm_state_buf
      buf = if ssm_buf
              ML::GGUF::Qwen35Metal.lowrank_project_state_buf(ssm_buf, shared_basis_bufs[il],
                h_k, h_v, s, rank, command_queue_name: "self_spec_gpu_pipeline_draft")
            else
              cpu_buf = ML::MetalBuffer.new(state_size_bytes)
              full_state = state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
              cpu_buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
              cpu_buf
            end
      bufs[il] = buf
    end
    bufs
  }

  warmup_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  warmup_lr = build_lr_states.call(warmup_state)
  ML::GGUF::Qwen35CPU.forward_self_draft_top1(weights, input_ids[0], calib_count, warmup_state,
    lowrank_set, warmup_lr, shared_basis_bufs, rank).not_nil!

  warmup_exact = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, input_ids[0], calib_count, warmup_exact)

  warmup_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, input_ids, calib_count, warmup_verify)

  self_draft_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  self_lr = build_lr_states.call(self_draft_state)
  self_draft_ids = [] of Int32
  t_self = Time.instant
  input_ids.each_with_index do |tok, i|
    out = ML::GGUF::Qwen35CPU.forward_self_draft_top1(weights, tok, calib_count + i, self_draft_state,
      lowrank_set, self_lr, shared_basis_bufs, rank).not_nil!
    self_draft_ids << out[0]
  end
  self_draft_ms = (Time.instant - t_self).total_milliseconds

  exact_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  exact_ids = [] of Int32
  t_exact = Time.instant
  input_ids.each_with_index do |tok, i|
    out = ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, exact_state)
    exact_ids << out[0]
  end
  exact_ms = (Time.instant - t_exact).total_milliseconds

  verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_verify = Time.instant
  verify_results = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, input_ids, calib_count, verify_state)
  verifier_ms = (Time.instant - t_verify).total_milliseconds
  verifier_ids = verify_results.map { |r| r[0] }

  agreement = self_draft_ids.zip(exact_ids).count { |pair| pair[0] == pair[1] }
  wall_total = self_draft_ms + verifier_ms

  {
    steps:                   n_draft,
    self_draft_ms:           self_draft_ms,
    exact_ms:                exact_ms,
    verifier_ms:             verifier_ms,
    self_draft_per_token_ms: self_draft_ms / n_draft,
    exact_per_token_ms:      exact_ms / n_draft,
    verifier_per_token_ms:   verifier_ms / n_draft,
    self_spec_wall_ratio:    wall_total > 0.0 ? exact_ms / wall_total : 0.0,
    agreement:               agreement,
    self_draft_ids:          self_draft_ids,
    exact_ids:               exact_ids,
    verifier_ids:            verifier_ids,
  }
end

private def simulate_self_draft_gpu_chain_run(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              calib_count : Int32,
                                              n_draft : Int32,
                                              layer_bases : LayerBasisMap,
                                              rank : Int32) : NamedTuple(steps: Int32, submit_ms: Float64, wait_ms: Float64, chain_ms: Float64, exact_ms: Float64, agreement: Int32, chain_ids: Array(Int32), exact_ids: Array(Int32))
  raise "self-draft GPU chain requires at least one held-out token" unless n_draft > 0
  raise "self-draft GPU chain requires Metal" unless ML::GGUF::Qwen35Metal.available?
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  first_token = token_ids[calib_count]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State) {
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      buf = ML::MetalBuffer.new(state_size_bytes)
      ssm_buf = state.layers[il].ssm_state_buf
      full_state = if ssm_buf
                     ssm_buf.read(full_state_size)
                   else
                     state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
                   end
      buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
      bufs[il] = buf
    end
    bufs
  }

  exact_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  exact_ids = [] of Int32
  exact_tok = first_token
  t_exact = Time.instant
  n_draft.times do |i|
    out = ML::GGUF::Qwen35CPU.forward_top1(weights, exact_tok, calib_count + i, exact_state)
    exact_ids << out[0]
    exact_tok = out[0]
  end
  exact_ms = (Time.instant - t_exact).total_milliseconds

  chain_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  chain_lr = build_lr_states.call(chain_state)
  initial_token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
  initial_token_buf.contents.as(Pointer(UInt32)).value = first_token.to_u32
  token_buf = initial_token_buf
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  wba = WbaTrace.maybe("self_draft_gpu_chain")

  t_chain = Time.instant
  t_submit = Time.instant
  n_draft.times do |i|
    t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_self_draft_top1_from_token_buf_async(weights, token_buf, 0, calib_count + i, chain_state,
      lowrank_set, chain_lr, shared_basis_bufs, rank, scratch_namespace: "self_draft_gpu_chain_#{i}").not_nil!
    wba.try(&.mark("draft", "submit_#{i}", t0, Time.instant))
    submissions << sub
    token_buf = sub.top1_id_buf.not_nil!
  end
  submit_ms = (Time.instant - t_submit).total_milliseconds

  chain_ids = [] of Int32
  t_wait = Time.instant
  submissions.each_with_index do |sub, i|
    t0 = Time.instant
    packed = ML::GGUF::Qwen35Metal.wait_forward_decode_wave(sub)
    wba.try(&.mark("draft", "wait_read_#{i}", t0, Time.instant))
    raise "GPU chain decode returned #{packed.size} values" unless packed.size == 2
    chain_ids << packed[0].to_i32
  end
  wba.try(&.flush)
  wait_ms = (Time.instant - t_wait).total_milliseconds
  chain_ms = (Time.instant - t_chain).total_milliseconds
  agreement = chain_ids.zip(exact_ids).count { |pair| pair[0] == pair[1] }

  {
    steps:     n_draft,
    submit_ms: submit_ms,
    wait_ms:   wait_ms,
    chain_ms:  chain_ms,
    exact_ms:  exact_ms,
    agreement: agreement,
    chain_ids: chain_ids,
    exact_ids: exact_ids,
  }
end

private def simulate_self_draft_gpu_state_only_run(weights : ML::GGUF::Qwen35Weights,
                                                   token_ids : Array(Int32),
                                                   calib_count : Int32,
                                                   n_draft : Int32,
                                                   layer_bases : LayerBasisMap,
                                                   rank : Int32) : NamedTuple(steps: Int32, project_ms: Float64, submit_ms: Float64, wait_ms: Float64, chain_ms: Float64, per_token_ms: Float64)
  raise "self-draft GPU state-only requires at least one held-out token" unless n_draft > 0
  raise "self-draft GPU state-only requires Metal" unless ML::GGUF::Qwen35Metal.available?
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  known_tokens = token_ids[calib_count, n_draft]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end

  state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_project = Time.instant
  lr_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    ssm_buf = state.layers[il].ssm_state_buf
    buf = if ssm_buf
            ML::GGUF::Qwen35Metal.lowrank_project_state_buf(ssm_buf, shared_basis_bufs[il],
              h_k, h_v, s, rank, command_queue_name: "self_draft_gpu_state_only")
          else
            cpu_buf = ML::MetalBuffer.new(state_size_bytes)
            full_state = state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
            cpu_buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
            cpu_buf
          end
    lr_bufs[il] = buf
  end
  project_ms = (Time.instant - t_project).total_milliseconds

  token_buf = ML::MetalBuffer.new(n_draft.to_i64 * sizeof(UInt32))
  token_ptr = token_buf.contents.as(Pointer(UInt32))
  known_tokens.each_with_index do |tok, i|
    token_ptr[i] = tok.to_u32
  end

  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  wba = WbaTrace.maybe("self_draft_gpu_state_only")
  cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_draft_gpu_state_only")

  t_chain = Time.instant
  t_submit = Time.instant
  n_draft.times do |i|
    t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_self_draft_state_from_token_buf_async(weights, token_buf, i, calib_count + i, state,
      lowrank_set, lr_bufs, shared_basis_bufs, rank,
      scratch_namespace: "self_draft_gpu_state_only_#{i}",
      command_queue_name: "self_draft_gpu_state_only",
      append_command_buffer: cmd).not_nil!
    wba.try(&.mark("draft", "submit_state_only_#{i}", t0, Time.instant))
    submissions << sub
  end
  cmd.commit
  submit_ms = (Time.instant - t_submit).total_milliseconds

  t_wait = Time.instant
  cmd.wait
  submissions.each do |sub|
    sub.pending_cmds.each(&.wait)
  end
  wait_ms = (Time.instant - t_wait).total_milliseconds
  wba.try(&.mark("draft", "wait_state_only_block", t_wait, Time.instant))
  wba.try(&.flush)
  chain_ms = (Time.instant - t_chain).total_milliseconds

  {
    steps:        n_draft,
    project_ms:   project_ms,
    submit_ms:    submit_ms,
    wait_ms:      wait_ms,
    chain_ms:     chain_ms,
    per_token_ms: chain_ms / n_draft,
  }
end

private def simulate_self_draft_gpu_chain_overlap_run(weights : ML::GGUF::Qwen35Weights,
                                                      token_ids : Array(Int32),
                                                      calib_count : Int32,
                                                      n_draft : Int32,
                                                      layer_bases : LayerBasisMap,
                                                      rank : Int32) : NamedTuple(steps: Int32, draft_alone_ms: Float64, verifier_ms: Float64, overlap_ms: Float64, draft_submit_ms: Float64, draft_wait_ms: Float64, hidden_ms: Float64, speedup: Float64, agreement: Int32, draft_ids: Array(Int32), exact_ids: Array(Int32), verifier_ids: Array(Int32))
  solo = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, n_draft, layer_bases, rank)
  raise "self-draft GPU chain overlap requires Metal" unless ML::GGUF::Qwen35Metal.available?
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  candidates = token_ids[calib_count, n_draft]
  first_token = candidates[0]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State) {
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      buf = ML::MetalBuffer.new(state_size_bytes)
      ssm_buf = state.layers[il].ssm_state_buf
      full_state = if ssm_buf
                     ssm_buf.read(full_state_size)
                   else
                     state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
                   end
      buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
      bufs[il] = buf
    end
    bufs
  }

  draft_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  draft_lr = build_lr_states.call(draft_state)
  verifier_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  initial_token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
  initial_token_buf.contents.as(Pointer(UInt32)).value = first_token.to_u32
  token_buf = initial_token_buf
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  wba = WbaTrace.maybe("self_draft_gpu_chain_overlap")

  t_overlap = Time.instant
  t_submit = Time.instant
  n_draft.times do |i|
    t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_self_draft_top1_from_token_buf_async(weights, token_buf, 0, calib_count + i, draft_state,
      lowrank_set, draft_lr, shared_basis_bufs, rank,
      scratch_namespace: "self_draft_gpu_chain_overlap_#{i}",
      command_queue_name: "self_draft_gpu_chain_overlap").not_nil!
    wba.try(&.mark("draft", "submit_#{i}", t0, Time.instant))
    submissions << sub
    token_buf = sub.top1_id_buf.not_nil!
  end
  draft_submit_ms = (Time.instant - t_submit).total_milliseconds

  t_verify = Time.instant
  verifier = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, verifier_state)
  verifier_ms = (Time.instant - t_verify).total_milliseconds
  wba.try(&.mark("verifier", "chunk_major", t_verify, Time.instant))

  draft_ids = [] of Int32
  t_wait = Time.instant
  submissions.each_with_index do |sub, i|
    t0 = Time.instant
    packed = ML::GGUF::Qwen35Metal.wait_forward_decode_wave(sub)
    wba.try(&.mark("draft", "wait_read_#{i}", t0, Time.instant))
    raise "GPU chain overlap decode returned #{packed.size} values" unless packed.size == 2
    draft_ids << packed[0].to_i32
  end
  draft_wait_ms = (Time.instant - t_wait).total_milliseconds
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.flush)

  serial_ms = solo[:chain_ms] + verifier_ms
  hidden_ms = serial_ms - overlap_ms
  {
    steps:           n_draft,
    draft_alone_ms:  solo[:chain_ms],
    verifier_ms:     verifier_ms,
    overlap_ms:      overlap_ms,
    draft_submit_ms: draft_submit_ms,
    draft_wait_ms:   draft_wait_ms,
    hidden_ms:       hidden_ms,
    speedup:         overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    agreement:       draft_ids.zip(solo[:exact_ids]).count { |pair| pair[0] == pair[1] },
    draft_ids:       draft_ids,
    exact_ids:       solo[:exact_ids],
    verifier_ids:    verifier.map { |r| r[0] },
  }
end

private def simulate_self_spec_gpu_pipeline_run(weights : ML::GGUF::Qwen35Weights,
                                                prompt_ids : Array(Int32),
                                                gen_tokens : Int32,
                                                gamma : Int32,
                                                layer_bases : LayerBasisMap,
                                                rank : Int32,
                                                use_verifier_backup : Bool = true,
                                                draft_block_tokens : Int32? = nil,
                                                draft_no_ffn : Bool = false,
                                                draft_skip_recurrent_ffn : Bool = false,
                                                gamma_schedule : Array(Int32)? = nil,
                                                draft_updown_rank : Int32? = nil,
                                                ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                                draft_updown_fallback_on_reject : Bool = false,
                                                draft_updown_after_full_accepts : Int32 = 0,
                                                draft_no_ffn_layer_indices : Set(Int32)? = nil,
                                                draft_updown_layer_indices : Set(Int32)? = nil) : NamedTuple(chunks: Int32, rejections: Int32, accepted_draft_tokens: Int32, proposed_tokens: Int32, draft_updown_chunks: Int32, draft_seed_ms: Float64, draft_next_ms: Float64, verifier_ms: Float64, draft_wait_ms: Float64, backup_ms: Float64, rebuild_ms: Float64, controller_ms: Float64, plain_exact_ms: Float64, serial_ms: Float64, overlap_ms: Float64, replay_ms: Float64, hidden_ms: Float64, speedup: Float64, plain_speedup: Float64, parity: Bool, gamma_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32), draft_steps: Int32, draft_blocks: Int32, draft_fork_ms: Float64, draft_token_buf_ms: Float64, draft_lr_project_ms: Float64, draft_submit_ms: Float64, draft_commit_ms: Float64, draft_wait_block_ms: Float64, draft_read_ids_ms: Float64, draft_resync_ms: Float64, verifier_initial_ms: Float64, verifier_prefill_ms: Float64, verifier_chunks: Int32, verifier_tokens: Int32, verifier_tail_skip_tokens: Int32)
  raise "GPU pipeline requires Metal" unless ML::GGUF::Qwen35Metal.available?
  raise "GPU pipeline gamma must be positive" unless gamma > 0
  raise "GPU pipeline gen_tokens must be positive" unless gen_tokens > 0
  raise "GPU pipeline pca-updown warmup must be non-negative" if draft_updown_after_full_accepts < 0
  raise "GPU pipeline requires non-empty prompt" if prompt_ids.empty?
  schedule = gamma_schedule && !gamma_schedule.not_nil!.empty? ? gamma_schedule.not_nil! : [gamma]
  raise "GPU pipeline schedule values must be positive" if schedule.any? { |v| v <= 0 }
  max_gamma = schedule.max

  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 8
  prefix_ids = prompt_ids[0, prompt_ids.size - 1]
  prompt_last_token = prompt_ids[-1]
  prompt_pos_last = prompt_ids.size - 1
  last_token = prompt_last_token
  pos_last = prompt_pos_last

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s
  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end
  if first_basis = shared_basis_bufs.values.first?
    warm_full_state = ML::MetalBuffer.new(full_state_size.to_i64 * sizeof(Float32))
    warm_full_state.contents.as(Pointer(UInt8)).clear(warm_full_state.size)
    ML::GGUF::Qwen35Metal.lowrank_project_state_buf(warm_full_state, first_basis,
      h_k, h_v, s, rank, command_queue_name: "self_spec_gpu_pipeline_draft")
  end
  updown_x_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_c_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_coeff_w_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_down_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_actual_rank = 0
  if updown_rank = draft_updown_rank
    raise "GPU pipeline pca-updown cannot be combined with global draft_no_ffn" if draft_no_ffn
    raise "GPU pipeline pca-updown cannot be combined with draft_skip_recurrent_ffn" if draft_skip_recurrent_ffn
    if no_ffn_set = draft_no_ffn_layer_indices
      updown_set_for_conflict = draft_updown_layer_indices || lowrank_set
      overlap = no_ffn_set.select { |il| updown_set_for_conflict.includes?(il) }
      raise "GPU pipeline pca-updown/no-ffn layer sets overlap: #{overlap.to_a.sort.join(',')}" unless overlap.empty?
    end
    adapters = ffn_updown_adapters || raise "GPU pipeline pca-updown requires FFN up/down adapters"
    maps = build_updown_adapter_buffer_maps(adapters, draft_updown_layer_indices || lowrank_set, updown_rank, hp.n_embd)
    updown_x_mean_bufs = maps[:x_mean]
    updown_c_mean_bufs = maps[:c_mean]
    updown_coeff_w_bufs = maps[:coeff_w]
    updown_down_bufs = maps[:down]
    updown_actual_rank = maps[:rank]
  end
  wba = WbaTrace.maybe("self_spec_gpu_pipeline")
  attr_collect = true
  draft_steps = 0
  draft_blocks = 0
  draft_fork_ms = 0.0
  draft_token_buf_ms = 0.0
  draft_lr_project_ms = 0.0
  draft_submit_ms = 0.0
  draft_commit_ms = 0.0
  draft_wait_block_ms = 0.0
  draft_read_ids_ms = 0.0
  draft_resync_ms = 0.0
  verifier_initial_ms = 0.0
  verifier_prefill_ms = 0.0
  verifier_chunks = 0
  verifier_tokens_count = 0
  verifier_tail_skip_tokens = 0

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State,
                       block_cmd : ML::Metal::CommandBuffer?) {
    t_lr_build = Time.instant
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      ssm_buf = state.layers[il].ssm_state_buf
      buf = if ssm_buf
              if cmd = block_cmd
                ML::GGUF::Qwen35Metal.lowrank_project_state_append(ssm_buf, shared_basis_bufs[il],
                  h_k, h_v, s, rank, cmd)
              else
                ML::GGUF::Qwen35Metal.lowrank_project_state_buf(ssm_buf, shared_basis_bufs[il],
                  h_k, h_v, s, rank, command_queue_name: "self_spec_gpu_pipeline_draft")
              end
            else
              cpu_buf = ML::MetalBuffer.new(state_size_bytes)
              full_state = state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
              cpu_buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
              cpu_buf
            end
      bufs[il] = buf
    end
    draft_lr_project_ms += (Time.instant - t_lr_build).total_milliseconds if attr_collect
    bufs
  }

  submit_block = ->(state : ML::GGUF::Qwen35CPU::State,
                    lr_bufs : Hash(Int32, ML::MetalBuffer),
                    token_buf : ML::MetalBuffer,
                    pos_start : Int32,
                    label : String,
                    block_cmd : ML::Metal::CommandBuffer?,
                    steps : Int32,
                    use_updown : Bool) {
    submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
    if attr_collect
      draft_blocks += 1
      draft_steps += steps
    end
    cur_token_buf = token_buf
    split_tokens = draft_block_tokens || (ENV["QWEN35_DRAFT_BLOCK_TOKENS"]?.try(&.to_i?) || DEFAULT_SELF_SPEC_GPU_PIPELINE_DRAFT_BLOCK_TOKENS)
    current_cmd = block_cmd || ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
    steps.times do |j|
      if split_tokens > 0 && j > 0 && (j % split_tokens) == 0
        t_commit = Time.instant
        current_cmd.commit
        draft_commit_ms += (Time.instant - t_commit).total_milliseconds if attr_collect
        wba.try(&.mark("draft", "commit_#{label}_part_#{j // split_tokens}", t_commit, Time.instant))
        current_cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
      end
      t_submit = Time.instant
      sub = ML::GGUF::Qwen35CPU.forward_self_draft_top1_from_token_buf_async(weights, cur_token_buf, 0, pos_start + j, state,
        lowrank_set, lr_bufs, shared_basis_bufs, rank,
        lowrank_skip_ffn: draft_no_ffn,
        skip_recurrent_ffn: draft_skip_recurrent_ffn,
        lowrank_skip_ffn_layer_indices: draft_no_ffn_layer_indices,
        lowrank_updown_x_mean_bufs: updown_x_mean_bufs,
        lowrank_updown_c_mean_bufs: updown_c_mean_bufs,
        lowrank_updown_coeff_w_bufs: updown_coeff_w_bufs,
        lowrank_updown_down_bufs: updown_down_bufs,
        lowrank_updown_rank: use_updown ? updown_actual_rank : 0,
        lowrank_updown_layer_indices: draft_updown_layer_indices,
        scratch_namespace: "#{label}_#{j}",
        append_command_buffer: current_cmd).not_nil!
      draft_submit_ms += (Time.instant - t_submit).total_milliseconds if attr_collect
      wba.try(&.mark("draft", "submit_#{label}_#{j}", t_submit, Time.instant))
      submissions << sub
      cur_token_buf = sub.top1_id_buf.not_nil!
    end
    t_commit = Time.instant
    current_cmd.commit
    draft_commit_ms += (Time.instant - t_commit).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "commit_#{label}", t_commit, Time.instant))
    GpuDraftBlock.new(submissions, state, lr_bufs, use_updown)
  }

  submit_seed = ->(base_state : ML::GGUF::Qwen35CPU::State,
                   token_id : Int32,
                   pos_start : Int32,
                   label : String,
                   steps : Int32,
                   use_updown : Bool) {
    t_fork = Time.instant
    state = base_state.fork
    draft_fork_ms += (Time.instant - t_fork).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "fork_#{label}", t_fork, Time.instant))
    t_token = Time.instant
    token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    token_buf.contents.as(Pointer(UInt32)).value = token_id.to_u32
    draft_token_buf_ms += (Time.instant - t_token).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "token_buf_#{label}", t_token, Time.instant))
    block_cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
    t_lr = Time.instant
    lr_bufs = build_lr_states.call(state, block_cmd)
    wba.try(&.mark("draft", "lr_states_#{label}", t_lr, Time.instant))
    submit_block.call(state, lr_bufs, token_buf, pos_start, label, block_cmd, steps, use_updown)
  }

  read_block = ->(block : GpuDraftBlock, limit : Int32, label : String) {
    active = block.submissions[0, limit]
    t_wait = Time.instant
    active.each do |sub|
      sub.pending_cmds.each(&.wait)
      sub.cmd.wait
    end
    draft_wait_block_ms += (Time.instant - t_wait).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "wait_block_#{label}", t_wait, Time.instant))

    ids = Array(Int32).new(active.size)
    t_read = Time.instant
    active.each do |sub|
      ids << sub.top1_id_buf.not_nil!.contents.as(Pointer(UInt32)).value.to_i32
    end
    draft_read_ids_ms += (Time.instant - t_read).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "read_ids_#{label}", t_read, Time.instant))
    ids
  }

  drain_block = ->(block : GpuDraftBlock?) {
    if b = block
      b.submissions.each do |sub|
        sub.pending_cmds.each(&.wait)
        sub.cmd.wait
      end
    end
  }

  state_before_last = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  verifier_state = state_before_last.fork
  verifier_backup = if use_verifier_backup
                      backup = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
                      ML::GGUF::Qwen35CPU.prepare_state_metal!(backup, hp)
                      backup
                    else
                      nil
                    end
  t_seed = Time.instant
  current_schedule_index = 0
  draft_updown_available = updown_actual_rank > 0
  draft_updown_enabled = draft_updown_available && draft_updown_after_full_accepts <= 0
  draft_updown_full_accept_streak = 0
  current_block = submit_seed.call(state_before_last, last_token, pos_last, "self_spec_seed", Math.min(schedule[current_schedule_index], gen_tokens), draft_updown_enabled)
  t_initial_target = Time.instant
  target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, last_token, pos_last, verifier_state)[0]
  verifier_initial_ms += (Time.instant - t_initial_target).total_milliseconds if attr_collect
  wba.try(&.mark("verifier", "initial_target", t_initial_target, Time.instant))
  current_proposal = read_block.call(current_block, Math.min(gamma, gen_tokens), "seed")
  draft_seed_ms = (Time.instant - t_seed).total_milliseconds
  wba.try(&.mark("pipeline", "seed_block", t_seed, Time.instant))

  emitted_tokens = 0
  chunks = 0
  rejections = 0
  accepted_draft_tokens = 0
  proposed_tokens = 0
  draft_updown_chunks = 0
  draft_next_ms = 0.0
  verifier_ms = 0.0
  draft_wait_ms = 0.0
  backup_ms = 0.0
  rebuild_ms = 0.0
  controller_ms = 0.0
  replay_ms = 0.0
  overlap_ms = draft_seed_ms
  exact_ids = [] of Int32
  emitted_ids = [] of Int32
  gamma_history = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    chunk_size = Math.min(current_proposal.size, gen_tokens - emitted_tokens)
    proposal = current_proposal[0, chunk_size]
    gamma_history << proposal.size
    proposed_tokens += proposal.size
    draft_updown_chunks += 1 if current_block.use_updown
    cycle_start_pos = prompt_ids.size + emitted_tokens
    final_chunk = emitted_tokens + proposal.size >= gen_tokens
    verifier_tokens = final_chunk && proposal.size > 1 ? proposal[0, proposal.size - 1] : proposal
    if attr_collect
      verifier_tail_skip_tokens += proposal.size - verifier_tokens.size
    end

    next_block = nil.as(GpuDraftBlock?)
    chunk_draft_next_ms = 0.0
    t_overlap = Time.instant
    if emitted_tokens + proposal.size < gen_tokens
      t_next = Time.instant
      last_proposed_buf = current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
      next_schedule_index = (current_schedule_index + 1) % schedule.size
      next_steps = Math.min(schedule[next_schedule_index], gen_tokens - emitted_tokens - proposal.size)
      next_block = submit_block.call(current_block.state, current_block.lr_bufs, last_proposed_buf, pos_last + proposal.size, "self_spec_next_#{chunks}", nil, next_steps, draft_updown_enabled)
      chunk_draft_next_ms += (Time.instant - t_next).total_milliseconds
    else
      next_schedule_index = current_schedule_index
    end

    if use_verifier_backup
      t_backup = Time.instant
      verifier_backup.not_nil!.copy_from!(verifier_state)
      backup_ms += (Time.instant - t_backup).total_milliseconds
      wba.try(&.mark("controller", "backup_#{chunks}", t_backup, Time.instant))
    end
    t_verify = Time.instant
    target_nexts = verifier_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verifier_tokens, cycle_start_pos, verifier_state)
    dt_verify = (Time.instant - t_verify).total_milliseconds
    verifier_ms += dt_verify
    if attr_collect
      verifier_prefill_ms += dt_verify
      verifier_chunks += 1
      verifier_tokens_count += verifier_tokens.size
    end
    wba.try(&.mark("verifier", "chunk_#{chunks}", t_verify, Time.instant))

    if next_block
      t_wait = Time.instant
      next_limit = Math.min(gamma, gen_tokens - emitted_tokens - proposal.size)
      next_proposal = read_block.call(next_block.not_nil!, next_limit, "next_#{chunks}")
      draft_wait_ms += (Time.instant - t_wait).total_milliseconds
      chunk_draft_next_ms += (Time.instant - t_wait).total_milliseconds
    else
      next_proposal = [] of Int32
    end
    overlap_ms += (Time.instant - t_overlap).total_milliseconds
    wba.try(&.mark("pipeline", "overlap_chunk_#{chunks}", t_overlap, Time.instant))

    t_controller = Time.instant
    correction_or_accepted = [] of Int32
    expected = target_next_id
    rejected = false
    proposal.each_with_index do |cand, i|
      exact_ids << expected
      emitted = if cand == expected
                  accepted_draft_tokens += 1
                  cand
                else
                  rejections += 1
                  rejected = true
                  expected
                end
      correction_or_accepted << emitted
      emitted_ids << emitted
      emitted_tokens += 1

      pos = cycle_start_pos + i
      last_token = emitted
      pos_last = pos
      expected = target_nexts[i][0] if cand == expected && i < target_nexts.size
      break if rejected || emitted_tokens >= gen_tokens
    end
    controller_ms += (Time.instant - t_controller).total_milliseconds
    wba.try(&.mark("controller", "accept_chunk_#{chunks}", t_controller, Time.instant))

    if rejected
      drain_block.call(next_block)
      draft_updown_full_accept_streak = 0
      if (draft_updown_fallback_on_reject || draft_updown_after_full_accepts > 0) && draft_updown_enabled
        draft_updown_enabled = false
      end
      resync_base = nil.as(ML::GGUF::Qwen35CPU::State?)
      if use_verifier_backup
        backup = verifier_backup.not_nil!
        verifier_state.copy_from!(backup)
        t_replay = Time.instant
        corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, cycle_start_pos, verifier_state)
        replay_ms += (Time.instant - t_replay).total_milliseconds
        target_next_id = corrected[-1][0]
        resync_base = backup.fork
        if correction_or_accepted.size > 1
          ML::GGUF::Qwen35CPU.prefill_tokens(weights, correction_or_accepted[0, correction_or_accepted.size - 1], cycle_start_pos, resync_base)
        end
      else
        t_rebuild = Time.instant
        consumed = prompt_ids + emitted_ids
        verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
        ML::GGUF::Qwen35CPU.prepare_state_metal!(verifier_state, hp)
        target_next_id = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, consumed, 0, verifier_state)[0]
        base_tokens = consumed[0, consumed.size - 1]
        resync_base = verifier_state_after_prefix(weights, base_tokens, max_seq)
        rebuild_ms += (Time.instant - t_rebuild).total_milliseconds
        wba.try(&.mark("controller", "rebuild_#{chunks}", t_rebuild, Time.instant))
      end
      if emitted_tokens < gen_tokens
        t_resync = Time.instant
        current_schedule_index = 0
        current_block = submit_seed.call(resync_base.not_nil!, last_token, pos_last, "self_spec_resync_#{chunks}", Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), draft_updown_enabled)
        current_proposal = read_block.call(current_block, Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), "resync_#{chunks}")
        dt_resync = (Time.instant - t_resync).total_milliseconds
        draft_seed_ms += dt_resync
        overlap_ms += dt_resync
        draft_resync_ms += dt_resync if attr_collect
        wba.try(&.mark("pipeline", "resync_#{chunks}", t_resync, Time.instant))
      end
    else
      if draft_updown_available && draft_updown_after_full_accepts > 0
        draft_updown_full_accept_streak += 1
        draft_updown_enabled = true if draft_updown_full_accept_streak >= draft_updown_after_full_accepts
      end
      if emitted_tokens < gen_tokens
        target_next_id = target_nexts[proposal.size - 1][0]
        draft_next_ms += chunk_draft_next_ms
        current_block = next_block.not_nil!
        current_proposal = next_proposal
        current_schedule_index = next_schedule_index
      end
    end
  end
  # Report real self-spec wall time. The phase counters above are diagnostic only:
  # they can overlap and previously missed reject replay/controller work.
  overlap_ms = (Time.instant - t_seed).total_milliseconds

  plain_state = state_before_last.fork
  t_plain = Time.instant
  plain_last_token = prompt_last_token
  plain_pos_last = prompt_pos_last
  plain_exact_ids = [] of Int32
  gen_tokens.times do
    id = ML::GGUF::Qwen35CPU.forward_top1(weights, plain_last_token, plain_pos_last, plain_state)[0]
    plain_exact_ids << id
    plain_last_token = id
    plain_pos_last += 1
  end
  plain_exact_ms = (Time.instant - t_plain).total_milliseconds
  raise "plain exact ids mismatch" unless plain_exact_ids == exact_ids
  wba.try(&.mark("pipeline", "plain_exact", t_plain, Time.instant))

  attr_collect = false
  serial_state_before_last = state_before_last
  serial_verifier_state = state_before_last.fork
  serial_backup = if use_verifier_backup
                    backup = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
                    ML::GGUF::Qwen35CPU.prepare_state_metal!(backup, hp)
                    backup
                  else
                    nil
                  end
  t_serial = Time.instant
  serial_target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, prompt_last_token, prompt_pos_last, serial_verifier_state)[0]
  serial_last_token = prompt_last_token
  serial_pos_last = prompt_pos_last
  serial_schedule_index = 0
  serial_draft_updown_available = updown_actual_rank > 0
  serial_draft_updown_enabled = serial_draft_updown_available && draft_updown_after_full_accepts <= 0
  serial_draft_updown_full_accept_streak = 0
  serial_current_block = submit_seed.call(serial_state_before_last, serial_last_token, serial_pos_last, "self_spec_serial_seed", Math.min(schedule[serial_schedule_index], gen_tokens), serial_draft_updown_enabled)
  serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens), "serial_seed")
  serial_emitted_tokens = 0
  serial_exact_ids = [] of Int32
  serial_emitted_ids = [] of Int32
  serial_chunks = 0

  while serial_emitted_tokens < gen_tokens
    serial_chunks += 1
    chunk_size = Math.min(serial_current_proposal.size, gen_tokens - serial_emitted_tokens)
    proposal = serial_current_proposal[0, chunk_size]
    cycle_start_pos = prompt_ids.size + serial_emitted_tokens
    final_chunk = serial_emitted_tokens + proposal.size >= gen_tokens
    verifier_tokens = final_chunk && proposal.size > 1 ? proposal[0, proposal.size - 1] : proposal

    serial_backup.try(&.copy_from!(serial_verifier_state))
    target_nexts = verifier_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verifier_tokens, cycle_start_pos, serial_verifier_state)

    correction_or_accepted = [] of Int32
    expected = serial_target_next_id
    rejected = false
    proposal.each_with_index do |cand, i|
      serial_exact_ids << expected
      emitted = if cand == expected
                  cand
                else
                  rejected = true
                  expected
                end
      correction_or_accepted << emitted
      serial_emitted_ids << emitted
      serial_emitted_tokens += 1
      serial_last_token = emitted
      serial_pos_last = cycle_start_pos + i
      expected = target_nexts[i][0] if cand == expected && i < target_nexts.size
      break if rejected || serial_emitted_tokens >= gen_tokens
    end

    if rejected
      serial_draft_updown_full_accept_streak = 0
      if (draft_updown_fallback_on_reject || draft_updown_after_full_accepts > 0) && serial_draft_updown_enabled
        serial_draft_updown_enabled = false
      end
      serial_resync_base = nil.as(ML::GGUF::Qwen35CPU::State?)
      if use_verifier_backup
        backup = serial_backup.not_nil!
        serial_verifier_state.copy_from!(backup)
        corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, cycle_start_pos, serial_verifier_state)
        serial_target_next_id = corrected[-1][0]
        serial_resync_base = backup.fork
        if correction_or_accepted.size > 1
          ML::GGUF::Qwen35CPU.prefill_tokens(weights, correction_or_accepted[0, correction_or_accepted.size - 1], cycle_start_pos, serial_resync_base)
        end
      else
        consumed = prompt_ids + serial_emitted_ids
        serial_verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
        ML::GGUF::Qwen35CPU.prepare_state_metal!(serial_verifier_state, hp)
        serial_target_next_id = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, consumed, 0, serial_verifier_state)[0]
        base_tokens = consumed[0, consumed.size - 1]
        serial_resync_base = verifier_state_after_prefix(weights, base_tokens, max_seq)
      end
      if serial_emitted_tokens < gen_tokens
        serial_schedule_index = 0
        serial_current_block = submit_seed.call(serial_resync_base.not_nil!, serial_last_token, serial_pos_last, "self_spec_serial_resync_#{serial_chunks}", Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
        serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_resync_#{serial_chunks}")
      end
    else
      if serial_draft_updown_available && draft_updown_after_full_accepts > 0
        serial_draft_updown_full_accept_streak += 1
        serial_draft_updown_enabled = true if serial_draft_updown_full_accept_streak >= draft_updown_after_full_accepts
      end
      if serial_emitted_tokens < gen_tokens
        serial_target_next_id = target_nexts[proposal.size - 1][0]
        last_proposed_buf = serial_current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
        serial_schedule_index = (serial_schedule_index + 1) % schedule.size
        serial_current_block = submit_block.call(serial_current_block.state, serial_current_block.lr_bufs, last_proposed_buf, serial_pos_last, "self_spec_serial_next_#{serial_chunks}", nil, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
        serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_next_#{serial_chunks}")
      end
    end
  end
  serial_ms = (Time.instant - t_serial).total_milliseconds
  raise "serial pipeline exact ids mismatch" unless serial_exact_ids == exact_ids
  raise "serial pipeline emitted ids mismatch" unless serial_emitted_ids == emitted_ids
  wba.try(&.mark("pipeline", "paired_serial", t_serial, Time.instant))
  wba.try(&.flush)
  hidden_ms = serial_ms - overlap_ms
  {
    chunks:                chunks,
    rejections:            rejections,
    accepted_draft_tokens: accepted_draft_tokens,
    proposed_tokens:       proposed_tokens,
    draft_updown_chunks:   draft_updown_chunks,
    draft_seed_ms:         draft_seed_ms,
    draft_next_ms:         draft_next_ms,
    verifier_ms:           verifier_ms,
    draft_wait_ms:         draft_wait_ms,
    backup_ms:             backup_ms,
    rebuild_ms:            rebuild_ms,
    controller_ms:         controller_ms,
    plain_exact_ms:        plain_exact_ms,
    serial_ms:             serial_ms,
    overlap_ms:            overlap_ms,
    replay_ms:             replay_ms,
    hidden_ms:             hidden_ms,
    speedup:               overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    plain_speedup:         overlap_ms > 0.0 ? plain_exact_ms / overlap_ms : 0.0,
    parity:                exact_ids == emitted_ids,
    gamma_history:         gamma_history,
    exact_ids:             exact_ids,
    emitted_ids:           emitted_ids,
    draft_steps:           draft_steps,
    draft_blocks:          draft_blocks,
    draft_fork_ms:         draft_fork_ms,
    draft_token_buf_ms:    draft_token_buf_ms,
    draft_lr_project_ms:   draft_lr_project_ms,
    draft_submit_ms:       draft_submit_ms,
    draft_commit_ms:       draft_commit_ms,
    draft_wait_block_ms:   draft_wait_block_ms,
    draft_read_ids_ms:     draft_read_ids_ms,
    draft_resync_ms:       draft_resync_ms,
    verifier_initial_ms:   verifier_initial_ms,
    verifier_prefill_ms:   verifier_prefill_ms,
    verifier_chunks:       verifier_chunks,
    verifier_tokens:       verifier_tokens_count,
    verifier_tail_skip_tokens: verifier_tail_skip_tokens,
  }
end

private def self_spec_pipeline_attr_note(pipe) : String
  draft_profiled_ms = pipe[:draft_fork_ms] + pipe[:draft_token_buf_ms] + pipe[:draft_lr_project_ms] +
                      pipe[:draft_submit_ms] + pipe[:draft_commit_ms] + pipe[:draft_wait_block_ms] +
                      pipe[:draft_read_ids_ms]
  verifier_total_ms = pipe[:verifier_initial_ms] + pipe[:verifier_prefill_ms]
  sprintf(" attr_draft_steps=%d attr_draft_blocks=%d attr_draft_profiled_ms=%.3f attr_draft_fork_ms=%.3f attr_draft_token_buf_ms=%.3f attr_draft_lr_project_ms=%.3f attr_draft_submit_ms=%.3f attr_draft_commit_ms=%.3f attr_draft_wait_block_ms=%.3f attr_draft_read_ids_ms=%.3f attr_draft_resync_ms=%.3f attr_verifier_total_ms=%.3f attr_verifier_initial_ms=%.3f attr_verifier_prefill_ms=%.3f attr_verifier_chunks=%d attr_verifier_tokens=%d attr_verifier_tail_skip_tokens=%d",
    pipe[:draft_steps], pipe[:draft_blocks],
    draft_profiled_ms,
    pipe[:draft_fork_ms], pipe[:draft_token_buf_ms], pipe[:draft_lr_project_ms],
    pipe[:draft_submit_ms], pipe[:draft_commit_ms], pipe[:draft_wait_block_ms],
    pipe[:draft_read_ids_ms], pipe[:draft_resync_ms],
    verifier_total_ms, pipe[:verifier_initial_ms], pipe[:verifier_prefill_ms],
    pipe[:verifier_chunks], pipe[:verifier_tokens], pipe[:verifier_tail_skip_tokens])
end

private def build_self_spec_hybrid_routes(layer_ids : Array(Int32),
                                          manual_noffn : Set(Int32)?,
                                          manual_updown : Set(Int32)?) : Array(HybridRoute)
  layers = layer_ids.uniq.sort
  routes = [] of HybridRoute
  seen = Set(String).new

  add_route = ->(name : String, noffn_values : Array(Int32), updown_values : Array(Int32)) {
    noffn_sorted = noffn_values.uniq.sort
    updown_sorted = updown_values.uniq.sort
    overlap = noffn_sorted & updown_sorted
    if overlap.empty?
      key = "#{noffn_sorted.join(',')}|#{updown_sorted.join(',')}"
      unless seen.includes?(key)
        seen << key
        noffn = noffn_sorted.empty? ? nil.as(Set(Int32)?) : Set(Int32).new(noffn_sorted).as(Set(Int32)?)
        updown = updown_sorted.empty? ? nil.as(Set(Int32)?) : Set(Int32).new(updown_sorted).as(Set(Int32)?)
        route_name = name.gsub(/[^A-Za-z0-9_.-]/, "_")
        routes << {name: route_name, noffn: noffn, updown: updown}
      end
    end
  }

  manual_noffn_values = manual_noffn ? manual_noffn.not_nil!.to_a : [] of Int32
  manual_updown_values = manual_updown ? manual_updown.not_nil!.to_a : [] of Int32
  add_route.call("manual", manual_noffn_values, manual_updown_values) unless manual_noffn_values.empty? && manual_updown_values.empty?

  add_route.call("pure", [] of Int32, [] of Int32)
  return routes if layers.empty?

  head1 = layers[0, 1]
  head2 = layers[0, Math.min(2, layers.size)]
  tail1 = layers[layers.size - 1, 1]
  tail2_start = Math.max(layers.size - 2, 0)
  tail2 = layers[tail2_start, layers.size - tail2_start]

  add_route.call("noffn_#{head1.join('_')}", head1, [] of Int32)
  add_route.call("noffn_#{head2.join('_')}", head2, [] of Int32)
  add_route.call("updown_#{tail1.join('_')}", [] of Int32, tail1)
  add_route.call("updown_#{tail2.join('_')}", [] of Int32, tail2)
  add_route.call("hybrid_n#{head1.join('_')}_u#{tail1.join('_')}", head1, tail1)
  add_route.call("hybrid_n#{head1.join('_')}_u#{tail2.join('_')}", head1, tail2)
  add_route.call("hybrid_n#{head2.join('_')}_u#{tail2.join('_')}", head2, tail2)
  routes
end

private def hybrid_route_note(route : HybridRoute, updown_rank : Int32?) : String
  noffn_note = route[:noffn] ? " draft_no_ffn_layers=#{route[:noffn].not_nil!.to_a.sort.join(',')}" : ""
  updown_note = (updown_rank && route[:updown]) ? " draft_pca_updown_layers=#{route[:updown].not_nil!.to_a.sort.join(',')}" : ""
  " hybrid_route=#{route[:name]}#{noffn_note}#{updown_note}"
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
simulate_lowrank_metal = false
simulate_lowrank_metal_project = false
simulate_lowrank_metal_chunk = false
simulate_lowrank_metal_chunk_out = false
simulate_lowrank_metal_layer_chunk = false
simulate_lowrank_metal_layer_full = false
simulate_lowrank_metal_layer_updown_rank : Int32? = nil
simulate_lowrank_metal_layer_overlap = false
simulate_lowrank_metal_verifier_overlap = false
simulate_lowrank_metal_decode_verifier_overlap = false
simulate_exact_verifier_ltp = false
simulate_lowrank_metal_chunk_thread_overlap = false
simulate_multilayer_overlap_n = 0
simulate_logit_rank : Int32? = nil
simulate_logit_layers = [] of Int32
simulate_fallback_threshold : Float64? = nil
simulate_fallback_thresholds = [] of Float64
simulate_generate_tokens = 0
simulate_output_margin_threshold : Float64? = nil
simulate_refresh_interval : Int32? = nil
simulate_oracle_refresh_interval : Int32? = nil
simulate_self_spec_gammas = [] of Int32
simulate_self_spec_adaptive = false
simulate_self_spec_adaptive_min = 4
simulate_self_spec_adaptive_start = 4
simulate_self_spec_adaptive_max = 16
simulate_self_spec_adaptive_grow_margin : Float64? = nil
simulate_self_spec_draft_margin : Float64? = nil
simulate_self_spec_draft_stop_margin : Float64? = nil
simulate_self_spec_topk_rescue : Int32? = nil
simulate_self_spec_tree_k : Int32? = nil
simulate_topk_oracle_k : Int32? = nil
simulate_topk_oracle_train_tokens : Int32? = nil
simulate_self_spec_progressive = [] of Int32
simulate_self_spec_wall_progressive = [] of Int32
simulate_cheap_self_draft_variants = [] of String
ffn_pca_calib_prompts = [] of String
simulate_self_spec_gpu_pipeline_suite_prompts = [] of NamedTuple(name: String, text: String)
simulate_ffn_updown_metal_rank : Int32? = nil
simulate_self_spec_wall_metal_lowrank = false
simulate_self_spec_wall_metal_project = false
simulate_self_spec_wall_metal_layer_updown = false
simulate_self_draft_metal_baseline = 0
simulate_self_draft_gpu_chain = 0
simulate_self_draft_gpu_state_only = 0
simulate_self_draft_gpu_chain_overlap = 0
simulate_self_spec_gpu_pipeline = 0
simulate_self_spec_gpu_pipeline_gammas = [] of Int32
simulate_self_spec_gpu_pipeline_schedules = [] of Array(Int32)
simulate_self_spec_gpu_pipeline_draft_splits = [] of Int32
simulate_self_spec_gpu_pipeline_no_backup = false
simulate_self_spec_gpu_pipeline_draft_no_ffn = false
simulate_self_spec_gpu_pipeline_draft_no_ffn_layers = [] of Int32
simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn = false
simulate_self_spec_gpu_pipeline_draft_updown_rank : Int32? = nil
simulate_self_spec_gpu_pipeline_draft_updown_ranks = [] of Int32
simulate_self_spec_gpu_pipeline_draft_updown_layers = [] of Int32
simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject = false
simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts = 0
simulate_self_spec_gpu_pipeline_attribution = ENV["QWEN35_SELF_SPEC_ATTR"]? == "1"
simulate_self_spec_gpu_pipeline_hybrid_sweep = false
self_spec_cost_model = false
self_spec_draft_cost = 0.0
self_spec_verifier_cost = 0.0
self_spec_chunk_overhead = 0.0
self_spec_correction_cost = 1.0
self_spec_overlap_cost = false
self_spec_overlap_efficiency = 1.0

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
  p.on("--simulate-lowrank-metal", "Compare Metal low-rank DeltaNet step against the CPU low-rank proof kernel") { simulate_lowrank_metal = true }
  p.on("--simulate-lowrank-metal-project", "Compare Metal Q/K projection plus low-rank DeltaNet step against the CPU proof kernel") { simulate_lowrank_metal_project = true }
  p.on("--simulate-lowrank-metal-chunk", "Compare one-command-buffer Metal low-rank chunk scan against CPU low-rank steps") { simulate_lowrank_metal_chunk = true }
  p.on("--simulate-lowrank-metal-chunk-out", "Compare fused Metal low-rank chunk scan+postnorm+ssm_out against CPU steps") { simulate_lowrank_metal_chunk_out = true }
  p.on("--simulate-lowrank-metal-layer-chunk", "Compare fused Metal low-rank recurrent attention chunk plus CPU FFN against CPU low-rank layer steps") { simulate_lowrank_metal_layer_chunk = true }
  p.on("--simulate-lowrank-metal-layer-full", "Compare one-command-buffer Metal low-rank recurrent layer chunk against CPU low-rank layer steps") { simulate_lowrank_metal_layer_full = true }
  p.on("--simulate-lowrank-metal-layer-updown=R", "Compare integrated Metal low-rank recurrent layer chunk with FFN pca-updown rank R against CPU pca-updown") { |v| simulate_lowrank_metal_layer_updown_rank = v.to_i }
  p.on("--simulate-lowrank-metal-layer-overlap", "Compare serial vs queued async full low-rank layer chunk submissions") { simulate_lowrank_metal_layer_overlap = true }
  p.on("--simulate-lowrank-metal-verifier-overlap", "Overlap one async low-rank layer chunk with exact prefill verifier on the held-out span") { simulate_lowrank_metal_verifier_overlap = true }
  p.on("--simulate-lowrank-metal-decode-verifier-overlap", "Overlap one async low-rank layer chunk with queued exact decode-wave verifier on the held-out span") { simulate_lowrank_metal_decode_verifier_overlap = true }
  p.on("--simulate-exact-verifier-ltp", "Compare exact verifier routes: serial decode, queued decode, and chunk-major prefill") { simulate_exact_verifier_ltp = true }
  p.on("--simulate-lowrank-metal-chunk-thread-overlap", "Overlap one async low-rank layer chunk with chunk-major verifier in a worker thread") { simulate_lowrank_metal_chunk_thread_overlap = true }
  p.on("--simulate-lowrank-multilayer-chunk-thread-overlap=N", "Overlap N chained async low-rank layer chunks on one lane queue with chunk-major verifier in a worker thread") { |v| simulate_multilayer_overlap_n = v.to_i }
  p.on("--simulate-logits-rank=N", "Run full-model logit drift gate for one rank") { |v| simulate_logit_rank = v.to_i }
  p.on("--simulate-logits-layers=LIST", "Comma-separated recurrent layers to approximate together during the logit drift gate") { |v| simulate_logit_layers = parse_int_list(v) }
  p.on("--simulate-fallback-threshold=F", "Fallback to exact DeltaNet step when max per-head K residual exceeds F") { |v| simulate_fallback_threshold = v.to_f64 }
  p.on("--simulate-fallback-thresholds=LIST", "Run multiple fallback thresholds in one process") { |v| simulate_fallback_thresholds = v.split(',').map(&.strip).reject(&.empty?).map(&.to_f64) }
  p.on("--simulate-output-margin-threshold=F", "Fallback to exact output when approximate top1/top2 margin is below F") { |v| simulate_output_margin_threshold = v.to_f64 }
  p.on("--simulate-refresh-interval=N", "Force an exact low-rank-state refresh every N approximate-eligible positions") { |v| simulate_refresh_interval = v.to_i }
  p.on("--simulate-oracle-refresh-interval=N", "Copy the paired exact shadow state into the approximate state every N positions") { |v| simulate_oracle_refresh_interval = v.to_i }
  p.on("--simulate-generate=N", "Run teacher-forced exact-greedy generation drift gate for N decode tokens") { |v| simulate_generate_tokens = v.to_i }
  p.on("--simulate-self-spec-gammas=LIST", "Run self-spec low-rank draft simulation for comma-separated gammas") { |v| simulate_self_spec_gammas = parse_int_list(v) }
  p.on("--simulate-self-spec-adaptive=MIN,START,MAX", "Run adaptive self-spec gamma: grow on full accept, shrink on reject") do |v|
    values = parse_int_list(v)
    raise "adaptive self-spec expects MIN,START,MAX" unless values.size == 3
    simulate_self_spec_adaptive = true
    simulate_self_spec_adaptive_min = values[0]
    simulate_self_spec_adaptive_start = values[1]
    simulate_self_spec_adaptive_max = values[2]
  end
  p.on("--simulate-self-spec-adaptive-grow-margin=F", "Only grow adaptive self-spec gamma when exact verifier min margin in the chunk is at least F") { |v| simulate_self_spec_adaptive_grow_margin = v.to_f64 }
  p.on("--simulate-self-spec-draft-margin=F", "Count low-margin draft proposal steps below F inside each self-spec chunk") { |v| simulate_self_spec_draft_margin = v.to_f64 }
  p.on("--simulate-self-spec-draft-stop-margin=F", "Stop a self-spec proposal chunk once draft margin falls below F after the min gamma") { |v| simulate_self_spec_draft_stop_margin = v.to_f64 }
  p.on("--simulate-self-spec-topk-rescue=K", "Treat a greedy reject as tree-rescued when exact token is in draft top-K") { |v| simulate_self_spec_topk_rescue = v.to_i }
  p.on("--simulate-self-spec-tree-k=K", "Run progressive top-K tree oracle using --simulate-self-spec-progressive as the schedule") { |v| simulate_self_spec_tree_k = v.to_i }
  p.on("--simulate-topk-oracle=K", "Train/test a lightweight token/rank-bias reranker inside low-rank draft top-K") { |v| simulate_topk_oracle_k = v.to_i }
  p.on("--simulate-topk-oracle-train-tokens=N", "Training samples from the start of --simulate-generate for --simulate-topk-oracle") { |v| simulate_topk_oracle_train_tokens = v.to_i }
  p.on("--simulate-self-spec-progressive=LIST", "Run progressive self-spec verifier chunks with a repeating comma-separated schedule, e.g. 4,4,8") { |v| simulate_self_spec_progressive = parse_int_list(v) }
  p.on("--simulate-self-spec-wall-progressive=LIST", "Measure wall-clock low-rank draft plus exact chunk verifier for a progressive schedule") { |v| simulate_self_spec_wall_progressive = parse_int_list(v) }
  p.on("--simulate-cheap-self-draft-variants=LIST", "Run wall self-spec with comma-separated draft variants: lowrank,lowrank-no-ffn,skip-layer,early-exit-N,lowrank-ffn-top-P,lowrank-ffn-pca-R,lowrank-ffn-pca-down-R,lowrank-ffn-pca-updown-R") { |v| simulate_cheap_self_draft_variants = v.split(',').map(&.strip).reject(&.empty?) }
  p.on("--ffn-pca-calib-prompt=TEXT", "Additional prompt used to build FFN PCA/PCA-down basis; may be repeated") { |v| ffn_pca_calib_prompts << v }
  p.on("--simulate-ffn-updown-metal=R", "Run Metal microkernel gate for FFN pca-updown rank R") { |v| simulate_ffn_updown_metal_rank = v.to_i }
  p.on("--simulate-self-spec-wall-metal-lowrank", "Use the Metal low-rank DeltaNet core inside wall-clock self-spec draft proposals") { simulate_self_spec_wall_metal_lowrank = true }
  p.on("--simulate-self-spec-wall-metal-project", "Also compute Q/K low-rank coefficients on Metal before the Metal low-rank step") { simulate_self_spec_wall_metal_lowrank = true; simulate_self_spec_wall_metal_project = true }
  p.on("--simulate-self-spec-wall-metal-layer-updown", "Route lowrank-ffn-pca-updown-R draft layers through the integrated Metal layer-updown path") { simulate_self_spec_wall_metal_lowrank = true; simulate_self_spec_wall_metal_project = true; simulate_self_spec_wall_metal_layer_updown = true }
  p.on("--simulate-self-draft-metal-baseline=N", "Wall-clock the Metal-only self-draft (low-rank on --simulate-logits-layers) vs exact greedy and chunk-major verifier on N held-out tokens") { |v| simulate_self_draft_metal_baseline = v.to_i }
  p.on("--simulate-self-draft-gpu-chain=N", "Queue N low-rank self-draft top1 steps with GPU top1_id -> next embedding and no intermediate CPU readback") { |v| simulate_self_draft_gpu_chain = v.to_i }
  p.on("--simulate-self-draft-gpu-state-only=N", "Queue N known-token low-rank draft state updates without lm-head/top1; lower-bound ablation for draft head/control cost") { |v| simulate_self_draft_gpu_state_only = v.to_i }
  p.on("--simulate-self-draft-gpu-chain-overlap=N", "Run GPU self-draft chain on a lane queue while chunk-major verifier runs on the default queue") { |v| simulate_self_draft_gpu_chain_overlap = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline=N", "Run real fixed-gamma self-spec block pipeline: draft[k+1] on lane queue while verifier validates draft[k]") { |v| simulate_self_spec_gpu_pipeline = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-gammas=LIST", "Run real GPU self-spec pipeline for comma-separated fixed gammas in one model load") { |v| simulate_self_spec_gpu_pipeline_gammas = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-schedule=LIST", "Run real GPU self-spec pipeline with a repeating gamma schedule that resets on reject, e.g. 4,4,8") { |v| simulate_self_spec_gpu_pipeline_schedules << parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-splits=LIST", "Run real GPU self-spec pipeline with comma-separated draft command-buffer split sizes; 0 keeps one command buffer per draft block") { |v| simulate_self_spec_gpu_pipeline_draft_splits = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-no-backup", "Skip verifier rollback backup on the hot full-accept path; rebuild exact state from emitted ids on reject") { simulate_self_spec_gpu_pipeline_no_backup = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn", "Use the research lowrank-no-ffn draft route for GPU self-spec proposals; exact verifier still enforces parity") { simulate_self_spec_gpu_pipeline_draft_no_ffn = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-layers=LIST", "Skip FFN only for the listed low-rank recurrent draft layers; enables hybrid draft bodies") { |v| simulate_self_spec_gpu_pipeline_draft_no_ffn_layers = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-skip-recurrent-ffn", "Research route: skip FFN on all recurrent draft layers; exact verifier still enforces parity") { simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown=R", "Use resident FFN pca-updown rank R on selected low-rank recurrent draft layers in the real GPU pipeline") { |v| simulate_self_spec_gpu_pipeline_draft_updown_rank = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updowns=LIST", "In-process A/B list for resident FFN pca-updown ranks in the real GPU pipeline; use 0 for lowrank baseline") { |v| simulate_self_spec_gpu_pipeline_draft_updown_ranks = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-layers=LIST", "Apply resident FFN pca-updown only to the listed low-rank recurrent draft layers; enables hybrid draft bodies") { |v| simulate_self_spec_gpu_pipeline_draft_updown_layers = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-fallback-on-reject", "After the first pca-updown draft rejection, resync future draft blocks with baseline lowrank") { simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-after-full-accepts=N", "Only enable pca-updown after N consecutive full-accept chunks; any reject disables and resets the streak") { |v| simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-attribution", "Append WBA attribution counters for the real GPU self-spec pipeline") { simulate_self_spec_gpu_pipeline_attribution = true }
  p.on("--simulate-self-spec-gpu-pipeline-hybrid-sweep", "Run an in-process route sweep over pure/no-FFN/pca-updown hybrid layer masks") { simulate_self_spec_gpu_pipeline_hybrid_sweep = true }
  p.on("--simulate-self-spec-gpu-pipeline-suite-prompt=NAME::TEXT", "Additional eval prompt for GPU self-spec pipeline suite; main --prompt still runs first") do |v|
    if sep = v.index("::")
      name = v[0, sep]
      text = v[(sep + 2)..]
    else
      name = "suite#{simulate_self_spec_gpu_pipeline_suite_prompts.size + 1}"
      text = v
    end
    safe_name = name.empty? ? "suite#{simulate_self_spec_gpu_pipeline_suite_prompts.size + 1}" : name.gsub(/[^A-Za-z0-9_.-]/, "_")
    simulate_self_spec_gpu_pipeline_suite_prompts << {name: safe_name, text: text}
  end
  p.on("--self-spec-draft-cost=F", "Relative cost per low-rank draft token (plain exact decode token = 1)") { |v| self_spec_cost_model = true; self_spec_draft_cost = v.to_f64 }
  p.on("--self-spec-verifier-cost=F", "Relative cost per exact verifier token in a chunk (plain exact decode token = 1)") { |v| self_spec_cost_model = true; self_spec_verifier_cost = v.to_f64 }
  p.on("--self-spec-chunk-overhead=F", "Relative fixed overhead per self-spec chunk") { |v| self_spec_cost_model = true; self_spec_chunk_overhead = v.to_f64 }
  p.on("--self-spec-correction-cost=F", "Relative cost per rejected-token correction step") { |v| self_spec_cost_model = true; self_spec_correction_cost = v.to_f64 }
  p.on("--self-spec-overlap-cost", "Estimate draft/verifier pipeline cost as max(draft, verifier) per chunk") { self_spec_cost_model = true; self_spec_overlap_cost = true }
  p.on("--self-spec-overlap-efficiency=F", "Fraction of min(draft, verifier) hidden by overlap, 0..1 (default: 1)") { |v| self_spec_cost_model = true; self_spec_overlap_cost = true; self_spec_overlap_efficiency = v.to_f64 }
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
token_ids = token_ids_for_prompt(tok, prompt, tokens_limit)
ffn_pca_calib_token_sets = ffn_pca_calib_prompts.map { |calib_prompt| token_ids_for_prompt(tok, calib_prompt, tokens_limit) }

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
per_head = recurrent_k_vectors_for_prompt(weights, token_ids, layer_index)
samples = (simulate_delta || simulate_lowrank || simulate_lowrank_metal || simulate_lowrank_metal_project || simulate_lowrank_metal_chunk || simulate_lowrank_metal_chunk_out || simulate_lowrank_metal_layer_chunk || simulate_lowrank_metal_layer_full || simulate_lowrank_metal_layer_updown_rank || simulate_lowrank_metal_layer_overlap || simulate_lowrank_metal_verifier_overlap || simulate_lowrank_metal_decode_verifier_overlap || simulate_lowrank_metal_chunk_thread_overlap || simulate_multilayer_overlap_n > 0) ? recurrent_samples_for_prompt(weights, token_ids, layer_index) : [] of RecurrentSample
max_rank = ranks.max
if rank = simulate_logit_rank
  max_rank = Math.max(max_rank, rank)
end
calib_count = Math.min(calib_tokens, token_ids.size - 1)
raise "need at least one held-out token" unless calib_count > 0 && calib_count < token_ids.size

bases = per_head.map { |vectors| build_basis(vectors[0, calib_count], max_rank, basis_mode, pca_iters) }

puts "Qwen35 DeltaNet fixed-basis K residual probe"
puts "model=#{File.basename(model)}"
puts "layer=#{layer_index} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_tokens=#{token_ids.size - calib_count}"
puts "heads=#{per_head.size} state_size=#{per_head[0][0].size} ranks=#{ranks.join(',')}"
puts "basis=#{basis_mode} pca_iters=#{pca_iters}; per-head basis over first calib_tokens; reports held-out L2 residual for normalized K vectors"
puts basis_rank_note(bases, max_rank)
puts "thresholds=#{thresholds.map { |t| t.round(4) }.join(',')}"

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
    rank_notes = simulate_logit_layers.uniq.sort.map do |il|
      "#{il}:#{basis_rank_note(layer_bases[il], rank)}"
    end
    puts "layer_basis_effective_ranks #{rank_notes.join(' ')}"
    ffn_pca_ranks = [] of Int32
    ffn_pca_down_ranks = [] of Int32
    ffn_pca_updown_ranks = [] of Int32
    if metal_updown_rank = simulate_ffn_updown_metal_rank
      ffn_pca_updown_ranks << metal_updown_rank
    end
    if layer_updown_rank = simulate_lowrank_metal_layer_updown_rank
      ffn_pca_updown_ranks << layer_updown_rank
    end
    if pipeline_updown_rank = simulate_self_spec_gpu_pipeline_draft_updown_rank
      ffn_pca_updown_ranks << pipeline_updown_rank
    end
    simulate_self_spec_gpu_pipeline_draft_updown_ranks.each do |pipeline_updown_rank|
      ffn_pca_updown_ranks << pipeline_updown_rank if pipeline_updown_rank > 0
    end
    simulate_cheap_self_draft_variants.each do |variant|
      if pca_rank = draft_variant_ffn_pca_rank(variant)
        ffn_pca_ranks << pca_rank
      end
      if pca_down_rank = draft_variant_ffn_pca_down_rank(variant)
        ffn_pca_down_ranks << pca_down_rank
      end
      if pca_updown_rank = draft_variant_ffn_pca_updown_rank(variant)
        ffn_pca_updown_ranks << pca_updown_rank
      end
    end
    ffn_activation_bases = nil.as(FFNBasisMap?)
    ffn_down_adapters = nil.as(FFNAdapterMap?)
    ffn_updown_adapters = nil.as(FFNUpDownAdapterMap?)
    all_ffn_pca_ranks = ffn_pca_ranks + ffn_pca_down_ranks + ffn_pca_updown_ranks
    unless all_ffn_pca_ranks.empty?
      max_ffn_pca_rank = all_ffn_pca_ranks.max
      ffn_vectors = if ffn_pca_calib_token_sets.empty?
                      ffn_activation_vectors_for_prompt(weights, token_ids, simulate_logit_layers.uniq, calib_count)
                    else
                      ffn_activation_vectors_for_token_sets(weights, ffn_pca_calib_token_sets, simulate_logit_layers.uniq, calib_tokens)
                    end
      built = {} of Int32 => Array(Array(Float64))
      ffn_vectors.each do |il, vectors|
        next if vectors.empty?
        built[il] = pca_basis(vectors, max_ffn_pca_rank, pca_iters)
      end
      ffn_activation_bases = built
      calib_source = ffn_pca_calib_token_sets.empty? ? "eval_prompt" : "external_prompts:#{ffn_pca_calib_token_sets.size}"
      puts "ffn_activation_pca_basis source=#{calib_source} layers=#{built.keys.sort.join(',')} max_rank=#{max_ffn_pca_rank} calib_vectors=#{built.map { |il, _| "#{il}:#{ffn_vectors[il].size}" }.join(',')} pca_iters=#{pca_iters}"
      unless (ffn_pca_down_ranks + ffn_pca_updown_ranks).empty?
        adapters = {} of Int32 => FFNAdapter
        built.each do |il, basis_set|
          layer = weights.layers[il].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "FFN PCA-down layer #{il} is not recurrent"
          down_basis = basis_set.map do |basis_vec|
            ML::GGUF::Qwen35CPU.qmatvec_nobias(layer.ffn_down_qw, basis_vec.map(&.to_f32))
          end
          adapters[il] = FFNAdapter.new(basis_set, down_basis)
        end
        ffn_down_adapters = adapters
        adapter_rank_note = (ffn_pca_down_ranks + ffn_pca_updown_ranks).max
        puts "ffn_down_pca_adapter layers=#{adapters.keys.sort.join(',')} max_rank=#{adapter_rank_note} precomputed_vectors=#{adapters.map { |il, adapter| "#{il}:#{adapter.down_basis.size}" }.join(',')}"
      end
      unless ffn_pca_updown_ranks.empty?
        updown_token_sets = ffn_pca_calib_token_sets.empty? ? [token_ids[0, calib_count]] : ffn_pca_calib_token_sets
        updown_samples = ffn_updown_samples_for_token_sets(weights, updown_token_sets, simulate_logit_layers.uniq, calib_tokens)
        updown = {} of Int32 => FFNUpDownAdapter
        max_updown_rank = ffn_pca_updown_ranks.max
        down_adapters = ffn_down_adapters || raise "FFN up/down adapter requires down adapters"
        built.each do |il, basis_set|
          samples_for_layer = updown_samples[il]? || [] of NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))
          updown[il] = train_ffn_updown_adapter(samples_for_layer, basis_set, down_adapters[il].down_basis, max_updown_rank)
        end
        ffn_updown_adapters = updown
        puts "ffn_updown_pca_adapter layers=#{updown.keys.sort.join(',')} max_rank=#{max_updown_rank} samples=#{updown_samples.map { |il, s| "#{il}:#{s.size}" }.join(',')}"
        if metal_rank = simulate_ffn_updown_metal_rank
          raise "Metal FFN up/down unavailable" unless ML::GGUF::Qwen35Metal.available?
          layer_id = simulate_logit_layers.uniq.find { |il| updown[il]? && updown_samples[il]? && !updown_samples[il].empty? } || raise "no FFN up/down sample for Metal gate"
          adapter = updown[layer_id]
          sample = updown_samples[layer_id][0]
          ffn_in = sample[:ffn_in].map(&.to_f32)
          hidden_dim = ffn_in.size
          bench_rank = Math.min(metal_rank, adapter.coeff_weights.size)
          raise "FFN up/down Metal rank must be positive" unless bench_rank > 0
          raise "FFN up/down Metal output dim mismatch" unless adapter.down_basis[0].size == hidden_dim

          x_mean = adapter.x_mean.map(&.to_f32)
          c_mean = adapter.c_mean.map(&.to_f32)
          coeff_weights = Array(Float32).new(bench_rank * hidden_dim)
          down_basis = Array(Float32).new(bench_rank * hidden_dim)
          bench_rank.times do |j|
            hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
            hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
          end

          cpu_out = [] of Float32
          cpu_reps = 3
          t_cpu = Time.instant
          cpu_reps.times { cpu_out = ffn_out_from_updown_adapter(ffn_in, adapter, bench_rank) }
          cpu_ms = (Time.instant - t_cpu).total_milliseconds / cpu_reps

          x_mean_buf = ML::MetalBuffer.from_array(x_mean[0, hidden_dim])
          c_mean_buf = ML::MetalBuffer.from_array(c_mean[0, bench_rank])
          coeff_w_buf = ML::MetalBuffer.from_array(coeff_weights)
          down_buf = ML::MetalBuffer.from_array(down_basis)

          metal_out = ML::GGUF::Qwen35Metal.ffn_pca_updown_out_resident(ffn_in, x_mean_buf, c_mean_buf, coeff_w_buf, down_buf, hidden_dim, bench_rank)
          metal_reps = 5
          t_metal = Time.instant
          metal_reps.times { metal_out = ML::GGUF::Qwen35Metal.ffn_pca_updown_out_resident(ffn_in, x_mean_buf, c_mean_buf, coeff_w_buf, down_buf, hidden_dim, bench_rank) }
          metal_ms = (Time.instant - t_metal).total_milliseconds / metal_reps

          sum_sq = 0.0
          max_delta = 0.0
          hidden_dim.times do |d|
            delta = (cpu_out[d] - metal_out[d]).abs.to_f64
            max_delta = delta if delta > max_delta
            sum_sq += delta * delta
          end
          rmse = Math.sqrt(sum_sq / hidden_dim)
          puts "ffn_updown_metal layer=#{layer_id} rank=#{bench_rank} hidden=#{hidden_dim} max_delta=#{max_delta.round(8)} rmse=#{rmse.round(8)} cpu_ms=#{cpu_ms.round(4)} metal_ms=#{metal_ms.round(4)} metal_note=resident_adapter_upload_x_readback"
        end
      end
    end
    thresholds_to_run = if simulate_fallback_thresholds.empty?
                          [simulate_fallback_threshold]
                        else
                          simulate_fallback_thresholds.map { |v| v.as(Float64?) }
                        end
    thresholds_to_run.each do |fallback_threshold|
      logit = simulate_logits_policy(weights, token_ids, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_oracle_refresh_interval, simulate_output_margin_threshold)
      total_steps = logit[:approx_steps] + logit[:fallback_steps]
      approx_rate = total_steps > 0 ? (100.0 * logit[:approx_steps] / total_steps) : 0.0
      fallback_note = fallback_threshold ? " fallback_threshold=#{fallback_threshold} approx_rate=#{approx_rate.round(2)}%" : ""
      output_note = simulate_output_margin_threshold ? " output_margin_threshold=#{simulate_output_margin_threshold} output_fallbacks=#{logit[:output_fallbacks]}" : ""
      refresh_note = simulate_refresh_interval ? " refresh_interval=#{simulate_refresh_interval}" : ""
      oracle_refresh_note = simulate_oracle_refresh_interval ? " oracle_refresh_interval=#{simulate_oracle_refresh_interval}" : ""
      puts "logit_drift_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} max_delta=#{logit[:max_delta].round(6)} top1_match=#{logit[:top1_match].round(2)}% top5_hit=#{logit[:top5_hit].round(2)}% mean_kl=#{logit[:mean_kl].round(8)} max_kl=#{logit[:max_kl].round(8)} min_margin=#{logit[:min_margin].round(6)} confident_mismatches=#{logit[:confident_mismatches]} approx_steps=#{logit[:approx_steps]} fallback_steps=#{logit[:fallback_steps]}#{fallback_note}#{refresh_note}#{oracle_refresh_note}#{output_note}"

      if simulate_generate_tokens > 0
        gen = simulate_greedy_policy(weights, token_ids, simulate_generate_tokens, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_oracle_refresh_interval, simulate_output_margin_threshold)
        gen_total_steps = gen[:approx_steps] + gen[:fallback_steps]
        gen_approx_rate = gen_total_steps > 0 ? (100.0 * gen[:approx_steps] / gen_total_steps) : 0.0
        gen_output_note = simulate_output_margin_threshold ? " output_margin_threshold=#{simulate_output_margin_threshold} output_fallbacks=#{gen[:output_fallbacks]}" : ""
        gen_refresh_note = simulate_refresh_interval ? " refresh_interval=#{simulate_refresh_interval}" : ""
        gen_oracle_refresh_note = simulate_oracle_refresh_interval ? " oracle_refresh_interval=#{simulate_oracle_refresh_interval}" : ""
        puts "greedy_drift_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} gen_tokens=#{simulate_generate_tokens} mean_cos=#{gen[:mean_cos].round(8)} min_cos=#{gen[:min_cos].round(8)} max_delta=#{gen[:max_delta].round(6)} top1_match=#{gen[:top1_match].round(2)}% top5_hit=#{gen[:top5_hit].round(2)}% mean_kl=#{gen[:mean_kl].round(8)} max_kl=#{gen[:max_kl].round(8)} min_margin=#{gen[:min_margin].round(6)} confident_mismatches=#{gen[:confident_mismatches]} approx_steps=#{gen[:approx_steps]} fallback_steps=#{gen[:fallback_steps]} approx_rate=#{gen_approx_rate.round(2)}%#{gen_refresh_note}#{gen_oracle_refresh_note}#{gen_output_note} exact_ids=#{gen[:exact_ids].join(',')} approx_ids=#{gen[:approx_ids].join(',')}"
      end

      if simulate_generate_tokens > 0 && !simulate_self_spec_gammas.empty?
        simulate_self_spec_gammas.each do |gamma|
          spec = simulate_self_spec_policy(weights, token_ids, simulate_generate_tokens, gamma, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, nil, nil, nil, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin, simulate_self_spec_topk_rescue)
          spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
          spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
          cost_note = ""
          if self_spec_cost_model
            estimated_cost = self_spec_estimated_cost(spec, self_spec_draft_cost, self_spec_verifier_cost,
              self_spec_chunk_overhead, self_spec_correction_cost, self_spec_overlap_cost, self_spec_overlap_efficiency)
            plain_cost = simulate_generate_tokens.to_f64
            estimated_speedup = estimated_cost > 0.0 ? plain_cost / estimated_cost : 0.0
            overlap_note = self_spec_overlap_cost ? ",overlap_eff:#{self_spec_overlap_efficiency.round(4)}" : ""
            cost_note = " cost_model=#{self_spec_overlap_cost ? "overlap" : "sum"}:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)}#{overlap_note} estimated_cost=#{estimated_cost.round(4)} estimated_speedup=#{estimated_speedup.round(4)}x"
          end
          rescue_note = simulate_self_spec_topk_rescue ? " topk_rescue=#{simulate_self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
          puts "self_spec_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{gamma} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}% draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')}#{cost_note} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
        end
      end
      if simulate_generate_tokens > 0 && simulate_self_spec_adaptive
        spec = simulate_self_spec_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_adaptive_start, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_self_spec_adaptive_min, simulate_self_spec_adaptive_max, simulate_self_spec_adaptive_grow_margin, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin, simulate_self_spec_topk_rescue)
        spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
        spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
        cost_note = ""
        if self_spec_cost_model
          estimated_cost = self_spec_estimated_cost(spec, self_spec_draft_cost, self_spec_verifier_cost,
            self_spec_chunk_overhead, self_spec_correction_cost, self_spec_overlap_cost, self_spec_overlap_efficiency)
          plain_cost = simulate_generate_tokens.to_f64
          estimated_speedup = estimated_cost > 0.0 ? plain_cost / estimated_cost : 0.0
          overlap_note = self_spec_overlap_cost ? ",overlap_eff:#{self_spec_overlap_efficiency.round(4)}" : ""
          cost_note = " cost_model=#{self_spec_overlap_cost ? "overlap" : "sum"}:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)}#{overlap_note} estimated_cost=#{estimated_cost.round(4)} estimated_speedup=#{estimated_speedup.round(4)}x"
        end
        grow_margin_note = simulate_self_spec_adaptive_grow_margin ? " grow_margin=#{simulate_self_spec_adaptive_grow_margin}" : ""
        rescue_note = simulate_self_spec_topk_rescue ? " topk_rescue=#{simulate_self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
        puts "self_spec_adaptive layers=#{simulate_logit_layers.join(',')} rank=#{rank} min_gamma=#{simulate_self_spec_adaptive_min} start_gamma=#{simulate_self_spec_adaptive_start} max_gamma=#{simulate_self_spec_adaptive_max}#{grow_margin_note} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}% draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')}#{cost_note} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && !simulate_self_spec_progressive.empty?
        spec = simulate_self_spec_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_progressive[0], layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, nil, nil, nil, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin, simulate_self_spec_topk_rescue, simulate_self_spec_progressive)
        spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
        spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
        cost_note = ""
        if self_spec_cost_model
          estimated_cost = self_spec_estimated_cost(spec, self_spec_draft_cost, self_spec_verifier_cost,
            self_spec_chunk_overhead, self_spec_correction_cost, self_spec_overlap_cost, self_spec_overlap_efficiency)
          plain_cost = simulate_generate_tokens.to_f64
          estimated_speedup = estimated_cost > 0.0 ? plain_cost / estimated_cost : 0.0
          overlap_note = self_spec_overlap_cost ? ",overlap_eff:#{self_spec_overlap_efficiency.round(4)}" : ""
          cost_note = " cost_model=#{self_spec_overlap_cost ? "overlap" : "sum"}:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)}#{overlap_note} estimated_cost=#{estimated_cost.round(4)} estimated_speedup=#{estimated_speedup.round(4)}x"
        end
        rescue_note = simulate_self_spec_topk_rescue ? " topk_rescue=#{simulate_self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
        puts "self_spec_progressive layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{simulate_self_spec_progressive.join(',')} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}% draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')}#{cost_note} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && (tree_k = simulate_self_spec_tree_k)
        tree_schedule = simulate_self_spec_progressive.empty? ? [2, 2, 4] : simulate_self_spec_progressive
        tree = simulate_self_spec_tree_oracle(weights, token_ids, simulate_generate_tokens, tree_k, tree_schedule, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval)
        tree_total_steps = tree[:approx_steps] + tree[:fallback_steps]
        tree_approx_rate = tree_total_steps > 0 ? (100.0 * tree[:approx_steps] / tree_total_steps) : 0.0
        parity = tree[:exact_ids] == tree[:emitted_ids]
        tree_cost_note = ""
        if self_spec_cost_model
          rank_cost = self_spec_tree_estimated_cost(tree, self_spec_draft_cost, self_spec_verifier_cost, self_spec_chunk_overhead, self_spec_correction_cost, tree[:branch_tokens_rank])
          full_cost = self_spec_tree_estimated_cost(tree, self_spec_draft_cost, self_spec_verifier_cost, self_spec_chunk_overhead, self_spec_correction_cost, tree[:branch_tokens_full])
          tree_cost_note = " cost_model=tree:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)} rank_cost=#{rank_cost.round(4)} rank_speedup=#{(simulate_generate_tokens / rank_cost).round(4)}x full_cost=#{full_cost.round(4)} full_speedup=#{(simulate_generate_tokens / full_cost).round(4)}x"
        end
        puts "self_spec_tree_oracle layers=#{simulate_logit_layers.join(',')} rank=#{rank} top_k=#{tree_k} schedule=#{tree_schedule.join(',')} gen_tokens=#{simulate_generate_tokens} chunks=#{tree[:chunks]} full_rescue_chunks=#{tree[:full_rescue_chunks]} misses=#{tree[:misses]} parity=#{parity} draft_steps=#{tree[:draft_steps]} top1_hits=#{tree[:top1_hits]} topk_hits=#{tree[:topk_hits]} top1_rate=#{tree[:top1_rate].round(2)}% topk_rate=#{tree[:topk_rate].round(2)}% branch_tokens_rank=#{tree[:branch_tokens_rank]} branch_tokens_full=#{tree[:branch_tokens_full]} avg_rank_branch_tokens=#{tree[:avg_rank_branch_tokens].round(3)} avg_full_branch_tokens=#{tree[:avg_full_branch_tokens].round(3)} correction_steps=#{tree[:correction_steps]} approx_steps=#{tree[:approx_steps]} fallback_steps=#{tree[:fallback_steps]} approx_rate=#{tree_approx_rate.round(2)}% schedule_history=#{tree[:schedule_history].join(',')}#{tree_cost_note} exact_ids=#{tree[:exact_ids].join(',')} emitted_ids=#{tree[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && (oracle_k = simulate_topk_oracle_k)
        oracle = simulate_topk_oracle_calibration(weights, token_ids, simulate_generate_tokens, oracle_k, simulate_topk_oracle_train_tokens, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval)
        delta_branch = oracle[:baseline_avg_branch_tokens] - oracle[:calibrated_avg_branch_tokens]
        puts "topk_oracle_calibration layers=#{simulate_logit_layers.join(',')} rank=#{rank} top_k=#{oracle_k} gen_tokens=#{simulate_generate_tokens} samples=#{oracle[:samples]} train=#{oracle[:train_samples]} test=#{oracle[:test_samples]} best_token_scale=#{oracle[:best_token_scale]} best_rank_scale=#{oracle[:best_rank_scale]} best_margin_threshold=#{oracle[:best_margin_threshold].round(4)} train_top1=#{oracle[:train_top1_rate].round(2)}% train_topk=#{oracle[:train_topk_rate].round(2)}% train_avg_branch=#{oracle[:train_avg_branch_tokens].round(3)} baseline_top1=#{oracle[:baseline_top1_rate].round(2)}% baseline_topk=#{oracle[:baseline_topk_rate].round(2)}% baseline_avg_branch=#{oracle[:baseline_avg_branch_tokens].round(3)} baseline_misses=#{oracle[:baseline_misses]} calibrated_top1=#{oracle[:calibrated_top1_rate].round(2)}% calibrated_topk=#{oracle[:calibrated_topk_rate].round(2)}% calibrated_avg_branch=#{oracle[:calibrated_avg_branch_tokens].round(3)} calibrated_misses=#{oracle[:calibrated_misses]} delta_avg_branch=#{delta_branch.round(3)} margin_gate_rate=#{oracle[:margin_gate_rate].round(2)}% margin_gate_topk=#{oracle[:margin_gate_topk_rate].round(2)}% margin_gate_avg_branch=#{oracle[:margin_gate_avg_branch_tokens].round(3)} margin_gate_misses=#{oracle[:margin_gate_misses]} margin_gate_cost=#{oracle[:margin_gate_cost].round(3)} exact_ids=#{oracle[:exact_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && !simulate_self_spec_wall_progressive.empty?
        wall = simulate_self_spec_wall_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_wall_progressive, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_self_spec_wall_metal_lowrank, simulate_self_spec_wall_metal_project, simulate_self_spec_wall_metal_layer_updown)
        metal_note = simulate_self_spec_wall_metal_layer_updown ? " metal_project=1 metal_layer_updown=1" : (simulate_self_spec_wall_metal_project ? " metal_project=1" : (simulate_self_spec_wall_metal_lowrank ? " metal_lowrank=1" : ""))
        puts "self_spec_wall_progressive layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{simulate_self_spec_wall_progressive.join(',')}#{metal_note} gen_tokens=#{simulate_generate_tokens} chunks=#{wall[:chunks]} rejections=#{wall[:rejections]} accepted_draft_tokens=#{wall[:accepted_draft_tokens]} proposed_tokens=#{wall[:proposed_tokens]} accept_rate=#{wall[:accept_rate].round(2)}% verifier_tokens=#{wall[:verifier_tokens]} correction_steps=#{wall[:correction_steps]} draft_ms=#{wall[:draft_ms].round(3)} verifier_ms=#{wall[:verifier_ms].round(3)} replay_ms=#{wall[:replay_ms].round(3)} serial_ms=#{wall[:serial_ms].round(3)} overlap_est_ms=#{wall[:overlap_est_ms].round(3)} speedup_est=#{wall[:speedup_est].round(4)}x exact_ids=#{wall[:exact_ids].join(',')} emitted_ids=#{wall[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && !simulate_self_spec_wall_progressive.empty? && !simulate_cheap_self_draft_variants.empty?
        simulate_cheap_self_draft_variants.each do |variant|
          unless cheap_draft_variant_valid?(variant)
            raise "unknown cheap self-draft variant #{variant.inspect}"
          end
          wall = simulate_self_spec_wall_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_wall_progressive, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_self_spec_wall_metal_lowrank, simulate_self_spec_wall_metal_project, simulate_self_spec_wall_metal_layer_updown, variant, ffn_activation_bases, ffn_down_adapters, ffn_updown_adapters)
          metal_note = simulate_self_spec_wall_metal_layer_updown ? " metal_project=1 metal_layer_updown=1" : (simulate_self_spec_wall_metal_project ? " metal_project=1" : (simulate_self_spec_wall_metal_lowrank ? " metal_lowrank=1" : ""))
          parity = wall[:exact_ids] == wall[:emitted_ids]
          puts "cheap_self_draft_variant=#{variant} layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{simulate_self_spec_wall_progressive.join(',')}#{metal_note} gen_tokens=#{simulate_generate_tokens} chunks=#{wall[:chunks]} rejections=#{wall[:rejections]} accepted_draft_tokens=#{wall[:accepted_draft_tokens]} proposed_tokens=#{wall[:proposed_tokens]} accept_rate=#{wall[:accept_rate].round(2)}% parity=#{parity} verifier_tokens=#{wall[:verifier_tokens]} correction_steps=#{wall[:correction_steps]} draft_ms=#{wall[:draft_ms].round(3)} verifier_ms=#{wall[:verifier_ms].round(3)} replay_ms=#{wall[:replay_ms].round(3)} serial_ms=#{wall[:serial_ms].round(3)} overlap_est_ms=#{wall[:overlap_est_ms].round(3)} speedup_est=#{wall[:speedup_est].round(4)}x exact_ids=#{wall[:exact_ids].join(',')} emitted_ids=#{wall[:emitted_ids].join(',')}"
        end
      end
    end
    if simulate_self_draft_metal_baseline > 0
      sd = simulate_self_draft_metal_baseline_run(weights, token_ids, calib_count, simulate_self_draft_metal_baseline, layer_bases, rank)
      puts "self_draft_metal_baseline layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{sd[:steps]} self_draft_ms=#{sd[:self_draft_ms].round(3)} exact_ms=#{sd[:exact_ms].round(3)} verifier_ms=#{sd[:verifier_ms].round(3)} self_draft_per_tok_ms=#{sd[:self_draft_per_token_ms].round(3)} exact_per_tok_ms=#{sd[:exact_per_token_ms].round(3)} verifier_per_tok_ms=#{sd[:verifier_per_token_ms].round(3)} self_spec_wall_ratio=#{sd[:self_spec_wall_ratio].round(4)} agreement=#{sd[:agreement]}/#{sd[:steps]} self_draft_ids=#{sd[:self_draft_ids].join(',')} exact_ids=#{sd[:exact_ids].join(',')} verifier_ids=#{sd[:verifier_ids].join(',')}"
    end
    if simulate_self_draft_gpu_chain > 0
      chain = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, simulate_self_draft_gpu_chain, layer_bases, rank)
      puts "self_draft_gpu_chain layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{chain[:steps]} submit_ms=#{chain[:submit_ms].round(3)} wait_ms=#{chain[:wait_ms].round(3)} chain_ms=#{chain[:chain_ms].round(3)} exact_ms=#{chain[:exact_ms].round(3)} agreement=#{chain[:agreement]}/#{chain[:steps]} chain_ids=#{chain[:chain_ids].join(',')} exact_ids=#{chain[:exact_ids].join(',')}"
    end
    if simulate_self_draft_gpu_state_only > 0
      state_only = simulate_self_draft_gpu_state_only_run(weights, token_ids, calib_count, simulate_self_draft_gpu_state_only, layer_bases, rank)
      puts "self_draft_gpu_state_only layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{state_only[:steps]} project_ms=#{state_only[:project_ms].round(3)} submit_ms=#{state_only[:submit_ms].round(3)} wait_ms=#{state_only[:wait_ms].round(3)} chain_ms=#{state_only[:chain_ms].round(3)} per_token_ms=#{state_only[:per_token_ms].round(3)}"
    end
    if simulate_self_draft_gpu_chain_overlap > 0
      ov = simulate_self_draft_gpu_chain_overlap_run(weights, token_ids, calib_count, simulate_self_draft_gpu_chain_overlap, layer_bases, rank)
      puts "self_draft_gpu_chain_overlap layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{ov[:steps]} draft_alone_ms=#{ov[:draft_alone_ms].round(3)} verifier_ms=#{ov[:verifier_ms].round(3)} overlap_ms=#{ov[:overlap_ms].round(3)} draft_submit_ms=#{ov[:draft_submit_ms].round(3)} draft_wait_ms=#{ov[:draft_wait_ms].round(3)} hidden_ms=#{ov[:hidden_ms].round(3)} speedup=#{ov[:speedup].round(4)} agreement=#{ov[:agreement]}/#{ov[:steps]} draft_ids=#{ov[:draft_ids].join(',')} exact_ids=#{ov[:exact_ids].join(',')} verifier_ids=#{ov[:verifier_ids].join(',')}"
    end
    draft_no_ffn_layer_set = simulate_self_spec_gpu_pipeline_draft_no_ffn_layers.empty? ? nil : Set(Int32).new(simulate_self_spec_gpu_pipeline_draft_no_ffn_layers)
    draft_updown_layer_set = simulate_self_spec_gpu_pipeline_draft_updown_layers.empty? ? nil : Set(Int32).new(simulate_self_spec_gpu_pipeline_draft_updown_layers)
    hybrid_routes = simulate_self_spec_gpu_pipeline_hybrid_sweep ? build_self_spec_hybrid_routes(simulate_logit_layers, draft_no_ffn_layer_set, draft_updown_layer_set) : [] of HybridRoute
    pipeline_gammas = simulate_self_spec_gpu_pipeline_gammas.dup
    if simulate_self_spec_gpu_pipeline > 0 && !pipeline_gammas.includes?(simulate_self_spec_gpu_pipeline)
      pipeline_gammas << simulate_self_spec_gpu_pipeline
    end
    if simulate_generate_tokens > 0 && (!pipeline_gammas.empty? || !simulate_self_spec_gpu_pipeline_schedules.empty?)
      default_draft_split = ENV["QWEN35_DRAFT_BLOCK_TOKENS"]?.try(&.to_i?) || DEFAULT_SELF_SPEC_GPU_PIPELINE_DRAFT_BLOCK_TOKENS
      pipeline_splits = simulate_self_spec_gpu_pipeline_draft_splits.empty? ? [default_draft_split.as(Int32?)] : simulate_self_spec_gpu_pipeline_draft_splits.map { |v| v.as(Int32?) }
      pipeline_updown_options = [] of Int32?
      if simulate_self_spec_gpu_pipeline_draft_updown_ranks.empty?
        pipeline_updown_options << simulate_self_spec_gpu_pipeline_draft_updown_rank
      else
        simulate_self_spec_gpu_pipeline_draft_updown_ranks.each do |v|
          pipeline_updown_options << (v > 0 ? v : nil)
        end
      end
      if !simulate_self_spec_gpu_pipeline_suite_prompts.empty? && pipeline_updown_options.any? { |option| !option.nil? }
        raise "GPU pipeline suite with pca-updown requires external --ffn-pca-calib-prompt so FFN adapters are not tied to the main prompt" if ffn_pca_calib_token_sets.empty?
      end
      if simulate_self_spec_gpu_pipeline_hybrid_sweep
        pipeline_gammas.each do |pipeline_gamma|
          pipeline_splits.each do |draft_split|
            pipeline_updown_options.each do |pipeline_updown_rank|
              hybrid_routes.each do |route|
                next if pipeline_updown_rank && route[:updown].nil?
                next if pipeline_updown_rank.nil? && route[:updown]
                route_updown_rank = route[:updown] ? pipeline_updown_rank : nil
                pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_gamma, layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, route_updown_rank, ffn_updown_adapters, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, route[:noffn], route[:updown])
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = route_updown_rank ? " draft_pca_updown=#{route_updown_rank}" : ""
                draft_updown_fallback_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                route_note = hybrid_route_note(route, route_updown_rank)
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                puts "self_spec_gpu_pipeline_hybrid layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{route_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
              end
            end
          end
        end
        simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
          next if pipeline_schedule.empty?
          pipeline_splits.each do |draft_split|
            pipeline_updown_options.each do |pipeline_updown_rank|
              hybrid_routes.each do |route|
                next if pipeline_updown_rank && route[:updown].nil?
                next if pipeline_updown_rank.nil? && route[:updown]
                route_updown_rank = route[:updown] ? pipeline_updown_rank : nil
                pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_schedule[0], layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, route_updown_rank, ffn_updown_adapters, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, route[:noffn], route[:updown])
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = route_updown_rank ? " draft_pca_updown=#{route_updown_rank}" : ""
                draft_updown_fallback_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                route_note = hybrid_route_note(route, route_updown_rank)
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                puts "self_spec_gpu_pipeline_hybrid layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{route_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
              end
            end
          end
        end
      else
      pipeline_gammas.each do |pipeline_gamma|
        pipeline_splits.each do |draft_split|
          pipeline_updown_options.each do |pipeline_updown_rank|
            pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_gamma, layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, pipeline_updown_rank, ffn_updown_adapters, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, draft_no_ffn_layer_set, draft_updown_layer_set)
            accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
            backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
            draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
            draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
            draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
            draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
            draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
            draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
            draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
            split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
            attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
            puts "self_spec_gpu_pipeline layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
          end
        end
      end
      simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
        next if pipeline_schedule.empty?
        pipeline_splits.each do |draft_split|
          pipeline_updown_options.each do |pipeline_updown_rank|
            pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_schedule[0], layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, pipeline_updown_rank, ffn_updown_adapters, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, draft_no_ffn_layer_set, draft_updown_layer_set)
            accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
            backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
            draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
            draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
            draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
            draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
            draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
            draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
            draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
            split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
            attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
            puts "self_spec_gpu_pipeline layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
          end
        end
      end
      end
      unless simulate_self_spec_gpu_pipeline_suite_prompts.empty?
        simulate_self_spec_gpu_pipeline_suite_prompts.each do |suite_prompt|
          suite_token_ids = token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit)
          suite_calib_count = Math.min(calib_tokens, suite_token_ids.size - 1)
          raise "suite prompt #{suite_prompt[:name]} needs at least one held-out token" unless suite_calib_count > 0 && suite_calib_count < suite_token_ids.size
          suite_layer_bases = {} of Int32 => BasisSet
          simulate_logit_layers.uniq.each do |il|
            suite_layer_bases[il] = recurrent_k_vectors_for_prompt(weights, suite_token_ids, il).map do |vectors|
              build_basis(vectors[0, suite_calib_count], max_rank, basis_mode, pca_iters)
            end
          end
          suite_rank_notes = simulate_logit_layers.uniq.sort.map do |il|
            "#{il}:#{basis_rank_note(suite_layer_bases[il], rank)}"
          end
          puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} token_vectors=#{suite_token_ids.size} calib_tokens=#{suite_calib_count} heldout_tokens=#{suite_token_ids.size - suite_calib_count} layer_basis_effective_ranks=#{suite_rank_notes.join(' ')}"
          pipeline_gammas.each do |pipeline_gamma|
            pipeline_splits.each do |draft_split|
              pipeline_updown_options.each do |pipeline_updown_rank|
                pipe = simulate_self_spec_gpu_pipeline_run(weights, suite_token_ids, simulate_generate_tokens, pipeline_gamma, suite_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, pipeline_updown_rank, ffn_updown_adapters, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, draft_no_ffn_layer_set, draft_updown_layer_set)
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
                draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
              end
            end
          end
          simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
            next if pipeline_schedule.empty?
            pipeline_splits.each do |draft_split|
              pipeline_updown_options.each do |pipeline_updown_rank|
                pipe = simulate_self_spec_gpu_pipeline_run(weights, suite_token_ids, simulate_generate_tokens, pipeline_schedule[0], suite_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, pipeline_updown_rank, ffn_updown_adapters, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, draft_no_ffn_layer_set, draft_updown_layer_set)
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
                draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
              end
            end
          end
        end
      end
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
  if simulate_lowrank_metal
    hp = weights.hparams
    lr_metal = simulate_lowrank_projected_delta_metal(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_metal_steps=#{lr_metal[:steps]} lr_cpu_ms=#{lr_metal[:cpu_ms].round(3)} lr_metal_ms=#{lr_metal[:metal_ms].round(3)} lr_metal_y_rmse=#{lr_metal[:y_rmse].round(8)} lr_metal_y_max=#{lr_metal[:y_max].round(8)} lr_metal_state_rmse=#{lr_metal[:state_rmse].round(8)} lr_metal_state_max=#{lr_metal[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_project
    hp = weights.hparams
    lr_project = simulate_lowrank_projected_delta_metal_project(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_project_steps=#{lr_project[:steps]} lr_project_cpu_ms=#{lr_project[:cpu_ms].round(3)} lr_project_metal_ms=#{lr_project[:metal_ms].round(3)} lr_project_y_rmse=#{lr_project[:y_rmse].round(8)} lr_project_y_max=#{lr_project[:y_max].round(8)} lr_project_state_rmse=#{lr_project[:state_rmse].round(8)} lr_project_state_max=#{lr_project[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_chunk
    hp = weights.hparams
    lr_chunk = simulate_lowrank_projected_delta_metal_chunk(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_chunk_steps=#{lr_chunk[:steps]} lr_chunk_cpu_ms=#{lr_chunk[:cpu_ms].round(3)} lr_chunk_metal_ms=#{lr_chunk[:metal_ms].round(3)} lr_chunk_y_rmse=#{lr_chunk[:y_rmse].round(8)} lr_chunk_y_max=#{lr_chunk[:y_max].round(8)} lr_chunk_state_rmse=#{lr_chunk[:state_rmse].round(8)} lr_chunk_state_max=#{lr_chunk[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_chunk_out
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_chunk_out = simulate_lowrank_projected_delta_metal_chunk_out(samples, bases, target_layer.ssm_out_qw, target_layer.ssm_norm, hp.rms_eps.to_f32, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_chunk_out_steps=#{lr_chunk_out[:steps]} lr_chunk_out_cpu_ms=#{lr_chunk_out[:cpu_ms].round(3)} lr_chunk_out_metal_ms=#{lr_chunk_out[:metal_ms].round(3)} lr_chunk_out_rmse=#{lr_chunk_out[:out_rmse].round(8)} lr_chunk_out_max=#{lr_chunk_out[:out_max].round(8)} lr_chunk_out_state_rmse=#{lr_chunk_out[:state_rmse].round(8)} lr_chunk_out_state_max=#{lr_chunk_out[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_layer_chunk
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_layer = simulate_lowrank_recurrent_layer_metal_chunk(samples, bases, target_layer, hp, rank, calib_count)
    line += " lr_layer_chunk_steps=#{lr_layer[:steps]} lr_layer_chunk_cpu_ms=#{lr_layer[:cpu_ms].round(3)} lr_layer_chunk_metal_ms=#{lr_layer[:metal_ms].round(3)} lr_layer_chunk_rmse=#{lr_layer[:layer_rmse].round(8)} lr_layer_chunk_max=#{lr_layer[:layer_max].round(8)} lr_layer_chunk_state_rmse=#{lr_layer[:state_rmse].round(8)} lr_layer_chunk_state_max=#{lr_layer[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_layer_full
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_full = simulate_lowrank_recurrent_layer_full_metal_chunk(samples, bases, target_layer, hp, rank, calib_count)
    line += " lr_layer_full_steps=#{lr_full[:steps]} lr_layer_full_cpu_ms=#{lr_full[:cpu_ms].round(3)} lr_layer_full_metal_ms=#{lr_full[:metal_ms].round(3)} lr_layer_full_rmse=#{lr_full[:layer_rmse].round(8)} lr_layer_full_max=#{lr_full[:layer_max].round(8)} lr_layer_full_state_rmse=#{lr_full[:state_rmse].round(8)} lr_layer_full_state_max=#{lr_full[:state_max].round(8)}"
  end
  if layer_updown_rank = simulate_lowrank_metal_layer_updown_rank
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    ffn_vectors = ffn_activation_vectors_for_prompt(weights, token_ids, [layer_index], calib_count)[layer_index]
    ffn_basis = pca_basis(ffn_vectors, layer_updown_rank, pca_iters)
    down_basis = ffn_basis.map do |basis_vec|
      ML::GGUF::Qwen35CPU.qmatvec_nobias(target_layer.ffn_down_qw, basis_vec.map(&.to_f32))
    end
    updown_samples = ffn_updown_samples_for_token_sets(weights, [token_ids[0, calib_count]], [layer_index], calib_count)[layer_index]
    updown_adapter = train_ffn_updown_adapter(updown_samples, ffn_basis, down_basis, layer_updown_rank)
    lr_updown = simulate_lowrank_recurrent_layer_updown_metal_chunk(samples, bases, target_layer, hp, rank, calib_count, updown_adapter, layer_updown_rank)
    line += " lr_layer_updown_steps=#{lr_updown[:steps]} lr_layer_updown_rank=#{lr_updown[:updown_rank]} lr_layer_updown_cpu_ms=#{lr_updown[:cpu_ms].round(3)} lr_layer_updown_metal_ms=#{lr_updown[:metal_ms].round(3)} lr_layer_updown_rmse=#{lr_updown[:layer_rmse].round(8)} lr_layer_updown_max=#{lr_updown[:layer_max].round(8)} lr_layer_updown_state_rmse=#{lr_updown[:state_rmse].round(8)} lr_layer_updown_state_max=#{lr_updown[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_layer_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_overlap = simulate_lowrank_recurrent_layer_full_async_overlap(samples, bases, target_layer, hp, rank, calib_count)
    line += " lr_layer_overlap_steps=#{lr_overlap[:steps]} lr_layer_overlap_serial_ms=#{lr_overlap[:serial_ms].round(3)} lr_layer_overlap_async_ms=#{lr_overlap[:async_ms].round(3)} lr_layer_overlap_speedup=#{lr_overlap[:speedup].round(4)} lr_layer_overlap_output_max=#{lr_overlap[:output_max].round(8)}"
  end
  if simulate_lowrank_metal_verifier_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_verify = simulate_lowrank_draft_exact_verifier_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count)
    line += " lr_verifier_overlap_steps=#{lr_verify[:steps]} lr_verifier_draft_ms=#{lr_verify[:draft_ms].round(3)} lr_verifier_verify_ms=#{lr_verify[:verifier_ms].round(3)} lr_verifier_serial_ms=#{lr_verify[:serial_ms].round(3)} lr_verifier_overlap_ms=#{lr_verify[:overlap_ms].round(3)} lr_verifier_speedup=#{lr_verify[:speedup].round(4)} lr_verifier_hidden_ms=#{lr_verify[:hidden_ms].round(3)} lr_verifier_draft_output_max=#{lr_verify[:draft_output_max].round(8)} lr_verifier_match=#{lr_verify[:verifier_match]}"
  end
  if simulate_lowrank_metal_decode_verifier_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_decode_verify = simulate_lowrank_draft_exact_decode_verifier_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count)
    line += " lr_decode_verify_steps=#{lr_decode_verify[:steps]} lr_decode_verify_draft_ms=#{lr_decode_verify[:draft_ms].round(3)} lr_decode_verify_serial_ms=#{lr_decode_verify[:verifier_serial_ms].round(3)} lr_decode_verify_async_ms=#{lr_decode_verify[:verifier_async_ms].round(3)} lr_decode_verify_overlap_ms=#{lr_decode_verify[:overlap_ms].round(3)} lr_decode_verify_async_speedup=#{lr_decode_verify[:async_speedup].round(4)} lr_decode_verify_overlap_speedup=#{lr_decode_verify[:overlap_speedup].round(4)} lr_decode_verify_hidden_ms=#{lr_decode_verify[:hidden_ms].round(3)} lr_decode_verify_draft_output_max=#{lr_decode_verify[:draft_output_max].round(8)} lr_decode_verify_match=#{lr_decode_verify[:verifier_match]}"
  end
  if simulate_exact_verifier_ltp
    ltp = simulate_exact_verifier_ltp_proxy(weights, token_ids, calib_count)
    line += " exact_ltp_steps=#{ltp[:steps]} exact_ltp_decode_serial_ms=#{ltp[:decode_serial_ms].round(3)} exact_ltp_decode_queued_ms=#{ltp[:decode_queued_ms].round(3)} exact_ltp_chunk_major_ms=#{ltp[:chunk_major_ms].round(3)} exact_ltp_queued_speedup=#{ltp[:queued_speedup].round(4)} exact_ltp_speedup=#{ltp[:ltp_speedup].round(4)} exact_ltp_queued_match=#{ltp[:queued_match]} exact_ltp_chunk_match=#{ltp[:chunk_match]}"
  end
  if simulate_lowrank_metal_chunk_thread_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    ltp_overlap = simulate_lowrank_draft_exact_chunk_verifier_thread_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count)
    line += " chunk_thread_steps=#{ltp_overlap[:steps]} chunk_thread_draft_ms=#{ltp_overlap[:draft_ms].round(3)} chunk_thread_verify_ms=#{ltp_overlap[:chunk_verifier_ms].round(3)} chunk_thread_serial_ms=#{ltp_overlap[:serial_ms].round(3)} chunk_thread_overlap_ms=#{ltp_overlap[:overlap_ms].round(3)} chunk_thread_speedup=#{ltp_overlap[:speedup].round(4)} chunk_thread_hidden_ms=#{ltp_overlap[:hidden_ms].round(3)} chunk_thread_draft_output_max=#{ltp_overlap[:draft_output_max].round(8)} chunk_thread_match=#{ltp_overlap[:verifier_match]}"
  end
  if simulate_multilayer_overlap_n > 0
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    multi = simulate_lowrank_multilayer_chunk_thread_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count, simulate_multilayer_overlap_n)
    line += " multi_thread_n_layers=#{multi[:n_layers]} multi_thread_steps=#{multi[:steps]} multi_thread_draft_ms=#{multi[:draft_ms].round(3)} multi_thread_draft_per_layer_ms=#{multi[:draft_per_layer_ms].round(3)} multi_thread_verify_ms=#{multi[:chunk_verifier_ms].round(3)} multi_thread_serial_ms=#{multi[:serial_ms].round(3)} multi_thread_overlap_ms=#{multi[:overlap_ms].round(3)} multi_thread_speedup=#{multi[:speedup].round(4)} multi_thread_hidden_ms=#{multi[:hidden_ms].round(3)} multi_thread_draft_output_max=#{multi[:draft_output_max].round(8)} multi_thread_match=#{multi[:verifier_match]}"
  end
  puts line
end
