require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/core/buffer"

# DeltaNet / GatedDeltaRule single-step kernel correctness.
# Compares the Metal `delta_net_step` against the CPU reference
# `Qwen35CPU.delta_net_step!` on the Qwen 3.5 9B shapes:
#   h_k = 16, h_v = 32, s = 128.
describe "Qwen35Metal.delta_net_step" do
  it "matches the CPU reference on Qwen 3.5 9B shapes" do
    next unless ML::GGUF::Qwen35Metal.available?

    h_k = 16
    h_v = 32
    s   = 128
    scale = (1.0 / Math.sqrt(s.to_f64)).to_f32

    # Seed so runs are reproducible (and so this test is deterministic).
    rng = Random.new(0xC0FFEE_u64)

    state_init = Array(Float32).new(h_v * s * s) { ((rng.next_float - 0.5) * 0.2_f64).to_f32 }
    q_conv     = Array(Float32).new(h_k * s)    { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    k_conv     = Array(Float32).new(h_k * s)    { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    v_conv     = Array(Float32).new(h_v * s)    { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    ghead      = Array(Float32).new(h_v)        { (0.9 + 0.1 * rng.next_float).to_f32 }
    beta       = Array(Float32).new(h_v)        { rng.next_float.to_f32 }

    # CPU reference
    cpu_state = state_init.dup
    cpu_y     = Array(Float32).new(h_v * s, 0.0_f32)
    ML::GGUF::Qwen35CPU.delta_net_step!(cpu_state, q_conv, k_conv, v_conv,
                                         ghead, beta, cpu_y,
                                         h_k, h_v, s, scale)

    # Metal path
    bytes     = state_init.size.to_i64 * sizeof(Float32)
    state_buf = ML::MetalBuffer.new(bytes)
    state_buf.write(state_init)

    gpu_y = ML::GGUF::Qwen35Metal.delta_net_step(
      state_buf, q_conv, k_conv, v_conv, ghead, beta,
      h_k, h_v, s, scale,
    )

    gpu_state = state_buf.read(state_init.size)

    # Output cosine must be ~1.0.
    y_dot   = 0.0_f64
    y_ncpu  = 0.0_f64
    y_ngpu  = 0.0_f64
    max_y_diff = 0.0_f32
    cpu_y.size.times do |i|
      a = cpu_y[i].to_f64
      b = gpu_y[i].to_f64
      y_dot  += a * b
      y_ncpu += a * a
      y_ngpu += b * b
      d = (cpu_y[i] - gpu_y[i]).abs
      max_y_diff = d if d > max_y_diff
    end
    y_cos = y_dot / (Math.sqrt(y_ncpu) * Math.sqrt(y_ngpu))

    # State cosine + max |Δ|
    s_dot = 0.0_f64
    s_ncpu = 0.0_f64
    s_ngpu = 0.0_f64
    max_s_diff = 0.0_f32
    cpu_state.size.times do |i|
      a = cpu_state[i].to_f64
      b = gpu_state[i].to_f64
      s_dot  += a * b
      s_ncpu += a * a
      s_ngpu += b * b
      d = (cpu_state[i] - gpu_state[i]).abs
      max_s_diff = d if d > max_s_diff
    end
    s_cos = s_dot / (Math.sqrt(s_ncpu) * Math.sqrt(s_ngpu))

    printf "  [delta_net_step] y: cos=%.12f max|Δ|=%g\n", y_cos, max_y_diff
    printf "                   state: cos=%.12f max|Δ|=%g\n", s_cos, max_s_diff

    y_cos.should be > 0.9999999
    s_cos.should be > 0.9999999
    max_y_diff.should be < 1.0e-4_f32
    max_s_diff.should be < 1.0e-5_f32

    state_buf.release
  end

  it "matches repeated CPU steps for an 8-token prefill chunk" do
    next unless ML::GGUF::Qwen35Metal.available?

    h_k = 16
    h_v = 32
    s = 128
    n_tokens = 8
    scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
    rng = Random.new(0xD371A_u64)

    state_init = Array(Float32).new(h_v * s * s) { ((rng.next_float - 0.5) * 0.2_f64).to_f32 }
    q_conv = Array(Float32).new(n_tokens * h_k * s) { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    k_conv = Array(Float32).new(n_tokens * h_k * s) { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    v_conv = Array(Float32).new(n_tokens * h_v * s) { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    ghead = Array(Float32).new(n_tokens * h_v) { (0.9 + 0.1 * rng.next_float).to_f32 }
    beta = Array(Float32).new(n_tokens * h_v) { rng.next_float.to_f32 }

    cpu_state = state_init.dup
    cpu_y = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
    q_stride = h_k * s
    v_stride = h_v * s
    h_stride = h_v
    n_tokens.times do |t|
      y_t = Array(Float32).new(h_v * s, 0.0_f32)
      ML::GGUF::Qwen35CPU.delta_net_step!(
        cpu_state,
        q_conv[t * q_stride, q_stride],
        k_conv[t * q_stride, q_stride],
        v_conv[t * v_stride, v_stride],
        ghead[t * h_stride, h_stride],
        beta[t * h_stride, h_stride],
        y_t,
        h_k, h_v, s, scale
      )
      y_t.each_with_index { |v, i| cpu_y[t * v_stride + i] = v }
    end

    state_buf = ML::MetalBuffer.new(state_init.size.to_i64 * sizeof(Float32))
    state_buf.write(state_init)

    gpu_y = ML::GGUF::Qwen35Metal.delta_net_chunk(
      state_buf, q_conv, k_conv, v_conv, ghead, beta,
      h_k, h_v, s, n_tokens, scale
    )
    gpu_state = state_buf.read(state_init.size)

    y_dot = 0.0_f64
    y_ncpu = 0.0_f64
    y_ngpu = 0.0_f64
    max_y_diff = 0.0_f32
    cpu_y.size.times do |i|
      a = cpu_y[i].to_f64
      b = gpu_y[i].to_f64
      y_dot += a * b
      y_ncpu += a * a
      y_ngpu += b * b
      d = (cpu_y[i] - gpu_y[i]).abs
      max_y_diff = d if d > max_y_diff
    end
    y_cos = y_dot / (Math.sqrt(y_ncpu) * Math.sqrt(y_ngpu))

    s_dot = 0.0_f64
    s_ncpu = 0.0_f64
    s_ngpu = 0.0_f64
    max_s_diff = 0.0_f32
    cpu_state.size.times do |i|
      a = cpu_state[i].to_f64
      b = gpu_state[i].to_f64
      s_dot += a * b
      s_ncpu += a * a
      s_ngpu += b * b
      d = (cpu_state[i] - gpu_state[i]).abs
      max_s_diff = d if d > max_s_diff
    end
    s_cos = s_dot / (Math.sqrt(s_ncpu) * Math.sqrt(s_ngpu))

    printf "  [delta_net_chunk] y: cos=%.12f max|Δ|=%g\n", y_cos, max_y_diff
    printf "                    state: cos=%.12f max|Δ|=%g\n", s_cos, max_s_diff

    y_cos.should be > 0.999999
    s_cos.should be > 0.999999
    max_y_diff.should be < 1.0e-3_f32
    max_s_diff.should be < 1.0e-4_f32

    state_buf.release
  end

  it "prepares recurrent Q/K/V and gates for an 8-token chunk" do
    next unless ML::GGUF::Qwen35Metal.available?

    h_k = 16
    h_v = 32
    s = 128
    conv_k = 4
    n_tokens = 8
    eps = 1.0e-6_f32
    qkv_dim = 2 * h_k * s + h_v * s
    q_dim = h_k * s
    v_dim = h_v * s
    rng = Random.new(0xC0DE_u64)

    conv_state_init = Array(Float32).new((conv_k - 1) * qkv_dim) { ((rng.next_float - 0.5) * 0.2_f64).to_f32 }
    qkv_mixed = Array(Float32).new(n_tokens * qkv_dim) { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    alpha = Array(Float32).new(n_tokens * h_v) { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    beta = Array(Float32).new(n_tokens * h_v) { ((rng.next_float - 0.5) * 0.5_f64).to_f32 }
    conv1d = Array(Float32).new(qkv_dim * conv_k) { ((rng.next_float - 0.5) * 0.2_f64).to_f32 }
    dt_bias = Array(Float32).new(h_v) { ((rng.next_float - 0.5) * 0.1_f64).to_f32 }
    ssm_a = Array(Float32).new(h_v) { (-(0.1 + rng.next_float)).to_f32 }

    cpu_state = conv_state_init.dup
    cpu_q = Array(Float32).new(n_tokens * q_dim, 0.0_f32)
    cpu_k = Array(Float32).new(n_tokens * q_dim, 0.0_f32)
    cpu_v = Array(Float32).new(n_tokens * v_dim, 0.0_f32)
    cpu_g = Array(Float32).new(n_tokens * h_v, 0.0_f32)
    cpu_b = Array(Float32).new(n_tokens * h_v, 0.0_f32)

    n_tokens.times do |tok|
      conv_out = Array(Float32).new(qkv_dim) do |ch|
        acc = 0.0_f32
        w_base = ch * conv_k
        (conv_k - 1).times do |t|
          acc += cpu_state[t * qkv_dim + ch] * conv1d[w_base + t]
        end
        acc += qkv_mixed[tok * qkv_dim + ch] * conv1d[w_base + conv_k - 1]
        sig = 1.0_f32 / (1.0_f32 + Math.exp(-acc).to_f32)
        acc * sig
      end

      (conv_k - 2).times do |t|
        src = (t + 1) * qkv_dim
        dst = t * qkv_dim
        qkv_dim.times { |ch| cpu_state[dst + ch] = cpu_state[src + ch] }
      end
      last = (conv_k - 2) * qkv_dim
      qkv_dim.times { |ch| cpu_state[last + ch] = qkv_mixed[tok * qkv_dim + ch] }

      q_dim.times do |i|
        cpu_q[tok * q_dim + i] = conv_out[i]
        cpu_k[tok * q_dim + i] = conv_out[q_dim + i]
      end
      v_dim.times { |i| cpu_v[tok * v_dim + i] = conv_out[2 * q_dim + i] }
      h_k.times do |h|
        ML::GGUF::Qwen35CPU.l2_norm_slice!(cpu_q, tok * q_dim + h * s, s, eps)
        ML::GGUF::Qwen35CPU.l2_norm_slice!(cpu_k, tok * q_dim + h * s, s, eps)
      end

      h_v.times do |h|
        idx = tok * h_v + h
        b = beta[idx]
        cpu_b[idx] = 1.0_f32 / (1.0_f32 + Math.exp(-b).to_f32)
        xi = alpha[idx] + dt_bias[h]
        sp = xi > 20.0_f32 ? xi : Math.log(1.0_f32 + Math.exp(xi).to_f32).to_f32
        cpu_g[idx] = Math.exp((sp * ssm_a[h]).to_f64).to_f32
      end
    end

    conv_buf = ML::MetalBuffer.new(conv_state_init.size.to_i64 * sizeof(Float32))
    conv_buf.write(conv_state_init)
    gpu_q, gpu_k, gpu_v, gpu_g, gpu_b = ML::GGUF::Qwen35Metal.recurrent_prep_chunk(
      conv_buf, qkv_mixed, alpha, beta, conv1d, dt_bias, ssm_a,
      h_k, h_v, s, conv_k, n_tokens, eps
    )
    gpu_state = conv_buf.read(conv_state_init.size)

    max_q = max_abs_diff(cpu_q, gpu_q)
    max_k = max_abs_diff(cpu_k, gpu_k)
    max_v = max_abs_diff(cpu_v, gpu_v)
    max_g = max_abs_diff(cpu_g, gpu_g)
    max_b = max_abs_diff(cpu_b, gpu_b)
    max_state = max_abs_diff(cpu_state, gpu_state)

    printf "  [recurrent_prep_chunk] max|Δ| q=%g k=%g v=%g g=%g b=%g state=%g\n",
      max_q, max_k, max_v, max_g, max_b, max_state

    max_q.should be < 1.0e-5_f32
    max_k.should be < 1.0e-5_f32
    max_v.should be < 1.0e-5_f32
    max_g.should be < 1.0e-6_f32
    max_b.should be < 1.0e-6_f32
    max_state.should be < 1.0e-6_f32

    conv_buf.release
  end
end

private def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  a.size.should eq(b.size)
  max = 0.0_f32
  a.size.times do |i|
    d = (a[i] - b[i]).abs
    max = d if d > max
  end
  max
end
