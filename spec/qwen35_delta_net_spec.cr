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
end
