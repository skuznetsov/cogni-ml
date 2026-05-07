require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"

private def patterned_f32(size : Int32, scale : Float32 = 0.05_f32) : Array(Float32)
  Array(Float32).new(size) do |i|
    (Math.sin(i.to_f64 * 0.173) * scale + Math.cos(i.to_f64 * 0.071) * (scale * 0.5)).to_f32
  end
end

private def metal_buffer_from(values : Array(Float32)) : ML::MetalBuffer
  buf = ML::MetalBuffer.new(values.size.to_i64 * sizeof(Float32))
  buf.write(values)
  buf
end

private def read_metal_f32(buf : ML::MetalBuffer, count : Int32) : Array(Float32)
  ptr = buf.contents.as(Pointer(Float32))
  Array(Float32).new(count) { |i| ptr[i] }
end

private def max_abs_delta(a : Array(Float32), b : Array(Float32)) : Float64
  raise "size mismatch" unless a.size == b.size
  max = 0.0
  a.each_with_index do |v, i|
    d = (v - b[i]).abs.to_f64
    max = d if d > max
  end
  max
end

describe ML::GGUF::Qwen35Metal do
  it "captures exact conv state at a chunk checkpoint" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    h_k = 2
    h_v = 3
    s = 8
    conv_k = 4
    n_tokens = 5
    checkpoint_index = 2
    qkv_dim = 2 * h_k * s + h_v * s
    conv_state_size = (conv_k - 1) * qkv_dim

    initial_conv = patterned_f32(conv_state_size, 0.031_f32)
    qkv = patterned_f32(n_tokens * qkv_dim, 0.025_f32)
    alpha = patterned_f32(n_tokens * h_v, 0.015_f32)
    beta = patterned_f32(n_tokens * h_v, 0.011_f32)
    conv1d = patterned_f32(qkv_dim * conv_k, 0.019_f32)
    dt_bias = patterned_f32(h_v, 0.013_f32)
    ssm_a = patterned_f32(h_v, 0.017_f32).map { |v| -v.abs }

    full_buf = metal_buffer_from(initial_conv)
    full = ML::GGUF::Qwen35Metal.recurrent_prep_chunk_checkpoint(full_buf, qkv, alpha, beta, conv1d, dt_bias, ssm_a,
      h_k, h_v, s, conv_k, n_tokens, 1.0e-6_f32, checkpoint_index)

    prefix_len = checkpoint_index + 1
    prefix_buf = metal_buffer_from(initial_conv)
    ML::GGUF::Qwen35Metal.recurrent_prep_chunk(prefix_buf,
      qkv[0, prefix_len * qkv_dim], alpha[0, prefix_len * h_v], beta[0, prefix_len * h_v], conv1d, dt_bias, ssm_a,
      h_k, h_v, s, conv_k, prefix_len, 1.0e-6_f32)

    full_ref_buf = metal_buffer_from(initial_conv)
    full_ref = ML::GGUF::Qwen35Metal.recurrent_prep_chunk(full_ref_buf, qkv, alpha, beta, conv1d, dt_bias, ssm_a,
      h_k, h_v, s, conv_k, n_tokens, 1.0e-6_f32)

    max_abs_delta(full[:checkpoint_conv], read_metal_f32(prefix_buf, conv_state_size)).should be < 1.0e-6
    max_abs_delta(read_metal_f32(full_buf, conv_state_size), read_metal_f32(full_ref_buf, conv_state_size)).should be < 1.0e-6
    max_abs_delta(full[:q], full_ref[0]).should be < 1.0e-6
    max_abs_delta(full[:k], full_ref[1]).should be < 1.0e-6
    max_abs_delta(full[:v], full_ref[2]).should be < 1.0e-6
    max_abs_delta(full[:g], full_ref[3]).should be < 1.0e-6
    max_abs_delta(full[:beta], full_ref[4]).should be < 1.0e-6
  end

  it "captures exact DeltaNet SSM state at a rowwise chunk checkpoint" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    h_k = 2
    h_v = 3
    s = 128
    n_tokens = 5
    checkpoint_index = 2
    state_size = h_v * s * s
    q_size = n_tokens * h_k * s
    v_size = n_tokens * h_v * s

    initial_state = patterned_f32(state_size, 0.002_f32)
    q = patterned_f32(q_size, 0.006_f32)
    k = patterned_f32(q_size, 0.005_f32)
    v = patterned_f32(v_size, 0.004_f32)
    g = Array(Float32).new(n_tokens * h_v) { |i| 0.90_f32 + (i % 3).to_f32 * 0.01_f32 }
    beta = Array(Float32).new(n_tokens * h_v) { |i| 0.20_f32 + (i % 5).to_f32 * 0.015_f32 }
    scale = 0.08838835_f32

    full_buf = metal_buffer_from(initial_state)
    full = ML::GGUF::Qwen35Metal.delta_net_chunk_checkpoint(full_buf, q, k, v, g, beta,
      h_k, h_v, s, n_tokens, scale, checkpoint_index)

    prefix_len = checkpoint_index + 1
    prefix_buf = metal_buffer_from(initial_state)
    ML::GGUF::Qwen35Metal.delta_net_chunk_checkpoint(prefix_buf,
      q[0, prefix_len * h_k * s], k[0, prefix_len * h_k * s], v[0, prefix_len * h_v * s],
      g[0, prefix_len * h_v], beta[0, prefix_len * h_v], h_k, h_v, s, prefix_len, scale, checkpoint_index)

    full_ref_buf = metal_buffer_from(initial_state)
    full_ref = ML::GGUF::Qwen35Metal.delta_net_chunk(full_ref_buf, q, k, v, g, beta, h_k, h_v, s, n_tokens, scale)

    max_abs_delta(full[:checkpoint_state], read_metal_f32(prefix_buf, state_size)).should be < 1.0e-6
    max_abs_delta(read_metal_f32(full_buf, state_size), read_metal_f32(full_ref_buf, state_size)).should be < 1.0e-6
    max_abs_delta(full[:out], full_ref).should be < 1.0e-6
  end
end
