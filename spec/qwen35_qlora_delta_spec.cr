require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_qlora_delta"

private def qwen35_qlora_delta_loss(
  q : Array(Float32), k : Array(Float32), v : Array(Float32),
  decay : Array(Float32), beta : Array(Float32), grad : Array(Float32),
  rows : Int32, h_k : Int32, h_v : Int32, size : Int32,
) : Float64
  output = ML::GGUF::Qwen35QLoRA::Delta.forward(q, k, v, decay, beta, rows, h_k, h_v, size)
  output.zip(grad).sum { |value, derivative| value.to_f64 * derivative.to_f64 }
end

describe ML::GGUF::Qwen35QLoRA::Delta do
  it "matches repeated native CPU steps with grouped Q/K heads" do
    rows = 4_i32
    h_k = 2_i32
    h_v = 3_i32
    size = 3_i32
    q = Array(Float32).new(rows * h_k * size) { |i| ((i * 7 % 13) - 6).to_f32 / 9.0_f32 }
    k = Array(Float32).new(rows * h_k * size) { |i| ((i * 5 % 11) - 5).to_f32 / 8.0_f32 }
    v = Array(Float32).new(rows * h_v * size) { |i| ((i * 3 % 17) - 8).to_f32 / 7.0_f32 }
    decay = Array(Float32).new(rows * h_v) { |i| 0.72_f32 + (i % 5).to_f32 * 0.035_f32 }
    beta = Array(Float32).new(rows * h_v) { |i| 0.15_f32 + (i % 4).to_f32 * 0.11_f32 }
    scale = 1.0_f32 / Math.sqrt(size.to_f64).to_f32

    expected = Array(Float32).new(rows * h_v * size, 0.0_f32)
    state = Array(Float32).new(h_v * size * size, 0.0_f32)
    rows.times do |row|
      q_row = Array(Float32).new(h_k * size) { |i| q[row * h_k * size + i] }
      k_row = Array(Float32).new(h_k * size) { |i| k[row * h_k * size + i] }
      v_row = Array(Float32).new(h_v * size) { |i| v[row * h_v * size + i] }
      decay_row = Array(Float32).new(h_v) { |i| decay[row * h_v + i] }
      beta_row = Array(Float32).new(h_v) { |i| beta[row * h_v + i] }
      y_row = Array(Float32).new(h_v * size, 0.0_f32)
      ML::GGUF::Qwen35CPU.delta_net_step!(
        state, q_row, k_row, v_row, decay_row, beta_row, y_row,
        h_k, h_v, size, scale,
      )
      size.times do |d|
        h_v.times do |head|
          expected[row * h_v * size + head * size + d] = y_row[head * size + d]
        end
      end
    end

    actual = ML::GGUF::Qwen35QLoRA::Delta.forward(q, k, v, decay, beta, rows, h_k, h_v, size)
    actual.size.should eq(expected.size)
    actual.zip(expected).each { |value, reference| value.should be_close(reference, 2.0e-5_f32) }
  end

  it "matches finite differences for Q, K, V, decay, and beta" do
    rows = 3_i32
    h_k = 2_i32
    h_v = 3_i32
    size = 2_i32
    q = [0.31_f32, -0.22_f32, 0.17_f32, 0.41_f32, -0.13_f32, 0.29_f32,
         0.27_f32, -0.34_f32, 0.19_f32, 0.12_f32, -0.23_f32, 0.36_f32]
    k = [-0.18_f32, 0.38_f32, 0.26_f32, -0.11_f32, 0.33_f32, 0.21_f32,
         -0.29_f32, 0.14_f32, 0.08_f32, 0.32_f32, -0.16_f32, 0.24_f32]
    v = Array(Float32).new(rows * h_v * size) { |i| ((i * 5 % 17) - 8).to_f32 / 10.0_f32 }
    decay = [0.78_f32, 0.81_f32, 0.74_f32, 0.85_f32, 0.77_f32, 0.89_f32, 0.73_f32, 0.82_f32, 0.87_f32]
    beta = [0.24_f32, 0.31_f32, 0.18_f32, 0.42_f32, 0.27_f32, 0.35_f32, 0.22_f32, 0.38_f32, 0.29_f32]
    grad = Array(Float32).new(rows * h_v * size) { |i| ((i * 7 % 19) - 9).to_f32 / 11.0_f32 }

    dq, dk, dv, ddecay, dbeta = ML::GGUF::Qwen35QLoRA::Delta.backward(
      q, k, v, decay, beta, grad, rows, h_k, h_v, size,
    )

    epsilon = 1.0e-3_f32
    probes = [
      {q, dq, 0}, {q, dq, 8}, {k, dk, 2}, {k, dk, 10},
      {v, dv, 7}, {v, dv, 16}, {decay, ddecay, 4}, {beta, dbeta, 7},
    ]
    probes.each do |input, derivative, index|
      original = input[index]
      input[index] = original + epsilon
      plus = qwen35_qlora_delta_loss(q, k, v, decay, beta, grad, rows, h_k, h_v, size)
      input[index] = original - epsilon
      minus = qwen35_qlora_delta_loss(q, k, v, decay, beta, grad, rows, h_k, h_v, size)
      input[index] = original
      numeric = ((plus - minus) / (2.0_f64 * epsilon.to_f64)).to_f32
      derivative[index].should be_close(numeric, 2.0e-3_f32)
    end
  end

  it "backpropagates across recomputed checkpoint boundaries" do
    rows = 19_i32
    h_k = 2_i32
    h_v = 3_i32
    size = 2_i32
    q = Array(Float32).new(rows * h_k * size) { |i| ((i * 7 % 23) - 11).to_f32 / 17.0_f32 }
    k = Array(Float32).new(rows * h_k * size) { |i| ((i * 5 % 19) - 9).to_f32 / 15.0_f32 }
    v = Array(Float32).new(rows * h_v * size) { |i| ((i * 3 % 29) - 14).to_f32 / 21.0_f32 }
    decay = Array(Float32).new(rows * h_v) { |i| 0.76_f32 + (i % 7).to_f32 * 0.025_f32 }
    beta = Array(Float32).new(rows * h_v) { |i| 0.12_f32 + (i % 6).to_f32 * 0.09_f32 }
    grad = Array(Float32).new(rows * h_v * size) { |i| ((i * 11 % 31) - 15).to_f32 / 23.0_f32 }

    dq, dk, dv, ddecay, dbeta = ML::GGUF::Qwen35QLoRA::Delta.backward(
      q, k, v, decay, beta, grad, rows, h_k, h_v, size,
    )

    epsilon = 1.0e-3_f32
    probes = [
      {q, dq, 16 * h_k * size},
      {k, dk, 17 * h_k * size + 3},
      {v, dv, 16 * h_v * size + 4},
      {decay, ddecay, 16 * h_v + 2},
      {beta, dbeta, 17 * h_v + 1},
    ]
    probes.each do |input, derivative, index|
      original = input[index]
      input[index] = original + epsilon
      plus = qwen35_qlora_delta_loss(q, k, v, decay, beta, grad, rows, h_k, h_v, size)
      input[index] = original - epsilon
      minus = qwen35_qlora_delta_loss(q, k, v, decay, beta, grad, rows, h_k, h_v, size)
      input[index] = original
      numeric = ((plus - minus) / (2.0_f64 * epsilon.to_f64)).to_f32
      derivative[index].should be_close(numeric, 3.0e-3_f32)
    end
  end

  it "rejects malformed dimensions and array shapes" do
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35QLoRA::Delta.forward([1.0_f32], [1.0_f32], [1.0_f32], [1.0_f32], [1.0_f32], 1, 1, 1, 0)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35QLoRA::Delta.forward([1.0_f32], [1.0_f32], [1.0_f32], [1.0_f32], [1.0_f32], 2, 1, 1, 1)
    end
  end
end
