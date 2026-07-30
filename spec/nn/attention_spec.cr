require "../spec_helper"

describe ML::NN::MultiHeadAttention do
  describe "#initialize" do
    it "creates attention with correct dimensions" do
      mha = ML::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: ML::Tensor::Device::CPU)
      mha.embed_dim.should eq(64)
      mha.num_heads.should eq(4)
      mha.head_dim.should eq(16)
    end

    it "validates head_dim divides embed_dim" do
      mha = ML::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: ML::Tensor::Device::CPU)
      mha.head_dim.should eq(16)
    end

    it "rejects invalid head geometry and dropout before projection allocation" do
      expect_raises(ArgumentError, /embed_dim must be positive/) do
        ML::NN::MultiHeadAttention.new(embed_dim: 0, num_heads: 1, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /num_heads must be positive/) do
        ML::NN::MultiHeadAttention.new(embed_dim: 8, num_heads: 0, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /embed_dim must be divisible by num_heads/) do
        ML::NN::MultiHeadAttention.new(embed_dim: 8, num_heads: 3, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /dropout must be finite and in/) do
        ML::NN::MultiHeadAttention.new(embed_dim: 8, num_heads: 2, dropout: Float32::NAN, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /dropout must be finite and in/) do
        ML::NN::MultiHeadAttention.new(embed_dim: 8, num_heads: 2, dropout: 1.0_f32, device: ML::Tensor::Device::CPU)
      end
    end
  end

  describe "#self_attention" do
    it "produces correct output shape" do
      mha = ML::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.randn(2, 8, 64, requires_grad: false, device: ML::Tensor::Device::CPU)
      output = mha.self_attention(input)
      output.shape.should eq(ML::Shape.new([2, 8, 64]))
    end
  end

  describe "#forward" do
    it "computes cross-attention correctly" do
      mha = ML::NN::MultiHeadAttention.new(embed_dim: 32, num_heads: 2, device: ML::Tensor::Device::CPU)
      query = ML::Autograd::Variable.randn(1, 4, 32, requires_grad: false, device: ML::Tensor::Device::CPU)
      key = ML::Autograd::Variable.randn(1, 6, 32, requires_grad: false, device: ML::Tensor::Device::CPU)
      value = ML::Autograd::Variable.randn(1, 6, 32, requires_grad: false, device: ML::Tensor::Device::CPU)

      output = mha.forward(query, key, value)
      output.shape.should eq(ML::Shape.new([1, 4, 32]))
    end

    it "uses a logical transposed mask for unequal cross-attention lengths and backward" do
      mha = ML::NN::MultiHeadAttention.new(
        embed_dim: 2,
        num_heads: 1,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
      [mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj].each do |projection|
        projection.weight.data.cpu_data.not_nil!.replace([
          1.0_f32, 0.0_f32,
          0.0_f32, 1.0_f32,
        ])
        projection.weight.requires_grad = false
      end

      query = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32],
          ML::Shape.new(1_i32, 2_i32, 2_i32)
        ),
        requires_grad: true
      )
      key = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 1.0_f32],
          ML::Shape.new(1_i32, 3_i32, 2_i32)
        ),
        requires_grad: true
      )
      value = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32, 50.0_f32, 60.0_f32],
          ML::Shape.new(1_i32, 3_i32, 2_i32)
        ),
        requires_grad: true
      )
      mask_base = ML::Tensor.from_array(
        [
          0.0_f32, -1e9_f32,
          -1e9_f32, 0.0_f32,
          -1e9_f32, -1e9_f32,
        ],
        ML::Shape.new(3_i32, 2_i32)
      )
      mask = mask_base.transpose
      mask_before = mask_base.to_a
      mask_strides = mask.strides.to_a

      output = mha.forward(query, key, value, mask)

      output.shape.should eq(ML::Shape.new(1_i32, 2_i32, 2_i32))
      output.data.to_a.should eq([
        10.0_f32, 20.0_f32,
        30.0_f32, 40.0_f32,
      ])
      mask_base.to_a.should eq(mask_before)
      mask.strides.to_a.should eq(mask_strides)

      output.backward(
        ML::Tensor.from_array(
          [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
          ML::Shape.new(1_i32, 2_i32, 2_i32)
        )
      )

      query.grad.not_nil!.to_a.each { |gradient| gradient.should be_close(0.0_f32, 1e-6_f32) }
      key.grad.not_nil!.to_a.each { |gradient| gradient.should be_close(0.0_f32, 1e-6_f32) }
      value.grad.not_nil!.to_a.should eq([
        1.0_f32, 2.0_f32,
        3.0_f32, 4.0_f32,
        0.0_f32, 0.0_f32,
      ])
    end

    it "matches finite differences for unequal cross-attention query, key, and value gradients" do
      mha = ML::NN::MultiHeadAttention.new(
        embed_dim: 2,
        num_heads: 1,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
      [mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj].each do |projection|
        projection.weight.data.cpu_data.not_nil!.replace([
          1.0_f32, 0.0_f32,
          0.0_f32, 1.0_f32,
        ])
        projection.weight.requires_grad = false
      end

      query_values = [0.2_f32, -0.7_f32, 1.1_f32, 0.3_f32]
      key_values = [0.5_f32, -0.4_f32, -0.3_f32, 0.8_f32, 0.9_f32, 0.2_f32]
      value_values = [1.2_f32, -0.5_f32, 0.7_f32, 1.1_f32, -0.2_f32, 0.4_f32]
      upstream_values = [0.3_f32, -0.6_f32, 0.9_f32, 0.2_f32]
      step = 1e-3_f32

      loss = ->(q_values : Array(Float32), k_values : Array(Float32), v_values : Array(Float32)) do
        output = mha.forward(
          ML::Autograd::Variable.new(
            ML::Tensor.from_array(q_values, ML::Shape.new(1_i32, 2_i32, 2_i32)),
            requires_grad: false
          ),
          ML::Autograd::Variable.new(
            ML::Tensor.from_array(k_values, ML::Shape.new(1_i32, 3_i32, 2_i32)),
            requires_grad: false
          ),
          ML::Autograd::Variable.new(
            ML::Tensor.from_array(v_values, ML::Shape.new(1_i32, 3_i32, 2_i32)),
            requires_grad: false
          )
        ).data.to_a
        total = 0.0_f32
        output.each_with_index { |value, index| total += value * upstream_values[index] }
        total
      end

      numeric_query = Array(Float32).new(query_values.size) do |index|
        plus = query_values.dup
        minus = query_values.dup
        plus[index] += step
        minus[index] -= step
        (loss.call(plus, key_values, value_values) -
          loss.call(minus, key_values, value_values)) / (2.0_f32 * step)
      end
      numeric_key = Array(Float32).new(key_values.size) do |index|
        plus = key_values.dup
        minus = key_values.dup
        plus[index] += step
        minus[index] -= step
        (loss.call(query_values, plus, value_values) -
          loss.call(query_values, minus, value_values)) / (2.0_f32 * step)
      end
      numeric_value = Array(Float32).new(value_values.size) do |index|
        plus = value_values.dup
        minus = value_values.dup
        plus[index] += step
        minus[index] -= step
        (loss.call(query_values, key_values, plus) -
          loss.call(query_values, key_values, minus)) / (2.0_f32 * step)
      end

      query = ML::Autograd::Variable.new(
        ML::Tensor.from_array(query_values, ML::Shape.new(1_i32, 2_i32, 2_i32)),
        requires_grad: true
      )
      key = ML::Autograd::Variable.new(
        ML::Tensor.from_array(key_values, ML::Shape.new(1_i32, 3_i32, 2_i32)),
        requires_grad: true
      )
      value = ML::Autograd::Variable.new(
        ML::Tensor.from_array(value_values, ML::Shape.new(1_i32, 3_i32, 2_i32)),
        requires_grad: true
      )
      mha.forward(query, key, value).backward(
        ML::Tensor.from_array(
          upstream_values,
          ML::Shape.new(1_i32, 2_i32, 2_i32)
        )
      )

      query.grad.not_nil!.to_a.each_with_index do |gradient, index|
        gradient.should be_close(numeric_query[index], 2e-3_f32)
      end
      key.grad.not_nil!.to_a.each_with_index do |gradient, index|
        gradient.should be_close(numeric_key[index], 2e-3_f32)
      end
      value.grad.not_nil!.to_a.each_with_index do |gradient, index|
        gradient.should be_close(numeric_value[index], 2e-3_f32)
      end
    end

    it "applies a separate rank-three mask to each batch item" do
      mha = ML::NN::MultiHeadAttention.new(
        embed_dim: 1,
        num_heads: 1,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
      [mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj].each do |projection|
        projection.weight.data.cpu_data.not_nil![0] = 1.0_f32
        projection.weight.requires_grad = false
      end

      query = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 1.0_f32],
          ML::Shape.new(2_i32, 1_i32, 1_i32)
        ),
        requires_grad: false
      )
      key = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32],
          ML::Shape.new(2_i32, 2_i32, 1_i32)
        ),
        requires_grad: false
      )
      value = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32],
          ML::Shape.new(2_i32, 2_i32, 1_i32)
        ),
        requires_grad: false
      )
      mask = ML::Tensor.from_array(
        [
          0.0_f32, -1e9_f32,
          -1e9_f32, 0.0_f32,
        ],
        ML::Shape.new(2_i32, 1_i32, 2_i32)
      )

      output = mha.forward(query, key, value, mask)

      output.shape.should eq(ML::Shape.new(2_i32, 1_i32, 1_i32))
      output.data.to_a.should eq([10.0_f32, 40.0_f32])
    end

    it "rejects inconsistent cross-attention and mask shapes before projection" do
      mha = ML::NN::MultiHeadAttention.new(
        embed_dim: 2,
        num_heads: 1,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
      query = ML::Autograd::Variable.ones(
        1,
        2,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      key = ML::Autograd::Variable.ones(
        1,
        3,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      short_value = ML::Autograd::Variable.ones(
        1,
        2,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )

      expect_raises(ArgumentError, /key and value sequence lengths/) do
        mha.forward(query, key, short_value)
      end

      rank_two_query = ML::Autograd::Variable.ones(
        2,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      expect_raises(ArgumentError, /query must have shape/) do
        mha.forward(rank_two_query, key, key)
      end

      other_batch = ML::Autograd::Variable.ones(
        2,
        3,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      expect_raises(ArgumentError, /batch sizes must match/) do
        mha.forward(query, other_batch, other_batch)
      end

      wrong_embedding = ML::Autograd::Variable.ones(
        1,
        3,
        1,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      expect_raises(ArgumentError, /key embedding dimension/) do
        mha.forward(query, wrong_embedding, wrong_embedding)
      end

      empty_query = ML::Autograd::Variable.zeros(
        1,
        0,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      expect_raises(ArgumentError, /query batch and sequence dimensions/) do
        mha.forward(empty_query, key, key)
      end

      value = ML::Autograd::Variable.ones(
        1,
        3,
        2,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
      invalid_mask = ML::Tensor.zeros(
        3,
        2,
        device: ML::Tensor::Device::CPU
      )
      expect_raises(ArgumentError, /attention mask shape/) do
        mha.forward(query, key, value, invalid_mask)
      end

      fully_masked = ML::Tensor.from_array(
        [
          -Float32::INFINITY, -Float32::INFINITY, -Float32::INFINITY,
          0.0_f32, 0.0_f32, 0.0_f32,
        ],
        ML::Shape.new(2_i32, 3_i32)
      )
      expect_raises(ArgumentError, /no finite attention source/) do
        mha.forward(query, key, value, fully_masked)
      end
    end

    it "supports backward on CPU" do
      mha = ML::NN::MultiHeadAttention.new(embed_dim: 8, num_heads: 2, device: ML::Tensor::Device::CPU)
      query = ML::Autograd::Variable.randn(1, 3, 8, requires_grad: true, device: ML::Tensor::Device::CPU)
      key = ML::Autograd::Variable.randn(1, 3, 8, requires_grad: true, device: ML::Tensor::Device::CPU)
      value = ML::Autograd::Variable.randn(1, 3, 8, requires_grad: true, device: ML::Tensor::Device::CPU)

      output = mha.forward(query, key, value)
      loss = output.mean
      loss.backward

      query.grad.should_not be_nil
      key.grad.should_not be_nil
      value.grad.should_not be_nil
    end
  end

  describe "#parameters" do
    it "returns all projection weights and biases" do
      mha = ML::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: ML::Tensor::Device::CPU)
      params = mha.parameters
      params.size.should eq(8)
    end
  end

  describe "GPU attention" do
    it "works on GPU" do
      if ML::Metal::Device.available?
        mha = ML::NN::MultiHeadAttention.new(embed_dim: 64, num_heads: 4, device: ML::Tensor::Device::GPU)
        input = ML::Autograd::Variable.randn(2, 8, 64, requires_grad: false, device: ML::Tensor::Device::GPU)
        output = mha.self_attention(input)
        output.shape.should eq(ML::Shape.new([2, 8, 64]))
      end
    end
  end
end
