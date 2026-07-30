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
