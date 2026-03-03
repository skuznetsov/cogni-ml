require "../spec_helper"

describe ML::NN::LayerNorm do
  describe "#initialize" do
    it "creates layer norm with correct size" do
      ln = ML::NN::LayerNorm.new(64, device: ML::Tensor::Device::CPU)
      ln.normalized_shape.should eq([64])
    end

    it "initializes weight (gamma) to ones" do
      ln = ML::NN::LayerNorm.new(32, device: ML::Tensor::Device::CPU)
      weight = ln.weight.data.cpu_data.not_nil!
      weight.all? { |x| (x - 1.0_f32).abs < 1e-6 }.should be_true
    end

    it "initializes bias (beta) to zeros" do
      ln = ML::NN::LayerNorm.new(32, device: ML::Tensor::Device::CPU)
      bias = ln.bias.data.cpu_data.not_nil!
      bias.all? { |x| x.abs < 1e-6 }.should be_true
    end
  end

  describe "#forward" do
    it "produces correct output shape" do
      ln = ML::NN::LayerNorm.new(64, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.randn(8, 64, requires_grad: false, device: ML::Tensor::Device::CPU)
      output = ln.forward(input)
      output.shape.should eq(ML::Shape.new([8, 64]))
    end

    it "normalizes to approximately zero mean" do
      ln = ML::NN::LayerNorm.new(128, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.randn(4, 128, requires_grad: false, device: ML::Tensor::Device::CPU)
      output = ln.forward(input)

      data = output.data.cpu_data.not_nil!

      4.times do |row|
        row_data = (0...128).map { |c| data[row * 128 + c] }
        mean = row_data.sum / 128
        mean.abs.should be < 0.1
      end
    end
  end

  describe "#parameters" do
    it "returns gamma and beta" do
      ln = ML::NN::LayerNorm.new(64, device: ML::Tensor::Device::CPU)
      params = ln.parameters
      params.size.should eq(2)
    end
  end
end

describe ML::NN::RMSNorm do
  describe "#forward" do
    it "produces correct output shape" do
      rms = ML::NN::RMSNorm.new(64, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.randn(8, 64, requires_grad: false, device: ML::Tensor::Device::CPU)
      output = rms.forward(input)
      output.shape.should eq(ML::Shape.new([8, 64]))
    end
  end
end
