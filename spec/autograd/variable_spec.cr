require "../spec_helper"

describe ML::Autograd::Variable do
  describe "#initialize" do
    it "creates variable from tensor" do
      t = ML::Tensor.new(3, 4, device: ML::Tensor::Device::CPU)
      v = ML::Autograd::Variable.new(t, requires_grad: true)
      v.shape.should eq(t.shape)
      v.requires_grad?.should be_true
    end
  end

  describe ".randn" do
    it "creates random variable" do
      v = ML::Autograd::Variable.randn(5, 5, requires_grad: true, device: ML::Tensor::Device::CPU)
      v.shape.should eq(ML::Shape.new([5, 5]))
      v.requires_grad?.should be_true
    end
  end

  describe ".zeros" do
    it "creates zero variable" do
      v = ML::Autograd::Variable.zeros(2, 3, requires_grad: false, device: ML::Tensor::Device::CPU)
      v.requires_grad?.should be_false
      data = v.data.cpu_data.not_nil!
      data.all? { |x| x == 0.0_f32 }.should be_true
    end
  end

  describe "#backward" do
    it "backward can be called on scalar variable" do
      x = ML::Autograd::Variable.randn(1, requires_grad: true, device: ML::Tensor::Device::CPU)
      x.backward
    end

    it "backward on non-scalar requires grad_output" do
      x = ML::Autograd::Variable.randn(2, 2, requires_grad: true, device: ML::Tensor::Device::CPU)
      grad_out = ML::Tensor.ones(2, 2, device: ML::Tensor::Device::CPU)
      x.backward(grad_out)
    end
  end

  describe "#zero_grad!" do
    it "resets gradient to nil" do
      v = ML::Autograd::Variable.randn(3, 3, requires_grad: true, device: ML::Tensor::Device::CPU)
      v.grad = ML::Tensor.ones(3, 3, device: ML::Tensor::Device::CPU)
      v.zero_grad!
      v.grad.should be_nil
    end
  end
end
