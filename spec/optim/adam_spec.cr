require "../spec_helper"

describe ML::Optim::Adam do
  describe "#initialize" do
    it "creates optimizer with parameters" do
      params = [
        ML::Autograd::Variable.randn(10, 10, requires_grad: true, device: ML::Tensor::Device::CPU),
        ML::Autograd::Variable.randn(10, requires_grad: true, device: ML::Tensor::Device::CPU),
      ]
      opt = ML::Optim::Adam.new(params, lr: 0.001_f32)
      opt.lr.should eq(0.001_f32)
    end
  end

  describe "#step" do
    it "updates parameters with gradients" do
      params = [
        ML::Autograd::Variable.randn(5, 5, requires_grad: true, device: ML::Tensor::Device::CPU),
      ]

      params[0].grad = ML::Tensor.ones(5, 5, device: ML::Tensor::Device::CPU)

      opt = ML::Optim::Adam.new(params, lr: 0.1_f32)

      original = params[0].data.cpu_data.not_nil![0]

      opt.step

      updated = params[0].data.cpu_data.not_nil![0]
      (original - updated).abs.should be > 0.0
    end
  end

  describe "#zero_grad" do
    it "zeros all gradients" do
      params = [
        ML::Autograd::Variable.randn(3, 3, requires_grad: true, device: ML::Tensor::Device::CPU),
        ML::Autograd::Variable.randn(3, requires_grad: true, device: ML::Tensor::Device::CPU),
      ]

      params[0].grad = ML::Tensor.ones(3, 3, device: ML::Tensor::Device::CPU)
      params[1].grad = ML::Tensor.ones(3, device: ML::Tensor::Device::CPU)

      opt = ML::Optim::Adam.new(params, lr: 0.001_f32)
      opt.zero_grad

      params.each { |p| p.grad.should be_nil }
    end
  end
end

describe "Adam with weight decay" do
  it "applies weight decay when configured" do
    params = [
      ML::Autograd::Variable.ones(5, 5, requires_grad: true, device: ML::Tensor::Device::CPU),
    ]

    params[0].grad = ML::Tensor.zeros(5, 5, device: ML::Tensor::Device::CPU)

    opt = ML::Optim::Adam.new(params, lr: 0.1_f32, weight_decay: 0.1_f32)

    original = params[0].data.cpu_data.not_nil![0]
    opt.step

    updated = params[0].data.cpu_data.not_nil![0]
    updated.should be < original
  end
end

describe ML::Optim::SGD do
  describe "#step" do
    it "performs simple gradient descent" do
      params = [
        ML::Autograd::Variable.randn(4, 4, requires_grad: true, device: ML::Tensor::Device::CPU),
      ]

      params[0].grad = ML::Tensor.ones(4, 4, device: ML::Tensor::Device::CPU)

      opt = ML::Optim::SGD.new(params, lr: 0.5_f32)

      original = params[0].data.cpu_data.not_nil![0]
      opt.step

      expected = original - 0.5_f32
      updated = params[0].data.cpu_data.not_nil![0]
      updated.should be_close(expected, 1e-5)
    end
  end
end
