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

    it "rejects hyperparameters that would produce invalid updates" do
      params = [] of ML::Autograd::Variable

      expect_raises(ArgumentError, /lr must be finite and non-negative/) do
        ML::Optim::Adam.new(params, lr: Float32::NAN)
      end
      expect_raises(ArgumentError, /beta1 must be finite and in/) do
        ML::Optim::Adam.new(params, beta1: 1.0_f32)
      end
      expect_raises(ArgumentError, /beta2 must be finite and in/) do
        ML::Optim::Adam.new(params, beta2: -0.1_f32)
      end
      expect_raises(ArgumentError, /eps must be finite and positive/) do
        ML::Optim::Adam.new(params, eps: 0.0_f32)
      end
      expect_raises(ArgumentError, /weight_decay must be finite and non-negative/) do
        ML::Optim::Adam.new(params, weight_decay: -0.1_f32)
      end
    end

    it "rejects invalid parameter-group values on every admission path" do
      params = [] of ML::Autograd::Variable

      expect_raises(ArgumentError, /lr must be finite and non-negative/) do
        ML::Optim::ParamGroup.new(params, lr: Float32::NAN)
      end

      group = ML::Optim::ParamGroup.new(params)
      group.weight_decay = -0.1_f32
      expect_raises(ArgumentError, /weight_decay must be finite and non-negative/) do
        ML::Optim::Adam.new([group])
      end

      optimizer = ML::Optim::Adam.new(params)
      expect_raises(ArgumentError, /weight_decay must be finite and non-negative/) do
        optimizer.add_param_group(group)
      end
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
  describe "#initialize" do
    it "rejects invalid update hyperparameters" do
      params = [] of ML::Autograd::Variable

      expect_raises(ArgumentError, /lr must be finite and non-negative/) do
        ML::Optim::SGD.new(params, lr: -0.1_f32)
      end
      expect_raises(ArgumentError, /momentum must be finite and non-negative/) do
        ML::Optim::SGD.new(params, momentum: Float32::NAN)
      end
      expect_raises(ArgumentError, /weight_decay must be finite and non-negative/) do
        ML::Optim::SGD.new(params, weight_decay: -0.1_f32)
      end
      expect_raises(ArgumentError, /dampening must be finite and non-negative/) do
        ML::Optim::SGD.new(params, dampening: -0.1_f32)
      end
      expect_raises(ArgumentError, /Nesterov requires positive momentum and zero dampening/) do
        ML::Optim::SGD.new(params, momentum: 0.0_f32, nesterov: true)
      end
    end
  end

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

describe "learning-rate schedulers" do
  it "rejects invalid step and decay parameters" do
    optimizer = ML::Optim::Adam.new([] of ML::Autograd::Variable)

    expect_raises(ArgumentError, /step_size must be positive/) do
      ML::Optim::StepLR.new(optimizer, 0)
    end
    expect_raises(ArgumentError, /gamma must be finite and non-negative/) do
      ML::Optim::StepLR.new(optimizer, 1, Float32::NAN)
    end
    expect_raises(ArgumentError, /gamma must be finite and non-negative/) do
      ML::Optim::ExponentialLR.new(optimizer, -0.1_f32)
    end
  end
end
