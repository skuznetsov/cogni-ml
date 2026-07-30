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

    it "rejects every non-contiguous parameter before state or data mutation" do
      valid = ML::Autograd::Variable.new(
        ML::Tensor.from_array([10.0_f32, 20.0_f32], ML::Shape.new(1_i32, 2_i32)),
        requires_grad: true
      )
      valid.grad = ML::Tensor.from_array([0.5_f32, 0.25_f32], ML::Shape.new(1_i32, 2_i32))

      base = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2_i32, 3_i32)
      )
      invalid = ML::Autograd::Variable.new(base.transpose, requires_grad: true)
      invalid.grad = ML::Tensor.from_array(
        [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32, 0.6_f32],
        ML::Shape.new(3_i32, 2_i32)
      )
      optimizer = ML::Optim::Adam.new([valid, invalid], lr: 0.1_f32, weight_decay: 0.1_f32)
      valid_data_before = valid.data.to_a
      valid_grad_before = valid.grad.not_nil!.to_a
      invalid_data_before = invalid.data.to_a
      invalid_grad_before = invalid.grad.not_nil!.to_a
      base_before = base.to_a
      invalid_data_id = invalid.data.object_id
      invalid_grad_id = invalid.grad.not_nil!.object_id
      base_storage_id = base.cpu_data.not_nil!.object_id

      expect_raises(ArgumentError) { optimizer.step }

      optimizer.state_dict.should be_empty
      valid.data.to_a.should eq(valid_data_before)
      valid.grad.not_nil!.to_a.should eq(valid_grad_before)
      invalid.data.to_a.should eq(invalid_data_before)
      invalid.grad.not_nil!.to_a.should eq(invalid_grad_before)
      base.to_a.should eq(base_before)
      invalid.data.object_id.should eq(invalid_data_id)
      invalid.grad.not_nil!.object_id.should eq(invalid_grad_id)
      base.cpu_data.not_nil!.object_id.should eq(base_storage_id)
    end

    it "rejects parameter-gradient storage aliases before mutation" do
      parameter = ML::Autograd::Variable.new(
        ML::Tensor.from_array([1.0_f32, 2.0_f32], ML::Shape.new(2_i32)),
        requires_grad: true
      )
      parameter.grad = parameter.data
      before = parameter.data.to_a
      optimizer = ML::Optim::Adam.new([parameter], lr: 0.1_f32)

      expect_raises(ArgumentError) { optimizer.step }

      parameter.data.to_a.should eq(before)
      parameter.grad.not_nil!.to_a.should eq(before)
      optimizer.state_dict.should be_empty
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

    it "rejects every non-contiguous parameter before any parameter or gradient mutation" do
      valid = ML::Autograd::Variable.new(
        ML::Tensor.from_array([10.0_f32, 20.0_f32], ML::Shape.new(1_i32, 2_i32)),
        requires_grad: true
      )
      valid.grad = ML::Tensor.from_array([0.5_f32, 0.25_f32], ML::Shape.new(1_i32, 2_i32))

      base = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2_i32, 3_i32)
      )
      invalid = ML::Autograd::Variable.new(base.transpose, requires_grad: true)
      invalid.grad = ML::Tensor.from_array(
        [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32, 0.6_f32],
        ML::Shape.new(3_i32, 2_i32)
      )
      params = [valid, invalid]
      optimizer = ML::Optim::SGD.new(
        params,
        lr: 0.1_f32,
        momentum: 0.9_f32,
        weight_decay: 0.1_f32
      )
      valid_data_before = valid.data.to_a
      valid_grad_before = valid.grad.not_nil!.to_a
      invalid_data_before = invalid.data.to_a
      invalid_grad_before = invalid.grad.not_nil!.to_a
      base_before = base.to_a
      invalid_data_id = invalid.data.object_id
      invalid_grad_id = invalid.grad.not_nil!.object_id
      base_storage_id = base.cpu_data.not_nil!.object_id

      expect_raises(ArgumentError) { optimizer.step }

      valid.data.to_a.should eq(valid_data_before)
      valid.grad.not_nil!.to_a.should eq(valid_grad_before)
      invalid.data.to_a.should eq(invalid_data_before)
      invalid.grad.not_nil!.to_a.should eq(invalid_grad_before)
      base.to_a.should eq(base_before)
      invalid.data.object_id.should eq(invalid_data_id)
      invalid.grad.not_nil!.object_id.should eq(invalid_grad_id)
      base.cpu_data.not_nil!.object_id.should eq(base_storage_id)

      # A successful step after removing the invalid parameter must behave like
      # the first momentum step. This catches hidden velocity mutation before
      # the rejected step returned.
      invalid.requires_grad = false
      optimizer.step
      valid.data.to_a[0].should be_close(9.85_f32, 1e-5_f32)
      valid.data.to_a[1].should be_close(19.775_f32, 1e-5_f32)
      valid.grad.not_nil!.to_a.should eq(valid_grad_before)
    end

    it "rejects active parameters that share storage before a double update" do
      storage = ML::Tensor.from_array([1.0_f32, 2.0_f32], ML::Shape.new(2_i32))
      first = ML::Autograd::Variable.new(storage, requires_grad: true)
      second = ML::Autograd::Variable.new(storage, requires_grad: true)
      first.grad = ML::Tensor.ones(2_i32, device: ML::Tensor::Device::CPU)
      second.grad = ML::Tensor.ones(2_i32, device: ML::Tensor::Device::CPU)
      before = storage.to_a
      optimizer = ML::Optim::SGD.new([first, second], lr: 0.1_f32)

      expect_raises(ArgumentError) { optimizer.step }

      storage.to_a.should eq(before)
      first.grad.not_nil!.to_a.should eq([1.0_f32, 1.0_f32])
      second.grad.not_nil!.to_a.should eq([1.0_f32, 1.0_f32])
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
