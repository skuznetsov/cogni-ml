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

    it "rejects invalid normalized shapes and epsilon" do
      expect_raises(ArgumentError, /normalized_shape must not be empty/) do
        ML::NN::LayerNorm.new([] of Int32, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /normalized_shape dimensions must be positive/) do
        ML::NN::LayerNorm.new([4_i32, 0_i32], device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /eps must be finite and positive/) do
        ML::NN::LayerNorm.new(4, eps: 0.0_f32, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /element count overflow/) do
        ML::NN::LayerNorm.new([46_341_i32, 46_341_i32], device: ML::Tensor::Device::CPU)
      end
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

    it "normalizes a transposed CPU view in logical row-major order" do
      base = ML::Tensor.from_array(
        [
          1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32,
          16.0_f32, 32.0_f32, 64.0_f32, 128.0_f32,
          3.0_f32, 6.0_f32, 12.0_f32, 24.0_f32,
          5.0_f32, 7.0_f32, 11.0_f32, 13.0_f32,
          17.0_f32, 19.0_f32, 23.0_f32, 29.0_f32,
          31.0_f32, 37.0_f32, 41.0_f32, 43.0_f32,
        ],
        ML::Shape.new(2_i32, 3_i32, 4_i32)
      )
      input = ML::Autograd::Variable.new(base.transpose, requires_grad: false)
      dense_input = ML::Autograd::Variable.new(input.data.contiguous, requires_grad: false)
      ln = ML::NN::LayerNorm.new(3, eps: 1e-5_f32, device: ML::Tensor::Device::CPU)
      ln.weight.data.cpu_data.not_nil!.replace([1.5_f32, -2.0_f32, 0.25_f32])
      ln.bias.data.cpu_data.not_nil!.replace([-0.5_f32, 1.25_f32, 3.0_f32])
      ln.weight.requires_grad = false
      ln.bias.requires_grad = false
      base_before = base.cpu_data.not_nil!.dup
      input_id = input.data.object_id
      input_strides = input.data.strides.to_a
      weight_before = ln.weight.data.to_a
      bias_before = ln.bias.data.to_a

      view_output = ln.forward(input)
      dense_output = ln.forward(dense_input)

      base.cpu_data.not_nil!.should eq(base_before)
      input.data.object_id.should eq(input_id)
      input.data.strides.to_a.should eq(input_strides)
      ln.weight.data.to_a.should eq(weight_before)
      ln.bias.data.to_a.should eq(bias_before)
      view_output.shape.should eq(ML::Shape.new(2_i32, 4_i32, 3_i32))
      view_output.requires_grad?.should be_false
      view_output.grad_fn.should be_nil
      actual = view_output.data.to_a
      expected = [
        -1.7781993_f32, -1.5570261_f32, 2.862155_f32,
        -1.7781993_f32, -1.5570261_f32, 2.862155_f32,
        -1.7781994_f32, -1.5570264_f32, 2.862155_f32,
        -1.7781994_f32, -1.5570264_f32, 2.862155_f32,
        -2.2882488_f32, 1.375491_f32, 3.3137279_f32,
        -2.2033248_f32, 1.5744429_f32, 3.3244429_f32,
        -2.2033248_f32, 1.5744429_f32, 3.3244429_f32,
        -2.3765526_f32, 1.1412145_f32, 3.2991605_f32,
      ]
      actual.each_with_index do |value, index|
        value.should be_close(expected[index], 1e-5_f32)
        value.should be_close(dense_output.data.to_a[index], 1e-5_f32)
      end
    end

    it "preserves LayerNorm gradients through a transposed CPU view" do
      values = [
        1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32,
        16.0_f32, 32.0_f32, 64.0_f32, 128.0_f32,
        3.0_f32, 6.0_f32, 12.0_f32, 24.0_f32,
      ]
      view_base = ML::Autograd::Variable.new(
        ML::Tensor.from_array(values, ML::Shape.new(1_i32, 3_i32, 4_i32)),
        requires_grad: true
      )
      view_input = view_base.transpose
      dense_input = ML::Autograd::Variable.new(view_input.data.contiguous, requires_grad: true)
      view_ln = ML::NN::LayerNorm.new(3, eps: 1e-5_f32, device: ML::Tensor::Device::CPU)
      dense_ln = ML::NN::LayerNorm.new(3, eps: 1e-5_f32, device: ML::Tensor::Device::CPU)
      [view_ln, dense_ln].each do |ln|
        ln.weight.data.cpu_data.not_nil!.replace([1.5_f32, -2.0_f32, 0.25_f32])
        ln.bias.data.cpu_data.not_nil!.replace([-0.5_f32, 1.25_f32, 3.0_f32])
      end
      upstream = ML::Tensor.from_array(
        [
          0.1_f32, -0.2_f32, 0.3_f32,
          0.4_f32, 0.5_f32, -0.6_f32,
          0.7_f32, -0.8_f32, 0.9_f32,
          -1.0_f32, 1.1_f32, -1.2_f32,
        ],
        ML::Shape.new(1_i32, 4_i32, 3_i32)
      )

      view_ln.forward(view_input).backward(upstream)
      dense_ln.forward(dense_input).backward(upstream)

      view_grad = view_base.grad.not_nil!.transpose.to_a
      dense_grad = dense_input.grad.not_nil!.to_a
      view_grad.each_with_index do |value, index|
        value.should be_close(dense_grad[index], 1e-5_f32)
      end
      view_ln.weight.grad.not_nil!.to_a.each_with_index do |value, index|
        value.should be_close(dense_ln.weight.grad.not_nil!.to_a[index], 1e-5_f32)
      end
      view_ln.bias.grad.not_nil!.to_a.each_with_index do |value, index|
        value.should be_close(dense_ln.bias.grad.not_nil!.to_a[index], 1e-5_f32)
      end
    end

    it "does not build a LayerNorm graph inside NoGrad" do
      ln = ML::NN::LayerNorm.new(3, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.ones(
        2,
        3,
        requires_grad: true,
        device: ML::Tensor::Device::CPU
      )

      output = ML::Autograd::NoGrad.with { ln.forward(input) }

      output.requires_grad?.should be_false
      output.grad_fn.should be_nil
      input.requires_grad?.should be_true
      ln.weight.requires_grad?.should be_true
      ln.bias.requires_grad?.should be_true
      ML::Autograd::NoGrad.enabled?.should be_false
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
  describe "#initialize" do
    it "rejects invalid dimension and epsilon" do
      expect_raises(ArgumentError, /dim must be positive/) do
        ML::NN::RMSNorm.new(0, device: ML::Tensor::Device::CPU)
      end
      expect_raises(ArgumentError, /eps must be finite and positive/) do
        ML::NN::RMSNorm.new(4, eps: Float32::NAN, device: ML::Tensor::Device::CPU)
      end
    end
  end

  describe "#forward" do
    it "produces correct output shape" do
      rms = ML::NN::RMSNorm.new(64, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.randn(8, 64, requires_grad: false, device: ML::Tensor::Device::CPU)
      output = rms.forward(input)
      output.shape.should eq(ML::Shape.new([8, 64]))
    end

    it "normalizes a transposed CPU view in logical row-major order" do
      base = ML::Tensor.from_array(
        [
          1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32,
          16.0_f32, 32.0_f32, 64.0_f32, 128.0_f32,
          3.0_f32, 6.0_f32, 12.0_f32, 24.0_f32,
          5.0_f32, 7.0_f32, 11.0_f32, 13.0_f32,
          17.0_f32, 19.0_f32, 23.0_f32, 29.0_f32,
          31.0_f32, 37.0_f32, 41.0_f32, 43.0_f32,
        ],
        ML::Shape.new(2_i32, 3_i32, 4_i32)
      )
      input = ML::Autograd::Variable.new(base.transpose, requires_grad: false)
      dense_input = ML::Autograd::Variable.new(input.data.contiguous, requires_grad: false)
      rms = ML::NN::RMSNorm.new(3, eps: 1e-5_f32, device: ML::Tensor::Device::CPU)
      rms.weight.data.cpu_data.not_nil!.replace([1.5_f32, -2.0_f32, 0.25_f32])
      rms.weight.requires_grad = false
      base_before = base.cpu_data.not_nil!.dup
      input_id = input.data.object_id
      input_strides = input.data.strides.to_a
      weight_before = rms.weight.data.to_a

      view_output = rms.forward(input)
      dense_output = rms.forward(dense_input)

      base.cpu_data.not_nil!.should eq(base_before)
      input.data.object_id.should eq(input_id)
      input.data.strides.to_a.should eq(input_strides)
      rms.weight.data.to_a.should eq(weight_before)
      view_output.shape.should eq(ML::Shape.new(2_i32, 4_i32, 3_i32))
      view_output.requires_grad?.should be_false
      view_output.grad_fn.should be_nil
      actual = view_output.data.to_a
      expected = [
        0.15929827_f32, -3.398363_f32, 0.079649135_f32,
        0.15929827_f32, -3.398363_f32, 0.079649135_f32,
        0.15929827_f32, -3.398363_f32, 0.079649135_f32,
        0.15929827_f32, -3.398363_f32, 0.079649135_f32,
        0.36380345_f32, -1.6492423_f32, 0.37593022_f32,
        0.43118334_f32, -1.560473_f32, 0.37985197_f32,
        0.5919342_f32, -1.6502408_f32, 0.3677167_f32,
        0.6316669_f32, -1.8788042_f32, 0.34822664_f32,
      ]
      actual.each_with_index do |value, index|
        value.should be_close(expected[index], 1e-5_f32)
        value.should be_close(dense_output.data.to_a[index], 1e-5_f32)
      end
    end

    it "preserves RMSNorm gradients through a transposed CPU view" do
      values = [
        1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32,
        16.0_f32, 32.0_f32, 64.0_f32, 128.0_f32,
        3.0_f32, 6.0_f32, 12.0_f32, 24.0_f32,
      ]
      view_base = ML::Autograd::Variable.new(
        ML::Tensor.from_array(values, ML::Shape.new(1_i32, 3_i32, 4_i32)),
        requires_grad: true
      )
      view_input = view_base.transpose
      dense_input = ML::Autograd::Variable.new(view_input.data.contiguous, requires_grad: true)
      view_rms = ML::NN::RMSNorm.new(3, eps: 1e-5_f32, device: ML::Tensor::Device::CPU)
      dense_rms = ML::NN::RMSNorm.new(3, eps: 1e-5_f32, device: ML::Tensor::Device::CPU)
      [view_rms, dense_rms].each do |rms|
        rms.weight.data.cpu_data.not_nil!.replace([1.5_f32, -2.0_f32, 0.25_f32])
      end
      upstream = ML::Tensor.from_array(
        [
          0.1_f32, -0.2_f32, 0.3_f32,
          0.4_f32, 0.5_f32, -0.6_f32,
          0.7_f32, -0.8_f32, 0.9_f32,
          -1.0_f32, 1.1_f32, -1.2_f32,
        ],
        ML::Shape.new(1_i32, 4_i32, 3_i32)
      )

      view_rms.forward(view_input).backward(upstream)
      dense_rms.forward(dense_input).backward(upstream)

      view_grad = view_base.grad.not_nil!.transpose.to_a
      dense_grad = dense_input.grad.not_nil!.to_a
      view_grad.each_with_index do |value, index|
        value.should be_close(dense_grad[index], 1e-5_f32)
      end
      view_rms.weight.grad.not_nil!.to_a.each_with_index do |value, index|
        value.should be_close(dense_rms.weight.grad.not_nil!.to_a[index], 1e-5_f32)
      end
    end

    it "does not build an RMSNorm graph inside NoGrad" do
      rms = ML::NN::RMSNorm.new(3, device: ML::Tensor::Device::CPU)
      input = ML::Autograd::Variable.ones(
        2,
        3,
        requires_grad: true,
        device: ML::Tensor::Device::CPU
      )

      output = ML::Autograd::NoGrad.with { rms.forward(input) }

      output.requires_grad?.should be_false
      output.grad_fn.should be_nil
      input.requires_grad?.should be_true
      rms.weight.requires_grad?.should be_true
      ML::Autograd::NoGrad.enabled?.should be_false
    end
  end
end
