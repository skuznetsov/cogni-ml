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

    it "reads a transposed upstream gradient in logical row-major order" do
      x = ML::Autograd::Variable.new(
        ML::Tensor.ones(3, 2, device: ML::Tensor::Device::CPU),
        requires_grad: true
      )
      scale = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [2.0_f32, 3.0_f32, 5.0_f32, 7.0_f32, 11.0_f32, 13.0_f32],
          ML::Shape.new(3, 2)
        ),
        requires_grad: false
      )
      upstream = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2, 3)
      ).transpose

      (x * scale).backward(upstream)

      x.grad.not_nil!.to_a.should eq([2.0_f32, 12.0_f32, 10.0_f32, 35.0_f32, 33.0_f32, 78.0_f32])
    end

    it "keeps a direct pointwise backward consumer stride-aware" do
      saved_input = ML::Tensor.ones(3, 2, device: ML::Tensor::Device::CPU)
      saved_scale = ML::Tensor.from_array(
        [2.0_f32, 3.0_f32, 5.0_f32, 7.0_f32, 11.0_f32, 13.0_f32],
        ML::Shape.new(3, 2)
      )
      upstream = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2, 3)
      ).transpose

      gradients = ML::Autograd::MulBackward.new(saved_input, saved_scale).backward(upstream)

      gradients[0].not_nil!.to_a.should eq([2.0_f32, 12.0_f32, 10.0_f32, 35.0_f32, 33.0_f32, 78.0_f32])
      gradients[1].not_nil!.to_a.should eq([1.0_f32, 4.0_f32, 2.0_f32, 5.0_f32, 3.0_f32, 6.0_f32])
    end

    it "keeps division and activation backward consumers stride-aware" do
      saved_input = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2, 3)
      ).transpose
      saved_scale = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 4.0_f32, 5.0_f32, 3.0_f32, 2.0_f32],
        ML::Shape.new(3, 2)
      )
      upstream = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2, 3)
      ).transpose

      div_gradients = ML::Autograd::DivBackward.new(saved_input, saved_scale).backward(upstream)
      div_gradients[0].not_nil!.to_a.should eq([1.0_f32, 2.0_f32, 0.5_f32, 1.0_f32, 1.0_f32, 3.0_f32])
      div_gradients[1].not_nil!.to_a.should eq([-1.0_f32, -4.0_f32, -0.25_f32, -1.0_f32, -1.0_f32, -9.0_f32])

      activation_input = ML::Tensor.from_array(
        [-1.0_f32, 2.0_f32, -3.0_f32, 4.0_f32, -5.0_f32, 6.0_f32],
        ML::Shape.new(2, 3)
      ).transpose
      relu_gradient = ML::Autograd::ReluBackward.new(activation_input).backward(upstream)[0].not_nil!
      relu_gradient.to_a.should eq([0.0_f32, 4.0_f32, 2.0_f32, 0.0_f32, 0.0_f32, 6.0_f32])

      sigmoid_output = ML::Tensor.from_array(
        [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32, 0.6_f32],
        ML::Shape.new(2, 3)
      ).transpose
      sigmoid_gradient = ML::Autograd::SigmoidBackward.new(sigmoid_output).backward(upstream)[0].not_nil!
      expected = [0.09_f32, 0.96_f32, 0.32_f32, 1.25_f32, 0.63_f32, 1.44_f32]
      sigmoid_gradient.to_a.each_with_index do |value, index|
        value.should be_close(expected[index], 1e-6_f32)
      end
    end

    it "keeps scalar multiplication backward stride-aware" do
      x = ML::Autograd::Variable.new(
        ML::Tensor.ones(3, 2, device: ML::Tensor::Device::CPU),
        requires_grad: true
      )
      upstream = ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(2, 3)
      ).transpose

      gradient = (x * 2.0_f32).grad_fn.not_nil!.backward(upstream)[0].not_nil!

      gradient.to_a.should eq([2.0_f32, 8.0_f32, 4.0_f32, 10.0_f32, 6.0_f32, 12.0_f32])
    end
  end

  describe "CPU numerical consumers" do
    it "reads a transposed operand in logical row-major order" do
      left = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
          ML::Shape.new(2, 3)
        ).transpose
      )
      right = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32, 50.0_f32, 60.0_f32],
          ML::Shape.new(3, 2)
        )
      )

      (left + right).data.to_a.should eq([11.0_f32, 24.0_f32, 32.0_f32, 45.0_f32, 53.0_f32, 66.0_f32])
    end

    it "keeps binary operations in logical row-major order" do
      left = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
          ML::Shape.new(2, 3)
        ).transpose
      )
      right = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [2.0_f32, 4.0_f32, 5.0_f32, 10.0_f32, 3.0_f32, 2.0_f32],
          ML::Shape.new(3, 2)
        )
      )

      (left - right).data.to_a.should eq([-1.0_f32, 0.0_f32, -3.0_f32, -5.0_f32, 0.0_f32, 4.0_f32])
      (left * right).data.to_a.should eq([2.0_f32, 16.0_f32, 10.0_f32, 50.0_f32, 9.0_f32, 12.0_f32])
      (left / right).data.to_a.should eq([0.5_f32, 1.0_f32, 0.4_f32, 0.5_f32, 1.0_f32, 3.0_f32])
    end

    it "keeps matrix multiplication in logical row-major order" do
      left = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
          ML::Shape.new(2, 3)
        ).transpose
      )
      right = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [7.0_f32, 8.0_f32, 9.0_f32, 10.0_f32],
          ML::Shape.new(2, 2)
        )
      )

      left.matmul(right).data.to_a.should eq([43.0_f32, 48.0_f32, 59.0_f32, 66.0_f32, 75.0_f32, 84.0_f32])
    end

    it "preserves logical order through a pointwise activation" do
      input = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [-1.0_f32, 2.0_f32, -3.0_f32, 4.0_f32, -5.0_f32, 6.0_f32],
          ML::Shape.new(2, 3)
        ).transpose
      )

      input.relu.data.to_a.should eq([0.0_f32, 4.0_f32, 2.0_f32, 0.0_f32, 0.0_f32, 6.0_f32])
    end

    it "preserves logical order through sigmoid" do
      input = ML::Autograd::Variable.new(
        ML::Tensor.from_array(
          [-1.0_f32, 0.0_f32, 1.0_f32, 2.0_f32, -2.0_f32, 3.0_f32],
          ML::Shape.new(2, 3)
        ).transpose
      )
      expected_inputs = [-1.0_f32, 2.0_f32, 0.0_f32, -2.0_f32, 1.0_f32, 3.0_f32]

      input.sigmoid.data.to_a.each_with_index do |value, index|
        expected = 1.0_f32 / (1.0_f32 + Math.exp(-expected_inputs[index]))
        value.should be_close(expected, 1e-6_f32)
      end
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
