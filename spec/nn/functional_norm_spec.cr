require "../spec_helper"

describe ML::Ops::CPU do
  it "evaluates graphless LayerNorm over logical rows and reports read materialization" do
    base = ML::Tensor.from_array(
      [
        1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32,
        16.0_f32, 32.0_f32, 64.0_f32, 128.0_f32,
        3.0_f32, 6.0_f32, 12.0_f32, 24.0_f32,
      ],
      ML::Shape.new(1_i32, 3_i32, 4_i32)
    )
    input = base.transpose
    weight = ML::Tensor.from_array(
      [1.5_f32, -2.0_f32, 0.25_f32],
      ML::Shape.new(3_i32)
    )
    bias = ML::Tensor.from_array(
      [-0.5_f32, 1.25_f32, 3.0_f32],
      ML::Shape.new(3_i32)
    )
    base_before = base.cpu_data.not_nil!.dup
    input_id = input.object_id
    input_strides = input.strides.to_a
    weight_before = weight.to_a
    bias_before = bias.to_a

    evaluation = ML::Ops::CPU.layer_norm_with_stats(
      input,
      weight,
      bias,
      [3_i32],
      1e-5_f32
    )

    base.cpu_data.not_nil!.should eq(base_before)
    input.object_id.should eq(input_id)
    input.strides.to_a.should eq(input_strides)
    weight.to_a.should eq(weight_before)
    bias.to_a.should eq(bias_before)
    evaluation.output.should be_a(ML::Tensor)
    evaluation.output.on_cpu?.should be_true
    evaluation.output.contiguous?.should be_true
    evaluation.output.shape.should eq(ML::Shape.new(1_i32, 4_i32, 3_i32))
    evaluation.read_stats.materialization_count.should eq(1)
    evaluation.read_stats.materialized_bytes.should eq(48_i64)
    evaluation.read_stats.input_materialized_bytes.should eq(48_i64)
    evaluation.read_stats.weight_materialized_bytes.should eq(0_i64)
    evaluation.read_stats.bias_materialized_bytes.should eq(0_i64)
    expected = [
      -1.7781993_f32, -1.5570261_f32, 2.862155_f32,
      -1.7781993_f32, -1.5570261_f32, 2.862155_f32,
      -1.7781994_f32, -1.5570264_f32, 2.862155_f32,
      -1.7781994_f32, -1.5570264_f32, 2.862155_f32,
    ]
    evaluation.output.to_a.each_with_index do |value, index|
      value.should be_close(expected[index], 1e-5_f32)
    end

    dense = ML::Ops::CPU.layer_norm_with_stats(
      input.contiguous,
      weight,
      bias,
      [3_i32],
      1e-5_f32
    )
    dense.read_stats.materialization_count.should eq(0)
    dense.read_stats.materialized_bytes.should eq(0_i64)
  end

  it "evaluates graphless RMSNorm over logical rows and validates the trailing shape" do
    base = ML::Tensor.from_array(
      [
        1.0_f32, 2.0_f32, 4.0_f32, 8.0_f32,
        16.0_f32, 32.0_f32, 64.0_f32, 128.0_f32,
        3.0_f32, 6.0_f32, 12.0_f32, 24.0_f32,
      ],
      ML::Shape.new(1_i32, 3_i32, 4_i32)
    )
    input = base.transpose
    weight = ML::Tensor.from_array(
      [1.5_f32, -2.0_f32, 0.25_f32],
      ML::Shape.new(3_i32)
    )
    base_before = base.cpu_data.not_nil!.dup
    input_id = input.object_id
    input_strides = input.strides.to_a
    weight_before = weight.to_a

    evaluation = ML::Ops::CPU.rms_norm_with_stats(input, weight, 3_i32, 1e-5_f32)

    base.cpu_data.not_nil!.should eq(base_before)
    input.object_id.should eq(input_id)
    input.strides.to_a.should eq(input_strides)
    weight.to_a.should eq(weight_before)
    evaluation.output.should be_a(ML::Tensor)
    evaluation.output.on_cpu?.should be_true
    evaluation.output.contiguous?.should be_true
    evaluation.output.shape.should eq(ML::Shape.new(1_i32, 4_i32, 3_i32))
    evaluation.read_stats.materialization_count.should eq(1)
    evaluation.read_stats.materialized_bytes.should eq(48_i64)
    evaluation.read_stats.input_materialized_bytes.should eq(48_i64)
    evaluation.read_stats.weight_materialized_bytes.should eq(0_i64)
    evaluation.read_stats.bias_materialized_bytes.should eq(0_i64)
    expected = [
      0.15929827_f32, -3.398363_f32, 0.079649135_f32,
      0.15929827_f32, -3.398363_f32, 0.079649135_f32,
      0.15929827_f32, -3.398363_f32, 0.079649135_f32,
      0.15929827_f32, -3.398363_f32, 0.079649135_f32,
    ]
    evaluation.output.to_a.each_with_index do |value, index|
      value.should be_close(expected[index], 1e-5_f32)
    end

    expect_raises(ArgumentError, /trailing dimension/) do
      ML::Ops::CPU.rms_norm(
        input,
        ML::Tensor.ones(4_i32, device: ML::Tensor::Device::CPU),
        4_i32,
        1e-5_f32
      )
    end
  end

  it "rejects invalid LayerNorm shapes and keeps the functional source free of autograd" do
    input = ML::Tensor.ones(2_i32, 3_i32, 4_i32, device: ML::Tensor::Device::CPU)
    weight = ML::Tensor.ones(12_i32, device: ML::Tensor::Device::CPU)
    bias = ML::Tensor.zeros(12_i32, device: ML::Tensor::Device::CPU)

    multi_axis = ML::Ops::CPU.layer_norm(
      input,
      weight,
      bias,
      [3_i32, 4_i32],
      1e-5_f32
    )
    multi_axis.shape.should eq(input.shape)

    expect_raises(ArgumentError, /trailing shape/) do
      ML::Ops::CPU.layer_norm(input, weight, bias, [2_i32, 6_i32], 1e-5_f32)
    end
    expect_raises(ArgumentError, /weight size/) do
      ML::Ops::CPU.layer_norm(
        input,
        ML::Tensor.ones(11_i32, device: ML::Tensor::Device::CPU),
        bias,
        [3_i32, 4_i32],
        1e-5_f32
      )
    end
    expect_raises(ArgumentError, /bias size/) do
      ML::Ops::CPU.layer_norm(
        input,
        weight,
        ML::Tensor.zeros(11_i32, device: ML::Tensor::Device::CPU),
        [3_i32, 4_i32],
        1e-5_f32
      )
    end
    expect_raises(ArgumentError, /eps must be finite and positive/) do
      ML::Ops::CPU.layer_norm(input, weight, bias, [3_i32, 4_i32], Float32::NAN)
    end

    source = File.read(File.expand_path("../../src/ml/ops/normalization.cr", __DIR__))
    source.should_not match(/Autograd|Variable/)
  end

  it "matches independent finite differences for LayerNorm and RMSNorm gradients" do
    input_values = [1.0_f32, 2.0_f32, 4.0_f32, 3.0_f32, 7.0_f32, 11.0_f32]
    weight_values = [1.5_f32, -2.0_f32, 0.25_f32]
    bias_values = [-0.5_f32, 1.25_f32, 3.0_f32]
    upstream = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32, -0.6_f32]
    epsilon = 1e-5_f32
    step = 1e-3_f32

    layer_loss = ->(x_values : Array(Float32), w_values : Array(Float32), b_values : Array(Float32)) do
      output = ML::Ops::CPU.layer_norm(
        ML::Tensor.from_array(x_values, ML::Shape.new(2_i32, 3_i32)),
        ML::Tensor.from_array(w_values, ML::Shape.new(3_i32)),
        ML::Tensor.from_array(b_values, ML::Shape.new(3_i32)),
        [3_i32],
        epsilon
      ).to_a
      total = 0.0_f32
      output.each_with_index { |value, index| total += value * upstream[index] }
      total
    end

    rms_loss = ->(x_values : Array(Float32), w_values : Array(Float32)) do
      output = ML::Ops::CPU.rms_norm(
        ML::Tensor.from_array(x_values, ML::Shape.new(2_i32, 3_i32)),
        ML::Tensor.from_array(w_values, ML::Shape.new(3_i32)),
        3_i32,
        epsilon
      ).to_a
      total = 0.0_f32
      output.each_with_index { |value, index| total += value * upstream[index] }
      total
    end

    numeric_layer_x = Array(Float32).new(input_values.size) do |index|
      plus = input_values.dup
      minus = input_values.dup
      plus[index] += step
      minus[index] -= step
      (layer_loss.call(plus, weight_values, bias_values) -
        layer_loss.call(minus, weight_values, bias_values)) / (2.0_f32 * step)
    end
    numeric_layer_weight = Array(Float32).new(weight_values.size) do |index|
      plus = weight_values.dup
      minus = weight_values.dup
      plus[index] += step
      minus[index] -= step
      (layer_loss.call(input_values, plus, bias_values) -
        layer_loss.call(input_values, minus, bias_values)) / (2.0_f32 * step)
    end
    numeric_layer_bias = Array(Float32).new(bias_values.size) do |index|
      plus = bias_values.dup
      minus = bias_values.dup
      plus[index] += step
      minus[index] -= step
      (layer_loss.call(input_values, weight_values, plus) -
        layer_loss.call(input_values, weight_values, minus)) / (2.0_f32 * step)
    end

    layer = ML::NN::LayerNorm.new(3, eps: epsilon, device: ML::Tensor::Device::CPU)
    layer.weight.data.cpu_data.not_nil!.replace(weight_values)
    layer.bias.data.cpu_data.not_nil!.replace(bias_values)
    layer_input = ML::Autograd::Variable.new(
      ML::Tensor.from_array(input_values, ML::Shape.new(2_i32, 3_i32)),
      requires_grad: true
    )
    layer.forward(layer_input).backward(
      ML::Tensor.from_array(upstream, ML::Shape.new(2_i32, 3_i32))
    )

    layer_input.grad.not_nil!.to_a.each_with_index do |value, index|
      value.should be_close(numeric_layer_x[index], 2e-3_f32)
    end
    layer.weight.grad.not_nil!.to_a.each_with_index do |value, index|
      value.should be_close(numeric_layer_weight[index], 2e-3_f32)
    end
    layer.bias.grad.not_nil!.to_a.each_with_index do |value, index|
      value.should be_close(numeric_layer_bias[index], 2e-3_f32)
    end

    numeric_rms_x = Array(Float32).new(input_values.size) do |index|
      plus = input_values.dup
      minus = input_values.dup
      plus[index] += step
      minus[index] -= step
      (rms_loss.call(plus, weight_values) -
        rms_loss.call(minus, weight_values)) / (2.0_f32 * step)
    end
    numeric_rms_weight = Array(Float32).new(weight_values.size) do |index|
      plus = weight_values.dup
      minus = weight_values.dup
      plus[index] += step
      minus[index] -= step
      (rms_loss.call(input_values, plus) -
        rms_loss.call(input_values, minus)) / (2.0_f32 * step)
    end

    rms = ML::NN::RMSNorm.new(3, eps: epsilon, device: ML::Tensor::Device::CPU)
    rms.weight.data.cpu_data.not_nil!.replace(weight_values)
    rms_input = ML::Autograd::Variable.new(
      ML::Tensor.from_array(input_values, ML::Shape.new(2_i32, 3_i32)),
      requires_grad: true
    )
    rms.forward(rms_input).backward(
      ML::Tensor.from_array(upstream, ML::Shape.new(2_i32, 3_i32))
    )

    rms_input.grad.not_nil!.to_a.each_with_index do |value, index|
      value.should be_close(numeric_rms_x[index], 2e-3_f32)
    end
    rms.weight.grad.not_nil!.to_a.each_with_index do |value, index|
      value.should be_close(numeric_rms_weight[index], 2e-3_f32)
    end
  end
end
