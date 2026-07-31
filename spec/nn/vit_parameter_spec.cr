require "../spec_helper"

describe ML::NN::PatchEmbedding do
  it "rejects non-positive geometry before division or allocation" do
    expect_raises(ArgumentError, /img_size must be positive/) do
      ML::NN::PatchEmbedding.new(img_size: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /patch_size must be positive/) do
      ML::NN::PatchEmbedding.new(patch_size: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /in_channels must be positive/) do
      ML::NN::PatchEmbedding.new(in_channels: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /embed_dim must be positive/) do
      ML::NN::PatchEmbedding.new(embed_dim: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /num_patches overflow/) do
      ML::NN::PatchEmbedding.new(img_size: Int32::MAX, patch_size: 1, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /patch_dim overflow/) do
      ML::NN::PatchEmbedding.new(
        img_size: Int32::MAX,
        patch_size: Int32::MAX,
        in_channels: Int32::MAX,
        embed_dim: 1,
        device: ML::Tensor::Device::CPU
      )
    end
  end

  it "rejects invalid NCHW image geometry before projection" do
    patch_embed = ML::NN::PatchEmbedding.new(
      img_size: 2,
      patch_size: 2,
      in_channels: 1,
      embed_dim: 1,
      device: ML::Tensor::Device::CPU
    )

    expect_raises(ArgumentError, /rank 4/) do
      patch_embed.forward(
        ML::Autograd::Variable.ones(
          1,
          2,
          2,
          requires_grad: false,
          device: ML::Tensor::Device::CPU
        )
      )
    end
    expect_raises(ArgumentError, /channel/) do
      patch_embed.forward(
        ML::Autograd::Variable.ones(
          1,
          2,
          2,
          2,
          requires_grad: false,
          device: ML::Tensor::Device::CPU
        )
      )
    end
    expect_raises(ArgumentError, /spatial/) do
      patch_embed.forward(
        ML::Autograd::Variable.ones(
          1,
          1,
          3,
          2,
          requires_grad: false,
          device: ML::Tensor::Device::CPU
        )
      )
    end
  end

  it "accepts positive divisible standalone image geometry" do
    patch_embed = ML::NN::PatchEmbedding.new(
      img_size: 2,
      patch_size: 1,
      in_channels: 1,
      embed_dim: 1,
      device: ML::Tensor::Device::CPU
    )

    output = patch_embed.forward(
      ML::Autograd::Variable.ones(
        1,
        1,
        4,
        1,
        requires_grad: false,
        device: ML::Tensor::Device::CPU
      )
    )

    output.shape.should eq(ML::Shape.new(1_i32, 4_i32, 1_i32))
  end

  it "preserves logical patch order and gradients for a transposed square image" do
    patch_embed = ML::NN::PatchEmbedding.new(
      img_size: 2,
      patch_size: 1,
      in_channels: 1,
      embed_dim: 1,
      device: ML::Tensor::Device::CPU
    )
    patch_embed.proj.weight.data.cpu_data.not_nil!.replace([1.0_f32])
    patch_embed.proj.bias.not_nil!.data.cpu_data.not_nil!.replace([0.0_f32])
    patch_embed.parameters.each { |parameter| parameter.requires_grad = false }

    view_base = ML::Autograd::Variable.new(
      ML::Tensor.from_array(
        [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        ML::Shape.new(1_i32, 1_i32, 2_i32, 2_i32)
      ),
      requires_grad: true
    )
    view_input = view_base.transpose
    dense_input = ML::Autograd::Variable.new(
      view_input.data.contiguous,
      requires_grad: true
    )
    upstream = ML::Tensor.from_array(
      [0.5_f32, -1.0_f32, 1.5_f32, -2.0_f32],
      ML::Shape.new(1_i32, 4_i32, 1_i32)
    )
    base_before = view_base.data.to_a
    view_strides = view_input.data.strides.to_a

    view_output = patch_embed.forward(view_input)
    dense_output = patch_embed.forward(dense_input)

    view_output.data.to_a.should eq([1.0_f32, 3.0_f32, 2.0_f32, 4.0_f32])
    view_output.data.to_a.should eq(dense_output.data.to_a)
    view_base.data.to_a.should eq(base_before)
    view_input.data.strides.to_a.should eq(view_strides)

    view_output.backward(upstream)
    dense_output.backward(upstream)

    view_base.grad.not_nil!.transpose.to_a.should eq(dense_input.grad.not_nil!.to_a)
    view_base.grad.not_nil!.transpose.to_a.should eq(upstream.to_a)
  end
end

describe ML::NN::MLP do
  it "rejects invalid dimensions and dropout before layer allocation" do
    expect_raises(ArgumentError, /in_features must be positive/) do
      ML::NN::MLP.new(0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /hidden_features must be positive/) do
      ML::NN::MLP.new(4, hidden_features: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /out_features must be positive/) do
      ML::NN::MLP.new(4, out_features: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /dropout must be finite and in/) do
      ML::NN::MLP.new(4, dropout: Float32::NAN, device: ML::Tensor::Device::CPU)
    end
  end
end

describe ML::NN::TransformerEncoderBlock do
  it "rejects invalid block parameters before sublayer allocation" do
    expect_raises(ArgumentError, /mlp_ratio must be finite and positive/) do
      ML::NN::TransformerEncoderBlock.new(
        4,
        2,
        mlp_ratio: Float32::NAN,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /MLP hidden dimension must be positive/) do
      ML::NN::TransformerEncoderBlock.new(
        4,
        2,
        mlp_ratio: 0.01_f32,
        device: ML::Tensor::Device::CPU
      )
    end
  end

  it "preserves block outputs and gradients for a transposed token view" do
    block = ML::NN::TransformerEncoderBlock.new(
      embed_dim: 2,
      num_heads: 1,
      mlp_ratio: 1.0_f32,
      device: ML::Tensor::Device::CPU
    )
    block.parameters.each_with_index do |parameter, parameter_index|
      data = parameter.data.cpu_data.not_nil!
      data.each_index do |index|
        data[index] = (((parameter_index + 1) * 3 + index) % 11 - 5).to_f32 * 0.04_f32
      end
      parameter.requires_grad = false
    end

    view_base = ML::Autograd::Variable.new(
      ML::Tensor.from_array(
        [0.2_f32, -0.7_f32, 1.1_f32, 0.3_f32, 0.5_f32, -0.4_f32],
        ML::Shape.new(1_i32, 2_i32, 3_i32)
      ),
      requires_grad: true
    )
    view_input = view_base.transpose
    dense_input = ML::Autograd::Variable.new(
      view_input.data.contiguous,
      requires_grad: true
    )
    upstream = ML::Tensor.from_array(
      [0.3_f32, -0.6_f32, 0.9_f32, 0.2_f32, -0.4_f32, 0.7_f32],
      ML::Shape.new(1_i32, 3_i32, 2_i32)
    )
    base_before = view_base.data.to_a
    view_strides = view_input.data.strides.to_a

    view_output = block.forward(view_input)
    dense_output = block.forward(dense_input)

    view_output.data.to_a.each_with_index do |actual, index|
      actual.should be_close(dense_output.data.to_a[index], 1e-5_f32)
    end
    view_base.data.to_a.should eq(base_before)
    view_input.data.strides.to_a.should eq(view_strides)

    view_output.backward(upstream)
    dense_output.backward(upstream)

    view_gradient = view_base.grad.not_nil!.transpose.to_a
    dense_gradient = dense_input.grad.not_nil!.to_a
    view_gradient.each_with_index do |actual, index|
      actual.should be_close(dense_gradient[index], 1e-5_f32)
    end
  end
end

describe ML::NN::ViTEncoder do
  it "rejects invalid encoder topology before model allocation" do
    expect_raises(ArgumentError, /depth must be non-negative/) do
      ML::NN::ViTEncoder.new(depth: -1, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /num_heads must be positive/) do
      ML::NN::ViTEncoder.new(depth: 0, num_heads: 0, device: ML::Tensor::Device::CPU)
    end
    expect_raises(ArgumentError, /mlp_ratio must be finite and positive/) do
      ML::NN::ViTEncoder.new(
        img_size: 4,
        patch_size: 2,
        embed_dim: 4,
        depth: 0,
        num_heads: 2,
        mlp_ratio: Float32::NAN,
        device: ML::Tensor::Device::CPU
      )
    end
  end

  it "rejects a divisible but unconfigured image size before patch projection" do
    encoder = ML::NN::ViTEncoder.new(
      img_size: 2,
      patch_size: 1,
      in_channels: 1,
      embed_dim: 2,
      depth: 0,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )

    expect_raises(ArgumentError, /spatial/) do
      encoder.forward(
        ML::Autograd::Variable.ones(
          1,
          1,
          4,
          1,
          requires_grad: false,
          device: ML::Tensor::Device::CPU
        )
      )
    end
  end

  it "matches a dense reference for a transposed image through the full CPU encoder" do
    encoder = ML::NN::ViTEncoder.new(
      img_size: 2,
      patch_size: 1,
      in_channels: 1,
      embed_dim: 3,
      depth: 1,
      num_heads: 1,
      mlp_ratio: 1.0_f32,
      device: ML::Tensor::Device::CPU
    )
    encoder.parameters.each_with_index do |parameter, parameter_index|
      data = parameter.data.cpu_data.not_nil!
      data.each_index do |index|
        data[index] = (((parameter_index + 2) * 5 + index) % 17 - 8).to_f32 * 0.025_f32
      end
      parameter.requires_grad = false
    end

    view_base = ML::Autograd::Variable.new(
      ML::Tensor.from_array(
        [0.4_f32, -0.8_f32, 1.2_f32, 0.1_f32],
        ML::Shape.new(1_i32, 1_i32, 2_i32, 2_i32)
      ),
      requires_grad: true
    )
    view_input = view_base.transpose
    dense_input = ML::Autograd::Variable.new(
      view_input.data.contiguous,
      requires_grad: true
    )
    upstream = ML::Tensor.from_array(
      [
        0.1_f32, -0.2_f32, 0.3_f32,
        0.4_f32, 0.5_f32, -0.6_f32,
        0.7_f32, -0.8_f32, 0.9_f32,
        -1.0_f32, 1.1_f32, -1.2_f32,
        1.3_f32, -1.4_f32, 1.5_f32,
      ],
      ML::Shape.new(1_i32, 5_i32, 3_i32)
    )

    view_output = encoder.forward(view_input)
    dense_output = encoder.forward(dense_input)

    view_output.shape.should eq(ML::Shape.new(1_i32, 5_i32, 3_i32))
    view_output.data.to_a.each_with_index do |actual, index|
      actual.should be_close(dense_output.data.to_a[index], 1e-5_f32)
    end

    view_output.backward(upstream)
    dense_output.backward(upstream)

    view_gradient = view_base.grad.not_nil!.transpose.to_a
    dense_gradient = dense_input.grad.not_nil!.to_a
    view_gradient.each_with_index do |actual, index|
      actual.should be_close(dense_gradient[index], 1e-5_f32)
    end
  end
end
