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
end
