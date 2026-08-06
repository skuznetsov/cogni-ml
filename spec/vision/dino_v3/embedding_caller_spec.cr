require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_test_embedding_parameters : ML::Vision::DinoV3::EmbeddingParameters
  config = ML::Vision::DinoV3::EmbeddingConfig.new(
    16_i32,
    3_i32,
    4_i32,
    1_i32,
    1_i32,
    100.0_f32
  )
  ML::Vision::DinoV3::EmbeddingParameters.new(
    config,
    Array(Float32).new(4 * 3 * 16 * 16) { |index| (index % 7).to_f32 / 32.0_f32 },
    Array(Float32).new(4) { |index| index.to_f32 / 16.0_f32 },
    Array(Float32).new(4) { |index| (index + 1).to_f32 / 16.0_f32 },
    Array(Float32).new(4) { |index| (index + 2).to_f32 / 16.0_f32 }
  )
end

describe ML::Vision::DinoV3::DinoV3EmbeddingCaller do
  it "forwards a bounded parameter set through the existing CPU embedding boundary" do
    caller = ML::Vision::DinoV3::DinoV3EmbeddingCaller.new(
      dino_v3_test_embedding_parameters
    )
    input = ML::Tensor.new(
      ML::Shape.new(1_i32, 3_i32, 512_i32, 512_i32),
      device: ML::Tensor::Device::CPU
    )

    result = caller.forward(input)

    result.patches.shape.to_a.should eq([1_i32, 1024_i32, 4_i32])
    result.embeddings.shape.to_a.should eq([1_i32, 1026_i32, 4_i32])
    result.patch_coordinates.shape.to_a.should eq([1024_i32, 2_i32])
    result.rope_cos.shape.to_a.should eq([1024_i32, 4_i32])
    result.rope_sin.shape.to_a.should eq([1024_i32, 4_i32])
    result.parameter_f32le_sha256.should_not be_empty
    result.embeddings.to_a.all?(&.finite?).should be_true
  end

  it "rejects aggregate parameter allocation above the caller budget" do
    expect_raises(
      ML::Vision::DinoV3::EmbeddingCallerError,
      /parameter byte budget/
    ) do
      ML::Vision::DinoV3::DinoV3EmbeddingCaller.new(
        dino_v3_test_embedding_parameters,
        max_parameter_bytes: 1024_i64
      )
    end
  end

  it "publishes only the four roles needed by the embedding boundary" do
    ML::Vision::DinoV3::DinoV3EmbeddingCaller::REQUIRED_ROLES.to_a.should eq([
      "embedding.patch_weight",
      "embedding.patch_bias",
      "embedding.cls_token",
      "embedding.register_tokens",
    ])
  end
end
