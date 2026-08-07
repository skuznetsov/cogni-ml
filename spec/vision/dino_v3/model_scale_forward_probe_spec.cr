{% unless flag?(:dinov3_model_scale_forward_probe) %}
  # The model-scale forward path is deliberately absent from the default
  # Crystal build. Compile this spec with -Ddinov3_model_scale_forward_probe
  # only for the bounded manual experiment.
{% else %}
  require "../../../src/ml/vision/dino_v3"
  require "../../spec_helper"

  private def dino_v3_probe_test_embedding_parameters : ML::Vision::DinoV3::EmbeddingParameters
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

  describe ML::Vision::DinoV3::EmbeddingCPU do
    it "admits the pinned 512 hidden-1024 work plan only in the probe build" do
      config = ML::Vision::DinoV3::EmbeddingConfig.new(
        16_i32,
        3_i32,
        1024_i32,
        16_i32,
        1_i32,
        100.0_f32
      )

      plan = ML::Vision::DinoV3::EmbeddingCPU.preflight_model_scale_probe(
        config,
        512_i32,
        max_multiply_adds: 805_306_368_i64
      )

      plan.multiply_adds.should eq(805_306_368_i64)
      plan.output_bytes.should eq(8_929_280_i64)
    end

    it "keeps the elevated budget path explicit and bounded" do
      caller = ML::Vision::DinoV3::DinoV3EmbeddingCaller.new(
        dino_v3_probe_test_embedding_parameters
      )
      input = ML::Tensor.new(
        ML::Shape.new(1_i32, 3_i32, 512_i32, 512_i32),
        device: ML::Tensor::Device::CPU
      )

      result = caller.forward_for_model_scale_probe(
        input,
        max_multiply_adds: ML::Vision::DinoV3::EmbeddingCPU::MAX_MULTIPLY_ADDS + 1_i64
      )

      result.embeddings.shape.to_a.should eq([1_i32, 1026_i32, 4_i32])
      result.embeddings.to_a.all?(&.finite?).should be_true

      expect_raises(
        ML::Vision::DinoV3::EmbeddingBudgetError,
        /1\.\.4294967296/
      ) do
        ML::Vision::DinoV3::EmbeddingCPU.preflight_model_scale_probe(
          dino_v3_probe_test_embedding_parameters.config,
          512_i32,
          max_multiply_adds: ML::Vision::DinoV3::EmbeddingCPU::MODEL_SCALE_PROBE_MAX_MULTIPLY_ADDS + 1_i64
        )
      end
    end
  end
{% end %}
