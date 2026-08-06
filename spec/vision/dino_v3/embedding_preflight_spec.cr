require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_preflight_certificate : ML::Vision::DinoV3::ConfigCertificate
  source = File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json")
  )
  ML::Vision::DinoV3::ConfigCertificate.parse(
    source,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

private def dino_v3_preflight_config : ML::Vision::DinoV3::EmbeddingConfig
  ML::Vision::DinoV3::EmbeddingConfig.new(
    16_i32,
    3_i32,
    4_i32,
    1_i32,
    1_i32,
    100.0_f32
  )
end

describe ML::Vision::DinoV3::EmbeddingCPU do
  it "builds an allocation-free geometry and work plan" do
    plan = ML::Vision::DinoV3::EmbeddingCPU.preflight(
      dino_v3_preflight_config,
      512_i32
    )

    plan.input_edge.should eq(512_i32)
    plan.patch_edge.should eq(32_i32)
    plan.patch_count.should eq(1024_i64)
    plan.input_bytes.should eq(3_145_728_i64)
    plan.output_elements.should eq(18_440_i64)
    plan.output_bytes.should eq(73_760_i64)
    plan.multiply_adds.should eq(3_145_728_i64)
    plan.max_multiply_adds.should eq(67_108_864_i64)
  end

  it "accepts an exact operation budget and rejects one below it" do
    config = dino_v3_preflight_config
    plan = ML::Vision::DinoV3::EmbeddingCPU.preflight(
      config,
      512_i32,
      max_multiply_adds: 3_145_728_i64
    )
    plan.multiply_adds.should eq(3_145_728_i64)

    expect_raises(ML::Vision::DinoV3::EmbeddingBudgetError, /multiply-add/) do
      ML::Vision::DinoV3::EmbeddingCPU.preflight(
        config,
        512_i32,
        max_multiply_adds: 3_145_727_i64
      )
    end
  end

  it "rejects the pinned real 512 geometry before payload or tensor work" do
    certificate = dino_v3_preflight_certificate
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)

    expect_raises(ML::Vision::DinoV3::EmbeddingBudgetError, /multiply-add/) do
      ML::Vision::DinoV3::DinoV3EmbeddingCaller.preflight(
        certificate,
        runtime,
        512_i32
      )
    end
  end

  it "rejects an unsupported input edge before computing a plan" do
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /resolution/) do
      ML::Vision::DinoV3::EmbeddingCPU.preflight(
        dino_v3_preflight_config,
        256_i32
      )
    end
  end
end
