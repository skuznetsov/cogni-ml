require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_quality_metrics"

describe ML::GGUF::QwenQBitQualityMetrics do
  it "distinguishes ranked top-2 parity from unordered top-2 coverage" do
    exact = ML::GGUF::QwenQBitQualityMetrics::Top2.new(10, 8.0_f32, 20, 7.0_f32)
    swapped = ML::GGUF::QwenQBitQualityMetrics::Top2.new(20, 7.1_f32, 10, 6.9_f32)

    comparison = ML::GGUF::QwenQBitQualityMetrics.compare_top2(exact, swapped)

    comparison.ranked_matches.should eq(0)
    comparison.set_overlap.should eq(2)
    comparison.exact_top1_covered.should be_true
    comparison.exact_top2_covered.should be_true
    comparison.margin_delta.should be_close(0.8_f32, 1.0e-6_f32)
  end

  it "uses stable token-id tie breaking when extracting top-2 logits" do
    top2 = ML::GGUF::QwenQBitQualityMetrics.top2([
      1.0_f32,
      3.0_f32,
      3.0_f32,
      2.0_f32,
    ])

    top2.first_id.should eq(1)
    top2.first_logit.should eq(3.0_f32)
    top2.second_id.should eq(2)
    top2.second_logit.should eq(3.0_f32)
  end

  it "rejects logits that cannot define a finite top-2" do
    expect_raises(ArgumentError, /at least two/) do
      ML::GGUF::QwenQBitQualityMetrics.top2([1.0_f32])
    end
    expect_raises(ArgumentError, /must be finite/) do
      ML::GGUF::QwenQBitQualityMetrics.top2([1.0_f32, Float32::NAN])
    end
  end

  it "computes embedding cosine similarity and rejects invalid instruments" do
    identity_ecs = ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(
      [1.0_f32, 2.0_f32, 3.0_f32],
      [2.0_f32, 4.0_f32, 6.0_f32],
    )
    orthogonal_ecs = ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(
      [1.0_f32, 0.0_f32],
      [0.0_f32, 1.0_f32],
    )
    opposite_ecs = ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(
      [1.0_f32, 0.0_f32],
      [-1.0_f32, 0.0_f32],
    )

    identity_ecs.should be_close(1.0_f64, 1.0e-12_f64)
    orthogonal_ecs.should be_close(0.0_f64, 1.0e-12_f64)
    opposite_ecs.should be_close(-1.0_f64, 1.0e-12_f64)

    expect_raises(ArgumentError, /same non-zero dimension/) do
      ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(
        [1.0_f32],
        [1.0_f32, 2.0_f32],
      )
    end
    expect_raises(ArgumentError, /finite non-zero vectors/) do
      ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(
        [0.0_f32, 0.0_f32],
        [1.0_f32, 0.0_f32],
      )
    end
  end
end
