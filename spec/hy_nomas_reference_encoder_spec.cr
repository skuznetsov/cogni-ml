require "spec"
require "../examples/support/hy_nomas_reference_encoder"

describe HyNomasReferenceEncoder do
  it "admits only the fixed reference geometry and counts every encoder-phase owner" do
    plan = HyNomasReferenceEncoder::ResidencyPlan.for(HyNomasReferenceEncoder::FIXED_CFG)

    plan.weight_bytes.should eq(1_833_310_016_i64)
    plan.workspace_bytes.should eq(2_714_664_960_i64)
    plan.clip_bytes.should eq(106_168_320_i64)
    plan.rgb_bytes.should eq(442_368_i64)
    plan.latent_bytes.should eq(276_480_i64)
    plan.named_bytes.should eq(4_654_862_144_i64)
    HyNomasReferenceEncoder.validate_budget!(plan, plan.named_bytes).should be_nil
    expect_raises(ArgumentError, /exceed budget/) do
      HyNomasReferenceEncoder.validate_budget!(plan, plan.named_bytes - 1)
    end
    expect_raises(ArgumentError, /fixed 512x288x120/) do
      HyNomasReferenceEncoder::ResidencyPlan.for(ML::Video::DCAEVEncoderConfig.new(32, 32, 4))
    end
  end

  it "requires all four reference identity flags as one opt-in set" do
    validate = ->(image_path : String, image_sha : String, encoder_path : String, encoder_sha : String, allow : Bool, plan_only : Bool) {
      HyNomasReferenceEncoder.validate_cli!(image_path, image_sha, encoder_path, encoder_sha,
        allow_unqualified: allow, plan_only: plan_only,
        cinematic_requested: false, legacy_text: false)
    }

    validate.call("", "", "", "", false, false).should be_false
    validate.call("manifest.json", "a" * 64, "encoder.safetensors", "b" * 64,
      true, false).should be_true
    validate.call("manifest.json", "a" * 64, "encoder.safetensors", "b" * 64,
      false, true).should be_true
    expect_raises(ArgumentError, /supplied together/) { validate.call("manifest.json", "", "encoder.safetensors", "b" * 64, false, false) }
    expect_raises(ArgumentError, /unqualified/) { validate.call("manifest.json", "a" * 64, "encoder.safetensors", "b" * 64, false, false) }
    expect_raises(ArgumentError, /requires all reference/) { validate.call("", "", "", "", true, false) }
  end

  it "rejects reference initialization with Cinematic or legacy text caching" do
    args = {"manifest.json", "a" * 64, "encoder.safetensors", "b" * 64}
    expect_raises(ArgumentError, /Cinematic/) do
      HyNomasReferenceEncoder.validate_cli!(*args,
        allow_unqualified: true, plan_only: false, cinematic_requested: true, legacy_text: false)
    end
    expect_raises(ArgumentError, /once-per-scene/) do
      HyNomasReferenceEncoder.validate_cli!(*args,
        allow_unqualified: true, plan_only: false, cinematic_requested: false, legacy_text: true)
    end
  end

  it "validates DiT encode metrics independently of prior encoder phase metrics" do
    metrics = {
      "reference_encoder_segments" => 2_i64,
      "reference_encoder_waits"    => 1_i64,
      "dit_encode_allocations"     => 0_i64,
      "dit_encode_readbacks"       => 0_i64,
      "dit_preterminal_waits"      => 0_i64,
    }

    HyNomasReferenceEncoder.validate_dit_encode_metrics!(metrics).should be_nil

    incomplete = metrics.dup
    incomplete.delete("dit_preterminal_waits")
    expect_raises(ArgumentError, /Missing DiT encode metric/) do
      HyNomasReferenceEncoder.validate_dit_encode_metrics!(incomplete)
    end

    {"dit_encode_allocations", "dit_encode_readbacks", "dit_preterminal_waits"}.each do |key|
      metrics[key] = 1_i64
      expect_raises(ArgumentError, /DiT encode introduced/) do
        HyNomasReferenceEncoder.validate_dit_encode_metrics!(metrics)
      end
      metrics[key] = 0_i64
    end
  end

  it "labels T0 separately and omits a cache writer after the final generated segment" do
    two = HyNomasReferenceEncoder::ReferenceHistoryPlan.for(2)
    two.generated_segment_indices.should eq([1])
    two.generated_cache_writer_segment_indices.should eq([] of Int32)
    two.final_generated_segment_index.should eq(1)
    two.generated_duration_seconds.should eq(5)
    two.total_timeline_duration_seconds.should eq(10)

    four = HyNomasReferenceEncoder::ReferenceHistoryPlan.for(4)
    four.generated_segment_indices.should eq([1, 2, 3])
    four.generated_cache_writer_segment_indices.should eq([1, 2])
    four.generated_duration_seconds.should eq(15)
    four.total_timeline_duration_seconds.should eq(20)

    six = HyNomasReferenceEncoder::ReferenceHistoryPlan.for(6)
    six.generated_segment_indices.should eq([1, 2, 3, 4, 5])
    six.generated_cache_writer_segment_indices.should eq([1, 2, 3, 4])
    six.final_generated_segment_index.should eq(5)
    six.generated_duration_seconds.should eq(25)
    six.total_timeline_duration_seconds.should eq(30)
  end
end
