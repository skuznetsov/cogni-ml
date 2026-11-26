require "spec"
require "../examples/support/hy_nomas_sequence"

describe "reference T1 step capture admission" do
  it "admits only the unmasked source-BF16 original-photo reference run" do
    HyNomasSequence.validate_reference_t1_step_capture!(true,
      source_bf16: true,
      reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
      segments: 2,
      generated_t0_control: false,
      t0_key_visible_latents: nil,
      t0_hide_caption_keys: false,
      t1_current_text_kv: false)

    HyNomasSequence.validate_reference_t1_step_capture!(false,
      source_bf16: false, reference_latent_bytes: 0_i64, segments: 3)
  end

  it "rejects generated, masked, edited, incomplete, and non-BF16 contexts" do
    base = {
      source_bf16:            true,
      reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
      segments:               2,
      generated_t0_control:   false,
      t0_key_visible_latents: nil,
      t0_hide_caption_keys:   false,
      t1_current_text_kv:     false,
    }

    invalid = [
      {source_bf16: false, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
       segments: 2, generated_t0_control: false, t0_key_visible_latents: nil,
       t0_hide_caption_keys: false, t1_current_text_kv: false},
      {source_bf16: true, reference_latent_bytes: 0_i64,
       segments: 2, generated_t0_control: true, t0_key_visible_latents: nil,
       t0_hide_caption_keys: false, t1_current_text_kv: false},
      {source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
       segments: 1, generated_t0_control: false, t0_key_visible_latents: nil,
       t0_hide_caption_keys: false, t1_current_text_kv: false},
      {source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
       segments: 2, generated_t0_control: false, t0_key_visible_latents: 30,
       t0_hide_caption_keys: false, t1_current_text_kv: false},
      {source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
       segments: 2, generated_t0_control: false, t0_key_visible_latents: 30,
       t0_hide_caption_keys: true, t1_current_text_kv: false},
      {source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
       segments: 2, generated_t0_control: false, t0_key_visible_latents: 30,
       t0_hide_caption_keys: true, t1_current_text_kv: true},
    ]
    invalid.each do |arguments|
      expect_raises(ArgumentError, /reference T1 step capture/) do
        HyNomasSequence.validate_reference_t1_step_capture!(true, **arguments)
      end
    end
  end

  it "accounts one BF16-expanded F32 copy of each of three values per step" do
    HyNomasSequence::REFERENCE_T1_STEP_CAPTURE_BUFFER_BYTES.should eq(552_960_i64)
    HyNomasSequence::REFERENCE_T1_STEP_CAPTURE_BYTES.should eq(6_635_520_i64)
  end

  it "uses the same four source-BF16 Euler sigmas and deltas as T1" do
    HyNomasSequence.reference_t1_step_capture_schedule.should eq([
      {1.0_f32, -0.0625_f32},
      {0.9375_f32, -0.1041666865_f32},
      {0.8333333135_f32, -0.2083333135_f32},
      {0.625_f32, -0.625_f32},
    ])
  end

  it "uses the source-BF16 effective velocity in the x0 estimate" do
    ML::Video::HyNomasReferenceT1StepCapturePlan.bf16_expanded_f32(1.00390625_f32).should eq(1.0_f32)
    ML::Video::HyNomasReferenceT1StepCapturePlan.x0_hat_bf16_expanded_f32(
      1.5_f32, 0.25_f32, 0.5_f32).should eq(1.375_f32)
  end
end
