require "spec"
require "../examples/support/hy_nomas_sequence"

describe ML::Video::HyNomasReferenceAction do
  it "binds exactly two segments to one authenticated action condition" do
    action = ML::Video::HyNomasReferenceAction.new(run_segments: 2,
      conditioning_manifest_sha256: "a" * 64,
      conditioning_payload_sha256: "b" * 64,
      prompt_sha256: "c" * 64,
      valid_text_tokens: 369)

    action.run_segments.should eq(2)
    action.identity.should eq({
      schema:                       "cogni_ml.hy_nomas_reference_action/v1",
      run_segments:                 2,
      conditioning_manifest_sha256: "a" * 64,
      conditioning_payload_sha256:  "b" * 64,
      prompt_sha256:                "c" * 64,
      valid_text_tokens:            369,
      cache_text:                   [true, true],
    })
    action.text_tokens(23).should eq([23, 369])
    action.cache_text.should eq([true, true])
    action.query_text_tokens(23).should eq(369)
    action.query_text_tokens(512).should eq(512)
  end

  it "rejects widened geometry, invalid token counts, and malformed identities" do
    invalid = [
      {3, 23, "a" * 64, "b" * 64, "c" * 64},
      {2, 0, "a" * 64, "b" * 64, "c" * 64},
      {2, 1001, "a" * 64, "b" * 64, "c" * 64},
      {2, 23, "A" * 64, "b" * 64, "c" * 64},
      {2, 23, "a" * 64, "bad", "c" * 64},
      {2, 23, "a" * 64, "b" * 64, "c" * 63},
    ]
    invalid.each do |segments, tokens, manifest_sha, payload_sha, prompt_sha|
      expect_raises(ArgumentError) do
        ML::Video::HyNomasReferenceAction.new(run_segments: segments,
          conditioning_manifest_sha256: manifest_sha,
          conditioning_payload_sha256: payload_sha,
          prompt_sha256: prompt_sha,
          valid_text_tokens: tokens)
      end
    end
  end
end

describe "Hy-Nomas reference-action scalar admission" do
  action = ML::Video::HyNomasReferenceAction.new(run_segments: 2,
    conditioning_manifest_sha256: "a" * 64,
    conditioning_payload_sha256: "b" * 64,
    prompt_sha256: Digest::SHA256.hexdigest(%({"segments":[{}]})),
    valid_text_tokens: 369)

  it "uses the maximum valid prompt extent while caching each segment's own rows" do
    tokens = action.text_tokens(23)
    profile = ML::Video::HyNomasContinuationHistory.geometry(2, 4320,
      action.query_text_tokens(23), 384, 54, 16, 128,
      text_cache_policy: HyNomasSequence::HISTORY_TEXT_POLICY,
      segment_text_tokens: tokens,
      segment_cache_text: action.cache_text)

    profile.query_rows.should eq(4320 + 369)
    profile.storage_rows.should eq(4320 + 384)
    profile.segment_text_tokens.should eq([23, 369])
    profile.segment_cache_text.should eq([true, true])
    profile.workspace_geometries[0].current_rows.should eq(4320 + 23)
    profile.writer_workspace_geometries[0].current_rows.should eq(4320 + 23)
    profile.workspace_geometries[1].current_rows.should eq(4320 + 369)
    profile.append_elements(0).should eq((4320 + 23) * 16 * 128)
  end

  it "allows the caption extent to exceed the action extent without changing the action tail" do
    tokens = action.text_tokens(512)
    action.query_text_tokens(512).should eq(512)
    profile = ML::Video::HyNomasContinuationHistory.geometry(2, 4320,
      action.query_text_tokens(512), 512, 54, 16, 128,
      text_cache_policy: HyNomasSequence::HISTORY_TEXT_POLICY,
      segment_text_tokens: tokens,
      segment_cache_text: action.cache_text)
    profile.query_rows.should eq(4320 + 512)
    profile.workspace_geometries[0].current_rows.should eq(4320 + 512)
    profile.workspace_geometries[1].current_rows.should eq(4320 + 369)
  end

  it "adds the second Qwen input and refiner residency to its pure estimate" do
    mask = Array.new(1280) { |i| i < action.valid_text_tokens ? 1 : 0 }
    expected = 1280_i64 * 3584 * sizeof(UInt16) +
               ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true).total
    HyNomasSequence.reference_action_bytes(action).should eq(expected)
    HyNomasSequence.reference_action_bytes(nil).should eq(0_i64)
  end

  it "rejects incompatible reference-action admission modes" do
    cinematic = HyNomasCinematic::IndexIdentity.new(
      "/opt/mldata_ssd/cinematic/index.json", "d" * 64,
      "/opt/mldata_ssd/cinematic/cache.json", "e" * 64)
    noise = ML::Video::HyNomasSegmentNoiseEdit.new(2, 1, 2)
    prompt_edit = ML::Video::HyNomasSegmentPromptEdit.new(2, 1,
      "f" * 64, "a" * 64, "b" * 64, 369)

    valid = {
      segments:               2,
      reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
      text_cache_policy:      HyNomasSequence::HISTORY_TEXT_POLICY,
    }
    HyNomasSequence.validate_reference_action_admission!(action, **valid)

    expect_raises(ArgumentError, /two segments/) do
      HyNomasSequence.validate_reference_action_admission!(action,
        segments: 3, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        text_cache_policy: HyNomasSequence::HISTORY_TEXT_POLICY)
    end
    expect_raises(ArgumentError, /reference latent/) do
      HyNomasSequence.validate_reference_action_admission!(action,
        segments: 2, reference_latent_bytes: 0_i64,
        text_cache_policy: HyNomasSequence::HISTORY_TEXT_POLICY)
    end
    expect_raises(ArgumentError, /OncePerScene/) do
      HyNomasSequence.validate_reference_action_admission!(action,
        segments: 2, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::PerChunk)
    end
    expect_raises(ArgumentError, /Cinematic|capture|edit/) do
      HyNomasSequence.validate_reference_action_admission!(action, **valid, cinematic: cinematic)
    end
    expect_raises(ArgumentError, /Cinematic|capture|edit/) do
      HyNomasSequence.validate_reference_action_admission!(action, **valid, prepare_capture: true)
    end
    expect_raises(ArgumentError, /Cinematic|capture|edit/) do
      HyNomasSequence.validate_reference_action_admission!(action, **valid, noise_edit: noise)
    end
    expect_raises(ArgumentError, /Cinematic|capture|edit/) do
      HyNomasSequence.validate_reference_action_admission!(action, **valid, prompt_edit: prompt_edit)
    end
  end

  it "binds the action descriptor to the authenticated v2 default-geometry prompt" do
    prompt = %({"segments":[{}]})
    sha = Digest::SHA256.hexdigest(prompt)
    bound = ML::Video::HyNomasReferenceAction.new(run_segments: 2,
      conditioning_manifest_sha256: "a" * 64,
      conditioning_payload_sha256: "b" * 64,
      prompt_sha256: sha,
      valid_text_tokens: 369)

    HyNomasSequence.validate_reference_action_identity!(bound,
      conditioning_manifest_sha256: "a" * 64,
      conditioning_payload_sha256: "b" * 64,
      conditioning_prompt: prompt,
      conditioning_valid_text_tokens: 369,
      conditioning_geometry: HyNomasGeneration::SpatialGeometry::DEFAULT,
      conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v2",
      conditioning_schema_version: 2)

    expect_raises(ArgumentError, /identity/) do
      HyNomasSequence.validate_reference_action_identity!(bound,
        conditioning_manifest_sha256: "x" * 64,
        conditioning_payload_sha256: "b" * 64,
        conditioning_prompt: prompt,
        conditioning_valid_text_tokens: 369,
        conditioning_geometry: HyNomasGeneration::SpatialGeometry::DEFAULT,
        conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v2",
        conditioning_schema_version: 2)
    end
    expect_raises(ArgumentError, /identity/) do
      HyNomasSequence.validate_reference_action_identity!(bound,
        conditioning_manifest_sha256: "a" * 64,
        conditioning_payload_sha256: "x" * 64,
        conditioning_prompt: prompt,
        conditioning_valid_text_tokens: 369,
        conditioning_geometry: HyNomasGeneration::SpatialGeometry::DEFAULT,
        conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v2",
        conditioning_schema_version: 2)
    end
    expect_raises(ArgumentError, /identity/) do
      HyNomasSequence.validate_reference_action_identity!(bound,
        conditioning_manifest_sha256: "a" * 64,
        conditioning_payload_sha256: "b" * 64,
        conditioning_prompt: %({"segments":[{"caption":"different"}]}),
        conditioning_valid_text_tokens: 369,
        conditioning_geometry: HyNomasGeneration::SpatialGeometry::DEFAULT,
        conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v2",
        conditioning_schema_version: 2)
    end
    expect_raises(ArgumentError, /identity/) do
      HyNomasSequence.validate_reference_action_identity!(bound,
        conditioning_manifest_sha256: "a" * 64,
        conditioning_payload_sha256: "b" * 64,
        conditioning_prompt: prompt,
        conditioning_valid_text_tokens: 368,
        conditioning_geometry: HyNomasGeneration::SpatialGeometry::DEFAULT,
        conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v2",
        conditioning_schema_version: 2)
    end
    expect_raises(ArgumentError, /identity/) do
      HyNomasSequence.validate_reference_action_identity!(bound,
        conditioning_manifest_sha256: "a" * 64,
        conditioning_payload_sha256: "b" * 64,
        conditioning_prompt: prompt,
        conditioning_valid_text_tokens: 369,
        conditioning_geometry: HyNomasGeneration::SpatialGeometry::CANDIDATE_768,
        conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v3",
        conditioning_schema_version: 3)
    end
    expect_raises(ArgumentError, /action tails/) do
      HyNomasSequence.validate_reference_action_identity!(bound,
        conditioning_manifest_sha256: "a" * 64,
        conditioning_payload_sha256: "b" * 64,
        conditioning_prompt: %({"segments":[{"contained_actions":[{}]}]}),
        conditioning_valid_text_tokens: 369,
        conditioning_geometry: HyNomasGeneration::SpatialGeometry::DEFAULT,
        conditioning_schema: "cogni_ml.hy_nomas_generation_inputs/v2",
        conditioning_schema_version: 2)
    end
  end

  it "requires the descriptor and separate conditioning object as a pair" do
    expect_raises(ArgumentError, /separate authenticated conditioning/) do
      HyNomasSequence.validate_reference_action!(action, nil,
        segments: 2,
        reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        text_cache_policy: HyNomasSequence::HISTORY_TEXT_POLICY)
    end
  end
end
