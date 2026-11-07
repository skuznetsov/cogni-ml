require "spec"
require "../examples/support/hy_nomas_sequence"
require "../examples/support/hy_nomas_decode_latent_admission"

describe "sequence decoder cache override" do
  it "preserves the default identity and passes an explicit path and digest pair" do
    HyNomasLatentDecodeAdmission.resolve_decoder_cache!("", "").should be_nil
    path = "/opt/mldata_ssd/recovered-pixelumm-cache.json"
    sha = "d" * 64
    HyNomasLatentDecodeAdmission.resolve_decoder_cache!(path, sha).should eq({path, sha})
  end

  it "rejects partial cache overrides before file authentication" do
    [{"/opt/mldata_ssd/recovered-pixelumm-cache.json", ""}, {"", "d" * 64}].each do |path, sha|
      expect_raises(ArgumentError, /must be supplied together/) do
        HyNomasLatentDecodeAdmission.resolve_decoder_cache!(path, sha)
      end
    end
  end

  it "rejects relative cache paths and malformed cache digests" do
    [{"relative/cache.json", "d" * 64}, {"/cache.json", "D" * 64},
     {"/cache.json", "d" * 63}, {"/cache.json", "g" * 64}].each do |path, sha|
      expect_raises(ArgumentError, /absolute path and lowercase SHA-256/) do
        HyNomasLatentDecodeAdmission.resolve_decoder_cache!(path, sha)
      end
    end
  end
end

describe "shared scene text policy" do
  it "uses one cached text block per admitted JSON scene" do
    HyNomasSequence::HISTORY_TEXT_POLICY.once_per_scene?.should be_true
  end
end

describe HyNomasSequence do
  it "binds selected BF16 weight index and lineage into one sequence identity" do
    cinematic = HyNomasCinematic::IndexIdentity.new(
      "/opt/mldata_ssd/cinematic/index.json", "c" * 64,
      "/opt/mldata_ssd/cinematic/cinematic-cache.json", "d" * 64)

    HyNomasSequence.weight_identity(cinematic).should eq({
      dit_variant:              "cinematic-active-blocks",
      dit_weight_index_sha256:  "c" * 64,
      cinematic_lineage:        "/opt/mldata_ssd/cinematic/cinematic-cache.json",
      cinematic_lineage_sha256: "d" * 64,
    })
    HyNomasSequence.weight_identity(nil).should eq({
      dit_variant:              "original",
      dit_weight_index_sha256:  HyNomasGeneration::INDEX_SHA,
      cinematic_lineage:        nil,
      cinematic_lineage_sha256: nil,
    })
  end

  it "keeps Cinematic outside the base-indexed checkpoint contract" do
    cinematic = HyNomasCinematic::IndexIdentity.new(
      "/opt/mldata_ssd/cinematic/index.json", "c" * 64,
      "/opt/mldata_ssd/cinematic/cinematic-cache.json", "d" * 64)

    expect_raises(ArgumentError, /base-only/) do
      HyNomasSequence.validate_checkpoint_variant!(cinematic, prepare_capture: true)
    end
    expect_raises(ArgumentError, /base-only/) do
      HyNomasSequence.validate_checkpoint_variant!(cinematic, checkpoint: true)
    end
    expect_raises(ArgumentError, /base-only/) do
      HyNomasSequence.validate_checkpoint_variant!(cinematic, restore: true)
    end
    HyNomasSequence.validate_checkpoint_variant!(nil, prepare_capture: true)
    HyNomasSequence.validate_checkpoint_variant!(nil, checkpoint: true)
    HyNomasSequence.validate_checkpoint_variant!(nil, restore: true)
  end

  it "admits one scene with no action tails" do
    ["{}", %({"contained_actions":null}), %({"contained_actions":[]})].each do |scene|
      HyNomasSequence.validate_shared_scene_prompt!(%({"segments":[#{scene}]}))
    end
  end

  it "rejects action tails before a shared prompt can enter history" do
    expect_raises(ArgumentError, /action tails/) do
      HyNomasSequence.validate_shared_scene_prompt!(
        %({"segments":[{"contained_actions":[{"action_description":"<Action1>Move</Action1>"}]}]}))
    end
    expect_raises(ArgumentError, /action tails/) do
      HyNomasSequence.validate_shared_scene_prompt!(%({"segments":[{"contained_actions":{}}]}))
    end
  end

  it "rejects multiple scene prompts and malformed scene boundaries" do
    ["not json", "[]", "{}", %({"segments":[]}),
     %({"segments":[{},{}]}), %({"segments":[null]})].each do |prompt|
      expect_raises(ArgumentError) { HyNomasSequence.validate_shared_scene_prompt!(prompt) }
    end
  end

  it "admits two reference subjects only for the sequential-action profile" do
    prompt = %({"subject_list":[{"subject_id":"<|Subject1|>","display_name":"Alex","subject_attribute":"Teal shirt, seated left"},{"subject_id":"<|Subject2|>","display_name":"Resume","subject_attribute":"Cream paper, white gloves, standing right"}],"segments":[{}]})
    HyNomasSequence.validate_reference_subject_roster!(prompt, sequential_actions: true)
    expect_raises(ArgumentError, /single T0 Subject1/) do
      HyNomasSequence.validate_reference_subject_roster!(prompt)
    end
  end

  it "rejects missing, swapped and malformed sequential reference subjects" do
    subject1 = %({"subject_id":"<|Subject1|>","display_name":"Alex","subject_attribute":"Teal shirt"})
    subject2 = %({"subject_id":"<|Subject2|>","display_name":"Resume","subject_attribute":"Cream paper"})
    rosters = ["[]", "[#{subject1}]", "[#{subject2},#{subject1}]",
               "[#{subject1},#{subject1}]", "[#{subject1},#{subject2},#{subject2}]",
               "[#{subject1},null]", "[#{subject1},{}]",
               "[#{subject1},#{subject2.sub("Resume", "Alex")}]",
               "[#{subject1},#{subject2.sub("Cream paper", " ")}]",
               "[#{subject1},#{subject2.sub("<|Subject2|>", "<|Subject3|>")}]"]
    rosters.each do |roster|
      expect_raises(ArgumentError, /ordered T0 Subject1/) do
        HyNomasSequence.validate_reference_subject_roster!(
          %({"subject_list":#{roster},"segments":[{}]}), sequential_actions: true)
      end
    end
    expect_raises(ArgumentError, /action tails/) do
      HyNomasSequence.validate_reference_subject_roster!(
        %({"subject_list":[#{subject1},#{subject2}],"segments":[{"contained_actions":[{}]}]}),
        sequential_actions: true)
    end
  end

  it "counts four denoise forwards plus one separate writer per nonfinal segment" do
    HyNomasSequence.forwards(2).should eq(9)
    HyNomasSequence.forwards(4).should eq(19)
    HyNomasSequence.forwards(6).should eq(29)
    [1, 7].each do |count|
      expect_raises(ArgumentError) { HyNomasSequence.forwards(count) }
    end
  end

  it "keeps the T0 reference writer unmasked and changes only the early T0 video keys" do
    all_visible = HyNomasSequence.t0_history_attention_profiles(263, 30)
    recent_only = HyNomasSequence.t0_history_attention_profiles(263, 10)
    all_visible.size.should eq(2)
    recent_only.size.should eq(2)
    all_visible[0].should be_nil
    recent_only[0].should be_nil

    all_keys = all_visible[1].not_nil!.key
    recent_keys = recent_only[1].not_nil!.key
    all_visible[1].not_nil!.query.size.should eq(4_583)
    all_keys.size.should eq(8_903)
    recent_keys.size.should eq(8_903)
    all_visible[1].not_nil!.query.all?(&.zero?).should be_true
    recent_only[1].not_nil!.query.all?(&.zero?).should be_true
    all_keys.all?(&.zero?).should be_true
    recent_keys[0, 2_880].should eq(Array(Int32).new(2_880, -1))
    recent_keys[2_880..].all?(&.zero?).should be_true

    # Both arms take the same masked workspace path; only the T1 key labels differ.
    common = {
      text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene,
      segment_text_tokens: [263, 263], segment_cache_text: [true, false],
      source_bf16_attention: true,
    }
    keep_all = ML::Video::HyNomasContinuationHistory.geometry(2, 4_320, 263, 320, 54, 16, 128,
      **common, segment_attention_profiles: all_visible)
    keep_recent = ML::Video::HyNomasContinuationHistory.geometry(2, 4_320, 263, 320, 54, 16, 128,
      **common, segment_attention_profiles: recent_only)
    keep_all.named_bytes.should eq(keep_recent.named_bytes)
    keep_all.workspace_geometries[0].masked.should be_false
    keep_all.writer_workspace_geometries[0].should eq(keep_all.workspace_geometries[0])
    keep_all.workspace_geometries[1].masked.should be_true
    keep_recent.workspace_geometries[1].masked.should be_true
  end

  it "hides only T0 caption keys for the all-visible T1 reader without changing workspace bytes" do
    all_visible = HyNomasSequence.t0_history_attention_profiles(263, 30)
    hide_caption = HyNomasSequence.t0_history_attention_profiles(263, 30,
      t0_hide_caption_keys: true)

    all_visible[0].should be_nil
    hide_caption[0].should be_nil
    baseline_t1 = all_visible[1].not_nil!
    hidden_t1 = hide_caption[1].not_nil!
    hidden_t1.query.size.should eq(4_583)
    hidden_t1.key.size.should eq(8_903)
    hidden_t1.query.should eq(baseline_t1.query)
    hidden_t1.query.all?(&.zero?).should be_true
    hidden_t1.key[0, 4_320].all?(&.zero?).should be_true
    hidden_t1.key[4_320, 263].should eq(Array(Int32).new(263, -1))
    hidden_t1.key[4_583..].all?(&.zero?).should be_true

    common = {
      text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene,
      segment_text_tokens: [263, 263], segment_cache_text: [true, false],
      source_bf16_attention: true,
    }
    all_visible_geometry = ML::Video::HyNomasContinuationHistory.geometry(2, 4_320, 263, 320, 54, 16, 128,
      **common, segment_attention_profiles: all_visible)
    hide_caption_geometry = ML::Video::HyNomasContinuationHistory.geometry(2, 4_320, 263, 320, 54, 16, 128,
      **common, segment_attention_profiles: hide_caption)
    hide_caption_geometry.named_bytes.should eq(all_visible_geometry.named_bytes)
    hide_caption_geometry.workspace_geometries[0].masked.should be_false
    hide_caption_geometry.writer_workspace_geometries[0].should eq(hide_caption_geometry.workspace_geometries[0])
    hide_caption_geometry.workspace_geometries[1].masked.should be_true

    expect_raises(ArgumentError, /all 30 T0 video latents visible/) do
      HyNomasSequence.t0_history_attention_profiles(263, 10, t0_hide_caption_keys: true)
    end
    expect_raises(ArgumentError, /all 30 T0 video latents visible/) do
      HyNomasSequence.validate_t0_history_admission!(nil, true,
        source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        segments: 2)
    end
    {
      {source_bf16: false, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES, segments: 2},
      {source_bf16: true, reference_latent_bytes: 0_i64, segments: 2},
      {source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES, segments: 3},
    }.each do |bounds|
      expect_raises(ArgumentError, /two-segment source-BF16 reference run without edits/) do
        HyNomasSequence.validate_t0_history_admission!(30, true, **bounds)
      end
    end
    expect_raises(ArgumentError, /two-segment source-BF16 reference run without edits/) do
      HyNomasSequence.validate_t0_history_admission!(30, true,
        source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        segments: 2, text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::PerChunk)
    end
    noise_edit = ML::Video::HyNomasSegmentNoiseEdit.new(2, 1, 2)
    expect_raises(ArgumentError, /two-segment source-BF16 reference run without edits/) do
      HyNomasSequence.validate_t0_history_admission!(30, true,
        source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        segments: 2, noise_edit: noise_edit)
    end
    prompt_edit = ML::Video::HyNomasSegmentPromptEdit.new(2, 1, "a" * 64, "b" * 64, "c" * 64, 263)
    expect_raises(ArgumentError, /two-segment source-BF16 reference run without edits/) do
      HyNomasSequence.validate_t0_history_admission!(30, true,
        source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        segments: 2, prompt_edit: prompt_edit)
    end
  end

  it "adds current T1 text keys only when hidden T0 caption keys remain all-video-visible" do
    hidden_t0_caption = HyNomasSequence.t0_history_attention_profiles(263, 30,
      t0_hide_caption_keys: true)
    current_t1_text = HyNomasSequence.t0_history_attention_profiles(263, 30,
      t0_hide_caption_keys: true, t1_current_text_kv: true)
    hidden_t1 = hidden_t0_caption[1].not_nil!
    current_t1 = current_t1_text[1].not_nil!

    current_t1.query.size.should eq(4_583)
    current_t1.query.should eq(hidden_t1.query)
    current_t1.key.size.should eq(9_166)
    current_t1.key[0, 4_320].all?(&.zero?).should be_true
    current_t1.key[4_320, 263].should eq(Array(Int32).new(263, -1))
    current_t1.key[4_583, 4_320].all?(&.zero?).should be_true
    current_t1.key[8_903, 263].all?(&.zero?).should be_true

    text_profile = HyNomasSequence.t1_current_text_profile(263, 2)
    text_profile[:segment_text_tokens].should eq([263, 263])
    text_profile[:segment_cache_text].should eq([true, true])
    current = ML::Video::HyNomasContinuationHistory.geometry(2, 4_320, 263, 320, 54, 16, 128,
      text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene,
      segment_text_tokens: text_profile[:segment_text_tokens],
      segment_cache_text: text_profile[:segment_cache_text],
      source_bf16_attention: true, segment_attention_profiles: current_t1_text)
    hidden = ML::Video::HyNomasContinuationHistory.geometry(2, 4_320, 263, 320, 54, 16, 128,
      text_cache_policy: ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene,
      segment_text_tokens: text_profile[:segment_text_tokens],
      segment_cache_text: [true, false],
      source_bf16_attention: true, segment_attention_profiles: hidden_t0_caption)
    current.query_rows.should eq(4_583)
    current.workspace_geometries[1].query_rows.should eq(4_583)
    current.workspace_geometries[1].current_rows.should eq(4_583)
    current.workspace_geometries[1].key_rows.should eq(9_166)
    current.workspace_geometries[0].key_rows.should eq(4_583)
    current.max_history_rows.should eq(hidden.max_history_rows)
    current.append_elements(0).should eq(hidden.append_elements(0))
    current.writer_workspace_geometries[0].should eq(hidden.writer_workspace_geometries[0])
    expect_raises(ArgumentError, /Final or invalid chunk/) do
      current.append_elements(1)
    end

    expect_raises(ArgumentError) do
      HyNomasSequence.t0_history_attention_profiles(263, 30, t1_current_text_kv: true)
    end
    expect_raises(ArgumentError) do
      HyNomasSequence.t0_history_attention_profiles(263, 10,
        t0_hide_caption_keys: true, t1_current_text_kv: true)
    end
    expect_raises(ArgumentError) do
      HyNomasSequence.validate_t0_history_admission!(30, false, true,
        source_bf16: true, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        segments: 2)
    end
    expect_raises(ArgumentError) do
      HyNomasSequence.validate_t0_history_admission!(30, true, true,
        source_bf16: false, reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        segments: 2)
    end
  end

  it "rejects unsupported T0 history visibility windows" do
    [0, 9, 11, 20, 29, 31].each do |visible|
      expect_raises(ArgumentError) do
        HyNomasSequence.t0_history_attention_profiles(263, visible)
      end
    end
  end
end
