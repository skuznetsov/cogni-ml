require "spec"
require "../examples/support/hy_nomas_sequence"

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

  it "counts four denoise forwards plus one separate writer per nonfinal segment" do
    HyNomasSequence.forwards(2).should eq(9)
    HyNomasSequence.forwards(4).should eq(19)
    HyNomasSequence.forwards(6).should eq(29)
    [1, 7].each do |count|
      expect_raises(ArgumentError) { HyNomasSequence.forwards(count) }
    end
  end
end
