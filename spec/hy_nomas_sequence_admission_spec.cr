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
end
