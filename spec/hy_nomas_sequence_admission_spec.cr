require "spec"
require "../examples/support/hy_nomas_sequence"

describe "shared scene text policy" do
  it "uses one cached text block per admitted JSON scene" do
    HyNomasSequence::HISTORY_TEXT_POLICY.once_per_scene?.should be_true
  end
end

describe HyNomasSequence do
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
