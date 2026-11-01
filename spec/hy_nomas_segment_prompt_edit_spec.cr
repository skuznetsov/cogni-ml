require "spec"
require "../src/ml/video/hy_nomas_segment_edit"

describe ML::Video::HyNomasSegmentPromptEdit do
  it "keeps the parent identity and binds all edited descendants" do
    edit = ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "b" * 64, "c" * 64, 123)
    edit.bind_contract("after-A", 1).should eq("after-A")
    edit.bind_contract("after-B", 2).should_not eq("after-B")
    doc = JSON.parse(edit.bind_contract("after-B", 2))
    doc["prompt_edit"]["conditioning_manifest_sha256"].as_s.should eq("a" * 64)
    doc["prompt_edit"]["target_segment"].as_i.should eq(1)
  end

  it "selects B conditioning only and returns C to the original as a new turn" do
    edit = ML::Video::HyNomasSegmentPromptEdit.new(4, 1, "a" * 64, "b" * 64, "c" * 64, 123)
    edit.edited_segment?(0).should be_false
    edit.edited_segment?(1).should be_true
    edit.edited_segment?(2).should be_false
    edit.text_tokens(91).should eq([91, 123, 91, 91])
    edit.cache_text.should eq([true, true, true, false])
  end

  it "invalidates descendant state for every conditioning identity change" do
    original = ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "b" * 64, "c" * 64, 123)
    variants = [
      ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "d" * 64, "b" * 64, "c" * 64, 123),
      ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "d" * 64, "c" * 64, 123),
      ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "b" * 64, "d" * 64, 123),
      ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "b" * 64, "c" * 64, 124),
    ]
    variants.each do |changed|
      changed.bind_contract("parent", 1).should eq(original.bind_contract("parent", 1))
      changed.bind_contract("child", 2).should_not eq(original.bind_contract("child", 2))
    end
    later = ML::Video::HyNomasSegmentPromptEdit.new(5, 3, "a" * 64, "b" * 64, "c" * 64, 123)
    later.text_tokens(91).should eq([91, 91, 91, 123, 91])
    later.cache_text.should eq([true, false, false, true, true])
    (1..3).each { |boundary| later.bind_contract("parent", boundary).should eq("parent") }
  end

  it "rejects malformed identities, initial targets and out-of-run selectors" do
    expect_raises(ArgumentError) { ML::Video::HyNomasSegmentPromptEdit.new(3, 0, "a" * 64, "b" * 64, "c" * 64, 123) }
    expect_raises(ArgumentError) { ML::Video::HyNomasSegmentPromptEdit.new(3, 3, "a" * 64, "b" * 64, "c" * 64, 123) }
    ["", "X" * 64, "a" * 63].each do |sha|
      expect_raises(ArgumentError) { ML::Video::HyNomasSegmentPromptEdit.new(3, 1, sha, "b" * 64, "c" * 64, 123) }
    end
    [0, 1001].each do |tokens|
      expect_raises(ArgumentError) { ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "b" * 64, "c" * 64, tokens) }
    end
    edit = ML::Video::HyNomasSegmentPromptEdit.new(3, 1, "a" * 64, "b" * 64, "c" * 64, 123)
    [-1, 3].each { |segment| expect_raises(ArgumentError) { edit.edited_segment?(segment) } }
    [0, 3].each { |boundary| expect_raises(ArgumentError) { edit.bind_contract("base", boundary) } }
  end
end
