require "spec"
require "../src/answer"

describe "lower bound" do
  it "handles boundaries and duplicates" do
    Answer.lower_bound([] of Int32, 0).should eq(0)
    Answer.lower_bound([1, 2, 2, 4], 2).should eq(1)
    Answer.lower_bound([1, 2, 2, 4], 5).should eq(4)
    Answer.lower_bound([Int32::MIN, 0, Int32::MAX], Int32::MIN).should eq(0)
    Answer.lower_bound([Int32::MIN, 0, Int32::MAX], Int32::MAX).should eq(2)
  end
  it "matches a linear oracle over 200 inputs without mutation" do
    rng = Random.new(918)
    200.times do
      values = Array.new(rng.rand(0..60)) { rng.rand(-20..20) }.sort
      original = values.dup
      target = rng.rand(-25..25)
      Answer.lower_bound(values, target).should eq(values.index { |v| v >= target } || values.size)
      values.should eq(original)
    end
  end
end
