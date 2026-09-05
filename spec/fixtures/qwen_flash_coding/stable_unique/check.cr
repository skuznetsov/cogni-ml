require "spec"
require "../src/answer"

describe "stable unique" do
  it "handles empty and repeated inputs without mutation" do
    Answer.stable_unique([] of Int32).should eq([] of Int32)
    values = [3, -1, 3, 0, -1, 2, 2]
    original = values.dup
    Answer.stable_unique(values).should eq([3, -1, 0, 2])
    values.should eq(original)
    Answer.stable_unique([Int32::MIN, Int32::MAX, Int32::MIN]).should eq([Int32::MIN, Int32::MAX])
  end
  it "matches a simple order-preserving reference over 100 inputs" do
    rng = Random.new(712)
    100.times do
      values = Array.new(rng.rand(0..50)) { rng.rand(-8..8) }
      Answer.stable_unique(values).should eq(values.uniq)
    end
  end
end
