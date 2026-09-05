require "spec"
require "../src/answer"

describe "merge ranges" do
  it "handles empty, overlapping, nested and endpoint-touching ranges" do
    Answer.merge_ranges([] of Tuple(Int32, Int32)).should eq([] of Tuple(Int32, Int32))
    values = [{5, 8}, {1, 3}, {3, 6}, {2, 2}, {10, 11}]
    original = values.dup
    Answer.merge_ranges(values).should eq([{1, 8}, {10, 11}])
    values.should eq(original)
    Answer.merge_ranges([{1, 2}, {3, 4}]).should eq([{1, 2}, {3, 4}])
    Answer.merge_ranges([{Int32::MAX, Int32::MAX}, {Int32::MIN, -1}, {-1, 0}]).should eq([{Int32::MIN, 0}, {Int32::MAX, Int32::MAX}])
  end
  it "matches a fixed-point pairwise oracle over 100 small inputs" do
    rng = Random.new(632)
    100.times do
      values = Array.new(rng.rand(0..15)) do
        a, b = rng.rand(-10..10), rng.rand(-10..10)
        {Math.min(a, b), Math.max(a, b)}
      end
      expected = values.dup
      loop do
        pair = (0...expected.size).to_a.combinations(2).find do |ij|
          a, b = expected[ij[0]], expected[ij[1]]
          a[0] <= b[1] && b[0] <= a[1]
        end
        break unless pair
        i, j = pair
        a, b = expected[i], expected[j]
        expected.delete_at(j)
        expected[i] = {Math.min(a[0], b[0]), Math.max(a[1], b[1])}
      end
      Answer.merge_ranges(values).should eq(expected.sort)
    end
  end
end
