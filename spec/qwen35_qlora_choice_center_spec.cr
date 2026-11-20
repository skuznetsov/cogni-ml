require "spec"
require "../eval/crystal_bugfixes/caller_transfer/real_pairs/choice_pair_center"

describe ChoicePairCenter do
  it "leaves equal and opposite log-odds untouched" do
    result = ChoicePairCenter.calculate(2_f32, -2_f32, 0.25_f32)
    result[:mean_log_odds].should eq(0.0)
    result[:penalty].should eq(0.0)
    result[:scale].should eq(0_f32)
  end

  it "computes the penalty at the precision of the mean loss" do
    left, right = {0.35_f32, -0.12_f32}
    expected = (left.to_f64 + right.to_f64) ** 2 / 16
    ChoicePairCenter.calculate(left, right, 0.25_f32)[:penalty].should be_close(expected, 1e-12)
  end

  it "has the same derivative on both raw logits and reverses it under a label swap" do
    weight = 0.25_f32
    result = ChoicePairCenter.calculate(1_f32, 3_f32, weight)
    result[:mean_log_odds].should eq(2.0)
    result[:penalty].should eq(1.0)
    result[:scale].should eq(0.5_f32)

    swapped = ChoicePairCenter.calculate(-1_f32, -3_f32, weight)
    swapped[:penalty].should eq(result[:penalty])
    swapped[:scale].should eq(-result[:scale])

    delta = 0.03125_f32
    [{1_f32 + delta, 3_f32}, {1_f32, 3_f32 + delta}].each do |left, right|
      upper = ChoicePairCenter.calculate(left, right, weight)[:penalty]
      lower = ChoicePairCenter.calculate(2_f32 - left, 6_f32 - right, weight)[:penalty]
      ((upper - lower) / (2 * delta)).should eq(result[:scale].to_f64)
    end
  end
end
