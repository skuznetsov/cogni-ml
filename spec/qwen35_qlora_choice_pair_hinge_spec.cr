require "spec"
require "../eval/crystal_bugfixes/caller_transfer/real_pairs/choice_pair_hinge"

describe ChoicePairHinge do
  it "pushes the weaker side in the direction of its own correct label" do
    pair = ChoicePairHinge.calculate(0.125_f32, 0.25_f32, true, 0.5_f32, 0.5_f32)
    pair[:margin_a].should eq(0.125_f32)
    pair[:margin_b].should eq(-0.25_f32)
    pair[:penalty].should eq(0.375)
    pair[:scale_a].should eq(0_f32)
    pair[:scale_b].should eq(0.5_f32)

    swapped = ChoicePairHinge.calculate(-0.125_f32, -0.25_f32, false, 0.5_f32, 0.5_f32)
    swapped[:penalty].should eq(pair[:penalty])
    swapped[:scale_a].should eq(-pair[:scale_a])
    swapped[:scale_b].should eq(-pair[:scale_b])

    delta = 0.03125_f32
    plus = ChoicePairHinge.calculate(0.125_f32, 0.25_f32 + delta, true, 0.5_f32, 0.5_f32)
    minus = ChoicePairHinge.calculate(0.125_f32, 0.25_f32 - delta, true, 0.5_f32, 0.5_f32)
    ((plus[:penalty] - minus[:penalty]) / (2 * delta)).should eq(pair[:scale_b].to_f64)
  end

  it "splits a tied shortfall and vanishes only when both sides reach the floor" do
    tied = ChoicePairHinge.calculate(0.25_f32, -0.25_f32, true, 0.5_f32, 0.5_f32)
    tied[:penalty].should eq(0.125)
    tied[:scale_a].should eq(-0.25_f32)
    tied[:scale_b].should eq(0.25_f32)

    left_below = ChoicePairHinge.calculate(0.25_f32, -1_f32, true, 0.5_f32, 0.5_f32)
    left_below[:penalty].should eq(0.125)
    left_below[:scale_a].should eq(-0.5_f32)
    left_below[:scale_b].should eq(0_f32)

    passed = ChoicePairHinge.calculate(0.5_f32, -1_f32, true, 0.5_f32, 0.5_f32)
    passed[:penalty].should eq(0.0)
    passed[:scale_a].should eq(0_f32)
    passed[:scale_b].should eq(0_f32)
  end

  it "differentiates the directed difference independently of the common offset" do
    pair = ChoicePairHinge.calculate(0.125_f32, 0.25_f32, true, 0.5_f32, 0.5_f32, on_difference: true)
    shifted = ChoicePairHinge.calculate(0.875_f32, 1_f32, true, 0.5_f32, 0.5_f32, on_difference: true)
    swapped = ChoicePairHinge.calculate(0.25_f32, 0.125_f32, false, 0.5_f32, 0.5_f32, on_difference: true)
    pair[:penalty].should eq(0.28125)
    shifted[:penalty].should eq(pair[:penalty])
    shifted[:scale_a].should eq(pair[:scale_a])
    shifted[:scale_b].should eq(pair[:scale_b])
    pair[:scale_a].should eq(-0.25_f32)
    pair[:scale_b].should eq(0.25_f32)
    swapped[:penalty].should eq(pair[:penalty])
    swapped[:scale_a].should eq(pair[:scale_b])
    swapped[:scale_b].should eq(pair[:scale_a])

    delta = 0.03125_f32
    plus = ChoicePairHinge.calculate(0.125_f32 + delta, 0.25_f32, true, 0.5_f32, 0.5_f32, on_difference: true)
    minus = ChoicePairHinge.calculate(0.125_f32 - delta, 0.25_f32, true, 0.5_f32, 0.5_f32, on_difference: true)
    ((plus[:penalty] - minus[:penalty]) / (2 * delta)).should eq(pair[:scale_a].to_f64)
  end

  it "stops at the directed floor even if one individual leader is wrong" do
    values = {-0.25_f32, -1.25_f32, true, 0.5_f32, 0.5_f32}
    weakest = ChoicePairHinge.calculate(*values)
    directed = ChoicePairHinge.calculate(*values, on_difference: true)
    weakest[:penalty].should eq(0.375)
    directed[:penalty].should eq(0.0)
    directed[:scale_a].should eq(0_f32)
    directed[:scale_b].should eq(0_f32)
  end
end
