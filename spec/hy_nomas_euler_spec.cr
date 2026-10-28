require "spec"
require "../src/ml/video/hy_nomas_euler"

describe ML::Video::HyNomasEulerPlan do
  it "matches the upstream eight-step shifted Euler schedule" do
    schedule = ML::Video::HyNomasEulerPlan.schedule(8, 5.0_f32)

    schedule.size.should eq(8)
    schedule.map(&.sigma).should eq([
      1.0_f32,
      0.9722222_f32,
      0.9375_f32,
      0.89285713_f32,
      0.8333333_f32,
      0.75_f32,
      0.625_f32,
      0.41666666_f32,
    ])
    schedule.map(&.timestep).should eq([
      1000.0_f32,
      972.22223_f32,
      937.5_f32,
      892.8571_f32,
      833.3333_f32,
      750.0_f32,
      625.0_f32,
      416.66666_f32,
    ])
    schedule.map(&.dt).should eq([
      -0.027777791_f32,
      -0.03472221_f32,
      -0.044642866_f32,
      -0.05952382_f32,
      -0.08333331_f32,
      -0.125_f32,
      -0.20833334_f32,
      -0.41666666_f32,
    ])
  end

  it "rejects unsupported step counts and invalid shifts before Metal setup" do
    expect_raises(ArgumentError, /between 1 and 8/) do
      ML::Video::HyNomasEulerPlan.schedule(0, 5.0_f32)
    end
    expect_raises(ArgumentError, /finite and positive/) do
      ML::Video::HyNomasEulerPlan.schedule(8, 0.0_f32)
    end
  end
end
