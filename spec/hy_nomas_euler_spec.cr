require "spec"
require "../src/ml/video/hy_nomas_euler"

describe ML::Video::HyNomasEulerPlan do
  it "preserves eight as the default while admitting bounded longer schedules" do
    ML::Video::HyNomasEulerPlan.schedule.size.should eq(8)
    [16, 24, 32, 64, 96, 128].each do |steps|
      schedule = ML::Video::HyNomasEulerPlan.schedule(steps, 5.0_f32)
      schedule.size.should eq(steps)
      schedule.first.sigma.should eq(1.0_f32)
      (schedule.last.sigma + schedule.last.dt).should eq(0.0_f32)
      schedule.all? { |s| s.sigma.finite? && s.timestep.finite? && s.dt.finite? && s.dt < 0 }.should be_true
    end
  end

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

  it "matches pinned upstream F32 anchors for longer experimental schedules" do
    # CPU torch.linspace followed by the pinned solver's shift and differences.
    expected = {
      16 => {
        [0x3f800000_u32, 0x3f555555_u32, 0x3e800000_u32],
        [0x447a0000_u32, 0x44505555_u32, 0x437a0000_u32],
        [0xbc579440_u32, 0xbd1b26c0_u32, 0xbe800000_u32],
      },
      24 => {
        [0x3f800000_u32, 0x3f555555_u32, 0x3e36db6f_u32],
        [0x447a0000_u32, 0x44505555_u32, 0x4332924a_u32],
        [0xbc0d3dc0_u32, 0xbcc8c8c0_u32, 0xbe36db6f_u32],
      },
      32 => {
        [0x3f800000_u32, 0x3f555555_u32, 0x3e0e38e4_u32],
        [0x447a0000_u32, 0x44505555_u32, 0x430ae38f_u32],
        [0xbbd20d00_u32, 0xbc9467e0_u32, 0xbe0e38e4_u32],
      },
      64 => {
        [0x3f800000_u32, 0x3f555555_u32, 0x3d969697_u32],
        [0x447a0000_u32, 0x44505555_u32, 0x42930f0f_u32],
        [0xbb4f6400_u32, 0xbc113f80_u32, 0xbd969697_u32],
      },
      96 => {
        [0x3f800000_u32, 0x3f555555_u32, 0x3d4cccce_u32],
        [0x447a0000_u32, 0x44505555_u32, 0x42480001_u32],
        [0xbb09ae00_u32, 0xbbc04d00_u32, 0xbd4cccce_u32],
      },
      128 => {
        [0x3f800000_u32, 0x3f555555_u32, 0x3d1b26ca_u32],
        [0x447a0000_u32, 0x44505555_u32, 0x421783e1_u32],
        [0xbace1600_u32, 0xbb8fb800_u32, 0xbd1b26ca_u32],
      },
    }
    expected.each do |steps, bits|
      schedule = ML::Video::HyNomasEulerPlan.schedule(steps, 5.0_f32)
      anchors = [0, steps // 2, steps - 1].map { |index| schedule[index] }
      anchors.map { |s| s.sigma.unsafe_as(UInt32) }.should eq(bits[0])
      anchors.map { |s| s.timestep.unsafe_as(UInt32) }.should eq(bits[1])
      anchors.map { |s| s.dt.unsafe_as(UInt32) }.should eq(bits[2])
    end
  end

  it "matches pinned 96-step F32 anchors around CPU linspace FMA positions" do
    schedule = ML::Video::HyNomasEulerPlan.schedule(96, 5.0_f32)

    [26, 29].map { |index| schedule[index].sigma.unsafe_as(UInt32) }.should eq([
      0x3f6e4c42_u32, 0x3f6b9aba_u32,
    ])
    [26, 29].map { |index| schedule[index].timestep.unsafe_as(UInt32) }.should eq([
      0x4468b678_u32, 0x4466151a_u32,
    ])
    [25, 26, 28, 29].map { |index| schedule[index].dt.unsafe_as(UInt32) }.should eq([
      0xbb5c2900_u32, 0xbb60e700_u32, 0xbb6ad600_u32, 0xbb700f00_u32,
    ])
  end

  it "matches both ends of the upstream F32 linspace for six steps" do
    # Pinned EulerSolver uses CPU torch.linspace before shifting. Directly
    # dividing (steps - index) / steps differs by one ULP at index two.
    schedule = ML::Video::HyNomasEulerPlan.schedule(6, 5.0_f32)
    schedule.map { |s| s.sigma.unsafe_as(UInt32) }.should eq([
      0x3f800000_u32, 0x3f762763_u32, 0x3f68ba2e_u32,
      0x3f555555_u32, 0x3f36db6d_u32, 0x3f000000_u32,
    ])
    schedule.map { |s| s.timestep.unsafe_as(UInt32) }.should eq([
      0x447a0000_u32, 0x44706277_u32, 0x446345d1_u32,
      0x44505555_u32, 0x44329248_u32, 0x43fa0000_u32,
    ])
    schedule.map { |s| s.dt.unsafe_as(UInt32) }.should eq([
      0xbd1d89d0_u32, 0xbd56d350_u32, 0xbd9b26c8_u32,
      0xbdf3cf40_u32, 0xbe5b6db4_u32, 0xbf000000_u32,
    ])
  end

  it "preserves upstream F32 linspace endpoint anchoring for odd counts" do
    expected = {
      3 => [0x3f800000_u32, 0x3f68ba2e_u32, 0x3f36db6d_u32],
      5 => [0x3f800000_u32, 0x3f73cf3e_u32, 0x3f61e1e1_u32, 0x3f44ec4f_u32, 0x3f0e38e4_u32],
      7 => [0x3f800000_u32, 0x3f77bdee_u32, 0x3f6d097c_u32, 0x3f5e9bd3_u32, 0x3f4a1af3_u32, 0x3f2aaaab_u32, 0x3ee8ba2f_u32],
    }
    expected.each do |steps, bits|
      schedule = ML::Video::HyNomasEulerPlan.schedule(steps, 5.0_f32)
      schedule.map { |s| s.sigma.unsafe_as(UInt32) }.should eq(bits)
      (schedule.last.sigma + schedule.last.dt).should eq(0.0_f32)
    end
  end

  it "rejects unsupported step counts and invalid shifts before Metal setup" do
    expect_raises(ArgumentError, /between 1 and 128/) do
      ML::Video::HyNomasEulerPlan.schedule(0, 5.0_f32)
    end
    expect_raises(ArgumentError, /between 1 and 128/) do
      ML::Video::HyNomasEulerPlan.schedule(129, 5.0_f32)
    end
    expect_raises(ArgumentError, /finite and positive/) do
      ML::Video::HyNomasEulerPlan.schedule(8, 0.0_f32)
    end
  end
end
