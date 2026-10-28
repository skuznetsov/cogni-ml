require "spec"
require "../examples/support/pixelumm_drift_probe"

describe PixelUMMDriftProbe do
  it "balances profiling on and off across time without changing the four-plan baseline" do
    schedule = PixelUMMDriftProbe.schedule
    schedule.size.should eq(102)
    schedule.map(&.[:index]).should eq((0...102).to_a)
    schedule.count { |item| item[:kind] == "correctness" }.should eq(2)
    timed = schedule.select { |item| item[:kind] == "timed" }
    timed.size.should eq(96)
    timed.count(&.[:profile]).should eq(48)
    timed.each_slice(8) do |items|
      items.map(&.[:profile]).should eq([false, true, true, false, true, false, false, true])
    end
    schedule.each do |item|
      entry = PixelUMMDriftProbe.entry(item)
      entry[:mode].should eq("baseline")
      PixelUMMFFNWeightReuseProbe.plan_order(entry[:mode], entry[:baseline_schedule]).should eq([0, 1, 2, 3])
      entry[:capture_mode].should eq(item[:kind] == "correctness")
    end
  end
end
