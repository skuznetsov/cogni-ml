require "spec"
require "../examples/support/pixelumm_cognigraph_probe"

alias Probe = PixelUMMCogniGraphProbe

describe PixelUMMCogniGraphProbe do
  it "schedules two capture correctness requests, two output-only checks, four ABBA warmups, and 32 balanced timed requests" do
    requests = Probe.schedule

    requests.size.should eq(40)
    requests.map(&.[:index]).should eq((0...40).map(&.to_i32))
    requests[0...4].map(&.[:enabled]).should eq([false, true, false, true])
    requests[0...4].map(&.[:capture_mode]).should eq([true, true, false, false])
    requests[0...4].map(&.[:kind]).should eq(["correctness"] * 4)
    requests[0][:positive_nan_control].should be_true
    requests[1..].all?(&.[:positive_nan_control]).should be_false

    requests[4...8].map(&.[:enabled]).should eq([false, true, true, false])
    requests[4...8].all? { |request| request[:kind] == "warmup" && !request[:capture_mode] }.should be_true

    timed = requests[8..]
    timed.size.should eq(32)
    Array.new(4) { |cycle| timed[(cycle * 8)...((cycle + 1) * 8)].map(&.[:enabled]) }.should eq(
      Array.new(4) { [false, true, true, false, true, false, false, true] })
    timed.count(&.[:enabled]).should eq(16)
    timed.count(&.[:timing_eligible]).should eq(32)
    timed.all? { |request| request[:kind] == "timed" && !request[:capture_mode] }.should be_true

    requests.each do |request|
      entry = Probe.owner_entry(request)
      entry[:mode].should eq("baseline")
      entry[:capture_mode].should eq(request[:capture_mode])
      entry[:index].should eq(request[:index])
    end
  end

  it "rejects altered schedules that lose balance or move capture boundaries" do
    requests = Probe.schedule

    mismatched_mode = requests.dup
    mismatched_mode[9] = mismatched_mode[9].merge({enabled: false})
    expect_raises(ArgumentError) { Probe.validate_schedule!(mismatched_mode) }

    moved_capture = requests.dup
    moved_capture[2] = moved_capture[2].merge({capture_mode: true})
    expect_raises(ArgumentError) { Probe.validate_schedule!(moved_capture) }

    wrong_control = requests.dup
    wrong_control[0] = wrong_control[0].merge({positive_nan_control: false})
    expect_raises(ArgumentError) { Probe.validate_schedule!(wrong_control) }
  end

  it "allows native island mode changes only between requests after one successful terminal wait" do
    mode = Probe::NativeIslandMode.new

    mode.prepare_request!(0, false).should be_true
    mode.begin_request!(0, false).should be_nil
    expect_raises(ArgumentError) { mode.prepare_request!(1, true) }
    expect_raises(ArgumentError) { mode.finish_request!(0, 0) }
    mode.finish_request!(0, 1).should be_nil

    mode.prepare_request!(1, true).should be_true
    mode.begin_request!(1, true).should be_nil
    mode.finish_request!(1, 1).should be_nil

    mode.prepare_request!(2, true).should be_false
    mode.begin_request!(2, true).should be_nil
    mode.finish_request!(2, 1).should be_nil
    mode.prepare_request!(3, false).should be_true
  end

  it "rejects skipped requests, mismatched mode selections, and repeated terminal completion" do
    mode = Probe::NativeIslandMode.new
    expect_raises(ArgumentError) { mode.prepare_request!(1, false) }
    mode.prepare_request!(0, false)
    expect_raises(ArgumentError) { mode.begin_request!(0, true) }
    mode.begin_request!(0, false)
    mode.finish_request!(0, 1)
    expect_raises(ArgumentError) { mode.finish_request!(0, 1) }
    expect_raises(ArgumentError) { mode.prepare_request!(2, true) }
  end
end
