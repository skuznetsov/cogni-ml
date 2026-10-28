require "spec"
require "../examples/support/pixelumm_gc_probe"

alias GCP = PixelUMMGCProbe

describe PixelUMMGCProbe do
  it "balances both GC policies and gives every primary a normal-GC tail" do
    schedule = GCP.schedule
    schedule.size.should eq(70)
    timed = schedule.select { |trial| trial[:kind] == "timed" }
    timed.size.should eq(64)
    timed.count { |trial| trial[:policy] == GCP::NORMAL }.should eq(32)
    timed.count { |trial| trial[:policy] == GCP::DISABLED }.should eq(32)
    timed.each_slice(8) do |trials|
      trials.map(&.[:policy]).should eq([GCP::NORMAL, GCP::DISABLED, GCP::DISABLED, GCP::NORMAL,
                                         GCP::DISABLED, GCP::NORMAL, GCP::NORMAL, GCP::DISABLED])
    end
    entries = schedule.flat_map { |trial| [GCP.entry(trial), GCP.entry(trial, tail: true)] }
    entries.map(&.[:index]).should eq((0...140).to_a)
    entries.each do |entry|
      entry[:mode].should eq("baseline")
      PixelUMMFFNWeightReuseProbe.plan_order(entry[:mode], entry[:baseline_schedule]).should eq([0, 1, 2, 3])
    end
    entries.count(&.[:capture_mode]).should eq(2)
    entries.count { |entry| entry[:kind] == "tail" }.should eq(70)
    entries.last[:kind].should eq("tail")
  end

  it "bounds the diagnostic request-count extension before allocating Metal buffers" do
    PixelUMMFFNWeightReuseProbe.validate_request_count!(14).should be_nil
    PixelUMMFFNWeightReuseProbe.validate_request_count!(140).should be_nil
    expect_raises(ArgumentError) { PixelUMMFFNWeightReuseProbe.validate_request_count!(0) }
    expect_raises(ArgumentError) { PixelUMMFFNWeightReuseProbe.validate_request_count!(141) }
  end

  it "restores GC after a disabled-scope exception and rejects nested or unknown policy" do
    GCP.require_boehm!
    expect_raises(Exception, "injected body failure") do
      GCP.with_policy(GCP::DISABLED) do
        (LibGC.is_disabled != 0).should be_true
        raise "injected body failure"
      end
    end
    LibGC.is_disabled.should eq(0)
    GCP.with_policy(GCP::NORMAL) { LibGC.is_disabled.should eq(0) }
    GCP.with_policy(GCP::DISABLED) do
      expect_raises(ArgumentError) { GCP.with_policy(GCP::DISABLED) { raise "unreachable" } }
    end
    expect_raises(ArgumentError) { GCP.with_policy("unknown") { raise "unreachable" } }
    LibGC.is_disabled.should eq(0)
  end

  it "observes real Boehm collections and allocates no bytes while sampling points" do
    GCP.start_observer!
    begin
      sizeof(GCP::Observer::Counters).should eq(88)
      first = GCP::Point.new
      second = GCP::Point.new
      GCP.delta(first, second)[:allocated_bytes].should eq(0)
      # Positive control only; GC.collect never appears in the GPU driver.
      GC.collect
      collected = GCP::Point.new
      GCP.delta(second, collected)[:collections].should be > 0
      GCP.delta(second, collected)[:rendezvous_pauses].should be > 0
      GCP.delta(second, collected)[:stopped_intervals].should be > 0
      GCP.delta(second, collected)[:collection_seconds].should be > 0
      collected.unmatched_event_count.should eq(0)
      GCP.with_policy(GCP::DISABLED) { GC.collect }
      suppressed = GCP::Point.new
      GCP.delta(collected, suppressed)[:collections].should eq(0)
    ensure
      GCP.stop_observer!
    end
    LibGC.get_on_collection_event.pointer.null?.should be_true
  end
end
