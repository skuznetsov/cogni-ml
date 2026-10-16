require "spec"
require "../src/ml/audio/drama3/host_profile"

include ML::Audio::Drama3

describe "DRAMA-3 fixed host profile" do
  it "keeps disabled scopes transparent and does not accumulate counters" do
    profile = HostProfile.new(false)

    profile.enabled?.should be_false
    profile.measure(HostEvent::CommandSubmit) { :unchanged }.should eq(:unchanged)
    profile.add(HostCounter::KernelDispatches, 3_i64)

    delta = JSON.parse(HostProfile.delta(profile.snapshot, profile.snapshot).to_json)
    delta["events"].as_a.should be_empty
    delta["counters"]["KernelDispatches"].as_i.should eq(0)
  end

  it "tracks nested inclusive and exclusive time and returns a JSON snapshot delta" do
    profile = HostProfile.new(true)
    before = profile.snapshot

    result = profile.measure(HostEvent::EncoderAcquire) do
      profile.add(HostCounter::PendingEntries, 2_i64)
      profile.measure(HostEvent::CompletionWait) { :waited }
      :acquired
    end
    result.should eq(:acquired)

    after = profile.snapshot
    # Snapshots are values. A later update must not mutate the captured one.
    profile.measure(HostEvent::CompletionWait) { nil }

    delta = JSON.parse(HostProfile.delta(before, after).to_json)
    events = delta["events"].as_a.to_h { |event| {event["name"].as_s, event} }
    acquire = events["EncoderAcquire"]
    wait = events["CompletionWait"]

    acquire["calls"].as_i.should eq(1)
    wait["calls"].as_i.should eq(1)
    acquire_inclusive = acquire["inclusive_ns"].as_i
    acquire_exclusive = acquire["exclusive_ns"].as_i
    wait_inclusive = wait["inclusive_ns"].as_i
    (acquire_exclusive + wait_inclusive).should eq(acquire_inclusive)
    acquire_exclusive.should be >= 0
    wait["exclusive_ns"].as_i.should be <= wait_inclusive
    delta["counters"]["PendingEntries"].as_i.should eq(2)
    delta["counters"]["KernelDispatches"].as_i.should eq(0)
  end

  it "records elapsed time when a measured operation raises and preserves the exception" do
    profile = HostProfile.new(true)
    before = profile.snapshot

    expect_raises(Exception, "measurement sentinel") do
      profile.measure(HostEvent::Retention) do
        raise "measurement sentinel"
      end
    end

    delta = JSON.parse(HostProfile.delta(before, profile.snapshot).to_json)
    retention = delta["events"].as_a.find { |event| event["name"].as_s == "Retention" }.not_nil!
    retention["calls"].as_i.should eq(1)
    retention["inclusive_ns"].as_i.should be >= 0
  end

  {% unless flag?(:gc_none) || flag?(:wasm32) %}
    it "reports allocation and collection deltas around an explicit test collection" do
      before = HostProfile.gc_snapshot
      retained = Array(UInt8).new(256 * 1024, 0x5a_u8)
      GC.collect
      after = HostProfile.gc_snapshot

      delta = JSON.parse(HostProfile.gc_delta(before, after).to_json)
      delta["allocated_bytes"].as_i.should be > 0
      delta["collection_count"].as_i.should be > 0
      retained.size.should eq(256 * 1024)
    end
  {% end %}
end
