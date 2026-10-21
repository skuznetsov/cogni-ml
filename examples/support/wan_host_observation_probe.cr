# CPU-only smoke test for benchmark host observations. This intentionally does
# not initialize Metal or force a collection.
require "json"
require "./wan_host_observation"

before = WanHostObservation::Snapshot.capture
begin_seconds = WanHostObservation.monotonic_seconds
deadline = begin_seconds + 0.100
iterations = 0_u64
while WanHostObservation.monotonic_seconds < deadline
  iterations &+= 1
end
end_seconds = WanHostObservation.monotonic_seconds
after = WanHostObservation::Snapshot.capture
observations = WanHostObservation.between(before, after)

raise "monotonic host clock did not advance" unless end_seconds > begin_seconds
raise "probe loop did not execute" unless iterations > 0
raise "GC allocated-byte readings are missing" unless after.gc_allocated_bytes >= before.gc_allocated_bytes
raise "GC collection readings are missing" unless after.gc_collection_count >= before.gc_collection_count

if (before.valid_mask & 1) != 0 && (after.valid_mask & 1) != 0
  if user_delta = observations[:user_ns_delta]
    raise "CPU positive control did not report user time" unless user_delta > 0_u64
  else
    raise "CPU positive control did not report user time"
  end
end
if (before.valid_mask & 4) != 0 && (after.valid_mask & 4) != 0
  raise "process-memory readings are missing" unless before.resident_bytes && after.resident_bytes && before.footprint_bytes && after.footprint_bytes
end

{% if flag?(:darwin) %}
  raise "Darwin process CPU sensor group is unavailable" unless (before.valid_mask & 1) != 0 && (after.valid_mask & 1) != 0
  raise "Darwin process-memory sensor group is unavailable" unless (before.valid_mask & 4) != 0 && (after.valid_mask & 4) != 0
{% end %}

missing_before = WanHostObservation::Snapshot.new(1.0, 0, 0.0, 0.0,
  StaticArray[10_u64, 20_u64, 30_u64, 40_u64, 50_u64, 60_u64, 70_u64, 80_u64], 100_u64, 1_u64)
missing_after = WanHostObservation::Snapshot.new(2.0, 0, 0.0, 0.0,
  StaticArray[11_u64, 21_u64, 31_u64, 41_u64, 51_u64, 61_u64, 71_u64, 81_u64], 110_u64, 2_u64)
missing_delta = WanHostObservation.between(missing_before, missing_after)
raise "unavailable sensor groups must produce null deltas" unless missing_delta[:user_ns_delta].nil? && missing_delta[:resident_bytes_delta].nil?

decreasing_before = WanHostObservation::Snapshot.new(3.0, 1, 0.0, 0.0,
  StaticArray[100_u64, 200_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64], 100_u64, 1_u64)
decreasing_after = WanHostObservation::Snapshot.new(4.0, 1, 0.0, 0.0,
  StaticArray[99_u64, 201_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64], 90_u64, 2_u64)
decreasing_delta = WanHostObservation.between(decreasing_before, decreasing_after)
raise "decreasing cumulative counters must produce null deltas" unless decreasing_delta[:user_ns_delta].nil?
raise "increasing cumulative counters must remain visible" unless decreasing_delta[:system_ns_delta] == 1_u64

gauge_before = WanHostObservation::Snapshot.new(5.0, 4, 0.0, 0.0,
  StaticArray[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 70_u64, 80_u64], 100_u64, 1_u64)
gauge_after = WanHostObservation::Snapshot.new(6.0, 4, 0.0, 0.0,
  StaticArray[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 60_u64, 90_u64], 110_u64, 2_u64)
gauge_delta = WanHostObservation.between(gauge_before, gauge_after)
raise "decreasing resident-memory gauges must remain visible" unless gauge_delta[:resident_bytes_delta] == -10_i64
raise "increasing footprint gauges must remain visible" unless gauge_delta[:footprint_bytes_delta] == 10_i64

puts({schema: "wan_host_observation_probe_v1", status: "passed",
      elapsed_seconds: end_seconds - begin_seconds, loop_iterations: iterations,
      observations: observations,
      missing_mask_negative_control_passed: true,
      decreasing_counter_negative_control_passed: true}.to_json)
