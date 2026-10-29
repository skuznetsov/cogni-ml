require "spec"
require "../examples/support/pixelumm_ffn_aa_contract"

private alias A = PixelUMMFFNAaContract

private def aa_metric(
  failures : UInt32 = 0_u32,
  different_words : UInt32 = 0_u32,
  nonfinite : UInt32 = 0_u32,
  elements : UInt32 = 0_u32,
) : A::Metric
  {failures: failures, different_words: different_words, nonfinite: nonfinite, elements: elements}
end

private def aa_valid_metrics : Array(A::Metric)
  metrics = Array.new(A::TOTAL_METRIC_SLOTS) do |slot|
    elements = if slot < A::NORMAL_METRIC_SLOTS
                 slot % A::CHECKS_PER_SAMPLE == A::PRODUCT_CHECK_OFFSET ? A::PRODUCT_ELEMENTS : A::DOWN_ELEMENTS
               elsif slot < A::NAN_CONTROL_METRIC_SLOT
                 A::RAW_FFN_ELEMENTS
               else
                 A::DOWN_ELEMENTS
               end
    aa_metric(elements: elements.to_u32)
  end
  metrics[A::NAN_CONTROL_METRIC_SLOT] = aa_metric(failures: 1_u32, nonfinite: 1_u32,
    elements: A::DOWN_ELEMENTS.to_u32)
  metrics
end

describe PixelUMMFFNAaContract do
  it "defines the fixed J5760 baseline-only schedule and comparator layout" do
    schedule = A.schedule
    schedule.size.should eq(18)
    schedule.map(&.index).should eq((0...18).map(&.to_i32))
    schedule.map(&.warmup).should eq([true, true] + Array.new(16, false))
    schedule.map(&.capture_raw).should eq([true] + Array.new(17, false))
    A.validate_schedule!(schedule)

    A::NORMAL_METRIC_SLOTS.should eq(54)
    A::RAW_GATE_METRIC_SLOT.should eq(54)
    A::RAW_UP_METRIC_SLOT.should eq(55)
    A::NAN_CONTROL_METRIC_SLOT.should eq(56)
    A::TOTAL_METRIC_SLOTS.should eq(57)
    A.product_metric_slot(0).should eq(0)
    A.down_reference_metric_slot(17).should eq(52)
    A.down_canonical_metric_slot(17).should eq(53)
  end

  it "rejects schedule mutations and the wrong sample count" do
    schedule = A.schedule
    wrong_index = schedule.dup
    wrong_index[7] = A::ScheduleEntry.new(8_i32, false, false)
    expect_raises(ArgumentError) { A.validate_schedule!(wrong_index) }

    wrong_warmup = schedule.dup
    wrong_warmup[1] = A::ScheduleEntry.new(1_i32, false, false)
    expect_raises(ArgumentError) { A.validate_schedule!(wrong_warmup) }

    wrong_capture = schedule.dup
    wrong_capture[0] = A::ScheduleEntry.new(0_i32, true, false)
    expect_raises(ArgumentError) { A.validate_schedule!(wrong_capture) }

    expect_raises(ArgumentError) { A.validate_schedule!(schedule[0, 17]) }
    expect_raises(ArgumentError) { A.product_metric_slot(-1_i32) }
    expect_raises(ArgumentError) { A.sample_metric_slot(0_i32, 3_i32) }
  end

  it "accepts the full comparator vector and pins geometry, raw capture, and exact down parity" do
    metrics = aa_valid_metrics
    metrics[0] = aa_metric(different_words: 3_u32, elements: A::PRODUCT_ELEMENTS.to_u32)
    metrics[1] = aa_metric(different_words: 2_u32, elements: A::DOWN_ELEMENTS.to_u32)
    metrics[A::RAW_GATE_METRIC_SLOT] = aa_metric(different_words: 5_u32, elements: A::RAW_FFN_ELEMENTS.to_u32)
    metrics[A::RAW_UP_METRIC_SLOT] = aa_metric(different_words: 7_u32, elements: A::RAW_FFN_ELEMENTS.to_u32)
    A.validate_metrics!(metrics)
  end

  it "rejects wrong counts, nonfinite values, inactive comparators, and a non-bitwise canonical down" do
    wrong_count = aa_valid_metrics
    wrong_count[0] = aa_metric(elements: (A::PRODUCT_ELEMENTS - 1).to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(wrong_count) }

    nonfinite = aa_valid_metrics
    nonfinite[1] = aa_metric(failures: 1_u32, different_words: 1_u32, nonfinite: 1_u32,
      elements: A::DOWN_ELEMENTS.to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(nonfinite) }

    inactive = aa_valid_metrics
    inactive[0] = aa_metric
    expect_raises(ArgumentError) { A.validate_metrics!(inactive) }

    noncanonical = aa_valid_metrics
    noncanonical[A.down_canonical_metric_slot(0_i32)] = aa_metric(different_words: 1_u32,
      elements: A::DOWN_ELEMENTS.to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(noncanonical) }

    wrong_raw_count = aa_valid_metrics
    wrong_raw_count[A::RAW_UP_METRIC_SLOT] = aa_metric(elements: (A::RAW_FFN_ELEMENTS - 1).to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(wrong_raw_count) }

    short_metrics = aa_valid_metrics[0, A::TOTAL_METRIC_SLOTS - 1]
    expect_raises(ArgumentError) { A.validate_metrics!(short_metrics) }
  end

  it "requires the NaN comparator control to fail exactly once with one nonfinite word" do
    inactive = aa_valid_metrics
    inactive[A::NAN_CONTROL_METRIC_SLOT] = aa_metric
    expect_raises(ArgumentError) { A.validate_metrics!(inactive) }

    wrong_control = aa_valid_metrics
    wrong_control[A::NAN_CONTROL_METRIC_SLOT] = aa_metric(
      failures: 1_u32, nonfinite: 0_u32, elements: A::DOWN_ELEMENTS.to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(wrong_control) }

    counted_after_injection = aa_valid_metrics
    counted_after_injection[A::NAN_CONTROL_METRIC_SLOT] = aa_metric(
      failures: 1_u32, different_words: 1_u32, nonfinite: 1_u32, elements: A::DOWN_ELEMENTS.to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(counted_after_injection) }

    too_many_failures = aa_valid_metrics
    too_many_failures[A::NAN_CONTROL_METRIC_SLOT] = aa_metric(
      failures: 2_u32, nonfinite: 1_u32, elements: A::DOWN_ELEMENTS.to_u32)
    expect_raises(ArgumentError) { A.validate_metrics!(too_many_failures) }
  end

  it "summarizes exactly 16 positive finite timed values without asserting root isolation" do
    values = (1..16).map(&.to_f64)
    summary = A.summarize(values)

    summary.schema.should eq(A::SCHEMA)
    summary.scope.should eq(A::SCOPE)
    summary.sample_count.should eq(16)
    summary.sample_seconds.should eq(values)
    summary.median_seconds.should eq(8.5)
    summary.min_seconds.should eq(1.0)
    summary.max_seconds.should eq(16.0)
    summary.p90_seconds.should eq(14.5)
    summary.mad_seconds.should eq(4.0)
    summary.mad_over_median.should be_close(4.0 / 8.5, 1e-12)
    summary.first4_median_seconds.should eq(2.5)
    summary.last4_median_seconds.should eq(14.5)
    summary.last4_over_first4_median.should be_close(14.5 / 2.5, 1e-12)
    summary.adjacent_pair_later_over_earlier.size.should eq(8)
    summary.adjacent_pair_later_over_earlier[0].should eq(2.0)
    summary.adjacent_pair_later_over_earlier[7].should be_close(16.0 / 15.0, 1e-12)

    json = JSON.parse(summary.to_json)
    json["sample_count"].as_i.should eq(16)
    json["adjacent_pair_later_over_earlier"].as_a.size.should eq(8)
    json.as_h.has_key?("root_isolated").should be_false
    json.as_h.has_key?("root_isolation_verified").should be_false
  end

  it "rejects wrong timed-sample count and nonfinite or nonpositive values" do
    expect_raises(ArgumentError) { A.summarize(Array.new(15, 1.0)) }

    nan = Array.new(16, 1.0)
    nan[4] = Float64::NAN
    expect_raises(ArgumentError) { A.summarize(nan) }

    infinity = Array.new(16, 1.0)
    infinity[4] = Float64::INFINITY
    expect_raises(ArgumentError) { A.summarize(infinity) }

    zero = Array.new(16, 1.0)
    zero[4] = 0.0
    expect_raises(ArgumentError) { A.summarize(zero) }
  end
end
