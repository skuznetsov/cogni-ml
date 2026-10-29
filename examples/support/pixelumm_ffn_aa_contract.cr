require "json"

# CPU-only contract and analysis for the fixed J5760 connected baseline A/A
# calibration. Root isolation is deliberately absent from this report: callers
# may claim it only after independently checking the native root intervals.
module PixelUMMFFNAaContract
  extend self

  SCHEMA = "cogni_ml.pixelumm_ffn_aa_baseline/v1"
  SCOPE  = "J5760_connected_baseline_aa"

  TOKENS =  5760
  HIDDEN =  4096
  FFN    = 12288

  WARMUPS       =  2
  TIMED_SAMPLES = 16
  TOTAL_SAMPLES = WARMUPS + TIMED_SAMPLES

  CHECKS_PER_SAMPLE           = 3
  PRODUCT_CHECK_OFFSET        = 0
  DOWN_REFERENCE_CHECK_OFFSET = 1
  DOWN_CANONICAL_CHECK_OFFSET = 2

  NORMAL_METRIC_SLOTS     = TOTAL_SAMPLES * CHECKS_PER_SAMPLE
  RAW_GATE_METRIC_SLOT    = NORMAL_METRIC_SLOTS
  RAW_UP_METRIC_SLOT      = RAW_GATE_METRIC_SLOT + 1
  NAN_CONTROL_METRIC_SLOT = RAW_UP_METRIC_SLOT + 1
  TOTAL_METRIC_SLOTS      = NAN_CONTROL_METRIC_SLOT + 1

  PRODUCT_ELEMENTS = TOKENS * FFN
  RAW_FFN_ELEMENTS = TOKENS * FFN
  DOWN_ELEMENTS    = TOKENS * HIDDEN

  alias Metric = NamedTuple(failures: UInt32, different_words: UInt32, nonfinite: UInt32, elements: UInt32)

  record ScheduleEntry, index : Int32, warmup : Bool, capture_raw : Bool do
    include JSON::Serializable
  end

  record Summary,
    schema : String,
    scope : String,
    sample_count : Int32,
    sample_seconds : Array(Float64),
    median_seconds : Float64,
    min_seconds : Float64,
    max_seconds : Float64,
    p90_seconds : Float64,
    mad_seconds : Float64,
    mad_over_median : Float64,
    first4_median_seconds : Float64,
    last4_median_seconds : Float64,
    last4_over_first4_median : Float64,
    adjacent_pair_later_over_earlier : Array(Float64) do
    include JSON::Serializable
  end

  def schedule : Array(ScheduleEntry)
    Array.new(TOTAL_SAMPLES) do |index|
      ScheduleEntry.new(index.to_i32, index < WARMUPS, index == 0)
    end
  end

  def validate_schedule!(entries : Array(ScheduleEntry)) : Nil
    unless entries.size == TOTAL_SAMPLES
      raise ArgumentError.new("A/A schedule must contain exactly #{TOTAL_SAMPLES} baseline samples")
    end

    entries.each_with_index do |entry, position|
      warmup = position < WARMUPS
      capture_raw = position == 0
      unless entry.index == position && entry.warmup == warmup && entry.capture_raw == capture_raw
        raise ArgumentError.new("A/A schedule differs from the fixed baseline order at sample #{position}")
      end
    end
  end

  def product_metric_slot(sample_index : Int32) : Int32
    sample_metric_slot(sample_index, PRODUCT_CHECK_OFFSET)
  end

  def down_reference_metric_slot(sample_index : Int32) : Int32
    sample_metric_slot(sample_index, DOWN_REFERENCE_CHECK_OFFSET)
  end

  def down_canonical_metric_slot(sample_index : Int32) : Int32
    sample_metric_slot(sample_index, DOWN_CANONICAL_CHECK_OFFSET)
  end

  def sample_metric_slot(sample_index : Int32, check_offset : Int32) : Int32
    unless (0...TOTAL_SAMPLES).includes?(sample_index)
      raise ArgumentError.new("A/A sample index is outside the fixed schedule")
    end
    unless (0...CHECKS_PER_SAMPLE).includes?(check_offset)
      raise ArgumentError.new("A/A metric check offset is outside the fixed sample layout")
    end

    sample_index * CHECKS_PER_SAMPLE + check_offset
  end

  # Validate all GPU-reported counts after the single terminal wait. The three
  # per-sample slots are product-vs-oracle, down-vs-oracle, and exact canonical
  # down words. Raw gate/up checks are captured once at sample 0; the final
  # slot is an intentional NaN comparator control over the full down tensor.
  def validate_metrics!(metrics : Array(Metric)) : Nil
    unless metrics.size == TOTAL_METRIC_SLOTS
      raise ArgumentError.new("A/A metric vector must contain exactly #{TOTAL_METRIC_SLOTS} slots")
    end

    NORMAL_METRIC_SLOTS.times do |slot|
      metric = metrics[slot]
      offset = slot % CHECKS_PER_SAMPLE
      expected_elements = offset == PRODUCT_CHECK_OFFSET ? PRODUCT_ELEMENTS : DOWN_ELEMENTS
      validate_active_finite_metric!(metric, expected_elements.to_u32, "normal slot #{slot}")

      if offset == DOWN_CANONICAL_CHECK_OFFSET && metric[:different_words] != 0
        raise ArgumentError.new("canonical down metric slot #{slot} is not bitwise equal")
      end
    end

    validate_active_finite_metric!(metrics[RAW_GATE_METRIC_SLOT], RAW_FFN_ELEMENTS.to_u32, "raw gate capture")
    validate_active_finite_metric!(metrics[RAW_UP_METRIC_SLOT], RAW_FFN_ELEMENTS.to_u32, "raw up capture")

    control = metrics[NAN_CONTROL_METRIC_SLOT]
    unless control == {failures: 1_u32, different_words: 0_u32, nonfinite: 1_u32, elements: DOWN_ELEMENTS.to_u32}
      raise ArgumentError.new("NaN comparator control must report one injected nonfinite failure over the down tensor")
    end
  end

  def median(values : Array(Float64)) : Float64
    raise ArgumentError.new("median requires at least one value") if values.empty?
    sorted = values.sort
    middle = sorted.size // 2
    if sorted.size.odd?
      sorted[middle]
    else
      (sorted[middle - 1] + sorted[middle]) / 2.0
    end
  end

  # p90 is linearly interpolated at rank (n - 1) * 0.9. Pair ratios use each
  # later sample divided by its immediately preceding sample.
  def summarize(sample_seconds : Array(Float64)) : Summary
    unless sample_seconds.size == TIMED_SAMPLES
      raise ArgumentError.new("A/A analysis requires exactly #{TIMED_SAMPLES} timed samples")
    end
    unless sample_seconds.all? { |seconds| seconds.finite? && seconds > 0.0 }
      raise ArgumentError.new("A/A sample times must be positive and finite")
    end

    center = median(sample_seconds)
    deviations = sample_seconds.map { |seconds| (seconds - center).abs }
    mad = median(deviations)
    first4 = sample_seconds[0, 4]
    last4 = sample_seconds[TIMED_SAMPLES - 4, 4]
    pairs = Array.new(TIMED_SAMPLES // 2) do |pair|
      earlier = sample_seconds[pair * 2]
      later = sample_seconds[pair * 2 + 1]
      later / earlier
    end

    Summary.new(
      SCHEMA,
      SCOPE,
      sample_seconds.size.to_i32,
      sample_seconds.dup,
      center,
      sample_seconds.min,
      sample_seconds.max,
      percentile(sample_seconds, 0.9),
      mad,
      mad / center,
      median(first4),
      median(last4),
      median(last4) / median(first4),
      pairs,
    )
  end

  private def validate_active_finite_metric!(metric : Metric, expected_elements : UInt32, label : String) : Nil
    unless metric[:elements] == expected_elements && metric[:failures] == 0 && metric[:nonfinite] == 0 &&
           metric[:different_words] <= metric[:elements]
      raise ArgumentError.new("#{label} comparator metric is inactive, malformed, or failed")
    end
  end

  private def percentile(values : Array(Float64), fraction : Float64) : Float64
    sorted = values.sort
    position = (sorted.size - 1) * fraction
    lower = position.floor.to_i
    upper = position.ceil.to_i
    weight = position - lower
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
  end
end
