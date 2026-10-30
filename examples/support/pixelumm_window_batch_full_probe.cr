require "./pixelumm_window_batch_probe"

module PixelUMMWindowBatchFullProbe
  extend self

  {% if flag?(:darwin) %}
    lib HostClock
      fun clock_ns = clock_gettime_nsec_np(clock_id : UInt32) : UInt64
    end
  {% end %}

  PLAN_COUNT             =     36
  METRICS_BYTES_PER_PLAN = 16_i64
  EXTRA_PANEL_BYTES      = PixelUMMWindowBatchProbe::EXTRA_PANEL_BYTES
  MAX_NAMED_BYTES        = 24_i64 * 1024 * 1024 * 1024

  EXPECTED_BASELINE_NAMED_BYTES = 24_108_145_128_i64
  CANDIDATE_NAMED_BYTES         = EXTRA_PANEL_BYTES + PLAN_COUNT * METRICS_BYTES_PER_PLAN
  EXPECTED_TOTAL_NAMED_BYTES    = EXPECTED_BASELINE_NAMED_BYTES + CANDIDATE_NAMED_BYTES

  record Arm, kind : String, alternate : Bool, selected : Bool, poison : Bool,
    compare : Bool, timing_eligible : Bool

  record Budget, baseline_named_bytes : Int64, candidate_named_bytes : Int64,
    total_named_bytes : Int64, max_named_bytes : Int64

  def host_thread_cpu_ns : UInt64
    {% if flag?(:darwin) %}
      value = HostClock.clock_ns(16_u32) # CLOCK_THREAD_CPUTIME_ID
      raise RuntimeError.new("invalid thread CPU clock sample") if value == 0_u64
      value
    {% else %}
      raise RuntimeError.new("full36 host CPU timing requires macOS")
    {% end %}
  end

  def validate_cli!(report : String, max_named_bytes : Int64, *,
                    timed : Bool, reverse : Bool) : Nil
    raise ArgumentError.new("full36 report path must be absolute") unless Path[report].absolute?
    unless 0 < max_named_bytes <= MAX_NAMED_BYTES
      raise ArgumentError.new("full36 named budget must be positive and at most 24 GiB")
    end
    if reverse && !timed
      raise ArgumentError.new("--reverse requires --timed")
    end
  end

  # The window panel is shared by the 36 wrappers; each plan still owns its
  # separate four-word metrics buffer. Keep this arithmetic CPU-testable.
  def candidate_named_bytes(plan_count : Int32 = PLAN_COUNT) : Int64
    unless plan_count == PLAN_COUNT
      raise ArgumentError.new("full36 window batching requires exactly 36 plans")
    end
    EXTRA_PANEL_BYTES + plan_count.to_i64 * METRICS_BYTES_PER_PLAN
  end

  def admit_candidate_budget!(baseline_named_bytes : Int64,
                              max_named_bytes : Int64) : Budget
    unless baseline_named_bytes == EXPECTED_BASELINE_NAMED_BYTES
      raise ArgumentError.new("full36 baseline named inventory mismatch: expected #{EXPECTED_BASELINE_NAMED_BYTES}, got #{baseline_named_bytes}")
    end
    unless 0 < max_named_bytes <= MAX_NAMED_BYTES
      raise ArgumentError.new("full36 named budget must be positive and at most 24 GiB")
    end
    candidate = candidate_named_bytes
    if baseline_named_bytes > Int64::MAX - candidate
      raise ArgumentError.new("full36 aggregate named-byte total overflows")
    end
    total = baseline_named_bytes + candidate
    unless total == EXPECTED_TOTAL_NAMED_BYTES
      raise ArgumentError.new("full36 aggregate named inventory mismatch")
    end
    unless total <= max_named_bytes
      raise ArgumentError.new("full36 baseline plus B2 candidate exceeds named budget: required #{total}, admitted #{max_named_bytes}")
    end
    Budget.new(baseline_named_bytes, candidate, total, max_named_bytes)
  end

  def schedule(*, timed : Bool, reverse : Bool = false) : Array(Arm)
    if reverse && !timed
      raise ArgumentError.new("reverse schedule requires timed mode")
    end

    gate_modes = [true, false, false, true]
    gate_inputs = [false, false, true, true]
    gate_names = ["gate_a_first", "gate_a_second", "gate_b_first", "gate_b_second"]
    arms = gate_modes.map_with_index do |candidate, index|
      selected = reverse ? !candidate : candidate
      Arm.new(gate_names[index], gate_inputs[index], selected, true, true, false)
    end

    return arms unless timed

    [false, true].each do |candidate|
      selected = reverse ? !candidate : candidate
      mode = selected ? "candidate" : "baseline"
      arms << Arm.new("warm_a_#{mode}", false, selected, false, false, false)
    end

    timed_modes = [false, true, true, false, true, false, false, true]
    timed_modes.each_with_index do |candidate, index|
      selected = reverse ? !candidate : candidate
      arms << Arm.new("timed_#{index}", false, selected, false, index == 7, true)
    end
    arms << Arm.new("final_candidate_tail", false, true, false, true, false)
    arms
  end
end
