require "spec"
require "../examples/support/pixelumm_ffn_stage_timing_contract"

alias T = PixelUMMFFNStageTimingContract
alias G = PixelUMMFFNGraphProbe
alias I = PixelUMMFFNIsolatedProbe

private def pixelumm_ffn_timing_sources : Hash(String, String)
  T::SOURCE_PATHS.to_h do |path|
    {File.realpath(path), "a" * 64}
  end
end

private def pixelumm_ffn_timing_parity_report(sources : Hash(String, String)) : JSON::Any
  stage_counts = G.expected_stage_counts(G::Topology::StageOutputs)
  stages = G::STAGE_NAMES.map_with_index do |name, index|
    count = stage_counts[index].to_i64
    if index == G::STAGE_NAMES.size - 1
      {name: name, expected_elements: count, failures: 1, different_words: 0,
       nonfinite: 1, elements: count, passed: true}
    else
      {name: name, expected_elements: count, failures: 0, different_words: 0,
       nonfinite: 0, elements: count, passed: true}
    end
  end
  operation_counts = G.operation_count_report(
    G.expected_candidate_counts(G::Topology::StageOutputs))
  JSON.parse({
    schema: G::SCHEMA,
    pass: true, probe_completed: true, numerical_pass: true,
    execution_pass: true, production_promoted: false,
    full_generation_executed: false, correctness_only: true,
    fixture: {
      path: "/fixture/reference.json",
      metadata_sha256: "b" * 64, payload_sha256: "c" * 64,
      weight_cache_metadata_sha256: "d" * 64,
      weight_cache_payload_sha256: "e" * 64,
      input_window: "w0", block: 0,
      normalized_input_sha256: "f" * 64,
      input_provenance: "saved_initial_noise_conditioning_upstream_stem",
      model_inference: false,
      upstream_revision: G::B::UPSTREAM_REVISION,
      upstream_model_sha256: G::B::MODEL_SHA256,
    },
    comparison: {atol: 2e-5, rtol: 2e-5},
    protocol: {
      tokens: G::C::TOKENS, hidden: G::C::HIDDEN, ffn: G::C::FFN,
      candidate_topology: "stage_outputs",
      stage_outputs_are_mandatory_targets: true,
      output_only_and_captured_graphs_compiled_separately: true,
      output_only_down_compared_to_saved_and_native_baseline: true,
      output_only_product_compared_to_saved_and_native_baseline_before_capture: true,
      output_only_front_stages_compared_before_capture: true,
      captured_stages_compared_to_saved_and_native_baseline: true,
      no_generation: true, no_promotion: true,
      candidate_timing_samples: 0, baseline_timing_samples: 0,
      terminal_waits: 1, comparator_readback_passes: 1,
      encoded_named_buffer_allocations_allowed: 0,
      preterminal_readbacks_allowed: 0,
      output_only_front_stage_outputs: ["graph.gate", "graph.up", "graph.silu", "graph.product"],
      expected_candidate_graph_operation_counts: operation_counts,
      candidate_graph_operation_count_names: G::GRAPH_OPERATION_NAMES,
    },
    execution: {
      named_bytes: 2_000_000_000_i64,
      maximum_named_bytes: T::NAMED_BUFFER_CAP_BYTES,
      encoding_named_allocation_delta: 0,
      preterminal_readbacks: 0, total_readbacks_after_wait: 1,
      wait_count: 1,
      baseline_operation_counts: [1, 1, 1, 1, 1, 1, 2],
      candidate_graph_operation_counts: operation_counts,
    },
    stages: stages,
    source_sha256: sources,
  }.to_json)
end

private def pixelumm_ffn_timing_metrics : Array(G::Metric)
  metrics = Array(G::Metric).new(T::METRIC_SLOTS) do
    {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32, elements: 0_u32}
  end
  10.times do |sample|
    metrics[sample * 2] = {failures: 0_u32, different_words: 0_u32,
      nonfinite: 0_u32, elements: G::C::PRODUCT_ELEMENTS.to_u32}
    metrics[sample * 2 + 1] = {failures: 0_u32, different_words: 0_u32,
      nonfinite: 0_u32, elements: G::C::DOWN_ELEMENTS.to_u32}
  end
  raw = (G::C::TOKENS * G::C::FFN).to_u32
  metrics[54] = {failures: 0_u32, different_words: 17_u32, nonfinite: 0_u32, elements: raw}
  metrics[55] = {failures: 0_u32, different_words: 19_u32, nonfinite: 0_u32, elements: raw}
  metrics[56] = {failures: 1_u32, different_words: 0_u32,
    nonfinite: 1_u32, elements: G::C::DOWN_ELEMENTS.to_u32}
  metrics
end

private def pixelumm_ffn_timing_boundaries : Array(I::StageTiming)
  T::SCHEDULE.map_with_index do |entry, index|
    first = (index * 2 + 1).to_i32
    {stage: entry[:label], first_root: first, end_root: first + 1,
     encode_wall_seconds: 0.25}
  end
end

private def pixelumm_ffn_timing_roots(overlap_sample : Bool = false) : Array(ML::Video::MPSCommandRootTiming)
  Array.new(21) do |sequence|
    start = sequence.to_f64 * 2.0 + 0.25
    finish = start + 0.5
    if overlap_sample && sequence == 0
      start = 2.0
      finish = 2.5
    end
    ML::Video::MPSCommandRootTiming.new(sequence, true, false, start, finish)
  end
end

describe PixelUMMFFNStageTimingContract do
  it "pins the fixed balanced A/B order and operation counts" do
    T.validate_schedule!(T::SCHEDULE)
    T.validate_baseline_counts!(T::BASELINE_COUNTS)
    T.validate_candidate_counts!(T::CANDIDATE_COUNTS)

    expect_raises(ArgumentError) do
      T.validate_schedule!(T::SCHEDULE.reverse)
    end
    expect_raises(ArgumentError) do
      T.validate_baseline_counts!([641_u64, 641_u64, 641_u64, 641_u64,
        641_u64, 640_u64, 2_u64])
    end
    expect_raises(ArgumentError) do
      T.validate_candidate_counts!([640_u64, 640_u64, 0_u64, 0_u64, 1_u64])
    end
  end

  it "requires every timed product/down comparison, zero unused coverage, and exact controls" do
    metrics = pixelumm_ffn_timing_metrics
    T.validate_metrics!(metrics).should be_empty

    missing = metrics.dup
    missing[4] = {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32, elements: 0_u32}
    T.validate_metrics!(missing).should_not be_empty

    used = metrics.dup
    used[53] = {failures: 1_u32, different_words: 1_u32, nonfinite: 0_u32, elements: 1_u32}
    T.validate_metrics!(used).should_not be_empty

    bad_control = metrics.dup
    bad_control[56] = {failures: 0_u32, different_words: 0_u32,
      nonfinite: 0_u32, elements: G::C::DOWN_ELEMENTS.to_u32}
    T.validate_metrics!(bad_control).should_not be_empty
  end

  it "conserves isolated per-sample GPU roots and rejects overlap outside a sample" do
    timings = pixelumm_ffn_timing_boundaries
    records = T.build_sample_timings(timings, pixelumm_ffn_timing_roots)
    records.size.should eq(10)
    records.map(&.[:mode]).should eq(["A", "B", "A", "B", "B", "A", "B", "A", "A", "B"])
    records.map(&.[:warmup]).should eq([true, true, false, false, false, false, false, false, false, false])
    records.each do |record|
      record[:repeats].should eq(128)
      record[:gpu_root_union_seconds].should eq(0.5)
      record[:outside_root_overlap_seconds].should eq(0.0)
    end

    expect_raises(ArgumentError) do
      T.build_sample_timings(timings, pixelumm_ffn_timing_roots(overlap_sample: true))
    end
    expect_raises(ArgumentError) do
      invalid = timings.dup
      invalid[1] = invalid[0]
      T.build_sample_timings(invalid, pixelumm_ffn_timing_roots)
    end
  end

  it "accepts only a positive, ordered Mach/UTC anchor with a sub-millisecond bracket" do
    T.validate_clock_anchor!(100_i64, 1_800_000_000_000_000_000_i64, 500_100_i64)
    expect_raises(ArgumentError) { T.validate_clock_anchor!(500_i64, 1000_i64, 499_i64) }
    expect_raises(ArgumentError) { T.validate_clock_anchor!(500_i64, 0_i64, 600_i64) }
    expect_raises(ArgumentError) do
      T.validate_clock_anchor!(1_i64, 2_i64, 1_000_002_i64)
    end
  end

  it "validates the live 8 GiB, 10 percent, 64 MiB, 600 second guard identity" do
    records = [
      JSON.parse({event: "limits", runtime_floor_pct: 10,
        runtime_floor_pct_exact: "10", max_footprint_bytes: 8_i64 * 1024 * 1024 * 1024,
        max_swap_growth_bytes: 64_i64 * 1024 * 1024, timeout_sec: 600}.to_json),
      JSON.parse({event: "sample"}.to_json),
      JSON.parse({event: "started", pid: 77, pgid: 77}.to_json),
    ]
    T.validate_guard_records!(records, 77_i64)["max_footprint_bytes"].as_i64.should eq(8_i64 * 1024 * 1024 * 1024)

    expect_raises(ArgumentError) { T.validate_guard_records!(records, 78_i64) }
    expect_raises(ArgumentError) do
      shifted = records.dup
      shifted[0] = JSON.parse({event: "limits", runtime_floor_pct: 10,
        runtime_floor_pct_exact: "10", max_footprint_bytes: 8_i64 * 1024 * 1024 * 1024,
        max_swap_growth_bytes: 64_i64 * 1024 * 1024, timeout_sec: 601}.to_json)
      T.validate_guard_records!(shifted, 77_i64)
    end
    expect_raises(ArgumentError) do
      T.validate_guard_records!(records + [JSON.parse({event: "stopping"}.to_json)], 77_i64)
    end
  end

  it "requires the passed stage_outputs correctness report to match current sources and fixtures" do
    sources = pixelumm_ffn_timing_sources
    report = pixelumm_ffn_timing_parity_report(sources)
    T.validate_stage_outputs_parity_report!(report, sources,
      "/fixture/reference.json", "b" * 64, "c" * 64, "d" * 64,
      "e" * 64, "f" * 64).should be_nil

    expect_raises(ArgumentError) do
      T.validate_stage_outputs_parity_report!(report, sources,
        "/fixture/other.json", "b" * 64, "c" * 64, "d" * 64,
        "e" * 64, "f" * 64)
    end
    expect_raises(ArgumentError) do
      wrong_tolerance = JSON.parse(report.to_json)
      wrong_tolerance.as_h["comparison"].as_h["rtol"] = JSON::Any.new(1e-4)
      T.validate_stage_outputs_parity_report!(wrong_tolerance, sources,
        "/fixture/reference.json", "b" * 64, "c" * 64, "d" * 64,
        "e" * 64, "f" * 64)
    end
    expect_raises(ArgumentError) do
      stale = sources.dup
      stale[stale.keys.first] = "0" * 64
      T.validate_stage_outputs_parity_report!(report, stale,
        "/fixture/reference.json", "b" * 64, "c" * 64, "d" * 64,
        "e" * 64, "f" * 64)
    end
    expect_raises(ArgumentError) do
      incomplete = JSON.parse(report.to_json)
      incomplete["stages"].as_a.pop
      T.validate_stage_outputs_parity_report!(incomplete, sources,
        "/fixture/reference.json", "b" * 64, "c" * 64, "d" * 64,
        "e" * 64, "f" * 64)
    end
  end
end
