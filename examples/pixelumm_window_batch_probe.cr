require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_block_probe"
require "./support/pixelumm_memory_probe"
require "./support/pixelumm_window_batch_probe"

reference, reference_sha, report = "", "", ""
budget = 0_i64
timed = false
OptionParser.parse do |parser|
  parser.banner = "Independent-window B2 attention, full-J block; external 10% memory guard required."
  parser.on("--reference PATH", "Authenticated full-J single-block fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata digest") { |v| reference_sha = v }
  parser.on("--report PATH", "Fresh absolute result path") { |v| report = v }
  parser.on("--max-named-bytes N", "Total baseline plus candidate named budget, at most 8 GiB") { |v| budget = v.to_i64 }
  parser.on("--timed", "Run ABBA/BAAB full-block timing only after all numerical gates pass") { timed = true }
end
unless ARGV.empty? && {reference, reference_sha, report}.all? { |v| !v.empty? } &&
       Path[report].absolute? && 0 < budget <= 8_i64 * 1024**3
  raise ArgumentError.new("expected pinned full-J reference, fresh absolute report and bounded memory")
end
fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("window batch report must be fresh and writable") if fd < 0
report_io = IO::FileDescriptor.new(fd)
observer = PixelUMMMemoryProbe::Recorder.system
session = nil.as(PixelUMMBlockProbe::Session?)
selection = nil.as(PixelUMMWindowBatchProbe::Selection?)
chain = nil.as(ML::Metal::MPSCommandChain?)
terminal_wait_attempted = false
quarantined = false
passed = false
failure = nil.as(String?)
requests = [] of JSON::Any
baseline_named, candidate_named = 0_i64, 0_i64
fixture_input_sha = ""
capture_order = [] of String
schedule = [
  {kind: "positive_comparator_control", selected: false, check: 3, compare: true, poison: true},
  {kind: "raw_qk_gate", selected: false, check: 1, compare: false, poison: false},
  {kind: "same_p_pv_gate", selected: false, check: 2, compare: false, poison: false},
  {kind: "connected_candidate_gate", selected: true, check: 0, compare: true, poison: true},
  {kind: "connected_baseline_gate", selected: false, check: 0, compare: true, poison: true},
]
if timed
  [false, true].each do |candidate|
    schedule << {kind: "warm", selected: candidate, check: 0, compare: false, poison: false}
  end
  [false, true, true, false, true, false, false, true].each_with_index do |candidate, index|
    schedule << {kind: "timed_#{index}", selected: candidate, check: 0,
                 compare: index == 7, poison: false}
  end
end
begin
  observer.emit("before_metadata")
  fixture = PixelUMMBlockProbe::Fixture.new(reference, reference_sha)
  raise ArgumentError.new("window batch is qualified only at full J") unless fixture.grid == [6, 44, 80]
  fixture_input_sha = fixture.inputs["x"].sha256
  capture_order = fixture.capture_order
  PixelUMMWindowBatchProbe.validate_capture_inventory!(capture_order)
  observer.emit("after_fixture_authentication")
  candidate_named = PixelUMMWindowBatchProbe::EXTRA_PANEL_BYTES + 16_i64
  # Reserve candidate storage before the first model allocation, rather than
  # testing the aggregate only after a large diagnostic pool is already live.
  raise ArgumentError.new("candidate bank exceeds aggregate named budget") unless budget > candidate_named
  session = PixelUMMBlockProbe::Session.new(fixture, budget - candidate_named, ffn_f32: true)
  baseline_named = session.named_bytes
  selection = PixelUMMWindowBatchProbe::Selection.new([session.plan])
  raise "candidate named inventory mismatch" unless selection.named_bytes == candidate_named && baseline_named + candidate_named <= budget
  observer.emit("session_ready", phase_bytes_kind: "aggregate_unique_named_bytes", phase_bytes: baseline_named + candidate_named)
  layout = session.plan.config.layout
  schedule.each_with_index do |arm, request|
    selection.configure!(arm[:selected], arm[:check])
    selection.poison_transients! if arm[:poison]
    # Timing has no CPU tensor diagnostics between its eight requests. Every
    # request retains the same input, buffers, plans and post-wait metadata path.
    GC.collect if request == 5
    observer.emit("request_before_encode", request: request)
    chain = ML::Metal::MPSCommandChain.new
    terminal_wait_attempted = false
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    metrics_reads = selection.metrics_read_count
    start = Time.instant
    session.encode(chain)
    encode_seconds = (Time.instant - start).total_seconds
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    raise "window batch encoded a named allocation, readback or wait" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
    inflight_change_rejected = false
    early_metrics_rejected = false
    if request == 0
      begin
        selection.configure!(true)
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("in flight")
        inflight_change_rejected = true
      end
      begin
        selection.metrics(chain)
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("terminal wait")
        early_metrics_rejected = true
      end
      raise "window batch lifecycle guard was not exercised" unless inflight_change_rejected && early_metrics_rejected && !selection.selected
    end
    terminal_wait_attempted = true
    wait_start = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - wait_start).total_seconds
    execution_seconds = (Time.instant - start).total_seconds
    raise "window batch requires exactly one terminal wait" unless chain.wait_count == 1 && chain.completed?
    metrics = arm[:check] == 0 ? nil : selection.metrics(chain)
    metrics_passed = if words = metrics
                       words[3] == PixelUMMWindowBatchProbe.expected_elements(arm[:check], layout) &&
                         words[2] == 0 && (arm[:check] == 3 ? words[0] > 0 : words[0] == 0)
                     else
                       true
                     end
    segments = chain.segment_count
    gpu_seconds = chain.gpu_elapsed_seconds
    roots = chain.completed_root_timings.map do |root|
      {sequence: root.sequence, completed: root.completed, has_error: root.has_error,
       gpu_start_seconds: root.gpu_start_seconds, gpu_end_seconds: root.gpu_end_seconds}
    end
    observer.emit("request_after_terminal_wait", request: request)
    # Compare before capture reuse. This read is after the sole terminal wait;
    # warm/timed intermediate requests deliberately perform no tensor readback.
    comparisons = arm[:compare] ? session.compare : [] of PixelUMMBlockProbe::Comparison
    capture_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    actual_metrics_reads = selection.metrics_read_count - metrics_reads
    unless capture_reads == (arm[:compare] ? capture_order.size : 0) &&
           actual_metrics_reads == (arm[:check] == 0 ? 0 : 1)
      raise "window batch post-wait readback inventory changed"
    end
    captures_passed = !arm[:compare] || (comparisons.map { |c| c[:stage] } == capture_order && comparisons.all? { |c| c[:finite] && c[:failures] == 0 })
    request_passed = metrics_passed && captures_passed
    requests << JSON.parse({
      request: request, kind: arm[:kind], candidate: arm[:selected], check_mode: arm[:check],
      passed: request_passed, timing_eligible: arm[:kind].starts_with?("timed_"),
      encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds, total_seconds: execution_seconds,
      gpu_elapsed_seconds: gpu_seconds, command_segments: segments, terminal_wait_count: chain.wait_count,
      named_allocations_during_encode: encode_allocations, activation_readbacks_before_terminal_wait: early_reads,
      capture_readbacks_after_terminal_wait: capture_reads,
      metrics_readbacks_after_terminal_wait: actual_metrics_reads,
      stage_metrics: metrics, metrics_layout: ["tolerance_failures", "finite_differences", "nonfinite_pairs", "elements"],
      comparisons: comparisons, transient_poisoned: arm[:poison],
      inflight_change_rejected: inflight_change_rejected, early_metrics_rejected: early_metrics_rejected,
      roots: roots,
    }.to_json)
    STDERR.puts "#{arm[:kind]} candidate=#{arm[:selected]} passed=#{request_passed} seconds=#{execution_seconds} metrics=#{metrics}"
    chain.release
    chain = nil
    unless request_passed
      raise "window batch numerical gate failed at #{arm[:kind]}"
    end
    GC.collect if request < 5
  end
  passed = requests.size == schedule.size
rescue error
  failure = "#{error.class}: #{error.message}"
  STDERR.puts failure
ensure
  begin
    if active = chain
      if terminal_wait_attempted && !active.completed?
        quarantined = true
        raise "failed terminal wait; retaining owners until process exit"
      end
      active.commit_and_wait unless active.completed?
      active.release
    end
    selection.try(&.release)
    session.try(&.release)
  rescue error
    quarantined = true
    passed = false
    failure = "#{failure}; cleanup: #{error.class}: #{error.message}"
  end
  GC.collect
  observer.emit("after_session_release")
  result = {
    schema: "cogni_ml.pixelumm_window_batch_probe/v1", generation: false, passed: passed,
    failure: failure, cleanup_quarantined: quarantined,
    reference: reference, reference_sha256: reference_sha, input_sha256: fixture_input_sha,
    upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
    oracle: "Authenticated saved upstream full-J single-block attention/output; raw operations compare to current native MPS control.",
    capture_order: capture_order,
    grid: [6, 44, 80], block_count: 1, heads: 32, head_dimension: 128,
    atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL,
    baseline_named_bytes: baseline_named, candidate_named_bytes: candidate_named,
    total_unique_named_bytes: baseline_named + candidate_named, named_budget_bytes: budget,
    shared_candidate_panel_bytes: PixelUMMWindowBatchProbe::EXTRA_PANEL_BYTES,
    full_model_qualified: false, production_promoted: false,
    expected_request_count: schedule.size, requests: requests, memory_phases: observer.events,
    timing_contract: "One full block encode through existing terminal wait; no CPU tensor diagnostics between timed requests; ABBA then BAAB; final candidate captures rechecked.",
  }
  report_io.puts result.to_pretty_json
  report_io.close
  puts({report: report, passed: passed, failure: failure}.to_json)
end
raise "PixelUMM window batch qualification failed: #{failure}" unless passed
