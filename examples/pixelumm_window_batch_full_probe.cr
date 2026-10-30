require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_decoder_probe"
require "./support/pixelumm_window_batch_probe"
require "./support/pixelumm_window_batch_full_probe"

reference, reference_sha, alternate, alternate_sha = "", "", "", ""
head_reference, head_reference_sha, head_alternate, head_alternate_sha = "", "", "", ""
cache_path, cache_sha, report = "", "", ""
budget = 0_i64
timed = false
reverse = false
OptionParser.parse do |parser|
  parser.banner = "PixelUMM 36 blocks plus global head, full J, B2 window batch; external 10% cache-aware memory guard required."
  parser.on("--reference PATH", "Authenticated full36 block fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted full36 metadata digest") { |v| reference_sha = v }
  parser.on("--alternate-reference PATH", "Alternate full36 fixture") { |v| alternate = v }
  parser.on("--alternate-reference-sha256 SHA", "Trusted alternate full36 digest") { |v| alternate_sha = v }
  parser.on("--head-reference PATH", "Head fixture linked to the first block35 edge") { |v| head_reference = v }
  parser.on("--head-reference-sha256 SHA", "Trusted head metadata digest") { |v| head_reference_sha = v }
  parser.on("--head-alternate-reference PATH", "Head fixture linked to the alternate block35 edge") { |v| head_alternate = v }
  parser.on("--head-alternate-reference-sha256 SHA", "Trusted alternate head digest") { |v| head_alternate_sha = v }
  parser.on("--bf16-cache PATH", "Authenticated full36 BF16 weight cache") { |v| cache_path = v }
  parser.on("--bf16-cache-sha256 SHA", "Trusted BF16 cache metadata digest") { |v| cache_sha = v }
  parser.on("--report PATH", "Fresh absolute 0600 result path") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate unique named budget, at most 24 GiB") { |v| budget = v.to_i64 }
  parser.on("--timed", "Run warm and ABBA/BAAB timing after all poisoned numerical gates pass") { timed = true }
  parser.on("--reverse", "Flip candidate/baseline assignment across gates, warmups and timed requests; requires --timed") { reverse = true }
end

unless ARGV.empty? && {reference, reference_sha, alternate, alternate_sha,
                       head_reference, head_reference_sha, head_alternate, head_alternate_sha,
                       cache_path, cache_sha, report}.all? { |value| !value.empty? }
  raise ArgumentError.new("expected paired authenticated full36/head fixtures, BF16 cache and report")
end
PixelUMMWindowBatchFullProbe.validate_cli!(report, budget, timed: timed, reverse: reverse)
schedule = PixelUMMWindowBatchFullProbe.schedule(timed: timed, reverse: reverse)

# Create the exclusive report before the memory recorder samples the process.
fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("full36 report must be fresh and writable") if fd < 0
report_io = IO::FileDescriptor.new(fd)
observer = PixelUMMMemoryProbe::Recorder.system

session = nil.as(PixelUMMDecoderProbe::Session?)
selection = nil.as(PixelUMMWindowBatchProbe::Selection?)
chain = nil.as(ML::Metal::MPSCommandChain?)
terminal_wait_attempted = false
terminal_wait_failed = false
cleanup_quarantined = false
passed = false
failure = nil.as(String?)
requests = [] of JSON::Any
capture_order = [] of String
fixture_input_sha = ""
alternate_input_sha = ""
baseline_named = 0_i64
candidate_named = 0_i64
total_named = 0_i64
head_input_alias = false
session_poison_count = 0
selection_change_rejected = false
selection_release_rejected = false
selection_poison_rejected = false
session_release_rejected = false
session_poison_rejected = false
read_rejected = false
budget_admission = nil.as(PixelUMMWindowBatchFullProbe::Budget?)

begin
  observer.emit("before_metadata")
  block_fixture = PixelUMMTwoBlockProbe::Fixture.new(reference, reference_sha, 36)
  block_replay = PixelUMMTwoBlockProbe::Fixture.new(alternate, alternate_sha, 36)
  head_fixture = PixelUMMHeadProbe::Fixture.new(head_reference, head_reference_sha)
  head_replay = PixelUMMHeadProbe::Fixture.new(head_alternate, head_alternate_sha)
  fixture = PixelUMMDecoderProbe::Fixture.new(block_fixture, head_fixture)
  replay = PixelUMMDecoderProbe::Fixture.new(block_replay, head_replay)
  fixture.compatible_replay!(replay)
  cache = PixelUMMTwoBlockProbe::WeightCache.new(cache_path, cache_sha, block_fixture)
  capture_order = fixture.capture_order
  fixture_input_sha = fixture.blocks.inputs["x"].sha256
  alternate_input_sha = replay.blocks.inputs["x"].sha256
  observer.emit("after_metadata_and_pair_compatibility")

  # This CPU-only inventory gate includes the candidate before Session begins
  # uploading fixture/cache tensors or allocating any Metal buffer.
  admitted = PixelUMMDecoderProbe.admission(fixture, PixelUMMWindowBatchFullProbe::MAX_NAMED_BYTES)
  budget_admission = PixelUMMWindowBatchFullProbe.admit_candidate_budget!(admitted.named_bytes, budget)
  baseline_named = admitted.named_bytes
  candidate_named = budget_admission.not_nil!.candidate_named_bytes
  total_named = budget_admission.not_nil!.total_named_bytes
  observer.emit("candidate_aggregate_budget_admitted",
    phase_bytes_kind: "baseline_plus_candidate_unique_named_bytes",
    phase_bytes: total_named)

  session_budget = budget_admission.not_nil!.max_named_bytes - budget_admission.not_nil!.candidate_named_bytes
  session = PixelUMMDecoderProbe::Session.new(fixture, cache,
    max_named_bytes: session_budget, prepare_comparison_scratch: true,
    phase_observer: observer)
  selection = PixelUMMWindowBatchProbe::Selection.new(session.blocks.plans)
  capture_order = session.capture_order
  head_input_alias = session.head_input_aliases_block35_output?
  unless session.blocks.plans.size == 36 && capture_order.size == 43 &&
         head_input_alias &&
         session.head_plan.input.handle == session.blocks.plans[35].output.handle &&
         session.named_bytes == budget_admission.not_nil!.baseline_named_bytes &&
         selection.named_bytes == budget_admission.not_nil!.candidate_named_bytes &&
         session.named_bytes + selection.named_bytes == budget_admission.not_nil!.total_named_bytes &&
         session.blocks.plans.all? { |plan| plan.ffn_f32 && !plan.fused_kv_gather && !plan.scale_softmax }
    raise "full36/head inventory, direct edge, baseline graph mode, or candidate named-byte admission mismatch"
  end
  observer.emit("session_ready", phase_bytes_kind: "aggregate_unique_named_bytes",
    phase_bytes: session.named_bytes + selection.named_bytes)
  STDERR.puts "PixelUMM B2 full36 admitted #{session.named_bytes + selection.named_bytes} unique named bytes, full J=21120"

  current_alternate = false
  schedule.each_with_index do |arm, request|
    if arm.alternate != current_alternate
      target = arm.alternate ? replay : fixture
      session.replay_input!(target)
      current_alternate = arm.alternate
      observer.emit("after_replay_input", request: request)
    end

    selection.configure!(arm.selected, 0)
    raise "B2 stage-check mode must remain disabled" unless selection.check_mode == 0
    poisoned_session_bytes = 0_i64
    if arm.poison
      selection.poison_transients!
      poisoned_session_bytes = session.poison_transients!
      session_poison_count += 1
      raise "full36 poison omitted session/head transient storage" unless poisoned_session_bytes > 0
      observer.emit("after_poisoned_replay_input", request: request,
        phase_bytes_kind: "session_transient_poison_bytes", phase_bytes: poisoned_session_bytes)
    end

    observer.emit("request_before_encode", request: request)
    chain = ML::Metal::MPSCommandChain.new
    terminal_wait_attempted = false
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    metrics_reads = selection.metrics_read_count
    encode_start_mach_ns = PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!
    execution_start = Time.instant
    encode_start = Time.instant
    request_cpu_start_ns = PixelUMMWindowBatchFullProbe.host_thread_cpu_ns
    encode_cpu_start_ns = request_cpu_start_ns
    session.encode(chain)
    encode_seconds = (Time.instant - encode_start).total_seconds
    encode_cpu_end_ns = PixelUMMWindowBatchFullProbe.host_thread_cpu_ns
    encode_end_mach_ns = PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0 &&
           session.head_input_aliases_block35_output?
      raise "full36/head encode allocated a named buffer, read an activation, waited, or copied block35 output"
    end

    if request == 0
      original_selected = selection.selected
      begin
        selection.configure!(!arm.selected, 0)
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("in flight")
        selection_change_rejected = true
      end
      unless selection_change_rejected && selection.selected == original_selected
        raise "in-flight B2 selection rejection did not preserve the live bank"
      end

      begin
        selection.poison_transients!
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("in flight")
        selection_poison_rejected = true
      end
      begin
        session.poison_transients!
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("in flight")
        session_poison_rejected = true
      end
      begin
        session.compare
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("terminal wait")
        read_rejected = true
      end
      begin
        selection.release
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("in flight")
        selection_release_rejected = true
      end
      begin
        session.release
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("in flight")
        session_release_rejected = true
      end
      unless selection_change_rejected && selection_poison_rejected && session_poison_rejected &&
             read_rejected && selection_release_rejected && session_release_rejected &&
             selection.selected == original_selected &&
             ML::MetalBuffer.hy_text_probe_readbacks == reads &&
             ML::MetalBuffer.hy_text_probe_allocations == allocations && chain.wait_count == 0
        raise "full36/head in-flight lifecycle probe admitted selection, release, poison, read, allocation, or wait"
      end
    end

    wait_start = Time.instant
    wait_cpu_start_ns = PixelUMMWindowBatchFullProbe.host_thread_cpu_ns
    terminal_wait_attempted = true
    begin
      chain.commit_and_wait
    rescue error
      terminal_wait_failed = true
      raise error
    end
    wait_seconds = (Time.instant - wait_start).total_seconds
    wait_cpu_end_ns = PixelUMMWindowBatchFullProbe.host_thread_cpu_ns
    total_seconds = (Time.instant - execution_start).total_seconds
    request_cpu_end_ns = wait_cpu_end_ns
    terminal_wait_end_mach_ns = PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!
    observer.emit("request_after_terminal_wait", request: request)
    unless chain.wait_count == 1 && chain.completed?
      raise "full36/head request requires exactly one completed terminal wait"
    end

    segments = chain.segment_count
    roots = chain.completed_root_timings
    unless segments > 0 && roots.size == segments &&
           roots.map(&.sequence) == (0...segments).to_a &&
           roots.all? { |root| root.completed && !root.has_error &&
           root.gpu_start_seconds.finite? && root.gpu_end_seconds.finite? &&
           root.gpu_start_seconds >= 0.0 && root.gpu_end_seconds > root.gpu_start_seconds }
      raise "full36/head terminal root inventory is incomplete or contains an error"
    end
    roots_json = roots.map do |root|
      {sequence: root.sequence, completed: root.completed, has_error: root.has_error,
       gpu_start_seconds: root.gpu_start_seconds, gpu_end_seconds: root.gpu_end_seconds}
    end
    gpu_seconds = chain.gpu_elapsed_seconds
    metrics_read_delta = selection.metrics_read_count - metrics_reads
    raise "B2 stage metrics were read despite check_mode=0" unless metrics_read_delta == 0

    # Tensor comparisons only occur for the four poisoned gates, the final
    # timed request, and the untimed candidate tail. Intermediate timed
    # requests retain the same metadata path without any tensor readback.
    comparisons = arm.compare ? session.compare(request: request) : [] of PixelUMMDecoderProbe::Comparison
    capture_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    expected_capture_reads = arm.compare ? capture_order.size : 0
    unless capture_reads == expected_capture_reads && capture_order.size == 43
      raise "full36/head post-wait capture readback inventory changed"
    end
    comparisons_passed = !arm.compare ||
                         (comparisons.map { |comparison| comparison[:stage] } == capture_order &&
                          comparisons.all? { |comparison| comparison[:finite] && comparison[:failures] == 0 })
    request_passed = comparisons_passed
    request_input_sha = session.blocks.fixture.inputs["x"].sha256
    requests << JSON.parse({
      request: request, kind: arm.kind, alternate_input: arm.alternate,
      input_sha256: request_input_sha, candidate: arm.selected,
      check_mode: selection.check_mode, passed: request_passed,
      timing_eligible: arm.timing_eligible,
      encode_seconds: encode_seconds,
      encode_host_cpu_seconds: (encode_cpu_end_ns - encode_cpu_start_ns).to_f64 / 1e9,
      terminal_wait_seconds: wait_seconds,
      terminal_wait_host_cpu_seconds: (wait_cpu_end_ns - wait_cpu_start_ns).to_f64 / 1e9,
      total_seconds: total_seconds,
      total_host_cpu_seconds: (request_cpu_end_ns - request_cpu_start_ns).to_f64 / 1e9,
      encode_start_mach_ns: encode_start_mach_ns, encode_end_mach_ns: encode_end_mach_ns,
      terminal_wait_end_mach_ns: terminal_wait_end_mach_ns,
      guard_clock: "mach_absolute_time",
      gpu_elapsed_seconds: gpu_seconds, command_segments: segments,
      root_count: roots.size, roots: roots_json,
      terminal_wait_count: chain.wait_count, named_allocations_during_encode: encode_allocations,
      activation_readbacks_before_terminal_wait: early_reads,
      capture_readbacks_after_terminal_wait: capture_reads,
      metrics_readbacks_after_terminal_wait: metrics_read_delta,
      comparisons: comparisons, poisoned: arm.poison,
      session_transient_poison_bytes: poisoned_session_bytes,
      head_input_is_block35_output: session.head_input_aliases_block35_output?,
      early_lifecycle_rejections: request == 0 ? {
        selection_change_rejected:  selection_change_rejected,
        selection_release_rejected: selection_release_rejected,
        selection_poison_rejected:  selection_poison_rejected,
        session_release_rejected:   session_release_rejected,
        session_poison_rejected:    session_poison_rejected,
        read_rejected:              read_rejected,
      } : nil,
    }.to_json)
    STDERR.puts "#{arm.kind}: candidate=#{arm.selected} passed=#{request_passed} total=#{total_seconds}s encode=#{encode_seconds}s wait=#{wait_seconds}s"

    chain.release
    chain = nil
    observer.emit("after_chain_release", request: request)
    unless request_passed
      raise "full36/head numerical gate failed at #{arm.kind}"
    end
  end
  passed = requests.size == schedule.size && requests.all? { |request| request["passed"].as_bool }
  failure ||= "full36/head schedule did not complete" unless passed
rescue error
  passed = false
  failure = "#{error.class}: #{error.message}"
  STDERR.puts failure
ensure
  if active = chain
    if terminal_wait_failed || (terminal_wait_attempted && !active.completed?)
      cleanup_quarantined = true
      failure = "#{failure}; failed terminal wait quarantined live PixelUMM owners"
    else
      unless active.completed?
        begin
          terminal_wait_attempted = true
          active.commit_and_wait
        rescue error
          cleanup_quarantined = true
          failure = "#{failure}; cleanup wait: #{error.class}: #{error.message}"
        end
      end
      if active.completed?
        begin
          active.release
          chain = nil
        rescue error
          cleanup_quarantined = true
          failure = "#{failure}; chain release: #{error.class}: #{error.message}"
        end
      else
        cleanup_quarantined = true
      end
    end
  end

  unless cleanup_quarantined
    begin
      selection.try(&.release)
    rescue error
      cleanup_quarantined = true
      failure = "#{failure}; selection release: #{error.class}: #{error.message}"
    end
  end
  unless cleanup_quarantined
    begin
      session.try(&.release)
    rescue error
      cleanup_quarantined = true
      failure = "#{failure}; session release: #{error.class}: #{error.message}"
    end
  end
  unless cleanup_quarantined
    GC.collect
    begin
      observer.emit("after_session_release")
    rescue error
      passed = false
      failure = "#{failure}; memory recorder: #{error.class}: #{error.message}"
    end
  end

  baseline_named = session.try(&.named_bytes) || baseline_named
  candidate_named = selection.try(&.named_bytes) || candidate_named
  total_named = baseline_named + candidate_named if total_named == 0
  admitted_budget = budget_admission
  report_result = {
    schema: "cogni_ml.pixelumm_window_batch_full_probe/v1",
    generation: false, production_promoted: false, passed: passed && !cleanup_quarantined,
    failure: failure, cleanup_quarantined: cleanup_quarantined,
    reference: reference, reference_sha256: reference_sha,
    alternate_reference: alternate, alternate_reference_sha256: alternate_sha,
    head_reference: head_reference, head_reference_sha256: head_reference_sha,
    head_alternate_reference: head_alternate,
    head_alternate_reference_sha256: head_alternate_sha,
    bf16_cache: cache_path, bf16_cache_sha256: cache_sha,
    input_sha256: fixture_input_sha, alternate_input_sha256: alternate_input_sha,
    upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
    oracle: "Saved upstream full36 block and head captures; capture compatibility is authenticated before GPU allocation.",
    candidate: "B2 paired window attention across all 36 serial full-J blocks",
    grid: [6, 44, 80], block_count: 36, includes_head: true,
    heads: 32, head_dimension: 128,
    atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL,
    capture_order: capture_order,
    capture_inventory: {block_attention: 36, retained_block_outputs: 2, head_stages: 5, total: 43},
    head_input_is_block35_output: head_input_alias,
    baseline_named_bytes: baseline_named, candidate_named_bytes: candidate_named,
    total_unique_named_bytes: total_named,
    named_budget_bytes: admitted_budget.try(&.max_named_bytes) || budget,
    candidate_panel_bytes: PixelUMMWindowBatchFullProbe::EXTRA_PANEL_BYTES,
    candidate_metrics_bytes_per_plan: PixelUMMWindowBatchFullProbe::METRICS_BYTES_PER_PLAN,
    candidate_plan_count: PixelUMMWindowBatchFullProbe::PLAN_COUNT,
    expected_total_unique_named_bytes: PixelUMMWindowBatchFullProbe::EXPECTED_TOTAL_NAMED_BYTES,
    full_model_qualified: false, timed: timed, reverse: reverse,
    expected_request_count: schedule.size, requests: requests,
    memory_phases: observer.events,
    timing_contract: "Four poisoned A/B baseline/candidate gates; optional A baseline/candidate warmups; eight same-input ABBA/BAAB timed requests; final timed capture plus untimed candidate capture tail. One existing terminal wait per request; no CPU tensor diagnostics between timed requests.",
  }
  report_io.puts report_result.to_pretty_json
  report_io.close
  puts({report: report, passed: passed && !cleanup_quarantined, failure: failure}.to_json)
end

raise "PixelUMM B2 full36/head diagnostic failed: #{failure}" unless passed && !cleanup_quarantined
