require "option_parser"
require "json"
require "./support/pixelumm_cognigraph_probe"

alias P = PixelUMMCogniGraphProbe
alias B = PixelUMMFFNWeightReuseProbe

reference = ""
reference_sha256 = ""
weight_cache = ""
report = ""
maximum_named_bytes = B::MAX_NAMED_BYTES

OptionParser.parse do |parser|
  parser.banner = "Four-plan PixelUMM native-island CogniGraph probe. No generation."
  parser.on("--reference PATH", "Authenticated upstream-stem window reference manifest") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted reference metadata SHA256") { |value| reference_sha256 = value }
  parser.on("--weight-cache PATH", "External pinned 36-block BF16 cache manifest") { |value| weight_cache = value }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--max-named-bytes N", "Named-buffer cap, at most 9 GiB") { |value| maximum_named_bytes = value.to_i64 }
end

unless ARGV.empty?
  raise ArgumentError.new("unexpected positional arguments")
end
budget_preflight = B.preflight_budget(maximum_named_bytes)

def canonical_input_file?(path : String) : Bool
  !path.empty? && Path[path].absolute? && File.file?(path) &&
    !File.symlink?(path) && File.realpath(path) == path
end

unless canonical_input_file?(reference) &&
       PixelUMMBlockProbe.valid_digest?(reference_sha256) &&
       canonical_input_file?(weight_cache) && Path[report].absolute? && !report.empty? &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("reference/cache must be canonical absolute files and report must be fresh under /Volumes")
end

# Resolve an optional volume symlink in the parent directory before reserving
# the output, so the report path itself is canonical and remains O_EXCL.
canonical_report = File.join(File.realpath(File.dirname(report)), File.basename(report))
unless !File.exists?(canonical_report) && !File.symlink?(canonical_report)
  raise ArgumentError.new("CogniGraph report must not already exist")
end
report_fd = LibC.open(canonical_report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("CogniGraph report must be fresh and writable") if report_fd < 0
report_io = IO::FileDescriptor.new(report_fd)

fixture = B::Fixture.new(reference, reference_sha256)
cache = B::WeightCache.new(weight_cache, fixture)
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
requests = [] of JSON::Any
owner = nil.as(B::Owner?)
completed = false

begin
  setup_started = Time.monotonic
  active = B::Owner.new(fixture, cache, maximum_named_bytes, memory,
    P::REQUEST_COUNT.to_i32, false)
  owner = active
  native_handles = active.native_plan_handles
  unprepared_enable_status = P.native_set_status(native_handles.first, 1)
  raise ArgumentError.new("enabling an unprepared native island unexpectedly succeeded") if unprepared_enable_status == 0
  P.prepare_native_plans!(active.native_plan_handles)
  duplicate_prepare_status = P.native_prepare_status(native_handles.first)
  raise ArgumentError.new("duplicate native-island preparation unexpectedly succeeded") if duplicate_prepare_status == 0
  invalid_enabled_status = P.native_set_status(native_handles.first, 2)
  raise ArgumentError.new("invalid native-island enabled value unexpectedly succeeded") if invalid_enabled_status == 0
  setup_seconds = (Time.monotonic - setup_started).total_seconds
  island_mode = P::NativeIslandMode.new
  all_trials = P.schedule
  late_prepare_status = nil.as(Int32?)

  all_trials.each do |trial|
    entry = P.owner_entry(trial)
    changed = island_mode.prepare_request!(trial[:index], trial[:enabled])
    P.set_native_plans!(active.native_plan_handles, trial[:enabled]) if changed
    island_mode.begin_request!(trial[:index], trial[:enabled])

    chain = nil.as(ML::Metal::MPSCommandChain?)
    wait_attempted = false
    host_start = P::HostPoint.now
    wall_start = Time.monotonic
    reads_before = ML::MetalBuffer.hy_text_probe_readbacks
    allocations_before = ML::MetalBuffer.hy_text_probe_allocations
    owner_started = false
    begin
      chain = ML::Metal::MPSCommandChain.new
      active.begin_request!(entry, active.expected_input_ids)
      owner_started = true
      active.track_chain!(chain.not_nil!)
      active.poison_gate_buffers!(chain.not_nil!)
      active.clear_metrics!(chain.not_nil!)

      encoded = active.encode_request!(chain.not_nil!, entry)
      unless encoded[:stage_conversion_count] == 0 &&
             encoded[:stage_identity_check_count] == 0
        raise ArgumentError.new("CogniGraph probe unexpectedly staged or checked FFN matrices")
      end
      preterminal_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      unless allocations == 0 && preterminal_reads == 0 && chain.not_nil!.wait_count == 0
        raise ArgumentError.new("CogniGraph request allocated named buffers, read back or waited before completion")
      end

      wait_started = Time.monotonic
      wait_attempted = true
      chain.not_nil!.commit_and_wait
      wait_seconds = (Time.monotonic - wait_started).total_seconds
      unless chain.not_nil!.completed? && chain.not_nil!.wait_count == 1
        raise ArgumentError.new("CogniGraph request did not complete after its sole terminal wait")
      end
      roots = chain.not_nil!.completed_root_timings
      native_gpu_seconds = chain.not_nil!.gpu_elapsed_seconds
      root_profile = ML::Video::MPSCommandPhaseProfile.build(roots,
        [ML::Video::MPSCommandPhaseRange.new("whole_request", 0, roots.size.to_i32)],
        native_gpu_seconds)
      root_count = roots.size

      metrics = active.finish_request!(chain.not_nil!, entry)
      island_mode.finish_request!(trial[:index], chain.not_nil!.wait_count)
      if trial[:index] == 0
        late_prepare_status = P.native_prepare_status(native_handles.first)
        raise ArgumentError.new("late native-island preparation after a drained encode unexpectedly succeeded") if late_prepare_status == 0
      end
      postterminal_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      unless allocations == 0 && postterminal_reads == 1 && chain.not_nil!.wait_count == 1
        raise ArgumentError.new("CogniGraph request requires one post-wait comparator readback and no named allocation")
      end

      host_end = P::HostPoint.now
      wall_seconds = (Time.monotonic - wall_start).total_seconds
      host_wall_seconds = (host_end.wall_ns - host_start.wall_ns).to_f64 / 1e9
      requests << JSON.parse({
        index: trial[:index], kind: trial[:kind], native_islands_enabled: trial[:enabled],
        native_mode_toggled_before_request: changed,
        capture_mode: trial[:capture_mode], positive_nan_control: trial[:positive_nan_control],
        timing_eligible: trial[:timing_eligible],
        full_request_wall_seconds: wall_seconds,
        host_clock_wall_seconds: host_wall_seconds,
        host_thread_cpu_seconds: P.host_thread_seconds(host_start, host_end),
        body_encode_seconds: encoded[:body_encode_seconds],
        comparator_encode_seconds: encoded[:comparator_encode_seconds],
        terminal_wait_wall_seconds: wait_seconds,
        native_gpu_elapsed_seconds: native_gpu_seconds,
        gpu_root_duration_sum_seconds: root_profile.root_duration_seconds,
        gpu_root_union_seconds: root_profile.union_seconds,
        gpu_root_overlap_seconds: root_profile.overlap_seconds,
        root_count: root_count,
        terminal_wait_count: chain.not_nil!.wait_count,
        named_buffer_allocations: allocations,
        preterminal_readbacks: preterminal_reads,
        comparator_readbacks_after_wait: postterminal_reads,
        numerical_metrics: metrics.map { |metric| P.metric_json(metric) },
      }.to_json)
    rescue error
      active.abort_request! if owner_started
      if command_chain = chain
        if !wait_attempted && !command_chain.completed? && command_chain.wait_count == 0
          begin
            command_chain.commit_and_wait
          rescue drain_error
            STDERR.puts("PIXELUMM_COGNIGRAPH_DRAIN_FAILURE #{drain_error.inspect_with_backtrace}")
          end
        end
        if command_chain.completed? && command_chain.wait_count == 1
          if owner_started && active.lifecycle.attempted && !active.lifecycle.terminal_wait_succeeded
            begin
              active.lifecycle.mark_cleanup_wait!(completed: true, wait_count: 1)
            rescue cleanup_error
              STDERR.puts("PIXELUMM_COGNIGRAPH_CLEANUP_WAIT_FAILURE #{cleanup_error.inspect_with_backtrace}")
            end
          end
          begin
            command_chain.release
          rescue cleanup_error
            STDERR.puts("PIXELUMM_COGNIGRAPH_CHAIN_CLEANUP_FAILURE #{cleanup_error.inspect_with_backtrace}")
          end
        end
      end
      raise error
    end
  end

  native_stats = P.native_plan_stats(active.native_plan_handles)
  P.validate_negative_controls!({unprepared_enable: unprepared_enable_status,
                                 duplicate_prepare: duplicate_prepare_status, invalid_enabled: invalid_enabled_status,
                                 late_prepare: late_prepare_status || 0})
  named_resources = active.named_resources_json
  active.release_after_terminal!
  owner = nil
  report_doc = {
    schema:                   P::SCHEMA,
    probe_process_id:         Process.pid,
    probe_completed:          true,
    production_promoted:      false,
    full_generation_executed: false,
    numerical_comparison:     {atol: B::ATOL, rtol: B::RTOL, result: "all comparator gates passed"},
    input_pins:               {
      reference_manifest_sha256:    fixture.sha256,
      reference_payload_sha256:     fixture.payload_sha256,
      weight_cache_manifest_sha256: cache.tensor_file.sha256,
      weight_cache_payload_sha256:  cache.tensor_file.document["payload_sha256"].as_s,
      upstream_model_sha256:        B::MODEL_SHA256,
      upstream_revision:            B::UPSTREAM_REVISION,
    },
    capture_contract: {request_capture_mode_changes_native_plan_capture_mask: true,
                       output_only_requests_retain_only_output_capture: true,
                       encoder_counts_are_prepared_plan_model_counters: true},
    measurement_semantics: {
      full_request_wall: "Time.monotonic around one full four-plan request through the terminal comparator metrics readback and chain release",
      host_thread_cpu:   "CLOCK_THREAD_CPUTIME_ID delta across the same full request",
      gpu_root_sum:      "sum of completed MPS root intervals; roots can overlap and this is not exclusive GPU busy time",
    },
    native_island_stats:                P.stats_json(native_stats),
    native_island_stats_interpretation: "encoder counts are prepared-plan model counters from the native bridge, not measured encoder API calls",
    native_api_negative_controls:       {unprepared_enable_status: unprepared_enable_status,
                                   duplicate_prepare_status: duplicate_prepare_status, invalid_enabled_status: invalid_enabled_status,
                                   late_prepare_after_drained_request_status: late_prepare_status},
    resource_guard: {max_process_footprint_mib: 16_384, minimum_free_memory_percent: 10,
                     maximum_swap_growth_mib: 64, enforced_by_external_memory_guard: true},
    request_count:   requests.size,
    named_resources: named_resources,
    setup_seconds:   setup_seconds,
    requests:        requests,
    summary:         {correctness_gates_passed: true, candidate_requests: all_trials.count(&.[:enabled]),
              baseline_requests: all_trials.count { |trial| !trial[:enabled] },
              timed_candidate_requests: all_trials.count { |trial| trial[:timing_eligible] && trial[:enabled] },
              timed_baseline_requests: all_trials.count { |trial| trial[:timing_eligible] && !trial[:enabled] }},
  }
  report_io << JSON.build(indent: "  ") { |json| report_doc.to_json(json) }
  report_io << '\n'
  report_io.flush
  completed = true
ensure
  if active = owner
    if active.lifecycle.release_allowed?
      begin
        active.release_unsubmitted!
      rescue cleanup_error
        STDERR.puts("PIXELUMM_COGNIGRAPH_OWNER_CLEANUP_FAILURE #{cleanup_error.inspect_with_backtrace}")
      end
    end
  end
  report_io.close
end

raise "CogniGraph probe did not write its report" unless completed
puts({prefix: "PIXELUMM_COGNIGRAPH", report: canonical_report,
      requests: requests.size, named_bytes: budget_preflight[:total_bytes]}.to_json)
