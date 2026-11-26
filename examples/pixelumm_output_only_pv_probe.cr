require "option_parser"
require "json"
require "../src/ml/video/pixelumm_decoder_block_mps"
require "./support/pixelumm_cognigraph_probe"
require "./support/pixelumm_ffn_weight_reuse_probe"

alias P = PixelUMMCogniGraphProbe
alias B = PixelUMMFFNWeightReuseProbe

module PixelUMMOutputOnlyPVProbe
  extend self

  SCHEMA      = "cogni_ml.pixelumm_output_only_pv_probe/v1"
  ERROR_BYTES = 1024

  def handle_pointer(handle : UInt64) : Pointer(Void)
    raise ArgumentError.new("null PixelUMM native plan handle") if handle == 0_u64
    Pointer(Void).new(handle)
  end

  private def error_text(bytes : Bytes) : String
    length = bytes.index(0_u8) || bytes.size
    String.new(bytes.to_unsafe, length)
  end

  {% unless flag?(:cpu_only) %}
    def prepare_status(handle : UInt64) : Pointer(Void)
      error = Bytes.new(ERROR_BYTES, 0_u8)
      status = PixelUMMDecoderBlockMPSFFI.plan_prepare_softmax_pv_output_only(
        handle_pointer(handle), 0, error.to_unsafe, error.size)
      status == 0 ? handle_pointer(handle) : Pointer(Void).null
    end

    def prepare_plans!(handles : Array(UInt64)) : Array(Pointer(Void))
      unless handles.size == 4 && handles.uniq.size == 4
        raise ArgumentError.new("output-only softmax/PV setup requires four distinct native plans")
      end
      wrappers = [] of Pointer(Void)
      begin
        handles.each_with_index do |handle, index|
          error = Bytes.new(ERROR_BYTES, 0_u8)
          status = PixelUMMDecoderBlockMPSFFI.plan_prepare_softmax_pv_output_only(
            handle_pointer(handle), 0, error.to_unsafe, error.size)
          wrapper = handle_pointer(handle)
          if status != 0
            raise ArgumentError.new("output-only softmax/PV preparation failed for plan #{index}: #{error_text(error)}")
          end
          wrappers << wrapper
        end
      rescue error
        release_wrappers!(wrappers)
        raise error
      end
      wrappers
    end

    def select_status(wrapper : Pointer(Void), enabled : Int32) : Int32
      error = Bytes.new(ERROR_BYTES, 0_u8)
      PixelUMMDecoderBlockMPSFFI.plan_set_softmax_pv(wrapper, enabled, error.to_unsafe, error.size)
    end

    def select_plans!(wrappers : Array(Pointer(Void)), enabled : Bool) : Nil
      unless wrappers.size == 4 && wrappers.none?(&.null?)
        raise ArgumentError.new("output-only softmax/PV selection requires four live wrappers")
      end
      wrappers.each_with_index do |wrapper, index|
        error = Bytes.new(ERROR_BYTES, 0_u8)
        status = PixelUMMDecoderBlockMPSFFI.plan_set_softmax_pv(wrapper, enabled ? 1 : 0,
          error.to_unsafe, error.size)
        unless status == 0
          raise ArgumentError.new("output-only softmax/PV selection failed for plan #{index} (#{status}): #{error_text(error)}")
        end
      end
    end

    def release_wrappers!(wrappers : Array(Pointer(Void))) : Nil
      wrappers.each do |wrapper|
        next if wrapper.null?
        error = Bytes.new(ERROR_BYTES, 0_u8)
        status = PixelUMMDecoderBlockMPSFFI.plan_set_softmax_pv(
          wrapper, 0, error.to_unsafe, error.size)
        raise ArgumentError.new("output-only softmax/PV idle restore failed: #{error_text(error)}") unless status == 0
      end
      wrappers.clear
    end
  {% else %}
    def prepare_status(handle : UInt64) : Pointer(Void)
      raise ArgumentError.new("output-only softmax/PV preparation requires macOS Metal")
    end

    def prepare_plans!(handles : Array(UInt64)) : Array(Pointer(Void))
      raise ArgumentError.new("output-only softmax/PV preparation requires macOS Metal")
    end

    def select_status(wrapper : Pointer(Void), enabled : Int32) : Int32
      raise ArgumentError.new("output-only softmax/PV selection requires macOS Metal")
    end

    def select_plans!(wrappers : Array(Pointer(Void)), enabled : Bool) : Nil
      raise ArgumentError.new("output-only softmax/PV selection requires macOS Metal")
    end

    def release_wrappers!(wrappers : Array(Pointer(Void))) : Nil
      raise ArgumentError.new("output-only softmax/PV release requires macOS Metal")
    end
  {% end %}

  def metrics_json(metrics : Array(B::Metric), trial : P::Trial)
    metrics.map_with_index do |metric, slot|
      kind = "unused"
      plan_slot = nil.as(Int32?)
      stage = nil.as(String?)
      compared = false
      if slot < B::STAGE_METRIC_SLOTS
        kind = "weight_stage_identity"
        plan_slot = slot.to_i32
      elsif slot == B::POSITIVE_CONTROL_SLOT
        kind = "positive_nan_control"
        compared = trial[:positive_nan_control]
      else
        capture_index = slot - B::CAPTURE_METRIC_BASE
        plan_slot = capture_index // B::CAPTURE_ORDER.size
        stage_index = capture_index % B::CAPTURE_ORDER.size
        stage = B::CAPTURE_ORDER[stage_index]
        kind = "capture_comparison"
        compared = trial[:capture_mode] || stage == "output"
      end
      {slot: slot, kind: kind, plan_slot: plan_slot, stage: stage,
       compared: compared, failures: metric[:failures],
       different_words: metric[:different_words], nonfinite: metric[:nonfinite],
       elements: metric[:elements]}
    end
  end

  def roots_json(roots : Array(ML::Video::MPSCommandRootTiming))
    roots.map do |root|
      {sequence: root.sequence, completed: root.completed, has_error: root.has_error,
       gpu_start_seconds: root.gpu_start_seconds,
       gpu_end_seconds: root.gpu_end_seconds,
       duration_seconds: root.gpu_end_seconds - root.gpu_start_seconds}
    end
  end
end

reference = ""
reference_sha256 = ""
weight_cache = ""
report = ""
maximum_named_bytes = B::MAX_NAMED_BYTES
trace_ready = trace_release = ""
instruments_sample = false
correctness_only = false

OptionParser.parse do |parser|
  parser.banner = "PixelUMM output-only softmax/PV full-block diagnostic. No generation."
  parser.on("--reference PATH", "Authenticated upstream-stem window reference manifest") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted reference metadata SHA256") { |value| reference_sha256 = value }
  parser.on("--weight-cache PATH", "External pinned 36-block BF16 cache manifest") { |value| weight_cache = value }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--max-named-bytes N", "Named-buffer cap, at most 9 GiB") { |value| maximum_named_bytes = value.to_i64 }
  parser.on("--instruments-sample", "Four correctness requests with phase metadata; excluded from speed claims") { instruments_sample = true }
  parser.on("--correctness-only", "Only the first four correctness requests without profiling") { correctness_only = true }
  parser.on("--trace-ready-file PATH", "Fresh CPU-only Instruments attach receipt") { |value| trace_ready = value }
  parser.on("--trace-release-file PATH", "Fresh CPU-only Instruments release gate") { |value| trace_release = value }
end

raise ArgumentError.new("correctness-only cannot collect Instruments metadata") if correctness_only && instruments_sample

unless trace_ready.empty? && trace_release.empty?
  unless instruments_sample && !trace_ready.empty? && !trace_release.empty? &&
         trace_ready != trace_release && [trace_ready, trace_release].all? { |path|
           Path[path].absolute? && File.directory?(File.dirname(path)) &&
           File.realpath(File.dirname(path)).starts_with?("/Volumes/") &&
           !File.exists?(path) && !File.symlink?(path)
         }
    raise ArgumentError.new("Instruments gate paths must be distinct fresh absolute paths under /Volumes in sample mode")
  end
  trace_ready = File.join(File.realpath(File.dirname(trace_ready)), File.basename(trace_ready))
  trace_release = File.join(File.realpath(File.dirname(trace_release)), File.basename(trace_release))
  raise ArgumentError.new("Instruments gate paths alias") if trace_ready == trace_release
end

raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
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

canonical_report = File.join(File.realpath(File.dirname(report)), File.basename(report))
raise ArgumentError.new("output-only softmax/PV report must not already exist") if File.exists?(canonical_report) || File.symlink?(canonical_report)
report_fd = LibC.open(canonical_report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("output-only softmax/PV report must be fresh and writable") if report_fd < 0
report_io = IO::FileDescriptor.new(report_fd)

fixture = nil.as(B::Fixture?)
cache = nil.as(B::WeightCache?)
owner = nil.as(B::Owner?)
wrappers = [] of Pointer(Void)
requests = [] of JSON::Any
setup_seconds = 0.0
failure = nil.as(String?)
named_resources = nil.as(JSON::Any?)
negative_controls = nil.as(JSON::Any?)
report_written = false
completed = false
trials = instruments_sample || correctness_only ? P.schedule.first(4) : P.schedule
phase_markers = instruments_sample ? Array.new(4) {
  Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker).new(11) { PixelUMMDecoderBlockMPSFFI::PhaseMarker.new }
} : nil

write_report = ->(success : Bool) do
  fixture_value = fixture
  cache_value = cache
  active = owner
  report_doc = {
    schema: PixelUMMOutputOnlyPVProbe::SCHEMA, probe_process_id: Process.pid,
    probe_completed: success, production_promoted: false,
    full_generation_executed: false,
    instruments_sample: instruments_sample,
    correctness_only: correctness_only,
    failure: failure,
    numerical_comparison: {atol: B::ATOL, rtol: B::RTOL,
                           result: success ? "all comparator gates passed" : "failed or incomplete"},
    input_pins: fixture_value && cache_value ? {
      reference_manifest_sha256:    fixture_value.sha256,
      reference_payload_sha256:     fixture_value.payload_sha256,
      weight_cache_manifest_sha256: cache_value.tensor_file.sha256,
      weight_cache_payload_sha256:  cache_value.tensor_file.document["payload_sha256"].as_s,
      upstream_model_sha256:        B::MODEL_SHA256,
      upstream_revision:            B::UPSTREAM_REVISION,
    } : nil,
    candidate_graph: {
      operation: "rank-4 axis-3 F32 softmax followed by rank-3 batch-32 PV; only attended output is bound",
      qk_pv_batch_rank: 3, softmax_rank: 4, softmax_axis: 3,
      scale: "unchanged standalone F32 1/sqrt(128) before softmax",
      probability_output_bound: false,
      probability_materialization_inside_mpsgraph: "unknown; not inferred from target binding",
      additional_named_metal_buffers: 0,
      graph_intermediate_memory: "opaque MPSGraph allocation; not in named-buffer inventory",
    },
    capture_contract: {
      correctness_requests:                           "full capture A/B then output-only A/B",
      compared_stages:                                B::CAPTURE_ORDER,
      direct_f32_attention_check_in_timed_path:       false,
      direct_f32_attention_qualification:             "separate plan-reuse --softmax-pv --softmax-pv-output-only --check-softmax-pv-f32 report; attended output only",
      output_only_requests_compare_only_final_output: true,
      tolerance:                                      {atol: B::ATOL, rtol: B::RTOL},
    },
    schedule: {
      warmup:                                         (instruments_sample || correctness_only) ? "none; correctness only" : "ABBA",
      timed:                                          (instruments_sample || correctness_only) ? "none; correctness only" : "four repetitions of ABBA/BAAB pair pattern",
      request_count:                                  trials.size,
      timed_request_count:                            trials.count(&.[:timing_eligible]),
      same_window_major_plan_order_for_both_variants: true,
    },
    measurement_semantics: {
      instruments_timings_eligible_for_speed_claims: false,
      phase_markers_collected:                       instruments_sample,
      full_request_wall:                             "monotonic wall interval through one terminal wait and one comparator metrics readback",
      host_thread_cpu:                               "CLOCK_THREAD_CPUTIME_ID over the same full request",
      body_encode:                                   "B::Owner native plan encode duration; excludes comparator encoding and request setup/selection",
      gpu_roots:                                     "raw completed MPS root intervals; roots may overlap and do not establish exclusive GPU busy time",
      gpu_elapsed:                                   "native command-chain aggregate for the full request",
    },
    resource_guard: {
      max_process_footprint_mib: 16_384, minimum_free_memory_percent: 10,
      maximum_swap_growth_mib: 64, enforced_by_external_memory_guard: true,
      named_buffer_cap_bytes: maximum_named_bytes,
      named_resources: named_resources || active.try(&.named_resources_json),
      graph_intermediate_memory_included_in_named_inventory: false,
    },
    request_count_recorded: requests.size,
    request_count_completed: requests.count { |request| request["request_failed"]?.try(&.as_bool) != true },
    negative_controls: negative_controls,
    requests: requests,
    summary: {
      correctness_gates_passed:         success,
      candidate_requests_planned:       trials.count(&.[:enabled]),
      baseline_requests_planned:        trials.count { |trial| !trial[:enabled] },
      timed_candidate_requests_planned: trials.count { |trial| trial[:timing_eligible] && trial[:enabled] },
      timed_baseline_requests_planned:  trials.count { |trial| trial[:timing_eligible] && !trial[:enabled] },
    },
    setup_seconds: setup_seconds,
  }
  report_io << JSON.build(indent: "  ") { |json| report_doc.to_json(json) }
  report_io << '\n'
  report_io.flush
  report_written = true
end

memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
begin
  setup_started = Time.instant
  fixture = B::Fixture.new(reference, reference_sha256)
  cache = B::WeightCache.new(weight_cache, fixture.not_nil!)
  active = B::Owner.new(fixture.not_nil!, cache.not_nil!, maximum_named_bytes,
    memory, trials.size.to_i32, false)
  owner = active
  handles = active.native_plan_handles
  wrappers = PixelUMMOutputOnlyPVProbe.prepare_plans!(handles)

  # Reject invalid operations before requests and prove the valid baseline can
  # still be selected after those no-op negative controls.
  duplicate_prepare_status = PixelUMMOutputOnlyPVProbe.prepare_status(handles.first)
  invalid_enable_status = PixelUMMOutputOnlyPVProbe.select_status(wrappers.first, 2)
  null_select_status = PixelUMMOutputOnlyPVProbe.select_status(Pointer(Void).null, 1)
  unless duplicate_prepare_status.null? && invalid_enable_status != 0 && null_select_status != 0
    raise ArgumentError.new("output-only softmax/PV rejected-operation negative control unexpectedly succeeded")
  end
  negative_controls = JSON.parse({duplicate_prepare_rejected: duplicate_prepare_status.null?,
                                  invalid_enable_status:      invalid_enable_status,
                                  null_select_status:         null_select_status}.to_json)
  PixelUMMOutputOnlyPVProbe.select_plans!(wrappers, false)
  setup_seconds = (Time.instant - setup_started).total_seconds

  unless trace_ready.empty?
    # CPU startup gate only, before encoding and request clocks. It allows
    # attaching Instruments to this live PID without any additional GPU wait.
    ready_fd = LibC.open(trace_ready, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("Instruments ready receipt must be fresh and writable") if ready_fd < 0
    IO::FileDescriptor.new(ready_fd).tap do |io|
      io.puts(Process.pid)
      io.close
    end
    deadline = Time.instant + 45.seconds
    until File.file?(trace_release)
      raise "Instruments CPU attach gate timed out" if Time.instant >= deadline
      sleep 10.milliseconds
    end
    raise ArgumentError.new("Instruments release gate is a symlink") if File.symlink?(trace_release)
  end

  trials.each do |trial|
    entry = P.owner_entry(trial)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    wait_attempted = false
    host_start = P::HostPoint.now
    wall_start = Time.instant
    reads_before = ML::MetalBuffer.hy_text_probe_readbacks
    allocations_before = ML::MetalBuffer.hy_text_probe_allocations
    owner_started = false
    terminal_wait_seconds = nil.as(Float64?)
    root_profile = nil.as(ML::Video::MPSCommandPhaseProfile?)
    raw_roots = [] of ML::Video::MPSCommandRootTiming
    native_gpu_seconds = nil.as(Float64?)
    encoded = nil.as(NamedTuple(body_encode_seconds: Float64,
      comparator_encode_seconds: Float64, stage_conversion_count: Int32,
      stage_identity_check_count: Int32)?)
    raw_metrics = nil.as(Array(B::Metric)?)
    captured_phase_profile = nil.as(JSON::Any?)

    begin
      chain = ML::Metal::MPSCommandChain.new
      chain.not_nil!.enable_api_profile!(phase_roots: false) if instruments_sample
      active.begin_request!(entry, active.expected_input_ids)
      owner_started = true
      # Select the attention variant only while the owner is idle and before
      # native encode; FFN remains on its unchanged baseline graph.
      PixelUMMOutputOnlyPVProbe.select_plans!(wrappers, trial[:enabled])
      active.track_chain!(chain.not_nil!)
      active.poison_gate_buffers!(chain.not_nil!)
      active.clear_metrics!(chain.not_nil!)

      encoded = active.encode_request!(chain.not_nil!, entry, phase_markers)
      unless encoded.not_nil![:stage_conversion_count] == 0 &&
             encoded.not_nil![:stage_identity_check_count] == 0
        raise ArgumentError.new("output-only softmax/PV probe unexpectedly staged or checked FFN matrices")
      end
      preterminal_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      unless allocations == 0 && preterminal_reads == 0 && chain.not_nil!.wait_count == 0
        raise ArgumentError.new("output-only softmax/PV request allocated named buffers, read back or waited before completion")
      end

      wait_started = Time.instant
      wait_attempted = true
      chain.not_nil!.commit_and_wait
      terminal_wait_seconds = (Time.instant - wait_started).total_seconds
      unless chain.not_nil!.completed? && chain.not_nil!.wait_count == 1
        raise ArgumentError.new("output-only softmax/PV request did not complete after its sole terminal wait")
      end
      raw_roots = chain.not_nil!.completed_root_timings
      native_gpu_seconds = chain.not_nil!.gpu_elapsed_seconds
      root_profile = ML::Video::MPSCommandPhaseProfile.build(raw_roots,
        [ML::Video::MPSCommandPhaseRange.new("whole_request", 0, raw_roots.size.to_i32)],
        native_gpu_seconds.not_nil!)
      if values = phase_markers
        captured_phase_profile = JSON.parse({
          markers: values.map do |plan_markers|
            plan_markers.map do |marker|
              {phase: marker.phase, wall_ns: marker.wall_ns,
               thread_cpu_ns: marker.thread_cpu_ns, root_count: marker.root_count}
            end.to_a
          end,
        }.to_json)
      end

      metrics = active.finish_request!(chain.not_nil!, entry) do |observed|
        # Capture the one existing post-wait read before B::Owner validates it,
        # so a numerical failure still leaves all raw slots in the report.
        raw_metrics = observed
      end
      unless metrics == raw_metrics
        raise ArgumentError.new("output-only softmax/PV metric callback differs from the returned single read")
      end
      postterminal_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      unless allocations == 0 && postterminal_reads == 1 && chain.not_nil!.wait_count == 1
        raise ArgumentError.new("output-only softmax/PV request requires one post-wait metrics read and no named allocation")
      end

      host_end = P::HostPoint.now
      wall_seconds = (Time.instant - wall_start).total_seconds
      requests << JSON.parse({
        index: trial[:index], kind: trial[:kind], output_only_pv_enabled: trial[:enabled],
        capture_mode: trial[:capture_mode], positive_nan_control: trial[:positive_nan_control],
        timing_eligible: trial[:timing_eligible], full_request_wall_seconds: wall_seconds,
        host_clock_wall_seconds: (host_end.wall_ns - host_start.wall_ns).to_f64 / 1e9,
        host_thread_cpu_seconds: P.host_thread_seconds(host_start, host_end),
        body_encode_seconds: encoded.not_nil![:body_encode_seconds],
        comparator_encode_seconds: encoded.not_nil![:comparator_encode_seconds],
        terminal_wait_wall_seconds: terminal_wait_seconds,
        native_gpu_elapsed_seconds: native_gpu_seconds,
        gpu_root_duration_sum_seconds: root_profile.not_nil!.root_duration_seconds,
        gpu_root_union_seconds: root_profile.not_nil!.union_seconds,
        gpu_root_overlap_seconds: root_profile.not_nil!.overlap_seconds,
        gpu_roots: PixelUMMOutputOnlyPVProbe.roots_json(raw_roots),
        root_count: raw_roots.size, terminal_wait_count: chain.not_nil!.wait_count,
        named_buffer_allocations: allocations, preterminal_readbacks: preterminal_reads,
        comparator_readbacks_after_wait: postterminal_reads,
        numerical_metrics: PixelUMMOutputOnlyPVProbe.metrics_json(metrics, trial),
        phase_profile: captured_phase_profile,
      }.to_json)
    rescue error
      active.abort_request! if owner_started
      command_chain = chain
      if command_chain && !wait_attempted && !command_chain.not_nil!.completed? && command_chain.not_nil!.wait_count == 0
        begin
          command_chain.not_nil!.commit_and_wait
        rescue drain_error
          STDERR.puts("PIXELUMM_OUTPUT_ONLY_PV_DRAIN_FAILURE #{drain_error.inspect_with_backtrace}")
        end
      end
      if command_chain && command_chain.not_nil!.completed? && command_chain.not_nil!.wait_count == 1
        if owner_started && active.lifecycle.attempted && !active.lifecycle.terminal_wait_succeeded
          begin
            active.lifecycle.mark_cleanup_wait!(completed: true, wait_count: 1)
          rescue cleanup_error
            STDERR.puts("PIXELUMM_OUTPUT_ONLY_PV_CLEANUP_WAIT_FAILURE #{cleanup_error.inspect_with_backtrace}")
          end
        end
        begin
          command_chain.not_nil!.release
        rescue cleanup_error
          STDERR.puts("PIXELUMM_OUTPUT_ONLY_PV_CHAIN_CLEANUP_FAILURE #{cleanup_error.inspect_with_backtrace}")
        end
      end
      host_end = P::HostPoint.now
      if raw_metrics
        requests << JSON.parse({
          index: trial[:index], kind: trial[:kind], output_only_pv_enabled: trial[:enabled],
          capture_mode: trial[:capture_mode], positive_nan_control: trial[:positive_nan_control],
          timing_eligible: trial[:timing_eligible], request_failed: true,
          failure: error.message, raw_metrics_captured_before_validation: true,
          numerical_metrics: PixelUMMOutputOnlyPVProbe.metrics_json(raw_metrics.not_nil!, trial),
          full_request_wall_seconds: (Time.instant - wall_start).total_seconds,
          host_clock_wall_seconds: (host_end.wall_ns - host_start.wall_ns).to_f64 / 1e9,
          host_thread_cpu_seconds: P.host_thread_seconds(host_start, host_end),
          body_encode_seconds: encoded ? encoded.not_nil![:body_encode_seconds] : nil,
          comparator_encode_seconds: encoded ? encoded.not_nil![:comparator_encode_seconds] : nil,
          terminal_wait_wall_seconds: terminal_wait_seconds,
          native_gpu_elapsed_seconds: native_gpu_seconds,
          gpu_roots: PixelUMMOutputOnlyPVProbe.roots_json(raw_roots),
          terminal_wait_count: command_chain ? command_chain.not_nil!.wait_count : nil,
          named_buffer_allocations: ML::MetalBuffer.hy_text_probe_allocations - allocations_before,
          comparator_readbacks_after_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads_before,
        }.to_json)
      else
        requests << JSON.parse({index: trial[:index], kind: trial[:kind],
                                output_only_pv_enabled: trial[:enabled], request_failed: true,
                                failure: error.message,
                                numerical_metrics: nil,
                                full_request_wall_seconds: (Time.instant - wall_start).total_seconds,
                                host_clock_wall_seconds: (host_end.wall_ns - host_start.wall_ns).to_f64 / 1e9,
                                host_thread_cpu_seconds: P.host_thread_seconds(host_start, host_end),
                                body_encode_seconds: encoded ? encoded.not_nil![:body_encode_seconds] : nil,
                                comparator_encode_seconds: encoded ? encoded.not_nil![:comparator_encode_seconds] : nil,
                                terminal_wait_wall_seconds: terminal_wait_seconds,
                                native_gpu_elapsed_seconds: native_gpu_seconds,
                                gpu_roots: PixelUMMOutputOnlyPVProbe.roots_json(raw_roots),
                                terminal_wait_count: command_chain ? command_chain.not_nil!.wait_count : nil,
                                named_buffer_allocations: ML::MetalBuffer.hy_text_probe_allocations - allocations_before,
                                comparator_readbacks_after_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads_before,
        }.to_json)
      end
      raise error
    end
  end

  named_resources = active.named_resources_json
  PixelUMMOutputOnlyPVProbe.release_wrappers!(wrappers)
  active.release_after_terminal!
  owner = nil
  completed = true
  write_report.call(true)
rescue error
  failure = error.inspect_with_backtrace
  if owner && owner.not_nil!.lifecycle.release_allowed?
    begin
      PixelUMMOutputOnlyPVProbe.release_wrappers!(wrappers)
      named_resources = owner.not_nil!.named_resources_json unless named_resources
      owner.not_nil!.release_unsubmitted!
      owner = nil
    rescue cleanup_error
      STDERR.puts("PIXELUMM_OUTPUT_ONLY_PV_OWNER_CLEANUP_FAILURE #{cleanup_error.inspect_with_backtrace}")
    end
  end
  begin
    write_report.call(false)
  rescue report_error
    STDERR.puts("PIXELUMM_OUTPUT_ONLY_PV_REPORT_WRITE_FAILURE #{report_error.inspect_with_backtrace}")
  end
  raise error
ensure
  report_io.close
end

unless report_written && completed
  raise "output-only softmax/PV probe did not write its report"
end
puts({prefix: "PIXELUMM_OUTPUT_ONLY_PV", report: canonical_report,
      requests: requests.size, named_bytes: budget_preflight[:total_bytes]}.to_json)
