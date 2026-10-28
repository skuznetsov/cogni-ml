require "option_parser"
require "digest/sha256"
require "json"
require "./support/pixelumm_ffn_weight_reuse_probe"

alias B = PixelUMMFFNWeightReuseProbe

reference = ""
reference_sha256 = ""
weight_cache = ""
report = ""
baseline_schedule = "window-major"
maximum_named_bytes = B::MAX_NAMED_BYTES

OptionParser.parse do |parser|
  parser.banner = "Bounded two-window/two-block PixelUMM FFN weight-reuse probe. No generation."
  parser.on("--reference PATH", "Authenticated upstream-stem window reference manifest") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted reference metadata SHA256") { |value| reference_sha256 = value }
  parser.on("--weight-cache PATH", "External pinned 36-block BF16 cache manifest") { |value| weight_cache = value }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--baseline-schedule ORDER", "Baseline order: window-major (default) or block-major") { |value| baseline_schedule = value }
  parser.on("--max-named-bytes N", "Named-buffer cap, at most 9 GiB") { |value| maximum_named_bytes = value.to_i64 }
end

unless ARGV.empty?
  raise ArgumentError.new("unexpected positional arguments")
end
B.validate_baseline_schedule!(baseline_schedule)
budget_preflight = B.preflight_budget(maximum_named_bytes)

def regular_absolute_file?(path : String) : Bool
  !path.empty? && Path[path].absolute? && File.file?(path) &&
    !File.symlink?(path) && File.realpath(path) == path
end

unless regular_absolute_file?(reference) &&
       PixelUMMBlockProbe.valid_digest?(reference_sha256) &&
       regular_absolute_file?(weight_cache) &&
       Path[report].absolute? && !report.empty? &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("reference/cache must be canonical absolute files and report must be fresh under /Volumes")
end

fixture = B::Fixture.new(reference, reference_sha256)
cache = B::WeightCache.new(weight_cache, fixture)
requests = B.schedule(baseline_schedule)
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(B::Owner?)
result = nil.as(JSON::Any?)
all_requests = [] of JSON::Any

begin
  setup_started = Time.instant
  active = B::Owner.new(fixture, cache, maximum_named_bytes, memory)
  owner = active
  setup_seconds = (Time.instant - setup_started).total_seconds

  requests.each do |request|
    index = request[:index]
    chain = nil.as(ML::Metal::MPSCommandChain?)
    wait_succeeded = false
    request_started = Time.instant
    readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
    allocations_before = ML::MetalBuffer.hy_text_probe_allocations
    begin
      chain = ML::Metal::MPSCommandChain.new
      active.begin_request!(request, active.expected_input_ids)
      active.track_chain!(chain.not_nil!)
      active.poison_gate_buffers!(chain.not_nil!)
      active.clear_metrics!(chain.not_nil!)

      encoded = active.encode_request!(chain.not_nil!, request)
      expected_conversions = request[:mode] == "staged_f32" ? 2 : 0
      expected_checks = request[:mode] == "staged_f32" && request[:capture_mode] ? 4 : 0
      unless encoded[:stage_conversion_count] == expected_conversions &&
             encoded[:stage_identity_check_count] == expected_checks
        raise ArgumentError.new("FFN request staged/checked an unexpected number of matrices")
      end

      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      unless allocations == 0 && preterminal_readbacks == 0 && chain.not_nil!.wait_count == 0
        raise ArgumentError.new("FFN request allocated named buffers, read back or waited before completion")
      end

      wait_started = Time.instant
      chain.not_nil!.commit_and_wait
      wait_succeeded = true
      wait_seconds = (Time.instant - wait_started).total_seconds
      gpu_elapsed_seconds = chain.not_nil!.gpu_elapsed_seconds
      metrics = active.finish_request!(chain.not_nil!, request)
      request_wall_seconds = (Time.instant - request_started).total_seconds
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      unless readbacks == 1
        raise ArgumentError.new("FFN request requires exactly one packed metrics readback after its terminal wait")
      end

      all_requests << JSON.parse({
        index:                     index,
        kind:                      request[:kind],
        mode:                      request[:mode],
        baseline_schedule:         request[:baseline_schedule],
        candidate_schedule:        request[:candidate_schedule],
        baseline_plan_order:       B.plan_order("baseline", baseline_schedule),
        candidate_plan_order:      B.plan_order("staged_f32", baseline_schedule),
        selected_plan_order:       B.plan_order(request[:mode], baseline_schedule),
        capture_mode:              request[:capture_mode],
        timing_eligible:           request[:timing_eligible],
        stage_conversions:         encoded[:stage_conversion_count],
        stage_identity_checks:     encoded[:stage_identity_check_count],
        stage_conversion_bytes:    encoded[:stage_conversion_count].to_i64 * 3_i64 * B::STAGE_BYTES,
        body_encode_seconds:       encoded[:body_encode_seconds],
        comparator_encode_seconds: encoded[:comparator_encode_seconds],
        request_wall_seconds:      request_wall_seconds,
        request_wait_seconds:      wait_seconds,
        gpu_elapsed_seconds:       gpu_elapsed_seconds,
        terminal_wait_count:       chain.not_nil!.wait_count,
        stage_metrics:             (0...B::STAGE_METRIC_SLOTS).map do |slot|
          {slot: slot, block: slot % 2, window: slot // 2,
           failures: metrics[slot][:failures], different_words: metrics[slot][:different_words],
           nonfinite: metrics[slot][:nonfinite], elements: metrics[slot][:elements]}
        end,
        capture_metrics: 4.times.flat_map do |plan_slot|
          B::CAPTURE_ORDER.each_with_index.map do |stage, stage_index|
            slot = B.metric_slot(plan_slot, stage_index)
            metric = metrics[slot]
            {slot: slot, plan_slot: plan_slot, window: plan_slot // 2,
             block: plan_slot % 2, stage: stage,
             compared: request[:capture_mode] || stage == "output",
             failures: metric[:failures], different_words: metric[:different_words],
             nonfinite: metric[:nonfinite], elements: metric[:elements]}
          end
        end,
        positive_control: {slot: B::POSITIVE_CONTROL_SLOT,
                           enabled: index == 0, failures: metrics[B::POSITIVE_CONTROL_SLOT][:failures],
                           different_words: metrics[B::POSITIVE_CONTROL_SLOT][:different_words],
                           nonfinite: metrics[B::POSITIVE_CONTROL_SLOT][:nonfinite],
                           elements: metrics[B::POSITIVE_CONTROL_SLOT][:elements]},
        named_buffer_allocations_during_request: allocations,
        preterminal_readbacks_during_request:    preterminal_readbacks,
        comparator_readbacks_after_wait:         readbacks,
        comparator_readback_bytes:               active.metric_bytes,
      }.to_json)
    rescue error
      STDERR.puts("PIXELUMM_FFN_WEIGHT_REUSE_REQUEST_FAILURE #{error.inspect_with_backtrace}")
      active.abort_request!
      # Drain at most once after partial submission. A failed terminal wait is
      # never retried and never authorizes release of resident buffers.
      if chain && !chain.not_nil!.completed? && chain.not_nil!.wait_count == 0
        begin
          chain.not_nil!.commit_and_wait
          wait_succeeded = true
        rescue
        end
      end
      if wait_succeeded && chain && chain.not_nil!.completed? && chain.not_nil!.wait_count == 1
        if active.lifecycle.attempted && !active.lifecycle.terminal_wait_succeeded
          begin
            active.lifecycle.mark_cleanup_wait!(completed: true,
              wait_count: chain.not_nil!.wait_count)
          rescue cleanup_error
            STDERR.puts("PIXELUMM_FFN_WEIGHT_REUSE_CLEANUP_GUARD_FAILURE #{cleanup_error.inspect_with_backtrace}")
          end
        end
        begin
          chain.not_nil!.release
        rescue
        end
      end
      raise error
    end
  end

  resources = active.named_resources_json
  native_plan_handles = active.native_plan_handles.dup
  active.release_after_terminal!
  result = JSON.parse({
    schema:   "cogni_ml.pixelumm_ffn_weight_reuse_probe/v1",
    scope:    "Four resident native PixelUMM plans over two overlapping saved initial-noise preview windows and blocks 0/1; no full generation or production promotion.",
    geometry: {grid: [10, 18, 32], tokens: B::TOKENS,
               hidden: B::HIDDEN, ffn: B::FFN, tile_tokens: 8, token_order: "tile_major"},
    source: {kind: "saved_initial_noise_conditioning_upstream_stem",
             model_inference: false,
             input_manifest_path: B::INPUT_MANIFEST_PATH,
             input_manifest_sha256: B::INPUT_MANIFEST_SHA256,
             input_payload_path: B::INPUT_PAYLOAD_PATH,
             input_payload_sha256: B::INPUT_PAYLOAD_SHA256,
             stem_subset_path: B::STEM_SUBSET_PATH,
             stem_subset_sha256: B::STEM_SUBSET_SHA256,
             stem_receipt_path: B::STEM_RECEIPT_PATH,
             stem_receipt_sha256: B::STEM_RECEIPT_SHA256,
             stem_weights_sha256: B::STEM_WEIGHTS_SHA256,
             windows: fixture.document["windows"].as_h.transform_values do |window|
               window["source_slices"]
             end,
             x_expression: fixture.document["input_provenance"]["x_expression"],
             window_offsets: fixture.document["input_provenance"]["window_offsets"]},
    fixture: {path: fixture.path, metadata_sha256: fixture.sha256,
              payload_sha256: fixture.payload_sha256,
              upstream_revision: B::UPSTREAM_REVISION, model_sha256: B::MODEL_SHA256,
              input_x_sha256: B::WINDOW_NAMES.to_h { |name| {name, fixture.windows[name].inputs["x"].sha256} },
              input_x_windows_differ: fixture.windows["w0"].inputs["x"].sha256 != fixture.windows["w1"].inputs["x"].sha256},
    weight_cache: {path: cache.tensor_file.path, metadata_sha256: cache.tensor_file.sha256,
                   payload_path: cache.tensor_file.payload, payload_sha256: cache.tensor_file.document["payload_sha256"],
                   payload_bytes: cache.tensor_file.payload_bytes,
                   block0_subset_sha256: B::BLOCK_SUBSET_SHA256["0"],
                   block1_subset_sha256: B::BLOCK_SUBSET_SHA256["1"]},
    comparison: {atol: B::ATOL, rtol: B::RTOL,
                 capture_order: B::CAPTURE_ORDER,
                 numerical_scope: "Per-window/per-block BF16 captures and outputs against the authenticated saved upstream fixture."},
    topology: {native_plans: 4, distinct_native_plan_handles: native_plan_handles.uniq.size == 4,
               candidate_schedule: "block-major", baseline_schedule_option: baseline_schedule,
               plans_reselected_before_every_request: true,
               f32_stage_buffers_shared_across_all_plans: true,
               native_bindings_checked_at_preparation_and_each_request: true,
               native_binding_fields: ["input", "cos", "sin", "output", "bf16_gate", "bf16_up", "bf16_down", "f32_gate", "f32_up", "f32_down"],
               block1_input_edge: "same-window block0 output", native_plan_handles: native_plan_handles},
    schedule: {correctness_requests: 2, warmup_requests: 4, timed_requests: 8,
               warmup_order: ["baseline", "staged_f32", "staged_f32", "baseline"],
               timed_order: ["baseline", "staged_f32", "staged_f32", "baseline",
                             "staged_f32", "baseline", "baseline", "staged_f32"],
               timing_stage_conversions_per_candidate_request: 2,
               timing_stage_identity_checks_per_request: 0,
               timing_comparator_scope: "four outputs compared after the two-block body; no activation readback",
               requests: all_requests},
    memory: {named_bytes: active.budget[:total_bytes], maximum_named_bytes: maximum_named_bytes,
             resources: resources, opaque_native_plan_bytes: nil,
             process_footprint_enforcement: "Run under examples/support/wan_memory_guard.py with --max-footprint-mib 16384; its sampled runtime floor is an outer guard.",
             memory_categories_additive: false, memory_events: memory.events},
    validation: {all_request_metrics_passed: true, positive_nan_control_passed: true,
                 strict_capture_tolerance: {atol: B::ATOL, rtol: B::RTOL},
                 no_encode_named_allocations_or_preterminal_readbacks: true,
                 exactly_one_terminal_wait_and_packed_readback_per_request: true,
                 staged_conversion_count: "two weight-only BF16-to-F32 conversions per candidate request, one per block",
                 candidate_correctness_stage_identity_checks: 4,
                 full_generation_executed: false},
    setup_seconds: setup_seconds,
    summary:       {probe_completed: true, candidate: "BF16 weights expanded once per block to shared F32 buffers for both windows", production_promoted: false},
  }.to_json)
ensure
  if owner && owner.not_nil!.lifecycle.release_allowed?
    begin
      owner.not_nil!.release_unsubmitted!
    rescue cleanup_error
      STDERR.puts("PIXELUMM_FFN_WEIGHT_REUSE_OWNER_CLEANUP_FAILURE #{cleanup_error.inspect_with_backtrace}")
    end
  end
end

File.write(report, JSON.build(indent: "  ") { |json| result.not_nil!.to_json(json) })
puts({prefix: "PIXELUMM_FFN_WEIGHT_REUSE", report: report,
      requests: all_requests.size, named_bytes: budget_preflight[:total_bytes],
      baseline_schedule: baseline_schedule}.to_json)
