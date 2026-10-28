require "option_parser"
require "digest/sha256"
require "json"
require "./support/pixelumm_plan_reuse_probe"

alias B = PixelUMMPlanReuseProbe

reference = ""
reference_sha256 = ""
report = ""
tokens = 21120
resident_plans = 1
shared_gate_up = false
softmax_pv = false
check_softmax_pv_f32 = false
reverse_warmups = false
reverse_first = false
maximum_named_bytes = PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES

OptionParser.parse do |parser|
  parser.banner = "One versus two native PixelUMM Plans with identical shared bindings; no generation."
  parser.on("--reference PATH", "Authenticated saved connected-block fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA256") { |value| reference_sha256 = value }
  parser.on("--rows N", "Exact saved fixture row count: 2048 or 21120") { |value| tokens = value.to_i32 }
  parser.on("--plans N", "Resident native Plans: one or two") { |value| resident_plans = value.to_i32 }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--shared", "Use shared gate/up in every resident Plan") { shared_gate_up = true }
  parser.on("--softmax-pv", "Compare separate attention graphs with fused F32 softmax/PV in two Plans") { softmax_pv = true }
  parser.on("--check-softmax-pv-f32", "Correctness-only direct F32 softmax/PV path comparison; requires --softmax-pv") { check_softmax_pv_f32 = true }
  parser.on("--reverse-warmups", "Reverse the two-Plan warmup order") { reverse_warmups = true }
  parser.on("--reverse-first", "Reverse the two-Plan measured ABBA order") { reverse_first = true }
  parser.on("--max-named-bytes N", "Named-buffer preflight cap, at most 8 GiB") { |value| maximum_named_bytes = value.to_i64 }
end

unless ARGV.empty? && {2048, 21120}.includes?(tokens) && {1, 2}.includes?(resident_plans)
  raise ArgumentError.new("invalid PixelUMM Plan reuse probe arguments")
end
B.validate_softmax_pv_configuration!(resident_plans, shared_gate_up, softmax_pv)
B.validate_softmax_pv_f32_configuration!(softmax_pv, check_softmax_pv_f32,
  resident_plans, shared_gate_up)
# Admit the complete named set before fixture reads or Metal initialization.
B.validate_budget!(tokens, resident_plans, maximum_named_bytes, check_softmax_pv_f32)
unless Path[reference].absolute? && File.file?(reference) &&
       Path[report].absolute? && !report.empty? &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("fixture must be absolute and report must be fresh under /Volumes")
end

fixture = PixelUMMBlockProbe::Fixture.new(reference, reference_sha256)
PixelUMMConnectedResourceProbe.validate_fixture!(fixture, tokens)
requests = B.schedule(resident_plans, reverse_warmups: reverse_warmups, reverse_first: reverse_first)
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(B::Owner?)
result = nil.as(JSON::Any?)
all_requests = [] of JSON::Any

begin
  setup_started = Time.instant
  active = B::Owner.new(fixture, maximum_named_bytes, resident_plans,
    shared_gate_up, memory, prepare_softmax_pv: softmax_pv,
    check_softmax_pv_f32: check_softmax_pv_f32)
  owner = active
  setup_seconds = (Time.instant - setup_started).total_seconds

  requests.each do |request|
    index = request[:index]
    chain = nil.as(ML::Metal::MPSCommandChain?)
    wait_succeeded = false
    readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
    allocations_before = ML::MetalBuffer.hy_text_probe_allocations
    begin
      chain = ML::Metal::MPSCommandChain.new
      chain.not_nil!.enable_api_profile!(phase_roots: false)
      active.preencode!(index)
      active.begin_request!(chain.not_nil!, request[:plan_index])
      request_started = Time.instant
      active.poison_captures_and_clear_metrics!(chain.not_nil!)
      started = Time.instant
      active.encode!(chain.not_nil!, request[:plan_index])
      encode_seconds = (Time.instant - started).total_seconds
      active.encode_comparisons!(chain.not_nil!)
      active.postencode!(index)
      allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      unless allocations == 0 && preterminal_readbacks == 0 && chain.not_nil!.wait_count == 0
        raise "PixelUMM Plan reuse request allocated named buffers, read back or waited before completion"
      end
      started = Time.instant
      chain.not_nil!.commit_and_wait
      wait_succeeded = true
      wait_seconds = (Time.instant - started).total_seconds
      request_wall_seconds = (Time.instant - request_started).total_seconds
      active.postwait!(chain.not_nil!, index)
      metrics = active.metrics(chain.not_nil!)
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      raise "PixelUMM Plan reuse requires exactly one metrics readback" unless readbacks == 1
      PixelUMMConnectedResourceProbe::CAPTURE_NAMES.each_with_index do |name, slot|
        metric = metrics[slot]
        unless metric[:elements] == (fixture.captures[name].nbytes // 2).to_u32 &&
               metric[:failures] == 0 && metric[:nonfinite] == 0
          raise "PixelUMM Plan reuse GPU upstream comparison failed: #{name}"
        end
      end
      PixelUMMConnectedResourceProbe.validate_positive_control!(metrics[6], metrics[7])
      unless metrics[7][:failures] == 1 && metrics[7][:nonfinite] == 1
        raise "PixelUMM Plan reuse NaN control was not exact"
      end
      if check_softmax_pv_f32
        B.validate_softmax_pv_f32_metrics!(metrics,
          active.f32_geometry || raise("direct F32 check has no geometry inventory"))
      end
      profile = active.phase_profile(chain.not_nil!)
      expected_fused_softmax_pv = active.fused_softmax_pv_for_plan(request[:plan_index])
      unless profile.shared_gate_up == shared_gate_up &&
             profile.fused_softmax_pv == expected_fused_softmax_pv &&
             profile.check_softmax_pv_f32 == check_softmax_pv_f32
        raise "PixelUMM Plan reuse encoded topology mismatch"
      end
      all_requests << JSON.parse({
        index: index,
        kind: check_softmax_pv_f32 ? "correctness_only" : (request[:warmup] ? "warmup" : "timed"),
        warmup: request[:warmup],
        timing_eligible: !check_softmax_pv_f32 && !request[:warmup],
        selected_plan: request[:plan_index],
        attention_mode: profile.fused_softmax_pv ? "fused_softmax_pv" : "separate_softmax_pv",
        attention_graph_topology: profile.attention_graph_topology,
        request_wall_seconds: request_wall_seconds,
        block_encode_call_seconds: encode_seconds,
        request_wait_seconds: wait_seconds, terminal_wait_count: chain.not_nil!.wait_count,
        phase_profile: JSON.parse(profile.to_json),
        gpu_metrics: metrics.map_with_index do |metric, slot|
          stage = case slot
                  when 0..6 then PixelUMMConnectedResourceProbe::CAPTURE_NAMES[slot]
                  when 7    then "output_positive_control"
                  when 8    then "local_probabilities_f32"
                  when 9    then "local_attended_f32"
                  when 10   then "local_f32_positive_control"
                  else           raise "unexpected PixelUMM metric slot #{slot}"
                  end
          {slot: slot, stage: stage,
           failures: metric[:failures], different_words: metric[:different_words],
           nonfinite: metric[:nonfinite], elements: metric[:elements]}
        end,
        named_buffer_allocations_during_request: allocations,
        preterminal_readbacks_during_request: preterminal_readbacks,
        comparator_readbacks_after_wait: readbacks,
        comparator_readback_bytes: active.metric_bytes,
      }.to_json)
      chain.not_nil!.release
      active.after_chain_release!(index)
    rescue error
      # Cleanup must not hide the first failure when a guard rejects teardown.
      STDERR.puts("PIXELUMM_PLAN_REUSE_REQUEST_FAILURE #{error.inspect_with_backtrace}")
      active.abort_request!
      # A partial enqueue may need the same sole terminal drain. A failed wait
      # is never retried and does not authorize releasing shared buffers.
      if chain && !chain.not_nil!.completed? && chain.not_nil!.wait_count == 0
        begin
          chain.not_nil!.commit_and_wait
          wait_succeeded = true
          active.postwait!(chain.not_nil!, index) if active.lifecycle.attempted?
        rescue
        end
      end
      if wait_succeeded && chain && chain.not_nil!.completed? && chain.not_nil!.wait_count == 1
        begin
          chain.not_nil!.release
        rescue
        end
      end
      raise error
    end
  end

  identity = active.named_resources_json
  native_plan_handles = active.native_plan_handles
  attention_plan_modes = Array.new(resident_plans) do |plan_index|
    fused = active.fused_softmax_pv_for_plan(plan_index.to_i32)
    {native_plan_index: plan_index, attention_mode: fused ? "fused_softmax_pv" : "separate_softmax_pv",
     attention_graph_topology: fused ? "separate_qk_fused_softmax_pv" : "separate_qk_separate_softmax_pv"}
  end
  f32_geometry = check_softmax_pv_f32 ? B.softmax_pv_f32_geometry(tokens) : nil
  active.release_after_terminal!
  fixture_document = JSON.parse(File.read(fixture.path))
  result = JSON.parse({
    schema:   check_softmax_pv_f32 ? B::SOFTMAX_PV_F32_SCHEMA : (softmax_pv ? B::SOFTMAX_PV_SCHEMA : B::SCHEMA),
    scope:    "One connected native block per request; one versus two resident native Plans with identical borrowed and owned bindings; observer-off production bridge; no generation or production promotion",
    geometry: {grid: fixture.grid, tokens: tokens, hidden: 4096, ffn: 12288, token_order: "tile_major", fixture_rows_used_exactly_once: true},
    fixture:  {path: fixture.path, metadata_sha256: fixture.sha256,
              payload_sha256: fixture_document["payload_sha256"].as_s,
              upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
              checkpoint_subset_sha256: PixelUMMBlockProbe::SUBSET_SHA256,
              capture_sha256: PixelUMMConnectedResourceProbe::CAPTURE_NAMES.to_h { |name| {name, fixture.captures[name].sha256} }},
    comparison: {atol: PixelUMMConnectedResourceProbe::ATOL, rtol: PixelUMMConnectedResourceProbe::RTOL,
                 oracle_scope: "Authenticated saved pinned upstream-source-method MPS fixture chain, not independent backend or newly executed whole upstream block",
                 numerical_claim: "Per-plan capture parity at the configured tolerances only; no losslessness or broader precision claim."},
    direct_f32_check: check_softmax_pv_f32 ? {
      enabled: true, correctness_only: true, timing_eligible: false,
      comparison: "GPU F32 softmax probabilities and attended outputs from separate and fused softmax/PV paths",
      atol: B::ATOL, rtol: B::RTOL,
      probability_snapshot_bytes: f32_geometry.not_nil![:probabilities_snapshot_bytes],
      attended_snapshot_bytes: f32_geometry.not_nil![:attended_snapshot_bytes],
      total_probability_elements: f32_geometry.not_nil![:probability_elements],
      total_attended_elements: f32_geometry.not_nil![:attended_elements],
      tile_count: f32_geometry.not_nil![:tiles].size,
      tile_geometries: f32_geometry.not_nil![:tiles],
      metrics: {probabilities_slot: 8, probabilities_stage: "local_probabilities_f32",
                attended_slot: 9, attended_stage: "local_attended_f32",
                positive_control_slot: 10, positive_control_stage: "local_f32_positive_control",
                metric_slots: B::F32_CHECK_METRIC_SLOTS, readback_bytes: B::F32_CHECK_METRIC_BYTES},
      comparison_semantics: "No BF16 probability cast; observed F32 parity is limited to the checked inputs and geometries.",
    } : nil,
    topology: {resident_native_plans: resident_plans, shared_gate_up: shared_gate_up,
               softmax_pv_experiment: softmax_pv,
               direct_f32_correctness_check: check_softmax_pv_f32,
               timing_eligible: !check_softmax_pv_f32,
               softmax_pv_plan_modes: attention_plan_modes,
               independent_native_preparation: true, shared_gate_up_prepared_in_every_plan: true,
               block_encodes_per_request: 1, per_graph_resource_observer: false,
               production_workspace_prefix_guards_changed: false, binding_identity: identity,
               native_plan_handles: native_plan_handles},
    schedule: {request_count: requests.size, warmup_count: 2,
               measured_count: check_softmax_pv_f32 ? 0 : 4,
               correctness_request_count: check_softmax_pv_f32 ? requests.size : 0,
               timing_matrix_eligible: !check_softmax_pv_f32,
               reverse_warmups: reverse_warmups, reverse_first: reverse_first,
               request_order_and_samples: all_requests},
    execution: {resources: active.resources, named_bytes: active.named_bytes, maximum_named_bytes: maximum_named_bytes,
                setup_seconds: setup_seconds, successful_terminal_waits: all_requests.size,
                memory_categories_additive: false,
                memory_accounting: "Named MetalBuffers are exact; device allocation, process footprint, RSS and GC counters overlap. Opaque native/MPS ownership is unknown; aggregate counters do not isolate it.",
                memory_events: memory.events},
    validation: {all_reference_comparisons_passed: true, positive_nonfinite_control_passed_each_request: true,
                 direct_f32_softmax_pv_comparisons_passed_each_request: check_softmax_pv_f32,
                 direct_f32_nan_positive_control_passed_each_request: check_softmax_pv_f32,
                 no_encode_named_allocations_or_preterminal_readbacks: true, sole_terminal_wait_each_request: true,
                 identical_buffers_and_distinct_native_plans_checked: true},
    source_hashes: {production_bridge_sha256: Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "..", "src", "ml", "video", "pixelumm_decoder_block_mps_bridge.mm"))),
                    comparator_sha256: Digest::SHA256.hexdigest(PixelUMMConnectedResourceProbe::SOURCE)},
    summary: {bounded_probe_completed: true, correctness_only: check_softmax_pv_f32,
              timing_matrix_eligible: !check_softmax_pv_f32, production_promoted: false},
  }.to_json)
ensure
  if owner && owner.not_nil!.lifecycle.release_allowed?
    owner.not_nil!.release_unsubmitted!
  end
end

File.write(report, JSON.build(indent: "  ") { |json| result.not_nil!.to_json(json) })
puts({prefix: "PIXELUMM_PLAN_REUSE", report: report, resident_plans: resident_plans,
      requests: all_requests.size, named_bytes: owner.not_nil!.named_bytes}.to_json)
