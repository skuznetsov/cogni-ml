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
  parser.on("--reverse-warmups", "Reverse the two-Plan warmup order") { reverse_warmups = true }
  parser.on("--reverse-first", "Reverse the two-Plan measured ABBA order") { reverse_first = true }
  parser.on("--max-named-bytes N", "Named-buffer preflight cap, at most 8 GiB") { |value| maximum_named_bytes = value.to_i64 }
end

unless ARGV.empty? && {2048, 21120}.includes?(tokens) && {1, 2}.includes?(resident_plans)
  raise ArgumentError.new("invalid PixelUMM Plan reuse probe arguments")
end
# Admit the complete named set before fixture reads or Metal initialization.
B.validate_budget!(tokens, resident_plans, maximum_named_bytes)
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
  active = B::Owner.new(fixture, maximum_named_bytes, resident_plans, shared_gate_up, memory)
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
      profile = active.phase_profile(chain.not_nil!)
      unless profile.shared_gate_up == shared_gate_up
        raise "PixelUMM Plan reuse encoded topology mismatch"
      end
      all_requests << JSON.parse({
        index: index, kind: request[:warmup] ? "warmup" : "timed", warmup: request[:warmup],
        selected_plan: request[:plan_index], block_encode_call_seconds: encode_seconds,
        request_wait_seconds: wait_seconds, terminal_wait_count: chain.not_nil!.wait_count,
        phase_profile: JSON.parse(profile.to_json),
        gpu_metrics: metrics.map_with_index do |metric, slot|
          {slot: slot, stage: slot < 7 ? PixelUMMConnectedResourceProbe::CAPTURE_NAMES[slot] : "output_positive_control",
           failures: metric[:failures], different_words: metric[:different_words],
           nonfinite: metric[:nonfinite], elements: metric[:elements]}
        end,
        named_buffer_allocations_during_request: allocations,
        preterminal_readbacks_during_request: preterminal_readbacks,
        comparator_readbacks_after_wait: readbacks,
        comparator_readback_bytes: PixelUMMConnectedResourceProbe::METRIC_BYTES,
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
  active.release_after_terminal!
  fixture_document = JSON.parse(File.read(fixture.path))
  result = JSON.parse({
    schema:   B::SCHEMA,
    scope:    "One connected native block per request; one versus two resident native Plans with identical borrowed and owned bindings; observer-off production bridge; no generation or production promotion",
    geometry: {grid: fixture.grid, tokens: tokens, hidden: 4096, ffn: 12288, token_order: "tile_major", fixture_rows_used_exactly_once: true},
    fixture:  {path: fixture.path, metadata_sha256: fixture.sha256,
              payload_sha256: fixture_document["payload_sha256"].as_s,
              upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
              checkpoint_subset_sha256: PixelUMMBlockProbe::SUBSET_SHA256,
              capture_sha256: PixelUMMConnectedResourceProbe::CAPTURE_NAMES.to_h { |name| {name, fixture.captures[name].sha256} }},
    comparison: {atol: PixelUMMConnectedResourceProbe::ATOL, rtol: PixelUMMConnectedResourceProbe::RTOL,
                 oracle_scope: "Authenticated saved pinned upstream-source-method MPS fixture chain, not independent backend or newly executed whole upstream block"},
    topology: {resident_native_plans: resident_plans, shared_gate_up: shared_gate_up,
               independent_native_preparation: true, shared_gate_up_prepared_in_every_plan: true,
               block_encodes_per_request: 1, per_graph_resource_observer: false,
               production_workspace_prefix_guards_changed: false, binding_identity: identity,
               native_plan_handles: native_plan_handles},
    schedule: {request_count: requests.size, warmup_count: 2, measured_count: 4,
               reverse_warmups: reverse_warmups, reverse_first: reverse_first,
               request_order_and_samples: all_requests},
    execution: {resources: active.resources, named_bytes: active.named_bytes, maximum_named_bytes: maximum_named_bytes,
                setup_seconds: setup_seconds, successful_terminal_waits: all_requests.size,
                memory_categories_additive: false,
                memory_accounting: "Named MetalBuffers are exact; device allocation, process footprint, RSS and GC counters overlap. Opaque native/MPS ownership is unknown; aggregate counters do not isolate it.",
                memory_events: memory.events},
    validation: {all_reference_comparisons_passed: true, positive_nonfinite_control_passed_each_request: true,
                 no_encode_named_allocations_or_preterminal_readbacks: true, sole_terminal_wait_each_request: true,
                 identical_buffers_and_distinct_native_plans_checked: true},
    source_hashes: {production_bridge_sha256: Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "..", "src", "ml", "video", "pixelumm_decoder_block_mps_bridge.mm"))),
                    comparator_sha256: Digest::SHA256.hexdigest(PixelUMMConnectedResourceProbe::SOURCE)},
    summary: {bounded_probe_completed: true, production_promoted: false},
  }.to_json)
ensure
  if owner && owner.not_nil!.lifecycle.release_allowed?
    owner.not_nil!.release_unsubmitted!
  end
end

File.write(report, JSON.build(indent: "  ") { |json| result.not_nil!.to_json(json) })
puts({prefix: "PIXELUMM_PLAN_REUSE", report: report, resident_plans: resident_plans,
      requests: all_requests.size, named_bytes: owner.not_nil!.named_bytes}.to_json)
