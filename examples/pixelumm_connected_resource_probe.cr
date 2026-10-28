require "option_parser"
require "digest/sha256"
require "json"
require "./support/pixelumm_connected_resource_probe"

alias B = PixelUMMConnectedResourceProbe

reference = ""
reference_sha256 = ""
report = ""
tokens = 21120
shared_gate_up = false
reverse_warmups = false
reverse_first = false
maximum_named_bytes = B::MAX_NAMED_BYTES

OptionParser.parse do |parser|
  parser.banner = "Connected one-block PixelUMM resource probe; no generation."
  parser.on("--reference PATH", "Authenticated saved full-J or true J2048 block fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA256") { |value| reference_sha256 = value }
  parser.on("--rows N", "Exact saved fixture row count: 2048 or 21120") { |value| tokens = value.to_i32 }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--shared", "Use the prepared shared F32 gate/up graph for this process") { shared_gate_up = true }
  parser.on("--reverse-warmups", "Reverse only the first two pool-mode requests") { reverse_warmups = true }
  parser.on("--reverse-first", "Reverse only the measured pool-mode ABBA order") { reverse_first = true }
  parser.on("--max-named-bytes N", "Named-buffer preflight cap, at most 8 GiB") { |value| maximum_named_bytes = value.to_i64 }
end

unless ARGV.empty? && {2048, 21120}.includes?(tokens)
  raise ArgumentError.new("invalid connected PixelUMM resource probe arguments")
end
unless Path[reference].absolute? && File.file?(reference) &&
       Path[report].absolute? && !report.empty? &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("fixture must be absolute and report must be fresh under /Volumes")
end

fixture = PixelUMMBlockProbe::Fixture.new(reference, reference_sha256)
unless fixture.grid.reduce(1) { |product, dim| product * dim } == tokens
  raise ArgumentError.new("connected PixelUMM row count must equal the saved fixture grid")
end
unless PixelUMMBlockProbe.valid_digest?(reference_sha256)
  raise ArgumentError.new("trusted fixture metadata SHA256 is required")
end
fixture_document = JSON.parse(File.read(fixture.path))
payload_sha256 = fixture_document["payload_sha256"].as_s
requests = B.schedule(shared: shared_gate_up,
  reverse_warmups: reverse_warmups, reverse_first: reverse_first)
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(B::Owner?)
result = nil.as(JSON::Any?)
all_requests = [] of JSON::Any
maximum_encode_named_allocations = 0_i64
total_postterminal_readbacks = 0_i64
preterminal_readbacks = 0_i64

begin
  setup_started = Time.instant
  active_owner = B::Owner.new(fixture, maximum_named_bytes, shared_gate_up, memory)
  owner = active_owner
  setup_seconds = (Time.instant - setup_started).total_seconds

  requests.each do |request|
    request_index = request[:index]
    chain = nil.as(ML::Metal::MPSCommandChain?)
    wait_succeeded = false
    graph_info = nil.as(B::NativeInfo?)
    request_readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
    request_allocations_before = ML::MetalBuffer.hy_text_probe_allocations
    wait_seconds = 0.0

    begin
      active_owner.lifecycle.begin_request!
      chain = ML::Metal::MPSCommandChain.new
      chain.not_nil!.enable_api_profile!(phase_roots: false)
      active_owner.preencode!(request_index)
      active_owner.poison_captures_and_clear_metrics!(chain.not_nil!)

      active_owner.begin_graph_events!(request[:pool_mode])
      active_owner.lifecycle.mark_attempt!
      encode_started = Time.instant
      active_owner.session.encode(chain.not_nil!, profile: true)
      encode_seconds = (Time.instant - encode_started).total_seconds
      graph_info = active_owner.finish_graph_events!
      active_owner.postencode!(request_index)
      active_owner.encode_comparisons!(chain.not_nil!)

      request_allocations = ML::MetalBuffer.hy_text_probe_allocations - request_allocations_before
      request_preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - request_readbacks_before
      maximum_encode_named_allocations = Math.max(maximum_encode_named_allocations, request_allocations)
      preterminal_readbacks += request_preterminal_readbacks
      unless request_allocations == 0 && request_preterminal_readbacks == 0 && chain.not_nil!.wait_count == 0
        raise "connected PixelUMM encode path allocated named buffers, read back, or waited before terminal completion"
      end

      wait_started = Time.instant
      chain.not_nil!.commit_and_wait
      wait_succeeded = true
      wait_seconds = (Time.instant - wait_started).total_seconds
      active_owner.postwait!(chain.not_nil!, request_index)
      unless chain.not_nil!.wait_count == 1 && chain.not_nil!.completed?
        raise "connected PixelUMM request did not complete its sole terminal wait"
      end

      info = graph_info.not_nil!
      graph_events = active_owner.read_graph_events!(info, request[:pool_mode], chain.not_nil!)
      metrics = active_owner.metrics(chain.not_nil!)
      request_readback_delta = ML::MetalBuffer.hy_text_probe_readbacks - request_readbacks_before
      total_postterminal_readbacks += request_readback_delta
      unless request_readback_delta == 1
        raise "connected PixelUMM request must perform exactly one post-wait 128-byte metrics readback"
      end

      expected_elements = B::CAPTURE_NAMES.map do |name|
        (fixture.captures[name].nbytes // 2).to_u32
      end
      B::CAPTURE_NAMES.each_with_index do |name, slot|
        metric = metrics[slot]
        unless metric[:elements] == expected_elements[slot] &&
               metric[:failures] == 0 && metric[:nonfinite] == 0
          raise "connected PixelUMM GPU comparison failed for #{name}"
        end
      end
      B.validate_positive_control!(metrics[6], metrics[7])
      unless metrics[7][:failures] == 1 && metrics[7][:nonfinite] == 1 &&
             metrics[7][:elements] == expected_elements[6]
        raise "connected PixelUMM injected nonfinite comparator control was not exact"
      end

      profile = active_owner.session.plan.phase_profile(chain.not_nil!)
      unless profile.shared_gate_up == request[:shared_gate_up] &&
             active_owner.session.plan.profiled_shared_gate_up == request[:shared_gate_up]
        raise "connected PixelUMM encoded topology snapshot differs from the process selection"
      end
      profile_json = JSON.parse(profile.to_json)
      metrics_json = metrics.map_with_index do |metric, slot|
        JSON.parse({slot: slot, stage: slot < 7 ? B::CAPTURE_NAMES[slot] : "output_positive_control",
                    failures: metric[:failures], different_words: metric[:different_words],
                    nonfinite: metric[:nonfinite], elements: metric[:elements]}.to_json)
      end
      events_json = active_owner.native_event_json(graph_events)

      chain.not_nil!.release
      active_owner.after_chain_release!(request_index)
      all_requests << JSON.parse({
        index:                     request_index,
        kind:                      request[:kind],
        warmup:                    request[:warmup],
        pool_mode:                 request[:pool_mode] ? "nested_autoreleasepool" : "outer_block_pool_only",
        requested_shared_gate_up:  request[:shared_gate_up],
        profiled_shared_gate_up:   profile.shared_gate_up,
        terminal_wait_count:       chain.not_nil!.wait_count,
        request_wait_seconds:      wait_seconds,
        block_encode_call_seconds: encode_seconds,
        native_graph_calls:        info.graph_calls,
        native_event_count:        info.event_count,
        native_snapshot_inventory: {
          overflow:       info.overflow != 0,
          sample_failed:  info.sample_failed != 0,
          unknown_graphs: info.unknown_graphs,
          graph_calls:    info.graph_calls,
        },
        graph_events:                            events_json,
        phase_profile:                           profile_json,
        gpu_metrics:                             metrics_json,
        comparator_readback_bytes:               B::METRIC_BYTES,
        comparator_readbacks_after_wait:         request_readback_delta,
        named_buffer_allocations_during_request: request_allocations,
        preterminal_readbacks_during_request:    request_preterminal_readbacks,
      }.to_json)
    rescue error
      if active_owner.native_collection_active?
        begin
          active_owner.finish_graph_events!
        rescue
        end
      end

      if chain && !chain.not_nil!.completed? && chain.not_nil!.wait_count == 0
        begin
          wait_started = Time.instant
          chain.not_nil!.commit_and_wait
          wait_succeeded = true
          wait_seconds = (Time.instant - wait_started).total_seconds
          active_owner.postwait!(chain.not_nil!, request_index) if active_owner.lifecycle.attempted?
        rescue
        end
      end

      if wait_succeeded && chain && chain.not_nil!.completed? && chain.not_nil!.wait_count == 1
        begin
          chain.not_nil!.release
          active_owner.after_chain_release!(request_index)
        rescue
        end
      end
      if active_owner.lifecycle.release_allowed?
        begin
          active_owner.release_after_terminal!
        rescue
        end
      end
      raise error
    end
  end

  active_owner.release_after_terminal!
  expected_graph_calls = B.expected_graph_ids(active_owner.tile_count, shared_gate_up).size
  result = JSON.parse({
    schema:   B::SCHEMA,
    scope:    "connected one-block saved-reference resource observation; production PixelUMM MPSGraph encode; no generation or causal shader attribution",
    geometry: {grid: fixture.grid, tokens: tokens, hidden: B::HIDDEN, ffn: B::FFN,
               token_order: "tile_major", fixture_rows_used_exactly_once: true},
    fixture: {
      path:                     fixture.path,
      metadata_sha256:          fixture.sha256,
      payload_sha256:           payload_sha256,
      upstream_revision:        PixelUMMBlockProbe::UPSTREAM_REVISION,
      checkpoint_subset_sha256: PixelUMMBlockProbe::SUBSET_SHA256,
      capture_sha256:           B::CAPTURE_NAMES.to_h { |name| {name, fixture.captures[name].sha256} },
    },
    comparison: {atol: B::ATOL, rtol: B::RTOL,
                 comparator: "GPU BF16 values; abs(actual-reference) <= atol + rtol*abs(reference); any nonfinite fails"},
    topology: {
      shared_gate_up_process_selection:               shared_gate_up,
      encoded_mode_snapshot_checked_per_request:      true,
      separate_ffn_graph_calls_per_block:             3,
      shared_ffn_graph_calls_per_block:               2,
      total_expected_graph_calls:                     expected_graph_calls,
      graph_builder_names:                            ["gate_projection", "up_projection", "shared_gate_up_projection", "down_projection"],
      down_projection_is_same_production_block_graph: true,
    },
    schedule: {
      request_count:                 requests.size,
      warmup_count:                  2,
      measured_count:                4,
      reverse_warmups:               reverse_warmups,
      reverse_first:                 reverse_first,
      pool_modes:                    requests.map { |request| request[:pool_mode] ? "nested_autoreleasepool" : "outer_block_pool_only" },
      topology_is_fixed_for_process: true,
      request_order_and_samples:     all_requests,
    },
    resource_sampling: {
      hook_location:           "around the unchanged production pixel_encode_graph body; diagnostic bridge macro only",
      per_graph_sample_phases: {pool_off: ["before_encode", "after_encode", "after_scope"],
                                pool_on: ["before_encode", "after_encode", "after_pool_drain"]},
      pool_scope:            "The nested autorelease pool contains only the production graph encode body and its Graph API/result temporaries; surrounding arrays and capture buffers remain under the outer Block pool.",
      timestamp_semantics:   "monotonic timestamp is sampled at observation start; device and process readings are sequential, non-atomic, and may observe prior asynchronous GPU progress.",
      device_counter_name:   "device-reported currentAllocatedSize",
      attribution_limit:     "Snapshots are observational and do not identify machinewide allocation, MPS opaque temporary bytes, exclusive causes, or shader-level effects.",
      native_event_capacity: B::MAX_EVENTS,
      event_abi_bytes:       sizeof(B::NativeEvent),
    },
    execution: {
      resources:                                       active_owner.resources,
      named_bytes:                                     active_owner.named_bytes,
      maximum_named_bytes:                             maximum_named_bytes,
      memory_categories_additive:                      false,
      memory_accounting:                               "Exact declared named buffers are summed. Process footprint, RSS, Metal device allocation, named MetalBuffer counters and Crystal GC readings overlap; opaque MPS temporary bytes are not estimated.",
      setup_seconds:                                   setup_seconds,
      successful_terminal_waits:                       all_requests.size,
      maximum_named_buffer_allocations_during_request: maximum_encode_named_allocations,
      preterminal_readbacks:                           preterminal_readbacks,
      postterminal_metric_readbacks:                   total_postterminal_readbacks,
      metric_readback_bytes_per_request:               B::METRIC_BYTES,
      memory_events:                                   memory.events,
    },
    validation: {
      all_reference_comparisons_passed:                  true,
      positive_nonfinite_control_passed_each_request:    true,
      exact_graph_event_inventory_passed_each_request:   true,
      encoded_topology_snapshot_passed_each_request:     true,
      resource_measurement_is_not_performance_promotion: true,
    },
    source_hashes: {
      diagnostic_bridge_sha256: Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "support", "pixelumm_connected_resource_bridge.mm"))),
      production_bridge_sha256: Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "..", "src", "ml", "video", "pixelumm_decoder_block_mps_bridge.mm"))),
      comparator_sha256:        Digest::SHA256.hexdigest(B::SOURCE),
    },
    summary: {resource_probe_completed: true, production_promoted: false},
  }.to_json)
rescue error
  if owner && owner.not_nil!.lifecycle.release_allowed?
    begin
      owner.not_nil!.release_after_terminal!
    rescue
    end
  end
  raise error
end

File.write(report, JSON.build(indent: "  ") { |json| result.not_nil!.to_json(json) })
puts({prefix: "PIXELUMM_CONNECTED_RESOURCE", report: report,
      requests: all_requests.size, shared_gate_up: shared_gate_up,
      named_bytes: owner.not_nil!.named_bytes, memory_events: memory.events.size}.to_json)
