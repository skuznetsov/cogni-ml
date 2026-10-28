require "option_parser"
require "digest/sha256"
require "./support/pixelumm_ffn_isolated_probe"

alias B = PixelUMMFFNIsolatedProbe

reference = ""
reference_sha = ""
report = ""
tokens = 21120
reverse_warmups = false
reverse_first = false
down_first = false
max_named_bytes = 4_i64 * 1024 * 1024 * 1024

OptionParser.parse do |parser|
  parser.banner = "Isolated PixelUMM FFN graph diagnostic; no attention, decoder or generation."
  parser.on("--reference PATH", "Authenticated full-J or true J2048 suffix fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA256") { |value| reference_sha = value }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--rows N", "Exact fixture row count: 2048 or 21120; no row repetition") { |value| tokens = value.to_i32 }
  parser.on("--reverse-warmups", "Reverse warmup order and graph preparation/release order") { reverse_warmups = true }
  parser.on("--reverse-first", "Reverse only the measured gate/up ABBA starting order") { reverse_first = true }
  parser.on("--down-first", "Run the unchanged down graph before gate/up in each request") { down_first = true }
  parser.on("--max-named-bytes N", "Preallocation cap, at most 4 GiB") { |value| max_named_bytes = value.to_i64 }
end

unless ARGV.empty? && {2048, 21120}.includes?(tokens)
  raise ArgumentError.new("invalid isolated FFN probe arguments")
end
unless Path[reference].absolute? && !report.empty? && Path[report].absolute? &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("fixture must be absolute and report must be fresh under /Volumes")
end

fixture = B::Fixture.new(reference, reference_sha)
unless fixture.metadata.tokens == tokens
  raise ArgumentError.new("isolated FFN row count must equal the saved fixture grid; repetition is forbidden")
end
named_bytes = B.validate_budget!(tokens, max_named_bytes)
requests = B.schedule(reverse_warmups, reverse_first)
stage_order = B.stage_order(down_first)
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(B::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
owner_released = false
terminal_wait_succeeded = false
result = nil.as(JSON::Any?)

begin
  setup_started = Time.instant
  owner = B::Owner.new(fixture, max_named_bytes, reverse_warmups, memory)
  setup_seconds = (Time.instant - setup_started).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  chain.enable_api_profile!(phase_roots: false)
  owner.preencode!

  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  samples = [] of NamedTuple(index: Int32, mode: String, warmup: Bool,
    stages: Hash(String, B::StageTiming))

  requests.each do |request|
    timings = {} of String => B::StageTiming
    stage_order.each do |stage|
      case stage
      when "gate_up"
        owner.poison!(chain, ["candidate.gate", "candidate.up"])
        timing = B.timed_stage(chain, stage) do
          owner.encode_gate_up(chain, request[:mode])
        end
        timings[stage] = timing
        owner.compare_gate_up(chain, request[:index])
      when "down"
        owner.poison!(chain, ["candidate.down"])
        timing = B.timed_stage(chain, stage) do
          owner.encode_down(chain)
        end
        timings[stage] = timing
        owner.copy_down_canonical!(chain) unless owner.canonical_ready?
        owner.compare_down(chain, request[:index])
      else
        raise ArgumentError.new("unknown isolated FFN stage")
      end
      owner.poststage!(stage, request[:index])
    end
    samples << {index: request[:index], mode: request[:mode], warmup: request[:warmup], stages: timings}
  end
  owner.inject_nonfinite_control(chain)

  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  unless encode_allocations == 0 && preterminal_readbacks == 0 && chain.wait_count == 0
    raise "isolated FFN encode path allocated named buffers, read back, or waited"
  end

  wait_started = Time.instant
  chain.commit_and_wait
  terminal_wait_succeeded = true
  terminal_wait_seconds = (Time.instant - wait_started).total_seconds
  unless chain.wait_count == 1 && chain.completed?
    raise "isolated FFN probe did not complete its sole terminal wait"
  end
  owner.postwait!(chain)
  roots = chain.completed_root_timings
  metrics = owner.metrics(chain)
  api_profile = chain.api_profile
  api_profile_json = owner.api_profile_json(api_profile)

  per_sample = samples.map do |sample|
    stage_metrics = {
      "gate_up" => {gate: metrics[sample[:index] * 4], up: metrics[sample[:index] * 4 + 1]},
      "down"    => {expected: metrics[sample[:index] * 4 + 2], canonical: metrics[sample[:index] * 4 + 3]},
    }
    stage_timing = {} of String => JSON::Any
    sample[:stages].each do |stage, timing|
      stage_timing[stage] = JSON.parse({
        stage: timing[:stage], first_root: timing[:first_root], end_root: timing[:end_root],
        encode_wall_seconds: timing[:encode_wall_seconds],
        gpu: B.root_timing(timing, roots),
      }.to_json)
    end
    JSON.parse({index: sample[:index], mode: sample[:mode], warmup: sample[:warmup],
                stages: stage_timing, metrics: stage_metrics}.to_json)
  end

  expected_gate_up_elements = (tokens * B::FFN).to_u32
  expected_down_elements = (tokens * B::HIDDEN).to_u32
  requests.each do |request|
    base = request[:index] * 4
    unless metrics[base][:elements] == expected_gate_up_elements &&
           metrics[base + 1][:elements] == expected_gate_up_elements &&
           metrics[base + 2][:elements] == expected_down_elements &&
           metrics[base + 3][:elements] == expected_down_elements
      raise "isolated FFN GPU comparator did not cover an exact output extent"
    end
  end
  unless metrics[0, 24].all? { |metric| metric[:failures] == 0 && metric[:nonfinite] == 0 } &&
         metrics.each_with_index.all? { |metric, index| index == 24 || metric[:elements] == (index % 4 < 2 ? expected_gate_up_elements : expected_down_elements) }
    raise "isolated FFN GPU comparator rejected a captured output or reported an invalid element count"
  end
  unless metrics[24][:failures] == 1 && metrics[24][:nonfinite] == 1 &&
         metrics[24][:elements] == expected_down_elements
    raise "isolated FFN positive control did not detect its injected nonfinite value"
  end
  unless requests.all? { |request| metrics[request[:index] * 4 + 3][:different_words] == 0 }
    raise "unchanged down graph output was not bitwise identical across requests"
  end

  summaries = ["separate", "shared"].map do |mode|
    selected = per_sample.select { |sample| sample["mode"].as_s == mode && !sample["warmup"].as_bool }
    stage_summary = ["gate_up", "down"].map do |stage|
      values = selected.map { |sample| sample["stages"][stage]["gpu"]["gpu_root_union_seconds"].as_f }
      {
        stage: stage, median_gpu_root_union_seconds: B.median(values),
        min_gpu_root_union_seconds: values.min, max_gpu_root_union_seconds: values.max,
        all_root_timings_isolated: selected.all? { |sample| sample["stages"][stage]["gpu"]["isolated_root_timing"].as_bool },
        maximum_outside_root_overlap_seconds: selected.max_of { |sample| sample["stages"][stage]["gpu"]["outside_root_overlap_seconds"].as_f },
      }
    end
    {mode: mode, measured_requests: selected.size, stages: stage_summary}
  end
  separate = summaries.find { |summary| summary[:mode] == "separate" }.not_nil!
  shared = summaries.find { |summary| summary[:mode] == "shared" }.not_nil!
  separate_gate = separate[:stages].find { |stage| stage[:stage] == "gate_up" }.not_nil![:median_gpu_root_union_seconds]
  shared_gate = shared[:stages].find { |stage| stage[:stage] == "gate_up" }.not_nil![:median_gpu_root_union_seconds]
  separate_down = separate[:stages].find { |stage| stage[:stage] == "down" }.not_nil![:median_gpu_root_union_seconds]
  shared_down = shared[:stages].find { |stage| stage[:stage] == "down" }.not_nil![:median_gpu_root_union_seconds]
  ratio_reference = {
    descriptive_slowdown_reference:   1.10,
    interpretation:                   "10% is descriptive only; this diagnostic applies no performance promotion threshold",
    gate_up_shared_to_separate_ratio: shared_gate / separate_gate,
    down_shared_to_separate_ratio:    shared_down / separate_down,
  }

  source = fixture.metadata.source_replay
  result = JSON.parse({
    schema:   "cogni_ml.pixelumm_ffn_isolated_probe/v1",
    scope:    "saved full-J or true J2048 FFN boundaries; production MPSGraph source-method diagnostic; no attention execution, decoder, generation, or causal shader claim",
    geometry: {grid: fixture.metadata.grid, tokens: tokens, hidden: B::HIDDEN, ffn: B::FFN,
               token_order: "tile_major", fixture_rows_used_exactly_once: true},
    fixture: {path: fixture.path, metadata_sha256: fixture.sha256,
              payload_sha256: fixture.metadata.payload_sha256,
              upstream_revision: B::UPSTREAM_REVISION, upstream_model_sha256: B::MODEL_SHA256,
              source_replay: source},
    comparison: {atol: B::ATOL, rtol: B::RTOL, comparator: "GPU BF16 values, abs(a-e) <= atol + rtol*abs(e); nonfinite always fails"},
    topology:   {gate_up_modes: ["separate_gate_then_up", "shared_input_pair"],
               production_builders: ["pixel_linear_step(castOperandsToF32=true)", "pixel_linear_shared_input_pair_step"],
               input_boundary: "BF16; F32 operands; BF16 output", down_graph_same_prepared_executable_for_all_requests: true,
               down_graph: "pixel_linear_step(castOperandsToF32=true)", common_candidate_output_buffers: true,
               down_canonical_buffer_bytes: fixture.metadata.captures["down_proj"].nbytes,
               down_canonical_word_differences_by_request: requests.map { |request| metrics[request[:index] * 4 + 3][:different_words] }},
    schedule: {warmup_order: requests[0, 2].map(&.[:mode]), measured_order: requests[2, 4].map(&.[:mode]),
               reverse_warmups: reverse_warmups, reverse_first: reverse_first,
               plan_compile_and_release_order: owner.plan_order, down_first: down_first, stage_order: stage_order,
               confounding_note: "reverse-warmups changes warmup order and graph preparation/release order together; reverse-first changes only measured gate/up order"},
    timing: {unit: "union of completed GPU root lifetimes for the named stage; overlaps are reported and can disqualify exclusive timing",
             excluded_from_stage: "fixture upload, graph preparation, poison/canonical blit, GPU comparisons, terminal wait and host readback",
             stage_host_encode_wall_scope: "gate_up combines both separate graph encodes or the one shared graph encode; no per-projection host attribution",
             api_profile_scope: "whole command chain; not attributed to individual stages",
             summaries: summaries, samples: per_sample, ratio_reference: ratio_reference},
    execution: {named_bytes: named_bytes, resources: owner.resources, maximum_named_bytes: max_named_bytes,
                memory_accounting: "named buffers are summed by exact declared extent; process/device readings are overlapping, nonadditive phase snapshots; opaque MPS temporary allocation bytes are not estimated",
                setup_seconds: setup_seconds, terminal_wait_seconds: terminal_wait_seconds, terminal_waits: chain.wait_count,
                preterminal_named_allocations: encode_allocations, preterminal_readbacks: preterminal_readbacks,
                postterminal_readbacks: ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before,
                command_roots: roots.size, api_profile: api_profile_json,
                root_timings: roots.map { |root| {sequence: root.sequence, completed: root.completed,
                                                  has_error: root.has_error, gpu_start_seconds: root.gpu_start_seconds, gpu_end_seconds: root.gpu_end_seconds} }},
    gpu_comparison: {metrics: metrics.map_with_index { |metric, index| {slot: index,
                                                                        failures: metric[:failures], different_words: metric[:different_words],
                                                                        nonfinite: metric[:nonfinite], elements: metric[:elements]} },
                     positive_control: metrics[24], all_reference_comparisons_passed: metrics[0, 24].all? { |metric| metric[:failures] == 0 && metric[:nonfinite] == 0 },
                     down_bitwise_unchanged_across_all_requests: requests.all? { |request| metrics[request[:index] * 4 + 3][:different_words] == 0 }},
    source_hashes: {diagnostic_bridge_sha256: Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "support", "pixelumm_ffn_isolated_bridge.mm"))),
                    production_bridge_sha256: Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "..", "src", "ml", "video", "pixelumm_decoder_block_mps_bridge.mm"))),
                    comparator_sha256: Digest::SHA256.hexdigest(B::SOURCE)},
    summary: {diagnostic_completed: true, production_promoted: false},
  }.to_json)

  owner.release_after_terminal!(chain)
  owner_released = true
  chain.release
  memory.emit("after_chain_release", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
  report_result = result.not_nil!
  report_result.as_h["memory_events"] = JSON.parse(memory.events.to_json)
  report_fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise IO::Error.from_errno("create isolated FFN report") if report_fd < 0
  report_file = IO::FileDescriptor.new(report_fd)
  begin
    report_file.puts(report_result.to_pretty_json)
  ensure
    report_file.close
  end
  puts report_result["summary"].to_json
ensure
  active_owner = owner
  if active_chain = chain
    unless active_chain.completed?
      active_chain.commit_and_wait
      terminal_wait_succeeded = true
    end
    if active_owner && !owner_released
      if !active_owner.lifecycle.attempted?
        active_owner.release_after_terminal!(active_chain)
        owner_released = true
      elsif terminal_wait_succeeded && active_chain.completed? && active_chain.wait_count == 1
        unless active_owner.lifecycle.terminal_wait_succeeded?
          active_owner.postwait!(active_chain)
        end
        active_owner.release_after_terminal!(active_chain)
        owner_released = true
      end
    end
    active_chain.release if terminal_wait_succeeded && active_chain.drained?
  elsif active_owner && !owner_released
    active_owner.release_unsubmitted!
    owner_released = true
  end
end
