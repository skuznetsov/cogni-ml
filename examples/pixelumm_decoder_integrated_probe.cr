require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_decoder_probe"
require "../src/ml/video/pixelumm_decoder_profile"

reference, reference_sha, alternate, alternate_sha = "", "", "", ""
head_reference, head_reference_sha, head_alternate, head_alternate_sha = "", "", "", ""
cache_path, cache_sha, report = "", "", ""
budget = 0_i64
repeats = 1
profile_variance = false
compare_shared_gate_up = false
OptionParser.parse do |parser|
  parser.banner = "PixelUMM 36 blocks plus global head, full J, one chain; external 10% memory guard required."
  parser.on("--reference PATH", "Authenticated full36 block fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted full36 metadata digest") { |v| reference_sha = v }
  parser.on("--alternate-reference PATH", "Alternate full36 fixture") { |v| alternate = v }
  parser.on("--alternate-reference-sha256 SHA", "Trusted alternate full36 digest") { |v| alternate_sha = v }
  parser.on("--head-reference PATH", "Head fixture linked to the first block35 edge") { |v| head_reference = v }
  parser.on("--head-reference-sha256 SHA", "Trusted head metadata digest") { |v| head_reference_sha = v }
  parser.on("--head-alternate-reference PATH", "Head fixture linked to the alternate block35 edge") { |v| head_alternate = v }
  parser.on("--head-alternate-reference-sha256 SHA", "Trusted alternate head metadata digest") { |v| head_alternate_sha = v }
  parser.on("--bf16-cache PATH", "Authenticated full36 BF16 weight cache") { |v| cache_path = v }
  parser.on("--bf16-cache-sha256 SHA", "Trusted BF16 cache metadata digest") { |v| cache_sha = v }
  parser.on("--report PATH", "New comparison and memory report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate unique named budget, at most 24 GiB") { |v| budget = v.to_i64 }
  parser.on("--repeats N", "Repeat requests after the gate, 1..3") { |v| repeats = v.to_i }
  parser.on("--profile-variance", "Six bounded phase/API/continuation controls; repeats must be 1") { profile_variance = true }
  parser.on("--compare-shared-gate-up", "Warm both F32 graph modes, unprofiled ABBA and two poisoned-input gates") { compare_shared_gate_up = true }
end
unless ARGV.empty? && {reference, reference_sha, alternate, alternate_sha,
                       head_reference, head_reference_sha, head_alternate, head_alternate_sha,
                       cache_path, cache_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 24_i64 * 1024**3 && 1 <= repeats <= 3 &&
       (!(profile_variance || compare_shared_gate_up) || repeats == 1) &&
       !(profile_variance && compare_shared_gate_up)
  raise ArgumentError.new("expected authenticated paired full-J fixtures, cache, budget and fresh report")
end
raise ArgumentError.new("integrated report must be absolute") unless Path[report].absolute?
fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("integrated report must be fresh and writable") if fd < 0
report_io = IO::FileDescriptor.new(fd)
observer = PixelUMMMemoryProbe::Recorder.system
session = nil.as(PixelUMMDecoderProbe::Session?)
chain = nil.as(ML::Metal::MPSCommandChain?)
requests = [] of JSON::Any
capture_order = [] of String
named_bytes, block_bytes, head_owned_bytes, head_weight_bytes = 0_i64, 0_i64, 0_i64, 0_i64
preparation_seconds = 0.0
failure = nil.as(String?)
passed = true
schedule = if compare_shared_gate_up
             PixelUMMDecoderProbe.shared_gate_up_schedule
           elsif profile_variance
             PixelUMMDecoderProbe.variance_schedule
           else
             Array(PixelUMMDecoderProbe::RequestArm).new(repeats + 2) do |index|
               poisoned = index == repeats + 1
               kind = poisoned ? "poisoned_different_input_gate" : (index == 0 ? "numerical_gate" : "timed")
               PixelUMMDecoderProbe::RequestArm.new(kind, poisoned, poisoned, false)
             end
           end
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
  observer.emit("after_metadata_and_head_payload_authentication")
  start = Time.instant
  session = PixelUMMDecoderProbe::Session.new(fixture, cache,
    max_named_bytes: budget, phase_observer: observer,
    prepare_shared_gate_up: compare_shared_gate_up)
  preparation_seconds = (Time.instant - start).total_seconds
  named_bytes, block_bytes = session.named_bytes, session.block_named_bytes
  head_owned_bytes, head_weight_bytes = session.head_owned_named_bytes, session.head_weight_named_bytes
  capture_order = session.capture_order
  unless capture_order.size == 43 && session.head_input_aliases_block35_output? &&
         session.head_plan.input.same?(session.blocks.workspace.output_buffer(35)) &&
         named_bytes == block_bytes + head_owned_bytes + head_weight_bytes
    raise "integrated unique named inventory, capture coverage or direct GPU edge mismatch"
  end
  observer.emit("session_ready")
  STDERR.puts "Integrated PixelUMM admitted #{named_bytes} unique named bytes, full J=21120"
  current_alternate = false
  head_poison_count = 0
  segments_by_mode = {} of Bool => Int32
  schedule.each_with_index do |arm, request|
    poisoned = arm.poisoned
    if arm.alternate != current_alternate
      target = arm.alternate ? replay : fixture
      session.replay_input!(target)
      current_alternate = arm.alternate
      observer.emit("after_replay_input", request: request)
    end
    if poisoned
      session.poison_transients!
      head_poison_count += 1
      unless session.head_plan.poison_count == head_poison_count && session.head_plan.last_poisoned_bytes == 1_124_766_720_i64
        raise "integrated replay omitted head transient poison"
      end
      observer.emit("after_poisoned_replay_input", request: request)
    end
    session.use_shared_gate_up!(arm.shared_gate_up) if compare_shared_gate_up
    unless session.blocks.plans.all? { |plan| plan.shared_gate_up == arm.shared_gate_up }
      raise "integrated gate/up selection must cover every block"
    end
    chain = ML::Metal::MPSCommandChain.new
    if arm.profile
      chain.enable_api_profile!(phase_roots: false)
      chain.enable_continuation_trace!
    end
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    head_before_blocks_rejected = false
    early_release_rejected, early_read_rejected, early_reuse_rejected, early_poison_rejected = false, false, false, false
    early_mode_change_rejected = false
    unsupported_profile_rejected_without_submission = false
    if request == 0
      if compare_shared_gate_up
        prior_segments = chain.segment_count
        begin
          session.encode_blocks(chain, profile: true)
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("not qualified")
          unsupported_profile_rejected_without_submission = true
        end
        # This fails if rejection incorrectly latched the empty chain. No
        # terminal wait is allowed between rejection and this idle operation.
        session.use_shared_gate_up!(arm.shared_gate_up)
        unless unsupported_profile_rejected_without_submission && chain.segment_count == prior_segments &&
               chain.wait_count == 0 && ML::MetalBuffer.hy_text_probe_readbacks == reads &&
               ML::MetalBuffer.hy_text_probe_allocations == allocations
          raise "unsupported shared gate/up profiling changed submission state"
        end
      end
      begin
        session.encode_head(chain)
      rescue error : ArgumentError
        raise error unless (error.message || "").downcase.includes?("block")
        head_before_blocks_rejected = true
      end
      raise "integrated head accepted encoding before block35" unless head_before_blocks_rejected
    end
    observer.emit("request_before_encode", request: request)
    execution_start = Time.instant
    start = Time.instant
    session.encode_blocks(chain, profile: arm.profile)
    blocks_encode_seconds = (Time.instant - start).total_seconds
    observer.emit("request_after_blocks_encode", request: request)
    start = Time.instant
    session.encode_head(chain)
    head_encode_seconds = (Time.instant - start).total_seconds
    encode_span_seconds = (Time.instant - execution_start).total_seconds
    observer.emit("request_after_head_encode", request: request)
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0 &&
           session.head_input_aliases_block35_output?
      raise "integrated encode allocated named buffers, read activations, waited or copied the edge"
    end
    early_profile_rejected = false
    if arm.profile
      begin
        session.blocks.plans.first.phase_markers(chain)
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("terminal wait")
        early_profile_rejected = true
      end
      raise "integrated profile read admitted before terminal wait" unless early_profile_rejected
    end
    if request == 0
      if compare_shared_gate_up
        before_modes = session.blocks.plans.map(&.shared_gate_up)
        begin
          session.use_shared_gate_up!(!arm.shared_gate_up)
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("in flight")
          early_mode_change_rejected = true
        end
        unless early_mode_change_rejected && session.blocks.plans.map(&.shared_gate_up) == before_modes
          raise "integrated in-flight gate/up selection mutated one or more blocks"
        end
      end
      begin
        session.release
      rescue error : ArgumentError
        early_release_rejected = true
      end
      begin
        session.compare
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("terminal wait")
        early_read_rejected = true
      end
      begin
        session.encode(chain)
      rescue error : ArgumentError
        early_reuse_rejected = true
      end
      begin
        session.poison_transients!
      rescue error : ArgumentError
        early_poison_rejected = true
      end
      unless early_release_rejected && early_read_rejected && early_reuse_rejected && early_poison_rejected &&
             ML::MetalBuffer.hy_text_probe_readbacks == reads &&
             ML::MetalBuffer.hy_text_probe_allocations == allocations && chain.wait_count == 0
        raise "integrated lifecycle checks admitted an unsafe in-flight operation"
      end
    end
    start = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - start).total_seconds
    execution_seconds = (Time.instant - execution_start).total_seconds
    observer.emit("request_after_terminal_wait", request: request)
    # All metadata extraction is after the existing wait and outside execution
    # timing. It does not read activations, allocate GPU buffers or enqueue work.
    root_profile = nil.as(JSON::Any?)
    phase_profile = nil.as(JSON::Any?)
    trace_profile = nil.as(JSON::Any?)
    if profile_variance || compare_shared_gate_up
      roots = chain.completed_root_timings
      whole = ML::Video::MPSCommandPhaseProfile.build(roots,
        [ML::Video::MPSCommandPhaseRange.new("whole_decoder", 0, roots.size.to_i32)],
        chain.gpu_elapsed_seconds)
      root_profile = JSON.parse({
        root_count: whole.root_count, root_duration_seconds: whole.root_duration_seconds,
        envelope_seconds: whole.envelope_seconds, union_seconds: whole.union_seconds,
        overlap_seconds: whole.overlap_seconds,
        roots: roots.map { |root| {sequence: root.sequence, completed: root.completed,
                                   has_error: root.has_error, gpu_start_seconds: root.gpu_start_seconds,
                                   gpu_end_seconds: root.gpu_end_seconds} },
      }.to_json)
      if arm.profile
        markers = session.blocks.plans.map(&.phase_markers(chain))
        phases = ML::Video::PixelUMMDecoderPhaseProfile.build(markers,
          session.blocks.plans.map(&.local_attention_tile_count),
          markers.last.last.root_count, roots.size.to_i32, roots,
          chain.gpu_elapsed_seconds, chain.api_profile)
        phase_profile = JSON.parse(phases.to_json)
        trace = chain.continuation_trace
        trace_profile = JSON.parse({info: trace.info, events: trace.events, roots: trace.roots}.to_json)
      end
      if profile_variance
        raise "integrated profile changed segment topology" unless chain.segment_count == 4579
      else
        segments_by_mode[arm.shared_gate_up] ||= chain.segment_count
        unless chain.segment_count == segments_by_mode[arm.shared_gate_up]
          raise "integrated gate/up mode changed segment topology between requests"
        end
      end
      unless chain.wait_count == 1 && ML::MetalBuffer.hy_text_probe_readbacks == reads &&
             ML::MetalBuffer.hy_text_probe_allocations == allocations
        raise "integrated profiling added waits, activation reads or named allocations"
      end
    end
    comparisons = session.compare(request: request)
    observer.emit("request_after_diagnostic_comparisons", request: request)
    unless comparisons.map { |c| c[:stage] } == capture_order &&
           ML::MetalBuffer.hy_text_probe_readbacks - reads == 43 && chain.wait_count == 1
      raise "integrated comparisons omitted a retained capture or changed terminal waits"
    end
    numerical_pass = comparisons.all? { |c| c[:finite] && c[:failures] == 0 }
    passed &&= numerical_pass
    unless numerical_pass
      first_bad = comparisons.find { |c| !c[:finite] || c[:failures] != 0 }.not_nil!
      failure = "numerical mismatch at request #{request}, stage #{first_bad[:stage]}"
    end
    kind = arm.kind
    request_input_sha = session.blocks.fixture.inputs["x"].sha256
    requests << JSON.parse({
      kind: kind, request: request, passed: numerical_pass, input_sha256: request_input_sha,
      head_input_is_block35_output: session.head_input_aliases_block35_output?,
      blocks_encode_seconds: blocks_encode_seconds, head_encode_seconds: head_encode_seconds,
      encode_seconds: blocks_encode_seconds + head_encode_seconds,
      observed_encode_span_seconds: encode_span_seconds,
      terminal_wait_seconds: wait_seconds, total_seconds: execution_seconds,
      gpu_elapsed_seconds: chain.gpu_elapsed_seconds, command_segments: chain.segment_count,
      terminal_wait_count: chain.wait_count, named_allocations_during_encode: encode_allocations,
      activation_readbacks_before_terminal_wait: early_reads,
      diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
      head_before_blocks_rejected: head_before_blocks_rejected,
      early_release_rejected: early_release_rejected, early_read_rejected: early_read_rejected,
      early_reuse_rejected: early_reuse_rejected, early_poison_rejected: early_poison_rejected,
      shared_scratch_poisoned_before_request: poisoned, comparisons: comparisons,
      profiling: arm.profile, early_profile_rejected: early_profile_rejected,
      shared_gate_up: arm.shared_gate_up,
      shared_gate_up_block_count: session.blocks.plans.count(&.shared_gate_up),
      shared_gate_up_prepared_block_count: session.blocks.plans.count(&.shared_gate_up_prepared),
      early_mode_change_rejected: early_mode_change_rejected,
      unsupported_profile_rejected_without_submission: unsupported_profile_rejected_without_submission,
      per_block_encode_host_seconds: session.blocks.plans.map(&.last_encode_host_seconds),
      root_profile: root_profile, phase_profile: phase_profile, continuation_trace: trace_profile,
    }.to_json)
    STDERR.puts "#{kind}: passed=#{numerical_pass}, encode+wait=#{blocks_encode_seconds + head_encode_seconds + wait_seconds}s"
    chain.release
    chain = nil
    observer.emit("after_chain_release", request: request)
    break unless passed
  end
rescue error
  passed = false
  failure = "#{error.class}: #{error.message}"
  STDERR.puts failure
ensure
  begin
    if active = chain
      active.commit_and_wait unless active.completed?
      active.release
    end
    session.try(&.release)
  rescue error
    passed = false
    failure = "#{failure}; cleanup: #{error.class}: #{error.message}"
  end
  GC.collect
  observer.emit("after_session_release")
  result = {
    schema: "cogni_ml.pixelumm_native_integrated_probe/v2", generation: false,
    scope: "native_36_blocks_plus_global_head_in_one_gpu_chain",
    reference: reference, reference_sha256: reference_sha,
    alternate_reference: alternate, alternate_reference_sha256: alternate_sha,
    head_reference: head_reference, head_reference_sha256: head_reference_sha,
    head_alternate_reference: head_alternate, head_alternate_reference_sha256: head_alternate_sha,
    bf16_cache: cache_path, bf16_cache_sha256: cache_sha,
    grid: [6, 44, 80], block_indices: (0...36).to_a, output_shape: [1, 3, 24, 704, 1280],
    atol: PixelUMMHeadProbe::ATOL, rtol: PixelUMMHeadProbe::RTOL,
    named_budget_bytes: budget, named_bytes: named_bytes, block_named_bytes: block_bytes,
    head_owned_named_bytes: head_owned_bytes, head_weight_named_bytes: head_weight_bytes,
    duplicate_head_input_bytes: 0, shared_block_workspace: true, persistent_block_output_banks: 2,
    head_scratch_aliases_block_workspace: false, compared_capture_order: capture_order,
    unretained_early_block_output_aliases: PixelUMMTwoBlockProbe.thirty_six_unretained_output_aliases,
    oracle: "Reused authenticated upstream MPS controls joined at an exact block35-output/head-input edge; no fresh connected upstream neural run.",
    handoff: "block35 output is the same MetalBuffer consumed by the head on the same command chain, without CPU copy or intermediate wait",
    preparation_seconds: preparation_seconds, repeats: repeats, requests: requests,
    profile_variance: profile_variance, expected_request_count: schedule.size,
    compare_shared_gate_up: compare_shared_gate_up,
    shared_gate_up_schedule: schedule.map { |arm| {kind: arm.kind, shared_gate_up: arm.shared_gate_up} },
    timing_scope: "encode_seconds sums native API calls; total_seconds includes intervening memory telemetry and first-request lifecycle checks, excludes fixture preparation and post-wait comparisons; not a throughput benchmark",
    memory_telemetry: {categories_additive: false, guard_timestamp_clock: "mach_absolute_time",
                       guard_join_timestamp_field: "guard_monotonic_ns", sampled_peaks_are_bounds: false},
    memory_phases: observer.events, passed: passed && requests.size == schedule.size, failure: failure,
  }
  report_io.puts(result.to_pretty_json)
  report_io.close
  puts({report: report, passed: result[:passed], named_bytes: named_bytes, failure: failure}.to_json)
end
raise "PixelUMM integrated qualification failed: #{failure}" unless passed && requests.size == schedule.size
