require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_decoder_probe"
require "./support/pixelumm_tiled_gate_up_probe"
require "./support/pixelumm_cpu_diagnostic_probe"
require "../src/ml/video/pixelumm_decoder_profile"

reference, reference_sha, alternate, alternate_sha = "", "", "", ""
head_reference, head_reference_sha, head_alternate, head_alternate_sha = "", "", "", ""
cache_path, cache_sha, report = "", "", ""
budget = 0_i64
repeats = 1
profile_variance = false
compare_shared_gate_up = false
profile_shared_gate_up = false
compare_combined_attention = false
profile_combined_attention = false
reverse_combined_attention = false
profile_aa = false
reverse_aa_profile = false
correctness_only = false
tiled_rows = 0
reverse_tiled = false
compare_cpu_diagnostics = false
reverse_cpu_diagnostics = false
align_cpu_diagnostic_lifecycle = false
cpu_diagnostic_pause_seconds = nil.as(Int32?)
diagnostic_snapshot_dir = ""
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
  parser.on("--profile-shared-gate-up", "Six same-input shared/baseline phase controls; no generation") { profile_shared_gate_up = true }
  parser.on("--compare-combined-attention", "Both inputs/modes upstream gates, then unprofiled combined K/V + scale/softmax ABBA") { compare_combined_attention = true }
  parser.on("--profile-combined-attention", "Same-input phase crossover with off warm/tail controls; no generation") { profile_combined_attention = true }
  parser.on("--reverse-combined-attention", "Reverse both warm and timed combined-attention orders") { reverse_combined_attention = true }
  parser.on("--profile-aa", "Fixed baseline/input A, six requests, 50s post-wait gaps and crossed API observers") { profile_aa = true }
  parser.on("--reverse-aa-profile", "Reverse only the middle A/A observer order") { reverse_aa_profile = true }
  parser.on("--tiled-gate-up-rows N", "Observer-off row-tiled FFN crossover, six paced requests") { |v| tiled_rows = v.to_i32 }
  parser.on("--reverse-tiled-gate-up", "Reverse both warm and timed row-tiled FFN orders") { reverse_tiled = true }
  parser.on("--correctness-only", "Only the four poisoned combined-attention gates") { correctness_only = true }
  parser.on("--compare-cpu-diagnostics", "Fixed baseline/input, matched snapshots, eager/deferred CPU comparisons") { compare_cpu_diagnostics = true }
  parser.on("--reverse-cpu-diagnostics", "Reverse the two CPU-comparison placements") { reverse_cpu_diagnostics = true }
  parser.on("--align-cpu-diagnostic-lifecycle", "Release completed chains and collect GC before either CPU comparison policy") { align_cpu_diagnostic_lifecycle = true }
  parser.on("--cpu-diagnostic-pause-seconds N", "Common post-wait interval for CPU diagnostics, 50 or 60 seconds") { |v| cpu_diagnostic_pause_seconds = v.to_i32 }
  parser.on("--diagnostic-snapshot-dir PATH", "Existing directory for two fresh disk snapshots") { |v| diagnostic_snapshot_dir = v }
end
PixelUMMTiledGateUpProbe.validate_rows!(21120, tiled_rows) if tiled_rows != 0
tiled_modes = tiled_rows > 0
paced_requests = profile_aa || tiled_modes || compare_cpu_diagnostics
unless ARGV.empty? && {reference, reference_sha, alternate, alternate_sha,
                       head_reference, head_reference_sha, head_alternate, head_alternate_sha,
                       cache_path, cache_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 24_i64 * 1024**3 && 1 <= repeats <= 3 &&
       (!(profile_variance || compare_shared_gate_up || profile_shared_gate_up || compare_combined_attention || profile_combined_attention || profile_aa || tiled_modes || compare_cpu_diagnostics) || repeats == 1) &&
       {profile_variance, compare_shared_gate_up, profile_shared_gate_up, compare_combined_attention, profile_combined_attention, profile_aa, tiled_modes, compare_cpu_diagnostics}.count(true) <= 1 &&
       (!reverse_combined_attention || compare_combined_attention || profile_combined_attention) &&
       (!reverse_aa_profile || profile_aa) &&
       (!reverse_tiled || tiled_modes) &&
       (!reverse_cpu_diagnostics || compare_cpu_diagnostics) &&
       (!align_cpu_diagnostic_lifecycle || compare_cpu_diagnostics) &&
       (cpu_diagnostic_pause_seconds.nil? || (compare_cpu_diagnostics && {50, 60}.includes?(cpu_diagnostic_pause_seconds.not_nil!))) &&
       (compare_cpu_diagnostics == !diagnostic_snapshot_dir.empty?) &&
       (!correctness_only || compare_combined_attention)
  raise ArgumentError.new("expected authenticated paired full-J fixtures, cache, budget and fresh report")
end
aa_pause_ns = (cpu_diagnostic_pause_seconds.try(&.to_i64) || 50_i64) * 1_000_000_000_i64
raise ArgumentError.new("integrated report must be absolute") unless Path[report].absolute?
if compare_cpu_diagnostics
  unless Path[diagnostic_snapshot_dir].absolute? && Dir.exists?(diagnostic_snapshot_dir)
    raise ArgumentError.new("CPU diagnostic snapshots require an existing absolute directory")
  end
end
fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("integrated report must be fresh and writable") if fd < 0
report_io = IO::FileDescriptor.new(fd)
observer = PixelUMMMemoryProbe::Recorder.system
session = nil.as(PixelUMMDecoderProbe::Session?)
tiled_selection = nil.as(PixelUMMTiledGateUpProbe::Selection?)
diagnostic_store = nil.as(PixelUMMCPUDiagnosticProbe::Store?)
chain = nil.as(ML::Metal::MPSCommandChain?)
terminal_wait_attempted = false
cleanup_quarantined = false
spacing_abort_before_submission = false
previous_terminal_ns = nil.as(Int64?)
requests = [] of JSON::Any
capture_order = [] of String
named_bytes, block_bytes, head_owned_bytes, head_weight_bytes = 0_i64, 0_i64, 0_i64, 0_i64
preparation_seconds = 0.0
failure = nil.as(String?)
passed = true
diagnostics_completed = !compare_cpu_diagnostics
shared_modes = compare_shared_gate_up || profile_shared_gate_up
attention_modes = compare_combined_attention || profile_combined_attention || profile_aa
cpu_arms = compare_cpu_diagnostics ? PixelUMMCPUDiagnosticProbe.schedule(reverse_cpu_diagnostics) : [] of PixelUMMCPUDiagnosticProbe::Arm
schedule = if compare_cpu_diagnostics
             cpu_arms.map_with_index do |arm, index|
               PixelUMMDecoderProbe::RequestArm.new(arm.kind, false, index == 0, false)
             end
           elsif tiled_modes
             PixelUMMDecoderProbe.tiled_gate_up_schedule(reverse_tiled)
           elsif profile_aa
             PixelUMMDecoderProbe.aa_profile_schedule(reverse_aa_profile)
           elsif profile_combined_attention
             PixelUMMDecoderProbe.combined_attention_profile_schedule(reverse_combined_attention)
           elsif compare_combined_attention
             PixelUMMDecoderProbe.combined_attention_schedule(reverse_combined_attention, correctness_only)
           elsif profile_shared_gate_up
             PixelUMMDecoderProbe.shared_gate_up_profile_schedule
           elsif compare_shared_gate_up
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
    prepare_shared_gate_up: shared_modes,
    prepare_combined_attention: attention_modes,
    prepare_comparison_scratch: compare_cpu_diagnostics)
  tiled_selection = PixelUMMTiledGateUpProbe::Selection.new(session.blocks.plans, tiled_rows) if tiled_modes
  preparation_seconds = (Time.instant - start).total_seconds
  named_bytes, block_bytes = session.named_bytes, session.block_named_bytes
  head_owned_bytes, head_weight_bytes = session.head_owned_named_bytes, session.head_weight_named_bytes
  capture_order = session.capture_order
  if compare_cpu_diagnostics
    captures = capture_order.map do |stage|
      PixelUMMCPUDiagnosticProbe::Capture.new(stage, session.diagnostic_capture_bytes(stage))
    end
    paths = (0...2).map { |slot| File.join(diagnostic_snapshot_dir, "condition-#{slot}.bf16") }
    diagnostic_store = PixelUMMCPUDiagnosticProbe::Store.new(paths, captures, session.diagnostic_comparison_scratch)
  end
  unless capture_order.size == 43 && session.head_input_aliases_block35_output? &&
         session.head_plan.input.same?(session.blocks.workspace.output_buffer(35)) &&
         named_bytes == block_bytes + head_owned_bytes + head_weight_bytes
    raise "integrated unique named inventory, capture coverage or direct GPU edge mismatch"
  end
  observer.emit("session_ready")
  STDERR.puts "Integrated PixelUMM admitted #{named_bytes} unique named bytes, full J=21120"
  current_alternate = false
  head_poison_count = 0
  segments_by_mode = {} of Tuple(Bool, Bool, Bool) => Int32
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
    tiled_selection.try(&.select!(arm.tiled_gate_up))
    session.use_shared_gate_up!(arm.shared_gate_up) if shared_modes
    session.use_combined_attention!(arm.combined_attention) if attention_modes
    unless session.blocks.plans.all? { |plan| plan.shared_gate_up == arm.shared_gate_up }
      raise "integrated gate/up selection must cover every block"
    end
    if attention_modes && !session.blocks.plans.all? { |plan| plan.fused_kv_gather == arm.combined_attention && plan.scale_softmax == arm.combined_attention && plan.scale_softmax_prepared }
      raise "integrated combined attention selection must cover every block"
    end
    aa_ready_ns = 0_i64
    aa_sleep_ns = 0_i64
    aa_target_ns = nil.as(Int64?)
    aa_spacing = nil.as(JSON::Any?)
    if paced_requests
      aa_ready_ns = PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!
      aa_sleep_ns = PixelUMMDecoderProbe.aa_pause_remaining_ns(previous_terminal_ns, aa_ready_ns, aa_pause_ns)
      aa_target_ns = previous_terminal_ns.try { |value| value + aa_pause_ns }
      sleep aa_sleep_ns.nanoseconds if aa_sleep_ns > 0
    end
    chain = ML::Metal::MPSCommandChain.new
    terminal_wait_attempted = false
    if arm.profile
      chain.enable_api_profile!(phase_roots: false)
      chain.enable_continuation_trace!
    end
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    head_before_blocks_rejected = false
    early_release_rejected, early_read_rejected, early_reuse_rejected, early_poison_rejected = false, false, false, false
    early_mode_change_rejected = false
    early_attention_change_rejected = false
    unsupported_profile_rejected_without_submission = false
    if request == 0
      if shared_modes || profile_combined_attention || paced_requests
        prior_segments = chain.segment_count
        begin
          session.encode_blocks(chain, profile: true)
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("API chain")
          unsupported_profile_rejected_without_submission = true
        end
        # This fails if rejection incorrectly latched the empty chain. No
        # terminal wait is allowed between rejection and this idle operation.
        session.use_shared_gate_up!(arm.shared_gate_up) if shared_modes
        session.use_combined_attention!(arm.combined_attention) if attention_modes
        unless unsupported_profile_rejected_without_submission && chain.segment_count == prior_segments &&
               chain.wait_count == 0 && ML::MetalBuffer.hy_text_probe_readbacks == reads &&
               ML::MetalBuffer.hy_text_probe_allocations == allocations
          raise "missing-API profiling rejection changed submission state"
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
    if paced_requests
      before_ns = PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!
      lateness_ns = aa_target_ns.try { |value| before_ns - value }
      if lateness_ns && !(0 <= lateness_ns <= PixelUMMDecoderProbe::AA_ALLOWED_LATENESS_NS)
        # This chain has no submitted work. Quarantine it until process exit
        # instead of submitting an empty command buffer during error cleanup.
        spacing_abort_before_submission = true
        raise "A/A next-encode deadline missed before GPU submission"
      end
      aa_spacing = JSON.parse({
        clock: "mach_absolute_time_nanoseconds", previous_terminal_end_ns: previous_terminal_ns,
        diagnostic_ready_ns: aa_ready_ns, target_encode_start_ns: aa_target_ns,
        before_encode_ns: before_ns, requested_gap_ns: aa_pause_ns,
        requested_sleep_ns: aa_sleep_ns,
        actual_gap_ns: previous_terminal_ns.try { |value| before_ns - value },
        lateness_ns: lateness_ns, allowed_lateness_ns: PixelUMMDecoderProbe::AA_ALLOWED_LATENESS_NS,
      }.to_json)
      STDERR.puts "AA_ENCODE request=#{request} pid=#{Process.pid} mach_ns=#{before_ns} profiling=#{arm.profile}"
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
    early_tiled_change_rejected = false
    early_snapshot_rejected = false
    if request == 0
      if compare_cpu_diagnostics
        begin
          session.read_diagnostic_actual!(capture_order.first, Bytes.empty)
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("terminal wait")
          early_snapshot_rejected = true
        end
        raise "integrated snapshot read admitted before terminal wait" unless early_snapshot_rejected
      end
      if selection = tiled_selection
        before_mode = selection.selected
        begin
          selection.select!(!arm.tiled_gate_up)
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("in flight")
          early_tiled_change_rejected = true
        end
        unless early_tiled_change_rejected && selection.selected == before_mode
          raise "integrated in-flight tiled gate/up selection mutated plans"
        end
      end
      if attention_modes
        before_modes = session.blocks.plans.map { |plan| {plan.fused_kv_gather, plan.scale_softmax} }
        begin
          session.use_combined_attention!(!arm.combined_attention)
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("in flight")
          early_attention_change_rejected = true
        end
        unless early_attention_change_rejected && session.blocks.plans.map { |plan| {plan.fused_kv_gather, plan.scale_softmax} } == before_modes
          raise "integrated in-flight attention selection mutated one or more blocks"
        end
      end
      if shared_modes
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
    terminal_wait_attempted = true
    chain.commit_and_wait
    wait_seconds = (Time.instant - start).total_seconds
    execution_seconds = (Time.instant - execution_start).total_seconds
    terminal_ns = paced_requests ? PixelUMMMemoryProbe.guard_monotonic_ns.not_nil! : nil
    previous_terminal_ns = terminal_ns if paced_requests
    STDERR.puts "AA_WAIT request=#{request} pid=#{Process.pid} mach_ns=#{terminal_ns}" if paced_requests
    observer.emit("request_after_terminal_wait", request: request)
    # All metadata extraction is after the existing wait and outside execution
    # timing. It does not read activations, allocate GPU buffers or enqueue work.
    root_profile = nil.as(JSON::Any?)
    phase_profile = nil.as(JSON::Any?)
    trace_profile = nil.as(JSON::Any?)
    if profile_variance || shared_modes || profile_combined_attention || paced_requests
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
          chain.gpu_elapsed_seconds, chain.api_profile, shared_gate_up: arm.shared_gate_up,
          fused_kv_gather: arm.combined_attention, scale_softmax: arm.combined_attention)
        unless session.blocks.plans.all? { |plan| plan.profiled_shared_gate_up == arm.shared_gate_up &&
               plan.profiled_fused_kv_gather == arm.combined_attention &&
               plan.profiled_scale_softmax == arm.combined_attention }
          raise "integrated profile topology differs from encoded request"
        end
        phase_profile = JSON.parse(phases.to_json)
        trace = chain.continuation_trace
        trace_profile = JSON.parse({info: trace.info, events: trace.events, roots: trace.roots}.to_json)
      end
      if profile_variance
        raise "integrated profile changed segment topology" unless chain.segment_count == 4579
      else
        mode_key = {arm.shared_gate_up, arm.combined_attention, arm.tiled_gate_up}
        segments_by_mode[mode_key] ||= chain.segment_count
        unless chain.segment_count == segments_by_mode[mode_key]
          raise "integrated mode changed segment topology between requests"
        end
      end
      unless chain.wait_count == 1 && ML::MetalBuffer.hy_text_probe_readbacks == reads &&
             ML::MetalBuffer.hy_text_probe_allocations == allocations
        raise "integrated profiling added waits, activation reads or named allocations"
      end
    end
    observer.emit("request_after_profile_extraction", request: request) if profile_combined_attention || paced_requests
    # Copy all native-backed metadata before the optional early native release.
    # Session-owned capture buffers remain valid after the chain is released.
    completed_chain_gpu_seconds = chain.gpu_elapsed_seconds
    completed_chain_segments = chain.segment_count
    completed_chain_waits = chain.wait_count
    completed_chain_completed = chain.completed?
    lifecycle_events = [] of NamedTuple(event: String, mach_ns: Int64)
    record_lifecycle = ->(event : String) {
      if align_cpu_diagnostic_lifecycle
        lifecycle_events << {event: event, mach_ns: PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!}
      end
    }
    if align_cpu_diagnostic_lifecycle
      unless completed_chain_completed && completed_chain_waits == 1 &&
             root_profile.not_nil!["root_count"].as_i == completed_chain_segments
        raise "aligned CPU diagnostics require completed chain metadata before release"
      end
      lifecycle_events << {event: "terminal_wait_complete", mach_ns: terminal_ns.not_nil!}
      record_lifecycle.call("completed_chain_metadata_captured")
    end
    diagnostic_start = Time.instant
    cpu_arm = compare_cpu_diagnostics ? cpu_arms[request] : nil
    deferred = cpu_arm.try(&.policy) == "deferred"
    comparisons = if align_cpu_diagnostic_lifecycle
                    store = diagnostic_store.not_nil!
                    if slot = cpu_arm.not_nil!.snapshot_slot
                      record_lifecycle.call("snapshot_write_begin")
                      store.save_snapshot(slot, request, !deferred,
                        ->(stage : String, bytes : Bytes) { session.not_nil!.read_diagnostic_actual!(stage, bytes) })
                      record_lifecycle.call("snapshot_write_complete")
                    end
                    record_lifecycle.call("chain_release_begin")
                    chain.release
                    chain = nil
                    record_lifecycle.call("chain_release_complete")
                    observer.emit("after_chain_release", request: request)
                    record_lifecycle.call("gc_begin")
                    observer.emit("before_aligned_diagnostic_gc", request: request)
                    GC.collect
                    record_lifecycle.call("gc_complete")
                    observer.emit("after_aligned_diagnostic_gc", request: request)
                    if deferred
                      record_lifecycle.call("comparison_deferred")
                      [] of PixelUMMDecoderProbe::Comparison
                    else
                      record_lifecycle.call("comparison_begin")
                      result = if slot = cpu_arm.not_nil!.snapshot_slot
                                 store.compare_eager(slot,
                                   ->(stage : String, bytes : Bytes) { session.not_nil!.read_diagnostic_expected!(stage, bytes) })
                               else
                                 session.compare(request: request)
                               end
                      record_lifecycle.call("comparison_complete")
                      result
                    end
                  elsif store = diagnostic_store
                    if slot = cpu_arm.not_nil!.snapshot_slot
                      store.snapshot(slot, request, !deferred,
                        ->(stage : String, bytes : Bytes) { session.not_nil!.read_diagnostic_expected!(stage, bytes) },
                        ->(stage : String, bytes : Bytes) { session.not_nil!.read_diagnostic_actual!(stage, bytes) })
                    else
                      session.compare(request: request)
                    end
                  else
                    session.compare(request: request)
                  end
    record_lifecycle.call("request_diagnostics_complete")
    diagnostic_seconds = (Time.instant - diagnostic_start).total_seconds
    observer.emit("request_after_diagnostic_comparisons", request: request)
    unless (deferred ? comparisons.empty? : comparisons.map { |c| c[:stage] } == capture_order) &&
           ML::MetalBuffer.hy_text_probe_readbacks - reads == 43 && completed_chain_waits == 1
      raise "integrated comparisons omitted a retained capture or changed terminal waits"
    end
    numerical_pass = deferred ? nil : comparisons.all? { |c| c[:finite] && c[:failures] == 0 }
    passed &&= numerical_pass != false
    if numerical_pass == false
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
      gpu_elapsed_seconds: completed_chain_gpu_seconds, command_segments: completed_chain_segments,
      terminal_wait_count: completed_chain_waits, named_allocations_during_encode: encode_allocations,
      activation_readbacks_before_terminal_wait: early_reads,
      diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
      diagnostic_cpu_scratch_bytes: session.diagnostic_cpu_scratch_bytes,
      cpu_diagnostic_role: cpu_arm.try(&.role), cpu_diagnostic_policy: cpu_arm.try(&.policy),
      preceding_cpu_diagnostic_policy: cpu_arm.try(&.preceding_policy),
      cpu_diagnostic_snapshot_slot: cpu_arm.try(&.snapshot_slot),
      cpu_diagnostic_snapshot_bytes: cpu_arm.try(&.snapshot_slot).nil? ? 0_i64 : diagnostic_store.not_nil!.per_slot_bytes,
      cpu_diagnostic_seconds_before_next_request: diagnostic_seconds,
      cpu_diagnostics_pending: deferred, early_snapshot_rejected: early_snapshot_rejected,
      cpu_diagnostic_lifecycle: align_cpu_diagnostic_lifecycle ? {
        events:                   lifecycle_events,
        completed_chain_metadata: {completed: completed_chain_completed, wait_count: completed_chain_waits,
                                   command_segments: completed_chain_segments,
                                   root_count: root_profile.not_nil!["root_count"].as_i},
      } : nil,
      head_before_blocks_rejected: head_before_blocks_rejected,
      early_release_rejected: early_release_rejected, early_read_rejected: early_read_rejected,
      early_reuse_rejected: early_reuse_rejected, early_poison_rejected: early_poison_rejected,
      shared_scratch_poisoned_before_request: poisoned, comparisons: comparisons,
      profiling: arm.profile, early_profile_rejected: early_profile_rejected,
      tiled_gate_up: arm.tiled_gate_up, tiled_gate_up_rows: tiled_rows,
      tiled_gate_up_block_count: tiled_selection.try { |selection| selection.selected ? selection.plans.size : 0 } || 0,
      tiled_gate_up_prepared_block_count: tiled_selection.try(&.plans.size) || 0,
      early_tiled_change_rejected: early_tiled_change_rejected,
      shared_gate_up: arm.shared_gate_up,
      shared_gate_up_block_count: session.blocks.plans.count(&.shared_gate_up),
      shared_gate_up_prepared_block_count: session.blocks.plans.count(&.shared_gate_up_prepared),
      combined_attention: arm.combined_attention,
      fused_kv_gather_block_count: session.blocks.plans.count(&.fused_kv_gather),
      scale_softmax_block_count: session.blocks.plans.count(&.scale_softmax),
      scale_softmax_prepared_block_count: session.blocks.plans.count(&.scale_softmax_prepared),
      early_attention_change_rejected: early_attention_change_rejected,
      early_mode_change_rejected: early_mode_change_rejected,
      unsupported_profile_rejected_without_submission: unsupported_profile_rejected_without_submission,
      missing_api_profile_rejected_without_submission: unsupported_profile_rejected_without_submission,
      per_block_encode_host_seconds: session.blocks.plans.map(&.last_encode_host_seconds),
      aa_spacing: aa_spacing, terminal_wait_end_mach_ns: terminal_ns,
      root_profile: root_profile, phase_profile: phase_profile, continuation_trace: trace_profile,
    }.to_json)
    STDERR.puts "#{kind}: passed=#{deferred ? "pending" : numerical_pass.to_s}, encode+wait=#{blocks_encode_seconds + head_encode_seconds + wait_seconds}s"
    unless align_cpu_diagnostic_lifecycle
      chain.not_nil!.release
      chain = nil
      observer.emit("after_chain_release", request: request)
    end
    if !align_cpu_diagnostic_lifecycle && (attention_modes || tiled_modes || compare_cpu_diagnostics)
      # Post-wait GC is outside timing. Prepared attention diagnostics reuse
      # comparison arrays; row-tiling diagnostics retain per-capture host buffers.
      observer.emit("before_post_diagnostic_gc", request: request)
      GC.collect
      observer.emit("after_post_diagnostic_gc", request: request)
    end
    break unless passed
  end
  if passed && requests.size == schedule.size && (store = diagnostic_store)
    store.finish_requests!(requests.size)
    cpu_arms.each_with_index do |arm, request|
      next unless arm.policy == "deferred"
      if align_cpu_diagnostic_lifecycle
        events = requests[request]["cpu_diagnostic_lifecycle"]["events"].as_a
        events << JSON.parse({event: "all_requests_finished", mach_ns: PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!}.to_json)
        events << JSON.parse({event: "deferred_comparison_begin", mach_ns: PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!}.to_json)
      end
      observer.emit("before_deferred_diagnostic_comparisons", request: request)
      start = Time.instant
      comparisons = store.compare_deferred(arm.snapshot_slot.not_nil!,
        ->(stage : String, bytes : Bytes) { session.not_nil!.read_diagnostic_expected!(stage, bytes) })
      if align_cpu_diagnostic_lifecycle
        requests[request]["cpu_diagnostic_lifecycle"]["events"].as_a << JSON.parse({
          event: "deferred_comparison_complete", mach_ns: PixelUMMMemoryProbe.guard_monotonic_ns.not_nil!,
        }.to_json)
      end
      unless comparisons.map { |c| c[:stage] } == capture_order
        raise "deferred integrated comparison omitted a retained capture"
      end
      numerical_pass = comparisons.all? { |c| c[:finite] && c[:failures] == 0 }
      requests[request].as_h["comparisons"] = JSON.parse(comparisons.to_json)
      requests[request].as_h["passed"] = JSON.parse(numerical_pass.to_json)
      requests[request].as_h["cpu_diagnostics_pending"] = JSON.parse("false")
      requests[request].as_h["deferred_comparison_seconds"] = JSON.parse((Time.instant - start).total_seconds.to_json)
      passed &&= numerical_pass
      failure = "deferred numerical mismatch at request #{request}" unless numerical_pass
      observer.emit("after_deferred_diagnostic_comparisons", request: request)
    end
    diagnostics_completed = store.completed?
    raise "CPU diagnostic snapshots were not all compared" unless diagnostics_completed
  end
rescue error
  passed = false
  failure = "#{error.class}: #{error.message}"
  STDERR.puts failure
ensure
  diagnostic_store.try(&.close)
  begin
    raise "unsubmitted A/A chain quarantined after spacing failure" if spacing_abort_before_submission
    if active = chain
      if terminal_wait_attempted && !active.completed?
        cleanup_quarantined = true
        raise "failed terminal wait: retaining chain and session until process exit"
      end
      unless active.completed?
        terminal_wait_attempted = true
        active.commit_and_wait
      end
      active.release
    end
    tiled_selection.try(&.release)
    session.try(&.release)
  rescue error
    cleanup_quarantined = true if spacing_abort_before_submission || (chain && terminal_wait_attempted && !chain.not_nil!.completed?)
    passed = false
    failure = "#{failure}; cleanup: #{error.class}: #{error.message}"
  end
  GC.collect
  observer.emit("after_session_release")
  result = {
    schema: compare_cpu_diagnostics ? (align_cpu_diagnostic_lifecycle ? "cogni_ml.pixelumm_cpu_diagnostic_placement/v2" : "cogni_ml.pixelumm_cpu_diagnostic_placement/v1") : tiled_modes ? "cogni_ml.pixelumm_native_row_tiled_ffn/v1" : profile_aa ? "cogni_ml.pixelumm_native_aa_profile/v1" : (profile_combined_attention ? "cogni_ml.pixelumm_native_combined_attention_profile/v1" : (compare_combined_attention ? "cogni_ml.pixelumm_native_combined_attention_probe/v1" : "cogni_ml.pixelumm_native_integrated_probe/v2")), generation: false,
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
    compare_shared_gate_up: compare_shared_gate_up, profile_shared_gate_up: profile_shared_gate_up,
    compare_combined_attention: compare_combined_attention,
    profile_combined_attention: profile_combined_attention,
    reverse_combined_attention: reverse_combined_attention, correctness_only: correctness_only,
    profile_aa: profile_aa, reverse_aa_profile: reverse_aa_profile,
    compare_cpu_diagnostics: compare_cpu_diagnostics, reverse_cpu_diagnostics: reverse_cpu_diagnostics,
    align_cpu_diagnostic_lifecycle: align_cpu_diagnostic_lifecycle,
    cpu_diagnostic_estimand: align_cpu_diagnostic_lifecycle ? "residual eager-versus-deferred CPU-placement effect conditional on common completed-chain release and GC; not an isolated early-release or GC effect" : nil,
    cpu_diagnostics_completed: diagnostics_completed,
    diagnostic_snapshot_dir: diagnostic_snapshot_dir,
    diagnostic_snapshot_total_bytes: diagnostic_store.try(&.total_bytes) || 0_i64,
    cpu_diagnostic_schedule: cpu_arms.map { |arm| {kind: arm.kind, role: arm.role, policy: arm.policy, snapshot_slot: arm.snapshot_slot, preceding_policy: arm.preceding_policy} },
    tiled_gate_up_rows: tiled_rows, reverse_tiled_gate_up: reverse_tiled,
    tiled_gate_up_schedule: tiled_modes ? schedule.map { |arm| {kind: arm.kind, tiled_gate_up: arm.tiled_gate_up, alternate: arm.alternate, poisoned: arm.poisoned, profiling: arm.profile} } : nil,
    row_ranges: tiled_modes ? PixelUMMTiledGateUpProbe.row_ranges(21120, tiled_rows) : nil,
    aa_pause_seconds: aa_pause_ns / 1e9,
    aa_allowed_lateness_seconds: PixelUMMDecoderProbe::AA_ALLOWED_LATENESS_NS / 1e9,
    spacing_abort_before_submission: spacing_abort_before_submission,
    aa_schedule: profile_aa ? schedule.map { |arm| {kind: arm.kind, combined_attention: arm.combined_attention, alternate: arm.alternate, poisoned: arm.poisoned, profiling: arm.profile} } : nil,
    cleanup_quarantined: cleanup_quarantined,
    combined_attention_schedule: schedule.map { |arm| {kind: arm.kind, combined_attention: arm.combined_attention, alternate: arm.alternate, poisoned: arm.poisoned, profiling: arm.profile} },
    shared_gate_up_schedule: schedule.map { |arm| {kind: arm.kind, shared_gate_up: arm.shared_gate_up, profiling: arm.profile} },
    timing_scope: "encode_seconds sums native API calls; total_seconds includes intervening memory telemetry and first-request lifecycle checks, excludes fixture preparation and post-wait comparisons; not a throughput benchmark",
    memory_telemetry: {categories_additive: false, guard_timestamp_clock: "mach_absolute_time",
                       guard_join_timestamp_field: "guard_monotonic_ns", sampled_peaks_are_bounds: false},
    memory_phases: observer.events, passed: passed && diagnostics_completed && requests.size == schedule.size, failure: failure,
  }
  report_io.puts(result.to_pretty_json)
  report_io.close
  puts({report: report, passed: result[:passed], named_bytes: named_bytes, failure: failure}.to_json)
end
raise "PixelUMM integrated qualification failed: #{failure}" unless passed && diagnostics_completed && requests.size == schedule.size
