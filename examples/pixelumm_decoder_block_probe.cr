require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_block_probe"

reference, expected_sha, report = "", "", ""
budget = 0_i64
warmups, repeats = 1, 3
profile_order = "none"
continuation_trace = false
ffn_f32 = false
compare_shared_gate_up = false
profile_shared_gate_up = false
trace_ready, trace_release = "", ""
OptionParser.parse do |parser|
  parser.banner = "One PixelUMM block only; external 10% memory guard required."
  parser.on("--reference PATH", "Pinned upstream-adapted GPU fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA-256") { |v| expected_sha = v }
  parser.on("--report PATH", "New comparison/timing report") { |v| report = v }
  parser.on("--max-named-bytes N", "Named-buffer admission budget, up to 8 GiB") { |v| budget = v.to_i64 }
  parser.on("--warmups N", "Untimed warm-up requests, 0..2") { |v| warmups = v.to_i }
  parser.on("--repeats N", "Timed requests after the numerical gate, 1..6") { |v| repeats = v.to_i }
  parser.on("--profile-order ORDER", "none, off-on or on-off; balanced pairs in one resident session") { |v| profile_order = v }
  parser.on("--continuation-trace", "Opt-in CPU continuation chronology on profiled arms") { continuation_trace = true }
  parser.on("--ffn-f32", "Opt-in F32 graph tensors for gate/up/down, retaining BF16 boundaries") { ffn_f32 = true }
  parser.on("--compare-shared-gate-up", "Compare separate and shared-input F32 gate/up graphs") { compare_shared_gate_up = true }
  parser.on("--profile-shared-gate-up", "Qualify both F32 gate/up phase topologies on all captures") { profile_shared_gate_up = true }
  parser.on("--trace-ready-file PATH", "Fresh PID receipt for bounded Instruments attach before encoding") { |v| trace_ready = v }
  parser.on("--trace-release-file PATH", "Fresh CPU release file paired with --trace-ready-file") { |v| trace_release = v }
end
unless ARGV.empty? && {reference, expected_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 8_i64 * 1024 * 1024 * 1024 && 0 <= warmups <= 2 && 1 <= repeats <= 6 &&
       {"none", "off-on", "on-off"}.includes?(profile_order)
  raise ArgumentError.new("expected authenticated reference, new report, bounded memory and request counts")
end
if profile_shared_gate_up
  raise ArgumentError.new("shared profiling requires --ffn-f32 and its own bounded schedule") unless ffn_f32 && !compare_shared_gate_up && profile_order == "none" && !continuation_trace
end
if compare_shared_gate_up && !(ffn_f32 && profile_order == "none" && !continuation_trace)
  raise ArgumentError.new("--compare-shared-gate-up requires --ffn-f32, --profile-order none, and continuation tracing off")
end
if continuation_trace && profile_order == "none"
  raise ArgumentError.new("--continuation-trace requires a profiled profile-order")
end
unless trace_ready.empty? == trace_release.empty? &&
       (trace_ready.empty? || (trace_ready != trace_release &&
       !File.info?(trace_ready, false) && !File.info?(trace_release, false)))
  raise ArgumentError.new("trace attach requires two distinct fresh CPU gate files")
end
raise ArgumentError.new("refusing to replace probe report") if File.info?(report, false)
report_fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("probe report must be fresh and writable") if report_fd < 0
report_io = IO::FileDescriptor.new(report_fd)
shared_modes = compare_shared_gate_up || profile_shared_gate_up
continuation_trace = true if profile_shared_gate_up
session = nil.as(PixelUMMBlockProbe::Session?)
chain = nil.as(ML::Metal::MPSCommandChain?)
requests = [] of JSON::Any
passed = true
schedule = [] of NamedTuple(kind: String, profile: Bool, pair: Int32, shared_gate_up: Bool)
if profile_shared_gate_up
  [{true, true}, {false, true}, {false, false}, {true, false}, {true, true}, {false, true}].each_with_index do |entry, index|
    schedule << {kind: index < 2 ? "numerical_gate" : "profile_control", profile: entry[1], pair: (index // 2).to_i32, shared_gate_up: entry[0]}
  end
elsif compare_shared_gate_up
  # Run the candidate gate before the baseline can write correct diagnostic
  # captures. Each repeat contributes AB and BA ordering to reduce order bias.
  schedule << {kind: "numerical_gate", profile: false, pair: -1, shared_gate_up: true}
  schedule << {kind: "numerical_gate", profile: false, pair: -1, shared_gate_up: false}
  warmups.times do |warmup|
    order = warmup.even? ? [false, true] : [true, false]
    order.each do |shared|
      schedule << {kind: "warmup", profile: false, pair: -1, shared_gate_up: shared}
    end
  end
  repeats.times do |repeat|
    [{false, true}, {true, false}].each_with_index do |order, order_index|
      pair = (repeat * 2 + order_index).to_i32
      order.each do |shared|
        schedule << {kind: "timed", profile: false, pair: pair, shared_gate_up: shared}
      end
    end
  end
elsif profile_order == "none"
  (1 + warmups + repeats).times do |index|
    schedule << {kind: index == 0 ? "numerical_gate" : (index <= warmups ? "warmup" : "timed"), profile: false, pair: -1, shared_gate_up: false}
  end
else
  initial_order = profile_order == "off-on" ? [false, true] : [true, false]
  initial_order.each { |profile| schedule << {kind: "numerical_gate", profile: profile, pair: -1, shared_gate_up: false} }
  warmups.times do
    initial_order.each { |profile| schedule << {kind: "warmup", profile: profile, pair: -1, shared_gate_up: false} }
  end
  repeats.times do |pair|
    order = pair.even? ? initial_order : initial_order.reverse
    order.each { |profile| schedule << {kind: "timed", profile: profile, pair: pair, shared_gate_up: false} }
  end
end
expected_segments = nil.as(Int32?)
expected_segments_by_mode = {} of String => Int32
capture_hashes = nil.as(Hash(String, String)?)
capture_hashes_by_mode = {} of String => Hash(String, String)
cross_mode_hashes_equal = nil.as(Bool?)
begin
  fixture = PixelUMMBlockProbe::Fixture.new(reference, expected_sha)
  preparation = Time.instant
  if shared_modes && fixture.capture_order.size != 26
    raise ArgumentError.new("shared gate/up comparison requires all 26 fixture captures")
  end
  session = PixelUMMBlockProbe::Session.new(fixture, budget, ffn_f32: ffn_f32,
    prepare_shared_gate_up: shared_modes)
  if shared_modes && (!session.plan.shared_gate_up_prepared || session.plan.shared_gate_up)
    raise ArgumentError.new("shared gate/up plan capability/default selection mismatch")
  end
  preparation_seconds = (Time.instant - preparation).total_seconds
  STDERR.puts "PixelUMM one block admitted #{session.named_bytes} named bytes, grid=#{fixture.grid}"
  unless trace_ready.empty?
    # This bounded startup gate precedes all request encoding and timing. It
    # lets Instruments attach to a live process without adding any GPU wait.
    ready_fd = LibC.open(trace_ready, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("trace ready receipt must be fresh and writable") if ready_fd < 0
    IO::FileDescriptor.new(ready_fd).tap do |io|
      io.puts(Process.pid)
      io.close
    end
    deadline = Time.instant + 45.seconds
    until File.file?(trace_release)
      raise "Instruments CPU attach gate timed out" if Time.instant >= deadline
      sleep 10.milliseconds
    end
  end
  # Numerical gate arms precede all measured requests.
  schedule.each_with_index do |arm, request|
    if shared_modes
      session.plan.use_shared_gate_up!(arm[:shared_gate_up])
      raise "PixelUMM shared gate/up mode selection was not applied" unless session.plan.shared_gate_up == arm[:shared_gate_up]
    end
    chain = ML::Metal::MPSCommandChain.new
    if arm[:profile]
      chain.enable_api_profile!(phase_roots: false)
      chain.enable_continuation_trace! if continuation_trace
    end
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    start = Time.instant
    session.encode(chain, profile: arm[:profile])
    encode_seconds = (Time.instant - start).total_seconds
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
      raise "PixelUMM encode allocated named Metal buffers, read back activations or waited"
    end
    early_release_rejected = false
    inflight_toggle_rejected = false
    if request == 0
      begin
        session.release
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("command chain")
        early_release_rejected = true
      end
      raise "PixelUMM accepted in-flight release" unless early_release_rejected
      if shared_modes
        begin
          session.plan.use_shared_gate_up!(!arm[:shared_gate_up])
        rescue error : ArgumentError
          message = error.message || ""
          raise error unless message.includes?("idle") || message.includes?("in flight") || message.includes?("in-flight")
          inflight_toggle_rejected = true
        end
        unless inflight_toggle_rejected && session.plan.shared_gate_up == arm[:shared_gate_up]
          raise "PixelUMM accepted or applied an in-flight shared gate/up toggle"
        end
      end
    end
    prewait_profile_rejected = false
    if arm[:profile]
      begin
        session.plan.phase_profile(chain)
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("terminal wait")
        prewait_profile_rejected = true
      end
      raise "PixelUMM accepted a preterminal profile snapshot" unless prewait_profile_rejected
    end
    start = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - start).total_seconds
    comparisons = (shared_modes || profile_order != "none" || request == 0 || request == schedule.size - 1) ? session.compare : [] of PixelUMMBlockProbe::Comparison
    numerical_pass = comparisons.all? { |comparison| comparison[:finite] && comparison[:failures] == 0 }
    capture_hashes_stable = true
    mode_name = arm[:shared_gate_up] ? "shared_gate_up" : "baseline"
    unless comparisons.empty?
      if shared_modes && comparisons.size != 26
        raise "PixelUMM shared gate/up arm did not compare all 26 captures"
      end
      hashes = comparisons.to_h { |comparison| {comparison[:stage], comparison[:actual_sha256]} }
      if shared_modes
        if previous_hashes = capture_hashes_by_mode[mode_name]?
          capture_hashes_stable = previous_hashes == hashes
        else
          capture_hashes_by_mode[mode_name] = hashes
        end
      else
        capture_hashes ||= hashes
        capture_hashes_stable = capture_hashes == hashes
      end
    end
    if shared_modes
      if expected = expected_segments_by_mode[mode_name]?
        raise "PixelUMM #{mode_name} command-root count changed across arms" unless chain.segment_count == expected
      else
        expected_segments_by_mode[mode_name] = chain.segment_count
      end
    else
      expected_segments ||= chain.segment_count
      raise "PixelUMM profiling changed command-root count" unless chain.segment_count == expected_segments
    end
    if shared_modes && capture_hashes_by_mode.has_key?("baseline") && capture_hashes_by_mode.has_key?("shared_gate_up")
      cross_mode_hashes_equal = capture_hashes_by_mode["baseline"] == capture_hashes_by_mode["shared_gate_up"]
    end
    snapshot_mode_change_rejected = false
    phase_profile = nil.as(JSON::Any?)
    if arm[:profile]
      # A completed profile belongs to the encoded mode, not the next selection.
      session.plan.use_shared_gate_up!(!arm[:shared_gate_up]) if profile_shared_gate_up
      profile = session.plan.phase_profile(chain)
      if profile_shared_gate_up
        snapshot_mode_change_rejected = profile.shared_gate_up == arm[:shared_gate_up] &&
                                        session.plan.shared_gate_up != arm[:shared_gate_up]
        raise "completed profile was relabelled by idle graph selection" unless snapshot_mode_change_rejected
        session.plan.use_shared_gate_up!(arm[:shared_gate_up])
      end
      phase_profile = JSON.parse(profile.to_json)
    end
    trace_profile = if arm[:profile] && continuation_trace
                      trace = chain.continuation_trace
                      JSON.parse({info: trace.info, events: trace.events, roots: trace.roots}.to_json)
                    end
    arm_passed = numerical_pass && capture_hashes_stable && chain.wait_count == 1 && cross_mode_hashes_equal != false
    passed &&= arm_passed
    kind = arm[:kind]
    requests << JSON.parse({request: request, kind: kind, passed: arm_passed,
                            profile_enabled: arm[:profile], pair: arm[:pair],
                            shared_gate_up_enabled: arm[:shared_gate_up],
                            shared_gate_up_inflight_toggle_rejected: inflight_toggle_rejected,
                            capture_hashes_stable: capture_hashes_stable,
                            prewait_profile_rejected: prewait_profile_rejected,
                            profile_mode_snapshot_preserved: snapshot_mode_change_rejected,
                            encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                            total_seconds: encode_seconds + wait_seconds, gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                            command_segments: chain.segment_count, terminal_wait_count: chain.wait_count,
                            named_allocations_during_encode: encode_allocations, activation_readbacks_before_terminal_wait: early_reads,
                            diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                            early_release_rejected: early_release_rejected, comparisons: comparisons,
                            phase_profile: phase_profile, continuation_trace: trace_profile}.to_json)
    STDERR.puts "#{kind} #{request} profile=#{arm[:profile]} shared_gate_up=#{arm[:shared_gate_up]}: passed=#{arm_passed}, total=#{encode_seconds + wait_seconds}s"
    chain.release
    chain = nil
    break unless passed
  end
  comparison_complete = !shared_modes ||
                        (capture_hashes_by_mode.has_key?("baseline") && capture_hashes_by_mode.has_key?("shared_gate_up") && cross_mode_hashes_equal == true)
  result = {schema: shared_modes ? "cogni_ml.pixelumm_native_block_probe/v2" : "cogni_ml.pixelumm_native_block_probe/v1", reference: reference,
            reference_sha256: expected_sha, grid: fixture.grid, block_index: 0,
            generation: false, precision: "BF16 weights and handoffs; F32 normalization/RoPE/attention intermediates; " +
                                          (session.plan.ffn_f32 ? "gate/up/down operands cast to F32 graph tensors, with BF16 output casts" : "BF16 gate/up/down graph feeds") +
                                          "; graph types do not certify backend physical arithmetic",
            oracle: "adapted upstream-semantics PyTorch MPS; tiny CPU Flex equivalence is separate",
            tolerance: {atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL},
            profile_order: profile_order, continuation_trace_enabled: continuation_trace,
            ffn_f32_graph_tensors: session.plan.ffn_f32,
            shared_gate_up_comparison: shared_modes, profile_shared_gate_up: profile_shared_gate_up,
            shared_gate_up_prepared: session.plan.shared_gate_up_prepared,
            shared_gate_up_capture_hashes_equal: cross_mode_hashes_equal,
            shared_gate_up_capture_hashes_by_mode: capture_hashes_by_mode,
            shared_gate_up_command_segments_by_mode: expected_segments_by_mode,
            instruments_attach_gate: !trace_ready.empty?,
            capture_order: fixture.capture_order,
            preparation_seconds: preparation_seconds, named_bytes: session.named_bytes,
            borrowed_bytes: session.borrowed_bytes, owned_named_bytes: session.plan.named_bytes,
            owned_named_bytes_by_name: session.plan.named_bytes_by_name, requests: requests,
            passed: passed && requests.size == schedule.size && comparison_complete}
  report_io.puts(result.to_pretty_json)
  report_io.flush
  puts({report: report, passed: result[:passed]}.to_json)
  raise "PixelUMM native/upstream gate failed at unchanged tolerance" unless result[:passed]
ensure
  begin
    # A failed terminal drain must retain the plan and its borrowed bindings.
    if active = chain
      begin
        active.commit_and_wait
      ensure
        if active.completed?
          session.try(&.release)
          active.release
        end
      end
    else
      session.try(&.release)
    end
  ensure
    report_io.close
  end
end
