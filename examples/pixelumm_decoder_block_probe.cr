require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_block_probe"

reference, expected_sha, report = "", "", ""
budget = 0_i64
warmups, repeats = 1, 3
profile_order = "none"
continuation_trace = false
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
  parser.on("--trace-ready-file PATH", "Fresh PID receipt for bounded Instruments attach before encoding") { |v| trace_ready = v }
  parser.on("--trace-release-file PATH", "Fresh CPU release file paired with --trace-ready-file") { |v| trace_release = v }
end
unless ARGV.empty? && {reference, expected_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 8_i64 * 1024 * 1024 * 1024 && 0 <= warmups <= 2 && 1 <= repeats <= 6 &&
       {"none", "off-on", "on-off"}.includes?(profile_order) && (!continuation_trace || profile_order != "none")
  raise ArgumentError.new("expected authenticated reference, new report, bounded memory and request counts")
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
session = nil.as(PixelUMMBlockProbe::Session?)
chain = nil.as(ML::Metal::MPSCommandChain?)
requests = [] of JSON::Any
passed = true
schedule = [] of NamedTuple(kind: String, profile: Bool, pair: Int32)
if profile_order == "none"
  (1 + warmups + repeats).times do |index|
    schedule << {kind: index == 0 ? "numerical_gate" : (index <= warmups ? "warmup" : "timed"), profile: false, pair: -1}
  end
else
  initial_order = profile_order == "off-on" ? [false, true] : [true, false]
  initial_order.each { |profile| schedule << {kind: "numerical_gate", profile: profile, pair: -1} }
  warmups.times do
    initial_order.each { |profile| schedule << {kind: "warmup", profile: profile, pair: -1} }
  end
  repeats.times do |pair|
    order = pair.even? ? initial_order : initial_order.reverse
    order.each { |profile| schedule << {kind: "timed", profile: profile, pair: pair} }
  end
end
expected_segments = nil.as(Int32?)
capture_hashes = nil.as(Hash(String, String)?)
begin
  fixture = PixelUMMBlockProbe::Fixture.new(reference, expected_sha)
  preparation = Time.instant
  session = PixelUMMBlockProbe::Session.new(fixture, budget)
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
  # The first request is a numerical gate, excluded from measured requests.
  schedule.each_with_index do |arm, request|
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
    if request == 0
      begin
        session.release
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("command chain")
        early_release_rejected = true
      end
      raise "PixelUMM accepted in-flight release" unless early_release_rejected
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
    comparisons = (profile_order != "none" || request == 0 || request == schedule.size - 1) ? session.compare : [] of PixelUMMBlockProbe::Comparison
    numerical_pass = comparisons.all? { |comparison| comparison[:finite] && comparison[:failures] == 0 }
    unless comparisons.empty?
      hashes = comparisons.to_h { |comparison| {comparison[:stage], comparison[:actual_sha256]} }
      capture_hashes ||= hashes
      raise "PixelUMM off/on capture words changed" unless capture_hashes == hashes
    end
    expected_segments ||= chain.segment_count
    raise "PixelUMM profiling changed command-root count" unless chain.segment_count == expected_segments
    phase_profile = arm[:profile] ? JSON.parse(session.plan.phase_profile(chain).to_json) : nil
    trace_profile = if arm[:profile] && continuation_trace
                      trace = chain.continuation_trace
                      JSON.parse({info: trace.info, events: trace.events, roots: trace.roots}.to_json)
                    end
    passed &&= numerical_pass && chain.wait_count == 1
    kind = arm[:kind]
    requests << JSON.parse({request: request, kind: kind, passed: numerical_pass,
                            profile_enabled: arm[:profile], pair: arm[:pair],
                            prewait_profile_rejected: prewait_profile_rejected,
                            encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                            total_seconds: encode_seconds + wait_seconds, gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                            command_segments: chain.segment_count, terminal_wait_count: chain.wait_count,
                            named_allocations_during_encode: encode_allocations, activation_readbacks_before_terminal_wait: early_reads,
                            diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                            early_release_rejected: early_release_rejected, comparisons: comparisons,
                            phase_profile: phase_profile, continuation_trace: trace_profile}.to_json)
    STDERR.puts "#{kind} #{request} profile=#{arm[:profile]}: passed=#{numerical_pass}, total=#{encode_seconds + wait_seconds}s"
    chain.release
    chain = nil
    break unless passed
  end
  result = {schema: "cogni_ml.pixelumm_native_block_probe/v1", reference: reference,
            reference_sha256: expected_sha, grid: fixture.grid, block_index: 0,
            generation: false, precision: "BF16 boundaries with F32 normalization/RoPE/attention accumulation",
            oracle: "adapted upstream-semantics PyTorch MPS; tiny CPU Flex equivalence is separate",
            tolerance: {atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL},
            profile_order: profile_order, continuation_trace_enabled: continuation_trace,
            instruments_attach_gate: !trace_ready.empty?,
            capture_order: fixture.capture_order,
            preparation_seconds: preparation_seconds, named_bytes: session.named_bytes,
            borrowed_bytes: session.borrowed_bytes, owned_named_bytes: session.plan.named_bytes,
            owned_named_bytes_by_name: session.plan.named_bytes_by_name, requests: requests,
            passed: passed && requests.size == schedule.size}
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
