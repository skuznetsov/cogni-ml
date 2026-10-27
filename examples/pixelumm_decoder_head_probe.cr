require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_head_probe"

reference, reference_sha, alternate, alternate_sha, report = "", "", "", "", ""
budget = 0_i64
repeats = 1
OptionParser.parse do |parser|
  parser.banner = "Global PixelUMM head only; full J, external 10% cache-aware memory guard required."
  parser.on("--reference PATH", "Authenticated upstream global-head fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata hash") { |v| reference_sha = v }
  parser.on("--alternate-reference PATH", "Different-input fixture for poisoned replay") { |v| alternate = v }
  parser.on("--alternate-reference-sha256 SHA", "Trusted alternate metadata hash") { |v| alternate_sha = v }
  parser.on("--report PATH", "Fresh comparison and timing report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate input/weights/head budget, up to 2 GiB") { |v| budget = v.to_i64 }
  parser.on("--repeats N", "Repeated requests after numerical gate, 1..3") { |v| repeats = v.to_i }
end
unless ARGV.empty? && {reference, reference_sha, alternate, alternate_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= PixelUMMHeadProbe::NAMED_LIMIT && 1 <= repeats <= 3
  raise ArgumentError.new("expected two authenticated full-J fixtures, bounded named memory and fresh report")
end
raise ArgumentError.new("head report must be absolute") unless Path[report].absolute?
fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("head report must be fresh and writable") if fd < 0
report_io = IO::FileDescriptor.new(fd)
observer = PixelUMMMemoryProbe::Recorder.system
session = nil.as(PixelUMMHeadProbe::Session?)
chain = nil.as(ML::Metal::MPSCommandChain?)
requests = [] of JSON::Any
preparation_seconds = 0.0
named_bytes, borrowed_bytes, owned_bytes = 0_i64, 0_i64, 0_i64
failure = nil.as(String?)
passed = true
begin
  observer.emit("before_metadata")
  fixture = PixelUMMHeadProbe::Fixture.new(reference, reference_sha)
  replay = PixelUMMHeadProbe::Fixture.new(alternate, alternate_sha)
  fixture.compatible_replay!(replay)
  observer.emit("after_fixture_authentication")
  start = Time.instant
  session = PixelUMMHeadProbe::Session.new(fixture, budget)
  preparation_seconds = (Time.instant - start).total_seconds
  named_bytes, borrowed_bytes, owned_bytes = session.named_bytes, session.borrowed_bytes, session.plan.named_bytes
  observer.emit("session_ready")
  STDERR.puts "PixelUMM head admitted #{named_bytes} unique named bytes, full J=21120"
  (repeats + 2).times do |request|
    poisoned = request == repeats + 1
    if poisoned
      session.replay_input!(replay)
      poisoned_bytes = session.plan.poison_transients!
      unless poisoned_bytes == 1_124_766_720_i64 && session.plan.last_poisoned_bytes == poisoned_bytes && session.plan.poison_count == 1
        raise "head scratch poisoning omitted a transient plane"
      end
      observer.emit("after_poisoned_replay_input", request: request)
    end
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    observer.emit("request_before_encode", request: request)
    start = Time.instant
    session.encode(chain)
    encode_seconds = (Time.instant - start).total_seconds
    observer.emit("request_after_encode", request: request)
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
      raise "head encode allocated named buffers, read activations or waited"
    end
    early_release_rejected, early_read_rejected, early_reuse_rejected, early_poison_rejected = false, false, false, false
    if request == 0
      begin
        session.plan.release
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("chain")
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
        raise error unless (error.message || "").includes?("chain")
        early_reuse_rejected = true
      end
      begin
        session.plan.poison_transients!
      rescue error : ArgumentError
        raise error unless (error.message || "").includes?("chain")
        early_poison_rejected = true
      end
      unless early_release_rejected && early_read_rejected && early_reuse_rejected && early_poison_rejected &&
             ML::MetalBuffer.hy_text_probe_readbacks == reads
        raise "head accepted in-flight release/read/reuse"
      end
    end
    start = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - start).total_seconds
    observer.emit("request_after_terminal_wait", request: request)
    comparisons = session.compare
    observer.emit("request_after_diagnostic_comparisons", request: request)
    unless comparisons.map { |value| value[:stage] } == PixelUMMHeadProbe::CAPTURES
      raise "head comparison omitted expected captures"
    end
    numerical_pass = comparisons.all? { |result| result[:finite] && result[:failures] == 0 }
    passed &&= numerical_pass && chain.wait_count == 1
    kind = poisoned ? "poisoned_different_input_gate" : (request == 0 ? "numerical_gate" : "timed")
    requests << JSON.parse({request: request, kind: kind, passed: numerical_pass,
                            input_sha256: session.fixture.inputs["block35_output"].sha256,
                            shared_scratch_poisoned_before_request: poisoned,
                            poisoned_bytes: poisoned ? session.plan.last_poisoned_bytes : 0_i64,
                            encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                            total_seconds: encode_seconds + wait_seconds,
                            gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                            command_segments: chain.segment_count, terminal_wait_count: chain.wait_count,
                            named_allocations_during_encode: encode_allocations,
                            activation_readbacks_before_terminal_wait: early_reads,
                            diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                            early_release_rejected: early_release_rejected, early_read_rejected: early_read_rejected,
                            early_reuse_rejected: early_reuse_rejected, early_poison_rejected: early_poison_rejected,
                            aliased_weights_rejected: session.aliased_weights_rejected, comparisons: comparisons}.to_json)
    STDERR.puts "#{kind}: passed=#{numerical_pass}, encode+wait=#{encode_seconds + wait_seconds}s"
    chain.release
    chain = nil
    observer.emit("after_chain_release", request: request)
    break unless passed
  end
rescue error
  failure = "#{error.class}: #{error.message}"
  passed = false
ensure
  if active = chain
    active.commit_and_wait unless active.completed?
    active.release
  end
  session.try(&.release)
  GC.collect
  observer.emit("after_session_release")
  result = {schema: "cogni_ml.pixelumm_native_head_probe/v1", generation: false,
            scope: "standalone_global_decoder_head_from_authenticated_block35_output",
            reference: reference, reference_sha256: reference_sha,
            alternate_reference: alternate, alternate_reference_sha256: alternate_sha,
            grid: [6, 44, 80], output_shape: [1, 3, 24, 704, 1280],
            atol: PixelUMMHeadProbe::ATOL, rtol: PixelUMMHeadProbe::RTOL,
            named_budget_bytes: budget, named_bytes: named_bytes,
            owned_named_bytes: owned_bytes, borrowed_named_bytes: borrowed_bytes,
            preparation_seconds: preparation_seconds, repeats: repeats,
            terminal_waits_per_request: 1, requests: requests,
            memory_phases: observer.events, passed: passed && requests.size == repeats + 2,
            failure: failure}
  report_io.puts(result.to_pretty_json)
  report_io.close
  puts({report: report, passed: result[:passed], named_bytes: named_bytes, failure: failure}.to_json)
end
raise "PixelUMM global head qualification failed: #{failure}" unless passed && requests.size == repeats + 2
