require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_block_probe"

reference, expected_sha, report = "", "", ""
budget = 0_i64
warmups, repeats = 1, 3
OptionParser.parse do |parser|
  parser.banner = "One PixelUMM block only; external 10% memory guard required."
  parser.on("--reference PATH", "Pinned upstream-adapted GPU fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA-256") { |v| expected_sha = v }
  parser.on("--report PATH", "New comparison/timing report") { |v| report = v }
  parser.on("--max-named-bytes N", "Named-buffer admission budget, up to 8 GiB") { |v| budget = v.to_i64 }
  parser.on("--warmups N", "Untimed warm-up requests, 0..2") { |v| warmups = v.to_i }
  parser.on("--repeats N", "Timed requests after the numerical gate, 1..6") { |v| repeats = v.to_i }
end
unless ARGV.empty? && {reference, expected_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 8_i64 * 1024 * 1024 * 1024 && 0 <= warmups <= 2 && 1 <= repeats <= 6
  raise ArgumentError.new("expected authenticated reference, new report, bounded memory and request counts")
end
raise ArgumentError.new("refusing to replace probe report") if File.info?(report, false)
report_fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
raise IO::Error.new("probe report must be fresh and writable") if report_fd < 0
report_io = IO::FileDescriptor.new(report_fd)
session = nil.as(PixelUMMBlockProbe::Session?)
chain = nil.as(ML::Metal::MPSCommandChain?)
requests = [] of JSON::Any
passed = true
begin
  fixture = PixelUMMBlockProbe::Fixture.new(reference, expected_sha)
  preparation = Time.instant
  session = PixelUMMBlockProbe::Session.new(fixture, budget)
  preparation_seconds = (Time.instant - preparation).total_seconds
  STDERR.puts "PixelUMM one block admitted #{session.named_bytes} named bytes, grid=#{fixture.grid}"
  # The first request is a numerical gate, excluded from measured requests.
  (1 + warmups + repeats).times do |request|
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    start = Time.instant
    session.encode(chain)
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
    start = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - start).total_seconds
    comparisons = (request == 0 || request == warmups + repeats) ? session.compare : [] of PixelUMMBlockProbe::Comparison
    numerical_pass = comparisons.all? { |comparison| comparison[:finite] && comparison[:failures] == 0 }
    passed &&= numerical_pass && chain.wait_count == 1
    kind = request == 0 ? "numerical_gate" : (request <= warmups ? "warmup" : "timed")
    requests << JSON.parse({request: request, kind: kind, passed: numerical_pass,
                            encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                            total_seconds: encode_seconds + wait_seconds, gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                            command_segments: chain.segment_count, terminal_wait_count: chain.wait_count,
                            named_allocations_during_encode: encode_allocations, activation_readbacks_before_terminal_wait: early_reads,
                            diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                            early_release_rejected: early_release_rejected, comparisons: comparisons}.to_json)
    STDERR.puts "#{kind} #{request}: passed=#{numerical_pass}, total=#{encode_seconds + wait_seconds}s"
    chain.release
    chain = nil
    break unless passed
  end
  result = {schema: "cogni_ml.pixelumm_native_block_probe/v1", reference: reference,
            reference_sha256: expected_sha, grid: fixture.grid, block_index: 0,
            generation: false, precision: "BF16 boundaries with F32 normalization/RoPE/attention accumulation",
            oracle: "adapted upstream-semantics PyTorch MPS; tiny CPU Flex equivalence is separate",
            tolerance: {atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL},
            preparation_seconds: preparation_seconds, named_bytes: session.named_bytes,
            borrowed_bytes: session.borrowed_bytes, owned_named_bytes: session.plan.named_bytes,
            owned_named_bytes_by_name: session.plan.named_bytes_by_name, requests: requests,
            passed: passed && requests.size == 1 + warmups + repeats}
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
