require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_decoder_probe"

reference, reference_sha, alternate, alternate_sha = "", "", "", ""
head_reference, head_reference_sha, head_alternate, head_alternate_sha = "", "", "", ""
cache_path, cache_sha, report = "", "", ""
budget = 0_i64
repeats = 1
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
end
unless ARGV.empty? && {reference, reference_sha, alternate, alternate_sha,
                       head_reference, head_reference_sha, head_alternate, head_alternate_sha,
                       cache_path, cache_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 24_i64 * 1024**3 && 1 <= repeats <= 3
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
    max_named_bytes: budget, phase_observer: observer)
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
  (repeats + 2).times do |request|
    poisoned = request == repeats + 1
    if poisoned
      session.replay_input!(block_replay, head_replay)
      session.poison_transients!
      unless session.head_plan.poison_count == 1 && session.head_plan.last_poisoned_bytes == 1_124_766_720_i64
        raise "integrated replay omitted head transient poison"
      end
      observer.emit("after_poisoned_replay_input", request: request)
    end
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    head_before_blocks_rejected = false
    early_release_rejected, early_read_rejected, early_reuse_rejected, early_poison_rejected = false, false, false, false
    if request == 0
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
    session.encode_blocks(chain)
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
    if request == 0
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
    kind = poisoned ? "poisoned_different_input_gate" : (request == 0 ? "numerical_gate" : "timed")
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
    schema: "cogni_ml.pixelumm_native_integrated_probe/v1", generation: false,
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
    timing_scope: "encode_seconds sums native API calls; total_seconds includes intervening memory telemetry and first-request lifecycle checks, excludes fixture preparation and post-wait comparisons; not a throughput benchmark",
    memory_telemetry: {categories_additive: false, guard_timestamp_clock: "mach_absolute_time",
                       guard_join_timestamp_field: "guard_monotonic_ns", sampled_peaks_are_bounds: false},
    memory_phases: observer.events, passed: passed && requests.size == repeats + 2, failure: failure,
  }
  report_io.puts(result.to_pretty_json)
  report_io.close
  puts({report: report, passed: result[:passed], named_bytes: named_bytes, failure: failure}.to_json)
end
raise "PixelUMM integrated qualification failed: #{failure}" unless passed && requests.size == repeats + 2
