require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/hy_nomas_q4_dit_probe"

reference, reference_sha, index, index_sha, report, mode = "", "", "", "", "", ""
budget = 0_i64
compact_text = false
compact_text_tokens = nil.as(Int32?)
OptionParser.parse do |parser|
  parser.banner = "One complete camera-off DiT forward; external memory guard required."
  parser.on("--reference PATH", "Pinned upstream full DiT fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted reference SHA256") { |v| reference_sha = v }
  parser.on("--cache-index PATH", "Authenticated ordered 54-block cache index") { |v| index = v }
  parser.on("--cache-index-sha256 SHA", "Trusted cache index SHA256") { |v| index_sha = v }
  parser.on("--precision MODE", "q4 or independent bf16 control") { |v| mode = v }
  parser.on("--max-named-bytes N", "Aggregate named residency budget") { |v| budget = v.to_i64 }
  parser.on("--compact-text", "Remove excluded text rows after the unchanged refiner") { compact_text = true }
  parser.on("--compact-text-tokens N", "DiT text storage rows, default 64; requires --compact-text") { |v| compact_text_tokens = v.to_i32 }
  parser.on("--report PATH", "New native comparison report") { |v| report = v }
end
unless ARGV.empty? && {reference, reference_sha, index, index_sha, report}.all? { |v| !v.empty? } &&
       {"bf16", "q4"}.includes?(mode) && 0 < budget <= 24_i64 * 1024 * 1024 * 1024
  raise ArgumentError.new("Expected authenticated inputs, precision bf16|q4, new report, and budget <=24 GiB")
end
# M64 is the smallest tested aligned shape that preserves the fixed fixture's
# projection arithmetic. Explicit sizes remain available as diagnostic probes.
text_storage = compact_text_tokens || (compact_text ? 64 : 23)
raise ArgumentError.new("Refusing to replace native report") if File.info?(report, false)
admitted_index = ML::Video::HyNomasQ4StackIndex.load!(index, index_sha, HyNomasQ4DiTIO::SOURCE_SHA, HyNomasQ4DiTIO::CONFIG)
fixture = HyNomasQ4DiTProbe::Fixture.new(reference, reference_sha, mode, admitted_index)
session = nil.as(HyNomasQ4DiTProbe::Session?)
chain = nil.as(ML::Metal::MPSCommandChain?)
requests = [] of JSON::Any
begin
  session = HyNomasQ4DiTProbe::Session.new(fixture, index, index_sha, admitted_index, mode, budget,
    compact_text: compact_text, compact_text_tokens: text_storage.not_nil!)
  STDERR.puts "admitted #{session.named_bytes} named bytes; mode=#{mode}; patch/refiner/54 blocks/final, no diffusion generation"
  2.times do |request|
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    started = Time.instant
    session.encode(chain, poison: request > 0, poison_text_suffix: compact_text && request > 0)
    encode_seconds = (Time.instant - started).total_seconds
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    raise "Full DiT encode allocated Metal buffers/read back activations/waited" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
    early_release_rejected = false
    if request == 0
      begin
        session.release
      rescue ex : ArgumentError
        raise ex unless (ex.message || "").includes?("command chain")
        early_release_rejected = true
      end
      raise "Full DiT accepted in-flight release" unless early_release_rejected
    end
    started = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - started).total_seconds
    comparisons = session.compare(request.to_i32)
    diagnostic_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    raise "Full DiT unexpected postterminal read count" unless diagnostic_reads == 14
    passed = chain.wait_count == 1 && comparisons.all? { |r| r[:finite] && r[:failures] == 0 && r[:first_request_sha256_match] != false }
    requests << JSON.parse({request: request, passed: passed, scratch_poisoned: request > 0,
                            excluded_suffix_poisoned: compact_text && request > 0,
                            poison_buffer_count: request > 0 ? session.poison_count : 0,
                            named_allocations_during_encode: encode_allocations,
                            activation_readbacks_before_terminal_wait: early_reads,
                            diagnostic_readbacks_after_terminal_wait: diagnostic_reads,
                            terminal_wait_count: chain.wait_count, early_release_rejected: early_release_rejected,
                            command_segments: chain.segment_count, encode_seconds: encode_seconds,
                            terminal_wait_seconds: wait_seconds, gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                            comparisons: comparisons}.to_json)
    STDERR.puts "request #{request + 1}/2: #{passed}; encode=#{encode_seconds}s, terminal=#{wait_seconds}s"
    chain.release
    chain = nil
    break unless passed
  end
  passed = requests.size == 2 && requests.all? { |r| r["passed"].as_bool }
  result = {schema: "cogni_ml.hy_nomas_full_dit_native_comparison", schema_version: 1,
            reference: fixture.path, reference_sha256: fixture.sha256, cache_index: index, cache_index_sha256: index_sha,
            precision_mode: mode, compact_text: compact_text,
            dit_text_tokens: session.dit_text_tokens, valid_text_tokens: 23,
            named_bytes: session.named_bytes, cache_payload_bytes: session.cache_payload_bytes,
            camera_conditioning: "off", action_conditioning: "off", diffusion_generation: false,
            qwen_input: "authenticated saved upstream state26 crop; raw upload then GPU BF16 conversion",
            neural_handoffs: "GPU buffers on one chain; one shared BF16 weight stage between 54 ordered blocks",
            precision_scope: "BF16 resident globals; Q4_K or BF16 resident blocks; F32 neural math and MPS attention",
            tolerance: {atol: HyNomasQ4DiTProbe::ATOL, rtol: HyNomasQ4DiTProbe::RTOL}, requests: requests, passed: passed}
  HyNomasQ4DiTIO.write_new!(report, result.to_pretty_json + "\n")
  puts({report: report, passed: passed, precision_mode: mode, named_bytes: session.named_bytes}.to_json)
  raise "Full DiT native/upstream gate failed; unchanged tolerances" unless passed
ensure
  chain.try(&.commit_and_wait)
  session.try(&.release)
  chain.try(&.release)
end
