require "option_parser"
require "./hy_nomas_probe_instrumentation"
require "./pixelumm_two_block_probe"

module PixelUMMSerialProbeRunner
  extend self

  def run(block_count : Int32) : Nil
    PixelUMMTwoBlockProbe.validate_block_count!(block_count)
    named_budget_limit = PixelUMMTwoBlockProbe.named_budget_limit(block_count)
    reference, expected_sha, cache_path, cache_sha, report = "", "", "", "", ""
    alternate, alternate_sha = "", ""
    budget = 0_i64
    warmups, repeats = 1, 3
    ffn_f32 = false
    OptionParser.parse do |parser|
      parser.banner = "#{block_count} sequential PixelUMM blocks only; named budget capped at #{named_budget_limit} bytes and external 10% memory guard required."
      parser.on("--reference PATH", "Sequential upstream-adapted GPU fixture") { |v| reference = v }
      parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA-256") { |v| expected_sha = v }
      parser.on("--bf16-cache PATH", "Authenticated preconverted BF16 weights") { |v| cache_path = v }
      parser.on("--bf16-cache-sha256 SHA", "Trusted BF16 cache metadata SHA-256") { |v| cache_sha = v }
      parser.on("--alternate-reference PATH", "Different-input fixture for poisoned shared-scratch replay") { |v| alternate = v }
      parser.on("--alternate-reference-sha256 SHA", "Trusted alternate fixture metadata SHA-256") { |v| alternate_sha = v }
      parser.on("--report PATH", "New comparison/timing report") { |v| report = v }
      parser.on("--max-named-bytes N", "Aggregate unique named budget, up to #{named_budget_limit} bytes") { |v| budget = v.to_i64 }
      parser.on("--warmups N", "Untimed warm-up requests, 0..2") { |v| warmups = v.to_i }
      parser.on("--repeats N", "Timed requests after numerical gate, 1..6") { |v| repeats = v.to_i }
      parser.on("--ffn-f32", "Opt-in F32 graph tensors for every gate/up/down, retaining BF16 boundaries") { ffn_f32 = true }
    end
    unless ARGV.empty? && {reference, expected_sha, cache_path, cache_sha, report}.all? { |v| !v.empty? } &&
           alternate.empty? == alternate_sha.empty? && 0 < budget <= named_budget_limit &&
           0 <= warmups <= 2 && 1 <= repeats <= 6
      raise ArgumentError.new("expected authenticated fixtures/cache, new report and bounded memory/request counts")
    end
    raise ArgumentError.new("refusing to replace serial probe report") if File.info?(report, false)
    fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("serial probe report must be fresh and writable") if fd < 0
    report_io = IO::FileDescriptor.new(fd)
    session = nil.as(PixelUMMTwoBlockProbe::Session?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    requests = [] of JSON::Any
    passed = true
    begin
      fixture = PixelUMMTwoBlockProbe::Fixture.new(reference, expected_sha, block_count)
      replay = alternate.empty? ? nil : PixelUMMTwoBlockProbe::Fixture.new(alternate, alternate_sha, block_count)
      if other = replay
        fixture.compatible_replay!(other)
        unless (0...block_count).all? { |slot| fixture.captures["block#{slot}.output"].sha256 != other.captures["block#{slot}.output"].sha256 }
          raise ArgumentError.new("alternate oracle outputs must distinguish every block")
        end
      end
      cache = PixelUMMTwoBlockProbe::WeightCache.new(cache_path, cache_sha, fixture)
      preparation = Time.instant
      session = PixelUMMTwoBlockProbe::Session.new(fixture, cache, budget, ffn_f32: ffn_f32)
      preparation_seconds = (Time.instant - preparation).total_seconds
      STDERR.puts "PixelUMM serial admitted #{session.named_bytes} unique named bytes, grid=#{fixture.grid}"
      total_requests = 1 + warmups + repeats + (replay ? 1 : 0)
      total_requests.times do |request|
        replay_request = request == 1 + warmups + repeats
        if replay_request
          session.replay_input!(replay.not_nil!)
          session.poison_transients!
        end
        chain = ML::Metal::MPSCommandChain.new
        wrong_order_rejected = false
        if request == 0
          begin
            session.plans[block_count - 1].encode(chain)
          rescue error : ArgumentError
            raise error unless (error.message || "").includes?("slot")
            wrong_order_rejected = true
          end
          raise "PixelUMM accepted the last block before block zero" unless wrong_order_rejected
        end
        allocations = ML::MetalBuffer.hy_text_probe_allocations
        reads = ML::MetalBuffer.hy_text_probe_readbacks
        start = Time.instant
        session.encode(chain)
        encode_seconds = (Time.instant - start).total_seconds
        encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
        early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
        unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
          raise "PixelUMM serial encode allocated named buffers, read activations or waited"
        end
        early_release_rejected, early_read_rejected, early_reuse_rejected = false, false, false
        if request == 0
          begin
            session.release
          rescue error : ArgumentError
            raise error unless (error.message || "").includes?("command chain")
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
            raise error unless (error.message || "").includes?("command chain")
            early_reuse_rejected = true
          end
          unless early_release_rejected && early_read_rejected && early_reuse_rejected &&
                 ML::MetalBuffer.hy_text_probe_readbacks == reads
            raise "PixelUMM serial accepted in-flight release/read/reuse"
          end
        end
        start = Time.instant
        chain.commit_and_wait
        wait_seconds = (Time.instant - start).total_seconds
        compare_request = request == 0 || request == warmups + repeats || replay_request
        comparisons = compare_request ? session.compare : [] of PixelUMMTwoBlockProbe::Comparison
        if compare_request && comparisons.size != fixture.captures.size
          raise "PixelUMM comparison omitted expected captures"
        end
        numerical_pass = comparisons.all? { |result| result[:finite] && result[:failures] == 0 }
        passed &&= numerical_pass && chain.wait_count == 1
        kind = replay_request ? "poisoned_different_input_gate" : (request == 0 ? "numerical_gate" : (request <= warmups ? "warmup" : "timed"))
        requests << JSON.parse({request: request, kind: kind, passed: numerical_pass,
                                encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                                total_seconds: encode_seconds + wait_seconds, gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                                command_segments: chain.segment_count, terminal_wait_count: chain.wait_count,
                                named_allocations_during_encode: encode_allocations, activation_readbacks_before_terminal_wait: early_reads,
                                diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                                wrong_order_rejected: wrong_order_rejected, early_release_rejected: early_release_rejected,
                                early_read_rejected: early_read_rejected, early_reuse_rejected: early_reuse_rejected,
                                shared_scratch_poisoned_before_request: replay_request, comparisons: comparisons}.to_json)
        STDERR.puts "#{kind} #{request}: passed=#{numerical_pass}, total=#{encode_seconds + wait_seconds}s"
        chain.release
        chain = nil
        break unless passed
      end
      schema = PixelUMMTwoBlockProbe.fixture_schema(block_count).sub("pixelumm_", "pixelumm_native_").sub("_reference/", "_probe/")
      result = {schema: schema, reference: reference,
                reference_sha256: expected_sha, alternate_reference: alternate, alternate_reference_sha256: alternate_sha,
                bf16_cache: cache_path, bf16_cache_sha256: cache_sha, grid: fixture.grid, block_indices: (0...block_count).to_a,
                generation: false, shared_transient_workspace: true, persistent_output_banks: block_count,
                handoff: "each block output is the next block input on GPU, without a copy or intermediate wait",
                precision: "BF16 weights and handoffs; F32 normalization/RoPE/attention intermediates; " +
                  (session.ffn_f32 ? "gate/up/down operands cast to F32 graph tensors, with BF16 output casts" : "BF16 gate/up/down graph feeds") +
                  "; graph types do not certify backend physical accumulation precision",
                ffn_f32_graph_tensors: session.ffn_f32, ffn_f32_by_slot: session.plans.map(&.ffn_f32),
                oracle: "adapted upstream-semantics PyTorch MPS; compiled CUDA Flex equivalence is not claimed",
                tolerance: {atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL},
                preparation_seconds: preparation_seconds, named_bytes: session.named_bytes,
                borrowed_bytes: session.borrowed_bytes, owned_named_bytes: session.workspace.named_bytes,
                requests: requests, passed: passed && requests.size == total_requests}
      report_io.puts(result.to_pretty_json)
      report_io.flush
      puts({report: report, passed: result[:passed]}.to_json)
      raise "PixelUMM serial failed at unchanged upstream tolerance" unless result[:passed]
    ensure
      begin
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
  end
end
