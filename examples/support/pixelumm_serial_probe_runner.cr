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
    reuse_outputs = false
    compare_common_ring_captures = false
    memory_phases = false
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
      parser.on("--reuse-output-banks", "Opt-in two output banks; omit overwritten diagnostic output aliases") { reuse_outputs = true }
      parser.on("--compare-common-ring-captures", "Compare only captures stable with two output banks, without changing allocated workspace") { compare_common_ring_captures = true }
      parser.on("--memory-phases", "Emit opt-in scalar process/Metal/GC memory phase events") { memory_phases = true }
    end
    PixelUMMTwoBlockProbe.validate_serial_execution_mode!(block_count,
      ffn_f32: ffn_f32, reuse_outputs: reuse_outputs)
    unless ARGV.empty? && {reference, expected_sha, cache_path, cache_sha, report}.all? { |v| !v.empty? } &&
           alternate.empty? == alternate_sha.empty? && 0 < budget <= named_budget_limit &&
           0 <= warmups <= 2 && 1 <= repeats <= 6
      raise ArgumentError.new("expected authenticated fixtures/cache, new report and bounded memory/request counts")
    end
    raise ArgumentError.new("refusing to replace serial probe report") if File.info?(report, false)
    fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("serial probe report must be fresh and writable") if fd < 0
    report_io = IO::FileDescriptor.new(fd)
    memory_observer = memory_phases ? PixelUMMMemoryProbe::Recorder.system : nil
    session = nil.as(PixelUMMTwoBlockProbe::Session?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    requests = [] of JSON::Any
    passed = true
    begin
      memory_phase(memory_observer, "before_metadata")
      fixture = PixelUMMTwoBlockProbe::Fixture.new(reference, expected_sha, block_count)
      memory_phase(memory_observer, "after_fixture_metadata_authentication")
      replay = alternate.empty? ? nil : PixelUMMTwoBlockProbe::Fixture.new(alternate, alternate_sha, block_count)
      if other = replay
        memory_phase(memory_observer, "after_alternate_fixture_metadata_authentication")
        fixture.compatible_replay!(other)
        PixelUMMTwoBlockProbe.validate_replay_output_distinguishability!(
          fixture.captures, other.captures, block_count)
      end
      memory_phase(memory_observer, "before_cache_metadata_authentication")
      cache = PixelUMMTwoBlockProbe::WeightCache.new(cache_path, cache_sha, fixture)
      memory_phase(memory_observer, "after_cache_metadata_authentication")
      memory_phase(memory_observer, "after_metadata")
      preparation = Time.instant
      session = PixelUMMTwoBlockProbe::Session.new(fixture, cache, budget,
        ffn_f32: ffn_f32, reuse_outputs: reuse_outputs,
        compare_common_ring_captures: compare_common_ring_captures,
        phase_observer: memory_observer)
      preparation_seconds = (Time.instant - preparation).total_seconds
      memory_phase(memory_observer, "session_ready")
      STDERR.puts "PixelUMM serial admitted #{session.named_bytes} unique named bytes, grid=#{fixture.grid}"
      total_requests = 1 + warmups + repeats + (replay ? 1 : 0)
      total_requests.times do |request|
        replay_request = request == 1 + warmups + repeats
        if replay_request
          replay_fixture = replay.not_nil!
          session.replay_input!(replay_fixture)
          unless session.fixture.inputs["x"].sha256 == replay_fixture.inputs["x"].sha256
            raise "PixelUMM poisoned replay did not bind the alternate initial input"
          end
          session.poison_transients!
        end
        request_input_sha256 = session.fixture.inputs["x"].sha256
        chain = ML::Metal::MPSCommandChain.new
        wrong_order_rejected = false
        stale_output_access_rejected = false
        wrong_order_slots_rejected = [] of Int32
        stale_output_slots_rejected = [] of Int32
        stable_output_slots_accessible = [] of Int32
        predecessor_input_alias_slots_verified = [] of Int32
        if request == 0
          if reuse_outputs && block_count > 2
            stale_slots = block_count == 36 ? [16, 18, 33] : [0]
            stale_slots.each do |slot|
              begin
                session.plans[slot].stage("output")
              rescue error : ArgumentError
                raise error unless (error.message || "").includes?("not captured")
                stale_output_slots_rejected << slot
              end
            end
            if block_count == 36
              [16, 18, 33, 34, 35].each do |slot|
                unless session.plans[slot].input.same?(session.workspace.output_buffer(slot - 1))
                  raise "PixelUMM block input alias did not use its immediate predecessor output bank"
                end
                predecessor_input_alias_slots_verified << slot
              end
              [34, 35].each do |slot|
                unless session.plans[slot].stage("output").same?(session.workspace.output_buffer(slot))
                  raise "PixelUMM final retained output was not backed by its assigned output bank"
                end
                stable_output_slots_accessible << slot
              end
            end
            stale_output_access_rejected = stale_output_slots_rejected == stale_slots &&
                                           (block_count != 36 || predecessor_input_alias_slots_verified == [16, 18, 33, 34, 35] &&
                                             stable_output_slots_accessible == [34, 35])
            raise "PixelUMM exposed an overwritten output or input alias as a persistent capture" unless stale_output_access_rejected
          end
          wrong_order_slots = block_count == 36 ? [16, 17, 18, 34, 35] : [block_count - 1]
          wrong_order_slots.each do |slot|
            begin
              session.plans[slot].encode(chain)
            rescue error : ArgumentError
              raise error unless (error.message || "").includes?("slot")
              wrong_order_slots_rejected << slot
            end
          end
          wrong_order_rejected = wrong_order_slots_rejected == wrong_order_slots
          raise "PixelUMM accepted an out-of-order block before block zero" unless wrong_order_rejected
        end
        allocations = ML::MetalBuffer.hy_text_probe_allocations
        reads = ML::MetalBuffer.hy_text_probe_readbacks
        memory_phase(memory_observer, "request_before_encode", request: request)
        start = Time.instant
        session.encode(chain)
        encode_seconds = (Time.instant - start).total_seconds
        memory_phase(memory_observer, "request_after_encode", request: request)
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
        memory_phase(memory_observer, "request_after_terminal_wait", request: request)
        compare_request = request == 0 || request == warmups + repeats || replay_request
        comparisons = if compare_request
                        memory_phase(memory_observer, "before_cpu_diagnostic_comparisons", request: request)
                        session.compare(request: request, phase_observer: memory_observer)
                      else
                        [] of PixelUMMTwoBlockProbe::Comparison
                      end
        memory_phase(memory_observer, "after_cpu_diagnostic_comparisons", request: request) if compare_request
        if compare_request && comparisons.map { |value| value[:stage] } != session.capture_order
          raise "PixelUMM comparison omitted expected captures"
        end
        numerical_pass = comparisons.all? { |result| result[:finite] && result[:failures] == 0 }
        passed &&= numerical_pass && chain.wait_count == 1
        kind = replay_request ? "poisoned_different_input_gate" : (request == 0 ? "numerical_gate" : (request <= warmups ? "warmup" : "timed"))
        requests << JSON.parse({request: request, kind: kind, passed: numerical_pass,
                                input_sha256: request_input_sha256,
                                encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                                total_seconds: encode_seconds + wait_seconds, gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                                command_segments: chain.segment_count, terminal_wait_count: chain.wait_count,
                                named_allocations_during_encode: encode_allocations, activation_readbacks_before_terminal_wait: early_reads,
                                diagnostic_readbacks_after_terminal_wait: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                                wrong_order_rejected: wrong_order_rejected, early_release_rejected: early_release_rejected,
                                wrong_order_slots_rejected: wrong_order_slots_rejected,
                                stale_output_slots_rejected: stale_output_slots_rejected,
                                predecessor_input_alias_slots_verified: predecessor_input_alias_slots_verified,
                                stable_output_slots_accessible: stable_output_slots_accessible,
                                early_read_rejected: early_read_rejected, early_reuse_rejected: early_reuse_rejected,
                                shared_scratch_poisoned_before_request: replay_request, comparisons: comparisons}.to_json)
        requests[-1].as_h["stale_output_access_rejected"] = JSON::Any.new(stale_output_access_rejected)
        STDERR.puts "#{kind} #{request}: passed=#{numerical_pass}, total=#{encode_seconds + wait_seconds}s"
        chain.release
        chain = nil
        memory_phase(memory_observer, "after_chain_release", request: request)
        break unless passed
      end
      schema = PixelUMMTwoBlockProbe.fixture_schema(block_count).sub("pixelumm_", "pixelumm_native_").sub("_reference/", "_probe/")
      result = {schema: schema, reference: reference,
                reference_sha256: expected_sha, alternate_reference: alternate, alternate_reference_sha256: alternate_sha,
                bf16_cache: cache_path, bf16_cache_sha256: cache_sha, grid: fixture.grid, block_indices: (0...block_count).to_a,
                generation: false, shared_transient_workspace: true, persistent_output_banks: session.workspace.output_bank_count,
                reuse_output_banks: reuse_outputs, output_bank_by_slot: session.output_bank_by_slot,
                compare_common_ring_captures: compare_common_ring_captures,
                comparison_coverage_mode: compare_common_ring_captures ? "common_ring_survivors" : "workspace_capture_order",
                compared_capture_order: session.capture_order,
                comparison_capture_omissions: fixture.capture_order - session.capture_order,
                omitted_comparison_capture_aliases: fixture.capture_order - session.capture_order,
                omitted_overwritten_capture_aliases: fixture.capture_order - session.workspace_capture_order,
                authenticated_fixture_capture_count: fixture.captures.size,
                handoff: "each physical block output is the next block input on GPU, without a copy or intermediate wait; reused banks are ephemeral",
                precision: "BF16 weights and handoffs; F32 normalization/RoPE/attention intermediates; " +
                  (session.ffn_f32 ? "gate/up/down operands cast to F32 graph tensors, with BF16 output casts" : "BF16 gate/up/down graph feeds") +
                  "; graph types do not certify backend physical accumulation precision",
                max_named_bytes: budget, named_byte_limit: named_budget_limit,
                warmups: warmups, repeats: repeats, request_count: requests.size,
                ffn_f32_graph_tensors: session.ffn_f32, ffn_f32_by_slot: session.plans.map(&.ffn_f32),
                oracle: "adapted upstream-semantics PyTorch MPS; compiled CUDA Flex equivalence is not claimed",
                tolerance: {atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL},
                preparation_seconds: preparation_seconds, named_bytes: session.named_bytes,
                borrowed_bytes: session.borrowed_bytes, owned_named_bytes: session.workspace.named_bytes,
                requests: requests, passed: passed && requests.size == total_requests}
      if memory_observer
        session.release
        memory_phase(memory_observer, "runner_after_session_release")
        session = nil
      end
      report_json = result.to_pretty_json
      if block_count == 36
        full36_report_document = JSON.parse(report_json)
        full36_report_document.as_h["full36_capture_policy"] = fixture.document["full36_capture_policy"]
        full36_report_document.as_h["unretained_early_output_aliases"] = JSON.parse(
          PixelUMMTwoBlockProbe.thirty_six_unretained_output_aliases.to_json)
        report_json = full36_report_document.to_pretty_json
      end
      if observer = memory_observer
        report_document = JSON.parse(report_json)
        report_document.as_h["memory_telemetry"] = JSON.parse({
          enabled: true,
          schema: PixelUMMMemoryProbe::SCHEMA,
          event_prefix: PixelUMMMemoryProbe::PREFIX,
          timestamp_clock: "CLOCK_MONOTONIC",
          guard_timestamp_clock: "mach_absolute_time",
          guard_join_timestamp_field: "guard_monotonic_ns",
          categories_additive: false,
          category_notes: "Process footprint, RSS, Metal allocation, named MetalBuffer counters, and Crystal GC counters overlap; do not sum them. Opaque MPS bytes are not estimated.",
        }.to_json)
        report_document.as_h["memory_phases"] = JSON::Any.new(observer.events)
        report_json = report_document.to_pretty_json
      end
      report_io.puts(report_json)
      report_io.flush
      puts({report: report, passed: result[:passed], memory_phases: memory_phases,
            compare_common_ring_captures: compare_common_ring_captures}.to_json)
      raise "PixelUMM serial failed at unchanged upstream tolerance" unless result[:passed]
    ensure
      begin
        if active = chain
          begin
            active.commit_and_wait
          ensure
            if active.completed?
              cleanup = PixelUMMMemoryProbe::Cleanup.new
              begin
                cleanup.attempt { session.try(&.release) }
              ensure
                cleanup.attempt { active.release }
              end
              cleanup.raise_first!
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

  private def memory_phase(observer : PixelUMMMemoryProbe::Recorder?, phase : String, *,
                           slot : Int32? = nil, request : Int32? = nil,
                           phase_bytes_kind : String? = nil, phase_bytes : Int64? = nil) : Nil
    observer.try do |active|
      active.emit(phase, slot: slot, request: request,
        phase_bytes_kind: phase_bytes_kind, phase_bytes: phase_bytes)
    end
    nil
  end
end
