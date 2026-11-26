require "./hy_nomas_bf16_cache_probe"

# Opt-in correctness replay, not a throughput benchmark. Saved upstream inputs
# are uploaded once; the predecessor's GPU outputs feed its successor directly.
module HyNomasBF16SerialProbe
  extend self
  alias Serial = HyNomasPretrainedSerialProbe
  alias Cache = HyNomasBF16CacheProbe

  # A caller-pinned oracle still needs an external producer receipt. At least
  # reject self-oracles (including copied bytes and hardlinks) at admission.
  def validate_source_oracle_pairs!(sources : Array(Serial::BF16Source),
                                    oracles : Array(Serial::BF16Source)) : Nil
    Serial.validate_bf16_sources!(sources)
    Serial.validate_bf16_sources!(oracles)
    sources.each do |source|
      oracles.each do |oracle|
        if source[:sha256] == oracle[:sha256] ||
           File.info(source[:path]).same_file?(File.info(oracle[:path]))
          raise ArgumentError.new("BF16 caches and independent oracles must not alias or have identical file digests")
        end
      end
    end
  end

  def run(reference_path : String, reference_sha : String, conditioning_path : String,
          sources : Array(Serial::BF16Source), oracle_sources : Array(Serial::BF16Source),
          report_path : String, budget : Int64, successor_block0_control : Bool = false) : Bool
    raise ArgumentError.new("refusing to replace serial cache report") if File.info?(report_path, false)
    unless Dir.exists?(File.dirname(File.expand_path(report_path)))
      raise ArgumentError.new("serial cache report parent does not exist")
    end
    validate_source_oracle_pairs!(sources, oracle_sources)
    reference = Serial::ReferenceFixture.new(reference_path, reference_sha)
    conditioning_sha = Serial::PINNED_CONDITIONING_SHA
    qwen_sha = Serial.conditioning_qwen_sha256(conditioning_path, conditioning_sha)
    conditioning = HyNomasTextProbe::Fixture.new(conditioning_path, conditioning_sha, qwen_sha)
    reference.validate_conditioning_inputs!(conditioning)
    reference.authenticate_weight_artifacts!(conditioning)

    owner = nil.as(Serial::RunOwner?)
    oracle_files = [] of ML::Video::CheckpointFile
    cache_files = [] of ML::Video::CheckpointFile
    weight_results = [] of Cache::WeightComparison
    begin
      # Authenticate both complete inventories, oracle pairings and plans before
      # constructing an owner that can allocate a Metal buffer. The owner's
      # actual consuming handles reauthenticate the same immutable sources.
      sources.each_with_index do |source, index|
        file = ML::Video::BF16WeightCache.authenticate(source[:path], source[:sha256])
        cache_files << file
        plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], Serial::CONFIG,
          Serial::NAMED_BUDGET_LIMIT_BYTES, resident_dtype: ML::Video::WeightDType::BF16,
          block_index: index.to_i32, require_all: true, allow_f32_to_bf16: false)
        Serial.validate_bf16_plan!(plan, index.to_i32)
        oracle_source = oracle_sources[index]
        oracle = ML::Video::BF16WeightCache.authenticate(oracle_source[:path], oracle_source[:sha256])
        oracle_files << oracle
        Cache.validate_oracle!(plan, oracle, index.to_i32)
      end
      cache_files.each(&.close)
      cache_files.clear

      owner = Serial::RunOwner.new(reference, conditioning, budget, bf16_sources: sources)
      active = owner
      raise "BF16 serial replay retained conversion dispatches" unless active.weight_conversion_dispatches == 0
      active.run_control!
      active.each_completed_weight_plan do |index, plan, arena|
        Cache.validate_oracle!(plan, oracle_files[index], index)
        weight_results.concat(Cache.compare_weight_buffers!(plan, arena, oracle_files[index]))
      end
      unless weight_results.size == 64 && weight_results.all? { |row| row[:different_words] == 0 }
        raise "two-block resident BF16 weights differ from their independent upstream conversions"
      end
      active.release_control_phase!
      active.prepare_serial!(successor_block0_control: successor_block0_control)
      if successor_block0_control
        active.run_shared_request!(1, capture_first_repeat: false, negative_mode: true)
      else
        active.run_shared_request!(1, capture_first_repeat: true, negative_mode: false)
        active.run_shared_request!(2, capture_first_repeat: false, negative_mode: false)
      end
      passed = active.control_passed? && active.shared_passed?(successor_block0_control)
      active.cleanup_after_terminal!
      report = {
        schema: "cogni_ml.hy_nomas_bf16_serial_probe.v1", passed: passed,
        mode: successor_block0_control ? "successor_block0_negative_control" : "bf16_shared_workspace_repeat",
        reference: {path: reference.path, sha256: reference.sha256,
                    scope: "authenticated saved actual two-block upstream MPS captures; no fresh upstream block rerun"},
        conditioning: {path: conditioning.path, sha256: conditioning.sha256},
        sources: sources.each_with_index.map { |source, index|
          pin = Cache.block_pin(index.to_i32)
          {block_index: index, path: File.realpath(source[:path]), sha256: source[:sha256],
           source_sha256: pin.file_sha256, source_manifest_sha256: pin.manifest_sha256}
        },
        oracles: oracle_sources.each_with_index.map { |source, index|
          {block_index: index, path: File.realpath(source[:path]), sha256: source[:sha256]}
        },
        geometry: {batches: 1, video_tokens: 6, text_tokens: 23, dim: 2048,
                   heads: 16, head_dim: 128, ffn_dim: 8192},
        precision: {native: {parameter_storage: "BF16", compute: "F32", attention: "MPS F32"},
                    upstream_reference: reference.document["precision"]},
        tolerance: {atol: Serial::ATOL, rtol: Serial::RTOL},
        resource_budget: active.named_bytes_report,
        source_conversion_dispatches: active.weight_conversion_dispatches,
        weights: weight_results, control_comparisons: active.control_results,
        shared_comparisons: active.shared_results, request_counters: active.request_metrics,
        owner_released_after_terminal: active.released? && !active.rooted?,
        negative_control_detected: successor_block0_control && passed,
        scope: "two real pretrained blocks; GPU hidden-state handoff, shared scratch/workspace; setup byte I/O and postterminal diagnostics only on CPU; no full-J, speed or generation claim",
      }
      temporary = File.tempfile("hy-nomas-bf16-serial-report-", ".partial", dir: File.dirname(report_path))
      begin
        temporary << report.to_pretty_json << '\n'
        temporary.fsync
        temporary.close
        File.link(temporary.path, report_path)
      ensure
        temporary.close unless temporary.closed?
        File.delete(temporary.path) if File.exists?(temporary.path)
      end
      puts passed ? "PASS: BF16 source-cache two-block replay" : "FAIL: BF16 source-cache two-block replay"
      passed
    ensure
      if active = owner
        active.drain_active_chain_after_error!
        active.cleanup_after_terminal!
      end
      cache_files.each(&.close)
      oracle_files.each(&.close)
    end
  end
end
