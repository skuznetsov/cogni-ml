require "./support/hy_nomas_q4_paired_benchmark"

alias B = HyNomasQ4PairedBenchmark
alias P = HyNomasQ4ReductionProbe

report_path = ""
rows = 23
repeats = 16
reverse_first = false
max_named_bytes = 768_i64 * 1024 * 1024
OptionParser.parse do |parser|
  parser.banner = "Bounded paired Q4/MPS projection benchmark; no generation."
  parser.on("--report PATH", "Fresh report in the dedicated external directory") { |v| report_path = v }
  parser.on("--rows N", "23, 32 or 1280; larger shapes repeat the real 23 rows on GPU") { |v| rows = v.to_i32 }
  parser.on("--repeats N", "1..32 complete projections per timed root, default 16") { |v| repeats = v.to_i32 }
  parser.on("--reverse-first", "Reverse the first order, retaining balanced forward/reverse rounds") { reverse_first = true }
  parser.on("--max-named-bytes N", "Named storage admission cap, default 768 MiB") { |v| max_named_bytes = v.to_i64 }
end
unless ARGV.empty? && !report_path.empty? && [23, 32, 1280].includes?(rows) && (1..32).includes?(repeats) &&
       max_named_bytes > 0 && max_named_bytes <= 1024_i64 * 1024 * 1024
  raise ArgumentError.new("Invalid paired benchmark report, geometry, repetitions or named cap")
end
report_path = B.validate_report_path!(report_path)
baseline_raw = File.read(B::BASELINE)
raise "Paired baseline digest changed" unless Digest::SHA256.hexdigest(baseline_raw) == B::BASELINE_SHA
baseline = JSON.parse(baseline_raw)
fixture = P::Fixture.new(P::FIXTURE_PATH)
cache = ML::Video::HyNomasQ4Cache::Block.new(P::CACHE_MANIFEST, P::CACHE_MANIFEST_SHA256, P::SOURCE_SHA256,
  ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280, text_tokens: 1280, dim: 2048,
    heads: 16, head_dim: 128, ffn_dim: 8192), P::BLOCK_INDEX)
unless cache.source_bytes == P::SOURCE_BYTES && cache.payload_bytes == P::CACHE_PAYLOAD_BYTES && cache.payload_sha256 == P::CACHE_PAYLOAD_SHA256
  raise "Paired Q4 cache identity changed"
end
P.verify_cache_tensor!(cache, "double_blocks.0.txt_mlp.fc2.weight", [2048_i64, 8192_i64], "q4_k_8range", P::Q4_WEIGHT_OFFSET, P::Q4_WEIGHT_BYTES)
P.verify_cache_tensor!(cache, "double_blocks.0.txt_mlp.fc2.bias", [2048_i64], "bf16-le", P::BIAS_OFFSET, P::BIAS_BYTES)
resource_plan, named_bytes = B.planned_named_bytes(rows)
raise ArgumentError.new("Paired named storage exceeds caller cap before Metal allocation") if named_bytes > max_named_bytes
schedule = B.schedule(reverse_first)
output_elements = rows * P::OUTPUT_DIM
samples = [] of NamedTuple(round: Int32, position: Int32, arm: String, warmup: Bool, reverse: Bool,
  root_sequence: Int32, encode_seconds: Float64)
owner = nil.as(B::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
setup_started = Time.instant
begin
  owner = B::Owner.new(cache, rows, fixture.input_values)
  setup_seconds = (Time.instant - setup_started).total_seconds
  raise "Paired resource plan mismatch" unless owner.resource_bytes == resource_plan && owner.named_bytes == named_bytes
  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  reads_before = ML::MetalBuffer.hy_text_probe_readbacks
  encode_started = Time.instant
  owner.encode_setup(chain)
  schedule.each_with_index do |sample, i|
    owner.prepare_sample(chain)
    B.continue_async(chain)
    sequence = chain.observed_root_count - 1
    started = Time.instant
    owner.encode_sample(chain, sample[:arm], repeats)
    encode_seconds = (Time.instant - started).total_seconds
    raise "Timed projection changed its root" unless chain.observed_root_count == sequence + 1
    samples << sample.merge({root_sequence: sequence, encode_seconds: encode_seconds})
    B.continue_async(chain)
    owner.check_sample(chain, sample[:arm], i)
  end
  owner.check_sample(chain, schedule.last[:arm], B::SAMPLE_COUNT, injected: true)
  owner.encode_pack(chain)
  total_encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
  raise "Paired encode allocated, read back or waited" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
  terminal_started = Time.instant
  chain.commit_and_wait
  terminal_wait_seconds = (Time.instant - terminal_started).total_seconds
  roots = chain.completed_root_timings
  raise "Paired root schedule changed" unless roots.size == 1 + 2 * B::SAMPLE_COUNT
  # A command-root interval is not isolated if another root overlaps it.
  # Keep the complete timestamps so the postterminal checker can recompute it.
  root_profile = ML::Video::MPSCommandPhaseProfile.build(roots,
    [ML::Video::MPSCommandPhaseRange.new("complete_diagnostic", 0, roots.size)], chain.gpu_elapsed_seconds)
  timing_overlap_seconds = samples.map do |sample|
    timed = roots[sample[:root_sequence]]
    covered_until = timed.gpu_start_seconds
    overlap = 0.0
    roots.reject { |r| r.sequence == timed.sequence }.sort_by(&.gpu_start_seconds).each do |other|
      left = Math.max(covered_until, Math.max(timed.gpu_start_seconds, other.gpu_start_seconds))
      right = Math.min(timed.gpu_end_seconds, other.gpu_end_seconds)
      if right > left
        overlap += right - left
        covered_until = right
      end
    end
    overlap
  end
  raise "Timed GPU roots overlap neighboring work" unless timing_overlap_seconds.all? { |v| v.finite? && v == 0.0 }
  packed = owner.values(chain)
  raise "Paired benchmark needs one terminal packed readback" unless ML::MetalBuffer.hy_text_probe_readbacks - reads_before == 1
  fixture.unchanged!

  bank_hashes = {} of String => String
  outputs = {} of String => Array(Float32)
  B::BANKS.each_with_index do |name, i|
    values = packed[i * output_elements, output_elements]
    raise "Nonfinite paired output #{name}" unless values.all?(&.finite?)
    raise "Constant paired output #{name}" unless values.any? { |v| v != values.first }
    outputs[name] = values
    bank_hashes[name] = Digest::SHA256.hexdigest(values.to_unsafe.as(UInt8*).to_slice(values.size * 4))
  end
  baseline_map = {
    "direct" => "direct", "dense.g2" => "group.contiguous.g2.fold",
    "packed.g1" => "packed.contiguous.g1", "packed.g2" => "packed.contiguous.g2",
    "mps.resident" => "mps.helper", "mps.expand" => "mps.helper",
  }
  if rows == 23
    baseline_map.each do |name, previous|
      ["raw", "bias"].each do |suffix|
        raise "Paired M23 baseline changed: #{name}.#{suffix}" unless bank_hashes["#{name}.#{suffix}"] == baseline["output_sha256"]["#{previous}.#{suffix}"].as_s
      end
    end
  end
  stats = packed[B::BANKS.size * output_elements, B::STATS_WORDS].map(&.unsafe_as(UInt32))
  metrics = (2 + B::SAMPLE_COUNT + 1).times.map { |i| stats[i * B::METRIC_WORDS, B::METRIC_WORDS] }.to_a
  metrics.each_with_index do |words, i|
    expected_differences = i == 1 || i == metrics.size - 1 ? 1_u32 : 0_u32
    count = i < 2 ? P::INPUT_DIM * P::OUTPUT_DIM : output_elements
    unless words == [expected_differences, 0_u32, ((count + 255) // 256).to_u32, B::MAGIC]
      raise "Paired GPU comparator/positive control failed at metric #{i}: #{words}"
    end
  end
  pairs = [["packed.g1", "direct"], ["packed.g2", "dense.g2"], ["mps.expand", "mps.resident"],
           ["packed.g1", "mps.resident"], ["packed.g2", "mps.resident"]]
  comparisons = pairs.flat_map do |pair|
    ["raw", "bias"].map { |suffix| P.comparison("#{pair[0]}.#{suffix}.vs.#{pair[1]}.#{suffix}", outputs["#{pair[0]}.#{suffix}"], outputs["#{pair[1]}.#{suffix}"]) }
  end
  upstream = Array(Float32).new(output_elements) { |i| fixture.upstream_output[i % P::OUTPUT_ELEMENTS] }
  lossy = ["packed.g1", "packed.g2", "mps.resident"].map do |arm|
    P.comparison("#{arm}.bias.vs.original_bf16_saved_upstream_rows", outputs["#{arm}.bias"], upstream)
  end
  rows_timing = samples.map do |sample|
    root = roots[sample[:root_sequence]]
    duration = root.gpu_end_seconds - root.gpu_start_seconds
    unless root.completed && !root.has_error && root.gpu_start_seconds.finite? && root.gpu_start_seconds > 0 && duration.finite? && duration > 0
      raise "Invalid isolated GPU timing root"
    end
    sample.merge({gpu_start_seconds: root.gpu_start_seconds, gpu_end_seconds: root.gpu_end_seconds,
                  gpu_seconds: duration, gpu_seconds_per_call: duration / repeats})
  end
  summaries = B::ARMS.map do |arm|
    measured = rows_timing.select { |s| s[:arm] == arm && !s[:warmup] }
    times = measured.map { |s| s[:gpu_seconds_per_call] }
    {arm: arm, samples: times.size, median_seconds: B.median(times), min_seconds: times.min, max_seconds: times.max,
     forward_median_seconds: B.median(measured.reject { |s| s[:reverse] }.map { |s| s[:gpu_seconds_per_call] }),
     reverse_median_seconds: B.median(measured.select { |s| s[:reverse] }.map { |s| s[:gpu_seconds_per_call] })}
  end
  raw_bytes = packed.to_unsafe.as(UInt8*).to_slice(packed.size * 4)
  result = {
    schema: "hy-nomas-q4-paired-projection-v1", schema_version: 1,
    scope: "one pretrained text FFN down projection; repeated authenticated rows at larger M; no generation",
    geometry: {m: rows, k: P::INPUT_DIM, n: P::OUTPUT_DIM, original_rows: P::ROWS,
               larger_m_input: "GPU cyclic repetition of the 23 saved real rows; not a new full-prefix activation distribution"},
    fixture: {path: P::FIXTURE_PATH, sha256: fixture.sha256, sidecar_sha256: fixture.sidecar_sha256, upstream_revision: P::UPSTREAM_REVISION},
    cache: {manifest: P::CACHE_MANIFEST, manifest_sha256: P::CACHE_MANIFEST_SHA256, payload_sha256: cache.payload_sha256,
            weight_offset: P::Q4_WEIGHT_OFFSET, weight_bytes: P::Q4_WEIGHT_BYTES, bias_offset: P::BIAS_OFFSET},
    baseline: {path: B::BASELINE, sha256: B::BASELINE_SHA, m23_hashes_checked: rows == 23 ? B::BANKS.size : 0},
    kernel_sources: {qwen_sha256: Digest::SHA256.hexdigest(ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE),
                     benchmark_sha256: Digest::SHA256.hexdigest(B.packed_source), diagnostic_sha256: Digest::SHA256.hexdigest(P::DIAGNOSTIC_SOURCE)},
    tolerance: {atol: P::ATOL, rtol: P::RTOL},
    benchmark: {repeats_per_root: repeats, warmup_rounds: B::WARMUP_ROUNDS, measured_rounds: B::MEASURED_ROUNDS,
                reverse_first: reverse_first, arms: B::ARMS, timing: "GPU command-root interval divided by complete projection repetitions",
                timed_packed: "tile decode + SIMD GEMM + F32 merge + in-place BF16 bias",
                timed_mps_resident: "prepared resident F32 matrix MPS projection + in-place BF16 bias",
                timed_mps_expand: "Q4 helper F16 decode widened to F32 + MPS projection + in-place BF16 bias",
                excluded: "setup/input preparation/qualification/poisoning/comparators/packing; no profiler enabled",
                one_time_resident_f32_matrix_bytes: P::INPUT_DIM.to_i64 * P::OUTPUT_DIM * 4,
                timed_root_overlap_seconds: timing_overlap_seconds,
                samples: rows_timing, summaries: summaries},
    execution: {terminal_waits: chain.wait_count, preterminal_readbacks: early_reads, terminal_packed_readbacks: 1,
                encode_named_allocations: encode_allocations, command_segments: chain.segment_count, named_bytes: named_bytes,
                named_buffers: resource_plan, setup_seconds: setup_seconds, encode_seconds: total_encode_seconds,
                terminal_wait_seconds: terminal_wait_seconds, combined_gpu_root_seconds: chain.gpu_elapsed_seconds,
                root_union_seconds: root_profile.union_seconds,
                root_timings: roots.map { |r| {sequence: r.sequence, completed: r.completed, has_error: r.has_error,
                                               gpu_start_seconds: r.gpu_start_seconds, gpu_end_seconds: r.gpu_end_seconds} },
                all_math_and_input_replication_on_gpu: true, diagnostic_async_boundaries: 2 * B::SAMPLE_COUNT},
    gpu_metrics: {weight_and_injected: metrics[0, 2], sample_repeatability: metrics[2, B::SAMPLE_COUNT], output_injected: metrics.last},
    comparisons: comparisons, original_bf16_lossy_comparisons: lossy, output_sha256: bank_hashes,
    packed_readback: {path: report_path + ".f32le", sha256: Digest::SHA256.hexdigest(raw_bytes), nbytes: raw_bytes.size,
                      banks: B::BANKS, elements_per_bank: output_elements, stats_words: B::STATS_WORDS,
                      encoding: "little-endian F32 output banks then U32 comparator words"},
    summary: {diagnostic_completed: true,
              packed_g2_same_operand_gate_passed: comparisons.select { |c| c[:name].starts_with?("packed.g2.") }.all? { |c| c[:gate_passed] },
              original_bf16_quality_gate_passed: lossy.all? { |c| c[:gate_passed] },
              production_promoted: false},
  }
  owner.release_after_terminal!(chain)
  chain.release
  owner = nil
  chain = nil
  P.write_new_artifact!(report_path + ".f32le", String.new(raw_bytes))
  P.write_new_artifact!(report_path, result.to_pretty_json + "\n")
  puts({report: report_path, rows: rows, named_bytes: named_bytes, summaries: summaries,
        packed_g2_gate: result[:summary][:packed_g2_same_operand_gate_passed]}.to_json)
ensure
  if active_chain = chain
    unless active_chain.completed?
      begin
        active_chain.commit_and_wait
      rescue ex
        STDERR.puts "Terminal cleanup failed; retain resources through process exit: #{ex.message}"
      end
    end
    if active_chain.completed?
      owner.try { |o| o.encoded? ? o.release_after_terminal!(active_chain) : o.release_setup_only! }
      active_chain.release
    end
  else
    owner.try(&.release_setup_only!)
  end
end
