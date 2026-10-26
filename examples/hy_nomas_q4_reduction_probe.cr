require "./support/hy_nomas_q4_reduction_probe"

report_path = ""
max_named_bytes = 384_i64 * 1024 * 1024
OptionParser.parse do |parser|
  parser.banner = "Bounded direct-Q4 decoder/reduction diagnostic; no generation."
  parser.on("--report PATH", "Fresh report under the dedicated external reports directory") { |v| report_path = v }
  parser.on("--max-named-bytes N", "Named GPU storage admission cap; default 384 MiB") { |v| max_named_bytes = v.to_i64 }
end
unless ARGV.empty? && !report_path.empty? && max_named_bytes > 0 && max_named_bytes <= 1024_i64 * 1024 * 1024
  raise ArgumentError.new("Expected report and 0 < named cap <= 1 GiB")
end
report_path = HyNomasQ4ReductionProbe.validate_report_path!(report_path)
raise ArgumentError.new("Refusing to replace existing packed sidecar") if File.info?(report_path + ".f32le", false)
previous_report_path = File.join(HyNomasQ4ReductionProbe::GROUP_ROOT, "group-v2.json")
previous_report_raw = File.read(previous_report_path)
previous_report_sha = Digest::SHA256.hexdigest(previous_report_raw)
raise "Previous diagnostic identity changed" unless previous_report_sha == "3a239292011d2648c8d519ce9374b7c3dcf5199dc24ea6f8bc53078387768b48"
previous_report = JSON.parse(previous_report_raw)

fixture = HyNomasQ4ReductionProbe::Fixture.new(HyNomasQ4ReductionProbe::FIXTURE_PATH)
cache = ML::Video::HyNomasQ4Cache::Block.new(
  HyNomasQ4ReductionProbe::CACHE_MANIFEST,
  HyNomasQ4ReductionProbe::CACHE_MANIFEST_SHA256,
  HyNomasQ4ReductionProbe::SOURCE_SHA256,
  ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
    text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192),
  HyNomasQ4ReductionProbe::BLOCK_INDEX)
raise ArgumentError.new("Unexpected Q4 cache source size") unless cache.source_bytes == HyNomasQ4ReductionProbe::SOURCE_BYTES
raise ArgumentError.new("Unexpected Q4 cache payload identity") unless cache.payload_bytes == HyNomasQ4ReductionProbe::CACHE_PAYLOAD_BYTES && cache.payload_sha256 == HyNomasQ4ReductionProbe::CACHE_PAYLOAD_SHA256
HyNomasQ4ReductionProbe.verify_cache_tensor!(cache,
  "double_blocks.0.txt_mlp.fc2.weight", [2048_i64, 8192_i64],
  "q4_k_8range", HyNomasQ4ReductionProbe::Q4_WEIGHT_OFFSET,
  HyNomasQ4ReductionProbe::Q4_WEIGHT_BYTES)
HyNomasQ4ReductionProbe.verify_cache_tensor!(cache,
  "double_blocks.0.txt_mlp.fc2.bias", [2048_i64],
  "bf16-le", HyNomasQ4ReductionProbe::BIAS_OFFSET,
  HyNomasQ4ReductionProbe::BIAS_BYTES)
resource_plan, required_named_bytes = HyNomasQ4ReductionProbe.planned_named_bytes
raise ArgumentError.new("Direct-Q4 named residency exceeds caller cap before Metal allocation") if required_named_bytes > max_named_bytes

owner = nil.as(HyNomasQ4ReductionProbe::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
setup_started = Time.instant
begin
  owner = HyNomasQ4ReductionProbe::Owner.new(cache, fixture.input_values)
  setup_seconds = (Time.instant - setup_started).total_seconds
  raise "Resource admission mismatch" unless owner.resource_bytes == resource_plan && owner.named_bytes == required_named_bytes
  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  reads_before = ML::MetalBuffer.hy_text_probe_readbacks
  encode_started = Time.instant
  owner.encode(chain)
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
  raise "Encode allocated/read/waited" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
  chain.commit_and_wait
  packed = owner.values(chain)
  terminal_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
  raise "Expected one packed terminal readback" unless terminal_reads == 1
  fixture.unchanged!

  outputs = {} of String => Array(Float32)
  hashes = {} of String => String
  HyNomasQ4ReductionProbe::OUTPUT_VARIANTS.each_with_index do |name, i|
    values = HyNomasQ4ReductionProbe.extract_output(packed, i.to_i32)
    raise "Nonfinite output: #{name}" unless values.all?(&.finite?)
    outputs[name] = values
    hashes[name] = Digest::SHA256.hexdigest(values.to_unsafe.as(UInt8*).to_slice(values.size * sizeof(Float32)))
  end
  stats = packed[HyNomasQ4ReductionProbe::OUTPUT_VARIANTS.size * HyNomasQ4ReductionProbe::OUTPUT_ELEMENTS,
    HyNomasQ4ReductionProbe::STATS_WORDS].map(&.unsafe_as(UInt32))
  raise "Decoder comparator controls failed" unless [0, 10, 20].all? { |i| stats[i + 8] == 65_536 && stats[i + 9] == 0x51443431 } &&
                                                    stats[10] == 0 && stats[20] == 1 && stats[23] == 17 && stats[24] == 1 && stats[25] == 0
  previous_report["output_sha256"].as_h.each do |name, sha|
    raise "Baseline output changed: #{name}" unless hashes[name] == sha.as_s
  end
  grouping_coverage = HyNomasQ4ReductionProbe::GROUP_CASES.map_with_index do |c, i|
    offset = HyNomasQ4ReductionProbe::DECODE_STATS_WORDS + i * HyNomasQ4ReductionProbe::GROUP_TILE_COUNT
    visits = stats[offset, HyNomasQ4ReductionProbe::GROUP_TILE_COUNT]
    raise "Full-K tile coverage failed: #{c[:name]}" unless visits.all? { |v| v == 1 }
    {name: c[:name], groups: c[:groups], interleaved_32_tiles: c[:interleaved],
     tiles_per_partition: HyNomasQ4ReductionProbe::GROUP_TILE_COUNT // c[:groups],
     partition_start_step_k: c[:interleaved] ? 32 : HyNomasQ4ReductionProbe::INPUT_DIM // c[:groups],
     within_partition_step_k: c[:interleaved] ? c[:groups] * 32 : 32,
     visited_tiles: visits.size, every_tile_once: true}
  end
  packed_grouping_coverage = HyNomasQ4ReductionProbe::PACKED_GROUPS.map_with_index do |groups, i|
    offset = HyNomasQ4ReductionProbe::DECODE_STATS_WORDS + HyNomasQ4ReductionProbe::GROUP_COVERAGE_WORDS +
             i * 2 * HyNomasQ4ReductionProbe::GROUP_TILE_COUNT
    visits = stats[offset, HyNomasQ4ReductionProbe::GROUP_TILE_COUNT]
    owners = stats[offset + HyNomasQ4ReductionProbe::GROUP_TILE_COUNT, HyNomasQ4ReductionProbe::GROUP_TILE_COUNT]
    tiles_per_partition = HyNomasQ4ReductionProbe::GROUP_TILE_COUNT // groups
    raise "Packed K tile count failed: G#{groups}" unless visits.all? { |v| v == 1 }
    unless owners.each_with_index.all? { |owner_mask, tile| owner_mask == (1_u32 << (tile // tiles_per_partition)) }
      raise "Packed K partition ownership failed: G#{groups}"
    end
    {name: "contiguous.g#{groups}", groups: groups, tiles_per_partition: tiles_per_partition,
     visited_tiles: visits.size, every_tile_once: true, partition_ownership_verified: true}
  end
  comparisons = [] of HyNomasQ4ReductionProbe::Comparison
  [["direct.raw", "dense.raw"], ["direct.bias", "dense.bias"],
   ["mps.canonical.raw", "mps.helper.raw"], ["mps.canonical.bias", "mps.helper.bias"],
   ["direct.raw", "mps.helper.raw"], ["direct.bias", "mps.helper.bias"],
   ["direct.raw", "compensated.raw"], ["mps.helper.raw", "compensated.raw"],
   ["direct.bias", "compensated.bias"], ["mps.helper.bias", "compensated.bias"]].each do |pair|
    comparisons << HyNomasQ4ReductionProbe.comparison("#{pair[0]}.vs.#{pair[1]}", outputs[pair[0]], outputs[pair[1]])
  end
  prefix_comparisons = HyNomasQ4ReductionProbe::PREFIXES.map do |k|
    HyNomasQ4ReductionProbe.comparison("dense.k#{k}.vs.mps.k#{k}", outputs["dense.k#{k}"], outputs["mps.k#{k}"])
  end
  grouping_comparisons = HyNomasQ4ReductionProbe::GROUP_OUTPUTS.map do |name|
    reference = name.ends_with?(".raw") ? "mps.helper.raw" : "mps.helper.bias"
    HyNomasQ4ReductionProbe.comparison("#{name}.vs.#{reference}", outputs[name], outputs[reference])
  end
  merge_comparisons = HyNomasQ4ReductionProbe::GROUP_CASES.flat_map do |c|
    ["raw", "bias"].map do |suffix|
      left = "group.#{c[:name]}.fold.#{suffix}"
      right = "group.#{c[:name]}.tree.#{suffix}"
      HyNomasQ4ReductionProbe.comparison("#{left}.vs.#{right}", outputs[left], outputs[right])
    end
  end
  HyNomasQ4ReductionProbe::GROUP_CASES.select { |c| c[:groups] == 1 }.each do |c|
    HyNomasQ4ReductionProbe::GROUP_MERGES.each do |merge|
      ["raw", "bias"].each do |suffix|
        control = HyNomasQ4ReductionProbe.comparison("one-partition-control",
          outputs["group.#{c[:name]}.#{merge}.#{suffix}"], outputs["dense.#{suffix}"])
        raise "One-partition SIMD control changed" unless control[:raw_bit_differences] == 0
      end
    end
  end
  grouping_upstream_lossy = HyNomasQ4ReductionProbe::GROUP_OUTPUTS.select(&.ends_with?(".bias")).map do |name|
    HyNomasQ4ReductionProbe.comparison("#{name}.vs.original_bf16_saved_upstream", outputs[name], fixture.upstream_output)
  end
  packed_grouping_comparisons = HyNomasQ4ReductionProbe::PACKED_GROUPS.flat_map do |groups|
    ["raw", "bias"].flat_map do |suffix|
      name = "packed.contiguous.g#{groups}.#{suffix}"
      ["group.contiguous.g#{groups}.fold.#{suffix}", "mps.helper.#{suffix}"].map do |reference|
        HyNomasQ4ReductionProbe.comparison("#{name}.vs.#{reference}", outputs[name], outputs[reference])
      end
    end
  end
  packed_upstream_lossy = HyNomasQ4ReductionProbe::PACKED_OUTPUTS.select(&.ends_with?(".bias")).map do |name|
    HyNomasQ4ReductionProbe.comparison("#{name}.vs.original_bf16_saved_upstream", outputs[name], fixture.upstream_output)
  end
  packed_g1_exact = ["raw", "bias"].all? { |suffix| hashes["packed.contiguous.g1.#{suffix}"] == hashes["direct.#{suffix}"] }
  packed_g2_dense_exact = ["raw", "bias"].all? { |suffix| hashes["packed.contiguous.g2.#{suffix}"] == hashes["group.contiguous.g2.fold.#{suffix}"] }
  packed_g2_mps_passed = packed_grouping_comparisons.select { |c| c[:name].starts_with?("packed.contiguous.g2.") && c[:name].includes?(".vs.mps.helper.") }.all? { |c| c[:gate_passed] }
  raise "Packed one-partition control changed" unless packed_g1_exact
  raise "Dense clone did not reproduce actual Q4 GEMM" unless comparisons[0][:raw_bit_differences] == 0 && comparisons[1][:raw_bit_differences] == 0
  decode_exact = stats[0] == 0 && stats[1] == 0 && stats[2] == 0
  mismatches = [] of NamedTuple(index: Int32, row: Int32, col: Int32, direct: Float32, mps: Float32,
    compensated: Float32, raw_direct: Float32, raw_mps: Float32, raw_compensated: Float32,
    tolerance: Float64, direct_abs_error_to_compensated: Float64, mps_abs_error_to_compensated: Float64)
  outputs["direct.bias"].each_with_index do |value, i|
    target = outputs["mps.helper.bias"][i]
    tolerance = HyNomasQ4ReductionProbe::ATOL.to_f64 + HyNomasQ4ReductionProbe::RTOL.to_f64 * target.to_f64.abs
    next unless (value.to_f64 - target.to_f64).abs > tolerance
    reference = outputs["compensated.bias"][i]
    mismatches << {index: i, row: i // HyNomasQ4ReductionProbe::OUTPUT_DIM, col: i % HyNomasQ4ReductionProbe::OUTPUT_DIM,
                   direct: value, mps: target, compensated: reference,
                   raw_direct: outputs["direct.raw"][i], raw_mps: outputs["mps.helper.raw"][i], raw_compensated: outputs["compensated.raw"][i],
                   tolerance: tolerance, direct_abs_error_to_compensated: (value.to_f64 - reference).abs,
                   mps_abs_error_to_compensated: (target.to_f64 - reference).abs}
  end
  raw_mismatch_indices = [] of Int32
  outputs["direct.raw"].each_with_index do |v, i|
    target = outputs["mps.helper.raw"][i].to_f64
    raw_mismatch_indices << i if (v.to_f64 - target).abs > HyNomasQ4ReductionProbe::ATOL.to_f64 + HyNomasQ4ReductionProbe::RTOL.to_f64 * target.abs
  end
  readback_bytes = packed.to_unsafe.as(UInt8*).to_slice(packed.size * sizeof(Float32))
  readback_path = report_path + ".f32le"
  saved_upstream = HyNomasQ4ReductionProbe.comparison("direct.bias.vs.original_bf16_saved_upstream", outputs["direct.bias"], fixture.upstream_output)
  result = {
    schema: "hy-nomas-q4-decoder-reduction-diagnostic-v3", schema_version: 3,
    scope: "One authenticated M23/K8192/N2048 text FF down projection; no block, diffusion or video generation",
    fixture: {path: HyNomasQ4ReductionProbe::FIXTURE_PATH, sha256: fixture.sha256,
              sidecar_sha256: fixture.sidecar_sha256, upstream_revision: HyNomasQ4ReductionProbe::UPSTREAM_REVISION},
    cache: {manifest: HyNomasQ4ReductionProbe::CACHE_MANIFEST, manifest_sha256: cache.manifest_sha256,
            source_sha256: cache.source_sha256, payload_sha256: cache.payload_sha256, payload_bytes: cache.payload_bytes,
            selected_weight_offset: HyNomasQ4ReductionProbe::Q4_WEIGHT_OFFSET, selected_bias_offset: HyNomasQ4ReductionProbe::BIAS_OFFSET},
    kernel_sources: {qwen_sha256: Digest::SHA256.hexdigest(ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE),
                     derived_sha256: Digest::SHA256.hexdigest(HyNomasQ4ReductionProbe.dense_source),
                     grouped_sha256: Digest::SHA256.hexdigest(HyNomasQ4ReductionProbe.grouped_source),
                     packed_grouped_sha256: Digest::SHA256.hexdigest(HyNomasQ4ReductionProbe.packed_grouped_source),
                     diagnostic_sha256: Digest::SHA256.hexdigest(HyNomasQ4ReductionProbe::DIAGNOSTIC_SOURCE)},
    previous_report: {path: previous_report_path, sha256: previous_report_sha,
                      common_output_hashes_verified: previous_report["output_sha256"].as_h.size},
    tolerance: {atol: HyNomasQ4ReductionProbe::ATOL, rtol: HyNomasQ4ReductionProbe::RTOL},
    decode: {elements: 16_777_216, bit_differences: stats[0], nonfinite_canonical: stats[1], nonfinite_helper: stats[2],
             first_difference: stats[0] == 0 ? nil : stats[3], low_nibble_differences: stats[4], high_nibble_differences: stats[5],
             self_control_differences: stats[10], injected_single_bit_differences: stats[20], compared_blocks: stats[8],
             all_coefficients_bitwise_equal: decode_exact},
    execution: {terminal_wait_count: chain.wait_count, command_segments: chain.segment_count,
                encode_named_allocations: encode_allocations, preterminal_readbacks: early_reads, terminal_packed_readbacks: terminal_reads,
                named_bytes: owner.named_bytes, named_buffers: owner.resource_bytes, setup_seconds: setup_seconds,
                encode_seconds: encode_seconds, combined_gpu_seconds: chain.gpu_elapsed_seconds,
                combined_timing_is_not_a_kernel_benchmark: true, neural_math_and_reconstruction_on_gpu: true},
    comparisons: comparisons, prefix_comparisons: prefix_comparisons,
    grouping_coverage: grouping_coverage, grouping_comparisons: grouping_comparisons,
    fold_vs_tree_comparisons: merge_comparisons, grouping_original_bf16_lossy_comparisons: grouping_upstream_lossy,
    packed_grouping_coverage: packed_grouping_coverage, packed_grouping_comparisons: packed_grouping_comparisons,
    packed_original_bf16_lossy_comparisons: packed_upstream_lossy,
    failures_after_bias: mismatches,
    failures_before_bias_indices: raw_mismatch_indices,
    packed_readback: {path: readback_path, sha256: Digest::SHA256.hexdigest(readback_bytes), nbytes: readback_bytes.size,
                      output_variants: HyNomasQ4ReductionProbe::OUTPUT_VARIANTS, elements_per_output: HyNomasQ4ReductionProbe::OUTPUT_ELEMENTS,
                      stats_words: HyNomasQ4ReductionProbe::STATS_WORDS,
                      decode_stats_words: HyNomasQ4ReductionProbe::DECODE_STATS_WORDS,
                      encoding: "little-endian F32 banks then U32 decoder words, dense K32 visits, packed K32 visits and owner masks"},
    original_bf16_upstream_lossy_comparison: saved_upstream, output_sha256: hashes,
    summary: {diagnostic_completed: true, dense_clone_bitwise_reproduces_actual_q4: true,
              decoder_bitwise_equal: decode_exact, original_17_failures_reproduced: mismatches.size == 17,
              direct_mps_same_f16_operand_gate_passed: comparisons[5][:gate_passed],
              packed_g1_bitwise_reproduces_direct: packed_g1_exact,
              packed_g2_bitwise_reproduces_dense: packed_g2_dense_exact,
              packed_g2_mps_same_f16_operand_gate_passed: packed_g2_mps_passed,
              compensated_is_diagnostic_not_correct_rounding_proof: true},
  }
  owner.release_after_terminal!(chain)
  chain.release
  owner = nil
  chain = nil
  HyNomasQ4ReductionProbe.write_new_artifact!(readback_path, String.new(readback_bytes))
  HyNomasQ4ReductionProbe.write_new_artifact!(report_path, result.to_pretty_json + "\n")
  puts({report: report_path, decoder_equal: decode_exact, direct_mps_failures: mismatches.size, named_bytes: required_named_bytes}.to_json)
ensure
  if active_chain = chain
    unless active_chain.completed?
      begin
        active_chain.commit_and_wait
      rescue ex
        STDERR.puts "Terminal cleanup failed; retain resources until exit: #{ex.message}"
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
