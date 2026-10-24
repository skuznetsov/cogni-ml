# Bounded native-only benchmark for compact and canonical Hy-Nomas MPS attention.
# This measures the standalone attention path, not a full block or generation.
require "json"
require "option_parser"
require "../src/ml/video/hy_nomas_attention_compact"
require "../src/ml/video/hy_nomas_attention_mps"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private HY_COMPACT_BENCHMARK_SCHEMA      = "hy_nomas_compact_attention_benchmark_v1"
private HY_COMPACT_DIAGNOSTIC_SCHEMA     = "hy_nomas_compact_attention_diagnostic_v1"
private HY_COMPACT_FAILURE_SCHEMA        = "hy_nomas_compact_attention_failure_v1"
private HY_COMPACT_BATCHES               =    1
private HY_COMPACT_HEADS                 =   16
private HY_COMPACT_HEAD_DIM              =  128
private HY_COMPACT_MAX_TOKENS            = 6560
private HY_COMPACT_OPAQUE_RESERVE_BYTES  = 256_i64 * 1024 * 1024
private HY_COMPACT_PREFLIGHT_LIMIT_BYTES = 4_i64 * 1024 * 1024 * 1024
private HY_COMPACT_SCORE_TILE_MULTIPLIER = 3_i64
private HY_COMPACT_INPUT_FORMULA         = "avalanche32(i + salt*0x9e3779b9); ((hash & 65535) - 32768) / 65536"

private record HyCompactBenchmarkGeometry,
  max_score_elements : Int64,
  graph_calls : Int32,
  resident_bytes : Int64

private record HyCompactBenchmarkSpec,
  label : String,
  query_chunk : Int32,
  heads_per_group : Int32,
  input_bf16_boundary : Bool,
  backend : String,
  plan_type : String,
  geometry : HyCompactBenchmarkGeometry

private record HyCompactBenchmarkArm,
  spec : HyCompactBenchmarkSpec,
  output : ML::MetalBuffer,
  plan : ML::Video::HyNomasCompactAttentionMPSPlan | ML::Video::HyNomasAttentionMPSPlan,
  setup_plan_wall_seconds : Float64

private record HyCompactBenchmarkSample,
  arm : String,
  pair_repetition : Int32,
  pair_order : String,
  pair_position : Int32,
  encode_wall_seconds : Float64,
  total_wall_including_terminal_wait_seconds : Float64,
  gpu_elapsed_seconds : Float64

private def hy_compact_median(samples : Array(Float64)) : Float64
  raise ArgumentError.new("Cannot calculate median of an empty sample") if samples.empty?
  ordered = samples.sort
  middle = ordered.size // 2
  if ordered.size.odd?
    ordered[middle]
  else
    (ordered[middle - 1] + ordered[middle]) / 2.0
  end
end

private def hy_compact_emit(record) : Nil
  STDOUT.puts(record.to_json)
  STDOUT.flush
end

private HY_COMPACT_KERNELS = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline float hy_compact_input(uint i, uint salt) {
    uint mixed = i + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int signed_value = int(mixed & 65535u) - 32768;
    return float(signed_value) / 65536.0f;
  }

  kernel void hy_nomas_compact_benchmark_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &salt [[buffer(2)]], constant uint &bf16_boundary [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float value = hy_compact_input(i, salt);
    if (bf16_boundary != 0u) {
      uint bits = as_type<uint>(value);
      bits += 0x7fffu + ((bits >> 16) & 1u);
      value = as_type<float>(bits & 0xffff0000u);
    }
    output[i] = value;
  }

  kernel void hy_nomas_compact_benchmark_zero(
      device atomic_uint *metrics [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < 5u) atomic_store_explicit(metrics + i, 0u, memory_order_relaxed);
  }

  inline uint hy_compact_bf16_rne(float value) {
    const uint bits = as_type<uint>(value);
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
  }

  kernel void hy_nomas_compact_benchmark_compare(
      device const float *actual [[buffer(0)]],
      device const float *reference [[buffer(1)]],
      device atomic_uint *metrics [[buffer(2)]],
      constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float got = actual[i];
    const float ref = reference[i];
    if (!isfinite(got) || !isfinite(ref)) {
      atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
      return;
    }
    const float error = abs(got - ref);
    const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
    if (error > tolerance)
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
    if (hy_compact_bf16_rne(got) != hy_compact_bf16_rne(ref))
      atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(metrics + 3, as_type<uint>(error), memory_order_relaxed);
    atomic_fetch_max_explicit(metrics + 4, as_type<uint>(error / tolerance), memory_order_relaxed);
  }
METAL

private def hy_compact_measure_arm(arm : HyCompactBenchmarkArm, phase : String,
                                   pair_repetition : Int32, pair_order : String,
                                   pair_position : Int32, named_bytes : Int64,
                                   score_reserve_bytes : Int64,
                                   fill_setup_wall_seconds : Float64,
                                   retained_chains : Array(ML::Metal::MPSCommandChain))
  chain = ML::Metal::MPSCommandChain.new
  retained_chains << chain
  begin
    allocations_before = ML::MetalBuffer.video_spec_allocations
    arm_started = Time.instant
    arm.plan.encode(chain)
    encode_wall_seconds = (Time.instant - arm_started).total_seconds
    allocations_during_encode = ML::MetalBuffer.video_spec_allocations - allocations_before
    raise "attention encode allocated #{allocations_during_encode} tracked MetalBuffers" unless allocations_during_encode == 0

    chain.commit_and_wait
    total_wall_seconds = (Time.instant - arm_started).total_seconds
    raise "compact arm did not use exactly one terminal wait" unless chain.wait_count == 1
    raise "compact arm did not cross the segmented execution path" unless chain.segment_count > 0

    result = {
      schema:                                                HY_COMPACT_BENCHMARK_SCHEMA,
      phase:                                                 phase,
      status:                                                "ok",
      backend:                                               arm.spec.backend,
      plan_type:                                             arm.spec.plan_type,
      arm:                                                   arm.spec.label,
      pair_repetition:                                       pair_repetition,
      pair_order:                                            pair_order,
      pair_position:                                         pair_position,
      batches:                                               HY_COMPACT_BATCHES,
      tokens:                                                arm.plan.tokens,
      heads:                                                 arm.plan.heads,
      head_dim:                                              arm.plan.head_dim,
      query_chunk:                                           arm.spec.query_chunk,
      heads_per_group:                                       arm.spec.heads_per_group,
      input_layout:                                          "B,T,H,D",
      output_layout:                                         "B,T,H,D",
      input_formula:                                         HY_COMPACT_INPUT_FORMULA,
      input_salts:                                           {q: 1, k: 2, v: 3},
      input_bf16_boundary_roundtrip:                         arm.spec.input_bf16_boundary,
      kv_pack_passes_per_encode:                             arm.spec.plan_type == "compact" ? 1 : 0,
      canonical_full_qkv_transpose_passes_per_encode:        arm.spec.plan_type == "canonical_dense" ? 1 : 0,
      query_gather_and_output_scatter_inside_encode:         true,
      setup_plan_wall_seconds:                               arm.setup_plan_wall_seconds,
      input_fill_setup_wall_including_terminal_wait_seconds: fill_setup_wall_seconds,
      encode_wall_seconds:                                   encode_wall_seconds,
      total_wall_including_terminal_wait_seconds:            total_wall_seconds,
      gpu_elapsed_seconds:                                   chain.gpu_elapsed_seconds,
      segments:                                              chain.segment_count,
      terminal_waits:                                        chain.wait_count,
      graph_calls:                                           arm.spec.geometry.graph_calls,
      tracked_metal_buffer_allocations_during_encode:        allocations_during_encode,
      named_arrays_and_stages_bytes:                         named_bytes,
      stage_buffers_bytes_for_arm:                           arm.spec.geometry.resident_bytes,
      max_score_bytes_per_graph:                             arm.spec.geometry.max_score_elements * sizeof(Float32),
      three_score_tile_reserve_bytes:                        score_reserve_bytes,
      opaque_mpsgraph_reserve_heuristic_bytes:               HY_COMPACT_OPAQUE_RESERVE_BYTES,
      preflight_budget_bytes:                                HY_COMPACT_PREFLIGHT_LIMIT_BYTES,
      backend_private_allocation_cap_established:            false,
    }
    chain.release
    result
  ensure
    unless chain.completed?
      begin
        chain.commit_and_wait
      rescue ex
        hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "sample_cleanup",
                         arm: arm.spec.label, error: ex.message || ex.class.name})
      end
    end
    chain.release if chain.completed?
  end
end

private def hy_compact_diagnose(actual : ML::MetalBuffer, reference : ML::MetalBuffer,
                                metrics : ML::MetalBuffer, zero : ML::Metal::ComputePipeline,
                                compare : ML::Metal::ComputePipeline, count : Int32,
                                comparison : String, tokens : Int32,
                                query_chunk_a : Int32, group_a : Int32,
                                query_chunk_b : Int32?, group_b : Int32?,
                                retained_chains : Array(ML::Metal::MPSCommandChain))
  chain = ML::Metal::MPSCommandChain.new
  retained_chains << chain
  begin
    started = Time.instant
    chain.encoder.set_pipeline(zero)
      .set_buffer(metrics, 0, ML::Metal::BufferAccess::Write)
      .dispatch_1d(5)
    chain.encoder.set_pipeline(compare)
      .set_buffer(actual, 0).set_buffer(reference, 1)
      .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
      .set_value(count.to_u32, 3)
      .dispatch_1d(count)
    chain.commit_and_wait
    wall_seconds = (Time.instant - started).total_seconds
    raise "diagnostic did not use exactly one terminal wait" unless chain.wait_count == 1

    # The only diagnostic data copied to the CPU is this metrics array, after
    # the dedicated diagnostic chain has completed.
    values = metrics.contents.as(UInt32*)
    result = {
      schema:                                    HY_COMPACT_DIAGNOSTIC_SCHEMA,
      status:                                    values[0] == 0 && values[1] == 0 && values[2] == 0 ? "pass" : "fail",
      comparison:                                comparison,
      tokens:                                    tokens,
      query_chunk_a:                             query_chunk_a,
      heads_per_group_a:                         group_a,
      query_chunk_b:                             query_chunk_b,
      heads_per_group_b:                         group_b,
      compared_values:                           count,
      nonfinite_pairs:                           values[0],
      tolerance_violations:                      values[1],
      bf16_rne_bin_mismatches:                   values[2],
      max_absolute_error:                        values[3].unsafe_as(Float32),
      max_tolerance_ratio:                       values[4].unsafe_as(Float32),
      host_wall_including_terminal_wait_seconds: wall_seconds,
      gpu_elapsed_seconds:                       chain.gpu_elapsed_seconds,
      terminal_waits:                            chain.wait_count,
      tolerance:                                 "2e-5 + 2e-5*abs(reference)",
      bf16_rounding:                             "round-to-nearest-even",
      pinned_pytorch_2_8_oracle_in_this_run:     false,
    }
    chain.release
    result
  ensure
    unless chain.completed?
      begin
        chain.commit_and_wait
      rescue ex
        hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "diagnostic_cleanup",
                         error: ex.message || ex.class.name})
      end
    end
    chain.release if chain.completed?
  end
end

tokens = 6560
query_chunk = 128
heads_per_group = 16
repeats = 4
input_bf16_boundary = false
compare_dense = false
compare_query_chunk = nil.as(Int32?)
compare_heads_per_group = nil.as(Int32?)
arena = nil.as(ML::Video::Arena?)
arms = [] of HyCompactBenchmarkArm
retained_chains = [] of ML::Metal::MPSCommandChain
retained_commands = [] of ML::Metal::CommandBuffer
failure_message = nil.as(String?)

begin
  OptionParser.parse do |parser|
    parser.banner = "Usage: crystal run examples/hy_nomas_attention_benchmark.cr -- [--tokens N] [--query-chunk N] [--heads-per-group N] [--repeats N] [--compare-query-chunk N] [--compare-heads-per-group N] [--compare-dense]"
    parser.on("--tokens=N", "Token count (1..#{HY_COMPACT_MAX_TOKENS}, default 6560)") { |value| tokens = value.to_i }
    parser.on("--query-chunk=N", "Primary query chunk (default 128)") { |value| query_chunk = value.to_i }
    parser.on("--heads-per-group=N", "Primary heads per group (default 16)") { |value| heads_per_group = value.to_i }
    parser.on("--repeats=N", "Measured samples or paired repetitions (default 4)") { |value| repeats = value.to_i }
    parser.on("--input-bf16-boundary", "Round Q/K/V through BF16 RNE on GPU before timing") { input_bf16_boundary = true }
    parser.on("--compare-query-chunk=N", "Enable paired arm B with this query chunk") { |value| compare_query_chunk = value.to_i }
    parser.on("--compare-heads-per-group=N", "Enable paired arm B with this head-group size") { |value| compare_heads_per_group = value.to_i }
    parser.on("--compare-dense", "Pair compact arm A with canonical dense MPS arm B") { compare_dense = true }
  end

  if compare_dense && (!compare_query_chunk.nil? || !compare_heads_per_group.nil?)
    raise ArgumentError.new("--compare-dense is mutually exclusive with --compare-query-chunk and --compare-heads-per-group")
  end
  paired = compare_dense || !compare_query_chunk.nil? || !compare_heads_per_group.nil?
  unless tokens > 0 && tokens <= HY_COMPACT_MAX_TOKENS
    raise ArgumentError.new("--tokens must be between 1 and #{HY_COMPACT_MAX_TOKENS}")
  end
  unless query_chunk > 0 && query_chunk <= tokens
    raise ArgumentError.new("--query-chunk must be positive and no greater than --tokens")
  end
  unless heads_per_group > 0 && heads_per_group <= HY_COMPACT_HEADS
    raise ArgumentError.new("--heads-per-group must be between 1 and #{HY_COMPACT_HEADS}")
  end
  raise ArgumentError.new("--repeats must be positive") unless repeats > 0
  raise ArgumentError.new("paired comparisons require at least two repetitions") if paired && repeats < 2

  query_chunk_b = compare_dense ? tokens : (compare_query_chunk || query_chunk).not_nil!
  heads_per_group_b = compare_dense ? HY_COMPACT_HEADS : (compare_heads_per_group || heads_per_group).not_nil!
  unless query_chunk_b > 0 && query_chunk_b <= tokens
    raise ArgumentError.new("--compare-query-chunk must be positive and no greater than --tokens")
  end
  unless heads_per_group_b > 0 && heads_per_group_b <= HY_COMPACT_HEADS
    raise ArgumentError.new("--compare-heads-per-group must be between 1 and #{HY_COMPACT_HEADS}")
  end

  specs = [] of HyCompactBenchmarkSpec
  compact_geometry_a = ML::Video::HyNomasCompactAttentionMPSPlan.geometry(
    HY_COMPACT_BATCHES, tokens, HY_COMPACT_HEADS, HY_COMPACT_HEAD_DIM,
    query_chunk, heads_per_group)
  geometry_a = HyCompactBenchmarkGeometry.new(compact_geometry_a.max_score_elements,
    compact_geometry_a.graph_calls, compact_geometry_a.resident_bytes)
  specs << HyCompactBenchmarkSpec.new("A", query_chunk, heads_per_group,
    input_bf16_boundary, "native_compact_mpsgraph", "compact", geometry_a)
  if compare_dense
    # Canonical mode 3 uses four full-size B,H,T,D staging arrays and one graph.
    dense_count = ML::Video::HyNomasDualStreamBlock.checked_count(
      HY_COMPACT_BATCHES, tokens, HY_COMPACT_HEADS, HY_COMPACT_HEAD_DIM)
    dense_score_elements = ML::Video::HyNomasDualStreamBlock.checked_count(
      HY_COMPACT_HEADS, tokens, tokens)
    dense_resident_bytes = 4_i64 * dense_count * sizeof(Float32)
    geometry_b = HyCompactBenchmarkGeometry.new(dense_score_elements.to_i64,
      1, dense_resident_bytes)
    specs << HyCompactBenchmarkSpec.new("B", tokens, HY_COMPACT_HEADS,
      input_bf16_boundary, "native_canonical_mpsgraph", "canonical_dense", geometry_b)
  elsif paired
    compact_geometry_b = ML::Video::HyNomasCompactAttentionMPSPlan.geometry(
      HY_COMPACT_BATCHES, tokens, HY_COMPACT_HEADS, HY_COMPACT_HEAD_DIM,
      query_chunk_b, heads_per_group_b)
    geometry_b = HyCompactBenchmarkGeometry.new(compact_geometry_b.max_score_elements,
      compact_geometry_b.graph_calls, compact_geometry_b.resident_bytes)
    specs << HyCompactBenchmarkSpec.new("B", query_chunk_b, heads_per_group_b,
      input_bf16_boundary, "native_compact_mpsgraph", "compact", geometry_b)
  end

  values = HY_COMPACT_BATCHES.to_i64 * tokens * HY_COMPACT_HEADS * HY_COMPACT_HEAD_DIM
  arm_count = specs.size.to_i64
  named_arrays_bytes = (3_i64 * values + arm_count * values + 5_i64) * sizeof(Float32)
  stage_bytes = specs.sum(0_i64) { |spec| spec.geometry.resident_bytes }
  score_bytes_per_graph = specs.sum(0_i64) { |spec| spec.geometry.max_score_elements * sizeof(Float32) }
  score_reserve_bytes = HY_COMPACT_SCORE_TILE_MULTIPLIER * score_bytes_per_graph
  named_and_stage_bytes = named_arrays_bytes + stage_bytes
  estimated_total = named_and_stage_bytes + score_reserve_bytes + HY_COMPACT_OPAQUE_RESERVE_BYTES
  preflight = {
    schema:   "hy_nomas_compact_attention_preflight_v1",
    status:   estimated_total <= HY_COMPACT_PREFLIGHT_LIMIT_BYTES ? "geometry_fits_reserve_estimate" : "geometry_exceeds_reserve_estimate",
    batches:  HY_COMPACT_BATCHES,
    tokens:   tokens,
    heads:    HY_COMPACT_HEADS,
    head_dim: HY_COMPACT_HEAD_DIM,
    arms:     specs.map { |spec| {label: spec.label, query_chunk: spec.query_chunk,
                              heads_per_group: spec.heads_per_group,
                              backend: spec.backend, plan_type: spec.plan_type,
                              graph_calls: spec.geometry.graph_calls,
                              max_score_bytes_per_graph: spec.geometry.max_score_elements * sizeof(Float32)} },
    named_arrays_bytes:                           named_arrays_bytes,
    stage_buffers_bytes:                          stage_bytes,
    named_arrays_and_stages_bytes:                named_and_stage_bytes,
    score_tile_multiplier:                        HY_COMPACT_SCORE_TILE_MULTIPLIER,
    score_tile_reserve_bytes:                     score_reserve_bytes,
    opaque_mpsgraph_reserve_heuristic_bytes:      HY_COMPACT_OPAQUE_RESERVE_BYTES,
    estimated_total_bytes:                        estimated_total,
    fits_reserve_estimate:                        estimated_total <= HY_COMPACT_PREFLIGHT_LIMIT_BYTES,
    budget_bytes:                                 HY_COMPACT_PREFLIGHT_LIMIT_BYTES,
    external_memory_guard_required:               true,
    external_guard_resident_memory_ceiling_bytes: HY_COMPACT_PREFLIGHT_LIMIT_BYTES,
    external_guard_min_free_memory_percent:       15,
    external_guard_policy:                        "sampled process group footprint <= 4 GiB and cache-aware host headroom >= 15%",
    backend_private_allocation_cap_established:   false,
  }
  hy_compact_emit(preflight)
  raise ArgumentError.new("attention geometry exceeds the 4 GiB reserve estimate") if estimated_total > HY_COMPACT_PREFLIGHT_LIMIT_BYTES

  arena = ML::Video::Arena.new
  fill = ML::Metal::ComputePipeline.new("hy_nomas_compact_benchmark_fill", HY_COMPACT_KERNELS)
  zero = ML::Metal::ComputePipeline.new("hy_nomas_compact_benchmark_zero", HY_COMPACT_KERNELS)
  compare = ML::Metal::ComputePipeline.new("hy_nomas_compact_benchmark_compare", HY_COMPACT_KERNELS)

  count = values.to_i32
  q = arena.not_nil!.alloc("q", count)
  k = arena.not_nil!.alloc("k", count)
  v = arena.not_nil!.alloc("v", count)
  output_a = arena.not_nil!.alloc("output_a", count)
  output_b = paired ? arena.not_nil!.alloc("output_b", count) : output_a
  metrics = arena.not_nil!.alloc("diagnostic_metrics", 5)

  specs.each do |spec|
    output = spec.label == "A" ? output_a : output_b
    setup_started = Time.instant
    plan = if spec.plan_type == "canonical_dense"
             ML::Video::HyNomasAttentionMPSPlan.new(
               HY_COMPACT_BATCHES, tokens, HY_COMPACT_HEADS, HY_COMPACT_HEAD_DIM,
               q, k, v, output)
           else
             ML::Video::HyNomasCompactAttentionMPSPlan.new(
               HY_COMPACT_BATCHES, tokens, HY_COMPACT_HEADS, HY_COMPACT_HEAD_DIM,
               q, k, v, output, spec.query_chunk, spec.heads_per_group)
           end
    setup_plan_wall_seconds = (Time.instant - setup_started).total_seconds
    arms << HyCompactBenchmarkArm.new(spec, output, plan, setup_plan_wall_seconds)
  end

  actual_named_bytes = arena.not_nil!.resident_bytes + arms.sum(0_i64) { |arm| arm.plan.resident_bytes }
  unless actual_named_bytes == named_and_stage_bytes
    raise "actual named buffer bytes (#{actual_named_bytes}) differ from preflight (#{named_and_stage_bytes})"
  end
  unless actual_named_bytes + score_reserve_bytes + HY_COMPACT_OPAQUE_RESERVE_BYTES <= HY_COMPACT_PREFLIGHT_LIMIT_BYTES
    raise "actual named buffers and reserves exceed the 4 GiB budget"
  end

  setup_started = Time.instant
  setup = ML::Metal::CommandBuffer.new
  retained_commands << setup
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  { {q, 1_u32}, {k, 2_u32}, {v, 3_u32} }.each do |buffer, salt|
    setup_encoder.set_pipeline(fill)
      .set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
      .set_value(buffer.element_count.to_u32, 1)
      .set_value(salt, 2)
      .set_value(input_bf16_boundary ? 1_u32 : 0_u32, 3)
      .dispatch_1d(buffer.element_count)
  end
  setup_encoder.end_encoding
  setup.commit
  setup.wait
  fill_setup_wall_seconds = (Time.instant - setup_started).total_seconds
  hy_compact_emit({schema: "hy_nomas_compact_attention_input_setup_v1",
                   status: "ok", formula: HY_COMPACT_INPUT_FORMULA,
                   salts: {q: 1, k: 2, v: 3}, gpu_generated_values_per_buffer: count,
                   input_bf16_boundary_roundtrip: input_bf16_boundary,
                   host_wall_including_terminal_wait_seconds: fill_setup_wall_seconds,
                   terminal_waits: 1, setup_outside_measured_arms: true})

  arms.each do |arm|
    sample = hy_compact_measure_arm(arm, "warmup", 0, "warmup", 1,
      actual_named_bytes, score_reserve_bytes, fill_setup_wall_seconds,
      retained_chains)
    hy_compact_emit(sample)
  end

  measured_samples = [] of HyCompactBenchmarkSample
  if paired
    repeats.times do |pair_index|
      pair_order = pair_index.even? ? "AB" : "BA"
      ordered = pair_index.even? ? arms : [arms[1], arms[0]]
      ordered.each_with_index do |arm, position|
        sample = hy_compact_measure_arm(arm, "measured", pair_index + 1, pair_order,
          position + 1, actual_named_bytes, score_reserve_bytes, fill_setup_wall_seconds,
          retained_chains)
        hy_compact_emit(sample)
        measured_samples << HyCompactBenchmarkSample.new(arm.spec.label,
          pair_index + 1, pair_order, position + 1,
          sample[:encode_wall_seconds],
          sample[:total_wall_including_terminal_wait_seconds],
          sample[:gpu_elapsed_seconds])
      end
    end
  else
    repeats.times do |sample_index|
      sample = hy_compact_measure_arm(arms[0], "measured", sample_index + 1, "A",
        1, actual_named_bytes, score_reserve_bytes, fill_setup_wall_seconds,
        retained_chains)
      hy_compact_emit(sample)
      measured_samples << HyCompactBenchmarkSample.new("A", sample_index + 1, "A", 1,
        sample[:encode_wall_seconds],
        sample[:total_wall_including_terminal_wait_seconds],
        sample[:gpu_elapsed_seconds])
    end
  end

  comparison = compare_dense ? "compact_arm_a_vs_canonical_dense_arm_b" : (paired ? "arm_a_vs_arm_b" : "self_finite_and_bin_check")
  diagnostic = hy_compact_diagnose(output_a, paired ? output_b : output_a, metrics,
    zero, compare, count, comparison, tokens, query_chunk, heads_per_group,
    paired ? query_chunk_b : nil, paired ? heads_per_group_b : nil,
    retained_chains)
  hy_compact_emit(diagnostic)
  unless diagnostic[:status] == "pass"
    raise "diagnostic failed: nonfinite=#{diagnostic[:nonfinite_pairs]}, tolerance=#{diagnostic[:tolerance_violations]}, bf16_rne=#{diagnostic[:bf16_rne_bin_mismatches]}"
  end

  if paired
    pair_comparisons = [] of NamedTuple(
      pair_repetition: Int32, pair_order: String,
      arm_a_total_wall_seconds: Float64, arm_b_total_wall_seconds: Float64,
      arm_a_over_b_total_wall_ratio: Float64, arm_b_minus_a_total_wall_seconds: Float64,
      arm_a_encode_wall_seconds: Float64, arm_b_encode_wall_seconds: Float64,
      arm_a_over_b_encode_wall_ratio: Float64, arm_b_minus_a_encode_wall_seconds: Float64,
      arm_a_gpu_elapsed_seconds: Float64, arm_b_gpu_elapsed_seconds: Float64,
      arm_a_over_b_gpu_elapsed_ratio: Float64?, arm_b_minus_a_gpu_elapsed_seconds: Float64)
    repeats.times do |pair_index|
      repetition = pair_index + 1
      sample_a = measured_samples.find do |sample|
        sample.arm == "A" && sample.pair_repetition == repetition
      end.not_nil!
      sample_b = measured_samples.find do |sample|
        sample.arm == "B" && sample.pair_repetition == repetition
      end.not_nil!
      unless sample_a.total_wall_including_terminal_wait_seconds > 0.0 &&
             sample_b.total_wall_including_terminal_wait_seconds > 0.0 &&
             sample_a.encode_wall_seconds > 0.0 && sample_b.encode_wall_seconds > 0.0
        raise "paired sample has a nonpositive wall-time denominator"
      end
      pair_comparisons << {
        pair_repetition:                   repetition,
        pair_order:                        sample_a.pair_order,
        arm_a_total_wall_seconds:          sample_a.total_wall_including_terminal_wait_seconds,
        arm_b_total_wall_seconds:          sample_b.total_wall_including_terminal_wait_seconds,
        arm_a_over_b_total_wall_ratio:     sample_a.total_wall_including_terminal_wait_seconds / sample_b.total_wall_including_terminal_wait_seconds,
        arm_b_minus_a_total_wall_seconds:  sample_b.total_wall_including_terminal_wait_seconds - sample_a.total_wall_including_terminal_wait_seconds,
        arm_a_encode_wall_seconds:         sample_a.encode_wall_seconds,
        arm_b_encode_wall_seconds:         sample_b.encode_wall_seconds,
        arm_a_over_b_encode_wall_ratio:    sample_a.encode_wall_seconds / sample_b.encode_wall_seconds,
        arm_b_minus_a_encode_wall_seconds: sample_b.encode_wall_seconds - sample_a.encode_wall_seconds,
        arm_a_gpu_elapsed_seconds:         sample_a.gpu_elapsed_seconds,
        arm_b_gpu_elapsed_seconds:         sample_b.gpu_elapsed_seconds,
        arm_a_over_b_gpu_elapsed_ratio:    sample_a.gpu_elapsed_seconds > 0.0 && sample_b.gpu_elapsed_seconds > 0.0 ? sample_a.gpu_elapsed_seconds / sample_b.gpu_elapsed_seconds : nil,
        arm_b_minus_a_gpu_elapsed_seconds: sample_b.gpu_elapsed_seconds - sample_a.gpu_elapsed_seconds,
      }
    end
    wall_a = measured_samples.select { |sample| sample.arm == "A" }.map(&.total_wall_including_terminal_wait_seconds)
    wall_b = measured_samples.select { |sample| sample.arm == "B" }.map(&.total_wall_including_terminal_wait_seconds)
    encode_a = measured_samples.select { |sample| sample.arm == "A" }.map(&.encode_wall_seconds)
    encode_b = measured_samples.select { |sample| sample.arm == "B" }.map(&.encode_wall_seconds)
    gpu_a = measured_samples.select { |sample| sample.arm == "A" }.map(&.gpu_elapsed_seconds)
    gpu_b = measured_samples.select { |sample| sample.arm == "B" }.map(&.gpu_elapsed_seconds)
    hy_compact_emit({schema: "hy_nomas_compact_attention_decision_v1",
                     status: "diagnostic_pass", comparison: comparison,
                     measured_pair_orders: pair_comparisons.map { |pair| pair[:pair_order] },
                     within_pair_comparisons: pair_comparisons,
                     arm_a_total_wall_samples_seconds: wall_a, arm_b_total_wall_samples_seconds: wall_b,
                     arm_a_total_wall_median_seconds: hy_compact_median(wall_a),
                     arm_b_total_wall_median_seconds: hy_compact_median(wall_b),
                     arm_a_encode_wall_median_seconds: hy_compact_median(encode_a),
                     arm_b_encode_wall_median_seconds: hy_compact_median(encode_b),
                     arm_a_gpu_elapsed_median_seconds: hy_compact_median(gpu_a),
                     arm_b_gpu_elapsed_median_seconds: hy_compact_median(gpu_b),
                     total_wall_includes_terminal_wait: true, diagnostic_passed: true,
                     benchmark_selects_or_changes_no_default: true})
  else
    hy_compact_emit({schema: "hy_nomas_compact_attention_decision_v1",
                     status: "single_arm_diagnostic_pass", measured_samples: repeats,
                     total_wall_includes_terminal_wait: true, diagnostic_passed: true,
                     independent_arm_comparison_available: false, benchmark_selects_or_changes_no_default: true})
  end
rescue ex
  failure_message = ex.message || ex.class.name
  hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "benchmark",
                   status: "failed", error: failure_message})
ensure
  cleanup_ok = true
  retained_commands.each do |command|
    begin
      if command.committed? && !command.completed?
        command.wait
      elsif !command.committed?
        command.discard
      end
    rescue ex
      cleanup_ok = false
      failure_message ||= ex.message || ex.class.name
      hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "setup_command_drain",
                       error: ex.message || ex.class.name, resident_buffers_preserved: true})
    end
  end
  retained_chains.each do |chain|
    begin
      chain.commit_and_wait unless chain.completed?
      chain.release if chain.completed?
    rescue ex
      cleanup_ok = false
      failure_message ||= ex.message || ex.class.name
      hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "command_chain_drain",
                       error: ex.message || ex.class.name, resident_buffers_preserved: true})
    end
  end
  if cleanup_ok
    arms.each do |arm|
      begin
        arm.plan.release
      rescue ex
        cleanup_ok = false
        failure_message ||= ex.message || ex.class.name
        hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "plan_release",
                         arm: arm.spec.label, error: ex.message || ex.class.name,
                         resident_buffers_preserved: true})
      end
    end
  end
  if cleanup_ok
    begin
      arena.try(&.release)
    rescue ex
      failure_message ||= ex.message || ex.class.name
      hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "arena_release",
                       error: ex.message || ex.class.name})
    end
  else
    failure_message ||= "GPU work could not be proven drained; resources remain retained"
    hy_compact_emit({schema: HY_COMPACT_FAILURE_SCHEMA, phase: "cleanup",
                     error: "GPU work could not be proven drained or released; plans and caller buffers remain retained"})
  end
end

exit 1 if failure_message
