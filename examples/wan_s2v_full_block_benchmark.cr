# Full-shape one-block cost and exactness diagnostic for Wan S2V.
# This is not a complete video inference or image-quality qualification.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/wan_s2v_block"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_attention_mps"
require "../spec/support/video_allocation_counter"
require "./support/wan_host_observation"

private FULL_SEQ                       = 36_608
private FULL_VIDEO_TOKENS              = 34_944
private FULL_CONTEXT                   =    512
private DIM                            =  5_120
private HEADS                          =     40
private HEAD_DIM                       =    128
private FFN                            = 13_824
private FRAMES                         =     21
private SPATIAL                        =  1_664
private BASELINE_CHUNK_Q               =    512
private KV_LAYOUT_CHUNK_Q              =    128
private EXPLICIT_SCORE_TILE_MULTIPLIER =  3_i64
private EXPLICIT_SCORE_TILE_BYTES      = FULL_SEQ.to_i64 * DIM.to_i64 * sizeof(Float32)
private EXPLICIT_SCORE_RESERVE_BYTES   = EXPLICIT_SCORE_TILE_MULTIPLIER * EXPLICIT_SCORE_TILE_BYTES
private MAX_EXPLICIT_BUDGET            = 16_i64 * 1024 * 1024 * 1024
private MAX_ATTENTION_GRAPH_BUDGET     = 18_i64 * 1024 * 1024 * 1024
private OPAQUE_RESERVE                 = 1_i64 * 1024 * 1024 * 1024
private EXPECTED_WEIGHTS_BYTES         = 1_017_447_344_i64
private EXPECTED_WEIGHTS_SHA256        = "c172519063c1c34fb36f7f930429ca7db655a0e335524d7487525db1df5ad220"
private FULL_VALUES                    = FULL_SEQ.to_i64 * DIM

private BENCH_METAL = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  kernel void wan_full_block_fill_f32(
      device float *out [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &salt [[buffer(2)]], constant float &scale [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const ulong n = ulong(i);
    const int v = int((n * 17ul + (n / 127ul) * 7ul + ulong(salt) * 11ul) % 251ul) - 125;
    out[i] = float(v) * scale;
  }

  kernel void wan_full_block_fill_positions(
      device float *out [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &video_tokens [[buffer(2)]], constant uint &spatial [[buffer(3)]],
      constant uint &width [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint token = i / 3u;
    const uint local = token < video_tokens ? token : token - video_tokens;
    const uint frame = token < video_tokens ? local / spatial : 0u;
    const uint spatial_index = local % spatial;
    const uint axis = i % 3u;
    out[i] = axis == 0u ? float(frame) : (axis == 1u ? float(spatial_index / width) : float(spatial_index % width));
  }

  kernel void wan_full_block_zero_metrics(
      device atomic_uint *m [[buffer(0)]], constant uint &count [[buffer(1)]],
      uint i [[thread_position_in_grid]]) {
    if (i < count) atomic_store_explicit(m + i, 0u, memory_order_relaxed);
  }

  kernel void wan_full_block_compare(
      device const float *reference [[buffer(0)]], device const float *actual [[buffer(1)]],
      device atomic_uint *m [[buffer(2)]], constant uint &count [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float a = reference[i], b = actual[i];
    if (as_type<uint>(a) != as_type<uint>(b))
      atomic_fetch_add_explicit(m, 1u, memory_order_relaxed);
    if (!isfinite(a) || !isfinite(b)) {
      atomic_fetch_add_explicit(m + 2, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
      return;
    }
    const float error = abs(a - b);
    const float ratio = error / (2.0e-5f + 2.0e-5f * abs(a));
    if (ratio > 1.0f) atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(m + 3, as_type<uint>(error), memory_order_relaxed);
    atomic_fetch_max_explicit(m + 4, as_type<uint>(ratio), memory_order_relaxed);
  }

  kernel void wan_full_block_corrupt_plus_one_and_nan(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0u) {
      x[0] += 1.0f;
      x[1] = as_type<float>(0x7fc00001u);
    }
  }
METAL

private record ArmResult, groups : Int32, query_chunk : Int32, order : String, warmup : Bool,
  host_wall_seconds : Float64, encode_host_elapsed_seconds : Float64,
  command_chain_setup_elapsed_seconds : Float64, terminal_wait_elapsed_seconds : Float64,
  monotonic_begin_seconds : Float64, monotonic_end_seconds : Float64,
  gpu_elapsed_seconds : Float64, segments : Int32, terminal_waits : Int32,
  encode_metal_buffer_allocations : Int64,
  phase_root_profile : ML::Video::MPSCommandPhaseProfile?,
  api_profile : ML::Video::MPSAPIProfile?,
  observation_before : WanHostObservation::Snapshot,
  observation_after : WanHostObservation::Snapshot

private def phase_root_profile_json(profile : ML::Video::MPSCommandPhaseProfile)
  {
    root_count:                profile.root_count,
    root_duration_sum_seconds: profile.root_duration_seconds,
    envelope_seconds:          profile.envelope_seconds,
    union_seconds:             profile.union_seconds,
    overlap_seconds:           profile.overlap_seconds,
    phase_union_seconds:       profile.phase_union_seconds,
    attribution_scope:         "one WanS2VBlock.encode with no same-root caller prelude, suffix, or audio work",
    interval_interpretation:   "roots and phases may overlap in GPU time; phase unions are not additive",
  }
end

private def mps_api_profile_json(profile : ML::Video::MPSAPIProfile)
  {
    schema: "wan_mps_api_profile_v1",
    enabled: true,
    available_after_existing_terminal_wait: true,
    timing_unit: "nanoseconds",
    wall_clock: "CLOCK_MONOTONIC",
    thread_cpu_clock: "CLOCK_THREAD_CPUTIME_ID",
    phase_attribution: "MPS API wrapper entry phase; phase 0 is outside block phases",
    timing_semantics: "Graph totals are inclusive; nested fields are nested commitAndContinue spans; graph exclusive subtracts nested spans; commit and linear exclusive equal their own totals",
    phases: profile.phases.map do |phase|
      {
        phase_id: phase.phase_id,
        name: phase.name,
        operations: phase.operations.map do |name, stats|
          {
            name: name,
            count: stats.count,
            wall_total_ns: stats.wall_total_ns,
            wall_max_ns: stats.wall_max_ns,
            thread_cpu_total_ns: stats.thread_cpu_total_ns,
            thread_cpu_max_ns: stats.thread_cpu_max_ns,
            nested_count: stats.nested_count,
            nested_wall_total_ns: stats.nested_wall_total_ns,
            nested_thread_cpu_total_ns: stats.nested_thread_cpu_total_ns,
            exclusive_wall_total_ns: stats.exclusive_wall_total_ns,
            exclusive_thread_cpu_total_ns: stats.exclusive_thread_cpu_total_ns,
          }
        end,
      }
    end,
  }
end

private def sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def checked_sum(values : Enumerable(Int64)) : Int64
  total = 0_i64
  values.each do |value|
    raise ArgumentError.new("Full-block memory estimate overflow") if value < 0 || total > Int64::MAX - value
    total += value
  end
  total
end

private def encode_arm(groups : Int32, order : String, warmup : Bool,
                       x : ML::MetalBuffer, modulation : ML::MetalBuffer,
                       positions : ML::MetalBuffer, context : ML::MetalBuffer,
                       output : ML::MetalBuffer, weights : ML::Video::Arena,
                       scratch : ML::Video::Arena, cfg : ML::Video::WanS2VConfig,
                       linear : ML::Video::WanS2VMPSLinearWorkspace,
                       attention : ML::Video::WanAttentionMPSChunkWorkspace,
                       precompute_rope : Bool = false,
                       phase_roots : Bool = false,
                       api_profile : Bool = false,
                       reuse_scratch : Bool = false,
                       fuse_kv_preparation : Bool = false) : ArmResult
  observation_before = WanHostObservation::Snapshot.capture
  monotonic_begin = WanHostObservation.monotonic_seconds
  wall_started = Time.instant
  chain = ML::Metal::MPSCommandChain.new
  if api_profile
    chain.enable_api_profile!(phase_roots: true)
  elsif phase_roots
    chain.enable_phase_root_profile!
  end
  allocations_before = ML::MetalBuffer.video_spec_allocations
  encode_started = Time.instant
  ML::Video::WanS2VBlock.encode(chain.encoder, x, modulation, positions, context,
    output, weights, scratch, cfg, weight_prefix: "block.", tiled_attention: true,
    execution_chain: chain, mps_workspace: linear, attention_workspace: attention,
    cross_attention_groups_per_segment: groups, precompute_rope: precompute_rope,
    reuse_scratch: reuse_scratch, fuse_kv_preparation: fuse_kv_preparation)
  encode_seconds = (Time.instant - encode_started).total_seconds
  wait_started = Time.instant
  # Diagnostic phase roots are source-bound to this block. Do no suffix or
  # audio work on the chain before the existing terminal wait.
  chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  wall_ended = Time.instant
  wall_seconds = (wall_ended - wall_started).total_seconds
  monotonic_end = WanHostObservation.monotonic_seconds
  observation_after = WanHostObservation::Snapshot.capture
  phase_profile = chain.phase_root_profile_enabled? ? chain.phase_root_profile : nil
  api_profile_result = if api_profile
                         profile = chain.api_profile
                         expected_graph_calls =
                           ((cfg.seq.to_i64 + attention.chunk_rows.to_i64 - 1) // attention.chunk_rows.to_i64).to_u64
                         expected_commit_calls = chain.segment_count.to_i64 - 1
                         raise "Full-block API profile has no continuation roots" unless expected_commit_calls > 0
                         # Full-block Wan uses 3 self-QKV, self-O, 3 cross-QKV,
                         # cross-O, and 2 FFN MPS linear encodes.
                         profile.validate_full_block_counts!(expected_graph_calls, 10_u64,
                           expected_commit_calls.to_u64)
                         profile
                       else
                         nil
                       end
  allocations = ML::MetalBuffer.video_spec_allocations - allocations_before
  raise "Full block encode allocated MetalBuffers" unless allocations == 0
  raise "Full block arm did not use exactly one terminal wait" unless chain.wait_count == 1
  raise "Full block arm did not segment its command chain" unless chain.segment_count > 0
  result = ArmResult.new(groups, attention.chunk_rows, order, warmup, wall_seconds, encode_seconds,
    (encode_started - wall_started).total_seconds, wait_seconds,
    monotonic_begin, monotonic_end, chain.gpu_elapsed_seconds, chain.segment_count,
    chain.wait_count, allocations, phase_profile, api_profile_result,
    observation_before, observation_after)
  chain.release
  result
rescue ex
  if chain && !chain.completed?
    begin
      chain.commit_and_wait
    rescue
    end
  end
  chain.release if chain && chain.completed?
  raise ex
end

private def encode_inputs(e : ML::Metal::ComputeEncoder,
                          fill : ML::Metal::ComputePipeline,
                          positions_fill : ML::Metal::ComputePipeline,
                          x : ML::MetalBuffer, modulation : ML::MetalBuffer,
                          positions : ML::MetalBuffer, context : ML::MetalBuffer,
                          local_audio : ML::MetalBuffer, cfg : ML::Video::WanS2VConfig) : Nil
  [
    {x, FULL_VALUES.to_i32, 1_u32, 0.0004_f32},
    {modulation, 12 * DIM, 2_u32, 0.0002_f32},
    {context, FULL_CONTEXT * DIM, 3_u32, 0.0003_f32},
    {local_audio, local_audio.element_count, 4_u32, 0.0_f32},
  ].each do |buffer, count, salt, scale|
    e.set_pipeline(fill).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 1).set_value(salt, 2).set_value(scale, 3).dispatch_1d(count)
  end
  e.set_pipeline(positions_fill).set_buffer(positions, 0, ML::Metal::BufferAccess::Write)
    .set_value((cfg.seq * 3).to_u32, 1).set_value(cfg.video_tokens.to_u32, 2)
    .set_value(SPATIAL.to_u32, 3).set_value(52_u32, 4).dispatch_1d(cfg.seq * 3)
end

private def validate_pair(reference : ML::MetalBuffer, actual : ML::MetalBuffer,
                          metrics : ML::MetalBuffer, zero : ML::Metal::ComputePipeline,
                          compare : ML::Metal::ComputePipeline) : NamedTuple(bit_differences: UInt32,
  tolerance_violations: UInt32, nonfinite: UInt32,
  max_abs: Float32, max_ratio: Float32)
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write)
    .set_value(5_u32, 1).dispatch_1d(5)
  e.set_pipeline(compare).set_buffer(reference, 0).set_buffer(actual, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(FULL_VALUES.to_u32, 3).dispatch_1d(FULL_VALUES.to_i32)
  e.end_encoding
  command.commit_and_wait
  values = metrics.contents.as(UInt32*)
  {
    bit_differences: values[0], tolerance_violations: values[1], nonfinite: values[2],
    max_abs: values[3].unsafe_as(Float32), max_ratio: values[4].unsafe_as(Float32),
  }
end

weights_path = ""
pairs = 3
comparison_mode = "cross-segments"
candidate_chunk = 256
warmup_order = "AB"
phase_roots_requested = false
prepared_kv_requested = false
api_profile_requested = false
reuse_block_scratch = false
plan_only = false
OptionParser.parse do |parser|
  parser.banner = "Usage: build/wan_s2v_full_block_benchmark --weights=FILE [--comparison=kv-prep-control|kv-prep-fusion --pairs=3|4] [--warmup-order=AB|BA] [--plan-only]"
  parser.on("--weights=FILE", "Official block-0 plus injector-0 BF16 safetensors sidecar") { |value| weights_path = value }
  parser.on("--comparison=MODE", "cross-segments (default), self-chunks, kv-layout, attention-graph, kv-prep-control, kv-prep-fusion, rope, or profile-replay") { |value| comparison_mode = value }
  parser.on("--candidate-self-chunk=N", "Candidate query rows for self-chunks: 128, 256 or 1024; K/V, attention-graph, rope and profile-replay modes require 128") { |value| candidate_chunk = value.to_i }
  parser.on("--pairs=N", "Measured pairs after warmup; K/V control/fusion accept 3 or balanced 4 (other modes retain 1..3)") { |value| pairs = value.to_i }
  parser.on("--warmup-order=AB|BA", "K/V control/fusion warmup order; reverse across fresh processes to expose order drift") { |value| warmup_order = value }
  parser.on("--phase-roots", "Report asynchronous command-root intervals; requires profile-replay") { phase_roots_requested = true }
  parser.on("--prepared-kv", "Profile replay with both arms using shared head-major K/V refreshed inside each block encode; requires profile-replay") { prepared_kv_requested = true }
  parser.on("--api-profile", "Profile candidate MPS API wall/thread CPU spans; requires profile-replay --phase-roots --prepared-kv") { api_profile_requested = true }
  parser.on("--reuse-block-scratch", "Use opt-in same-object block scratch lifetime aliases for both arms") { reuse_block_scratch = true }
  parser.on("--plan-only", "Print the full-shape memory plan and exit before GPU setup or dispatch") { plan_only = true }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end
raise ArgumentError.new("--weights is required") if weights_path.empty?
raise ArgumentError.new("--pairs must be within 1..4") unless (1..4).includes?(pairs)
raise ArgumentError.new("--comparison must be cross-segments, self-chunks, kv-layout, attention-graph, kv-prep-control, kv-prep-fusion, rope, or profile-replay") unless {"cross-segments", "self-chunks", "kv-layout", "attention-graph", "kv-prep-control", "kv-prep-fusion", "rope", "profile-replay"}.includes?(comparison_mode)
raise ArgumentError.new("--warmup-order must be AB or BA") unless {"AB", "BA"}.includes?(warmup_order)
raise ArgumentError.new("--candidate-self-chunk must be 128, 256 or 1024") unless {128, 256, 1024}.includes?(candidate_chunk)
compare_self_chunks = comparison_mode == "self-chunks"
compare_kv_layout = comparison_mode == "kv-layout"
compare_attention_graph = comparison_mode == "attention-graph"
compare_kv_prep_control = comparison_mode == "kv-prep-control"
compare_kv_prep_fusion = comparison_mode == "kv-prep-fusion"
compare_rope = comparison_mode == "rope"
compare_profile_replay = comparison_mode == "profile-replay"
baseline_self_attention_backend = "mpsgraph_sdpa_chunked"
candidate_self_attention_backend = compare_attention_graph ? "mpsgraph_explicit_qk_softmax_pv_chunked" : baseline_self_attention_backend
prepared_kv_replay = compare_profile_replay && prepared_kv_requested
prepared_kv_for_both_arms = compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || prepared_kv_replay
prepared_kv_for_candidate = compare_kv_layout || prepared_kv_for_both_arms
raise ArgumentError.new("--comparison=profile-replay requires --phase-roots") if compare_profile_replay && !phase_roots_requested
raise ArgumentError.new("--phase-roots is supported only with --comparison=profile-replay") if phase_roots_requested && !compare_profile_replay
raise ArgumentError.new("--prepared-kv is supported only with --comparison=profile-replay") if prepared_kv_requested && !compare_profile_replay
raise ArgumentError.new("--api-profile is supported only with --comparison=profile-replay") if api_profile_requested && !compare_profile_replay
raise ArgumentError.new("--api-profile requires --phase-roots and --prepared-kv") if api_profile_requested && !(phase_roots_requested && prepared_kv_requested)
raise ArgumentError.new("--comparison=rope requires --pairs=3") if compare_rope && pairs != 3
raise ArgumentError.new("--comparison=attention-graph requires --pairs=3") if compare_attention_graph && pairs != 3
raise ArgumentError.new("--comparison=kv-prep-control requires --pairs=3 or 4") if compare_kv_prep_control && !{3, 4}.includes?(pairs)
raise ArgumentError.new("--comparison=kv-prep-fusion requires --pairs=3 or 4") if compare_kv_prep_fusion && !{3, 4}.includes?(pairs)
raise ArgumentError.new("--pairs=4 is supported only by kv-prep-control and kv-prep-fusion") if pairs == 4 && !(compare_kv_prep_control || compare_kv_prep_fusion)
raise ArgumentError.new("--warmup-order=BA is supported only by kv-prep-control and kv-prep-fusion") if warmup_order == "BA" && !(compare_kv_prep_control || compare_kv_prep_fusion)
explicit_budget_bytes = compare_attention_graph ? MAX_ATTENTION_GRAPH_BUDGET : MAX_EXPLICIT_BUDGET
if compare_kv_layout || compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || compare_rope || compare_profile_replay
  raise ArgumentError.new("--comparison=#{comparison_mode} requires --candidate-self-chunk=128 so both arms use the same chunk") unless candidate_chunk == KV_LAYOUT_CHUNK_Q
end
require_exact = !(compare_self_chunks || compare_kv_layout || compare_attention_graph)
candidate_groups = (compare_self_chunks || compare_kv_layout || compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || compare_rope || compare_profile_replay) ? 320 : 1024
baseline_chunk = (compare_kv_layout || compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || compare_rope || compare_profile_replay) ? KV_LAYOUT_CHUNK_Q : BASELINE_CHUNK_Q
candidate_chunk = (compare_kv_layout || compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || compare_rope || compare_profile_replay) ? KV_LAYOUT_CHUNK_Q : (compare_self_chunks ? candidate_chunk : BASELINE_CHUNK_Q)
weights_path = File.expand_path(weights_path)
raise ArgumentError.new("Missing safetensors file #{weights_path}") unless File.file?(weights_path)
raise ArgumentError.new("Unexpected official block sidecar length") unless File.size(weights_path) == EXPECTED_WEIGHTS_BYTES
weights_sha = sha256_file(weights_path)
raise ArgumentError.new("Official block sidecar SHA-256 mismatch") unless weights_sha == EXPECTED_WEIGHTS_SHA256

cfg = ML::Video::WanS2VConfig.new(1, FULL_SEQ, FULL_CONTEXT, DIM, HEADS, FFN,
  FULL_VIDEO_TOKENS, 10_000.0_f32, FRAMES, SPATIAL)
# The block plans use full sequence rows. The unused five audio plans need only
# a one-frame/one-spatial-token shape; no audio injector is encoded here.
plan_cfg = ML::Video::WanS2VConfig.new(1, FULL_SEQ, FULL_CONTEXT, DIM, HEADS, FFN,
  1, 10_000.0_f32, 1, 1)
audio_cfg = ML::Video::WanS2VAudioConfig.new(1, FULL_SEQ, DIM, HEADS, 1, 1, 5)
bindings = ML::Video::WanS2VBlock.checkpoint_bindings(cfg,
  ML::GGUF::SafeTensorDType::BF16, block_target_prefix: "block.", audio_target_prefix: "audio.")
checkpoint = ML::Video::CheckpointFile.new(weights_path)
prepared : ML::Video::PreparedWeights? = nil
weight_arena : ML::Video::Arena? = nil
scratch : ML::Video::Arena? = nil
audio_scratch : ML::Video::Arena? = nil
inputs : ML::Video::Arena? = nil
projection_stage : ML::Video::WanS2VMPSLinearStage? = nil
projection_workspace : ML::Video::WanS2VMPSLinearWorkspace? = nil
attention_workspace : ML::Video::WanAttentionMPSChunkWorkspace? = nil
candidate_attention_workspace : ML::Video::WanAttentionMPSChunkWorkspace? = nil
begin
  explicit_io_counts = [
    FULL_VALUES, 12_i64 * DIM, 3_i64 * FULL_SEQ,
    FULL_CONTEXT.to_i64 * DIM, audio_cfg.audio_rows.to_i64 * DIM,
    2_i64 * FULL_VALUES, 5_i64,
  ]
  input_bytes = checked_sum(explicit_io_counts) * sizeof(Float32)
  default_block_scratch_bytes = ML::Video::WanS2VBlock.scratch_bytes(cfg, precompute_rope: compare_rope)
  block_scratch_bytes = ML::Video::WanS2VBlock.scratch_bytes(cfg,
    precompute_rope: compare_rope, reuse_scratch: reuse_block_scratch)
  block_scratch_bytes_saved = default_block_scratch_bytes - block_scratch_bytes
  tiny_audio_scratch_bytes = ML::Video::WanS2VAudioInjector.scratch_bytes(audio_cfg)
  projection_stage_bytes = ML::Video::WanS2VMPSLinearStage.required_weight_count(DIM, FFN).to_i64 * sizeof(Float32)
  attention_workspace_bytes = 3_i64 * baseline_chunk * DIM * sizeof(Float32)
  attention_workspace_bytes += 3_i64 * candidate_chunk * DIM * sizeof(Float32) if compare_self_chunks || compare_kv_layout || compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || compare_profile_replay
  prepared_kv_bytes = prepared_kv_for_candidate ? 2_i64 * FULL_VALUES * sizeof(Float32) : 0_i64
  attention_workspace_bytes += prepared_kv_bytes
  explicit_score_reserve_bytes = compare_attention_graph ? EXPLICIT_SCORE_RESERVE_BYTES : 0_i64
  explicit_nonweight_bytes = checked_sum([
    input_bytes, block_scratch_bytes, tiny_audio_scratch_bytes,
    projection_stage_bytes, attention_workspace_bytes,
  ])
  reserve_bytes = explicit_nonweight_bytes + explicit_score_reserve_bytes + OPAQUE_RESERVE
  files = [checkpoint]
  plan = ML::Video::WeightPlan.new(files, bindings, explicit_budget_bytes,
    reserve_bytes, ML::Video::WeightDType::BF16, require_all: true)
  total_estimate = plan.peak_bytes
  raise ArgumentError.new("Full block exceeds the #{explicit_budget_bytes} byte explicit-plus-reserve budget for #{comparison_mode}: #{total_estimate}") if total_estimate > explicit_budget_bytes

  puts({schema: "wan_s2v_full_block_memory_plan_v1",
        status: plan_only ? "plan_only" : "admitted_for_benchmark", plan_only: plan_only,
        sequence_tokens: FULL_SEQ, video_tokens: FULL_VIDEO_TOKENS, context_tokens: FULL_CONTEXT,
        dim: DIM, heads: HEADS, head_dim: HEAD_DIM, ffn_dim: FFN,
        weight_dtype: "BF16", operator_math: "F32", checkpoint_bytes: File.size(weights_path),
        checkpoint_sha256: weights_sha, bound_tensors: bindings.size,
        resident_weight_bytes: plan.destination_bytes, block_scratch_bytes: block_scratch_bytes,
        reuse_block_scratch: reuse_block_scratch,
        block_scratch_layout: reuse_block_scratch ? "lifetime-aliased" : "unaliased",
        default_block_scratch_bytes: default_block_scratch_bytes,
        block_scratch_bytes_saved: block_scratch_bytes_saved,
        unused_audio_plan_scratch_bytes: tiny_audio_scratch_bytes, input_output_bytes: input_bytes,
        projection_stage_bytes: projection_stage_bytes,
        self_attention_chunk_bytes: attention_workspace_bytes - prepared_kv_bytes,
        prepared_kv_buffer_bytes: prepared_kv_bytes,
        prepared_kv_pair_shared_between_arms: prepared_kv_for_both_arms,
        prepared_kv_buffer_bytes_refreshed_per_arm_encode: prepared_kv_bytes,
        prepared_kv_requested: prepared_kv_requested,
        prepared_kv_refreshed_inside_each_prepared_arm_encode: prepared_kv_bytes > 0,
        self_attention_chunk_and_prepared_kv_bytes: attention_workspace_bytes,
        explicit_score_tile_bytes: compare_attention_graph ? EXPLICIT_SCORE_TILE_BYTES : 0_i64,
        explicit_score_tile_multiplier: compare_attention_graph ? EXPLICIT_SCORE_TILE_MULTIPLIER : 0_i64,
        explicit_score_additional_reserve_bytes: explicit_score_reserve_bytes,
        explicit_score_reserve_is_heuristic: true,
        opaque_mps_reserve_bytes: OPAQUE_RESERVE, estimated_bytes: total_estimate,
        budget_bytes: explicit_budget_bytes,
        budget_limit_gib: explicit_budget_bytes / (1024_i64 * 1024 * 1024),
        attention_backend: "mps_chunked",
        comparison_mode: comparison_mode,
        baseline_precompute_rope: false,
        candidate_precompute_rope: compare_rope,
        rope_coefficients_built_per_candidate_encode: compare_rope,
        baseline_query_chunk: baseline_chunk,
        candidate_query_chunk: candidate_chunk,
        baseline_kv_layout: prepared_kv_for_both_arms ? "B,H,K,D" : "B,K,H,D",
        candidate_kv_layout: prepared_kv_for_candidate ? "B,H,K,D" : "B,K,H,D",
        query_chunk_rows_equal_between_arms: baseline_chunk == candidate_chunk,
        layout_preparation_inside_block_encode: prepared_kv_for_candidate,
        baseline_self_attention_backend: baseline_self_attention_backend,
        candidate_self_attention_backend: candidate_self_attention_backend,
        cross_attention_backend: "native_tiled",
        baseline_fuse_kv_preparation: false,
        candidate_fuse_kv_preparation: compare_kv_prep_fusion,
        baseline_explicit_qk_softmax_pv: false,
        candidate_explicit_qk_softmax_pv: compare_attention_graph,
        exact_required: require_exact,
        cross_groups_per_segment: [320, candidate_groups], scope: "one real block-0 forward, synthetic inputs; not video quality or full-stack qualification"}.to_json)
  STDOUT.flush
  exit if plan_only

  weight_arena = ML::Video::Arena.new
  scratch = ML::Video::Arena.new
  audio_scratch = ML::Video::Arena.new
  inputs = ML::Video::Arena.new
  prepared = ML::Video::PreparedWeights.new(plan, weight_arena.not_nil!)
  ML::Video::WanS2VBlock.prepare!(precompute_rope: compare_rope)
  ML::Video::WanS2VAudioInjector.prepare!
  ML::Video::MetalOps.prepare_tiled_attention!

  ML::Video::WanS2VBlock.prepare_scratch!(scratch.not_nil!, cfg,
    precompute_rope: compare_rope, reuse_scratch: reuse_block_scratch)
  raise "Prepared block scratch differs from its memory plan" unless scratch.not_nil!.resident_bytes == block_scratch_bytes
  ML::Video::WanS2VAudioInjector.prepare_scratch!(audio_scratch.not_nil!, audio_cfg)
  x = inputs.not_nil!.alloc("x", FULL_VALUES.to_i32)
  modulation = inputs.not_nil!.alloc("modulation", 12 * DIM)
  positions = inputs.not_nil!.alloc("positions", 3 * FULL_SEQ)
  context = inputs.not_nil!.alloc("context", FULL_CONTEXT * DIM)
  local_audio = inputs.not_nil!.alloc("unused_local_audio", audio_cfg.audio_rows * DIM)
  output_baseline = inputs.not_nil!.alloc("output_baseline", FULL_VALUES.to_i32)
  output_candidate = inputs.not_nil!.alloc("output_candidate", FULL_VALUES.to_i32)
  metrics = inputs.not_nil!.alloc("comparison_metrics", 5)

  projection_stage = ML::Video::WanS2VMPSLinearStage.new(
    ML::Video::WanS2VMPSLinearStage.required_weight_count(DIM, FFN))
  projection_workspace = ML::Video::WanS2VMPSLinearWorkspace.new(
    plan_cfg, audio_cfg, scratch.not_nil!, audio_scratch.not_nil!, context,
    local_audio, projection_stage.not_nil!)
  chunk_q = scratch.not_nil!.alloc("chunk_q", baseline_chunk * DIM)
  chunk_a = scratch.not_nil!.alloc("chunk_a", baseline_chunk * DIM)
  chunk_b = scratch.not_nil!.alloc("chunk_b", baseline_chunk * DIM)
  # Keep attention-owned destinations outside the block scratch resources
  # checked by fused preparation. Both arms retain and refresh this same pair.
  candidate_prepared_k = prepared_kv_for_candidate ? inputs.not_nil!.alloc("shared_prepared_k_bhkd", FULL_VALUES.to_i32) : nil
  candidate_prepared_v = prepared_kv_for_candidate ? inputs.not_nil!.alloc("shared_prepared_v_bhkd", FULL_VALUES.to_i32) : nil
  attention_workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(
    1, FULL_SEQ, FULL_SEQ, HEADS, HEAD_DIM, baseline_chunk,
    scratch.not_nil!["wan_b_self_q"], scratch.not_nil!["wan_b_self_k"],
    scratch.not_nil!["wan_b_self_v"], scratch.not_nil!["wan_b_self_attn"],
    chunk_q, chunk_a, chunk_b,
    prepared_kv_for_both_arms ? candidate_prepared_k : nil,
    prepared_kv_for_both_arms ? candidate_prepared_v : nil)
  if compare_self_chunks || compare_kv_layout || compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || compare_profile_replay
    candidate_attention_workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(
      1, FULL_SEQ, FULL_SEQ, HEADS, HEAD_DIM, candidate_chunk,
      scratch.not_nil!["wan_b_self_q"], scratch.not_nil!["wan_b_self_k"],
      scratch.not_nil!["wan_b_self_v"], scratch.not_nil!["wan_b_self_attn"],
      scratch.not_nil!.alloc("candidate_chunk_q", candidate_chunk * DIM),
      scratch.not_nil!.alloc("candidate_chunk_a", candidate_chunk * DIM),
      scratch.not_nil!.alloc("candidate_chunk_b", candidate_chunk * DIM),
      candidate_prepared_k, candidate_prepared_v,
      explicit_qk_softmax_pv: compare_attention_graph)
  end
  candidate_attention = candidate_attention_workspace || attention_workspace.not_nil!

  fill = ML::Metal::ComputePipeline.new("wan_full_block_fill_f32", BENCH_METAL)
  positions_fill = ML::Metal::ComputePipeline.new("wan_full_block_fill_positions", BENCH_METAL)
  zero = ML::Metal::ComputePipeline.new("wan_full_block_zero_metrics", BENCH_METAL)
  compare = ML::Metal::ComputePipeline.new("wan_full_block_compare", BENCH_METAL)
  corrupt = ML::Metal::ComputePipeline.new("wan_full_block_corrupt_plus_one_and_nan", BENCH_METAL)
  setup = ML::Metal::CommandBuffer.new
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  encode_inputs(setup_encoder, fill, positions_fill, x, modulation, positions,
    context, local_audio, cfg)
  setup_encoder.end_encoding
  setup.commit_and_wait

  warmup_labels = warmup_order == "AB" ? ["A", "B"] : ["B", "A"]
  measured_pair_orders = if pairs == 4
                           ["AB", "BA", "BA", "AB"]
                         else
                           ["AB", "BA", "AB"][0, pairs]
                         end
  chronology = warmup_labels + measured_pair_orders.flat_map do |pair_order|
    pair_order == "AB" ? ["A", "B"] : ["B", "A"]
  end
  puts({schema: "wan_s2v_full_block_timing_order_v1",
        comparison_mode: comparison_mode, warmup_order: warmup_order,
        measured_pair_orders: measured_pair_orders,
        chronology: chronology.join(" -> "),
        chronology_arms: chronology,
        paired_order_balanced: measured_pair_orders.count("AB") == measured_pair_orders.count("BA")}.to_json)
  STDOUT.flush

  warm_results = {} of String => ArmResult
  warmup_labels.each do |label|
    baseline = label == "A"
    attention = baseline ? attention_workspace.not_nil! : candidate_attention
    groups = baseline ? 320 : candidate_groups
    output = baseline ? output_baseline : output_candidate
    warm_results[label] = encode_arm(groups, label, true, x, modulation, positions,
      context, output, weight_arena.not_nil!, scratch.not_nil!, cfg,
      projection_workspace.not_nil!, attention,
      precompute_rope: compare_rope && !baseline,
      phase_roots: compare_profile_replay && !baseline && phase_roots_requested && !api_profile_requested,
      api_profile: compare_profile_replay && !baseline && api_profile_requested,
      reuse_scratch: reuse_block_scratch,
      fuse_kv_preparation: compare_kv_prep_fusion && !baseline)
  end
  warm_baseline = warm_results["A"]
  warm_candidate = warm_results["B"]
  if compare_profile_replay
    baseline_root_count = warm_baseline.segments
    candidate_profile = warm_candidate.phase_root_profile.not_nil!
    unless baseline_root_count == warm_candidate.segments && candidate_profile.root_count == baseline_root_count
      raise "Profile-replay warmup command-root counts differ"
    end
  end
  warm_compare = validate_pair(output_baseline, output_candidate, metrics, zero, compare)
  puts({schema: "wan_s2v_full_block_warm_comparison_v1", compared_values: FULL_VALUES,
        bit_differences: warm_compare[:bit_differences],
        tolerance_violations: warm_compare[:tolerance_violations], nonfinite: warm_compare[:nonfinite],
        max_abs: warm_compare[:max_abs], max_normalized_error: warm_compare[:max_ratio],
        tolerance: "2e-5+2e-5*abs(reference)",
        exact_equal: warm_compare[:bit_differences] == 0,
        strict_pass: warm_compare[:tolerance_violations] == 0 && warm_compare[:nonfinite] == 0,
        comparison_mode: comparison_mode, exact_required: require_exact,
        reuse_block_scratch: reuse_block_scratch,
        block_scratch_layout: reuse_block_scratch ? "lifetime-aliased" : "unaliased",
        block_scratch_bytes: block_scratch_bytes,
        default_block_scratch_bytes: default_block_scratch_bytes,
        block_scratch_bytes_saved: block_scratch_bytes_saved,
        baseline_kv_layout: prepared_kv_for_both_arms ? "B,H,K,D" : "B,K,H,D",
        candidate_kv_layout: prepared_kv_for_candidate ? "B,H,K,D" : "B,K,H,D",
        baseline_query_chunk_rows: baseline_chunk, candidate_query_chunk_rows: candidate_chunk,
        layout_preparation_inside_block_encode: prepared_kv_for_candidate,
        prepared_kv_requested: prepared_kv_requested,
        prepared_kv_refreshed_inside_each_prepared_arm_encode: prepared_kv_bytes > 0,
        prepared_kv_pair_shared_between_arms: prepared_kv_for_both_arms,
        baseline_explicit_qk_softmax_pv: false,
        candidate_explicit_qk_softmax_pv: compare_attention_graph,
        phase_root_diagnostics_enabled: compare_profile_replay && phase_roots_requested,
        api_profile_diagnostics_enabled: api_profile_requested,
        baseline_self_attention_backend: baseline_self_attention_backend,
        candidate_self_attention_backend: candidate_self_attention_backend,
        cross_attention_backend: "native_tiled",
        baseline_fuse_kv_preparation: false,
        candidate_fuse_kv_preparation: compare_kv_prep_fusion,
        profile_replay_is_not_a_speed_comparison: compare_profile_replay,
        comparison_outside_timed_arms: true}.to_json)
  STDOUT.flush
  unless (!require_exact || warm_compare[:bit_differences] == 0) && warm_compare[:tolerance_violations] == 0 && warm_compare[:nonfinite] == 0
    raise "Warmup block outputs failed the required comparison"
  end
  {warm_baseline, warm_candidate}.each do |arm|
    arm_uses_prepared_kv = compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || prepared_kv_replay || (arm.order == "B" && compare_kv_layout)
    arm_uses_explicit_qk_softmax_pv = compare_attention_graph && arm.order == "B"
    arm_uses_fused_kv_preparation = compare_kv_prep_fusion && arm.order == "B"
    puts({schema: "wan_s2v_full_block_arm_v1", groups_per_segment: arm.groups, query_chunk_rows: arm.query_chunk,
          order: arm.order, warmup: arm.warmup, host_wall_seconds: arm.host_wall_seconds,
          encode_cpu_seconds: arm.encode_host_elapsed_seconds,
          encode_host_elapsed_seconds: arm.encode_host_elapsed_seconds,
          encode_host_elapsed_semantics: "elapsed host wall time inside the synchronous encode/enqueue call; not process CPU time",
          command_chain_setup_elapsed_seconds: arm.command_chain_setup_elapsed_seconds,
          terminal_wait_elapsed_seconds: arm.terminal_wait_elapsed_seconds,
          monotonic_begin_seconds: arm.monotonic_begin_seconds,
          monotonic_end_seconds: arm.monotonic_end_seconds,
          observations: WanHostObservation.between(arm.observation_before, arm.observation_after),
          gpu_elapsed_seconds: arm.gpu_elapsed_seconds,
          command_segments: arm.segments, terminal_waits: arm.terminal_waits,
          encode_metal_buffer_allocations: arm.encode_metal_buffer_allocations,
          comparison_mode: comparison_mode,
          reuse_block_scratch: reuse_block_scratch,
          block_scratch_layout: reuse_block_scratch ? "lifetime-aliased" : "unaliased",
          block_scratch_bytes: block_scratch_bytes,
          default_block_scratch_bytes: default_block_scratch_bytes,
          block_scratch_bytes_saved: block_scratch_bytes_saved,
          kv_layout: arm_uses_prepared_kv ? "B,H,K,D" : "B,K,H,D",
          baseline_query_chunk_rows: baseline_chunk, candidate_query_chunk_rows: candidate_chunk,
          query_chunk_rows_equal_between_arms: baseline_chunk == candidate_chunk,
          layout_preparation_inside_block_encode: arm_uses_prepared_kv,
          prepared_kv_buffer_bytes: arm_uses_prepared_kv ? prepared_kv_bytes : 0_i64,
          prepared_kv_pair_shared_between_arms: prepared_kv_for_both_arms,
          prepared_kv_buffer_bytes_refreshed_per_encode: arm_uses_prepared_kv ? prepared_kv_bytes : 0_i64,
          prepared_kv_requested: prepared_kv_requested,
          self_attention_backend: arm_uses_explicit_qk_softmax_pv ? candidate_self_attention_backend : baseline_self_attention_backend,
          cross_attention_backend: "native_tiled",
          explicit_qk_softmax_pv: arm_uses_explicit_qk_softmax_pv,
          fuse_kv_preparation: arm_uses_fused_kv_preparation,
          precompute_rope: compare_rope && arm.order == "B",
          rope_coefficient_build_inside_block_encode: compare_rope && arm.order == "B",
          phase_root_diagnostics_enabled: arm.phase_root_profile != nil,
          phase_root_profile: arm.phase_root_profile.try { |profile| phase_root_profile_json(profile) },
          api_profile_diagnostics_enabled: arm.api_profile != nil,
          mps_api_profile: arm.api_profile.try { |profile| mps_api_profile_json(profile) },
          profile_replay_is_not_a_speed_comparison: compare_profile_replay,
          inter_operator_host_waits: 0, activation_readbacks: 0}.to_json)
  end
  STDOUT.flush

  pairs.times do |pair|
    order = measured_pair_orders[pair]
    baseline_root_count = nil.as(Int32?)
    candidate_root_count = nil.as(Int32?)
    order.each_char do |label|
      baseline = label == 'A'
      output = baseline ? output_baseline : output_candidate
      attention = baseline ? attention_workspace.not_nil! : candidate_attention
      groups = baseline ? 320 : candidate_groups
      arm = encode_arm(groups, baseline ? "A" : "B", false,
        x, modulation, positions, context, output, weight_arena.not_nil!,
        scratch.not_nil!, cfg, projection_workspace.not_nil!, attention,
        precompute_rope: compare_rope && !baseline,
        phase_roots: compare_profile_replay && !baseline && phase_roots_requested && !api_profile_requested,
        api_profile: compare_profile_replay && !baseline && api_profile_requested,
        reuse_scratch: reuse_block_scratch,
        fuse_kv_preparation: compare_kv_prep_fusion && !baseline)
      if baseline
        baseline_root_count = arm.segments
      elsif profile = arm.phase_root_profile
        unless profile.root_count == arm.segments
          raise "Profile-replay root summary count differs from command segment count"
        end
        candidate_root_count = profile.root_count
      end
      arm_uses_prepared_kv = compare_attention_graph || compare_kv_prep_control || compare_kv_prep_fusion || prepared_kv_replay || (!baseline && compare_kv_layout)
      arm_uses_explicit_qk_softmax_pv = compare_attention_graph && !baseline
      arm_uses_fused_kv_preparation = compare_kv_prep_fusion && !baseline
      puts({schema: "wan_s2v_full_block_arm_v1", pair: pair,
            groups_per_segment: arm.groups, query_chunk_rows: arm.query_chunk,
            order: arm.order, warmup: false,
            host_wall_seconds: arm.host_wall_seconds,
            encode_cpu_seconds: arm.encode_host_elapsed_seconds,
            encode_host_elapsed_seconds: arm.encode_host_elapsed_seconds,
            encode_host_elapsed_semantics: "elapsed host wall time inside the synchronous encode/enqueue call; not process CPU time",
            command_chain_setup_elapsed_seconds: arm.command_chain_setup_elapsed_seconds,
            terminal_wait_elapsed_seconds: arm.terminal_wait_elapsed_seconds,
            monotonic_begin_seconds: arm.monotonic_begin_seconds,
            monotonic_end_seconds: arm.monotonic_end_seconds,
            observations: WanHostObservation.between(arm.observation_before, arm.observation_after),
            gpu_elapsed_seconds: arm.gpu_elapsed_seconds, command_segments: arm.segments,
            terminal_waits: arm.terminal_waits,
            encode_metal_buffer_allocations: arm.encode_metal_buffer_allocations,
            comparison_mode: comparison_mode,
            reuse_block_scratch: reuse_block_scratch,
            block_scratch_layout: reuse_block_scratch ? "lifetime-aliased" : "unaliased",
            block_scratch_bytes: block_scratch_bytes,
            default_block_scratch_bytes: default_block_scratch_bytes,
            block_scratch_bytes_saved: block_scratch_bytes_saved,
            kv_layout: arm_uses_prepared_kv ? "B,H,K,D" : "B,K,H,D",
            baseline_query_chunk_rows: baseline_chunk, candidate_query_chunk_rows: candidate_chunk,
            query_chunk_rows_equal_between_arms: baseline_chunk == candidate_chunk,
            layout_preparation_inside_block_encode: arm_uses_prepared_kv,
            prepared_kv_buffer_bytes: arm_uses_prepared_kv ? prepared_kv_bytes : 0_i64,
            prepared_kv_pair_shared_between_arms: prepared_kv_for_both_arms,
            prepared_kv_buffer_bytes_refreshed_per_encode: arm_uses_prepared_kv ? prepared_kv_bytes : 0_i64,
            prepared_kv_requested: prepared_kv_requested,
            self_attention_backend: arm_uses_explicit_qk_softmax_pv ? candidate_self_attention_backend : baseline_self_attention_backend,
            cross_attention_backend: "native_tiled",
            explicit_qk_softmax_pv: arm_uses_explicit_qk_softmax_pv,
            fuse_kv_preparation: arm_uses_fused_kv_preparation,
            precompute_rope: compare_rope && !baseline,
            rope_coefficient_build_inside_block_encode: compare_rope && !baseline,
            phase_root_diagnostics_enabled: arm.phase_root_profile != nil,
            phase_root_profile: arm.phase_root_profile.try { |profile| phase_root_profile_json(profile) },
            api_profile_diagnostics_enabled: arm.api_profile != nil,
            mps_api_profile: arm.api_profile.try { |profile| mps_api_profile_json(profile) },
            profile_replay_is_not_a_speed_comparison: compare_profile_replay,
            inter_operator_host_waits: 0, activation_readbacks: 0}.to_json)
      STDOUT.flush
    end
    if compare_profile_replay
      unless baseline_root_count == candidate_root_count && baseline_root_count == warm_baseline.segments
        raise "Profile-replay command-root counts differ"
      end
    end
    comparison = validate_pair(output_baseline, output_candidate, metrics, zero, compare)
    puts({schema: "wan_s2v_full_block_comparison_v1", pair: pair,
          compared_values: FULL_VALUES, bit_differences: comparison[:bit_differences],
          tolerance_violations: comparison[:tolerance_violations], nonfinite: comparison[:nonfinite],
          max_abs: comparison[:max_abs], max_normalized_error: comparison[:max_ratio],
          tolerance: "2e-5+2e-5*abs(reference)", exact_equal: comparison[:bit_differences] == 0,
          strict_pass: comparison[:tolerance_violations] == 0 && comparison[:nonfinite] == 0,
          comparison_mode: comparison_mode, exact_required: require_exact,
          reuse_block_scratch: reuse_block_scratch,
          block_scratch_layout: reuse_block_scratch ? "lifetime-aliased" : "unaliased",
          block_scratch_bytes: block_scratch_bytes,
          default_block_scratch_bytes: default_block_scratch_bytes,
          block_scratch_bytes_saved: block_scratch_bytes_saved,
          baseline_kv_layout: prepared_kv_for_both_arms ? "B,H,K,D" : "B,K,H,D",
          candidate_kv_layout: prepared_kv_for_candidate ? "B,H,K,D" : "B,K,H,D",
          layout_preparation_inside_block_encode: prepared_kv_for_candidate,
          prepared_kv_requested: prepared_kv_requested,
          prepared_kv_refreshed_inside_each_prepared_arm_encode: prepared_kv_bytes > 0,
          prepared_kv_pair_shared_between_arms: prepared_kv_for_both_arms,
          baseline_self_attention_backend: baseline_self_attention_backend,
          candidate_self_attention_backend: candidate_self_attention_backend,
          cross_attention_backend: "native_tiled",
          baseline_fuse_kv_preparation: false,
          candidate_fuse_kv_preparation: compare_kv_prep_fusion,
          baseline_explicit_qk_softmax_pv: false,
          candidate_explicit_qk_softmax_pv: compare_attention_graph,
          comparison_outside_timed_arms: true}.to_json)
    raise "Block output comparison failed" unless (!require_exact || comparison[:bit_differences] == 0) && comparison[:tolerance_violations] == 0 && comparison[:nonfinite] == 0
    STDOUT.flush
  end

  control = ML::Metal::CommandBuffer.new
  control_encoder = ML::Metal::ComputeEncoder.new(control)
  # Establish identical GPU backings before the finite/NaN negative control;
  # different query chunks need only pass the existing strict numerical gate.
  ML::Video::MetalOps.copy(control_encoder, output_candidate, output_baseline,
    FULL_VALUES.to_i32)
  control_encoder.set_pipeline(corrupt).set_buffer(output_candidate, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  control_encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write)
    .set_value(5_u32, 1).dispatch_1d(5)
  control_encoder.set_pipeline(compare).set_buffer(output_baseline, 0).set_buffer(output_candidate, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(FULL_VALUES.to_u32, 3).dispatch_1d(FULL_VALUES.to_i32)
  control_encoder.end_encoding
  control.commit_and_wait
  control_metrics = metrics.contents.as(UInt32*)
  control_rejected = control_metrics[0] == 2 && control_metrics[1] == 2 && control_metrics[2] == 1
  puts({schema: "wan_s2v_full_block_corruption_control_v1", injected_changes: 2,
        bit_differences: control_metrics[0], tolerance_violations: control_metrics[1],
        nonfinite: control_metrics[2], rejected: control_rejected,
        control_contains_finite_tolerance_failure: true,
        control_contains_nonfinite_failure: true,
        control_outside_timed_arms: true, control_uses_existing_single_wait: true}.to_json)
  raise "Full-block comparator failed its finite/non-finite corruption control" unless control_rejected
ensure
  candidate_attention_workspace.try(&.release)
  attention_workspace.try(&.release)
  projection_workspace.try(&.release_after_terminal_wait!)
  projection_stage.try(&.release_after_terminal_wait!)
  prepared.try(&.release)
  weight_arena.try(&.release)
  audio_scratch.try(&.release)
  scratch.try(&.release)
  inputs.try(&.release)
  checkpoint.close
end
