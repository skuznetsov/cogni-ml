# Bounded native-only profile of one real Hy-Nomas block-0 MPS forward.
# This is a block-cost and same-run-repeatability probe, not video generation.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/hy_nomas_block_mps"
require "../src/ml/video/hy_nomas_block_profile"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private HY_BLOCK_SCHEMA             = "hy_nomas_block_benchmark_v1"
private HY_BLOCK_DIAGNOSTIC_SCHEMA  = "hy_nomas_block_diagnostic_v1"
private HY_BLOCK_FAILURE_SCHEMA     = "hy_nomas_block_failure_v1"
private HY_BLOCK_BATCHES            =     1
private HY_BLOCK_VIDEO_TOKENS       = 5_280
private HY_BLOCK_TEXT_TOKENS        = 1_280
private HY_BLOCK_DIM                = 2_048
private HY_BLOCK_HEADS              =    16
private HY_BLOCK_HEAD_DIM           =   128
private HY_BLOCK_FFN                = 8_192
private HY_BLOCK_JOINT_TOKENS       = HY_BLOCK_VIDEO_TOKENS + HY_BLOCK_TEXT_TOKENS
private HY_BLOCK_QUERY_CHUNK        = 256
private HY_BLOCK_HEADS_PER_GROUP    = HY_BLOCK_HEADS
private HY_BLOCK_MEMORY_LIMIT_BYTES = 4_i64 * 1024 * 1024 * 1024
private HY_BLOCK_SCORE_TILES        = 3_i64
private HY_BLOCK_OPAQUE_RESERVE     = 256_i64 * 1024 * 1024
private HY_BLOCK_SOURCE_BYTES       = 604_231_456_i64
private HY_BLOCK_SOURCE_SHA256      = "6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1"
private HY_BLOCK_DEFAULT_WEIGHTS    = "/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/weights/distilled-block0-f32.safetensors"
private HY_BLOCK_FILL_FORMULA       = "avalanche32(i + salt*0x9e3779b9); ((hash & 65535) - 32768) / 65536 * scale"
private HY_BLOCK_ROPE_FORMULA       = "synthetic unit-circle RoPE; position=video token, theta=10000, full 128-channel table; not upstream theta=256 axes=[16,56,56] grid=[6,22,40]"
private HY_BLOCK_RUN_QUALIFICATION  = "performance-only at J6560 with deterministic synthetic inputs and synthetic RoPE; no upstream full-block output or model-quality claim"
private HY_BLOCK_PROFILE_ORDER      = [true, false, false, true]

private record HyNomasBlockSample,
  profile_enabled : Bool,
  group : Int32,
  position : Int32,
  encode_wall_seconds : Float64,
  terminal_wait_wall_seconds : Float64,
  total_wall_seconds : Float64,
  gpu_root_duration_sum_seconds : Float64,
  root_union_seconds : Float64,
  root_envelope_seconds : Float64,
  root_count : Int32

private def hy_block_emit(record) : Nil
  STDOUT.puts(record.to_json)
  STDOUT.flush
end

private def hy_block_median(samples : Array(Float64)) : Float64
  raise ArgumentError.new("Cannot calculate median of an empty sample") if samples.empty?
  ordered = samples.sort
  middle = ordered.size // 2
  ordered.size.odd? ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2.0
end

private def hy_block_diagnose(image_output : ML::MetalBuffer, text_output : ML::MetalBuffer,
                              img_reference : ML::MetalBuffer, txt_reference : ML::MetalBuffer,
                              metrics : ML::MetalBuffer,
                              zero_metrics : ML::Metal::ComputePipeline,
                              compare : ML::Metal::ComputePipeline,
                              group : Int32, position : Int32, arm : String,
                              retained_commands : Array(ML::Metal::CommandBuffer)) : Bool
  started = Time.instant
  command = ML::Metal::CommandBuffer.new
  retained_commands << command
  encoder = ML::Metal::ComputeEncoder.new(command)
  encoder.set_pipeline(zero_metrics)
    .set_buffer(metrics, 0, ML::Metal::BufferAccess::Write)
    .dispatch_1d(6)
  encoder.set_pipeline(compare)
    .set_buffer(image_output, 0).set_buffer(img_reference, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(img_reference.element_count.to_u32, 3)
    .dispatch_1d(img_reference.element_count)
  encoder.set_pipeline(compare)
    .set_buffer(text_output, 0).set_buffer(txt_reference, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(txt_reference.element_count.to_u32, 3)
    .dispatch_1d(txt_reference.element_count)
  encoder.end_encoding
  command.commit
  wait_started = Time.instant
  command.wait
  wait_wall_seconds = (Time.instant - wait_started).total_seconds
  total_wall_seconds = (Time.instant - started).total_seconds

  values = metrics.contents.as(UInt32*)
  passed = values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0
  hy_block_emit({
    schema:                                      HY_BLOCK_DIAGNOSTIC_SCHEMA,
    status:                                      passed ? "pass" : "fail",
    attention_precision:                         "BF16BoundaryControl",
    run_qualification:                           HY_BLOCK_RUN_QUALIFICATION,
    rope_formula:                                HY_BLOCK_ROPE_FORMULA,
    profile_group:                               group,
    profile_position:                            position,
    arm:                                         arm,
    comparison:                                  "this measured sample versus the first profile-off warmup output",
    qualification:                               "same-run repeatability only; not an upstream full-block oracle",
    compared_values:                             img_reference.element_count + txt_reference.element_count,
    gpu_metric_words:                            6,
    diagnostic_metric_bytes_read:                6_i64 * sizeof(UInt32),
    raw_f32_bit_mismatches:                      values[0],
    full_output_exact_raw_f32_bits:              values[0] == 0,
    nonfinite_values:                            values[1],
    tolerance_violations:                        values[2],
    bf16_rne_bin_mismatches:                     values[3],
    max_absolute_error:                          values[4].unsafe_as(Float32),
    max_tolerance_ratio:                         values[5].unsafe_as(Float32),
    tolerance:                                   "2e-5 + 2e-5*abs(reference)",
    bf16_rounding:                               "round-to-nearest-even",
    host_wall_including_diagnostic_wait_seconds: total_wall_seconds,
    diagnostic_terminal_wait_seconds:            wait_wall_seconds,
    diagnostic_terminal_waits:                   1,
    diagnostic_wait_excluded_from_block_timings: true,
    upstream_fixture_oracle_used:                false,
  })
  passed
end

private def hy_block_sample(block : ML::Video::HyNomasDualStreamBlock,
                            workspace : ML::Video::HyNomasMPSWorkspace,
                            chain_img : ML::MetalBuffer, chain_txt : ML::MetalBuffer,
                            vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                            rope_cos : ML::MetalBuffer, rope_sin : ML::MetalBuffer,
                            profile_enabled : Bool, group : Int32, position : Int32,
                            named_bytes : Int64, score_reserve_bytes : Int64,
                            retained_chains : Array(ML::Metal::MPSCommandChain))
  chain_setup_started = Time.instant
  chain = ML::Metal::MPSCommandChain.new
  retained_chains << chain
  chain_setup_wall_seconds = (Time.instant - chain_setup_started).total_seconds
  marks = [] of ML::Video::HyNomasBlockPhaseMarker
  started = Time.instant
  observer = ->(name : String) : Nil {
    observed_roots = chain.observed_root_count
    elapsed = (Time.instant - started).total_seconds
    marks << ML::Video::HyNomasBlockPhaseMarker.new(name, elapsed, observed_roots)
  }

  begin
    allocations_before = ML::MetalBuffer.video_spec_allocations
    block.encode_mps(chain, chain_img, chain_txt, vec, vec_txt,
      rope_cos, rope_sin, workspace,
      phase_observer: profile_enabled ? observer : nil)
    encode_wall_seconds = (Time.instant - started).total_seconds
    allocations_during_encode = ML::MetalBuffer.video_spec_allocations - allocations_before
    raise "encode allocated #{allocations_during_encode} tracked MetalBuffers" unless allocations_during_encode == 0
    raise "profile callback omitted block boundaries" if profile_enabled && marks.size != 9
    raise "profile callback order changed" if profile_enabled && marks.map(&.name) !=
                                                ["validate", "modulation", "qkv", "joint_pack", "attention",
                                                 "attention_projection_residual", "image_ffn", "text_ffn", "complete"]

    wait_started = Time.instant
    chain.commit_and_wait
    terminal_wait_wall_seconds = (Time.instant - wait_started).total_seconds
    total_wall_seconds = (Time.instant - started).total_seconds
    raise "Block sample did not use exactly one terminal wait" unless chain.wait_count == 1
    raise "Block sample did not cross the segmented MPS execution path" unless chain.segment_count > 0
    roots = chain.completed_root_timings
    block_profile = if profile_enabled
                      ML::Video::HyNomasBlockPhaseProfile.build(
                        marks, roots, chain.gpu_elapsed_seconds)
                    else
                      nil
                    end
    root_profile = block_profile.try(&.root_profile) ||
                   ML::Video::MPSCommandPhaseProfile.build(roots,
                     [ML::Video::MPSCommandPhaseRange.new("whole_block", 0, roots.size.to_i32)],
                     chain.gpu_elapsed_seconds)
    phase_root_candidates = if profile = block_profile
                              profile.root_candidates
                            else
                              Array(Array(String)).new(roots.size) { ["whole_block"] }
                            end

    phase_root_rows = roots.map_with_index do |root, sequence|
      {sequence: root.sequence, gpu_start_seconds: root.gpu_start_seconds,
       gpu_end_seconds: root.gpu_end_seconds,
       duration_seconds: root.gpu_end_seconds - root.gpu_start_seconds,
       phase_candidates: phase_root_candidates[sequence]}
    end
    sample = HyNomasBlockSample.new(profile_enabled, group, position,
      encode_wall_seconds, terminal_wait_wall_seconds, total_wall_seconds,
      root_profile.root_duration_seconds, root_profile.union_seconds,
      root_profile.envelope_seconds, roots.size.to_i32)
    hy_block_emit({
      schema:                                          HY_BLOCK_SCHEMA,
      phase:                                           "sample",
      status:                                          "ok",
      arm:                                             profile_enabled ? "profile_on" : "profile_off",
      profile_enabled:                                 profile_enabled,
      profile_group:                                   group,
      profile_order:                                   "ABBA",
      profile_position:                                position,
      batches:                                         HY_BLOCK_BATCHES,
      video_tokens:                                    HY_BLOCK_VIDEO_TOKENS,
      text_tokens:                                     HY_BLOCK_TEXT_TOKENS,
      joint_tokens:                                    HY_BLOCK_JOINT_TOKENS,
      dim:                                             HY_BLOCK_DIM,
      heads:                                           HY_BLOCK_HEADS,
      head_dim:                                        HY_BLOCK_HEAD_DIM,
      ffn_dim:                                         HY_BLOCK_FFN,
      attention_query_chunk:                           HY_BLOCK_QUERY_CHUNK,
      attention_heads_per_group:                       HY_BLOCK_HEADS_PER_GROUP,
      resident_weight_dtype:                           "BF16",
      source_weight_dtype:                             "F32",
      attention_precision:                             "BF16BoundaryControl",
      run_qualification:                               HY_BLOCK_RUN_QUALIFICATION,
      named_buffers_and_plans_bytes:                   named_bytes,
      total_host_wall_seconds_including_terminal_wait: total_wall_seconds,
      chain_setup_wall_seconds:                        chain_setup_wall_seconds,
      encode_wall_seconds:                             encode_wall_seconds,
      terminal_wait_wall_seconds:                      terminal_wait_wall_seconds,
      native_gpu_root_duration_sum_seconds:            chain.gpu_elapsed_seconds,
      gpu_root_duration_sum_seconds:                   root_profile.root_duration_seconds,
      gpu_root_union_seconds:                          root_profile.union_seconds,
      gpu_root_envelope_seconds:                       root_profile.envelope_seconds,
      score_tile_reserve_bytes:                        score_reserve_bytes,
      opaque_mpsgraph_reserve_heuristic_bytes:         HY_BLOCK_OPAQUE_RESERVE,
      root_count:                                      roots.size,
      segments:                                        chain.segment_count,
      terminal_waits:                                  chain.wait_count,
      tracked_metal_buffer_allocations_during_encode:  allocations_during_encode,
      phase_marks:                                     marks.map do |mark|
        {name: mark.name, elapsed_seconds: mark.elapsed_seconds,
         observed_root_count: mark.root_count}
      end,
      phase_cpu_seconds:         block_profile.try(&.phase_cpu_seconds),
      phase_cpu_timing_limit:    "profile-on CPU intervals include observer timestamp/root-count overhead; profile-off is the clean encode baseline",
      phase_root_candidates:     phase_root_candidates,
      root_timing_records:       phase_root_rows,
      profile_attribution_limit: "phase root lists are inclusive candidates; shared roots can span semantic boundaries; no exact per-phase GPU costs",
    })
    chain.release
    sample
  ensure
    unless chain.completed?
      begin
        chain.commit_and_wait
      rescue ex
        hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "sample_cleanup",
                       group: group, position: position,
                       error: ex.message || ex.class.name})
      end
    end
    chain.release if chain.completed?
  end
end

private HY_BLOCK_KERNELS = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline float hy_block_input(uint i, uint salt, float scale) {
    uint mixed = i + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int signed_value = int(mixed & 65535u) - 32768;
    return (float(signed_value) / 65536.0f) * scale;
  }

  kernel void hy_nomas_block_benchmark_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &salt [[buffer(2)]], constant float &scale [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = hy_block_input(i, salt, scale);
  }

  kernel void hy_nomas_block_benchmark_rope(
      device float *cosine [[buffer(0)]], device float *sine [[buffer(1)]],
      constant uint &count [[buffer(2)]], constant uint &head_dim [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint token = i / head_dim;
    const uint channel = i % head_dim;
    const uint pair = channel / 2u;
    const float frequency = pow(10000.0f, -2.0f * float(pair) / float(head_dim));
    const float angle = float(token) * frequency;
    cosine[i] = cos(angle);
    sine[i] = sin(angle);
  }

  kernel void hy_nomas_block_benchmark_copy(
      device const float *source [[buffer(0)]], device float *destination [[buffer(1)]],
      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) destination[i] = source[i];
  }

  kernel void hy_nomas_block_benchmark_zero_metrics(
      device atomic_uint *metrics [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < 6u) atomic_store_explicit(metrics + i, 0u, memory_order_relaxed);
  }

  inline uint hy_block_bf16_rne(float value) {
    const uint bits = as_type<uint>(value);
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
  }

  kernel void hy_nomas_block_benchmark_compare(
      device const float *actual [[buffer(0)]], device const float *reference [[buffer(1)]],
      device atomic_uint *metrics [[buffer(2)]], constant uint &count [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float got = actual[i], ref = reference[i];
    if (as_type<uint>(got) != as_type<uint>(ref))
      atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
    if (!isfinite(got) || !isfinite(ref)) {
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
      return;
    }
    const float error = abs(got - ref);
    const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
    const float ratio = error / tolerance;
    if (ratio > 1.0f) atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
    if (hy_block_bf16_rne(got) != hy_block_bf16_rne(ref))
      atomic_fetch_add_explicit(metrics + 3, 1u, memory_order_relaxed);
    if (error > 0.0f) {
      atomic_fetch_max_explicit(metrics + 4, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 5, as_type<uint>(ratio), memory_order_relaxed);
    }
  }
METAL

weights_path = ENV["HY_NOMAS_BLOCK_WEIGHTS"]? || HY_BLOCK_DEFAULT_WEIGHTS
groups = 1
plan_only = false
failure_message = nil.as(String?)
input_arena = nil.as(ML::Video::Arena?)
weight_arena = nil.as(ML::Video::Arena?)
scratch_arena = nil.as(ML::Video::Arena?)
prepared = nil.as(ML::Video::PreparedWeights?)
block = nil.as(ML::Video::HyNomasDualStreamBlock?)
workspace = nil.as(ML::Video::HyNomasMPSWorkspace?)
source = nil.as(ML::Video::CheckpointFile?)
retained_chains = [] of ML::Metal::MPSCommandChain
retained_commands = [] of ML::Metal::CommandBuffer

begin
  OptionParser.parse do |parser|
    parser.banner = "Usage: crystal run examples/hy_nomas_block_benchmark.cr -- [--weights=PATH] [--groups=N] [--preflight-only]"
    parser.on("--weights=PATH", "Block-0 SafeTensors source; must match the pinned SHA256") { |value| weights_path = value }
    parser.on("--groups=N", "ABBA profile-on/off groups, each with four measured samples (default 1)") { |value| groups = value.to_i }
    parser.on("--preflight-only", "Verify source identity and bounded memory preflight without initializing Metal or allocating GPU buffers") { plan_only = true }
    parser.on("--plan-only", "Alias for --preflight-only") { plan_only = true }
  end
  raise ArgumentError.new("--groups must be positive") unless groups > 0
  raise ArgumentError.new("Source checkpoint is missing: #{weights_path}") unless File.file?(weights_path)
  file_size = File.size(weights_path).to_i64
  raise ArgumentError.new("Pinned source checkpoint size mismatch: #{file_size}") unless file_size == HY_BLOCK_SOURCE_BYTES
  source_sha256 = Digest::SHA256.new.file(weights_path).hexfinal
  raise ArgumentError.new("Pinned source checkpoint SHA256 mismatch: #{source_sha256}") unless source_sha256 == HY_BLOCK_SOURCE_SHA256

  config = ML::Video::HyNomasConfig.new(
    batches: HY_BLOCK_BATCHES,
    video_tokens: HY_BLOCK_VIDEO_TOKENS,
    text_tokens: HY_BLOCK_TEXT_TOKENS,
    dim: HY_BLOCK_DIM,
    heads: HY_BLOCK_HEADS,
    head_dim: HY_BLOCK_HEAD_DIM,
    ffn_dim: HY_BLOCK_FFN,
    attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl,
  )
  config.validate!
  source_file = ML::Video::CheckpointFile.new(weights_path)
  source = source_file
  unless source_file.tensors.size == 32 && source_file.tensors.keys.all?(&.starts_with?("double_blocks.0."))
    raise ArgumentError.new("Pinned source checkpoint must contain exactly the 32 block-0 tensors")
  end
  unless source_file.tensors.values.all?(&.dtype.f32?)
    raise ArgumentError.new("Pinned block-0 checkpoint tensors must all be F32")
  end
  scratch_bytes = ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum(0_i64) do |(_name, count)|
    count.to_i64 * sizeof(Float32)
  end
  input_elements = config.image_rows.to_i64 * config.dim + config.text_rows.to_i64 * config.dim +
                   config.effective_img_modulation_rows.to_i64 * config.dim +
                   config.effective_txt_modulation_rows.to_i64 * config.dim +
                   config.video_tokens.to_i64 * config.head_dim * 2
  input_bytes = input_elements * sizeof(Float32)
  reference_bytes = (config.image_rows.to_i64 + config.text_rows) * config.dim * sizeof(Float32)
  diagnostic_bytes = 6_i64 * sizeof(UInt32)
  compact_shape = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
    config.batches, config.joint_tokens, config.heads, config.head_dim,
    HY_BLOCK_QUERY_CHUNK, HY_BLOCK_HEADS_PER_GROUP)
  weight_stage_bytes = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * sizeof(Float32)
  rms_image_rows = config.image_rows.to_i64 * config.heads
  rms_text_rows = config.text_rows.to_i64 * config.heads
  rms_image_bytes = (rms_image_rows * config.head_dim + rms_image_rows) * sizeof(Float32)
  rms_text_bytes = (rms_text_rows * config.head_dim + rms_text_rows) * sizeof(Float32)
  rms_bytes = 2_i64 * (rms_image_bytes + rms_text_bytes)
  workspace_bytes = weight_stage_bytes + compact_shape.resident_bytes + rms_bytes
  named_reserve_bytes = scratch_bytes + input_bytes + reference_bytes + diagnostic_bytes + workspace_bytes
  score_reserve_bytes = HY_BLOCK_SCORE_TILES * compact_shape.max_score_elements * sizeof(Float32)
  available_checkpoint_budget = HY_BLOCK_MEMORY_LIMIT_BYTES - score_reserve_bytes - HY_BLOCK_OPAQUE_RESERVE
  raise ArgumentError.new("Score and opaque reserves exceed the 4 GiB preflight budget") if available_checkpoint_budget <= 0
  checkpoint_plan = ML::Video::HyNomasDualStreamBlock.weight_plan(
    [source_file], config, available_checkpoint_budget, reserve_bytes: named_reserve_bytes,
    resident_dtype: ML::Video::WeightDType::BF16, block_index: 0, require_all: true,
    allow_f32_to_bf16: true)
  estimated_peak_bytes = checkpoint_plan.peak_bytes + score_reserve_bytes + HY_BLOCK_OPAQUE_RESERVE
  preflight = {
    schema:                                     "hy_nomas_block_preflight_v1",
    status:                                     estimated_peak_bytes <= HY_BLOCK_MEMORY_LIMIT_BYTES ? "geometry_fits_reserve_estimate" : "geometry_exceeds_reserve_estimate",
    source_checkpoint:                          File.basename(weights_path),
    source_checkpoint_bytes:                    file_size,
    source_checkpoint_sha256:                   source_sha256,
    source_checkpoint_tensors:                  source_file.tensors.size,
    resident_weight_dtype:                      "BF16",
    blocks:                                     1,
    batches:                                    config.batches,
    video_tokens:                               config.video_tokens,
    text_tokens:                                config.text_tokens,
    joint_tokens:                               config.joint_tokens,
    dim:                                        config.dim,
    heads:                                      config.heads,
    head_dim:                                   config.head_dim,
    ffn_dim:                                    config.ffn_dim,
    attention_query_chunk:                      HY_BLOCK_QUERY_CHUNK,
    attention_heads_per_group:                  HY_BLOCK_HEADS_PER_GROUP,
    attention_precision:                        "BF16BoundaryControl",
    upstream_video_grid:                        [6, 22, 40],
    upstream_rope_axes:                         [16, 56, 56],
    upstream_rope_theta:                        256.0,
    synthetic_rope_used:                        true,
    input_rope_is_upstream_nd_rotary:           false,
    attention_graph_calls:                      compact_shape.graph_calls,
    max_score_bytes_per_graph:                  compact_shape.max_score_elements * sizeof(Float32),
    scratch_bytes:                              scratch_bytes,
    synthetic_input_bytes:                      input_bytes,
    repeatability_reference_bytes:              reference_bytes,
    diagnostic_bytes:                           diagnostic_bytes,
    weight_stage_bytes:                         weight_stage_bytes,
    compact_attention_stages_bytes:             compact_shape.resident_bytes,
    rms_plan_scratch_bytes:                     rms_bytes,
    bf16_resident_weights_bytes:                checkpoint_plan.destination_bytes,
    f32_weight_conversion_staging_bytes:        checkpoint_plan.staging_bytes,
    explicitly_named_peak_bytes:                checkpoint_plan.peak_bytes,
    score_tile_multiplier:                      HY_BLOCK_SCORE_TILES,
    score_tile_reserve_bytes:                   score_reserve_bytes,
    opaque_mpsgraph_reserve_heuristic_bytes:    HY_BLOCK_OPAQUE_RESERVE,
    estimated_peak_bytes:                       estimated_peak_bytes,
    preflight_limit_bytes:                      HY_BLOCK_MEMORY_LIMIT_BYTES,
    fits_reserve_estimate:                      estimated_peak_bytes <= HY_BLOCK_MEMORY_LIMIT_BYTES,
    external_memory_guard_required:             true,
    external_guard_policy:                      "cache-aware free+filecache headroom >=15%; process-group footprint <=4 GiB; swap growth <=64 MiB; timeout 600 seconds",
    backend_private_allocation_cap_established: false,
  }
  hy_block_emit(preflight)
  raise ArgumentError.new("Block named buffers and reserves exceed the 4 GiB estimate") if estimated_peak_bytes > HY_BLOCK_MEMORY_LIMIT_BYTES
  if plan_only
    hy_block_emit({schema: "hy_nomas_block_plan_v1", status: "preflight_only", metal_initialized: false, gpu_buffers_allocated: false,
                   upstream_fixture_oracle_used: false, benchmark_selects_or_changes_no_default: true})
    source_file.close
    exit 0
  end
  raise ArgumentError.new("Hy-Nomas full-block benchmark requires macOS Metal") unless ML::Metal::Device.init!

  input_arena = ML::Video::Arena.new
  weight_arena = ML::Video::Arena.new
  scratch_arena = ML::Video::Arena.new
  input_count = ->(rows : Int32) { ML::Video::HyNomasDualStreamBlock.checked_count(rows, config.dim) }
  img = input_arena.not_nil!.alloc("img", input_count.call(config.image_rows))
  txt = input_arena.not_nil!.alloc("txt", input_count.call(config.text_rows))
  vec = input_arena.not_nil!.alloc("vec", input_count.call(config.effective_img_modulation_rows))
  vec_txt = input_arena.not_nil!.alloc("vec_txt", input_count.call(config.effective_txt_modulation_rows))
  rope_count = ML::Video::HyNomasDualStreamBlock.checked_count(config.video_tokens, config.head_dim)
  rope_cos = input_arena.not_nil!.alloc("video_rope_cos", rope_count)
  rope_sin = input_arena.not_nil!.alloc("video_rope_sin", rope_count)
  img_reference = input_arena.not_nil!.alloc("img_reference", input_count.call(config.image_rows))
  txt_reference = input_arena.not_nil!.alloc("txt_reference", input_count.call(config.text_rows))
  metrics = input_arena.not_nil!.alloc("repeatability_metrics", 6)
  ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch_arena.not_nil!, config)

  caller_named_bytes = input_arena.not_nil!.resident_bytes + scratch_arena.not_nil!.resident_bytes
  unless caller_named_bytes == input_bytes + scratch_bytes + reference_bytes + diagnostic_bytes
    raise "Allocated input and scratch bytes do not match the preflight"
  end
  weight_reserve = caller_named_bytes + workspace_bytes
  plan = ML::Video::HyNomasDualStreamBlock.weight_plan(
    [source_file], config, available_checkpoint_budget, reserve_bytes: weight_reserve,
    resident_dtype: ML::Video::WeightDType::BF16, block_index: 0, require_all: true,
    allow_f32_to_bf16: true)
  unless plan.destination_bytes == checkpoint_plan.destination_bytes &&
         plan.staging_bytes == checkpoint_plan.staging_bytes && plan.peak_bytes == checkpoint_plan.peak_bytes
    raise "Checkpoint memory plan changed after caller buffers were allocated"
  end
  prepared = ML::Video::PreparedWeights.new(plan, weight_arena.not_nil!)
  block = ML::Video::HyNomasDualStreamBlock.new(config, weight_arena.not_nil!, scratch_arena.not_nil!)
  workspace = ML::Video::HyNomasMPSWorkspace.new(block.not_nil!, img, txt, vec, vec_txt,
    rope_cos, rope_sin, attention_query_chunk: HY_BLOCK_QUERY_CHUNK,
    attention_heads_per_group: HY_BLOCK_HEADS_PER_GROUP)
  unless workspace.not_nil!.resident_bytes == workspace_bytes
    raise "MPS workspace residency differs from preflight: #{workspace.not_nil!.resident_bytes} != #{workspace_bytes}"
  end

  fill = ML::Metal::ComputePipeline.new("hy_nomas_block_benchmark_fill", HY_BLOCK_KERNELS)
  rope = ML::Metal::ComputePipeline.new("hy_nomas_block_benchmark_rope", HY_BLOCK_KERNELS)
  copy = ML::Metal::ComputePipeline.new("hy_nomas_block_benchmark_copy", HY_BLOCK_KERNELS)
  zero_metrics = ML::Metal::ComputePipeline.new("hy_nomas_block_benchmark_zero_metrics", HY_BLOCK_KERNELS)
  compare = ML::Metal::ComputePipeline.new("hy_nomas_block_benchmark_compare", HY_BLOCK_KERNELS)

  setup_started = Time.instant
  setup = ML::Metal::CommandBuffer.new
  retained_commands << setup
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  prepared.not_nil!.encode(setup_encoder)
  { {img, 1_u32, 0.20_f32}, {txt, 2_u32, 0.20_f32},
   {vec, 3_u32, 0.10_f32}, {vec_txt, 4_u32, 0.10_f32} }.each do |buffer, salt, scale|
    setup_encoder.set_pipeline(fill)
      .set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
      .set_value(buffer.element_count.to_u32, 1)
      .set_value(salt, 2)
      .set_value(scale, 3)
      .dispatch_1d(buffer.element_count)
  end
  setup_encoder.set_pipeline(rope)
    .set_buffer(rope_cos, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(rope_sin, 1, ML::Metal::BufferAccess::Write)
    .set_value(rope_count.to_u32, 2)
    .set_value(HY_BLOCK_HEAD_DIM.to_u32, 3)
    .dispatch_1d(rope_count)
  setup_encoder.end_encoding
  setup.commit
  setup.wait
  setup_wall_seconds = (Time.instant - setup_started).total_seconds
  prepared.not_nil!.release
  prepared = nil
  hy_block_emit({schema: "hy_nomas_block_input_setup_v1", status: "ok",
                 formula: HY_BLOCK_FILL_FORMULA,
                 salts: {img: 1, txt: 2, vec: 3, vec_txt: 4},
                 rope_formula: HY_BLOCK_ROPE_FORMULA,
                 attention_precision: "BF16BoundaryControl",
                 run_qualification: HY_BLOCK_RUN_QUALIFICATION,
                 gpu_generated_input_elements: input_elements,
                 source_weights_converted_f32_to_bf16_on_gpu: true,
                 setup_host_wall_including_terminal_wait_seconds: setup_wall_seconds,
                 setup_terminal_waits: 1, setup_outside_measured_samples: true})

  named_bytes = caller_named_bytes + weight_arena.not_nil!.resident_bytes + workspace_bytes
  warmup = hy_block_sample(block.not_nil!, workspace.not_nil!, img, txt, vec, vec_txt,
    rope_cos, rope_sin, false, 0, 0, named_bytes, score_reserve_bytes, retained_chains)
  capture_started = Time.instant
  capture = ML::Metal::CommandBuffer.new
  retained_commands << capture
  capture_encoder = ML::Metal::ComputeEncoder.new(capture)
  image_output = block.not_nil!.image_output
  text_output = block.not_nil!.text_output
  capture_encoder.set_pipeline(copy).set_buffer(image_output, 0)
    .set_buffer(img_reference, 1, ML::Metal::BufferAccess::Write)
    .set_value(img_reference.element_count.to_u32, 2)
    .dispatch_1d(img_reference.element_count)
  capture_encoder.set_pipeline(copy).set_buffer(text_output, 0)
    .set_buffer(txt_reference, 1, ML::Metal::BufferAccess::Write)
    .set_value(txt_reference.element_count.to_u32, 2)
    .dispatch_1d(txt_reference.element_count)
  capture_encoder.end_encoding
  capture.commit
  capture.wait
  hy_block_emit({schema: "hy_nomas_block_reference_capture_v1", status: "ok",
                 capture_host_wall_including_terminal_wait_seconds: (Time.instant - capture_started).total_seconds,
                 reference: "first warmup output on same weights and deterministic inputs; not an upstream fixture oracle",
                 terminal_waits: 1})

  samples = [] of HyNomasBlockSample
  diagnostics_passed = true
  groups.times do |group_index|
    HY_BLOCK_PROFILE_ORDER.each_with_index do |profile_enabled, position|
      sample = hy_block_sample(block.not_nil!, workspace.not_nil!, img, txt, vec, vec_txt,
        rope_cos, rope_sin, profile_enabled, group_index + 1, position + 1,
        named_bytes, score_reserve_bytes, retained_chains)
      unless sample.root_count == warmup.root_count
        raise "Measured MPS root count changed from the warmup: #{sample.root_count} != #{warmup.root_count}"
      end
      samples << sample
      arm = profile_enabled ? "profile_on" : "profile_off"
      sample_passed = hy_block_diagnose(image_output, text_output, img_reference,
        txt_reference, metrics, zero_metrics, compare, group_index + 1,
        position + 1, arm, retained_commands)
      diagnostics_passed &&= sample_passed
      raise "Same-run diagnostic failed for #{arm} group=#{group_index + 1} position=#{position + 1}" unless sample_passed
    end
  end

  profile_on = samples.select(&.profile_enabled)
  profile_off = samples.reject(&.profile_enabled)
  hy_block_emit({
    schema:                                                      "hy_nomas_block_profile_summary_v1",
    status:                                                      diagnostics_passed ? "diagnostic_pass" : "diagnostic_fail",
    attention_precision:                                         "BF16BoundaryControl",
    run_qualification:                                           HY_BLOCK_RUN_QUALIFICATION,
    profile_order_per_group:                                     "ABBA (profile_on, profile_off, profile_off, profile_on)",
    groups:                                                      groups,
    warmup_root_count:                                           warmup.root_count,
    root_counts_equal_to_warmup_across_measured_arms:            true,
    profile_on_encode_wall_median_seconds:                       hy_block_median(profile_on.map(&.encode_wall_seconds)),
    profile_off_encode_wall_median_seconds:                      hy_block_median(profile_off.map(&.encode_wall_seconds)),
    profile_on_total_wall_median_seconds:                        hy_block_median(profile_on.map(&.total_wall_seconds)),
    profile_off_total_wall_median_seconds:                       hy_block_median(profile_off.map(&.total_wall_seconds)),
    profile_on_terminal_wait_median_seconds:                     hy_block_median(profile_on.map(&.terminal_wait_wall_seconds)),
    profile_off_terminal_wait_median_seconds:                    hy_block_median(profile_off.map(&.terminal_wait_wall_seconds)),
    profile_on_gpu_root_duration_median_seconds:                 hy_block_median(profile_on.map(&.gpu_root_duration_sum_seconds)),
    profile_off_gpu_root_duration_median_seconds:                hy_block_median(profile_off.map(&.gpu_root_duration_sum_seconds)),
    profile_on_gpu_root_union_median_seconds:                    hy_block_median(profile_on.map(&.root_union_seconds)),
    profile_off_gpu_root_union_median_seconds:                   hy_block_median(profile_off.map(&.root_union_seconds)),
    profile_on_gpu_root_envelope_median_seconds:                 hy_block_median(profile_on.map(&.root_envelope_seconds)),
    profile_off_gpu_root_envelope_median_seconds:                hy_block_median(profile_off.map(&.root_envelope_seconds)),
    warmup_encode_wall_seconds:                                  warmup.encode_wall_seconds,
    repeatability_diagnostic_passed:                             diagnostics_passed,
    diagnostic_samples_checked:                                  samples.size,
    diagnostic_waits_excluded_from_block_timings:                true,
    total_wall_includes_terminal_wait:                           true,
    profile_callback_cost_is_included_in_profile_on_encode_wall: true,
    profile_off_is_clean_encode_baseline:                        true,
    exact_repeatability_uses_raw_f32_bit_comparison:             true,
    root_union_and_envelope_are_not_per_phase_costs:             true,
    benchmark_selects_or_changes_no_default:                     true,
  })
rescue ex
  failure_message = ex.message || ex.class.name
  hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "benchmark",
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
      hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "setup_command_drain",
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
      hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "command_chain_drain",
                     error: ex.message || ex.class.name, resident_buffers_preserved: true})
    end
  end
  if cleanup_ok
    begin
      workspace.try(&.release_after_terminal_wait!)
      prepared.try(&.release)
      input_arena.try(&.release)
      scratch_arena.try(&.release)
      weight_arena.try(&.release)
    rescue ex
      cleanup_ok = false
      failure_message ||= ex.message || ex.class.name
      hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "resource_release",
                     error: ex.message || ex.class.name})
    end
  else
    failure_message ||= "GPU work could not be proven drained; resident resources remain retained"
    hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "cleanup",
                   error: "GPU work could not be proven drained or released; caller buffers remain retained"})
  end
  begin
    source.try(&.close)
  rescue ex
    cleanup_ok = false
    failure_message ||= ex.message || ex.class.name
    hy_block_emit({schema: HY_BLOCK_FAILURE_SCHEMA, phase: "checkpoint_close",
                   error: ex.message || ex.class.name})
  end
end

exit 1 if failure_message
