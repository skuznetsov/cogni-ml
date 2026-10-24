# Bounded native-only comparison of two serial applications of real block-0
# weights. This is a scratch/workspace benchmark, not video generation.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/hy_nomas_serial_mps"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private HY_SERIAL_SOURCE_BYTES  = 604_231_456_i64
private HY_SERIAL_SOURCE_SHA256 = "6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1"
private HY_SERIAL_WEIGHTS       = "/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/weights/distilled-block0-f32.safetensors"
private HY_SERIAL_MEMORY_LIMIT  = 4_i64 * 1024 * 1024 * 1024
private HY_SERIAL_SCORE_TILES   = 3_i64
private HY_SERIAL_OPAQUE_BYTES  = 256_i64 * 1024 * 1024
private HY_SERIAL_QUERY_CHUNK   = 256
private HY_SERIAL_HEADS_GROUP   =  16
private HY_SERIAL_FILL_FORMULA  = "avalanche32(i + salt*0x9e3779b9); ((hash & 65535) - 32768) / 65536 * scale"
private HY_SERIAL_ROPE_FORMULA  = "synthetic unit-circle RoPE; position=video token, theta=10000; not upstream theta=256 axes=[16,56,56] grid=[6,22,40]"
private HY_SERIAL_QUALIFICATION = "performance-only; two serial applications of block-0 weights with deterministic synthetic inputs and synthetic RoPE; not model generation or an upstream full-block oracle"

private class HyNomasBlockProbe
  SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;
    inline float probe_input(uint i, uint salt, float scale) {
      uint mixed = i + salt * 0x9e3779b9u;
      mixed ^= mixed >> 16; mixed *= 0x7feb352du;
      mixed ^= mixed >> 15; mixed *= 0x846ca68bu; mixed ^= mixed >> 16;
      const int value = int(mixed & 65535u) - 32768;
      return (float(value) / 65536.0f) * scale;
    }
    kernel void hy_nomas_serial_fill(device float *out [[buffer(0)]],
        constant uint &count [[buffer(1)]], constant uint &salt [[buffer(2)]],
        constant float &scale [[buffer(3)]], uint i [[thread_position_in_grid]]) {
      if (i < count) out[i] = probe_input(i, salt, scale);
    }
    kernel void hy_nomas_serial_rope(device float *c [[buffer(0)]],
        device float *s [[buffer(1)]], constant uint &count [[buffer(2)]],
        constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {
      if (i >= count) return;
      uint token = i / dim, channel = i % dim, pair = channel / 2u;
      float frequency = pow(10000.0f, -2.0f * float(pair) / float(dim));
      float angle = float(token) * frequency; c[i] = cos(angle); s[i] = sin(angle);
    }
    kernel void hy_nomas_serial_zero(device atomic_uint *m [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
      if (i == 0u) atomic_store_explicit(m, 0u, memory_order_relaxed);
    }
    kernel void hy_nomas_serial_finite(device const float *actual [[buffer(0)]],
        device atomic_uint *m [[buffer(1)]], constant uint &count [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
      if (i >= count) return;
      if (!isfinite(actual[i])) atomic_fetch_add_explicit(m, 1u, memory_order_relaxed);
    }
  METAL

  @fill = ML::Metal::ComputePipeline.new("hy_nomas_serial_fill", SOURCE)
  @rope = ML::Metal::ComputePipeline.new("hy_nomas_serial_rope", SOURCE)
  @zero = ML::Metal::ComputePipeline.new("hy_nomas_serial_zero", SOURCE)
  @finite = ML::Metal::ComputePipeline.new("hy_nomas_serial_finite", SOURCE)

  def encode_inputs(e : ML::Metal::ComputeEncoder, config : ML::Video::HyNomasConfig,
                    img : ML::MetalBuffer, txt : ML::MetalBuffer,
                    vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                    cos : ML::MetalBuffer, sin : ML::MetalBuffer) : Nil
    { {img, 1_u32, 0.20_f32}, {txt, 2_u32, 0.20_f32},
     {vec, 3_u32, 0.10_f32}, {vec_txt, 4_u32, 0.10_f32} }.each do |buffer, salt, scale|
      e.set_pipeline(@fill).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
        .set_value(buffer.element_count.to_u32, 1).set_value(salt, 2).set_value(scale, 3)
        .dispatch_1d(buffer.element_count)
    end
    count = ML::Video::HyNomasDualStreamBlock.checked_count(config.video_tokens, config.head_dim)
    e.set_pipeline(@rope).set_buffer(cos, 0, ML::Metal::BufferAccess::Write)
      .set_buffer(sin, 1, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 2).set_value(config.head_dim.to_u32, 3).dispatch_1d(count)
  end

  def encode_warmup_finiteness(encoder : ML::Metal::ComputeEncoder,
                               actual_img : ML::MetalBuffer, actual_txt : ML::MetalBuffer,
                               metrics : ML::MetalBuffer) : Nil
    encoder.set_pipeline(@zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(1)
    {actual_img, actual_txt}.each do |actual|
      encoder.set_pipeline(@finite).set_buffer(actual, 0)
        .set_buffer(metrics, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_value(actual.element_count.to_u32, 2).dispatch_1d(actual.element_count)
    end
  end

  def metrics_record(metrics : ML::MetalBuffer) : Hash(String, JSON::Any)
    words = metrics.contents.as(UInt32*)
    passed = words[0] == 0
    {
      "status"                              => JSON::Any.new(passed ? "pass" : "fail"),
      "nonfinite_values"                    => JSON::Any.new(words[0].to_i64),
      "scope"                               => JSON::Any.new("warmup GPU self-finiteness check only"),
      "diagnostic_appended_to_warmup_chain" => JSON::Any.new(true),
    }
  end
end

private def hy_serial_emit(record) : Nil
  STDOUT.puts(record.to_json)
  STDOUT.flush
end

private def hy_serial_sum_scratch(config : ML::Video::HyNomasConfig) : Int64
  ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum(0_i64) do |(_name, count)|
    count.to_i64 * sizeof(Float32)
  end
end

private def hy_serial_hash(buffer : ML::MetalBuffer) : String
  bytes = Slice.new(buffer.contents.as(UInt8*), buffer.size.to_i32)
  Digest::SHA256.new.update(bytes).hexfinal
end

private def hy_serial_median(values : Array(Float64)) : Float64
  ordered = values.sort
  middle = ordered.size // 2
  ordered.size.odd? ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2.0
end

private def hy_serial_encode(mode : String,
                             serial : ML::Video::HyNomasMPSSerialWorkspace?,
                             blocks : Array(ML::Video::HyNomasDualStreamBlock),
                             workspaces : Array(ML::Video::HyNomasMPSWorkspace),
                             chain : ML::Metal::MPSCommandChain,
                             img : ML::MetalBuffer, txt : ML::MetalBuffer,
                             vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                             cos : ML::MetalBuffer, sin : ML::MetalBuffer) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
  if mode == "shared"
    return serial.not_nil!.encode(chain)
  end
  next_img, next_txt = blocks[0].encode_mps(chain, img, txt, vec, vec_txt, cos, sin, workspaces[0])
  blocks[1].encode_mps(chain, next_img, next_txt, vec, vec_txt, cos, sin, workspaces[1])
end

private def hy_serial_sample(mode : String, index : Int32,
                             serial : ML::Video::HyNomasMPSSerialWorkspace?,
                             blocks : Array(ML::Video::HyNomasDualStreamBlock),
                             workspaces : Array(ML::Video::HyNomasMPSWorkspace),
                             img : ML::MetalBuffer, txt : ML::MetalBuffer,
                             vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                             cos : ML::MetalBuffer, sin : ML::MetalBuffer,
                             probe : HyNomasBlockProbe,
                             metrics : ML::MetalBuffer,
                             expected_hashes : Tuple(String, String)?,
                             retained : Array(ML::Metal::MPSCommandChain))
  chain = ML::Metal::MPSCommandChain.new
  retained << chain
  allocations_before = ML::MetalBuffer.video_spec_allocations
  started = Time.instant
  output_img, output_txt = hy_serial_encode(mode, serial, blocks, workspaces,
    chain, img, txt, vec, vec_txt, cos, sin)
  encode_wall = (Time.instant - started).total_seconds
  allocations = ML::MetalBuffer.video_spec_allocations - allocations_before
  raise "Encode allocated #{allocations} tracked MetalBuffers" unless allocations == 0
  # Warmup checks output finiteness only. Measured requests contain only both
  # block encodes and this chain's terminal wait.
  probe.encode_warmup_finiteness(chain.encoder, output_img, output_txt, metrics) unless expected_hashes
  raise "Unexpected wait before terminal commit" unless chain.wait_count == 0
  wait_started = Time.instant
  chain.commit_and_wait
  wait_wall = (Time.instant - wait_started).total_seconds
  total_wall = encode_wall + wait_wall
  raise "Sample must have exactly one terminal wait" unless chain.wait_count == 1
  raise "Sample did not use segmented MPS execution" unless chain.segment_count > 0
  roots = chain.completed_root_timings
  image_hash = hy_serial_hash(output_img)
  text_hash = hy_serial_hash(output_txt)
  if expected_hashes
    unless image_hash == expected_hashes[0] && text_hash == expected_hashes[1]
      raise "Measured raw F32 output hash changed from warmup"
    end
    diagnostic = {
      "status"                           => JSON::Any.new("pass"),
      "scope"                            => JSON::Any.new("hash exact repeatability; no GPU diagnostic in timed chain"),
      "gpu_diagnostic_in_measured_chain" => JSON::Any.new(false),
    }
  else
    diagnostic = probe.metrics_record(metrics)
    raise "Warmup GPU self-finiteness check failed" unless diagnostic["status"].as_s == "pass"
  end
  sample = {
    schema: "hy_nomas_serial_sample_v1", mode: mode, sample_index: index,
    status: "ok", run_qualification: HY_SERIAL_QUALIFICATION,
    encode_wall_seconds: encode_wall, terminal_wait_wall_seconds: wait_wall,
    total_wall_seconds: total_wall, native_gpu_duration_seconds: chain.gpu_elapsed_seconds,
    root_count: roots.size, segment_count: chain.segment_count, wait_count: chain.wait_count,
    root_count_is_observational: true,
    tracked_metal_buffer_allocations_during_encode: allocations,
    image_output_raw_f32_sha256: image_hash,
    text_output_raw_f32_sha256: text_hash,
    timings_include_only_block_encodes_and_one_terminal_wait: expected_hashes != nil,
    diagnostic: diagnostic,
  }
  chain.release
  sample
ensure
  if chain && !chain.completed?
    begin
      chain.commit_and_wait
    rescue
    end
  end
  chain.release if chain && chain.completed?
end

weights_path = ENV["HY_NOMAS_BLOCK_WEIGHTS"]? || HY_SERIAL_WEIGHTS
mode = "shared"
video_tokens = 768
text_tokens = 256
preflight_only = false
failure_message = nil.as(String?)
source = nil.as(ML::Video::CheckpointFile?)
input_arena = nil.as(ML::Video::Arena?)
weight_arena = nil.as(ML::Video::Arena?)
scratch_arenas = [] of ML::Video::Arena
prepared = nil.as(ML::Video::PreparedWeights?)
serial = nil.as(ML::Video::HyNomasMPSSerialWorkspace?)
blocks = [] of ML::Video::HyNomasDualStreamBlock
workspaces = [] of ML::Video::HyNomasMPSWorkspace
chains = [] of ML::Metal::MPSCommandChain
commands = [] of ML::Metal::CommandBuffer

begin
  OptionParser.parse do |parser|
    parser.banner = "Usage: crystal run examples/hy_nomas_serial_benchmark.cr -- [--mode=shared|separate] [--weights=PATH] [--video-tokens=N --text-tokens=N | --large] [--preflight-only]"
    parser.on("--mode=MODE", "shared scratch/workspace or separate block workspaces (default shared)") { |value| mode = value }
    parser.on("--weights=PATH", "Pinned block-0 SafeTensors source") { |value| weights_path = value }
    parser.on("--video-tokens=N", "Video token count (default 768)") { |value| video_tokens = value.to_i }
    parser.on("--text-tokens=N", "Text token count (default 256)") { |value| text_tokens = value.to_i }
    parser.on("--large", "Use video=5280 and text=1280; admitted only in shared mode") { video_tokens = 5280; text_tokens = 1280 }
    parser.on("--preflight-only", "Check pins and bounded named-memory estimate without Metal initialization") { preflight_only = true }
  end
  raise ArgumentError.new("--mode must be shared or separate") unless {"shared", "separate"}.includes?(mode)
  raise ArgumentError.new("Token counts must be positive") unless video_tokens > 0 && text_tokens > 0
  raise ArgumentError.new("--large is admitted only for --mode=shared") if mode == "separate" && video_tokens == 5280 && text_tokens == 1280
  raise ArgumentError.new("Source checkpoint is missing: #{weights_path}") unless File.file?(weights_path)
  source_size = File.size(weights_path).to_i64
  raise ArgumentError.new("Pinned source checkpoint size mismatch: #{source_size}") unless source_size == HY_SERIAL_SOURCE_BYTES
  source_sha = Digest::SHA256.new.file(weights_path).hexfinal
  raise ArgumentError.new("Pinned source checkpoint SHA256 mismatch: #{source_sha}") unless source_sha == HY_SERIAL_SOURCE_SHA256

  config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: video_tokens,
    text_tokens: text_tokens, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192,
    attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl)
  config.validate!
  source_file = ML::Video::CheckpointFile.new(weights_path)
  source = source_file
  unless source_file.tensors.size == 32 && source_file.tensors.keys.all?(&.starts_with?("double_blocks.0.")) &&
         source_file.tensors.values.all?(&.dtype.f32?)
    raise ArgumentError.new("Pinned checkpoint must contain exactly 32 F32 block-0 tensors")
  end

  base_scratch_bytes = hy_serial_sum_scratch(config)
  output_bytes = (config.image_rows.to_i64 + config.text_rows) * config.dim * sizeof(Float32)
  scratch_bytes = mode == "shared" ? base_scratch_bytes + output_bytes : 2_i64 * base_scratch_bytes
  input_elements = config.image_rows.to_i64 * config.dim + config.text_rows.to_i64 * config.dim +
                   config.effective_img_modulation_rows.to_i64 * config.dim + config.effective_txt_modulation_rows.to_i64 * config.dim +
                   2_i64 * config.video_tokens * config.head_dim
  input_bytes = input_elements * sizeof(Float32)
  reference_bytes = 0_i64
  diagnostic_bytes = sizeof(UInt32).to_i64
  compact = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
    config.batches, config.joint_tokens, config.heads, config.head_dim,
    HY_SERIAL_QUERY_CHUNK, HY_SERIAL_HEADS_GROUP)
  weight_stage_bytes = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * sizeof(Float32)
  image_rms_rows = config.image_rows.to_i64 * config.heads
  text_rms_rows = config.text_rows.to_i64 * config.heads
  rms_bytes = 2_i64 * ((image_rms_rows * config.head_dim + image_rms_rows) +
                       (text_rms_rows * config.head_dim + text_rms_rows)) * sizeof(Float32)
  workspaces_count = mode == "shared" ? 1_i64 : 2_i64
  one_workspace_bytes = weight_stage_bytes + compact.resident_bytes + rms_bytes
  mps_workspace_bytes = workspaces_count * one_workspace_bytes
  named_reserve_bytes = input_bytes + reference_bytes + diagnostic_bytes + scratch_bytes + mps_workspace_bytes
  score_multiplier = HY_SERIAL_SCORE_TILES * (mode == "shared" ? 1_i64 : 2_i64)
  score_reserve_bytes = score_multiplier * compact.max_score_elements * sizeof(Float32)
  available_budget = HY_SERIAL_MEMORY_LIMIT - score_reserve_bytes - HY_SERIAL_OPAQUE_BYTES
  raise ArgumentError.new("Score and opaque reserves exceed the 4 GiB preflight budget") if available_budget <= 0
  checkpoint_plan = ML::Video::HyNomasDualStreamBlock.weight_plan([source_file], config,
    available_budget, reserve_bytes: named_reserve_bytes, resident_dtype: ML::Video::WeightDType::BF16,
    block_index: 0, require_all: true, allow_f32_to_bf16: true)
  estimated_peak = checkpoint_plan.peak_bytes + score_reserve_bytes + HY_SERIAL_OPAQUE_BYTES
  hy_serial_emit({
    schema: "hy_nomas_serial_preflight_v1", status: estimated_peak <= HY_SERIAL_MEMORY_LIMIT ? "fits_reserve_estimate" : "exceeds_reserve_estimate",
    mode: mode, source_checkpoint: File.basename(weights_path), source_checkpoint_bytes: source_size,
    source_checkpoint_sha256: source_sha, source_checkpoint_tensors: source_file.tensors.size,
    source_weight_dtype: "F32", resident_weight_dtype: "BF16", blocks: 2,
    batches: config.batches, video_tokens: config.video_tokens, text_tokens: config.text_tokens,
    joint_tokens: config.joint_tokens, dim: config.dim, heads: config.heads,
    head_dim: config.head_dim, ffn_dim: config.ffn_dim,
    attention_precision: "BF16BoundaryControl", attention_query_chunk: HY_SERIAL_QUERY_CHUNK,
    attention_heads_per_group: HY_SERIAL_HEADS_GROUP, synthetic_rope_used: true,
    input_bytes: input_bytes, repeatability_reference_bytes: reference_bytes,
    diagnostic_bytes: diagnostic_bytes, scratch_bytes: scratch_bytes,
    weight_stage_bytes_per_workspace: weight_stage_bytes,
    compact_attention_stage_bytes_per_workspace: compact.resident_bytes,
    rms_plan_scratch_bytes_per_workspace: rms_bytes,
    mps_workspace_count: workspaces_count, mps_workspace_bytes: mps_workspace_bytes,
    bf16_resident_weights_bytes: checkpoint_plan.destination_bytes,
    f32_weight_conversion_staging_bytes: checkpoint_plan.staging_bytes,
    explicitly_named_peak_bytes: checkpoint_plan.peak_bytes,
    score_tile_reserve_bytes: score_reserve_bytes,
    opaque_mpsgraph_reserve_heuristic_bytes: HY_SERIAL_OPAQUE_BYTES,
    estimated_peak_bytes: estimated_peak, preflight_limit_bytes: HY_SERIAL_MEMORY_LIMIT,
    external_memory_guard_required: true,
    external_guard_policy: "cache-aware free+filecache headroom >=15%; process-group footprint <=4 GiB; swap growth <=64 MiB; timeout 600 seconds",
    backend_private_allocation_cap_established: false,
    run_qualification: HY_SERIAL_QUALIFICATION,
  })
  raise ArgumentError.new("Explicit buffers and conservative reserves exceed 4 GiB") if estimated_peak > HY_SERIAL_MEMORY_LIMIT
  if preflight_only
    hy_serial_emit({schema: "hy_nomas_serial_plan_v1", status: "preflight_only",
                    metal_initialized: false, gpu_buffers_allocated: false, benchmark_selects_or_changes_no_default: true})
    source_file.close
    exit 0
  end
  raise ArgumentError.new("Hy-Nomas serial benchmark requires macOS Metal") unless ML::Metal::Device.init!

  input_arena = ML::Video::Arena.new
  weight_arena = ML::Video::Arena.new
  count = ->(rows : Int32) { ML::Video::HyNomasDualStreamBlock.checked_count(rows, config.dim) }
  img = input_arena.not_nil!.alloc("img", count.call(config.image_rows))
  txt = input_arena.not_nil!.alloc("txt", count.call(config.text_rows))
  vec = input_arena.not_nil!.alloc("vec", count.call(config.effective_img_modulation_rows))
  vec_txt = input_arena.not_nil!.alloc("vec_txt", count.call(config.effective_txt_modulation_rows))
  rope_count = ML::Video::HyNomasDualStreamBlock.checked_count(config.video_tokens, config.head_dim)
  cos = input_arena.not_nil!.alloc("rope.cos", rope_count)
  sin = input_arena.not_nil!.alloc("rope.sin", rope_count)
  metrics = input_arena.not_nil!.alloc("diagnostic.metrics", 1)

  if mode == "separate"
    2.times { scratch_arenas << ML::Video::Arena.new }
    scratch_arenas.each { |arena| ML::Video::HyNomasDualStreamBlock.prepare_scratch!(arena, config) }
  end

  caller_named = input_arena.not_nil!.resident_bytes + scratch_arenas.sum(&.resident_bytes)
  expected_caller_named = input_bytes + reference_bytes + diagnostic_bytes +
                          (mode == "separate" ? scratch_bytes : 0_i64)
  raise "Caller buffer bytes differ from preflight" unless caller_named == expected_caller_named
  plan = ML::Video::HyNomasDualStreamBlock.weight_plan([source_file], config, available_budget,
    reserve_bytes: named_reserve_bytes, resident_dtype: ML::Video::WeightDType::BF16,
    block_index: 0, require_all: true, allow_f32_to_bf16: true)
  unless plan.destination_bytes == checkpoint_plan.destination_bytes && plan.staging_bytes == checkpoint_plan.staging_bytes &&
         plan.peak_bytes == checkpoint_plan.peak_bytes
    raise "Checkpoint memory plan changed after buffer allocation"
  end
  prepared = ML::Video::PreparedWeights.new(plan, weight_arena.not_nil!)

  initial_workspace_count = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
  if mode == "shared"
    serial = ML::Video::HyNomasMPSSerialWorkspace.new(config, [weight_arena.not_nil!, weight_arena.not_nil!],
      img, txt, vec, vec_txt, cos, sin, attention_query_chunk: HY_SERIAL_QUERY_CHUNK,
      attention_heads_per_group: HY_SERIAL_HEADS_GROUP)
    blocks = [serial.not_nil!.blocks[0], serial.not_nil!.blocks[1]]
  else
    blocks << ML::Video::HyNomasDualStreamBlock.new(config, weight_arena.not_nil!, scratch_arenas[0])
    blocks << ML::Video::HyNomasDualStreamBlock.new(config, weight_arena.not_nil!, scratch_arenas[1])
    ws0 = ML::Video::HyNomasMPSWorkspace.new(blocks[0], img, txt, vec, vec_txt, cos, sin,
      attention_query_chunk: HY_SERIAL_QUERY_CHUNK, attention_heads_per_group: HY_SERIAL_HEADS_GROUP)
    ws1 = ML::Video::HyNomasMPSWorkspace.new(blocks[1], blocks[0].image_output, blocks[0].text_output,
      vec, vec_txt, cos, sin, attention_query_chunk: HY_SERIAL_QUERY_CHUNK,
      attention_heads_per_group: HY_SERIAL_HEADS_GROUP)
    workspaces << ws0 << ws1
  end
  actual_workspace_bytes = serial.try(&.mps_workspace_bytes) || workspaces.sum(&.resident_bytes)
  unless actual_workspace_bytes == mps_workspace_bytes
    raise "MPS workspace residency differs from preflight: #{actual_workspace_bytes} != #{mps_workspace_bytes}"
  end

  probe = HyNomasBlockProbe.new
  setup_started = Time.instant
  setup = ML::Metal::CommandBuffer.new
  commands << setup
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  prepared.not_nil!.encode(setup_encoder)
  probe.encode_inputs(setup_encoder, config, img, txt, vec, vec_txt, cos, sin)
  setup_encoder.end_encoding
  setup.commit
  setup.wait
  setup_wall = (Time.instant - setup_started).total_seconds
  prepared.not_nil!.release
  prepared = nil
  hy_serial_emit({schema: "hy_nomas_serial_setup_v1", status: "ok", mode: mode,
                  setup_host_wall_including_terminal_wait_seconds: setup_wall, setup_terminal_waits: 1,
                  weights_converted_f32_to_bf16_on_gpu: true, upload_staging_released_after_wait: true,
                  input_fill_formula: HY_SERIAL_FILL_FORMULA, rope_formula: HY_SERIAL_ROPE_FORMULA,
                  setup_outside_measured_samples: true})

  actual_resident = input_arena.not_nil!.resident_bytes + weight_arena.not_nil!.resident_bytes +
                    (serial.try(&.resident_bytes) || scratch_arenas.sum(&.resident_bytes) + actual_workspace_bytes)
  expected_resident = input_bytes + reference_bytes + diagnostic_bytes + checkpoint_plan.destination_bytes + scratch_bytes + mps_workspace_bytes
  raise "Named resident bytes differ from preflight accounting" unless actual_resident == expected_resident
  created_workspace_count = ML::Video::HyNomasMPSWorkspace.retained_workspace_count - initial_workspace_count
  expected_workspace_count = mode == "shared" ? 1 : 2
  raise "Unexpected retained workspace count: #{created_workspace_count}" unless created_workspace_count == expected_workspace_count
  hy_serial_emit({schema: "hy_nomas_serial_workspace_v1", status: "ready", mode: mode,
                  workspace_count: created_workspace_count, retained_workspace_count: ML::Video::HyNomasMPSWorkspace.retained_workspace_count,
                  named_resident_bytes: actual_resident, input_bytes: input_bytes, weight_bytes: weight_arena.not_nil!.resident_bytes,
                  scratch_bytes: scratch_bytes, mps_workspace_bytes: mps_workspace_bytes,
                  score_tile_reserve_bytes: score_reserve_bytes, opaque_mpsgraph_reserve_heuristic_bytes: HY_SERIAL_OPAQUE_BYTES,
                  source_sha256: source_sha, run_qualification: HY_SERIAL_QUALIFICATION})

  warmup = hy_serial_sample(mode, 0, serial, blocks, workspaces, img, txt, vec, vec_txt, cos, sin,
    probe, metrics, nil, chains)
  warmup_hashes = {warmup[:image_output_raw_f32_sha256], warmup[:text_output_raw_f32_sha256]}
  hy_serial_emit({schema: "hy_nomas_serial_warmup_v1", status: "ok", mode: mode,
                  root_count: warmup[:root_count], output_hashes: {
                    image: warmup[:image_output_raw_f32_sha256], text: warmup[:text_output_raw_f32_sha256],
                  },
                  gpu_self_finiteness_check: warmup[:diagnostic],
                  warmup_self_finiteness_check_excluded_from_performance_summary: true})

  samples = [] of typeof(warmup)
  diagnostics = [] of Hash(String, JSON::Any)
  4.times do |i|
    sample = hy_serial_sample(mode, i + 1, serial, blocks, workspaces,
      img, txt, vec, vec_txt, cos, sin, probe, metrics, warmup_hashes,
      chains)
    diagnostic = sample[:diagnostic]
    diagnostics << diagnostic
    samples << sample
    hy_serial_emit(sample.merge({
      diagnostic: diagnostic, workspace_count: created_workspace_count,
      named_resident_bytes: actual_resident,
      cross_mode_hash_validation: "external paired-result receipt",
      cross_mode_hash_validation_in_benchmark: false,
    }))
  end
  measured_root_counts = samples.map(&.[:root_count])
  measured_root_count_distribution = measured_root_counts.uniq.sort.map do |root_count|
    {root_count: root_count, requests: measured_root_counts.count(root_count)}
  end
  hy_serial_emit({schema: "hy_nomas_serial_summary_v1", status: "repeatability_hashes_match_warmup",
                  mode: mode, measured_requests: samples.size, warmups: 1, workspace_count: created_workspace_count,
                  retained_workspace_count_after_setup: ML::Video::HyNomasMPSWorkspace.retained_workspace_count,
                  all_measured_root_counts_stable: measured_root_counts.uniq.size == 1,
                  measured_root_count_distribution: measured_root_count_distribution,
                  root_count_reporting_scope: "observed automatic MPS segmentation per request; not a deterministic operation-count invariant",
                  root_counts_are_observational: true, all_samples_raw_hashes_match_warmup: true,
                  all_samples_zero_tracked_buffer_allocations_during_encode: true,
                  all_samples_one_terminal_wait: true,
                  all_samples_hash_exact_repeatability_passed: diagnostics.all? { |d| d["status"].as_s == "pass" },
                  warmup_gpu_self_finiteness_check_passed: warmup[:diagnostic]["status"].as_s == "pass",
                  warmup_self_finiteness_check_excluded_from_performance_summary: true,
                  measured_diagnostic_scope: "hash exact repeatability; no GPU diagnostic in timed chain",
                  timings_include_only_block_encodes_and_one_terminal_wait: true,
                  image_output_raw_f32_sha256: samples.last[:image_output_raw_f32_sha256],
                  text_output_raw_f32_sha256: samples.last[:text_output_raw_f32_sha256],
                  cross_mode_hash_validation: "external paired-result receipt",
                  cross_mode_hash_validation_in_benchmark: false,
                  median_encode_wall_seconds: hy_serial_median(samples.map(&.[:encode_wall_seconds])),
                  median_terminal_wait_wall_seconds: hy_serial_median(samples.map(&.[:terminal_wait_wall_seconds])),
                  median_total_wall_seconds: hy_serial_median(samples.map(&.[:total_wall_seconds])),
                  median_native_gpu_duration_seconds: hy_serial_median(samples.map(&.[:native_gpu_duration_seconds])),
                  named_resident_bytes: actual_resident, explicitly_named_setup_peak_bytes: checkpoint_plan.peak_bytes,
                  estimated_peak_bytes: estimated_peak, score_tile_reserve_bytes: score_reserve_bytes,
                  opaque_mpsgraph_reserve_heuristic_bytes: HY_SERIAL_OPAQUE_BYTES,
                  attention_precision: "BF16BoundaryControl", source_checkpoint_sha256: source_sha,
                  run_qualification: HY_SERIAL_QUALIFICATION,
                  no_full_model_or_video_generation: true})
rescue ex
  failure_message = ex.message || ex.class.name
  hy_serial_emit({schema: "hy_nomas_serial_failure_v1", phase: "benchmark", status: "failed", error: failure_message})
ensure
  cleanup_ok = true
  commands.each do |command|
    begin
      command.wait if command.committed? && !command.completed?
      command.discard unless command.committed?
    rescue ex
      cleanup_ok = false
      hy_serial_emit({schema: "hy_nomas_serial_failure_v1", phase: "setup_command_drain", error: ex.message || ex.class.name})
    end
  end
  chains.each do |chain|
    begin
      chain.commit_and_wait unless chain.completed?
      chain.release if chain.completed?
    rescue ex
      cleanup_ok = false
      hy_serial_emit({schema: "hy_nomas_serial_failure_v1", phase: "chain_drain", error: ex.message || ex.class.name})
    end
  end
  if cleanup_ok
    begin
      serial.try(&.release)
      workspaces.each(&.release)
      prepared.try(&.release)
      scratch_arenas.each(&.release)
      input_arena.try(&.release)
      weight_arena.try(&.release)
    rescue ex
      cleanup_ok = false
      failure_message ||= ex.message || ex.class.name
      hy_serial_emit({schema: "hy_nomas_serial_failure_v1", phase: "resource_release", error: ex.message || ex.class.name})
    end
  else
    failure_message ||= "GPU work could not be proven drained; resident resources retained"
    hy_serial_emit({schema: "hy_nomas_serial_failure_v1", phase: "cleanup", error: failure_message})
  end
  source.try(&.close)
end

exit(1) if failure_message
