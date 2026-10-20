# Full-geometry Wan S2V self-attention chunk-size comparison.
# Requires a reference fixture from examples/support/wan_s2v_self_attention_benchmark_reference.py.
# This measures standalone F32 self attention only, not a full S2V block.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_attention_mps"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private SELF_Q_ROWS                         = 36_608
private SELF_K_ROWS                         = 36_608
private SELF_HEADS                          =     40
private SELF_HEAD_DIM                       =    128
private SELF_CHANNELS                       = SELF_HEADS * SELF_HEAD_DIM
private SELF_VALUES                         = SELF_Q_ROWS.to_i64 * SELF_CHANNELS
private SELF_BASELINE_CHUNK_ROWS            = 512
private SELF_CHUNK_CHOICES                  = {128, 256, 1024}
private SELF_FIXTURE_SCHEMA                 = "wan_s2v_self_attention_reference_v1"
private SELF_BENCHMARK_SCHEMA               = "wan_s2v_self_attention_benchmark_v1"
private SELF_FORMULA                        = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
private SELF_PREFLIGHT_LIMIT                = 8_i64 * 1024 * 1024 * 1024
private SELF_OPAQUE_RESERVE_HEURISTIC_BYTES = 2_i64 * 1024 * 1024 * 1024

private def self_emit_json(json : String) : Nil
  STDOUT.puts(json)
  STDOUT.flush
end

private def self_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def self_load_expected(path : String, buffer : ML::MetalBuffer,
                               expected_bytes : Int64) : Nil
  raise ArgumentError.new("Reference output byte size mismatch") unless File.size(path) == expected_bytes
  destination = buffer.contents.as(UInt8*)
  offset = 0_i64
  chunk = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(chunk)) > 0
      (destination + offset).copy_from(chunk.to_unsafe, count)
      offset += count
    end
  end
  raise ArgumentError.new("Truncated reference output") unless offset == expected_bytes
end

private def self_expected_explicit_buffer_bytes(candidate_chunk_rows : Int32) : Int64
  q_values = SELF_Q_ROWS.to_i64 * SELF_CHANNELS
  kv_values = SELF_K_ROWS.to_i64 * SELF_CHANNELS
  full_values = q_values * 4 + kv_values * 2 # Q, K, V, expected, and two backend outputs
  chunk_values = (SELF_BASELINE_CHUNK_ROWS.to_i64 + candidate_chunk_rows) * SELF_CHANNELS * 3
  scalar_values = 12_i64 # ten comparator metrics and two corruption-control values
  (full_values + chunk_values + scalar_values) * sizeof(Float32)
end

private def self_fixture_expected_path(fixture_dir : String) : String
  manifest_path = File.join(fixture_dir, "manifest.json")
  raise ArgumentError.new("Missing self-attention reference manifest") unless File.file?(manifest_path)
  manifest = JSON.parse(File.read(manifest_path))
  raise ArgumentError.new("Unsupported self-attention reference schema") unless manifest["schema"].as_s == SELF_FIXTURE_SCHEMA

  geometry = manifest["geometry"]
  {
    {"batches", 1_i64}, {"query_rows", SELF_Q_ROWS.to_i64},
    {"key_rows", SELF_K_ROWS.to_i64}, {"heads", SELF_HEADS.to_i64},
    {"head_dim", SELF_HEAD_DIM.to_i64},
  }.each do |key, expected|
    raise ArgumentError.new("Reference geometry mismatch for #{key}") unless geometry[key].as_i64 == expected
  end
  raise ArgumentError.new("Reference layout must be B,H,Q,D") unless geometry["reference_layout"].as_s == "B,H,Q,D"
  raise ArgumentError.new("Native input layout must be B,Q,H,D") unless geometry["native_layout"].as_s == "B,Q,H,D"
  raise ArgumentError.new("Fixture input formula does not match native generator") unless manifest["input_formula"].as_s == SELF_FORMULA
  salts = manifest["input_salts"]
  unless salts["q"].as_i64 == 1 && salts["k"].as_i64 == 2 && salts["v"].as_i64 == 3
    raise ArgumentError.new("Fixture input salts do not match native generator")
  end

  reference = manifest["reference"]
  unless reference["operator"].as_s == "torch.nn.functional.scaled_dot_product_attention" &&
         reference["device"].as_s == "mps" && reference["dtype"].as_s == "float32" &&
         reference["mps_fallback_disabled"].as_bool &&
         reference["comparison_tolerance"].as_s == "2e-5 + 2e-5 * abs(reference)" &&
         reference["chunking_semantics"].as_s == "independent query rows with full K/V; no mask, causal=false, dropout=0" &&
         reference["source_adapter"].as_s == "wan_s2v_real_stack_reference.py: transpose contiguous B,H,Q,D, PyTorch MPS SDPA, return B,H,Q,D"
    raise ArgumentError.new("Reference oracle metadata is not the required F32 MPS SDPA contract")
  end
  raise ArgumentError.new("Reference torch version must be PyTorch 2.8.x") unless reference["torch_version"].as_s.starts_with?("2.8.")
  unless {64_i64, 128_i64, 256_i64}.includes?(reference["query_chunk_rows"].as_i64)
    raise ArgumentError.new("Reference query chunk rows are outside the bounded oracle choices")
  end

  source = manifest["source"]
  {
    {"wan_attention_source_path", "wan_attention_source_sha256"},
    {"wan_s2v_reference_adapter_path", "wan_s2v_reference_adapter_sha256"},
  }.each do |path_key, digest_key|
    source_path = source[path_key].as_s
    raise ArgumentError.new("Recorded reference source is missing: #{path_key}") unless File.file?(source_path)
    expected_digest = source[digest_key].as_s
    unless expected_digest.matches?(/\A[0-9a-f]{64}\z/) && self_sha256_file(source_path) == expected_digest
      raise ArgumentError.new("Recorded reference source SHA-256 mismatch: #{path_key}")
    end
  end
  generator_digest = source["generator_sha256"].as_s
  generator_path = File.expand_path("support/wan_s2v_self_attention_benchmark_reference.py", __DIR__)
  unless generator_digest.matches?(/\A[0-9a-f]{64}\z/) && self_sha256_file(generator_path) == generator_digest
    raise ArgumentError.new("Reference generator SHA-256 mismatch")
  end

  expected = manifest["expected"]
  expected_bytes = SELF_VALUES * sizeof(Float32)
  unless expected["dtype"].as_s == "F32_LE" && expected["shape"].as_a.map(&.as_i64) ==
           [1_i64, SELF_HEADS.to_i64, SELF_Q_ROWS.to_i64, SELF_HEAD_DIM.to_i64] &&
         expected["bytes"].as_i64 == expected_bytes
    raise ArgumentError.new("Reference output shape or dtype mismatch")
  end
  name = expected["file"].as_s
  raise ArgumentError.new("Reference output path must be a basename") unless File.basename(name) == name
  path = File.join(fixture_dir, name)
  raise ArgumentError.new("Missing reference output") unless File.file?(path)
  raise ArgumentError.new("Reference output SHA-256 mismatch") unless self_sha256_file(path) == expected["sha256"].as_s
  path
end

private SELF_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  inline float wan_self_probe_value(uint index, uint salt) {
    uint mixed = index + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int quantized = int(mixed & 1023u) - 512;
    return float(quantized) / 256.0f;
  }
  kernel void wan_s2v_self_probe_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &salt [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    output[i] = wan_self_probe_value(i, salt);
  }
  kernel void wan_s2v_self_probe_zero(device atomic_uint *metrics [[buffer(0)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i < 10u) atomic_store_explicit(metrics + i, 0u, memory_order_relaxed);
  }
  // Native storage is B,Q,H,D; the upstream MPS fixture is contiguous B,H,Q,D.
  kernel void wan_s2v_self_probe_compare(
      device const float *actual [[buffer(0)]],
      device const float *expected_bhqd [[buffer(1)]],
      device const float *baseline [[buffer(2)]],
      device atomic_uint *metrics [[buffer(3)]],
      constant uint &count [[buffer(4)]],
      constant uint &query_rows [[buffer(5)]],
      constant uint &heads [[buffer(6)]],
      constant uint &head_dim [[buffer(7)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint channels = heads * head_dim;
    const uint query = i / channels;
    const uint head_channel = i % channels;
    const uint head = head_channel / head_dim;
    const uint channel = head_channel % head_dim;
    const uint reference_index = head * query_rows * head_dim + query * head_dim + channel;
    const float got = actual[i], ref = expected_bhqd[reference_index], prior = baseline[i];
    if (as_type<uint>(got) != as_type<uint>(ref))
      atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
    if (!isfinite(got) || !isfinite(ref)) {
      atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
    } else {
      const float error = abs(got - ref);
      const float ratio = error / (2.0e-5f + 2.0e-5f * abs(ref));
      if (ratio > 1.0f) atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 3, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 4, as_type<uint>(ratio), memory_order_relaxed);
    }
    if (as_type<uint>(got) != as_type<uint>(prior))
      atomic_fetch_add_explicit(metrics + 5, 1u, memory_order_relaxed);
    if (!isfinite(got) || !isfinite(prior)) {
      atomic_fetch_add_explicit(metrics + 7, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 6, 1u, memory_order_relaxed);
    } else {
      const float error = abs(got - prior);
      const float ratio = error / (2.0e-5f + 2.0e-5f * abs(prior));
      if (ratio > 1.0f) atomic_fetch_add_explicit(metrics + 6, 1u, memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 8, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 9, as_type<uint>(ratio), memory_order_relaxed);
    }
  }
  kernel void wan_s2v_self_probe_corrupt_one(
      device float *actual [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) actual[0] += 0.125f;
  }
METAL

private def self_validate(actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                          baseline : ML::MetalBuffer, metrics : ML::MetalBuffer,
                          zero : ML::Metal::ComputePipeline,
                          compare : ML::Metal::ComputePipeline,
                          count : Int32, query_rows : Int32,
                          heads : Int32, head_dim : Int32)
  validation = ML::Metal::MPSCommandChain.new
  validation_started = Time.instant
  validation.encoder.set_pipeline(zero)
    .set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(10)
  validation.encoder.set_pipeline(compare)
    .set_buffer(actual, 0).set_buffer(expected, 1).set_buffer(baseline, 2)
    .set_buffer(metrics, 3, ML::Metal::BufferAccess::ReadWrite)
    .set_value(count.to_u32, 4).set_value(query_rows.to_u32, 5)
    .set_value(heads.to_u32, 6).set_value(head_dim.to_u32, 7)
    .dispatch_1d(count)
  validation.commit_and_wait
  host_seconds = (Time.instant - validation_started).total_seconds
  raise "validation did not use one terminal wait" unless validation.wait_count == 1
  values = metrics.contents.as(UInt32*)
  result = {
    host_seconds:   host_seconds,
    gpu_seconds:    validation.gpu_elapsed_seconds,
    terminal_waits: validation.wait_count,
    upstream:       {
      bit_differences: values[0], tolerance_violations: values[1], nonfinite: values[2],
      max_abs: values[3].unsafe_as(Float32), max_tolerance_ratio: values[4].unsafe_as(Float32),
    },
    vs_baseline: {
      bit_differences: values[5], tolerance_violations: values[6], nonfinite: values[7],
      max_abs: values[8].unsafe_as(Float32), max_tolerance_ratio: values[9].unsafe_as(Float32),
    },
  }
  validation.release
  result
end

private def self_require_strict!(validation) : Nil
  unless validation[:upstream][:tolerance_violations] == 0 && validation[:upstream][:nonfinite] == 0 &&
         validation[:vs_baseline][:tolerance_violations] == 0 && validation[:vs_baseline][:nonfinite] == 0
    raise "strict full-output upstream or baseline comparison failed"
  end
end

fixture_dir = ""
candidate_chunk_rows = 0
OptionParser.parse do |parser|
  parser.banner = "Usage: crystal run examples/wan_s2v_self_attention_benchmark.cr -- --fixture DIR --candidate 128|256|1024"
  parser.on("--fixture=DIR", "Fixture directory from wan_s2v_self_attention_benchmark_reference.py") { |value| fixture_dir = value }
  parser.on("--candidate=ROWS", "Candidate MPS query chunk rows: 128, 256, or 1024 (baseline is 512)") { |value| candidate_chunk_rows = value.to_i }
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
unless SELF_CHUNK_CHOICES.includes?(candidate_chunk_rows)
  raise ArgumentError.new("--candidate must be one of #{SELF_CHUNK_CHOICES.join(", ")}")
end

fixture_dir = File.expand_path(fixture_dir)
expected_path = self_fixture_expected_path(fixture_dir)
fixture_manifest_path = File.join(fixture_dir, "manifest.json")
fixture_manifest = JSON.parse(File.read(fixture_manifest_path))
fixture_manifest_sha256 = self_sha256_file(fixture_manifest_path)
fixture_torch_version = fixture_manifest["reference"]["torch_version"].as_s
fixture_wan_attention_sha256 = fixture_manifest["source"]["wan_attention_source_sha256"].as_s
fixture_adapter_sha256 = fixture_manifest["source"]["wan_s2v_reference_adapter_sha256"].as_s
fixture_generator_sha256 = fixture_manifest["source"]["generator_sha256"].as_s
estimated_explicit_bytes = self_expected_explicit_buffer_bytes(candidate_chunk_rows)
estimated_with_opaque_reserve = estimated_explicit_bytes + SELF_OPAQUE_RESERVE_HEURISTIC_BYTES
raise ArgumentError.new("Self-attention preflight exceeds its 8 GiB explicit-plus-opaque-reserve-heuristic budget") if estimated_with_opaque_reserve > SELF_PREFLIGHT_LIMIT

arena = ML::Video::Arena.new
baseline_workspace = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
candidate_workspace = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
active_chain = nil.as(ML::Metal::MPSCommandChain?)
begin
  fill = ML::Metal::ComputePipeline.new("wan_s2v_self_probe_fill", SELF_PROBE_SOURCE)
  zero = ML::Metal::ComputePipeline.new("wan_s2v_self_probe_zero", SELF_PROBE_SOURCE)
  compare = ML::Metal::ComputePipeline.new("wan_s2v_self_probe_compare", SELF_PROBE_SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("wan_s2v_self_probe_corrupt_one", SELF_PROBE_SOURCE)

  q = arena.alloc("q", SELF_VALUES.to_i32)
  k = arena.alloc("k", (SELF_K_ROWS.to_i64 * SELF_CHANNELS).to_i32)
  v = arena.alloc("v", (SELF_K_ROWS.to_i64 * SELF_CHANNELS).to_i32)
  expected = arena.alloc("expected", SELF_VALUES.to_i32)
  baseline_output = arena.alloc("output_baseline_512", SELF_VALUES.to_i32)
  candidate_output = arena.alloc("output_candidate_#{candidate_chunk_rows}", SELF_VALUES.to_i32)
  metrics = arena.alloc("metrics", 10)
  control_expected = arena.alloc("control_expected", 1)
  control_actual = arena.alloc("control_actual", 1)

  baseline_chunk_q = arena.alloc("baseline_chunk_q", SELF_BASELINE_CHUNK_ROWS * SELF_CHANNELS)
  baseline_chunk_a = arena.alloc("baseline_chunk_output_a", SELF_BASELINE_CHUNK_ROWS * SELF_CHANNELS)
  baseline_chunk_b = arena.alloc("baseline_chunk_output_b", SELF_BASELINE_CHUNK_ROWS * SELF_CHANNELS)
  candidate_chunk_q = arena.alloc("candidate_chunk_q", candidate_chunk_rows * SELF_CHANNELS)
  candidate_chunk_a = arena.alloc("candidate_chunk_output_a", candidate_chunk_rows * SELF_CHANNELS)
  candidate_chunk_b = arena.alloc("candidate_chunk_output_b", candidate_chunk_rows * SELF_CHANNELS)
  self_load_expected(expected_path, expected, SELF_VALUES * sizeof(Float32))
  control_expected.contents.as(Float32*)[0] = 0.5_f32
  control_actual.contents.as(Float32*)[0] = 0.5_f32

  baseline_workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(
    1, SELF_Q_ROWS, SELF_K_ROWS, SELF_HEADS, SELF_HEAD_DIM, SELF_BASELINE_CHUNK_ROWS,
    q, k, v, baseline_output, baseline_chunk_q, baseline_chunk_a, baseline_chunk_b)
  candidate_workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(
    1, SELF_Q_ROWS, SELF_K_ROWS, SELF_HEADS, SELF_HEAD_DIM, candidate_chunk_rows,
    q, k, v, candidate_output, candidate_chunk_q, candidate_chunk_a, candidate_chunk_b)
  explicit_bytes = arena.resident_bytes
  unless explicit_bytes == estimated_explicit_bytes &&
         explicit_bytes + SELF_OPAQUE_RESERVE_HEURISTIC_BYTES <= SELF_PREFLIGHT_LIMIT
    raise ArgumentError.new("Actual explicit MetalBuffer bytes differ from the bounded preflight")
  end

  # Generate full Q/K/V on GPU. This one terminal wait is outside timed arms.
  setup_started = Time.instant
  setup = ML::Metal::CommandBuffer.new
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  { {q, 1_u32}, {k, 2_u32}, {v, 3_u32} }.each do |buffer, salt|
    count = buffer.element_count
    setup_encoder.set_pipeline(fill).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 1).set_value(salt, 2).dispatch_1d(count)
  end
  setup_encoder.end_encoding
  setup.commit
  setup.wait
  setup_seconds = (Time.instant - setup_started).total_seconds

  baseline_arm = {"baseline_512", SELF_BASELINE_CHUNK_ROWS, baseline_workspace.not_nil!, baseline_output}
  candidate_arm = {"candidate_#{candidate_chunk_rows}", candidate_chunk_rows, candidate_workspace.not_nil!, candidate_output}
  {baseline_arm, candidate_arm}.each do |name, chunk_rows, workspace, output|
    active_chain = ML::Metal::MPSCommandChain.new
    allocations_before = ML::MetalBuffer.video_spec_allocations
    arm_started = Time.instant
    workspace.encode(active_chain.not_nil!)
    encode_seconds = (Time.instant - arm_started).total_seconds
    raise "attention encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocations_before
    active_chain.not_nil!.commit_and_wait
    wall_seconds = (Time.instant - arm_started).total_seconds
    raise "warmup did not use one terminal wait" unless active_chain.not_nil!.wait_count == 1
    raise "warmup did not cross the segmented execution path" unless active_chain.not_nil!.segment_count > 0
    gpu_seconds = active_chain.not_nil!.gpu_elapsed_seconds
    segments = active_chain.not_nil!.segment_count
    active_chain.not_nil!.release
    active_chain = nil

    self_emit_json({schema: SELF_BENCHMARK_SCHEMA, phase: "warmup", backend: name,
                    chunk_rows: chunk_rows, query_rows: SELF_Q_ROWS, key_rows: SELF_K_ROWS,
                    heads: SELF_HEADS, head_dim: SELF_HEAD_DIM, compared_values: SELF_VALUES,
                    validation_after_both_arms: true,
                    attention_encode_seconds: encode_seconds,
                    host_attention_wall_seconds_including_terminal_wait: wall_seconds,
                    gpu_attention_seconds_excluding_validation: gpu_seconds,
                    attention_command_segments: segments, attention_terminal_waits: 1,
                    encode_metal_buffer_allocations: 0,
                    explicit_buffer_bytes: explicit_bytes,
                    opaque_mpsgraph_reserve_heuristic_bytes: SELF_OPAQUE_RESERVE_HEURISTIC_BYTES,
                    backend_private_allocation_cap_established: false,
                    preflight_budget_bytes: SELF_PREFLIGHT_LIMIT,
                    input_setup_host_wall_seconds_including_one_terminal_wait: setup_seconds,
                    fixture_manifest: fixture_manifest_path}.to_json)
  end

  { {"baseline_512", baseline_output}, {"candidate_#{candidate_chunk_rows}", candidate_output} }.each do |name, output|
    validation = self_validate(output, expected, baseline_output, metrics, zero, compare,
      SELF_VALUES.to_i32, SELF_Q_ROWS, SELF_HEADS, SELF_HEAD_DIM)
    self_emit_json({schema: "wan_s2v_self_attention_validation_v1", phase: "warmup", backend: name,
                    compared_values: SELF_VALUES, upstream: validation[:upstream], vs_baseline: validation[:vs_baseline],
                    comparison_is_self_for_baseline: name == "baseline_512",
                    validation_host_seconds_including_terminal_wait: validation[:host_seconds],
                    validation_gpu_seconds: validation[:gpu_seconds], validation_terminal_waits: validation[:terminal_waits],
                    validation_after_both_arms: true}.to_json)
    self_require_strict!(validation)
  end

  baseline_wall_samples = [] of Float64
  candidate_wall_samples = [] of Float64
  {0, 1, 2}.each do |pair_index|
    pair_order = pair_index == 1 ? "BA" : "AB"
    2.times do |position|
      use_baseline = pair_index == 1 ? position == 1 : position == 0
      name, chunk_rows, workspace, output = use_baseline ? baseline_arm : candidate_arm
      active_chain = ML::Metal::MPSCommandChain.new
      allocations_before = ML::MetalBuffer.video_spec_allocations
      arm_started = Time.instant
      workspace.encode(active_chain.not_nil!)
      encode_seconds = (Time.instant - arm_started).total_seconds
      raise "attention encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocations_before
      active_chain.not_nil!.commit_and_wait
      wall_seconds = (Time.instant - arm_started).total_seconds
      raise "attention arm did not use one terminal wait" unless active_chain.not_nil!.wait_count == 1
      raise "attention arm did not cross the segmented execution path" unless active_chain.not_nil!.segment_count > 0
      gpu_seconds = active_chain.not_nil!.gpu_elapsed_seconds
      segments = active_chain.not_nil!.segment_count
      active_chain.not_nil!.release
      active_chain = nil

      if name == "baseline_512"
        baseline_wall_samples << wall_seconds
      else
        candidate_wall_samples << wall_seconds
      end
      self_emit_json({schema: SELF_BENCHMARK_SCHEMA, phase: "measured", backend: name,
                      candidate_chunk_rows: candidate_chunk_rows, chunk_rows: chunk_rows,
                      query_rows: SELF_Q_ROWS, key_rows: SELF_K_ROWS, heads: SELF_HEADS,
                      head_dim: SELF_HEAD_DIM, compared_values: SELF_VALUES,
                      pair_repetition: pair_index + 1, pair_order: pair_order,
                      pair_position: position + 1, validation_after_both_arms: true,
                      attention_encode_seconds: encode_seconds,
                      host_attention_wall_seconds_including_terminal_wait: wall_seconds,
                      gpu_attention_seconds_excluding_validation: gpu_seconds,
                      attention_command_segments: segments, attention_terminal_waits: 1,
                      encode_metal_buffer_allocations: 0,
                      explicit_buffer_bytes: explicit_bytes,
                      opaque_mpsgraph_reserve_heuristic_bytes: SELF_OPAQUE_RESERVE_HEURISTIC_BYTES,
                      backend_private_allocation_cap_established: false,
                      input_setup_host_wall_seconds_including_one_terminal_wait: setup_seconds,
                      fixture_manifest: fixture_manifest_path,
                      fixture_manifest_sha256: fixture_manifest_sha256,
                      reference_torch_version: fixture_torch_version,
                      wan_attention_source_sha256: fixture_wan_attention_sha256,
                      wan_s2v_reference_adapter_sha256: fixture_adapter_sha256,
                      generator_sha256: fixture_generator_sha256}.to_json)
    end
    { {"baseline_512", baseline_output}, {"candidate_#{candidate_chunk_rows}", candidate_output} }.each do |name, output|
      validation = self_validate(output, expected, baseline_output, metrics, zero, compare,
        SELF_VALUES.to_i32, SELF_Q_ROWS, SELF_HEADS, SELF_HEAD_DIM)
      self_emit_json({schema: "wan_s2v_self_attention_validation_v1", phase: "measured", backend: name,
                      pair_repetition: pair_index + 1, pair_order: pair_order,
                      compared_values: SELF_VALUES, upstream: validation[:upstream], vs_baseline: validation[:vs_baseline],
                      comparison_is_self_for_baseline: name == "baseline_512",
                      validation_host_seconds_including_terminal_wait: validation[:host_seconds],
                      validation_gpu_seconds: validation[:gpu_seconds], validation_terminal_waits: validation[:terminal_waits],
                      validation_after_both_arms: true}.to_json)
      self_require_strict!(validation)
    end
  end

  baseline_sorted = baseline_wall_samples.sort
  candidate_sorted = candidate_wall_samples.sort
  baseline_median = baseline_sorted[baseline_sorted.size // 2]
  candidate_median = candidate_sorted[candidate_sorted.size // 2]
  paired_savings = baseline_wall_samples.zip(candidate_wall_samples).map { |baseline, candidate| baseline - candidate }
  candidate_faster = candidate_median < baseline_median && paired_savings.all? { |saving| saving > 0.0 }
  self_emit_json({schema:                                        "wan_s2v_self_attention_candidate_decision_v1",
                  baseline_chunk_rows:                           SELF_BASELINE_CHUNK_ROWS,
                  candidate_chunk_rows:                          candidate_chunk_rows,
                  measured_pair_orders:                          ["AB", "BA", "AB"],
                  baseline_host_attention_wall_samples_seconds:  baseline_wall_samples,
                  candidate_host_attention_wall_samples_seconds: candidate_wall_samples,
                  baseline_host_attention_wall_median_seconds:   baseline_median,
                  candidate_host_attention_wall_median_seconds:  candidate_median,
                  paired_baseline_minus_candidate_seconds:       paired_savings,
                  all_pairs_faster:                              paired_savings.all? { |saving| saving > 0.0 },
                  strict_all_values_upstream_and_baseline_pass:  true,
                  candidate_attention_wall_improves:             candidate_faster,
                  candidate_admitted_for_further_review:         candidate_faster,
                  default_changed:                               false,
                  timing_decision_metric:                        "all three paired host attention savings positive and median improves; encode plus one terminal wait; validation excluded"}.to_json)

  # Prove the full-output comparator rejects a deliberately corrupted value.
  control = ML::Metal::MPSCommandChain.new
  control.encoder.set_pipeline(corrupt)
    .set_buffer(control_actual, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  control.encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(10)
  control.encoder.set_pipeline(compare).set_buffer(control_actual, 0)
    .set_buffer(control_expected, 1).set_buffer(control_expected, 2)
    .set_buffer(metrics, 3, ML::Metal::BufferAccess::ReadWrite)
    .set_value(1_u32, 4).set_value(1_u32, 5).set_value(1_u32, 6).set_value(1_u32, 7)
    .dispatch_1d(1)
  control.commit_and_wait
  raise "corruption negative control did not use one terminal wait" unless control.wait_count == 1
  control_values = metrics.contents.as(UInt32*)
  unless control_values[0] == 1 && control_values[1] == 1 && control_values[2] == 0 &&
         control_values[5] == 1 && control_values[6] == 1 && control_values[7] == 0
    raise "strict comparator failed to reject one corrupted value"
  end
  self_emit_json({schema: "wan_s2v_self_attention_comparator_negative_control_v1",
                  corrupted_values: 1, bit_differences: control_values[0],
                  tolerance_violations: control_values[1], nonfinite: control_values[2],
                  terminal_waits: control.wait_count}.to_json)
  control.release
ensure
  if active = active_chain
    begin
      active.commit_and_wait unless active.completed?
      active.release if active.completed?
    rescue
    end
  end
  candidate_workspace.try(&.release)
  baseline_workspace.try(&.release)
  arena.release
end
