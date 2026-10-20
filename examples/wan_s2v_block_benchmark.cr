# Full-geometry cross-attention segmentation diagnostic.
# Requires a reference fixture from examples/support/wan_s2v_block_benchmark_reference.py.
# This measures standalone F32 cross attention only, not a full S2V block.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_attention"
require "../src/ml/video/metal_attention_mps"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private CROSS_Q_ROWS                = 36_608
private CROSS_K_ROWS                =    512
private CROSS_HEADS                 =     40
private CROSS_HEAD_DIM              =    128
private CROSS_CHUNK_ROWS            =    512
private CROSS_VALUES                = CROSS_Q_ROWS.to_i64 * CROSS_HEADS * CROSS_HEAD_DIM
private CROSS_CHANNELS              = CROSS_HEADS * CROSS_HEAD_DIM
private CROSS_EXPLICIT_BUFFER_LIMIT = 4_i64 * 1024 * 1024 * 1024
private CROSS_FIXTURE_SCHEMA        = "wan_s2v_cross_attention_reference_v1"
private CROSS_INPUT_FORMULA         = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"

private def cross_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def cross_load_expected(path : String, buffer : ML::MetalBuffer,
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

private def cross_expected_explicit_buffer_bytes : Int64
  full_sized_values = CROSS_VALUES * 4                        # Q, expected, baseline output, candidate output
  key_value_values = CROSS_K_ROWS.to_i64 * CROSS_CHANNELS * 2 # K and V
  chunk_values = CROSS_CHUNK_ROWS.to_i64 * CROSS_CHANNELS * 3 # Q scratch and two Graph outputs
  scalar_values = 12_i64                                      # ten metrics and two corruption-control scalars
  (full_sized_values + key_value_values + chunk_values + scalar_values) * sizeof(Float32)
end

private def cross_fixture_expected_path(fixture_dir : String) : String
  manifest_path = File.join(fixture_dir, "manifest.json")
  raise ArgumentError.new("Missing cross-attention reference manifest") unless File.file?(manifest_path)
  manifest = JSON.parse(File.read(manifest_path))
  raise ArgumentError.new("Unsupported cross-attention reference schema") unless manifest["schema"].as_s == CROSS_FIXTURE_SCHEMA

  geometry = manifest["geometry"]
  {
    {"batches", 1_i64}, {"query_rows", CROSS_Q_ROWS.to_i64},
    {"key_rows", CROSS_K_ROWS.to_i64}, {"heads", CROSS_HEADS.to_i64},
    {"head_dim", CROSS_HEAD_DIM.to_i64},
  }.each do |key, expected|
    raise ArgumentError.new("Reference geometry mismatch for #{key}") unless geometry[key].as_i64 == expected
  end
  raise ArgumentError.new("Reference layout must be B,H,Q,D") unless geometry["reference_layout"].as_s == "B,H,Q,D"
  raise ArgumentError.new("Native input layout must be B,Q,H,D") unless geometry["native_layout"].as_s == "B,Q,H,D"
  raise ArgumentError.new("Fixture input formula does not match native generator") unless manifest["input_formula"].as_s == CROSS_INPUT_FORMULA
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
         reference["source_adapter"].as_s == "transpose to contiguous B,H,Q,D, PyTorch MPS SDPA, return B,H,Q,D"
    raise ArgumentError.new("Reference oracle metadata is not the pinned F32 MPS SDPA contract")
  end
  raise ArgumentError.new("Reference chunk rows must be positive") unless reference["query_chunk_rows"].as_i64 > 0
  source = manifest["source"]
  {
    {"wan_attention_path", "wan_attention_sha256"},
    {"wan_s2v_reference_adapter_path", "wan_s2v_reference_adapter_sha256"},
  }.each do |path_key, digest_key|
    source_path = source[path_key].as_s
    raise ArgumentError.new("Pinned reference source is missing: #{path_key}") unless File.file?(source_path)
    expected_digest = source[digest_key].as_s
    unless expected_digest.matches?(/\A[0-9a-f]{64}\z/) && cross_sha256_file(source_path) == expected_digest
      raise ArgumentError.new("Pinned reference source SHA-256 mismatch: #{path_key}")
    end
  end
  adapter_digest = source["adapter_sha256"].as_s
  unless adapter_digest.matches?(/\A[0-9a-f]{64}\z/) && cross_sha256_file(File.expand_path("support/wan_s2v_block_benchmark_reference.py", __DIR__)) == adapter_digest
    raise ArgumentError.new("Reference generator SHA-256 mismatch")
  end
  expected = manifest["expected"]
  expected_bytes = CROSS_VALUES * sizeof(Float32)
  unless expected["dtype"].as_s == "F32_LE" && expected["shape"].as_a.map(&.as_i64) ==
           [1_i64, CROSS_HEADS, CROSS_Q_ROWS, CROSS_HEAD_DIM] && expected["bytes"].as_i64 == expected_bytes
    raise ArgumentError.new("Reference output shape or dtype mismatch")
  end
  name = expected["file"].as_s
  raise ArgumentError.new("Reference output path must be a basename") unless File.basename(name) == name
  path = File.join(fixture_dir, name)
  raise ArgumentError.new("Missing reference output") unless File.file?(path)
  raise ArgumentError.new("Reference output SHA-256 mismatch") unless cross_sha256_file(path) == expected["sha256"].as_s
  path
end

private CROSS_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  inline float wan_cross_probe_value(uint index, uint salt) {
    uint mixed = index + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int quantized = int(mixed & 1023u) - 512;
    return float(quantized) / 256.0f;
  }
  kernel void wan_s2v_cross_probe_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &salt [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    output[i] = wan_cross_probe_value(i, salt);
  }
  kernel void wan_s2v_cross_probe_zero(device atomic_uint *metrics [[buffer(0)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i < 10u) atomic_store_explicit(metrics + i, 0u, memory_order_relaxed);
  }
  // Native storage is B,Q,H,D. The upstream MPS fixture is contiguous B,H,Q,D.
  kernel void wan_s2v_cross_probe_compare(
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
  kernel void wan_s2v_cross_probe_corrupt_one(
      device float *actual [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) actual[0] += 0.125f;
  }
METAL

fixture_dir = ""
repetitions = 3
OptionParser.parse do |parser|
  parser.banner = "Usage: crystal run examples/wan_s2v_block_benchmark.cr -- --fixture DIR [--repetitions N]"
  parser.on("--fixture=DIR", "Fixture directory from wan_s2v_block_benchmark_reference.py") { |value| fixture_dir = value }
  parser.on("--repetitions=N", "Measured repetitions per backend after one warmup (1..5)") { |value| repetitions = value.to_i }
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
raise ArgumentError.new("--repetitions must be within 1..5") unless (1..5).includes?(repetitions)
fixture_dir = File.expand_path(fixture_dir)
expected_path = cross_fixture_expected_path(fixture_dir)
fixture_manifest_path = File.join(fixture_dir, "manifest.json")
fixture_manifest = JSON.parse(File.read(fixture_manifest_path))
fixture_manifest_sha256 = cross_sha256_file(fixture_manifest_path)
fixture_torch_version = fixture_manifest["reference"]["torch_version"].as_s
fixture_wan_attention_sha256 = fixture_manifest["source"]["wan_attention_sha256"].as_s
fixture_adapter_sha256 = fixture_manifest["source"]["wan_s2v_reference_adapter_sha256"].as_s
estimated_explicit_bytes = cross_expected_explicit_buffer_bytes
raise ArgumentError.new("Cross benchmark exceeds its 4 GiB explicit-buffer cap") if estimated_explicit_bytes > CROSS_EXPLICIT_BUFFER_LIMIT

arena = ML::Video::Arena.new
workspace = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
chain = nil.as(ML::Metal::MPSCommandChain?)
begin
  ML::Video::MetalOps.prepare_tiled_attention!
  fill = ML::Metal::ComputePipeline.new("wan_s2v_cross_probe_fill", CROSS_PROBE_SOURCE)
  zero = ML::Metal::ComputePipeline.new("wan_s2v_cross_probe_zero", CROSS_PROBE_SOURCE)
  compare = ML::Metal::ComputePipeline.new("wan_s2v_cross_probe_compare", CROSS_PROBE_SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("wan_s2v_cross_probe_corrupt_one", CROSS_PROBE_SOURCE)

  q = arena.alloc("q", CROSS_VALUES.to_i32)
  k = arena.alloc("k", CROSS_K_ROWS * CROSS_CHANNELS)
  v = arena.alloc("v", CROSS_K_ROWS * CROSS_CHANNELS)
  expected = arena.alloc("expected", CROSS_VALUES.to_i32)
  output_320 = arena.alloc("output_320", CROSS_VALUES.to_i32)
  output_work = arena.alloc("output_work", CROSS_VALUES.to_i32)
  metrics = arena.alloc("metrics", 10)
  control_expected = arena.alloc("control_expected", 1)
  control_actual = arena.alloc("control_actual", 1)
  cross_load_expected(expected_path, expected, CROSS_VALUES * sizeof(Float32))

  chunk_q = arena.alloc("chunk_q", CROSS_CHUNK_ROWS * CROSS_CHANNELS)
  chunk_a = arena.alloc("chunk_output_a", CROSS_CHUNK_ROWS * CROSS_CHANNELS)
  chunk_b = arena.alloc("chunk_output_b", CROSS_CHUNK_ROWS * CROSS_CHANNELS)
  control_expected.contents.as(Float32*)[0] = 0.5_f32
  control_actual.contents.as(Float32*)[0] = 0.5_f32
  workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(
    1, CROSS_Q_ROWS, CROSS_K_ROWS, CROSS_HEADS, CROSS_HEAD_DIM, CROSS_CHUNK_ROWS,
    q, k, v, output_work, chunk_q, chunk_a, chunk_b)
  explicit_bytes = arena.resident_bytes
  raise ArgumentError.new("Cross benchmark exceeds its 4 GiB explicit-buffer cap") if explicit_bytes > CROSS_EXPLICIT_BUFFER_LIMIT

  # Generate inputs once; this one setup wait is outside every timed arm.
  setup = ML::Metal::CommandBuffer.new
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  { {q, 1_u32}, {k, 2_u32}, {v, 3_u32} }.each do |buffer, salt|
    count = buffer.element_count
    setup_encoder.set_pipeline(fill).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 1).set_value(salt, 2).dispatch_1d(count)
  end
  setup_encoder.end_encoding
  setup_started = Time.instant
  setup.commit
  setup.wait
  setup_seconds = (Time.instant - setup_started).total_seconds

  backends = { {"tiled_320", 320}, {"tiled_1024", 1024}, {"mps_chunk_512", 0} }
  # Warm baseline first, then each candidate, so all graph/pipeline setup is
  # excluded from measured arms and candidate order does not bias the baseline.
  {backends[0], backends[1], backends[2]}.each_with_index do |entry, index|
    name = entry[0]
    groups = entry[1]
    output = index == 0 ? output_320 : output_work
    chain = ML::Metal::MPSCommandChain.new
    allocations_before = ML::MetalBuffer.video_spec_allocations
    arm_started = Time.instant
    case name
    when "tiled_320", "tiled_1024"
      ML::Video::MetalOps.attention_tiled(chain.not_nil!, q, k, v, output,
        CROSS_Q_ROWS, CROSS_K_ROWS, CROSS_HEADS, CROSS_HEAD_DIM, 1, groups)
    when "mps_chunk_512"
      workspace.not_nil!.encode(chain.not_nil!)
    end
    raise "attention encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocations_before
    chain.not_nil!.commit_and_wait
    raise "attention arm did not use one terminal wait" unless chain.not_nil!.wait_count == 1
    raise "attention arm did not cross the segmented execution path" unless chain.not_nil!.segment_count > 0
    chain.not_nil!.release
    chain = nil
    validation = ML::Metal::MPSCommandChain.new
    validation.encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(10)
    validation.encoder.set_pipeline(compare).set_buffer(output, 0).set_buffer(expected, 1)
      .set_buffer(output_320, 2).set_buffer(metrics, 3, ML::Metal::BufferAccess::ReadWrite)
      .set_value(CROSS_VALUES.to_u32, 4).set_value(CROSS_Q_ROWS.to_u32, 5)
      .set_value(CROSS_HEADS.to_u32, 6).set_value(CROSS_HEAD_DIM.to_u32, 7)
      .dispatch_1d(CROSS_VALUES.to_i32)
    validation.commit_and_wait
    raise "warmup validation did not use one terminal wait" unless validation.wait_count == 1
    values = metrics.contents.as(UInt32*)
    puts({schema: "wan_s2v_cross_attention_benchmark_v1", backend: name,
          segment_groups: groups == 0 ? nil : groups, query_rows: CROSS_Q_ROWS,
          key_rows: CROSS_K_ROWS, heads: CROSS_HEADS, head_dim: CROSS_HEAD_DIM,
          compared_values: CROSS_VALUES, warmup: true, repetition: nil,
          upstream: {bit_differences: values[0], tolerance_violations: values[1],
                     nonfinite: values[2], max_abs: values[3].unsafe_as(Float32),
                     max_tolerance_ratio: values[4].unsafe_as(Float32)},
          vs_tiled_320: {bit_differences: values[5], tolerance_violations: values[6],
                         nonfinite: values[7], max_abs: values[8].unsafe_as(Float32),
                         max_tolerance_ratio: values[9].unsafe_as(Float32),
                         comparison_is_self_for_tiled_320: name == "tiled_320"},
          explicit_buffer_bytes: explicit_bytes,
          fixture_manifest: File.join(fixture_dir, "manifest.json")}.to_json)
    unless values[1] == 0 && values[2] == 0 && values[6] == 0 && values[7] == 0
      raise "strict upstream or native segmentation comparison failed for warmup #{name}"
    end
    validation.release
  end

  repetitions.times do |iteration|
    # Rotate each backend through first, middle, and last position to balance
    # thermal/clock drift across the three measured rounds.
    rotation = iteration % 3
    order = case rotation
            when 0 then backends
            when 1 then {backends[1], backends[2], backends[0]}
            else        {backends[2], backends[0], backends[1]}
            end
    order.each do |entry|
      name = entry[0]
      groups = entry[1]
      output = name == "tiled_320" ? output_320 : output_work
      chain = ML::Metal::MPSCommandChain.new
      allocations_before = ML::MetalBuffer.video_spec_allocations
      arm_started = Time.instant
      case name
      when "tiled_320", "tiled_1024"
        ML::Video::MetalOps.attention_tiled(chain.not_nil!, q, k, v, output,
          CROSS_Q_ROWS, CROSS_K_ROWS, CROSS_HEADS, CROSS_HEAD_DIM, 1, groups)
      when "mps_chunk_512"
        workspace.not_nil!.encode(chain.not_nil!)
      end
      attention_encode_seconds = (Time.instant - arm_started).total_seconds
      raise "attention encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocations_before
      chain.not_nil!.commit_and_wait
      attention_wall_seconds = (Time.instant - arm_started).total_seconds
      raise "attention arm did not use one terminal wait" unless chain.not_nil!.wait_count == 1
      raise "attention arm did not cross the segmented execution path" unless chain.not_nil!.segment_count > 0
      attention_gpu_seconds = chain.not_nil!.gpu_elapsed_seconds
      attention_segments = chain.not_nil!.segment_count
      chain.not_nil!.release
      chain = nil

      validation = ML::Metal::MPSCommandChain.new
      validation_started = Time.instant
      validation.encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(10)
      validation.encoder.set_pipeline(compare).set_buffer(output, 0).set_buffer(expected, 1)
        .set_buffer(output_320, 2).set_buffer(metrics, 3, ML::Metal::BufferAccess::ReadWrite)
        .set_value(CROSS_VALUES.to_u32, 4).set_value(CROSS_Q_ROWS.to_u32, 5)
        .set_value(CROSS_HEADS.to_u32, 6).set_value(CROSS_HEAD_DIM.to_u32, 7)
        .dispatch_1d(CROSS_VALUES.to_i32)
      validation.commit_and_wait
      validation_wall_seconds = (Time.instant - validation_started).total_seconds
      raise "validation did not use one terminal wait" unless validation.wait_count == 1
      values = metrics.contents.as(UInt32*)
      report = {
        schema:          "wan_s2v_cross_attention_benchmark_v1",
        backend:         name,
        segment_groups:  groups == 0 ? nil : groups,
        query_rows:      CROSS_Q_ROWS,
        key_rows:        CROSS_K_ROWS,
        heads:           CROSS_HEADS,
        head_dim:        CROSS_HEAD_DIM,
        compared_values: CROSS_VALUES,
        warmup:          false,
        repetition:      iteration,
        upstream:        {
          bit_differences:      values[0],
          tolerance_violations: values[1],
          nonfinite:            values[2],
          max_abs:              values[3].unsafe_as(Float32),
          max_tolerance_ratio:  values[4].unsafe_as(Float32),
        },
        vs_tiled_320: {
          bit_differences:                  values[5],
          tolerance_violations:             values[6],
          nonfinite:                        values[7],
          max_abs:                          values[8].unsafe_as(Float32),
          max_tolerance_ratio:              values[9].unsafe_as(Float32),
          comparison_is_self_for_tiled_320: name == "tiled_320",
        },
        attention_encode_seconds:                            attention_encode_seconds,
        host_attention_wall_seconds_including_terminal_wait: attention_wall_seconds,
        gpu_attention_seconds_excluding_validation:          attention_gpu_seconds,
        attention_command_segments:                          attention_segments,
        attention_terminal_waits:                            1,
        measured_order_rotation:                             rotation,
        validation_host_seconds_including_terminal_wait:     validation_wall_seconds,
        validation_gpu_seconds:                              validation.gpu_elapsed_seconds,
        validation_terminal_waits:                           validation.wait_count,
        encode_metal_buffer_allocations:                     0,
        explicit_buffer_bytes:                               explicit_bytes,
        setup_input_generation_wait_seconds:                 setup_seconds,
        fixture_manifest:                                    fixture_manifest_path,
        fixture_manifest_sha256:                             fixture_manifest_sha256,
        reference_torch_version:                             fixture_torch_version,
        wan_attention_source_sha256:                         fixture_wan_attention_sha256,
        wan_s2v_reference_adapter_sha256:                    fixture_adapter_sha256,
      }
      puts report.to_json
      unless values[1] == 0 && values[2] == 0 && values[6] == 0 && values[7] == 0
        raise "strict upstream or native segmentation comparison failed for #{name}"
      end
      validation.release
    end
  end

  # Prove the validation kernel rejects one deliberately corrupted output.
  # This is a one-element guard exercise, not part of any timed arm.
  control = ML::Metal::MPSCommandChain.new
  control.encoder.set_pipeline(corrupt)
    .set_buffer(control_actual, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  control.encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(10)
  control.encoder.set_pipeline(compare).set_buffer(control_actual, 0).set_buffer(control_expected, 1)
    .set_buffer(control_expected, 2).set_buffer(metrics, 3, ML::Metal::BufferAccess::ReadWrite)
    .set_value(1_u32, 4).set_value(1_u32, 5).set_value(1_u32, 6).set_value(1_u32, 7)
    .dispatch_1d(1)
  control.commit_and_wait
  raise "corruption negative control did not use one terminal wait" unless control.wait_count == 1
  control_values = metrics.contents.as(UInt32*)
  unless control_values[0] == 1 && control_values[1] == 1 && control_values[2] == 0 &&
         control_values[5] == 1 && control_values[6] == 1 && control_values[7] == 0
    raise "strict comparator failed to reject one corrupted value"
  end
  puts({schema: "wan_s2v_cross_attention_comparator_negative_control_v1",
        corrupted_values: 1, bit_differences: control_values[0],
        tolerance_violations: control_values[1], nonfinite: control_values[2],
        terminal_waits: control.wait_count}.to_json)
  control.release
ensure
  if active = chain
    begin
      active.commit_and_wait unless active.completed?
      active.release if active.completed?
    rescue
    end
  end
  workspace.try(&.release)
  arena.release
end
