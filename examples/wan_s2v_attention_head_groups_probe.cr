# Selected-query experiment for head-grouped, full-key Wan self-attention.
# This is a primitive probe; it does not load model weights or run generation.
require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/metal_attention_mps"
require "../src/ml/video/metal_attention"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"
require "./support/wan_s2v_steel_attention_probe"

private HG_Q_ROWS          = 36_608
private HG_K_ROWS          = 36_608
private HG_HEADS           =     40
private HG_HEAD_DIM        =    128
private HG_CHANNELS        = HG_HEADS * HG_HEAD_DIM
private HG_BASELINE_CHUNK  = 128
private HG_QUERY_ROWS      = {128, 257, 8_192}
private HG_CHUNK_ROWS      = {128, 256, 512, 1_024}
private HG_EXPLICIT_CAP    = 5_i64 * 1024 * 1024 * 1024
private HG_MAX_ROOTS       = 4_096
private HG_MANIFEST_SHA256 = "7586da395e4acedcd667639c91350119135183a0ed692fa994faeb683c6b850a"
private HG_EXPECTED_SHA256 = "8bf37c04b5effec694ec888a0a3c052160e827fcb7e0fb3603f5bfd939a709ef"
private HG_FORMULA         = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
private HG_MANIFEST_SCHEMA = "wan_s2v_self_attention_reference_v1"

private HG_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline float hg_value(uint index, uint salt) {
    uint mixed = index + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    return float(int(mixed & 1023u) - 512) / 256.0f;
  }

  kernel void hg_fill(device float *output [[buffer(0)]],
      constant uint &count [[buffer(1)]], constant uint &flat_base [[buffer(2)]],
      constant uint &salt [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = hg_value(flat_base + index, salt);
  }

  kernel void hg_nan_fill(device float *output [[buffer(0)]],
      constant uint &count [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = as_type<float>(0x7fc00000u);
  }

  kernel void hg_zero(device atomic_uint *metrics [[buffer(0)]],
      uint index [[thread_position_in_grid]]) {
    if (index < 5u) atomic_store_explicit(metrics + index, 0u, memory_order_relaxed);
  }

  // mode=1 maps the selected BHQD fixture slice; mode=0 compares native BQHD.
  kernel void hg_compare(device const float *reference [[buffer(0)]],
      device const float *actual [[buffer(1)]], device atomic_uint *metrics [[buffer(2)]],
      constant uint &count [[buffer(3)]], constant uint &query_rows [[buffer(4)]],
      constant uint &heads [[buffer(5)]], constant uint &head_dim [[buffer(6)]],
      constant uint &reference_bhqd [[buffer(7)]], uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    uint reference_index = index;
    if (reference_bhqd != 0u) {
      const uint channels = heads * head_dim;
      const uint query = index / channels;
      const uint head_channel = index % channels;
      const uint head = head_channel / head_dim;
      const uint channel = head_channel % head_dim;
      reference_index = head * query_rows * head_dim + query * head_dim + channel;
    }
    const float ref = reference[reference_index], got = actual[index];
    const uint ref_bits = as_type<uint>(ref), got_bits = as_type<uint>(got);
    const bool same_bits = ref_bits == got_bits;
    if (!isfinite(ref) || !isfinite(got)) {
      if (!same_bits) atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
      return;
    }
    if (same_bits) return;
    atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
    const float error = abs(ref - got);
    const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
    const float ratio = error / tolerance;
    if (error > tolerance) atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
    if (error > 0.0f) {
      atomic_fetch_max_explicit(metrics + 3, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 4, as_type<uint>(ratio), memory_order_relaxed);
    }
  }
METAL

# Read root summaries only after the arm's existing terminal wait.
class ML::Metal::MPSCommandChain
  def hg_root_timings_after_wait!(root_limit : Int32)
    unless completed? && wait_count == 1
      raise ArgumentError.new("Root timing requires one completed terminal wait")
    end
    count = segment_count
    raise ArgumentError.new("MPS root count is invalid") if count <= 0 || count > root_limit
    roots = [] of NamedTuple(sequence: Int32, gpu_start_seconds: Float64,
      gpu_end_seconds: Float64, gpu_interval_seconds: Float64)
    {% if flag?(:darwin) && !flag?(:cpu_only) %}
      count.times do |sequence|
        completed = 0_i32
        has_error = 0_i32
        gpu_start = 0.0
        gpu_end = 0.0
        status = MPSCommandChainFFI.root_timing(@native, sequence.to_i32,
          pointerof(completed), pointerof(has_error), pointerof(gpu_start), pointerof(gpu_end))
        raise ArgumentError.new("Failed to read root timing #{sequence} (#{status})") unless status == 0
        unless completed == 1 && has_error == 0 && gpu_start.finite? &&
               gpu_end.finite? && gpu_start > 0.0 && gpu_end > gpu_start
          raise ArgumentError.new("MPS root #{sequence} did not complete cleanly")
        end
        roots << {sequence: sequence.to_i32, gpu_start_seconds: gpu_start,
                  gpu_end_seconds: gpu_end, gpu_interval_seconds: gpu_end - gpu_start}
      end
    {% else %}
      raise ArgumentError.new("MPS root timing requires macOS Metal")
    {% end %}
    roots
  end
end

private record HGFixture, manifest_path : String, expected_path : String,
  manifest_sha256 : String, expected_sha256 : String, torch_version : String,
  attention_source_sha256 : String, adapter_sha256 : String, generator_sha256 : String

private record HGArm, name : String, heads_per_group : Int32, chunk_rows : Int32,
  output : ML::MetalBuffer,
  workspace : ML::Video::WanAttentionMPSChunkWorkspace | WanS2VSteelAttentionProbe

private def hg_sha256(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def hg_verify_source!(source : JSON::Any, path_key : String, digest_key : String) : String
  path = source[path_key].as_s
  raise ArgumentError.new("Missing pinned reference source for #{path_key}") unless File.file?(path)
  digest = source[digest_key].as_s
  unless digest.matches?(/\A[0-9a-f]{64}\z/) && hg_sha256(path) == digest
    raise ArgumentError.new("Reference source SHA-256 mismatch for #{path_key}")
  end
  digest
end

private def hg_validate_fixture!(fixture_dir : String) : HGFixture
  manifest_path = File.join(fixture_dir, "manifest.json")
  expected_path = File.join(fixture_dir, "expected_bhqd.f32")
  raise ArgumentError.new("Missing pinned attention fixture manifest") unless File.file?(manifest_path)
  raise ArgumentError.new("Missing pinned full-query expected output") unless File.file?(expected_path)
  manifest_sha = hg_sha256(manifest_path)
  expected_sha = hg_sha256(expected_path)
  raise ArgumentError.new("Pinned attention manifest SHA mismatch") unless manifest_sha == HG_MANIFEST_SHA256
  raise ArgumentError.new("Pinned expected output SHA mismatch") unless expected_sha == HG_EXPECTED_SHA256
  expected_bytes = HG_Q_ROWS.to_i64 * HG_CHANNELS * sizeof(Float32)
  raise ArgumentError.new("Full attention expected output byte size mismatch") unless File.size(expected_path) == expected_bytes

  manifest = JSON.parse(File.read(manifest_path))
  raise ArgumentError.new("Unsupported attention fixture schema") unless manifest["schema"].as_s == HG_MANIFEST_SCHEMA
  geometry = manifest["geometry"]
  { {"batches", 1_i64}, {"query_rows", HG_Q_ROWS.to_i64},
   {"key_rows", HG_K_ROWS.to_i64}, {"heads", HG_HEADS.to_i64},
   {"head_dim", HG_HEAD_DIM.to_i64} }.each do |key, value|
    raise ArgumentError.new("Attention geometry mismatch for #{key}") unless geometry[key].as_i64 == value
  end
  unless geometry["native_layout"].as_s == "B,Q,H,D" &&
         geometry["reference_layout"].as_s == "B,H,Q,D"
    raise ArgumentError.new("Attention fixture layouts mismatch")
  end
  raise ArgumentError.new("Attention input formula mismatch") unless manifest["input_formula"].as_s == HG_FORMULA
  salts = manifest["input_salts"]
  unless salts["q"].as_i64 == 1 && salts["k"].as_i64 == 2 && salts["v"].as_i64 == 3
    raise ArgumentError.new("Attention fixture input salts mismatch")
  end
  reference = manifest["reference"]
  unless reference["operator"].as_s == "torch.nn.functional.scaled_dot_product_attention" &&
         reference["device"].as_s == "mps" && reference["dtype"].as_s == "float32" &&
         reference["mps_fallback_disabled"].as_bool &&
         reference["comparison_tolerance"].as_s == "2e-5 + 2e-5 * abs(reference)" &&
         reference["chunking_semantics"].as_s == "independent query rows with full K/V; no mask, causal=false, dropout=0" &&
         reference["source_adapter"].as_s == "wan_s2v_real_stack_reference.py: transpose contiguous B,H,Q,D, PyTorch MPS SDPA, return B,H,Q,D"
    raise ArgumentError.new("Attention fixture oracle metadata mismatch")
  end
  torch_version = reference["torch_version"].as_s
  raise ArgumentError.new("Attention fixture requires PyTorch 2.8.x") unless torch_version.starts_with?("2.8.")
  unless [64_i64, 128_i64, 256_i64].includes?(reference["query_chunk_rows"].as_i64)
    raise ArgumentError.new("Attention fixture oracle chunking metadata mismatch")
  end
  source = manifest["source"]
  attention_sha = hg_verify_source!(source, "wan_attention_source_path", "wan_attention_source_sha256")
  adapter_sha = hg_verify_source!(source, "wan_s2v_reference_adapter_path", "wan_s2v_reference_adapter_sha256")
  generator_sha = source["generator_sha256"].as_s
  generator = File.expand_path("support/wan_s2v_self_attention_benchmark_reference.py", __DIR__)
  unless generator_sha.matches?(/\A[0-9a-f]{64}\z/) && hg_sha256(generator) == generator_sha
    raise ArgumentError.new("Attention fixture generator SHA mismatch")
  end
  expected = manifest["expected"]
  unless expected["file"].as_s == "expected_bhqd.f32" &&
         File.basename(expected["file"].as_s) == expected["file"].as_s &&
         expected["dtype"].as_s == "F32_LE" &&
         expected["shape"].as_a.map(&.as_i64) == [1_i64, HG_HEADS.to_i64, HG_Q_ROWS.to_i64, HG_HEAD_DIM.to_i64] &&
         expected["bytes"].as_i64 == expected_bytes && expected["sha256"].as_s == HG_EXPECTED_SHA256
    raise ArgumentError.new("Attention fixture expected-output metadata mismatch")
  end
  HGFixture.new(manifest_path, expected_path, manifest_sha, expected_sha,
    torch_version, attention_sha, adapter_sha, generator_sha)
end

private def hg_load_expected_slice(path : String, destination : ML::MetalBuffer,
                                   query_start : Int32, query_rows : Int32) : Nil
  destination_pointer = destination.contents.as(UInt8*)
  bytes_per_head = query_rows.to_i64 * HG_HEAD_DIM * sizeof(Float32)
  File.open(path, "rb") do |file|
    HG_HEADS.times do |head|
      source_offset = (head.to_i64 * HG_Q_ROWS + query_start) * HG_HEAD_DIM * sizeof(Float32)
      file.seek(source_offset, IO::Seek::Set)
      written = 0_i64
      while written < bytes_per_head
        amount = Math.min(1_i64 << 20, bytes_per_head - written).to_i32
        slice = Bytes.new(destination_pointer + head.to_i64 * bytes_per_head + written, amount)
        read = file.read(slice)
        raise ArgumentError.new("Truncated selected expected-output slice") if read == 0
        written += read
      end
    end
  end
end

private def hg_plan(query_rows : Int32, candidate_chunk : Int32,
                    candidate_group : Int32, baseline_group : Int32,
                    same_chunk_control : Bool, key_shard_rows : Int32 = 0)
  query_values = query_rows.to_i64 * HG_CHANNELS
  kv_values = HG_K_ROWS.to_i64 * HG_CHANNELS
  arms = same_chunk_control ? 3 : 2
  base_scratch = 3_i64 * HG_BASELINE_CHUNK * baseline_group * HG_HEAD_DIM
  candidate_scratch = 3_i64 * candidate_chunk * candidate_group * HG_HEAD_DIM
  scratch_values = base_scratch + candidate_scratch + base_scratch + candidate_scratch
  scratch_values += 3_i64 * candidate_chunk * HG_HEADS * HG_HEAD_DIM if same_chunk_control
  explicit_values = query_values * (4 + arms) + 4_i64 * kv_values + scratch_values
  # Fresh and reused candidate workspaces each own an F32 (max, sum) per row/head.
  explicit_values += 4_i64 * query_rows * HG_HEADS if key_shard_rows > 0
  metric_sets = arms + 1 + (same_chunk_control ? 1 : 0) + 1 + 2 + 2
  explicit_values += 4 + metric_sets * 5
  {query_values: query_values, kv_values: kv_values, scratch_values: scratch_values,
   active_arms: arms, metric_sets: metric_sets, explicit_bytes: explicit_values * sizeof(Float32)}
end

private def hg_encode_fill(encoder : ML::Metal::ComputeEncoder,
                           pipeline : ML::Metal::ComputePipeline,
                           buffer : ML::MetalBuffer, count : Int32,
                           flat_base : UInt32, salt : UInt32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).set_value(flat_base, 2).set_value(salt, 3)
  encoder.dispatch_1d(count)
end

private def hg_encode_nan(encoder : ML::Metal::ComputeEncoder,
                          pipeline : ML::Metal::ComputePipeline,
                          buffer : ML::MetalBuffer, count : Int32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).dispatch_1d(count)
end

private def hg_encode_compare(encoder : ML::Metal::ComputeEncoder,
                              zero : ML::Metal::ComputePipeline,
                              compare : ML::Metal::ComputePipeline,
                              reference : ML::MetalBuffer, actual : ML::MetalBuffer,
                              metrics : ML::MetalBuffer, count : Int32,
                              query_rows : Int32, fixture_layout : Bool) : Nil
  encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  encoder.set_pipeline(compare).set_buffer(reference, 0).set_buffer(actual, 1)
  encoder.set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
  encoder.set_value(count.to_u32, 3).set_value(query_rows.to_u32, 4)
  encoder.set_value(HG_HEADS.to_u32, 5).set_value(HG_HEAD_DIM.to_u32, 6)
  encoder.set_value(fixture_layout ? 1_u32 : 0_u32, 7).dispatch_1d(count)
end

private def hg_metrics(buffer : ML::MetalBuffer)
  values = buffer.contents.as(UInt32*)
  {bit_differences: values[0], tolerance_violations: values[1], nonfinite: values[2],
   max_abs_error: values[3].unsafe_as(Float32), max_tolerance_ratio: values[4].unsafe_as(Float32)}
end

private def hg_run_arm(arm : HGArm, query_rows : Int32,
                       nan_fill : ML::Metal::ComputePipeline)
  total_started = Time.instant
  allocation_count = ML::MetalBuffer.video_spec_allocations
  chain = ML::Metal::MPSCommandChain.new
  begin
    encode_started = Time.instant
    encoder = chain.encoder
    hg_encode_nan(encoder, nan_fill, arm.output, query_rows * HG_CHANNELS)
    arm.workspace.encode(chain)
    encode_seconds = (Time.instant - encode_started).total_seconds
    chain.commit_and_wait
    completed_seconds = (Time.instant - total_started).total_seconds
    allocations = ML::MetalBuffer.video_spec_allocations - allocation_count
    raise "attention arm allocated a tracked MetalBuffer" unless allocations == 0
    raise "attention arm did not use one terminal wait" unless chain.wait_count == 1
    roots = chain.hg_root_timings_after_wait!(HG_MAX_ROOTS)
    result = {
      schema: "wan_s2v_attention_head_groups_arm_v1", arm: arm.name,
      heads_per_group: arm.heads_per_group, group_count: HG_HEADS // arm.heads_per_group,
      chunk_rows: arm.chunk_rows, completed_wall_seconds: completed_seconds,
      backend: arm.workspace.is_a?(WanS2VSteelAttentionProbe) ? "native_steel_f32" : "mpsgraph_sdpa",
      encode_wall_seconds: encode_seconds,
      summed_command_root_gpu_interval_seconds: chain.gpu_elapsed_seconds,
      distinct_executable_count: arm.workspace.distinct_executable_count,
      segment_count: chain.segment_count, terminal_waits: chain.wait_count,
      tracked_metal_buffer_allocations: allocations, output_prefilled_nan: true,
      root_spans: roots,
      root_timing_boundary: "GPU root intervals are command-buffer intervals, not kernel-only timestamps",
    }
    chain.release
    result
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def hg_validate_original(arms : Array(HGArm), expected : ML::MetalBuffer,
                                 metrics : Hash(String, ML::MetalBuffer),
                                 zero : ML::Metal::ComputePipeline,
                                 compare : ML::Metal::ComputePipeline,
                                 control_reference_plus : ML::MetalBuffer,
                                 control_actual_plus : ML::MetalBuffer,
                                 control_reference_nan : ML::MetalBuffer,
                                 control_actual_nan : ML::MetalBuffer,
                                 query_rows : Int32, require_bitexact : Bool = false) : Nil
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    count = query_rows * HG_CHANNELS
    arms.each do |arm|
      hg_encode_compare(encoder, zero, compare, expected, arm.output,
        metrics["fixture_#{arm.name}"], count, query_rows, true)
    end
    hg_encode_compare(encoder, zero, compare, arms[0].output, arms[1].output,
      metrics["baseline_candidate"], count, query_rows, false)
    if arms.size == 3
      hg_encode_compare(encoder, zero, compare, arms[1].output, arms[2].output,
        metrics["candidate_same_chunk"], count, query_rows, false)
    end
    hg_encode_compare(encoder, zero, compare, control_reference_plus,
      control_actual_plus, metrics["positive_control"], 1, 1, false)
    hg_encode_compare(encoder, zero, compare, control_reference_nan,
      control_actual_nan, metrics["nan_control"], 1, 1, false)
    chain.commit_and_wait
    raise "attention diagnostics did not use one terminal wait" unless chain.wait_count == 1
    result_names = arms.map { |arm| "fixture_#{arm.name}" }
    result_names << "baseline_candidate"
    result_names << "candidate_same_chunk" if arms.size == 3
    result_names << "positive_control" << "nan_control"
    results = Hash(String, typeof(hg_metrics(metrics["positive_control"]))).new
    result_names.each { |name| results[name] = hg_metrics(metrics[name]) }
    arms.each do |arm|
      result = results["fixture_#{arm.name}"]
      unless result[:tolerance_violations] == 0 && result[:nonfinite] == 0
        raise "#{arm.name} failed strict upstream fixture comparison"
      end
    end
    pair = results["baseline_candidate"]
    unless pair[:tolerance_violations] == 0 && pair[:nonfinite] == 0
      raise "candidate differs from baseline beyond the strict tolerance"
    end
    if require_bitexact && pair[:bit_differences] != 0
      raise "sharded candidate is not bit-identical to the unsharded native control"
    end
    if arms.size == 3
      control = results["candidate_same_chunk"]
      unless control[:tolerance_violations] == 0 && control[:nonfinite] == 0
        raise "same-chunk control differs beyond the strict tolerance"
      end
    end
    plus = results["positive_control"]
    nan = results["nan_control"]
    unless plus[:tolerance_violations] > 0 && plus[:nonfinite] == 0 && nan[:nonfinite] > 0
      raise "strict comparator negative controls were not rejected"
    end
    puts({schema: "wan_s2v_attention_head_groups_validation_v1",
          phase: "original_fixture", query_rows: query_rows, compared_values: count,
          metrics: results, positive_control_rejected: true, nan_control_rejected: true,
          native_bitexact_required: require_bitexact,
          terminal_waits: chain.wait_count, passed: true}.to_json)
    STDOUT.flush
    chain.release
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def hg_validate_mutated(original_baseline : ML::MetalBuffer,
                                original_candidate : ML::MetalBuffer,
                                mutated_baseline : ML::MetalBuffer,
                                mutated_candidate : ML::MetalBuffer,
                                metrics : Hash(String, ML::MetalBuffer),
                                zero : ML::Metal::ComputePipeline,
                                compare : ML::Metal::ComputePipeline,
                                query_rows : Int32, require_bitexact : Bool = false) : Nil
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    count = query_rows * HG_CHANNELS
    hg_encode_compare(encoder, zero, compare, mutated_baseline, mutated_candidate,
      metrics["mutated_baseline_candidate"], count, query_rows, false)
    hg_encode_compare(encoder, zero, compare, original_baseline, mutated_baseline,
      metrics["mutation_changed_baseline"], count, query_rows, false)
    hg_encode_compare(encoder, zero, compare, original_candidate, mutated_candidate,
      metrics["mutation_changed_candidate"], count, query_rows, false)
    chain.commit_and_wait
    raise "changed-K/V diagnostics did not use one terminal wait" unless chain.wait_count == 1
    results = {
      mutated_baseline_candidate: hg_metrics(metrics["mutated_baseline_candidate"]),
      mutation_changed_baseline:  hg_metrics(metrics["mutation_changed_baseline"]),
      mutation_changed_candidate: hg_metrics(metrics["mutation_changed_candidate"]),
    }
    strict = results[:mutated_baseline_candidate]
    unless strict[:tolerance_violations] == 0 && strict[:nonfinite] == 0
      raise "changed-K/V candidate differs from fresh baseline beyond strict tolerance"
    end
    if require_bitexact && strict[:bit_differences] != 0
      raise "changed-K/V sharded candidate is not bit-identical to native control"
    end
    {results[:mutation_changed_baseline], results[:mutation_changed_candidate]}.each do |changed|
      unless changed[:bit_differences] > 0 && changed[:nonfinite] == 0
        raise "changed-K/V replay did not change a finite output"
      end
    end
    puts({schema: "wan_s2v_attention_head_groups_validation_v1",
          phase: "changed_kv_replay", k_salt: 5, v_salt: 6, query_rows: query_rows,
          compared_values: count, metrics: results, terminal_waits: chain.wait_count,
          native_bitexact_required: require_bitexact,
          changed_kv_attested: true, passed: true}.to_json)
    STDOUT.flush
    chain.release
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def hg_validate_reused_changed_kv(arms : Array(HGArm),
                                          mutated_baseline : ML::MetalBuffer,
                                          mutated_candidate : ML::MetalBuffer,
                                          metrics : Hash(String, ML::MetalBuffer),
                                          zero : ML::Metal::ComputePipeline,
                                          compare : ML::Metal::ComputePipeline,
                                          query_rows : Int32, require_bitexact : Bool = false) : Nil
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    count = query_rows * HG_CHANNELS
    hg_encode_compare(encoder, zero, compare, mutated_baseline, arms[0].output,
      metrics["baseline_candidate"], count, query_rows, false)
    hg_encode_compare(encoder, zero, compare, mutated_candidate, arms[1].output,
      metrics["mutated_baseline_candidate"], count, query_rows, false)
    chain.commit_and_wait
    raise "reused-workspace diagnostics did not use one terminal wait" unless chain.wait_count == 1
    baseline = hg_metrics(metrics["baseline_candidate"])
    candidate = hg_metrics(metrics["mutated_baseline_candidate"])
    {baseline, candidate}.each do |result|
      unless result[:tolerance_violations] == 0 && result[:nonfinite] == 0
        raise "reused workspace differs from its fresh changed-K/V counterpart"
      end
      if require_bitexact && result[:bit_differences] != 0
        raise "reused changed-K/V workspace lost bit identity"
      end
    end
    puts({schema: "wan_s2v_attention_head_groups_validation_v1",
          phase: "reused_changed_kv_workspace", k_salt: 5, v_salt: 6,
          query_rows: query_rows, compared_values_per_arm: count,
          metrics: {baseline: baseline, candidate: candidate},
          terminal_waits: chain.wait_count, original_workspaces_reused: true,
          native_bitexact_required: require_bitexact,
          matched_fresh_changed_kv_outputs: true, passed: true}.to_json)
    STDOUT.flush
    chain.release
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def hg_setup_inputs(q : ML::MetalBuffer, k : ML::MetalBuffer,
                            v : ML::MetalBuffer, outputs : Array(ML::MetalBuffer),
                            fill : ML::Metal::ComputePipeline,
                            nan_fill : ML::Metal::ComputePipeline,
                            query_rows : Int32, query_start : Int32,
                            query_values : Int64, kv_values : Int64,
                            k_salt : UInt32, v_salt : UInt32) : Float64
  started = Time.instant
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    hg_encode_fill(encoder, fill, q, query_values.to_i32,
      (query_start.to_i64 * HG_CHANNELS).to_u32, 1_u32)
    hg_encode_fill(encoder, fill, k, kv_values.to_i32, 0_u32, k_salt)
    hg_encode_fill(encoder, fill, v, kv_values.to_i32, 0_u32, v_salt)
    outputs.each { |output| hg_encode_nan(encoder, nan_fill, output, query_values.to_i32) }
    chain.commit_and_wait
    raise "input setup did not use one terminal wait" unless chain.wait_count == 1
    elapsed = (Time.instant - started).total_seconds
    chain.release
    elapsed
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def hg_setup_kv(k : ML::MetalBuffer, v : ML::MetalBuffer,
                        fill : ML::Metal::ComputePipeline, kv_values : Int64,
                        k_salt : UInt32, v_salt : UInt32) : Float64
  started = Time.instant
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    hg_encode_fill(encoder, fill, k, kv_values.to_i32, 0_u32, k_salt)
    hg_encode_fill(encoder, fill, v, kv_values.to_i32, 0_u32, v_salt)
    chain.commit_and_wait
    raise "changed K/V setup did not use one terminal wait" unless chain.wait_count == 1
    elapsed = (Time.instant - started).total_seconds
    chain.release
    elapsed
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

fixture_dir = ""
query_rows = 128
query_start = 0
query_start_set = false
tail = false
baseline_group = HG_HEADS
candidate_group = 1
candidate_chunk = 1_024
pairs = 1
order = "AB"
same_chunk_control = false
plan_only = false
steel_source_path = ""
baseline_steel_source_path = ""
key_shard_rows = 0
require_native_bitexact = false
OptionParser.parse do |parser|
  parser.banner = "Usage: wan_s2v_attention_head_groups_probe --fixture=DIR [--query-rows=128|257|8192] [--baseline-heads-per-group=G] [--candidate-heads-per-group=G] [--candidate-chunk=128|256|512|1024] [--pairs=1|3] [--order=AB|BA] [--same-chunk-control]"
  parser.on("--fixture=DIR", "Pinned full-query upstream attention fixture") { |value| fixture_dir = value }
  parser.on("--query-rows=N", "Selected query rows (default 128)") { |value| query_rows = value.to_i }
  parser.on("--query-start=N", "First full-sequence query row (default 0)") do |value|
    query_start = value.to_i
    query_start_set = true
  end
  parser.on("--tail", "Select the final query rows") { tail = true }
  parser.on("--baseline-heads-per-group=N", "Baseline group size dividing 40 at chunk 128 (default 40)") { |value| baseline_group = value.to_i }
  parser.on("--candidate-heads-per-group=N", "Candidate group size dividing 40 (default 1)") { |value| candidate_group = value.to_i }
  parser.on("--candidate-chunk=N", "Candidate query chunk rows: 128, 256, 512, 1024 (default 1024)") { |value| candidate_chunk = value.to_i }
  parser.on("--pairs=N", "Measured A/B pairs: 1 or 3 (default 1)") { |value| pairs = value.to_i }
  parser.on("--order=AB|BA", "Pair order; 3 pairs alternate to balance chronology") { |value| order = value }
  parser.on("--same-chunk-control", "Opt in to all-head G=40 at candidate chunk size") { same_chunk_control = true }
  parser.on("--plan-only", "Print CPU-only plan without Metal or fixture access") { plan_only = true }
  parser.on("--candidate-steel-source=FILE", "Opt in to the pinned standalone F32 STEEL probe shader; group/chunk metadata applies to the MPS control only") { |value| steel_source_path = File.expand_path(value) }
  parser.on("--baseline-steel-source=FILE", "Use an unsharded native F32 STEEL control") { |value| baseline_steel_source_path = File.expand_path(value) }
  parser.on("--candidate-steel-key-shard=N", "Continue exact online state across 2048-key GPU shards") { |value| key_shard_rows = value.to_i }
  parser.on("--require-native-bitexact", "Require candidate/native control bit identity") { require_native_bitexact = true }
end

raise ArgumentError.new("--query-rows must be 128, 257, or 8192") unless HG_QUERY_ROWS.includes?(query_rows)
unless baseline_group > 0 && baseline_group <= HG_HEADS && HG_HEADS % baseline_group == 0
  raise ArgumentError.new("--baseline-heads-per-group must divide 40")
end
unless candidate_group > 0 && candidate_group <= HG_HEADS && HG_HEADS % candidate_group == 0
  raise ArgumentError.new("--candidate-heads-per-group must divide 40")
end
raise ArgumentError.new("--candidate-chunk must be 128, 256, 512, or 1024") unless HG_CHUNK_ROWS.includes?(candidate_chunk)
raise ArgumentError.new("--pairs must be 1 or 3") unless {1, 3}.includes?(pairs)
raise ArgumentError.new("--order must be AB or BA") unless {"AB", "BA"}.includes?(order)
raise ArgumentError.new("--tail cannot combine with --query-start") if tail && query_start_set
raise ArgumentError.new("All-head candidate is already the same-chunk control") if same_chunk_control && candidate_group == HG_HEADS
raise ArgumentError.new("STEEL probe does not support the same-chunk control") if !steel_source_path.empty? && same_chunk_control
unless {0, 2_048}.includes?(key_shard_rows)
  raise ArgumentError.new("--candidate-steel-key-shard must be zero or 2048")
end
if key_shard_rows > 0 && steel_source_path.empty?
  raise ArgumentError.new("Key shards require --candidate-steel-source")
end
if !baseline_steel_source_path.empty? && (steel_source_path.empty? || same_chunk_control)
  raise ArgumentError.new("Native baseline requires a native candidate and no same-chunk control")
end
if require_native_bitexact && (baseline_steel_source_path.empty? || key_shard_rows == 0)
  raise ArgumentError.new("Bit identity requires native baseline and sharded candidate")
end
query_start = HG_Q_ROWS - query_rows if tail
unless query_start >= 0 && query_start.to_i64 + query_rows <= HG_Q_ROWS
  raise ArgumentError.new("--query-start and --query-rows exceed full query extent")
end
fixture_dir = File.expand_path(fixture_dir) unless fixture_dir.empty?
buffer_plan = hg_plan(query_rows, candidate_chunk, candidate_group, baseline_group, same_chunk_control, key_shard_rows)
raise ArgumentError.new("Explicit MetalBuffer plan exceeds 5 GiB") if buffer_plan[:explicit_bytes] > HG_EXPLICIT_CAP

if plan_only
  puts({schema: "wan_s2v_attention_head_groups_plan_v1", query_start: query_start,
        query_rows: query_rows, key_rows: HG_K_ROWS, heads: HG_HEADS, head_dim: HG_HEAD_DIM,
        baseline: {heads_per_group: baseline_group, group_count: HG_HEADS // baseline_group, chunk_rows: HG_BASELINE_CHUNK},
        candidate: {heads_per_group: candidate_group, group_count: HG_HEADS // candidate_group,
                    chunk_rows: candidate_chunk},
        same_chunk_control: same_chunk_control ? {heads_per_group: HG_HEADS, chunk_rows: candidate_chunk} : nil,
        candidate_backend: steel_source_path.empty? ? "mpsgraph_sdpa" : "native_steel_f32",
        baseline_backend: baseline_steel_source_path.empty? ? "mpsgraph_sdpa" : "native_steel_f32",
        key_shard_rows: key_shard_rows, native_bitexact_required: require_native_bitexact,
        steel_geometry: steel_source_path.empty? ? nil : {query_tile: WanS2VSteelAttentionProbe::QUERY_TILE, key_tile: WanS2VSteelAttentionProbe::KEY_TILE, simd_groups: WanS2VSteelAttentionProbe::SIMD_GROUPS, head_grouping_applies: false},
        changed_kv_replay: {k_salt: 5, v_salt: 6}, pairs: pairs, order: order,
        explicit_buffer_bytes: buffer_plan[:explicit_bytes], explicit_buffer_cap_bytes: HG_EXPLICIT_CAP,
        explicit_buffer_cap_passed: true, opaque_mps_private_workspace_included: false,
        fixture_required_for_plan_only: false, model_weights_loaded: false,
        scope: "selected-query full-K attention primitive only; no complete block, model, or generation"}.to_json)
  exit 0
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
fixture = hg_validate_fixture!(fixture_dir)
steel_source = steel_source_path.empty? ? "" : File.read(steel_source_path)
baseline_steel_source = baseline_steel_source_path.empty? ? "" : File.read(baseline_steel_source_path)

puts({schema: "wan_s2v_attention_head_groups_setup_v1",
      device_name: ML::Metal::Device.instance.name,
      query_start: query_start, query_rows: query_rows, key_rows: HG_K_ROWS,
      heads: HG_HEADS, head_dim: HG_HEAD_DIM, channels: HG_CHANNELS,
      input_formula: HG_FORMULA, q_salt: 1, k_salt: 2, v_salt: 3,
      fixture_manifest_path: fixture.manifest_path,
      fixture_manifest_sha256: fixture.manifest_sha256,
      expected_full_file_sha256: fixture.expected_sha256,
      candidate_backend: steel_source.empty? ? "mpsgraph_sdpa" : "native_steel_f32",
      baseline_backend: baseline_steel_source.empty? ? "mpsgraph_sdpa" : "native_steel_f32",
      baseline_shader_sha256: baseline_steel_source.empty? ? nil : Digest::SHA256.hexdigest(baseline_steel_source),
      key_shard_rows: key_shard_rows, native_bitexact_required: require_native_bitexact,
      candidate_shader_sha256: steel_source.empty? ? nil : Digest::SHA256.hexdigest(steel_source),
      steel_geometry: steel_source.empty? ? nil : {query_tile: WanS2VSteelAttentionProbe::QUERY_TILE, key_tile: WanS2VSteelAttentionProbe::KEY_TILE, simd_groups: WanS2VSteelAttentionProbe::SIMD_GROUPS, head_grouping_applies: false},
      reference_torch_version: fixture.torch_version,
      wan_attention_source_sha256: fixture.attention_source_sha256,
      wan_s2v_reference_adapter_sha256: fixture.adapter_sha256,
      generator_sha256: fixture.generator_sha256,
      baseline: {heads_per_group: baseline_group, group_count: HG_HEADS // baseline_group, chunk_rows: HG_BASELINE_CHUNK},
      candidate: {heads_per_group: candidate_group, group_count: HG_HEADS // candidate_group,
                  chunk_rows: candidate_chunk},
      same_chunk_control: same_chunk_control ? {heads_per_group: HG_HEADS,
                                                group_count: 1, chunk_rows: candidate_chunk} : nil,
      explicit_buffer_bytes: buffer_plan[:explicit_bytes], explicit_buffer_cap_bytes: HG_EXPLICIT_CAP,
      opaque_mps_private_workspace_bytes: "not included in explicit MetalBuffer accounting",
      pairs: pairs, requested_order: order, model_weights_loaded: false,
      validation_tolerance: "2e-5+2e-5*abs(reference)",
      timed_arm_scope: "NaN poison, full BTHD-to-head-major K/V refresh, selected backend attention, and one terminal wait",
      all_outputs_compared_to_pinned_upstream: true,
      scope: "selected-query attention only; no full block/model/video-throughput claim"}.to_json)
STDOUT.flush

arena = ML::Video::Arena.new
arms = [] of HGArm
mutation_arms = [] of HGArm
begin
  query_values = buffer_plan[:query_values]
  kv_values = buffer_plan[:kv_values]
  q = arena.alloc("hg_query_input", query_values.to_i32)
  k = arena.alloc("hg_full_key_input", kv_values.to_i32)
  v = arena.alloc("hg_full_value_input", kv_values.to_i32)
  prepared_k = arena.alloc("hg_head_major_key", kv_values.to_i32)
  prepared_v = arena.alloc("hg_head_major_value", kv_values.to_i32)
  expected = arena.alloc("hg_upstream_expected_bhqd", query_values.to_i32)

  baseline_name = "baseline_g#{baseline_group}_c#{HG_BASELINE_CHUNK}"
  arm_specs = [{baseline_name, baseline_group, HG_BASELINE_CHUNK},
               {"candidate", candidate_group, candidate_chunk}]
  arm_specs << {"same_chunk_g40", HG_HEADS, candidate_chunk} if same_chunk_control
  outputs = [] of ML::MetalBuffer
  arm_specs.each do |name, group, chunk|
    output = arena.alloc("hg_output_#{name}", query_values.to_i32)
    outputs << output
    scratch_values = chunk.to_i64 * group * HG_HEAD_DIM
    chunk_q = arena.alloc("hg_#{name}_chunk_q", scratch_values.to_i32)
    chunk_a = arena.alloc("hg_#{name}_chunk_output_a", scratch_values.to_i32)
    chunk_b = arena.alloc("hg_#{name}_chunk_output_b", scratch_values.to_i32)
    workspace = if name == "candidate" && !steel_source.empty?
                  row_state = key_shard_rows > 0 ? arena.alloc("hg_candidate_row_state", query_rows * HG_HEADS * 2) : nil
                  WanS2VSteelAttentionProbe.new(query_rows, HG_K_ROWS, HG_HEADS,
                    q, k, v, output, prepared_k, prepared_v, steel_source, key_shard_rows, row_state)
                elsif name == baseline_name && !baseline_steel_source.empty?
                  WanS2VSteelAttentionProbe.new(query_rows, HG_K_ROWS, HG_HEADS,
                    q, k, v, output, prepared_k, prepared_v, baseline_steel_source)
                else
                  ML::Video::WanAttentionMPSChunkWorkspace.new(1, query_rows,
                    HG_K_ROWS, HG_HEADS, HG_HEAD_DIM, chunk, q, k, v, output,
                    chunk_q, chunk_a, chunk_b, prepared_k, prepared_v, heads_per_group: group)
                end
    arms << HGArm.new(name, group, chunk, output, workspace)
  end

  mutated_baseline = arena.alloc("hg_mutated_baseline_output", query_values.to_i32)
  mutated_candidate = arena.alloc("hg_mutated_candidate_output", query_values.to_i32)
  outputs << mutated_baseline << mutated_candidate
  {arms[0], arms[1]}.each do |original|
    output = original.name == baseline_name ? mutated_baseline : mutated_candidate
    scratch_values = original.chunk_rows.to_i64 * original.heads_per_group * HG_HEAD_DIM
    chunk_q = arena.alloc("hg_mutation_#{original.name}_chunk_q", scratch_values.to_i32)
    chunk_a = arena.alloc("hg_mutation_#{original.name}_chunk_output_a", scratch_values.to_i32)
    chunk_b = arena.alloc("hg_mutation_#{original.name}_chunk_output_b", scratch_values.to_i32)
    workspace = if original.name == "candidate" && !steel_source.empty?
                  row_state = key_shard_rows > 0 ? arena.alloc("hg_mutation_candidate_row_state", query_rows * HG_HEADS * 2) : nil
                  WanS2VSteelAttentionProbe.new(query_rows, HG_K_ROWS, HG_HEADS,
                    q, k, v, output, prepared_k, prepared_v, steel_source, key_shard_rows, row_state)
                elsif original.name == baseline_name && !baseline_steel_source.empty?
                  WanS2VSteelAttentionProbe.new(query_rows, HG_K_ROWS, HG_HEADS,
                    q, k, v, output, prepared_k, prepared_v, baseline_steel_source)
                else
                  ML::Video::WanAttentionMPSChunkWorkspace.new(1, query_rows,
                    HG_K_ROWS, HG_HEADS, HG_HEAD_DIM, original.chunk_rows, q, k, v, output,
                    chunk_q, chunk_a, chunk_b, prepared_k, prepared_v,
                    heads_per_group: original.heads_per_group)
                end
    mutation_arms << HGArm.new("mutated_#{original.name}", original.heads_per_group,
      original.chunk_rows, output, workspace)
  end

  control_reference_plus = arena.alloc("hg_control_plus_reference", 1)
  control_actual_plus = arena.alloc("hg_control_plus_actual", 1)
  control_reference_nan = arena.alloc("hg_control_nan_reference", 1)
  control_actual_nan = arena.alloc("hg_control_nan_actual", 1)
  control_reference_plus.contents.as(Float32*)[0] = 0.0f32
  control_actual_plus.contents.as(Float32*)[0] = 1.0f32
  control_reference_nan.contents.as(Float32*)[0] = 0.0f32
  control_actual_nan.contents.as(Float32*)[0] = Float32::NAN

  metrics = Hash(String, ML::MetalBuffer).new
  metric_names = arms.map { |arm| "fixture_#{arm.name}" }
  metric_names << "baseline_candidate" << "mutated_baseline_candidate"
  metric_names << "mutation_changed_baseline" << "mutation_changed_candidate"
  metric_names << "positive_control" << "nan_control"
  metric_names << "candidate_same_chunk" if same_chunk_control
  metric_names.each { |name| metrics[name] = arena.alloc("hg_metrics_#{name}", 5) }
  unless arena.resident_bytes == buffer_plan[:explicit_bytes]
    raise ArgumentError.new("Actual explicit buffer bytes differ from plan (#{arena.resident_bytes} vs #{buffer_plan[:explicit_bytes]})")
  end

  fill_pipeline = ML::Metal::ComputePipeline.new("hg_fill", HG_PROBE_SOURCE)
  nan_pipeline = ML::Metal::ComputePipeline.new("hg_nan_fill", HG_PROBE_SOURCE)
  zero_pipeline = ML::Metal::ComputePipeline.new("hg_zero", HG_PROBE_SOURCE)
  compare_pipeline = ML::Metal::ComputePipeline.new("hg_compare", HG_PROBE_SOURCE)
  hg_load_expected_slice(fixture.expected_path, expected, query_start, query_rows)

  setup_seconds = hg_setup_inputs(q, k, v, outputs, fill_pipeline, nan_pipeline,
    query_rows, query_start, query_values, kv_values, 2_u32, 3_u32)
  puts({schema: "wan_s2v_attention_head_groups_input_setup_v1",
        host_wall_seconds: setup_seconds, terminal_waits: 1,
        explicit_buffer_bytes: arena.resident_bytes, qkv_fill_gpu: true,
        expected_slice_loaded: true, outputs_prefilled_nan: true}.to_json)
  STDOUT.flush

  warm_order = order == "AB" ? [0, 1] : [1, 0]
  warm_order.each { |index| puts(hg_run_arm(arms[index], query_rows, nan_pipeline).to_json) }
  puts(hg_run_arm(arms[2], query_rows, nan_pipeline).to_json) if arms.size == 3
  hg_validate_original(arms, expected, metrics, zero_pipeline, compare_pipeline,
    control_reference_plus, control_actual_plus, control_reference_nan,
    control_actual_nan, query_rows, require_native_bitexact)

  measured_wall = Hash(String, Array(Float64)).new { |hash, key| hash[key] = [] of Float64 }
  pairs.times do |pair_index|
    pair_order = if order == "AB"
                   pair_index.odd? ? [1, 0] : [0, 1]
                 else
                   pair_index.odd? ? [0, 1] : [1, 0]
                 end
    pair_label = pair_order == [0, 1] ? "AB" : "BA"
    pair_order.each do |index|
      result = hg_run_arm(arms[index], query_rows, nan_pipeline)
      measured_wall[arms[index].name] << result[:completed_wall_seconds]
      puts result.merge({phase: "measured", pair: pair_index + 1, pair_order: pair_label}).to_json
      STDOUT.flush
    end
    if arms.size == 3
      result = hg_run_arm(arms[2], query_rows, nan_pipeline)
      measured_wall[arms[2].name] << result[:completed_wall_seconds]
      puts result.merge({phase: "measured", pair: pair_index + 1,
                         pair_order: "control_after_#{pair_label}"}).to_json
      STDOUT.flush
    end
    hg_validate_original(arms, expected, metrics, zero_pipeline, compare_pipeline,
      control_reference_plus, control_actual_plus, control_reference_nan,
      control_actual_nan, query_rows, require_native_bitexact)
  end

  mutation_setup = hg_setup_kv(k, v, fill_pipeline, kv_values, 5_u32, 6_u32)
  mutation_order = order == "AB" ? [0, 1] : [1, 0]
  mutation_order.each do |index|
    result = hg_run_arm(mutation_arms[index], query_rows, nan_pipeline)
    puts result.merge({phase: "changed_kv_replay", k_salt: 5, v_salt: 6}).to_json
    STDOUT.flush
  end
  hg_validate_mutated(arms[0].output, arms[1].output, mutated_baseline,
    mutated_candidate, metrics, zero_pipeline, compare_pipeline, query_rows, require_native_bitexact)
  mutation_order.each do |index|
    result = hg_run_arm(arms[index], query_rows, nan_pipeline)
    puts result.merge({phase: "reused_changed_kv_workspace", k_salt: 5, v_salt: 6}).to_json
    STDOUT.flush
  end
  hg_validate_reused_changed_kv(arms, mutated_baseline, mutated_candidate,
    metrics, zero_pipeline, compare_pipeline, query_rows, require_native_bitexact)
  puts({schema: "wan_s2v_attention_head_groups_summary_v1", query_start: query_start,
        query_rows: query_rows, baseline_heads_per_group: baseline_group,
        baseline_chunk_rows: HG_BASELINE_CHUNK, candidate_heads_per_group: candidate_group,
        candidate_chunk_rows: candidate_chunk, measured_pairs: pairs, requested_order: order,
        candidate_backend: steel_source.empty? ? "mpsgraph_sdpa" : "native_steel_f32",
        baseline_backend: baseline_steel_source.empty? ? "mpsgraph_sdpa" : "native_steel_f32",
        key_shard_rows: key_shard_rows, native_bitexact_required: require_native_bitexact,
        measured_completed_wall_seconds: measured_wall, mutation_setup_host_seconds: mutation_setup,
        explicit_buffer_bytes: arena.resident_bytes, full_kv_rows: HG_K_ROWS,
        default_production_workspace_changed: false,
        scope: "selected-query attention only; timings include K/V preparation and backend layout work"}.to_json)
  STDOUT.flush
ensure
  (mutation_arms + arms).reverse_each do |arm|
    begin
      arm.workspace.release
    rescue
    end
  end
  arena.release
end
