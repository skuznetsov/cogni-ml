# Bounded long-key MPS self-attention profiler. This is a selected-query
# primitive probe, not a complete block or model benchmark.
require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/metal_attention_mps"
require "../src/ml/video/metal_attention"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private WAN_PROFILE_Q_ROWS          = 36_608
private WAN_PROFILE_K_ROWS          = 36_608
private WAN_PROFILE_HEADS           =     40
private WAN_PROFILE_HEAD_DIM        =    128
private WAN_PROFILE_CHANNELS        = WAN_PROFILE_HEADS * WAN_PROFILE_HEAD_DIM
private WAN_PROFILE_CHUNK_ROWS      = 128
private WAN_PROFILE_QUERY_ROWS      = {128, 256, 257, 512, 1_024, 4_096, 8_192}
private WAN_PROFILE_EXPLICIT_CAP    = 3_i64 * 1024 * 1024 * 1024
private WAN_PROFILE_MAX_ROOTS       = 4_096
private WAN_PROFILE_MANIFEST_SHA256 = "7586da395e4acedcd667639c91350119135183a0ed692fa994faeb683c6b850a"
private WAN_PROFILE_EXPECTED_SHA256 = "8bf37c04b5effec694ec888a0a3c052160e827fcb7e0fb3603f5bfd939a709ef"
private WAN_PROFILE_FORMULA         = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
private WAN_PROFILE_MANIFEST_SCHEMA = "wan_s2v_self_attention_reference_v1"

private WAN_PROFILE_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline float wan_profile_value(uint index, uint salt) {
    uint mixed = index + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int quantized = int(mixed & 1023u) - 512;
    return float(quantized) / 256.0f;
  }

  kernel void wan_profile_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &flat_base [[buffer(2)]], constant uint &salt [[buffer(3)]],
      uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = wan_profile_value(flat_base + index, salt);
  }

  kernel void wan_profile_nan_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = as_type<float>(0x7fc00000u);
  }

  kernel void wan_profile_zero(device atomic_uint *metrics [[buffer(0)]],
                               uint index [[thread_position_in_grid]]) {
    if (index < 5u)
      atomic_store_explicit(metrics + index, 0u, memory_order_relaxed);
  }

  // mode=1 maps the selected contiguous BHQD reference slice; mode=0 compares
  // two native BQHD buffers. Equal finite values take the zero-atomic path.
  kernel void wan_profile_compare(
      device const float *reference [[buffer(0)]],
      device const float *actual [[buffer(1)]],
      device atomic_uint *metrics [[buffer(2)]],
      constant uint &count [[buffer(3)]],
      constant uint &query_rows [[buffer(4)]],
      constant uint &heads [[buffer(5)]],
      constant uint &head_dim [[buffer(6)]],
      constant uint &reference_bhqd [[buffer(7)]],
      uint index [[thread_position_in_grid]]) {
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
    const float ref = reference[reference_index];
    const float got = actual[index];
    const uint ref_bits = as_type<uint>(ref), got_bits = as_type<uint>(got);
    const bool same_bits = ref_bits == got_bits;
    if (!isfinite(ref) || !isfinite(got)) {
      if (!same_bits)
        atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
      return;
    }
    if (same_bits) return;

    atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
    const float error = abs(ref - got);
    const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
    const float ratio = error / tolerance;
    if (error > tolerance)
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
    if (error > 0.0f) {
      atomic_fetch_max_explicit(metrics + 3, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 4, as_type<uint>(ratio), memory_order_relaxed);
    }
  }
METAL

# This probe-local extension reads only already-completed root summaries. It
# does not add boundaries, continuations, synchronization, or production API.
class ML::Metal::MPSCommandChain
  def wan_profile_root_timings_after_wait!(root_limit : Int32)
    unless completed? && wait_count == 1
      raise ArgumentError.new("Root timing is available only after one terminal wait")
    end
    count = segment_count
    raise ArgumentError.new("MPS root count is invalid") if count <= 0
    raise ArgumentError.new("MPS root count exceeds probe metadata limit") if count > root_limit

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
        raise ArgumentError.new("Failed to read MPS root timing summary #{sequence} (#{status})") unless status == 0
        unless completed == 1 && has_error == 0 &&
               gpu_start.finite? && gpu_end.finite? && gpu_start > 0.0 &&
               gpu_end > gpu_start
          raise ArgumentError.new("MPS root #{sequence} did not complete cleanly")
        end
        roots << {
          sequence:             sequence.to_i32,
          gpu_start_seconds:    gpu_start,
          gpu_end_seconds:      gpu_end,
          gpu_interval_seconds: gpu_end - gpu_start,
        }
      end
    {% else %}
      raise ArgumentError.new("MPS root timing requires macOS Metal")
    {% end %}
    roots
  end
end

private def wan_profile_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def wan_profile_verify_source!(source : JSON::Any,
                                       path_key : String,
                                       digest_key : String) : String
  path = source[path_key].as_s
  raise ArgumentError.new("Missing pinned reference source for #{path_key}") unless File.file?(path)
  digest = source[digest_key].as_s
  unless digest.matches?(/\A[0-9a-f]{64}\z/) && wan_profile_sha256_file(path) == digest
    raise ArgumentError.new("Reference source SHA-256 mismatch for #{path_key}")
  end
  digest
end

private def wan_profile_validate_fixture!(fixture_dir : String) : NamedTuple(
  manifest_path: String, expected_path: String, manifest_sha256: String,
  expected_sha256: String, torch_version: String, attention_source_sha256: String,
  adapter_sha256: String, generator_sha256: String)
  manifest_path = File.join(fixture_dir, "manifest.json")
  expected_path = File.join(fixture_dir, "expected_bhqd.f32")
  raise ArgumentError.new("Missing pinned attention fixture manifest") unless File.file?(manifest_path)
  raise ArgumentError.new("Missing pinned full-query expected output") unless File.file?(expected_path)

  manifest_sha256 = wan_profile_sha256_file(manifest_path)
  unless manifest_sha256 == WAN_PROFILE_MANIFEST_SHA256
    raise ArgumentError.new("Pinned attention manifest SHA-256 mismatch")
  end
  expected_bytes = WAN_PROFILE_Q_ROWS.to_i64 * WAN_PROFILE_CHANNELS * sizeof(Float32)
  raise ArgumentError.new("Full attention expected output byte size mismatch") unless File.size(expected_path) == expected_bytes
  expected_sha256 = wan_profile_sha256_file(expected_path)
  unless expected_sha256 == WAN_PROFILE_EXPECTED_SHA256
    raise ArgumentError.new("Pinned full attention expected SHA-256 mismatch")
  end

  # Verify both full-file pins before parsing or loading any expected values.
  manifest = JSON.parse(File.read(manifest_path))
  unless manifest["schema"].as_s == WAN_PROFILE_MANIFEST_SCHEMA
    raise ArgumentError.new("Unsupported attention fixture schema")
  end
  geometry = manifest["geometry"]
  {
    {"batches", 1_i64}, {"query_rows", WAN_PROFILE_Q_ROWS.to_i64},
    {"key_rows", WAN_PROFILE_K_ROWS.to_i64}, {"heads", WAN_PROFILE_HEADS.to_i64},
    {"head_dim", WAN_PROFILE_HEAD_DIM.to_i64},
  }.each do |key, expected|
    raise ArgumentError.new("Attention fixture geometry mismatch for #{key}") unless geometry[key].as_i64 == expected
  end
  unless geometry["native_layout"].as_s == "B,Q,H,D" &&
         geometry["reference_layout"].as_s == "B,H,Q,D"
    raise ArgumentError.new("Attention fixture layouts mismatch")
  end
  unless manifest["input_formula"].as_s == WAN_PROFILE_FORMULA
    raise ArgumentError.new("Attention fixture input formula mismatch")
  end
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
  attention_source_sha256 = wan_profile_verify_source!(source,
    "wan_attention_source_path", "wan_attention_source_sha256")
  adapter_sha256 = wan_profile_verify_source!(source,
    "wan_s2v_reference_adapter_path", "wan_s2v_reference_adapter_sha256")
  generator_sha256 = source["generator_sha256"].as_s
  generator_path = File.expand_path("support/wan_s2v_self_attention_benchmark_reference.py", __DIR__)
  unless generator_sha256.matches?(/\A[0-9a-f]{64}\z/) &&
         wan_profile_sha256_file(generator_path) == generator_sha256
    raise ArgumentError.new("Attention fixture generator SHA-256 mismatch")
  end

  expected = manifest["expected"]
  unless expected["file"].as_s == "expected_bhqd.f32" &&
         File.basename(expected["file"].as_s) == expected["file"].as_s &&
         expected["dtype"].as_s == "F32_LE" &&
         expected["shape"].as_a.map(&.as_i64) == [
           1_i64, WAN_PROFILE_HEADS.to_i64, WAN_PROFILE_Q_ROWS.to_i64,
           WAN_PROFILE_HEAD_DIM.to_i64,
         ] && expected["bytes"].as_i64 == expected_bytes &&
         expected["sha256"].as_s == WAN_PROFILE_EXPECTED_SHA256
    raise ArgumentError.new("Attention fixture expected-output metadata mismatch")
  end

  {
    manifest_path: manifest_path, expected_path: expected_path,
    manifest_sha256: manifest_sha256, expected_sha256: expected_sha256,
    torch_version: torch_version, attention_source_sha256: attention_source_sha256,
    adapter_sha256: adapter_sha256, generator_sha256: generator_sha256,
  }
end

private def wan_profile_load_expected_slice(path : String, destination : ML::MetalBuffer,
                                            query_start : Int32, query_rows : Int32) : Nil
  destination_pointer = destination.contents.as(UInt8*)
  bytes_per_head = query_rows.to_i64 * WAN_PROFILE_HEAD_DIM * sizeof(Float32)
  File.open(path, "rb") do |file|
    WAN_PROFILE_HEADS.times do |head|
      source_offset = (head.to_i64 * WAN_PROFILE_Q_ROWS + query_start) *
                      WAN_PROFILE_HEAD_DIM * sizeof(Float32)
      file.seek(source_offset, IO::Seek::Set)
      written = 0_i64
      while written < bytes_per_head
        amount = Math.min(1_i64 << 20, bytes_per_head - written).to_i32
        slice = Bytes.new(destination_pointer + head.to_i64 * bytes_per_head + written, amount)
        count = file.read(slice)
        raise ArgumentError.new("Truncated selected expected-output slice") if count == 0
        written += count
      end
    end
  end
end

private def wan_profile_plan(query_rows : Int32)
  query_values = query_rows.to_i64 * WAN_PROFILE_CHANNELS
  kv_values = WAN_PROFILE_K_ROWS.to_i64 * WAN_PROFILE_CHANNELS
  chunk_values = WAN_PROFILE_CHUNK_ROWS.to_i64 * WAN_PROFILE_CHANNELS
  # Q, expected BHQD, native output, full K/V, three chunk buffers, metrics,
  # and four scalar values for the +1/NaN comparator controls.
  explicit_bytes = (3_i64 * query_values + 2_i64 * kv_values +
                    3_i64 * chunk_values + 19_i64) * sizeof(Float32)
  {
    query_values:   query_values,
    kv_values:      kv_values,
    chunk_values:   chunk_values,
    explicit_bytes: explicit_bytes,
  }
end

private def wan_profile_encode_fill(encoder : ML::Metal::ComputeEncoder,
                                    pipeline : ML::Metal::ComputePipeline,
                                    buffer : ML::MetalBuffer, count : Int32,
                                    flat_base : UInt32, salt : UInt32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).set_value(flat_base, 2).set_value(salt, 3)
  encoder.dispatch_1d(count)
end

private def wan_profile_encode_nan_fill(encoder : ML::Metal::ComputeEncoder,
                                        pipeline : ML::Metal::ComputePipeline,
                                        buffer : ML::MetalBuffer, count : Int32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).dispatch_1d(count)
end

private def wan_profile_encode_compare(encoder : ML::Metal::ComputeEncoder,
                                       zero : ML::Metal::ComputePipeline,
                                       compare : ML::Metal::ComputePipeline,
                                       reference : ML::MetalBuffer,
                                       actual : ML::MetalBuffer,
                                       metrics : ML::MetalBuffer,
                                       count : Int32, query_rows : Int32,
                                       reference_bhqd : Bool) : Nil
  encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  encoder.set_pipeline(compare)
  encoder.set_buffer(reference, 0).set_buffer(actual, 1)
  encoder.set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
  encoder.set_value(count.to_u32, 3).set_value(query_rows.to_u32, 4)
  encoder.set_value(WAN_PROFILE_HEADS.to_u32, 5).set_value(WAN_PROFILE_HEAD_DIM.to_u32, 6)
  encoder.set_value(reference_bhqd ? 1_u32 : 0_u32, 7).dispatch_1d(count)
end

private def wan_profile_read_metrics(buffer : ML::MetalBuffer)
  values = buffer.contents.as(UInt32*)
  {
    bit_differences: values[0], tolerance_violations: values[1],
    nonfinite: values[2], max_abs_error: values[3].unsafe_as(Float32),
    max_tolerance_ratio: values[4].unsafe_as(Float32),
  }
end

private def wan_profile_run_arm(phase : String, sample_index : Int32,
                                query_rows : Int32, query_start : Int32,
                                q : ML::MetalBuffer, k : ML::MetalBuffer,
                                v : ML::MetalBuffer, output : ML::MetalBuffer,
                                chunk_q : ML::MetalBuffer,
                                chunk_output_a : ML::MetalBuffer,
                                plan : ML::Video::WanAttentionMPSPlan,
                                profile_host : Bool)
  total_started = Time.instant
  allocation_count = ML::MetalBuffer.video_spec_allocations
  chain = ML::Metal::MPSCommandChain.new
  chain_creation_wall_seconds = (Time.instant - total_started).total_seconds
  begin
    encode_started = Time.instant
    q_copy_total = 0.0
    q_tail_zero_total = 0.0
    plan_encode_total = 0.0
    output_copy_total = 0.0
    chunk_records = [] of NamedTuple(offset_rows: Int32, valid_rows: Int32,
      q_copy_host_seconds: Float64?, q_tail_zero_host_seconds: Float64?,
      plan_encode_host_seconds: Float64?,
      output_copy_host_seconds: Float64?)
    channels = WAN_PROFILE_CHANNELS.to_i64
    query_offset = 0
    while query_offset < query_rows
      valid_rows = Math.min(WAN_PROFILE_CHUNK_ROWS, query_rows - query_offset)
      valid_elements = valid_rows.to_i64 * channels
      source_offset_bytes = query_offset.to_i64 * channels * sizeof(Float32)
      destination_offset_bytes = query_offset.to_i64 * channels * sizeof(Float32)
      valid_bytes = valid_elements * sizeof(Float32)
      q_copy_seconds = nil.as(Float64?)
      q_tail_zero_seconds = nil.as(Float64?)
      plan_encode_seconds = nil.as(Float64?)
      output_copy_seconds = nil.as(Float64?)

      if profile_host
        operation_started = Time.instant
        ML::Video::WanAttentionMPSPlan.copy(chain, q, chunk_q, valid_bytes,
          source_offset_bytes, 0_i64)
        q_copy_seconds = (Time.instant - operation_started).total_seconds
        q_copy_total += q_copy_seconds.not_nil!
      else
        ML::Video::WanAttentionMPSPlan.copy(chain, q, chunk_q, valid_bytes,
          source_offset_bytes, 0_i64)
      end

      if valid_rows < WAN_PROFILE_CHUNK_ROWS
        chunk_elements = WAN_PROFILE_CHUNK_ROWS.to_i64 * channels
        tail_elements = (chunk_elements - valid_elements).to_i32
        if profile_host
          operation_started = Time.instant
          ML::Video::MetalAttentionMPS.zero_q_tail!(chain, chunk_q,
            valid_elements.to_i32, tail_elements)
          q_tail_zero_seconds = (Time.instant - operation_started).total_seconds
          q_tail_zero_total += q_tail_zero_seconds.not_nil!
        else
          ML::Video::MetalAttentionMPS.zero_q_tail!(chain, chunk_q,
            valid_elements.to_i32, tail_elements)
        end
      end

      if profile_host
        operation_started = Time.instant
        plan.encode(chain, 0)
        plan_encode_seconds = (Time.instant - operation_started).total_seconds
        plan_encode_total += plan_encode_seconds.not_nil!
      else
        plan.encode(chain, 0)
      end

      if profile_host
        operation_started = Time.instant
        ML::Video::WanAttentionMPSPlan.copy(chain, chunk_output_a, output, valid_bytes,
          0_i64, destination_offset_bytes)
        output_copy_seconds = (Time.instant - operation_started).total_seconds
        output_copy_total += output_copy_seconds.not_nil!
      else
        ML::Video::WanAttentionMPSPlan.copy(chain, chunk_output_a, output, valid_bytes,
          0_i64, destination_offset_bytes)
      end

      chunk_records << {
        offset_rows:              query_offset,
        valid_rows:               valid_rows,
        q_copy_host_seconds:      q_copy_seconds,
        q_tail_zero_host_seconds: q_tail_zero_seconds,
        plan_encode_host_seconds: plan_encode_seconds,
        output_copy_host_seconds: output_copy_seconds,
      }
      query_offset += valid_rows
    end
    encode_wall_seconds = (Time.instant - encode_started).total_seconds

    wait_started = Time.instant
    chain.commit_and_wait
    wait_wall_seconds = (Time.instant - wait_started).total_seconds
    completed_wall_seconds = (Time.instant - total_started).total_seconds
    allocations_during_arm = ML::MetalBuffer.video_spec_allocations - allocation_count
    raise "MPS attention arm allocated a tracked MetalBuffer" unless allocations_during_arm == 0
    raise "MPS attention arm did not use one terminal wait" unless chain.wait_count == 1
    raise "MPS attention arm produced no command roots" unless chain.segment_count > 0
    roots = chain.wan_profile_root_timings_after_wait!(WAN_PROFILE_MAX_ROOTS)
    roots_by_start = roots.sort_by { |root| root[:gpu_start_seconds] }
    root_start_seconds = roots_by_start.first[:gpu_start_seconds]
    root_end_seconds = roots_by_start.max_of { |root| root[:gpu_end_seconds] }
    root_envelope_seconds = root_end_seconds - root_start_seconds
    root_interval_sum_seconds = roots.sum { |root| root[:gpu_interval_seconds] }
    root_uncovered_seconds = 0.0
    root_gaps = [] of NamedTuple(after_sequence: Int32, before_sequence: Int32,
      seconds: Float64)
    root_overlaps = [] of NamedTuple(prior_union_end_sequence: Int32,
      sequence: Int32, seconds: Float64)
    previous_end = roots_by_start.first[:gpu_end_seconds]
    previous_sequence = roots_by_start.first[:sequence]
    roots_by_start.skip(1).each do |root|
      if root[:gpu_start_seconds] > previous_end
        gap = root[:gpu_start_seconds] - previous_end
        root_uncovered_seconds += gap
        root_gaps << {after_sequence: previous_sequence, before_sequence: root[:sequence], seconds: gap}
      elsif root[:gpu_start_seconds] < previous_end
        # Sorted intervals accumulate a union. Count only the part of this
        # root already covered by that union, never pairwise intersections.
        overlap = Math.min(root[:gpu_end_seconds], previous_end) - root[:gpu_start_seconds]
        if overlap > 0.0
          root_overlaps << {
            prior_union_end_sequence: previous_sequence,
            sequence:                 root[:sequence],
            seconds:                  overlap,
          }
        end
      end
      if root[:gpu_end_seconds] > previous_end
        previous_end = root[:gpu_end_seconds]
        previous_sequence = root[:sequence]
      end
    end
    root_union_seconds = root_envelope_seconds - root_uncovered_seconds
    root_overlap_seconds = root_interval_sum_seconds - root_union_seconds
    root_overlap_seconds = 0.0 if root_overlap_seconds < 0.0

    result = {
      schema:                                   "wan_s2v_self_attention_profile_arm_v1",
      domain:                                   "wan.s2v.self_attention",
      phase:                                    phase,
      sample_index:                             sample_index,
      backend:                                  "mps_chunk128_bthd",
      query_start:                              query_start,
      query_rows:                               query_rows,
      key_rows:                                 WAN_PROFILE_K_ROWS,
      heads:                                    WAN_PROFILE_HEADS,
      head_dim:                                 WAN_PROFILE_HEAD_DIM,
      host_profile_enabled:                     profile_host,
      completed_wall_seconds:                   completed_wall_seconds,
      completed_wall_boundary:                  "Chain creation through encode calls and commit_and_wait completion",
      chain_creation_wall_seconds:              chain_creation_wall_seconds,
      encode_wall_seconds:                      encode_wall_seconds,
      terminal_wait_wall_seconds:               wait_wall_seconds,
      terminal_wait_wall_boundary:              "Wall time around commit_and_wait, including encoder pause and command completion callback/fallback processing",
      q_copy_host_seconds:                      profile_host ? q_copy_total : nil,
      q_tail_zero_host_seconds:                 profile_host ? q_tail_zero_total : nil,
      mps_plan_encode_host_seconds:             profile_host ? plan_encode_total : nil,
      output_copy_host_seconds:                 profile_host ? output_copy_total : nil,
      chunk_host_spans:                         chunk_records,
      summed_command_root_gpu_interval_seconds: chain.gpu_elapsed_seconds,
      root_gpu_envelope_seconds:                root_envelope_seconds,
      root_interval_sum_gpu_seconds:            root_interval_sum_seconds,
      root_gpu_uncovered_envelope_seconds:      root_uncovered_seconds,
      root_gpu_overlap_seconds:                 root_overlap_seconds,
      root_gpu_overlap_semantics:               "sum of root durations minus interval union; per-root entries are covered duration from prior union, not pairwise intersections",
      root_gpu_gaps:                            root_gaps,
      root_gpu_overlaps:                        root_overlaps,
      segment_count:                            chain.segment_count,
      terminal_waits:                           chain.wait_count,
      metal_buffer_allocations_during_arm:      allocations_during_arm,
      root_metadata_limit:                      WAN_PROFILE_MAX_ROOTS,
      root_spans:                               roots,
      kernel_time_boundary:                     "GPU root intervals cover command-buffer work and scheduling gaps; kernel-only timestamps are unavailable",
      validation_follows_on_separate_chain:     true,
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

private def wan_profile_validate(output : ML::MetalBuffer,
                                 expected : ML::MetalBuffer,
                                 metrics : ML::MetalBuffer,
                                 zero : ML::Metal::ComputePipeline,
                                 compare : ML::Metal::ComputePipeline,
                                 query_rows : Int32, label : String,
                                 prefill_next : Bool,
                                 nan_fill : ML::Metal::ComputePipeline,
                                 controls : Tuple(ML::MetalBuffer, ML::MetalBuffer,
                                                  ML::MetalBuffer, ML::MetalBuffer,
                                                  ML::MetalBuffer, ML::MetalBuffer)) : Nil
  control_reference_plus, control_actual_plus, metrics_plus, control_reference_nan, control_actual_nan, metrics_nan = controls
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    wan_profile_encode_compare(encoder, zero, compare, expected, output,
      metrics, query_rows * WAN_PROFILE_CHANNELS, query_rows, true)
    wan_profile_encode_compare(encoder, zero, compare, control_reference_plus,
      control_actual_plus, metrics_plus, 1, 1, false)
    wan_profile_encode_compare(encoder, zero, compare, control_reference_nan,
      control_actual_nan, metrics_nan, 1, 1, false)
    if prefill_next
      wan_profile_encode_nan_fill(encoder, nan_fill, output,
        query_rows * WAN_PROFILE_CHANNELS)
    end
    chain.commit_and_wait
    raise "attention diagnostics did not use one terminal wait" unless chain.wait_count == 1
    actual_result = wan_profile_read_metrics(metrics)
    plus_result = wan_profile_read_metrics(metrics_plus)
    nan_result = wan_profile_read_metrics(metrics_nan)
    actual_passed = actual_result[:tolerance_violations] == 0 && actual_result[:nonfinite] == 0
    plus_rejected = plus_result[:tolerance_violations] > 0 && plus_result[:nonfinite] == 0
    nan_rejected = nan_result[:nonfinite] > 0

    STDOUT.puts({
      schema:                      "wan_s2v_self_attention_profile_validation_v1",
      domain:                      "wan.s2v.self_attention",
      label:                       label,
      compared_values:             query_rows.to_i64 * WAN_PROFILE_CHANNELS,
      bit_differences:             actual_result[:bit_differences],
      strict_tolerance_violations: actual_result[:tolerance_violations],
      nonfinite:                   actual_result[:nonfinite],
      max_abs_error:               actual_result[:max_abs_error],
      max_tolerance_ratio:         actual_result[:max_tolerance_ratio],
      tolerance:                   "2e-5+2e-5*abs(reference)",
      positive_control_violations: plus_result[:tolerance_violations],
      positive_control_rejected:   plus_rejected,
      nan_control_nonfinite:       nan_result[:nonfinite],
      nan_control_rejected:        nan_rejected,
      prefilled_for_next_arm:      prefill_next,
      terminal_waits:              chain.wait_count,
      passed:                      actual_passed && plus_rejected && nan_rejected,
    }.to_json)
    STDOUT.flush
    chain.release
    raise "strict attention profile validation failed" unless actual_passed && plus_rejected && nan_rejected
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def wan_profile_setup(q : ML::MetalBuffer, k : ML::MetalBuffer,
                              v : ML::MetalBuffer, output : ML::MetalBuffer,
                              fill : ML::Metal::ComputePipeline,
                              nan_fill : ML::Metal::ComputePipeline,
                              query_rows : Int32, query_start : Int32,
                              query_values : Int64, kv_values : Int64) : Float64
  started = Time.instant
  chain = ML::Metal::MPSCommandChain.new
  begin
    encoder = chain.encoder
    wan_profile_encode_fill(encoder, fill, q, query_values.to_i32,
      (query_start.to_i64 * WAN_PROFILE_CHANNELS).to_u32, 1_u32)
    wan_profile_encode_fill(encoder, fill, k, kv_values.to_i32, 0_u32, 2_u32)
    wan_profile_encode_fill(encoder, fill, v, kv_values.to_i32, 0_u32, 3_u32)
    wan_profile_encode_nan_fill(encoder, nan_fill, output,
      query_rows * WAN_PROFILE_CHANNELS)
    chain.pause!
    chain.commit_and_wait
    raise "input setup did not use one terminal wait" unless chain.wait_count == 1
    seconds = (Time.instant - started).total_seconds
    chain.release
    seconds
  ensure
    begin
      chain.commit_and_wait unless chain.completed?
    ensure
      chain.release if chain.completed?
    end
  end
end

private def wan_profile_wait_for_trace!(release_file : String) : Nil
  raise ArgumentError.new("Trace release file already exists; remove stale gate first") if File.exists?(release_file)
  STDOUT.puts({
    schema:                       "wan_s2v_self_attention_profile_trace_gate_v1",
    event:                        "awaiting_trace_release",
    process_id:                   Process.pid,
    release_file:                 release_file,
    timeout_seconds:              30,
    gate_is_outside_measured_arm: true,
  }.to_json)
  STDOUT.flush
  started = Time.instant
  until File.exists?(release_file)
    if (Time.instant - started).total_seconds >= 30.0
      raise "Timed out waiting for trace release file"
    end
    sleep 0.025
  end
  STDOUT.puts({
    schema:     "wan_s2v_self_attention_profile_trace_gate_v1",
    event:      "trace_release_received",
    process_id: Process.pid,
  }.to_json)
  STDOUT.flush
end

fixture_dir = ""
query_rows = 1_024
query_start = 0
query_start_set = false
tail = false
samples = 3
profile_host = true
plan_only = false
trace_release_file = ""
OptionParser.parse do |parser|
  parser.banner = "Usage: wan_s2v_self_attention_profile --fixture=DIR [--query-rows=N] [--query-start=N|--tail] [--samples=0|1|3]"
  parser.on("--fixture=DIR", "Pinned full-query upstream attention fixture directory") { |value| fixture_dir = value }
  parser.on("--query-rows=N", "Selected rows: 128, 256, 257, 512, 1024, 4096, or 8192 (default 1024)") { |value| query_rows = value.to_i }
  parser.on("--query-start=N", "First full-sequence query row (default 0)") do |value|
    query_start = value.to_i
    query_start_set = true
  end
  parser.on("--tail", "Select the final query rows in the full sequence") { tail = true }
  parser.on("--samples=N", "Measured arms after one warmup: 0, 1, or 3 (default 3)") { |value| samples = value.to_i }
  parser.on("--profile-host=BOOL", "Time each copy and plan.encode call (default true)") do |value|
    unless {"true", "false"}.includes?(value.downcase)
      raise ArgumentError.new("--profile-host must be true or false")
    end
    profile_host = value.downcase == "true"
  end
  parser.on("--trace-release-file=PATH", "Wait up to 30 seconds for an external trace gate before measured arms") { |value| trace_release_file = value }
  parser.on("--plan-only", "Print the CPU-only buffer and scope plan without Metal or fixture access") { plan_only = true }
end

unless WAN_PROFILE_QUERY_ROWS.includes?(query_rows)
  raise ArgumentError.new("--query-rows must be 128, 256, 257, 512, 1024, 4096, or 8192")
end
raise ArgumentError.new("--samples must be 0, 1, or 3") unless {0, 1, 3}.includes?(samples)
raise ArgumentError.new("--tail cannot be combined with --query-start") if tail && query_start_set
query_start = WAN_PROFILE_Q_ROWS - query_rows if tail
unless query_start >= 0 && query_start.to_i64 + query_rows <= WAN_PROFILE_Q_ROWS
  raise ArgumentError.new("--query-start and --query-rows exceed full query extent")
end
raise ArgumentError.new("--trace-release-file requires at least one measured sample") if !trace_release_file.empty? && samples == 0

buffer_plan = wan_profile_plan(query_rows)
if buffer_plan[:explicit_bytes] > WAN_PROFILE_EXPLICIT_CAP
  raise ArgumentError.new("Explicit MetalBuffer plan exceeds 3 GiB")
end
if plan_only
  STDOUT.puts({
    schema:                         "wan_s2v_self_attention_profile_plan_v1",
    domain:                         "wan.s2v.self_attention",
    backend:                        "mps_chunk128_bthd",
    query_start:                    query_start,
    query_rows:                     query_rows,
    key_rows:                       WAN_PROFILE_K_ROWS,
    heads:                          WAN_PROFILE_HEADS,
    head_dim:                       WAN_PROFILE_HEAD_DIM,
    query_values:                   buffer_plan[:query_values],
    kv_values_per_tensor:           buffer_plan[:kv_values],
    chunk_values_per_tensor:        buffer_plan[:chunk_values],
    explicit_buffer_bytes:          buffer_plan[:explicit_bytes],
    explicit_buffer_cap_bytes:      WAN_PROFILE_EXPLICIT_CAP,
    explicit_buffer_cap_passed:     true,
    mps_private_workspace_included: false,
    mps_private_workspace_note:     "opaque MPSGraph storage is outside explicit MetalBuffer accounting",
    allowed_query_rows:             WAN_PROFILE_QUERY_ROWS,
    chunk_rows:                     WAN_PROFILE_CHUNK_ROWS,
    measured_samples:               samples,
    host_call_profile:              profile_host,
    fixture_required_for_plan_only: false,
    scope:                          "selected-query full-K self-attention primitive; no complete block, model, or video-throughput claim",
  }.to_json)
  exit 0
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
fixture_dir = File.expand_path(fixture_dir)
fixture = wan_profile_validate_fixture!(fixture_dir)

STDOUT.puts({
  schema:                           "wan_s2v_self_attention_profile_setup_v1",
  domain:                           "wan.s2v.self_attention",
  backend:                          "mps_chunk128_bthd",
  device_name:                      ML::Metal::Device.instance.name,
  query_start:                      query_start,
  query_rows:                       query_rows,
  key_rows:                         WAN_PROFILE_K_ROWS,
  heads:                            WAN_PROFILE_HEADS,
  head_dim:                         WAN_PROFILE_HEAD_DIM,
  channels:                         WAN_PROFILE_CHANNELS,
  chunk_rows:                       WAN_PROFILE_CHUNK_ROWS,
  native_input_formula:             WAN_PROFILE_FORMULA,
  q_salt:                           1,
  k_salt:                           2,
  v_salt:                           3,
  fixture_manifest_path:            fixture[:manifest_path],
  fixture_manifest_sha256:          fixture[:manifest_sha256],
  expected_full_file_sha256:        fixture[:expected_sha256],
  expected_slice_layout:            "B,H,Q,D selected per head; native output B,Q,H,D",
  reference_torch_version:          fixture[:torch_version],
  wan_attention_source_sha256:      fixture[:attention_source_sha256],
  wan_s2v_reference_adapter_sha256: fixture[:adapter_sha256],
  generator_sha256:                 fixture[:generator_sha256],
  explicit_buffer_bytes:            buffer_plan[:explicit_bytes],
  explicit_buffer_cap_bytes:        WAN_PROFILE_EXPLICIT_CAP,
  mps_private_workspace_bytes:      "opaque and not included in explicit MetalBuffer accounting",
  samples:                          samples,
  host_call_profile:                profile_host,
  output_nan_prefill:               "GPU dispatch during setup and after diagnostics before the next arm",
  operator_root_waits:              "one terminal wait per setup, arm, and diagnostic chain",
  trace_release_file:               trace_release_file.empty? ? nil : trace_release_file,
  model_weights_loaded:             false,
  full_kv_used:                     true,
  scope:                            "selected-query attention primitive only; no complete block/model/video throughput inference",
}.to_json)
STDOUT.flush

arena = ML::Video::Arena.new
plan = nil.as(ML::Video::WanAttentionMPSPlan?)
begin
  query_values = buffer_plan[:query_values]
  kv_values = buffer_plan[:kv_values]
  chunk_values = buffer_plan[:chunk_values]
  q = arena.alloc("profile_query_input", query_values.to_i32)
  k = arena.alloc("profile_full_key_input", kv_values.to_i32)
  v = arena.alloc("profile_full_value_input", kv_values.to_i32)
  expected = arena.alloc("profile_upstream_expected_bhqd", query_values.to_i32)
  output = arena.alloc("profile_native_output_bqhd", query_values.to_i32)
  chunk_q = arena.alloc("profile_mps_chunk_query", chunk_values.to_i32)
  chunk_output_a = arena.alloc("profile_mps_chunk_output_a", chunk_values.to_i32)
  chunk_output_b = arena.alloc("profile_mps_chunk_output_b", chunk_values.to_i32)
  metrics = arena.alloc("profile_metrics_upstream", 5)
  control_reference_plus = arena.alloc("profile_control_plus_reference", 1)
  control_actual_plus = arena.alloc("profile_control_plus_actual", 1)
  metrics_plus = arena.alloc("profile_control_plus_metrics", 5)
  control_reference_nan = arena.alloc("profile_control_nan_reference", 1)
  control_actual_nan = arena.alloc("profile_control_nan_actual", 1)
  metrics_nan = arena.alloc("profile_control_nan_metrics", 5)
  controls = {control_reference_plus, control_actual_plus, metrics_plus,
              control_reference_nan, control_actual_nan, metrics_nan}
  unless arena.resident_bytes == buffer_plan[:explicit_bytes]
    raise ArgumentError.new("Actual explicit buffer bytes differ from the CPU plan")
  end

  expected_pointer = expected.contents.as(UInt8*)
  control_reference_plus.contents.as(Float32*)[0] = 0.0f32
  control_actual_plus.contents.as(Float32*)[0] = 1.0f32
  control_reference_nan.contents.as(Float32*)[0] = 0.0f32
  control_actual_nan.contents.as(Float32*)[0] = Float32::NAN

  fill_pipeline = ML::Metal::ComputePipeline.new("wan_profile_fill", WAN_PROFILE_PROBE_SOURCE)
  nan_fill_pipeline = ML::Metal::ComputePipeline.new("wan_profile_nan_fill", WAN_PROFILE_PROBE_SOURCE)
  zero_pipeline = ML::Metal::ComputePipeline.new("wan_profile_zero", WAN_PROFILE_PROBE_SOURCE)
  compare_pipeline = ML::Metal::ComputePipeline.new("wan_profile_compare", WAN_PROFILE_PROBE_SOURCE)
  ML::Video::MetalAttentionMPS.prepare_chunk!
  plan = ML::Video::WanAttentionMPSPlan.new(1, WAN_PROFILE_CHUNK_ROWS,
    WAN_PROFILE_K_ROWS, WAN_PROFILE_HEADS, WAN_PROFILE_HEAD_DIM,
    chunk_q, k, v, chunk_output_a, chunk_output_b)
  wan_profile_load_expected_slice(fixture[:expected_path], expected, query_start, query_rows)

  setup_seconds = wan_profile_setup(q, k, v, output, fill_pipeline,
    nan_fill_pipeline, query_rows, query_start, query_values, kv_values)
  STDOUT.puts({
    schema:                "wan_s2v_self_attention_profile_input_setup_v1",
    query_start:           query_start,
    query_rows:            query_rows,
    key_rows:              WAN_PROFILE_K_ROWS,
    host_wall_seconds:     setup_seconds,
    terminal_waits:        1,
    explicit_buffer_bytes: arena.resident_bytes,
    qkv_fill_gpu:          true,
    expected_slice_loaded: true,
    output_prefilled_nan:  true,
  }.to_json)
  STDOUT.flush

  warm = wan_profile_run_arm("warmup", 0, query_rows, query_start,
    q, k, v, output, chunk_q, chunk_output_a, plan.not_nil!, profile_host)
  STDOUT.puts(warm.to_json)
  STDOUT.flush
  wan_profile_validate(output, expected, metrics, zero_pipeline, compare_pipeline,
    query_rows, "warmup", samples > 0, nan_fill_pipeline, controls)

  unless trace_release_file.empty?
    wan_profile_wait_for_trace!(File.expand_path(trace_release_file))
  end

  measured_wall = [] of Float64
  samples.times do |sample_index|
    arm = wan_profile_run_arm("measured", sample_index + 1, query_rows,
      query_start, q, k, v, output, chunk_q, chunk_output_a, plan.not_nil!, profile_host)
    measured_wall << arm[:completed_wall_seconds]
    STDOUT.puts(arm.to_json)
    STDOUT.flush
    wan_profile_validate(output, expected, metrics, zero_pipeline, compare_pipeline,
      query_rows, "measured_#{sample_index + 1}", sample_index + 1 < samples,
      nan_fill_pipeline, controls)
  end

  sorted_wall = measured_wall.sort
  median_wall = sorted_wall.empty? ? nil : sorted_wall[sorted_wall.size // 2]
  STDOUT.puts({
    schema:                          "wan_s2v_self_attention_profile_summary_v1",
    domain:                          "wan.s2v.self_attention",
    backend:                         "mps_chunk128_bthd",
    query_start:                     query_start,
    query_rows:                      query_rows,
    measured_samples:                measured_wall.size,
    measured_completed_wall_seconds: measured_wall,
    median_completed_wall_seconds:   median_wall,
    host_call_profile:               profile_host,
    full_kv_rows:                    WAN_PROFILE_K_ROWS,
    model_weights_loaded:            false,
    scope:                           "MPS chunk128 selected-query self-attention only; host spans are encode-call attribution and root GPU intervals are not kernel-only time",
  }.to_json)
  STDOUT.flush
ensure
  begin
    if attention_plan = plan
      attention_plan.release
    end
  ensure
    arena.release
  end
end
