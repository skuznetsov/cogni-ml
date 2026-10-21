# Bounded full-K Wan self-attention discriminator. This is a selected-query
# primitive probe, not a complete block or model benchmark.
require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/metal_attention_mps"
require "../src/ml/video/metal_attention"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private WAN_BOUNDED_Q_ROWS             = 36_608
private WAN_BOUNDED_K_ROWS             = 36_608
private WAN_BOUNDED_HEADS              =     40
private WAN_BOUNDED_HEAD_DIM           =    128
private WAN_BOUNDED_CHANNELS           = WAN_BOUNDED_HEADS * WAN_BOUNDED_HEAD_DIM
private WAN_BOUNDED_CHUNK_ROWS         =   128
private WAN_BOUNDED_GROUPS_PER_SEGMENT =   320
private WAN_BOUNDED_QUERY_LIMIT        = 1_024
private WAN_BOUNDED_EXPLICIT_CAP       = 2_i64 * 1024 * 1024 * 1024
private WAN_BOUNDED_MANIFEST_SHA256    = "7586da395e4acedcd667639c91350119135183a0ed692fa994faeb683c6b850a"
private WAN_BOUNDED_EXPECTED_SHA256    = "8bf37c04b5effec694ec888a0a3c052160e827fcb7e0fb3603f5bfd939a709ef"
private WAN_BOUNDED_FORMULA            = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
private WAN_BOUNDED_MANIFEST_SCHEMA    = "wan_s2v_self_attention_reference_v1"
private WAN_BOUNDED_SHARED_KV_SOURCE   = {{ read_file("#{__DIR__}/support/wan_s2v_shared_kv_attention.metal") }}
private WAN_BOUNDED_SHARED_KV_BYTES    = 2_i64 * 32 * WAN_BOUNDED_HEAD_DIM * sizeof(Float32)
private WAN_BOUNDED_REGISTER_KV_SOURCE = {{ read_file("#{__DIR__}/support/wan_s2v_register_kv_attention.metal") }}
private WAN_BOUNDED_REGISTER_Q_TILES   = 2
private WAN_BOUNDED_PREFETCH_KV_SOURCE = {{ read_file("#{__DIR__}/support/wan_s2v_prefetch_kv_attention.metal") }}

lib WanBoundedAttentionLimits
  fun max_threadgroup_memory_length = wan_bounded_max_threadgroup_memory_length(handle : Pointer(Void)) : Int64
end

private WAN_BOUNDED_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline float wan_bounded_value(uint index, uint salt) {
    uint mixed = index + salt * 0x9e3779b9u;
    mixed ^= mixed >> 16;
    mixed *= 0x7feb352du;
    mixed ^= mixed >> 15;
    mixed *= 0x846ca68bu;
    mixed ^= mixed >> 16;
    const int quantized = int(mixed & 1023u) - 512;
    return float(quantized) / 256.0f;
  }

  kernel void wan_bounded_attention_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      constant uint &flat_base [[buffer(2)]], constant uint &salt [[buffer(3)]],
      uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = wan_bounded_value(flat_base + index, salt);
  }

  kernel void wan_bounded_attention_nan_fill(
      device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
      uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = as_type<float>(0x7fc00000u);
  }

  kernel void wan_bounded_attention_zero(device atomic_uint *metrics [[buffer(0)]],
                                         uint index [[thread_position_in_grid]]) {
    if (index < 5u)
      atomic_store_explicit(metrics + index, 0u, memory_order_relaxed);
  }

  // mode=1 maps the selected contiguous per-head BHQD slice; mode=0 compares
  // two native BQHD buffers. Equal finite values take the zero-atomic fast path.
  kernel void wan_bounded_attention_compare(
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

  kernel void wan_bounded_attention_corrupt_one(
      device float *output [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0u) output[0] += 1.0f;
  }

  // The existing single-wait negative-control chain checks that non-finite
  // candidate output is rejected by the strict upstream comparator.
  kernel void wan_bounded_attention_nan_one(
      device float *output [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0u) output[1] = as_type<float>(0x7fc00000u);
  }
METAL

private def wan_bounded_emit(json : String) : Nil
  STDOUT.puts(json)
  STDOUT.flush
end

private def wan_bounded_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def wan_bounded_verify_source!(source : JSON::Any,
                                       path_key : String,
                                       digest_key : String) : String
  path = source[path_key].as_s
  raise ArgumentError.new("Missing reference source for #{path_key}: #{path}") unless File.file?(path)
  digest = source[digest_key].as_s
  unless digest.matches?(/\A[0-9a-f]{64}\z/) && wan_bounded_sha256_file(path) == digest
    raise ArgumentError.new("Reference source SHA-256 mismatch: #{path_key}")
  end
  digest
end

private def wan_bounded_validate_fixture!(fixture_dir : String) : NamedTuple(
  manifest_path: String, expected_path: String, manifest_sha256: String,
  expected_sha256: String, torch_version: String, attention_source_sha256: String,
  adapter_sha256: String, generator_sha256: String)
  manifest_path = File.join(fixture_dir, "manifest.json")
  expected_path = File.join(fixture_dir, "expected_bhqd.f32")
  raise ArgumentError.new("Missing pinned attention fixture manifest") unless File.file?(manifest_path)
  raise ArgumentError.new("Missing pinned full-query expected output") unless File.file?(expected_path)

  manifest_sha256 = wan_bounded_sha256_file(manifest_path)
  raise ArgumentError.new("Pinned attention manifest SHA-256 mismatch") unless manifest_sha256 == WAN_BOUNDED_MANIFEST_SHA256
  expected_bytes = WAN_BOUNDED_Q_ROWS.to_i64 * WAN_BOUNDED_CHANNELS * sizeof(Float32)
  raise ArgumentError.new("Full attention expected output byte size mismatch") unless File.size(expected_path) == expected_bytes
  expected_sha256 = wan_bounded_sha256_file(expected_path)
  raise ArgumentError.new("Pinned full attention expected SHA-256 mismatch") unless expected_sha256 == WAN_BOUNDED_EXPECTED_SHA256

  # Both pinned full-file hashes are checked before parsing metadata or reading
  # any output bytes. The fixture is assumed immutable for the rest of this run.
  manifest = JSON.parse(File.read(manifest_path))
  raise ArgumentError.new("Unsupported attention fixture schema") unless manifest["schema"].as_s == WAN_BOUNDED_MANIFEST_SCHEMA
  geometry = manifest["geometry"]
  {
    {"batches", 1_i64}, {"query_rows", WAN_BOUNDED_Q_ROWS.to_i64},
    {"key_rows", WAN_BOUNDED_K_ROWS.to_i64}, {"heads", WAN_BOUNDED_HEADS.to_i64},
    {"head_dim", WAN_BOUNDED_HEAD_DIM.to_i64},
  }.each do |key, expected|
    raise ArgumentError.new("Attention fixture geometry mismatch for #{key}") unless geometry[key].as_i64 == expected
  end
  unless geometry["native_layout"].as_s == "B,Q,H,D" && geometry["reference_layout"].as_s == "B,H,Q,D"
    raise ArgumentError.new("Attention fixture layouts mismatch")
  end
  unless manifest["input_formula"].as_s == WAN_BOUNDED_FORMULA
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
  attention_source_sha256 = wan_bounded_verify_source!(source,
    "wan_attention_source_path", "wan_attention_source_sha256")
  adapter_sha256 = wan_bounded_verify_source!(source,
    "wan_s2v_reference_adapter_path", "wan_s2v_reference_adapter_sha256")
  generator_sha256 = source["generator_sha256"].as_s
  generator_path = File.expand_path("support/wan_s2v_self_attention_benchmark_reference.py", __DIR__)
  unless generator_sha256.matches?(/\A[0-9a-f]{64}\z/) &&
         wan_bounded_sha256_file(generator_path) == generator_sha256
    raise ArgumentError.new("Attention fixture generator SHA-256 mismatch")
  end

  expected = manifest["expected"]
  unless expected["file"].as_s == "expected_bhqd.f32" &&
         File.basename(expected["file"].as_s) == expected["file"].as_s &&
         expected["dtype"].as_s == "F32_LE" &&
         expected["shape"].as_a.map(&.as_i64) == [
           1_i64, WAN_BOUNDED_HEADS.to_i64, WAN_BOUNDED_Q_ROWS.to_i64,
           WAN_BOUNDED_HEAD_DIM.to_i64,
         ] && expected["bytes"].as_i64 == expected_bytes &&
         expected["sha256"].as_s == WAN_BOUNDED_EXPECTED_SHA256
    raise ArgumentError.new("Attention fixture expected-output metadata mismatch")
  end

  {
    manifest_path: manifest_path, expected_path: expected_path,
    manifest_sha256: manifest_sha256, expected_sha256: expected_sha256,
    torch_version: torch_version, attention_source_sha256: attention_source_sha256,
    adapter_sha256: adapter_sha256, generator_sha256: generator_sha256,
  }
end

private def wan_bounded_load_expected_slice(path : String, destination : ML::MetalBuffer,
                                            query_start : Int32, query_rows : Int32) : Nil
  destination_pointer = destination.contents.as(UInt8*)
  bytes_per_head = query_rows.to_i64 * WAN_BOUNDED_HEAD_DIM * sizeof(Float32)
  File.open(path, "rb") do |file|
    WAN_BOUNDED_HEADS.times do |head|
      source_offset = (head.to_i64 * WAN_BOUNDED_Q_ROWS + query_start) *
                      WAN_BOUNDED_HEAD_DIM * sizeof(Float32)
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

private def wan_bounded_qcache_source : String
  source = ML::Video::MetalOps::TILED_ATTENTION_SOURCE
  replacements = {
    {
      "  const uint qr = qbase + fm;\n",
      "  const uint qr = qbase + fm;\n" +
      "  float2 qfragments[D / 8];\n" +
      "  for (uint dd = 0; dd < D / 8; ++dd) {\n" +
      "    const uint qc = dd * 8 + fn;\n" +
      "    qfragments[dd] = float2(\n" +
      "      qr < nq ? q[qb + ulong(qr) * channels + head * D + qc] : 0.f,\n" +
      "      qr < nq ? q[qb + ulong(qr) * channels + head * D + qc + 1] : 0.f);\n" +
      "  }\n",
    },
    {
      "        const uint qc = dd * 8 + fn + j;\n",
      "",
    },
    {
      "          aq.thread_elements()[j] = qr < nq ? q[qb + ulong(qr) * channels + head * D + qc] : 0.f;\n",
      "          aq.thread_elements()[j] = qfragments[dd][j];\n",
    },
    {
      "kernel void video_tiled_attention_##D",
      "kernel void wan_bounded_qcache_attention_##D",
    },
  }
  replacements.each do |needle, replacement|
    count = source.split(needle).size - 1
    raise "Tiled attention Q-cache source anchor count mismatch" unless count == 1
    source = source.gsub(needle, replacement)
  end
  source
end

private def wan_bounded_qcache_pipeline(source : String) : ML::Metal::ComputePipeline
  name = "wan_bounded_qcache_attention_#{WAN_BOUNDED_HEAD_DIM}"
  ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, source) }
end

private def wan_bounded_native_range(encoder : ML::Metal::ComputeEncoder,
                                     pipeline : ML::Metal::ComputePipeline,
                                     q : ML::MetalBuffer, k : ML::MetalBuffer,
                                     v : ML::MetalBuffer, output : ML::MetalBuffer,
                                     query_rows : Int32, first_tile : Int32,
                                     tile_count : Int32, threads : Int32 = 32) : Nil
  scale = 1_f32 / Math.sqrt(WAN_BOUNDED_HEAD_DIM.to_f32)
  params = StaticArray[query_rows.to_u32, WAN_BOUNDED_K_ROWS.to_u32,
    WAN_BOUNDED_HEADS.to_u32, scale.unsafe_as(UInt32), first_tile.to_u32]
  encoder.set_pipeline(pipeline)
  encoder.set_buffer(q, 0).set_buffer(k, 1).set_buffer(v, 2)
  encoder.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
  encoder.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 4)
  encoder.dispatch_threadgroups({tile_count, 1, 1}, {threads, 1, 1})
end

private def wan_bounded_qcache_attention(chain : ML::Metal::MPSCommandChain,
                                         pipeline : ML::Metal::ComputePipeline,
                                         q : ML::MetalBuffer, k : ML::MetalBuffer,
                                         v : ML::MetalBuffer, output : ML::MetalBuffer,
                                         query_rows : Int32) : Nil
  total_tiles = ((query_rows + 7) // 8) * WAN_BOUNDED_HEADS
  first_tile = 0
  while first_tile < total_tiles
    count = Math.min(WAN_BOUNDED_GROUPS_PER_SEGMENT, total_tiles - first_tile)
    wan_bounded_native_range(chain.encoder, pipeline, q, k, v, output,
      query_rows, first_tile, count)
    first_tile += count
    unless first_tile == total_tiles
      chain.pause!
      chain.continue_async!
      chain.resume!
    end
  end
end

private def wan_bounded_shared_kv_attention(chain : ML::Metal::MPSCommandChain,
                                            pipeline : ML::Metal::ComputePipeline,
                                            simd_groups : Int32,
                                            q : ML::MetalBuffer, k : ML::MetalBuffer,
                                            v : ML::MetalBuffer, output : ML::MetalBuffer,
                                            query_rows : Int32) : Nil
  # Keep the existing upper bound on eight-query SIMD groups per submission.
  # Tail-only groups still execute the barriers and count toward this budget.
  groups_per_segment = WAN_BOUNDED_GROUPS_PER_SEGMENT // simd_groups
  query_rows_per_group = 8 * simd_groups
  total_tiles = ((query_rows + query_rows_per_group - 1) // query_rows_per_group) * WAN_BOUNDED_HEADS
  first_tile = 0
  while first_tile < total_tiles
    count = Math.min(groups_per_segment, total_tiles - first_tile)
    wan_bounded_native_range(chain.encoder, pipeline, q, k, v, output,
      query_rows, first_tile, count, simd_groups * 32)
    first_tile += count
    unless first_tile == total_tiles
      chain.pause!
      chain.continue_async!
      chain.resume!
    end
  end
end

private def wan_bounded_register_kv_attention(chain : ML::Metal::MPSCommandChain,
                                              pipeline : ML::Metal::ComputePipeline,
                                              q : ML::MetalBuffer, k : ML::MetalBuffer,
                                              v : ML::MetalBuffer, output : ML::MetalBuffer,
                                              query_rows : Int32) : Nil
  # Bound work by eight-query tile equivalents, including masked tail tiles.
  groups_per_segment = WAN_BOUNDED_GROUPS_PER_SEGMENT // WAN_BOUNDED_REGISTER_Q_TILES
  rows_per_group = 8 * WAN_BOUNDED_REGISTER_Q_TILES
  total_tiles = ((query_rows + rows_per_group - 1) // rows_per_group) * WAN_BOUNDED_HEADS
  first_tile = 0
  while first_tile < total_tiles
    count = Math.min(groups_per_segment, total_tiles - first_tile)
    wan_bounded_native_range(chain.encoder, pipeline, q, k, v, output,
      query_rows, first_tile, count)
    first_tile += count
    unless first_tile == total_tiles
      chain.pause!
      chain.continue_async!
      chain.resume!
    end
  end
end

private def wan_bounded_encode_fill(encoder : ML::Metal::ComputeEncoder,
                                    pipeline : ML::Metal::ComputePipeline,
                                    buffer : ML::MetalBuffer, count : Int32,
                                    flat_base : UInt32, salt : UInt32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).set_value(flat_base, 2).set_value(salt, 3)
  encoder.dispatch_1d(count)
end

private def wan_bounded_encode_nan_fill(encoder : ML::Metal::ComputeEncoder,
                                        pipeline : ML::Metal::ComputePipeline,
                                        buffer : ML::MetalBuffer, count : Int32) : Nil
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(count.to_u32, 1).dispatch_1d(count)
end

private def wan_bounded_encode_compare(encoder : ML::Metal::ComputeEncoder,
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
  encoder.set_value(WAN_BOUNDED_HEADS.to_u32, 5).set_value(WAN_BOUNDED_HEAD_DIM.to_u32, 6)
  encoder.set_value(reference_bhqd ? 1_u32 : 0_u32, 7).dispatch_1d(count)
end

private def wan_bounded_read_metrics(buffer : ML::MetalBuffer)
  values = buffer.contents.as(UInt32*)
  {
    bit_differences: values[0], tolerance_violations: values[1],
    nonfinite: values[2], max_abs_error: values[3].unsafe_as(Float32),
    max_tolerance_ratio: values[4].unsafe_as(Float32),
  }
end

private def wan_bounded_emit_check(name : String, result, bitwise_required : Bool) : Bool
  passed = result[:tolerance_violations] == 0 && result[:nonfinite] == 0 &&
           (!bitwise_required || result[:bit_differences] == 0)
  wan_bounded_emit({
    schema: "wan_s2v_bounded_attention_validation_v1",
    domain: "wan.s2v.self_attention", check: name,
    bit_differences: result[:bit_differences],
    strict_tolerance_violations: result[:tolerance_violations],
    nonfinite: result[:nonfinite], max_abs_error: result[:max_abs_error],
    max_tolerance_ratio: result[:max_tolerance_ratio],
    tolerance: "2e-5+2e-5*abs(reference)", bitwise_required: bitwise_required,
    passed: passed,
  }.to_json)
  passed
end

private def wan_bounded_validate_pair(expected : ML::MetalBuffer,
                                      baseline : ML::MetalBuffer,
                                      candidate : ML::MetalBuffer,
                                      metrics_baseline : ML::MetalBuffer,
                                      metrics_candidate : ML::MetalBuffer,
                                      metrics_pair : ML::MetalBuffer,
                                      zero : ML::Metal::ComputePipeline,
                                      compare : ML::Metal::ComputePipeline,
                                      query_rows : Int32, comparison : String,
                                      phase : String, pair_index : Int32,
                                      pair_order : String) : Nil
  count = query_rows * WAN_BOUNDED_CHANNELS
  validation = ML::Metal::MPSCommandChain.new
  begin
    e = validation.encoder
    wan_bounded_encode_compare(e, zero, compare, expected, baseline,
      metrics_baseline, count, query_rows, true)
    wan_bounded_encode_compare(e, zero, compare, expected, candidate,
      metrics_candidate, count, query_rows, true)
    wan_bounded_encode_compare(e, zero, compare, baseline, candidate,
      metrics_pair, count, query_rows, false)
    validation.commit_and_wait
    raise "attention validation did not use one terminal wait" unless validation.wait_count == 1
    baseline_result = wan_bounded_read_metrics(metrics_baseline)
    candidate_result = wan_bounded_read_metrics(metrics_candidate)
    pair_result = wan_bounded_read_metrics(metrics_pair)
    validation.release
    baseline_passed = wan_bounded_emit_check("baseline_vs_upstream", baseline_result, false)
    candidate_passed = wan_bounded_emit_check("candidate_vs_upstream", candidate_result, false)
    pair_passed = wan_bounded_emit_check("candidate_vs_baseline", pair_result,
      !comparison.starts_with?("mps-vs-"))
    passed = baseline_passed && candidate_passed && pair_passed
    wan_bounded_emit({
      schema: "wan_s2v_bounded_attention_validation_batch_v1",
      phase: phase, comparison: comparison, pair: pair_index,
      pair_order: pair_order, values: count, terminal_waits: validation.wait_count,
      all_checks_passed: passed,
    }.to_json)
    raise "strict selected-query attention qualification failed" unless passed
  ensure
    unless validation.completed?
      begin
        validation.commit_and_wait
      rescue
      end
    end
    if validation.completed?
      begin
        validation.release
      rescue
      end
    end
  end
end

private def wan_bounded_run_arm(backend : String,
                                query_rows : Int32,
                                q : ML::MetalBuffer, k : ML::MetalBuffer,
                                v : ML::MetalBuffer, output : ML::MetalBuffer,
                                mps_workspace : ML::Video::WanAttentionMPSChunkWorkspace?,
                                qcache_pipeline : ML::Metal::ComputePipeline?,
                                shared_kv_pipeline : ML::Metal::ComputePipeline?,
                                shared_kv_simd_groups : Int32,
                                register_kv_pipeline : ML::Metal::ComputePipeline?,
                                prefetch_kv_pipeline : ML::Metal::ComputePipeline?)
  chain = ML::Metal::MPSCommandChain.new
  begin
    allocation_count = ML::MetalBuffer.video_spec_allocations
    started = Time.instant
    case backend
    when "mps_chunk128"
      mps_workspace.not_nil!.encode(chain)
    when "native_tiled"
      ML::Video::MetalOps.attention_tiled(chain, q, k, v, output, query_rows,
        WAN_BOUNDED_K_ROWS, WAN_BOUNDED_HEADS, WAN_BOUNDED_HEAD_DIM,
        1, WAN_BOUNDED_GROUPS_PER_SEGMENT)
    when "tiled_q_cache"
      wan_bounded_qcache_attention(chain, qcache_pipeline.not_nil!, q, k, v,
        output, query_rows)
    when "shared_kv_sg2", "shared_kv_sg4"
      raise "shared K/V backend/group mismatch" unless backend == "shared_kv_sg#{shared_kv_simd_groups}"
      wan_bounded_shared_kv_attention(chain, shared_kv_pipeline.not_nil!, shared_kv_simd_groups,
        q, k, v, output, query_rows)
    when "register_kv_q2"
      wan_bounded_register_kv_attention(chain, register_kv_pipeline.not_nil!,
        q, k, v, output, query_rows)
    when "prefetch_kv"
      # Match the Q-cache baseline's one-SIMD-group, eight-query-tile mapping
      # and 320-tile-equivalent segment bound.
      wan_bounded_qcache_attention(chain, prefetch_kv_pipeline.not_nil!,
        q, k, v, output, query_rows)
    else
      raise ArgumentError.new("Unknown bounded attention backend: #{backend}")
    end
    encode_seconds = (Time.instant - started).total_seconds
    raise "attention encode allocated a tracked MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == allocation_count
    chain.commit_and_wait
    wall_seconds = (Time.instant - started).total_seconds
    raise "attention arm did not use one terminal wait" unless chain.wait_count == 1
    raise "attention arm did not cross the segmented path" unless chain.segment_count > 0
    result = {
      backend: backend, encode_wall_seconds: encode_seconds,
      arm_wall_seconds: wall_seconds,
      summed_command_root_gpu_interval_seconds: chain.gpu_elapsed_seconds,
      segments: chain.segment_count, terminal_waits: chain.wait_count,
      encode_metal_buffer_allocations: ML::MetalBuffer.video_spec_allocations - allocation_count,
    }
    chain.release
    result
  ensure
    unless chain.completed?
      begin
        chain.commit_and_wait
      rescue
      end
    end
    if chain.completed?
      begin
        chain.release
      rescue
      end
    end
  end
end

fixture_dir = ""
query_rows = 256
query_start = 0
comparison = "mps-vs-tiled"
OptionParser.parse do |parser|
  parser.banner = "Usage: wan_s2v_bounded_attention_benchmark --fixture=DIR [--query-rows=N] [--query-start=N] [--comparison=MODE]"
  parser.on("--fixture=DIR", "Pinned full-query upstream attention fixture directory (required)") { |value| fixture_dir = value }
  parser.on("--query-rows=N", "Selected query rows (1..#{WAN_BOUNDED_QUERY_LIMIT}; default 256)") { |value| query_rows = value.to_i }
  parser.on("--query-start=N", "First full-sequence query row (default 0)") { |value| query_start = value.to_i }
  parser.on("--comparison=MODE", "mps-vs-tiled, tiled-q-cache, q-cache-vs-shared-kv-sg2, q-cache-vs-shared-kv-sg4, q-cache-vs-register-kv-q2, mps-vs-register-kv-q2, q-cache-vs-prefetch-kv or mps-vs-prefetch-kv") { |value| comparison = value }
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
raise ArgumentError.new("--query-rows must be within 1..#{WAN_BOUNDED_QUERY_LIMIT}") unless query_rows >= 1 && query_rows <= WAN_BOUNDED_QUERY_LIMIT
raise ArgumentError.new("--query-start and --query-rows exceed full query extent") unless query_start >= 0 && query_start.to_i64 + query_rows <= WAN_BOUNDED_Q_ROWS
unless {"mps-vs-tiled", "tiled-q-cache", "q-cache-vs-shared-kv-sg2", "q-cache-vs-shared-kv-sg4", "q-cache-vs-register-kv-q2", "mps-vs-register-kv-q2", "q-cache-vs-prefetch-kv", "mps-vs-prefetch-kv"}.includes?(comparison)
  raise ArgumentError.new("Unsupported --comparison mode")
end
mps_baseline = comparison.starts_with?("mps-vs-")
register_kv_enabled = comparison.ends_with?("register-kv-q2")
prefetch_kv_enabled = comparison.ends_with?("prefetch-kv")
shared_kv_simd_groups = case comparison
                        when "q-cache-vs-shared-kv-sg2" then 2
                        when "q-cache-vs-shared-kv-sg4" then 4
                        else                                 0
                        end

fixture_dir = File.expand_path(fixture_dir)
fixture = wan_bounded_validate_fixture!(fixture_dir)
query_values = query_rows.to_i64 * WAN_BOUNDED_CHANNELS
kv_values = WAN_BOUNDED_K_ROWS.to_i64 * WAN_BOUNDED_CHANNELS
chunk_values = WAN_BOUNDED_CHUNK_ROWS.to_i64 * WAN_BOUNDED_CHANNELS
mps_scratch_values = mps_baseline ? chunk_values * 3 : 0_i64
metrics_values = 15_i64
# Four query-sized F32 buffers: Q, expected BHQD slice, baseline and candidate.
explicit_plan_bytes = (query_values * 4 + kv_values * 2 + mps_scratch_values + metrics_values) * sizeof(Float32)
raise "bounded attention explicit plan exceeds 2 GiB" if explicit_plan_bytes > WAN_BOUNDED_EXPLICIT_CAP

wan_bounded_emit({
  schema: "wan_s2v_bounded_attention_setup_v1",
  domain: "wan.s2v.self_attention", comparison: comparison,
  device_name: ML::Metal::Device.instance.name,
  query_start: query_start, query_rows: query_rows,
  key_rows: WAN_BOUNDED_K_ROWS, heads: WAN_BOUNDED_HEADS,
  head_dim: WAN_BOUNDED_HEAD_DIM, channels: WAN_BOUNDED_CHANNELS,
  native_input_formula: WAN_BOUNDED_FORMULA, q_salt: 1, k_salt: 2, v_salt: 3,
  fixture_manifest_path: fixture[:manifest_path], fixture_manifest_sha256: fixture[:manifest_sha256],
  expected_full_file_sha256: fixture[:expected_sha256], fixture_immutable_during_run_assumption: true,
  expected_slice_layout: "B,H,Q,D selected by per-head seek; native output B,Q,H,D",
  expected_source_weight_dtype: "F32", reference_torch_version: fixture[:torch_version],
  wan_attention_source_sha256: fixture[:attention_source_sha256],
  wan_s2v_reference_adapter_sha256: fixture[:adapter_sha256],
  generator_sha256: fixture[:generator_sha256],
  explicit_buffer_plan_bytes: explicit_plan_bytes,
  explicit_buffer_cap_bytes: WAN_BOUNDED_EXPLICIT_CAP,
  mps_private_workspace_bytes: "opaque and not included in explicit MetalBuffer accounting",
  model_weights_loaded: false, full_kv_used: true,
  scope: "selected-query attention primitive only; no full block/model/video throughput inference",
}.to_json)

# Check the compiled candidate on the actual bridge device before allocating
# large input buffers. No dynamic threadgroup storage is added at dispatch.
shared_kv_pipeline = nil.as(ML::Metal::ComputePipeline?)
if shared_kv_simd_groups > 0
  name = "wan_bounded_shared_kv_attention_#{WAN_BOUNDED_HEAD_DIM}_sg#{shared_kv_simd_groups}"
  shared_kv_pipeline = ML::Metal::ComputePipeline.new(name, WAN_BOUNDED_SHARED_KV_SOURCE)
  device = ML::Metal::Device.instance
  memory_limit = WanBoundedAttentionLimits.max_threadgroup_memory_length(device.device_handle)
  static_bytes = shared_kv_pipeline.static_threadgroup_memory_length
  width = shared_kv_pipeline.thread_execution_width
  max_threads = shared_kv_pipeline.max_total_threads_per_threadgroup
  required_threads = shared_kv_simd_groups * 32
  unless width == 32 && max_threads >= required_threads &&
         static_bytes >= WAN_BOUNDED_SHARED_KV_BYTES && static_bytes <= memory_limit
    raise "shared K/V pipeline exceeds the device or pipeline capability limits"
  end
  wan_bounded_emit({
    schema: "wan_s2v_bounded_attention_shared_kv_capabilities_v1",
    kernel: name, kernel_source_sha256: Digest::SHA256.hexdigest(WAN_BOUNDED_SHARED_KV_SOURCE),
    device_max_threadgroup_memory_bytes: memory_limit,
    pipeline_static_threadgroup_memory_bytes: static_bytes,
    planned_threadgroup_memory_bytes: WAN_BOUNDED_SHARED_KV_BYTES,
    thread_execution_width: width, pipeline_max_threads: max_threads,
    threads_per_threadgroup: required_threads, simd_groups_per_threadgroup: shared_kv_simd_groups,
    max_threadgroups_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT // shared_kv_simd_groups,
    max_simd_groups_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT,
    passed: true,
  }.to_json)
end

register_kv_pipeline = nil.as(ML::Metal::ComputePipeline?)
if register_kv_enabled
  name = "wan_bounded_register_kv_attention_#{WAN_BOUNDED_HEAD_DIM}_q#{WAN_BOUNDED_REGISTER_Q_TILES}"
  register_kv_pipeline = ML::Metal::ComputePipeline.new(name, WAN_BOUNDED_REGISTER_KV_SOURCE)
  static_bytes = register_kv_pipeline.static_threadgroup_memory_length
  width = register_kv_pipeline.thread_execution_width
  max_threads = register_kv_pipeline.max_total_threads_per_threadgroup
  unless width == 32 && max_threads >= 32 && static_bytes == 0
    raise "register K/V pipeline exceeds the admitted capability limits"
  end
  wan_bounded_emit({
    schema: "wan_s2v_bounded_attention_register_kv_capabilities_v1",
    kernel: name, kernel_source_sha256: Digest::SHA256.hexdigest(WAN_BOUNDED_REGISTER_KV_SOURCE),
    pipeline_static_threadgroup_memory_bytes: static_bytes,
    thread_execution_width: width, pipeline_max_threads: max_threads,
    threads_per_threadgroup: 32, simd_groups_per_threadgroup: 1,
    query_tiles_per_simd_group: WAN_BOUNDED_REGISTER_Q_TILES,
    max_threadgroups_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT // WAN_BOUNDED_REGISTER_Q_TILES,
    max_eight_query_tiles_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT,
    passed: true,
  }.to_json)
end

prefetch_kv_pipeline = nil.as(ML::Metal::ComputePipeline?)
if prefetch_kv_enabled
  name = "wan_bounded_attention_prefetch_kv_#{WAN_BOUNDED_HEAD_DIM}"
  prefetch_kv_pipeline = ML::Metal::ComputePipeline.new(name, WAN_BOUNDED_PREFETCH_KV_SOURCE)
  static_bytes = prefetch_kv_pipeline.static_threadgroup_memory_length
  width = prefetch_kv_pipeline.thread_execution_width
  max_threads = prefetch_kv_pipeline.max_total_threads_per_threadgroup
  unless width == 32 && max_threads >= 32 && static_bytes == 0
    raise "prefetch K/V pipeline exceeds the admitted capability limits"
  end
  wan_bounded_emit({
    schema: "wan_s2v_bounded_attention_prefetch_kv_capabilities_v1",
    kernel: name, kernel_source_sha256: Digest::SHA256.hexdigest(WAN_BOUNDED_PREFETCH_KV_SOURCE),
    pipeline_static_threadgroup_memory_bytes: static_bytes,
    thread_execution_width: width, pipeline_max_threads: max_threads,
    threads_per_threadgroup: 32, simd_groups_per_threadgroup: 1,
    query_tiles_per_simd_group: 1,
    max_threadgroups_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT,
    max_eight_query_tiles_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT,
    passed: true,
  }.to_json)
end

arena = ML::Video::Arena.new
mps_workspace = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
begin
  q = arena.alloc("bounded_query_input", query_values.to_i32)
  k = arena.alloc("bounded_full_key_input", kv_values.to_i32)
  v = arena.alloc("bounded_full_value_input", kv_values.to_i32)
  expected = arena.alloc("bounded_upstream_expected_slice", query_values.to_i32)
  baseline_output = arena.alloc("bounded_baseline_output", query_values.to_i32)
  candidate_output = arena.alloc("bounded_candidate_output", query_values.to_i32)
  metrics_baseline = arena.alloc("bounded_metrics_baseline_upstream", 5)
  metrics_candidate = arena.alloc("bounded_metrics_candidate_upstream", 5)
  metrics_pair = arena.alloc("bounded_metrics_candidate_baseline", 5)

  baseline_backend, candidate_backend = if register_kv_enabled
                                          {mps_baseline ? "mps_chunk128" : "tiled_q_cache", "register_kv_q2"}
                                        elsif comparison == "mps-vs-tiled"
                                          {"mps_chunk128", "native_tiled"}
                                        elsif comparison == "tiled-q-cache"
                                          {"native_tiled", "tiled_q_cache"}
                                        elsif prefetch_kv_enabled
                                          {mps_baseline ? "mps_chunk128" : "tiled_q_cache", "prefetch_kv"}
                                        else
                                          {"tiled_q_cache", "shared_kv_sg#{shared_kv_simd_groups}"}
                                        end
  baseline_chunk_q = nil.as(ML::MetalBuffer?)
  baseline_chunk_a = nil.as(ML::MetalBuffer?)
  baseline_chunk_b = nil.as(ML::MetalBuffer?)
  if mps_baseline
    baseline_chunk_q = arena.alloc("bounded_mps_chunk_q", chunk_values.to_i32)
    baseline_chunk_a = arena.alloc("bounded_mps_chunk_output_a", chunk_values.to_i32)
    baseline_chunk_b = arena.alloc("bounded_mps_chunk_output_b", chunk_values.to_i32)
  end

  wan_bounded_load_expected_slice(fixture[:expected_path], expected, query_start, query_rows)
  fill_pipeline = ML::Metal::ComputePipeline.new("wan_bounded_attention_fill", WAN_BOUNDED_PROBE_SOURCE)
  nan_fill_pipeline = ML::Metal::ComputePipeline.new("wan_bounded_attention_nan_fill", WAN_BOUNDED_PROBE_SOURCE)
  zero_pipeline = ML::Metal::ComputePipeline.new("wan_bounded_attention_zero", WAN_BOUNDED_PROBE_SOURCE)
  compare_pipeline = ML::Metal::ComputePipeline.new("wan_bounded_attention_compare", WAN_BOUNDED_PROBE_SOURCE)
  corrupt_pipeline = ML::Metal::ComputePipeline.new("wan_bounded_attention_corrupt_one", WAN_BOUNDED_PROBE_SOURCE)
  nan_corrupt_pipeline = ML::Metal::ComputePipeline.new("wan_bounded_attention_nan_one", WAN_BOUNDED_PROBE_SOURCE)
  ML::Video::MetalOps.tiled_attention_pipeline(WAN_BOUNDED_HEAD_DIM)
  ML::Video::MetalOps.pipeline("video_copy_f32")
  qcache_pipeline = nil.as(ML::Metal::ComputePipeline?)
  unless mps_baseline
    qcache_source = wan_bounded_qcache_source
    qcache_pipeline = wan_bounded_qcache_pipeline(qcache_source)
  end

  if mps_baseline
    mps_workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(1,
      query_rows, WAN_BOUNDED_K_ROWS, WAN_BOUNDED_HEADS, WAN_BOUNDED_HEAD_DIM,
      WAN_BOUNDED_CHUNK_ROWS, q, k, v, baseline_output,
      baseline_chunk_q.not_nil!, baseline_chunk_a.not_nil!, baseline_chunk_b.not_nil!)
  end
  actual_explicit_bytes = arena.resident_bytes
  unless actual_explicit_bytes == explicit_plan_bytes && actual_explicit_bytes <= WAN_BOUNDED_EXPLICIT_CAP
    raise ArgumentError.new("Actual explicit buffer bytes differ from the admitted plan")
  end

  # Reproduce the reference's full-flat-index Q slice and full K/V salts.
  setup_started = Time.instant
  setup = ML::Metal::CommandBuffer.new
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  wan_bounded_encode_fill(setup_encoder, fill_pipeline, q, query_values.to_i32,
    (query_start.to_i64 * WAN_BOUNDED_CHANNELS).to_u32, 1_u32)
  wan_bounded_encode_fill(setup_encoder, fill_pipeline, k, kv_values.to_i32, 0_u32, 2_u32)
  wan_bounded_encode_fill(setup_encoder, fill_pipeline, v, kv_values.to_i32, 0_u32, 3_u32)
  # Any output element an arm fails to write remains a NaN and fails the
  # subsequent finite-value gate, including the final partial query tile.
  wan_bounded_encode_nan_fill(setup_encoder, nan_fill_pipeline, baseline_output, query_values.to_i32)
  wan_bounded_encode_nan_fill(setup_encoder, nan_fill_pipeline, candidate_output, query_values.to_i32)
  setup_encoder.end_encoding
  setup.commit_and_wait
  setup_seconds = (Time.instant - setup_started).total_seconds

  wan_bounded_emit({
    schema: "wan_s2v_bounded_attention_plan_v1",
    domain: "wan.s2v.self_attention", comparison: comparison,
    baseline_backend: baseline_backend, candidate_backend: candidate_backend,
    query_start: query_start, query_rows: query_rows,
    key_rows: WAN_BOUNDED_K_ROWS, heads: WAN_BOUNDED_HEADS, head_dim: WAN_BOUNDED_HEAD_DIM,
    explicit_buffer_bytes: actual_explicit_bytes, explicit_buffer_cap_bytes: WAN_BOUNDED_EXPLICIT_CAP,
    input_fill_waits: 1, input_fill_host_wall_seconds: setup_seconds,
    mps_chunk_rows: mps_baseline ? WAN_BOUNDED_CHUNK_ROWS : nil,
    native_groups_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT,
    shared_kv_simd_groups_per_threadgroup: shared_kv_simd_groups,
    shared_kv_threadgroup_memory_bytes: shared_kv_simd_groups > 0 ? WAN_BOUNDED_SHARED_KV_BYTES : 0_i64,
    both_arms_cache_q: shared_kv_simd_groups > 0 ||
                       ((register_kv_enabled || prefetch_kv_enabled) && !mps_baseline),
    register_kv_query_tiles_per_simd_group: register_kv_enabled ? WAN_BOUNDED_REGISTER_Q_TILES : 0,
    register_kv_max_threadgroups_per_segment: register_kv_enabled ? WAN_BOUNDED_GROUPS_PER_SEGMENT // WAN_BOUNDED_REGISTER_Q_TILES : 0,
    prefetch_kv_query_tiles_per_simd_group: prefetch_kv_enabled ? 1 : 0,
    prefetch_kv_max_threadgroups_per_segment: prefetch_kv_enabled ? WAN_BOUNDED_GROUPS_PER_SEGMENT : 0,
    max_eight_query_tiles_per_segment: WAN_BOUNDED_GROUPS_PER_SEGMENT,
    fixture_manifest_path: fixture[:manifest_path], fixture_manifest_sha256: fixture[:manifest_sha256],
    source_checkpoint_weights_loaded: false,
  }.to_json)

  # One warmup AB then exactly three measured AB/BA/AB pairs.
  pair_orders = {"AB", "BA", "AB"}
  warmup_order = {baseline_backend, candidate_backend}
  warmup_order.each do |backend|
    output = backend == baseline_backend ? baseline_output : candidate_output
    result = wan_bounded_run_arm(backend, query_rows, q, k, v, output,
      mps_workspace, qcache_pipeline, shared_kv_pipeline, shared_kv_simd_groups,
      register_kv_pipeline, prefetch_kv_pipeline)
    wan_bounded_emit({
      schema: "wan_s2v_bounded_attention_arm_v1", phase: "warmup",
      comparison: comparison, pair: 0, pair_order: "AB", backend: backend,
      query_start: query_start, query_rows: query_rows,
      key_rows: WAN_BOUNDED_K_ROWS, heads: WAN_BOUNDED_HEADS, head_dim: WAN_BOUNDED_HEAD_DIM,
      encode_wall_seconds: result[:encode_wall_seconds],
      arm_wall_seconds: result[:arm_wall_seconds],
      summed_command_root_gpu_interval_seconds: result[:summed_command_root_gpu_interval_seconds],
      gpu_interval_interpretation: "sum of command-root GPU intervals; not additive kernel times",
      segments: result[:segments], terminal_waits: result[:terminal_waits],
      encode_metal_buffer_allocations: result[:encode_metal_buffer_allocations],
      full_kv_used: true,
    }.to_json)
  end
  wan_bounded_validate_pair(expected, baseline_output, candidate_output,
    metrics_baseline, metrics_candidate, metrics_pair,
    zero_pipeline, compare_pipeline, query_rows, comparison, "warmup", 0, "AB")

  baseline_wall_samples = [] of Float64
  candidate_wall_samples = [] of Float64
  measured_orders = [] of String
  pair_orders.each_with_index do |pair_order, pair_index|
    order = pair_order == "AB" ? {baseline_backend, candidate_backend} : {candidate_backend, baseline_backend}
    order.each do |backend|
      output = backend == baseline_backend ? baseline_output : candidate_output
      result = wan_bounded_run_arm(backend, query_rows, q, k, v, output,
        mps_workspace, qcache_pipeline, shared_kv_pipeline, shared_kv_simd_groups,
        register_kv_pipeline, prefetch_kv_pipeline)
      if backend == baseline_backend
        baseline_wall_samples << result[:arm_wall_seconds]
      else
        candidate_wall_samples << result[:arm_wall_seconds]
      end
      wan_bounded_emit({
        schema: "wan_s2v_bounded_attention_arm_v1", phase: "measured",
        comparison: comparison, pair: pair_index + 1, pair_order: pair_order,
        backend: backend, query_start: query_start, query_rows: query_rows,
        key_rows: WAN_BOUNDED_K_ROWS, heads: WAN_BOUNDED_HEADS, head_dim: WAN_BOUNDED_HEAD_DIM,
        encode_wall_seconds: result[:encode_wall_seconds],
        arm_wall_seconds: result[:arm_wall_seconds],
        summed_command_root_gpu_interval_seconds: result[:summed_command_root_gpu_interval_seconds],
        gpu_interval_interpretation: "sum of command-root GPU intervals; not additive kernel times",
        segments: result[:segments], terminal_waits: result[:terminal_waits],
        encode_metal_buffer_allocations: result[:encode_metal_buffer_allocations],
        full_kv_used: true,
      }.to_json)
    end
    measured_orders << pair_order
    wan_bounded_validate_pair(expected, baseline_output, candidate_output,
      metrics_baseline, metrics_candidate, metrics_pair,
      zero_pipeline, compare_pipeline, query_rows, comparison,
      "measured", pair_index + 1, pair_order)
  end

  baseline_sorted = baseline_wall_samples.sort
  candidate_sorted = candidate_wall_samples.sort
  baseline_median = baseline_sorted[baseline_sorted.size // 2]
  candidate_median = candidate_sorted[candidate_sorted.size // 2]
  paired_savings = baseline_wall_samples.zip(candidate_wall_samples).map { |base, candidate| base - candidate }
  all_pairs_faster = paired_savings.all? { |saving| saving > 0.0 }
  candidate_faster = all_pairs_faster && candidate_median < baseline_median
  # Tail/other shapes supply correctness diagnostics, never timing admission.
  timing_admission_eligible = query_rows == 256 && query_start == 0
  wan_bounded_emit({
    schema: "wan_s2v_bounded_attention_decision_v1", domain: "wan.s2v.self_attention",
    comparison: comparison, baseline_backend: baseline_backend, candidate_backend: candidate_backend,
    query_start: query_start, query_rows: query_rows, key_rows: WAN_BOUNDED_K_ROWS,
    heads: WAN_BOUNDED_HEADS, head_dim: WAN_BOUNDED_HEAD_DIM,
    measured_pair_orders: measured_orders,
    baseline_arm_wall_seconds: baseline_wall_samples,
    candidate_arm_wall_seconds: candidate_wall_samples,
    baseline_median_seconds: baseline_median, candidate_median_seconds: candidate_median,
    paired_baseline_minus_candidate_seconds: paired_savings,
    all_three_pairs_candidate_faster: all_pairs_faster,
    median_candidate_faster: candidate_median < baseline_median,
    timing_admission_eligible: timing_admission_eligible,
    primitive_candidate_admitted_for_further_review: candidate_faster && timing_admission_eligible,
    full_block_or_video_speedup_claim: false, model_throughput_extrapolation: false,
    timing_scope: "selected-query attention arm wall including one terminal wait; validation excluded",
  }.to_json)

  # All numerical checks for the final pair are complete before the control.
  control = ML::Metal::MPSCommandChain.new
  begin
    e = control.encoder
    ML::Video::MetalOps.copy(e, baseline_output, candidate_output, query_values.to_i32)
    e.set_pipeline(corrupt_pipeline)
      .set_buffer(candidate_output, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
    e.set_pipeline(nan_corrupt_pipeline)
      .set_buffer(candidate_output, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
    wan_bounded_encode_compare(e, zero_pipeline, compare_pipeline,
      baseline_output, candidate_output, metrics_pair, query_values.to_i32,
      query_rows, false)
    control.commit_and_wait
    raise "attention corruption control did not use one terminal wait" unless control.wait_count == 1
    control_metrics = wan_bounded_read_metrics(metrics_pair)
    unless control_metrics[:bit_differences] == 2 && control_metrics[:tolerance_violations] == 2 &&
           control_metrics[:nonfinite] == 1
      raise "attention corruption control did not reject one +1 value and one NaN"
    end
    wan_bounded_emit({
      schema: "wan_s2v_bounded_attention_negative_control_v1",
      corrupted_values: 2, corruption: "candidate[0] += 1.0f; candidate[1] = quiet_NaN()",
      compared_values: query_values, bit_differences: control_metrics[:bit_differences],
      strict_tolerance_violations: control_metrics[:tolerance_violations],
      nonfinite: control_metrics[:nonfinite], terminal_waits: control.wait_count,
    }.to_json)
    control.release
  ensure
    unless control.completed?
      begin
        control.commit_and_wait
      rescue
      end
    end
    if control.completed?
      begin
        control.release
      rescue
      end
    end
  end
ensure
  if workspace = mps_workspace
    begin
      workspace.release
    rescue
    end
  end
  arena.release
end
