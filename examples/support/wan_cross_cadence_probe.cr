require "digest/sha256"
require "json"

module WanCrossCadenceProbe
  QUERY_ROWS          = 36_608
  KEY_ROWS            =    512
  HEADS               =     40
  HEAD_DIM            =    128
  CHANNELS            = HEADS * HEAD_DIM
  GROUPS_PER_DISPATCH = 320
  VALUES_PER_ROW      = CHANNELS
  FULL_VALUES         = QUERY_ROWS.to_i64 * VALUES_PER_ROW
  FULL_BYTES          = FULL_VALUES * sizeof(Float32)
  BASE_MEMORY_BYTES   = 3_i64 * FULL_BYTES + 2_i64 * KEY_ROWS * CHANNELS * sizeof(Float32)
  MEMORY_CAP_BYTES    = 3_i64 * 1024 * 1024 * 1024
  IO_CHUNK_BYTES      = 8 * 1024 * 1024

  FORMULA                       = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
  FIXTURE_SCHEMA                = "wan_s2v_text_cross_cadence_reference_v1"
  FIXTURE_MANIFEST_SHA256       = "113550335981d111a37121f1258205cd97a5b8a33ece3d5d4c6613b7fc8f54ea"
  FIXTURE_EXPECTED_SHA256       = "bc8136211a4fb6251f73a810e70f63dfe5999a0475d27fa9e87f91d6bbe75739"
  WAN_ATTENTION_SOURCE          = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit/wan_modules_attention.py"
  WAN_ATTENTION_SOURCE_SHA256   = "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc"
  ADAPTER_SOURCE_SHA256         = "863f0020418f3da40caf6f2863dbef9ed40d147127c34b0748cf3415433333a3"
  GENERATOR_SOURCE_SHA256       = "54ca338f6abd0fa09958d38b82e1e15e03279fdd8afdb13a1ed39814d35a6f00"
  NATIVE_OPERATOR_SOURCE_SHA256 = "9665db4c98a582ce84470c75936fb31f78235ee02b364dbdd9acfbf83879803d"
  NATIVE_KERNEL_SOURCE_SHA256   = "f79fa9bd1fdb77b98861f436b1f4db4cdd8918965de6bebc9b2a0b9000c8768a"

  METAL_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    inline float wan_cross_cadence_value(uint index, uint salt) {
      uint n = index + salt * 0x9e3779b9u;
      n ^= n >> 16;
      n *= 0x7feb352du;
      n ^= n >> 15;
      n *= 0x846ca68bu;
      n ^= n >> 16;
      return float(int(n & 1023u) - 512) / 256.0f;
    }

    kernel void wan_cross_cadence_input_fill(
        device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
        constant uint &flat_base [[buffer(2)]], constant uint &salt [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
      if (index < count) output[index] = wan_cross_cadence_value(flat_base + index, salt);
    }

    kernel void wan_cross_cadence_nan_fill(
        device float *output [[buffer(0)]], constant uint &count [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      if (index < count) output[index] = as_type<float>(0x7fc00000u);
    }

    kernel void wan_cross_cadence_zero_metrics(
        device atomic_uint *metrics [[buffer(0)]], uint index [[thread_position_in_grid]]) {
      if (index < 5u) atomic_store_explicit(metrics + index, 0u, memory_order_relaxed);
    }

    // Each group reduces 256 output values before touching global metrics.
    // This keeps the full 187M-value oracle comparison from issuing dense atomics.
    kernel void wan_cross_cadence_compare(
        device const float *reference [[buffer(0)]],
        device const float *actual [[buffer(1)]],
        device atomic_uint *metrics [[buffer(2)]],
        constant uint &count [[buffer(3)]],
        uint index [[thread_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
      threadgroup uint group_differences[256];
      threadgroup uint group_violations[256];
      threadgroup uint group_nonfinite[256];
      threadgroup uint group_max_abs[256];
      threadgroup uint group_max_ratio[256];

      uint differences = 0u;
      uint violations = 0u;
      uint nonfinite = 0u;
      uint max_abs = 0u;
      uint max_ratio = 0u;
      if (index < count) {
        const float ref = reference[index];
        const float got = actual[index];
        const uint ref_bits = as_type<uint>(ref);
        const uint got_bits = as_type<uint>(got);
        if (ref_bits != got_bits) differences = 1u;
        if (!isfinite(ref) || !isfinite(got)) {
          nonfinite = 1u;
        } else if (ref_bits != got_bits) {
          const float error = abs(ref - got);
          const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
          const float ratio = error / tolerance;
          if (error > tolerance) violations = 1u;
          if (error > 0.0f) {
            max_abs = as_type<uint>(error);
            max_ratio = as_type<uint>(ratio);
          }
        }
      }
      group_differences[lane] = differences;
      group_violations[lane] = violations;
      group_nonfinite[lane] = nonfinite;
      group_max_abs[lane] = max_abs;
      group_max_ratio[lane] = max_ratio;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
          group_differences[lane] += group_differences[lane + stride];
          group_violations[lane] += group_violations[lane + stride];
          group_nonfinite[lane] += group_nonfinite[lane + stride];
          group_max_abs[lane] = max(group_max_abs[lane], group_max_abs[lane + stride]);
          group_max_ratio[lane] = max(group_max_ratio[lane], group_max_ratio[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (lane == 0u) {
        atomic_fetch_add_explicit(metrics, group_differences[0], memory_order_relaxed);
        atomic_fetch_add_explicit(metrics + 1, group_violations[0], memory_order_relaxed);
        atomic_fetch_add_explicit(metrics + 2, group_nonfinite[0], memory_order_relaxed);
        atomic_fetch_max_explicit(metrics + 3, group_max_abs[0], memory_order_relaxed);
        atomic_fetch_max_explicit(metrics + 4, group_max_ratio[0], memory_order_relaxed);
      }
    }

    kernel void wan_cross_cadence_add_one(
        device float *output [[buffer(0)]], uint index [[thread_position_in_grid]]) {
      if (index == 0u) output[0] += 1.0f;
    }

    kernel void wan_cross_cadence_make_nan(
        device float *output [[buffer(0)]], uint index [[thread_position_in_grid]]) {
      if (index == 0u) output[0] = as_type<float>(0x7fc00000u);
    }
  METAL

  def self.sha256_file(path : String) : String
    digest = Digest::SHA256.new
    chunk = Bytes.new(1 << 20)
    File.open(path, "rb") do |file|
      while (count = file.read(chunk)) > 0
        digest.update(chunk[0, count])
      end
    end
    digest.final.hexstring
  end

  private def self.verify_source!(source : JSON::Any, path_key : String,
                                  digest_key : String, expected_path : String,
                                  expected_digest : String) : Nil
    path = source[path_key].as_s
    unless File.real_path(path) == File.real_path(expected_path)
      raise ArgumentError.new("Pinned source path mismatch for #{path_key}")
    end
    digest = source[digest_key].as_s
    unless digest == expected_digest && sha256_file(path) == expected_digest
      raise ArgumentError.new("Pinned source SHA-256 mismatch for #{path_key}")
    end
  end

  def self.validate_fixture!(fixture_dir : String) : NamedTuple(
    manifest_path: String, expected_path: String, expected_sha256: String,
    manifest_sha256: String, query_chunk_rows: Int64)
    manifest_path = File.join(fixture_dir, "manifest.json")
    expected_path = File.join(fixture_dir, "expected_bqhd.f32")
    raise ArgumentError.new("Missing cadence reference manifest") unless File.file?(manifest_path)
    raise ArgumentError.new("Missing BQHD reference output") unless File.file?(expected_path)

    manifest_sha = sha256_file(manifest_path)
    unless manifest_sha == FIXTURE_MANIFEST_SHA256
      raise ArgumentError.new("Pinned cadence manifest SHA-256 mismatch")
    end
    raise ArgumentError.new("Reference output byte size mismatch") unless File.size(expected_path) == FULL_BYTES

    manifest = JSON.parse(File.read(manifest_path))
    raise ArgumentError.new("Unsupported cadence fixture schema") unless manifest["schema"].as_s == FIXTURE_SCHEMA
    geometry = manifest["geometry"]
    {
      {"batches", 1_i64}, {"query_rows", QUERY_ROWS.to_i64},
      {"key_rows", KEY_ROWS.to_i64}, {"heads", HEADS.to_i64},
      {"head_dim", HEAD_DIM.to_i64},
    }.each do |key, expected|
      raise ArgumentError.new("Cadence fixture geometry mismatch for #{key}") unless geometry[key].as_i64 == expected
    end
    unless geometry["native_layout"].as_s == "B,Q,H,D" && geometry["reference_layout"].as_s == "B,Q,H,D"
      raise ArgumentError.new("Cadence fixture layout mismatch")
    end
    raise ArgumentError.new("Cadence fixture input formula mismatch") unless manifest["input_formula"].as_s == FORMULA
    salts = manifest["input_salts"]
    unless salts["q"].as_i64 == 1 && salts["k"].as_i64 == 2 && salts["v"].as_i64 == 3
      raise ArgumentError.new("Cadence fixture input salts mismatch")
    end

    reference = manifest["reference"]
    unless reference["operator"].as_s == "torch.nn.functional.scaled_dot_product_attention" &&
           reference["device"].as_s == "mps" && reference["dtype"].as_s == "float32" &&
           reference["mps_fallback_disabled"].as_bool &&
           reference["chunking_semantics"].as_s == "independent query spans with full K/V; no mask, noncausal, dropout=0" &&
           reference["source_adapter"].as_s == "transpose Q/K/V to contiguous B,H,Q,D; PyTorch MPS SDPA; return contiguous B,Q,H,D" &&
           reference["comparison_tolerance"].as_s == "2e-5 + 2e-5 * abs(reference)" &&
           reference["terminal_sync_semantics"].as_s.starts_with?("one explicit torch.mps.synchronize() after all chunks;")
      raise ArgumentError.new("Cadence fixture oracle metadata mismatch")
    end
    raise ArgumentError.new("Cadence fixture must use PyTorch 2.8.x") unless reference["torch_version"].as_s.starts_with?("2.8.")
    query_chunk_rows = reference["query_chunk_rows"].as_i64
    raise ArgumentError.new("Cadence fixture chunk size is unsupported") unless {128_i64, 256_i64}.includes?(query_chunk_rows)

    experiment = manifest["experiment"]
    unless experiment["dispatches_per_command_root"].as_a.map(&.as_i64) == [1_i64, 2_i64, 4_i64] &&
           experiment["groups_per_segment"].as_i64 == GROUPS_PER_DISPATCH &&
           experiment["kernel"].as_s == "attention_tiled_range" &&
           experiment["kernel_unchanged"].as_bool
      raise ArgumentError.new("Cadence experiment metadata mismatch")
    end

    expected = manifest["expected"]
    unless expected["file"].as_s == "expected_bqhd.f32" &&
           expected["dtype"].as_s == "F32_LE" &&
           expected["shape"].as_a.map(&.as_i64) == [1_i64, QUERY_ROWS.to_i64, HEADS.to_i64, HEAD_DIM.to_i64] &&
           expected["bytes"].as_i64 == FULL_BYTES &&
           expected["sha256"].as_s == FIXTURE_EXPECTED_SHA256
      raise ArgumentError.new("Cadence fixture output metadata mismatch")
    end

    source = manifest["source"]
    repo_root = File.expand_path("../..", __DIR__)
    verify_source!(source, "wan_attention_source_path", "wan_attention_source_sha256",
      WAN_ATTENTION_SOURCE, WAN_ATTENTION_SOURCE_SHA256)
    verify_source!(source, "wan_s2v_reference_adapter_path", "wan_s2v_reference_adapter_sha256",
      File.join(repo_root, "spec/support/wan_s2v_real_stack_reference.py"), ADAPTER_SOURCE_SHA256)
    generator = File.join(__DIR__, "wan_s2v_cross_cadence_reference.py")
    verify_source!(source, "generator_path", "generator_sha256", generator, GENERATOR_SOURCE_SHA256)

    native_operator = File.join(repo_root, "src/ml/video/metal_attention.cr")
    native_kernel = File.join(repo_root, "src/ml/video/kernels/attention_tiled.metal")
    unless sha256_file(native_operator) == NATIVE_OPERATOR_SOURCE_SHA256 &&
           sha256_file(native_kernel) == NATIVE_KERNEL_SOURCE_SHA256
      raise ArgumentError.new("Pinned native attention implementation changed")
    end

    {manifest_path: manifest_path, expected_path: expected_path,
     expected_sha256: expected["sha256"].as_s, manifest_sha256: manifest_sha,
     query_chunk_rows: query_chunk_rows}
  end

  # Hash the complete oracle output while copying only the selected BQHD prefix
  # into the shared GPU-visible buffer. The 750 MB file is never materialized.
  def self.load_expected_prefix!(path : String, target : ML::MetalBuffer,
                                 loaded_bytes : Int64) : String
    destination = target.contents.as(UInt8*)
    digest = Digest::SHA256.new
    chunk = Bytes.new(IO_CHUNK_BYTES)
    file_offset = 0_i64
    File.open(path, "rb") do |file|
      while (count = file.read(chunk)) > 0
        digest.update(chunk[0, count])
        copy_bytes = Math.min(count.to_i64, Math.max(loaded_bytes - file_offset, 0_i64)).to_i
        if copy_bytes > 0
          (destination + file_offset).copy_from(chunk.to_unsafe, copy_bytes)
          file_offset += copy_bytes
        end
      end
    end
    raise ArgumentError.new("Reference prefix load was incomplete") unless file_offset == loaded_bytes
    actual_sha = digest.final.hexstring
    raise ArgumentError.new("Pinned cadence expected-output SHA-256 mismatch") unless actual_sha == FIXTURE_EXPECTED_SHA256
    actual_sha
  end

  def self.read_metrics(buffer : ML::MetalBuffer)
    values = buffer.contents.as(UInt32*)
    {
      bit_differences:      values[0],
      tolerance_violations: values[1],
      nonfinite:            values[2],
      max_abs_error:        values[3].unsafe_as(Float32),
      max_tolerance_ratio:  values[4].unsafe_as(Float32),
    }
  end
end
