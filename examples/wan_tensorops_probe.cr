# Opt-in TensorOps/shared-panel diagnostic. Small mode has an independent sequential-FMA
# oracle; dense/QBit modes compare against the pinned t050 upstream projection.
# QBit reconstructs coefficients inside the local B tile, with no global
# dense weight stage. The diagnostic retains dense weights for its MPS control.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_linear_mps"
require "./support/wan_tensorops_kernel"
require "./support/wan_shared_panel_kernel"
require "../spec/support/video_allocation_counter"

private WTP_DIN             =  5120
private WTP_DOUT            = 13824
private WTP_SOURCE_ROWS     =     4
private WTP_ROWS            =  4480
private WTP_TILE            =  1024
private WTP_WEIGHTS         = WTP_DIN * WTP_DOUT
private WTP_OUTPUT_ELEMENTS = WTP_ROWS * WTP_DOUT
private WTP_REFERENCE_ELEMS = WTP_SOURCE_ROWS * WTP_DOUT
private WTP_ARM             = "positional_tile1024_t050_mode0"
private WTP_FIXTURE_SHA     = "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703"
private WTP_REFERENCE_SHA   = "d77557f7d01798f91bbc3ad765f0147e33c0702e4cdb735f849b9641a0aaee3f"
private WTP_FIXTURE_DIR     = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022/fixture-4480-v3"
private WTP_REFERENCE_DIR   = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-positional-20261022/fixture-v1"
private WTP_ABSOLUTE_TOL    = 2.0e-5
private WTP_RELATIVE_TOL    = 2.0e-5
private WTP_DENSE_CAP       = 2_i64 * 1024**3

mode = nil.as(String?)
tile = "32x32"
backend = "tensorops"
OptionParser.parse do |parser|
  parser.banner = "Usage: wan_tensorops_probe (--small | --dense | --qbit) [--tile MxN] [--backend tensorops|shared-panel]"
  parser.on("--backend BACKEND", "Opt-in GEMM backend: tensorops or shared-panel") { |value| backend = value }
  parser.on("--tile TILE", "Probe tile: 32x32, 64x32, 32x64 or 64x64") { |value| tile = value }
  parser.on("--small", "Run GPU-generated small-shape qualification") do
    raise ArgumentError.new("Choose exactly one mode") if mode
    mode = "small"
  end
  parser.on("--dense", "Run pinned dense TensorOps/MPS qualification and timing") do
    raise ArgumentError.new("Choose exactly one mode") if mode
    mode = "dense"
  end
  parser.on("--qbit", "Run pinned no-stage QBit TensorOps/MPS qualification and timing") do
    raise ArgumentError.new("Choose exactly one mode") if mode
    mode = "qbit"
  end
end
raise ArgumentError.new("Unknown GEMM backend #{backend}") unless {"tensorops", "shared-panel"}.includes?(backend)
raise ArgumentError.new("Unknown arguments: #{ARGV.join(' ')}") unless ARGV.empty?
raise ArgumentError.new("Select exactly one of --small, --dense or --qbit") unless mode
unless {"32x32", "64x32", "32x64", "64x64"}.includes?(tile)
  raise ArgumentError.new("Unadmitted TensorOps tile #{tile}")
end
if backend == "shared-panel" && !{"32x32", "64x32"}.includes?(tile)
  raise ArgumentError.new("Unadmitted shared-panel tile #{tile}")
end
tile_m, tile_n = tile.split('x').map(&.to_i)

private WTP_METAL = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  #pragma clang fp contract(off)

  kernel void wtp_zero(device uint *values [[buffer(0)]], constant uint &n [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) values[i] = 0u;
  }

  kernel void wtp_poison(device float *values [[buffer(0)]], constant uint &n [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) values[i] = as_type<float>(0x7fc00000u);
  }

  kernel void wtp_seed_small(device float *input [[buffer(0)]],
      device float *weight [[buffer(1)]], device float *bias [[buffer(2)]],
      constant uint4 &shape [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    const uint M = shape.x, K = shape.y, N = shape.z;
    const bool one_hot = (shape.w & 0x80000000u) != 0u;
    const bool heterogeneous = (shape.w & 0x40000000u) != 0u;
    const bool preserve_weight_bias = (shape.w & 0x20000000u) != 0u;
    const uint salt = shape.w & 0x1fffffffu;
    if (i < M * K) {
      const uint row = i / K, k = i % K;
      float value;
      if (heterogeneous) {
        uint h = (row + 1u) * 0x9e3779b9u ^ (k + salt) * 0x85ebca6bu;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
        value = float(int(h % 2001u) - 1000) * (1.0f / 4093.0f);
      } else if (one_hot) {
        const uint selected_k = (row * 7u + 3u) % K;
        value = k == selected_k ? 1.0f : 0.0f;
      } else if (k == 0u) value = 512.0f + float(row) * 0.5f;
      else if (k == 1u) value = -511.5f - float(row) * 0.25f;
      else if (k == K - 1u && (K & 1u) != 0u)
        value = (row & 1u) != 0u ? -0.375f : 0.25f;
      else {
        const uint pair = k >> 1u;
        const uint magnitude = 1u + ((pair * 5u + row * 3u + salt) % 7u);
        value = float(magnitude) * (1.0f / 32.0f) * ((k & 1u) == 0u ? 1.0f : -1.0f);
      }
      input[i] = value;
    }
    if (!preserve_weight_bias && i < N * K) {
      const uint n = i / K, k = i % K, pair = k >> 1u;
      float value;
      if (heterogeneous) {
        uint h = (n + 7u) * 0xc2b2ae35u ^ (k + salt) * 0x27d4eb2fu;
        h ^= h >> 16; h *= 0x846ca68bu; h ^= h >> 15;
        value = float(int(h % 2001u) - 1000) * (1.0f / 4093.0f);
      } else if (one_hot) {
        const int code = int((n * 17u + k * 5u + salt) % 29u) - 14;
        value = float(code) * (1.0f / 64.0f);
      } else if (k < 2u) value = float(1u + (n % 3u)) * 0.125f;
      else if (k == K - 1u && (K & 1u) != 0u)
        value = float(1u + (n % 5u)) * 0.0625f;
      else {
        const uint magnitude = 1u + ((n * 3u + pair * 7u + salt) % 11u);
        value = float(magnitude) * (1.0f / 64.0f) * ((n & 1u) == 0u ? 1.0f : -1.0f);
      }
      weight[i] = value;
    }
    if (!preserve_weight_bias && i < N) {
      const int code = one_hot ? int((i * 11u + salt) % 13u) - 6 : int(i % 9u) - 4;
      bias[i] = float(code) * (1.0f / 32.0f);
    }
  }

  kernel void wtp_repeat(device const float *source [[buffer(0)]], device float *out [[buffer(1)]],
      constant uint &period [[buffer(2)]], constant uint &n [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = source[i % period];
  }

  // Independent scalar reference: sequential F32 fma in increasing K order,
  // followed by one separately rounded F32 bias addition.
  kernel void wtp_sequential_fma(device const float *input [[buffer(0)]],
      device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]],
      device float *output [[buffer(3)]], constant uint4 &shape [[buffer(4)]],
      uint i [[thread_position_in_grid]]) {
    const uint M = shape.x, K = shape.y, N = shape.z;
    if (i >= M * N) return;
    const uint row = i / N, n = i % N;
    float sum = 0.0f;
    for (uint k = 0u; k < K; ++k)
      sum = fma(input[ulong(row) * K + k], weight[ulong(n) * K + k], sum);
    output[i] = sum + bias[n];
  }

  kernel void wtp_seed_control(device float *actual [[buffer(0)]],
      device float *expected [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < 2u) {
      actual[i] = i == 0u ? 1.0f : as_type<float>(0x7fc00000u);
      expected[i] = 0.0f;
    }
  }

  // Separate tiny fixture for fail-closed controls; never mutates pinned data.
  // Every dyadic seed coefficient is exactly BF16-representable.
  kernel void wtp_seed_qbit_control(device const float *weight [[buffer(0)]],
      device uchar *blob [[buffer(1)]], device uint *offsets [[buffer(2)]],
      device uint *status [[buffer(3)]], constant uint &test [[buffer(4)]],
      uint i [[thread_position_in_grid]]) {
    constexpr uint K = 1024u, N = 3u, RECORD_BYTES = 2052u;
    if (i < K * N) {
      uint n = i / K, k = i % K, base = n * RECORD_BYTES;
      uint bits = as_type<uint>(weight[i]) >> 16;
      if (test == 3u && n == 0u && k == 0u) bits = 0x7fc0u;
      blob[base + 4u + k * 2u] = uchar(bits & 255u);
      blob[base + 5u + k * 2u] = uchar(bits >> 8);
      if (k == 0u) {
        blob[base] = 16u;
        blob[base + 1u] = test == 2u && n == 0u ? 1u : 0u;
        blob[base + 2u] = 0u;
        blob[base + 3u] = 0u;
      }
    }
    if (i <= N)
      offsets[i] = i * RECORD_BYTES + (test == 1u && i == N ? 4u : 0u);
    if (i < 8u) status[i] = test == 4u && i == 6u ? 1u : 0u;
  }

  kernel void wtp_snapshot_status(device const uint *status [[buffer(0)]],
      device uint *snapshots [[buffer(1)]], constant uint &test [[buffer(2)]],
      uint i [[thread_position_in_grid]]) {
    if (i < 8u) snapshots[test * 8u + i] = status[i];
  }

  // Stress-only raw records. Round each seeded coefficient down to its exact
  // BF16 expansion and retain that same F32 coefficient for both GPU controls.
  kernel void wtp_pack_raw_stress(device float *weight [[buffer(0)]],
      device uchar *blob [[buffer(1)]], device uint *offsets [[buffer(2)]],
      device uint *status [[buffer(3)]], constant uint4 &shape [[buffer(4)]],
      uint i [[thread_position_in_grid]]) {
    const uint K = shape.y, N = shape.z, record_bytes = 2052u;
    const uint tiles = N * (K / 1024u);
    if (i < N * K) {
      uint bits = as_type<uint>(weight[i]) & 0xffff0000u;
      weight[i] = as_type<float>(bits);
      uint within = i % 1024u, base = (i / 1024u) * record_bytes;
      blob[base + 4u + within * 2u] = uchar((bits >> 16) & 255u);
      blob[base + 5u + within * 2u] = uchar(bits >> 24);
      if (within == 0u) {
        blob[base] = 16u; blob[base + 1u] = 0u;
        blob[base + 2u] = 0u; blob[base + 3u] = 0u;
      }
    }
    if (i <= tiles) offsets[i] = i * record_bytes;
    if (i < 8u) status[i] = 0u;
  }

  kernel void wtp_compare(device const float *actual [[buffer(0)]],
      device const float *expected [[buffer(1)]], device atomic_uint *metrics [[buffer(2)]],
      constant uint &slot [[buffer(3)]], constant uint &n [[buffer(4)]],
      constant uint &period [[buffer(5)]], uint i [[thread_position_in_grid]],
      uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint4 counts[256];
    threadgroup float2 maxima[256];
    uint4 c = uint4(0u);
    float2 m = float2(0.0f);
    if (i < n) {
      const float a = actual[i], b = expected[i % period];
      c.x = uint(as_type<uint>(a) != as_type<uint>(b));
      if (!isfinite(a) || !isfinite(b)) {
        c.y = 1u;
        c.z = 1u;
      } else {
        const float d = abs(a - b);
        const float r = d / (2.0e-5f + 2.0e-5f * abs(b));
        c.y = uint(r > 1.0f);
        m = float2(d, r);
      }
    }
    counts[t] = c;
    maxima[t] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (t < stride) {
        counts[t] += counts[t + stride];
        maxima[t] = max(maxima[t], maxima[t + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
      const uint base = slot * 5u;
      atomic_fetch_add_explicit(metrics + base, counts[0].x, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + base + 1u, counts[0].y, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + base + 2u, counts[0].z, memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + base + 3u, as_type<uint>(maxima[0].x), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + base + 4u, as_type<uint>(maxima[0].y), memory_order_relaxed);
    }
  }
METAL

private def wtp_pipeline(name : String) : ML::Metal::ComputePipeline
  ML::Metal::ComputePipeline.new(name, WTP_METAL)
end

private def wtp_zero(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                     buffer : ML::MetalBuffer, n : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    .set_value(n.to_u32, 1).dispatch_1d(n)
end

private def wtp_poison(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                       buffer : ML::MetalBuffer, n : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    .set_value(n.to_u32, 1).dispatch_1d(n)
end

private def wtp_seed_small(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                           input : ML::MetalBuffer, weight : ML::MetalBuffer, bias : ML::MetalBuffer,
                           rows : Int32, din : Int32, dout : Int32, salt : UInt32) : Nil
  shape = StaticArray[rows.to_u32, din.to_u32, dout.to_u32, salt]
  e.set_pipeline(pipeline).set_buffer(input, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(weight, 1, ML::Metal::BufferAccess::Write)
    .set_buffer(bias, 2, ML::Metal::BufferAccess::Write)
    .set_bytes(shape.to_unsafe.as(Void*), 16, 3)
    .dispatch_1d(Math.max(rows * din, din * dout))
end

private def wtp_repeat(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                       source : ML::MetalBuffer, output : ML::MetalBuffer,
                       period : Int32, n : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(source, 0)
    .set_buffer(output, 1, ML::Metal::BufferAccess::Write)
    .set_value(period.to_u32, 2).set_value(n.to_u32, 3).dispatch_1d(n)
end

private def wtp_sequential_fma(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                               input : ML::MetalBuffer, weight : ML::MetalBuffer,
                               bias : ML::MetalBuffer, output : ML::MetalBuffer,
                               rows : Int32, din : Int32, dout : Int32) : Nil
  shape = StaticArray[rows.to_u32, din.to_u32, dout.to_u32, 0_u32]
  e.set_pipeline(pipeline).set_buffer(input, 0).set_buffer(weight, 1).set_buffer(bias, 2)
    .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
    .set_bytes(shape.to_unsafe.as(Void*), 16, 4).dispatch_1d(rows * dout)
end

private def wtp_seed_control(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                             actual : ML::MetalBuffer, expected : ML::MetalBuffer) : Nil
  e.set_pipeline(pipeline).set_buffer(actual, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(expected, 1, ML::Metal::BufferAccess::Write).dispatch_1d(2)
end

private def wtp_compare(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                        actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                        metrics : ML::MetalBuffer, slot : Int32, n : Int32, period : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(slot.to_u32, 3).set_value(n.to_u32, 4).set_value(period.to_u32, 5)
    .dispatch_threadgroups({(n + 255) // 256, 1, 1}, {256, 1, 1})
end

private def wtp_metric(buffer : ML::MetalBuffer, slot : Int32)
  p = buffer.contents.as(UInt32*) + slot * 5
  {bit_differences: p[0], tolerance_violations: p[1], nonfinite: p[2],
   max_abs: p[3].unsafe_as(Float32), max_ratio: p[4].unsafe_as(Float32)}
end

private def wtp_metric_json(buffer : ML::MetalBuffer, slot : Int32, elements : Int32) : JSON::Any
  metric = wtp_metric(buffer, slot)
  JSON.parse({compared_elements: elements, bit_differences: metric[:bit_differences],
              tolerance_violations: metric[:tolerance_violations], nonfinite: metric[:nonfinite],
              max_abs: metric[:max_abs], max_ratio: metric[:max_ratio],
              absolute_tolerance: WTP_ABSOLUTE_TOL, relative_tolerance: WTP_RELATIVE_TOL}.to_json)
end

private def wtp_tolerance_ok(metric) : Bool
  metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
end

private def wtp_positive_ok(metric) : Bool
  metric[:bit_differences] == 2 && metric[:tolerance_violations] == 2 && metric[:nonfinite] == 1
end

private def wtp_upload(path : String, buffer : ML::MetalBuffer, bytes : Int64) : Nil
  raise "Upload byte count mismatch for #{path}" unless File.size(path) == bytes && buffer.size >= bytes
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), bytes.to_i32)) }
end

# Caller has already verified the complete pinned fixture and its SHA-256.
private def wtp_upload_prefix(path : String, buffer : ML::MetalBuffer, bytes : Int64) : Nil
  raise "Invalid fixture prefix extent" unless bytes > 0 && File.size(path) >= bytes && buffer.size >= bytes
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), bytes.to_i32)) }
end

private def wtp_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

private def wtp_verify(directory : String, record : JSON::Any,
                       dtype : String, shape : Array(Int32)) : String
  path = record["path"].as_s
  raise "Unsafe fixture path: #{path}" unless !path.empty? && File.basename(path) == path
  raise "Fixture dtype mismatch for #{path}" unless record["dtype"].as_s == dtype
  actual_shape = record["shape"].as_a.map(&.as_i)
  raise "Fixture shape mismatch for #{path}: #{actual_shape}" unless actual_shape == shape
  element_bytes = case dtype
                  when "U8", "QBitBlockBytes" then 1
                  when "BF16"                 then 2
                  when "F32", "U32LE"         then 4
                  else                             raise "Unexpected fixture dtype #{dtype}"
                  end
  expected_bytes = shape.product.to_i64 * element_bytes
  raise "Fixture byte record mismatch for #{path}" unless record["bytes"].as_i64 == expected_bytes
  full = File.join(directory, path)
  unless File.size(full) == expected_bytes && wtp_sha256(full) == record["sha256"].as_s
    raise "Fixture file hash mismatch: #{path}"
  end
  full
end

private def wtp_validate_dense_fixture : NamedTuple(
  fixture_hash: String, reference_hash: String, input_path: String,
  bias_path: String, weight_path: String, reference_path: String,
  arm_blob_sha256: String, blob_path: String, blob_bytes: Int32,
  offsets_path: String, offset_count: Int32, lut4_path: String, lut5_path: String)
  fixture_manifest_path = File.join(WTP_FIXTURE_DIR, "manifest.json")
  reference_manifest_path = File.join(WTP_REFERENCE_DIR, "manifest.json")
  fixture_hash = wtp_sha256(fixture_manifest_path)
  reference_hash = wtp_sha256(reference_manifest_path)
  raise "Unexpected source manifest SHA-256" unless fixture_hash == WTP_FIXTURE_SHA
  raise "Unexpected positional manifest SHA-256" unless reference_hash == WTP_REFERENCE_SHA
  fixture = JSON.parse(File.read(fixture_manifest_path))
  reference = JSON.parse(File.read(reference_manifest_path))
  unless fixture["schema"].as_s == "wan_quantized_projection_reference_v1" &&
         reference["schema"].as_s == "wan_qbit_positional_weight_reference_v1" &&
         reference["source_manifest_sha256"].as_s == fixture_hash &&
         reference["source_hashes"]["source_fixture_manifest_sha256"].as_s == fixture_hash
    raise "Fixture schema or manifest lineage mismatch"
  end
  geometry = fixture["geometry"]
  ref_geometry = reference["geometry"]
  unless geometry["source_rows"].as_i == WTP_SOURCE_ROWS &&
         geometry["input_dim"].as_i == WTP_DIN && geometry["output_dim"].as_i == WTP_DOUT &&
         geometry["rows"].as_i == WTP_ROWS && ref_geometry["rows"].as_i == WTP_ROWS &&
         ref_geometry["source_rows"].as_i == WTP_SOURCE_ROWS &&
         ref_geometry["input_dim"].as_i == WTP_DIN && ref_geometry["output_dim"].as_i == WTP_DOUT &&
         ref_geometry["normalization_block_size"].as_i == 256
    raise "Fixture geometry mismatch"
  end
  provenance = fixture["input_provenance"]
  unless provenance["kind"].as_s == "upstream_expected_ffn_mod_seeded_capture" &&
         provenance["capture_seed"].as_i == 7031 && !provenance["real_video_derived"].as_bool
    raise "Input is not the pinned seeded, non-video-derived fixture"
  end
  input_scope = ref_geometry["input_scope"]
  unless input_scope["kind"].as_s == provenance["kind"].as_s &&
         input_scope["capture_seed"].as_i == provenance["capture_seed"].as_i &&
         input_scope["real_video_derived"].as_bool == provenance["real_video_derived"].as_bool
    raise "Positional input provenance differs from source fixture"
  end

  source_tensor = fixture["source_tensor"]
  input_path = wtp_verify(WTP_FIXTURE_DIR, fixture["input"], "F32", [WTP_SOURCE_ROWS, WTP_DIN])
  wtp_verify(WTP_FIXTURE_DIR, source_tensor["weight_file"], "BF16", [WTP_DOUT, WTP_DIN])
  wtp_verify(WTP_FIXTURE_DIR, source_tensor["bias_bf16_file"], "BF16", [WTP_DOUT])
  bias_path = wtp_verify(WTP_FIXTURE_DIR, source_tensor["bias_f32_file"], "F32", [WTP_DOUT])
  wtp_verify(WTP_FIXTURE_DIR, fixture["variants"]["bf16"]["decoded_weight"], "F32", [WTP_DOUT, WTP_DIN])

  source_hashes = reference["source_hashes"]
  expected_hashes = {
    "checkpoint_weight_bf16_sha256"  => source_tensor["weight_file"]["sha256"].as_s,
    "decoded_bf16_source_f32_sha256" => fixture["variants"]["bf16"]["decoded_weight"]["sha256"].as_s,
    "input_f32_sha256"               => fixture["input"]["sha256"].as_s,
    "bias_bf16_sha256"               => source_tensor["bias_bf16_file"]["sha256"].as_s,
    "bias_f32_sha256"                => source_tensor["bias_f32_file"]["sha256"].as_s,
  }
  expected_hashes.each do |name, digest|
    raise "Positional lineage mismatch for #{name}" unless source_hashes[name].as_s == digest
  end
  lut4_record = fixture["variants"]["qbit_p4"]["lut"]
  lut5_record = fixture["variants"]["qbit_p5"]["lut"]
  lut4_path = wtp_verify(WTP_FIXTURE_DIR, lut4_record, "F32", [256])
  lut5_path = wtp_verify(WTP_FIXTURE_DIR, lut5_record, "F32", [256])
  unless source_hashes["qbit_lut_p4_sha256"].as_s == lut4_record["sha256"].as_s &&
         source_hashes["qbit_lut_p5_sha256"].as_s == lut5_record["sha256"].as_s
    raise "Positional LUT lineage mismatch"
  end

  arm = reference["arms"].as_a.find { |candidate| candidate["name"].as_s == WTP_ARM } ||
        raise "Missing pinned positional arm #{WTP_ARM}"
  unless arm["tile_size"].as_i == WTP_TILE && arm["value_mode"].as_i == 0 &&
         arm["threshold_f32_hex"].as_s == "0x3f000000" &&
         arm["precision_tie_order"].as_a.map(&.as_i) == [16, 5, 4]
    raise "Unadmitted positional arm policy"
  end
  arm_provenance = arm["projection_provenance"]
  unless arm_provenance["rows_computed"].as_i == WTP_ROWS &&
         arm_provenance["rows_saved"].as_i == WTP_SOURCE_ROWS &&
         arm_provenance["all_periodic_repeats_bitexact"].as_bool
    raise "Pinned projection lacks four-row periodic-repeat provenance"
  end
  tiles = WTP_WEIGHTS // WTP_TILE
  wtp_verify(WTP_REFERENCE_DIR, arm["plan"], "U32LE", [tiles, 4])
  offsets_path = wtp_verify(WTP_REFERENCE_DIR, arm["offsets"], "U32LE", [tiles + 1])
  blob_bytes = arm["blob"]["bytes"].as_i
  blob_path = wtp_verify(WTP_REFERENCE_DIR, arm["blob"], "U8", [blob_bytes])
  weight_path = wtp_verify(WTP_REFERENCE_DIR, arm["decoded_weight"], "F32", [WTP_DOUT, WTP_DIN])
  reference_path = wtp_verify(WTP_REFERENCE_DIR, arm["projection"], "F32", [WTP_SOURCE_ROWS, WTP_DOUT])
  {
    fixture_hash: fixture_hash, reference_hash: reference_hash, input_path: input_path,
    bias_path: bias_path, weight_path: weight_path, reference_path: reference_path,
    arm_blob_sha256: arm["blob"]["sha256"].as_s,
    blob_path: blob_path, blob_bytes: blob_bytes, offsets_path: offsets_path,
    offset_count: tiles + 1, lut4_path: lut4_path, lut5_path: lut5_path,
  }
end

# Read only diagnostic status after the route's existing terminal completion.
# These eight words are not activation data and do not add a synchronization.
private def wtp_status!(kernel : WanTensorOpsKernel) : Array(UInt32)?
  if status = kernel.status_buffer
    words = Array.new(8) { |i| status.contents.as(UInt32*)[i] }
    raise "QBit GPU decode/status failure: #{words}" unless words.all?(&.zero?)
    words
  end
end

private def wtp_dense_kernel(backend : String, tile_m : Int32, tile_n : Int32) : WanTensorOpsKernel
  backend == "shared-panel" ? WanSharedPanelKernel.new(tile_m, tile_n) : WanTensorOpsKernel.new(tile_m, tile_n)
end

private def wtp_qbit_kernel(backend : String, tile_m : Int32, tile_n : Int32,
                            blob : ML::MetalBuffer, offsets : ML::MetalBuffer,
                            lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                            status : ML::MetalBuffer, blob_bytes : Int32,
                            offset_count : Int32) : WanTensorOpsKernel
  if backend == "shared-panel"
    WanSharedPanelQBitKernel.new(tile_m, tile_n, blob, offsets, lut4, lut5,
      status, blob_bytes, offset_count)
  else
    WanTensorOpsQBitKernel.new(tile_m, tile_n, blob, offsets, lut4, lut5,
      status, blob_bytes, offset_count)
  end
end

private def wtp_repeat_seed(e : ML::Metal::ComputeEncoder,
                            pipeline : ML::Metal::ComputePipeline,
                            source : ML::MetalBuffer, output : ML::MetalBuffer,
                            rows : Int32, din : Int32) : Nil
  wtp_repeat(e, pipeline, source, output, WTP_SOURCE_ROWS * din, rows * din)
end

private def wtp_compare_triplet(e : ML::Metal::ComputeEncoder,
                                compare : ML::Metal::ComputePipeline,
                                tensorops_output : ML::MetalBuffer,
                                mps_output : ML::MetalBuffer,
                                expected : ML::MetalBuffer,
                                metrics : ML::MetalBuffer,
                                elements : Int32, period : Int32) : Nil
  wtp_compare(e, compare, tensorops_output, expected, metrics, 0, elements, period)
  wtp_compare(e, compare, mps_output, expected, metrics, 1, elements, period)
  wtp_compare(e, compare, tensorops_output, mps_output, metrics, 2, elements, elements)
end

private def wtp_new_plan(rows : Int32, din : Int32, dout : Int32,
                         input : ML::MetalBuffer, weight : ML::MetalBuffer,
                         output : ML::MetalBuffer) : ML::Video::WanT5MPSLinearPlan
  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16,
    heads: 1, layers: 1, batches: 1, tokens: rows)
  ML::Video::WanT5MPSLinearPlan.new(cfg, din, dout, input, weight, output)
end

private def wtp_small_qualification(kernel : WanTensorOpsKernel,
                                    zero : ML::Metal::ComputePipeline,
                                    poison : ML::Metal::ComputePipeline,
                                    seed : ML::Metal::ComputePipeline,
                                    oracle : ML::Metal::ComputePipeline,
                                    compare : ML::Metal::ComputePipeline,
                                    seed_control : ML::Metal::ComputePipeline,
                                    rows : Int32, din : Int32, dout : Int32,
                                    salt : UInt32, raw_qbit_stress : Bool = false,
                                    qbit_fixture = nil) : JSON::Any
  arena = ML::Video::Arena.new
  plan : ML::Video::WanT5MPSLinearPlan? = nil
  begin
    input = arena.alloc("small_input", rows * din)
    weight = arena.alloc("small_weight", din * dout)
    bias = arena.alloc("small_bias", dout)
    tensorops_output = arena.alloc("small_tensorops_output", rows * dout)
    oracle_output = arena.alloc("small_sequential_fma_oracle", rows * dout)
    mps_output = arena.alloc("small_preexpanded_mps_output", rows * dout)
    metrics = arena.alloc("small_metrics", 20)
    control_actual = arena.alloc("small_control_actual", 2)
    control_expected = arena.alloc("small_control_expected", 2)
    active_kernel = kernel
    pack = nil.as(ML::Metal::ComputePipeline?)
    raw_blob = nil.as(ML::MetalBuffer?)
    raw_offsets = nil.as(ML::MetalBuffer?)
    raw_status = nil.as(ML::MetalBuffer?)
    if raw_qbit_stress
      raise "Raw QBit stress requires shared panels and whole records" unless kernel.route == "shared_panel_f32" && din % 1024 == 0
      records = dout * (din // 1024)
      raw_blob = arena.alloc("stress_raw_qbit_blob", records * 2052 // 4)
      raw_offsets = arena.alloc("stress_raw_qbit_offsets", records + 1)
      lut4 = arena.alloc("stress_raw_qbit_lut4", 256)
      lut5 = arena.alloc("stress_raw_qbit_lut5", 256)
      raw_status = arena.alloc("stress_raw_qbit_status", 8)
      active_kernel = wtp_qbit_kernel("shared-panel", kernel.tile_m, kernel.tile_n,
        raw_blob, raw_offsets, lut4, lut5, raw_status, records * 2052, records + 1)
      pack = wtp_pipeline("wtp_pack_raw_stress")
    end
    if fixture = qbit_fixture
      raise "Pinned QBit stress must use the original K and a channel prefix" unless din == WTP_DIN && dout <= WTP_DOUT && (salt & 0x20000000_u32) != 0
      blob = arena.alloc("stress_pinned_qbit_blob", (fixture[:blob_bytes] + 3) // 4)
      count = dout * (din // 1024) + 1
      offsets = arena.alloc("stress_pinned_qbit_offsets", count)
      lut4 = arena.alloc("stress_pinned_lut4", 256)
      lut5 = arena.alloc("stress_pinned_lut5", 256)
      raw_status = arena.alloc("stress_pinned_status", 8)
      wtp_upload(fixture[:blob_path], blob, fixture[:blob_bytes].to_i64)
      wtp_upload_prefix(fixture[:offsets_path], offsets, count.to_i64 * 4)
      wtp_upload(fixture[:lut4_path], lut4, 1024_i64)
      wtp_upload(fixture[:lut5_path], lut5, 1024_i64)
      wtp_upload_prefix(fixture[:weight_path], weight, din.to_i64 * dout * 4)
      wtp_upload_prefix(fixture[:bias_path], bias, dout.to_i64 * 4)
      active_kernel = wtp_qbit_kernel("shared-panel", kernel.tile_m, kernel.tile_n,
        blob, offsets, lut4, lut5, raw_status, fixture[:blob_bytes], count)
    end
    plan = wtp_new_plan(rows, din, dout, input, weight, mps_output)

    before = ML::MetalBuffer.video_spec_allocations
    command = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(command)
    wtp_zero(e, zero, metrics, 20)
    wtp_poison(e, poison, tensorops_output, rows * dout)
    wtp_poison(e, poison, oracle_output, rows * dout)
    wtp_poison(e, poison, mps_output, rows * dout)
    wtp_seed_small(e, seed, input, weight, bias, rows, din, dout, salt)
    if pack
      shape = StaticArray[rows.to_u32, din.to_u32, dout.to_u32, 1024_u32]
      e.set_pipeline(pack).set_buffer(weight, 0, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(raw_blob.not_nil!, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(raw_offsets.not_nil!, 2, ML::Metal::BufferAccess::Write)
        .set_buffer(raw_status.not_nil!, 3, ML::Metal::BufferAccess::Write)
        .set_bytes(shape.to_unsafe.as(Void*), 16, 4).dispatch_1d(din * dout)
    elsif raw_status
      wtp_zero(e, zero, raw_status, 8)
    end
    active_kernel.encode(e, input, weight, bias, tensorops_output, rows, din, dout)
    ML::Video::MetalLinearMPS.project_preexpanded(e, input, bias, mps_output,
      rows, din, dout, ML::Video::WeightDType::F32, plan.not_nil!, weight)
    wtp_sequential_fma(e, oracle, input, weight, bias, oracle_output, rows, din, dout)
    wtp_compare(e, compare, tensorops_output, oracle_output, metrics, 0, rows * dout, rows * dout)
    wtp_compare(e, compare, mps_output, oracle_output, metrics, 1, rows * dout, rows * dout)
    wtp_compare(e, compare, tensorops_output, mps_output, metrics, 2, rows * dout, rows * dout)
    wtp_seed_control(e, seed_control, control_actual, control_expected)
    wtp_compare(e, compare, control_actual, control_expected, metrics, 3, 2, 2)
    e.end_encoding
    command.commit
    command.wait
    gpu_status = wtp_status!(active_kernel)
    allocations = ML::MetalBuffer.video_spec_allocations - before
    raise "Small qualification encode allocated tracked MetalBuffers" unless allocations == 0

    tensorops_metric = wtp_metric(metrics, 0)
    mps_metric = wtp_metric(metrics, 1)
    pair_metric = wtp_metric(metrics, 2)
    positive_metric = wtp_metric(metrics, 3)
    passed = wtp_tolerance_ok(tensorops_metric) && wtp_tolerance_ok(mps_metric) &&
             wtp_tolerance_ok(pair_metric) && wtp_positive_ok(positive_metric)
    one_hot = (salt & 0x80000000_u32) != 0
    heterogeneous = (salt & 0x40000000_u32) != 0
    result = JSON.parse({schema: "wan_tensorops_small_qualification_v1",
                         candidate_route: active_kernel.route, qbit_gpu_status: gpu_status,
                         qbit_raw_record_stress: raw_qbit_stress,
                         qbit_pinned_record_tail_stress: !qbit_fixture.nil?,
                         qbit_pinned_arm: qbit_fixture.nil? ? nil : WTP_ARM,
                         rows: rows, input_dim: din, output_dim: dout, input_seed_salt: salt,
                         gpu_seed_source_sha256: Digest::SHA256.hexdigest(WTP_METAL),
                         input_pattern: heterogeneous ? "independent uint-mixed row/K and N/K values, non-dyadic F32 scale 1/4093, nonzero bias" : (one_hot ? "one_hot_k=(row*7+3)%K with asymmetric dyadic row-major weight and nonzero bias" : "near-cancelling first pair, exact cancelling pairs, dyadic unmatched odd-K tail"),
                         cancellation_pair: one_hot || heterogeneous ? "not_applicable" : "x[0]*w[0] and x[1]*w[1] nearly cancel; remaining pairs cancel exactly",
                         odd_input_tail: din.odd?, oracle: "sequential F32 fma in increasing K, then F32 bias add",
                         tensorops_vs_oracle: wtp_metric_json(metrics, 0, rows * dout),
                         preexpanded_mps_vs_oracle: wtp_metric_json(metrics, 1, rows * dout),
                         tensorops_vs_mps: wtp_metric_json(metrics, 2, rows * dout),
                         comparator_positive_control: wtp_metric_json(metrics, 3, 2),
                         comparator_positive_control_passed: wtp_positive_ok(positive_metric),
                         qualification_passed: passed, encode_metal_buffer_allocations: allocations,
                         command_buffers: 1, encoders: 1, terminal_waits: 1,
                         inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
    puts(result.to_json)
    STDOUT.flush
    raise "Small TensorOps numerical qualification rejected for #{rows}x#{din}x#{dout}" unless passed
    result
  ensure
    plan.try(&.release)
    arena.release
  end
end

private def wtp_dense_qualification(kernel : WanTensorOpsKernel,
                                    zero : ML::Metal::ComputePipeline,
                                    poison : ML::Metal::ComputePipeline,
                                    repeat : ML::Metal::ComputePipeline,
                                    compare : ML::Metal::ComputePipeline,
                                    seed_control : ML::Metal::ComputePipeline,
                                    source : ML::MetalBuffer, input : ML::MetalBuffer,
                                    weight : ML::MetalBuffer, bias : ML::MetalBuffer,
                                    expected : ML::MetalBuffer,
                                    tensorops_output : ML::MetalBuffer,
                                    mps_output : ML::MetalBuffer,
                                    metrics : ML::MetalBuffer,
                                    control_actual : ML::MetalBuffer,
                                    control_expected : ML::MetalBuffer,
                                    plan : ML::Video::WanT5MPSLinearPlan) : JSON::Any
  before = ML::MetalBuffer.video_spec_allocations
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  wtp_zero(e, zero, metrics, 20)
  if status = kernel.status_buffer
    wtp_zero(e, zero, status, 8)
  end
  wtp_poison(e, poison, tensorops_output, WTP_OUTPUT_ELEMENTS)
  wtp_poison(e, poison, mps_output, WTP_OUTPUT_ELEMENTS)
  wtp_repeat_seed(e, repeat, source, input, WTP_ROWS, WTP_DIN)
  kernel.encode(e, input, weight, bias, tensorops_output, WTP_ROWS, WTP_DIN, WTP_DOUT)
  ML::Video::MetalLinearMPS.project_preexpanded(e, input, bias, mps_output,
    WTP_ROWS, WTP_DIN, WTP_DOUT, ML::Video::WeightDType::F32, plan, weight)
  wtp_compare_triplet(e, compare, tensorops_output, mps_output, expected,
    metrics, WTP_OUTPUT_ELEMENTS, WTP_REFERENCE_ELEMS)
  wtp_seed_control(e, seed_control, control_actual, control_expected)
  wtp_compare(e, compare, control_actual, control_expected, metrics, 3, 2, 2)
  e.end_encoding
  command.commit
  command.wait
  gpu_status = wtp_status!(kernel)
  allocations = ML::MetalBuffer.video_spec_allocations - before
  raise "Dense qualification encode allocated tracked MetalBuffers" unless allocations == 0

  tensorops_metric = wtp_metric(metrics, 0)
  mps_metric = wtp_metric(metrics, 1)
  pair_metric = wtp_metric(metrics, 2)
  positive_metric = wtp_metric(metrics, 3)
  passed = wtp_tolerance_ok(tensorops_metric) && wtp_tolerance_ok(mps_metric) &&
           wtp_tolerance_ok(pair_metric) && wtp_positive_ok(positive_metric)
  result = JSON.parse({schema: "wan_tensorops_dense_qualification_v1", arm: WTP_ARM,
                       candidate_route: kernel.route, qbit_gpu_status: gpu_status,
                       rows: WTP_ROWS, input_dim: WTP_DIN, output_dim: WTP_DOUT,
                       compared_outputs: WTP_OUTPUT_ELEMENTS, upstream_period_elements: WTP_REFERENCE_ELEMS,
                       tensorops_vs_upstream: wtp_metric_json(metrics, 0, WTP_OUTPUT_ELEMENTS),
                       preexpanded_mps_vs_upstream: wtp_metric_json(metrics, 1, WTP_OUTPUT_ELEMENTS),
                       tensorops_vs_preexpanded_mps: wtp_metric_json(metrics, 2, WTP_OUTPUT_ELEMENTS),
                       comparator_positive_control: wtp_metric_json(metrics, 3, 2),
                       comparator_positive_control_passed: wtp_positive_ok(positive_metric),
                       qualification_passed: passed, encode_metal_buffer_allocations: allocations,
                       command_buffers: 1, encoders: 1, terminal_waits: 1,
                       inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
  puts(result.to_json)
  STDOUT.flush
  raise "Dense TensorOps numerical qualification rejected; timing suppressed" unless passed
  result
end

private def wtp_dense_timed(route : String, kernel : WanTensorOpsKernel,
                            input : ML::MetalBuffer, weight : ML::MetalBuffer,
                            bias : ML::MetalBuffer, tensorops_output : ML::MetalBuffer,
                            mps_output : ML::MetalBuffer,
                            plan : ML::Video::WanT5MPSLinearPlan) : NamedTuple(
  encode_seconds: Float64, gpu_seconds: Float64, wall_seconds: Float64,
  terminal_waits: Int32, inter_operator_host_waits: Int32,
  device_activation_readbacks: Int32, encode_metal_buffer_allocations: Int64)
  before = ML::MetalBuffer.video_spec_allocations
  started = Time.instant
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  case route
  when "tensorops_f32", "tensorops_qbit_positional", "shared_panel_f32", "shared_panel_qbit_positional"
    kernel.encode(e, input, weight, bias, tensorops_output,
      WTP_ROWS, WTP_DIN, WTP_DOUT)
  when "preexpanded_mps_f32"
    ML::Video::MetalLinearMPS.project_preexpanded(e, input, bias, mps_output,
      WTP_ROWS, WTP_DIN, WTP_DOUT, ML::Video::WeightDType::F32, plan, weight)
  else
    raise "Unknown dense timing route #{route}"
  end
  e.end_encoding
  encode_seconds = (Time.instant - started).total_seconds
  gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
  wall_seconds = (Time.instant - started).total_seconds
  wtp_status!(kernel)
  allocations = ML::MetalBuffer.video_spec_allocations - before
  raise "Timed route allocated tracked MetalBuffers" unless allocations == 0
  {encode_seconds: encode_seconds, gpu_seconds: gpu_seconds, wall_seconds: wall_seconds,
   terminal_waits: 1, inter_operator_host_waits: 0, device_activation_readbacks: 0,
   encode_metal_buffer_allocations: allocations}
end

private def wtp_run_small(kernel : WanTensorOpsKernel) : Nil
  ML::Video::MetalLinearMPS.prepare!
  zero = wtp_pipeline("wtp_zero")
  poison = wtp_pipeline("wtp_poison")
  seed = wtp_pipeline("wtp_seed_small")
  oracle = wtp_pipeline("wtp_sequential_fma")
  compare = wtp_pipeline("wtp_compare")
  seed_control = wtp_pipeline("wtp_seed_control")
  shapes = [{32, 64, 32, 11_u32}, {35, 73, 37, 23_u32}, {17, 33, 9, 37_u32}]
  if kernel.route == "shared_panel_f32"
    shapes << {65, 1031, 49, 0x40000035_u32}
    shapes << {97, 257, 65, 0x40000061_u32}
  end
  results = shapes.map do |shape|
    wtp_small_qualification(kernel, zero, poison, seed, oracle, compare, seed_control,
      shape[0], shape[1], shape[2], shape[3])
  end
  one_hot = wtp_small_qualification(kernel, zero, poison, seed, oracle, compare, seed_control,
    35, 73, 37, 0x80000017_u32)
  results << one_hot
  puts({schema: "wan_tensorops_small_complete_v1", shapes: shapes.map { |s| [s[0], s[1], s[2]] },
        extra_one_hot_coordinate_case: {rows: 35, input_dim: 73, output_dim: 37},
        qualification_records: results.size, all_passed: results.all? { |r| r["qualification_passed"].as_bool },
        full_model_loaded: false, generation: false}.to_json)
end

# One command/terminal completion for a finite raw-BF16 control and malformed
# offset, mode, decoded nonfinite, and incoming-capacity failures. Output starts
# finite in failure cases so unwritten NaNs cannot masquerade as poisoning.
private def wtp_qbit_failure_controls(backend : String, tile_m : Int32, tile_n : Int32, fixture) : Nil
  rows, din, dout = 35, 1024, 3
  elements = rows * dout
  zero = wtp_pipeline("wtp_zero")
  seed = wtp_pipeline("wtp_seed_small")
  seed_qbit = wtp_pipeline("wtp_seed_qbit_control")
  snapshot = wtp_pipeline("wtp_snapshot_status")
  oracle = wtp_pipeline("wtp_sequential_fma")
  compare = wtp_pipeline("wtp_compare")
  if backend == "shared-panel"
    wtp_small_qualification(wtp_dense_kernel(backend, tile_m, tile_n), zero,
      wtp_pipeline("wtp_poison"), seed, oracle, compare, wtp_pipeline("wtp_seed_control"),
      65, 2048, 35, 0x40000079_u32, true)
    wtp_small_qualification(wtp_dense_kernel(backend, tile_m, tile_n), zero,
      wtp_pipeline("wtp_poison"), seed, oracle, compare, wtp_pipeline("wtp_seed_control"),
      65, WTP_DIN, 35, 0x60000083_u32, false, fixture)
  end
  arena = ML::Video::Arena.new
  begin
    input = arena.alloc("control_input", rows * din)
    weight = arena.alloc("control_weight", din * dout)
    bias = arena.alloc("control_bias", dout)
    output = arena.alloc("control_output", elements)
    expected = arena.alloc("control_expected", elements)
    blob_bytes = dout * 2052
    blob = arena.alloc("control_blob", blob_bytes // 4)
    offsets = arena.alloc("control_offsets", dout + 1)
    lut4 = arena.alloc("control_lut4", 256)
    lut5 = arena.alloc("control_lut5", 256)
    status = arena.alloc("control_status", 8)
    snapshots = arena.alloc("control_status_snapshots", 5 * 8)
    metrics = arena.alloc("control_metrics", 5 * 5)
    kernel = wtp_qbit_kernel(backend, tile_m, tile_n, blob, offsets, lut4, lut5,
      status, blob_bytes, dout + 1)
    before = ML::MetalBuffer.video_spec_allocations
    command = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(command)
    wtp_zero(e, zero, metrics, 25)
    wtp_zero(e, zero, lut4, 256)
    wtp_zero(e, zero, lut5, 256)
    wtp_seed_small(e, seed, input, weight, bias, rows, din, dout, 0x80000017_u32)
    wtp_sequential_fma(e, oracle, input, weight, bias, expected, rows, din, dout)
    5.times do |test|
      wtp_zero(e, zero, output, elements)
      e.set_pipeline(seed_qbit).set_buffer(weight, 0)
        .set_buffer(blob, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(offsets, 2, ML::Metal::BufferAccess::Write)
        .set_buffer(status, 3, ML::Metal::BufferAccess::Write)
        .set_value(test.to_u32, 4).dispatch_1d(din * dout)
      kernel.encode(e, input, weight, bias, output, rows, din, dout)
      wtp_compare(e, compare, output, expected, metrics, test, elements, elements)
      e.set_pipeline(snapshot).set_buffer(status, 0)
        .set_buffer(snapshots, 1, ML::Metal::BufferAccess::Write)
        .set_value(test.to_u32, 2).dispatch_1d(8)
    end
    e.end_encoding
    command.commit
    command.wait
    allocations = ML::MetalBuffer.video_spec_allocations - before
    raise "QBit control encoding allocated tracked MetalBuffers" unless allocations == 0
    names = ["finite_raw_bf16", "out_of_blob_offset", "unsupported_mode", "nonfinite_raw_value", "incoming_capacity_failure"]
    5.times do |test|
      words = Array.new(8) { |i| snapshots.contents.as(UInt32*)[test * 8 + i] }
      metric = wtp_metric(metrics, test)
      status_ok = if test == 0
                    words.all?(&.zero?)
                  elsif test == 4
                    words[6] == 1 && words.each_with_index.all? { |v, i| i == 6 || v == 0 }
                  else
                    words[7] > 0 && words.first(7).all?(&.zero?)
                  end
      output_ok = test == 0 ? (wtp_tolerance_ok(metric) && metric[:bit_differences] == 0) : (metric[:tolerance_violations] == elements && metric[:nonfinite] == elements)
      passed = status_ok && output_ok
      puts({schema: "wan_tensorops_qbit_failure_control_v1", test: names[test],
            qbit_gpu_status: words, output: wtp_metric_json(metrics, test, elements),
            passed: passed, encode_metal_buffer_allocations: allocations,
            batch_command_buffers: 1, batch_terminal_waits: 1,
            inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
      STDOUT.flush
      raise "QBit failure control rejected: #{names[test]}" unless passed
    end
  ensure
    arena.release
  end
end

private def wtp_run_dense(backend : String, tile_m : Int32, tile_n : Int32, fixture, qbit : Bool = false) : Nil
  ML::Video::MetalLinearMPS.prepare!
  zero = wtp_pipeline("wtp_zero")
  poison = wtp_pipeline("wtp_poison")
  repeat = wtp_pipeline("wtp_repeat")
  compare = wtp_pipeline("wtp_compare")
  seed_control = wtp_pipeline("wtp_seed_control")

  arena = ML::Video::Arena.new
  plan : ML::Video::WanT5MPSLinearPlan? = nil
  begin
    source = arena.alloc("four_seeded_source_rows", WTP_SOURCE_ROWS * WTP_DIN)
    input = arena.alloc("gpu_repeated_input", WTP_ROWS * WTP_DIN)
    weight = arena.alloc("pinned_t050_decoded_f32_weight", WTP_WEIGHTS)
    bias = arena.alloc("pinned_f32_bias", WTP_DOUT)
    expected = arena.alloc("pinned_upstream_four_row_projection", WTP_REFERENCE_ELEMS)
    tensorops_output = arena.alloc("tensorops_full_output", WTP_OUTPUT_ELEMENTS)
    mps_output = arena.alloc("mps_full_output", WTP_OUTPUT_ELEMENTS)
    metrics = arena.alloc("comparison_metrics", 35)
    control_actual = arena.alloc("comparator_control_actual", 2)
    control_expected = arena.alloc("comparator_control_expected", 2)
    wtp_upload(fixture[:input_path], source, WTP_SOURCE_ROWS.to_i64 * WTP_DIN * 4)
    wtp_upload(fixture[:weight_path], weight, WTP_WEIGHTS.to_i64 * 4)
    wtp_upload(fixture[:bias_path], bias, WTP_DOUT.to_i64 * 4)
    wtp_upload(fixture[:reference_path], expected, WTP_REFERENCE_ELEMS.to_i64 * 4)
    packed_bytes = 0_i64
    kernel = if qbit
               blob = arena.alloc("pinned_qbit_blob", (fixture[:blob_bytes] + 3) // 4)
               offsets = arena.alloc("pinned_qbit_offsets", fixture[:offset_count])
               lut4 = arena.alloc("pinned_qbit_p4_lut", 256)
               lut5 = arena.alloc("pinned_qbit_p5_lut", 256)
               status = arena.alloc("qbit_sticky_status", 8)
               wtp_upload(fixture[:blob_path], blob, fixture[:blob_bytes].to_i64)
               wtp_upload(fixture[:offsets_path], offsets, fixture[:offset_count].to_i64 * 4)
               wtp_upload(fixture[:lut4_path], lut4, 1024_i64)
               wtp_upload(fixture[:lut5_path], lut5, 1024_i64)
               packed_bytes = fixture[:blob_bytes].to_i64 + fixture[:offset_count].to_i64 * 4 + 2048
               wtp_qbit_kernel(backend, tile_m, tile_n, blob, offsets, lut4, lut5, status,
                 fixture[:blob_bytes], fixture[:offset_count])
             else
               wtp_dense_kernel(backend, tile_m, tile_n)
             end
    wtp_describe_kernel(kernel)
    explicit_bytes = arena.resident_bytes
    raise "Explicit TensorOps buffers exceed the 2 GiB cap" unless explicit_bytes < WTP_DENSE_CAP
    plan = wtp_new_plan(WTP_ROWS, WTP_DIN, WTP_DOUT, input, weight, mps_output)

    puts({schema: "wan_tensorops_dense_setup_v1", mode: qbit ? "qbit" : "dense",
          candidate_route: kernel.route,
          source_manifest_sha256: fixture[:fixture_hash],
          positional_manifest_sha256: fixture[:reference_hash],
          selected_arm: WTP_ARM, selected_arm_blob_sha256: fixture[:arm_blob_sha256],
          rows: WTP_ROWS, input_dim: WTP_DIN, output_dim: WTP_DOUT,
          input_rows_repeated_on_gpu: WTP_SOURCE_ROWS,
          decoded_weight_dtype: "F32", decoded_weight_bytes: WTP_WEIGHTS.to_i64 * 4,
          weight_stage_and_mps_input_share_identity: !qbit,
          candidate_binds_dense_weight: !qbit, candidate_global_dense_weight_stage_bytes: qbit ? 0_i64 : WTP_WEIGHTS.to_i64 * 4,
          candidate_packed_weight_bytes_including_offsets_and_luts: packed_bytes,
          dense_control_weight_retained_by_diagnostic: true,
          output_elements_per_route: WTP_OUTPUT_ELEMENTS,
          explicit_buffer_bytes: explicit_bytes, explicit_buffer_cap_bytes: WTP_DENSE_CAP,
          full_model_loaded: false, generation: false}.to_json)
    STDOUT.flush

    qualification = wtp_dense_qualification(kernel, zero, poison, repeat, compare, seed_control,
      source, input, weight, bias, expected, tensorops_output, mps_output,
      metrics, control_actual, control_expected, plan.not_nil!)
    raise "Dense qualification result missing pass signal" unless qualification["qualification_passed"].as_bool

    route_a = kernel.route
    route_b = "preexpanded_mps_f32"
    warm_order = [route_a, route_b, route_b, route_a,
                  route_b, route_a, route_a, route_b]
    warm_order.each_with_index do |route, position|
      timing = wtp_dense_timed(route, kernel, input, weight, bias,
        tensorops_output, mps_output, plan.not_nil!)
      puts({schema: "wan_tensorops_dense_timing_v1", arm: WTP_ARM,
            sample_kind: "warmup", segment: route, position: position,
            order: position < 4 ? "ABBA" : "BAAB", measured: false,
            same_decoded_coefficients_and_input: true,
            correctness_checked_before_and_after_batch: true, timing: timing}.to_json)
      STDOUT.flush
    end
    measured_order = [route_a, route_b, route_b, route_a,
                      route_b, route_a, route_a, route_b]
    measured_order.each_with_index do |route, position|
      timing = wtp_dense_timed(route, kernel, input, weight, bias,
        tensorops_output, mps_output, plan.not_nil!)
      puts({schema: "wan_tensorops_dense_timing_v1", arm: WTP_ARM,
            sample_kind: "paired", segment: route, position: position,
            pair: position // 2, quartet: position // 4,
            order: position < 4 ? "ABBA" : "BAAB", measured: true,
            same_decoded_coefficients_and_input: true,
            correctness_checked_before_and_after_batch: true, timing: timing}.to_json)
      STDOUT.flush
    end

    # Check the final timed outputs, then re-execute both routes from poisoned
    # outputs in this diagnostic command. The latter catches incomplete writes;
    # it is outside every measured interval and adds no inter-operator wait.
    post = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(post)
    before = ML::MetalBuffer.video_spec_allocations
    wtp_zero(e, zero, metrics, 35)
    wtp_compare_triplet(e, compare, tensorops_output, mps_output, expected,
      metrics, WTP_OUTPUT_ELEMENTS, WTP_REFERENCE_ELEMS)
    wtp_seed_control(e, seed_control, control_actual, control_expected)
    wtp_compare(e, compare, control_actual, control_expected, metrics, 3, 2, 2)
    wtp_poison(e, poison, tensorops_output, WTP_OUTPUT_ELEMENTS)
    wtp_poison(e, poison, mps_output, WTP_OUTPUT_ELEMENTS)
    kernel.encode(e, input, weight, bias, tensorops_output, WTP_ROWS, WTP_DIN, WTP_DOUT)
    ML::Video::MetalLinearMPS.project_preexpanded(e, input, bias, mps_output,
      WTP_ROWS, WTP_DIN, WTP_DOUT, ML::Video::WeightDType::F32, plan.not_nil!, weight)
    wtp_compare(e, compare, tensorops_output, expected, metrics, 4, WTP_OUTPUT_ELEMENTS, WTP_REFERENCE_ELEMS)
    wtp_compare(e, compare, mps_output, expected, metrics, 5, WTP_OUTPUT_ELEMENTS, WTP_REFERENCE_ELEMS)
    wtp_compare(e, compare, tensorops_output, mps_output, metrics, 6, WTP_OUTPUT_ELEMENTS, WTP_OUTPUT_ELEMENTS)
    e.end_encoding
    post.commit
    post.wait
    gpu_status = wtp_status!(kernel)
    allocations = ML::MetalBuffer.video_spec_allocations - before
    raise "Post-timing qualification encode allocated tracked MetalBuffers" unless allocations == 0
    tensorops_metric = wtp_metric(metrics, 0)
    mps_metric = wtp_metric(metrics, 1)
    pair_metric = wtp_metric(metrics, 2)
    positive_metric = wtp_metric(metrics, 3)
    post_passed = wtp_tolerance_ok(tensorops_metric) && wtp_tolerance_ok(mps_metric) &&
                  wtp_tolerance_ok(pair_metric) && wtp_positive_ok(positive_metric) &&
                  (4..6).all? { |slot| wtp_tolerance_ok(wtp_metric(metrics, slot)) }
    puts({schema: "wan_tensorops_dense_post_timing_qualification_v1", arm: WTP_ARM,
          candidate_route: kernel.route, qbit_gpu_status: gpu_status,
          tensorops_vs_upstream: wtp_metric_json(metrics, 0, WTP_OUTPUT_ELEMENTS),
          preexpanded_mps_vs_upstream: wtp_metric_json(metrics, 1, WTP_OUTPUT_ELEMENTS),
          tensorops_vs_preexpanded_mps: wtp_metric_json(metrics, 2, WTP_OUTPUT_ELEMENTS),
          poisoned_reexecution_tensorops_vs_upstream: wtp_metric_json(metrics, 4, WTP_OUTPUT_ELEMENTS),
          poisoned_reexecution_mps_vs_upstream: wtp_metric_json(metrics, 5, WTP_OUTPUT_ELEMENTS),
          poisoned_reexecution_tensorops_vs_mps: wtp_metric_json(metrics, 6, WTP_OUTPUT_ELEMENTS),
          poisoned_reexecution_outside_timed_intervals: true,
          individual_timed_invocations_poisoned: false,
          comparator_positive_control: wtp_metric_json(metrics, 3, 2),
          comparator_positive_control_passed: wtp_positive_ok(positive_metric),
          qualification_passed: post_passed,
          encode_metal_buffer_allocations: allocations, terminal_waits: 1,
          inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
    STDOUT.flush
    raise "Dense post-timing qualification rejected" unless post_passed
    puts({schema: "wan_tensorops_dense_complete_v1", mode: qbit ? "qbit" : "dense", arm: WTP_ARM,
          candidate_route: kernel.route,
          source_manifest_sha256: fixture[:fixture_hash],
          positional_manifest_sha256: fixture[:reference_hash],
          warmup_records: warm_order.size, measured_records: measured_order.size,
          measured_pairs: 4, measured_orders: ["ABBA", "BAAB"],
          pre_timing_qualification_passed: qualification["qualification_passed"].as_bool,
          post_timing_qualification_passed: post_passed,
          full_model_loaded: false, generation: false, complete: true}.to_json)
    STDOUT.flush
  ensure
    plan.try(&.release)
    arena.release
  end
end

private def wtp_describe_kernel(kernel : WanTensorOpsKernel) : Nil
  shared_panel = kernel.route.starts_with?("shared_panel_")
  puts({schema: "wan_tensorops_kernel_v1", tile_m: kernel.tile_m, tile_n: kernel.tile_n, tile_k: 32,
        candidate_route: kernel.route,
        execution_scope: shared_panel ? "threadgroup" : "simdgroup",
        threads_per_threadgroup: shared_panel ? 128 : 32,
        simdgroups_per_threadgroup: shared_panel ? 4 : 1,
        mma_backend: shared_panel ? "metal_simdgroup_matrix" : "mpp_tensor_ops",
        coefficient_dtype: "F32", accumulator_dtype: "F32",
        thread_execution_width: kernel.pipeline.thread_execution_width,
        source_sha256: kernel.source_sha256, relaxed_precision: false,
        metal_language: "4.0", math_mode: "safe"}.to_json)
  STDOUT.flush
end

case mode
when "small"
  kernel = wtp_dense_kernel(backend, tile_m, tile_n)
  wtp_describe_kernel(kernel)
  wtp_run_small(kernel)
when "dense", "qbit"
  fixture = wtp_validate_dense_fixture
  wtp_qbit_failure_controls(backend, tile_m, tile_n, fixture) if mode == "qbit"
  wtp_run_dense(backend, tile_m, tile_n, fixture, mode == "qbit")
else
  raise "Unreachable TensorOps probe mode"
end
