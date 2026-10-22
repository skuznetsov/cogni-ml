# Opt-in native arithmetic falsifier for the pinned four-row Wan FFN fixture.
# It does not load a model or generate video; the positional wire was produced
# separately and is consumed here only after manifest/hash/layout validation.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private DOT_DIN           =  5120
private DOT_DOUT          = 13824
private DOT_ROWS          =     4
private DOT_TILE          =  1024
private DOT_WEIGHTS       = DOT_DIN * DOT_DOUT
private DOT_OUTPUTS       = DOT_ROWS * DOT_DOUT
private DOT_FIXTURE_SHA   = "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703"
private DOT_REFERENCE_SHA = "d77557f7d01798f91bbc3ad765f0147e33c0702e4cdb735f849b9641a0aaee3f"
private DOT_FIXTURE_DIR   = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022/fixture-4480-v3"
private DOT_REFERENCE_DIR = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-positional-20261022/fixture-v1"
private DOT_ARM_NAMES     = [
  "positional_tile1024_t020_mode0",
  "positional_tile1024_t020_mode1",
  "positional_tile1024_t050_mode0",
  "positional_tile1024_t050_mode1",
]

private def dot_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

private def dot_verify(directory : String, record : JSON::Any,
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
  unless File.size(full) == expected_bytes && dot_sha256(full) == record["sha256"].as_s
    raise "Fixture file hash mismatch: #{path}"
  end
  full
end

private def dot_upload(path : String, buffer : ML::MetalBuffer, bytes : Int64) : Nil
  raise "Upload byte count mismatch for #{path}" unless File.size(path) == bytes && buffer.size >= bytes
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), bytes.to_i32)) }
end

private def dot_verify_distinct_rows(path : String) : Bool
  content = File.read(path)
  bytes = content.to_slice
  row_bytes = DOT_DIN * 4
  raise "Input fixture byte length mismatch" unless bytes.size == DOT_ROWS * row_bytes
  DOT_ROWS.times do |left|
    ((left + 1)...DOT_ROWS).each do |right|
      if bytes[left * row_bytes, row_bytes] == bytes[right * row_bytes, row_bytes]
        raise "Seeded input rows #{left} and #{right} are identical"
      end
    end
  end
  true
end

private def dot_u16(bytes : Bytes, offset : Int32) : UInt32
  bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
end

private def dot_u32(bytes : Bytes, offset : Int32) : UInt32
  dot_u16(bytes, offset) | (dot_u16(bytes, offset + 2) << 16)
end

private def dot_validate_wire(arm : JSON::Any, plan_path : String,
                              offsets_path : String, blob_path : String) : Nil
  plans_string = File.read(plan_path)
  offsets_string = File.read(offsets_path)
  blob_string = File.read(blob_path)
  plans = plans_string.to_slice
  offsets = offsets_string.to_slice
  blob = blob_string.to_slice
  tiles = DOT_WEIGHTS // DOT_TILE
  raise "Bad plan byte length" unless plans.size == tiles * 16
  raise "Bad offsets byte length" unless offsets.size == (tiles + 1) * 4
  raise "Bad blob byte length" unless blob.size == arm["blob"]["bytes"].as_i
  raise "Offsets must begin at zero" unless dot_u32(offsets, 0) == 0
  mode = arm["value_mode"].as_i.to_u32
  selected = {4_u32 => 0_i64, 5_u32 => 0_i64, 16_u32 => 0_i64}
  corrections_total = 0_i64

  tiles.times do |tile|
    p = tile * 16
    record_bytes = dot_u32(plans, p)
    precision = dot_u32(plans, p + 4)
    corrections = dot_u32(plans, p + 8)
    invalid = dot_u32(plans, p + 12)
    start = dot_u32(offsets, tile * 4)
    finish = dot_u32(offsets, (tile + 1) * 4)
    unless invalid == 0 && finish >= start && finish - start == record_bytes &&
           (start & 3) == 0 && (finish & 3) == 0
      raise "Invalid plan/offset contract at tile #{tile}"
    end
    base_bytes = case precision
                 when  4_u32 then 4 * 136
                 when  5_u32 then 4 * 168
                 when 16_u32 then DOT_TILE * 2
                 else             raise "Unsupported precision at tile #{tile}"
                 end
    raise "Raw tile has corrections" if precision == 16_u32 && corrections != 0
    raise "Correction count exceeds tile size" if corrections > DOT_TILE
    unpadded = 4 + base_bytes + corrections.to_i32 * 4
    expected_bytes = (unpadded + 3) & ~3
    unless expected_bytes.to_u32 == record_bytes && finish <= blob.size
      raise "Record cost/extent mismatch at tile #{tile}"
    end
    header = start.to_i32
    unless blob[header].to_u32 == precision && blob[header + 1].to_u32 == mode &&
           dot_u16(blob, header + 2) == corrections
      raise "Wire header disagrees with manifest plan at tile #{tile}"
    end
    selected[precision] += 1
    corrections_total += corrections
    width = 2
    entries = header + 4 + base_bytes
    previous = -1
    corrections.to_i32.times do |i|
      position = dot_u16(blob, entries + i * (width + 2)).to_i32
      unless position > previous && position < DOT_TILE
        raise "Non-increasing/out-of-range correction at tile #{tile}"
      end
      previous = position
    end
    expected_bytes.times do |i|
      if i >= unpadded && blob[header + i] != 0
        raise "Nonzero record padding at tile #{tile}"
      end
    end
  end
  raise "Final offset differs from blob length" unless dot_u32(offsets, tiles * 4).to_i == blob.size
  summary = arm["plan_summary"]
  counts = summary["selected_tile_counts"]
  unless selected[4_u32] == counts["p4"].as_i64 &&
         selected[5_u32] == counts["p5"].as_i64 &&
         selected[16_u32] == counts["raw"].as_i64 &&
         corrections_total == summary["correction_count"].as_i64
    raise "Wire histogram disagrees with reference manifest"
  end
end

private DOT_METAL = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  #pragma clang fp contract(off)
  constant uint QDIN = 5120u;
  constant uint QDOUT = 13824u;
  constant uint QTILE = 1024u;

  static inline uint qdot_u16(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8);
  }
  static inline uint qdot_u32(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
  }
  static inline float qdot_bf16(device const uchar *p) {
    return as_type<float>(qdot_u16(p) << 16);
  }
  static inline uint qdot_code(device const uchar *block, uint within, uint precision) {
    uint code = 0u;
    uint byte_index = 31u - within / 8u;
    uint mask = 1u << (within & 7u);
    for (uint plane = 0u; plane < precision; ++plane)
      if ((uint(block[8u + plane * 32u + byte_index]) & mask) != 0u)
        code |= 1u << (7u - plane);
    return code;
  }

  kernel void wan_qbit_dot(device const float *dense [[buffer(0)]],
      device const float *input [[buffer(1)]], device const uint *offsets [[buffer(2)]],
      device const uchar *blob [[buffer(3)]], device const float *lut4 [[buffer(4)]],
      device const float *lut5 [[buffer(5)]], device const float *bias [[buffer(6)]],
      device float *output [[buffer(7)]], constant uint &method [[buffer(8)]],
      uint3 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float4 partial[4][256];
    threadgroup float code_lut[4][32];
    threadgroup float means[4];
    threadgroup float sigmas[4];
    float4 accumulated = float4(0.0f);
    uint row = group.y;
    uint out0 = group.x * 4u;

    for (uint channel_base = 0u; channel_base < QDIN; channel_base += 256u) {
      if (method != 0u) {
        for (uint b = 0u; b < 4u; ++b) {
          uint out_index = out0 + b;
          uint flat = out_index * QDIN + channel_base;
          uint tile_index = flat / QTILE;
          uint local = flat % QTILE;
          uint begin = offsets[tile_index];
          device const uchar *record = blob + begin;
          uint precision = uint(record[0]);
          if (t == 0u) {
            means[b] = 0.0f;
            sigmas[b] = 0.0f;
            if (precision != 16u) {
              device const uchar *block = record + 4u + (local / 256u) * (8u + precision * 32u);
              means[b] = as_type<float>(qdot_u32(block));
              sigmas[b] = as_type<float>(qdot_u32(block + 4u));
            }
          }
          if (method == 3u && precision != 16u && t < (1u << precision)) {
            device const uchar *block = record + 4u + (local / 256u) * (8u + precision * 32u);
            device const float *lut = precision == 4u ? lut4 : lut5;
            float mean = as_type<float>(qdot_u32(block));
            float sigma = as_type<float>(qdot_u32(block + 4u));
            code_lut[b][t] = mean + sigma * lut[t << (8u - precision)];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint b = 0u; b < 4u; ++b) {
        uint out_index = out0 + b;
        uint channel = channel_base + t;
        uint weight_index = out_index * QDIN + channel;
        float x = input[row * QDIN + channel];
        float actual = method == 0u ? dense[weight_index] : 0.0f;
        float mean = 0.0f, sigma = 0.0f, level = 0.0f, base = actual, correction_delta = 0.0f;
        if (method != 0u) {
          uint tile_index = weight_index / QTILE;
          uint local = weight_index % QTILE;
          uint begin = offsets[tile_index];
          device const uchar *record = blob + begin;
          uint precision = uint(record[0]);
          uint value_mode = uint(record[1]);
          uint correction_count = qdot_u16(record + 2u);
          device const uchar *payload = record + 4u;
          if (precision == 16u) {
            base = 0.0f;
            actual = qdot_bf16(payload + local * 2u);
            // Raw BF16 uses a zero affine base, so its complete value is residual.
            correction_delta = actual;
          } else {
            uint block_stride = 8u + precision * 32u;
            device const uchar *block = payload + (local / 256u) * block_stride;
            mean = as_type<float>(qdot_u32(block));
            sigma = as_type<float>(qdot_u32(block + 4u));
            uint code = qdot_code(block, local & 255u, precision);
            device const float *lut = precision == 4u ? lut4 : lut5;
            level = lut[code];
            base = method == 3u ? code_lut[b][code >> (8u - precision)] : mean + sigma * level;
            actual = base;
            uint base_bytes = 4u * block_stride;
            device const uchar *entries = payload + base_bytes;
            uint lo = 0u, hi = correction_count;
            while (lo < hi) {
              uint mid = lo + (hi - lo) / 2u;
              uint position = qdot_u16(entries + mid * 4u);
              if (position < (local & 1023u)) lo = mid + 1u;
              else hi = mid;
            }
            if (lo < correction_count && qdot_u16(entries + lo * 4u) == (local & 1023u)) {
              uint value_bits = qdot_u16(entries + lo * 4u + 2u);
              if (value_mode == 0u) actual = as_type<float>(value_bits << 16);
              else actual = base + float(as_type<half>(ushort(value_bits)));
              // Mode 1 residual follows rounded reconstruction, not the stored half delta.
              correction_delta = actual - base;
            }
          }
        }
        partial[b][t] = method == 2u
          ? float4(x, x * level, x * correction_delta, 0.0f)
          : float4(actual * x, 0.0f, 0.0f, 0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (t < stride)
          for (uint b = 0u; b < 4u; ++b)
            partial[b][t] += partial[b][t + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (t == 0u) {
        for (uint b = 0u; b < 4u; ++b) {
          float group_sum = partial[b][0].x;
          if (method == 2u)
            group_sum = means[b] * partial[b][0].x + sigmas[b] * partial[b][0].y + partial[b][0].z;
          accumulated[b] += group_sum;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u)
      for (uint b = 0u; b < 4u; ++b)
        output[row * QDOUT + out0 + b] = accumulated[b] + bias[out0 + b];
  }

  kernel void wan_qbit_dot_zero(device atomic_uint *metrics [[buffer(0)]],
      constant uint &words [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < words) atomic_store_explicit(metrics + i, 0u, memory_order_relaxed);
  }
  kernel void wan_qbit_dot_compare(device const float *actual [[buffer(0)]],
      device const float *expected [[buffer(1)]], device atomic_uint *metrics [[buffer(2)]],
      constant uint &slot [[buffer(3)]], constant uint &count [[buffer(4)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float a = actual[i], b = expected[i];
    uint base = slot * 3u;
    if (as_type<uint>(a) != as_type<uint>(b))
      atomic_fetch_add_explicit(metrics + base, 1u, memory_order_relaxed);
    if (!isfinite(a) || !isfinite(b)) {
      atomic_fetch_add_explicit(metrics + base + 1u, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + base + 2u, 1u, memory_order_relaxed);
    } else if (abs(a - b) > 2.0e-5f + 2.0e-5f * abs(b)) {
      atomic_fetch_add_explicit(metrics + base + 1u, 1u, memory_order_relaxed);
    }
  }
  kernel void wan_qbit_dot_seed_comparator(device float *actual [[buffer(0)]],
      device float *expected [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i == 0u) { actual[0] = 1.0f; expected[0] = 0.0f; }
    if (i == 1u) { actual[1] = as_type<float>(0x7fc00000u); expected[1] = 0.0f; }
  }
METAL

private def dot_compare(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                        actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                        metrics : ML::MetalBuffer, slot : Int32, count : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(slot.to_u32, 3).set_value(count.to_u32, 4).dispatch_1d(count)
end

fixture_dir = DOT_FIXTURE_DIR
reference_dir = DOT_REFERENCE_DIR
OptionParser.parse do |parser|
  parser.on("--fixture=DIR", "Pinned four-row source fixture") { |value| fixture_dir = value }
  parser.on("--positional-fixture=DIR", "Pinned positional oracle fixture") { |value| reference_dir = value }
end

fixture_manifest_path = File.join(fixture_dir, "manifest.json")
reference_manifest_path = File.join(reference_dir, "manifest.json")
fixture_hash = dot_sha256(fixture_manifest_path)
reference_hash = dot_sha256(reference_manifest_path)
raise "Unexpected source manifest SHA-256" unless fixture_hash == DOT_FIXTURE_SHA
raise "Unexpected positional manifest SHA-256" unless reference_hash == DOT_REFERENCE_SHA
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
unless geometry["source_rows"].as_i == DOT_ROWS && geometry["input_dim"].as_i == DOT_DIN &&
       geometry["output_dim"].as_i == DOT_DOUT && ref_geometry["rows"].as_i == 4480 &&
       ref_geometry["source_rows"].as_i == DOT_ROWS && ref_geometry["input_dim"].as_i == DOT_DIN &&
       ref_geometry["output_dim"].as_i == DOT_DOUT && ref_geometry["normalization_block_size"].as_i == 256
  raise "Fixture geometry mismatch"
end
provenance = fixture["input_provenance"]
unless provenance["kind"].as_s == "upstream_expected_ffn_mod_seeded_capture" &&
       provenance["capture_seed"].as_i == 7031 && !provenance["real_video_derived"].as_bool
  raise "Input is not the pinned seeded, non-video-derived fixture"
end
raise "Unexpected projection provenance" unless ref_geometry["projection_rows_saved"].as_i == 4

source_tensor = fixture["source_tensor"]
input_path = dot_verify(fixture_dir, fixture["input"], "F32", [DOT_ROWS, DOT_DIN])
distinct_rows = dot_verify_distinct_rows(input_path)
dot_verify(fixture_dir, source_tensor["weight_file"], "BF16", [DOT_DOUT, DOT_DIN])
dot_verify(fixture_dir, source_tensor["bias_bf16_file"], "BF16", [DOT_DOUT])
bias_path = dot_verify(fixture_dir, source_tensor["bias_f32_file"], "F32", [DOT_DOUT])
dot_verify(fixture_dir, fixture["variants"]["bf16"]["decoded_weight"], "F32", [DOT_DOUT, DOT_DIN])
p4 = fixture["variants"]["qbit_p4"]
p5 = fixture["variants"]["qbit_p5"]
dot_verify(fixture_dir, p4["packed_weight"], "QBitBlockBytes", [DOT_DOUT, DOT_DIN // 256, 136])
dot_verify(fixture_dir, p5["packed_weight"], "QBitBlockBytes", [DOT_DOUT, DOT_DIN // 256, 168])
lut4_path = dot_verify(fixture_dir, p4["lut"], "F32", [256])
lut5_path = dot_verify(fixture_dir, p5["lut"], "F32", [256])

source_hashes = reference["source_hashes"]
expected_hashes = {
  "checkpoint_weight_bf16_sha256"  => source_tensor["weight_file"]["sha256"].as_s,
  "decoded_bf16_source_f32_sha256" => fixture["variants"]["bf16"]["decoded_weight"]["sha256"].as_s,
  "input_f32_sha256"               => fixture["input"]["sha256"].as_s,
  "bias_bf16_sha256"               => source_tensor["bias_bf16_file"]["sha256"].as_s,
  "bias_f32_sha256"                => source_tensor["bias_f32_file"]["sha256"].as_s,
  "qbit_p4_payload_sha256"         => p4["packed_weight"]["sha256"].as_s,
  "qbit_p5_payload_sha256"         => p5["packed_weight"]["sha256"].as_s,
  "qbit_lut_p4_sha256"             => p4["lut"]["sha256"].as_s,
  "qbit_lut_p5_sha256"             => p5["lut"]["sha256"].as_s,
}
expected_hashes.each do |name, digest|
  raise "Positional lineage mismatch for #{name}" unless source_hashes[name].as_s == digest
end

all_arms = reference["arms"].as_a
arms = DOT_ARM_NAMES.map do |name|
  arm = all_arms.find { |candidate| candidate["name"].as_s == name } || raise "Missing arm #{name}"
  threshold_hex = name.includes?("t020") ? "0x3e4ccccd" : "0x3f000000"
  mode = name.ends_with?("mode0") ? 0 : 1
  unless arm["tile_size"].as_i == DOT_TILE && arm["threshold_f32_hex"].as_s == threshold_hex &&
         arm["value_mode"].as_i == mode && arm["precision_tie_order"].as_a.map(&.as_i) == [16, 5, 4]
    raise "Unadmitted arm policy: #{name}"
  end
  prov = arm["projection_provenance"]
  unless prov["rows_computed"].as_i == 4480 && prov["rows_saved"].as_i == 4 &&
         prov["all_periodic_repeats_bitexact"].as_bool
    raise "Reference projection lacks four-row repeat provenance: #{name}"
  end
  plan_path = dot_verify(reference_dir, arm["plan"], "U32LE", [DOT_WEIGHTS // DOT_TILE, 4])
  offsets_path = dot_verify(reference_dir, arm["offsets"], "U32LE", [DOT_WEIGHTS // DOT_TILE + 1])
  blob_bytes = arm["blob"]["bytes"].as_i
  blob_path = dot_verify(reference_dir, arm["blob"], "U8", [blob_bytes])
  dot_validate_wire(arm, plan_path, offsets_path, blob_path)
  dot_verify(reference_dir, arm["decoded_weight"], "F32", [DOT_DOUT, DOT_DIN])
  dot_verify(reference_dir, arm["projection"], "F32", [DOT_ROWS, DOT_DOUT])
  arm
end

arena = ML::Video::Arena.new
begin
  input = arena.alloc("four_seeded_input_rows", DOT_ROWS * DOT_DIN)
  dense = arena.alloc("arm_decoded_f32_weight", DOT_WEIGHTS)
  bias = arena.alloc("captured_bias_f32", DOT_DOUT)
  lut4 = arena.alloc("qbit_lut_p4", 256)
  lut5 = arena.alloc("qbit_lut_p5", 256)
  offsets = arena.alloc("positional_offsets_u32", DOT_WEIGHTS // DOT_TILE + 1)
  max_blob = arms.max_of { |arm| arm["blob"]["bytes"].as_i }
  blob = arena.alloc("positional_blob", (max_blob + 3) // 4)
  dense_output = arena.alloc("dense_dot_output", DOT_OUTPUTS)
  fused_output = arena.alloc("fused_decode_dot_output", DOT_OUTPUTS)
  factor_output = arena.alloc("factored_dot_output", DOT_OUTPUTS)
  lookup_output = arena.alloc("onchip_codebook_dot_output", DOT_OUTPUTS)
  reference_output = arena.alloc("mps_oracle_output", DOT_OUTPUTS)
  metrics = arena.alloc("comparison_counters", 24)
  control_actual = arena.alloc("comparator_control_actual", 2)
  control_expected = arena.alloc("comparator_control_expected", 2)
  resident_bytes = arena.resident_bytes
  unless resident_bytes < (1_i64 << 30)
    raise "Explicit probe buffers exceed 1 GiB: #{resident_bytes}"
  end

  dot_upload(input_path, input, DOT_ROWS.to_i64 * DOT_DIN * 4)
  dot_upload(bias_path, bias, DOT_DOUT.to_i64 * 4)
  dot_upload(lut4_path, lut4, 256_i64 * 4)
  dot_upload(lut5_path, lut5, 256_i64 * 4)

  dot_source = ML::Metal::ComputePipeline.new("wan_qbit_dot", DOT_METAL)
  dot_zero = ML::Metal::ComputePipeline.new("wan_qbit_dot_zero", DOT_METAL)
  dot_compare_pipeline = ML::Metal::ComputePipeline.new("wan_qbit_dot_compare", DOT_METAL)
  dot_seed = ML::Metal::ComputePipeline.new("wan_qbit_dot_seed_comparator", DOT_METAL)
  puts({schema: "wan_qbit_dot_setup_v1", source_manifest_sha256: fixture_hash,
        positional_manifest_sha256: reference_hash, source_schema: fixture["schema"].as_s,
        positional_schema: reference["schema"].as_s, seeded_source_rows: DOT_ROWS,
        input_provenance: provenance["kind"].as_s, capture_seed: provenance["capture_seed"].as_i,
        real_video_derived: provenance["real_video_derived"].as_bool,
        source_input_tiled_for_execution: false, distinct_seeded_rows: distinct_rows,
        dimensions: {rows: DOT_ROWS, input: DOT_DIN, output: DOT_DOUT},
        explicit_buffer_bytes: resident_bytes, explicit_buffer_limit_bytes: 1_i64 << 30,
        methods: ["dense_decoded_f32", "onchip_positional_decode_dot", "code_domain_mean_scale_factoring",
                  "onchip_per_group_codebook_expansion"], timing_claim: false,
        full_model_loaded: false, generation: false}.to_json)
  STDOUT.flush

  arms.each do |arm|
    name = arm["name"].as_s
    blob_bytes = arm["blob"]["bytes"].as_i
    blob_path = File.join(reference_dir, arm["blob"]["path"].as_s)
    offsets_path = File.join(reference_dir, arm["offsets"]["path"].as_s)
    dense_path = File.join(reference_dir, arm["decoded_weight"]["path"].as_s)
    reference_path = File.join(reference_dir, arm["projection"]["path"].as_s)
    dot_upload(dense_path, dense, DOT_WEIGHTS.to_i64 * 4)
    dot_upload(reference_path, reference_output, DOT_OUTPUTS.to_i64 * 4)
    dot_upload(offsets_path, offsets, (DOT_WEIGHTS // DOT_TILE + 1).to_i64 * 4)
    dot_upload(blob_path, blob, blob_bytes.to_i64)

    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    allocations_before_encode = ML::MetalBuffer.video_spec_allocations
    encoder.set_pipeline(dot_zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write)
      .set_value(24_u32, 1).dispatch_1d(24)
    {dense_output, fused_output, factor_output, lookup_output}.each_with_index do |output, method|
      encoder.set_pipeline(dot_source).set_buffer(dense, 0).set_buffer(input, 1)
        .set_buffer(offsets, 2).set_buffer(blob, 3).set_buffer(lut4, 4).set_buffer(lut5, 5)
        .set_buffer(bias, 6).set_buffer(output, 7, ML::Metal::BufferAccess::Write)
        .set_value(method.to_u32, 8).dispatch_threadgroups({DOT_DOUT // 4, DOT_ROWS, 1}, {256, 1, 1})
    end
    {dense_output, fused_output, factor_output, lookup_output}.each_with_index do |output, slot|
      dot_compare(encoder, dot_compare_pipeline, output, reference_output, metrics, slot.to_i32, DOT_OUTPUTS)
    end
    dot_compare(encoder, dot_compare_pipeline, fused_output, dense_output, metrics, 4, DOT_OUTPUTS)
    dot_compare(encoder, dot_compare_pipeline, factor_output, dense_output, metrics, 5, DOT_OUTPUTS)
    dot_compare(encoder, dot_compare_pipeline, lookup_output, fused_output, metrics, 6, DOT_OUTPUTS)
    encoder.set_pipeline(dot_seed).set_buffer(control_actual, 0, ML::Metal::BufferAccess::Write)
      .set_buffer(control_expected, 1, ML::Metal::BufferAccess::Write).dispatch_1d(2)
    dot_compare(encoder, dot_compare_pipeline, control_actual, control_expected, metrics, 7, 2)
    encoder.end_encoding
    command.commit
    command.wait
    allocated_during_encode = ML::MetalBuffer.video_spec_allocations - allocations_before_encode
    raise "Encode-time MetalBuffer allocation detected: #{allocated_during_encode}" unless allocated_during_encode == 0

    words = metrics.contents.as(UInt32*)
    comparison_names = ["dense_vs_mps", "positional_vs_mps", "factored_vs_mps", "codebook_vs_mps",
                        "positional_vs_dense", "factored_vs_dense", "codebook_vs_positional", "seeded_comparator"]
    comparisons = {} of String => NamedTuple(bit_differences: UInt32, tolerance_violations: UInt32, nonfinite: UInt32)
    comparison_names.each_with_index do |label, slot|
      base = slot * 3
      comparisons[label] = {bit_differences: words[base], tolerance_violations: words[base + 1], nonfinite: words[base + 2]}
    end
    comparator_ok = comparisons["seeded_comparator"][:bit_differences] == 2 &&
                    comparisons["seeded_comparator"][:tolerance_violations] == 2 &&
                    comparisons["seeded_comparator"][:nonfinite] == 1
    fused_control_ok = comparisons["positional_vs_dense"][:bit_differences] == 0 &&
                       comparisons["positional_vs_dense"][:tolerance_violations] == 0 &&
                       comparisons["positional_vs_dense"][:nonfinite] == 0
    codebook_control_ok = comparisons["codebook_vs_positional"][:bit_differences] == 0 &&
                          comparisons["codebook_vs_positional"][:tolerance_violations] == 0 &&
                          comparisons["codebook_vs_positional"][:nonfinite] == 0
    result = {schema: "wan_qbit_dot_result_v1", arm: name, tile_size: DOT_TILE,
              threshold_f32_hex: arm["threshold_f32_hex"].as_s, value_mode: arm["value_mode"].as_i,
              executed_rows: DOT_ROWS, distinct_seeded_rows: distinct_rows, compared_outputs_per_path: DOT_OUTPUTS,
              dense_decoded_f32: comparisons["dense_vs_mps"], exact_decode_dot: comparisons["positional_vs_mps"],
              factored_mean_scale: comparisons["factored_vs_mps"],
              cached_codebook_expansion: comparisons["codebook_vs_mps"],
              fused_vs_control: comparisons["positional_vs_dense"],
              factored_vs_control: comparisons["factored_vs_dense"],
              cached_vs_fused: comparisons["codebook_vs_positional"],
              mps_gate: {dense: comparisons["dense_vs_mps"][:tolerance_violations] == 0 && comparisons["dense_vs_mps"][:nonfinite] == 0,
                         exact_decode_dot: comparisons["positional_vs_mps"][:tolerance_violations] == 0 && comparisons["positional_vs_mps"][:nonfinite] == 0,
                         factoring: comparisons["factored_vs_mps"][:tolerance_violations] == 0 && comparisons["factored_vs_mps"][:nonfinite] == 0,
                         codebook_expansion: comparisons["codebook_vs_mps"][:tolerance_violations] == 0 && comparisons["codebook_vs_mps"][:nonfinite] == 0},
              comparator_positive_control_passed: comparator_ok, fused_control_passed: fused_control_ok,
              codebook_equivalence_passed: codebook_control_ok, encode_metal_buffer_allocations: allocated_during_encode,
              terminal_waits: 1, inter_operator_host_waits: 0, device_activation_readbacks: 0,
              blob_and_offsets_bytes: blob_bytes + offsets.size, production_consumer: false, timing_claim: false}
    puts result.to_json
    STDOUT.flush
    raise "Comparator +1/NaN controls did not reject" unless comparator_ok
    raise "Fused positional decode does not match dense reduction control" unless fused_control_ok
    raise "Cached codebook path differs from direct positional decode" unless codebook_control_ok
  end
  puts({schema: "wan_qbit_dot_controls_v1", all_arms_completed: true,
        fixture_manifest_sha256: fixture_hash, positional_manifest_sha256: reference_hash,
        terminal_waits: arms.size, inter_operator_host_waits: 0, device_activation_readbacks: 0,
        full_model_loaded: false, generation: false, production_consumer: false}.to_json)
ensure
  arena.release
end
