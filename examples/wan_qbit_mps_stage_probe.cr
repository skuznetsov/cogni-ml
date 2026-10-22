# Opt-in packed positional-weight to shared F32 stage to MPS diagnostic.
# Uses pinned seeded inputs and reference data; no model loading or generation.
require "json"
require "digest/sha256"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_linear_mps"
require "../src/ml/video/qbit_positional_weights"
require "../src/ml/video/qbit_positional_mps_linear"
require "../spec/support/video_allocation_counter"

private QMS_DIN             =  5120
private QMS_DOUT            = 13824
private QMS_SOURCE_ROWS     =     4
private QMS_ROWS            =  4480
private QMS_TILE            =  1024
private QMS_WEIGHTS         = QMS_DIN * QMS_DOUT
private QMS_WEIGHT_BYTES    = QMS_WEIGHTS.to_i64 * 4
private QMS_OUTPUT_ELEMENTS = QMS_ROWS * QMS_DOUT
private QMS_ORACLE_ELEMENTS = QMS_SOURCE_ROWS * QMS_DOUT
private QMS_ARM_A           = "positional_tile1024_t050_mode0"
private QMS_ARM_B           = "positional_tile1024_t020_mode0"
private QMS_FIXTURE_SHA     = "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703"
private QMS_REFERENCE_SHA   = "d77557f7d01798f91bbc3ad765f0147e33c0702e4cdb735f849b9641a0aaee3f"
private QMS_FIXTURE_DIR     = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022/fixture-4480-v3"
private QMS_REFERENCE_DIR   = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-positional-20261022/fixture-v1"
private QMS_STAGE_ID        = "shared_f32_stage_0"
private QMS_INPUT_ID        = "seeded_input_0"
private QMS_BIAS_ID         = "bias_f32_0"
private QMS_TOLERANCE       = 2.0e-5

unless ARGV.empty?
  raise ArgumentError.new("wan_qbit_mps_stage_probe accepts no arguments; fixture paths are pinned")
end

private def qms_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

private def qms_verify(directory : String, record : JSON::Any,
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
  unless File.size(full) == expected_bytes && qms_sha256(full) == record["sha256"].as_s
    raise "Fixture file hash mismatch: #{path}"
  end
  full
end

private def qms_upload(path : String, buffer : ML::MetalBuffer, bytes : Int64) : Nil
  raise "Upload byte count mismatch for #{path}" unless File.size(path) == bytes && buffer.size >= bytes
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), bytes.to_i32)) }
end

private def qms_verify_distinct_rows(path : String) : Bool
  bytes = File.read(path).to_slice
  row_bytes = QMS_DIN * 4
  raise "Input fixture byte length mismatch" unless bytes.size == QMS_SOURCE_ROWS * row_bytes
  QMS_SOURCE_ROWS.times do |left|
    ((left + 1)...QMS_SOURCE_ROWS).each do |right|
      raise "Seeded input rows #{left} and #{right} are identical" if bytes[left * row_bytes, row_bytes] == bytes[right * row_bytes, row_bytes]
    end
  end
  true
end

private def qms_u16(bytes : Bytes, offset : Int32) : UInt32
  bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
end

private def qms_u32(bytes : Bytes, offset : Int32) : UInt32
  qms_u16(bytes, offset) | (qms_u16(bytes, offset + 2) << 16)
end

private def qms_validate_wire(arm : JSON::Any, plan_path : String,
                              offsets_path : String, blob_path : String) : Nil
  plans = File.read(plan_path).to_slice
  offsets = File.read(offsets_path).to_slice
  blob = File.read(blob_path).to_slice
  tiles = QMS_WEIGHTS // QMS_TILE
  raise "Bad plan byte length" unless plans.size == tiles * 16
  raise "Bad offsets byte length" unless offsets.size == (tiles + 1) * 4
  raise "Bad blob byte length" unless blob.size == arm["blob"]["bytes"].as_i
  raise "Offsets must begin at zero" unless qms_u32(offsets, 0) == 0
  mode = arm["value_mode"].as_i.to_u32
  selected = {4_u32 => 0_i64, 5_u32 => 0_i64, 16_u32 => 0_i64}
  corrections_total = 0_i64

  tiles.times do |tile|
    p = tile * 16
    record_bytes = qms_u32(plans, p)
    precision = qms_u32(plans, p + 4)
    corrections = qms_u32(plans, p + 8)
    invalid = qms_u32(plans, p + 12)
    start = qms_u32(offsets, tile * 4)
    finish = qms_u32(offsets, (tile + 1) * 4)
    unless invalid == 0 && finish >= start && finish - start == record_bytes &&
           (start & 3) == 0 && (finish & 3) == 0
      raise "Invalid plan/offset contract at tile #{tile}"
    end
    base_bytes = case precision
                 when  4_u32 then 4 * 136
                 when  5_u32 then 4 * 168
                 when 16_u32 then QMS_TILE * 2
                 else             raise "Unsupported precision at tile #{tile}"
                 end
    raise "Raw tile has corrections" if precision == 16_u32 && corrections != 0
    raise "Correction count exceeds tile size" if corrections > QMS_TILE
    unpadded = 4 + base_bytes + corrections.to_i32 * 4
    expected_bytes = (unpadded + 3) & ~3
    unless expected_bytes.to_u32 == record_bytes && finish <= blob.size
      raise "Record cost/extent mismatch at tile #{tile}"
    end
    header = start.to_i32
    unless blob[header].to_u32 == precision && blob[header + 1].to_u32 == mode &&
           qms_u16(blob, header + 2) == corrections
      raise "Wire header disagrees with manifest plan at tile #{tile}"
    end
    selected[precision] += 1
    corrections_total += corrections
    entries = header + 4 + base_bytes
    previous = -1
    corrections.to_i32.times do |i|
      position = qms_u16(blob, entries + i * 4).to_i32
      unless position > previous && position < QMS_TILE
        raise "Non-increasing/out-of-range correction at tile #{tile}"
      end
      previous = position
    end
    expected_bytes.times do |i|
      raise "Nonzero record padding at tile #{tile}" if i >= unpadded && blob[header + i] != 0
    end
  end
  raise "Final offset differs from blob length" unless qms_u32(offsets, tiles * 4).to_i == blob.size
  counts = arm["plan_summary"]["selected_tile_counts"]
  unless selected[4_u32] == counts["p4"].as_i64 && selected[5_u32] == counts["p5"].as_i64 &&
         selected[16_u32] == counts["raw"].as_i64 &&
         corrections_total == arm["plan_summary"]["correction_count"].as_i64
    raise "Wire histogram disagrees with reference manifest"
  end
end

private QMS_METAL = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void qms_zero(device uint *values [[buffer(0)]], constant uint &n [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) { if (i<n) values[i]=0u; }
  kernel void qms_repeat(device const float *source [[buffer(0)]], device float *out [[buffer(1)]],
      constant uint &period [[buffer(2)]], constant uint &n [[buffer(3)]],
      uint i [[thread_position_in_grid]]) { if (i<n) out[i]=source[i%period]; }
  kernel void qms_copy(device const float *source [[buffer(0)]], device float *out [[buffer(1)]],
      constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i<n) out[i]=source[i];
  }
  kernel void qms_seed(device float *actual [[buffer(0)]], device float *expected [[buffer(1)]],
      uint i [[thread_position_in_grid]]) {
    if (i<2u) { actual[i]=i==0u?1.0f:as_type<float>(0x7fc00000u); expected[i]=0.0f; }
  }
  kernel void qms_compare(device const float *actual [[buffer(0)]],
      device const float *expected [[buffer(1)]], device atomic_uint *metrics [[buffer(2)]],
      constant uint &slot [[buffer(3)]], constant uint &n [[buffer(4)]],
      constant uint &period [[buffer(5)]], uint i [[thread_position_in_grid]],
      uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint4 counts[256]; threadgroup float2 maxima[256];
    uint4 c=uint4(0u); float2 m=float2(0.0f);
    if (i<n) {
      float a=actual[i], b=expected[i%period];
      c.x=uint(as_type<uint>(a)!=as_type<uint>(b));
      if (!isfinite(a)||!isfinite(b)) {c.y=1u;c.z=1u;}
      else {float d=abs(a-b);float r=d/(2.0e-5f+2.0e-5f*abs(b));
        c.y=uint(r>1.0f);m=float2(d,r);}
    }
    counts[t]=c;maxima[t]=m;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=128u;s>0u;s>>=1u) {
      if(t<s){counts[t]+=counts[t+s];maxima[t]=max(maxima[t],maxima[t+s]);}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(t==0u) {
      uint base=slot*5u;
      atomic_fetch_add_explicit(metrics+base,counts[0].x,memory_order_relaxed);
      atomic_fetch_add_explicit(metrics+base+1u,counts[0].y,memory_order_relaxed);
      atomic_fetch_add_explicit(metrics+base+2u,counts[0].z,memory_order_relaxed);
      atomic_fetch_max_explicit(metrics+base+3u,as_type<uint>(maxima[0].x),memory_order_relaxed);
      atomic_fetch_max_explicit(metrics+base+4u,as_type<uint>(maxima[0].y),memory_order_relaxed);
    }
  }
METAL

private def qms_zero(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                     buffer : ML::MetalBuffer, n : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    .set_value(n.to_u32, 1).dispatch_1d(n)
end

private def qms_repeat(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                       source : ML::MetalBuffer, output : ML::MetalBuffer,
                       period : Int32, n : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(source, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
    .set_value(period.to_u32, 2).set_value(n.to_u32, 3).dispatch_1d(n)
end

private def qms_copy(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                     source : ML::MetalBuffer, output : ML::MetalBuffer, n : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(source, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
    .set_value(n.to_u32, 2).dispatch_1d(n)
end

private def qms_compare(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                        actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                        metrics : ML::MetalBuffer, slot : Int32, n : Int32, period : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(slot.to_u32, 3).set_value(n.to_u32, 4).set_value(period.to_u32, 5)
    .dispatch_threadgroups({(n + 255)//256, 1, 1}, {256, 1, 1})
end

private def qms_metric(buffer : ML::MetalBuffer, slot : Int32)
  p = buffer.contents.as(UInt32*) + slot*5
  {bit_differences: p[0], tolerance_violations: p[1], nonfinite: p[2],
   max_abs: p[3].unsafe_as(Float32), max_ratio: p[4].unsafe_as(Float32)}
end

private def qms_metric_json(buffer : ML::MetalBuffer, slot : Int32, elements : Int32) : JSON::Any
  metric = qms_metric(buffer, slot)
  JSON.parse({compared_elements: elements, bit_differences: metric[:bit_differences],
              tolerance_violations: metric[:tolerance_violations], nonfinite: metric[:nonfinite],
              max_abs: metric[:max_abs], max_ratio: metric[:max_ratio]}.to_json)
end

private def qms_tolerance_ok(metric) : Bool
  metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
end

private def qms_exact_ok(metric) : Bool
  metric[:bit_differences] == 0 && metric[:nonfinite] == 0
end

private def qms_seed_control(e : ML::Metal::ComputeEncoder, seed : ML::Metal::ComputePipeline,
                             actual : ML::MetalBuffer, expected : ML::MetalBuffer) : Nil
  e.set_pipeline(seed).set_buffer(actual, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(expected, 1, ML::Metal::BufferAccess::Write).dispatch_1d(2)
end

private def qms_route(route : String, e : ML::Metal::ComputeEncoder,
                      input : ML::MetalBuffer, offsets : ML::MetalBuffer, blob : ML::MetalBuffer,
                      lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                      oracle_weight : ML::MetalBuffer, positional_output : ML::MetalBuffer,
                      legacy_output : ML::MetalBuffer, status : ML::MetalBuffer,
                      stage : ML::MetalBuffer,
                      positional_plan : ML::Video::WanT5MPSLinearPlan,
                      legacy_plan : ML::Video::WanT5MPSLinearPlan) : Nil
  case route
  when "positional_mps"
    ML::Video::QBitPositionalMPSLinear.project(e, input, offsets, blob, lut4, lut5, bias,
      positional_output, status, QMS_ROWS, QMS_DIN, QMS_DOUT, QMS_TILE,
      ML::Video::WeightDType::F32, positional_plan, stage)
  when "legacy_f32_mps"
    ML::Video::MetalLinearMPS.project(e, input, oracle_weight, bias, legacy_output,
      QMS_ROWS, QMS_DIN, QMS_DOUT, ML::Video::WeightDType::F32,
      ML::Video::WeightDType::F32, legacy_plan, stage)
  else
    raise "Unknown MPS stage route #{route}"
  end
end

private def qms_qualification(phase : String, zero : ML::Metal::ComputePipeline,
                              compare : ML::Metal::ComputePipeline, seed : ML::Metal::ComputePipeline,
                              input : ML::MetalBuffer, offsets : ML::MetalBuffer, blob : ML::MetalBuffer,
                              lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                              bias : ML::MetalBuffer, oracle_weight : ML::MetalBuffer,
                              oracle_output : ML::MetalBuffer, positional_output : ML::MetalBuffer,
                              legacy_output : ML::MetalBuffer, stage : ML::MetalBuffer,
                              status : ML::MetalBuffer, metrics : ML::MetalBuffer,
                              control_actual : ML::MetalBuffer, control_expected : ML::MetalBuffer,
                              positional_plan : ML::Video::WanT5MPSLinearPlan,
                              legacy_plan : ML::Video::WanT5MPSLinearPlan)
  before = ML::MetalBuffer.video_spec_allocations
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  qms_zero(e, zero, metrics, 45)
  qms_route("positional_mps", e, input, offsets, blob, lut4, lut5, bias, oracle_weight,
    positional_output, legacy_output, status, stage, positional_plan, legacy_plan)
  qms_compare(e, compare, positional_output, oracle_output, metrics, 0,
    QMS_OUTPUT_ELEMENTS, QMS_ORACLE_ELEMENTS)
  qms_compare(e, compare, stage, oracle_weight, metrics, 1, QMS_WEIGHTS, QMS_WEIGHTS)
  qms_route("legacy_f32_mps", e, input, offsets, blob, lut4, lut5, bias, oracle_weight,
    positional_output, legacy_output, status, stage, positional_plan, legacy_plan)
  qms_compare(e, compare, legacy_output, oracle_output, metrics, 2,
    QMS_OUTPUT_ELEMENTS, QMS_ORACLE_ELEMENTS)
  qms_compare(e, compare, stage, oracle_weight, metrics, 3, QMS_WEIGHTS, QMS_WEIGHTS)
  qms_compare(e, compare, positional_output, legacy_output, metrics, 4,
    QMS_OUTPUT_ELEMENTS, QMS_OUTPUT_ELEMENTS)
  qms_seed_control(e, seed, control_actual, control_expected)
  qms_compare(e, compare, control_actual, control_expected, metrics, 5, 2, 2)
  e.end_encoding
  command.commit
  command.wait

  allocated = ML::MetalBuffer.video_spec_allocations - before
  raise "#{phase} qualification allocated tracked MetalBuffers" unless allocated == 0
  positional = qms_metric(metrics, 0)
  positional_stage = qms_metric(metrics, 1)
  legacy = qms_metric(metrics, 2)
  legacy_stage = qms_metric(metrics, 3)
  paired = qms_metric(metrics, 4)
  positive = qms_metric(metrics, 5)
  status_words = status.contents.as(UInt32*).to_slice(8).to_a
  positive_ok = positive[:bit_differences] == 2 && positive[:tolerance_violations] == 2 && positive[:nonfinite] == 1
  passed = qms_tolerance_ok(positional) && qms_exact_ok(positional_stage) &&
           qms_tolerance_ok(legacy) && qms_exact_ok(legacy_stage) &&
           qms_tolerance_ok(paired) && positive_ok && status_words.all? { |word| word == 0 }
  record = {
    schema: "wan_qbit_mps_stage_qualification_v1", phase: phase,
    arm: QMS_ARM_A, rows: QMS_ROWS, input_dim: QMS_DIN, output_dim: QMS_DOUT,
    weight_stage_id: QMS_STAGE_ID, weight_stage_bytes: QMS_WEIGHT_BYTES,
    comparator_tolerance_absolute: QMS_TOLERANCE, comparator_tolerance_relative: QMS_TOLERANCE,
    routes: [
      JSON.parse({segment: "positional_mps", qbit_decode_encode_calls: 1,
                  f32_weight_expand_encode_calls: 0, mps_projection_encode_calls: 1,
                  bias_add_encode_calls: 1, decode_inside_qualification_command: true,
                  output_vs_pinned_upstream: qms_metric_json(metrics, 0, QMS_OUTPUT_ELEMENTS),
                  stage_vs_decoded_f32_oracle: qms_metric_json(metrics, 1, QMS_WEIGHTS)}.to_json),
      JSON.parse({segment: "legacy_f32_mps", qbit_decode_encode_calls: 0,
                  f32_weight_expand_encode_calls: 1, mps_projection_encode_calls: 1,
                  bias_add_encode_calls: 1, decode_inside_qualification_command: false,
                  output_vs_pinned_upstream: qms_metric_json(metrics, 2, QMS_OUTPUT_ELEMENTS),
                  stage_vs_decoded_f32_oracle: qms_metric_json(metrics, 3, QMS_WEIGHTS)}.to_json),
    ],
    positional_vs_legacy: qms_metric_json(metrics, 4, QMS_OUTPUT_ELEMENTS),
    positive_control: qms_metric_json(metrics, 5, 2),
    positive_control_passed: positive_ok, status: status_words,
    qualification_passed: passed, command_buffers: 1, encoders: 1,
    encode_metal_buffer_allocations: allocated, terminal_waits: 1,
    inter_operator_host_waits: 0, device_activation_readbacks: 0,
  }
  JSON.parse(record.to_json)
end

private def qms_timed(route : String, input : ML::MetalBuffer,
                      offsets : ML::MetalBuffer, blob : ML::MetalBuffer,
                      lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                      oracle_weight : ML::MetalBuffer, positional_output : ML::MetalBuffer,
                      legacy_output : ML::MetalBuffer, status : ML::MetalBuffer,
                      stage : ML::MetalBuffer,
                      positional_plan : ML::Video::WanT5MPSLinearPlan,
                      legacy_plan : ML::Video::WanT5MPSLinearPlan)
  before = ML::MetalBuffer.video_spec_allocations
  started = Time.instant
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  qms_route(route, e, input, offsets, blob, lut4, lut5, bias, oracle_weight,
    positional_output, legacy_output, status, stage, positional_plan, legacy_plan)
  e.end_encoding
  encode_seconds = (Time.instant - started).total_seconds
  gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
  wall_seconds = (Time.instant - started).total_seconds
  allocated = ML::MetalBuffer.video_spec_allocations - before
  raise "Timed route allocated tracked MetalBuffers" unless allocated == 0
  status_words = status.contents.as(UInt32*).to_slice(8).to_a
  raise "Timed route reported a record error" unless status_words.all? { |word| word == 0 }
  {encode_seconds: encode_seconds, gpu_seconds: gpu_seconds, wall_seconds: wall_seconds,
   terminal_waits: 1, inter_operator_host_waits: 0, device_activation_readbacks: 0,
   encode_metal_buffer_allocations: allocated, status: status_words}
end

fixture_manifest_path = File.join(QMS_FIXTURE_DIR, "manifest.json")
reference_manifest_path = File.join(QMS_REFERENCE_DIR, "manifest.json")
fixture_hash = qms_sha256(fixture_manifest_path)
reference_hash = qms_sha256(reference_manifest_path)
raise "Unexpected source manifest SHA-256" unless fixture_hash == QMS_FIXTURE_SHA
raise "Unexpected positional manifest SHA-256" unless reference_hash == QMS_REFERENCE_SHA
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
unless geometry["source_rows"].as_i == QMS_SOURCE_ROWS && geometry["input_dim"].as_i == QMS_DIN &&
       geometry["output_dim"].as_i == QMS_DOUT && ref_geometry["rows"].as_i == QMS_ROWS &&
       ref_geometry["source_rows"].as_i == QMS_SOURCE_ROWS && ref_geometry["input_dim"].as_i == QMS_DIN &&
       ref_geometry["output_dim"].as_i == QMS_DOUT && ref_geometry["normalization_block_size"].as_i == 256
  raise "Fixture geometry mismatch"
end
provenance = fixture["input_provenance"]
unless provenance["kind"].as_s == "upstream_expected_ffn_mod_seeded_capture" &&
       provenance["capture_seed"].as_i == 7031 && !provenance["real_video_derived"].as_bool
  raise "Input is not the pinned seeded, non-video-derived fixture"
end
raise "Unexpected projection provenance" unless ref_geometry["projection_rows_saved"].as_i == QMS_SOURCE_ROWS

source_tensor = fixture["source_tensor"]
input_path = qms_verify(QMS_FIXTURE_DIR, fixture["input"], "F32", [QMS_SOURCE_ROWS, QMS_DIN])
distinct_rows = qms_verify_distinct_rows(input_path)
qms_verify(QMS_FIXTURE_DIR, source_tensor["weight_file"], "BF16", [QMS_DOUT, QMS_DIN])
qms_verify(QMS_FIXTURE_DIR, source_tensor["bias_bf16_file"], "BF16", [QMS_DOUT])
bias_path = qms_verify(QMS_FIXTURE_DIR, source_tensor["bias_f32_file"], "F32", [QMS_DOUT])
qms_verify(QMS_FIXTURE_DIR, fixture["variants"]["bf16"]["decoded_weight"], "F32", [QMS_DOUT, QMS_DIN])
p4 = fixture["variants"]["qbit_p4"]
p5 = fixture["variants"]["qbit_p5"]
qms_verify(QMS_FIXTURE_DIR, p4["packed_weight"], "QBitBlockBytes", [QMS_DOUT, QMS_DIN // 256, 136])
qms_verify(QMS_FIXTURE_DIR, p5["packed_weight"], "QBitBlockBytes", [QMS_DOUT, QMS_DIN // 256, 168])
lut4_path = qms_verify(QMS_FIXTURE_DIR, p4["lut"], "F32", [256])
lut5_path = qms_verify(QMS_FIXTURE_DIR, p5["lut"], "F32", [256])

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
arms = [QMS_ARM_A, QMS_ARM_B].map do |name|
  arm = all_arms.find { |candidate| candidate["name"].as_s == name } || raise "Missing arm #{name}"
  threshold_hex = name == QMS_ARM_A ? "0x3f000000" : "0x3e4ccccd"
  unless arm["tile_size"].as_i == QMS_TILE && arm["threshold_f32_hex"].as_s == threshold_hex &&
         arm["value_mode"].as_i == 0 && arm["precision_tie_order"].as_a.map(&.as_i) == [16, 5, 4]
    raise "Unadmitted arm policy: #{name}"
  end
  prov = arm["projection_provenance"]
  unless prov["rows_computed"].as_i == QMS_ROWS && prov["rows_saved"].as_i == QMS_SOURCE_ROWS &&
         prov["all_periodic_repeats_bitexact"].as_bool
    raise "Reference projection lacks seeded four-row repeat provenance: #{name}"
  end
  plan_path = qms_verify(QMS_REFERENCE_DIR, arm["plan"], "U32LE", [QMS_WEIGHTS // QMS_TILE, 4])
  offsets_path = qms_verify(QMS_REFERENCE_DIR, arm["offsets"], "U32LE", [QMS_WEIGHTS // QMS_TILE + 1])
  blob_bytes = arm["blob"]["bytes"].as_i
  blob_path = qms_verify(QMS_REFERENCE_DIR, arm["blob"], "U8", [blob_bytes])
  qms_validate_wire(arm, plan_path, offsets_path, blob_path)
  qms_verify(QMS_REFERENCE_DIR, arm["decoded_weight"], "F32", [QMS_DOUT, QMS_DIN])
  qms_verify(QMS_REFERENCE_DIR, arm["projection"], "F32", [QMS_SOURCE_ROWS, QMS_DOUT])
  arm
end
arm_a = arms[0]
arm_b = arms[1]

arena = ML::Video::Arena.new
positional_plan : ML::Video::WanT5MPSLinearPlan? = nil
legacy_plan : ML::Video::WanT5MPSLinearPlan? = nil
begin
  ML::Video::QBitPositionalMPSLinear.prepare!
  zero = ML::Metal::ComputePipeline.new("qms_zero", QMS_METAL)
  repeat = ML::Metal::ComputePipeline.new("qms_repeat", QMS_METAL)
  copy = ML::Metal::ComputePipeline.new("qms_copy", QMS_METAL)
  compare = ML::Metal::ComputePipeline.new("qms_compare", QMS_METAL)
  seed = ML::Metal::ComputePipeline.new("qms_seed", QMS_METAL)

  source = arena.alloc("four_seeded_source_rows", QMS_SOURCE_ROWS * QMS_DIN)
  input = arena.alloc("full_repeated_input", QMS_ROWS * QMS_DIN)
  stage = arena.alloc("shared_f32_weight_stage", QMS_WEIGHTS)
  oracle_weight_a = arena.alloc("oracle_weight_a_diagnostic", QMS_WEIGHTS)
  oracle_weight_b = arena.alloc("oracle_weight_b_diagnostic", QMS_WEIGHTS)
  oracle_output_a = arena.alloc("oracle_output_a_four_rows", QMS_ORACLE_ELEMENTS)
  oracle_output_b = arena.alloc("oracle_output_b_four_rows", QMS_ORACLE_ELEMENTS)
  positional_output = arena.alloc("positional_mps_output", QMS_OUTPUT_ELEMENTS)
  legacy_output = arena.alloc("legacy_f32_mps_output", QMS_OUTPUT_ELEMENTS)
  first_a_output = arena.alloc("first_a_output_snapshot", QMS_OUTPUT_ELEMENTS)
  bias = arena.alloc("bias_f32", QMS_DOUT)
  lut4 = arena.alloc("lut4", 256)
  lut5 = arena.alloc("lut5", 256)
  offsets_a = arena.alloc("arm_a_offsets", QMS_WEIGHTS // QMS_TILE + 1)
  offsets_b = arena.alloc("arm_b_offsets", QMS_WEIGHTS // QMS_TILE + 1)
  blob_a_bytes = arm_a["blob"]["bytes"].as_i
  blob_b_bytes = arm_b["blob"]["bytes"].as_i
  blob_a = arena.alloc("arm_a_blob", (blob_a_bytes + 3) // 4)
  blob_b = arena.alloc("arm_b_blob", (blob_b_bytes + 3) // 4)
  status = arena.alloc("sticky_status", 8)
  metrics = arena.alloc("comparison_metrics", 45)
  control_actual = arena.alloc("comparator_control_actual", 2)
  control_expected = arena.alloc("comparator_control_expected", 2)
  explicit_buffer_bytes = arena.resident_bytes
  explicit_buffer_cap_bytes = 2_i64 * 1024**3 + 512_i64 * 1024**2
  raise "Explicit buffers exceed the 2.5 GiB cap" unless explicit_buffer_bytes < explicit_buffer_cap_bytes

  qms_upload(input_path, source, QMS_SOURCE_ROWS.to_i64 * QMS_DIN * 4)
  qms_upload(bias_path, bias, QMS_DOUT.to_i64 * 4)
  qms_upload(lut4_path, lut4, 1024_i64)
  qms_upload(lut5_path, lut5, 1024_i64)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_a["offsets"]["path"].as_s), offsets_a, offsets_a.size)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_b["offsets"]["path"].as_s), offsets_b, offsets_b.size)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_a["blob"]["path"].as_s), blob_a, blob_a_bytes.to_i64)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_b["blob"]["path"].as_s), blob_b, blob_b_bytes.to_i64)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_a["decoded_weight"]["path"].as_s), oracle_weight_a, QMS_WEIGHT_BYTES)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_b["decoded_weight"]["path"].as_s), oracle_weight_b, QMS_WEIGHT_BYTES)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_a["projection"]["path"].as_s), oracle_output_a, QMS_ORACLE_ELEMENTS.to_i64 * 4)
  qms_upload(File.join(QMS_REFERENCE_DIR, arm_b["projection"]["path"].as_s), oracle_output_b, QMS_ORACLE_ELEMENTS.to_i64 * 4)

  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
    layers: 1, batches: 1, tokens: QMS_ROWS)
  positional_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, QMS_DIN, QMS_DOUT,
    input, stage, positional_output)
  legacy_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, QMS_DIN, QMS_DOUT,
    input, stage, legacy_output)

  setup = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(setup)
  qms_repeat(e, repeat, source, input, QMS_SOURCE_ROWS * QMS_DIN, QMS_ROWS * QMS_DIN)
  qms_zero(e, zero, status, 8)
  qms_zero(e, zero, metrics, 45)
  e.end_encoding
  setup.commit
  setup.wait

  puts({schema: "wan_qbit_mps_stage_setup_v1", mode: "packed_positional_to_shared_f32_mps",
        selected_arm: QMS_ARM_A, alternate_arm: QMS_ARM_B,
        source_manifest_sha256: fixture_hash, positional_manifest_sha256: reference_hash,
        selected_arm_blob_sha256: arm_a["blob"]["sha256"].as_s,
        alternate_arm_blob_sha256: arm_b["blob"]["sha256"].as_s,
        rows: QMS_ROWS, input_dim: QMS_DIN, output_dim: QMS_DOUT, tile_values: QMS_TILE,
        unique_seeded_rows: QMS_SOURCE_ROWS, capture_seed: 7031, real_video_derived: false,
        weight_elements: QMS_WEIGHTS, output_elements: QMS_OUTPUT_ELEMENTS,
        weight_stage_id: QMS_STAGE_ID, weight_stage_elements: QMS_WEIGHTS,
        weight_stage_bytes: QMS_WEIGHT_BYTES,
        diagnostic_oracle_weight_a_bytes: QMS_WEIGHT_BYTES,
        diagnostic_oracle_weight_b_bytes: QMS_WEIGHT_BYTES,
        explicit_buffer_bytes: explicit_buffer_bytes, explicit_buffer_cap_bytes: explicit_buffer_cap_bytes,
        both_packed_arms_resident_before_reuse: true, status_zeroed_once_before_chain: true,
        full_model_loaded: false, generation: false, passed: true}.to_json)
  STDOUT.flush

  # One encoder and one completion boundary cover the exact A/B/A stage reuse.
  reuse = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(reuse)
  before = ML::MetalBuffer.video_spec_allocations
  qms_zero(e, zero, metrics, 45)
  ML::Video::QBitPositionalMPSLinear.project(e, input, offsets_a, blob_a, lut4, lut5, bias,
    positional_output, status, QMS_ROWS, QMS_DIN, QMS_DOUT, QMS_TILE,
    ML::Video::WeightDType::F32, positional_plan.not_nil!, stage)
  qms_compare(e, compare, stage, oracle_weight_a, metrics, 0, QMS_WEIGHTS, QMS_WEIGHTS)
  qms_compare(e, compare, positional_output, oracle_output_a, metrics, 1,
    QMS_OUTPUT_ELEMENTS, QMS_ORACLE_ELEMENTS)
  qms_copy(e, copy, positional_output, first_a_output, QMS_OUTPUT_ELEMENTS)
  ML::Video::QBitPositionalMPSLinear.project(e, input, offsets_b, blob_b, lut4, lut5, bias,
    positional_output, status, QMS_ROWS, QMS_DIN, QMS_DOUT, QMS_TILE,
    ML::Video::WeightDType::F32, positional_plan.not_nil!, stage)
  qms_compare(e, compare, stage, oracle_weight_b, metrics, 2, QMS_WEIGHTS, QMS_WEIGHTS)
  qms_compare(e, compare, positional_output, oracle_output_b, metrics, 3,
    QMS_OUTPUT_ELEMENTS, QMS_ORACLE_ELEMENTS)
  qms_compare(e, compare, positional_output, first_a_output, metrics, 7,
    QMS_OUTPUT_ELEMENTS, QMS_OUTPUT_ELEMENTS)
  ML::Video::QBitPositionalMPSLinear.project(e, input, offsets_a, blob_a, lut4, lut5, bias,
    positional_output, status, QMS_ROWS, QMS_DIN, QMS_DOUT, QMS_TILE,
    ML::Video::WeightDType::F32, positional_plan.not_nil!, stage)
  qms_compare(e, compare, stage, oracle_weight_a, metrics, 4, QMS_WEIGHTS, QMS_WEIGHTS)
  qms_compare(e, compare, positional_output, oracle_output_a, metrics, 5,
    QMS_OUTPUT_ELEMENTS, QMS_ORACLE_ELEMENTS)
  qms_compare(e, compare, positional_output, first_a_output, metrics, 6,
    QMS_OUTPUT_ELEMENTS, QMS_OUTPUT_ELEMENTS)
  qms_seed_control(e, seed, control_actual, control_expected)
  qms_compare(e, compare, control_actual, control_expected, metrics, 8, 2, 2)
  e.end_encoding
  reuse.commit
  reuse.wait

  allocated = ML::MetalBuffer.video_spec_allocations - before
  raise "A/B/A reuse encode allocated tracked MetalBuffers" unless allocated == 0
  stage_a1 = qms_metric(metrics, 0)
  output_a1 = qms_metric(metrics, 1)
  stage_b = qms_metric(metrics, 2)
  output_b = qms_metric(metrics, 3)
  stage_a3 = qms_metric(metrics, 4)
  output_a3 = qms_metric(metrics, 5)
  a1_a3 = qms_metric(metrics, 6)
  b_a1 = qms_metric(metrics, 7)
  positive = qms_metric(metrics, 8)
  status_words = status.contents.as(UInt32*).to_slice(8).to_a
  positive_ok = positive[:bit_differences] == 2 && positive[:tolerance_violations] == 2 && positive[:nonfinite] == 1
  reuse_passed = qms_exact_ok(stage_a1) && qms_exact_ok(stage_b) && qms_exact_ok(stage_a3) &&
                 qms_tolerance_ok(output_a1) && qms_tolerance_ok(output_b) &&
                 qms_tolerance_ok(output_a3) && qms_exact_ok(a1_a3) &&
                 b_a1[:bit_differences] > 0 && b_a1[:nonfinite] == 0 && positive_ok &&
                 status_words.all? { |word| word == 0 }
  operations = [
    JSON.parse({sequence_index: 0, label: "A1", arm: QMS_ARM_A,
                weight_stage_id: QMS_STAGE_ID, stage_write_count: 1,
                stage_vs_arm_decoded_f32_oracle: qms_metric_json(metrics, 0, QMS_WEIGHTS),
                output_vs_arm_pinned_upstream: qms_metric_json(metrics, 1, QMS_OUTPUT_ELEMENTS)}.to_json),
    JSON.parse({sequence_index: 1, label: "B", arm: QMS_ARM_B,
                weight_stage_id: QMS_STAGE_ID, stage_write_count: 1,
                stage_vs_arm_decoded_f32_oracle: qms_metric_json(metrics, 2, QMS_WEIGHTS),
                output_vs_arm_pinned_upstream: qms_metric_json(metrics, 3, QMS_OUTPUT_ELEMENTS),
                output_vs_first_a: qms_metric_json(metrics, 7, QMS_OUTPUT_ELEMENTS)}.to_json),
    JSON.parse({sequence_index: 2, label: "A3", arm: QMS_ARM_A,
                weight_stage_id: QMS_STAGE_ID, stage_write_count: 1,
                stage_vs_arm_decoded_f32_oracle: qms_metric_json(metrics, 4, QMS_WEIGHTS),
                output_vs_arm_pinned_upstream: qms_metric_json(metrics, 5, QMS_OUTPUT_ELEMENTS),
                output_vs_first_a: qms_metric_json(metrics, 6, QMS_OUTPUT_ELEMENTS)}.to_json),
  ]
  puts({schema: "wan_qbit_mps_stage_reuse_v1", operations: operations,
        shared_weight_stage_id: QMS_STAGE_ID, sequence: [QMS_ARM_A, QMS_ARM_B, QMS_ARM_A],
        packed_arms_resident_before_chain: true, one_command_buffer: true, one_encoder: true,
        stage_write_count: 3, first_a_output_preserved_on_gpu: true,
        a1_vs_a3_bitexact: qms_exact_ok(a1_a3), b_vs_a1_differs: b_a1[:bit_differences] > 0,
        positive_control: qms_metric_json(metrics, 8, 2), positive_control_passed: positive_ok,
        status: status_words, reuse_passed: reuse_passed,
        encode_metal_buffer_allocations: allocated, terminal_waits: 1,
        inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
  STDOUT.flush
  raise "Packed A/B/A shared-stage reuse qualification rejected" unless reuse_passed

  pre = qms_qualification("pre_timing", zero, compare, seed, input, offsets_a, blob_a,
    lut4, lut5, bias, oracle_weight_a, oracle_output_a, positional_output, legacy_output,
    stage, status, metrics, control_actual, control_expected,
    positional_plan.not_nil!, legacy_plan.not_nil!)
  puts(pre.to_json); STDOUT.flush
  raise "Pre-timing MPS route qualification rejected" unless pre["qualification_passed"].as_bool

  warm_order = ["legacy_f32_mps", "positional_mps", "positional_mps", "legacy_f32_mps",
                "positional_mps", "legacy_f32_mps", "legacy_f32_mps", "positional_mps"]
  warm_order.each_with_index do |route, position|
    result = qms_timed(route, input, offsets_a, blob_a, lut4, lut5, bias, oracle_weight_a,
      positional_output, legacy_output, status, stage, positional_plan.not_nil!, legacy_plan.not_nil!)
    puts({schema: "wan_qbit_mps_stage_timing_v1", arm: QMS_ARM_A,
          weight_stage_id: QMS_STAGE_ID, input_id: QMS_INPUT_ID, bias_id: QMS_BIAS_ID,
          rows: QMS_ROWS, input_dim: QMS_DIN, output_dim: QMS_DOUT,
          sample_kind: "warmup", segment: route, position: position, measured: false,
          pair: position // 2, quartet: position // 4, order: position < 4 ? "ABBA" : "BAAB",
          decode_inside_timed_segment: route == "positional_mps",
          f32_weight_expand_inside_timed_segment: route == "legacy_f32_mps",
          route_action_counts: route == "positional_mps" ? {qbit_decode_encode_calls: 1, f32_weight_expand_encode_calls: 0,
                                                            mps_projection_encode_calls: 1, bias_add_encode_calls: 1} : {qbit_decode_encode_calls: 0, f32_weight_expand_encode_calls: 1,
                                                                                                                         mps_projection_encode_calls: 1, bias_add_encode_calls: 1},
          timing: result}.to_json)
    STDOUT.flush
  end

  measured_order = ["legacy_f32_mps", "positional_mps", "positional_mps", "legacy_f32_mps",
                    "positional_mps", "legacy_f32_mps", "legacy_f32_mps", "positional_mps"]
  measured_order.each_with_index do |route, position|
    result = qms_timed(route, input, offsets_a, blob_a, lut4, lut5, bias, oracle_weight_a,
      positional_output, legacy_output, status, stage, positional_plan.not_nil!, legacy_plan.not_nil!)
    puts({schema: "wan_qbit_mps_stage_timing_v1", arm: QMS_ARM_A,
          weight_stage_id: QMS_STAGE_ID, input_id: QMS_INPUT_ID, bias_id: QMS_BIAS_ID,
          rows: QMS_ROWS, input_dim: QMS_DIN, output_dim: QMS_DOUT,
          sample_kind: "paired", segment: route, position: position, pair: position // 2,
          quartet: position // 4, order: position < 4 ? "ABBA" : "BAAB", measured: true,
          decode_inside_timed_segment: route == "positional_mps",
          f32_weight_expand_inside_timed_segment: route == "legacy_f32_mps",
          route_action_counts: route == "positional_mps" ? {qbit_decode_encode_calls: 1, f32_weight_expand_encode_calls: 0,
                                                            mps_projection_encode_calls: 1, bias_add_encode_calls: 1} : {qbit_decode_encode_calls: 0, f32_weight_expand_encode_calls: 1,
                                                                                                                         mps_projection_encode_calls: 1, bias_add_encode_calls: 1},
          correctness_outside_timed_segment: true, timing: result}.to_json)
    STDOUT.flush
  end

  post = qms_qualification("post_timing", zero, compare, seed, input, offsets_a, blob_a,
    lut4, lut5, bias, oracle_weight_a, oracle_output_a, positional_output, legacy_output,
    stage, status, metrics, control_actual, control_expected,
    positional_plan.not_nil!, legacy_plan.not_nil!)
  puts(post.to_json); STDOUT.flush
  raise "Post-timing MPS route qualification rejected" unless post["qualification_passed"].as_bool
  puts({schema: "wan_qbit_mps_stage_completion_v1", mode: "packed_positional_to_shared_f32_mps",
        source_manifest_sha256: fixture_hash, positional_manifest_sha256: reference_hash,
        selected_arm: QMS_ARM_A, alternate_arm: QMS_ARM_B,
        reuse_records: 1, reuse_operations: 3, pre_qualification_records: 1,
        warmup_records: warm_order.size, measured_records: measured_order.size,
        post_qualification_records: 1, reuse_passed: reuse_passed,
        pre_timing_qualification_passed: pre["qualification_passed"].as_bool,
        post_timing_qualification_passed: post["qualification_passed"].as_bool,
        full_model_loaded: false, generation: false, complete: true}.to_json)
ensure
  legacy_plan.try(&.release)
  positional_plan.try(&.release)
  arena.release
end
