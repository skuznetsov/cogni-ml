# Opt-in native QBit positional-weight codec probe. It consumes a pinned
# four-row Wan block-0 projection fixture and never loads a Wan model/video.
# Run under support/wan_memory_guard.py with the 15% reserve and an explicit
# process-tree budget; fixture uploads and terminal metric reads are untimed.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_linear_mps"
require "../src/ml/video/packed_weight_decode"
require "../src/ml/video/qbit_positional_weights"
require "../spec/support/video_allocation_counter"

private POS_DIN            =  5120
private POS_DOUT           = 13824
private POS_SOURCE_ROWS    =     4
private POS_ROWS           =  4480
private POS_REPETITIONS    = POS_ROWS // POS_SOURCE_ROWS
private POS_WEIGHTS        = POS_DIN * POS_DOUT
private POS_OUTPUTS        = POS_ROWS * POS_DOUT
private POS_SOURCE_OUTPUTS = POS_SOURCE_ROWS * POS_DOUT
private POS_THRESHOLD_BITS = {0.1_f32, 0.2_f32, 0.5_f32}
private POS_TILES          = {256, 512, 1024}
private POS_REF_DIR        = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-positional-20261022/fixture-v1"
private POS_SOURCE_DIR     = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022/fixture-4480-v3"

private def pos_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

private def pos_verify_record(directory : String, record : JSON::Any,
                              dtype : String, shape : Array(Int32)) : String
  path = record["path"].as_s
  unless File.basename(path) == path && !path.empty?
    raise "Unsafe positional fixture path: #{path}"
  end
  raise "Positional fixture dtype mismatch for #{path}" unless record["dtype"].as_s == dtype
  got_shape = record["shape"].as_a.map(&.as_i)
  raise "Positional fixture shape mismatch for #{path}: #{got_shape}" unless got_shape == shape
  bytes = record["bytes"].as_i64
  element_bytes = dtype == "U8" || dtype == "QBitBlockBytes" ? 1 : dtype == "BF16" ? 2 : 4
  expected_bytes = shape.product.to_i64 * element_bytes
  raise "Positional fixture byte count mismatch for #{path}" unless bytes == expected_bytes
  full = File.join(directory, path)
  unless File.size(full) == bytes && pos_sha256(full) == record["sha256"].as_s
    raise "Positional fixture bytes/hash mismatch: #{path}"
  end
  full
end

private def pos_upload(path : String, buffer : ML::MetalBuffer, bytes : Int64) : Nil
  raise "Positional fixture byte count mismatch: #{path}" unless File.size(path) == bytes && buffer.size >= bytes
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), bytes.to_i32)) }
end

private def pos_write_u32le(buffer : ML::MetalBuffer, index : Int32, value : UInt32) : Nil
  p = buffer.contents.as(UInt8*) + index * 4
  p[0] = (value & 0xff_u32).to_u8
  p[1] = ((value >> 8) & 0xff_u32).to_u8
  p[2] = ((value >> 16) & 0xff_u32).to_u8
  p[3] = ((value >> 24) & 0xff_u32).to_u8
end

private def pos_write_u16le(buffer : ML::MetalBuffer, offset : Int32, value : UInt16) : Nil
  p = buffer.contents.as(UInt8*) + offset
  p[0] = (value & 0xff_u16).to_u8
  p[1] = ((value >> 8) & 0xff_u16).to_u8
end

private def pos_bytes_u32le(bytes : Bytes, offset : Int32) : UInt32
  bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8) |
    (bytes[offset + 2].to_u32 << 16) | (bytes[offset + 3].to_u32 << 24)
end

private def pos_validate_plan_files(plan_path : String, offsets_path : String,
                                    tile_count : Int32, tile_values : Int32) : UInt32
  plan_data = File.read(plan_path).to_slice
  offset_data = File.read(offsets_path).to_slice
  raise "Positional plan file has wrong size" unless plan_data.size == tile_count * 16
  raise "Positional offsets file has wrong size" unless offset_data.size == (tile_count + 1) * 4
  raise "Positional offsets must start at zero" unless pos_bytes_u32le(offset_data, 0) == 0
  tile_count.times do |tile|
    base = tile * 16
    length = pos_bytes_u32le(plan_data, base)
    precision = pos_bytes_u32le(plan_data, base + 4)
    corrections = pos_bytes_u32le(plan_data, base + 8)
    invalid = pos_bytes_u32le(plan_data, base + 12)
    start = pos_bytes_u32le(offset_data, tile * 4)
    finish = pos_bytes_u32le(offset_data, (tile + 1) * 4)
    unless finish >= start && finish - start == length && invalid == 0
      raise "Positional plan word 0 must be record_size_bytes at tile #{tile}"
    end
    width = tile_values == 256 ? 1 : 2
    base_bytes = case precision
                 when  4_u32 then tile_values // 256 * 136
                 when  5_u32 then tile_values // 256 * 168
                 when 16_u32 then tile_values * 2
                 else             raise "Unsupported precision in oracle plan tile #{tile}"
                 end
    raise "Raw positional tile contains corrections" if precision == 16 && corrections != 0
    raise "Positional corrections exceed tile size" if corrections > tile_values
    unpadded = 4 + base_bytes + corrections.to_i32 * (width + 2)
    expected_length = (unpadded + 3) & ~3
    raise "Positional plan record cost mismatch at tile #{tile}" unless length == expected_length.to_u32
    raise "Positional offset table is not four-byte aligned" unless (start & 3) == 0 && (finish & 3) == 0
  end
  pos_bytes_u32le(offset_data, tile_count * 4)
end

private def pos_read_u32(buffer : ML::MetalBuffer, index : Int32) : UInt32
  p = buffer.contents.as(UInt8*) + index * 4
  p[0].to_u32 | (p[1].to_u32 << 8) | (p[2].to_u32 << 16) | (p[3].to_u32 << 24)
end

private def pos_read_u16(buffer : ML::MetalBuffer, offset : Int32) : UInt16
  p = buffer.contents.as(UInt8*) + offset
  p[0].to_u16 | (p[1].to_u16 << 8)
end

private POS_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void pos_zero_words(device atomic_uint *words [[buffer(0)]],
      constant uint &count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < count) atomic_store_explicit(words+i, 0u, memory_order_relaxed);
  }
  kernel void pos_copy_f32(device const float *src [[buffer(0)]],
      device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
      uint i [[thread_position_in_grid]]) { if (i < count) dst[i] = src[i]; }
  kernel void pos_repeat_f32(device const float *src [[buffer(0)]],
      device float *dst [[buffer(1)]], constant uint &source_count [[buffer(2)]],
      constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = src[i % source_count];
  }
  kernel void pos_make_bf16_nan(device ushort *source [[buffer(0)]],
      uint i [[thread_position_in_grid]]) { if (i == 0u) source[0] = 0x7fc0u; }
  kernel void pos_corrupt_byte(device uchar *blob [[buffer(0)]],
      constant uint &index [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i == 0u) blob[index] = 1u;
  }
  kernel void pos_count_nonfinite(device const float *source [[buffer(0)]],
      device atomic_uint *count [[buffer(1)]], constant uint &length [[buffer(2)]],
      uint i [[thread_position_in_grid]]) {
    if (i < length && !isfinite(source[i]))
      atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
  }
  kernel void pos_compare_bytes(device const uchar *actual [[buffer(0)]],
      device const uchar *expected [[buffer(1)]], device atomic_uint *metrics [[buffer(2)]],
      constant uint &length [[buffer(3)]], uint i [[thread_position_in_grid]],
      uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint errors[256];
    errors[t] = i < length && actual[i] != expected[i] ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128u; stride>0u; stride>>=1u) {
      if (t < stride) errors[t] += errors[t+stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u && errors[0] != 0u)
      atomic_fetch_add_explicit(metrics, errors[0], memory_order_relaxed);
  }
  kernel void pos_compare_f32(device const float *actual [[buffer(0)]],
      device const float *expected [[buffer(1)]], device atomic_uint *metrics [[buffer(2)]],
      device float4 *partials [[buffer(3)]], constant uint &length [[buffer(4)]],
      uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
      uint g [[threadgroup_position_in_grid]]) {
    threadgroup uint4 counts[256]; threadgroup float2 maxima[256];
    threadgroup float4 sums[256];
    uint4 c=uint4(0u); float2 mx=float2(0.0f); float4 s=float4(0.0f);
    if (i < length) {
      float a=actual[i], b=expected[i]; c.x=as_type<uint>(a)!=as_type<uint>(b);
      if (!isfinite(a)||!isfinite(b)) { c.y=1u; c.z=1u; }
      else {
        float d=a-b, error=abs(d), ratio=error/(2.0e-5f+2.0e-5f*abs(b));
        c.y=ratio>1.0f; mx=float2(error,ratio); s=float4(d*d,b*b,a*a,a*b);
      }
    }
    counts[t]=c; maxima[t]=mx; sums[t]=s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128u; stride>0u; stride>>=1u) {
      if (t < stride) {
        counts[t]+=counts[t+stride]; maxima[t]=max(maxima[t],maxima[t+stride]);
        sums[t]+=sums[t+stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
      atomic_fetch_add_explicit(metrics,counts[0].x,memory_order_relaxed);
      atomic_fetch_add_explicit(metrics+1,counts[0].y,memory_order_relaxed);
      atomic_fetch_add_explicit(metrics+2,counts[0].z,memory_order_relaxed);
      atomic_fetch_max_explicit(metrics+3,as_type<uint>(maxima[0].x),memory_order_relaxed);
      atomic_fetch_max_explicit(metrics+4,as_type<uint>(maxima[0].y),memory_order_relaxed);
      partials[g]=sums[0];
    }
  }
  kernel void pos_fold_f32(device const float4 *partials [[buffer(0)]],
      device float4 *total [[buffer(1)]], constant uint &groups [[buffer(2)]],
      uint t [[thread_index_in_threadgroup]]) {
    threadgroup float4 sums[256]; float4 value=0.0f;
    for (uint g=t;g<groups;g+=256u) value+=partials[g];
    sums[t]=value; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128u;stride>0u;stride>>=1u) {
      if (t<stride) sums[t]+=sums[t+stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u) total[0]=sums[0];
  }
METAL

private def pos_zero(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                     buffer : ML::MetalBuffer, words : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    .set_value(words.to_u32, 1).dispatch_1d(words)
end

private def pos_compare_float(e : ML::Metal::ComputeEncoder,
                              zero : ML::Metal::ComputePipeline,
                              compare : ML::Metal::ComputePipeline,
                              fold : ML::Metal::ComputePipeline,
                              actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                              metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
                              total : ML::MetalBuffer, count : Int32) : Nil
  groups = (count + 255) // 256
  pos_zero(e, zero, metrics, 5)
  e.set_pipeline(compare).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(partials, 3, ML::Metal::BufferAccess::Write).set_value(count.to_u32, 4)
    .dispatch_threadgroups({groups, 1, 1}, {256, 1, 1})
  e.set_pipeline(fold).set_buffer(partials, 0)
    .set_buffer(total, 1, ML::Metal::BufferAccess::Write).set_value(groups.to_u32, 2)
    .dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
end

private def pos_compare_bytes(e : ML::Metal::ComputeEncoder,
                              zero : ML::Metal::ComputePipeline,
                              compare : ML::Metal::ComputePipeline,
                              actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                              metrics : ML::MetalBuffer, byte_count : Int32) : Nil
  pos_zero(e, zero, metrics, 1)
  e.set_pipeline(compare).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(byte_count.to_u32, 3)
    .dispatch_threadgroups({(byte_count + 255) // 256, 1, 1}, {256, 1, 1})
end

private def pos_repeat(e : ML::Metal::ComputeEncoder, pipeline : ML::Metal::ComputePipeline,
                       source : ML::MetalBuffer, destination : ML::MetalBuffer,
                       source_count : Int32, count : Int32) : Nil
  e.set_pipeline(pipeline).set_buffer(source, 0).set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
    .set_value(source_count.to_u32, 2).set_value(count.to_u32, 3).dispatch_1d(count)
end

private def pos_float_metrics(metrics : ML::MetalBuffer, total : ML::MetalBuffer, count : Int32)
  m = metrics.contents.as(UInt32*); s = total.contents.as(Float32*)
  raise "Nonfinite positional comparison reduction" unless (0...4).all? { |i| s[i].finite? }
  error = s[0].to_f64; reference = s[1].to_f64; actual = s[2].to_f64; dot = s[3].to_f64
  {compared_values: count, bit_differences: m[0], tolerance_violations: m[1], nonfinite: m[2],
   max_abs: m[3].unsafe_as(Float32), max_ratio: m[4].unsafe_as(Float32),
   rmse: Math.sqrt(error/count), relative_l2: Math.sqrt(error/Math.max(reference, 1e-300)),
   cosine: dot/Math.sqrt(Math.max(reference*actual, 1e-300))}
end

private def pos_project(e : ML::Metal::ComputeEncoder, plan : ML::Video::WanT5MPSLinearPlan,
                        output : ML::MetalBuffer, bias : ML::MetalBuffer) : Nil
  e.encode_wan_t5_mps_fc2(plan.handle)
  e.set_pipeline(ML::Video::MetalLinearMPS.pipeline).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(bias, 1).set_buffer(bias, 2).set_value(POS_DOUT.to_u32, 3)
    .set_value(0_u32, 4).set_value(POS_OUTPUTS.to_u32, 5).dispatch_1d(POS_OUTPUTS)
end

private def pos_run_timed_segment(kind : String,
                                  source_bf16 : ML::MetalBuffer,
                                  p4 : ML::MetalBuffer, p5 : ML::MetalBuffer,
                                  lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                                  plans : ML::MetalBuffer, offsets : ML::MetalBuffer,
                                  scan : ML::MetalBuffer, blob : ML::MetalBuffer,
                                  status : ML::MetalBuffer, stage : ML::MetalBuffer,
                                  plan : ML::Video::WanT5MPSLinearPlan,
                                  output : ML::MetalBuffer, bias : ML::MetalBuffer,
                                  tile_values : Int32,
                                  threshold : Float32, mode : Int32)
  command = ML::Metal::CommandBuffer.new
  encoder = ML::Metal::ComputeEncoder.new(command)
  before = ML::MetalBuffer.video_spec_allocations
  started = Time.instant
  case kind
  when "pack_only"
    pos_pack(encoder, source_bf16, p4, p5, lut4, lut5, plans, offsets, scan,
      blob, status, POS_WEIGHTS, tile_values, threshold, mode)
  when "positional_decode_projection"
    pos_decode(encoder, offsets, blob, lut4, lut5, stage, status, POS_WEIGHTS, tile_values)
    pos_project(encoder, plan, output, bias)
  when "bf16_expand_projection"
    encoder.set_pipeline(ML::Video::WanT5.pipeline("wan_t5_expand_fc2"))
      .set_buffer(source_bf16, 0).set_buffer(source_bf16, 1)
      .set_buffer(stage, 2, ML::Metal::BufferAccess::Write)
      .set_value(POS_WEIGHTS.to_u32, 3).set_value(1_u32, 4).dispatch_1d(POS_WEIGHTS)
    pos_project(encoder, plan, output, bias)
  else
    raise "Unknown timed positional segment: #{kind}"
  end
  encoder.end_encoding
  encode_seconds = (Time.instant - started).total_seconds
  gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
  wall_seconds = (Time.instant - started).total_seconds
  raise "Timed positional encode allocated a MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
  {encode_seconds: encode_seconds, gpu_seconds: gpu_seconds, wall_seconds: wall_seconds}
end

private def pos_pack(e : ML::Metal::ComputeEncoder, original : ML::MetalBuffer,
                     p4 : ML::MetalBuffer, p5 : ML::MetalBuffer,
                     lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                     plans : ML::MetalBuffer, offsets : ML::MetalBuffer,
                     scan : ML::MetalBuffer, blob : ML::MetalBuffer,
                     status : ML::MetalBuffer, count : Int32, tile_values : Int32,
                     threshold : Float32, mode : Int32) : Nil
  ML::Video::QBitPositionalWeights.pack(e, original, p4, p5, lut4, lut5,
    plans, offsets, scan, blob, status, count, tile_values, threshold, mode)
end

private def pos_decode(e : ML::Metal::ComputeEncoder, offsets : ML::MetalBuffer,
                       blob : ML::MetalBuffer, lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
                       destination : ML::MetalBuffer, status : ML::MetalBuffer,
                       count : Int32, tile_values : Int32) : Nil
  ML::Video::QBitPositionalWeights.decode(e, offsets, blob, lut4, lut5,
    destination, status, count, tile_values)
end

private def pos_record_stats(record : JSON::Any) : JSON::Any
  record["decoded_weight_stats"]
end

fixture_dir = POS_SOURCE_DIR
reference_dir = POS_REF_DIR
rounds = 4
OptionParser.parse do |parser|
  parser.on("--fixture=DIR", "Pinned four-row Wan block-0 projection fixture") { |value| fixture_dir = value }
  parser.on("--reference=DIR", "Independent positional-codec oracle fixture") { |value| reference_dir = value }
  parser.on("--positional-fixture=DIR", "Independent positional-codec oracle fixture") { |value| reference_dir = value }
  parser.on("--rounds=N", "Bounded timing samples per segment (must be 4)") { |value| rounds = value.to_i }
end
raise ArgumentError.new("--rounds must be 4") unless rounds == 4

fixture_manifest_path = File.join(fixture_dir, "manifest.json")
reference_manifest_path = File.join(reference_dir, "manifest.json")
raise "Missing positional source fixture manifest" unless File.file?(fixture_manifest_path)
raise "Missing positional oracle manifest" unless File.file?(reference_manifest_path)
fixture_hash = pos_sha256(fixture_manifest_path)
fixture = JSON.parse(File.read(fixture_manifest_path))
reference_hash = pos_sha256(reference_manifest_path)
reference = JSON.parse(File.read(reference_manifest_path))
raise "Unexpected positional oracle schema" unless reference["schema"].as_s == "wan_qbit_positional_weight_reference_v1"
raise "Positional oracle has wrong source fixture" unless reference["source_manifest_sha256"].as_s == fixture_hash
raise "Positional oracle source hash lineage mismatch" unless reference["source_hashes"]["source_fixture_manifest_sha256"].as_s == fixture_hash

geometry = reference["geometry"]
unless geometry["rows"].as_i == POS_ROWS && geometry["source_rows"].as_i == POS_SOURCE_ROWS &&
       geometry["input_dim"].as_i == POS_DIN && geometry["output_dim"].as_i == POS_DOUT &&
       geometry["normalization_block_size"].as_i == 256
  raise "Positional oracle geometry is outside the pinned FFN projection"
end
unless fixture["schema"].as_s == "wan_quantized_projection_reference_v1" &&
       fixture["geometry"]["source_rows"].as_i == POS_SOURCE_ROWS &&
       fixture["geometry"]["input_dim"].as_i == POS_DIN &&
       fixture["geometry"]["output_dim"].as_i == POS_DOUT &&
       fixture["input_provenance"]["real_video_derived"].as_bool == false
  raise "Input fixture is not the admitted four-row seeded capture"
end

source_tensor = fixture["source_tensor"]
weight_path = pos_verify_record(fixture_dir, source_tensor["weight_file"], "BF16", [POS_DOUT, POS_DIN])
source_f32_path = pos_verify_record(fixture_dir, fixture["variants"]["bf16"]["decoded_weight"], "F32", [POS_DOUT, POS_DIN])
pos_verify_record(fixture_dir, source_tensor["bias_bf16_file"], "BF16", [POS_DOUT])
bias_path = pos_verify_record(fixture_dir, source_tensor["bias_f32_file"], "F32", [POS_DOUT])
input_path = pos_verify_record(fixture_dir, fixture["input"], "F32", [POS_SOURCE_ROWS, POS_DIN])
p4_record = fixture["variants"]["qbit_p4"]["packed_weight"]
p5_record = fixture["variants"]["qbit_p5"]["packed_weight"]
lut4_record = fixture["variants"]["qbit_p4"]["lut"]
lut5_record = fixture["variants"]["qbit_p5"]["lut"]
p4_path = pos_verify_record(fixture_dir, p4_record, "QBitBlockBytes", [POS_DOUT, POS_DIN // 256, 136])
p5_path = pos_verify_record(fixture_dir, p5_record, "QBitBlockBytes", [POS_DOUT, POS_DIN // 256, 168])
lut4_path = pos_verify_record(fixture_dir, lut4_record, "F32", [256])
lut5_path = pos_verify_record(fixture_dir, lut5_record, "F32", [256])

source_hashes = reference["source_hashes"]
expected_source_hashes = {
  "checkpoint_weight_bf16_sha256"  => source_tensor["weight_file"]["sha256"].as_s,
  "decoded_bf16_source_f32_sha256" => fixture["variants"]["bf16"]["decoded_weight"]["sha256"].as_s,
  "input_f32_sha256"               => fixture["input"]["sha256"].as_s,
  "bias_bf16_sha256"               => source_tensor["bias_bf16_file"]["sha256"].as_s,
  "bias_f32_sha256"                => source_tensor["bias_f32_file"]["sha256"].as_s,
  "qbit_p4_payload_sha256"         => p4_record["sha256"].as_s,
  "qbit_p5_payload_sha256"         => p5_record["sha256"].as_s,
  "qbit_lut_p4_sha256"             => lut4_record["sha256"].as_s,
  "qbit_lut_p5_sha256"             => lut5_record["sha256"].as_s,
}
expected_source_hashes.each do |name, expected|
  raise "Positional oracle lineage hash mismatch: #{name}" unless source_hashes[name].as_s == expected
end

arms = reference["arms"].as_a
expected_keys = [] of Tuple(Int32, UInt32, Int32)
POS_TILES.each do |tile|
  POS_THRESHOLD_BITS.each do |threshold|
    {0, 1}.each { |mode| expected_keys << {tile, threshold.unsafe_as(UInt32), mode} }
  end
end
arm_keys = arms.map do |arm|
  tile = arm["tile_size"].as_i
  threshold = arm["threshold"].as_f.to_f32
  mode = arm["value_mode"].as_i
  unless POS_TILES.includes?(tile) && POS_THRESHOLD_BITS.any? { |known| known == threshold } && {0, 1}.includes?(mode)
    raise "Unexpected positional arm geometry/policy"
  end
  unless arm["precision_tie_order"].as_a.map(&.as_i) == [16, 5, 4]
    raise "Positional arm uses an unadmitted precision tie order"
  end
  {tile, threshold.unsafe_as(UInt32), mode}
end
raise "Positional oracle must contain all eighteen unique arms" unless arms.size == 18 && arm_keys.uniq.size == 18 && arm_keys.sort == expected_keys.sort

arms.each do |arm|
  tile = arm["tile_size"].as_i
  tile_count = POS_WEIGHTS // tile
  plan_path = pos_verify_record(reference_dir, arm["plan"], "U32LE", [tile_count, 4])
  offsets_path = pos_verify_record(reference_dir, arm["offsets"], "U32LE", [tile_count + 1])
  oracle_blob_end = pos_validate_plan_files(plan_path, offsets_path, tile_count, tile)
  blob_record = arm["blob"]
  blob_path = blob_record["path"].as_s
  pos_verify_record(reference_dir, blob_record, "U8", [blob_record["bytes"].as_i])
  raise "Oracle blob length disagrees with final offset: #{blob_path}" unless oracle_blob_end.to_i64 == blob_record["bytes"].as_i64
  unless blob_record["bytes"].as_i64 > 0 && blob_record["bytes"].as_i64 <= POS_WEIGHTS.to_i64 * 2 + tile_count.to_i64 * 4
    raise "Positional oracle blob exceeds raw BF16 upper bound"
  end
  pos_verify_record(reference_dir, arm["decoded_weight"], "F32", [POS_DOUT, POS_DIN])
  pos_verify_record(reference_dir, arm["projection"], "F32", [POS_SOURCE_ROWS, POS_DOUT])
  unless arm["plan_summary"]["selected_tile_counts"]["p4"].as_i +
           arm["plan_summary"]["selected_tile_counts"]["p5"].as_i +
           arm["plan_summary"]["selected_tile_counts"]["raw"].as_i == tile_count
    raise "Positional oracle plan histogram does not cover its tiles: #{blob_path}"
  end
end

arena = ML::Video::Arena.new
linear_plan : ML::Video::WanT5MPSLinearPlan? = nil
begin
  ML::Video::MetalLinearMPS.prepare!
  ML::Video::QBitPositionalWeights.prepare!
  source_bf16 = arena.alloc_weight("source_weight_bf16", POS_WEIGHTS, ML::Video::WeightDType::BF16)
  source_f32 = arena.alloc("source_weight_f32", POS_WEIGHTS)
  p4 = arena.alloc("source_qbit_p4", POS_WEIGHTS // 256 * 136 // 4)
  p5 = arena.alloc("source_qbit_p5", POS_WEIGHTS // 256 * 168 // 4)
  lut4 = arena.alloc("source_lut_p4", 256)
  lut5 = arena.alloc("source_lut_p5", 256)
  source_input = arena.alloc("seeded_input_source_4_rows", POS_SOURCE_ROWS * POS_DIN)
  input = arena.alloc("seeded_input_repeated_4480_rows", POS_ROWS * POS_DIN)
  bias = arena.alloc("captured_bias_f32", POS_DOUT)
  stage = arena.alloc("decoded_weight_stage", POS_WEIGHTS)
  expected_weight = arena.alloc("oracle_decoded_weight", POS_WEIGHTS)
  output = arena.alloc("projection_output", POS_OUTPUTS)
  oracle_projection_source = arena.alloc("oracle_projection_source_4_rows", POS_SOURCE_OUTPUTS)
  expected_output = arena.alloc("oracle_projection_repeated_4480_rows", POS_OUTPUTS)
  baseline_output = arena.alloc("bf16_projection", POS_OUTPUTS)

  max_tiles = POS_WEIGHTS // 256
  plans = arena.alloc("native_plans", max_tiles * ML::Video::QBitPositionalWeights::PLAN_WORDS)
  offsets = arena.alloc("native_offsets", max_tiles + 1)
  scan = arena.alloc("native_scan", (max_tiles + 255) // 256)
  max_blob_bytes = POS_WEIGHTS.to_i64 * 2 + max_tiles.to_i64 * 4
  blob = arena.alloc("native_blob_capacity", (max_blob_bytes // 4).to_i32)
  status = arena.alloc("native_status", ML::Video::QBitPositionalWeights::STATUS_WORDS)

  groups_weight = (POS_WEIGHTS + 255) // 256
  float_metrics = {} of String => ML::MetalBuffer
  float_partials = arena.alloc("float_compare_partials", groups_weight * 4)
  float_totals = {} of String => ML::MetalBuffer
  {"decoder", "weight_loss", "projection", "projection_loss"}.each do |name|
    float_metrics[name] = arena.alloc("metrics_#{name}", 5)
    float_totals[name] = arena.alloc("total_#{name}", 4)
  end
  byte_metrics = {} of String => ML::MetalBuffer
  {"plan", "offsets", "blob"}.each do |name|
    byte_metrics[name] = arena.alloc("byte_metrics_#{name}", 1)
  end
  expected_plan = arena.alloc("oracle_plan", max_tiles * ML::Video::QBitPositionalWeights::PLAN_WORDS)
  expected_offsets = arena.alloc("oracle_offsets", max_tiles + 1)
  max_oracle_blob = POS_WEIGHTS * 2 + max_tiles * 4
  expected_blob = arena.alloc("oracle_blob", max_oracle_blob // 4)
  # Dedicated small synthetic buffers keep adversarial controls independent of
  # the full matrix while exercising the exact production pack/decode entrypoints.
  ctrl_original = arena.alloc_weight("control_bf16", 1024, ML::Video::WeightDType::BF16)
  ctrl_p4 = arena.alloc("control_p4", 2 * 136 // 4)
  ctrl_p5 = arena.alloc("control_p5", 2 * 168 // 4)
  ctrl_lut4 = arena.alloc("control_lut4", 256)
  ctrl_lut5 = arena.alloc("control_lut5", 256)
  ctrl_plans = arena.alloc("control_plans", 4 * 4)
  ctrl_offsets = arena.alloc("control_offsets", 5)
  ctrl_scan = arena.alloc("control_scan", 1)
  ctrl_blob = arena.alloc("control_blob", 2048 // 4)
  ctrl_tiny_blob = arena.alloc("control_tiny_blob", 1)
  ctrl_status = arena.alloc("control_status", 8)
  ctrl_output = arena.alloc("control_output", 1024)
  ctrl_reference = arena.alloc("control_reference", 1024)
  ctrl_nonfinite = arena.alloc("control_nonfinite", 1)

  zero = ML::Metal::ComputePipeline.new("pos_zero_words", POS_PROBE_SOURCE)
  copy = ML::Metal::ComputePipeline.new("pos_copy_f32", POS_PROBE_SOURCE)
  repeat = ML::Metal::ComputePipeline.new("pos_repeat_f32", POS_PROBE_SOURCE)
  make_nan = ML::Metal::ComputePipeline.new("pos_make_bf16_nan", POS_PROBE_SOURCE)
  corrupt_byte = ML::Metal::ComputePipeline.new("pos_corrupt_byte", POS_PROBE_SOURCE)
  count_nonfinite = ML::Metal::ComputePipeline.new("pos_count_nonfinite", POS_PROBE_SOURCE)
  compare_bytes = ML::Metal::ComputePipeline.new("pos_compare_bytes", POS_PROBE_SOURCE)
  compare_float = ML::Metal::ComputePipeline.new("pos_compare_f32", POS_PROBE_SOURCE)
  fold_float = ML::Metal::ComputePipeline.new("pos_fold_f32", POS_PROBE_SOURCE)

  raise "Explicit native probe buffers exceed 3 GiB" if arena.resident_bytes > 3_i64 * 1024**3
  pos_upload(weight_path, source_bf16, POS_WEIGHTS.to_i64 * 2)
  pos_upload(source_f32_path, expected_weight, POS_WEIGHTS.to_i64 * 4)
  pos_upload(p4_path, p4, p4.size)
  pos_upload(p5_path, p5, p5.size)
  pos_upload(lut4_path, lut4, 1024_i64)
  pos_upload(lut5_path, lut5, 1024_i64)
  pos_upload(input_path, source_input, POS_SOURCE_ROWS.to_i64 * POS_DIN * 4)
  pos_upload(bias_path, bias, POS_DOUT.to_i64 * 4)

  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
    layers: 1, batches: 1, tokens: POS_ROWS)
  linear_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, POS_DIN, POS_DOUT, input, stage, output)
  setup = ML::Metal::CommandBuffer.new
  setup_encoder = ML::Metal::ComputeEncoder.new(setup)
  setup_encoder.set_pipeline(ML::Video::WanT5.pipeline("wan_t5_expand_fc2"))
    .set_buffer(source_bf16, 0).set_buffer(source_bf16, 1)
    .set_buffer(source_f32, 2, ML::Metal::BufferAccess::Write)
    .set_value(POS_WEIGHTS.to_u32, 3).set_value(1_u32, 4).dispatch_1d(POS_WEIGHTS)
  pos_compare_bytes(setup_encoder, zero, compare_bytes, source_f32, expected_weight,
    byte_metrics["plan"], POS_WEIGHTS * 4)
  setup_encoder.set_pipeline(copy).set_buffer(source_f32, 0)
    .set_buffer(stage, 1, ML::Metal::BufferAccess::Write)
    .set_value(POS_WEIGHTS.to_u32, 2).dispatch_1d(POS_WEIGHTS)
  pos_repeat(setup_encoder, repeat, source_input, input,
    POS_SOURCE_ROWS * POS_DIN, POS_ROWS * POS_DIN)
  pos_project(setup_encoder, linear_plan.not_nil!, output, bias)
  setup_encoder.set_pipeline(copy).set_buffer(output, 0)
    .set_buffer(baseline_output, 1, ML::Metal::BufferAccess::Write)
    .set_value(POS_OUTPUTS.to_u32, 2).dispatch_1d(POS_OUTPUTS)
  setup_encoder.end_encoding
  setup.commit
  setup.wait
  source_expansion_differences = byte_metrics["plan"].contents.as(UInt32*)[0]
  raise "Native BF16 expansion differs from fixture reference" unless source_expansion_differences == 0

  puts({schema: "wan_qbit_positional_probe_setup_v1", source_manifest_sha256: fixture_hash,
        reference_manifest_sha256: reference_hash, rows: POS_ROWS, repeated_reference_rows: geometry["rows"].as_i,
        input_scope: "four seeded upstream expected.ffn_mod rows tiled #{POS_REPETITIONS} times; not real-video denoising activations",
        explicit_resident_capacity_bytes: arena.resident_bytes, resident_limit_bytes: 3_i64 * 1024**3,
        memory_guard_required_free_pct: 15, max_swap_growth_mib: 64,
        bf16_raw_bytes: POS_WEIGHTS.to_i64 * 2, p4_payload_bytes: p4.size,
        p5_payload_bytes: p5.size, resident_blob_capacity_bytes: blob.size,
        bf16_expansion_vs_fixture_bit_differences: source_expansion_differences,
        oracle_decoded_weight_bytes_per_arm: POS_WEIGHTS.to_i64 * 4,
        global_container_descriptor_included: false, timing_claim: false,
        projection_output_bytes: POS_OUTPUTS.to_i64 * 4,
        activation_readbacks: 0, inter_operator_host_waits: 0}.to_json)
  STDOUT.flush

  # Host-side negative controls reject malformed public API parameters without
  # submitting work. The guard encoder is deliberately never committed.
  rejected_host_calls = 0
  guard_command = ML::Metal::CommandBuffer.new
  guard_encoder = ML::Metal::ComputeEncoder.new(guard_command)
  begin
    pos_pack(guard_encoder, source_bf16, p4, p5, lut4, lut5, plans, offsets, scan, blob, status,
      POS_WEIGHTS, 257, 0.1_f32, 0)
  rescue ArgumentError
    rejected_host_calls += 1
  end
  begin
    pos_pack(guard_encoder, source_bf16, p4, p5, lut4, lut5, plans, offsets, scan, blob, status,
      POS_WEIGHTS, 256, -0.1_f32, 0)
  rescue ArgumentError
    rejected_host_calls += 1
  end
  begin
    pos_pack(guard_encoder, source_bf16, p4, p5, lut4, lut5, plans, offsets, scan, blob, status,
      POS_WEIGHTS, 256, 0.1_f32, 2)
  rescue ArgumentError
    rejected_host_calls += 1
  end
  begin
    pos_decode(guard_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5, ctrl_output, ctrl_status, 257, 256)
  rescue ArgumentError
    rejected_host_calls += 1
  end
  guard_encoder.end_encoding
  raise "Native positional host API accepted an invalid argument" unless rejected_host_calls == 4

  # Comparator adversary: a finite +1 defect and NaN must both register as
  # tolerance violations, with NaN separately counted as nonfinite.
  ctrl_output.contents.as(Float32*).to_slice(2).fill(0.0_f32)
  ctrl_reference.contents.as(Float32*).to_slice(2).fill(0.0_f32)
  ctrl_output.contents.as(Float32*)[0] = 1.0_f32
  ctrl_output.contents.as(Float32*)[1] = Float32::NAN
  comparator_command = ML::Metal::CommandBuffer.new
  comparator_encoder = ML::Metal::ComputeEncoder.new(comparator_command)
  pos_compare_float(comparator_encoder, zero, compare_float, fold_float, ctrl_output,
    ctrl_reference, float_metrics["decoder"], float_partials, float_totals["decoder"], 2)
  comparator_encoder.end_encoding; comparator_command.commit; comparator_command.wait
  comparator_result = pos_float_metrics(float_metrics["decoder"], float_totals["decoder"], 2)
  raise "Comparator failed +1/NaN adversary" unless comparator_result[:bit_differences] == 2 &&
                                                    comparator_result[:tolerance_violations] == 2 && comparator_result[:nonfinite] == 1
  puts({schema: "wan_qbit_positional_control_v1", control: "comparator_plus_one_and_nan",
        comparator: comparator_result, passed: true}.to_json)
  STDOUT.flush

  # Boundary control: error/sigma == threshold is excluded by the strict >
  # escape predicate. Equal-cost records choose raw, but p4 is strictly cheaper.
  ctrl_original.contents.as(UInt16*).to_slice(256).fill(0_u16)
  ctrl_original.contents.as(UInt16*)[0] = 0x3f00_u16 # BF16 0.5
  ctrl_p4.contents.as(UInt8*).to_slice(272).fill(0_u8)
  ctrl_p5.contents.as(UInt8*).to_slice(336).fill(0_u8)
  pos_write_u32le(ctrl_p4, 1, 0x3f800000_u32)
  pos_write_u32le(ctrl_p5, 1, 0x3f800000_u32)
  ctrl_lut4.contents.as(Float32*).to_slice(256).fill(0.0_f32)
  ctrl_lut5.contents.as(Float32*).to_slice(256).fill(0.0_f32)
  boundary_command = ML::Metal::CommandBuffer.new
  boundary_encoder = ML::Metal::ComputeEncoder.new(boundary_command)
  pos_pack(boundary_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_blob, ctrl_status, 256, 256, 0.5_f32, 0)
  pos_decode(boundary_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5,
    ctrl_output, ctrl_status, 256, 256)
  boundary_encoder.end_encoding; boundary_command.commit; boundary_command.wait
  boundary_plan = ctrl_plans.contents.as(UInt32*)
  boundary_status = ctrl_status.contents.as(UInt32*)
  raise "Exact threshold boundary selected a correction or wrong tier" unless boundary_plan[1] == 4 && boundary_plan[2] == 0 && boundary_status[1] == 1 &&
                                                                              boundary_status[4] == 0 && boundary_status[5] == 0 && boundary_status[6] == 0 && boundary_status[7] == 0
  puts({schema: "wan_qbit_positional_control_v1", control: "strict_threshold_equality",
        threshold: 0.5, normalized_error: 0.5, selected_precision: boundary_plan[1],
        corrections: boundary_plan[2], passed: true}.to_json)
  STDOUT.flush

  # A cost tie between p4 and raw must retain raw because p4 is not strictly
  # cheaper. For 125 corrections in a 256-value tile both records cost 516 B.
  ctrl_original.contents.as(UInt16*).to_slice(256).fill(0_u16)
  125.times { |i| ctrl_original.contents.as(UInt16*)[i] = 0x3f00_u16 }
  ctrl_p4.contents.as(UInt8*).to_slice(272).fill(0_u8)
  ctrl_p5.contents.as(UInt8*).to_slice(336).fill(0_u8)
  ctrl_lut4.contents.as(Float32*).to_slice(256).fill(0.0_f32)
  ctrl_lut5.contents.as(Float32*).to_slice(256).fill(0.0_f32)
  tie_command = ML::Metal::CommandBuffer.new
  tie_encoder = ML::Metal::ComputeEncoder.new(tie_command)
  pos_pack(tie_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_blob, ctrl_status, 256, 256, 0.1_f32, 0)
  pos_decode(tie_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5,
    ctrl_output, ctrl_status, 256, 256)
  tie_encoder.end_encoding; tie_command.commit; tie_command.wait
  tie_plan = ctrl_plans.contents.as(UInt32*)
  tie_status = ctrl_status.contents.as(UInt32*)
  raise "Equal-cost p4/raw candidates did not retain raw" unless tie_plan[1] == 16 &&
                                                                 tie_plan[2] == 0 && tie_plan[0] == 516 && tie_status[0] == 516 &&
                                                                 tie_status[3] == 1 && tie_status[5] == 0 && tie_status[6] == 0 && tie_status[7] == 0
  puts({schema: "wan_qbit_positional_control_v1", control: "strict_cost_tie_prefers_raw",
        p4_record_bytes: 516, raw_record_bytes: 516, selected_precision: tie_plan[1], passed: true}.to_json)
  STDOUT.flush

  # With sigma zero, a nonzero reconstruction error is always an escape.
  ctrl_original.contents.as(UInt16*).to_slice(256).fill(0_u16)
  ctrl_original.contents.as(UInt16*)[0] = 0x3f00_u16
  pos_write_u32le(ctrl_p4, 1, 0_u32)
  pos_write_u32le(ctrl_p5, 1, 0_u32)
  sigma_command = ML::Metal::CommandBuffer.new
  sigma_encoder = ML::Metal::ComputeEncoder.new(sigma_command)
  pos_pack(sigma_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_blob, ctrl_status, 256, 256, 0.1_f32, 0)
  pos_decode(sigma_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5,
    ctrl_output, ctrl_status, 256, 256)
  sigma_encoder.end_encoding; sigma_command.commit; sigma_command.wait
  sigma_plan = ctrl_plans.contents.as(UInt32*)
  sigma_status = ctrl_status.contents.as(UInt32*)
  sigma_decoded = ctrl_output.contents.as(Float32*)
  raise "Zero-sigma nonzero error did not become a p4 correction" unless sigma_plan[1] == 4 &&
                                                                         sigma_plan[2] == 1 && sigma_status[4] == 1 && sigma_status[5] == 0 &&
                                                                         sigma_status[6] == 0 && sigma_status[7] == 0 && sigma_decoded[0] == 0.5_f32
  puts({schema: "wan_qbit_positional_control_v1", control: "zero_sigma_escape",
        selected_precision: sigma_plan[1], corrections: sigma_plan[2], decoded_first_value: sigma_decoded[0],
        passed: true}.to_json)
  STDOUT.flush

  # Nonzero padding must invalidate the whole record before any values escape.
  padding_command = ML::Metal::CommandBuffer.new
  padding_encoder = ML::Metal::ComputeEncoder.new(padding_command)
  pos_pack(padding_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_blob, ctrl_status, 256, 256, 0.1_f32, 0)
  padding_encoder.set_pipeline(corrupt_byte).set_buffer(ctrl_blob, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_value(143_u32, 1).dispatch_1d(1)
  pos_decode(padding_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5,
    ctrl_output, ctrl_status, 256, 256)
  pos_zero(padding_encoder, zero, ctrl_nonfinite, 1)
  padding_encoder.set_pipeline(count_nonfinite).set_buffer(ctrl_output, 0)
    .set_buffer(ctrl_nonfinite, 1, ML::Metal::BufferAccess::ReadWrite)
    .set_value(256_u32, 2).dispatch_1d(256)
  padding_encoder.end_encoding; padding_command.commit; padding_command.wait
  padding_status = ctrl_status.contents.as(UInt32*)
  padding_nan_count = ctrl_nonfinite.contents.as(UInt32*)[0]
  raise "Nonzero record padding was accepted" unless padding_status[7] == 1 && padding_nan_count == 256
  puts({schema: "wan_qbit_positional_control_v1", control: "nonzero_padding_fails_closed",
        decode_invalid_tiles: padding_status[7], all_nan_values: padding_nan_count, passed: true}.to_json)
  STDOUT.flush

  # Strong stale-buffer adversary: write a valid record, poison the source,
  # re-pack into the same blob (which leaves the previous bytes behind on the
  # invalid tile), and decode without an intermediate wait. Global pack status
  # must force NaNs rather than expose stale valid bytes.
  ctrl_original.contents.as(UInt16*).to_slice(256).fill(0_u16)
  ctrl_original.contents.as(UInt16*)[0] = 0x3f00_u16
  stale_command = ML::Metal::CommandBuffer.new
  stale_encoder = ML::Metal::ComputeEncoder.new(stale_command)
  pos_pack(stale_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_blob, ctrl_status, 256, 256, 0.5_f32, 0)
  stale_encoder.set_pipeline(make_nan).set_buffer(ctrl_original, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  pos_pack(stale_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_blob, ctrl_status, 256, 256, 0.5_f32, 0)
  pos_decode(stale_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5,
    ctrl_output, ctrl_status, 256, 256)
  pos_zero(stale_encoder, zero, ctrl_nonfinite, 1)
  stale_encoder.set_pipeline(count_nonfinite).set_buffer(ctrl_output, 0)
    .set_buffer(ctrl_nonfinite, 1, ML::Metal::BufferAccess::ReadWrite)
    .set_value(256_u32, 2).dispatch_1d(256)
  stale_encoder.end_encoding; stale_command.commit; stale_command.wait
  stale_status = ctrl_status.contents.as(UInt32*)
  stale_plan = ctrl_plans.contents.as(UInt32*)
  stale_header = ctrl_blob.contents.as(UInt8*)
  stale_nan_count = ctrl_nonfinite.contents.as(UInt32*)[0]
  raise "Failed-pack stale blob was not rejected" unless stale_status[5] == 1 &&
                                                         stale_status[6] == 0 && stale_status[7] == 1 && stale_plan[3] == 1 &&
                                                         stale_header[0] == 4 && stale_nan_count == 256
  puts({schema: "wan_qbit_positional_control_v1", control: "invalid_pack_rejects_stale_blob",
        valid_blob_seeded: true, invalid_repack_same_encoder: true, old_header_precision: stale_header[0],
        pack_invalid_tiles: stale_status[5], decode_invalid_tiles: stale_status[7],
        all_nan_values: stale_nan_count, terminal_waits: 1, passed: true}.to_json)
  STDOUT.flush

  # Capacity failure is another fail-closed path. The buffer is the minimum
  # admitted size (four bytes), while the selected complete record is larger.
  ctrl_original.contents.as(UInt16*).to_slice(256).fill(0_u16)
  ctrl_original.contents.as(UInt16*)[0] = 0x3f00_u16
  capacity_command = ML::Metal::CommandBuffer.new
  capacity_encoder = ML::Metal::ComputeEncoder.new(capacity_command)
  pos_pack(capacity_encoder, ctrl_original, ctrl_p4, ctrl_p5, ctrl_lut4, ctrl_lut5,
    ctrl_plans, ctrl_offsets, ctrl_scan, ctrl_tiny_blob, ctrl_status, 256, 256, 0.5_f32, 0)
  pos_decode(capacity_encoder, ctrl_offsets, ctrl_tiny_blob, ctrl_lut4, ctrl_lut5,
    ctrl_output, ctrl_status, 256, 256)
  pos_zero(capacity_encoder, zero, ctrl_nonfinite, 1)
  capacity_encoder.set_pipeline(count_nonfinite).set_buffer(ctrl_output, 0)
    .set_buffer(ctrl_nonfinite, 1, ML::Metal::BufferAccess::ReadWrite)
    .set_value(256_u32, 2).dispatch_1d(256)
  capacity_encoder.end_encoding; capacity_command.commit; capacity_command.wait
  capacity_status = ctrl_status.contents.as(UInt32*)
  capacity_nan_count = ctrl_nonfinite.contents.as(UInt32*)[0]
  raise "Capacity failure did not fail closed" unless capacity_status[6] == 1 &&
                                                      capacity_status[7] == 1 && capacity_nan_count == 256
  puts({schema: "wan_qbit_positional_control_v1", control: "undersized_blob_capacity",
        capacity_bytes: ctrl_tiny_blob.size, capacity_failures: capacity_status[6],
        decode_invalid_tiles: capacity_status[7], all_nan_values: capacity_nan_count, passed: true}.to_json)
  STDOUT.flush

  # Decoder corruption controls cover header, bounds, position and order. Each
  # gets a dedicated tiny one-tile command and checks status plus GPU NaN count.
  malformed = ["bad_precision", "out_of_range_offset", "position_out_of_range", "positions_not_increasing"]
  malformed.each do |kind|
    tile_values = kind == "position_out_of_range" || kind == "positions_not_increasing" ? 512 : 256
    tile_count = tile_values
    record_bytes = case kind
                   when "bad_precision"            then 4
                   when "out_of_range_offset"      then 4
                   when "position_out_of_range"    then 280
                   when "positions_not_increasing" then 284
                   else                                 raise "Unknown malformed control"
                   end
    ctrl_blob.contents.as(UInt8*).to_slice(2048).fill(0_u8)
    ctrl_offsets.contents.as(UInt32*).to_slice(2).fill(0_u32)
    if kind == "out_of_range_offset"
      pos_write_u32le(ctrl_offsets, 1, 4096_u32)
    else
      pos_write_u32le(ctrl_offsets, 1, record_bytes.to_u32)
      header = ctrl_blob.contents.as(UInt8*)
      header[0] = kind == "bad_precision" ? 6_u8 : 4_u8
      header[1] = 0_u8
      header[2] = kind == "positions_not_increasing" ? 2_u8 : kind == "position_out_of_range" ? 1_u8 : 0_u8
      header[3] = 0_u8
      if kind != "bad_precision"
        pos_write_u32le(ctrl_blob, 2, 0x3f800000_u32) # sigma of first base block
        if tile_values == 512
          pos_write_u32le(ctrl_blob, (4 + 136) // 4 + 1, 0x3f800000_u32)
          if kind == "position_out_of_range"
            pos_write_u16le(ctrl_blob, 4 + 272, 512_u16)
          else
            pos_write_u16le(ctrl_blob, 4 + 272, 2_u16)
            pos_write_u16le(ctrl_blob, 4 + 276, 1_u16)
          end
        end
      end
    end
    ctrl_status.contents.as(UInt32*).to_slice(8).fill(0_u32)
    malformed_command = ML::Metal::CommandBuffer.new
    malformed_encoder = ML::Metal::ComputeEncoder.new(malformed_command)
    pos_decode(malformed_encoder, ctrl_offsets, ctrl_blob, ctrl_lut4, ctrl_lut5,
      ctrl_output, ctrl_status, tile_count, tile_values)
    pos_zero(malformed_encoder, zero, ctrl_nonfinite, 1)
    malformed_encoder.set_pipeline(count_nonfinite).set_buffer(ctrl_output, 0)
      .set_buffer(ctrl_nonfinite, 1, ML::Metal::BufferAccess::ReadWrite)
      .set_value(tile_values.to_u32, 2).dispatch_1d(tile_values)
    malformed_encoder.end_encoding; malformed_command.commit; malformed_command.wait
    bad_status = ctrl_status.contents.as(UInt32*)[7]
    nan_count = ctrl_nonfinite.contents.as(UInt32*)[0]
    raise "Malformed tile was partially decoded: #{kind}" unless bad_status == 1 && nan_count == tile_values
    puts({schema: "wan_qbit_positional_control_v1", control: "malformed_#{kind}",
          invalid_tiles: bad_status, all_nan_values: nan_count, passed: true}.to_json)
    STDOUT.flush
  end

  # Four live diagnostics per geometry/policy combination. GPU byte output is
  # produced natively; oracle bytes are only read into the comparator, never
  # passed to the native decoder as its input.
  arms.each do |arm|
    tile_values = arm["tile_size"].as_i
    threshold = arm["threshold"].as_f.to_f32
    mode = arm["value_mode"].as_i
    tile_count = POS_WEIGHTS // tile_values
    plan_bytes = tile_count * 4 * 4
    offset_bytes = (tile_count + 1) * 4
    blob_bytes = arm["blob"]["bytes"].as_i
    pos_upload(pos_verify_record(reference_dir, arm["plan"], "U32LE", [tile_count, 4]), expected_plan, plan_bytes.to_i64)
    pos_upload(pos_verify_record(reference_dir, arm["offsets"], "U32LE", [tile_count + 1]), expected_offsets, offset_bytes.to_i64)
    pos_upload(pos_verify_record(reference_dir, arm["blob"], "U8", [blob_bytes]), expected_blob, blob_bytes.to_i64)
    pos_upload(pos_verify_record(reference_dir, arm["decoded_weight"], "F32", [POS_DOUT, POS_DIN]), expected_weight, POS_WEIGHTS.to_i64 * 4)
    pos_upload(pos_verify_record(reference_dir, arm["projection"], "F32", [POS_SOURCE_ROWS, POS_DOUT]),
      oracle_projection_source, POS_SOURCE_OUTPUTS.to_i64 * 4)

    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    before = ML::MetalBuffer.video_spec_allocations
    pos_pack(encoder, source_bf16, p4, p5, lut4, lut5, plans, offsets, scan, blob,
      status, POS_WEIGHTS, tile_values, threshold, mode)
    pos_repeat(encoder, repeat, oracle_projection_source, expected_output,
      POS_SOURCE_OUTPUTS, POS_OUTPUTS)
    pos_compare_bytes(encoder, zero, compare_bytes, plans, expected_plan, byte_metrics["plan"], plan_bytes)
    pos_compare_bytes(encoder, zero, compare_bytes, offsets, expected_offsets, byte_metrics["offsets"], offset_bytes)
    pos_compare_bytes(encoder, zero, compare_bytes, blob, expected_blob, byte_metrics["blob"], blob_bytes)
    pos_decode(encoder, offsets, blob, lut4, lut5, stage, status, POS_WEIGHTS, tile_values)
    pos_compare_float(encoder, zero, compare_float, fold_float, stage, expected_weight,
      float_metrics["decoder"], float_partials, float_totals["decoder"], POS_WEIGHTS)
    pos_compare_float(encoder, zero, compare_float, fold_float, stage, source_f32,
      float_metrics["weight_loss"], float_partials, float_totals["weight_loss"], POS_WEIGHTS)
    pos_project(encoder, linear_plan.not_nil!, output, bias)
    pos_compare_float(encoder, zero, compare_float, fold_float, output, expected_output,
      float_metrics["projection"], float_partials, float_totals["projection"], POS_OUTPUTS)
    pos_compare_float(encoder, zero, compare_float, fold_float, output, baseline_output,
      float_metrics["projection_loss"], float_partials, float_totals["projection_loss"], POS_OUTPUTS)
    encoder.end_encoding; command.commit; command.wait
    raise "Correctness encode allocated a MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before

    status_words = status.contents.as(UInt32*)
    p4_count = status_words[1]; p5_count = status_words[2]; raw_count = status_words[3]
    corrections = status_words[4]
    summary = arm["plan_summary"]
    expected_counts = summary["selected_tile_counts"]
    raise "Native pack status reports invalid plans/capacity/decode" unless status_words[5] == 0 && status_words[6] == 0 && status_words[7] == 0
    raise "Native tier histogram differs from independent oracle" unless p4_count.to_i64 == expected_counts["p4"].as_i64 &&
                                                                         p5_count.to_i64 == expected_counts["p5"].as_i64 &&
                                                                         raw_count.to_i64 == expected_counts["raw"].as_i64 &&
                                                                         corrections.to_i64 == summary["correction_count"].as_i64
    raise "Native packed bytes differ from oracle logical length" unless status_words[0].to_i64 == blob_bytes.to_i64
    byte_differences = {"plan"    => byte_metrics["plan"].contents.as(UInt32*)[0],
                        "offsets" => byte_metrics["offsets"].contents.as(UInt32*)[0],
                        "blob"    => byte_metrics["blob"].contents.as(UInt32*)[0]}
    raise "Native packed wire differs from oracle: #{byte_differences}" unless byte_differences.values.all?(&.zero?)
    decoded = pos_float_metrics(float_metrics["decoder"], float_totals["decoder"], POS_WEIGHTS)
    projected = pos_float_metrics(float_metrics["projection"], float_totals["projection"], POS_OUTPUTS)
    weight_loss = pos_float_metrics(float_metrics["weight_loss"], float_totals["weight_loss"], POS_WEIGHTS)
    projection_loss = pos_float_metrics(float_metrics["projection_loss"], float_totals["projection_loss"], POS_OUTPUTS)
    raise "Native decoded weights do not match independent oracle" unless decoded[:tolerance_violations] == 0 && decoded[:nonfinite] == 0
    raise "Native projection does not match independent oracle" unless projected[:tolerance_violations] == 0 && projected[:nonfinite] == 0

    offset_table_bytes = offset_bytes
    tile_record_header_bytes = tile_count.to_i64 * 4
    blob_and_offsets_bytes = status_words[0].to_i64 + offset_table_bytes
    raw_bf16_payload_bytes = POS_WEIGHTS.to_i64 * 2
    raw_blob_and_offsets_bytes = raw_bf16_payload_bytes + tile_record_header_bytes + offset_table_bytes
    puts({schema: "wan_qbit_positional_probe_result_v1", arm: arm["name"].as_s,
          tile_values: tile_values, threshold: threshold, value_mode: mode,
          projection_rows: POS_ROWS, source_rows: POS_SOURCE_ROWS, input_repeat_factor: POS_REPETITIONS,
          selected_tiles: {p4: p4_count, p5: p5_count, raw: raw_count}, corrections: corrections,
          blob_logical_bytes: status_words[0], offset_table_bytes: offset_table_bytes,
          tile_record_header_bytes_in_raw_comparison: tile_record_header_bytes,
          blob_and_offsets_bytes: blob_and_offsets_bytes,
          blob_and_offsets_bits_per_weight: blob_and_offsets_bytes.to_f64 * 8.0 / POS_WEIGHTS,
          raw_bf16_payload_bytes: raw_bf16_payload_bytes,
          raw_blob_and_offsets_bytes: raw_blob_and_offsets_bytes,
          resident_blob_capacity_bytes: blob.size,
          resident_offsets_capacity_bytes: offsets.size,
          global_container_descriptor_included: false,
          native_vs_oracle_exact_byte_differences: byte_differences,
          decoded_weight_vs_oracle: decoded, decoded_weight_vs_bf16: weight_loss,
          projection_vs_oracle: projected, projection_vs_bf16: projection_loss,
          oracle_decoded_weight_stats: pos_record_stats(arm),
          f16_policy_misses: summary["f16_policy_misses"],
          encode_metal_buffer_allocations: 0, terminal_waits: 1,
          inter_operator_host_waits: 0, activation_readbacks: 0,
          native_payload_input: "source BF16 + canonical p4/p5 payloads; oracle wire used only for exact comparison"}.to_json)
    STDOUT.flush
  end

  # Timing is limited to the smallest BF16-value-mode arm per tile geometry,
  # plus the smallest F16-delta arm at tile size 512. Correctness stays outside
  # timed command buffers; each paired schedule is ABBA then BAAB.
  timing_arms = [] of JSON::Any
  POS_TILES.each do |tile|
    selected = arms.select { |arm| arm["tile_size"].as_i == tile && arm["value_mode"].as_i == 0 }
      .min_by { |arm| arm["blob"]["bytes"].as_i }
    timing_arms << (selected || raise "Missing BF16 timing arm for tile #{tile}")
  end
  delta_timing_arm = arms.select { |arm| arm["tile_size"].as_i == 512 && arm["value_mode"].as_i == 1 }
    .min_by { |arm| arm["blob"]["bytes"].as_i }
  timing_arms << (delta_timing_arm || raise "Missing F16-delta timing arm")
  timing_arms.each do |arm|
    tile_values = arm["tile_size"].as_i
    threshold = arm["threshold"].as_f.to_f32
    mode = arm["value_mode"].as_i
    tile_count = POS_WEIGHTS // tile_values
    pos_upload(pos_verify_record(reference_dir, arm["plan"], "U32LE", [tile_count, 4]), expected_plan, (tile_count * 16).to_i64)
    pos_upload(pos_verify_record(reference_dir, arm["offsets"], "U32LE", [tile_count + 1]), expected_offsets, ((tile_count + 1) * 4).to_i64)
    blob_bytes = arm["blob"]["bytes"].as_i
    pos_upload(pos_verify_record(reference_dir, arm["blob"], "U8", [blob_bytes]), expected_blob, blob_bytes.to_i64)
    # Warm each distinct segment once, then take four matched observations of
    # each segment in two counterbalanced ABBA/BAAB quartets. Pack timing is
    # separate from projection timing and is sampled four times.
    {"pack_only", "positional_decode_projection", "bf16_expand_projection"}.each do |kind|
      sample = pos_run_timed_segment(kind, source_bf16, p4, p5, lut4, lut5,
        plans, offsets, scan, blob, status, stage, linear_plan.not_nil!, output,
        bias, tile_values, threshold, mode)
      puts({schema: "wan_qbit_positional_probe_timing_v1", arm: arm["name"].as_s,
            tile_values: tile_values, threshold: threshold, value_mode: mode,
            sample_kind: "warmup", segment: kind, encode_seconds: sample[:encode_seconds],
            gpu_seconds: sample[:gpu_seconds], wall_seconds: sample[:wall_seconds],
            terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
            encode_metal_buffer_allocations: 0, measured: false}.to_json)
      STDOUT.flush
    end

    rounds.times do |round|
      sample = pos_run_timed_segment("pack_only", source_bf16, p4, p5, lut4, lut5,
        plans, offsets, scan, blob, status, stage, linear_plan.not_nil!, output,
        bias, tile_values, threshold, mode)
      puts({schema: "wan_qbit_positional_probe_timing_v1", arm: arm["name"].as_s,
            tile_values: tile_values, threshold: threshold, value_mode: mode, round: round,
            segment: "pack_only", encode_seconds: sample[:encode_seconds],
            gpu_seconds: sample[:gpu_seconds], wall_seconds: sample[:wall_seconds],
            terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
            encode_metal_buffer_allocations: 0, measured: true,
            correctness_outside_timed_segment: true}.to_json)
      STDOUT.flush
    end

    projection_order = ["positional_decode_projection", "bf16_expand_projection",
                        "bf16_expand_projection", "positional_decode_projection",
                        "bf16_expand_projection", "positional_decode_projection",
                        "positional_decode_projection", "bf16_expand_projection"]
    projection_order.each_with_index do |kind, position|
      sample = pos_run_timed_segment(kind, source_bf16, p4, p5, lut4, lut5,
        plans, offsets, scan, blob, status, stage, linear_plan.not_nil!, output,
        bias, tile_values, threshold, mode)
      puts({schema: "wan_qbit_positional_probe_timing_v1", arm: arm["name"].as_s,
            tile_values: tile_values, threshold: threshold, value_mode: mode,
            quartet: position // 4, order: position < 4 ? "ABBA" : "BAAB",
            pair: position // 2, position: position % 4, segment: kind,
            encode_seconds: sample[:encode_seconds], gpu_seconds: sample[:gpu_seconds],
            wall_seconds: sample[:wall_seconds], terminal_waits: 1,
            inter_operator_host_waits: 0, activation_readbacks: 0,
            encode_metal_buffer_allocations: 0, measured: true,
            correctness_outside_timed_segment: true}.to_json)
      STDOUT.flush
    end
  end

  puts({schema: "wan_qbit_positional_probe_controls_v1", host_api_rejections: rejected_host_calls,
        malformed_decoder_cases: malformed, stale_blob_case_passed: true,
        capacity_failure_case_passed: true, equality_boundary_case_passed: true,
        all_native_arms_passed: true, timing_arms: timing_arms.map { |arm| arm["name"].as_s },
        reference_rows: geometry["rows"].as_i, projection_rows_executed: POS_ROWS,
        generated_video_activations: false, full_generation: false}.to_json)
ensure
  linear_plan.try(&.release)
  arena.release
end
