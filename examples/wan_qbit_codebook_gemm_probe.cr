# Opt-in native local-codebook GEMM diagnostic for a real Wan projection.
# Inputs are four seeded upstream rows; full-shape timing repeats them on GPU.
# Use the existing 15% memory guard. No model loading or generation.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_linear_mps"
require "../src/ml/video/qbit_positional_weights"
require "../src/ml/video/qbit_codebook_linear"
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

private CG_METAL = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void cg_zero(device uint *values [[buffer(0)]], constant uint &n [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) { if (i<n) values[i]=0u; }
  kernel void cg_repeat(device const float *source [[buffer(0)]], device float *out [[buffer(1)]],
      constant uint &period [[buffer(2)]], constant uint &n [[buffer(3)]],
      uint i [[thread_position_in_grid]]) { if (i<n) out[i]=source[i%period]; }
  kernel void cg_seed(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
      uint i [[thread_position_in_grid]]) {
    if (i<2u) { a[i]=i==0u?1.0f:as_type<float>(0x7fc00000u); b[i]=0.0f; }
  }
  inline void cg_u16(device uchar *p, uint value) { p[0]=uchar(value);p[1]=uchar(value>>8u); }
  inline void cg_u32(device uchar *p, uint value) {
    for(uint j=0u;j<4u;++j)p[j]=uchar(value>>(8u*j));
  }
  // Nine output channels cover p4, p5 and raw BF16, both correction modes,
  // correction endpoints, and a partial output tile. No host-side decoding.
  kernel void cg_synthetic_blob(device uchar *blob [[buffer(0)]],
      device uint *offsets [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if(i!=0u)return;
    uint at=0u;
    for(uint tile=0u;tile<9u;++tile) {
      offsets[tile]=at;
      uint precision=tile%3u==0u?4u:(tile%3u==1u?5u:16u);
      uint mode=(tile/3u)&1u;
      uint corrections=precision==16u?0u:3u;
      uint payload=precision==16u?512u:(precision==4u?136u:168u);
      uint bytes=(4u+payload+corrections*3u+3u)&~3u;
      for(uint j=0u;j<bytes;++j)blob[at+j]=0u;
      blob[at]=uchar(precision);blob[at+1u]=uchar(mode);cg_u16(blob+at+2u,corrections);
      if(precision==16u) {
        for(uint j=0u;j<256u;++j)cg_u16(blob+at+4u+j*2u,0x3f00u+(j&127u));
      } else {
        cg_u32(blob+at+4u,as_type<uint>(0.125f));
        cg_u32(blob+at+8u,as_type<uint>(0.5f));
        for(uint j=0u;j<precision*32u;++j)blob[at+12u+j]=uchar((j*37u+tile*19u)&255u);
        for(uint j=0u;j<3u;++j) {
          uint pos=j==0u?0u:(j==1u?127u:255u);
          uint value=mode==0u?(j==0u?0x3fc0u:(j==1u?0xbf40u:0x4000u)):
            (j==0u?0x3000u:(j==1u?0xb400u:0x3800u));
          blob[at+4u+payload+j*3u]=uchar(pos);
          cg_u16(blob+at+5u+payload+j*3u,value);
        }
      }
      at+=bytes;
    }
    offsets[9]=at;
  }
  kernel void cg_synthetic_input(device float *x [[buffer(0)]],
      device float *bias [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<19u*256u)x[i]=float(int((i*13u+i/256u*7u)%31u)-15)*0.03125f;
    if(i<9u)bias[i]=float(int(i)-4)*0.0625f;
  }
  kernel void cg_corrupt(device uchar *blob [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i==0u)blob[0]=3u;
  }
  // Five words per comparison: bit differences, tolerance failures, nonfinite,
  // maximum absolute error and maximum normalized error (positive float bits).
  kernel void cg_compare(device const float *actual [[buffer(0)]],
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

private def cg_zero(e : ML::Metal::ComputeEncoder, p : ML::Metal::ComputePipeline,
                    buffer : ML::MetalBuffer, n : Int32) : Nil
  e.set_pipeline(p).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    .set_value(n.to_u32, 1).dispatch_1d(n)
end

private def cg_compare(e : ML::Metal::ComputeEncoder, p : ML::Metal::ComputePipeline,
                       actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                       metrics : ML::MetalBuffer, slot : Int32, n : Int32, period : Int32) : Nil
  e.set_pipeline(p).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(slot.to_u32, 3).set_value(n.to_u32, 4).set_value(period.to_u32, 5)
    .dispatch_threadgroups({(n + 255)//256, 1, 1}, {256, 1, 1})
end

private def cg_metric(buffer : ML::MetalBuffer, slot : Int32)
  p = buffer.contents.as(UInt32*) + slot*5
  {bit_differences: p[0], tolerance_violations: p[1], nonfinite: p[2],
   max_abs: p[3].unsafe_as(Float32), max_ratio: p[4].unsafe_as(Float32)}
end

private def cg_fused(e : ML::Metal::ComputeEncoder,
                     x : ML::MetalBuffer, offsets : ML::MetalBuffer, blob : ML::MetalBuffer,
                     lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                     destination : ML::MetalBuffer, status : ML::MetalBuffer, rows : Int32) : Nil
  first = 0
  while first < rows
    count = Math.min(64, rows - first)
    ML::Video::QBitCodebookLinear.linear_range(e, x, offsets, blob, lut4, lut5, bias,
      destination, status, rows, DOT_DIN, DOT_DOUT, first, count, DOT_TILE)
    first += count
  end
end

private def cg_dense(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                     w : ML::MetalBuffer, bias : ML::MetalBuffer, destination : ML::MetalBuffer,
                     rows : Int32) : Nil
  first = 0
  while first < rows
    count = Math.min(64, rows - first)
    ML::Video::QBitCodebookLinear.dense_linear_range(e, x, w, bias, destination,
      rows, DOT_DIN, DOT_DOUT, first, count)
    first += count
  end
end

private def cg_mps(e : ML::Metal::ComputeEncoder, plan : ML::Video::WanT5MPSLinearPlan,
                   output : ML::MetalBuffer, bias : ML::MetalBuffer, rows : Int32) : Nil
  e.encode_wan_t5_mps_fc2(plan.handle)
  e.set_pipeline(ML::Video::MetalLinearMPS.pipeline)
    .set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(bias, 1).set_buffer(bias, 2).set_value(DOT_DOUT.to_u32, 3)
    .set_value(0_u32, 4).set_value((rows*DOT_DOUT).to_u32, 5).dispatch_1d(rows*DOT_DOUT)
end

private def cg_timed(kind : String, zero : ML::Metal::ComputePipeline,
                     x : ML::MetalBuffer, offsets : ML::MetalBuffer, blob : ML::MetalBuffer,
                     lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                     fused : ML::MetalBuffer, control : ML::MetalBuffer, stage : ML::MetalBuffer,
                     status : ML::MetalBuffer, plan : ML::Video::WanT5MPSLinearPlan, rows : Int32)
  before = ML::MetalBuffer.video_spec_allocations
  started = Time.instant
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  cg_zero(e, zero, status, 8)
  case kind
  when "fused_codebook_gemm"
    cg_fused(e, x, offsets, blob, lut4, lut5, bias, fused, status, rows)
  when "decode_mps_projection"
    ML::Video::QBitPositionalWeights.decode(e, offsets, blob, lut4, lut5, stage, status, DOT_WEIGHTS, DOT_TILE)
    cg_mps(e, plan, control, bias, rows)
  when "dense_tiled_gemm"
    cg_dense(e, x, stage, bias, fused, rows)
  else
    raise "Unknown timing route #{kind}"
  end
  e.end_encoding
  encode_seconds = (Time.instant - started).total_seconds
  gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
  wall_seconds = (Time.instant - started).total_seconds
  allocations = ML::MetalBuffer.video_spec_allocations - before
  raise "Timing route allocated tracked MetalBuffers" unless allocations == 0
  status_words = status.contents.as(UInt32*).to_slice(8).to_a
  raise "Timed route reported a record error" unless status_words.all? { |word| word == 0 }
  {encode_seconds: encode_seconds, gpu_seconds: gpu_seconds, wall_seconds: wall_seconds,
   terminal_waits: 1, inter_operator_host_waits: 0, device_activation_readbacks: 0,
   encode_metal_buffer_allocations: allocations, status: status_words}
end

private def cg_synthetic(zero : ML::Metal::ComputePipeline,
                         compare : ML::Metal::ComputePipeline,
                         lut4_path : String, lut5_path : String) : Nil
  arena = ML::Video::Arena.new
  plan : ML::Video::WanT5MPSLinearPlan? = nil
  begin
    seed_blob = ML::Metal::ComputePipeline.new("cg_synthetic_blob", CG_METAL)
    seed_input = ML::Metal::ComputePipeline.new("cg_synthetic_input", CG_METAL)
    corrupt = ML::Metal::ComputePipeline.new("cg_corrupt", CG_METAL)
    x = arena.alloc("synthetic_input", 19*256)
    bias = arena.alloc("synthetic_bias", 9)
    stage = arena.alloc("synthetic_dense_weights", 9*256)
    fused = arena.alloc("synthetic_fused", 19*9)
    control = arena.alloc("synthetic_control", 19*9)
    blob = arena.alloc("synthetic_blob", 1024)
    offsets = arena.alloc("synthetic_offsets", 10)
    status = arena.alloc("synthetic_status", 8)
    invalid_status = arena.alloc("synthetic_invalid_status", 8)
    metrics = arena.alloc("synthetic_metrics", 35)
    lut4 = arena.alloc("synthetic_lut4", 256)
    lut5 = arena.alloc("synthetic_lut5", 256)
    dot_upload(lut4_path, lut4, 1024_i64)
    dot_upload(lut5_path, lut5, 1024_i64)
    cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
      layers: 1, batches: 1, tokens: 19)
    plan = ML::Video::WanT5MPSLinearPlan.new(cfg, 256, 9, x, stage, control)
    command = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(command)
    before = ML::MetalBuffer.video_spec_allocations
    cg_zero(e, zero, status, 8)
    cg_zero(e, zero, invalid_status, 8)
    cg_zero(e, zero, metrics, 35)
    e.set_pipeline(seed_blob).set_buffer(blob, 0, ML::Metal::BufferAccess::Write)
      .set_buffer(offsets, 1, ML::Metal::BufferAccess::Write).dispatch_1d(1)
    e.set_pipeline(seed_input).set_buffer(x, 0, ML::Metal::BufferAccess::Write)
      .set_buffer(bias, 1, ML::Metal::BufferAccess::Write).dispatch_1d(19*256)
    ML::Video::QBitPositionalWeights.decode(e, offsets, blob, lut4, lut5, stage, status, 9*256, 256)
    { {0, 16}, {16, 3} }.each do |range|
      ML::Video::QBitCodebookLinear.dense_linear_range(e, x, stage, bias, control,
        19, 256, 9, range[0], range[1])
      ML::Video::QBitCodebookLinear.linear_range(e, x, offsets, blob, lut4, lut5, bias,
        fused, status, 19, 256, 9, range[0], range[1], 256)
    end
    cg_compare(e, compare, fused, control, metrics, 0, 19*9, 19*9)
    e.encode_wan_t5_mps_fc2(plan.not_nil!.handle)
    e.set_pipeline(ML::Video::MetalLinearMPS.pipeline)
      .set_buffer(control, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(bias, 1).set_buffer(bias, 2).set_value(9_u32, 3)
      .set_value(0_u32, 4).set_value((19*9).to_u32, 5).dispatch_1d(19*9)
    cg_compare(e, compare, fused, control, metrics, 1, 19*9, 19*9)
    e.set_pipeline(corrupt).set_buffer(blob, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
    ML::Video::QBitCodebookLinear.linear_range(e, x, offsets, blob, lut4, lut5, bias,
      fused, invalid_status, 19, 256, 9, 0, 19, 256)
    cg_compare(e, compare, fused, control, metrics, 2, 19*9, 19*9)
    e.end_encoding
    command.commit
    command.wait
    same = cg_metric(metrics, 0)
    mps = cg_metric(metrics, 1)
    invalid = cg_metric(metrics, 2)
    valid_status_words = status.contents.as(UInt32*).to_slice(8).to_a
    invalid_status_words = invalid_status.contents.as(UInt32*).to_slice(8).to_a
    allocations = ML::MetalBuffer.video_spec_allocations - before
    passed = same[:bit_differences] == 0 && same[:nonfinite] == 0 &&
             mps[:tolerance_violations] == 0 && mps[:nonfinite] == 0 &&
             valid_status_words.all? { |word| word == 0 } && invalid_status_words[7] > 0 &&
             invalid[:nonfinite] == 19*8 && allocations == 0
    puts({schema: "wan_qbit_codebook_gemm_synthetic_v1", rows: 19, input_dim: 256,
          output_dim: 9, tile_values: 256, precision_counts: {p4: 3, p5: 3, raw: 3},
          both_correction_modes: true, correction_positions: [0, 127, 255],
          fused_vs_dense_tiled: same, fused_vs_mps: mps,
          malformed_record_comparison: invalid, valid_status: valid_status_words,
          malformed_expected_nonfinite: 19*8,
          invalid_status: invalid_status_words, passed: passed,
          encode_metal_buffer_allocations: allocations, terminal_waits: 1,
          inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
    STDOUT.flush
    raise "Synthetic raw/correction/tail/malformed control rejected" unless passed
  ensure
    plan.try(&.release)
    arena.release
  end
end

fixture_dir = DOT_FIXTURE_DIR
reference_dir = DOT_REFERENCE_DIR
mode = "parity"
selected_arm = "positional_tile1024_t050_mode0"
OptionParser.parse do |parser|
  parser.on("--fixture=DIR", "Pinned source fixture") { |value| fixture_dir = value }
  parser.on("--positional-fixture=DIR", "Pinned positional reference") { |value| reference_dir = value }
  parser.on("--mode=MODE", "parity or timing") { |value| mode = value }
  parser.on("--arm=NAME", "One pinned tile1024 arm for full-shape timing") { |value| selected_arm = value }
end
raise "Unknown probe mode" unless {"parity", "timing"}.includes?(mode)
raise "Unknown timing arm" unless DOT_ARM_NAMES.includes?(selected_arm)
rows = mode == "timing" ? 4480 : DOT_ROWS

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
  arm_mode = name.ends_with?("mode0") ? 0 : 1
  unless arm["tile_size"].as_i == DOT_TILE && arm["threshold_f32_hex"].as_s == threshold_hex &&
         arm["value_mode"].as_i == arm_mode && arm["precision_tie_order"].as_a.map(&.as_i) == [16, 5, 4]
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
plan : ML::Video::WanT5MPSLinearPlan? = nil
begin
  ML::Video::MetalLinearMPS.prepare!
  ML::Video::QBitPositionalWeights.prepare!
  ML::Video::QBitCodebookLinear.prepare!
  zero = ML::Metal::ComputePipeline.new("cg_zero", CG_METAL)
  repeat = ML::Metal::ComputePipeline.new("cg_repeat", CG_METAL)
  compare = ML::Metal::ComputePipeline.new("cg_compare", CG_METAL)
  seed = ML::Metal::ComputePipeline.new("cg_seed", CG_METAL)
  cg_synthetic(zero, compare, lut4_path, lut5_path)
  source = arena.alloc("four_source_rows", DOT_ROWS*DOT_DIN)
  input = arena.alloc("input", rows*DOT_DIN)
  stage = arena.alloc("decoded_weight_stage_control_only", DOT_WEIGHTS)
  oracle_weight = arena.alloc("oracle_weight_control_only", DOT_WEIGHTS)
  oracle_output = arena.alloc("four_oracle_output_rows", DOT_OUTPUTS)
  fused = arena.alloc("fused_output", rows*DOT_DOUT)
  control = arena.alloc("control_output", rows*DOT_DOUT)
  bias = arena.alloc("bias", DOT_DOUT)
  lut4 = arena.alloc("lut4", 256)
  lut5 = arena.alloc("lut5", 256)
  offsets = arena.alloc("packed_offsets", DOT_WEIGHTS//DOT_TILE + 1)
  max_blob = arms.max_of { |arm| arm["blob"]["bytes"].as_i }
  blob = arena.alloc("packed_blob_capacity", (max_blob + 3)//4)
  status = arena.alloc("status", 8)
  metrics = arena.alloc("comparison_metrics", 35)
  control_a = arena.alloc("comparator_a", 2)
  control_b = arena.alloc("comparator_b", 2)
  raise "Explicit buffers exceed 2 GiB" unless arena.resident_bytes < 2_i64*1024**3
  dot_upload(input_path, source, DOT_ROWS.to_i64*DOT_DIN*4)
  dot_upload(bias_path, bias, DOT_DOUT.to_i64*4)
  dot_upload(lut4_path, lut4, 1024_i64)
  dot_upload(lut5_path, lut5, 1024_i64)
  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
    layers: 1, batches: 1, tokens: rows)
  plan = ML::Video::WanT5MPSLinearPlan.new(cfg, DOT_DIN, DOT_DOUT, input, stage, control)
  setup = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(setup)
  e.set_pipeline(repeat).set_buffer(source, 0).set_buffer(input, 1, ML::Metal::BufferAccess::Write)
    .set_value((DOT_ROWS*DOT_DIN).to_u32, 2).set_value((rows*DOT_DIN).to_u32, 3).dispatch_1d(rows*DOT_DIN)
  e.end_encoding; setup.commit; setup.wait
  puts({schema: "wan_qbit_codebook_gemm_setup_v1", mode: mode,
        source_manifest_sha256: fixture_hash, positional_manifest_sha256: reference_hash,
        rows: rows, input_dim: DOT_DIN, output_dim: DOT_DOUT, unique_seeded_rows: 4,
        capture_seed: 7031, real_video_derived: false, explicit_buffer_bytes: arena.resident_bytes,
        row_tile: ML::Video::QBitCodebookLinear::ROW_TILE,
        output_tile: ML::Video::QBitCodebookLinear::OUTPUT_TILE,
        rows_per_dispatch: 64, f32_weight_stage_bytes_control_only: stage.size,
        full_model_loaded: false, generation: false}.to_json)
  STDOUT.flush
  executing_arms = mode == "timing" ? arms.select { |arm| arm["name"].as_s == selected_arm } : arms
  executing_arms.each do |arm|
    name = arm["name"].as_s
    dot_upload(File.join(reference_dir, arm["offsets"]["path"].as_s), offsets, offsets.size)
    dot_upload(File.join(reference_dir, arm["blob"]["path"].as_s), blob, arm["blob"]["bytes"].as_i64)
    dot_upload(File.join(reference_dir, arm["decoded_weight"]["path"].as_s), oracle_weight, oracle_weight.size)
    dot_upload(File.join(reference_dir, arm["projection"]["path"].as_s), oracle_output, DOT_OUTPUTS.to_i64*4)
    command = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(command)
    before = ML::MetalBuffer.video_spec_allocations
    cg_zero(e, zero, status, 8); cg_zero(e, zero, metrics, 35)
    ML::Video::QBitPositionalWeights.decode(e, offsets, blob, lut4, lut5, stage, status, DOT_WEIGHTS, DOT_TILE)
    cg_compare(e, compare, stage, oracle_weight, metrics, 0, DOT_WEIGHTS, DOT_WEIGHTS)
    cg_dense(e, input, stage, bias, control, rows)
    cg_fused(e, input, offsets, blob, lut4, lut5, bias, fused, status, rows)
    cg_compare(e, compare, fused, oracle_output, metrics, 1, rows*DOT_DOUT, DOT_OUTPUTS)
    cg_compare(e, compare, control, oracle_output, metrics, 2, rows*DOT_DOUT, DOT_OUTPUTS)
    cg_compare(e, compare, fused, control, metrics, 3, rows*DOT_DOUT, rows*DOT_DOUT)
    cg_mps(e, plan.not_nil!, control, bias, rows)
    cg_compare(e, compare, control, oracle_output, metrics, 4, rows*DOT_DOUT, DOT_OUTPUTS)
    e.set_pipeline(seed).set_buffer(control_a, 0, ML::Metal::BufferAccess::Write)
      .set_buffer(control_b, 1, ML::Metal::BufferAccess::Write).dispatch_1d(2)
    cg_compare(e, compare, control_a, control_b, metrics, 5, 2, 2)
    e.end_encoding; command.commit; command.wait
    allocated = ML::MetalBuffer.video_spec_allocations - before
    raise "Qualification encode allocated buffers" unless allocated == 0
    decoded_metric = cg_metric(metrics, 0)
    fused_metric = cg_metric(metrics, 1)
    dense_metric = cg_metric(metrics, 2)
    same_metric = cg_metric(metrics, 3)
    mps_metric = cg_metric(metrics, 4)
    positive = cg_metric(metrics, 5)
    status_words = status.contents.as(UInt32*).to_slice(8).to_a
    decoder_ok = decoded_metric[:bit_differences] == 0 && decoded_metric[:nonfinite] == 0
    comparator_ok = positive[:bit_differences] == 2 && positive[:tolerance_violations] == 2 && positive[:nonfinite] == 1
    same_ok = same_metric[:bit_differences] == 0 && same_metric[:nonfinite] == 0
    fused_ok = fused_metric[:tolerance_violations] == 0 && fused_metric[:nonfinite] == 0
    dense_ok = dense_metric[:tolerance_violations] == 0 && dense_metric[:nonfinite] == 0
    mps_ok = mps_metric[:tolerance_violations] == 0 && mps_metric[:nonfinite] == 0
    status_ok = status_words.all? { |word| word == 0 }
    qualification_ok = decoder_ok && comparator_ok && same_ok && fused_ok && dense_ok && mps_ok && status_ok
    puts({schema: "wan_qbit_codebook_gemm_qualification_v1", arm: name, rows: rows,
          compared_weights: DOT_WEIGHTS, compared_outputs: rows*DOT_DOUT,
          decoded_weights_vs_oracle: decoded_metric, fused_vs_upstream: fused_metric,
          dense_tiled_vs_upstream: dense_metric, fused_vs_dense_tiled: same_metric,
          staged_mps_vs_upstream: mps_metric, positive_control: positive,
          decoder_bitexact: decoder_ok, fused_dense_bitexact: same_ok,
          fused_gate: fused_ok, dense_gate: dense_ok, staged_mps_gate: mps_ok,
          comparator_control_passed: comparator_ok, status: status_words,
          qualification_passed: qualification_ok, encode_metal_buffer_allocations: allocated,
          terminal_waits: 1, inter_operator_host_waits: 0, device_activation_readbacks: 0,
          blob_and_offsets_bytes: arm["storage_accounting"]["blob_and_offsets_bytes"].as_i64}.to_json)
    STDOUT.flush
    raise "Numerical qualification rejected: #{name}" unless qualification_ok
    next unless mode == "timing"
    # Warm both AB and BA orders; identical resident buffers and command boundary.
    warm_order = ["fused_codebook_gemm", "decode_mps_projection", "decode_mps_projection", "fused_codebook_gemm"]
    warm_order.each_with_index do |kind, position|
      result = cg_timed(kind, zero, input, offsets, blob, lut4, lut5, bias, fused,
        control, stage, status, plan.not_nil!, rows)
      puts({schema: "wan_qbit_codebook_gemm_timing_v1", arm: name, rows: rows,
            sample_kind: "warmup", segment: kind, position: position, measured: false,
            timing: result}.to_json); STDOUT.flush
    end
    measured_order = ["fused_codebook_gemm", "decode_mps_projection", "decode_mps_projection", "fused_codebook_gemm",
                      "decode_mps_projection", "fused_codebook_gemm", "fused_codebook_gemm", "decode_mps_projection"]
    measured_order.each_with_index do |kind, position|
      result = cg_timed(kind, zero, input, offsets, blob, lut4, lut5, bias, fused,
        control, stage, status, plan.not_nil!, rows)
      puts({schema: "wan_qbit_codebook_gemm_timing_v1", arm: name, rows: rows,
            sample_kind: "paired", segment: kind, position: position, pair: position//2,
            quartet: position//4, order: position < 4 ? "ABBA" : "BAAB", measured: true,
            correctness_outside_timed_segment: true, timing: result}.to_json); STDOUT.flush
    end
    result = cg_timed("dense_tiled_gemm", zero, input, offsets, blob, lut4, lut5, bias,
      fused, control, stage, status, plan.not_nil!, rows)
    puts({schema: "wan_qbit_codebook_gemm_timing_v1", arm: name, rows: rows,
          sample_kind: "diagnostic", segment: "dense_tiled_gemm", measured: false,
          timing: result}.to_json); STDOUT.flush
    raise "Timed route reported a record error" unless status.contents.as(UInt32*).to_slice(8).all? { |v| v == 0 }
  end
  puts({schema: "wan_qbit_codebook_gemm_complete_v1", mode: mode, rows: rows,
        arms: executing_arms.map { |arm| arm["name"].as_s }, full_model_loaded: false,
        generation: false, inter_operator_host_waits: 0, device_activation_readbacks: 0}.to_json)
ensure
  plan.try(&.release)
  arena.release
end
