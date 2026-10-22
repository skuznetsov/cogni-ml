# Opt-in Gaussian-prefix diagnostic. No model, video inference or timing claim.
# Prepare references with support/wan_quantized_projection_diagnostics.py.
require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/video/metal_linear_mps"
require "../src/ml/video/packed_weight_decode"
require "../spec/support/video_allocation_counter"

private def packed_probe_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

private def packed_probe_upload(path : String, buffer : ML::MetalBuffer, count : Int64) : Nil
  raise "Fixture byte count mismatch: #{path}" unless File.size(path) == count && buffer.size >= count
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), count.to_i32)) }
end

private def packed_probe_verify_file(directory : String, record : JSON::Any, name : String) : Nil
  unless record["path"].as_s == name && File.basename(name) == name
    raise "Fixture descriptor path mismatch: #{name}"
  end
  path = File.join(directory, name)
  unless File.size(path) == record["bytes"].as_i64 && packed_probe_sha256(path) == record["sha256"].as_s
    raise "Fixture bytes/hash mismatch: #{name}"
  end
end

private PACKED_PROBE_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void packed_probe_tile(device const float *src [[buffer(0)]],
      device float *dst [[buffer(1)]], constant uint &source_count [[buffer(2)]],
      constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = src[i % source_count];
  }
  kernel void packed_probe_copy(device const float *src [[buffer(0)]],
      device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
      uint i [[thread_position_in_grid]]) { if (i < count) dst[i] = src[i]; }
  kernel void packed_probe_zero(device atomic_uint *m [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i < 5u) atomic_store_explicit(m+i, 0u, memory_order_relaxed);
  }
  kernel void packed_probe_corrupt(device float *x [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i == 0u) { x[0] += 1.0f; x[1] = as_type<float>(0x7fc00000u); }
  }
  kernel void packed_probe_compare(device const float *a [[buffer(0)]],
      device const float *b [[buffer(1)]], device atomic_uint *m [[buffer(2)]],
      device float4 *partials [[buffer(3)]], constant uint &count [[buffer(4)]],
      uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
      uint g [[threadgroup_position_in_grid]]) {
    threadgroup uint4 counts[256];
    threadgroup float2 maxima[256];
    threadgroup float4 sums[256];
    uint4 c = uint4(0u); float2 mx = float2(0.0f); float4 s = float4(0.0f);
    if (i < count) {
      float av=a[i], bv=b[i]; c.x = as_type<uint>(av) != as_type<uint>(bv);
      if (!isfinite(av) || !isfinite(bv)) { c.y=1u; c.z=1u; }
      else {
        float d=av-bv, error=abs(d), ratio=error/(2.0e-5f+2.0e-5f*abs(bv));
        c.y = ratio > 1.0f; mx=float2(error,ratio);
        s=float4(d*d,bv*bv,av*av,av*bv);
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
    if (t==0u) {
      atomic_fetch_add_explicit(m,counts[0].x,memory_order_relaxed);
      atomic_fetch_add_explicit(m+1,counts[0].y,memory_order_relaxed);
      atomic_fetch_add_explicit(m+2,counts[0].z,memory_order_relaxed);
      atomic_fetch_max_explicit(m+3,as_type<uint>(maxima[0].x),memory_order_relaxed);
      atomic_fetch_max_explicit(m+4,as_type<uint>(maxima[0].y),memory_order_relaxed);
      partials[g]=sums[0];
    }
  }
  kernel void packed_probe_fold(device float4 *partials [[buffer(0)]],
      constant uint &groups [[buffer(1)]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float4 sums[256];
    float4 value=0.0f;
    for (uint g=t;g<groups;g+=256u) value+=partials[g];
    sums[t]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=128u;s>0u;s>>=1u) {
      if (t<s) sums[t]+=sums[t+s];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u) partials[0]=sums[0];
  }
METAL

private def packed_probe_compare(e : ML::Metal::ComputeEncoder,
                                 zero : ML::Metal::ComputePipeline,
                                 compare : ML::Metal::ComputePipeline,
                                 fold : ML::Metal::ComputePipeline,
                                 actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                                 metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
                                 count : Int32) : Nil
  e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  e.set_pipeline(compare).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(partials, 3, ML::Metal::BufferAccess::Write).set_value(count.to_u32, 4)
    .dispatch_threadgroups({(count + 255) // 256, 1, 1}, {256, 1, 1})
  e.set_pipeline(fold).set_buffer(partials, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_value(((count + 255) // 256).to_u32, 1).dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
end

private def packed_probe_metrics(metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
                                 count : Int32)
  words = metrics.contents.as(UInt32*)
  data = partials.contents.as(Float32*)
  # Only terminal, GPU-reduced statistics cross this boundary, never activations.
  sum_error = data[0].to_f64; sum_ref = data[1].to_f64
  sum_actual = data[2].to_f64; dot = data[3].to_f64
  {compared_values: count, bit_differences: words[0], tolerance_violations: words[1],
   nonfinite: words[2], max_abs: words[3].unsafe_as(Float32),
   max_ratio: words[4].unsafe_as(Float32), rmse: Math.sqrt(sum_error / count),
   relative_l2: Math.sqrt(sum_error / Math.max(sum_ref, 1e-300)),
   cosine: dot / Math.sqrt(Math.max(sum_ref * sum_actual, 1e-300))}
end

private GAUSSIAN_DIAGNOSTIC_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  #pragma clang fp contract(off)
  static inline float diagnostic_f32(device const uchar *p) {
    uint u=uint(p[0])|(uint(p[1])<<8)|(uint(p[2])<<16)|(uint(p[3])<<24);
    return as_type<float>(u);
  }
  static inline bool adaptive_accept(float error, float sigma, float threshold) {
    return isfinite(error) && isfinite(sigma) && error>=0.0f && sigma>=0.0f &&
      (sigma==0.0f ? error==0.0f : error/sigma<=threshold);
  }
  static inline uint adaptive_tier(float e4,float e5,float sigma,float t4,float t5) {
    return adaptive_accept(e4,sigma,t4) ? 4u :
      (adaptive_accept(e5,sigma,t5) ? 5u : 16u);
  }
  // Diagnostic selection only: candidates remain homogeneous input payloads.
  // One group selects an entire original 256-value normalization block.
  kernel void diagnostic_adaptive_select(device float *p4 [[buffer(0)]],
      device const float *p5 [[buffer(1)]], device const float *original [[buffer(2)]],
      device const uchar *moments [[buffer(3)]], device const uchar *expected [[buffer(4)]],
      device atomic_uint *metrics [[buffer(5)]], constant float &t4 [[buffer(6)]],
      constant float &t5 [[buffer(7)]], uint i [[thread_position_in_grid]],
      uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup float2 errors[256]; threadgroup uint tier;
    float a=p4[i],b=p5[i],w=original[i];
    errors[t]=float2(abs(a-w),abs(b-w));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=128u;s>0u;s>>=1u) {
      if (t<s) errors[t]=max(errors[t],errors[t+s]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u) {
      tier=adaptive_tier(errors[0].x,errors[0].y,diagnostic_f32(moments+g*136u+4u),t4,t5);
      uint k=tier==4u ? 0u : (tier==5u ? 1u : 2u);
      atomic_fetch_add_explicit(metrics+k,1u,memory_order_relaxed);
      if (tier!=uint(expected[g])) atomic_fetch_add_explicit(metrics+3,1u,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p4[i]=tier==4u ? a : (tier==5u ? b : w);
  }
  kernel void diagnostic_adaptive_control(device atomic_uint *metrics [[buffer(0)]],
      constant float &t4 [[buffer(1)]], constant float &t5 [[buffer(2)]],
      uint i [[thread_position_in_grid]]) {
    if (i>=12u) return;
    float e4=t4*2.0f,e5=t5*2.0f,sigma=1.0f; uint expected=16u;
    if (i<3u) {
      e4=as_type<float>(uint(int(as_type<uint>(t4))+int(i)-1));
      expected=i<2u ? 4u : 16u;
    } else if (i<6u) {
      e5=as_type<float>(uint(int(as_type<uint>(t5))+int(i)-4));
      expected=i<5u ? 5u : 16u;
    } else if (i<9u) {
      sigma=0.0f;e4=i==6u ? 0.0f : 1.0f;e5=i<8u ? 0.0f : 1.0f;
      expected=i==6u ? 4u : (i==7u ? 5u : 16u);
    } else if (i==9u) {
      e4=as_type<float>(0x7fc00000u);e5=0.0f;expected=5u;
    } else if (i==10u) e4=e5=INFINITY;
    else { sigma=as_type<float>(0x7fc00000u);e4=e5=0.0f; }
    if (adaptive_tier(e4,e5,sigma,t4,t5)!=expected)
      atomic_fetch_add_explicit(metrics+4,1u,memory_order_relaxed);
  }
  // Destructive scratch-only control after all projection checks: +/-4 source,
  // zero reconstruction, mean=0 and sigma=1. All tail buckets include each
  // element; every expected sum is an exactly representable power-of-two sum.
  kernel void diagnostic_stats_control(device float *stage [[buffer(0)]],
      device float *original [[buffer(1)]], device uchar *moments [[buffer(2)]],
      constant uint &count [[buffer(3)]], constant uint &mode [[buffer(4)]],
      uint i [[thread_position_in_grid]]) {
    if (i>=count) return;
    float w=(i&1u) ? -4.0f : 4.0f;
    if (mode==1u) {
      uint j=i&15u;
      if (j>=1u && j<=12u) {
        float thresholds[4]={1.7038403749465942f,2.2929420471191406f,3.0f,4.0f};
        uint k=(j-1u)/3u;
        int offset=int((j-1u)%3u)-1;
        w=as_type<float>(uint(int(as_type<uint>(thresholds[k]))+offset));
      } else w=j==13u ? 0.5f : (j==14u ? 2.0f : (j==15u ? 2.5f : 0.0f));
      if ((i&16u)!=0u) w=-w;
    }
    original[i]=w; stage[i]=mode==1u ? w*0.5f : 0.0f;
    if ((i&255u)==0u) {
      device uchar *p=moments+(i/256u)*136u;
      for (uint j=0u;j<8u;j++) p[j]=0u;
      p[6]=128u; p[7]=63u;
    }
  }
  kernel void diagnostic_transform(device float *stage [[buffer(0)]],
      device const float *original [[buffer(1)]], device const uchar *packed [[buffer(2)]],
      device const float *lut [[buffer(3)]], constant uint &count [[buffer(4)]],
      constant uint &precision [[buffer(5)]], constant uint &mode [[buffer(6)]],
      constant float &threshold [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i>=count) return;
    device const uchar *block=packed+(i/256u)*(8u+precision*32u);
    float mean=diagnostic_f32(block), sigma=diagnostic_f32(block+4);
    float z=sigma==0.0f ? 0.0f : (original[i]-mean)/sigma;
    if (mode==1u) {
      if (abs(z)>=threshold) stage[i]=original[i];
    } else {
      uint width=1u<<(8u-precision), lo=0u, hi=(1u<<(precision-1u))-1u;
      while (lo<hi) {
        uint mid=(lo+hi)/2u;
        float boundary=(lut[mid*width]+lut[(mid+1u)*width])*0.5f;
        if (abs(z)<=boundary) hi=mid; else lo=mid+1u;
      }
      float center=lut[lo*width];
      if (z<0.0f) center=-center;
      float scaled=sigma*center;
      stage[i]=mean+scaled;
    }
  }
  // One group per original 256-value normalization block. All formats use
  // the same original p4 moments to define their standardized tail buckets.
  kernel void diagnostic_stats(device const float *actual [[buffer(0)]],
      device const float *original [[buffer(1)]], device const uchar *moments [[buffer(2)]],
      device float4 *partials [[buffer(3)]], constant uint &count [[buffer(4)]],
      uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
      uint g [[threadgroup_position_in_grid]]) {
    threadgroup float4 a[256],b[256],c[256],m[256];
    float4 av=0.0f,bv=0.0f,cv=0.0f,mv=0.0f;
    if (i<count) {
      float w=original[i], d=actual[i]-w;
      float mean=diagnostic_f32(moments+g*136u);
      float sigma=diagnostic_f32(moments+g*136u+4u);
      float z=sigma==0.0f ? 0.0f : (w-mean)/sigma, z2=z*z, az=abs(z);
      float4 mask=float4(az>=1.7038403749465942f,az>=2.2929420471191406f,
                         az>=3.0f,az>=4.0f);
      av=float4(d*d,w*w,z*z2,z2*z2); bv=mask*(d*d); cv=mask;
      mv=float4(az,abs(d),d,0.0f);
    }
    a[t]=av;b[t]=bv;c[t]=cv;m[t]=mv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=128u;s>0u;s>>=1u) {
      if (t<s) {
        a[t]+=a[t+s];b[t]+=b[t+s];c[t]+=c[t+s];
        m[t].xy=max(m[t].xy,m[t+s].xy);m[t].z+=m[t+s].z;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u) {partials[g*4u]=a[0];partials[g*4u+1u]=b[0];
      partials[g*4u+2u]=c[0];partials[g*4u+3u]=m[0];}
  }
  kernel void diagnostic_reduce(device const float4 *partials [[buffer(0)]],
      device float4 *total [[buffer(1)]], constant uint &groups [[buffer(2)]],
      uint t [[thread_index_in_threadgroup]]) {
    threadgroup float4 a[256],b[256],c[256],m[256];
    float4 av=0.0f,bv=0.0f,cv=0.0f,mv=0.0f;
    for (uint g=t;g<groups;g+=256u) {
      av+=partials[g*4u];bv+=partials[g*4u+1u];cv+=partials[g*4u+2u];
      float4 p=partials[g*4u+3u];mv.xy=max(mv.xy,p.xy);mv.z+=p.z;
    }
    a[t]=av;b[t]=bv;c[t]=cv;m[t]=mv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s=128u;s>0u;s>>=1u) {
      if (t<s) {a[t]+=a[t+s];b[t]+=b[t+s];c[t]+=c[t+s];
        m[t].xy=max(m[t].xy,m[t+s].xy);m[t].z+=m[t+s].z;}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t==0u) {total[0]=a[0];total[1]=b[0];total[2]=c[0];total[3]=m[0];}
  }
METAL

private def gaussian_diagnostic_stats(total : ML::MetalBuffer, count : Int32)
  v = total.contents.as(Float32*)
  raise "Nonfinite diagnostic reduction" unless (0...16).all? { |i| v[i].finite? }
  # This bounded fixture uses exact Float32 integer sums. Refuse a larger
  # occupancy rather than silently reporting rounded counts above 2**24.
  raise "Tail count exceeds exact Float32 integer range" unless (8...12).all? { |i| v[i] >= 0 && v[i] <= (1 << 24) && v[i] == v[i].to_i64.to_f32 }
  error = v[0].to_f64
  {weight_relative_l2: Math.sqrt(error / Math.max(v[1].to_f64, 1e-300)),
   weight_rmse: Math.sqrt(error / count), weight_error_squared: error,
   mean_z3: v[2].to_f64 / count, mean_z4: v[3].to_f64 / count,
   z_convention: "raw moments; common original p4 block mean/sigma",
   tail_thresholds: [1.7038403749465942, 2.2929420471191406, 3.0, 4.0],
   tail_counts: (0...4).map { |i| v[8 + i].to_i64 },
   tail_weight_error_fractions: (0...4).map { |i| v[4 + i].to_f64 / Math.max(error, 1e-300) },
   max_abs_standardized_weight: v[12], max_abs_weight_error: v[13],
   mean_weight_error: v[14].to_f64 / count}
end

private def gaussian_diagnostic_boundary_expected : Array(Float64)
  thresholds = [1.7038403749465942_f32, 2.2929420471191406_f32, 3.0_f32, 4.0_f32]
  pattern = [0.0_f32]
  thresholds.each do |value|
    bits = value.unsafe_as(UInt32)
    pattern.concat([(bits - 1_u32).unsafe_as(Float32), value, (bits + 1_u32).unsafe_as(Float32)])
  end
  pattern.concat([0.5_f32, 2.0_f32, 2.5_f32])
  expected = Array(Float64).new(16, 0.0)
  256.times do |i|
    w = pattern[i & 15].to_f64
    w = -w if (i & 16) != 0
    error = -w * 0.5
    expected[0] += error * error; expected[1] += w * w
    expected[2] += w**3; expected[3] += w**4
    thresholds.each_with_index do |threshold, k|
      if w.abs >= threshold
        expected[4 + k] += error * error; expected[8 + k] += 1.0
      end
    end
    expected[12] = Math.max(expected[12], w.abs)
    expected[13] = Math.max(expected[13], error.abs)
    expected[14] += error
  end
  expected
end

fixture_dir = ""
diagnostic_dir = ""
adaptive_dir = ""
OptionParser.parse do |p|
  p.on("--fixture=DIR", "Original immutable packed projection fixture") { |v| fixture_dir = v }
  p.on("--diagnostic-fixture=DIR", "Independent immutable MPS diagnostic references") { |v| diagnostic_dir = v }
  p.on("--adaptive-fixture=DIR", "Optional MPS error-dependent block selection references") { |v| adaptive_dir = v }
end
raise ArgumentError.new("Both fixture directories are required") if fixture_dir.empty? || diagnostic_dir.empty?
manifest_hash = packed_probe_sha256(File.join(fixture_dir, "manifest.json"))
raise "Unexpected original fixture" unless manifest_hash == "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703"
manifest = JSON.parse(File.read(File.join(fixture_dir, "manifest.json")))
diagnostic_path = File.join(diagnostic_dir, "manifest.json")
diagnostic_hash = packed_probe_sha256(diagnostic_path)
diagnostic = JSON.parse(File.read(diagnostic_path))
unless diagnostic["schema"].as_s == "wan_quantized_projection_diagnostics_v1" &&
       diagnostic["source_manifest_sha256"].as_s == manifest_hash
  raise "Diagnostic references have the wrong origin"
end
{manifest, diagnostic}.each do |m|
  g = m["geometry"]
  unless g["rows"].as_i == 4480 && g["source_rows"].as_i == 4 &&
         g["input_dim"].as_i == 5120 && g["output_dim"].as_i == 13824
    raise "Diagnostic geometry mismatch"
  end
end
rows = 4480; source_rows = 4; din = 5120; dout = 13824
weight_count = din * dout; output_count = rows * dout
variants = ["bf16", "q4_k", "qbit_p4", "qbit_p5"]
vm = manifest["variants"]
packed_probe_verify_file(fixture_dir, manifest["input"], "input.f32")
variants.each do |variant|
  v = vm[variant]
  {"packed_weight", "decoded_weight", "expected_projection", "bias"}.each do |key|
    packed_probe_verify_file(fixture_dir, v[key], v["#{key}_file"].as_s)
  end
  packed_probe_verify_file(fixture_dir, v["lut"], v["lut_file"].as_s) if variant.starts_with?("qbit")
end
arms = diagnostic["arms"].as_a
raise "Expected eight unique diagnostic arms" unless arms.size == 8 && arms.map { |a| a["name"].as_s }.uniq.size == 8
expected_arms = {} of String => Tuple(Int32, String, Float64?)
{4, 5}.each do |precision|
  expected_arms["qbit_p#{precision}_arithmetic"] = {precision, "arithmetic", nil}
  expected_arms["qbit_p#{precision}_tail_z3"] = {precision, "tail_restore", 3.0}
  expected_arms["qbit_p#{precision}_tail_z4"] = {precision, "tail_restore", 4.0}
  expected_arms["qbit_p#{precision}_nearest"] = {precision, "nearest", nil}
end
arms.each do |arm|
  expected = expected_arms[arm["name"].as_s]? || raise "Unexpected diagnostic arm name"
  precision = arm["precision"].as_i
  mode = arm["mode"].as_s
  threshold = arm["threshold"].raw.nil? ? nil : arm["threshold"].as_f
  raise "Diagnostic intervention does not match its name" unless {precision, mode, threshold} == expected
  ref = arm["projection"]
  raise "Invalid diagnostic projection shape" unless ref["dtype"].as_s == "F32" && ref["bytes"].as_i64 == 4_i64 * dout * 4 && ref["shape"].as_a.map(&.as_i) == [4, dout]
  packed_probe_verify_file(diagnostic_dir, ref, ref["path"].as_s)
  lut = arm["lut"]
  raise "Invalid LUT shape" unless lut["dtype"].as_s == "F32" && lut["bytes"].as_i64 == 1024 && lut["shape"].as_a.map(&.as_i) == [256]
  packed_probe_verify_file(diagnostic_dir, lut, lut["path"].as_s)
  if mode != "arithmetic"
    raise "Intervention changes the Gaussian codebook" unless lut["sha256"].as_s == vm["qbit_p#{precision}"]["lut"]["sha256"].as_s
  end
end

adaptive = adaptive_dir.empty? ? nil : JSON.parse(File.read(File.join(adaptive_dir, "manifest.json")))
adaptive_hash = adaptive_dir.empty? ? nil : packed_probe_sha256(File.join(adaptive_dir, "manifest.json"))
adaptive_arms = adaptive.try(&.["policies"].as_a) || [] of JSON::Any
unless adaptive.nil?
  raise "Wrong adaptive fixture origin" unless adaptive.not_nil!["schema"].as_s == "wan_qbit_adaptive_weight_reference_v1" && adaptive.not_nil!["source_manifest_sha256"].as_s == manifest_hash
  a = adaptive.not_nil!
  raise "Wrong adaptive reference generator" unless a["created_by"].as_s == "wan_qbit_adaptive_weight_reference.py"
  oracle = a["oracle"]
  raise "Adaptive reference must use MPS without fallback" unless oracle["device"].as_s == "MPS" && oracle["fallback_disabled"].as_bool && oracle["no_full_wan_model_loaded"].as_bool && oracle["projection"].as_s == "torch.nn.functional.linear(input, decoded_weight, bias)"
  {"adaptive_script_sha256" => "wan_qbit_adaptive_weight_reference.py",
   "diagnostic_helper_script_sha256" => "wan_quantized_projection_diagnostics.py"}.each do |key, name|
    raise "Adaptive reference generator source changed" unless a["source_hashes"][key].as_s == packed_probe_sha256(File.join(__DIR__, "support", name))
  end
  g = adaptive.not_nil!["geometry"]
  raise "Adaptive geometry mismatch" unless g["rows"].as_i == rows && g["source_rows"].as_i == source_rows && g["input_dim"].as_i == din && g["output_dim"].as_i == dout
  expected_policies = {"adaptive_p4_020_p5_010" => {0.20_f32, 0.10_f32},
                       "adaptive_p4_050_p5_050" => {0.50_f32, 0.50_f32},
                       "adaptive_p4_100_p5_050" => {1.00_f32, 0.50_f32}}
  raise "Expected three unique adaptive policies" unless adaptive_arms.size == 3 && adaptive_arms.map { |a| a["name"].as_s }.uniq.size == 3
  adaptive_arms.each do |arm|
    thresholds = expected_policies[arm["name"].as_s]? || raise "Unknown adaptive policy"
    raise "Adaptive threshold mismatch" unless {arm["p4_max_error"].as_f.to_f32, arm["p5_max_error"].as_f.to_f32} == thresholds
    ref = arm["projection"]; tiers = arm["tiers"]
    raise "Invalid adaptive projection" unless ref["dtype"].as_s == "F32" && ref["bytes"].as_i64 == 4_i64 * dout * 4 && ref["shape"].as_a.map(&.as_i) == [4, dout]
    raise "Invalid adaptive tier record" unless tiers["dtype"].as_s == "U8" && tiers["bytes"].as_i64 == weight_count // 256 && tiers["shape"].as_a.map(&.as_i) == [weight_count // 256]
    {ref, tiers}.each { |record| packed_probe_verify_file(adaptive_dir, record, record["path"].as_s) }
  end
end

arena = ML::Video::Arena.new
plan : ML::Video::WanT5MPSLinearPlan? = nil
begin
  ML::Video::MetalLinearMPS.prepare!
  ML::Video::PackedWeightDecode.prepare!
  tile = ML::Metal::ComputePipeline.new("packed_probe_tile", PACKED_PROBE_SOURCE)
  copy = ML::Metal::ComputePipeline.new("packed_probe_copy", PACKED_PROBE_SOURCE)
  zero = ML::Metal::ComputePipeline.new("packed_probe_zero", PACKED_PROBE_SOURCE)
  compare = ML::Metal::ComputePipeline.new("packed_probe_compare", PACKED_PROBE_SOURCE)
  fold = ML::Metal::ComputePipeline.new("packed_probe_fold", PACKED_PROBE_SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("packed_probe_corrupt", PACKED_PROBE_SOURCE)
  transform = ML::Metal::ComputePipeline.new("diagnostic_transform", GAUSSIAN_DIAGNOSTIC_SOURCE)
  stats_control = ML::Metal::ComputePipeline.new("diagnostic_stats_control", GAUSSIAN_DIAGNOSTIC_SOURCE)
  stats = ML::Metal::ComputePipeline.new("diagnostic_stats", GAUSSIAN_DIAGNOSTIC_SOURCE)
  reduce = ML::Metal::ComputePipeline.new("diagnostic_reduce", GAUSSIAN_DIAGNOSTIC_SOURCE)
  adaptive_select = ML::Metal::ComputePipeline.new("diagnostic_adaptive_select", GAUSSIAN_DIAGNOSTIC_SOURCE)
  adaptive_control = ML::Metal::ComputePipeline.new("diagnostic_adaptive_control", GAUSSIAN_DIAGNOSTIC_SOURCE)
  source = arena.alloc("input_source", source_rows * din)
  input = arena.alloc("input", rows * din)
  original = arena.alloc("original", weight_count)
  stage = arena.alloc("stage", weight_count)
  expected_weight = arena.alloc("expected_weight", weight_count)
  output = arena.alloc("output", output_count)
  baseline = arena.alloc("baseline", output_count)
  expected_output = arena.alloc("expected_output", output_count)
  expected_source = arena.alloc("expected_source", source_rows * dout)
  bias = arena.alloc("bias", dout)
  lut4 = arena.alloc("lut4", 256); lut5 = arena.alloc("lut5", 256)
  ablation_lut = arena.alloc("ablation_lut", 256)
  packed = {} of String => ML::MetalBuffer
  variants.each do |variant|
    n = vm[variant]["packed_weight"]["bytes"].as_i64
    packed[variant] = arena.alloc("packed_#{variant}", (n // 4).to_i32)
    packed_probe_upload(File.join(fixture_dir, vm[variant]["packed_weight_file"].as_s), packed[variant], n)
  end
  wm = arena.alloc("weight_metrics", 5); om = arena.alloc("output_metrics", 5)
  lm = arena.alloc("loss_metrics", 5)
  wp = arena.alloc("weight_partials", ((weight_count + 255) // 256) * 4)
  op = arena.alloc("output_partials", ((output_count + 255) // 256) * 4)
  lp = arena.alloc("loss_partials", ((output_count + 255) // 256) * 4)
  sp = arena.alloc("stats_partials", (weight_count // 256) * 16)
  st = arena.alloc("stats_total", 16)
  boundary_stats = arena.alloc("boundary_stats_total", 16)
  adaptive_tiers = arena.alloc("adaptive_expected_tiers", weight_count // 256 // 4)
  adaptive_metrics = arena.alloc("adaptive_metrics", 5)
  raise "Explicit probe buffers exceed 2 GiB" if arena.resident_bytes > 2_i64 * 1024**3
  packed_probe_upload(File.join(fixture_dir, "input.f32"), source, source.size)
  packed_probe_upload(File.join(fixture_dir, "bias.f32"), bias, bias.size)
  packed_probe_upload(File.join(fixture_dir, "lut_p4.f32"), lut4, lut4.size)
  packed_probe_upload(File.join(fixture_dir, "lut_p5.f32"), lut5, lut5.size)
  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1, layers: 1, batches: 1, tokens: rows)
  plan = ML::Video::WanT5MPSLinearPlan.new(cfg, din, dout, input, stage, output)
  setup = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(setup)
  e.set_pipeline(tile).set_buffer(source, 0).set_buffer(input, 1, ML::Metal::BufferAccess::Write)
    .set_value((source_rows * din).to_u32, 2).set_value((rows * din).to_u32, 3).dispatch_1d(rows * din)
  e.set_pipeline(ML::Video::WanT5.pipeline("wan_t5_expand_fc2"))
    .set_buffer(packed["bf16"], 0).set_buffer(packed["bf16"], 1)
    .set_buffer(original, 2, ML::Metal::BufferAccess::Write)
    .set_value(weight_count.to_u32, 3).set_value(1_u32, 4).dispatch_1d(weight_count)
  e.end_encoding; setup.commit; setup.wait
  decode = ->(e : ML::Metal::ComputeEncoder, variant : String, lut : ML::MetalBuffer) {
    case variant
    when "bf16"
      e.set_pipeline(copy).set_buffer(original, 0).set_buffer(stage, 1, ML::Metal::BufferAccess::Write)
        .set_value(weight_count.to_u32, 2).dispatch_1d(weight_count)
    when "q4_k"
      ML::Video::PackedWeightDecode.q4_k(e, packed[variant], stage, weight_count)
    else
      ML::Video::PackedWeightDecode.qbit(e, packed[variant], stage, lut, weight_count, variant == "qbit_p4" ? 4 : 5)
    end
    nil
  }
  project = ->(e : ML::Metal::ComputeEncoder) {
    e.encode_wan_t5_mps_fc2(plan.not_nil!.handle)
    e.set_pipeline(ML::Video::MetalLinearMPS.pipeline).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(bias, 1).set_buffer(bias, 2).set_value(dout.to_u32, 3)
      .set_value(0_u32, 4).set_value(output_count.to_u32, 5).dispatch_1d(output_count)
    nil
  }
  measure = ->(e : ML::Metal::ComputeEncoder, n : Int32, total : ML::MetalBuffer) {
    e.set_pipeline(stats).set_buffer(stage, 0).set_buffer(original, 1).set_buffer(packed["qbit_p4"], 2)
      .set_buffer(sp, 3, ML::Metal::BufferAccess::Write).set_value(n.to_u32, 4)
      .dispatch_threadgroups({n // 256, 1, 1}, {256, 1, 1})
    e.set_pipeline(reduce).set_buffer(sp, 0).set_buffer(total, 1, ML::Metal::BufferAccess::Write)
      .set_value((n // 256).to_u32, 2).dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
    nil
  }
  puts({schema: "wan_gaussian_diagnostic_setup_v1", manifest_sha256: manifest_hash,
        diagnostic_manifest_sha256: diagnostic_hash, explicit_buffer_bytes: arena.resident_bytes,
        rows: rows, source_rows: source_rows, input_scope: "four seeded block rows repeated; no denoising trajectory",
        timing_claim: false, inter_operator_host_waits: 0, activation_readbacks: 0}.to_json)
  variants.each do |variant|
    v = vm[variant]
    packed_probe_upload(File.join(fixture_dir, v["decoded_weight_file"].as_s), expected_weight, expected_weight.size)
    packed_probe_upload(File.join(fixture_dir, v["expected_projection_file"].as_s), expected_output, expected_output.size)
    command = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(command)
    before = ML::MetalBuffer.video_spec_allocations
    decode.call(e, variant, variant == "qbit_p5" ? lut5 : lut4)
    packed_probe_compare(e, zero, compare, fold, stage, expected_weight, wm, wp, weight_count)
    measure.call(e, weight_count, st); project.call(e)
    if variant == "bf16"
      e.set_pipeline(copy).set_buffer(output, 0).set_buffer(baseline, 1, ML::Metal::BufferAccess::Write)
        .set_value(output_count.to_u32, 2).dispatch_1d(output_count)
    end
    packed_probe_compare(e, zero, compare, fold, output, expected_output, om, op, output_count)
    packed_probe_compare(e, zero, compare, fold, output, baseline, lm, lp, output_count)
    e.end_encoding; command.commit; command.wait
    raise "Encode allocated MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
    w = packed_probe_metrics(wm, wp, weight_count); o = packed_probe_metrics(om, op, output_count)
    raise "Original format parity failed: #{variant}" unless w[:tolerance_violations] == 0 && w[:nonfinite] == 0 && o[:tolerance_violations] == 0 && o[:nonfinite] == 0
    puts({schema: "wan_gaussian_diagnostic_result_v1", variant: variant, mode: "original",
          decoder: w, projection: o, lossy_vs_bf16: packed_probe_metrics(lm, lp, output_count),
          stats: gaussian_diagnostic_stats(st, weight_count)}.to_json); STDOUT.flush
  end
  arms.each do |arm|
    name = arm["name"].as_s; precision = arm["precision"].as_i; mode = arm["mode"].as_s
    lut = precision == 4 ? lut4 : lut5
    if mode == "arithmetic"
      packed_probe_upload(File.join(diagnostic_dir, arm["lut"]["path"].as_s), ablation_lut, 1024_i64)
      lut = ablation_lut
    end
    packed_probe_upload(File.join(diagnostic_dir, arm["projection"]["path"].as_s), expected_source, expected_source.size)
    command = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(command)
    before = ML::MetalBuffer.video_spec_allocations
    e.set_pipeline(tile).set_buffer(expected_source, 0).set_buffer(expected_output, 1, ML::Metal::BufferAccess::Write)
      .set_value((source_rows * dout).to_u32, 2).set_value(output_count.to_u32, 3).dispatch_1d(output_count)
    decode.call(e, "qbit_p#{precision}", lut)
    if mode != "arithmetic"
      e.set_pipeline(transform).set_buffer(stage, 0, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(original, 1).set_buffer(packed["qbit_p#{precision}"], 2).set_buffer(lut, 3)
        .set_value(weight_count.to_u32, 4).set_value(precision.to_u32, 5)
        .set_value(mode == "tail_restore" ? 1_u32 : 2_u32, 6)
        .set_value(mode == "tail_restore" ? arm["threshold"].as_f.to_f32 : 0.0_f32, 7)
        .dispatch_1d(weight_count)
    end
    measure.call(e, weight_count, st); project.call(e)
    packed_probe_compare(e, zero, compare, fold, output, expected_output, om, op, output_count)
    packed_probe_compare(e, zero, compare, fold, output, baseline, lm, lp, output_count)
    e.end_encoding; command.commit; command.wait
    raise "Encode allocated MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
    o = packed_probe_metrics(om, op, output_count)
    raise "Diagnostic projection parity failed: #{name}" unless o[:tolerance_violations] == 0 && o[:nonfinite] == 0
    puts({schema: "wan_gaussian_diagnostic_result_v1", variant: name, mode: mode,
          projection: o, lossy_vs_bf16: packed_probe_metrics(lm, lp, output_count),
          stats: gaussian_diagnostic_stats(st, weight_count), production_codec: false}.to_json); STDOUT.flush
  end
  adaptive_arms.each do |arm|
    packed_probe_upload(File.join(adaptive_dir, arm["projection"]["path"].as_s), expected_source, expected_source.size)
    packed_probe_upload(File.join(adaptive_dir, arm["tiers"]["path"].as_s), adaptive_tiers, adaptive_tiers.size)
    t4=arm["p4_max_error"].as_f.to_f32; t5=arm["p5_max_error"].as_f.to_f32
    command=ML::Metal::CommandBuffer.new
    e=ML::Metal::ComputeEncoder.new(command)
    before=ML::MetalBuffer.video_spec_allocations
    e.set_pipeline(tile).set_buffer(expected_source, 0).set_buffer(expected_output, 1, ML::Metal::BufferAccess::Write)
      .set_value((source_rows * dout).to_u32, 2).set_value(output_count.to_u32, 3).dispatch_1d(output_count)
    decode.call(e,"qbit_p5",lut5)
    e.set_pipeline(copy).set_buffer(stage,0).set_buffer(expected_weight,1,ML::Metal::BufferAccess::Write)
      .set_value(weight_count.to_u32,2).dispatch_1d(weight_count)
    decode.call(e,"qbit_p4",lut4)
    e.set_pipeline(zero).set_buffer(adaptive_metrics,0,ML::Metal::BufferAccess::Write).dispatch_1d(5)
    e.set_pipeline(adaptive_control).set_buffer(adaptive_metrics,0,ML::Metal::BufferAccess::ReadWrite)
      .set_value(t4,1).set_value(t5,2).dispatch_1d(12)
    e.set_pipeline(adaptive_select).set_buffer(stage,0,ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(expected_weight,1).set_buffer(original,2).set_buffer(packed["qbit_p4"],3)
      .set_buffer(adaptive_tiers,4).set_buffer(adaptive_metrics,5,ML::Metal::BufferAccess::ReadWrite)
      .set_value(t4,6).set_value(t5,7).dispatch_threadgroups({weight_count // 256,1,1},{256,1,1})
    measure.call(e,weight_count,st); project.call(e)
    packed_probe_compare(e,zero,compare,fold,output,expected_output,om,op,output_count)
    packed_probe_compare(e,zero,compare,fold,output,baseline,lm,lp,output_count)
    e.end_encoding;command.commit;command.wait
    raise "Adaptive encode allocated MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
    m=adaptive_metrics.contents.as(UInt32*)
    raise "Adaptive tier/control gate failed" unless m[0]+m[1]+m[2] == (weight_count // 256).to_u32 && m[3]==0 && m[4]==0
    o=packed_probe_metrics(om,op,output_count)
    raise "Adaptive projection parity failed" unless o[:tolerance_violations]==0 && o[:nonfinite]==0
    # Same 4-byte per-block metadata forecast as the old Q6 weight probe.
    bytes=m[0].to_i64*140+m[1].to_i64*172+m[2].to_i64*516
    puts({schema: "wan_qbit_adaptive_weight_result_v1",name: arm["name"].as_s,
          p4_max_error: t4,p5_max_error: t5,tier_counts: {p4: m[0],p5: m[1],bf16: m[2]},
          tier_mismatches: m[3],selector_control_failures: m[4],projection: o,
          lossy_vs_bf16: packed_probe_metrics(lm,lp,output_count),stats: gaussian_diagnostic_stats(st,weight_count),
          forecast_metadata_bytes_per_block: 4,forecast_weight_bytes: bytes,
          forecast_bits_per_weight: bytes*8.0/weight_count,production_codec: false,
          compact_mixed_payload_measured: false,adaptive_manifest_sha256: adaptive_hash}.to_json);STDOUT.flush
  end
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  e.set_pipeline(corrupt).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  packed_probe_compare(e, zero, compare, fold, output, expected_output, om, op, output_count)
  e.set_pipeline(stats_control).set_buffer(stage, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(original, 1, ML::Metal::BufferAccess::Write)
    .set_buffer(packed["qbit_p4"], 2, ML::Metal::BufferAccess::Write)
    .set_value(weight_count.to_u32, 3).set_value(0_u32, 4).dispatch_1d(weight_count)
  measure.call(e, weight_count, st)
  e.set_pipeline(stats_control).set_buffer(stage, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(original, 1, ML::Metal::BufferAccess::Write)
    .set_buffer(packed["qbit_p4"], 2, ML::Metal::BufferAccess::Write)
    .set_value(256_u32, 3).set_value(1_u32, 4).dispatch_1d(256)
  measure.call(e, 256, boundary_stats)
  e.end_encoding; command.commit; command.wait
  control = packed_probe_metrics(om, op, output_count)
  raise "Corruption comparator failed" unless control[:tolerance_violations] == 2 && control[:nonfinite] == 1
  stats_values = st.contents.as(Float32*)
  expected_stats = [16.0_f32 * weight_count, 16.0_f32 * weight_count, 0.0_f32, 256.0_f32 * weight_count,
                    16.0_f32 * weight_count, 16.0_f32 * weight_count, 16.0_f32 * weight_count, 16.0_f32 * weight_count,
                    weight_count.to_f32, weight_count.to_f32, weight_count.to_f32, weight_count.to_f32,
                    4.0_f32, 4.0_f32, 0.0_f32, 0.0_f32]
  raise "Diagnostic statistics control failed" unless (0...16).all? { |i| stats_values[i] == expected_stats[i] }
  expected_boundary = gaussian_diagnostic_boundary_expected
  boundary_values = boundary_stats.contents.as(Float32*)
  raise "Diagnostic boundary statistics control failed" unless (0...16).all? do |i|
                                                                 actual = boundary_values[i].to_f64
                                                                 actual.finite? && (actual - expected_boundary[i]).abs <= 2e-5 + 2e-5 * expected_boundary[i].abs
                                                               end
  puts({schema: "wan_gaussian_diagnostic_controls_v1", corruption_rejected: true, comparator: control,
        statistics_control_passed: true, statistics_expected: expected_stats,
        boundary_control_passed: true, boundary_expected: expected_boundary,
        boundary_actual: (0...16).map { |i| boundary_values[i] },
        manifest_sha256: manifest_hash, diagnostic_manifest_sha256: diagnostic_hash}.to_json)
ensure
  plan.try(&.release)
  arena.release
end
