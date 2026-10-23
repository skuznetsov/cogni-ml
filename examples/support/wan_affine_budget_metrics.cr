# GPU-reduced diagnostics shared by the opt-in equal-budget probe.
def affine_budget_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

def affine_budget_upload(path : String, buffer : ML::MetalBuffer, count : Int64) : Nil
  raise "Fixture byte count mismatch: #{path}" unless File.size(path) == count && buffer.size >= count
  File.open(path, "rb") { |file| file.read_fully(Slice.new(buffer.contents.as(UInt8*), count.to_i32)) }
end

def affine_budget_verify_file(directory : String, record : JSON::Any, name : String) : Nil
  unless record["path"].as_s == name && !Path[name].absolute? && !name.split('/').includes?("..")
    raise "Fixture descriptor path mismatch: #{name}"
  end
  path = File.join(directory, name)
  unless File.size(path) == record["bytes"].as_i64 && affine_budget_sha256(path) == record["sha256"].as_s
    raise "Fixture bytes/hash mismatch: #{name}"
  end
end

AFFINE_BUDGET_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void affine_budget_tile(device const float *src [[buffer(0)]],
      device float *dst [[buffer(1)]], constant uint &source_count [[buffer(2)]],
      constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = src[i % source_count];
  }
  kernel void affine_budget_copy(device const float *src [[buffer(0)]],
      device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
      uint i [[thread_position_in_grid]]) { if (i < count) dst[i] = src[i]; }
  kernel void affine_budget_zero(device atomic_uint *m [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i < 5u) atomic_store_explicit(m+i, 0u, memory_order_relaxed);
  }
  kernel void affine_budget_corrupt(device float *x [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i == 0u) { x[0] += 1.0f; x[1] = as_type<float>(0x7fc00000u); }
  }
  kernel void affine_budget_bytes(device const uint *a [[buffer(0)]],
      device const uint *b [[buffer(1)]], device atomic_uint *m [[buffer(2)]],
      constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < count && a[i] != b[i]) atomic_fetch_add_explicit(m, 1u, memory_order_relaxed);
  }
  kernel void affine_budget_compare(device const float *a [[buffer(0)]],
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
        float d=av-bv, error=abs(d), tolerance=2.0e-5f+2.0e-5f*abs(bv);
        float ratio=error/tolerance;
        c.y = error > tolerance; mx=float2(error,ratio);
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
METAL

def affine_budget_compare(e : ML::Metal::ComputeEncoder,
                          zero : ML::Metal::ComputePipeline,
                          compare : ML::Metal::ComputePipeline,
                          actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                          metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
                          count : Int32, offset : Int32 = 0) : Nil
  e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  e.set_pipeline(compare).set_buffer(actual, 0, ML::Metal::BufferAccess::Read, offset.to_i64 * 4)
    .set_buffer(expected, 1, ML::Metal::BufferAccess::Read, offset.to_i64 * 4)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(partials, 3, ML::Metal::BufferAccess::Write).set_value(count.to_u32, 4)
    .dispatch_threadgroups({(count + 255) // 256, 1, 1}, {256, 1, 1})
end

def affine_budget_metrics(metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
                          count : Int32)
  words = metrics.contents.as(UInt32*)
  data = partials.contents.as(Float32*)
  # Only terminal, GPU-reduced statistics cross this boundary, never activations.
  sum_error = 0.0; sum_ref = 0.0; sum_actual = 0.0; dot = 0.0
  ((count + 255) // 256).times do |group|
    sum_error += data[group * 4].to_f64
    sum_ref += data[group * 4 + 1].to_f64
    sum_actual += data[group * 4 + 2].to_f64
    dot += data[group * 4 + 3].to_f64
  end
  {compared_values: count, bit_differences: words[0], tolerance_violations: words[1],
   nonfinite: words[2], max_abs: words[3].unsafe_as(Float32),
   max_ratio: words[4].unsafe_as(Float32), rmse: Math.sqrt(sum_error / count),
   relative_l2: Math.sqrt(sum_error / Math.max(sum_ref, 1e-300)),
   cosine: dot / Math.sqrt(Math.max(sum_ref * sum_actual, 1e-300))}
end
