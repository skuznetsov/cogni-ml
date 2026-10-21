# Selected-query precision comparison; no model or complete block is run.
require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/metal_attention_mps"
require "../src/ml/video/metal_ops"
require "../spec/support/video_allocation_counter"

private PRECISION_K         = 36_608
private PRECISION_H         =     40
private PRECISION_D         =    128
private PRECISION_C         = PRECISION_H * PRECISION_D
private PRECISION_CHUNK     = 128
private PRECISION_FORMULA   = "avalanche32(native_flat_index + salt*0x9e3779b9); ((hash & 1023) - 512) / 256"
private PRECISION_TOLERANCE = "2e-5 + 2e-5 * abs(reference)"
private PRECISION_SOURCE    = <<-METAL
#include <metal_stdlib>
using namespace metal;
inline float value(uint i, uint salt) {
  uint x=i+salt*0x9e3779b9u;
  x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16;
  return float(int(x&1023u)-512)/256.0f;
}
inline float load(device const uchar *p, uint i, uint type) {
  if (type==0u) return ((device const float *)p)[i];
  if (type==1u) return float(((device const half *)p)[i]);
  return as_type<float>(uint(((device const ushort *)p)[i])<<16);
}
inline void store(device uchar *p, uint i, uint type, float x) {
  if (type==0u) ((device float *)p)[i]=x;
  else if (type==1u) ((device half *)p)[i]=half(x);
  else {
    uint u=as_type<uint>(x);
    // Round-to-nearest-even; preserve NaNs instead of turning them into infinity.
    ushort b=ushort((u+0x7fffu+((u>>16)&1u))>>16);
    if ((u&0x7fffffffu)>0x7f800000u) b=ushort((u>>16)|0x40u);
    ((device ushort *)p)[i]=b;
  }
}
kernel void precision_fill(device uchar *out [[buffer(0)]],
    constant uint &n [[buffer(1)]], constant uint &base [[buffer(2)]],
    constant uint &salt [[buffer(3)]], constant uint &type [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if(i<n) store(out,i,type,value(base+i,salt));
}
kernel void precision_nan(device uchar *out [[buffer(0)]],
    constant uint &n [[buffer(1)]], constant uint &type [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if(i<n) store(out,i,type,as_type<float>(0x7fc00000u));
}
kernel void precision_tail(device uchar *out [[buffer(0)]],
    constant uint &base [[buffer(1)]], constant uint &n [[buffer(2)]],
    constant uint &type [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if(i<n) store(out,base+i,type,0.0f);
}
kernel void precision_convert(device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]], constant uint &n [[buffer(2)]],
    constant uint &src_type [[buffer(3)]], constant uint &dst_type [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if(i<n) store(output,i,dst_type,load(input,i,src_type));
}
kernel void precision_transpose_kv(device const uchar *k [[buffer(0)]],
    device uchar *head_major_k [[buffer(1)]], device const uchar *v [[buffer(2)]],
    device uchar *head_major_v [[buffer(3)]], constant uint &batches [[buffer(4)]],
    constant uint &key_rows [[buffer(5)]], constant uint &heads [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]], constant uint &count [[buffer(8)]],
    constant uint &type [[buffer(9)]], uint i [[thread_position_in_grid]]) {
  if(i>=count) return;
  uint d=i%head_dim, key=(i/head_dim)%key_rows;
  uint head=(i/(head_dim*key_rows))%heads;
  uint batch=i/(head_dim*key_rows*heads);
  if(batch>=batches) return;
  uint source=((batch*key_rows+key)*heads+head)*head_dim+d;
  store(head_major_k,i,type,load(k,source,type));
  store(head_major_v,i,type,load(v,source,type));
}
kernel void precision_zero(device atomic_uint *m [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
  if(i<5u) atomic_store_explicit(m+i,0u,memory_order_relaxed);
}
kernel void precision_compare(device const float *reference [[buffer(0)]],
    device const uchar *actual [[buffer(1)]],
    device atomic_uint *m [[buffer(2)]], constant uint &n [[buffer(3)]],
    constant uint &qrows [[buffer(4)]], constant uint &type [[buffer(5)]],
    constant uint &bhqd [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if(i>=n) return;
  uint r=i;
  if(bhqd!=0u) {
    uint query=i/5120u, h=(i%5120u)/128u, d=i%128u;
    r=h*qrows*128u+query*128u+d;
  }
  float ref=reference[r], got=load(actual,i,type);
  if(!isfinite(ref)||!isfinite(got)) {
    atomic_fetch_add_explicit(m+1,1u,memory_order_relaxed);
    atomic_fetch_add_explicit(m+2,1u,memory_order_relaxed);
    return;
  }
  if(as_type<uint>(ref)==as_type<uint>(got)) return;
  atomic_fetch_add_explicit(m,1u,memory_order_relaxed);
  float e=abs(ref-got), ratio=e/(2e-5f+2e-5f*abs(ref));
  if(ratio>1.0f) atomic_fetch_add_explicit(m+1,1u,memory_order_relaxed);
  atomic_fetch_max_explicit(m+3,as_type<uint>(e),memory_order_relaxed);
  atomic_fetch_max_explicit(m+4,as_type<uint>(ratio),memory_order_relaxed);
}
METAL

lib WanPrecisionFFI
  # Darwin _time.h clock IDs; these CPU clocks do not wait for GPU work.
  fun cpu_clock = clock_gettime_nsec_np(clock_id : UInt32) : UInt64
  fun profile_queue_create = gs_wan_attention_profile_queue_create(device : Void*, capacity : UInt32) : Void*
  fun profile_queue_release = gs_wan_attention_profile_queue_release(queue : Void*) : Void
  fun create = gs_wan_attention_mps_plan_create_typed(device : Void*, batches : Int32,
                                                      query_rows : Int32, key_rows : Int32, heads : Int32, head_dim : Int32,
                                                      q : Void*, k : Void*, v : Void*, a : Void*, b : Void*, dtype : Int32) : Void*
  fun create_connected = gs_wan_attention_mps_plan_create_connected_typed(device : Void*, batches : Int32,
                                                                          query_rows : Int32, key_rows : Int32, heads : Int32, head_dim : Int32,
                                                                          q : Void*, k : Void*, v : Void*, a : Void*, b : Void*, dtype : Int32,
                                                                          head_major_kv : Int32) : Void*
end

private class PrecisionQueue
  @handle : Void* = Pointer(Void).null

  def initialize(@capacity : Int32)
    if @capacity != 0
      @handle = WanPrecisionFFI.profile_queue_create(ML::Metal::Device.instance.device_handle, @capacity.to_u32)
      raise "Precision queue unavailable" if @handle.null?
    end
  end

  def chain
    @capacity == 0 ? ML::Metal::MPSCommandChain.new : ML::Metal::MPSCommandChain.new(@handle)
  end

  def release
    WanPrecisionFFI.profile_queue_release(@handle) unless @handle.null?
    @handle = Pointer(Void).null
  end
end

private record PrecisionHostStamp, wall : Time::Instant, thread_ns : UInt64, process_ns : UInt64 do
  def self.now
    new(Time.instant, WanPrecisionFFI.cpu_clock(16_u32), WanPrecisionFFI.cpu_clock(12_u32))
  end
end

# Opt-in observer, constructed before the arm. No per-chunk records or GPU calls.
private class PrecisionHostProfile
  @wall = StaticArray(Float64, 6).new(0.0)
  @thread = StaticArray(Float64, 6).new(0.0)
  @process = StaticArray(Float64, 6).new(0.0)
  @calls = StaticArray(Int32, 6).new(0)

  def initialize(@enabled : Bool)
  end

  def measure(slot : Int32, & : -> Nil) : Nil
    if @enabled
      before = PrecisionHostStamp.now
      yield
      after = PrecisionHostStamp.now
      @wall[slot] += (after.wall - before.wall).total_seconds
      @thread[slot] += (after.thread_ns - before.thread_ns)*1e-9
      @process[slot] += (after.process_ns - before.process_ns)*1e-9
      @calls[slot] += 1
    else
      yield
    end
  end

  def summary(total_before : PrecisionHostStamp?, total_after : PrecisionHostStamp?)
    return nil unless @enabled
    before = total_before.not_nil!; after = total_after.not_nil!
    {clock:                     "Darwin calling-thread and process CPU clocks; elapsed intervals can include backpressure",
     total_thread_cpu_seconds:  (after.thread_ns - before.thread_ns)*1e-9,
     total_process_cpu_seconds: (after.process_ns - before.process_ns)*1e-9,
     stages:                    {"prepare", "query_copy", "tail", "graph", "output_copy", "final_convert"}.map_with_index do |name, i|
       {name: name, calls: @calls[i], wall_seconds: @wall[i],
        thread_cpu_seconds: @thread[i], process_cpu_seconds: @process[i]}
     end}
  end
end

class ML::Metal::MPSCommandChain
  # Probe-only constructor. The default runtime queue is unchanged. Each arm's
  # existing terminal wait orders its inputs/outputs with setup and validation.
  def initialize(queue : Pointer(Void))
    raise "Missing precision queue" if queue.null?
    @native = MPSCommandChainFFI.create(queue)
    raise "Failed to create precision chain" if @native.null?
    resume!
  end

  # Read only timestamps already retained by the chain, after its existing wait.
  def precision_root_timing_summary
    raise "Root timing requires the existing terminal wait" unless completed? && wait_count == 1
    raise "Precision root metadata cap exceeded" unless 0 < segment_count <= 256
    earliest = Float64::INFINITY
    latest = 0.0
    gaps = 0.0
    segment_count.times do |sequence|
      completed = 0_i32; has_error = 0_i32
      start = 0.0; finish = 0.0
      status = MPSCommandChainFFI.root_timing(@native, sequence.to_i32,
        pointerof(completed), pointerof(has_error), pointerof(start), pointerof(finish))
      raise "Invalid completed root timing" unless status == 0 && completed == 1 && has_error == 0 &&
                                                   start.finite? && finish.finite? && start > 0 && finish >= start
      earliest = Math.min(earliest, start)
      gaps += Math.max(0.0, start - latest) if sequence > 0
      latest = Math.max(latest, finish)
    end
    {gpu_envelope_seconds: latest - earliest, inter_root_gap_seconds: gaps,
     boundary: "completed root timestamp envelope; cannot separate same-root CPU/GPU work"}
  end
end

# Probe-local typed wrapper. Existing production Crystal plans remain F32.
private class PrecisionPlan
  getter graph : String
  getter kv_layout : String
  @handle : Void*
  @chain : ML::Metal::MPSCommandChain? = nil

  def initialize(@buffers : Array(ML::MetalBuffer), type : Int32,
                 @graph : String, @kv_layout : String)
    q, k, v, a, b = @buffers
    device = ML::Metal::Device.instance.device_handle
    @handle = if @graph == "connected"
                WanPrecisionFFI.create_connected(device, 1, PRECISION_CHUNK,
                  PRECISION_K, PRECISION_H, PRECISION_D, q.handle, k.handle,
                  v.handle, a.handle, b.handle, type, @kv_layout == "head-major" ? 1 : 0)
              else
                WanPrecisionFFI.create(device, 1, PRECISION_CHUNK, PRECISION_K,
                  PRECISION_H, PRECISION_D, q.handle, k.handle, v.handle,
                  a.handle, b.handle, type)
              end
    raise "Typed MPSGraph plan unavailable (dtype=#{type})" if @handle.null?
  end

  def encode(chain : ML::Metal::MPSCommandChain, *, encoder_paused : Bool = false) : Nil
    raise "Released precision plan" if @handle.null?
    if prior = @chain
      raise "Precision plan overlaps an incomplete chain" unless prior.same?(chain) || prior.completed?
    end
    raise "Invalid precision buffer" unless @buffers.all?(&.valid?)
    @chain = chain
    chain.pause! unless encoder_paused
    begin
      status = WanAttentionMPSFFI.encode(@handle, chain.handle, 0)
      raise "Typed attention encode failed (#{status})" unless status == 0
    ensure
      chain.resume! unless encoder_paused
    end
  end

  def release
    return if @handle.null?
    if chain = @chain
      raise "Complete precision chain before release" unless chain.completed?
    end
    WanAttentionMPSFFI.plan_release(@handle)
    @handle = Pointer(Void).null
  end
end

private def precision_hash(path : String)
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def precision_fixture(dir : String, dtype : String, rows : Int32,
                              start : Int32, q_salt : UInt32)
  path = File.join(dir, "manifest.json")
  manifest = JSON.parse(File.read(path))
  raise "Unsupported precision fixture" unless manifest["schema"].as_s == "wan_s2v_attention_precision_reference_v1"
  g = manifest["geometry"]
  {"batches" => 1_i64, "query_rows" => rows.to_i64, "query_start" => start.to_i64,
   "key_rows" => PRECISION_K.to_i64, "heads" => PRECISION_H.to_i64,
   "head_dim" => PRECISION_D.to_i64}.each do |key, v|
    raise "Precision fixture geometry mismatch: #{key}" unless g[key].as_i64 == v
  end
  raise "Precision fixture formula mismatch" unless manifest["input_formula"].as_s == PRECISION_FORMULA
  salts = manifest["input_salts"]
  raise "Precision fixture salts mismatch" unless salts["q"].as_i64 == q_salt.to_i64 && salts["k"].as_i64 == 2 && salts["v"].as_i64 == 3
  ref = manifest["reference"]
  oracle_dtype = {"f32" => "float32", "f16" => "float16", "bf16" => "bfloat16"}[dtype]
  raise "Precision fixture dtype mismatch" unless ref["dtype"].as_s == oracle_dtype
  raise "Precision fixture backend mismatch" unless ref["device"].as_s == "mps" && ref["mps_fallback_disabled"].as_bool
  raise "Precision fixture tolerance mismatch" unless ref["comparison_tolerance"].as_s == PRECISION_TOLERANCE
  raise "Precision fixture chunk mismatch" unless ref["query_chunk_rows"].as_i64 == PRECISION_CHUNK
  source = manifest["source"]
  raise "Official attention source mismatch" unless source["wan_attention_source_sha256"].as_s == "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc"
  unless ref["operator"].as_s == "pinned wan_modules_attention.py::attention SDPA fallback" &&
         !ref["flash_attention_2_available"].as_bool && !ref["flash_attention_3_available"].as_bool
    raise "Precision oracle must execute the official MPS fallback"
  end
  {"wan_attention_source_path" => "wan_attention_source_sha256", "generator_path" => "generator_sha256"}.each do |pk, hk|
    raise "Precision source pin mismatch" unless precision_hash(source[pk].as_s) == source[hk].as_s
  end
  e = manifest["expected"]
  raise "Precision expected filename mismatch" unless e["file"].as_s == "expected_bhqd.f32"
  expected = File.join(dir, "expected_bhqd.f32")
  bytes = rows.to_i64*PRECISION_C*4
  raise "Precision expected layout mismatch" unless e["shape"].as_a.map(&.as_i64) == [1_i64, PRECISION_H.to_i64, rows.to_i64, PRECISION_D.to_i64] && e["dtype"].as_s == "F32_LE"
  raise "Precision expected size mismatch" unless e["bytes"].as_i64 == bytes && File.size(expected) == bytes
  raise "Precision expected hash mismatch" unless precision_hash(expected) == e["sha256"].as_s
  {manifest_path: path, manifest_sha256: precision_hash(path), expected_path: expected,
   expected_sha256: e["sha256"].as_s, source: source}
end

private def precision_load(path : String, buffer : ML::MetalBuffer) : Nil
  File.open(path, "rb") do |file|
    pointer = buffer.contents.as(UInt8*)
    offset = 0_i64
    while offset < buffer.size
      n = Math.min(1_i64 << 20, buffer.size - offset).to_i32
      count = file.read(Bytes.new(pointer + offset, n))
      raise "Truncated precision fixture" if count == 0
      offset += count
    end
  end
end

private def precision_native_guards(buffers : Array(ML::MetalBuffer), short : ML::MetalBuffer,
                                    type : Int32, graph : String,
                                    kv_layout : String) : Nil
  q, k, v, a, b = buffers
  device = ML::Metal::Device.instance.device_handle
  # Reject malformed bindings before graph compilation or command submission.
  invalid = [
    WanPrecisionFFI.create(device, 1, 128, PRECISION_K, 40, 128,
      q.handle, k.handle, v.handle, a.handle, b.handle, -1),
    WanPrecisionFFI.create(device, 1, 128, PRECISION_K, 40, 128,
      short.handle, k.handle, v.handle, a.handle, b.handle, type),
    WanPrecisionFFI.create(device, 1, 128, PRECISION_K, 40, 128,
      q.handle, k.handle, k.handle, a.handle, b.handle, type),
    WanPrecisionFFI.create(device, 1, 0, PRECISION_K, 40, 128,
      q.handle, k.handle, v.handle, a.handle, b.handle, type),
    WanPrecisionFFI.create(device, Int32::MAX, Int32::MAX, Int32::MAX, 40, 128,
      q.handle, k.handle, v.handle, a.handle, b.handle, type),
  ]
  invalid.each do |handle|
    unless handle.null?
      WanAttentionMPSFFI.plan_release(handle)
      raise "Malformed typed attention binding was admitted"
    end
  end
  connected_rejected_cases = 0
  if graph == "connected"
    connected_invalid = [
      WanPrecisionFFI.create_connected(device, 1, 128, PRECISION_K, 40, 128,
        q.handle, k.handle, v.handle, a.handle, b.handle, 2,
        kv_layout == "head-major" ? 1 : 0),
      WanPrecisionFFI.create_connected(device, 1, 128, PRECISION_K, 40, 128,
        q.handle, k.handle, v.handle, a.handle, b.handle, type, 2),
      WanPrecisionFFI.create_connected(device, 1, 128, PRECISION_K, 40, 128,
        short.handle, k.handle, v.handle, a.handle, b.handle, type,
        kv_layout == "head-major" ? 1 : 0),
      WanPrecisionFFI.create_connected(device, 1, 128, PRECISION_K, 40, 128,
        q.handle, k.handle, k.handle, a.handle, b.handle, type,
        kv_layout == "head-major" ? 1 : 0),
      WanPrecisionFFI.create_connected(device, 1, 0, PRECISION_K, 40, 128,
        q.handle, k.handle, v.handle, a.handle, b.handle, type,
        kv_layout == "head-major" ? 1 : 0),
      WanPrecisionFFI.create_connected(device, Int32::MAX, Int32::MAX, Int32::MAX, 40, 128,
        q.handle, k.handle, v.handle, a.handle, b.handle, type,
        kv_layout == "head-major" ? 1 : 0),
    ]
    connected_invalid.each do |handle|
      unless handle.null?
        WanAttentionMPSFFI.plan_release(handle)
        raise "Malformed connected typed attention binding was admitted"
      end
    end
    connected_rejected_cases = connected_invalid.size
  end
  puts({schema: "wan_attention_precision_binding_guards_v1", dtype: type,
        graph: graph, kv_layout: kv_layout, rejected_cases: invalid.size,
        connected_rejected_cases: connected_rejected_cases,
        gpu_commands_submitted: false}.to_json)
end

private def precision_nan(encoder, pipeline, buffer, n : Int32, type : Int32)
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(n.to_u32, 1).set_value(type.to_u32, 2).dispatch_1d(n)
end

private def precision_fill(encoder, pipeline, buffer, n : Int32, base : UInt32,
                           salt : UInt32, type : Int32)
  encoder.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
  encoder.set_value(n.to_u32, 1).set_value(base, 2).set_value(salt, 3)
  encoder.set_value(type.to_u32, 4).dispatch_1d(n)
end

private def precision_convert(encoder, pipeline, src, dst, n : Int32,
                              src_type : Int32, dst_type : Int32)
  encoder.set_pipeline(pipeline).set_buffer(src, 0).set_buffer(dst, 1, ML::Metal::BufferAccess::Write)
  encoder.set_value(n.to_u32, 2).set_value(src_type.to_u32, 3)
  encoder.set_value(dst_type.to_u32, 4).dispatch_1d(n)
end

private def precision_transpose_kv(encoder, pipeline, k, head_major_k, v, head_major_v,
                                   type : Int32)
  encoder.set_pipeline(pipeline).set_buffer(k, 0).set_buffer(head_major_k, 1, ML::Metal::BufferAccess::Write)
  encoder.set_buffer(v, 2).set_buffer(head_major_v, 3, ML::Metal::BufferAccess::Write)
  encoder.set_value(1_u32, 4).set_value(PRECISION_K.to_u32, 5)
  encoder.set_value(PRECISION_H.to_u32, 6).set_value(PRECISION_D.to_u32, 7)
  encoder.set_value((PRECISION_K*PRECISION_C).to_u32, 8).set_value(type.to_u32, 9)
  encoder.dispatch_1d(PRECISION_K*PRECISION_C)
end

private def precision_compare(encoder, zero, compare, ref, actual, metrics,
                              n : Int32, rows : Int32, type : Int32, bhqd : Bool)
  encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  encoder.set_pipeline(compare).set_buffer(ref, 0).set_buffer(actual, 1)
  encoder.set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
  encoder.set_value(n.to_u32, 3).set_value(rows.to_u32, 4).set_value(type.to_u32, 5)
  encoder.set_value(bhqd ? 1_u32 : 0_u32, 6).dispatch_1d(n)
end

private def precision_metrics(buffer)
  p = buffer.contents.as(UInt32*)
  {bit_differences: p[0], strict_tolerance_violations: p[1], nonfinite: p[2],
   max_abs_error: p[3].unsafe_as(Float32), max_tolerance_ratio: p[4].unsafe_as(Float32),
   passed: p[1] == 0 && p[2] == 0}
end

private def precision_finish(chain)
  begin
    chain.commit_and_wait unless chain.completed?
  ensure
    chain.release if chain.completed?
  end
end

private def precision_expect_paused_rejection(& : -> Nil)
  begin
    yield
  rescue ArgumentError
    return
  end
  raise "Paused encoder guard accepted an invalid operation"
end

private def precision_paused_guards(chain, plan, q, cq)
  # Exercise the new explicit state precondition before the existing setup work.
  # No operation below reaches a native encode/copy and no extra wait is added.
  precision_expect_paused_rejection do
    ML::Video::WanAttentionMPSPlan.copy(chain, q, cq, 4_i64, encoder_paused: true)
  end
  precision_expect_paused_rejection { plan.encode(chain, encoder_paused: true) }
  chain.pause!
  begin
    precision_expect_paused_rejection { ML::Video::WanAttentionMPSPlan.copy(chain, q, cq, 4_i64) }
    precision_expect_paused_rejection { plan.encode(chain) }
    precision_expect_paused_rejection do
      ML::Video::WanAttentionMPSPlan.copy(chain, q, cq, 3_i64, encoder_paused: true)
    end
    precision_expect_paused_rejection do
      ML::Video::WanAttentionMPSPlan.copy(chain, q, q, 4_i64, encoder_paused: true)
    end
  ensure
    chain.resume!
  end
  STDOUT.puts({schema: "wan_attention_precision_paused_guards_v1", rejected_cases: 6,
               gpu_commands_submitted: false, additional_terminal_waits: 0}.to_json)
end

private def precision_arm(plan, q, k, v, output, cq, ca, sources,
                          converted_output, convert_pipeline, tail_pipeline,
                          rows : Int32, type : Int32, mode : String,
                          phase : String, sample : Int32,
                          fill_pipeline, transpose_pipeline,
                          head_major_k, head_major_v, query_start : Int32,
                          profile_host : Bool, encoder_policy : String,
                          queue : PrecisionQueue, queue_capacity : Int32)
  width = type == 0 ? 4_i64 : 2_i64
  paused = encoder_policy == "paused"
  host = PrecisionHostProfile.new(profile_host)
  before = ML::MetalBuffer.video_spec_allocations
  started = Time.instant
  chain = queue.chain
  begin
    encoding = Time.instant
    cpu_before = profile_host ? PrecisionHostStamp.now : nil
    host.measure(0) do
      if phase == "changed_query"
        src = sources
        precision_fill(chain.encoder, fill_pipeline, src ? src[0] : q, rows*PRECISION_C,
          (query_start.to_i64*PRECISION_C).to_u32, 4_u32, src ? 0 : type)
      end
      if mode == "f32-boundary"
        src = sources.not_nil!
        { {src[0], q, rows*PRECISION_C}, {src[1], k, PRECISION_K*PRECISION_C},
         {src[2], v, PRECISION_K*PRECISION_C} }.each do |source, dest, n|
          precision_convert(chain.encoder, convert_pipeline, source, dest, n, 0, type)
        end
      end
      if prepared_k = head_major_k
        precision_transpose_kv(chain.encoder, transpose_pipeline.not_nil!, k, prepared_k,
          v, head_major_v.not_nil!, type)
      end
    end
    chain.pause! if paused
    offset = 0
    while offset < rows
      valid = Math.min(PRECISION_CHUNK, rows - offset)
      n = valid.to_i64*PRECISION_C
      host.measure(1) do
        ML::Video::WanAttentionMPSPlan.copy(chain, q, cq, n*width, offset.to_i64*PRECISION_C*width,
          encoder_paused: paused)
      end
      if valid < PRECISION_CHUNK
        host.measure(2) do
          chain.resume! if paused
          tail_n = (PRECISION_CHUNK - valid)*PRECISION_C
          chain.encoder.set_pipeline(tail_pipeline).set_buffer(cq, 0, ML::Metal::BufferAccess::Write)
          chain.encoder.set_value(n.to_u32, 1).set_value(tail_n.to_u32, 2)
          chain.encoder.set_value(type.to_u32, 3).dispatch_1d(tail_n)
          chain.pause! if paused
        end
      end
      host.measure(3) { plan.encode(chain, encoder_paused: paused) }
      host.measure(4) do
        ML::Video::WanAttentionMPSPlan.copy(chain, ca, output, n*width, 0_i64, offset.to_i64*PRECISION_C*width,
          encoder_paused: paused)
      end
      offset += valid
    end
    if mode == "f32-boundary"
      host.measure(5) do
        chain.resume! if paused
        precision_convert(chain.encoder, convert_pipeline, output, converted_output.not_nil!,
          rows*PRECISION_C, type, 0)
      end
    end
    cpu_after = profile_host ? PrecisionHostStamp.now : nil
    encode_seconds = (Time.instant - encoding).total_seconds
    chain.commit_and_wait
    wall = (Time.instant - started).total_seconds
    allocations = ML::MetalBuffer.video_spec_allocations - before
    raise "Precision arm allocated MetalBuffer" unless allocations == 0
    raise "Precision arm has extra wait" unless chain.wait_count == 1
    {schema: "wan_attention_precision_arm_v1", phase: phase, sample: sample,
     mode: mode, dtype: type, query_rows: rows, key_rows: PRECISION_K, chunk_rows: PRECISION_CHUNK,
     graph: plan.graph, kv_layout: plan.kv_layout,
     encoder_policy: encoder_policy,
     queue_capacity: queue_capacity == 0 ? "default64" : queue_capacity.to_s,
     expected_compute_encoder_creations: paused ? 1 + (rows % PRECISION_CHUNK == 0 ? 0 : 1) + (mode == "f32-boundary" ? 1 : 0) : 1 + 3*((rows + PRECISION_CHUNK - 1)//PRECISION_CHUNK),
     kv_preparation_included: plan.kv_layout == "head-major",
     completed_wall_seconds: wall, encode_wall_seconds: encode_seconds,
     summed_root_gpu_seconds: chain.gpu_elapsed_seconds, command_roots: chain.segment_count,
     host_profile: host.summary(cpu_before, cpu_after),
     root_timing: profile_host ? chain.precision_root_timing_summary : nil,
     terminal_waits: chain.wait_count, tracked_metal_buffer_allocations: allocations,
     device_current_allocated_bytes_after_wait: ML::Metal::Device.instance.current_allocated_size,
     boundary: "chain creation through encoding and one terminal wait; conversions included only in f32-boundary mode; BTHD-to-BHTD K/V transpose included for head-major layout"}
  ensure
    precision_finish(chain)
  end
end

private def precision_validate(actual, expected, f32_expected, output_type : Int32,
                               metrics, controls, zero, compare, rows : Int32,
                               label : String, dtype : String, mode : String,
                               graph : String, kv_layout : String,
                               output, converted_output, nan_pipeline,
                               type : Int32)
  chain = ML::Metal::MPSCommandChain.new
  begin
    precision_compare(chain.encoder, zero, compare, expected, actual, metrics[0],
      rows*PRECISION_C, rows, output_type, true)
    precision_compare(chain.encoder, zero, compare, f32_expected, actual, metrics[1],
      rows*PRECISION_C, rows, output_type, true)
    precision_compare(chain.encoder, zero, compare, controls[0], controls[1], metrics[2], 1, 1, 0, false)
    precision_compare(chain.encoder, zero, compare, controls[0], controls[2], metrics[3], 1, 1, 0, false)
    # Reuse the existing diagnostic terminal boundary to poison the next arm.
    # There is no additional setup chain or host wait between arms.
    precision_nan(chain.encoder, nan_pipeline, output, rows*PRECISION_C, type)
    if widened = converted_output
      precision_nan(chain.encoder, nan_pipeline, widened, rows*PRECISION_C, 0)
    end
    chain.commit_and_wait
    raise "Precision diagnostic has extra wait" unless chain.wait_count == 1
    matched = precision_metrics(metrics[0]); drift = precision_metrics(metrics[1])
    plus = precision_metrics(metrics[2]); nan = precision_metrics(metrics[3])
    raise "Precision comparator controls failed" unless plus[:strict_tolerance_violations] > 0 && plus[:nonfinite] == 0 && nan[:nonfinite] > 0
    record = {schema: "wan_attention_precision_validation_v1", label: label, dtype: dtype, mode: mode,
              graph: graph, kv_layout: kv_layout,
              compared_values: rows.to_i64*PRECISION_C, dtype_matched: matched, versus_f32: drift,
              tolerance: PRECISION_TOLERANCE, positive_control_rejected: true, nan_control_rejected: true,
              diagnostic_terminal_waits: chain.wait_count, metric_words_read: 20,
              activations_read_back: false}
    STDOUT.puts(record.to_json); STDOUT.flush
    {matched_passed: matched[:passed], f32_passed: drift[:passed]}
  ensure
    precision_finish(chain)
  end
end

fixture_dir = ""
f32_fixture_dir = ""
alternate_dir = ""
alternate_f32_dir = ""
dtype = "f32"
mode = "resident"
rows = 128
start = 0
samples = 3
profile_host = false
encoder_policy = "eager"
queue_capacity = 0
graph = "opaque"
kv_layout = "bthd"
plan_only = false
OptionParser.parse do |o|
  o.banner = "Usage: wan_s2v_attention_precision_profile --fixture=DIR --f32-fixture=DIR --dtype=f32|f16|bf16"
  o.on("--fixture=DIR", "Dtype-matched official Wan fallback fixture") { |v| fixture_dir = v }
  o.on("--f32-fixture=DIR", "Unrounded F32 fixture for the same query span") { |v| f32_fixture_dir = v }
  o.on("--alternate-fixture=DIR", "Optional salt-4 fixture for same-plan changed-input check") { |v| alternate_dir = v }
  o.on("--alternate-f32-fixture=DIR", "Salt-4 F32 reference") { |v| alternate_f32_dir = v }
  o.on("--dtype=TYPE", "f32, f16 or bf16") { |v| dtype = v }
  o.on("--mode=MODE", "resident or f32-boundary") { |v| mode = v }
  o.on("--graph=GRAPH", "opaque or connected; opaque preserves the existing SDPA path") { |v| graph = v }
  o.on("--kv-layout=LAYOUT", "bthd or head-major; head-major requires the connected graph") { |v| kv_layout = v }
  o.on("--query-rows=N", "128, 257 or 8192") { |v| rows = v.to_i }
  o.on("--query-start=N", "Global query start") { |v| start = v.to_i }
  o.on("--samples=N", "0, 1 or 3 measured arms after warmup") { |v| samples = v.to_i }
  o.on("--profile-host=BOOL", "Opt-in CPU/wall stage clocks and completed-root timeline") do |v|
    raise ArgumentError.new("Unsupported host profiling flag") unless {"true", "false"}.includes?(v)
    profile_host = v == "true"
  end
  o.on("--encoder-policy=POLICY", "eager or paused; paused avoids empty compute encoders between blits/Graph") { |v| encoder_policy = v }
  o.on("--queue-capacity=N", "0 (unchanged default), 32, 64 or 128; probe-only queue ablation") { |v| queue_capacity = v.to_i }
  o.on("--plan-only", "Print memory plan without Metal/fixture access") { plan_only = true }
end
type = {"f32" => 0, "f16" => 1, "bf16" => 2}[dtype]? || raise ArgumentError.new("Unsupported dtype")
raise ArgumentError.new("Unsupported mode") unless {"resident", "f32-boundary"}.includes?(mode)
raise ArgumentError.new("Unsupported encoder policy") unless {"eager", "paused"}.includes?(encoder_policy)
raise ArgumentError.new("Unsupported queue capacity") unless {0, 32, 64, 128}.includes?(queue_capacity)
raise ArgumentError.new("Unsupported graph") unless {"opaque", "connected"}.includes?(graph)
raise ArgumentError.new("Unsupported K/V layout") unless {"bthd", "head-major"}.includes?(kv_layout)
raise ArgumentError.new("Opaque graph requires the existing BTHD K/V layout") if graph == "opaque" && kv_layout != "bthd"
raise ArgumentError.new("Connected graph supports F32 or F16 only") if graph == "connected" && type == 2
raise ArgumentError.new("Unsupported query rows") unless {128, 257, 8192}.includes?(rows)
raise ArgumentError.new("Query range out of bounds") unless start >= 0 && start.to_i64 + rows <= PRECISION_K
raise ArgumentError.new("Unsupported samples") unless {0, 1, 3}.includes?(samples)
raise ArgumentError.new("Alternate fixtures must be paired") if alternate_dir.empty? != alternate_f32_dir.empty?
raise ArgumentError.new("Alternate reuse check is bounded to 128 rows") if !alternate_dir.empty? && rows != 128
raise ArgumentError.new("F32-boundary mode requires a lower storage dtype") if mode == "f32-boundary" && type == 0
width = type == 0 ? 4_i64 : 2_i64
nq = rows*PRECISION_C
nkv = PRECISION_K*PRECISION_C
nc = PRECISION_CHUNK*PRECISION_C
typed_bytes = (2_i64*nq + 2_i64*nkv + 3_i64*nc)*width
kv_preparation_bytes = kv_layout == "head-major" ? 2_i64*nkv*width : 0_i64
reference_bytes = 2_i64*nq*4
boundary_bytes = mode == "f32-boundary" ? (2_i64*nq + 2_i64*nkv)*4 : 0_i64
# Four five-word metrics and three scalar comparator controls.
diagnostic_bytes = 23_i64*4
explicit_bytes = typed_bytes + kv_preparation_bytes + reference_bytes + boundary_bytes + diagnostic_bytes
raise ArgumentError.new("Precision buffer cap exceeded") if explicit_bytes > 4_i64*1024*1024*1024
buffer_plan = {schema: "wan_attention_precision_plan_v1", dtype: dtype, mode: mode,
               profile_host: profile_host,
               encoder_policy: encoder_policy,
               queue_capacity: queue_capacity,
               graph: graph, kv_layout: kv_layout, query_rows: rows,
               query_start: start, key_rows: PRECISION_K, heads: PRECISION_H, head_dim: PRECISION_D, chunk_rows: PRECISION_CHUNK,
               typed_attention_bytes: typed_bytes, kv_preparation_bytes: kv_preparation_bytes,
               reference_bytes: reference_bytes, boundary_bytes: boundary_bytes,
               kv_preparation_in_measured_arm: kv_layout == "head-major",
               diagnostic_bytes: diagnostic_bytes, explicit_buffer_bytes: explicit_bytes,
               opaque_mps_workspace_included: false, model_weights_loaded: false,
               scope: "selected-query full-key attention only; no complete block/model/video"}
if plan_only
  puts buffer_plan.to_json
  exit
end
raise ArgumentError.new("Both fixture paths are required") if fixture_dir.empty? || f32_fixture_dir.empty?
fixture = precision_fixture(fixture_dir, dtype, rows, start, 1_u32)
f32_fixture = precision_fixture(f32_fixture_dir, "f32", rows, start, 1_u32)
alternate = alternate_dir.empty? ? nil : precision_fixture(alternate_dir, dtype, rows, start, 4_u32)
alternate_f32 = alternate_f32_dir.empty? ? nil : precision_fixture(alternate_f32_dir, "f32", rows, start, 4_u32)

arena = ML::Video::Arena.new
plan = nil.as(PrecisionPlan?)
queue = nil.as(PrecisionQueue?)
passed = true
f32_passed = true
begin
  # Arena.alloc uses four-byte allocation units. These raw backings are sized
  # in bytes; the probe's explicit dtype controls every access and graph binding.
  # Every supported extent is divisible by four even with two-byte elements.
  q = arena.alloc("q", (nq.to_i64 * width // 4).to_i32)
  k = arena.alloc("k", (nkv.to_i64 * width // 4).to_i32)
  v = arena.alloc("v", (nkv.to_i64 * width // 4).to_i32)
  output = arena.alloc("out", (nq.to_i64 * width // 4).to_i32)
  cq = arena.alloc("chunk_q", (nc.to_i64 * width // 4).to_i32)
  ca = arena.alloc("chunk_a", (nc.to_i64 * width // 4).to_i32)
  cb = arena.alloc("chunk_b", (nc.to_i64 * width // 4).to_i32)
  expected = arena.alloc("expected", nq)
  f32_expected = arena.alloc("f32_expected", nq)
  metrics = 4.times.map { |i| arena.alloc("metric_#{i}", 5) }.to_a
  controls = 3.times.map { |i| arena.alloc("control_#{i}", 1) }.to_a
  controls[0].contents.as(Float32*)[0] = 0.0f32
  controls[1].contents.as(Float32*)[0] = 1.0f32
  controls[2].contents.as(Float32*)[0] = Float32::NAN
  sources = nil.as(Array(ML::MetalBuffer)?)
  converted_output = nil.as(ML::MetalBuffer?)
  prepared_kv = nil.as(Array(ML::MetalBuffer)?)
  if mode == "f32-boundary"
    sources = [arena.alloc("source_q", nq), arena.alloc("source_k", nkv), arena.alloc("source_v", nkv)]
    converted_output = arena.alloc("converted_output", nq)
  end
  if kv_layout == "head-major"
    prepared_kv = [arena.alloc("head_major_k", (nkv.to_i64*width//4).to_i32),
                   arena.alloc("head_major_v", (nkv.to_i64*width//4).to_i32)]
  end
  raise "Explicit allocation accounting mismatch" unless arena.resident_bytes == explicit_bytes
  pipelines = {} of String => ML::Metal::ComputePipeline
  {"fill", "nan", "tail", "convert", "zero", "compare"}.each do |name|
    pipelines[name] = ML::Metal::ComputePipeline.new("precision_#{name}", PRECISION_SOURCE)
  end
  transpose_pipeline = nil.as(ML::Metal::ComputePipeline?)
  if kv_layout == "head-major"
    transpose_pipeline = ML::Metal::ComputePipeline.new("precision_transpose_kv", PRECISION_SOURCE)
  end
  plan_k = prepared_kv ? prepared_kv.not_nil![0] : k
  plan_v = prepared_kv ? prepared_kv.not_nil![1] : v
  plan_buffers = [cq, plan_k, plan_v, ca, cb]
  precision_native_guards(plan_buffers, metrics[0], type, graph, kv_layout)
  plan = PrecisionPlan.new(plan_buffers, type, graph, kv_layout)
  queue = PrecisionQueue.new(queue_capacity)
  head_major_k = nil.as(ML::MetalBuffer?)
  head_major_v = nil.as(ML::MetalBuffer?)
  if buffers = prepared_kv
    head_major_k = buffers[0]
    head_major_v = buffers[1]
  end
  precision_load(fixture[:expected_path], expected)
  precision_load(f32_fixture[:expected_path], f32_expected)
  STDOUT.puts({schema: "wan_attention_precision_setup_v1", plan: buffer_plan, fixture: fixture,
               f32_fixture: f32_fixture, alternate_fixture: alternate, alternate_f32_fixture: alternate_f32,
               device_name: ML::Metal::Device.instance.name, samples: samples, process_id: Process.pid,
               accumulation_precision: "not specified by MPSGraph tensor dtypes or matmul/SDPA APIs",
               input_preparation: "GPU avalanche fill; F16 exact on this fixture, BF16 round-to-nearest-even",
               input_kv_layout: "BTHD source buffers; head-major mode transposes K and V on each measured chain",
               graph: graph, kv_layout: kv_layout,
               diagnostic_boundary: "separate chain after measured arm; 20 metric words only",
               production_defaults_changed: false}.to_json)
  STDOUT.flush
  setup = ML::Metal::MPSCommandChain.new
  begin
    precision_paused_guards(setup, plan.not_nil!, q, cq) if encoder_policy == "paused"
    if src = sources
      precision_fill(setup.encoder, pipelines["fill"], src[0], nq, (start.to_i64*PRECISION_C).to_u32, 1_u32, 0)
      precision_fill(setup.encoder, pipelines["fill"], src[1], nkv, 0_u32, 2_u32, 0)
      precision_fill(setup.encoder, pipelines["fill"], src[2], nkv, 0_u32, 3_u32, 0)
    else
      precision_fill(setup.encoder, pipelines["fill"], q, nq, (start.to_i64*PRECISION_C).to_u32, 1_u32, type)
      precision_fill(setup.encoder, pipelines["fill"], k, nkv, 0_u32, 2_u32, type)
      precision_fill(setup.encoder, pipelines["fill"], v, nkv, 0_u32, 3_u32, type)
    end
    precision_nan(setup.encoder, pipelines["nan"], output, nq, type)
    if widened = converted_output
      precision_nan(setup.encoder, pipelines["nan"], widened, nq, 0)
    end
  ensure
    precision_finish(setup)
  end
  walls = [] of Float64
  (samples + 1).times do |i|
    phase = i == 0 ? "warmup" : "measured"
    arm = precision_arm(plan.not_nil!, q, k, v, output, cq, ca, sources, converted_output,
      pipelines["convert"], pipelines["tail"], rows, type, mode, phase, i,
      pipelines["fill"], transpose_pipeline, head_major_k, head_major_v, start, profile_host, encoder_policy,
      queue.not_nil!, queue_capacity)
    STDOUT.puts(arm.to_json); STDOUT.flush
    walls << arm[:completed_wall_seconds] if i > 0
    actual = converted_output || output
    output_type = converted_output ? 0 : type
    ok = precision_validate(actual, expected, f32_expected, output_type, metrics, controls,
      pipelines["zero"], pipelines["compare"], rows, phase + "_" + i.to_s, dtype, mode,
      graph, kv_layout, output, converted_output, pipelines["nan"], type)
    passed = false unless ok[:matched_passed]
    f32_passed = false unless ok[:f32_passed]
  end
  if alt = alternate
    precision_load(alt[:expected_path], expected)
    precision_load(alternate_f32.not_nil![:expected_path], f32_expected)
    arm = precision_arm(plan.not_nil!, q, k, v, output, cq, ca, sources, converted_output,
      pipelines["convert"], pipelines["tail"], rows, type, mode, "changed_query", 0,
      pipelines["fill"], transpose_pipeline, head_major_k, head_major_v, start, profile_host, encoder_policy,
      queue.not_nil!, queue_capacity)
    STDOUT.puts(arm.to_json)
    ok = precision_validate(converted_output || output, expected, f32_expected,
      converted_output ? 0 : type, metrics, controls, pipelines["zero"], pipelines["compare"],
      rows, "changed_query", dtype, mode, graph, kv_layout,
      output, converted_output, pipelines["nan"], type)
    passed = false unless ok[:matched_passed]
    f32_passed = false unless ok[:f32_passed]
  end
  sorted = walls.sort
  puts({schema: "wan_attention_precision_summary_v1", dtype: dtype, mode: mode,
        graph: graph, kv_layout: kv_layout, query_rows: rows,
        query_start: start, measured_wall_seconds: walls, median_wall_seconds: sorted.empty? ? nil : sorted[sorted.size//2],
        all_dtype_matched_checks_passed: passed, all_strict_f32_checks_passed: f32_passed,
        admitted_at_unchanged_precision: passed && f32_passed,
        explicit_buffer_bytes: arena.resident_bytes,
        device_current_allocated_bytes: ML::Metal::Device.instance.current_allocated_size,
        full_kv: true, model_weights_loaded: false}.to_json)
ensure
  begin
    plan.try(&.release)
  ensure
    queue.try(&.release)
    arena.release
  end
end
exit(passed ? (f32_passed ? 0 : 3) : 2)
