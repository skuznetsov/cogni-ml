# Full-shape preparation-only diagnostic. Production kernels and source-dtype
# bias are reused; no projections, attention or model weights are loaded.
require "json"
require "option_parser"
require "../src/ml/video/wan_s2v_block"
require "../spec/support/video_allocation_counter"
require "./support/wan_host_observation"

private KV_SEQ       = 36_608
private KV_DIM       =  5_120
private KV_HEADS     =     40
private KV_VALUES    = KV_SEQ * KV_DIM
private KV_CAP_BYTES = 6_i64 * 1024 * 1024 * 1024

private KV_BENCH_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void wan_kv_bench_fill(device float *k [[buffer(0)]],
      device float *v [[buffer(1)]], constant uint &count [[buffer(2)]],
      constant uint &revision [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint n = i * 747796405u + 2891336453u + revision * 37u;
    n = ((n >> ((n >> 28u) + 4u)) ^ n) * 277803737u;
    n = (n >> 22u) ^ n;
    k[i] = float(int(n & 2047u) - 1024) / 1024.0f;
    v[i] = float(int((n >> 11u) & 2047u) - 1024) / 2048.0f;
  }
  kernel void wan_kv_bench_positions(device float *p [[buffer(0)]],
      constant uint &count [[buffer(1)]], constant uint &revision [[buffer(2)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint token = i / 3u, axis = i % 3u;
    p[i] = axis == 0u ? float(int(token % 4099u) - 2049) * 0.5f :
          (axis == 1u ? float(token / 52u) * 0.25f : float(token % 52u));
    p[i] += float(revision) * 0.0625f;
  }
  kernel void wan_kv_bench_bias(device float *f [[buffer(0)]],
      device ushort *b [[buffer(1)]], constant uint &bf16 [[buffer(2)]],
      constant uint &dim [[buffer(3)]], constant uint &revision [[buffer(4)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= dim) return;
    const float x = float(int((i * 7u + revision * 3u) % 37u) - 18) * 0.0007f;
    if (bf16) b[i] = ushort(as_type<uint>(x) >> 16u);
    else f[i] = x;
  }
  kernel void wan_kv_bench_zero(device atomic_uint *m [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i < 5u) atomic_store_explicit(m + i, 0u, memory_order_relaxed);
  }
  kernel void wan_kv_bench_compare(device const float *a [[buffer(0)]],
      device const float *b [[buffer(1)]], device atomic_uint *m [[buffer(2)]],
      constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float av = a[i], bv = b[i];
    if (as_type<uint>(av) != as_type<uint>(bv))
      atomic_fetch_add_explicit(m, 1u, memory_order_relaxed);
    if (!isfinite(av) || !isfinite(bv)) {
      atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(m + 2, 1u, memory_order_relaxed);
    } else {
      const float error = abs(av - bv);
      const float ratio = error / (2.0e-5f + 2.0e-5f * abs(av));
      if (ratio > 1.0f) atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
      atomic_fetch_max_explicit(m + 3, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(m + 4, as_type<uint>(ratio), memory_order_relaxed);
    }
  }
  kernel void wan_kv_bench_corrupt(device float *x [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i == 0u) { x[0] += 1.0f; x[1] = as_type<float>(0x7fc00001u); }
  }
METAL

# Exact bindings of WanS2VBlock's private K-only RoPE enqueue. Obtain its
# already-prepared production pipeline instead of compiling substitute math or
# charging the baseline for an unrelated Q rotation.
private def encode_separate_kv(chain : ML::Metal::MPSCommandChain,
                               k : ML::MetalBuffer, v : ML::MetalBuffer,
                               kr : ML::MetalBuffer, bias : ML::MetalBuffer,
                               dtype : ML::Video::WeightDType,
                               positions : ML::MetalBuffer,
                               hi : ML::MetalBuffer, lo : ML::MetalBuffer,
                               pk : ML::MetalBuffer, pv : ML::MetalBuffer,
                               cfg : ML::Video::WanS2VConfig,
                               rope : ML::Metal::ComputePipeline) : Nil
  e = chain.encoder
  # Preserve the production relative order of these independent operations:
  # V projection bias precedes K RoPE, followed by head-major packing.
  e.set_pipeline(ML::Video::MetalLinearMPS.pipeline)
    .set_buffer(v, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(bias, 1).set_buffer(bias, 2)
    .set_value(cfg.dim.to_u32, 3).set_value(dtype.bf16? ? 1_u32 : 0_u32, 4)
    .set_value(cfg.values.to_u32, 5).dispatch_1d(cfg.values)
  e.set_pipeline(rope).set_buffer(k, 0).set_buffer(positions, 1)
    .set_buffer(kr, 2, ML::Metal::BufferAccess::Write)
    .set_buffer(hi, 3).set_buffer(lo, 4)
    .set_value(cfg.batches.to_u32, 5).set_value(cfg.seq.to_u32, 6)
    .set_value(cfg.heads.to_u32, 7).set_value(cfg.head_dim.to_u32, 8)
    .dispatch_1d(cfg.values // 2)
  ML::Video::MetalAttentionMPS.transpose_kv!(chain, kr, v, pk, pv,
    cfg.batches, cfg.seq, cfg.heads, cfg.head_dim)
end

pairs = 4
dtype = ML::Video::WeightDType::BF16
warmup_order = "AB"
OptionParser.parse do |parser|
  parser.on("--pairs=N", "Four balanced AB/BA/BA/AB measured pairs") { |value| pairs = value.to_i }
  parser.on("--bias-dtype=TYPE", "Source bias bf16 (default) or f32") do |value|
    dtype = case value
            when "bf16" then ML::Video::WeightDType::BF16
            when "f32"  then ML::Video::WeightDType::F32
            else             raise ArgumentError.new("bias dtype must be bf16 or f32")
            end
  end
  parser.on("--warmup-order=ORDER", "AB (default) or BA") { |value| warmup_order = value }
end
raise ArgumentError.new("preparation probe requires exactly four pairs") unless pairs == 4
raise ArgumentError.new("warmup order must be AB or BA") unless {"AB", "BA"}.includes?(warmup_order)

cfg = ML::Video::WanS2VConfig.new(1, KV_SEQ, 512, KV_DIM, KV_HEADS,
  13_824, 34_944, 10_000.0_f32, 21, 1_664)
estimate = (8_i64 * KV_VALUES + 3_i64 * KV_SEQ + cfg.head_dim + 5_i64) * sizeof(Float32) +
           KV_DIM.to_i64 * (dtype.bf16? ? 2 : 4)
raise "K/V preparation explicit memory cap exceeded" if estimate > KV_CAP_BYTES
puts({schema: "wan_kv_prep_memory_plan_v1", explicit_estimate_bytes: estimate,
      explicit_cap_bytes: KV_CAP_BYTES, source_bias_dtype: dtype.to_s,
      model_weights_loaded: false, opaque_runtime_reserve_not_in_explicit_estimate: true,
      scope: "isolated full-shape synthetic K RoPE, V bias and K/V packing; not full-block latency"}.to_json)
STDOUT.flush

arena = ML::Video::Arena.new
scratch = ML::Video::Arena.new
begin
  ML::Video::WanS2VBlock.prepare!
  ML::Video::MetalLinearMPS.prepare!
  ML::Video::MetalAttentionMPS.prepare_head_major_kv!
  ML::Video::WanS2VBlock.prepare_rope_scratch!(scratch, cfg)
  hi = scratch[scratch.buffers.keys.find { |name| name.includes?("rope_freq_hi") }.not_nil!]
  lo = scratch[scratch.buffers.keys.find { |name| name.includes?("rope_freq_lo") }.not_nil!]
  rope = ML::Metal::PipelineCache.get("wan_rope_f32") { raise "production RoPE pipeline not prepared" }
  k = arena.alloc("k_source", KV_VALUES)
  v = arena.alloc("v_source", KV_VALUES)
  vr = arena.alloc("v_baseline", KV_VALUES)
  kr = arena.alloc("k_rotated", KV_VALUES)
  ak = arena.alloc("separate_k", KV_VALUES)
  av = arena.alloc("separate_v", KV_VALUES)
  bk = arena.alloc("fused_k", KV_VALUES)
  bv = arena.alloc("fused_v", KV_VALUES)
  positions = arena.alloc("positions", KV_SEQ * 3)
  bias = arena.alloc_weight("bias", KV_DIM, dtype)
  metrics = arena.alloc("metrics", 5)
  raise "K/V explicit allocation accounting mismatch" unless arena.resident_bytes + scratch.resident_bytes == estimate
  fill = ML::Metal::ComputePipeline.new("wan_kv_bench_fill", KV_BENCH_SOURCE)
  fill_positions = ML::Metal::ComputePipeline.new("wan_kv_bench_positions", KV_BENCH_SOURCE)
  fill_bias = ML::Metal::ComputePipeline.new("wan_kv_bench_bias", KV_BENCH_SOURCE)
  zero = ML::Metal::ComputePipeline.new("wan_kv_bench_zero", KV_BENCH_SOURCE)
  compare = ML::Metal::ComputePipeline.new("wan_kv_bench_compare", KV_BENCH_SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("wan_kv_bench_corrupt", KV_BENCH_SOURCE)

  (-1...pairs).each do |pair|
    # Every input, including the baseline's in-place biased V copy, is refreshed
    # on GPU before each pair and outside timing. Both arms consume equal data.
    setup = ML::Metal::MPSCommandChain.new
    begin
      e = setup.encoder
      e.set_pipeline(fill).set_buffer(k, 0, ML::Metal::BufferAccess::Write)
        .set_buffer(v, 1, ML::Metal::BufferAccess::Write).set_value(KV_VALUES.to_u32, 2)
        .set_value((pair + 2).to_u32, 3).dispatch_1d(KV_VALUES)
      ML::Video::MetalOps.copy(e, v, vr, KV_VALUES)
      e.set_pipeline(fill_positions).set_buffer(positions, 0, ML::Metal::BufferAccess::Write)
        .set_value((KV_SEQ * 3).to_u32, 1).set_value((pair + 2).to_u32, 2)
        .dispatch_1d(KV_SEQ * 3)
      e.set_pipeline(fill_bias).set_buffer(bias, 0, ML::Metal::BufferAccess::Write)
        .set_buffer(bias, 1, ML::Metal::BufferAccess::Write)
        .set_value(dtype.bf16? ? 1_u32 : 0_u32, 2).set_value(KV_DIM.to_u32, 3)
        .set_value((pair + 2).to_u32, 4).dispatch_1d(KV_DIM)
      setup.commit_and_wait
    ensure
      setup.commit_and_wait unless setup.completed?
      setup.release if setup.completed?
    end

    order = pair < 0 ? warmup_order : ({0, 3}.includes?(pair) ? "AB" : "BA")
    order.each_char.with_index do |arm, slot|
      observation_before = WanHostObservation::Snapshot.capture
      allocations = ML::MetalBuffer.video_spec_allocations
      started = Time.instant
      chain = ML::Metal::MPSCommandChain.new
      begin
        encode_started = Time.instant
        if arm == 'A'
          encode_separate_kv(chain, k, vr, kr, bias, dtype, positions, hi, lo, ak, av, cfg, rope)
        else
          ML::Video::WanS2VBlock.encode_kv_rope_bias_pack(chain.encoder, k, v,
            bias, dtype, positions, hi, lo, bk, bv, cfg)
        end
        encoded_seconds = (Time.instant - encode_started).total_seconds
        wait_started = Time.instant
        chain.commit_and_wait
        wall_seconds = (Time.instant - started).total_seconds
        wait_seconds = (Time.instant - wait_started).total_seconds
        observation_after = WanHostObservation::Snapshot.capture
        allocated = ML::MetalBuffer.video_spec_allocations - allocations
        raise "preparation arm allocated MetalBuffers" unless allocated == 0
        raise "preparation arm changed terminal boundaries" unless chain.wait_count == 1 && chain.segment_count == 1
        puts({schema: "wan_kv_prep_arm_v1", pair: pair, warmup: pair < 0,
              order: order, slot: slot, arm: arm.to_s, fused: arm == 'B',
              seq: cfg.seq, dim: cfg.dim, heads: cfg.heads, head_dim: cfg.head_dim,
              source_bias_dtype: dtype.to_s, wall_seconds: wall_seconds,
              encode_host_elapsed_seconds: encoded_seconds, terminal_wait_elapsed_seconds: wait_seconds,
              gpu_root_duration_sum_seconds: chain.gpu_elapsed_seconds,
              native_dispatches: arm == 'A' ? 3 : 1, segments: chain.segment_count,
              terminal_waits: chain.wait_count, inter_operator_host_waits: 0,
              activation_readbacks: 0, encode_metal_buffer_allocations: allocated,
              refreshed_inputs_outside_timing: true, phase_profiler: false,
              observations: WanHostObservation.between(observation_before, observation_after)}.to_json)
        STDOUT.flush
      ensure
        chain.commit_and_wait unless chain.completed?
        chain.release if chain.completed?
      end
    end

    validation = ML::Metal::MPSCommandChain.new
    begin
      e = validation.encoder
      e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
      { {ak, bk}, {av, bv} }.each do |reference, actual|
        e.set_pipeline(compare).set_buffer(reference, 0).set_buffer(actual, 1)
          .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
          .set_value(KV_VALUES.to_u32, 3).dispatch_1d(KV_VALUES)
      end
      validation.commit_and_wait
      m = metrics.contents.as(UInt32*)
      puts({schema: "wan_kv_prep_comparison_v1", pair: pair, warmup: pair < 0,
            compared_values: 2_i64 * KV_VALUES, bit_differences: m[0],
            tolerance_violations: m[1], nonfinite: m[2],
            max_abs: m[3].unsafe_as(Float32), max_ratio: m[4].unsafe_as(Float32),
            comparison_outside_timed_arms: true}.to_json)
      STDOUT.flush
      raise "preparation comparison failed" unless m[0] == 0 && m[1] == 0 && m[2] == 0
    ensure
      validation.commit_and_wait unless validation.completed?
      validation.release if validation.completed?
    end
  end

  control = ML::Metal::MPSCommandChain.new
  begin
    e = control.encoder
    e.set_pipeline(corrupt).set_buffer(bk, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
    e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
    e.set_pipeline(compare).set_buffer(ak, 0).set_buffer(bk, 1)
      .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
      .set_value(KV_VALUES.to_u32, 3).dispatch_1d(KV_VALUES)
    control.commit_and_wait
    m = metrics.contents.as(UInt32*)
    rejected = m[0] == 2 && m[1] == 2 && m[2] == 1
    puts({schema: "wan_kv_prep_negative_control_v1", bit_differences: m[0],
          tolerance_violations: m[1], nonfinite: m[2], rejected: rejected,
          outside_timed_arms: true}.to_json)
    raise "preparation corruption control was not rejected" unless rejected
  ensure
    control.commit_and_wait unless control.completed?
    control.release if control.completed?
  end
ensure
  scratch.release
  arena.release
end
