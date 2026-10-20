# Bounded, full-shape RoPE pair diagnostic; no model weights are loaded.
# Coefficient construction is charged to every precomputed arm. Input fill and
# GPU comparison have separate terminal boundaries outside the timed arms.
require "json"
require "option_parser"
require "../src/ml/video/wan_s2v_block"
require "../spec/support/video_allocation_counter"

private SEQ                = 36_608
private DIM                =  5_120
private HEADS              =     40
private VALUES             = SEQ * DIM
private MAX_EXPLICIT_BYTES = 3_i64 * 1024 * 1024 * 1024

private SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void wan_rope_bench_fill(
      device float *q [[buffer(0)]], device float *k [[buffer(1)]],
      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint n = i * 747796405u + 2891336453u;
    n = ((n >> ((n >> 28u) + 4u)) ^ n) * 277803737u;
    n = (n >> 22u) ^ n;
    q[i] = float(int(n & 2047u) - 1024) / 1024.0f;
    k[i] = float(int((n >> 11u) & 2047u) - 1024) / 2048.0f;
  }
  kernel void wan_rope_bench_positions(
      device float *p [[buffer(0)]], constant uint &count [[buffer(1)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint token = i / 3u, axis = i % 3u;
    // Includes negative reference offsets and fractional/high coordinates.
    p[i] = axis == 0u ? float(int(token % 4099u) - 2049) * 0.5f :
          (axis == 1u ? float(token / 52u) * 0.25f : float(token % 52u));
  }
  kernel void wan_rope_bench_zero(device atomic_uint *m [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i < 5u) atomic_store_explicit(m + i, 0u, memory_order_relaxed);
  }
  kernel void wan_rope_bench_compare(
      device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
      device atomic_uint *m [[buffer(2)]], constant uint &count [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
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
  kernel void wan_rope_bench_corrupt(device float *x [[buffer(0)]],
      uint i [[thread_position_in_grid]]) {
    if (i == 0u) x[0] += 1.0f;
  }
METAL

pairs = 3
OptionParser.parse do |parser|
  parser.on("--pairs=N", "Exactly three AB/BA/AB pairs") { |value| pairs = value.to_i }
end
raise ArgumentError.new("RoPE comparison requires exactly three pairs") unless pairs == 3

cfg = ML::Video::WanS2VConfig.new(1, SEQ, 1, DIM, HEADS, 1,
  1, 10_000.0_f32, 1, 1)
# Four independent output backings; corresponding input/output are in-place.
# Count table, positions, constants and metrics before creating any Metal buffer.
estimate = (4_i64 * VALUES + 3_i64 * SEQ + SEQ.to_i64 * cfg.head_dim +
            cfg.head_dim + 5_i64) * sizeof(Float32)
raise "RoPE explicit allocation preflight failed" if estimate > MAX_EXPLICIT_BYTES
puts({schema: "wan_rope_memory_plan_v1", explicit_estimate_bytes: estimate,
      explicit_cap_bytes: MAX_EXPLICIT_BYTES, model_weights_loaded: false,
      opaque_runtime_reserve_not_in_explicit_estimate: true}.to_json)
STDOUT.flush

arena = ML::Video::Arena.new
scratch = ML::Video::Arena.new
begin
  ML::Video::WanS2VBlock.prepare!(precompute_rope: true)
  ML::Video::WanS2VBlock.prepare_rope_scratch!(scratch, cfg, precompute_rope: true)
  aq = arena.alloc("inline_q", VALUES)
  ak = arena.alloc("inline_k", VALUES)
  bq = arena.alloc("table_q", VALUES)
  bk = arena.alloc("table_k", VALUES)
  positions = arena.alloc("positions", SEQ * 3)
  metrics = arena.alloc("metrics", 5)
  raise "RoPE allocation accounting mismatch" unless arena.resident_bytes + scratch.resident_bytes == estimate
  fill = ML::Metal::ComputePipeline.new("wan_rope_bench_fill", SOURCE)
  fill_positions = ML::Metal::ComputePipeline.new("wan_rope_bench_positions", SOURCE)
  zero = ML::Metal::ComputePipeline.new("wan_rope_bench_zero", SOURCE)
  compare = ML::Metal::ComputePipeline.new("wan_rope_bench_compare", SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("wan_rope_bench_corrupt", SOURCE)

  (-1...pairs).each do |pair|
    # Identical inputs are regenerated on GPU before each pair. This is outside
    # both timed arms, including BA. Each arm refreshes its table inside timing.
    setup = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(setup)
    { {aq, ak}, {bq, bk} }.each do |q, k|
      e.set_pipeline(fill).set_buffer(q, 0, ML::Metal::BufferAccess::Write)
        .set_buffer(k, 1, ML::Metal::BufferAccess::Write)
        .set_value(VALUES.to_u32, 2).dispatch_1d(VALUES)
    end
    e.set_pipeline(fill_positions).set_buffer(positions, 0, ML::Metal::BufferAccess::Write)
      .set_value((SEQ * 3).to_u32, 1).dispatch_1d(SEQ * 3)
    e.end_encoding
    setup.commit_and_wait

    order = pair < 0 || pair.even? ? {false, true} : {true, false}
    order.each_with_index do |precompute, index|
      q, k = precompute ? {bq, bk} : {aq, ak}
      command = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(command)
      allocations = ML::MetalBuffer.video_spec_allocations
      started = Time.instant
      ML::Video::WanS2VBlock.encode_rope_pair(e, q, k, positions, q, k, cfg, scratch,
        precompute_rope: precompute)
      e.end_encoding
      encoded_seconds = (Time.instant - started).total_seconds
      gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
      wall_seconds = (Time.instant - started).total_seconds
      raise "RoPE encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocations
      puts({schema: "wan_rope_arm_v1", pair: pair, warmup: pair < 0,
            order: index, precompute_rope: precompute,
            seq: SEQ, heads: HEADS, head_dim: cfg.head_dim,
            coefficient_build_inside_timing: precompute,
            wall_seconds: wall_seconds, encode_wall_seconds: encoded_seconds,
            gpu_seconds: gpu_seconds, terminal_waits: 1,
            inter_operator_host_waits: 0, activation_readbacks: 0,
            encode_metal_buffer_allocations: 0,
            scope: "full-shape synthetic RoPE pair; not full-block or video latency"}.to_json)
      STDOUT.flush
    end

    validation = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(validation)
    e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
    { {aq, bq}, {ak, bk} }.each do |reference, actual|
      e.set_pipeline(compare).set_buffer(reference, 0).set_buffer(actual, 1)
        .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_value(VALUES.to_u32, 3).dispatch_1d(VALUES)
    end
    e.end_encoding
    validation.commit_and_wait
    result = metrics.contents.as(UInt32*)
    puts({schema: "wan_rope_comparison_v1", pair: pair, warmup: pair < 0,
          compared_values: 2_i64 * VALUES, bit_differences: result[0],
          tolerance_violations: result[1], nonfinite: result[2],
          max_abs: result[3].unsafe_as(Float32), max_ratio: result[4].unsafe_as(Float32),
          comparison_outside_timed_arms: true}.to_json)
    raise "RoPE strict bit comparison failed" unless result[0] == 0 && result[1] == 0 && result[2] == 0
    STDOUT.flush
  end

  # Negative control starts from an exact copy; only one deliberately changed
  # value may be rejected. It is never included in the measured intervals.
  control = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(control)
  ML::Video::MetalOps.copy(e, aq, bq, VALUES)
  e.set_pipeline(corrupt).set_buffer(bq, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  e.set_pipeline(compare).set_buffer(aq, 0).set_buffer(bq, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(VALUES.to_u32, 3).dispatch_1d(VALUES)
  e.end_encoding
  control.commit_and_wait
  result = metrics.contents.as(UInt32*)
  rejected = result[0] == 1 && result[1] == 1 && result[2] == 0
  puts({schema: "wan_rope_corruption_control_v1", rejected: rejected,
        bit_differences: result[0], tolerance_violations: result[1], nonfinite: result[2]}.to_json)
  raise "RoPE comparator did not reject corruption" unless rejected
ensure
  scratch.release
  arena.release
end
