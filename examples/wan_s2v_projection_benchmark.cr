# Standalone cost diagnostic. Synthetic dyadic data, production matrix shapes.
# This does not qualify upstream parity or complete video latency.
require "json"
require "option_parser"
require "../src/ml/video/wan_t5"
require "../spec/support/video_allocation_counter"

operator = "hidden"
pairs = 3
OptionParser.parse do |parser|
  parser.on("--operator=NAME", "hidden, ffn-in, ffn-out, cross-kv") { |value| operator = value }
  parser.on("--pairs=N", "AB/BA measured pairs") { |value| pairs = value.to_i }
end
raise ArgumentError.new("pairs must be 2..8") unless 2 <= pairs <= 8
rows, din, dout = case operator
                  when "hidden"   then {36_608, 5_120, 5_120}
                  when "ffn-in"   then {36_608, 5_120, 13_824}
                  when "ffn-out"  then {36_608, 13_824, 5_120}
                  when "cross-kv" then {512, 5_120, 5_120}
                  else                 raise ArgumentError.new("Unknown operator")
                  end
source = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void wan_projection_bench_fill(
      device float *x [[buffer(0)]], device ushort *w [[buffer(1)]],
      constant uint &count [[buffer(2)]], constant uint &kind [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float value = float(int((i * 17u + i / 127u) % 31u) - 15) / 1024.0f;
    if (kind == 0u) x[i] = value;
    else w[i] = ushort(as_type<uint>(value) >> 16);
  }
  kernel void wan_projection_bench_zero(device atomic_uint *m [[buffer(0)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i < 5u) atomic_store_explicit(m + i, 0u, memory_order_relaxed);
  }
  kernel void wan_projection_bench_corrupt(device float *x [[buffer(0)]],
                                           uint i [[thread_position_in_grid]]) {
    if (i == 0u) x[0] += 1.0f;
  }
  kernel void wan_projection_bench_compare(
      device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
      device atomic_uint *m [[buffer(2)]], constant uint &count [[buffer(3)]],
      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float av = a[i], bv = b[i];
    if (as_type<uint>(av) != as_type<uint>(bv))
      atomic_fetch_add_explicit(m, 1u, memory_order_relaxed);
    if (!isfinite(av) || !isfinite(bv)) {
      atomic_fetch_add_explicit(m + 2, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
    } else {
      float error = abs(av - bv);
      float ratio = error / (2.0e-5f + 2.0e-5f * abs(bv));
      if (ratio > 1.0f) atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
      atomic_fetch_max_explicit(m + 3, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(m + 4, as_type<uint>(ratio), memory_order_relaxed);
    }
  }
METAL

arena = ML::Video::Arena.new
baseline_plan : ML::Video::WanT5MPSLinearPlan? = nil
candidate_plan : ML::Video::WanT5MPSLinearPlan? = nil
begin
  ML::Video::WanT5.prepare!
  fill = ML::Metal::ComputePipeline.new("wan_projection_bench_fill", source)
  zero = ML::Metal::ComputePipeline.new("wan_projection_bench_zero", source)
  compare = ML::Metal::ComputePipeline.new("wan_projection_bench_compare", source)
  corrupt = ML::Metal::ComputePipeline.new("wan_projection_bench_corrupt", source)
  input = arena.alloc("input", rows * din)
  weights = arena.alloc_weight("bf16", din * dout, ML::Video::WeightDType::BF16)
  stage = arena.alloc("stage", din * dout)
  baseline = arena.alloc("baseline", rows * dout)
  candidate = arena.alloc("candidate", rows * dout)
  metrics = arena.alloc("metrics", 5)
  raise "explicit probe buffer budget exceeded" if arena.resident_bytes > 6_000_000_000_i64
  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
    layers: 1, batches: 1, tokens: rows)
  baseline_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, din, dout, input, stage, baseline)
  candidate_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, din, dout, input, stage, candidate)

  # Setup is outside measured arms. There is no wait inside either projection.
  setup = ML::Metal::CommandBuffer.new
  encoder = ML::Metal::ComputeEncoder.new(setup)
  encoder.set_pipeline(fill).set_buffer(input, 0, ML::Metal::BufferAccess::Write)
    .set_buffer(weights, 1).set_value((rows * din).to_u32, 2).set_value(0_u32, 3)
    .dispatch_1d(rows * din)
  encoder.set_pipeline(fill).set_buffer(input, 0)
    .set_buffer(weights, 1, ML::Metal::BufferAccess::Write)
    .set_value((din * dout).to_u32, 2).set_value(1_u32, 3).dispatch_1d(din * dout)
  encoder.end_encoding
  setup.commit
  setup.wait

  # Each baseline expands exactly the same BF16 weights. The next resident arm
  # consumes that F32 stage directly; it measures an upper bound for caching.
  # Before BA, the stage still contains these immutable weights from warmup/AB.
  (-1...pairs).each do |pair|
    order = pair < 0 || pair.even? ? {"expand", "resident"} : {"resident", "expand"}
    order.each do |variant|
      command = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(command)
      allocations = ML::MetalBuffer.video_spec_allocations
      started = Time.instant
      if variant == "expand"
        ML::Video::WanT5.mps_projection(e, input, weights, baseline, rows, din, dout,
          ML::Video::WeightDType::BF16, baseline_plan.not_nil!, stage)
      else
        e.encode_wan_t5_mps_fc2(candidate_plan.not_nil!.handle)
      end
      e.end_encoding
      encode_seconds = (Time.instant - started).total_seconds
      gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
      wall_seconds = (Time.instant - started).total_seconds
      raise "projection encode allocated MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == allocations
      puts({schema: "wan_projection_benchmark_v1", operator: operator, pair: pair,
            warmup: pair < 0, variant: variant, rows: rows, din: din, dout: dout,
            encode_seconds: encode_seconds, wall_seconds: wall_seconds,
            gpu_seconds: gpu_seconds, terminal_waits: 1, inter_operator_host_waits: 0,
            activation_readbacks: 0, explicit_buffer_bytes: arena.resident_bytes,
            encode_metal_buffer_allocations: 0,
            scope: "synthetic full-shape projection; resident is a cache upper-bound diagnostic"}.to_json)
    end
    validation = ML::Metal::CommandBuffer.new
    e = ML::Metal::ComputeEncoder.new(validation)
    e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
    e.set_pipeline(compare).set_buffer(candidate, 0).set_buffer(baseline, 1)
      .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
      .set_value((rows * dout).to_u32, 3).dispatch_1d(rows * dout)
    e.end_encoding
    validation.commit
    validation.wait
    values = metrics.contents.as(UInt32*)
    puts({schema: "wan_projection_comparison_v1", operator: operator, pair: pair,
          compared_values: rows.to_i64 * dout, bit_differences: values[0],
          tolerance_violations: values[1], nonfinite: values[2],
          max_abs: values[3].unsafe_as(Float32), max_ratio: values[4].unsafe_as(Float32),
          tolerance: "2e-5+2e-5*abs(reference)", validation_outside_timed_arm: true}.to_json)
    raise "resident projection differs from baseline" unless values[0] == 0 && values[1] == 0 && values[2] == 0
  end
  control = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(control)
  e.set_pipeline(corrupt).set_buffer(candidate, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  e.set_pipeline(compare).set_buffer(candidate, 0).set_buffer(baseline, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value((rows * dout).to_u32, 3).dispatch_1d(rows * dout)
  e.end_encoding
  control.commit
  control.wait
  values = metrics.contents.as(UInt32*)
  raise "GPU comparator failed corruption control" unless values[0] == 1 && values[1] == 1 && values[2] == 0
  puts({schema: "wan_projection_comparator_control_v1", operator: operator,
        injected_corruptions: 1, bit_differences: values[0], tolerance_violations: values[1],
        nonfinite: values[2], rejected: true, validation_outside_timed_arm: true}.to_json)
ensure
  baseline_plan.try(&.release)
  candidate_plan.try(&.release)
  arena.release
end
