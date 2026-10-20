# Bounded FFN bias+GELU A/B probe. Default mode uses synthetic post-MPS inputs;
# the optional upstream mode loads only one projection's weights and fixed inputs.
# Host-memory admission belongs to the outer cache-aware guard.
require "json"
require "option_parser"
require "../src/ml/video/metal_linear_mps"
require "../spec/support/video_allocation_counter"
require "./support/wan_ffn_epilogue_upstream"

private DEFAULT_ROWS         = 4_096
private DEFAULT_WIDTH        = 13_824
private PAIRS                = 3
private MAX_EXPLICIT_BYTES   = 1_i64 * 1024 * 1024 * 1024

private SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline float wan_ffn_bias_value(device const float *bias32,
                                  device const ushort *bias16,
                                  uint column, uint bias_is_bf16) {
    return bias_is_bf16 != 0u
      ? as_type<float>(uint(bias16[column]) << 16u)
      : bias32[column];
  }

  // Keep the source order and |v| >= 10 branch identical to video_gelu_f32.
  // The bridge compiles device pipelines with fast math disabled.
  inline float wan_ffn_gelu_approx(float v) {
    if (abs(v) >= 10.0f)
      return v > 0.0f ? v : 0.0f;
    return 0.5f * v * (1.0f + tanh(0.7978845608028654f *
      (v + 0.044715f * v * v * v)));
  }

  // Both A/B arms receive exactly the same GPU-generated pre-bias values.
  // Rows deliberately cover zero, signed zero, +/-10 and adjacent F32 values,
  // cancellation/rounding, and large finite magnitudes.
  kernel void wan_ffn_epilogue_fill(
      device float *baseline [[buffer(0)]], device float *candidate [[buffer(1)]],
      device const float *bias32 [[buffer(2)]], device const ushort *bias16 [[buffer(3)]],
      constant uint &width [[buffer(4)]], constant uint &bias_is_bf16 [[buffer(5)]],
      constant uint &count [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const uint row = index / width, column = index % width;
    const float bias = wan_ffn_bias_value(bias32, bias16, column, bias_is_bf16);
    float input = 0.0f;
    if (row < 16u) {
      float target = 0.0f;
      switch (row) {
        case 0u:  target = 0.0f; break;
        case 1u:  target = -0.0f; break;
        case 2u:  target = 10.0f; break;
        case 3u:  target = as_type<float>(0x411fffffu); break;
        case 4u:  target = as_type<float>(0x41200001u); break;
        case 5u:  target = -10.0f; break;
        case 6u:  target = as_type<float>(0xc11fffffu); break;
        case 7u:  target = as_type<float>(0xc1200001u); break;
        case 8u:  target = 1.0e-7f; break;
        case 9u:  target = -1.0e-7f; break;
        case 10u: target = 16384.0f; break;
        case 11u: target = -16384.0f; break;
        case 12u: target = 1.0e20f; break;
        case 13u: target = -1.0e20f; break;
        case 14u: target = as_type<float>(0x7f7fffffu); break;
        default:  target = as_type<float>(0xff7fffffu); break;
      }
      input = target - bias;
    } else {
      // Hashed, deterministic, mostly unsaturated FFN-like activations.
      uint n = index * 747796405u + 2891336453u;
      n = ((n >> ((n >> 28u) + 4u)) ^ n) * 277803737u;
      n = (n >> 22u) ^ n;
      input = float(int(n & 0x00ffffffu) - 0x00800000) / 2097152.0f;
    }
    baseline[index] = input;
    candidate[index] = input;
  }

  // Candidate: one in-place dispatch. `biased` is an F32 value, matching the
  // baseline bias kernel's F32 store before its separate GELU dispatch.
  kernel void wan_ffn_epilogue_fused(
      device float *output [[buffer(0)]],
      device const float *bias32 [[buffer(1)]], device const ushort *bias16 [[buffer(2)]],
      constant uint &width [[buffer(3)]], constant uint &bias_is_bf16 [[buffer(4)]],
      constant uint &count [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const uint column = index % width;
    const float biased = output[index] +
      wan_ffn_bias_value(bias32, bias16, column, bias_is_bf16);
    output[index] = wan_ffn_gelu_approx(biased);
  }

  // Optional trace variant writes the exact pre-GELU F32 bias sum to a caller-
  // owned buffer while producing the normal fused output.
  kernel void wan_ffn_epilogue_fused_trace(
      device float *output [[buffer(0)]], device float *trace [[buffer(1)]],
      device const float *bias32 [[buffer(2)]], device const ushort *bias16 [[buffer(3)]],
      constant uint &width [[buffer(4)]], constant uint &bias_is_bf16 [[buffer(5)]],
      constant uint &count [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const uint column = index % width;
    const float biased = output[index] +
      wan_ffn_bias_value(bias32, bias16, column, bias_is_bf16);
    trace[index] = biased;
    output[index] = wan_ffn_gelu_approx(biased);
  }

  kernel void wan_ffn_epilogue_zero(device atomic_uint *metrics [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index < 5u)
      atomic_store_explicit(metrics + index, 0u, memory_order_relaxed);
  }

  // Metrics: bit differences, strict-tolerance violations, non-finite values,
  // maximum absolute error bits, and maximum normalized error bits. Equal
  // finite values perform no atomics; equal-bit NaNs still fail closed.
  kernel void wan_ffn_epilogue_compare(
      device const float *reference [[buffer(0)]], device const float *actual [[buffer(1)]],
      device atomic_uint *metrics [[buffer(2)]], constant uint &count [[buffer(3)]],
      uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float ref = reference[index], got = actual[index];
    const uint ref_bits = as_type<uint>(ref), got_bits = as_type<uint>(got);
    const bool same_bits = ref_bits == got_bits;
    if (!isfinite(ref) || !isfinite(got)) {
      if (!same_bits)
        atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
      atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
      return;
    }
    if (same_bits) return;

    atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
    const float error = abs(ref - got);
    const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
    const float ratio = error / tolerance;
    if (error > tolerance)
      atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
    if (error > 0.0f) {
      atomic_fetch_max_explicit(metrics + 3, as_type<uint>(error), memory_order_relaxed);
      atomic_fetch_max_explicit(metrics + 4, as_type<uint>(ratio), memory_order_relaxed);
    }
  }

  kernel void wan_ffn_epilogue_corrupt(device float *x [[buffer(0)]],
                                       uint index [[thread_position_in_grid]]) {
    if (index == 0u) x[0] += 1.0f;
  }
METAL

private def wan_ffn_bias_value(column : Int32) : Float32
  case column
  when 0 then 0.0_f32
  when 1 then 0.5_f32
  when 2 then -0.5_f32
  when 3 then 1.0_f32
  when 4 then -1.0_f32
  when 5 then 4_096.0_f32
  when 6 then -4_096.0_f32
  when 7 then 1.0e-6_f32
  else        ((column % 257).to_f32 - 128.0_f32) / 1_280.0_f32
  end
end

private def wan_ffn_encode_bias(e : ML::Metal::ComputeEncoder,
                                pipeline : ML::Metal::ComputePipeline,
                                output : ML::MetalBuffer, bias32 : ML::MetalBuffer,
                                bias16 : ML::MetalBuffer, width : Int32,
                                count : Int32, bf16 : Bool) : Nil
  e.set_pipeline(pipeline)
  e.set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
  # The existing baseline kernel selects one pointer; bind the same typed
  # source buffer at both slots, exactly as MetalLinearMPS.project does.
  e.set_buffer(bf16 ? bias16 : bias32, 1).set_buffer(bf16 ? bias16 : bias32, 2)
  e.set_value(width.to_u32, 3).set_value(bf16 ? 1_u32 : 0_u32, 4)
  e.set_value(count.to_u32, 5).dispatch_1d(count)
end

private def wan_ffn_encode_fill(e : ML::Metal::ComputeEncoder,
                                pipeline : ML::Metal::ComputePipeline,
                                baseline : ML::MetalBuffer, candidate : ML::MetalBuffer,
                                bias32 : ML::MetalBuffer, bias16 : ML::MetalBuffer,
                                width : Int32, count : Int32, bf16 : Bool) : Nil
  e.set_pipeline(pipeline)
  e.set_buffer(baseline, 0, ML::Metal::BufferAccess::Write)
  e.set_buffer(candidate, 1, ML::Metal::BufferAccess::Write)
  e.set_buffer(bias32, 2).set_buffer(bias16, 3)
  e.set_value(width.to_u32, 4).set_value(bf16 ? 1_u32 : 0_u32, 5)
  e.set_value(count.to_u32, 6).dispatch_1d(count)
end

private def wan_ffn_report_device : Nil
  device = ML::Metal::Device.instance
  puts({schema: "wan_ffn_epilogue_device_v1", device_name: device.name,
        recommended_working_set_bytes: device.recommended_working_set_size}.to_json)
  STDOUT.flush
end

rows = DEFAULT_ROWS
width = DEFAULT_WIDTH
trace_pre_gelu = false
upstream_fixture = nil.as(String?)
OptionParser.parse do |parser|
  parser.on("--rows=N", "Synthetic row count (default #{DEFAULT_ROWS})") { |value| rows = value.to_i }
  parser.on("--width=N", "FFN width (default #{DEFAULT_WIDTH})") { |value| width = value.to_i }
  parser.on("--trace-pre-gelu", "Run an untimed candidate pre-GELU trace comparison") { trace_pre_gelu = true }
  parser.on("--upstream-fixture=PATH", "Qualify the isolated FFN slice on fixed upstream ffn_mod") { |value| upstream_fixture = value }
end

if fixture_path = upstream_fixture
  fused_trace = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_fused_trace", SOURCE)
  compare = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_compare", SOURCE)
  zero = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_zero", SOURCE)
  prepared = WanFFNEpilogueUpstream::Prepared.new(fixture_path, fused_trace, compare, zero)
  begin
    wan_ffn_report_device
    puts prepared.setup_json
    STDOUT.flush
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    allocations_before = ML::MetalBuffer.video_spec_allocations
    prepared.encode!(encoder)
    encoder.end_encoding
    allocations = ML::MetalBuffer.video_spec_allocations - allocations_before
    raise "FFN upstream encode allocated MetalBuffers" unless allocations == 0
    gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
    prepared.validate_after_terminal_wait!
    puts({schema: "wan_ffn_epilogue_upstream_completion_v1", domain: "wan.s2v.ffn.epilogue",
          gpu_elapsed_seconds: gpu_seconds, terminal_waits: 1,
          inter_operator_host_waits: 0, activation_readbacks: 0,
          encode_metal_buffer_allocations: allocations,
          scope: "isolated projection/bias/GELU given upstream ffn_mod; not full-block parity"}.to_json)
    STDOUT.flush
  ensure
    prepared.release_after_terminal_wait!
  end
  exit 0
end

raise ArgumentError.new("rows must be at least 16 and width at least 8") unless rows >= 16 && width >= 8
count64 = rows.to_i64 * width
raise ArgumentError.new("FFN dispatch exceeds Int32") if count64 > Int32::MAX
count = count64.to_i32
trace_bytes = trace_pre_gelu ? count64 * sizeof(Float32) : 0_i64
estimated_bytes = 2_i64 * count64 * sizeof(Float32) +
  width.to_i64 * (sizeof(Float32) + sizeof(UInt16)) +
  5_i64 * sizeof(UInt32) + trace_bytes
raise "FFN epilogue explicit estimate exceeds the 1 GiB cap" if estimated_bytes > MAX_EXPLICIT_BYTES
puts({schema: "wan_ffn_epilogue_memory_plan_v1", domain: "wan.s2v.ffn.epilogue",
      rows: rows, width: width, values_per_arm: count64,
      arm_bytes: count64 * sizeof(Float32), trace_pre_gelu: trace_pre_gelu,
      explicit_estimate_bytes: estimated_bytes, explicit_cap_bytes: MAX_EXPLICIT_BYTES,
      host_memory_admission: "external cache-aware guard",
      model_weights_loaded: false, scope: "synthetic post-MPS bias plus GELU only"}.to_json)
STDOUT.flush

arena = ML::Video::Arena.new
begin
  wan_ffn_report_device
  baseline = arena.alloc("ffn_baseline", count)
  candidate = arena.alloc("ffn_candidate", count)
  bias32 = arena.alloc("bias_f32", width)
  bias16 = arena.alloc_weight("bias_bf16", width, ML::Video::WeightDType::BF16)
  metrics = arena.alloc("metrics", 5)
  trace = trace_pre_gelu ? arena.alloc("pre_gelu_trace", count) : nil

  bias_values = Array(Float32).new(width) { |column| wan_ffn_bias_value(column) }
  bf16_values = Array(UInt16).new(width) do |column|
    (bias_values[column].unsafe_as(UInt32) >> 16).to_u16
  end
  bias32.write(bias_values)
  bias16.write_bytes(bf16_values.to_unsafe.as(Pointer(UInt8)), width * sizeof(UInt16))
  raise "FFN epilogue allocation accounting mismatch" unless arena.resident_bytes == estimated_bytes

  # Resolve every pipeline before any timed arm. The baseline uses only the
  # existing MPS bias kernel and common approximate GELU pipeline.
  baseline_bias_pipeline = ML::Video::MetalLinearMPS.pipeline
  ML::Video::MetalOps.pipeline("video_gelu_f32")
  fill_pipeline = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_fill", SOURCE)
  candidate_pipeline = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_fused", SOURCE)
  trace_pipeline = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_fused_trace", SOURCE)
  zero_pipeline = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_zero", SOURCE)
  compare_pipeline = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_compare", SOURCE)
  corrupt_pipeline = ML::Metal::ComputePipeline.new("wan_ffn_epilogue_corrupt", SOURCE)

  # One setup wait restores identical inputs before every pair. The order is
  # warmup AB, then exactly AB/BA/AB measured pairs for each source bias dtype.
  {false, true}.each do |bf16|
    dtype = bf16 ? "bf16" : "f32"
    (-1...PAIRS).each do |pair|
      setup = ML::Metal::CommandBuffer.new
      setup_encoder = ML::Metal::ComputeEncoder.new(setup)
      wan_ffn_encode_fill(setup_encoder, fill_pipeline, baseline, candidate,
        bias32, bias16, width, count, bf16)
      setup_encoder.end_encoding
      setup.commit_and_wait

      order = pair < 0 || pair.even? ? {"baseline", "candidate"} : {"candidate", "baseline"}
      order.each_with_index do |variant, order_index|
        command = ML::Metal::CommandBuffer.new
        e = ML::Metal::ComputeEncoder.new(command)
        output = variant == "baseline" ? baseline : candidate
        allocations_before = ML::MetalBuffer.video_spec_allocations
        started = Time.instant
        if variant == "baseline"
          wan_ffn_encode_bias(e, baseline_bias_pipeline, output, bias32, bias16,
            width, count, bf16)
          ML::Video::MetalOps.gelu(e, output, output, count, approximate: true)
        else
          e.set_pipeline(candidate_pipeline)
          e.set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
          e.set_buffer(bias32, 1).set_buffer(bias16, 2)
          e.set_value(width.to_u32, 3).set_value(bf16 ? 1_u32 : 0_u32, 4)
          e.set_value(count.to_u32, 5).dispatch_1d(count)
        end
        e.end_encoding
        encode_seconds = (Time.instant - started).total_seconds
        gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
        arm_wall_seconds = (Time.instant - started).total_seconds
        encode_allocations = ML::MetalBuffer.video_spec_allocations - allocations_before
        raise "FFN epilogue encode allocated MetalBuffers" unless encode_allocations == 0
        puts({schema: "wan_ffn_epilogue_arm_v1", domain: "wan.s2v.ffn.epilogue",
              pair: pair, warmup: pair < 0, order: order_index, variant: variant,
              rows: rows, width: width, dtype: dtype,
              encode_wall_seconds: encode_seconds, arm_wall_seconds: arm_wall_seconds,
              gpu_elapsed_seconds: gpu_seconds,
              dispatches: variant == "baseline" ? 2 : 1,
              terminal_waits: 1, inter_operator_host_waits: 0,
              activation_readbacks: 0, encode_metal_buffer_allocations: encode_allocations,
              explicit_bytes: arena.resident_bytes,
              scope: "synthetic post-MPS bias plus approximate GELU; not a full block or model speedup"}.to_json)
        STDOUT.flush
      end

      validation = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(validation)
      e.set_pipeline(zero_pipeline).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
      e.set_pipeline(compare_pipeline).set_buffer(baseline, 0).set_buffer(candidate, 1)
        .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_value(count.to_u32, 3).dispatch_1d(count)
      e.end_encoding
      validation.commit_and_wait
      result = metrics.contents.as(UInt32*)
      passed = result[0] == 0 && result[1] == 0 && result[2] == 0
      puts({schema: "wan_ffn_epilogue_comparison_v1", domain: "wan.s2v.ffn.epilogue",
            pair: pair, warmup: pair < 0, rows: rows, width: width, dtype: dtype,
            compared_values: count64, bit_differences: result[0],
            tolerance_violations: result[1], nonfinite: result[2],
            max_abs: result[3].unsafe_as(Float32), max_ratio: result[4].unsafe_as(Float32),
            tolerance: "2e-5+2e-5*abs(reference)", bitwise_equal: result[0] == 0,
            passed: passed, validation_outside_timed_arms: true,
            scope: "synthetic post-MPS bias plus approximate GELU"}.to_json)
      STDOUT.flush
      raise "FFN epilogue strict comparison failed for #{dtype} pair #{pair}" unless passed
    end

    if trace_pre_gelu
      trace_buffer = trace.not_nil!
      validation = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(validation)
      wan_ffn_encode_fill(e, fill_pipeline, baseline, candidate,
        bias32, bias16, width, count, bf16)
      wan_ffn_encode_bias(e, baseline_bias_pipeline, baseline, bias32, bias16,
        width, count, bf16)
      e.set_pipeline(trace_pipeline)
      e.set_buffer(candidate, 0, ML::Metal::BufferAccess::ReadWrite)
      e.set_buffer(trace_buffer, 1, ML::Metal::BufferAccess::Write)
      e.set_buffer(bias32, 2).set_buffer(bias16, 3)
      e.set_value(width.to_u32, 4).set_value(bf16 ? 1_u32 : 0_u32, 5)
      e.set_value(count.to_u32, 6).dispatch_1d(count)
      e.set_pipeline(zero_pipeline).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
      e.set_pipeline(compare_pipeline).set_buffer(baseline, 0).set_buffer(trace_buffer, 1)
        .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_value(count.to_u32, 3).dispatch_1d(count)
      e.end_encoding
      validation.commit_and_wait
      result = metrics.contents.as(UInt32*)
      passed = result[0] == 0 && result[1] == 0 && result[2] == 0
      puts({schema: "wan_ffn_epilogue_trace_comparison_v1", domain: "wan.s2v.ffn.epilogue",
            rows: rows, width: width, dtype: dtype, compared_values: count64,
            bit_differences: result[0], tolerance_violations: result[1], nonfinite: result[2],
            max_abs: result[3].unsafe_as(Float32), max_ratio: result[4].unsafe_as(Float32),
            tolerance: "2e-5+2e-5*abs(reference)", bitwise_equal: result[0] == 0,
            passed: passed, trace_bytes: trace_bytes,
            validation_outside_timed_arms: true,
            scope: "optional candidate pre-GELU trace versus existing F32 bias output"}.to_json)
      STDOUT.flush
      raise "FFN pre-GELU trace comparison failed for #{dtype}" unless passed

      # The trace check leaves baseline at the bias sum and candidate after
      # GELU. Also compare complete outputs so trace mode validates both
      # candidate products and leaves equivalent tensors for the corruption test.
      validation = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(validation)
      ML::Video::MetalOps.gelu(e, baseline, baseline, count, approximate: true)
      e.set_pipeline(zero_pipeline).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
      e.set_pipeline(compare_pipeline).set_buffer(baseline, 0).set_buffer(candidate, 1)
        .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_value(count.to_u32, 3).dispatch_1d(count)
      e.end_encoding
      validation.commit_and_wait
      result = metrics.contents.as(UInt32*)
      passed = result[0] == 0 && result[1] == 0 && result[2] == 0
      puts({schema: "wan_ffn_epilogue_trace_output_comparison_v1", domain: "wan.s2v.ffn.epilogue",
            rows: rows, width: width, dtype: dtype, compared_values: count64,
            bit_differences: result[0], tolerance_violations: result[1], nonfinite: result[2],
            max_abs: result[3].unsafe_as(Float32), max_ratio: result[4].unsafe_as(Float32),
            tolerance: "2e-5+2e-5*abs(reference)", bitwise_equal: result[0] == 0,
            passed: passed, validation_outside_timed_arms: true,
            scope: "optional trace candidate final output versus existing bias plus GELU"}.to_json)
      STDOUT.flush
      raise "FFN trace candidate output comparison failed for #{dtype}" unless passed
    end
  end

  # Positive control: a one-element +1 perturbation must be caught by both the
  # bitwise and strict-tolerance counters after all timed/comparison work.
  control = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(control)
  e.set_pipeline(corrupt_pipeline).set_buffer(candidate, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  e.set_pipeline(zero_pipeline).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  e.set_pipeline(compare_pipeline).set_buffer(baseline, 0).set_buffer(candidate, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(count.to_u32, 3).dispatch_1d(count)
  e.end_encoding
  control.commit_and_wait
  result = metrics.contents.as(UInt32*)
  rejected = result[0] == 1 && result[1] == 1 && result[2] == 0
  puts({schema: "wan_ffn_epilogue_corruption_control_v1", domain: "wan.s2v.ffn.epilogue",
        rows: rows, width: width, dtype: "bf16", injected_corruptions: 1,
        bit_differences: result[0], tolerance_violations: result[1],
        nonfinite: result[2], rejected: rejected,
        scope: "post-completion positive control; outside timed arms"}.to_json)
  STDOUT.flush
  raise "FFN epilogue comparator failed the corruption positive control" unless rejected
ensure
  arena.release
end
