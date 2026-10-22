# Opt-in real-checkpoint projection probe. No model or video inference.
# Prepare the immutable fixture with support/wan_quantized_projection_reference.py.
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
METAL

private def packed_probe_compare(e : ML::Metal::ComputeEncoder,
                                 zero : ML::Metal::ComputePipeline,
                                 compare : ML::Metal::ComputePipeline,
                                 actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                                 metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
                                 count : Int32) : Nil
  e.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
  e.set_pipeline(compare).set_buffer(actual, 0).set_buffer(expected, 1)
    .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_buffer(partials, 3, ML::Metal::BufferAccess::Write).set_value(count.to_u32, 4)
    .dispatch_threadgroups({(count + 255) // 256, 1, 1}, {256, 1, 1})
end

private def packed_probe_metrics(metrics : ML::MetalBuffer, partials : ML::MetalBuffer,
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

fixture_dir = ""
rounds = 8
OptionParser.parse do |parser|
  parser.on("--fixture=DIR", "Immutable upstream fixture directory") { |v| fixture_dir = v }
  parser.on("--rounds=N", "Balanced forward/reverse rotations; 8 or 16") { |v| rounds = v.to_i }
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
raise ArgumentError.new("rounds must be 8 or 16") unless {8, 16}.includes?(rounds)
manifest_path = File.join(fixture_dir, "manifest.json")
manifest = JSON.parse(File.read(manifest_path))
raise "Unsupported fixture schema" unless manifest["schema"].as_s == "wan_quantized_projection_reference_v1"
geometry = manifest["geometry"]
rows = geometry["rows"].as_i
source_rows = geometry["source_rows"].as_i
din = geometry["input_dim"].as_i
dout = geometry["output_dim"].as_i
unless {4, 4480}.includes?(rows) && source_rows == 4 && din == 5120 && dout == 13824 &&
       geometry["expected_projection_rows"].as_i == rows
  raise "Probe admits only the captured four rows or the 4480-row repeated geometry"
end
weight_count = din * dout
output_count = rows * dout
variants = ["bf16", "q4_k", "qbit_p4", "qbit_p5"]
variant_manifest = manifest["variants"]
manifest_hash = packed_probe_sha256(manifest_path)
unless manifest["comparison"]["atol"].as_f == 2e-5 && manifest["comparison"]["rtol"].as_f == 2e-5 &&
       manifest["source_tensor"]["weight_name"].as_s == "blocks.0.ffn.0.weight" &&
       manifest["oracle"]["device"].as_s == "MPS" && manifest["oracle"]["fallback_disabled"].as_bool
  raise "Fixture is outside the unchanged projection reference contract"
end
packed_probe_verify_file(fixture_dir, manifest["input"], "input.f32")
variants.each do |variant|
  v = variant_manifest[variant]
  {"packed_weight", "decoded_weight", "expected_projection", "bias"}.each do |key|
    packed_probe_verify_file(fixture_dir, v[key], v["#{key}_file"].as_s)
  end
  if variant.starts_with?("qbit")
    packed_probe_verify_file(fixture_dir, v["lut"], "lut_p#{variant[-1]}.f32")
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
  corrupt = ML::Metal::ComputePipeline.new("packed_probe_corrupt", PACKED_PROBE_SOURCE)
  source = arena.alloc("captured_input", source_rows * din)
  input = arena.alloc("input", rows * din)
  stage = arena.alloc("weight_stage", weight_count)
  output = arena.alloc("output", output_count)
  expected_weight = arena.alloc("expected_weight", weight_count)
  expected_output = arena.alloc("expected_output", output_count)
  baseline_output = arena.alloc("baseline_output", output_count)
  bias = arena.alloc("bias", dout)
  lut_p4 = arena.alloc("lut_p4", 256)
  lut_p5 = arena.alloc("lut_p5", 256)
  packed = {} of String => ML::MetalBuffer
  variants.each do |variant|
    bytes = case variant
            when "bf16"    then weight_count.to_i64 * 2
            when "q4_k"    then weight_count.to_i64 // 256 * 144
            when "qbit_p4" then weight_count.to_i64 // 256 * 136
            else                weight_count.to_i64 // 256 * 168
            end
    packed[variant] = arena.alloc("packed_#{variant}", (bytes // 4).to_i32)
    path = File.join(fixture_dir, variant_manifest[variant]["packed_weight_file"].as_s)
    packed_probe_upload(path, packed[variant], bytes)
  end
  weight_metrics = arena.alloc("weight_metrics", 5)
  output_metrics = arena.alloc("output_metrics", 5)
  loss_metrics = arena.alloc("loss_metrics", 5)
  partial_count = ((weight_count + 255) // 256) * 4
  weight_partials = arena.alloc("weight_partials", partial_count)
  output_partials = arena.alloc("output_partials", partial_count)
  loss_partials = arena.alloc("loss_partials", partial_count)
  raise "Explicit probe buffers exceed 2 GiB" if arena.resident_bytes > 2_i64 * 1024**3
  packed_probe_upload(File.join(fixture_dir, "input.f32"), source, source_rows.to_i64 * din * 4)
  packed_probe_upload(File.join(fixture_dir, "bias.f32"), bias, dout.to_i64 * 4)
  packed_probe_upload(File.join(fixture_dir, "lut_p4.f32"), lut_p4, 1024_i64)
  packed_probe_upload(File.join(fixture_dir, "lut_p5.f32"), lut_p5, 1024_i64)
  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
    layers: 1, batches: 1, tokens: rows)
  plan = ML::Video::WanT5MPSLinearPlan.new(cfg, din, dout, input, stage, output)
  setup = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(setup)
  e.set_pipeline(tile).set_buffer(source, 0).set_buffer(input, 1, ML::Metal::BufferAccess::Write)
    .set_value((source_rows * din).to_u32, 2).set_value((rows * din).to_u32, 3)
    .dispatch_1d(rows * din)
  e.end_encoding
  setup.commit
  setup.wait

  decode = ->(encoder : ML::Metal::ComputeEncoder, variant : String) {
    case variant
    when "bf16"
      encoder.set_pipeline(ML::Video::WanT5.pipeline("wan_t5_expand_fc2"))
        .set_buffer(packed[variant], 0).set_buffer(packed[variant], 1)
        .set_buffer(stage, 2, ML::Metal::BufferAccess::Write)
        .set_value(weight_count.to_u32, 3).set_value(1_u32, 4).dispatch_1d(weight_count)
    when "q4_k"
      ML::Video::PackedWeightDecode.q4_k(encoder, packed[variant], stage, weight_count)
    else
      precision = variant == "qbit_p4" ? 4 : 5
      ML::Video::PackedWeightDecode.qbit(encoder, packed[variant], stage,
        precision == 4 ? lut_p4 : lut_p5, weight_count, precision)
    end
    nil
  }
  project = ->(encoder : ML::Metal::ComputeEncoder) {
    encoder.encode_wan_t5_mps_fc2(plan.not_nil!.handle)
    encoder.set_pipeline(ML::Video::MetalLinearMPS.pipeline)
      .set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(bias, 1).set_buffer(bias, 2).set_value(dout.to_u32, 3)
      .set_value(0_u32, 4).set_value(output_count.to_u32, 5).dispatch_1d(output_count)
    nil
  }

  puts({schema: "wan_packed_projection_setup_v1", manifest_sha256: manifest_hash,
        projection: "blocks.0.ffn.0", rows: rows, source_rows: source_rows, din: din, dout: dout,
        input_scope: "upstream-captured seeded activation; periodic row tiling, not a video clip",
        explicit_buffer_bytes: arena.resident_bytes,
        packed_bytes: packed.transform_values(&.size), f32_weight_stage_bytes: stage.size,
        input_bytes: input.size, output_bytes: output.size,
        mps_plan_storage: "opaque, excluded from explicit buffer accounting",
        tolerance: "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)"}.to_json)

  validate = ->(phase : String) {
    variants.each do |variant|
      v = variant_manifest[variant]
      packed_probe_upload(File.join(fixture_dir, v["decoded_weight_file"].as_s), expected_weight, stage.size)
      packed_probe_upload(File.join(fixture_dir, v["expected_projection_file"].as_s), expected_output, output.size)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      before = ML::MetalBuffer.video_spec_allocations
      decode.call(encoder, variant)
      packed_probe_compare(encoder, zero, compare, stage, expected_weight,
        weight_metrics, weight_partials, weight_count)
      project.call(encoder)
      if variant == "bf16"
        encoder.set_pipeline(copy).set_buffer(output, 0)
          .set_buffer(baseline_output, 1, ML::Metal::BufferAccess::Write)
          .set_value(output_count.to_u32, 2).dispatch_1d(output_count)
      end
      packed_probe_compare(encoder, zero, compare, output, expected_output,
        output_metrics, output_partials, output_count)
      packed_probe_compare(encoder, zero, compare, output, baseline_output,
        loss_metrics, loss_partials, output_count)
      encoder.end_encoding
      command.commit
      command.wait
      raise "Validation encode allocated MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
      wm = packed_probe_metrics(weight_metrics, weight_partials, weight_count)
      om = packed_probe_metrics(output_metrics, output_partials, output_count)
      lm = packed_probe_metrics(loss_metrics, loss_partials, output_count)
      puts({schema: "wan_packed_projection_parity_v1", phase: phase, variant: variant,
            manifest_sha256: manifest_hash, decoder: wm, projection: om,
            lossy_vs_native_bf16: lm, validation_outside_timed_arm: true}.to_json)
      STDOUT.flush
      raise "GPU decoded weights failed unchanged upstream gate: #{variant}" unless wm[:tolerance_violations] == 0 && wm[:nonfinite] == 0
      raise "GPU projection failed unchanged upstream gate: #{variant}" unless om[:tolerance_violations] == 0 && om[:nonfinite] == 0
    end
    nil
  }
  validate.call("before_timing")

  # Reject invalid wire dimensions/precision/aliasing before dispatch.
  rejected_guards = 0
  guard_command = ML::Metal::CommandBuffer.new
  guard_encoder = ML::Metal::ComputeEncoder.new(guard_command)
  [0, 257].each do |invalid|
    begin
      ML::Video::PackedWeightDecode.q4_k(guard_encoder, packed["q4_k"], stage, invalid)
    rescue ArgumentError
      rejected_guards += 1
    end
  end
  begin
    ML::Video::PackedWeightDecode.qbit(guard_encoder, packed["qbit_p4"], stage, lut_p4, weight_count, 3)
  rescue ArgumentError
    rejected_guards += 1
  end
  begin
    ML::Video::PackedWeightDecode.q4_k(guard_encoder, stage, stage, 256)
  rescue ArgumentError
    rejected_guards += 1
  end
  begin
    ML::Video::PackedWeightDecode.q4_k(guard_encoder, packed["qbit_p4"], stage, weight_count)
  rescue ArgumentError
    rejected_guards += 1
  end
  begin
    ML::Video::PackedWeightDecode.qbit(guard_encoder, packed["qbit_p4"], stage,
      packed["qbit_p4"], 256, 4)
  rescue ArgumentError
    rejected_guards += 1
  end
  guard_encoder.end_encoding
  guard_command.commit
  guard_command.wait
  raise "Invalid packed format guard accepted input" unless rejected_guards == 6

  {"projection", "decode_only"}.each do |mode|
    (-2...rounds).each do |round|
      order = variants.rotate(Math.max(round, 0) % 4)
      order = order.reverse if round == -1 || round >= rounds // 2
      order.each_with_index do |variant, position|
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        before = ML::MetalBuffer.video_spec_allocations
        started = Time.instant
        decode.call(encoder, variant)
        project.call(encoder) if mode == "projection"
        encoder.end_encoding
        encode_seconds = (Time.instant - started).total_seconds
        gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
        wall_seconds = (Time.instant - started).total_seconds
        raise "Timed encode allocated MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
        puts({schema: "wan_packed_projection_timing_v1", mode: mode, variant: variant,
              round: round, position: position, warmup: round < 0, rows: rows, din: din, dout: dout,
              encode_seconds: encode_seconds, gpu_seconds: gpu_seconds, wall_seconds: wall_seconds,
              terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
              encode_metal_buffer_allocations: 0, manifest_sha256: manifest_hash}.to_json)
        STDOUT.flush
      end
    end
  end
  validate.call("after_timing")

  # Positive controls: a finite error and a NaN must both fail the same gate.
  command = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(command)
  e.set_pipeline(corrupt).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  packed_probe_compare(e, zero, compare, output, expected_output, output_metrics, output_partials, output_count)
  e.end_encoding
  command.commit
  command.wait
  result = packed_probe_metrics(output_metrics, output_partials, output_count)
  raise "Corruption comparator failed" unless result[:tolerance_violations] == 2 && result[:nonfinite] == 1
  puts({schema: "wan_packed_projection_controls_v1", rejected_invalid_encodes: rejected_guards,
        corruption_rejected: true, corruption: result, manifest_sha256: manifest_hash}.to_json)
ensure
  plan.try(&.release)
  arena.release
end
