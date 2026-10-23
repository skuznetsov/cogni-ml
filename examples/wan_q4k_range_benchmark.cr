# Opt-in equal-storage projection experiment. Does not load a Wan model.
require "json"
require "digest/sha256"
require "option_parser"
require "path"
require "../src/ml/video/metal_linear_mps"
require "../src/ml/video/packed_weight_decode"
require "../src/ml/video/q4k_range_weights"
require "../spec/support/video_allocation_counter"
require "./support/wan_affine_budget_metrics"

fixture_dir = ""
rounds = 0
reverse_first = false
OptionParser.parse do |p|
  p.on("--fixture=DIR", "Immutable independent upstream fixture") { |v| fixture_dir = v }
  p.on("--reverse-first", "Reverse the first warmup and paired order") { reverse_first = true }
  p.on("--rounds=N", "Two or four complete forward/reverse arm rotations") { |v| rounds = v.to_i }
end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
rounds = 12 if rounds == 0
raise ArgumentError.new("Invalid balanced round count") unless {12, 24}.includes?(rounds)
manifest_path = File.join(fixture_dir, "manifest.json")
manifest = JSON.parse(File.read(manifest_path))
raise "Unsupported fixture" unless manifest["schema"].as_s == "wan_q4k_range_reference_v1"
geometry = manifest["geometry"]
rows = geometry["rows"].as_i
din = geometry["input_dim"].as_i
dout = geometry["output_dim"].as_i
unless {4, 4480}.includes?(rows) && geometry["source_rows"].as_i == 4 && din == 5120 && dout == 13824
  raise "Only the fixed real projection with four unique seeded source rows is admitted"
end
unless manifest["comparison"]["atol"].as_f == 2e-5 && manifest["comparison"]["rtol"].as_f == 2e-5 &&
       manifest["comparison"]["finite"].as_bool &&
       manifest["oracle"]["device"].as_s == "MPS" && manifest["oracle"]["fallback_disabled"].as_bool
  raise "Unchanged finite upstream gate is required"
end
variants = ["q4_k", "q4_k_minmax8", "q4_k_minmax1"]
variant_manifest = manifest["variants"]
raise "Unexpected fixture arm set" unless variant_manifest.as_h.keys.sort == variants.sort
source_fixture = manifest["source_fixture"]
{
  "manifest_sha256"          => "0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703",
  "checkpoint_weight_sha256" => "cd1037a3eb9c98cf5baed570e6825e7bb2463165563707cf7f1c3b367afca557",
  "input_sha256"             => "2d1462e25bfc05523da2ac8a7487c21b08986ef3eb9fe2f0a2f9527c9b65618c",
  "bias_f32_sha256"          => "6d264e0db1ffc55271a30cc6c8e60501effb4385b4200c007dc7a9c54d549b2f",
  "q4_k_sha256"              => "418e4e461b5cc931b9af01e4f3c216f0da61e4fc02087c7deab425fe2e15d737",
}.each do |key, expected|
  raise "Substituted source fixture: #{key}" unless source_fixture[key].as_s == expected
end
{"input" => "input_sha256", "bias" => "bias_f32_sha256", "original_weight" => "checkpoint_weight_sha256"}.each do |key, source_key|
  raise "Substituted source tensor: #{key}" unless manifest[key]["sha256"] == source_fixture[source_key]
end
raise "Substituted Q4_K payload" unless variant_manifest["q4_k"]["payload"]["sha256"] == source_fixture["q4_k_sha256"]
range_map = {"q4_k" => 8, "q4_k_minmax8" => 8, "q4_k_minmax1" => 1}
weight_count = din * dout
output_count = rows * dout
budget = ML::Video::Q4KRangeWeights.payload_bytes(weight_count)
raise "Incorrect common budget" unless budget == 39_813_120
record_path = ->(record : JSON::Any) {
  name = record["path"].as_s
  path = File.expand_path(name, fixture_dir)
  root = File.expand_path(fixture_dir) + File::SEPARATOR
  raise "Fixture path escapes its directory" unless path.starts_with?(root) && File.real_path(path).starts_with?(File.real_path(fixture_dir) + File::SEPARATOR) && !File.symlink?(path)
  affine_budget_verify_file(fixture_dir, record, name)
  path
}
input_path = record_path.call(manifest["input"])
bias_path = record_path.call(manifest["bias"])
original_path = record_path.call(manifest["original_weight"])
baseline = manifest["baseline"]
baseline_projection_path = record_path.call(baseline["expected_projection"])
record_path.call(baseline["decoded_weight"])
variants.each do |name|
  v = variant_manifest[name]
  raise "Unexpected range count" unless v["ranges"].as_i == range_map[name]
  raise "Unequal packed budget" unless v["payload"]["bytes"].as_i64 == budget
  descriptor_path = record_path.call(v["descriptor"])
  raise "Descriptor must occupy exactly 16 bytes" unless File.size(descriptor_path) == 16
  format_id = 0
  File.open(descriptor_path, "rb") do |file|
    descriptor = Array.new(4) { file.read_bytes(UInt32, IO::ByteFormat::LittleEndian) }
    raise "Invalid common descriptor" unless descriptor == [0x31425157_u32, format_id.to_u32, dout.to_u32, din.to_u32]
  end
  {"payload", "decoded_weight", "expected_projection"}.each { |key| record_path.call(v[key]) }
end

manifest_hash = affine_budget_sha256(manifest_path)
arena = ML::Video::Arena.new
plan : ML::Video::WanT5MPSLinearPlan? = nil
begin
  ML::Video::MetalLinearMPS.prepare!
  ML::Video::PackedWeightDecode.prepare!
  ML::Video::Q4KRangeWeights.prepare!
  tile = ML::Metal::ComputePipeline.new("affine_budget_tile", AFFINE_BUDGET_SOURCE)
  zero = ML::Metal::ComputePipeline.new("affine_budget_zero", AFFINE_BUDGET_SOURCE)
  compare = ML::Metal::ComputePipeline.new("affine_budget_compare", AFFINE_BUDGET_SOURCE)
  bytes_compare = ML::Metal::ComputePipeline.new("affine_budget_bytes", AFFINE_BUDGET_SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("affine_budget_corrupt", AFFINE_BUDGET_SOURCE)
  source = arena.alloc("source", 4 * din)
  input = arena.alloc("input", rows * din)
  original = arena.alloc("original", (weight_count.to_i64 // 2).to_i32)
  stage = arena.alloc("stage", weight_count)
  expected_weight = arena.alloc("expected_weight", weight_count)
  output = arena.alloc("output", output_count)
  expected_output = arena.alloc("expected_output", output_count)
  baseline_output = arena.alloc("baseline_output", output_count)
  bias = arena.alloc("bias", dout)
  status = arena.alloc("status", ML::Video::Q4KRangeWeights::STATUS_WORDS)
  status.contents.as(UInt32*).clear(ML::Video::Q4KRangeWeights::STATUS_WORDS)
  expected_payload = arena.alloc("expected_payload", (budget // 4).to_i32)
  payloads = {} of String => ML::MetalBuffer
  variants.each { |name| payloads[name] = arena.alloc(name, (budget // 4).to_i32) }
  metrics = arena.alloc("metrics", 5)
  weight_metrics = arena.alloc("weight_metrics", 5)
  partial_count = ((Math.max(weight_count, output_count) + 255) // 256) * 4
  partials = arena.alloc("partials", partial_count)
  weight_partials = arena.alloc("weight_partials", ((weight_count + 255) // 256) * 4)
  row_metrics = Array.new(4) { |r| arena.alloc("row_metrics_#{r}", 5) }
  row_partials = Array.new(4) { |r| arena.alloc("row_partials_#{r}", ((dout + 255) // 256) * 4) }
  affine_budget_upload(input_path, source, source.size)
  affine_budget_upload(bias_path, bias, bias.size)
  affine_budget_upload(original_path, original, original.size)
  affine_budget_upload(baseline_projection_path, baseline_output, baseline_output.size)
  affine_budget_upload(File.join(fixture_dir, variant_manifest["q4_k"]["payload"]["path"].as_s), payloads["q4_k"], budget)
  raise "Explicit buffers exceed 3 GiB" if arena.resident_bytes > 3_i64 * 1024**3
  cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
    layers: 1, batches: 1, tokens: rows)
  plan = ML::Video::WanT5MPSLinearPlan.new(cfg, din, dout, input, stage, output)
  setup = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(setup)
  e.set_pipeline(tile).set_buffer(source, 0).set_buffer(input, 1, ML::Metal::BufferAccess::Write)
    .set_value((4 * din).to_u32, 2).set_value((rows * din).to_u32, 3).dispatch_1d(rows * din)
  e.end_encoding
  setup.commit
  setup.wait

  check_status = -> {
    words = Slice.new(status.contents.as(UInt32*), ML::Video::Q4KRangeWeights::STATUS_WORDS).to_a
    raise "Sticky codec failure: #{words}" unless words.all?(&.zero?)
    nil
  }
  puts({schema: "wan_q4k_range_setup_v1", manifest_sha256: manifest_hash,
        rows: rows, din: din, dout: dout, unique_source_rows: 4, real_video_derived: false,
        payload_bytes_per_arm: budget, common_descriptor_bytes_per_arm: 16,
        package_bytes_per_arm: budget + 16, shared_bias_bytes: bias.size,
        f32_stage_bytes: stage.size, explicit_buffer_bytes_during_pack: arena.resident_bytes,
        packing_workspace_bytes: 0,
        reverse_first: reverse_first, variants: variants,
        mps_storage: "opaque and included in the process guard, excluded from explicit byte sum",
        tolerance: "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)"}.to_json)
  STDOUT.flush

  # One-time GPU packing has its own terminal measurement; it is excluded from
  # resident-packed inference. Full byte qualification is a separate diagnostic.
  variants.reject { |v| v == "q4_k" }.each do |name|
    v = variant_manifest[name]
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    before = ML::MetalBuffer.video_spec_allocations
    started = Time.instant
    ML::Video::Q4KRangeWeights.pack(encoder, original, payloads[name], status, weight_count, range_map[name])
    encoder.end_encoding
    encode = (Time.instant - started).total_seconds
    gpu = cmd.commit_and_wait_gpu_elapsed_seconds
    wall = (Time.instant - started).total_seconds
    raise "GPU pack encoded an allocation" unless ML::MetalBuffer.video_spec_allocations == before
    check_status.call
    affine_budget_upload(File.join(fixture_dir, v["payload"]["path"].as_s), expected_payload, budget)
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
    encoder.set_pipeline(bytes_compare).set_buffer(payloads[name], 0).set_buffer(expected_payload, 1)
      .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite).set_value((budget // 4).to_u32, 3)
      .dispatch_1d((budget // 4).to_i32)
    encoder.end_encoding
    cmd.commit
    cmd.wait
    differences = metrics.contents.as(UInt32*)[0]
    puts({schema: "wan_q4k_range_packing_v1", variant: name, gpu_seconds: gpu,
          wall_seconds: wall, encode_seconds: encode, payload_word_differences: differences,
          encode_metal_buffer_allocations: 0, activation_readbacks: 0,
          terminal_waits: 1, byte_validation_outside_packing_interval: true}.to_json)
    STDOUT.flush
    raise "GPU packing differs from independent serialized oracle: #{name}" unless differences == 0
  end

  decode = ->(encoder : ML::Metal::ComputeEncoder, name : String) {
    ML::Video::PackedWeightDecode.q4_k(encoder, payloads[name], stage, weight_count)
    nil
  }
  project = ->(encoder : ML::Metal::ComputeEncoder) {
    ML::Video::MetalLinearMPS.project_preexpanded(encoder, input, bias, output, rows,
      din, dout, ML::Video::WeightDType::F32, plan.not_nil!, stage)
    nil
  }
  # Qualify the unchanged native BF16 control before treating its saved
  # upstream output as the distortion reference for every compressed arm.
  affine_budget_upload(File.join(fixture_dir, baseline["decoded_weight"]["path"].as_s), expected_weight, stage.size)
  cmd = ML::Metal::CommandBuffer.new
  encoder = ML::Metal::ComputeEncoder.new(cmd)
  encoder.set_pipeline(ML::Video::WanT5.pipeline("wan_t5_expand_fc2"))
    .set_buffer(original, 0).set_buffer(original, 1).set_buffer(stage, 2, ML::Metal::BufferAccess::Write)
    .set_value(weight_count.to_u32, 3).set_value(1_u32, 4).dispatch_1d(weight_count)
  affine_budget_compare(encoder, zero, compare, stage, expected_weight, weight_metrics, weight_partials, weight_count)
  project.call(encoder)
  affine_budget_compare(encoder, zero, compare, output, baseline_output, metrics, partials, output_count)
  encoder.end_encoding
  cmd.commit
  cmd.wait
  bw = affine_budget_metrics(weight_metrics, weight_partials, weight_count)
  bo = affine_budget_metrics(metrics, partials, output_count)
  raise "Native BF16 control failed" unless bw[:tolerance_violations] == 0 && bw[:nonfinite] == 0 && bo[:tolerance_violations] == 0 && bo[:nonfinite] == 0
  puts({schema: "wan_q4k_range_bf16_control_v1", decoder: bw, projection: bo}.to_json)
  validate = ->(phase : String) {
    variants.each do |name|
      v = variant_manifest[name]
      affine_budget_upload(File.join(fixture_dir, v["decoded_weight"]["path"].as_s), expected_weight, stage.size)
      affine_budget_upload(File.join(fixture_dir, v["expected_projection"]["path"].as_s), expected_output, output.size)
      cmd = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(cmd)
      before = ML::MetalBuffer.video_spec_allocations
      decode.call(encoder, name)
      affine_budget_compare(encoder, zero, compare, stage, expected_weight, weight_metrics, weight_partials, weight_count)
      project.call(encoder)
      affine_budget_compare(encoder, zero, compare, output, expected_output, metrics, partials, output_count)
      4.times do |r|
        affine_budget_compare(encoder, zero, compare, output, baseline_output,
          row_metrics[r], row_partials[r], dout, r * dout)
      end
      encoder.end_encoding
      cmd.commit
      cmd.wait
      wm = affine_budget_metrics(weight_metrics, weight_partials, weight_count)
      om = affine_budget_metrics(metrics, partials, output_count)
      check_status.call
      loss = Array.new(4) { |r| affine_budget_metrics(row_metrics[r], row_partials[r], dout) }
      raise "Validation encoded an allocation" unless ML::MetalBuffer.video_spec_allocations == before
      puts({schema: "wan_q4k_range_parity_v1", phase: phase, variant: name,
            decoder: wm, projection: om, lossy_vs_bf16_per_source_row: loss,
            calibration_rows: [] of Int32,
            unique_evaluation_rows: [0, 1, 2, 3]}.to_json)
      STDOUT.flush
      raise "Weight parity failed #{name}: #{wm}" unless wm[:tolerance_violations] == 0 && wm[:nonfinite] == 0
      raise "Projection parity failed #{name}: #{om}" unless om[:tolerance_violations] == 0 && om[:nonfinite] == 0
    end
    nil
  }
  validate.call("before_timing")
  {"projection", "decode_only"}.each do |mode|
    (-2...rounds).each do |round|
      order = variants.rotate(Math.max(round, 0) % variants.size)
      order = order.reverse if (round == -1 || round >= rounds // 2) != reverse_first
      order.each_with_index do |name, position|
        cmd = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(cmd)
        before = ML::MetalBuffer.video_spec_allocations
        started = Time.instant
        decode.call(encoder, name)
        project.call(encoder) if mode == "projection"
        encoder.end_encoding
        encode = (Time.instant - started).total_seconds
        gpu = cmd.commit_and_wait_gpu_elapsed_seconds
        wall = (Time.instant - started).total_seconds
        raise "Timed encode allocated a MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
        check_status.call
        puts({schema: "wan_q4k_range_timing_v1", mode: mode, variant: name,
              round: round, position: position, warmup: round < 0, rows: rows,
              encode_seconds: encode, gpu_seconds: gpu, wall_seconds: wall,
              terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
              encode_metal_buffer_allocations: 0, manifest_sha256: manifest_hash}.to_json)
        STDOUT.flush
      end
    end
  end
  validate.call("after_timing")
  cmd = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(cmd)
  e.set_pipeline(corrupt).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  affine_budget_compare(e, zero, compare, output, expected_output, metrics, partials, output_count)
  e.end_encoding
  cmd.commit
  cmd.wait
  corruption = affine_budget_metrics(metrics, partials, output_count)
  raise "Comparator accepted finite/NaN corruption" unless corruption[:tolerance_violations] == 2 && corruption[:nonfinite] == 1
  puts({schema: "wan_q4k_range_controls_v1", corruption_rejected: true,
        corruption: corruption, final_status_zero: true}.to_json)
ensure
  plan.try(&.release)
  arena.release
end
