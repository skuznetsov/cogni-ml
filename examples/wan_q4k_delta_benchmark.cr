# Opt-in sparse-delta projection experiment. Does not load a Wan model.
require "json"
require "digest/sha256"
require "option_parser"
require "path"
require "../src/ml/video/metal_linear_mps"
require "../src/ml/video/packed_weight_decode"
require "../src/ml/video/q4k_delta_weights"
require "../src/ml/video/q4k_delta_linear"
require "../spec/support/video_allocation_counter"
require "./support/wan_affine_budget_metrics"

fixture_dir = ""
rounds = 0
reverse_first = false
compare_block_validation = false
compare_gemm = false
compare_fused = false
gemm_variant_id = ""
gemm_variant_selected = false
OptionParser.parse do |p|
  p.on("--fixture=DIR", "Immutable independent upstream fixture") { |v| fixture_dir = v }
  p.on("--reverse-first", "Reverse the first warmup and paired order") { reverse_first = true }
  p.on("--rounds=N", "Balanced forward/reverse arm rotations") { |v| rounds = v.to_i }
  p.on("--compare-block-validation", "Pair per-value and per-block correction validation") { compare_block_validation = true }
  p.on("--compare-gemm", "Compare legacy and candidate native dense GEMM against MPS only") { compare_gemm = true }
  p.on("--compare-fused", "Pair fused Q4_K delta projection against block-validated staged controls") { compare_fused = true }
  p.on("--gemm-variant=ID", "Native GEMM geometry: legacy, candidate-b128n32-sg4-tk32, candidate-b64n32-sg4-tk32-bt, or candidate-b64n32-sg4-tk64-bt") do |v|
    gemm_variant_id = v
    gemm_variant_selected = true
  end
end
raise ArgumentError.new("Comparison modes are mutually exclusive") if
  (compare_block_validation && compare_gemm) || (compare_block_validation && compare_fused) ||
  (compare_gemm && compare_fused)
raise ArgumentError.new("--gemm-variant requires --compare-gemm or --compare-fused") if
  gemm_variant_selected && !compare_gemm && !compare_fused
gemm_variant_id = compare_gemm ? "candidate-b64n32-sg4-tk32-bt" : "legacy" if gemm_variant_id.empty?
raise ArgumentError.new("--compare-gemm requires a candidate variant") if
  compare_gemm && gemm_variant_id == "legacy"
gemm_variant = case gemm_variant_id
               when "legacy"
                 ML::Video::Q4KDeltaLinear::GemmVariant::Legacy
               when "candidate-b128n32-sg4-tk32"
                 ML::Video::Q4KDeltaLinear::GemmVariant::B128N32S4K32
               when "candidate-b64n32-sg4-tk32-bt"
                 ML::Video::Q4KDeltaLinear::GemmVariant::B64N32S4K32BT
               when "candidate-b64n32-sg4-tk64-bt"
                 ML::Video::Q4KDeltaLinear::GemmVariant::B64N32S4K64BT
               else
                 raise ArgumentError.new("Unknown GEMM variant: #{gemm_variant_id}")
               end
raise ArgumentError.new("--fixture is required") if fixture_dir.empty?
rounds = 16 if rounds == 0
raise ArgumentError.new("Invalid balanced round count") unless {16, 32}.includes?(rounds)
raise ArgumentError.new("--compare-gemm uses exactly 16 measured rounds") if compare_gemm && rounds != 16
manifest_path = File.join(fixture_dir, "manifest.json")
manifest = JSON.parse(File.read(manifest_path))
raise "Unsupported fixture" unless manifest["schema"].as_s == "wan_q4k_delta_reference_v1"
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
variants = ["q4_k", "q4_k_delta1", "q4_k_delta4", "q4_k_delta8"]
execution_arms = variants.dup
execution_arms.concat(variants.map { |name| "#{name}_block" }) if compare_block_validation
execution_arms = variants.flat_map { |name| [name, "#{name}_fused"] } if compare_fused
execution_arms = ["dense_native_legacy", "dense_native_candidate", "dense_mps"] if compare_gemm
format_by_arm = variants.to_h { |name| {name, name} }
variants.each { |name| format_by_arm["#{name}_block"] = name } if compare_block_validation
variants.each { |name| format_by_arm["#{name}_fused"] = name } if compare_fused
format_by_arm.merge!({"dense_native_legacy" => "q4_k", "dense_native_candidate" => "q4_k", "dense_mps" => "q4_k"}) if compare_gemm
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
patch_map = {"q4_k" => 0, "q4_k_delta1" => 1, "q4_k_delta4" => 4, "q4_k_delta8" => 8}
weight_count = din * dout
output_count = rows * dout
budget = (weight_count.to_i64 // 256) * ML::Video::PackedWeightDecode::Q4_K_BLOCK_BYTES
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
  raise "Unexpected correction quota" unless v["patches"].as_i == patch_map[name]
  sidecar_bytes = patch_map[name] == 0 ? 0_i64 : ML::Video::Q4KDeltaWeights.sidecar_bytes(weight_count, patch_map[name])
  raise "Incorrect sidecar size" unless v["sidecar"]["bytes"].as_i64 == sidecar_bytes
  record_path.call(v["sidecar"])
  raise "Substituted canonical base" unless v["payload"]["sha256"] == source_fixture["q4_k_sha256"]
  raise "Unequal packed budget" unless v["payload"]["bytes"].as_i64 == budget
  descriptor_path = record_path.call(v["descriptor"])
  raise "Descriptor must occupy exactly 16 bytes" unless File.size(descriptor_path) == 16
  format_id = patch_map[name] == 0 ? 0 : 0x100 + patch_map[name]
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
  ML::Video::Q4KDeltaWeights.prepare!
  ML::Video::Q4KDeltaLinear.prepare! if compare_fused || compare_gemm
  tile = ML::Metal::ComputePipeline.new("affine_budget_tile", AFFINE_BUDGET_SOURCE)
  zero = ML::Metal::ComputePipeline.new("affine_budget_zero", AFFINE_BUDGET_SOURCE)
  compare = ML::Metal::ComputePipeline.new("affine_budget_compare", AFFINE_BUDGET_SOURCE)
  bytes_compare = ML::Metal::ComputePipeline.new("affine_budget_bytes", AFFINE_BUDGET_SOURCE)
  corrupt = ML::Metal::ComputePipeline.new("affine_budget_corrupt", AFFINE_BUDGET_SOURCE)
  copy = ML::Metal::ComputePipeline.new("affine_budget_copy", AFFINE_BUDGET_SOURCE)
  poison = ML::Metal::ComputePipeline.new("affine_budget_poison", <<-METAL)
    #include <metal_stdlib>
    using namespace metal;
    kernel void affine_budget_poison(device float *dst [[buffer(0)]],
        constant uint &count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
      if (i < count) dst[i] = as_type<float>(0x7fc00000u);
    }
    METAL
  source = arena.alloc("source", 4 * din)
  input = arena.alloc("input", rows * din)
  original = arena.alloc("original", (weight_count.to_i64 // 2).to_i32)
  stage = arena.alloc("stage", weight_count)
  expected_weight = arena.alloc("expected_weight", weight_count)
  output = arena.alloc("output", output_count)
  expected_output = arena.alloc("expected_output", output_count)
  baseline_output = arena.alloc("baseline_output", output_count)
  bias = arena.alloc("bias", dout)
  status = arena.alloc("status", ML::Video::Q4KDeltaWeights::STATUS_WORDS)
  dummy_sidecar = arena.alloc("q4k_fused_dummy_sidecar", 1) if compare_fused
  status.contents.as(UInt32*).clear(ML::Video::Q4KDeltaWeights::STATUS_WORDS)
  payload = arena.alloc("canonical_q4k", (budget // 4).to_i32)
  sidecars = {} of String => ML::MetalBuffer
  variants.reject { |name| name == "q4_k" }.each do |name|
    bytes = ML::Video::Q4KDeltaWeights.sidecar_bytes(weight_count, patch_map[name])
    sidecars[name] = arena.alloc(name, ((bytes + 3) // 4).to_i32)
  end
  expected_sidecar = arena.alloc("expected_sidecar", (ML::Video::Q4KDeltaWeights.sidecar_bytes(weight_count, 8) // 4).to_i32)
  bf16_weight = arena.alloc("bf16_weight", weight_count)
  loss_metrics = arena.alloc("loss_metrics", 5)
  loss_partials = arena.alloc("loss_partials", ((weight_count + 255) // 256) * 4)
  metrics = arena.alloc("metrics", 5)
  weight_metrics = arena.alloc("weight_metrics", 5)
  partial_count = ((Math.max(weight_count, output_count) + 255) // 256) * 4
  partials = arena.alloc("partials", partial_count)
  weight_partials = arena.alloc("weight_partials", ((weight_count + 255) // 256) * 4)
  row_metrics = Array.new(4) { |r| arena.alloc("row_metrics_#{r}", 5) }
  row_partials = Array.new(4) { |r| arena.alloc("row_partials_#{r}", ((dout + 255) // 256) * 4) }
  staged_output = arena.alloc("staged_output", output_count) if compare_fused
  staged_metrics = arena.alloc("staged_metrics", 5) if compare_fused
  staged_partials = arena.alloc("staged_partials", ((output_count + 255) // 256) * 4) if compare_fused
  dense_native_metrics = arena.alloc("dense_native_metrics", 5) if compare_fused || compare_gemm
  dense_native_partials = arena.alloc("dense_native_partials", ((output_count + 255) // 256) * 4) if compare_fused || compare_gemm
  dense_mps_metrics = arena.alloc("dense_mps_metrics", 5) if compare_fused || compare_gemm
  dense_mps_partials = arena.alloc("dense_mps_partials", ((output_count + 255) // 256) * 4) if compare_fused || compare_gemm
  gemm_candidate_output = arena.alloc("gemm_candidate_output", output_count) if compare_gemm
  gemm_candidate_metrics = arena.alloc("gemm_candidate_metrics", 5) if compare_gemm
  gemm_candidate_partials = arena.alloc("gemm_candidate_partials", ((output_count + 255) // 256) * 4) if compare_gemm
  gemm_legacy_output = arena.alloc("gemm_legacy_output", output_count) if compare_gemm
  gemm_mps_output = arena.alloc("gemm_mps_output", output_count) if compare_gemm
  gemm_candidate_legacy_metrics = arena.alloc("gemm_candidate_legacy_metrics", 5) if compare_gemm
  gemm_candidate_legacy_partials = arena.alloc("gemm_candidate_legacy_partials", ((output_count + 255) // 256) * 4) if compare_gemm
  gemm_candidate_mps_metrics = arena.alloc("gemm_candidate_mps_metrics", 5) if compare_gemm
  gemm_candidate_mps_partials = arena.alloc("gemm_candidate_mps_partials", ((output_count + 255) // 256) * 4) if compare_gemm
  affine_budget_upload(input_path, source, source.size)
  affine_budget_upload(bias_path, bias, bias.size)
  affine_budget_upload(original_path, original, original.size)
  affine_budget_upload(baseline_projection_path, baseline_output, baseline_output.size)
  affine_budget_upload(File.join(fixture_dir, baseline["decoded_weight"]["path"].as_s), bf16_weight, bf16_weight.size)
  affine_budget_upload(File.join(fixture_dir, variant_manifest["q4_k"]["payload"]["path"].as_s), payload, budget)
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
    words = Slice.new(status.contents.as(UInt32*), ML::Video::Q4KDeltaWeights::STATUS_WORDS).to_a
    raise "Sticky codec failure: #{words}" unless words.all?(&.zero?)
    true
  }
  puts({schema: "wan_q4k_delta_setup_v1", manifest_sha256: manifest_hash,
        rows: rows, din: din, dout: dout, unique_source_rows: 4, real_video_derived: false,
        per_arm_storage: variants.to_h { |name|
          patches = patch_map[name]
          extra = patches == 0 ? 0_i64 : ML::Video::Q4KDeltaWeights.sidecar_bytes(weight_count, patches)
          total = budget + extra + 16
          {name, {base_payload_bytes: budget, sidecar_bytes: extra, descriptor_bytes: 16,
                  package_bytes: total, patches_per_block: patches,
                  correction_count: (weight_count.to_i64 // 256) * patches,
                  bits_per_weight: total * 8.0 / weight_count}}
        }, shared_bias_bytes: bias.size,
        f32_stage_bytes: stage.size, explicit_buffer_bytes_during_pack: arena.resident_bytes,
        packing_workspace_bytes: 0,
        reverse_first: reverse_first, variants: variants,
        block_validation_comparison: compare_block_validation, execution_arms: execution_arms,
        fused_comparison: compare_fused,
        format_by_arm: format_by_arm,
        fused_staged_control_decoder: compare_fused ? "block_validated" : nil,
        candidate_global_f32_weight_stage_bytes: compare_fused ? 0_i64 : nil,
        control_f32_weight_stage_bytes: compare_fused ? stage.size : nil,
        retained_harness_bytes_excluding_stage: compare_fused ? arena.resident_bytes - stage.size : nil,
        q4_k_fused_dummy_sidecar_bytes: compare_fused ? dummy_sidecar.not_nil!.size : nil,
        gemm_comparison: compare_gemm,
        gemm_variant: gemm_variant_id,
        gemm_variant_name: ML::Video::Q4KDeltaLinear.variant_name(gemm_variant),
        gemm_tile_shape: {tm: ML::Video::Q4KDeltaLinear.tile_m(gemm_variant),
                          tn: ML::Video::Q4KDeltaLinear.tile_n(gemm_variant),
                          tk: ML::Video::Q4KDeltaLinear.tile_k(gemm_variant)},
        gemm_threadgroup_threads: ML::Video::Q4KDeltaLinear.threadgroup_threads(gemm_variant),
        dense_static_threadgroup_memory_bytes: (compare_gemm || compare_fused) ?
          ML::Video::Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(gemm_variant) : nil,
        gemm_legacy_dense_static_threadgroup_memory_bytes: compare_gemm ?
          ML::Video::Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(
            ML::Video::Q4KDeltaLinear::GemmVariant::Legacy) : nil,
        gemm_candidate_dense_static_threadgroup_memory_bytes: compare_gemm ?
          ML::Video::Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(gemm_variant) : nil,
        gemm_legacy_dense_pipeline: compare_gemm ?
          ML::Video::Q4KDeltaLinear.dense_pipeline(
            ML::Video::Q4KDeltaLinear::GemmVariant::Legacy).name : nil,
        gemm_candidate_dense_pipeline: compare_gemm ?
          ML::Video::Q4KDeltaLinear.dense_pipeline(gemm_variant).name : nil,
        candidate_pipeline: compare_fused ? ML::Video::Q4KDeltaLinear.linear_pipeline(gemm_variant).name : nil,
        candidate_static_threadgroup_memory_bytes: compare_fused ?
          ML::Video::Q4KDeltaLinear.static_threadgroup_memory_bytes(gemm_variant) : nil,
        dense_mode_execution_arms: compare_gemm ? execution_arms : (compare_fused ? ["dense_native", "dense_mps"] : nil),
        dense_mode_weight_format: (compare_gemm || compare_fused) ? "q4_k" : nil,
        dense_mode_internal_tile_shape: (compare_gemm || compare_fused) ?
          {tm: ML::Video::Q4KDeltaLinear.tile_m(gemm_variant),
           tn: ML::Video::Q4KDeltaLinear.tile_n(gemm_variant),
           tk: ML::Video::Q4KDeltaLinear.tile_k(gemm_variant)} : nil,
        gemm_schedule: compare_gemm ? "six_sample_palindrome_opposite_base_direction_by_half_v1" : nil,
        gemm_schedule_directions: compare_gemm ?
          {first_half: reverse_first ? "reverse" : "forward",
           second_half: reverse_first ? "forward" : "reverse"} : nil,
        mps_storage: "opaque and included in the process guard, excluded from explicit byte sum",
        tolerance: "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)"}.to_json)
  STDOUT.flush

  # One-time GPU packing has its own terminal measurement; it is excluded from
  # resident-packed inference. Full byte qualification is a separate diagnostic.
  unless compare_gemm
  variants.reject { |v| v == "q4_k" }.each do |name|
    v = variant_manifest[name]
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    before = ML::MetalBuffer.video_spec_allocations
    started = Time.instant
    ML::Video::Q4KDeltaWeights.pack(encoder, original, payload, sidecars[name], status, weight_count, patch_map[name])
    encoder.end_encoding
    encode = (Time.instant - started).total_seconds
    gpu = cmd.commit_and_wait_gpu_elapsed_seconds
    wall = (Time.instant - started).total_seconds
    raise "GPU pack encoded an allocation" unless ML::MetalBuffer.video_spec_allocations == before
    check_status.call
    sidecar_bytes = ML::Video::Q4KDeltaWeights.sidecar_bytes(weight_count, patch_map[name])
    affine_budget_upload(File.join(fixture_dir, v["sidecar"]["path"].as_s), expected_sidecar, sidecar_bytes)
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    encoder.set_pipeline(zero).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
    encoder.set_pipeline(bytes_compare).set_buffer(sidecars[name], 0).set_buffer(expected_sidecar, 1)
      .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite).set_value((sidecar_bytes // 4).to_u32, 3)
      .dispatch_1d((sidecar_bytes // 4).to_i32)
    encoder.end_encoding
    cmd.commit
    cmd.wait
    differences = metrics.contents.as(UInt32*)[0]
    puts({schema: "wan_q4k_delta_packing_v1", variant: name, gpu_seconds: gpu,
          wall_seconds: wall, encode_seconds: encode, sidecar_word_differences: differences,
          encode_metal_buffer_allocations: 0, activation_readbacks: 0,
          terminal_waits: 1, byte_validation_outside_packing_interval: true}.to_json)
    STDOUT.flush
    raise "GPU packing differs from independent serialized oracle: #{name}" unless differences == 0
  end
  end

  sidecar_for = ->(format : String) {
    format == "q4_k" ? dummy_sidecar.not_nil! : sidecars[format]
  }
  decode = ->(encoder : ML::Metal::ComputeEncoder, name : String) {
    format = format_by_arm[name]
    if compare_fused && name.ends_with?("_fused")
      ML::Video::Q4KDeltaLinear.decode_probe(encoder, payload, sidecar_for.call(format), stage,
        status, weight_count, patch_map[format])
    elsif format == "q4_k"
      ML::Video::PackedWeightDecode.q4_k(encoder, payload, stage, weight_count)
    elsif compare_fused || name != format
      ML::Video::Q4KDeltaWeights.decode_block_validated(encoder, payload, sidecars[format], stage, status, weight_count, patch_map[format])
    else
      ML::Video::Q4KDeltaWeights.decode(encoder, payload, sidecars[format], stage, status, weight_count, patch_map[format])
    end
    nil
  }
  project = ->(encoder : ML::Metal::ComputeEncoder) {
    ML::Video::MetalLinearMPS.project_preexpanded(encoder, input, bias, output, rows,
      din, dout, ML::Video::WeightDType::F32, plan.not_nil!, stage)
    nil
  }
  fused_linear = ->(encoder : ML::Metal::ComputeEncoder, name : String) {
    format = format_by_arm[name]
    ML::Video::Q4KDeltaLinear.linear(encoder, input, payload, sidecar_for.call(format), bias,
      output, status, rows, din, dout, patch_map[format], gemm_variant)
    nil
  }
  poison_buffer = ->(encoder : ML::Metal::ComputeEncoder, buffer : ML::MetalBuffer, count : Int32) {
    encoder.set_pipeline(poison).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 1).dispatch_1d(count)
    nil
  }
  preserve_output = ->(encoder : ML::Metal::ComputeEncoder, destination : ML::MetalBuffer) {
    encoder.set_pipeline(copy).set_buffer(output, 0).set_buffer(destination, 1)
      .set_value(output_count.to_u32, 2).dispatch_1d(output_count)
    nil
  }
  unless compare_gemm
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
  puts({schema: "wan_q4k_delta_bf16_control_v1", decoder: bw, projection: bo}.to_json)
  end
  validate_dense = ->(phase : String) {
    q4k = variant_manifest["q4_k"]
    affine_budget_upload(File.join(fixture_dir, q4k["decoded_weight"]["path"].as_s), expected_weight, stage.size)
    affine_budget_upload(File.join(fixture_dir, q4k["expected_projection"]["path"].as_s), expected_output, output.size)
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    before = ML::MetalBuffer.video_spec_allocations
    poison_buffer.call(encoder, stage, weight_count)
    ML::Video::PackedWeightDecode.q4_k(encoder, payload, stage, weight_count)
    affine_budget_compare(encoder, zero, compare, stage, expected_weight, weight_metrics, weight_partials, weight_count)
    poison_buffer.call(encoder, output, output_count)
    ML::Video::Q4KDeltaLinear.dense_linear(encoder, input, stage, bias, output, rows, din, dout, gemm_variant)
    affine_budget_compare(encoder, zero, compare, output, expected_output,
      dense_native_metrics.not_nil!, dense_native_partials.not_nil!, output_count)
    preserve_output.call(encoder, staged_output.not_nil!)
    poison_buffer.call(encoder, output, output_count)
    project.call(encoder)
    affine_budget_compare(encoder, zero, compare, output, expected_output,
      dense_mps_metrics.not_nil!, dense_mps_partials.not_nil!, output_count)
    affine_budget_compare(encoder, zero, compare, output, staged_output.not_nil!,
      staged_metrics.not_nil!, staged_partials.not_nil!, output_count)
    encoder.end_encoding
    cmd.commit
    cmd.wait
    decoder_metric = affine_budget_metrics(weight_metrics, weight_partials, weight_count)
    native_metric = affine_budget_metrics(dense_native_metrics.not_nil!, dense_native_partials.not_nil!, output_count)
    mps_metric = affine_budget_metrics(dense_mps_metrics.not_nil!, dense_mps_partials.not_nil!, output_count)
    cross_metric = affine_budget_metrics(staged_metrics.not_nil!, staged_partials.not_nil!, output_count)
    check_status.call
    raise "Dense Q4_K decoder parity failed" unless decoder_metric[:tolerance_violations] == 0 && decoder_metric[:nonfinite] == 0
    {"dense_native" => native_metric, "dense_mps" => mps_metric, "dense_native_vs_dense_mps" => cross_metric}.each do |name, metric|
      raise "Dense Q4_K parity failed #{name}" unless metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
    end
    raise "Dense validation encoded an allocation" unless ML::MetalBuffer.video_spec_allocations == before
    {"dense_native", "dense_mps"}.each do |name|
      puts({schema: "wan_q4k_delta_dense_parity_v1", phase: phase, variant: name,
            format: "q4_k", reference_arm: "q4_k", decoder: decoder_metric,
            dense_native: native_metric, dense_mps: mps_metric,
            dense_native_vs_dense_mps: cross_metric}.to_json)
      STDOUT.flush
    end
    nil
  }
  validate_gemm = ->(phase : String) {
    q4k = variant_manifest["q4_k"]
    affine_budget_upload(File.join(fixture_dir, q4k["decoded_weight"]["path"].as_s), expected_weight, stage.size)
    affine_budget_upload(File.join(fixture_dir, q4k["expected_projection"]["path"].as_s), expected_output, output.size)
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    before = ML::MetalBuffer.video_spec_allocations
    poison_buffer.call(encoder, stage, weight_count)
    ML::Video::PackedWeightDecode.q4_k(encoder, payload, stage, weight_count)
    affine_budget_compare(encoder, zero, compare, stage, expected_weight, weight_metrics, weight_partials, weight_count)

    poison_buffer.call(encoder, output, output_count)
    ML::Video::Q4KDeltaLinear.dense_linear(encoder, input, stage, bias, output, rows, din, dout,
      ML::Video::Q4KDeltaLinear::GemmVariant::Legacy)
    affine_budget_compare(encoder, zero, compare, output, expected_output,
      dense_native_metrics.not_nil!, dense_native_partials.not_nil!, output_count)
    preserve_output.call(encoder, gemm_legacy_output.not_nil!)

    poison_buffer.call(encoder, output, output_count)
    ML::Video::Q4KDeltaLinear.dense_linear(encoder, input, stage, bias, output, rows, din, dout, gemm_variant)
    affine_budget_compare(encoder, zero, compare, output, expected_output,
      gemm_candidate_metrics.not_nil!, gemm_candidate_partials.not_nil!, output_count)
    preserve_output.call(encoder, gemm_candidate_output.not_nil!)

    poison_buffer.call(encoder, output, output_count)
    project.call(encoder)
    affine_budget_compare(encoder, zero, compare, output, expected_output,
      dense_mps_metrics.not_nil!, dense_mps_partials.not_nil!, output_count)
    preserve_output.call(encoder, gemm_mps_output.not_nil!)

    affine_budget_compare(encoder, zero, compare, gemm_candidate_output.not_nil!, gemm_legacy_output.not_nil!,
      gemm_candidate_legacy_metrics.not_nil!, gemm_candidate_legacy_partials.not_nil!, output_count)
    affine_budget_compare(encoder, zero, compare, gemm_candidate_output.not_nil!, gemm_mps_output.not_nil!,
      gemm_candidate_mps_metrics.not_nil!, gemm_candidate_mps_partials.not_nil!, output_count)
    encoder.end_encoding
    cmd.commit
    cmd.wait
    decoder_metric = affine_budget_metrics(weight_metrics, weight_partials, weight_count)
    legacy_metric = affine_budget_metrics(dense_native_metrics.not_nil!, dense_native_partials.not_nil!, output_count)
    candidate_metric = affine_budget_metrics(gemm_candidate_metrics.not_nil!, gemm_candidate_partials.not_nil!, output_count)
    mps_metric = affine_budget_metrics(dense_mps_metrics.not_nil!, dense_mps_partials.not_nil!, output_count)
    candidate_legacy_metric = affine_budget_metrics(gemm_candidate_legacy_metrics.not_nil!, gemm_candidate_legacy_partials.not_nil!, output_count)
    candidate_mps_metric = affine_budget_metrics(gemm_candidate_mps_metrics.not_nil!, gemm_candidate_mps_partials.not_nil!, output_count)
    check_status.call
    {"Q4_K decoder" => decoder_metric, "legacy GEMM" => legacy_metric,
     "candidate GEMM" => candidate_metric, "MPS GEMM" => mps_metric}.each do |name, metric|
      raise "Dense GEMM parity failed #{name}" unless metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
    end
    {"candidate/legacy" => candidate_legacy_metric, "candidate/MPS" => candidate_mps_metric}.each do |name, metric|
      raise "Dense GEMM pair comparison failed #{name}" unless metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
    end
    raise "Dense GEMM validation encoded an allocation" unless ML::MetalBuffer.video_spec_allocations == before
    puts({schema: "wan_q4k_delta_gemm_parity_v1", phase: phase, format: "q4_k",
          reference_arm: "q4_k", decoder: decoder_metric,
          dense_native_legacy: legacy_metric, dense_native_candidate: candidate_metric,
          dense_mps: mps_metric, candidate_vs_legacy: candidate_legacy_metric,
          candidate_vs_mps: candidate_mps_metric}.to_json)
    STDOUT.flush
    nil
  }
  validate = ->(phase : String) {
    execution_arms.each do |name|
      v = variant_manifest[format_by_arm[name]]
      fused_arm = compare_fused && name.ends_with?("_fused")
      affine_budget_upload(File.join(fixture_dir, v["decoded_weight"]["path"].as_s), expected_weight, stage.size)
      affine_budget_upload(File.join(fixture_dir, v["expected_projection"]["path"].as_s), expected_output, output.size)
      cmd = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(cmd)
      before = ML::MetalBuffer.video_spec_allocations
      poison_buffer.call(encoder, stage, weight_count) if compare_fused
      decode.call(encoder, name)
      affine_budget_compare(encoder, zero, compare, stage, expected_weight, weight_metrics, weight_partials, weight_count)
      affine_budget_compare(encoder, zero, compare, stage, bf16_weight, loss_metrics, loss_partials, weight_count)
      if fused_arm
        poison_buffer.call(encoder, output, output_count)
        fused_linear.call(encoder, name)
      else
        poison_buffer.call(encoder, output, output_count) if compare_fused
        project.call(encoder)
      end
      affine_budget_compare(encoder, zero, compare, output, expected_output, metrics, partials, output_count)
      if fused_arm
        affine_budget_compare(encoder, zero, compare, output, staged_output.not_nil!,
          staged_metrics.not_nil!, staged_partials.not_nil!, output_count)
      elsif compare_fused
        preserve_output.call(encoder, staged_output.not_nil!)
      end
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
      weight_loss = affine_budget_metrics(loss_metrics, loss_partials, weight_count)
      raise "Nonfinite BF16 weight distortion" unless weight_loss[:nonfinite] == 0
      fused_staged = affine_budget_metrics(staged_metrics.not_nil!, staged_partials.not_nil!, output_count) if fused_arm
      loss = Array.new(4) { |r| affine_budget_metrics(row_metrics[r], row_partials[r], dout) }
      raise "Validation encoded an allocation" unless ML::MetalBuffer.video_spec_allocations == before
      if fused_arm
        puts({schema: "wan_q4k_delta_parity_v1", phase: phase, variant: name,
              decoder: wm, projection: om, fused_vs_staged: fused_staged,
              lossy_weight_vs_bf16: weight_loss, lossy_vs_bf16_per_source_row: loss,
              calibration_rows: [] of Int32,
              unique_evaluation_rows: [0, 1, 2, 3]}.to_json)
      else
        puts({schema: "wan_q4k_delta_parity_v1", phase: phase, variant: name,
              decoder: wm, projection: om, lossy_weight_vs_bf16: weight_loss,
              lossy_vs_bf16_per_source_row: loss, calibration_rows: [] of Int32,
              unique_evaluation_rows: [0, 1, 2, 3]}.to_json)
      end
      STDOUT.flush
      raise "Weight parity failed #{name}: #{wm}" unless wm[:tolerance_violations] == 0 && wm[:nonfinite] == 0
      raise "Projection parity failed #{name}: #{om}" unless om[:tolerance_violations] == 0 && om[:nonfinite] == 0
      if fused_staged
        raise "Fused/staged output parity failed #{name}: #{fused_staged}" unless fused_staged[:tolerance_violations] == 0 && fused_staged[:nonfinite] == 0
      end
      if compare_block_validation && (wm[:bit_differences] != 0 || om[:bit_differences] != 0)
        raise "Validation placement changed exact fixture output: #{name}"
      end
    end
    nil
  }
  prepare_dense_weight = -> {
    cmd = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(cmd)
    poison_buffer.call(encoder, stage, weight_count)
    ML::Video::PackedWeightDecode.q4_k(encoder, payload, stage, weight_count)
    encoder.end_encoding
    cmd.commit
    cmd.wait
    check_status.call
    nil
  }
  if compare_gemm
    validate_gemm.call("before_timing")
    prepare_dense_weight.call
    gemm_methods = ["dense_native_legacy", "dense_native_candidate", "dense_mps"]
    (-2...rounds).each do |round|
      rotation = Math.max(round, 0) % gemm_methods.size
      rotated = gemm_methods.rotate(rotation)
      reverse_phase = round < 0 ? round == -1 : round >= rounds // 2
      reverse_direction = reverse_phase != reverse_first
      rotated = [rotated[0], rotated[2], rotated[1]] if reverse_direction
      order = rotated + rotated.reverse
      order.each_with_index do |name, position|
        cmd = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(cmd)
        before = ML::MetalBuffer.video_spec_allocations
        started = Time.instant
        case name
        when "dense_native_legacy"
          ML::Video::Q4KDeltaLinear.dense_linear(encoder, input, stage, bias, output,
            rows, din, dout, ML::Video::Q4KDeltaLinear::GemmVariant::Legacy)
        when "dense_native_candidate"
          ML::Video::Q4KDeltaLinear.dense_linear(encoder, input, stage, bias, output,
            rows, din, dout, gemm_variant)
        else
          project.call(encoder)
        end
        encoder.end_encoding
        encode = (Time.instant - started).total_seconds
        gpu = cmd.commit_and_wait_gpu_elapsed_seconds
        wall = (Time.instant - started).total_seconds
        raise "Timed dense GEMM encode allocated a MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
        puts({schema: "wan_q4k_delta_timing_v1", mode: "dense_gemm", variant: name,
              round: round, position: position, warmup: round < 0, rows: rows,
              encode_seconds: encode, gpu_seconds: gpu, wall_seconds: wall,
              terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
              encode_metal_buffer_allocations: 0, manifest_sha256: manifest_hash}.to_json)
        STDOUT.flush
      end
    end
    validate_gemm.call("after_timing")
  elsif compare_fused
    validate_dense.call("before_timing")
    validate.call("before_timing")
    prepare_dense_weight.call

    # The dense control uses the same Q4_K-decoded coefficients and geometry as
    # fused_projection. Decode and parity checks are outside each timed sample.
    (-2...rounds).each do |round|
      if round < 0
        dense_order = round == -2 ? ["dense_native", "dense_mps"] : ["dense_mps", "dense_native"]
        dense_order = dense_order.reverse if reverse_first
      else
        abba = (round < rounds // 2) != reverse_first
        dense_order = abba ? ["dense_native", "dense_mps", "dense_mps", "dense_native"] : ["dense_mps", "dense_native", "dense_native", "dense_mps"]
      end
      dense_order.each_with_index do |name, position|
        cmd = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(cmd)
        before = ML::MetalBuffer.video_spec_allocations
        started = Time.instant
        if name == "dense_native"
          ML::Video::Q4KDeltaLinear.dense_linear(encoder, input, stage, bias, output, rows, din, dout, gemm_variant)
        else
          project.call(encoder)
        end
        encoder.end_encoding
        encode = (Time.instant - started).total_seconds
        gpu = cmd.commit_and_wait_gpu_elapsed_seconds
        wall = (Time.instant - started).total_seconds
        raise "Timed dense encode allocated a MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
        check_status.call
        puts({schema: "wan_q4k_delta_timing_v1", mode: "dense_mode", variant: name,
              round: round, position: position, warmup: round < 0, rows: rows,
              encode_seconds: encode, gpu_seconds: gpu, wall_seconds: wall,
              terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
              encode_metal_buffer_allocations: 0, manifest_sha256: manifest_hash}.to_json)
        STDOUT.flush
      end
    end

    (-2...rounds).each do |round|
      order = variants.rotate(Math.max(round, 0) % variants.size).flat_map do |name|
        [name, "#{name}_fused"]
      end
      swap_pair = round >= 0 && (round // variants.size).odd?
      order = order.each_slice(2).flat_map { |pair| swap_pair ? pair.reverse : pair }.to_a
      order = order.reverse if (round == -1 || round >= rounds // 2) != reverse_first
      order.each_with_index do |name, position|
        cmd = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(cmd)
        before = ML::MetalBuffer.video_spec_allocations
        started = Time.instant
        if name.ends_with?("_fused")
          ML::Video::Q4KDeltaLinear.linear(encoder, input, payload,
            sidecar_for.call(format_by_arm[name]), bias, output, status, rows, din, dout,
            patch_map[format_by_arm[name]], gemm_variant)
        else
          decode.call(encoder, name)
          project.call(encoder)
        end
        encoder.end_encoding
        encode = (Time.instant - started).total_seconds
        gpu = cmd.commit_and_wait_gpu_elapsed_seconds
        wall = (Time.instant - started).total_seconds
        raise "Timed fused encode allocated a MetalBuffer" unless ML::MetalBuffer.video_spec_allocations == before
        check_status.call
        puts({schema: "wan_q4k_delta_timing_v1", mode: "fused_projection", variant: name,
              round: round, position: position, warmup: round < 0, rows: rows,
              encode_seconds: encode, gpu_seconds: gpu, wall_seconds: wall,
              terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
              encode_metal_buffer_allocations: 0, manifest_sha256: manifest_hash}.to_json)
        STDOUT.flush
      end
    end
    validate_dense.call("after_timing")
    validate.call("after_timing")
  else
    validate.call("before_timing")
    {"projection", "decode_only"}.each do |mode|
      (-2...rounds).each do |round|
        order = variants.rotate(Math.max(round, 0) % variants.size)
        if compare_block_validation
          swap_pair = round >= 0 && (round // variants.size).odd?
          order = order.flat_map do |name|
            pair = [name, "#{name}_block"]
            swap_pair ? pair.reverse : pair
          end
        end
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
          puts({schema: "wan_q4k_delta_timing_v1", mode: mode, variant: name,
                round: round, position: position, warmup: round < 0, rows: rows,
                encode_seconds: encode, gpu_seconds: gpu, wall_seconds: wall,
                terminal_waits: 1, inter_operator_host_waits: 0, activation_readbacks: 0,
                encode_metal_buffer_allocations: 0, manifest_sha256: manifest_hash}.to_json)
          STDOUT.flush
        end
      end
    end
    validate.call("after_timing")
  end
  cmd = ML::Metal::CommandBuffer.new
  e = ML::Metal::ComputeEncoder.new(cmd)
  e.set_pipeline(corrupt).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
  affine_budget_compare(e, zero, compare, output, expected_output, metrics, partials, output_count)
  affine_budget_compare(e, zero, compare, output, staged_output.not_nil!,
    staged_metrics.not_nil!, staged_partials.not_nil!, output_count) if compare_fused
  e.end_encoding
  cmd.commit
  cmd.wait
  corruption = affine_budget_metrics(metrics, partials, output_count)
  raise "Comparator accepted finite/NaN corruption" unless corruption[:tolerance_violations] == 2 && corruption[:nonfinite] == 1
  control = {schema: "wan_q4k_delta_controls_v1", corruption_rejected: true,
             corruption: corruption, final_status_zero: check_status.call}
  if compare_fused
    staged_corruption = affine_budget_metrics(staged_metrics.not_nil!, staged_partials.not_nil!, output_count)
    raise "Comparator accepted fused/staged finite/NaN corruption" unless staged_corruption[:tolerance_violations] == 2 && staged_corruption[:nonfinite] == 1
    control = control.merge({fused_vs_staged_corruption_rejected: true,
                             fused_vs_staged_corruption:          staged_corruption})
  end
  puts(control.to_json)
ensure
  plan.try(&.release)
  arena.release
end
