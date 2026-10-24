# Based on Tencent Hy-Nomas at 911cea6458a89c2663513ff7099561ec190dd491.
# Copyright (C) 2026 Tencent. Apache-2.0. Modified for native Metal validation.
require "spec"
require "json"
require "digest/sha256"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_block"

private def hy_config : ML::Video::HyNomasConfig
  ML::Video::HyNomasConfig.new(
    batches: 1, video_tokens: 6, text_tokens: 3,
    dim: 64, heads: 2, head_dim: 32, ffn_dim: 256,
  )
end

private def hy_floats(value : JSON::Any) : Array(Float32)
  values = value.as_h? ? value["values"] : value
  values.as_a.map(&.as_f.to_f32)
end

private def hy_sha256(path : String) : String
  digest = Digest::SHA256.new
  bytes = Bytes.new(1024 * 1024)
  File.open(path, "rb") do |file|
    while (count = file.read(bytes)) > 0
      digest.update(bytes[0, count])
    end
  end
  digest.final.hexstring
end

# The expected digest must come from the trusted fixture-generation flow, not
# from mutable metadata inside the fixture itself. Parse the exact hashed bytes.
private def hy_load_fixture(path : String, expected_sha256 : String) : JSON::Any
  unless expected_sha256.matches?(/\A[0-9a-f]{64}\z/)
    raise ArgumentError.new("Hy-Nomas fixture requires a trusted SHA-256 digest")
  end
  contents = File.read(path)
  unless Digest::SHA256.hexdigest(contents) == expected_sha256
    raise ArgumentError.new("Hy-Nomas fixture SHA-256 mismatch")
  end
  JSON.parse(contents)
end

private def hy_case_config(fixture : JSON::Any, data : JSON::Any) : ML::Video::HyNomasConfig
  values = fixture["config"].as_h.dup
  data["config_overrides"]?.try(&.as_h.each { |key, value| values[key] = value })
  ML::Video::HyNomasConfig.new(
    batches: values["batches"].as_i,
    video_tokens: values["video_tokens"].as_i,
    text_tokens: values["text_tokens"].as_i,
    dim: values["dim"].as_i,
    heads: values["heads"].as_i,
    head_dim: values["head_dim"].as_i,
    ffn_dim: values["ffn_dim"].as_i,
    img_modulation_rows: values["img_modulation_rows"]?.try(&.as_i) || 0,
    txt_modulation_rows: values["txt_modulation_rows"]?.try(&.as_i) || 0,
  )
end

# This gate rejects nonfinite values even when a NaN comparison might otherwise
# appear to have no positive excess. A changed tolerance is not a valid repair.
private def hy_matches(actual : Array(Float32), reference : Array(Float32)) : Bool
  return false unless actual.size == reference.size
  actual.each_with_index do |value, index|
    target = reference[index]
    return false unless value.finite? && target.finite?
    return false unless (value - target).abs <= 2e-5_f32 + 2e-5_f32 * target.abs
  end
  true
end

private def hy_compare(actual : Array(Float32), expected : JSON::Any, label : String,
                       failures : Array(String), reports : Array(String)) : Nil
  reference = hy_floats(expected)
  violations = 0
  max_abs = 0.0_f32
  max_ratio = 0.0_f32
  Math.min(actual.size, reference.size).times do |index|
    value = actual[index]
    target = reference[index]
    unless value.finite? && target.finite?
      violations += 1
      next
    end
    delta = (value - target).abs
    tolerance = 2e-5_f32 + 2e-5_f32 * target.abs
    max_abs = Math.max(max_abs, delta)
    max_ratio = Math.max(max_ratio, delta / tolerance)
    violations += 1 if delta > tolerance
  end
  passed = actual.size == reference.size && violations == 0
  detail = "#{label} n=#{actual.size}/#{reference.size} violations=#{violations} max_abs=#{max_abs} max_gate_ratio=#{max_ratio}"
  reports << detail
  failures << detail unless passed
end

private def hy_run_variant(fixture : JSON::Any, data : JSON::Any,
                           variant : JSON::Any, fixture_path : String) : Nil
  config = hy_case_config(fixture, data)
  manifest = variant["weights_file"]
  sidecar = hy_sidecar(fixture_path, manifest)
  source_dtype = ML::GGUF::SafeTensorDType.parse(manifest["dtype"].as_s)
  resident_dtype = source_dtype.bf16? ? ML::Video::WeightDType::BF16 : ML::Video::WeightDType::F32
  bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config, source_dtype)
  variant["compute_dtype"].as_s.should eq("F32")
  fixture_bindings = variant["checkpoint_bindings"].as_a
  fixture_bindings.map { |item| item["source"].as_s }.sort.should eq(bindings.map(&.source).sort)
  fixture_bindings.each do |item|
    binding = bindings.find! { |entry| entry.source == item["source"].as_s }
    item["block_relative"].as_s.should eq(binding.source.lchop("double_blocks.0."))
    item["shape"].as_a.map(&.as_i64).should eq(binding.shape)
    ML::GGUF::SafeTensorDType.parse(item["dtype"].as_s).should eq(binding.dtype)
  end

  file = ML::Video::CheckpointFile.new(sidecar)
  arenas = [] of ML::Video::Arena
  prepared = nil.as(ML::Video::PreparedWeights?)
  begin
    allocation_count = ML::MetalBuffer.video_spec_allocations
    expect_raises(ArgumentError, /memory budget/) do
      ML::Video::HyNomasDualStreamBlock.weight_plan([file], config, 1_i64, resident_dtype: resident_dtype, require_all: true)
    end
    wrong_bindings = bindings.dup
    wrong_bindings[0] = bindings[0].copy_with(shape: [1_i64])
    expect_raises(ArgumentError, /shape mismatch/) do
      ML::Video::WeightPlan.new([file], wrong_bindings, 2_i64 * File.size(sidecar), resident_dtype: resident_dtype)
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocation_count),
      "invalid checkpoint admission must precede device allocation"

    inputs = ML::Video::Arena.new
    arenas << inputs
    weights = ML::Video::Arena.new
    arenas << weights
    first_scratch = ML::Video::Arena.new
    arenas << first_scratch
    chain_scratch = ML::Video::Arena.new
    arenas << chain_scratch
    perturbed_scratch = ML::Video::Arena.new
    arenas << perturbed_scratch
    data["inputs"].as_h.each { |name, values| inputs.upload(name, hy_floats(values)) }
    inputs.upload("img_perturbed", hy_floats(variant["video_perturbation"]["input_img"]))
    {first_scratch, chain_scratch, perturbed_scratch}.each do |scratch|
      ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
    end
    reserve = inputs.resident_bytes + first_scratch.resident_bytes + chain_scratch.resident_bytes + perturbed_scratch.resident_bytes
    plan = ML::Video::HyNomasDualStreamBlock.weight_plan(
      [file], config, 2_i64 * File.size(sidecar) + reserve + 1024_i64 * 1024,
      reserve_bytes: reserve, resident_dtype: resident_dtype, require_all: true,
    )
    plan.destination_bytes.should eq(bindings.sum { |binding| binding.shape.product * resident_dtype.bytes_per_element })
    prepared = ML::Video::PreparedWeights.new(plan, weights)
    first = ML::Video::HyNomasDualStreamBlock.new(config, weights, first_scratch)
    chain = ML::Video::HyNomasDualStreamBlock.new(config, weights, chain_scratch)
    perturbed = ML::Video::HyNomasDualStreamBlock.new(config, weights, perturbed_scratch)
    {variant["checkpoints"], variant["chain_expected"]["checkpoints"], variant["video_perturbation"]["checkpoints"]}.each do |expected|
      expected.as_h.keys.sort.should eq(first.checkpoints.keys.sort),
        "every named intermediate must have an upstream reference"
    end

    allocations_before_encode = ML::MetalBuffer.video_spec_allocations
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    prepared.not_nil!.encode(encoder)
    expect_raises(ArgumentError, /alias/) do
      first.encode(encoder, first.image_output, inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
    end
    first.encode(encoder, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
    chain.encode(encoder, first.image_output, first.text_output, inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
    # The second first-block encode must overwrite all scratch deterministically.
    first.encode(encoder, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
    perturbed.encode(encoder, inputs["img_perturbed"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
    encoder.end_encoding
    ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_encode),
      "encode must not allocate even a temporary MetalBuffer"
    gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
    # All activation reads occur after this one existing terminal wait.
    snapshots = hy_snapshots(first)
    chain_snapshots = hy_snapshots(chain)
    perturbed_snapshots = hy_snapshots(perturbed)
    failures = [] of String
    reports = [] of String
    {"first"     => {snapshots, variant["checkpoints"]},
     "chain"     => {chain_snapshots, variant["chain_expected"]["checkpoints"]},
     "perturbed" => {perturbed_snapshots, variant["video_perturbation"]["checkpoints"]}}.each do |group, pair|
      values, expected = pair
      expected.as_h.each do |name, reference|
        hy_compare(values[name], reference, "#{data["name"]}/#{manifest["dtype"]}/#{group}/#{name}", failures, reports)
      end
    end
    summary = "Hy-Nomas #{data["name"]}/#{manifest["dtype"]}: checkpoints=#{reports.size} failures=#{failures.size} resident_weights_bytes=#{plan.destination_bytes} scratch_bytes=#{reserve - inputs.resident_bytes} encode_metal_buffer_allocations=#{ML::MetalBuffer.video_spec_allocations - allocations_before_encode} gpu_seconds_four_encodes_plus_weight_decode=#{gpu_seconds} reference_kind=upstream_block_f32_math_mps_sdpa"
    STDERR.puts summary
    if report_path = ENV["HY_NOMAS_REPORT"]?
      File.open(report_path, "a") do |report|
        report.puts summary
        reports.each { |line| report.puts line }
      end
    end
    failures.empty?.should be_true, failures.join("\n")
    # A video perturbation must propagate through joint attention into text.
    {"img.output", "txt.output"}.each do |name|
      snapshots[name].zip(perturbed_snapshots[name]).any? { |left, right| (left - right).abs > 1e-5_f32 }.should be_true,
        "the fixture must exercise #{name} rather than an identity/no-op"
    end
    hy_matches(snapshots["img.output"], chain_snapshots["img.output"]).should be_false,
      "the chained block must consume the first block's resident output"
  ensure
    prepared.try(&.release)
    file.close
    arenas.reverse_each(&.release)
  end
end

private def hy_snapshots(block : ML::Video::HyNomasDualStreamBlock) : Hash(String, Array(Float32))
  block.checkpoints.transform_values do |buffer|
    buffer.read((buffer.size // sizeof(Float32)).to_i).map(&.to_f32)
  end
end

private def hy_sidecar(fixture_path : String, manifest : JSON::Any) : String
  relative = manifest["path"].as_s
  (File.basename(relative) == relative && relative != "." && relative != "..").should be_true,
    "sidecar must be a simple filename beside the fixture"
  path = File.join(File.dirname(File.expand_path(fixture_path)), relative)
  File.size(path).should eq(manifest["bytes"].as_i64)
  hy_sha256(path).should eq(manifest["sha256"].as_s)
  path
end

describe ML::Video::HyNomasDualStreamBlock do
  it "rejects unsupported model modes and invalid geometries before device allocation" do
    config = hy_config
    invalid = [
      config.copy_with(batches: 0),
      config.copy_with(video_tokens: -1),
      config.copy_with(text_tokens: 0),
      config.copy_with(dim: 63),
      config.copy_with(heads: -2, head_dim: -32),
      config.copy_with(dim: 66, head_dim: 33, ffn_dim: 264),
      config.copy_with(dim: 516, head_dim: 258, ffn_dim: 2064),
      config.copy_with(ffn_dim: 255),
      config.copy_with(batches: Int32::MAX),
      config.copy_with(img_modulation_rows: 2),
      config.copy_with(txt_modulation_rows: 2),
      config.copy_with(segment_count: 2),
      config.copy_with(chunk_size_tokens: 1),
      config.copy_with(history_tokens: 1),
      config.copy_with(has_mask: true),
      config.copy_with(has_camera: true),
      config.copy_with(action_windows: true),
      config.copy_with(norm_eps: 0.0_f32),
      config.copy_with(norm_eps: Float32::NAN),
      config.copy_with(norm_eps: 1e-5_f32),
      config.copy_with(norm_eps: Float32::INFINITY),
    ]
    allocations = ML::MetalBuffer.video_spec_allocations
    config.validate!
    invalid.each do |item|
      expect_raises(ArgumentError) { item.validate! }
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "binds exactly the pinned block weights and rejects unsupported checkpoint dtypes" do
    dtype = ML::GGUF::SafeTensorDType::BF16
    bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(hy_config, dtype, 7)
    bindings.size.should eq(32)
    bindings.map(&.source).uniq.size.should eq(32)
    bindings.map(&.target).uniq.size.should eq(32)
    bindings.all? { |binding| binding.source.starts_with?("double_blocks.7.") && binding.dtype == dtype }.should be_true
    bindings.find!(&.target.==("img_mod_weight")).shape.should eq([384_i64, 64_i64])
    bindings.find!(&.target.==("txt_attn_q_norm_weight")).shape.should eq([32_i64])
    bindings.find!(&.target.==("txt_mlp_fc2_weight")).shape.should eq([64_i64, 256_i64])
    expect_raises(ArgumentError) do
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(hy_config, ML::GGUF::SafeTensorDType::F16)
    end
    expect_raises(ArgumentError) do
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(hy_config, block_index: -1)
    end
  end

  it "rejects an unsupported on-disk dtype through model admission before device allocation" do
    header = %({"double_blocks.0.img_mod.linear.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}})
    file = File.tempfile("hy-nomas-unsupported", ".safetensors")
    checkpoint = nil.as(ML::Video::CheckpointFile?)
    allocations = ML::MetalBuffer.video_spec_allocations
    begin
      file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      file << header
      file.write(Bytes[0, 0])
      file.flush
      checkpoint = ML::Video::CheckpointFile.new(file.path)
      expect_raises(ArgumentError, /must be F32 or BF16/) do
        ML::Video::HyNomasDualStreamBlock.weight_plan([checkpoint.not_nil!], hy_config, 1_000_000_i64)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      checkpoint.try(&.close)
      path = file.path
      file.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "keeps the fixed comparison sensitive to corruption and nonfinite values" do
    hy_matches([0.25_f32], [0.25_f32]).should be_true
    hy_matches([0.25_f32], [0.26_f32]).should be_false
    hy_matches([Float32::NAN], [0.25_f32]).should be_false
    hy_matches([0.25_f32], [Float32::NAN]).should be_false
    hy_matches([Float32::INFINITY], [Float32::INFINITY]).should be_false
    hy_matches([0.25_f32], [] of Float32).should be_false
  end

  it "rejects changed fixture bytes against the trusted generation digest before device allocation" do
    file = File.tempfile("hy-nomas-fixture", ".json")
    allocations = ML::MetalBuffer.video_spec_allocations
    begin
      contents = %({"checkpoints":[0.25]})
      file << contents
      file.flush
      digest = Digest::SHA256.hexdigest(contents)
      hy_load_fixture(file.path, digest)["checkpoints"][0].as_f.should eq(0.25)
      File.write(file.path, %({"checkpoints":[0.26]}))
      expect_raises(ArgumentError, /SHA-256 mismatch/) { hy_load_fixture(file.path, digest) }
      expect_raises(ArgumentError, /trusted SHA-256/) { hy_load_fixture(file.path, "") }
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      path = file.path
      file.close
      File.delete(path) if File.exists?(path)
    end
  end

  it "matches the declared upstream F32 math control, resident chaining and perturbation without encode allocations" do
    fixture_path = ENV["HY_NOMAS_BLOCK_FIXTURE"]?
    pending "Set HY_NOMAS_BLOCK_FIXTURE to the external pinned MPS oracle JSON" unless fixture_path
    fixture = hy_load_fixture(fixture_path.not_nil!, ENV["HY_NOMAS_FIXTURE_SHA256"]? || "")
    fixture["schema_version"].as_i.should eq(1)
    fixture["provenance"]["revision"].as_s.should eq("911cea6458a89c2663513ff7099561ec190dd491")
    scope = fixture["scope"]
    scope["reference_kind"].as_s.should eq("upstream_block_f32_math_mps_sdpa")
    scope["pinned_magi_runtime_numeric_parity"].as_bool.should be_false
    scope["segment_mode"].as_s.should eq("single_full")
    scope["text_mask"].as_s.should eq("none_all_valid")
    scope["history"].as_s.should eq("none")
    scope["camera"].as_s.should eq("unsupported_absent")
    scope["chunk_size_tokens"].as_i.should eq(0)
    cases = fixture["cases"].as_a
    cases.map { |data| data["name"].as_s }.sort.should eq(["broadcast", "per_token"])
    cases.each do |data|
      data["variants"].as_h.keys.sort.should eq(["bf16_weights_roundtrip_f32_compute", "f32"])
      data["variants"].as_h.each_value do |variant|
        hy_run_variant(fixture, data, variant, fixture_path.not_nil!)
      end
    end
  end
end
