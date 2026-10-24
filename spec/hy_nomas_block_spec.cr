# Based on Tencent Hy-Nomas at 911cea6458a89c2663513ff7099561ec190dd491.
# Copyright (C) 2026 Tencent. Apache-2.0. Modified for native Metal validation.
require "spec"
require "json"
require "digest/sha256"
require "./support/video_allocation_counter"
require "./support/hy_nomas_norm_controls"
require "./support/hy_nomas_modulation_controls"
require "./support/hy_nomas_activation_controls"
require "../src/ml/video/hy_nomas_block_mps"
require "../src/ml/video/hy_nomas_serial_mps"

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

private def hy_fixture_variant_labels(fixture : JSON::Any) : Array(String)
  unless fixture["schema_version"].as_i == 2
    raise ArgumentError.new("Hy-Nomas fixture variant selection requires schema version 2")
  end
  cases = fixture["cases"].as_a
  case_names = cases.map { |data| data["name"].as_s }.sort
  unless case_names == ["broadcast", "per_token"]
    raise ArgumentError.new("Hy-Nomas fixture cases must be broadcast and per_token")
  end

  labels = [] of String
  cases.each do |data|
    variants = data["variants"].as_h
    unless variants.keys.sort == ["bf16_weights_roundtrip_f32_compute", "f32"]
      raise ArgumentError.new("Hy-Nomas fixture variants must be f32 and bf16_weights_roundtrip_f32_compute")
    end
    variants.each do |variant_name, variant|
      expected_dtype = variant_name == "f32" ? "F32" : "BF16"
      resident_dtype = variant["resident_dtype"].as_s
      unless resident_dtype == expected_dtype
        raise ArgumentError.new("Hy-Nomas fixture variant #{variant_name} must use resident dtype #{expected_dtype}")
      end
      labels << "#{data["name"].as_s}/#{resident_dtype}"
    end
  end
  labels
end

private def hy_validate_fixture_variant_selector!(fixture : JSON::Any, selector : String) : Nil
  unless selector.matches?(/\A(?:broadcast|per_token)\/(?:F32|BF16)\z/)
    raise ArgumentError.new("HY_NOMAS_FIXTURE_VARIANT must be case/resident_dtype, for example broadcast/BF16")
  end
  matches = hy_fixture_variant_labels(fixture).count(&.== selector)
  unless matches == 1
    raise ArgumentError.new("HY_NOMAS_FIXTURE_VARIANT #{selector} does not identify exactly one fixture variant")
  end
end

private def hy_preflight_fixture_variant_selector! : JSON::Any?
  selector = ENV["HY_NOMAS_FIXTURE_VARIANT"]?
  return nil unless selector
  if selector.empty?
    raise ArgumentError.new("HY_NOMAS_FIXTURE_VARIANT must not be empty")
  end
  unless selector.matches?(/\A(?:broadcast|per_token)\/(?:F32|BF16)\z/)
    raise ArgumentError.new("HY_NOMAS_FIXTURE_VARIANT must be case/resident_dtype, for example broadcast/BF16")
  end
  fixture_path = ENV["HY_NOMAS_BLOCK_FIXTURE"]?
  unless fixture_path
    raise ArgumentError.new("HY_NOMAS_BLOCK_FIXTURE is required when HY_NOMAS_FIXTURE_VARIANT is set")
  end
  fixture = hy_load_fixture(fixture_path, ENV["HY_NOMAS_FIXTURE_SHA256"]? || "")
  hy_validate_fixture_variant_selector!(fixture, selector)
  fixture
end

# The generic oracle also supports synthetic weights. An actual-checkpoint
# claim must independently select that mode and bind its exact source digest.
private def hy_require_checkpoint_source!(fixture : JSON::Any, expected_sha256 : String) : Nil
  unless expected_sha256.matches?(/\A[0-9a-f]{64}\z/)
    raise ArgumentError.new("Actual checkpoint mode requires a trusted source SHA-256")
  end
  unless fixture["weights_are_synthetic"].as_bool == false && fixture["source"].as_h?
    raise ArgumentError.new("Actual checkpoint mode rejects synthetic or missing source provenance")
  end
  source = fixture["source"]
  unless source["weights_are_synthetic"].as_bool == false &&
         source["file_sha256"].as_s == expected_sha256 &&
         source["format"].as_s == "safetensors" &&
         {"F32", "BF16"}.includes?(source["source_dtype"].as_s) &&
         source["tensor_prefix"].as_s == "double_blocks.0." &&
         source["tensor_count"].as_i == 32
    raise ArgumentError.new("Actual checkpoint source provenance differs from the trusted block")
  end
  fixture["cases"].as_a.each do |data|
    data["variants"].as_h.each_value do |variant|
      unless variant["weights_are_synthetic"].as_bool == false && variant["source_checkpoint"] == source
        raise ArgumentError.new("Variant is not bound to the trusted actual checkpoint source")
      end
    end
  end
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
    attention_precision: hy_attention_precision(fixture["scope"]["attention_precision"].as_s),
  )
end

private def hy_attention_precision(value : String) : ML::Video::HyNomasAttentionPrecision
  case value
  when "f32"             then ML::Video::HyNomasAttentionPrecision::F32Control
  when "bf16-boundaries" then ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl
  else                        raise ArgumentError.new("Unknown Hy-Nomas fixture attention precision")
  end
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
  bitwise_mismatches = 0
  first_bit_difference = nil.as(Int32?)
  first_violation = nil.as(String?)
  Math.min(actual.size, reference.size).times do |index|
    value = actual[index]
    target = reference[index]
    if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
      bitwise_mismatches += 1
      first_bit_difference ||= index
    end
    unless value.finite? && target.finite?
      violations += 1
      first_violation ||= " first_index=#{index} actual=#{value} reference=#{target}"
      next
    end
    delta = (value - target).abs
    tolerance = 2e-5_f32 + 2e-5_f32 * target.abs
    max_abs = Math.max(max_abs, delta)
    max_ratio = Math.max(max_ratio, delta / tolerance)
    if delta > tolerance
      violations += 1
      first_violation ||= " first_index=#{index} actual=#{value} reference=#{target} actual_bits=#{value.unsafe_as(UInt32).to_s(16)} reference_bits=#{target.unsafe_as(UInt32).to_s(16)}"
    end
  end
  passed = actual.size == reference.size && violations == 0
  detail = "#{label} n=#{actual.size}/#{reference.size} violations=#{violations} max_abs=#{max_abs} max_gate_ratio=#{max_ratio} bitwise_mismatches=#{bitwise_mismatches} first_bit_index=#{first_bit_difference}#{first_violation}"
  reports << detail
  failures << detail unless passed
end

private def hy_compare_arrays(actual : Array(Float32), reference : Array(Float32), label : String,
                              failures : Array(String), reports : Array(String)) : Nil
  violations = 0
  max_abs = 0.0_f32
  max_ratio = 0.0_f32
  bitwise_mismatches = 0
  first_violation = nil.as(String?)
  Math.min(actual.size, reference.size).times do |index|
    value = actual[index]
    target = reference[index]
    bitwise_mismatches += 1 if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
    unless value.finite? && target.finite?
      violations += 1
      first_violation ||= " first_index=#{index} actual=#{value} reference=#{target}"
      next
    end
    delta = (value - target).abs
    tolerance = 2e-5_f32 + 2e-5_f32 * target.abs
    max_abs = Math.max(max_abs, delta)
    max_ratio = Math.max(max_ratio, delta / tolerance)
    if delta > tolerance
      violations += 1
      first_violation ||= " first_index=#{index} actual=#{value} reference=#{target}"
    end
  end
  passed = actual.size == reference.size && violations == 0
  detail = "#{label} n=#{actual.size}/#{reference.size} violations=#{violations} max_abs=#{max_abs} max_gate_ratio=#{max_ratio} bitwise_mismatches=#{bitwise_mismatches}#{first_violation}"
  reports << detail
  failures << detail unless passed
end

# Terminal diagnostics only: relate a discrete BF16 mismatch to the F32
# projection that preceded packing. No model computation is moved to the CPU.
private def hy_boundary_diagnostics(config : ML::Video::HyNomasConfig,
                                    actual : Hash(String, Array(Float32)),
                                    expected : JSON::Any, label : String,
                                    reports : Array(String)) : Nil
  {"query" => {"img.attn.q_rope", "txt.attn.q_norm"},
   "key"   => {"img.attn.k_rope", "txt.attn.k_norm"},
   "value" => {"img.attn.v", "txt.attn.v"}}.each do |part, names|
    name = "joint.#{part}"
    reference = hy_floats(expected[name])
    raw_references = {hy_floats(expected[names[0]]), hy_floats(expected[names[1]])}
    # The ordinary comparison owns shape failures. Diagnostics must not mask
    # that report by indexing a malformed fixture after comparison failed.
    next unless actual[name].size == reference.size && reference.size == config.joint_rows * config.dim
    next unless actual[names[0]].size == config.image_rows * config.dim &&
                actual[names[1]].size == config.text_rows * config.dim &&
                raw_references[0].size == actual[names[0]].size &&
                raw_references[1].size == actual[names[1]].size
    count = 0
    actual[name].each_with_index do |packed, index|
      target = reference[index]
      next if packed == target
      joint_row, channel = index.divmod(config.dim)
      batch, token = joint_row.divmod(config.joint_tokens)
      stream = token < config.video_tokens ? 0 : 1
      stream_token = stream == 0 ? token : token - config.video_tokens
      stream_tokens = stream == 0 ? config.video_tokens : config.text_tokens
      raw_index = (batch * stream_tokens + stream_token) * config.dim + channel
      raw = actual[names[stream]][raw_index]
      raw_ref = raw_references[stream][raw_index]
      reports << "boundary_input #{label}/#{name} index=#{index} native_pre=#{raw} reference_pre=#{raw_ref} native_pre_bits=#{raw.unsafe_as(UInt32).to_s(16)} reference_pre_bits=#{raw_ref.unsafe_as(UInt32).to_s(16)} native_post=#{packed} reference_post=#{target}"
      count += 1
      break if count == 4
    end
  end
end

private def hy_run_variant(fixture : JSON::Any, data : JSON::Any,
                           variant : JSON::Any, fixture_path : String,
                           selection_label : String, selected_variant_count : Int32,
                           total_variant_count : Int32) : Array(String)
  config = hy_case_config(fixture, data)
  manifest = variant["weights_file"]
  sidecar = hy_sidecar(fixture_path, manifest)
  source_dtype = ML::GGUF::SafeTensorDType.parse(manifest["dtype"].as_s)
  resident_dtype = ML::Video::WeightDType.parse(variant["resident_dtype"].as_s)
  case_label = "#{data["name"]}/source_#{manifest["dtype"]}/resident_#{resident_dtype}"
  allow_conversion = variant["allow_f32_to_bf16"].as_bool
  variant["attention_precision"].as_s.should eq(fixture["scope"]["attention_precision"].as_s)
  replay = variant["attention_replay"]
  {"query", "key", "value", "attention"}.each do |name|
    hy_floats(replay[name]).should eq(hy_floats(variant["checkpoints"]["joint.#{name}"])),
      "attention replay must come from this variant's first block application"
  end
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
  submission = nil.as(ML::Metal::MPSCommandChain?)
  serial_reuse_submission = nil.as(ML::Metal::MPSCommandChain?)
  workspaces = [] of ML::Video::HyNomasMPSWorkspace
  serial_workspace = nil.as(ML::Video::HyNomasMPSSerialWorkspace?)
  serial_snapshot_arena = nil.as(ML::Video::Arena?)
  serial_snapshot_buffers = {} of String => ML::MetalBuffer
  serial_after_block_indices = [] of Int32
  clean_first_root_delta = nil.as(Int32?)
  clean_second_root_delta = nil.as(Int32?)
  clean_two_block_root_delta = nil.as(Int32?)
  serial_two_block_root_delta = nil.as(Int32?)
  serial_reuse_root_delta = nil.as(Int32?)
  serial_reuse_allocation_delta = nil.as(Int64?)
  serial_reuse_wait_count = nil.as(Int32?)
  serial_owner_scratch_reserve_bytes = 0_i64
  serial_snapshot_reserve_bytes = 0_i64
  retained_workspaces_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
  replay_plan = nil.as((ML::Video::HyNomasAttentionMPSPlan | ML::Video::HyNomasCompactAttentionMPSPlan)?)
  mps_mode = ENV["HY_NOMAS_BLOCK_MPS"]? || "0"
  {"0", "1"}.includes?(mps_mode).should be_true
  serial_mode = ENV["HY_NOMAS_SERIAL_WORKSPACE"]? || "0"
  {"0", "1"}.includes?(serial_mode).should be_true
  raise ArgumentError.new("Serial shared scratch requires the MPS block backend") if serial_mode == "1" && mps_mode != "1"
  attention_query_chunk = ENV.fetch("HY_NOMAS_ATTENTION_QUERY_CHUNK", "0").to_i32
  attention_heads_per_group = ENV.fetch("HY_NOMAS_ATTENTION_HEADS_PER_GROUP", "0").to_i32
  if attention_query_chunk != 0 || attention_heads_per_group != 0
    raise ArgumentError.new("Compact attention requires the MPS block backend") unless mps_mode == "1"
    ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(config.batches, config.joint_tokens,
      config.heads, config.head_dim, attention_query_chunk, attention_heads_per_group)
  end
  norm_mode = ENV["HY_NOMAS_NORM_CONTROLS"]? || "0"
  {"0", "1"}.includes?(norm_mode).should be_true
  norm_control = norm_mode == "1" ? HyNomasNormProbe::Controls.new(config.dim, config.norm_eps) : nil
  modulation_mode = ENV["HY_NOMAS_MODULATION_CONTROLS"]? || "0"
  {"0", "1"}.includes?(modulation_mode).should be_true
  modulation_control = nil.as(HyNomasModulationControls?)
  activation_mode = ENV["HY_NOMAS_ACTIVATION_CONTROLS"]? || "0"
  {"0", "1"}.includes?(activation_mode).should be_true
  raise ArgumentError.new("Activation graph controls require the MPS command chain") if activation_mode == "1" && mps_mode != "1"
  activation_control = nil.as(HyNomasActivationControls?)
  begin
    allocation_count = ML::MetalBuffer.video_spec_allocations
    expect_raises(ArgumentError, /memory budget/) do
      ML::Video::HyNomasDualStreamBlock.weight_plan([file], config, 1_i64, resident_dtype: resident_dtype, require_all: true,
        allow_f32_to_bf16: allow_conversion)
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
    {"query", "key", "value"}.each do |name|
      inputs.upload("attention_replay.#{name}", hy_floats(replay[name]))
    end
    joint_count = config.joint_rows * config.dim
    inputs.alloc("attention_replay.raw", joint_count)
    inputs.alloc("attention_replay.rounded", joint_count)
    inputs.alloc("attention_replay.img", config.image_rows * config.dim)
    inputs.alloc("attention_replay.txt", config.text_rows * config.dim)
    if norm_control
      inputs.alloc("norm_control.img", config.image_rows * config.dim)
      inputs.alloc("norm_control.txt", config.text_rows * config.dim)
    end
    if modulation_mode == "1"
      modulation_control = HyNomasModulationControls.new(config, inputs, variant, data)
    end
    if activation_mode == "1"
      activation_control = HyNomasActivationControls.new(config, inputs, variant)
    end
    {first_scratch, chain_scratch, perturbed_scratch}.each do |scratch|
      ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
    end
    # Weight-plan validation covers these named buffers. Prepared MPS stages
    # are added below and reported separately; the external guard caps the
    # actual process, including opaque MPSGraph allocations.
    if serial_mode == "1"
      serial_plan = ML::Video::HyNomasDualStreamBlock.scratch_plan(config)
      serial_owner_scratch_reserve_bytes = serial_plan.sum do |name, count|
        count.to_i64 * sizeof(Float32) * (name == "hy_nomas.img.output" || name == "hy_nomas.txt.output" ? 2_i64 : 1_i64)
      end
      serial_snapshot_reserve_bytes = variant["checkpoints"].as_h.sum do |name, _|
        first_scratch["hy_nomas.#{name}"].size
      end
    end
    reserve = inputs.resident_bytes + first_scratch.resident_bytes + chain_scratch.resident_bytes +
              perturbed_scratch.resident_bytes + serial_owner_scratch_reserve_bytes + serial_snapshot_reserve_bytes
    plan = ML::Video::HyNomasDualStreamBlock.weight_plan(
      [file], config, 2_i64 * File.size(sidecar) + reserve + 1024_i64 * 1024,
      reserve_bytes: reserve, resident_dtype: resident_dtype, require_all: true,
      allow_f32_to_bf16: allow_conversion,
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

    if mps_mode == "1"
      # Exercise constructor admission directly: a helper-only rejection test
      # would not catch allocating workspace stages before checking geometry.
      allocations_before_rejection = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /diagnostic only/) do
        ML::Video::HyNomasMPSWorkspace.new(first,
          inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"],
          attention_query_chunk: config.joint_tokens, attention_heads_per_group: config.heads - 1)
      end
      partial_rejection = config.batches == 1 ? /Short compact/ : /Multi-batch compact/
      expect_raises(ArgumentError, partial_rejection) do
        ML::Video::HyNomasMPSWorkspace.new(first,
          inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"],
          attention_query_chunk: 4, attention_heads_per_group: config.heads)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_rejection)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before)
      workspaces << ML::Video::HyNomasMPSWorkspace.new(first,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"],
        attention_query_chunk: attention_query_chunk, attention_heads_per_group: attention_heads_per_group)
      workspaces << ML::Video::HyNomasMPSWorkspace.new(chain,
        first.image_output, first.text_output, inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"],
        attention_query_chunk: attention_query_chunk, attention_heads_per_group: attention_heads_per_group)
      workspaces << ML::Video::HyNomasMPSWorkspace.new(perturbed,
        inputs["img_perturbed"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"],
        attention_query_chunk: attention_query_chunk, attention_heads_per_group: attention_heads_per_group)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before + 3)
      replay_plan = if attention_query_chunk > 0
                      ML::Video::HyNomasCompactAttentionMPSPlan.new(config.batches, config.joint_tokens,
                        config.heads, config.head_dim, inputs["attention_replay.query"], inputs["attention_replay.key"],
                        inputs["attention_replay.value"], inputs["attention_replay.raw"],
                        attention_query_chunk, attention_heads_per_group)
                    else
                      ML::Video::HyNomasAttentionMPSPlan.new(config.batches, config.joint_tokens,
                        config.heads, config.head_dim, inputs["attention_replay.query"], inputs["attention_replay.key"],
                        inputs["attention_replay.value"], inputs["attention_replay.raw"])
                    end
    end
    if serial_mode == "1"
      # This paired control deliberately applies the same pinned block-0
      # weights twice. The fixture's chain_expected oracle has that same scope.
      serial_workspace = ML::Video::HyNomasMPSSerialWorkspace.new(config, [weights, weights],
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["video_cos"], inputs["video_sin"],
        attention_query_chunk: attention_query_chunk,
        attention_heads_per_group: attention_heads_per_group)
      serial_workspace.not_nil!.blocks.size.should eq(2)
      serial_workspace.not_nil!.blocks.each do |block|
        block.checkpoints.keys.sort.should eq(first.checkpoints.keys.sort)
      end
      serial_workspace.not_nil!.scratch_bytes.should be > 0
      serial_workspace.not_nil!.scratch_bytes.should eq(serial_owner_scratch_reserve_bytes)
      serial_workspace.not_nil!.mps_workspace_bytes.should be > 0
      serial_workspace.not_nil!.resident_bytes.should eq(
        serial_workspace.not_nil!.scratch_bytes + serial_workspace.not_nil!.mps_workspace_bytes)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before + 4),
        "the three independent controls plus serial owner must retain exactly four MPS workspaces"

      serial_snapshot_arena = ML::Video::Arena.new
      arenas << serial_snapshot_arena.not_nil!
      serial_first = serial_workspace.not_nil!.blocks[0]
      serial_first.checkpoints.each do |name, source|
        (source.size % sizeof(Float32)).should eq(0)
        serial_snapshot_buffers[name] = serial_snapshot_arena.not_nil!.alloc(name,
          (source.size // sizeof(Float32)).to_i32)
      end
      serial_first.checkpoints.size.should eq(40)
      serial_snapshot_buffers.size.should eq(40)
      serial_snapshot_arena.not_nil!.resident_bytes.should eq(serial_snapshot_reserve_bytes)
    end
    allocations_before_encode = ML::MetalBuffer.video_spec_allocations
    command = nil.as(ML::Metal::CommandBuffer?)
    encoder = if mps_mode == "1"
                submission = ML::Metal::MPSCommandChain.new
                submission.not_nil!.encoder
              else
                command = ML::Metal::CommandBuffer.new
                ML::Metal::ComputeEncoder.new(command.not_nil!)
              end
    if mps_mode == "1"
      operation = first.mps_linear_operations.first
      expect_raises(ArgumentError, /preflight/) do
        workspaces[0].project(submission.not_nil!, operation.name, operation.input,
          operation.weights, operation.output, operation.rows, operation.input_dim,
          operation.output_dim)
      end
    end
    prepared.not_nil!.encode(encoder)
    if norm = norm_control
      norm.encode(encoder, inputs["img"], inputs["norm_control.img"], config.image_rows)
      norm.encode(encoder, inputs["txt"], inputs["norm_control.txt"], config.text_rows)
    end
    modulation_control.try(&.encode(encoder))
    if control = activation_control
      control.encode(submission.not_nil!)
      encoder = submission.not_nil!.encoder
    end
    # Isolate attention on the exact upstream Q/K/V. No F32 pre-attention drift
    # can cross a BF16 midpoint and contaminate this independent discriminator.
    if mps_mode == "1"
      replay_plan.not_nil!.encode(submission.not_nil!)
      encoder = submission.not_nil!.encoder
    else
      ML::Video::MetalOps.attention(encoder,
        inputs["attention_replay.query"], inputs["attention_replay.key"], inputs["attention_replay.value"],
        inputs["attention_replay.raw"], config.joint_tokens, config.joint_tokens,
        config.heads, config.head_dim, config.batches)
    end
    ML::Video::MetalOps.copy(encoder, inputs["attention_replay.raw"], inputs["attention_replay.rounded"], joint_count)
    encoder.set_pipeline(ML::Video::HyNomasDualStreamBlock.pipeline("hy_nomas_split_f32"))
    encoder.set_buffer(inputs["attention_replay.rounded"], 0, ML::Metal::BufferAccess::ReadWrite)
    encoder.set_buffer(inputs["attention_replay.img"], 1, ML::Metal::BufferAccess::Write)
    encoder.set_buffer(inputs["attention_replay.txt"], 2, ML::Metal::BufferAccess::Write)
    encoder.set_value(config.batches.to_u32, 3).set_value(config.video_tokens.to_u32, 4)
    encoder.set_value(config.text_tokens.to_u32, 5).set_value(config.dim.to_u32, 6)
    encoder.set_value(config.attention_precision.bf16_boundary_control? ? 1_u32 : 0_u32, 7)
    encoder.dispatch_1d(joint_count)
    expect_raises(ArgumentError, /alias/) do
      first.encode(encoder, first.image_output, inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
    end
    if mps_mode == "1"
      expect_raises(ArgumentError) do
        first.encode_mps(submission.not_nil!, first.image_output, inputs["txt"], inputs["vec"],
          inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[0])
      end
      observed_markers = [] of String
      observed_counts = [] of Int32
      observer = ->(name : String) do
        observed_markers << name
        observed_counts << submission.not_nil!.observed_root_count
        submission.not_nil!.wait_count.should eq(0)
        nil
      end
      clean_two_block_root_start = submission.not_nil!.observed_root_count
      first.encode_mps(submission.not_nil!, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[0],
        phase_observer: observer)
      observed_markers.should eq(["validate", "modulation", "qkv", "joint_pack", "attention",
                                  "attention_projection_residual", "image_ffn", "text_ffn", "complete"])
      observed_counts.all? { |count| count > 0 }.should be_true
      observed_counts.should eq(observed_counts.sort)
      clean_after_first_block = submission.not_nil!.observed_root_count
      chain.encode_mps(submission.not_nil!, first.image_output, first.text_output, inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[1])
      clean_after_second_block = submission.not_nil!.observed_root_count
      clean_first_root_delta = clean_after_first_block - clean_two_block_root_start
      clean_second_root_delta = clean_after_second_block - clean_after_first_block
      clean_two_block_root_delta = clean_after_second_block - clean_two_block_root_start
      # Repeat the same prepared bindings; no state or scratch may accumulate.
      first.encode_mps(submission.not_nil!, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[0])
      perturbed.encode_mps(submission.not_nil!, inputs["img_perturbed"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[2])
      if shared = serial_workspace
        serial_root_start = submission.not_nil!.observed_root_count
        serial_blocks = shared.blocks
        callback = ->(index : Int32, block : ML::Video::HyNomasDualStreamBlock) do
          serial_after_block_indices << index
          block.same?(serial_blocks[index]).should be_true
          if index == 0
            block.checkpoints.keys.sort.should eq(serial_snapshot_buffers.keys.sort)
            block.checkpoints.each do |name, source|
              count = (source.size // sizeof(Float32)).to_i32
              ML::Video::MetalOps.copy(submission.not_nil!.encoder, source,
                serial_snapshot_buffers[name], count)
            end
          end
          nil
        end
        returned_img, returned_txt = shared.encode(submission.not_nil!, after_block: callback)
        returned_img.same?(shared.blocks[1].image_output).should be_true
        returned_txt.same?(shared.blocks[1].text_output).should be_true
        serial_after_shared_encode = submission.not_nil!.observed_root_count
        serial_two_block_root_delta = serial_after_shared_encode - serial_root_start
        serial_after_block_indices.should eq([0, 1])

        # Reject a distinct live chain before dispatching either block. Reuse
        # this empty chain after the main chain drains for the lifecycle test.
        serial_reuse_submission = ML::Metal::MPSCommandChain.new
        main_roots_before_rejection = submission.not_nil!.observed_root_count
        reuse_roots_before_rejection = serial_reuse_submission.not_nil!.observed_root_count
        expect_raises(ArgumentError, /cannot span in-flight command chains/) do
          shared.encode(serial_reuse_submission.not_nil!)
        end
        submission.not_nil!.observed_root_count.should eq(main_roots_before_rejection)
        serial_reuse_submission.not_nil!.observed_root_count.should eq(reuse_roots_before_rejection)
        serial_after_block_indices.should eq([0, 1]), "rejected chain must not reach a block callback"
        submission.not_nil!.wait_count.should eq(0)
        serial_reuse_submission.not_nil!.wait_count.should eq(0)
        expect_raises(ArgumentError, /Complete Hy-Nomas serial command chain before release/) { shared.release }
      end
      expect_raises(ArgumentError, /Complete/) { workspaces[0].release }
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before + 3 + (serial_workspace ? 1 : 0))
    else
      first.encode(encoder, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      chain.encode(encoder, first.image_output, first.text_output, inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      first.encode(encoder, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      perturbed.encode(encoder, inputs["img_perturbed"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      encoder.end_encoding
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_encode),
      "encode must not allocate even a temporary MetalBuffer"
    gpu_seconds = if mps_mode == "1"
                    submission.not_nil!.commit_and_wait
                    submission.not_nil!.wait_count.should eq(1)
                    submission.not_nil!.gpu_elapsed_seconds
                  else
                    command.not_nil!.commit_and_wait_gpu_elapsed_seconds
                  end
    # All activation reads occur after this one existing terminal wait.
    snapshots = hy_snapshots(first)
    chain_snapshots = hy_snapshots(chain)
    perturbed_snapshots = hy_snapshots(perturbed)
    failures = [] of String
    reports = [] of String
    if control = activation_control
      control.snapshots.each do |stream, actual|
        expected = variant["checkpoints"]["#{stream}.ff.activated"]
        label = "#{case_label}/activation_control/#{stream}/gelu_tanh"
        hy_compare(actual, expected, label, failures, reports)
        reference = hy_floats(expected)
        mismatches = actual.zip(reference).count { |left, right| left.unsafe_as(UInt32) != right.unsafe_as(UInt32) }
        failures << "#{label} raw F32 bit mismatches=#{mismatches}" unless actual.size == reference.size && mismatches == 0
      end
    end
    if control = modulation_control
      control.snapshots.each do |name, actual|
        stream, kind, mode = name.split('.')
        checkpoint = kind == "modulate" ? "#{stream}.attn.modulated" : "#{stream}.after_attention"
        hy_compare(actual, variant["checkpoints"][checkpoint],
          "#{case_label}/modulation_control/#{name}", failures, reports)
        if mode == "uncontracted"
          reference = hy_floats(variant["checkpoints"][checkpoint])
          unless actual.size == reference.size && actual.zip(reference).all? { |left, right| left.unsafe_as(UInt32) == right.unsafe_as(UInt32) }
            failures << "#{case_label}/modulation_control/#{name} must match frozen upstream inputs bitwise"
          end
        end
      end
    end
    if norm_control
      {"img" => config.image_rows, "txt" => config.text_rows}.each do |stream, rows|
        actual = inputs["norm_control.#{stream}"].read(rows * config.dim)
        expected = variant["checkpoints"]["#{stream}.attn.norm"]
        label = "#{case_label}/norm_control/#{stream}"
        hy_compare(actual, expected, label, failures, reports)
        mismatches = actual.zip(hy_floats(expected)).count { |left, right| left.unsafe_as(UInt32) != right.unsafe_as(UInt32) }
        failures << "#{label} raw F32 bit mismatches=#{mismatches}" unless mismatches == 0
      end
    end
    hy_compare(inputs["attention_replay.raw"].read(joint_count), replay["attention_f32"],
      "#{case_label}/attention_replay/f32", failures, reports)
    hy_compare(inputs["attention_replay.rounded"].read(joint_count), replay["attention"],
      "#{case_label}/attention_replay/post_boundary", failures, reports)
    {"first"     => {snapshots, variant["checkpoints"]},
     "chain"     => {chain_snapshots, variant["chain_expected"]["checkpoints"]},
     "perturbed" => {perturbed_snapshots, variant["video_perturbation"]["checkpoints"]}}.each do |group, pair|
      values, expected = pair
      expected.as_h.each do |name, reference|
        hy_compare(values[name], reference, "#{case_label}/#{group}/#{name}", failures, reports)
        if config.attention_precision.bf16_boundary_control? &&
           {"joint.query", "joint.key", "joint.value", "joint.attention", "img.attn.raw", "txt.attn.raw"}.includes?(name)
          reference_bits = hy_floats(reference).map(&.unsafe_as(UInt32))
          actual_bits = values[name].map(&.unsafe_as(UInt32))
          mismatches = actual_bits.zip(reference_bits).count { |left, right| left != right }
          detail = "boundary_bits #{case_label}/#{group}/#{name} mismatches=#{mismatches} n=#{actual_bits.size}/#{reference_bits.size}"
          reports << detail
          failures << detail unless actual_bits.size == reference_bits.size && mismatches == 0
        end
      end
      if config.attention_precision.bf16_boundary_control?
        hy_boundary_diagnostics(config, values, expected, "#{case_label}/#{group}", reports)
      end
    end
    if shared = serial_workspace
      clean_two_block_root_delta.not_nil!.should eq(serial_two_block_root_delta.not_nil!),
        "shared two-block encoding must preserve the separate-control MPS root-count delta"
      serial_first_snapshots = serial_snapshot_buffers.transform_values do |buffer|
        buffer.read((buffer.size // sizeof(Float32)).to_i).map(&.to_f32)
      end
      serial_second_snapshots = hy_snapshots(shared.blocks[1])
      {"serial_first"  => {serial_first_snapshots, variant["checkpoints"]},
       "serial_second" => {serial_second_snapshots, variant["chain_expected"]["checkpoints"]}}.each do |group, pair|
        values, expected = pair
        expected.as_h.each do |name, reference|
          hy_compare(values[name], reference, "#{case_label}/#{group}/#{name}", failures, reports)
          if config.attention_precision.bf16_boundary_control? &&
             {"joint.query", "joint.key", "joint.value", "joint.attention", "img.attn.raw", "txt.attn.raw"}.includes?(name)
            reference_bits = hy_floats(reference).map(&.unsafe_as(UInt32))
            actual_bits = values[name].map(&.unsafe_as(UInt32))
            mismatches = actual_bits.zip(reference_bits).count { |left, right| left != right }
            detail = "boundary_bits #{case_label}/#{group}/#{name} mismatches=#{mismatches} n=#{actual_bits.size}/#{reference_bits.size}"
            reports << detail
            failures << detail unless actual_bits.size == reference_bits.size && mismatches == 0
          end
        end
        if config.attention_precision.bf16_boundary_control?
          hy_boundary_diagnostics(config, values, expected, "#{case_label}/#{group}", reports)
        end
      end

      # Once the first request is drained, the same workspace and same input
      # buffers may be encoded on a fresh chain. Compare both final streams to
      # their first-request values using the unchanged finite numerical gate.
      serial_before_reuse = {
        "img.output" => shared.blocks[1].image_output.read(config.image_rows * config.dim),
        "txt.output" => shared.blocks[1].text_output.read(config.text_rows * config.dim),
      }
      reuse_chain = serial_reuse_submission.not_nil!
      reuse_roots_start = reuse_chain.observed_root_count
      reuse_allocations_start = ML::MetalBuffer.video_spec_allocations
      reused_img, reused_txt = shared.encode(reuse_chain)
      reused_img.same?(shared.blocks[1].image_output).should be_true
      reused_txt.same?(shared.blocks[1].text_output).should be_true
      serial_reuse_root_delta = reuse_chain.observed_root_count - reuse_roots_start
      serial_reuse_allocation_delta = ML::MetalBuffer.video_spec_allocations - reuse_allocations_start
      serial_reuse_allocation_delta.should eq(0), "serial re-encode must not allocate MetalBuffers"
      reuse_chain.wait_count.should eq(0)
      reuse_chain.commit_and_wait
      reuse_chain.wait_count.should eq(1)
      serial_reuse_wait_count = reuse_chain.wait_count
      hy_compare_arrays(reused_img.read(config.image_rows * config.dim), serial_before_reuse["img.output"],
        "#{case_label}/serial_reencode/img.output", failures, reports)
      hy_compare_arrays(reused_txt.read(config.text_rows * config.dim), serial_before_reuse["txt.output"],
        "#{case_label}/serial_reencode/txt.output", failures, reports)
    end
    comparison_count = reports.count { |line| !line.starts_with?("boundary_input ") && !line.starts_with?("boundary_bits ") }
    origin = variant["weights_are_synthetic"].as_bool ? "synthetic" : "checkpoint"
    workspace_bytes = workspaces.sum(&.resident_bytes)
    replay_bytes = replay_plan.try(&.resident_bytes) || 0_i64
    serial_metrics = if shared = serial_workspace
                       " serial_workspace=1 serial_scratch_bytes=#{shared.scratch_bytes} serial_snapshot_bytes=#{serial_snapshot_arena.not_nil!.resident_bytes} serial_mps_workspace_bytes=#{shared.mps_workspace_bytes} serial_resident_bytes=#{shared.resident_bytes} clean_first_root_delta=#{clean_first_root_delta.not_nil!} clean_second_root_delta=#{clean_second_root_delta.not_nil!} clean_two_block_root_delta=#{clean_two_block_root_delta.not_nil!} serial_two_block_root_delta=#{serial_two_block_root_delta.not_nil!} serial_reencode_root_delta=#{serial_reuse_root_delta.not_nil!} serial_reencode_metal_buffer_allocations=#{serial_reuse_allocation_delta.not_nil!} serial_reencode_terminal_waits=#{serial_reuse_wait_count.not_nil!}"
                     else
                       " serial_workspace=0"
                     end
    summary = "Hy-Nomas #{data["name"]}/source_#{manifest["dtype"]}/resident_#{resident_dtype}/#{fixture["scope"]["attention_precision"].as_s}: fixture_selection=#{selection_label} fixture_variants=#{selected_variant_count}/#{total_variant_count} backend=#{mps_mode == "1" ? "mps" : "native"} comparisons=#{comparison_count} failures=#{failures.size} weights_origin=#{origin} resident_weights_bytes=#{plan.destination_bytes} weight_staging_bytes=#{plan.staging_bytes} scratch_bytes=#{reserve - inputs.resident_bytes} mps_workspace_bytes=#{workspace_bytes} mps_replay_attention_bytes=#{replay_bytes} mps_named_stage_total_bytes=#{workspace_bytes + replay_bytes}#{serial_metrics} attention_query_chunk=#{attention_query_chunk} attention_heads_per_group=#{attention_heads_per_group} encode_metal_buffer_allocations=#{ML::MetalBuffer.video_spec_allocations - allocations_before_encode} terminal_waits=1 gpu_seconds_diagnostic_command=#{gpu_seconds} reference_kind=#{fixture["scope"]["reference_kind"].as_s}"
    STDERR.puts summary
    if report_path = ENV["HY_NOMAS_REPORT"]?
      File.open(report_path, "a") do |report|
        report.puts summary
        reports.each { |line| report.puts line }
      end
    end
    # A video perturbation must propagate through joint attention into text.
    {"img.output", "txt.output"}.each do |name|
      snapshots[name].zip(perturbed_snapshots[name]).any? { |left, right| (left - right).abs > 1e-5_f32 }.should be_true,
        "the fixture must exercise #{name} rather than an identity/no-op"
    end
    hy_matches(snapshots["img.output"], chain_snapshots["img.output"]).should be_false,
      "the chained block must consume the first block's resident output"
    failures
  ensure
    begin
      submission.try { |pending| pending.commit_and_wait unless pending.completed? }
    ensure
      # A failed drain may leave GPU work in flight. Retain its resources in
      # that case; a completed chain, including a terminal GPU error, is safe
      # to clean up. The external guard remains the final process boundary.
      begin
        serial_reuse_submission.try { |pending| pending.commit_and_wait unless pending.completed? }
      ensure
        if (submission.nil? || submission.not_nil!.completed?) &&
           (serial_reuse_submission.nil? || serial_reuse_submission.not_nil!.completed?)
          begin
            activation_control.try { |control| control.release(submission) }
            submission.try(&.release)
            serial_reuse_submission.try(&.release)
            serial_workspace.try(&.release)
          ensure
            replay_plan.try(&.release)
            workspaces.reverse_each(&.release_after_terminal_wait!)
            if !workspaces.empty? && (expired_encoder = encoder) && (bound_inputs = inputs)
              # Invalid geometry prevents any dispatch if the released guard
              # regresses; the expired encoder must never be touched here.
              expect_raises(ArgumentError, /released/) do
                workspaces[0].rope(expired_encoder, bound_inputs["img"], bound_inputs["txt"],
                  bound_inputs["video_cos"], bound_inputs["video_sin"], 0)
              end
            end
            ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before)
            prepared.try(&.release)
            file.close
            arenas.reverse_each(&.release)
          end
        end
      end
    end
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

# Retain the digest-verified parse for selected runs. The real oracle JSON is
# hundreds of MiB, so the parity example must not parse a second copy.
HY_NOMAS_FIXTURE_VARIANT_PREFLIGHT = hy_preflight_fixture_variant_selector!

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
      config.copy_with(attention_precision: ML::Video::HyNomasAttentionPrecision.new(99)),
    ]
    allocations = ML::MetalBuffer.video_spec_allocations
    config.validate!
    config.attention_precision.should eq(ML::Video::HyNomasAttentionPrecision::F32Control)
    config.copy_with(attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl).validate!
    invalid.each do |item|
      expect_raises(ArgumentError) { item.validate! }
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "rounds both attention boundaries on GPU with ties-even and preserves the F32 control" do
    # Expected words are literal independent boundary cases, not values computed
    # by copying the shader's integer rounding formula into the test.
    source = [
      0x00000000_u32, 0x80000000_u32,
      0x3f807fff_u32, 0x3f808000_u32, 0x3f808001_u32, 0x3f818000_u32,
      0xbf808000_u32, 0xbf818000_u32,
      0x00008000_u32, 0x00018000_u32, 0x007f8000_u32,
      0x7f7f7fff_u32, 0x7f7f8000_u32,
      0x7f800000_u32, 0xff800000_u32,
      0x7f800001_u32, 0xff800001_u32, 0x7fc10000_u32,
    ]
    rounded = [
      0x00000000_u32, 0x80000000_u32,
      0x3f800000_u32, 0x3f800000_u32, 0x3f810000_u32, 0x3f820000_u32,
      0xbf800000_u32, 0xbf820000_u32,
      0x00000000_u32, 0x00020000_u32, 0x00800000_u32,
      0x7f7f0000_u32, 0x7f800000_u32,
      0x7f800000_u32, 0xff800000_u32,
      0x7fc00000_u32, 0xffc00000_u32, 0x7fc10000_u32,
    ]
    source.should_not eq(rounded), "an omitted BF16 boundary must be detectable"
    arena = ML::Video::Arena.new
    begin
      arena.upload("left", source.map(&.unsafe_as(Float32)))
      arena.upload("right", source.reverse.map(&.unsafe_as(Float32)))
      arena.upload("raw_attention", (source + source.reverse).map(&.unsafe_as(Float32)))
      {"f32", "bf16"}.each do |mode|
        arena.alloc("#{mode}.packed", 2 * source.size)
        arena.alloc("#{mode}.attention", 2 * source.size)
        arena.alloc("#{mode}.img", source.size)
        arena.alloc("#{mode}.txt", source.size)
      end
      ML::Video::HyNomasDualStreamBlock.prepare!
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      allocations = ML::MetalBuffer.video_spec_allocations
      {"f32" => 0_u32, "bf16" => 1_u32}.each do |mode, flag|
        encoder.set_pipeline(ML::Video::HyNomasDualStreamBlock.pipeline("hy_nomas_concat_f32"))
        encoder.set_buffer(arena["left"], 0).set_buffer(arena["right"], 1)
        encoder.set_buffer(arena["#{mode}.packed"], 2, ML::Metal::BufferAccess::Write)
        encoder.set_value(1_u32, 3).set_value(1_u32, 4).set_value(1_u32, 5)
        encoder.set_value(source.size.to_u32, 6).set_value(flag, 7)
        encoder.dispatch_1d(2 * source.size)
        ML::Video::MetalOps.copy(encoder, arena["raw_attention"], arena["#{mode}.attention"], 2 * source.size)
        encoder.set_pipeline(ML::Video::HyNomasDualStreamBlock.pipeline("hy_nomas_split_f32"))
        encoder.set_buffer(arena["#{mode}.attention"], 0, ML::Metal::BufferAccess::ReadWrite)
        encoder.set_buffer(arena["#{mode}.img"], 1, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(arena["#{mode}.txt"], 2, ML::Metal::BufferAccess::Write)
        encoder.set_value(1_u32, 3).set_value(1_u32, 4).set_value(1_u32, 5)
        encoder.set_value(source.size.to_u32, 6).set_value(flag, 7)
        encoder.dispatch_1d(2 * source.size)
      end
      encoder.end_encoding
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      command.commit_and_wait
      {"f32" => source, "bf16" => rounded}.each do |mode, expected|
        {"packed", "attention"}.each do |name|
          arena["#{mode}.#{name}"].read(2 * source.size).map(&.unsafe_as(UInt32)).should eq(expected + expected.reverse)
        end
        arena["#{mode}.img"].read(source.size).map(&.unsafe_as(UInt32)).should eq(expected)
        arena["#{mode}.txt"].read(source.size).map(&.unsafe_as(UInt32)).should eq(expected.reverse)
      end
    ensure
      arena.release
    end
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

  it "rejects a synthetic fixture or a different source digest in actual checkpoint mode" do
    digest = "a" * 64
    source = %({"format":"safetensors","file_sha256":"#{digest}","weights_are_synthetic":false,"source_dtype":"F32","tensor_prefix":"double_blocks.0.","tensor_count":32})
    variant = %({"weights_are_synthetic":false,"source_checkpoint":#{source}})
    contents = %({"weights_are_synthetic":false,"source":#{source},"cases":[{"variants":{"f32":#{variant},"bf16_weights_roundtrip_f32_compute":#{variant}}}]})
    fixture = JSON.parse(contents)
    allocations = ML::MetalBuffer.video_spec_allocations
    hy_require_checkpoint_source!(fixture, digest)
    expect_raises(ArgumentError, /differs from the trusted block/) { hy_require_checkpoint_source!(fixture, "b" * 64) }
    expect_raises(ArgumentError, /trusted source SHA-256/) { hy_require_checkpoint_source!(fixture, "") }
    expect_raises(ArgumentError, /rejects synthetic/) do
      hy_require_checkpoint_source!(JSON.parse(contents.sub("\"weights_are_synthetic\":false", "\"weights_are_synthetic\":true")), digest)
    end
    expect_raises(ArgumentError, /differs from the trusted block/) do
      hy_require_checkpoint_source!(JSON.parse(contents.sub("\"tensor_count\":32", "\"tensor_count\":31")), digest)
    end
    fixture["cases"][0]["variants"]["f32"].as_h["source_checkpoint"] = JSON::Any.new(nil)
    expect_raises(ArgumentError, /Variant is not bound/) { hy_require_checkpoint_source!(fixture, digest) }
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "preflights an exact fixture variant selector without device allocation" do
    contents = %({"schema_version":2,"cases":[{"name":"broadcast","variants":{"f32":{"resident_dtype":"F32"},"bf16_weights_roundtrip_f32_compute":{"resident_dtype":"BF16"}}},{"name":"per_token","variants":{"f32":{"resident_dtype":"F32"},"bf16_weights_roundtrip_f32_compute":{"resident_dtype":"BF16"}}}]})
    fixture = JSON.parse(contents)
    allocations = ML::MetalBuffer.video_spec_allocations
    hy_fixture_variant_labels(fixture).sort.should eq(["broadcast/BF16", "broadcast/F32", "per_token/BF16", "per_token/F32"])
    {"broadcast/BF16", "broadcast/F32", "per_token/BF16", "per_token/F32"}.each do |selector|
      hy_validate_fixture_variant_selector!(fixture, selector)
    end
    expect_raises(ArgumentError, /case\/resident_dtype/) do
      hy_validate_fixture_variant_selector!(fixture, "unknown/F32")
    end
    expect_raises(ArgumentError, /case\/resident_dtype/) do
      hy_validate_fixture_variant_selector!(fixture, "")
    end

    invalid_variants = JSON.parse(contents)
    invalid_variants["cases"][0]["variants"].as_h.delete("f32")
    expect_raises(ArgumentError, /fixture variants must be/) do
      hy_validate_fixture_variant_selector!(invalid_variants, "broadcast/F32")
    end
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "matches the declared upstream precision control, resident chaining and perturbation without encode allocations" do
    fixture_path = ENV["HY_NOMAS_BLOCK_FIXTURE"]?
    pending! "Set HY_NOMAS_BLOCK_FIXTURE to the external pinned MPS oracle JSON" unless fixture_path
    fixture = HY_NOMAS_FIXTURE_VARIANT_PREFLIGHT ||
              hy_load_fixture(fixture_path.not_nil!, ENV["HY_NOMAS_FIXTURE_SHA256"]? || "")
    if expected_source = ENV["HY_NOMAS_CHECKPOINT_SHA256"]?
      hy_require_checkpoint_source!(fixture, expected_source)
    end
    fixture["schema_version"].as_i.should eq(2)
    fixture["provenance"]["revision"].as_s.should eq("911cea6458a89c2663513ff7099561ec190dd491")
    scope = fixture["scope"]
    precision = scope["attention_precision"].as_s
    hy_attention_precision(precision)
    scope["reference_kind"].as_s.should eq(precision == "f32" ? "upstream_block_f32_math_mps_sdpa" : "upstream_block_bf16_boundary_control_mps_sdpa")
    scope["pinned_magi_runtime_numeric_parity"].as_bool.should be_false
    scope["segment_mode"].as_s.should eq("single_full")
    scope["text_mask"].as_s.should eq("none_all_valid")
    scope["history"].as_s.should eq("none")
    scope["camera"].as_s.should eq("unsupported_absent")
    scope["chunk_size_tokens"].as_i.should eq(0)
    cases = fixture["cases"].as_a
    cases.map { |data| data["name"].as_s }.sort.should eq(["broadcast", "per_token"])
    selector = ENV["HY_NOMAS_FIXTURE_VARIANT"]?
    if selector
      hy_validate_fixture_variant_selector!(fixture, selector)
    end
    total_variant_count = hy_fixture_variant_labels(fixture).size
    selected_variant_count = selector ? 1 : total_variant_count
    selection_label = selector || "all"
    selection_report = "Hy-Nomas fixture selection=#{selection_label} selected_variants=#{selected_variant_count}/#{total_variant_count}"
    STDERR.puts selection_report
    if report_path = ENV["HY_NOMAS_REPORT"]?
      File.open(report_path, "a") { |report| report.puts selection_report }
    end
    failures = [] of String
    cases.each do |data|
      data["variants"].as_h.keys.sort.should eq(["bf16_weights_roundtrip_f32_compute", "f32"])
      data["variants"].as_h.each do |_variant_name, variant|
        variant_label = "#{data["name"].as_s}/#{variant["resident_dtype"].as_s}"
        next if selector && variant_label != selector
        failures.concat(hy_run_variant(fixture, data, variant, fixture_path.not_nil!,
          selection_label, selected_variant_count, total_variant_count))
      end
    end
    failures.empty?.should be_true, failures.first(16).join("\n")
  end
end
