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
                           variant : JSON::Any, fixture_path : String) : Array(String)
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
  workspaces = [] of ML::Video::HyNomasMPSWorkspace
  retained_workspaces_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
  replay_plan = nil.as(ML::Video::HyNomasAttentionMPSPlan?)
  mps_mode = ENV["HY_NOMAS_BLOCK_MPS"]? || "0"
  {"0", "1"}.includes?(mps_mode).should be_true
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
    reserve = inputs.resident_bytes + first_scratch.resident_bytes + chain_scratch.resident_bytes + perturbed_scratch.resident_bytes
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
      workspaces << ML::Video::HyNomasMPSWorkspace.new(first,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      workspaces << ML::Video::HyNomasMPSWorkspace.new(chain,
        first.image_output, first.text_output, inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      workspaces << ML::Video::HyNomasMPSWorkspace.new(perturbed,
        inputs["img_perturbed"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"])
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before + 3)
      replay_plan = ML::Video::HyNomasAttentionMPSPlan.new(config.batches, config.joint_tokens,
        config.heads, config.head_dim, inputs["attention_replay.query"], inputs["attention_replay.key"],
        inputs["attention_replay.value"], inputs["attention_replay.raw"])
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
      first.encode_mps(submission.not_nil!, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[0])
      chain.encode_mps(submission.not_nil!, first.image_output, first.text_output, inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[1])
      # Repeat the same prepared bindings; no state or scratch may accumulate.
      first.encode_mps(submission.not_nil!, inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[0])
      perturbed.encode_mps(submission.not_nil!, inputs["img_perturbed"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["video_cos"], inputs["video_sin"], workspaces[2])
      expect_raises(ArgumentError, /Complete/) { workspaces[0].release }
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained_workspaces_before + 3)
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
    comparison_count = reports.count { |line| !line.starts_with?("boundary_input ") && !line.starts_with?("boundary_bits ") }
    origin = variant["weights_are_synthetic"].as_bool ? "synthetic" : "checkpoint"
    workspace_bytes = workspaces.sum(&.resident_bytes)
    summary = "Hy-Nomas #{data["name"]}/source_#{manifest["dtype"]}/resident_#{resident_dtype}/#{fixture["scope"]["attention_precision"].as_s}: backend=#{mps_mode == "1" ? "mps" : "native"} comparisons=#{comparison_count} failures=#{failures.size} weights_origin=#{origin} resident_weights_bytes=#{plan.destination_bytes} weight_staging_bytes=#{plan.staging_bytes} scratch_bytes=#{reserve - inputs.resident_bytes} mps_workspace_bytes=#{workspace_bytes} encode_metal_buffer_allocations=#{ML::MetalBuffer.video_spec_allocations - allocations_before_encode} terminal_waits=1 gpu_seconds_diagnostic_command=#{gpu_seconds} reference_kind=#{fixture["scope"]["reference_kind"].as_s}"
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
      if submission.nil? || submission.not_nil!.completed?
        begin
          activation_control.try { |control| control.release(submission) }
          submission.try(&.release)
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

  it "matches the declared upstream precision control, resident chaining and perturbation without encode allocations" do
    fixture_path = ENV["HY_NOMAS_BLOCK_FIXTURE"]?
    pending! "Set HY_NOMAS_BLOCK_FIXTURE to the external pinned MPS oracle JSON" unless fixture_path
    fixture = hy_load_fixture(fixture_path.not_nil!, ENV["HY_NOMAS_FIXTURE_SHA256"]? || "")
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
    failures = [] of String
    cases.each do |data|
      data["variants"].as_h.keys.sort.should eq(["bf16_weights_roundtrip_f32_compute", "f32"])
      data["variants"].as_h.each_value do |variant|
        failures.concat(hy_run_variant(fixture, data, variant, fixture_path.not_nil!))
      end
    end
    failures.empty?.should be_true, failures.first(16).join("\n")
  end
end
