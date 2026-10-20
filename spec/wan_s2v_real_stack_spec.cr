require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/wan_s2v_transformer"
require "../src/ml/video/wan_s2v_memory"
require "../src/ml/video/wan_s2v_audio"
require "./support/video_allocation_counter"

private WAN_S2V_REAL_STACK_REVISION            = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
private WAN_S2V_REAL_STACK_CHECKPOINT_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
private WAN_S2V_REAL_STACK_MOTIONER_SHA256     = "db348a3ca0dfb22122a78aebdcf162ba76d4d4e79b7b763c684d819279ac957a"
private WAN_S2V_REAL_STACK_SOURCE_SHA256       = {
  "wan_modules_model.py"               => "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
  "wan_modules_s2v_model_s2v.py"       => "91fb14df44edab2d01b5634420b463c5f1563aecdf6dac905da6b3acac3145cc",
  "wan_modules_s2v_audio_utils.py"     => "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd",
  "wan_modules_s2v_s2v_utils.py"       => "e45ae6c274b17e4fdddcb0439b598897a37679b8934ebd71f87255f92d5f0b2f",
  "wan_modules_s2v_auxi_blocks.py"     => "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6",
  "diffusers-normalization-v0.31.0.py" => "7f2d2086f669e19bc008d71743f3963017274fbc6ad86d0ab9bf18db211e7c5a",
}
private WAN_S2V_REAL_STACK_INJECTIONS = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]

private def wan_real_stack_f32(values : JSON::Any) : Array(Float32)
  values.as_a.map(&.as_f.to_f32)
end

private def wan_real_stack_sha256(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |file|
    buffer = Bytes.new(1024 * 1024)
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.hexfinal
end

private def wan_real_stack_shape(values : JSON::Any) : Array(Int32)
  values.as_a.map(&.as_i64.to_i32)
end

private def wan_real_stack_count(shape : Array(Int32)) : Int32
  shape.reduce(1_i64) do |count, dimension|
    raise ArgumentError.new("Invalid real-stack trace shape") unless dimension > 0 && count <= Int32::MAX // dimension
    count * dimension
  end.to_i32
end

private def wan_real_stack_cfg(config : JSON::Any, drop_motion_frames : Bool = false) : ML::Video::WanS2VTransformerConfig
  ML::Video::WanS2VTransformerConfig.new(
    batches: config["batches"].as_i,
    context_tokens: config["context_tokens"].as_i,
    dim: config["dim"].as_i,
    heads: config["heads"].as_i,
    ffn_dim: config["ffn_dim"].as_i,
    layers: config["layers"].as_i,
    video_frames: config["video_frames"].as_i,
    spatial_tokens: config["spatial_tokens"].as_i,
    latent_height: config["latent_height"].as_i,
    latent_width: config["latent_width"].as_i,
    injection_ids: config["injection_ids"].as_a.map(&.as_i),
    theta: config["theta"].as_f.to_f32,
    text_dim: config["text_dim"].as_i,
    text_tokens: config["text_tokens"].as_i,
    freq_dim: config["freq_dim"].as_i,
    in_dim: config["in_dim"].as_i,
    out_dim: config["out_dim"].as_i,
    audio_dim: config["audio_dim"].as_i,
    patch_tile_rows: config["patch_tile_rows"].as_i,
    drop_motion_frames: drop_motion_frames,
  )
end

private def wan_real_stack_reference(file : ML::GGUF::SafetensorsFile,
                                     name : String, shape : Array(Int32)) : Array(Float32)
  info = file.tensor(name) || raise "Missing same-prefix real-stack input: #{name}"
  info.dtype.should eq(ML::GGUF::SafeTensorDType::F32), name
  info.shape.should eq(shape.map(&.to_i64)), name
  values = file.read_tensor_f32(info)
  values.all?(&.finite?).should be_true, name
  values
end

private def wan_real_stack_metrics(name : String, actual : Array(Float32), expected : Array(Float32))
  actual.size.should eq(expected.size), name
  violations = 0
  non_finite = 0
  max_abs = 0.0_f64
  worst = 0.0_f64
  worst_index = 0
  actual.each_with_index do |value, index|
    target = expected[index]
    unless value.finite? && target.finite?
      non_finite += 1
      violations += 1
      next
    end
    difference = (value.to_f64 - target.to_f64).abs
    max_abs = Math.max(max_abs, difference)
    ratio = difference / (2e-5_f64 + 2e-5_f64 * target.abs.to_f64)
    violations += 1 if ratio > 1.0
    if ratio > worst
      worst = ratio
      worst_index = index
    end
  end
  {name: name, values: actual.size, non_finite: non_finite, violations: violations,
   max_abs: max_abs, max_normalized_error: worst, worst_index: worst_index}
end

private def wan_real_stack_checkpoint_bindings(
  cfg : ML::Video::WanS2VTransformerConfig,
  fixture : JSON::Any,
) : Array(ML::Video::WeightBinding)
  bindings = ML::Video::WanS2VTransformer.checkpoint_bindings(
    cfg, ML::GGUF::SafeTensorDType::BF16,
  ).select do |binding|
    binding.source.starts_with?("blocks.") || binding.source.starts_with?("audio_injector.")
  end
  contract = fixture["weight_contract"].as_a.select do |entry|
    source = entry["source"].as_s
    source.starts_with?("blocks.") || source.starts_with?("audio_injector.")
  end
  expected_count = cfg.layers * 27 + cfg.injection_ids.size * 12
  bindings.size.should eq(expected_count)
  contract.size.should eq(expected_count)
  bindings.map(&.source).sort.should eq(contract.map { |entry| entry["source"].as_s }.sort)
  bindings.each do |binding|
    entry = contract.find! { |item| item["source"].as_s == binding.source }
    entry["target"].as_s.should eq(binding.target)
    wan_real_stack_shape(entry["shape"]).map(&.to_i64).should eq(binding.shape)
    ML::GGUF::SafeTensorDType.parse(entry["dtype"].as_s).should eq(ML::GGUF::SafeTensorDType::BF16)
    binding.dtype.should eq(ML::GGUF::SafeTensorDType::BF16)
  end
  bindings
end

private def wan_real_stack_encode_branch(
  encoder : ML::Metal::ComputeEncoder,
  branch : String,
  input : ML::MetalBuffer,
  alternate : ML::MetalBuffer,
  block_output : ML::MetalBuffer,
  modulation : ML::MetalBuffer,
  positions : ML::MetalBuffer,
  context : ML::MetalBuffer,
  local_audio : ML::MetalBuffer,
  global_audio : ML::MetalBuffer,
  weights : ML::Video::Arena,
  scratch : ML::Video::Arena,
  traces : Hash(String, ML::MetalBuffer),
  cfg : ML::Video::WanS2VTransformerConfig,
  tiled_attention : Bool,
) : ML::MetalBuffer
  block_cfg = cfg.block_config
  audio_cfg = cfg.audio_config
  injection_index = 0
  current = input
  spare = alternate
  cfg.layers.times do |layer|
    injected = injection_index < cfg.injection_ids.size && cfg.injection_ids[injection_index] == layer
    layer_key = layer.to_s.rjust(2, '0')
    block_trace = {"output" => traces["#{branch}.block.#{layer_key}"]}
    if injected
      audio_trace = {"audio_output" => traces["#{branch}.audio.#{injection_index.to_s.rjust(2, '0')}"]}
      ML::Video::WanS2VBlock.encode_with_audio(
        encoder, current, modulation, positions, context, local_audio, global_audio,
        block_output, spare, weights, scratch, block_cfg, audio_cfg,
        block_trace, audio_trace, "block.#{layer}.", "audio.#{injection_index}.", tiled_attention,
      )
      injection_index += 1
    else
      ML::Video::WanS2VBlock.encode(
        encoder, current, modulation, positions, context, spare, weights, scratch,
        block_cfg, block_trace, "block.#{layer}.", tiled_attention,
      )
    end
    current, spare = spare, current
  end
  injection_index.should eq(cfg.injection_ids.size), "#{branch} did not execute every official audio injection"
  current
end

describe ML::Video::WanS2VTransformer do
  it "propagates official BF16 checkpoint bindings through every distinct S2V block and injector" do
    fixture_path = ENV["WAN_S2V_REAL_STACK_FIXTURE"]?
    unless fixture_path
      pending! "Set WAN_S2V_REAL_STACK_FIXTURE to a streamed official MPS fixture (generate with --layers 40)"
    end
    checkpoint_dir = ENV["WAN_S2V_REAL_STACK_CHECKPOINT_DIR"]?
    unless checkpoint_dir
      pending! "Set WAN_S2V_REAL_STACK_CHECKPOINT_DIR to the pinned local Wan checkpoint directory"
    end
    budget_text = ENV["WAN_S2V_REAL_STACK_BUDGET_BYTES"]?
    unless budget_text
      pending! "Set WAN_S2V_REAL_STACK_BUDGET_BYTES to an explicit device-residency cap"
    end

    fixture = JSON.parse(File.read(fixture_path))
    upstream = fixture["upstream"]
    upstream["repo_revision"].as_s.should eq(WAN_S2V_REAL_STACK_REVISION)
    upstream["source_sha256"].as_h.each do |name, digest|
      WAN_S2V_REAL_STACK_SOURCE_SHA256[name]?.should eq(digest.as_s), name
    end
    upstream["source_sha256"].as_h.size.should eq(WAN_S2V_REAL_STACK_SOURCE_SHA256.size)
    upstream["motioner_source"].as_s.should eq("wan_modules_s2v_motioner.py")
    upstream["motioner_source_sha256"].as_s.should eq(WAN_S2V_REAL_STACK_MOTIONER_SHA256)
    upstream["device"].as_s.should eq("mps")
    upstream["neural_arithmetic"].as_s.should eq("F32")
    upstream["weight_upload"].as_s.should eq("original BF16 tensor upload, then MPS F32 conversion")
    upstream["fallback_disabled"].as_bool.should be_true
    upstream["oracle_layer_syncs"].as_i.should eq(fixture["config"]["layers"].as_i)
    upstream["activation_cpu_bounce"].as_bool.should be_false

    checkpoint = fixture["checkpoint"]
    checkpoint["repo"].as_s.should eq("Wan-AI/Wan2.2-S2V-14B")
    checkpoint["revision"].as_s.should eq(WAN_S2V_REAL_STACK_CHECKPOINT_REVISION)
    checkpoint["streamed_weight_dtype"].as_s.should eq("BF16")

    config_json = fixture["config"]
    cfg = wan_real_stack_cfg(config_json)
    cfg.validate!
    cfg.batches.should eq(1)
    cfg.dim.should eq(5_120)
    cfg.heads.should eq(40)
    cfg.ffn_dim.should eq(13_824)
    cfg.video_frames.should eq(2)
    cfg.spatial_tokens.should eq(16)
    cfg.latent_height.should eq(8)
    cfg.latent_width.should eq(8)
    [1, 2, 40].includes?(cfg.layers).should be_true
    selected_injections = WAN_S2V_REAL_STACK_INJECTIONS.select { |id| id < cfg.layers }
    cfg.injection_ids.should eq(selected_injections)
    config_json["official_layers"].as_i.should eq(40)
    bindings = wan_real_stack_checkpoint_bindings(cfg, fixture)

    trace_manifest = fixture["traces"]["manifest"].as_a
    selected_trace_names = [] of String
    (0...cfg.layers).each do |layer|
      key = layer.to_s.rjust(2, '0')
      selected_trace_names << "normal.block.#{key}"
      selected_trace_names << "drop.block.#{key}"
    end
    cfg.injection_ids.size.times do |injector|
      key = injector.to_s.rjust(2, '0')
      selected_trace_names << "normal.audio.#{key}"
      selected_trace_names << "drop.audio.#{key}"
    end
    trace_entries = {} of String => JSON::Any
    trace_manifest.each do |entry|
      name = entry["name"].as_s
      next unless selected_trace_names.includes?(name)
      trace_entries[name] = entry
    end
    trace_entries.keys.sort.should eq(selected_trace_names.sort)

    expected_traces = nil.as(ML::GGUF::SafetensorsFile?)
    checkpoint_files = [] of ML::Video::CheckpointFile
    inputs = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    outputs = ML::Video::Arena.new
    trace_buffers = {} of String => ML::MetalBuffer
    prepared = nil.as(ML::Video::PreparedWeights?)
    begin
      fixture_full_path = File.expand_path(fixture_path)
      traces_meta = fixture["traces"]
      traces_meta["format"].as_s.should eq("safetensors-f32")
      trace_filename = traces_meta["file"].as_s
      (File.basename(trace_filename) == trace_filename && trace_filename != "." && trace_filename != "..").should be_true
      trace_path = File.join(File.dirname(fixture_full_path), trace_filename)
      File.size(trace_path).should eq(traces_meta["bytes"].as_i64)
      wan_real_stack_sha256(trace_path).should eq(traces_meta["sha256"].as_s)
      expected_traces = ML::GGUF::SafetensorsFile.new(trace_path)

      index_filename = checkpoint["index_file"].as_s
      (File.basename(index_filename) == index_filename && index_filename != "." && index_filename != "..").should be_true
      index_path = File.join(checkpoint_dir, index_filename)
      wan_real_stack_sha256(index_path).should eq(checkpoint["index_sha256"].as_s)
      index_map = JSON.parse(File.read(index_path))["weight_map"].as_h

      files_by_name = {} of String => ML::Video::CheckpointFile
      fixture["weight_contract"].as_a.each do |entry|
        source = entry["source"].as_s
        next unless source.starts_with?("blocks.") || source.starts_with?("audio_injector.")
        shard = entry["shard"].as_s
        (File.basename(shard) == shard && shard != "." && shard != "..").should be_true
        index_map[source].as_s.should eq(shard)
        unless files_by_name.has_key?(shard)
          file = ML::Video::CheckpointFile.new(File.join(checkpoint_dir, shard))
          files_by_name[shard] = file
          checkpoint_files << file
        end
      end
      checkpoint_files.size.should be > 0
      drop_cfg = wan_real_stack_cfg(config_json, drop_motion_frames: true)
      drop_cfg.layers.should eq(cfg.layers)
      drop_cfg.injection_ids.should eq(cfg.injection_ids)
      normal_rows = cfg.rows
      drop_rows = drop_cfg.rows
      normal_values = normal_rows.to_i64 * cfg.dim
      drop_values = drop_rows.to_i64 * cfg.dim
      max_values = Math.max(normal_values, drop_values).to_i32

      # Prefix activations are captured in the hash-checked trace sidecar, not
      # the raw-input JSON. Feeding the exact oracle prefix isolates the stack
      # from independently rounded patch/text/time conditioning.
      prefix = expected_traces.not_nil!
      normal_sequence_values = wan_real_stack_reference(prefix, "normal.initial_sequence",
        [cfg.batches, cfg.sequence_tokens, cfg.dim])
      drop_sequence_values = wan_real_stack_reference(prefix, "drop.initial_sequence",
        [cfg.batches, drop_cfg.sequence_tokens, cfg.dim])
      normal_sequence_values.size.should eq(normal_values)
      drop_sequence_values.size.should eq(drop_values)
      inputs.upload("normal.sequence", normal_sequence_values)
      inputs.upload("drop.sequence", drop_sequence_values)

      normal_context = wan_real_stack_reference(prefix, "normal.projected_context",
        [cfg.batches, cfg.context_tokens, cfg.dim])
      drop_context = wan_real_stack_reference(prefix, "drop.projected_context",
        [cfg.batches, cfg.context_tokens, cfg.dim])
      drop_context.should eq(normal_context)
      inputs.upload("context", normal_context)

      normal_modulation = wan_real_stack_reference(prefix, "normal.modulation", [cfg.batches, 6, 2, cfg.dim])
      drop_modulation = wan_real_stack_reference(prefix, "drop.modulation", [cfg.batches, 6, 2, cfg.dim])
      drop_modulation.should eq(normal_modulation)
      inputs.upload("modulation", normal_modulation)

      positions = wan_real_stack_f32(fixture["inputs"]["positions"])
      drop_positions = wan_real_stack_f32(fixture["inputs"]["drop_positions"])
      positions.should eq(ML::Video::WanS2VTransformer.position_metadata(cfg))
      drop_positions.should eq(ML::Video::WanS2VTransformer.position_metadata(drop_cfg))
      inputs.upload("positions", positions)
      inputs.upload("drop_positions", drop_positions)
      inputs.upload("local_audio", wan_real_stack_f32(fixture["inputs"]["local_audio"]))
      inputs.upload("global_audio", wan_real_stack_f32(fixture["inputs"]["global_audio"]))

      scratch_cfg = cfg.block_config
      audio_scratch_cfg = cfg.audio_config
      ML::Video::WanS2VBlock.prepare!
      ML::Video::WanS2VAudioInjector.prepare!
      ML::Video::WanS2VBlock.prepare_scratch!(scratch, scratch_cfg)
      ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, audio_scratch_cfg)

      outputs.alloc("alternate", max_values)
      outputs.alloc("block_output", max_values)
      trace_entries.each do |name, entry|
        shape = wan_real_stack_shape(entry["shape"])
        expected_sequence = name.starts_with?("normal.") ? cfg.sequence_tokens : drop_cfg.sequence_tokens
        shape.should eq([1, expected_sequence, cfg.dim])
        entry["dtype"].as_s.should eq("F32")
        info = expected_traces.tensor(name) || raise "Expected real-stack trace missing: #{name}"
        info.shape.should eq(shape.map(&.to_i64))
        info.dtype.should eq(ML::GGUF::SafeTensorDType::F32)
      end

      # One reserve covers every non-weight Metal allocation plus explicit 15%
      # headroom for driver/pipeline residency. The test never guesses a budget.
      trace_entries.each do |name, entry|
        trace_buffers[name] = outputs.alloc("trace.#{name}", wan_real_stack_count(wan_real_stack_shape(entry["shape"])))
      end
      reserve_bytes = inputs.resident_bytes + scratch.resident_bytes + outputs.resident_bytes
      estimated_weight_bytes = bindings.sum do |binding|
        binding.shape.reduce(1_i64) { |product, dimension| product * dimension } * 2_i64
      end
      safety_reserve = Math.max(1_i64 * 1024 * 1024 * 1024,
        (estimated_weight_bytes * 15 + 99) // 100)
      reserve_bytes += safety_reserve
      budget_bytes = budget_text.to_i64
      budget_bytes.should be > 0
      plan = ML::Video::WeightPlan.new(checkpoint_files, bindings, budget_bytes,
        reserve_bytes, resident_dtype: ML::Video::WeightDType::BF16, require_all: false)
      plan.destination_bytes.should eq(estimated_weight_bytes)
      plan.staging_bytes.should eq(0_i64)
      plan.reserve_bytes.should eq(reserve_bytes)
      plan.peak_bytes.should be <= budget_bytes
      prepared = ML::Video::PreparedWeights.new(plan, weights)
      weights.resident_bytes.should eq(plan.destination_bytes)
      bindings.each do |binding|
        weights.dtype(binding.target).should eq(ML::Video::WeightDType::BF16)
      end

      positions_buffer = inputs["positions"]
      modulation_buffer = inputs["modulation"]
      context_buffer = inputs["context"]
      local_audio = inputs["local_audio"]
      global_audio = inputs["global_audio"]
      tiled_attention = ENV["WAN_S2V_TILED_ATTENTION"]? == "1"
      ML::Video::MetalOps.prepare_tiled_attention! if tiled_attention

      before_encode_allocations = ML::MetalBuffer.video_spec_allocations
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      prepared.not_nil!.encode(encoder)
      normal_final = wan_real_stack_encode_branch(encoder, "normal",
        inputs["normal.sequence"], outputs["alternate"], outputs["block_output"],
        modulation_buffer, positions_buffer, context_buffer, local_audio, global_audio,
        weights, scratch, trace_buffers, cfg, tiled_attention)
      drop_final = wan_real_stack_encode_branch(encoder, "drop",
        inputs["drop.sequence"], outputs["alternate"], outputs["block_output"],
        modulation_buffer, inputs["drop_positions"], context_buffer, local_audio, global_audio,
        weights, scratch, trace_buffers, drop_cfg, tiled_attention)
      normal_final.valid?.should be_true
      drop_final.valid?.should be_true
      encoder.end_encoding
      command.commit
      command.wait
      ML::MetalBuffer.video_spec_allocations.should eq(before_encode_allocations),
        "native encoding must not allocate Metal buffers"

      metrics = selected_trace_names.map do |name|
        info = expected_traces.tensor(name).not_nil!
        reference = expected_traces.read_tensor_f32(info)
        actual = trace_buffers[name].read(reference.size.to_i32)
        wan_real_stack_metrics(name, actual, reference)
      end
      if report_path = ENV["WAN_S2V_REAL_STACK_REPORT"]?
        File.write(report_path, {
          schema: "cogni.wan.real-stack-comparison.v1",
          fixture_sha256: wan_real_stack_sha256(fixture_path),
          traces_sha256: traces_meta["sha256"].as_s,
          source_revision: WAN_S2V_REAL_STACK_REVISION,
          checkpoint_revision: WAN_S2V_REAL_STACK_CHECKPOINT_REVISION,
          absolute_tolerance: 2e-5, relative_tolerance: 2e-5,
          all_values_finite: metrics.all? { |point| point[:non_finite] == 0 },
          strict_gate_passed: metrics.all? { |point| point[:violations] == 0 },
          tiled_attention: tiled_attention,
          points: metrics,
        }.to_json)
      end
      metrics.each do |point|
        point[:non_finite].should eq(0), point[:name]
        point[:violations].should eq(0),
          "#{point[:name]}: violations=#{point[:violations]}, max_abs=#{point[:max_abs]}, worst normalized error=#{point[:max_normalized_error]}"
      end
    ensure
      prepared.try(&.release)
      outputs.release
      scratch.release
      weights.release
      inputs.release
      checkpoint_files.each(&.close)
      expected_traces.try(&.close)
    end
  end
end
