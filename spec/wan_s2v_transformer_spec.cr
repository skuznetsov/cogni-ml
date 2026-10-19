require "spec"
require "json"
require "../src/ml/video/wan_s2v_transformer"
require "../src/ml/video/wan_s2v_memory"
require "../src/ml/video/wan_s2v_audio"
require "./support/video_allocation_counter"

private WAN_TRANSFORMER_REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"

private def wan_transformer_f32(values : JSON::Any) : Array(Float32)
  values.as_a.map(&.as_f.to_f32)
end

private def wan_transformer_gate(name : String, actual : Array(Float32), expected : JSON::Any) : Nil
  reference = wan_transformer_f32(expected)
  actual.size.should eq(reference.size), name
  violations = 0
  worst = 0.0_f64
  actual.each_with_index do |value, index|
    target = reference[index]
    ratio = (value - target).abs.to_f64 / (2e-5_f64 + 2e-5_f64 * target.abs.to_f64)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1.0
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

private def wan_transformer_cfg(c : JSON::Any, layers : Int32? = nil,
                                injection_ids : Array(Int32)? = nil,
                                drop_motion_frames : Bool = false) : ML::Video::WanS2VTransformerConfig
  ML::Video::WanS2VTransformerConfig.new(
    batches: c["batches"].as_i,
    context_tokens: c["context_tokens"].as_i,
    dim: c["dim"].as_i,
    heads: c["heads"].as_i,
    ffn_dim: c["ffn_dim"].as_i,
    layers: layers || c["layers"].as_i,
    video_frames: c["video_frames"].as_i,
    spatial_tokens: c["spatial_tokens"].as_i,
    latent_height: c["latent_height"].as_i,
    latent_width: c["latent_width"].as_i,
    injection_ids: injection_ids || c["injection_ids"].as_a.map(&.as_i),
    theta: c["theta"].as_f.to_f32,
    text_dim: c["text_dim"].as_i,
    text_tokens: c["text_tokens"].as_i,
    freq_dim: c["freq_dim"].as_i,
    in_dim: c["in_dim"].as_i,
    out_dim: c["out_dim"].as_i,
    audio_dim: c["audio_dim"].as_i,
    patch_tile_rows: c["patch_tile_rows"].as_i,
    drop_motion_frames: drop_motion_frames,
  )
end

describe ML::Video::WanS2VTransformerConfig do
  it "admits exact production geometry, binding coverage, and reusable scratch budgets" do
    cfg = ML::Video::WanS2VTransformerConfig.new
    cfg.video_tokens.should eq(32_760)
    cfg.reference_tokens.should eq(1_560)
    cfg.framepack_tokens.should eq(2_314)
    cfg.sequence_tokens.should eq(36_634)
    cfg.rows.should eq(36_634)
    cfg.injection_ids.should eq([0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39])

    block_bindings = ML::Video::WanS2VTransformer.checkpoint_bindings(cfg)
    block_bindings.size.should eq(1_248)
    block_bindings.map(&.source).uniq.size.should eq(block_bindings.size)
    block_bindings.map(&.target).uniq.size.should eq(block_bindings.size)
    block_bindings.all? { |binding| binding.dtype == ML::GGUF::SafeTensorDType::BF16 }.should be_true
    block_bindings.any? { |b| b.source == "blocks.39.modulation" && b.target == "block.39.block_modulation" }.should be_true
    block_bindings.any? { |b| b.source == "audio_injector.injector.11.q.weight" && b.target == "audio.11.audio_q_weight" }.should be_true
    block_bindings.any? { |b| b.source == "patch_embedding.weight" && b.shape == [5_242_880_i64] }.should be_false
    block_bindings.any? { |b| b.source == "patch_embedding.weight" && b.shape == [5_120_i64, 16_i64, 1_i64, 2_i64, 2_i64] }.should be_true
    block_bindings.any? { |b| b.source.starts_with?("casual_audio_encoder.") }.should be_false

    audio_cfg = ML::Video::WanS2VCausalAudioConfig.new
    audio_bindings = ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(audio_cfg)
    audio_bindings.size.should eq(12)
    audio_bindings.all? { |b| b.target.starts_with?("s2v_audio.") }.should be_true
    merged = block_bindings + audio_bindings
    merged.size.should eq(1_260)
    merged.map(&.source).uniq.size.should eq(merged.size)
    merged.map(&.target).uniq.size.should eq(merged.size)

    ML::Video::WanS2VTransformer.transformer_scratch_bytes(cfg).should eq(5_090_304_i64)
    block_scratch = ML::Video::WanS2VBlock.scratch_bytes(cfg.block_config)
    audio_scratch = ML::Video::WanS2VAudioInjector.scratch_bytes(cfg.audio_config)
    block_scratch.should eq(8_799_085_056_i64)
    audio_scratch.should eq(2_689_310_720_i64)
    (block_scratch + audio_scratch).should eq(11_488_395_776_i64)
    ML::Video::WanS2VBlock.unaliased_scratch_bytes(cfg.block_config).should eq(19_838_456_320_i64)
    ML::Video::WanS2VAudioInjector.unaliased_scratch_bytes(cfg.audio_config).should eq(4_704_235_520_i64)
    (ML::Video::WanS2VBlock.unaliased_scratch_bytes(cfg.block_config) +
      ML::Video::WanS2VAudioInjector.unaliased_scratch_bytes(cfg.audio_config)).should eq(24_542_691_840_i64)
  end

  it "rejects invalid patch, FramePack, injection, and RoPE geometry" do
    expect_raises(ArgumentError) { ML::Video::WanS2VTransformerConfig.new(dim: 33, heads: 4) }
    expect_raises(ArgumentError) { ML::Video::WanS2VTransformerConfig.new(latent_height: 7) }
    expect_raises(ArgumentError) { ML::Video::WanS2VTransformerConfig.new(injection_ids: [0, 0]) }
    expect_raises(ArgumentError) { ML::Video::WanS2VTransformerConfig.new(injection_ids: [40]) }
  end

  it "prepares one exclusive block and injector workspace with exact union accounting" do
    cfg = ML::Video::WanS2VTransformerConfig.new(
      context_tokens: 3, dim: 16, heads: 2, ffn_dim: 32, layers: 2,
      video_frames: 2, spatial_tokens: 16, latent_height: 8, latent_width: 8,
      injection_ids: [0, 1], text_dim: 4, text_tokens: 3, freq_dim: 4,
      in_dim: 4, out_dim: 2, audio_dim: 8, patch_tile_rows: 4)
    block = ML::Video::Arena.new
    audio = ML::Video::Arena.new
    begin
      expect_raises(ArgumentError) { ML::Video::WanS2VTransformer.prepare_reused_scratch!(block, block, cfg) }
      ML::Video::WanS2VTransformer.prepare_reused_scratch!(block, audio, cfg)
      ML::Video::WanS2VTransformer::REUSED_AUDIO_SCRATCH.each do |target, source|
        audio[target].same?(block[source]).should be_true
      end
      ML::Video::WanS2VTransformer.workspace_resident_bytes(block, audio).should eq(
        ML::Video::WanS2VTransformer.reused_scratch_bytes(cfg))
      ML::Video::WanS2VTransformer.reused_scratch_bytes(cfg).should be < (
        ML::Video::WanS2VBlock.scratch_bytes(cfg.block_config) +
        ML::Video::WanS2VAudioInjector.scratch_bytes(cfg.audio_config)
      )
      expect_raises(ArgumentError) { ML::Video::WanS2VTransformer.prepare_reused_scratch!(block, audio, cfg) }
    ensure
      audio.release
      block.release
    end
  end

  it "keeps setup-only position metadata ordered across noisy, reference, and FramePack suffixes" do
    cfg = ML::Video::WanS2VTransformerConfig.new(
      context_tokens: 3, dim: 12, heads: 2, ffn_dim: 24, layers: 2,
      video_frames: 2, spatial_tokens: 16, latent_height: 8, latent_width: 8,
      injection_ids: [] of Int32, text_dim: 4, text_tokens: 3, freq_dim: 4,
      in_dim: 4, out_dim: 2, audio_dim: 8, patch_tile_rows: 4,
    )
    positions = ML::Video::WanS2VTransformer.position_metadata(cfg)
    cfg.sequence_tokens.should eq(72)
    positions.size.should eq(cfg.sequence_tokens * 3)
    positions[0, 3].should eq([0.0_f32, 0.0_f32, 0.0_f32])
    positions[15 * 3, 3].should eq([0.0_f32, 3.0_f32, 3.0_f32])
    positions[16 * 3, 3].should eq([1.0_f32, 0.0_f32, 0.0_f32])
    positions[32 * 3, 3].should eq([30.0_f32, 0.0_f32, 0.0_f32])
    positions[48 * 3, 3].should eq([-1.0_f32, 0.0_f32, 0.0_f32])
    positions[64 * 3, 3].should eq([-3.0_f32, 0.0_f32, 0.0_f32])
    positions[68 * 3, 3].should eq([-19.0_f32, 0.0_f32, 0.0_f32])

    empty_cfg = ML::Video::WanS2VTransformerConfig.new(
      context_tokens: 3, dim: 12, heads: 2, ffn_dim: 24, layers: 2,
      video_frames: 2, spatial_tokens: 16, latent_height: 8, latent_width: 8,
      injection_ids: [] of Int32, theta: 10000.0_f32, text_dim: 4,
      text_tokens: 3, freq_dim: 4, in_dim: 4, out_dim: 2, audio_dim: 8,
      patch_tile_rows: 4, drop_motion_frames: true,
    )
    empty_cfg.framepack_tokens.should eq(0)
    empty_cfg.sequence_tokens.should eq(48)
    empty_positions = ML::Video::WanS2VTransformer.position_metadata(empty_cfg)
    empty_positions.size.should eq(empty_cfg.sequence_tokens * 3)
    empty_positions[32 * 3, 3].should eq([30.0_f32, 0.0_f32, 0.0_f32])
    empty_positions[47 * 3, 3].should eq([30.0_f32, 3.0_f32, 3.0_f32])
  end
end

describe ML::Video::WanS2VTransformer do
  it "matches a pinned two-layer raw-to-head forward with FramePack and layer traces" do
    fixture_path = ENV["WAN_S2V_TRANSFORMER_FIXTURE"]?
    unless fixture_path
      pending! "Set WAN_S2V_TRANSFORMER_FIXTURE to the pinned MPS transformer fixture"
    end
    fixture = JSON.parse(File.read(fixture_path))
    upstream = fixture["upstream"]
    upstream["revision"].as_s.should eq(WAN_TRANSFORMER_REVISION)
    upstream["device"].as_s.should eq("mps")
    upstream["arithmetic"].as_s.should eq("F32")
    upstream["fallback_disabled"].as_bool.should be_true
    cfg = wan_transformer_cfg(fixture["config"])
    one_layer_cfg = wan_transformer_cfg(fixture["config"], layers: 1, injection_ids: [0])
    cfg.sequence_tokens.should eq(72)
    cfg.layers.should eq(2)
    cfg.injection_ids.should eq([0, 1])

    bindings = ML::Video::WanS2VTransformer.checkpoint_bindings(cfg, ML::GGUF::SafeTensorDType::F32)
    fixture_weights = fixture["weights"].as_h
    bindings.map(&.source).sort.should eq(fixture_weights.keys.sort)
    fixture["weight_shapes"].as_h.each do |source, dims|
      binding = bindings.find! { |item| item.source == source }
      dims.as_a.map(&.as_i).should eq(binding.shape)
    end

    inputs = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    transformer_scratch = ML::Video::Arena.new
    block_scratch = ML::Video::Arena.new
    audio_scratch = ML::Video::Arena.new
    output = ML::Video::Arena.new
    begin
      fixture["inputs"].as_h.each do |name, values|
        inputs.upload(name, wan_transformer_f32(values))
      end
      bindings.each do |binding|
        weights.upload(binding.target, wan_transformer_f32(fixture_weights[binding.source]))
      end
      ML::Video::WanS2VTransformer.prepare!
      ML::Video::WanS2VTransformer.prepare_transformer_scratch!(transformer_scratch, cfg)
      if ENV["WAN_S2V_REUSE_SCRATCH"]? == "1"
        ML::Video::WanS2VTransformer.prepare_reused_scratch!(block_scratch, audio_scratch, cfg)
      else
        ML::Video::WanS2VTransformer.prepare_scratch!(block_scratch, audio_scratch, cfg)
      end
      ML::Video::WanS2VTransformer.transformer_scratch_bytes(cfg).should eq(transformer_scratch.resident_bytes)

      raw_sequence = output.alloc("raw_sequence", cfg.rows * cfg.dim)
      raw_context = output.alloc("raw_context", cfg.batches * cfg.text_tokens * cfg.dim)
      raw_text_hidden = output.alloc("raw_text_hidden", cfg.batches * cfg.text_tokens * cfg.dim)
      raw_time = output.alloc("raw_time", cfg.batches * cfg.dim)
      raw_modulation = output.alloc("raw_modulation", cfg.batches * 12 * cfg.dim)
      raw_command = ML::Metal::CommandBuffer.new
      raw_encoder = ML::Metal::ComputeEncoder.new(raw_command)
      ML::Video::WanS2VTransformer.encode_patch_inputs(
        raw_encoder, inputs["video_latents"], inputs["condition_latents"],
        inputs["reference_latents"], inputs["motion_latents"], raw_sequence,
        weights, transformer_scratch, cfg, fixture["config"]["motion_frames"].as_i,
      )
      ML::Video::WanS2VTransformer.encode_text(raw_encoder, inputs["raw_context"],
        raw_context, raw_text_hidden, weights, cfg)
      ML::Video::WanS2VTransformer.encode_time(raw_encoder, inputs["timestep"],
        raw_time, raw_modulation, weights, transformer_scratch, cfg)
      raw_encoder.end_encoding
      raw_command.commit
      raw_command.wait

      position_values = wan_transformer_f32(fixture["inputs"]["positions"])
      ML::Video::WanS2VTransformer.position_metadata(cfg).should eq(position_values)
      wan_transformer_gate("raw FramePack patch sequence", raw_sequence.read(cfg.rows * cfg.dim), fixture["expected"]["sequence"])
      wan_transformer_gate("raw projected UMT5 context", raw_context.read(cfg.batches * cfg.text_tokens * cfg.dim), fixture["expected"]["projected_context"])
      wan_transformer_gate("raw timestep embedding", raw_time.read(cfg.batches * cfg.dim), fixture["expected"]["time_embedding"])
      wan_transformer_gate("raw zero-time modulation", raw_modulation.read(cfg.batches * 12 * cfg.dim), fixture["expected"]["modulation"])

      positions = inputs["positions"]
      alternate = output.alloc("alternate", cfg.rows * cfg.dim)
      one_layer_alternate = output.alloc("one_layer_alternate", cfg.rows * cfg.dim)
      one_layer_trace_output = output.alloc("one_layer_trace_output", cfg.rows * cfg.dim)
      one_layer_trace = {"output" => one_layer_trace_output}
      one_layer_audio_trace_output = output.alloc("one_layer_audio_trace_output", cfg.rows * cfg.dim)
      one_layer_audio_trace = {"audio_output" => one_layer_audio_trace_output}
      local_audio = inputs["local_audio"]
      global_audio = inputs["global_audio"]

      future_weight_name = "block.1.self_q_weight"
      future_weight = weights.buffers[future_weight_name]
      rejected_stack = ML::Metal::CommandBuffer.new
      rejected_stack_encoder = ML::Metal::ComputeEncoder.new(rejected_stack)
      begin
        weights.buffers[future_weight_name] = block_scratch["wan_b_self_q"]
        expect_raises(ArgumentError, "Wan S2V buffer aliases weight #{future_weight_name}") do
          ML::Video::WanS2VTransformer.encode_layers(rejected_stack_encoder,
            raw_sequence, one_layer_alternate, raw_modulation, positions,
            raw_context, local_audio, global_audio, weights, block_scratch,
            audio_scratch, cfg)
        end
      ensure
        weights.buffers[future_weight_name] = future_weight
        rejected_stack_encoder.end_encoding
        rejected_stack.discard
      end

      layer_command = ML::Metal::CommandBuffer.new
      layer_encoder = ML::Metal::ComputeEncoder.new(layer_command)
      ML::Video::WanS2VTransformer.encode_layers(layer_encoder, raw_sequence, one_layer_alternate,
        raw_modulation, positions, raw_context, local_audio, global_audio, weights,
        block_scratch, audio_scratch, one_layer_cfg, 0, one_layer_trace,
        one_layer_audio_trace)
      layer_encoder.end_encoding
      layer_command.commit
      layer_command.wait
      wan_transformer_gate("distinct block 0 output", one_layer_trace_output.read(cfg.rows * cfg.dim), fixture["expected"]["block0_output"])
      wan_transformer_gate("audio injection 0 output", one_layer_audio_trace_output.read(cfg.rows * cfg.dim), fixture["expected"]["audio0_output"])

      sequence = output.alloc("sequence", cfg.rows * cfg.dim)
      projected_context = output.alloc("projected_context", cfg.batches * cfg.text_tokens * cfg.dim)
      text_hidden = output.alloc("text_hidden", cfg.batches * cfg.text_tokens * cfg.dim)
      time_embedding = output.alloc("time_embedding", cfg.batches * cfg.dim)
      modulation = output.alloc("modulation", cfg.batches * 12 * cfg.dim)
      prediction = output.alloc("prediction", cfg.batches * cfg.video_tokens * cfg.out_dim * 4)
      latent_output = output.alloc("latent_output", cfg.batches * cfg.out_dim * cfg.video_frames * cfg.latent_height * cfg.latent_width)
      block1_trace_output = output.alloc("block1_trace_output", cfg.rows * cfg.dim)
      block1_trace = {"output" => block1_trace_output}
      audio1_trace_output = output.alloc("audio1_trace_output", cfg.rows * cfg.dim)
      audio1_trace = {"audio_output" => audio1_trace_output}

      alias_buffer = output.alloc("alias_guard", Math.max(cfg.batches * cfg.in_dim * cfg.video_frames * cfg.latent_height * cfg.latent_width,
        cfg.rows * cfg.dim))
      rejected = ML::Metal::CommandBuffer.new
      rejected_encoder = ML::Metal::ComputeEncoder.new(rejected)
      # The public error contract is an ArgumentError for forbidden overlap;
      # keep the assertion independent of the kernel-specific detail string.
      expect_raises(ArgumentError) do
        ML::Video::WanS2VTransformer.encode_patch_inputs(
          rejected_encoder, alias_buffer, inputs["condition_latents"],
          inputs["reference_latents"], inputs["motion_latents"], alias_buffer,
          weights, transformer_scratch, cfg, fixture["config"]["motion_frames"].as_i,
        )
      end
      rejected_encoder.end_encoding
      rejected.discard

      full_command = ML::Metal::CommandBuffer.new
      full_encoder = ML::Metal::ComputeEncoder.new(full_command)
      before_encode = ML::MetalBuffer.video_spec_allocations
      ML::Video::WanS2VTransformer.encode(full_encoder,
        inputs["video_latents"], inputs["condition_latents"],
        inputs["reference_latents"], inputs["motion_latents"], inputs["raw_context"],
        text_hidden, projected_context, inputs["timestep"], time_embedding, modulation,
        positions, local_audio, global_audio, sequence, alternate, prediction,
        latent_output, weights, transformer_scratch, block_scratch, audio_scratch,
        cfg, fixture["config"]["motion_frames"].as_i,
        trace_layer: 1, block_trace: block1_trace, audio_trace: audio1_trace)
      full_encoder.end_encoding
      ML::MetalBuffer.video_spec_allocations.should eq(before_encode), "transformer forward must allocate no GPU buffers"
      full_command.commit
      full_command.wait

      wan_transformer_gate("full projected UMT5 context", projected_context.read(cfg.batches * cfg.text_tokens * cfg.dim), fixture["expected"]["projected_context"])
      wan_transformer_gate("full timestep embedding", time_embedding.read(cfg.batches * cfg.dim), fixture["expected"]["time_embedding"])
      wan_transformer_gate("full zero-time modulation", modulation.read(cfg.batches * 12 * cfg.dim), fixture["expected"]["modulation"])
      wan_transformer_gate("distinct block 1 output", block1_trace_output.read(cfg.rows * cfg.dim), fixture["expected"]["block1_output"])
      wan_transformer_gate("audio injection 1 output", audio1_trace_output.read(cfg.rows * cfg.dim), fixture["expected"]["audio1_output"])
      wan_transformer_gate("prediction head", prediction.read(cfg.batches * cfg.video_tokens * cfg.out_dim * 4), fixture["expected"]["prediction"])
      wan_transformer_gate("unpatchified latent", latent_output.read(cfg.batches * cfg.out_dim * cfg.video_frames * cfg.latent_height * cfg.latent_width), fixture["expected"]["latent_output"])

      # The official first request drops the entire FramePack suffix and
      # replaces the VAE-produced no-pose COND with COND[0] * 0. Keep this as
      # a separate full-forward gate so the retained-history path above stays
      # independently exercised.
      drop_cfg = wan_transformer_cfg(fixture["config"], drop_motion_frames: true)
      drop_expected = fixture["expected"]["drop_no_pose"]
      drop_cfg.framepack_tokens.should eq(0)
      drop_cfg.sequence_tokens.should eq(48)
      wan_transformer_gate("empty FramePack positions",
        ML::Video::WanS2VTransformer.position_metadata(drop_cfg), fixture["inputs"]["drop_positions"])

      drop_sequence = output.alloc("drop_sequence", drop_cfg.rows * drop_cfg.dim)
      # Capture the encoder output before the transformer layers overwrite the
      # ping-pong sequence buffers. This stays in the same command encoder as
      # the full forward, so the diagnostic adds no submit or wait.
      drop_initial_sequence = output.alloc("drop_initial_sequence", drop_cfg.rows * drop_cfg.dim)
      drop_alternate = output.alloc("drop_alternate", drop_cfg.rows * drop_cfg.dim)
      drop_text_hidden = output.alloc("drop_text_hidden", drop_cfg.batches * drop_cfg.text_tokens * drop_cfg.dim)
      drop_projected_context = output.alloc("drop_projected_context", drop_cfg.batches * drop_cfg.text_tokens * drop_cfg.dim)
      drop_time_embedding = output.alloc("drop_time_embedding", drop_cfg.batches * drop_cfg.dim)
      drop_modulation = output.alloc("drop_modulation", drop_cfg.batches * 12 * drop_cfg.dim)
      drop_prediction = output.alloc("drop_prediction", drop_cfg.batches * drop_cfg.video_tokens * drop_cfg.out_dim * 4)
      drop_latent_output = output.alloc("drop_latent_output", drop_cfg.batches * drop_cfg.out_dim * drop_cfg.video_frames * drop_cfg.latent_height * drop_cfg.latent_width)
      drop_block_trace_output = output.alloc("drop_block1_trace_output", drop_cfg.rows * drop_cfg.dim)
      drop_audio_trace_output = output.alloc("drop_audio1_trace_output", drop_cfg.rows * drop_cfg.dim)
      drop_block_trace = {"output" => drop_block_trace_output}
      drop_audio_trace = {"audio_output" => drop_audio_trace_output}
      before_drop = ML::MetalBuffer.video_spec_allocations
      drop_command = ML::Metal::CommandBuffer.new
      drop_encoder = ML::Metal::ComputeEncoder.new(drop_command)
      ML::Video::WanS2VTransformer.zero_no_pose_condition(
        drop_encoder, inputs["no_pose_condition_latents"], drop_cfg,
      )
      ML::Video::WanS2VTransformer.encode_patch_inputs(
        drop_encoder, inputs["video_latents"], inputs["no_pose_condition_latents"],
        inputs["reference_latents"], inputs["motion_latents"], drop_initial_sequence,
        weights, transformer_scratch, drop_cfg,
        fixture["config"]["motion_frames"].as_i,
      )
      ML::Video::WanS2VTransformer.encode(drop_encoder,
        inputs["video_latents"], inputs["no_pose_condition_latents"],
        inputs["reference_latents"], inputs["motion_latents"], inputs["raw_context"],
        drop_text_hidden, drop_projected_context, inputs["timestep"], drop_time_embedding,
        drop_modulation, inputs["drop_positions"], local_audio, global_audio,
        drop_sequence, drop_alternate, drop_prediction, drop_latent_output,
        weights, transformer_scratch, block_scratch, audio_scratch, drop_cfg,
        fixture["config"]["motion_frames"].as_i,
        trace_layer: 1, block_trace: drop_block_trace, audio_trace: drop_audio_trace)
      drop_encoder.end_encoding
      ML::MetalBuffer.video_spec_allocations.should eq(before_drop),
        "no-pose first-request forward must allocate no GPU buffers"
      drop_command.commit
      drop_command.wait

      wan_transformer_gate("zeroed no-pose condition",
        inputs["no_pose_condition_latents"].read(cfg.batches * cfg.in_dim * cfg.video_frames * cfg.latent_height * cfg.latent_width),
        fixture["expected"]["zero_condition"])
      wan_transformer_gate("empty FramePack no-pose sequence",
        drop_initial_sequence.read(drop_cfg.rows * drop_cfg.dim), drop_expected["sequence"])
      wan_transformer_gate("empty FramePack projected UMT5 context",
        drop_projected_context.read(drop_cfg.batches * drop_cfg.text_tokens * drop_cfg.dim),
        drop_expected["projected_context"])
      wan_transformer_gate("empty FramePack timestep embedding",
        drop_time_embedding.read(drop_cfg.batches * drop_cfg.dim), drop_expected["time_embedding"])
      wan_transformer_gate("empty FramePack zero-time modulation",
        drop_modulation.read(drop_cfg.batches * 12 * drop_cfg.dim), drop_expected["modulation"])
      wan_transformer_gate("empty FramePack block 1 output",
        drop_block_trace_output.read(drop_cfg.rows * drop_cfg.dim), drop_expected["block1_output"])
      wan_transformer_gate("empty FramePack audio 1 output",
        drop_audio_trace_output.read(drop_cfg.rows * drop_cfg.dim), drop_expected["audio1_output"])
      wan_transformer_gate("empty FramePack prediction head",
        drop_prediction.read(drop_cfg.batches * drop_cfg.video_tokens * drop_cfg.out_dim * 4),
        drop_expected["prediction"])
      wan_transformer_gate("empty FramePack unpatchified latent",
        drop_latent_output.read(drop_cfg.batches * drop_cfg.out_dim * drop_cfg.video_frames * drop_cfg.latent_height * drop_cfg.latent_width),
        drop_expected["latent_output"])
    ensure
      output.release
      audio_scratch.release
      block_scratch.release
      transformer_scratch.release
      weights.release
      inputs.release
    end
  end
end
