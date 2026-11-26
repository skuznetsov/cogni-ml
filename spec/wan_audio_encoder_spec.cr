require "spec"
require "../src/ml/video/wan_audio_encoder"
require "../src/ml/video/metal_mps_command_chain"
require "digest/sha256"
require "./support/video_allocation_counter"

private def wan_audio_use_mps? : Bool
  ENV["WAN_AUDIO_USE_MPS"]? == "1"
end

private def wan_audio_use_mps_normalizer? : Bool
  ENV["WAN_AUDIO_USE_MPS_NORMALIZER"]? == "1"
end

private def wan_audio_use_mps_boundary_normalizer? : Bool
  ENV["WAN_AUDIO_USE_MPS_NORMALIZER_BOUNDARIES"]? == "1"
end

private def wan_audio_use_any_mps_normalizer? : Bool
  wan_audio_use_mps_normalizer? || wan_audio_use_mps_boundary_normalizer?
end

private def wan_audio_use_mps_gelu? : Bool
  ENV["WAN_AUDIO_USE_MPS_GELU"]? == "1"
end

private def wan_audio_use_mps_position? : Bool
  ENV["WAN_AUDIO_USE_MPS_POSITION"]? == "1"
end

private def wan_audio_run_pair(cfg : ML::Video::WanAudioEncoderConfig,
                               input : ML::MetalBuffer, output : ML::MetalBuffer,
                               repeated : ML::MetalBuffer,
                               weights : ML::Video::WanAudioEncoderWeights,
                               scratch : ML::Video::WanAudioEncoderScratch,
                               prepared : ML::Video::PreparedWeights,
                               plans : ML::Video::WanAudioMPSPlans?,
                               diagnostic : ML::Video::WanAudioEncoderTrace? = nil,
                               normalizer : ML::Video::WanAudioMPSNormalizationPlan? = nil,
                               gelu_plans : ML::Video::WanAudioMPSGeluPlans? = nil,
                               position_plan : ML::Video::WanAudioMPSPositionPlan? = nil) : Nil
  before = ML::MetalBuffer.video_spec_allocations
  if mps_plans = plans
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      active = ML::Metal::MPSCommandChain.new
      chain = active
      prepared.encode(active.encoder)
      ML::Video::WanAudioEncoder.encode(active, cfg, input, output, weights, scratch,
        mps_plans, diagnostic, normalizer, gelu_plans, position_plan)
      ML::Video::WanAudioEncoder.encode(active, cfg, input, repeated, weights, scratch,
        mps_plans, nil, normalizer, gelu_plans, position_plan)
      ML::MetalBuffer.video_spec_allocations.should eq(before),
        "MPS encode must allocate no GPU buffers"
      active.commit_and_wait
      active.wait_count.should eq(1), "MPS encode must use one terminal wait"
      active.segment_count.should be > 0, "MPS encode must submit prepared graph work"
      # Touch the plan after completion so its graph executables and bound
      # buffers remain strongly owned for the chain's entire GPU lifetime.
      mps_plans.validate!(cfg, weights, scratch)
      normalizer.try(&.validate!(cfg, input, scratch))
      gelu_plans.try(&.validate!(cfg, scratch))
      position_plan.try(&.validate!(cfg, weights, scratch))
    ensure
      if active = chain
        active.commit_and_wait unless active.completed?
        active.release if active.completed?
      end
    end
  elsif normalizer || gelu_plans || position_plan
    raise "MPS waveform normalization, GELU, and position plans require WAN_AUDIO_USE_MPS=1"
  else
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    prepared.encode(encoder)
    ML::Video::WanAudioEncoder.encode(encoder, cfg, input, output, weights, scratch, diagnostic)
    ML::Video::WanAudioEncoder.encode(encoder, cfg, input, repeated, weights, scratch)
    encoder.end_encoding
    ML::MetalBuffer.video_spec_allocations.should eq(before), "encode must allocate no GPU buffers"
    command.commit
    command.wait
  end
end

private def wan_audio_assert_gate(name : String, actual : Array(Float32), expected : JSON::Any) : Nil
  reference = expected.as_a
  wan_audio_assert_gate_values(name, actual, reference.map(&.as_f.to_f32))
end

private def wan_audio_assert_gate_values(name : String, actual : Array(Float32), reference : Array(Float32)) : Nil
  actual.size.should eq(reference.size), name
  worst = 0_f64
  violations = 0
  actual.each_with_index do |value, i|
    target = reference[i]
    ratio = (value - target).abs.to_f64 / (2e-5 + 2e-5 * target.abs)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

private def wan_audio_expected_tensor(file : ML::GGUF::SafetensorsFile, name : String) : Array(Float32)
  info = file.tensor(name)
  raise "production XLSR oracle is missing #{name}" unless info
  file.read_tensor_f32(info)
end

private def wan_audio_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |io|
    bytes = Bytes.new(1024 * 1024)
    while (size = io.read(bytes)) > 0
      digest.update(bytes[0, size])
    end
  end
  digest.hexfinal
end

private def wan_audio_config_matches_fixture(cfg : ML::Video::WanAudioEncoderConfig, c : JSON::Any) : Nil
  cfg.input_samples.should eq(c["input_samples"].as_i.to_i32)
  cfg.conv_dims.should eq(c["conv_dims"].as_a.map(&.as_i.to_i32))
  cfg.conv_kernels.should eq(c["conv_kernels"].as_a.map(&.as_i.to_i32))
  cfg.conv_strides.should eq(c["conv_strides"].as_a.map(&.as_i.to_i32))
  cfg.hidden_size.should eq(c["hidden_size"].as_i.to_i32)
  cfg.intermediate_size.should eq(c["intermediate_size"].as_i.to_i32)
  cfg.heads.should eq(c["heads"].as_i.to_i32)
  cfg.layers.should eq(c["layers"].as_i.to_i32)
  cfg.position_kernel.should eq(c["position_kernel"].as_i.to_i32)
  cfg.position_groups.should eq(c["position_groups"].as_i.to_i32)
  cfg.feature_norm.should eq(c["feature_norm"].as_s)
  cfg.stable_layer_norm.should eq(c["stable_layer_norm"].as_bool)
  cfg.layer_norm_eps.should eq(c["eps"].as_f.to_f32)
  cfg.video_fps.should eq(c["video_fps"].as_i.to_i32)
  cfg.target_fps.should eq(c["target_fps"].as_i.to_i32)
  cfg.target_frames.should eq(c["target_frames"].as_i.to_i32)
end

private def wan_audio_required_trace_names(cfg : ML::Video::WanAudioEncoderConfig) : Array(String)
  names = [
    "normalized_waveform", "feature_conv_0_raw", "feature_conv_0_norm", "feature_conv_1_raw",
    "feature_norm", "feature_projection", "position_scale", "position_conv_raw",
    "position_conv", "position_add", "pre_attention_norm", "query", "key", "value",
    "attention", "attention_projection", "attention_residual", "ffn_norm",
    "ffn_pre_gelu", "ffn_post_gelu", "ffn_output", "layer_output",
  ]
  cfg.conv_dims.size.times { |i| names << "feature_conv_#{i}" }
  names
end

private def wan_audio_read_trace(trace : ML::Video::WanAudioEncoderTrace,
                                 cfg : ML::Video::WanAudioEncoderConfig) : Hash(String, Array(Float32))
  actual = {
    "normalized_waveform"  => trace.normalized_waveform.read(cfg.input_samples),
    "feature_conv_0_raw"   => trace.first_conv_raw.read(cfg.convolution_output_lengths[0] * cfg.conv_dims[0]),
    "feature_conv_0_norm"  => trace.first_conv_norm.read(cfg.convolution_output_lengths[0] * cfg.conv_dims[0]),
    "feature_conv_1_raw"   => trace.second_conv_raw.read(cfg.convolution_output_lengths[1] * cfg.conv_dims[1]),
    "feature_norm"         => trace.feature_norm.read(cfg.feature_frames * cfg.conv_dims.last),
    "feature_projection"   => trace.feature_projection.read(cfg.sequence_values),
    "position_scale"       => trace.position_scale.read(cfg.position_kernel),
    "position_conv_raw"    => trace.position_conv_raw.read(cfg.sequence_values),
    "position_conv"        => trace.position_conv.read(cfg.sequence_values),
    "position_add"         => trace.position_add.read(cfg.sequence_values),
    "pre_attention_norm"   => trace.pre_attention_norm.read(cfg.sequence_values),
    "query"                => trace.query.read(cfg.sequence_values),
    "key"                  => trace.key.read(cfg.sequence_values),
    "value"                => trace.value.read(cfg.sequence_values),
    "attention"            => trace.attention.read(cfg.sequence_values),
    "attention_projection" => trace.attention_projection.read(cfg.sequence_values),
    "attention_residual"   => trace.attention_residual.read(cfg.sequence_values),
    "ffn_norm"             => trace.ffn_norm.read(cfg.sequence_values),
    "ffn_pre_gelu"         => trace.ffn_pre_gelu.read(cfg.ffn_values),
    "ffn_post_gelu"        => trace.ffn_post_gelu.read(cfg.ffn_values),
    "ffn_output"           => trace.ffn_output.read(cfg.sequence_values),
    "layer_output"         => trace.layer_output.read(cfg.sequence_values),
  }
  cfg.convolution_output_lengths.each_with_index do |length, i|
    actual["feature_conv_#{i}"] = trace.feature_convolutions[i].read(length * cfg.conv_dims[i])
  end
  actual
end

describe ML::Video::WanAudioEncoderConfig do
  it "matches the pinned five-second XLSR and Wan audio bucket geometry" do
    cfg = ML::Video::WanAudioEncoderConfig.new
    cfg.validate!
    cfg.convolution_output_lengths.should eq([15_999, 7_999, 3_999, 1_999, 999, 499, 249])
    cfg.feature_rate.should eq(50)
    cfg.interpolation_frames.should eq(149)
    cfg.bucket_repeats.should eq(1)
    cfg.bucket_frames.should eq(84)
    cfg.padded_interpolation_frames.should eq(9)
    cfg.bucket_source_index(79).should eq(148)
    cfg.bucket_source_index(80).should eq(150)
    cfg.bucket_source_index(83).should eq(156)
    cfg.bucket_source_valid?(79).should be_true
    cfg.bucket_source_valid?(80).should be_false
    # NumPy round-to-even behavior in the pinned Wan sampler.
    cfg.bucket_source_index(4).should eq(8)
    cfg.bucket_source_index(12).should eq(22)
    cfg.output_values.should eq(25 * 1_024 * 84)
    cfg.scratch_bytes.should eq(83_510_792_i64)
  end

  it "derives operator bindings from the real XLSR JSON without binding CTC logits" do
    cfg = ML::Video::WanAudioEncoderConfig.new
    bindings = ML::Video::WanAudioEncoder.checkpoint_bindings(cfg)
    bindings.any? { |b| b.source == "wav2vec2.feature_extractor.conv_layers.0.conv.weight" && b.shape == [512_i64, 1_i64, 10_i64] }.should be_true
    bindings.any? { |b| b.source == "wav2vec2.encoder.pos_conv_embed.conv.weight_g" && b.shape == [1_i64, 1_i64, 128_i64] }.should be_true
    bindings.any? { |b| b.source == "wav2vec2.encoder.layers.23.feed_forward.output_dense.weight" && b.shape == [1_024_i64, 4_096_i64] }.should be_true
    bindings.any? { |b| b.source == "wav2vec2.encoder.layer_norm.weight" && b.shape == [1_024_i64] }.should be_true
    bindings.any? { |b| b.source.starts_with?("wav2vec2.lm_head.") }.should be_false
    bindings.any? { |b| b.source == "wav2vec2.masked_spec_embed" }.should be_false
  end

  it "validates the local real config and safetensors header when the pinned assets are present" do
    config_path = ENV["WAN_AUDIO_XLSR_CONFIG"]? ||
                  "/opt/mldata_ssd/cogni-ml/video-scene-20261019/checkpoints/wan/wav2vec2-large-xlsr-53-english/config.json"
    model_path = ENV["WAN_AUDIO_XLSR_MODEL"]? ||
                 "/opt/mldata_ssd/cogni-ml/video-scene-20261019/checkpoints/wan/wav2vec2-large-xlsr-53-english/model.safetensors"
    unless File.exists?(config_path) && File.exists?(model_path)
      pending!("Pinned local XLSR config and safetensors are unavailable")
    end

    cfg = ML::Video::WanAudioEncoderConfig.from_hf_json(config_path)
    cfg.convolution_output_lengths.should eq([15_999, 7_999, 3_999, 1_999, 999, 499, 249])
    cfg.interpolation_frames.should eq(149)
    cfg.bucket_frames.should eq(84)

    file = ML::Video::CheckpointFile.new(model_path)
    begin
      plan = ML::Video::WeightPlan.new([file], ML::Video::WanAudioEncoder.checkpoint_bindings(cfg),
        4_i64 * 1024 * 1024 * 1024, resident_dtype: ML::Video::WeightDType::F32)
      plan.entries.size.should eq(ML::Video::WanAudioEncoder.checkpoint_bindings(cfg).size)
      plan.destination_bytes.should be > 1_000_000_000_i64
      plan.staging_bytes.should be > 1_000_000_000_i64
    ensure
      file.close
    end
  end
end

describe ML::Video::WanAudioEncoder do
  it "keeps a source-derived tiny architecture within the same operator contract" do
    cfg = ML::Video::WanAudioEncoderConfig.new(
      input_samples: 16_000,
      conv_dims: [8, 8, 8, 8, 8, 8, 8],
      hidden_size: 16,
      intermediate_size: 32,
      heads: 4,
      layers: 2,
      position_kernel: 8,
      position_groups: 4,
      target_frames: 6
    )
    cfg.validate!
    cfg.feature_frames.should eq(49)
    cfg.interpolation_frames.should eq(29)
    cfg.hidden_state_count.should eq(3)
    cfg.bucket_repeats.should eq(3)
    cfg.bucket_frames.should eq(18)
    cfg.padded_interpolation_frames.should eq(5)
    ML::Video::WanAudioEncoder::HF_TRANSFORMERS_VERSION.should eq("4.57.3")
    ML::Video::WanAudioEncoder::HF_MODEL_REVISION.should eq("dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b")
    ML::Video::WanAudioEncoder::HF_MODELING_SOURCE_SHA256.should eq("1c19d14f99c623c71aa8a058235978a06a8af4defb5e1f54321f7a6b7984e572")
  end

  it "matches every captured hidden state with repeated buckets and reuses resident scratch" do
    fixture_path = ENV["WAN_AUDIO_FIXTURE"]?
    unless fixture_path
      pending!("Set WAN_AUDIO_FIXTURE to a tiny pinned Wav2Vec2 reference fixture")
    end
    fixture = JSON.parse(File.read(fixture_path))
    fixture["transformers_version"].as_s.should eq(ML::Video::WanAudioEncoder::HF_TRANSFORMERS_VERSION)
    fixture["model_revision"].as_s.should eq(ML::Video::WanAudioEncoder::HF_MODEL_REVISION)
    fixture["modeling_source_sha256"].as_s.should eq(ML::Video::WanAudioEncoder::HF_MODELING_SOURCE_SHA256)
    fixture["device"].as_s.should eq("mps")
    fixture["arithmetic"].as_s.should eq("F32")
    weights_path = fixture["weights"].as_s
    Digest::SHA256.hexdigest(File.read(weights_path)).should eq(fixture["weight_sha256"].as_s)

    c = fixture["config"]
    cfg = ML::Video::WanAudioEncoderConfig.new(
      input_samples: c["input_samples"].as_i,
      conv_dims: c["conv_dims"].as_a.map(&.as_i),
      conv_kernels: c["conv_kernels"].as_a.map(&.as_i),
      conv_strides: c["conv_strides"].as_a.map(&.as_i),
      hidden_size: c["hidden_size"].as_i,
      intermediate_size: c["intermediate_size"].as_i,
      heads: c["heads"].as_i,
      layers: c["layers"].as_i,
      position_kernel: c["position_kernel"].as_i,
      position_groups: c["position_groups"].as_i,
      feature_norm: c["feature_norm"].as_s,
      stable_layer_norm: c["stable_layer_norm"].as_bool,
      layer_norm_eps: c["eps"].as_f.to_f32,
      video_fps: c["video_fps"].as_i,
      target_fps: c["target_fps"].as_i,
      target_frames: c["target_frames"].as_i
    )
    cfg.validate!
    cfg.feature_frames.should eq(fixture["feature_frames"].as_i.to_i32)
    cfg.interpolation_frames.should eq(fixture["interpolation_frames"].as_i.to_i32)
    cfg.bucket_repeats.should eq(fixture["bucket_repeats"].as_i.to_i32)
    cfg.bucket_frames.should eq(fixture["bucket_frames"].as_i.to_i32)
    cfg.padded_interpolation_frames.should eq(fixture["padded_interpolation_frames"].as_i.to_i32)
    cfg.bucket_frames.should be > cfg.target_frames
    cfg.bucket_source_valid?(cfg.bucket_frames - 1).should be_false

    file = ML::Video::CheckpointFile.new(weights_path)
    arena = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    mps_plans = nil.as(ML::Video::WanAudioMPSPlans?)
    mps_normalizer = nil.as(ML::Video::WanAudioMPSNormalizationPlan?)
    mps_gelu = nil.as(ML::Video::WanAudioMPSGeluPlans?)
    mps_position = nil.as(ML::Video::WanAudioMPSPositionPlan?)
    begin
      bindings = ML::Video::WanAudioEncoder.checkpoint_bindings(cfg)
      plan = ML::Video::WeightPlan.new([file], bindings,
        512_i64 * 1024 * 1024, cfg.scratch_bytes,
        resident_dtype: ML::Video::WeightDType::F32)
      plan.entries.size.should eq(bindings.size), "the CTC head and inference-unused mask embedding are excluded"
      prepared = ML::Video::PreparedWeights.new(plan, arena)
      weights = ML::Video::WanAudioEncoder.bind_weights(arena, cfg)
      scratch = ML::Video::WanAudioEncoder.prepare_scratch!(arena, cfg)
      if wan_audio_use_mps_gelu?
        raise "WAN_AUDIO_USE_MPS_GELU requires WAN_AUDIO_USE_MPS=1" unless wan_audio_use_mps?
      end
      mps_plans = ML::Video::WanAudioMPSPlans.new(cfg, weights, scratch,
        graph_gelu: wan_audio_use_mps_gelu?) if wan_audio_use_mps?
      if wan_audio_use_mps_gelu?
        mps_gelu = ML::Video::WanAudioMPSGeluPlans.new(cfg, scratch)
      end
      if wan_audio_use_mps_position?
        raise "WAN_AUDIO_USE_MPS_POSITION requires WAN_AUDIO_USE_MPS=1" unless mps_plans
        mps_position = ML::Video::WanAudioMPSPositionPlan.new(cfg, weights, scratch)
      end
      if wan_audio_use_mps_normalizer? && wan_audio_use_mps_boundary_normalizer?
        raise "Select only one Wan MPS waveform normalizer mode"
      end
      if wan_audio_use_any_mps_normalizer? && !mps_plans
        raise "Wan MPS waveform normalizers require WAN_AUDIO_USE_MPS=1"
      end
      diagnostic = ML::Video::WanAudioEncoder.prepare_trace!(arena, cfg)
      input = arena.upload("audio.input", fixture["input_values"].as_a.map(&.as_f.to_f32))
      if wan_audio_use_any_mps_normalizer?
        mps_normalizer = ML::Video::WanAudioMPSNormalizationPlan.new(cfg, input, scratch,
          source_boundaries: wan_audio_use_mps_boundary_normalizer?)
      end
      output = arena.alloc("audio.output", cfg.output_values)
      repeated = arena.alloc("audio.repeated", cfg.output_values)

      # The record keeps mutable config arrays by reference; setup signatures
      # must catch a changed array before any encoder command is emitted.
      original_first_conv = cfg.conv_dims[0]
      cfg.conv_dims[0] = original_first_conv + 1
      guard_before = ML::MetalBuffer.video_spec_allocations
      rejected_command = ML::Metal::CommandBuffer.new
      rejected_encoder = ML::Metal::ComputeEncoder.new(rejected_command)
      expect_raises(ArgumentError, "config changed after") do
        ML::Video::WanAudioEncoder.encode(rejected_encoder, cfg, input, output, weights, scratch)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(guard_before)
      cfg.conv_dims[0] = original_first_conv
      rejected_encoder.end_encoding

      wan_audio_run_pair(cfg, input, output, repeated, weights, scratch, prepared,
        mps_plans, diagnostic, mps_normalizer, mps_gelu, mps_position)

      actual = output.read(cfg.output_values)
      expected = fixture["expected"]
      actual_trace = wan_audio_read_trace(diagnostic, cfg)
      expected_trace = expected["trace"].as_h
      required_trace_names = wan_audio_required_trace_names(cfg)
      expected_trace.keys.sort.should eq(required_trace_names.sort),
        "the pinned oracle must expose every first-block diagnostic checkpoint"
      actual_trace.keys.sort.should eq(required_trace_names.sort),
        "native capture must expose the same complete first-block checkpoint set"
      if actual_path = ENV["WAN_AUDIO_ACTUAL"]?
        File.write(actual_path, {
          "output" => actual,
          "trace"  => actual_trace,
        }.to_json)
      end
      actual_trace.each do |name, values|
        wan_audio_assert_gate("trace.#{name}", values, expected_trace[name])
      end
      layer_values = cfg.hidden_size * cfg.bucket_frames
      cfg.hidden_state_count.times do |i|
        layer = actual[i * layer_values, layer_values]
        wan_audio_assert_gate("hidden_state_#{i}", layer, expected["hidden_state_#{i}"])
      end
      wan_audio_assert_gate("output", actual, expected["output"])
      repeated.read(cfg.output_values).map(&.unsafe_as(UInt32)).should eq(actual.map(&.unsafe_as(UInt32)))
    ensure
      mps_position.try(&.release)
      mps_gelu.try(&.release)
      mps_normalizer.try(&.release)
      mps_plans.try(&.release)
      prepared.try(&.release)
      arena.release
      file.close
    end
  end

  it "matches the real XLSR checkpoint at production widths on MPS" do
    fixture_path = ENV["WAN_AUDIO_PRODUCTION_FIXTURE"]?
    unless fixture_path
      pending!("Set WAN_AUDIO_PRODUCTION_FIXTURE to a real-checkpoint MPS fixture")
    end
    fixture = JSON.parse(File.read(fixture_path))
    fixture["mode"].as_s.should eq("production")
    fixture["transformers_version"].as_s.should eq(ML::Video::WanAudioEncoder::HF_TRANSFORMERS_VERSION)
    fixture["model_revision"].as_s.should eq(ML::Video::WanAudioEncoder::HF_MODEL_REVISION)
    fixture["modeling_source_sha256"].as_s.should eq(ML::Video::WanAudioEncoder::HF_MODELING_SOURCE_SHA256)
    fixture["device"].as_s.should eq("mps")
    fixture["arithmetic"].as_s.should eq("F32")

    model_path = ENV["WAN_AUDIO_XLSR_MODEL"]? || fixture["weights"].as_s
    config_path = ENV["WAN_AUDIO_XLSR_CONFIG"]? || File.join(File.dirname(model_path), "config.json")
    File.exists?(config_path).should be_true
    File.exists?(model_path).should be_true
    File.size(model_path).should eq(fixture["checkpoint_bytes"].as_i64)
    fixture["checkpoint"].as_s.should eq(File.expand_path(model_path))

    c = fixture["config"]
    cfg = ML::Video::WanAudioEncoderConfig.from_hf_json(config_path,
      input_samples: c["input_samples"].as_i.to_i32,
      target_frames: c["target_frames"].as_i.to_i32)
    wan_audio_config_matches_fixture(cfg, c)
    cfg.validate!
    cfg.feature_frames.should eq(fixture["feature_frames"].as_i)
    cfg.interpolation_frames.should eq(fixture["interpolation_frames"].as_i)
    cfg.bucket_repeats.should eq(fixture["bucket_repeats"].as_i)
    cfg.bucket_frames.should eq(fixture["bucket_frames"].as_i)
    cfg.padded_interpolation_frames.should eq(fixture["padded_interpolation_frames"].as_i)
    cfg.output_values.to_i64.should eq(fixture["output_shape"].as_a.reduce(1_i64) { |p, v| p * v.as_i64 })

    expected_path = fixture["expected_safetensors"].as_s
    wan_audio_sha256_file(expected_path).should eq(fixture["expected_safetensors_sha256"].as_s)
    expected_file = ML::GGUF::SafetensorsFile.new(expected_path)
    file = ML::Video::CheckpointFile.new(model_path)
    arena = ML::Video::Arena.new
    traces = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    mps_plans = nil.as(ML::Video::WanAudioMPSPlans?)
    mps_normalizer = nil.as(ML::Video::WanAudioMPSNormalizationPlan?)
    mps_gelu = nil.as(ML::Video::WanAudioMPSGeluPlans?)
    mps_position = nil.as(ML::Video::WanAudioMPSPositionPlan?)
    position_probe_file = nil.as(ML::GGUF::SafetensorsFile?)
    begin
      expected_keys = ["output"] + (0..cfg.layers).map { |i| "hidden_state_#{i}" }
      required_trace_names = wan_audio_required_trace_names(cfg)
      expected_keys.concat(required_trace_names.map { |name| "trace.#{name}" })
      expected_file.tensors.map(&.name).sort.should eq(expected_keys.sort)

      bindings = ML::Video::WanAudioEncoder.checkpoint_bindings(cfg)
      budget = fixture["memory_budget_bytes"].as_i64
      feature_conv_trace_values = cfg.convolution_output_lengths.each_with_index.sum(0_i64) do |length, i|
        length.to_i64 * cfg.conv_dims[i]
      end
      feature_trace_values = cfg.input_samples.to_i64 + feature_conv_trace_values +
                             2_i64 * cfg.convolution_output_lengths[0] * cfg.conv_dims[0] +
                             cfg.convolution_output_lengths[1].to_i64 * cfg.conv_dims[1] +
                             cfg.feature_frames.to_i64 * cfg.conv_dims.last +
                             cfg.position_kernel.to_i64 + 4_i64 * cfg.sequence_values
      trace_values = 10_i64 * cfg.sequence_values + 2_i64 * cfg.ffn_values + feature_trace_values
      reserve_bytes = cfg.scratch_bytes +
                      cfg.input_samples.to_i64 * sizeof(Float32) +
                      2_i64 * cfg.output_values * sizeof(Float32) +
                      trace_values * sizeof(Float32)
      # The CTC head is deliberately present in this safetensors file but is
      # not part of Wan's hidden-state conditioning path.
      plan = ML::Video::WeightPlan.new([file], bindings, budget, reserve_bytes,
        resident_dtype: ML::Video::WeightDType::F32)
      plan.entries.size.should eq(bindings.size)
      plan.destination_bytes.should be > 1_000_000_000_i64
      plan.staging_bytes.should be > 1_000_000_000_i64
      plan.peak_bytes.should be <= budget, "real XLSR weights, staging, scratch, input/output and complete trace exceed the fixture budget"
      plan.reserve_bytes.should eq(reserve_bytes)

      prepared = ML::Video::PreparedWeights.new(plan, arena)
      weights = ML::Video::WanAudioEncoder.bind_weights(arena, cfg)
      scratch = ML::Video::WanAudioEncoder.prepare_scratch!(arena, cfg)
      if wan_audio_use_mps_gelu?
        raise "WAN_AUDIO_USE_MPS_GELU requires WAN_AUDIO_USE_MPS=1" unless wan_audio_use_mps?
      end
      mps_plans = ML::Video::WanAudioMPSPlans.new(cfg, weights, scratch,
        graph_gelu: wan_audio_use_mps_gelu?) if wan_audio_use_mps?
      if wan_audio_use_mps_gelu?
        mps_gelu = ML::Video::WanAudioMPSGeluPlans.new(cfg, scratch)
      end
      if wan_audio_use_mps_position?
        raise "WAN_AUDIO_USE_MPS_POSITION requires WAN_AUDIO_USE_MPS=1" unless mps_plans
        mps_position = ML::Video::WanAudioMPSPositionPlan.new(cfg, weights, scratch)
      end
      if wan_audio_use_mps_normalizer? && wan_audio_use_mps_boundary_normalizer?
        raise "Select only one Wan MPS waveform normalizer mode"
      end
      if wan_audio_use_any_mps_normalizer? && !mps_plans
        raise "Wan MPS waveform normalizers require WAN_AUDIO_USE_MPS=1"
      end
      diagnostic = ML::Video::WanAudioEncoder.prepare_trace!(traces, cfg)
      input = arena.upload("audio.production.input", fixture["input_values"].as_a.map(&.as_f.to_f32))
      if wan_audio_use_any_mps_normalizer?
        mps_normalizer = ML::Video::WanAudioMPSNormalizationPlan.new(cfg, input, scratch,
          source_boundaries: wan_audio_use_mps_boundary_normalizer?)
      end
      output = arena.alloc("audio.production.output", cfg.output_values)
      repeated = arena.alloc("audio.production.repeated", cfg.output_values)

      wan_audio_run_pair(cfg, input, output, repeated, weights, scratch, prepared,
        mps_plans, diagnostic, mps_normalizer, mps_gelu, mps_position)
      (plan.staging_bytes + arena.resident_bytes + traces.resident_bytes).should be <= budget,
        "resident production tensors exceed the fixture memory cap"

      actual = output.read(cfg.output_values)
      trace_actual = wan_audio_read_trace(diagnostic, cfg)
      trace_actual.keys.sort.should eq(required_trace_names.sort)
      if actual_path = ENV["WAN_AUDIO_ACTUAL"]?
        File.write(actual_path, {"output" => actual, "trace" => trace_actual}.to_json)
      end

      if position_probe_path = ENV["WAN_AUDIO_POSITION_PROBE"]?
        native_actual_path = ENV["WAN_AUDIO_ACTUAL"]?
        raise "WAN_AUDIO_POSITION_PROBE requires WAN_AUDIO_ACTUAL for input identity" unless native_actual_path
        position_probe_file = ML::GGUF::SafetensorsFile.new(position_probe_path)
        position_probe_file.metadata["transformers_version"].should eq(ML::Video::WanAudioEncoder::HF_TRANSFORMERS_VERSION)
        position_probe_file.metadata["model_revision"].should eq(ML::Video::WanAudioEncoder::HF_MODEL_REVISION)
        position_probe_file.metadata["modeling_source_sha256"].should eq(ML::Video::WanAudioEncoder::HF_MODELING_SOURCE_SHA256)
        position_probe_file.metadata["arithmetic"].should eq("F32")
        position_probe_file.metadata["device"].should eq("mps")
        position_probe_file.metadata["samepad"].should eq("symmetric K//2 conv then crop final value for even kernels")
        position_probe_file.metadata["native_capture_sha256"].should eq(wan_audio_sha256_file(native_actual_path))
        wan_audio_assert_gate_values("same-input.position_conv_raw", trace_actual["position_conv_raw"],
          wan_audio_expected_tensor(position_probe_file, "position_conv_raw"))
        wan_audio_assert_gate_values("same-input.position_conv", trace_actual["position_conv"],
          wan_audio_expected_tensor(position_probe_file, "position_conv"))
      end
      trace_actual.each do |name, values|
        wan_audio_assert_gate_values("trace.#{name}", values,
          wan_audio_expected_tensor(expected_file, "trace.#{name}"))
      end
      layer_values = cfg.hidden_size * cfg.bucket_frames
      cfg.hidden_state_count.times do |i|
        layer = actual[i * layer_values, layer_values]
        wan_audio_assert_gate_values("hidden_state_#{i}", layer,
          wan_audio_expected_tensor(expected_file, "hidden_state_#{i}"))
      end
      wan_audio_assert_gate_values("output", actual, wan_audio_expected_tensor(expected_file, "output"))
      repeated.read(cfg.output_values).map(&.unsafe_as(UInt32)).should eq(actual.map(&.unsafe_as(UInt32)))
    ensure
      position_probe_file.try(&.close)
      mps_position.try(&.release)
      mps_gelu.try(&.release)
      mps_normalizer.try(&.release)
      mps_plans.try(&.release)
      prepared.try(&.release)
      traces.release
      arena.release
      file.close
      expected_file.close
    end
  end
end
