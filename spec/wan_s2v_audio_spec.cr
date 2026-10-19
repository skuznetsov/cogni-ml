require "spec"
require "../src/ml/video/wan_s2v_audio"
require "digest/sha256"
require "json"
require "./support/video_allocation_counter"

private def wan_s2v_audio_config(json : JSON::Any) : ML::Video::WanS2VCausalAudioConfig
  ML::Video::WanS2VCausalAudioConfig.new(
    feature_layers: json["feature_layers"].as_i.to_i32,
    input_dim: json["input_dim"].as_i.to_i32,
    hidden_dim: json["hidden_dim"].as_i.to_i32,
    local_tokens: json["local_tokens"].as_i.to_i32,
    bucket_frames: json["bucket_frames"].as_i.to_i32,
    prefix_frames: json["prefix_frames"].as_i.to_i32,
    drop_frames: json["drop_frames"].as_i.to_i32
  )
end

private def wan_s2v_audio_sha256(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |io|
    bytes = Bytes.new(1024 * 1024)
    while (size = io.read(bytes)) > 0
      digest.update(bytes[0, size])
    end
  end
  digest.hexfinal
end

private def wan_s2v_audio_assert_gate(name : String, actual : Array(Float32), expected : JSON::Any) : Nil
  reference = expected.as_a
  actual.size.should eq(reference.size), name
  worst = 0_f64
  violations = 0
  actual.each_with_index do |value, i|
    target = reference[i].as_f.to_f32
    ratio = (value - target).abs.to_f64 / (2e-5 + 2e-5 * target.abs)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

private def wan_s2v_audio_assert_mps_near_constant_row(
  cfg : ML::Video::WanS2VCausalAudioConfig,
  expected : Hash(String, JSON::Any),
  actual : Hash(String, Array(Float32)),
) : Nil
  # The pinned tiny fixture deliberately includes a near-constant first-local-
  # convolution row. PyTorch 2.8.0 MPS LayerNorm uses E[x^2]-E[x]^2 and clamps
  # variance below 1e-6 to zero before rsqrt; the centered-variance fallback
  # produces a measurably different row here.
  cfg.quarter_dim.should eq(2)
  cfg.conv1_frames.should eq(13)
  offset = 5 * cfg.quarter_dim
  conv = expected["local_conv1"].as_a
  norm = expected["local_norm1"].as_a
  x0 = conv[offset].as_f.to_f32
  x1 = conv[offset + 1].as_f.to_f32
  mean = (x0 + x1) / 2.0_f32
  raw_variance = (x0 * x0 + x1 * x1) / 2.0_f32 - mean * mean
  raw_variance.should be < 1e-6_f32, "fixture row must trigger the MPS raw-moment clamp"

  expected0 = norm[offset].as_f.to_f32
  expected1 = norm[offset + 1].as_f.to_f32
  expected0.should be_close((x0 - mean) * 1_000.0_f32, 2e-5_f32)
  expected1.should be_close((x1 - mean) * 1_000.0_f32, 2e-5_f32)

  centered_variance = ((x0 - mean) * (x0 - mean) + (x1 - mean) * (x1 - mean)) / 2.0_f32
  centered_inverse = 1.0_f32 / Math.sqrt(centered_variance + ML::Video::WanS2VCausalAudioEncoder::NORM_EPS).to_f32
  centered0 = (x0 - mean) * centered_inverse
  (expected0 - centered0).abs.should be > 0.1_f32,
    "fixture must distinguish MPS raw-moment clamp from centered variance"

  native0 = actual["local_norm1"][offset]
  native1 = actual["local_norm1"][offset + 1]
  (native0 - expected0).abs.should be < 2e-5_f32
  (native1 - expected1).abs.should be < 2e-5_f32
end

private def wan_s2v_audio_run_fixture(path : String, mode : String) : Nil
  fixture = JSON.parse(File.read(path))
  fixture["mode"].as_s.should eq(mode)
  fixture["wan_source_commit"].as_s.should eq(ML::Video::WanS2VCausalAudioEncoder::WAN_SOURCE_COMMIT)
  fixture["wan_audio_utils_sha256"].as_s.should eq(ML::Video::WanS2VCausalAudioEncoder::WAN_AUDIO_UTILS_SHA256)
  fixture["wan_auxi_blocks_sha256"].as_s.should eq(ML::Video::WanS2VCausalAudioEncoder::WAN_AUXI_BLOCKS_SHA256)
  fixture["device"].as_s.should eq("mps")
  fixture["arithmetic"].as_s.should eq("F32")

  cfg = wan_s2v_audio_config(fixture["config"])
  cfg.validate!
  fixture["input_shape"].as_a.map(&.as_i.to_i32).should eq([
    cfg.feature_layers, cfg.input_dim, cfg.bucket_frames,
  ])
  fixture["conv_frames"].as_a.map(&.as_i.to_i32).should eq([
    cfg.conv1_frames, cfg.conv2_frames, cfg.conv3_frames,
  ])
  fixture["video_frames"].as_i.to_i32.should eq(cfg.video_frames)

  expected = fixture["expected"].as_h
  required_names = cfg.trace_plan.map(&.first).sort
  expected.keys.sort.should eq(required_names), "oracle must capture every native trace checkpoint"

  weights_path = fixture["weights"].as_s
  wan_s2v_audio_sha256(weights_path).should eq(fixture["weight_sha256"].as_s)
  file = ML::Video::CheckpointFile.new(weights_path)
  arena = ML::Video::Arena.new
  traces = ML::Video::Arena.new
  prepared = nil.as(ML::Video::PreparedWeights?)
  begin
    bindings = ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(cfg)
    bindings.size.should eq(12)
    budget = fixture["memory_budget_bytes"].as_i64
    reserve_bytes = cfg.scratch_bytes +
                    cfg.feature_values.to_i64 * sizeof(Float32) +
                    (cfg.local_output_values.to_i64 * 2 + cfg.global_output_values.to_i64 * 2) * sizeof(Float32) +
                    cfg.trace_plan.sum { |(_, count)| count.to_i64 * sizeof(Float32) }
    plan = ML::Video::WeightPlan.new([file], bindings, budget, reserve_bytes,
      resident_dtype: ML::Video::WeightDType::F32, require_all: true)
    plan.peak_bytes.should be <= budget, "weight, staging, scratch, IO and complete trace must fit configured budget"
    plan.reserve_bytes.should eq(reserve_bytes)

    prepared = ML::Video::PreparedWeights.new(plan, arena)
    weights = ML::Video::WanS2VCausalAudioEncoder.bind_weights(arena, cfg)
    scratch = ML::Video::WanS2VCausalAudioEncoder.prepare_scratch!(arena, cfg)
    scratch_bytes = scratch.weighted_features.size + scratch.activation_a.size + scratch.activation_b.size
    scratch_bytes.should eq(cfg.scratch_bytes)

    input_values = fixture["input_features"].as_a.map(&.as_f.to_f32)
    input_values.size.should eq(cfg.feature_values)
    features = arena.upload("s2v_audio.input", input_values)
    local_output = arena.alloc("s2v_audio.local_output", cfg.local_output_values)
    global_output = arena.alloc("s2v_audio.global_output", cfg.global_output_values)
    local_repeat = arena.alloc("s2v_audio.local_repeat", cfg.local_output_values)
    global_repeat = arena.alloc("s2v_audio.global_repeat", cfg.global_output_values)

    trace = {} of String => ML::MetalBuffer
    cfg.trace_plan.each do |name, count|
      trace[name] = traces.alloc("s2v_audio.trace.#{name}", count)
    end
    expected.keys.sort.should eq(trace.keys.sort)

    # Reject aliased inputs/outputs before enqueueing work. The shared buffer
    # is sized for either role so the alias check, rather than a size check,
    # must reject it.
    alias_buffer = arena.alloc("s2v_audio.alias", Math.max(cfg.feature_values, cfg.local_output_values))
    before_rejection = ML::MetalBuffer.video_spec_allocations
    rejected_command = ML::Metal::CommandBuffer.new
    rejected_encoder = ML::Metal::ComputeEncoder.new(rejected_command)
    expect_raises(ArgumentError, "Unsafe video buffer alias") do
      ML::Video::WanS2VCausalAudioEncoder.encode(rejected_encoder, cfg, alias_buffer,
        alias_buffer, global_output, weights, scratch)
    end
    ML::MetalBuffer.video_spec_allocations.should eq(before_rejection)
    rejected_encoder.end_encoding
    rejected_command.discard
    arena.remove("s2v_audio.alias")

    before = ML::MetalBuffer.video_spec_allocations
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    prepared.encode(encoder)
    ML::Video::WanS2VCausalAudioEncoder.encode(encoder, cfg, features, local_output,
      global_output, weights, scratch, trace)
    ML::Video::WanS2VCausalAudioEncoder.encode(encoder, cfg, features, local_repeat,
      global_repeat, weights, scratch)
    encoder.end_encoding
    ML::MetalBuffer.video_spec_allocations.should eq(before), "encode must allocate no GPU buffers"

    accounted_peak = plan.staging_bytes + arena.resident_bytes + traces.resident_bytes
    accounted_peak.should be <= budget, "resident weights, staging, scratch, IO and traces exceed configured cap"

    command.commit
    command.wait
    actual = trace.to_h { |name, buffer| {name, buffer.read((buffer.size // 4).to_i)} }
    if actual_path = ENV["WAN_S2V_AUDIO_ACTUAL"]?
      File.write(actual_path, actual.to_json)
    end
    actual.keys.sort.should eq(required_names), "native capture must expose the complete trace set"
    if mode == "tiny"
      wan_s2v_audio_assert_mps_near_constant_row(cfg, expected, actual)
    end
    actual.each do |name, values|
      wan_s2v_audio_assert_gate(name, values, expected[name])
    end
    wan_s2v_audio_assert_gate("repeat.local_output", local_repeat.read(cfg.local_output_values), expected["local_output"])
    wan_s2v_audio_assert_gate("repeat.global_output", global_repeat.read(cfg.global_output_values), expected["global_output"])
    local_repeat.read(cfg.local_output_values).map(&.unsafe_as(UInt32)).should eq(
      local_output.read(cfg.local_output_values).map(&.unsafe_as(UInt32)))
    global_repeat.read(cfg.global_output_values).map(&.unsafe_as(UInt32)).should eq(
      global_output.read(cfg.global_output_values).map(&.unsafe_as(UInt32)))
  ensure
    prepared.try(&.release)
    traces.release
    arena.release
    file.close
  end
end

describe ML::Video::WanS2VCausalAudioConfig do
  it "matches the pinned production five-second, 84-bucket geometry" do
    cfg = ML::Video::WanS2VCausalAudioConfig.new
    cfg.validate!
    cfg.feature_values.should eq(25 * 1_024 * 84)
    cfg.input_frames.should eq(101)
    cfg.conv1_frames.should eq(101)
    cfg.conv2_frames.should eq(51)
    cfg.conv3_frames.should eq(26)
    cfg.video_frames.should eq(21)
    cfg.local_output_values.should eq(21 * 5 * 5_120)
    cfg.global_output_values.should eq(21 * 5_120)
    cfg.scratch_bytes.should eq(4_673_536_i64)

    bindings = ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(cfg)
    bindings.size.should eq(12)
    bindings.map(&.target).uniq.size.should eq(12)
    bindings.map(&.source).uniq.size.should eq(12)
    bindings.any? do |binding|
      binding.target == "s2v_audio.conv3.weight" &&
        binding.source == "casual_audio_encoder.encoder.conv3.conv.weight" &&
        binding.shape == [5_120_i64, 2_560_i64, 3_i64]
    end.should be_true
    bindings.any? do |binding|
      binding.target == "s2v_audio.layer_weights" &&
        binding.source == "casual_audio_encoder.weights" &&
        binding.shape == [1_i64, 25_i64, 1_i64, 1_i64]
    end.should be_true
  end

  it "keeps synthetic tiny geometry on the production operator contract" do
    cfg = ML::Video::WanS2VCausalAudioConfig.new(
      feature_layers: 3, input_dim: 6, hidden_dim: 8, local_tokens: 2,
      bucket_frames: 9, prefix_frames: 4, drop_frames: 2)
    cfg.validate!
    cfg.input_frames.should eq(13)
    cfg.conv2_frames.should eq(7)
    cfg.conv3_frames.should eq(4)
    cfg.video_frames.should eq(2)
    cfg.trace_plan.size.should eq(22)
    ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(cfg).size.should eq(12)
  end
end

describe ML::Video::WanS2VCausalAudioEncoder do
  it "matches every pinned tiny MPS intermediate and repeats without encoding allocations" do
    fixture_path = ENV["WAN_S2V_AUDIO_FIXTURE"]?
    unless fixture_path
      pending!("Set WAN_S2V_AUDIO_FIXTURE to the source-pinned tiny MPS fixture")
    end
    wan_s2v_audio_run_fixture(fixture_path, "tiny")
  end

  it "matches production-width CausalAudio against the twelve checkpoint tensors" do
    fixture_path = ENV["WAN_S2V_AUDIO_PRODUCTION_FIXTURE"]?
    unless fixture_path
      pending!("Set WAN_S2V_AUDIO_PRODUCTION_FIXTURE to the pinned production MPS fixture")
    end
    wan_s2v_audio_run_fixture(fixture_path, "production")
  end
end
