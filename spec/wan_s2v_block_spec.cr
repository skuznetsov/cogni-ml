# Copyright 2024-2025 The Alibaba Wan Team Authors.
# Apache-2.0. Adapted fixture/spec harness for Wan2.2 S2V at
# 1ea34ff48f87168174e12956e200b1d908b1c5ff.
require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/wan_s2v_block"
require "./support/video_allocation_counter"

lib LibWanVideoSpecMemory
  fun posix_memalign(ptr : Void**, alignment : LibC::SizeT, size : LibC::SizeT) : LibC::Int
  fun getpagesize : LibC::Int
  fun free(ptr : Void*) : Void
end

private WAN_SOURCE_SHA256 = {
  "wan_modules_model.py"               => "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
  "wan_modules_s2v_model_s2v.py"       => "91fb14df44edab2d01b5634420b463c5f1563aecdf6dac905da6b3acac3145cc",
  "wan_modules_s2v_audio_utils.py"     => "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd",
  "wan_modules_s2v_s2v_utils.py"       => "e45ae6c274b17e4fdddcb0439b598897a37679b8934ebd71f87255f92d5f0b2f",
  "wan_modules_s2v_auxi_blocks.py"     => "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6",
  "diffusers-normalization-v0.31.0.py" => "7f2d2086f669e19bc008d71743f3963017274fbc6ad86d0ab9bf18db211e7c5a",
}

private def wan_json_f32(value : JSON::Any) : Array(Float32)
  value.as_a.map(&.as_f.to_f32)
end

private def wan_comparison(name : String, actual : Array(Float32), expected : JSON::Any) : Tuple(Bool, String)
  reference = wan_json_f32(expected)
  unless actual.size == reference.size
    return {false, "#{name}: checkpoint size mismatch native=#{actual.size} oracle=#{reference.size}"}
  end

  violations = 0
  finite_count = 0
  native_non_finite = 0
  oracle_non_finite = 0
  worst_ratio = -1.0_f64
  worst_index = 0
  worst_actual = 0.0_f32
  worst_reference = 0.0_f32
  worst_difference = 0.0_f32
  worst_limit = 0.0_f32

  actual.each_with_index do |value, index|
    target = reference[index]
    native_non_finite += 1 unless value.finite?
    oracle_non_finite += 1 unless target.finite?
    unless value.finite? && target.finite?
      violations += 1
      if worst_ratio < Float64::INFINITY
        worst_ratio = Float64::INFINITY
        worst_index = index
        worst_actual = value
        worst_reference = target
        worst_difference = (value - target).abs
        worst_limit = 2e-5_f32 + 2e-5_f32 * target.abs
      end
      next
    end

    finite_count += 1
    difference = (value - target).abs
    limit = 2e-5_f32 + 2e-5_f32 * target.abs
    ratio = difference.to_f64 / limit.to_f64
    if difference > limit
      violations += 1
    end
    if ratio > worst_ratio
      worst_ratio = ratio
      worst_index = index
      worst_actual = value
      worst_reference = target
      worst_difference = difference
      worst_limit = limit
    end
  end

  detail = String.build do |io|
    io << "#{name}: checked=#{actual.size} finite=#{finite_count}"
    io << " native_non_finite=#{native_non_finite} oracle_non_finite=#{oracle_non_finite}"
    io << " violations=#{violations}"
    io << " max_normalized_error=#{worst_ratio} at flat=#{worst_index}"
    io << " native=#{worst_actual} oracle=#{worst_reference}"
    io << " abs=#{worst_difference} limit=#{worst_limit}"
  end
  {violations == 0, detail}
end

private def wan_matches(actual : Array(Float32), expected : JSON::Any) : Bool
  wan_comparison("fixture value", actual, expected)[0]
end

private def wan_snapshot_json(snapshots : Hash(String, Array(Float32))) : JSON::Any
  encoded = {} of String => JSON::Any
  snapshots.each do |name, values|
    encoded_values = [] of JSON::Any
    values.each do |value|
      if value.finite?
        encoded_values << JSON::Any.new(value.to_f64)
      else
        encoded_values << JSON::Any.new(value.to_s)
      end
    end
    encoded[name] = JSON::Any.new(encoded_values)
  end
  JSON::Any.new(encoded)
end

private def wan_rope_diagnostic(actual : Array(Float32), expected : JSON::Any,
                                snapshots : Hash(String, Array(Float32)), fixture : JSON::Any) : String
  reference = wan_json_f32(expected)
  raw_q_reference = wan_json_f32(fixture["expected"]["self_q"])
  norm_q_reference = wan_json_f32(fixture["expected"]["self_q_norm"])
  dim = fixture["config"]["dim"].as_i
  seq = fixture["config"]["seq"].as_i
  heads = fixture["config"]["heads"].as_i
  head_dim = dim // heads
  worst_index = 0
  worst_ratio = 0.0_f64
  violations = 0
  actual.each_with_index do |value, index|
    target = reference[index]
    difference = (value - target).abs.to_f64
    limit = 2e-5_f64 + 2e-5_f64 * target.abs.to_f64
    ratio = difference / limit
    violations += 1 if ratio > 1.0
    if ratio > worst_ratio
      worst_ratio = ratio
      worst_index = index
    end
  end

  row = worst_index // dim
  batch = row // seq
  token = row % seq
  channel = worst_index % dim
  head = channel // head_dim
  lane = channel % head_dim
  position_values = fixture["inputs"]["positions"].as_a
  position_offset = (batch * seq + token) * 3
  position = (0...3).map { |axis| position_values[position_offset + axis].as_f }
  difference = (actual[worst_index] - reference[worst_index]).abs
  limit = 2e-5_f32 + 2e-5_f32 * reference[worst_index].abs
  String.build do |io|
    io << "worst normalized error=#{worst_ratio} violations=#{violations} flat=#{worst_index}"
    io << " b=#{batch} token=#{token} head=#{head} channel_in_head=#{lane} pair=#{lane // 2} lane=#{lane % 2}"
    io << " position=#{position} native=#{actual[worst_index]} oracle=#{reference[worst_index]}"
    io << " abs=#{difference} limit=#{limit}"
    io << " raw_q(native/oracle)=#{snapshots["self_q"][worst_index]}/#{raw_q_reference[worst_index]}"
    io << " norm_q(native/oracle)=#{snapshots["self_q_norm"][worst_index]}/#{norm_q_reference[worst_index]}"
  end
end

private def wan_checkpoint_size(name : String, cfg : JSON::Any) : Int32
  dim = cfg["dim"].as_i
  rows = cfg["batches"].as_i * cfg["seq"].as_i
  context = cfg["batches"].as_i * cfg["context_tokens"].as_i
  ffn_dim = cfg["ffn_dim"].as_i
  case name
  when "cross_k", "cross_v"
    context * dim
  when "ffn_in", "ffn_gelu"
    rows * ffn_dim
  else
    rows * dim
  end
end

private def wan_audio_checkpoint_size(name : String, cfg : JSON::Any) : Int32
  dim = cfg["dim"].as_i
  frames = cfg["batches"].as_i * cfg["video_frames"].as_i
  video_rows = frames * cfg["spatial_tokens"].as_i
  audio_rows = frames * cfg["audio_tokens"].as_i
  case name
  when "global_silu"
    frames * dim
  when "adain_projection"
    frames * 2 * dim
  when "audio_k", "audio_v"
    audio_rows * dim
  when "audio_output"
    cfg["batches"].as_i * cfg["seq"].as_i * dim
  else
    video_rows * dim
  end
end

private def wan_fixture : JSON::Any
  path = ENV["WAN_S2V_BLOCK_FIXTURE"]? || raise "Set WAN_S2V_BLOCK_FIXTURE to the pinned MPS oracle fixture"
  JSON.parse(File.read(path))
end

private def wan_fixture_path : String
  ENV["WAN_S2V_BLOCK_FIXTURE"]? || raise "Set WAN_S2V_BLOCK_FIXTURE to the pinned MPS oracle fixture"
end

private def wan_resident_weight_dtype : ML::Video::WeightDType
  case ENV["WAN_S2V_WEIGHT_DTYPE"]? || "f32"
  when "f32"
    ML::Video::WeightDType::F32
  when "bf16"
    ML::Video::WeightDType::BF16
  else
    raise ArgumentError.new("WAN_S2V_WEIGHT_DTYPE must be f32 or bf16")
  end
end

private def wan_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1024 * 1024)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def wan_run_fixture(fixture : JSON::Any) : Hash(String, Array(Float32))
  resident_dtype = wan_resident_weight_dtype
  inputs = ML::Video::Arena.new
  weights = ML::Video::Arena.new
  scratch = ML::Video::Arena.new
  traces = ML::Video::Arena.new
  output = ML::Video::Arena.new
  checkpoint_file = nil.as(ML::Video::CheckpointFile?)
  prepared_weights = nil.as(ML::Video::PreparedWeights?)
  begin
    fixture["inputs"].as_h.each do |name, values|
      inputs.upload(name, wan_json_f32(values))
    end
    config_json = fixture["config"]
    block_cfg = ML::Video::WanS2VConfig.new(
      batches: config_json["batches"].as_i,
      seq: config_json["seq"].as_i,
      context_tokens: config_json["context_tokens"].as_i,
      dim: config_json["dim"].as_i,
      heads: config_json["heads"].as_i,
      ffn_dim: config_json["ffn_dim"].as_i,
      split: config_json["split"].as_i,
      theta: config_json["theta"].as_f.to_f32,
      video_frames: config_json["video_frames"].as_i,
      spatial_tokens: config_json["spatial_tokens"].as_i,
    )
    audio_cfg = ML::Video::WanS2VAudioConfig.new(
      batches: config_json["batches"].as_i,
      seq: config_json["seq"].as_i,
      dim: config_json["dim"].as_i,
      heads: config_json["heads"].as_i,
      video_frames: config_json["video_frames"].as_i,
      spatial_tokens: config_json["spatial_tokens"].as_i,
      audio_tokens: config_json["audio_tokens"].as_i,
    )

    if weight_manifest = fixture["weights_file"]?
      fixture["upstream"]["fixture_mode"].as_s.should eq("production_width")
      fixture_path = wan_fixture_path
      relative_path = weight_manifest["path"].as_s
      (File.basename(relative_path) == relative_path && relative_path != "." && relative_path != "..").should be_true,
        "weight sidecar must stay beside the fixture"
      sidecar_path = File.join(File.dirname(File.expand_path(fixture_path)), relative_path)
      File.size(sidecar_path).should eq(weight_manifest["bytes"].as_i64)
      wan_sha256_file(sidecar_path).should eq(weight_manifest["sha256"].as_s)

      fixture_bindings = fixture["weight_bindings"].as_a
      fixture_bindings.size.should eq(39)
      dtypes = fixture_bindings.map { |item| item["dtype"].as_s }.uniq
      dtypes.size.should eq(1), "all checkpoint fixture tensors must use one source dtype"
      checkpoint_dtype = ML::GGUF::SafeTensorDType.parse(dtypes.first)
      if resident_dtype.bf16?
        checkpoint_dtype.should eq(ML::GGUF::SafeTensorDType::BF16),
          "resident BF16 mode requires an exact BF16 source checkpoint"
      end
      bindings = ML::Video::WanS2VBlock.checkpoint_bindings(block_cfg, checkpoint_dtype)
      fixture_bindings.map { |item| item["source"].as_s }.sort.should eq(bindings.map(&.source).sort)
      fixture_bindings.map { |item| item["target"].as_s }.sort.should eq(bindings.map(&.target).sort)
      fixture_bindings.each do |item|
        native = bindings.find! { |binding| binding.source == item["source"].as_s }
        item["shape"].as_a.map(&.as_i64).should eq(native.shape)
        ML::GGUF::SafeTensorDType.parse(item["dtype"].as_s).should eq(native.dtype)
      end

      checkpoint_file = ML::Video::CheckpointFile.new(sidecar_path)
      plan = ML::Video::WeightPlan.new(
        [checkpoint_file.not_nil!], bindings,
        8_i64 * 1024 * 1024 * 1024, 1024_i64 * 1024 * 1024,
        resident_dtype: resident_dtype, require_all: true,
      )
      plan.entries.size.should eq(39)
      expected_destination_bytes = resident_dtype.bf16? ? 1_017_443_328_i64 : 2_034_886_656_i64
      plan.destination_bytes.should eq(expected_destination_bytes)
      expected_staging_bytes = if resident_dtype.bf16?
                                 0_i64
                               else
                                 expected_destination_bytes * checkpoint_dtype.bytes_per_element // 4
                               end
      if resident_dtype.bf16?
        expected_staging_bytes.should eq(0_i64),
          "native BF16 residency must not decode or stage a second copy"
      else
        expected_staging_bytes.should eq(
          checkpoint_dtype.f32? ? 2_034_886_656_i64 : 1_017_443_328_i64
        )
      end
      plan.staging_bytes.should eq(expected_staging_bytes)
      plan.peak_bytes.should be <= 8_i64 * 1024 * 1024 * 1024
      prepared_weights = ML::Video::PreparedWeights.new(plan, weights)
      bindings.each do |binding|
        weights.dtype(binding.target).should eq(resident_dtype)
      end
    else
      resident_dtype.should eq(ML::Video::WeightDType::F32),
        "the synthetic JSON fixture contains F32 parameters only"
      fixture["weights"].as_h.each do |name, values|
        weights.upload(name, wan_json_f32(values))
      end
    end

    ML::Video::WanS2VBlock.prepare!
    ML::Video::WanS2VAudioInjector.prepare!
    ML::Video::WanS2VBlock.prepare_scratch!(scratch, block_cfg)
    ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, audio_cfg)
    block_output = output.alloc("block_output", block_cfg.values)
    audio_output = output.alloc("audio_output", audio_cfg.values)
    repeat_block_output = output.alloc("repeat_block_output", block_cfg.values)
    repeat_audio_output = output.alloc("repeat_audio_output", audio_cfg.values)
    chained_block_output = output.alloc("chained_block_output", block_cfg.values)
    {inputs, scratch, traces, output}.each do |arena|
      arena.buffers.each_key do |name|
        arena.dtype(name).should eq(ML::Video::WeightDType::F32),
          "activations, scratch, and checkpoints must remain F32"
      end
    end

    expected = fixture["expected"].as_h
    block_trace = {} of String => ML::MetalBuffer
    audio_trace = {} of String => ML::MetalBuffer
    expected.each_key do |name|
      if ML::Video::WanS2VBlock::CHECKPOINTS.has_key?(name)
        block_trace[name] = traces.alloc("block.#{name}", wan_checkpoint_size(name, config_json))
      elsif ML::Video::WanS2VAudioInjector::CHECKPOINTS.has_key?(name)
        audio_trace[name] = traces.alloc("audio.#{name}", wan_audio_checkpoint_size(name, config_json))
      end
    end

    mismatched_cfg = ML::Video::WanS2VConfig.new(
      batches: block_cfg.batches, seq: block_cfg.seq,
      context_tokens: block_cfg.context_tokens, dim: block_cfg.dim,
      heads: block_cfg.heads, ffn_dim: block_cfg.ffn_dim,
      split: block_cfg.split, theta: block_cfg.theta,
      video_frames: block_cfg.video_frames, spatial_tokens: block_cfg.spatial_tokens + 1,
    )
    mismatched_audio_cfg = ML::Video::WanS2VAudioConfig.new(
      batches: audio_cfg.batches, seq: audio_cfg.seq, dim: audio_cfg.dim,
      heads: audio_cfg.heads, video_frames: audio_cfg.video_frames,
      spatial_tokens: audio_cfg.spatial_tokens + 1, audio_tokens: audio_cfg.audio_tokens,
    )
    rejected_command = ML::Metal::CommandBuffer.new
    rejected_encoder = ML::Metal::ComputeEncoder.new(rejected_command)
    expect_raises(ArgumentError, "Wan S2V block split must equal the shared video prefix") do
      ML::Video::WanS2VBlock.encode_with_audio(
        rejected_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], inputs["local_audio"], inputs["global_audio"],
        block_output, audio_output, weights, scratch, mismatched_cfg, mismatched_audio_cfg,
      )
    end
    rejected_encoder.end_encoding
    rejected_command.discard

    different_rope_cfg = ML::Video::WanS2VConfig.new(
      batches: block_cfg.batches, seq: block_cfg.seq,
      context_tokens: block_cfg.context_tokens, dim: block_cfg.dim,
      heads: block_cfg.heads, ffn_dim: block_cfg.ffn_dim,
      split: block_cfg.split, theta: block_cfg.theta + 0.5_f32,
      video_frames: block_cfg.video_frames, spatial_tokens: block_cfg.spatial_tokens,
    )
    rope_guard_command = ML::Metal::CommandBuffer.new
    rope_guard_encoder = ML::Metal::ComputeEncoder.new(rope_guard_command)
    expect_raises(KeyError) do
      ML::Video::WanS2VBlock.encode_with_audio(
        rope_guard_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], inputs["local_audio"], inputs["global_audio"],
        block_output, audio_output, weights, scratch, different_rope_cfg, audio_cfg,
      )
    end
    rope_guard_encoder.end_encoding
    rope_guard_command.discard

    allocation_start = ML::MetalBuffer.video_spec_allocations
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    prepared_weights.try(&.encode(encoder))
    ML::Video::WanS2VBlock.encode_with_audio(
      encoder, inputs["x"], inputs["modulation"], inputs["positions"],
      inputs["context"], inputs["local_audio"], inputs["global_audio"],
      block_output, audio_output, weights, scratch, block_cfg, audio_cfg,
      block_trace, audio_trace,
    )
    ML::Video::WanS2VBlock.encode_with_audio(
      encoder, inputs["x"], inputs["modulation"], inputs["positions"],
      inputs["context"], inputs["local_audio"], inputs["global_audio"],
      repeat_block_output, repeat_audio_output, weights, scratch, block_cfg, audio_cfg,
    )
    ML::Video::WanS2VBlock.encode(
      encoder, block_output, inputs["modulation"], inputs["positions"],
      inputs["context"], chained_block_output, weights, scratch, block_cfg,
    )
    encoder.end_encoding
    ML::MetalBuffer.video_spec_allocations.should eq(allocation_start),
      "Wan encode must not allocate MetalBuffers after setup"
    command.commit
    command.wait
    prepared_weights.try(&.release)
    checkpoint_file.try(&.close)

    snapshots = {} of String => Array(Float32)
    block_trace.each { |name, buffer| snapshots[name] = buffer.read((buffer.size // 4).to_i).map(&.to_f32) }
    audio_trace.each { |name, buffer| snapshots[name] = buffer.read((buffer.size // 4).to_i).map(&.to_f32) }
    snapshots["repeat_block_output"] = repeat_block_output.read((repeat_block_output.size // 4).to_i).map(&.to_f32)
    snapshots["repeat_audio_output"] = repeat_audio_output.read((repeat_audio_output.size // 4).to_i).map(&.to_f32)
    snapshots["chained_block_output"] = chained_block_output.read((chained_block_output.size // 4).to_i).map(&.to_f32)
    if snapshot_path = ENV["WAN_S2V_SNAPSHOT_DUMP"]?
      File.write(snapshot_path, wan_snapshot_json(snapshots).to_json)
    end
    snapshots
  ensure
    prepared_weights.try(&.release)
    checkpoint_file.try(&.close)
    output.release
    traces.release
    scratch.release
    weights.release
    inputs.release
  end
end

describe ML::Video::WanS2VConfig do
  it "preserves the tiny upstream 3-axis RoPE partition" do
    config = ML::Video::WanS2VConfig.new(
      batches: 1, seq: 12, context_tokens: 3, dim: 32, heads: 2,
      ffn_dim: 64, split: 8, theta: 10_000.0_f32,
      video_frames: 2, spatial_tokens: 4,
    )
    config.head_dim.should eq(16)
    config.rope_axis_dims.should eq({4, 2, 2})
    config.video_tokens.should eq(8)
  end

  it "rejects invalid segment and attention dimensions before encoding" do
    expect_raises(ArgumentError) do
      ML::Video::WanS2VConfig.new(
        batches: 1, seq: 12, context_tokens: 3, dim: 32, heads: 2,
        ffn_dim: 64, split: 0, theta: 10_000.0_f32,
        video_frames: 2, spatial_tokens: 4,
      )
    end
    expect_raises(ArgumentError) do
      ML::Video::WanS2VConfig.new(
        batches: 1, seq: 12, context_tokens: 3, dim: 30, heads: 2,
        ffn_dim: 64, split: 8, theta: 10_000.0_f32,
        video_frames: 2, spatial_tokens: 4,
      )
    end
  end
end

describe ML::Video::WanS2VAudioConfig do
  it "requires the upstream five local audio tokens per frame" do
    config = ML::Video::WanS2VAudioConfig.new(
      batches: 1, seq: 12, dim: 32, heads: 2,
      video_frames: 2, spatial_tokens: 4, audio_tokens: 5,
    )
    config.audio_rows.should eq(10)
    expect_raises(ArgumentError) do
      ML::Video::WanS2VAudioConfig.new(
        batches: 1, seq: 12, dim: 32, heads: 2,
        video_frames: 2, spatial_tokens: 4, audio_tokens: 4,
      )
    end
  end
end

describe ML::Video::WanS2VBlock do
  it "matches pinned block/audio checkpoints including distinct multi-head RoPE positions" do
    unless ENV["WAN_S2V_BLOCK_FIXTURE"]?
      pending "MPS oracle fixture is external; set WAN_S2V_BLOCK_FIXTURE"
    end
    fixture = wan_fixture
    fixture["upstream"]["revision"].as_s.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    fixture["upstream"]["diffusers_version"].as_s.should eq("0.31.0")
    fixture["upstream"]["oracle_device"].as_s.should eq("mps")
    fixture["upstream"]["fallback_disabled"].as_bool.should be_true
    fixture_mode = fixture["upstream"]["fixture_mode"].as_s
    {"tiny", "production_width"}.includes?(fixture_mode).should be_true
    source_hashes = fixture["upstream"]["source_sha256"].as_h
    source_hashes.keys.sort.should eq(WAN_SOURCE_SHA256.keys.sort)
    WAN_SOURCE_SHA256.each { |name, expected| source_hashes[name].as_s.should eq(expected) }
    fixture["config"]["audio_attention_query_rows_per_frame"].as_i.should eq(fixture["config"]["spatial_tokens"].as_i)
    fixture["config"]["batches"].as_i.should eq(2), "the MPS parity fixture must exercise two batches"
    fixture["config"]["heads"].as_i.should be > 1
    config = fixture["config"]
    if fixture_mode == "production_width"
      config["dim"].as_i.should eq(5120)
      config["ffn_dim"].as_i.should eq(13824)
      config["heads"].as_i.should eq(40)
      (config["dim"].as_i // config["heads"].as_i).should eq(128)
      config["qk_norm"].as_bool.should be_true
      config["cross_attn_norm"].as_bool.should be_true
      config["enable_adain"].as_bool.should be_true
      config["adain_mode"].as_s.should eq("attn_norm")
      config["audio_inject_layers"].as_a.map(&.as_i).should eq([0])
      config["split"].as_i.should eq(1)
      config["video_frames"].as_i.should eq(1)
      config["spatial_tokens"].as_i.should eq(1)
      config["audio_tokens"].as_i.should eq(5)
    end
    dim = config["dim"].as_i
    seq = config["seq"].as_i
    positions = fixture["inputs"]["positions"].as_a
    position_at = ->(batch : Int32, token : Int32) {
      offset = (batch * seq + token) * 3
      (0...3).map { |axis| positions[offset + axis].as_f }
    }
    position_at.call(0, 0).should eq([255.0, 511.0, 1022.0])
    position_at.call(1, 0).should eq([301.0, 700.0, 800.0])
    if fixture_mode == "production_width"
      position_at.call(0, 1).should eq([-30.0, 0.0, 0.0])
      position_at.call(1, 1).should eq([-45.0, 10.0, 20.0])
    else
      position_at.call(0, 7).should eq([256.0, 512.0, 1023.0])
      position_at.call(1, 7).should eq([302.0, 701.0, 801.0])
      position_at.call(0, 8).should eq([-30.0, 0.0, 0.0])
      position_at.call(1, 8).should eq([-45.0, 10.0, 20.0])
    end
    position_at.call(0, 0).should_not eq(position_at.call(0, 1)),
      "RoPE regression fixture must distinguish neighboring token positions"
    position_at.call(0, 0).should_not eq(position_at.call(1, 0)),
      "RoPE regression fixture must distinguish batches"
    batch_sizes = {
      "x"            => seq * dim,
      "modulation"   => 12 * dim,
      "context"      => config["context_tokens"].as_i * dim,
      "local_audio"  => config["video_frames"].as_i * config["audio_tokens"].as_i * dim,
      "global_audio" => config["video_frames"].as_i * dim,
    }
    batch_sizes.each do |name, batch_size|
      values = fixture["inputs"][name].as_a
      values[0, batch_size].should_not eq(values[batch_size, batch_size]),
        "fixture input #{name} must differ between batches"
    end
    required_checkpoints = ML::Video::WanS2VBlock::CHECKPOINTS.keys + ML::Video::WanS2VAudioInjector::CHECKPOINTS.keys
    fixture["expected"].as_h.keys.sort.should eq(required_checkpoints.sort)
    snapshots = wan_run_fixture(fixture)
    checkpoint_failures = [] of String
    fixture["expected"].as_h.each do |name, expected|
      unless snapshots.has_key?(name)
        checkpoint_failures << "missing Wan checkpoint #{name}"
        next
      end

      passed, detail = wan_comparison(name, snapshots[name], expected)
      unless passed
        if name == "self_q_rope"
          detail += " (#{wan_rope_diagnostic(snapshots[name], expected, snapshots, fixture)})"
        end
        checkpoint_failures << detail
      end
    end
    checkpoint_failures.empty?.should be_true, checkpoint_failures.join("\n")
    corrupted_audio_reference = wan_json_f32(fixture["expected"]["audio_output"])
    corrupted_audio_reference[0] += 0.01_f32
    wan_matches(snapshots["audio_output"], JSON.parse(corrupted_audio_reference.to_json)).should be_false,
      "the fixed gate must reject a corrupted value from the actual audio output fixture"

    split = config["split"].as_i
    config["batches"].as_i.times do |batch|
      start = (batch * seq + split) * dim
      count = (seq - split) * dim
      snapshots["audio_output"][start, count].should eq(snapshots["output"][start, count]),
        "audio injection must preserve the reference/motion suffix exactly"
    end
    snapshots["repeat_block_output"].should eq(snapshots["output"])
    snapshots["repeat_audio_output"].should eq(snapshots["audio_output"])
    snapshots["chained_block_output"].all?(&.finite?).should be_true,
      "a second block must consume the first block's resident output"
    snapshots["chained_block_output"].should_not eq(snapshots["output"]),
      "the chained block must execute on the prior block output"
    fixture["inputs"]["local_audio"].as_a.any? { |value| value.as_f != 0 }.should be_true
    prefix = fixture["config"]["split"].as_i * dim
    snapshots["audio_output"][0, prefix].should_not eq(snapshots["output"][0, prefix]),
      "nonzero audio must perturb the video prefix"
  end

  it "covers every channel in AdaIN when C exceeds 256" do
    dim = 320
    inputs = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    trace = ML::Video::Arena.new
    outputs = ML::Video::Arena.new
    begin
      cfg = ML::Video::WanS2VAudioConfig.new(
        batches: 1, seq: 2, dim: dim, heads: 20,
        video_frames: 1, spatial_tokens: 1, audio_tokens: 5,
      )
      video_row = Array(Float32).new(dim) { |i| Math.sin(i * 0.071).to_f32 }
      inputs.upload("x", video_row + Array(Float32).new(dim, 7.0_f32))
      inputs.upload("local_audio", Array(Float32).new(cfg.audio_rows * dim, 0.0_f32))
      inputs.upload("global_audio", Array(Float32).new(dim, 0.0_f32))

      weights.upload("adain_weight", Array(Float32).new(2 * dim * dim, 0.0_f32))
      adain_bias = Array(Float32).new(2 * dim, 0.0_f32)
      (256...dim).each { |i| adain_bias[i] = 0.2_f32 + (i - 256).to_f32 * 0.003_f32 }
      weights.upload("adain_bias", adain_bias)
      {"audio_q", "audio_k", "audio_v", "audio_o"}.each do |prefix|
        weights.upload("#{prefix}_weight", Array(Float32).new(dim * dim, 0.0_f32))
        weights.upload("#{prefix}_bias", Array(Float32).new(dim, 0.0_f32))
      end
      weights.upload("audio_q_norm_weight", Array(Float32).new(dim, 1.0_f32))
      weights.upload("audio_k_norm_weight", Array(Float32).new(dim, 1.0_f32))

      ML::Video::WanS2VAudioInjector.prepare!
      ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, cfg)
      y = outputs.alloc("y", cfg.values)
      adain_trace = trace.alloc("adain_video", dim)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      ML::Video::WanS2VAudioInjector.encode(
        encoder, inputs["x"], inputs["local_audio"], inputs["global_audio"],
        y, weights, scratch, cfg, {"adain_video" => adain_trace},
      )
      encoder.end_encoding
      command.commit
      command.wait

      mean = video_row.sum(&.to_f64) / dim
      variance = video_row.sum { |value| (value.to_f64 - mean) ** 2 } / dim
      denominator = Math.sqrt(variance + 1e-5)
      actual = adain_trace.read(dim)
      dim.times do |i|
        normalized = (video_row[i].to_f64 - mean) / denominator
        expected = normalized + adain_bias[i].to_f64
        actual[i].should be_close(expected, 2e-5 + 2e-5 * expected.abs),
          "AdaIN channel #{i} was not covered"
      end
      (256...dim).any? { |i| actual[i] != 0.0_f32 }.should be_true
    ensure
      outputs.release
      trace.release
      scratch.release
      weights.release
      inputs.release
    end
  end

  it "rejects overlapping no-copy Wan audio scratch wrappers" do
    page = LibWanVideoSpecMemory.getpagesize.to_u64
    ptr = Pointer(Void).null
    LibWanVideoSpecMemory.posix_memalign(pointerof(ptr), page, 3_u64 * page).should eq(0)
    inputs = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    outputs = ML::Video::Arena.new
    first = second = nil.as(ML::MetalBuffer?)
    first_in_scratch = false
    second_in_scratch = false
    begin
      dim = 8
      cfg = ML::Video::WanS2VAudioConfig.new(
        batches: 1, seq: 2, dim: dim, heads: 1,
        video_frames: 1, spatial_tokens: 1, audio_tokens: 5,
      )
      inputs.upload("x", Array(Float32).new(2 * dim, 0.1_f32))
      inputs.upload("local_audio", Array(Float32).new(5 * dim, 0.1_f32))
      inputs.upload("global_audio", Array(Float32).new(dim, 0.1_f32))
      weights.upload("adain_weight", Array(Float32).new(2 * dim * dim, 0.0_f32))
      weights.upload("adain_bias", Array(Float32).new(2 * dim, 0.0_f32))
      {"audio_q", "audio_k", "audio_v", "audio_o"}.each do |prefix|
        weights.upload("#{prefix}_weight", Array(Float32).new(dim * dim, 0.0_f32))
        weights.upload("#{prefix}_bias", Array(Float32).new(dim, 0.0_f32))
      end
      weights.upload("audio_q_norm_weight", Array(Float32).new(dim, 1.0_f32))
      weights.upload("audio_k_norm_weight", Array(Float32).new(dim, 1.0_f32))
      ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, cfg)
      outputs.alloc("y", cfg.values)
      first = ML::MetalBuffer.wrap_no_copy(ptr, (2 * page).to_i64)
      second = ML::MetalBuffer.wrap_no_copy((ptr.as(UInt8*) + page).as(Void*), (2 * page).to_i64)
      first.handle.should_not eq(second.handle)
      scratch.buffers["wan_a_gather"].release
      scratch.buffers["wan_a_norm"].release
      scratch.buffers["wan_a_gather"] = first
      first_in_scratch = true
      scratch.buffers["wan_a_norm"] = second
      second_in_scratch = true

      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError, "Wan audio scratch buffers overlap") do
        ML::Video::WanS2VAudioInjector.encode(
          encoder, inputs["x"], inputs["local_audio"], inputs["global_audio"],
          outputs["y"], weights, scratch, cfg,
        )
      end
      encoder.end_encoding
      command.discard
    ensure
      outputs.release
      scratch.release
      weights.release
      inputs.release
      first.try(&.release) unless first_in_scratch
      second.try(&.release) unless second_in_scratch
      LibWanVideoSpecMemory.free(ptr)
    end
  end
end
