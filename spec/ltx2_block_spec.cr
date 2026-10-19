# Adapted from Lightricks/LTX-2 BasicAVTransformerBlock at
# e98f6d927d83c92ef2be6aec76c8440b19306e39. Changed for native Metal/Crystal;
# covered by the LTX-2.x Community License (LICENSE-2_x).
require "spec"
require "json"
require "./support/video_allocation_counter"
require "../src/ml/video/ltx2_block"

private def ltx2_float_array(value : JSON::Any) : Array(Float32)
  value.as_a.map(&.as_f.to_f32)
end

private def ltx2_fixture : JSON::Any
  path = ENV["LTX2_BLOCK_FIXTURE"]? || raise "Set LTX2_BLOCK_FIXTURE to the pinned MPS oracle fixture"
  JSON.parse(File.read(path))
end

private def ltx2_upload(arena : ML::Video::Arena, inputs : JSON::Any, weights : JSON::Any) : Nil
  inputs.as_h.each do |name, value|
    arena.upload(name, ltx2_float_array(value))
  end
  weights.as_h.each do |name, value|
    arena.upload("w.#{name}", ltx2_float_array(value))
  end
end

private def ltx2_config(fixture : JSON::Any) : ML::Video::LTX2DualStreamConfig
  ML::Video::LTX2DualStreamConfig.new(
    video_dim: fixture["config"]["video_dim"].as_i,
    video_context_dim: fixture["config"]["video_context_dim"].as_i,
    video_tokens: fixture["config"]["video_tokens"].as_i,
    video_heads: fixture["config"]["video_heads"].as_i,
    video_head_dim: fixture["config"]["video_head_dim"].as_i,
    audio_dim: fixture["config"]["audio_dim"].as_i,
    audio_context_dim: fixture["config"]["audio_context_dim"].as_i,
    audio_context_tokens: fixture["config"]["audio_context_tokens"].as_i,
    audio_tokens: fixture["config"]["audio_tokens"].as_i,
    audio_heads: fixture["config"]["audio_heads"].as_i,
    audio_head_dim: fixture["config"]["audio_head_dim"].as_i,
    video_context_tokens: fixture["config"]["video_context_tokens"].as_i,
    video_modulation_rows: fixture["config"]["video_modulation_rows"].as_i,
    audio_modulation_rows: fixture["config"]["audio_modulation_rows"].as_i,
    norm_eps: fixture["config"]["norm_eps"].as_f.to_f32,
  )
end

private def ltx2_matches(actual : Array(Float32), expected : JSON::Any) : Bool
  reference = ltx2_float_array(expected)
  return false unless actual.size == reference.size
  actual.each_with_index do |value, index|
    target = reference[index]
    return false unless value.finite? && target.finite?
    tolerance = 2e-5_f32 + 2e-5_f32 * target.abs
    return false unless (value - target).abs <= tolerance
  end
  true
end

private def ltx2_compare(actual : Array(Float32), expected : JSON::Any, label : String) : Nil
  ltx2_matches(actual, expected).should be_true, "#{label} exceeds abs 2e-5 + rel 2e-5 gate"
end

private def ltx2_snapshots(block : ML::Video::LTX2DualStreamBlock) : Hash(String, Array(Float32))
  snapshots = {} of String => Array(Float32)
  block.checkpoints.each do |name, buffer|
    snapshots[name] = buffer.read((buffer.size // sizeof(Float32)).to_i).map(&.to_f32)
  end
  snapshots
end

private def ltx2_run_case(fixture : JSON::Any, case_data : JSON::Any) : Tuple(Hash(String, Array(Float32)), Hash(String, Array(Float32)), Int32)
  arena = ML::Video::Arena.new
  begin
    ltx2_upload(arena, case_data["inputs"], fixture["weights"])
    config = ltx2_config(fixture)
    block = ML::Video::LTX2DualStreamBlock.new(arena, config, "ltx2_first")

    # Construct a second block against the first block's resident outputs, then
    # restore the uploaded seed-input entries so Arena still owns each buffer
    # exactly once. Both blocks are encoded into the same command encoder below.
    seed_video = arena["video_x"]
    seed_audio = arena["audio_x"]
    arena.buffers["video_x"] = block.video_output
    arena.buffers["audio_x"] = block.audio_output
    chained = begin
      ML::Video::LTX2DualStreamBlock.new(arena, config, "ltx2_second")
    ensure
      arena.buffers["video_x"] = seed_video
      arena.buffers["audio_x"] = seed_audio
    end
    case_data["expected"].as_h.keys.sort.should eq(block.checkpoints.keys.sort),
      "fixture must contain exactly one checkpoint per first-block output"
    case_data["chain_expected"].as_h.keys.sort.should eq(chained.checkpoints.keys.sort),
      "fixture must contain exactly one checkpoint per chained-block output"

    initial_allocations = ML::MetalBuffer.video_spec_allocations
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    block.encode(encoder)
    chained.encode(encoder)
    # Reuse the first block's preallocated scratch on the same encoder with the
    # same seed inputs; its output must remain deterministic and allocation-free.
    block.encode(encoder)
    encoder.end_encoding
    ML::MetalBuffer.video_spec_allocations.should eq(initial_allocations),
      "either encode must not allocate a MetalBuffer, including short-lived scratch"
    command.commit
    command.wait

    {ltx2_snapshots(block), ltx2_snapshots(chained), initial_allocations.to_i}
  ensure
    arena.release
  end
end

describe ML::Video::LTX2DualStreamBlock do
  it "rejects attention head dimensions above the Metal primitive limit during setup" do
    unless ENV["LTX2_BLOCK_FIXTURE"]?
      pending "MPS oracle fixture is external; set LTX2_BLOCK_FIXTURE"
    end

    fixture = ltx2_fixture
    arena = ML::Video::Arena.new
    begin
      expect_raises(ArgumentError, /cannot exceed 256/) do
        ML::Video::LTX2DualStreamBlock.new(
          arena, ltx2_config(fixture).copy_with(video_head_dim: 258)
        )
      end
      expect_raises(ArgumentError, /cannot exceed 256/) do
        ML::Video::LTX2DualStreamBlock.new(
          arena, ltx2_config(fixture).copy_with(audio_head_dim: 258)
        )
      end
    ensure
      arena.release
    end
  end

  it "matches every pinned upstream checkpoint and catches a seeded input perturbation" do
    unless ENV["LTX2_BLOCK_FIXTURE"]?
      pending "MPS oracle fixture is external; set LTX2_BLOCK_FIXTURE"
    end

    fixture = ltx2_fixture
    fixture["upstream"]["revision"].as_s.should eq("e98f6d927d83c92ef2be6aec76c8440b19306e39")
    cases = fixture["cases"].as_a
    cases.map { |item| item["name"].as_s }.should eq(["baseline", "seeded_perturbation"])

    results = {} of String => Hash(String, Array(Float32))
    cases.each do |case_data|
      snapshots, chain_snapshots, _ = ltx2_run_case(fixture, case_data)
      expected = case_data["expected"].as_h
      expected.each do |name, values|
        ltx2_compare(snapshots[name], values, "#{case_data["name"]}.#{name}")
      end
      chain_expected = case_data["chain_expected"].as_h
      chain_expected.each do |name, values|
        ltx2_compare(chain_snapshots[name], values, "#{case_data["name"]}.chain.#{name}")
      end
      snapshots["video.output"].zip(chain_snapshots["video.output"])
        .any? { |left, right| (left - right).abs > 1e-5_f32 }.should be_true,
        "second block must consume the first block's resident output"
      results[case_data["name"].as_s] = snapshots
    end

    baseline = results["baseline"]["video.output"]
    perturbed = results["seeded_perturbation"]["video.output"]
    baseline.zip(perturbed).any? { |left, right| (left - right).abs > 1e-5_f32 }.should be_true
    baseline_audio = results["baseline"]["audio.output"]
    perturbed_audio = results["seeded_perturbation"]["audio.output"]
    baseline_audio.zip(perturbed_audio).any? { |left, right| (left - right).abs > 1e-5_f32 }.should be_true,
      "video-only perturbation must propagate through V2A into audio output"

    # Sensitivity control: a deliberately corrupted target must fail the exact fixed gate.
    corrupted = ltx2_float_array(cases[0]["expected"]["video.output"])
    corrupted[0] += 0.01_f32
    corrupted_json = JSON::Any.new(corrupted.map { |value| JSON::Any.new(value.to_f64) })
    ltx2_matches(baseline, corrupted_json).should be_false
  end
end
