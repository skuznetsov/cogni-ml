# Adapted from Lightricks/LTX-2 BasicAVTransformerBlock at
# e98f6d927d83c92ef2be6aec76c8440b19306e39. Changed for native Metal/Crystal;
# covered by the LTX-2.x Community License (LICENSE-2_x).
require "spec"
require "json"
require "digest/sha256"
require "./support/video_allocation_counter"
require "../src/ml/video/ltx2_block"

private def ltx2_float_array(value : JSON::Any) : Array(Float32)
  value.as_a.map(&.as_f.to_f32)
end

private def ltx2_fixture : JSON::Any
  path = ENV["LTX2_BLOCK_FIXTURE"]? || raise "Set LTX2_BLOCK_FIXTURE to the pinned MPS oracle fixture"
  JSON.parse(File.read(path))
end

private def ltx2_upload_inputs(arena : ML::Video::Arena, inputs : JSON::Any) : Nil
  inputs.as_h.each do |name, value|
    arena.upload(name, ltx2_float_array(value))
  end
end

private def ltx2_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1024 * 1024)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def ltx2_bool(config : JSON::Any, name : String, default : Bool) : Bool
  value = config[name]?
  value ? value.as_bool : default
end

private def ltx2_config(fixture : JSON::Any) : ML::Video::LTX2DualStreamConfig
  config = fixture["config"]
  ML::Video::LTX2DualStreamConfig.new(
    video_dim: config["video_dim"].as_i,
    video_context_dim: config["video_context_dim"].as_i,
    video_tokens: config["video_tokens"].as_i,
    video_heads: config["video_heads"].as_i,
    video_head_dim: config["video_head_dim"].as_i,
    audio_dim: config["audio_dim"].as_i,
    audio_context_dim: config["audio_context_dim"].as_i,
    audio_context_tokens: config["audio_context_tokens"].as_i,
    audio_tokens: config["audio_tokens"].as_i,
    audio_heads: config["audio_heads"].as_i,
    audio_head_dim: config["audio_head_dim"].as_i,
    video_context_tokens: config["video_context_tokens"].as_i,
    video_modulation_rows: config["video_modulation_rows"].as_i,
    audio_modulation_rows: config["audio_modulation_rows"].as_i,
    video_prompt_modulation_rows: config["video_prompt_modulation_rows"]?.try(&.as_i) || 1,
    audio_prompt_modulation_rows: config["audio_prompt_modulation_rows"]?.try(&.as_i) || 1,
    cross_attention_adaln: ltx2_bool(config, "cross_attention_adaln", false),
    video_apply_gated_attention: ltx2_bool(config, "video_apply_gated_attention",
      ltx2_bool(config, "apply_gated_attention", false)),
    audio_apply_gated_attention: ltx2_bool(config, "audio_apply_gated_attention",
      ltx2_bool(config, "apply_gated_attention", false)),
    video_ff_bias: ltx2_bool(config, "video_ff_bias", true),
    audio_ff_bias: ltx2_bool(config, "audio_ff_bias", true),
    norm_eps: config["norm_eps"].as_f.to_f32,
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

private def ltx2_metrics(actual : Array(Float32), expected : JSON::Any,
                         label : String) : Tuple(Bool, String)
  reference = ltx2_float_array(expected)
  count = Math.min(actual.size, reference.size)
  max_abs = 0.0_f64
  max_abs_index = -1
  max_excess = -Float64::MAX
  max_excess_index = -1
  nonfinite_count = 0
  first_nonfinite = "none"

  count.times do |index|
    value = actual[index]
    target = reference[index]
    unless value.finite? && target.finite?
      nonfinite_count += 1
      first_nonfinite = "index=#{index} actual=#{value} expected=#{target}" if first_nonfinite == "none"
      next
    end

    delta = (value - target).abs
    tolerance = 2e-5_f32 + 2e-5_f32 * target.abs
    excess = delta.to_f64 - tolerance.to_f64
    if delta.to_f64 > max_abs
      max_abs = delta.to_f64
      max_abs_index = index
    end
    if excess > max_excess
      max_excess = excess
      max_excess_index = index
    end
  end

  shape_matches = actual.size == reference.size
  passed = shape_matches && nonfinite_count == 0 && max_excess <= 0.0_f64
  max_abs_detail = if max_abs_index >= 0
                     "index=#{max_abs_index} actual=#{actual[max_abs_index]} expected=#{reference[max_abs_index]}"
                   else
                     "index=none"
                   end
  max_excess_detail = if max_excess_index >= 0
                        index = max_excess_index
                        target = reference[index]
                        tolerance = 2e-5_f32 + 2e-5_f32 * target.abs
                        "index=#{index} actual=#{actual[index]} expected=#{target} tolerance=#{tolerance}"
                      else
                        "index=none"
                      end
  shape_detail = shape_matches ? "" : " shape_mismatch=true"
  detail = "#{label} n=#{actual.size}/#{reference.size} nonfinite=#{nonfinite_count} first_nonfinite={#{first_nonfinite}} " +
           "max_abs=#{max_abs} {#{max_abs_detail}} max_gate_excess=#{max_excess} {#{max_excess_detail}}#{shape_detail}"
  {passed, detail}
end

private def ltx2_record_comparison(actual : Array(Float32), expected : JSON::Any,
                                   label : String, reports : Array(String),
                                   failures : Array(String)) : Nil
  passed, detail = ltx2_metrics(actual, expected, label)
  reports << detail
  failures << detail unless passed
end

private def ltx2_snapshots(block : ML::Video::LTX2DualStreamBlock) : Hash(String, Array(Float32))
  snapshots = {} of String => Array(Float32)
  block.checkpoints.each do |name, buffer|
    snapshots[name] = buffer.read((buffer.size // sizeof(Float32)).to_i).map(&.to_f32)
  end
  snapshots
end

private def ltx2_dump_snapshots(path : String,
                                snapshots_by_case : Hash(String, Hash(String, Hash(String, Array(Float32))))) : Nil
  serialized = JSON.build do |json|
    json.object do
      json.field "schema", "ltx2-terminal-snapshot-dump-v1"
      json.field "fixture", ENV["LTX2_BLOCK_FIXTURE"]? || ""
      json.field "read_boundary", "Existing checkpoint buffer reads after command wait; no extra GPU read or wait"
      json.field "snapshots" do
        json.object do
          snapshots_by_case.each do |case_name, groups|
            json.field case_name do
              json.object do
                groups.each do |group_name, values|
                  json.field group_name do
                    json.object do
                      values.each do |name, array|
                        json.field name do
                          json.array do
                            array.each do |value|
                              if value.finite?
                                json.number(value.to_f64)
                              else
                                json.string(value.to_s)
                              end
                            end
                          end
                        end
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end
  File.write(path, serialized)
  STDERR.puts "LTX2 snapshot dump written from existing terminal reads: #{path}"
end

private def ltx2_run_case(fixture : JSON::Any, case_data : JSON::Any) : Tuple(Hash(String, Array(Float32)), Hash(String, Array(Float32)), Int32)
  arena = ML::Video::Arena.new
  checkpoint_file = nil.as(ML::Video::CheckpointFile?)
  prepared_weights = nil.as(ML::Video::PreparedWeights?)
  begin
    ltx2_upload_inputs(arena, case_data["inputs"])
    config = ltx2_config(fixture)
    if manifest = fixture["weights_file"]?
      expected_mode = fixture["config"]["production_width_probe"]?.try(&.as_bool) || false
      fixture["upstream"]["fixture_mode"].as_s.should eq(expected_mode ? "synthetic_builder_width_probe" : "reduced_width")
      fixture_path = ENV["LTX2_BLOCK_FIXTURE"]? || raise "Set LTX2_BLOCK_FIXTURE to the pinned MPS oracle fixture"
      relative_path = manifest["path"].as_s
      (File.basename(relative_path) == relative_path && relative_path != "." && relative_path != "..").should be_true,
        "weight sidecar must stay beside the fixture"
      sidecar_path = File.join(File.dirname(File.expand_path(fixture_path)), relative_path)
      File.size(sidecar_path).should eq(manifest["bytes"].as_i64)
      ltx2_sha256_file(sidecar_path).should eq(manifest["sha256"].as_s)

      fixture_bindings = fixture["weight_bindings"].as_a
      dtypes = fixture_bindings.map { |item| item["dtype"].as_s }.uniq
      dtypes.should eq(["F32"]), "synthetic LTX oracle weights must all be F32"
      checkpoint_dtype = ML::GGUF::SafeTensorDType.parse(dtypes.first)
      bindings = ML::Video::LTX2DualStreamBlock.checkpoint_bindings(config, checkpoint_dtype)
      fixture_bindings.size.should eq(bindings.size)
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
        fixture["weight_budget_bytes"].as_i64,
        fixture["weight_reserve_bytes"].as_i64,
      )
      plan.entries.size.should eq(bindings.size)
      plan.peak_bytes.should be <= fixture["weight_budget_bytes"].as_i64
      prepared_weights = ML::Video::PreparedWeights.new(plan, arena)
    elsif weights = fixture["weights"]?
      weights.as_h.each do |name, values|
        arena.upload("w.#{name}", ltx2_float_array(values))
      end
    else
      raise "fixture must contain a safetensors sidecar or inline weights"
    end
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
    prepared_weights.try(&.encode(encoder))
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
    prepared_weights.try(&.release)
    checkpoint_file.try(&.close)
    arena.release
  end
end

describe ML::Video::LTX2DualStreamBlock do
  it "rejects attention head dimensions above the Metal primitive limit during setup" do
    unless ENV["LTX2_BLOCK_FIXTURE"]?
      pending "MPS oracle fixture is external; set LTX2_BLOCK_FIXTURE"
    end

    fixture = ltx2_fixture
    config = ltx2_config(fixture)
    arena = ML::Video::Arena.new
    begin
      expect_raises(ArgumentError, /cannot exceed 256/) do
        ML::Video::LTX2DualStreamBlock.new(
          arena, config.copy_with(video_dim: config.video_heads * 258, video_head_dim: 258)
        )
      end
      expect_raises(ArgumentError, /cannot exceed 256/) do
        ML::Video::LTX2DualStreamBlock.new(
          arena, config.copy_with(audio_dim: config.audio_heads * 258, audio_head_dim: 258)
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
    snapshots_by_case = {} of String => Hash(String, Hash(String, Array(Float32)))
    reports = [] of String
    failures = [] of String
    cases.each do |case_data|
      snapshots, chain_snapshots, _ = ltx2_run_case(fixture, case_data)
      expected = case_data["expected"].as_h
      expected.each do |name, values|
        ltx2_record_comparison(snapshots[name], values, "#{case_data["name"]}.#{name}", reports, failures)
      end
      chain_expected = case_data["chain_expected"].as_h
      chain_expected.each do |name, values|
        ltx2_record_comparison(chain_snapshots[name], values, "#{case_data["name"]}.chain.#{name}", reports, failures)
      end
      snapshots["video.output"].zip(chain_snapshots["video.output"])
        .any? { |left, right| (left - right).abs > 1e-5_f32 }.should be_true,
        "second block must consume the first block's resident output"
      results[case_data["name"].as_s] = snapshots
      snapshots_by_case[case_data["name"].as_s] = {"first" => snapshots, "chain" => chain_snapshots}
    end

    if dump_path = ENV["LTX2_SNAPSHOT_DUMP"]?
      ltx2_dump_snapshots(dump_path, snapshots_by_case)
    end
    STDERR.puts reports.join("\n")
    failures.empty?.should be_true,
      "#{failures.size} LTX2 checkpoint(s) exceeded abs 2e-5 + rel 2e-5 gate:\n#{failures.join("\n")}"

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
