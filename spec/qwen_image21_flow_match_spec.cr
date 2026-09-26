require "./spec_helper"
require "json"
require "digest/sha256"
require "../src/ml/gguf/qwen_image21_flow_match"

describe ML::GGUF::QwenImage21FlowMatch do
  it "matches the model scheduler's dynamic four-step sigma schedule" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(4, 256)

    schedule.mu.should be_close(0.5_f32, 1e-7_f32)
    schedule.sigmas.zip([
      1.0_f32,
      0.744611382_f32,
      0.426673472_f32,
      0.019999981_f32,
      0.0_f32,
    ]).each do |actual, expected|
      actual.should be_close(expected, 2e-7_f32)
    end
    schedule.timesteps.zip([
      1000.0_f32,
      744.611389_f32,
      426.673462_f32,
      19.999981_f32,
    ]).each do |actual, expected|
      actual.should be_close(expected, 2e-4_f32)
    end
  end

  it "matches pinned Diffusers sigma and timestep arrays at 2304 target tokens" do
    reference = JSON.parse(File.read(
      File.join(__DIR__, "fixtures", "qwen_image21_flow_match_diffusers.json")
    )).as_h
    provenance = reference["provenance"].as_h
    provenance["model_revision"].as_s.should eq("790c92633540aa0cb11d9abf19eb46d861714758")
    provenance["scheduler_config_sha256"].as_s.should eq(
      "5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1"
    )
    provenance["diffusers_commit"].as_s.should eq("8b3c707ebd3ec4881f4190cf42931da07eaf3b65")

    cases = reference["schedules"].as_a.select do |entry|
      entry["image_seq_len"].as_i == 2304
    end
    cases.map { |entry| entry["steps"].as_i }.should eq([20, 24, 40])

    cases.each do |entry|
      steps = entry["steps"].as_i
      schedule = ML::GGUF::QwenImage21FlowMatch.schedule(
        steps, entry["image_seq_len"].as_i
      )
      schedule.mu.should eq(entry["mu"].as_f32)
      schedule.sigmas.should eq(entry["sigmas"].as_a.map(&.as_f32))
      reference_timesteps = entry["timesteps"].as_a.map(&.as_f32)
      schedule.timesteps.should eq(reference_timesteps)
      # The pinned pipeline sends t.to(latents.dtype) / 1000, not raw sigma.
      reference_timesteps.each_with_index do |timestep, index|
        schedule.model_timestep(index).should eq(timestep / 1000.0_f32)
      end
    end
  end

  it "uses the exact Qwen-Image 2.1 resolution shift endpoints" do
    ML::GGUF::QwenImage21FlowMatch.calculate_mu(256).should be_close(0.5_f32, 1e-7_f32)
    ML::GGUF::QwenImage21FlowMatch.calculate_mu(8192).should be_close(0.9_f32, 1e-7_f32)
  end

  it "matches an independent four-step Euler trajectory" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(4, 256)
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.25_f32, -0.5_f32, 1.25_f32],
      schedule,
    ) do |latents, timestep, index|
      latents.map { |value| value * 0.25_f32 + timestep * (index + 1) }
    end

    result.zip([
      -0.960330307_f32,
      -1.538025737_f32,
      -0.190069795_f32,
    ]).each do |actual, expected|
      actual.should be_close(expected, 3e-6_f32)
    end
  end

  it "keeps the default solver identical to explicitly selected Euler" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(4, 256)
    initial = [0.25_f32, -0.5_f32, 1.25_f32]
    predictor = ->(latents : Array(Float32), _timestep : Float32, _index : Int32) {
      latents.map { |value| value * 0.25_f32 }
    }
    default_result = ML::GGUF::QwenImage21FlowMatch.denoise(initial, schedule) do |latents, timestep, index|
      predictor.call(latents, timestep, index)
    end
    explicit_euler_result = ML::GGUF::QwenImage21FlowMatch.denoise(
      initial, schedule,
      solver: ML::GGUF::QwenImage21FlowMatchSolver::Euler,
    ) do |latents, timestep, index|
      predictor.call(latents, timestep, index)
    end

    default_result.should eq(explicit_euler_result)
  end

  it "uses nonuniform Adams-Bashforth 2 steps to improve the analytic exponential ODE" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(5, 256)
    first_interval = schedule.sigmas[1] - schedule.sigmas[0]
    second_interval = schedule.sigmas[2] - schedule.sigmas[1]
    first_interval.should_not eq(second_interval)

    euler = ML::GGUF::QwenImage21FlowMatch.denoise([1.0_f32], schedule) do |latents, _timestep, _index|
      latents
    end
    ab2 = ML::GGUF::QwenImage21FlowMatch.denoise(
      [1.0_f32], schedule,
      solver: ML::GGUF::QwenImage21FlowMatchSolver::AdamsBashforth2,
    ) do |latents, _timestep, _index|
      latents
    end

    exact = Math.exp(-1.0).to_f32
    (ab2[0] - exact).abs.should be < (euler[0] - exact).abs
  end

  it "integrates a constant field exactly under variable-step Adams-Bashforth 2" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(5, 256)
    initial = [0.25_f32, -0.5_f32, 1.25_f32]
    field = [2.0_f32, -3.0_f32, 0.5_f32]

    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      initial, schedule,
      solver: ML::GGUF::QwenImage21FlowMatchSolver::AdamsBashforth2,
    ) do |_latents, _timestep, _index|
      field
    end

    total_interval = schedule.sigmas.last - schedule.sigmas.first
    result.zip(initial.zip(field).map { |value, derivative| value + total_interval * derivative }).each do |actual, expected|
      actual.should be_close(expected, 2e-6_f32)
    end
  end

  it "uses each nonuniform h_i / h_(i-1) ratio in Adams-Bashforth 2" do
    schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 0.8_f32, 0.3_f32, 0.0_f32],
      [1000.0_f32, 800.0_f32, 300.0_f32],
      0.5_f32,
    )
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [1.0_f32], schedule,
      solver: ML::GGUF::QwenImage21FlowMatchSolver::AdamsBashforth2,
    ) do |latents, _timestep, _index|
      latents
    end

    # Euler startup gives y_1=0.8. The next step has h_1/h_0=2.5,
    # then h_2/h_1=0.6, yielding y_3=0.39225.
    result[0].should be_close(0.39225_f32, 1e-7_f32)
  end

  it "rejects a zero previous sigma interval in Adams-Bashforth 2" do
    schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 1.0_f32, 0.5_f32],
      [1000.0_f32, 1000.0_f32],
      0.5_f32,
    )
    expect_raises(ArgumentError, "AB2 requires finite nonzero sigma intervals") do
      schedule.step_ab2([1.0_f32], [1.0_f32], [1.0_f32], 1)
    end
  end

  it "rejects unsupported solver names used by the opt-in environment setting" do
    expect_raises(ArgumentError, "solver must be euler or ab2") do
      ML::GGUF::QwenImage21FlowMatchSolver.parse("rk4")
    end
  end

  it "reports opt-in per-step elapsed time without changing the trajectory" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(2, 256)
    predictor = ->(latents : Array(Float32), timestep : Float32, index : Int32) {
      latents.map { |value| value * 0.25_f32 + timestep * (index + 1) }
    }
    baseline = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.25_f32, -0.5_f32, 1.25_f32], schedule,
    ) do |latents, timestep, index|
      predictor.call(latents, timestep, index)
    end
    observations = [] of Tuple(Int32, Float32, Float32, Float64)
    instrumented = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.25_f32, -0.5_f32, 1.25_f32], schedule,
      step_observer: ->(index : Int32, sigma : Float32, timestep : Float32, elapsed : Time::Span) {
        observations << {index, sigma, timestep, elapsed.total_milliseconds}
        nil
      },
    ) do |latents, timestep, index|
      predictor.call(latents, timestep, index)
    end

    instrumented.should eq(baseline)
    observations.map(&.[0]).should eq([0, 1])
    observations.map(&.[1]).should eq(schedule.sigmas.first(2))
    observations.map(&.[2]).should eq(schedule.timesteps)
    observations.all? { |_, _, _, elapsed_ms| elapsed_ms >= 0.0 }.should be_true
  end

  it "reports hashes of each post-Euler latent state when requested" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(3, 256)
    initial = [0.25_f32, -0.5_f32, 1.25_f32]
    prediction = [1.0_f32, 2.0_f32, 3.0_f32]
    expected_latents = initial.dup
    expected_hashes = [] of String
    schedule.step_count.times do |index|
      dt = schedule.sigmas[index + 1] - schedule.sigmas[index]
      expected_latents = Array(Float32).new(expected_latents.size) do |latent_index|
        expected_latents[latent_index] + dt * prediction[latent_index]
      end
      bytes = IO::Memory.new
      expected_latents.each { |value| bytes.write_bytes(value, IO::ByteFormat::LittleEndian) }
      expected_hashes << Digest::SHA256.hexdigest(bytes.to_slice)
    end

    observations = [] of Tuple(Int32, Float32, String)
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      initial, schedule,
      step_latent_hash_observer: ->(index : Int32, timestep : Float32, sha256 : String) {
        observations << {index, timestep, sha256}
        nil
      },
    ) do |_latents, _timestep, _index|
      prediction
    end

    observations.map(&.[0]).should eq([0, 1, 2])
    observations.map(&.[1]).should eq(schedule.timesteps)
    observations.map(&.[2]).should eq(expected_hashes)
    result.should eq(expected_latents)
  end

  it "rejects schedules too short for terminal stretching" do
    expect_raises(ArgumentError, "num_inference_steps must be at least two") do
      ML::GGUF::QwenImage21FlowMatch.schedule(0, 256)
    end
    expect_raises(ArgumentError, "num_inference_steps must be at least two") do
      ML::GGUF::QwenImage21FlowMatch.schedule(1, 256)
    end
  end

  it "fails at the first non-finite transformer output" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(2, 256)
    expect_raises(ArgumentError, "transformer produced non-finite output at denoising step 0") do
      ML::GGUF::QwenImage21FlowMatch.denoise([0.0_f32], schedule) do |_latents, _timestep, _index|
        [Float32::NAN]
      end
    end

    expect_raises(ArgumentError, "transformer produced non-finite output at denoising step 0") do
      ML::GGUF::QwenImage21FlowMatch.denoise(
        [0.0_f32], schedule,
        solver: ML::GGUF::QwenImage21FlowMatchSolver::AdamsBashforth2,
      ) do |_latents, _timestep, _index|
        [Float32::NAN]
      end
    end
  end

  it "fails when an Adams-Bashforth update produces non-finite latents" do
    schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 0.5_f32, 0.0_f32],
      [1000.0_f32, 500.0_f32],
      0.5_f32,
    )
    expect_raises(ArgumentError, "non-finite latents after denoising step 1") do
      ML::GGUF::QwenImage21FlowMatch.denoise(
        [Float32::MAX], schedule,
        solver: ML::GGUF::QwenImage21FlowMatchSolver::AdamsBashforth2,
      ) do |_latents, _timestep, index|
        [index == 0 ? Float32::MAX : 3.0e38_f32]
      end
    end
  end
end
