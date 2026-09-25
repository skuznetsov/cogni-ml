require "./spec_helper"
require "../src/ml/gguf/qwen_image21_flow_match"

describe ML::GGUF::QwenImage21FlowMatch do
  it "matches the model scheduler's dynamic four-step sigma schedule" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(4, 256)

    # Golden values are produced by
    # spec/support/qwen_image21_flow_match_reference.py.
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
  end
end
