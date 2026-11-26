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
        expected_float32_time = timestep / 1000.0_f32
        schedule.model_timestep(index).should eq(expected_float32_time)
        schedule.model_timestep(
          index,
          ML::GGUF::QwenImage21ModelTimestepPrecision::Float32,
        ).should eq(expected_float32_time)
      end
    end
  end

  it "matches the pinned official BF16 timestep path for the 1024-token Russian run" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(40, 1024)
    {
      {20, 0x441bc0d7_u32},
      {21, 0x4415b8e1_u32},
    }.each do |index, expected_bits|
      schedule.timesteps[index].unsafe_as(UInt32).should eq(expected_bits)
    end

    official_bf16_time_bits = [
      0x3f80_u16, 0x3f7c_u16, 0x3f78_u16, 0x3f74_u16, 0x3f70_u16,
      0x3f6c_u16, 0x3f67_u16, 0x3f63_u16, 0x3f5e_u16, 0x3f5a_u16,
      0x3f55_u16, 0x3f51_u16, 0x3f4c_u16, 0x3f47_u16, 0x3f42_u16,
      0x3f3c_u16, 0x3f36_u16, 0x3f31_u16, 0x3f2b_u16, 0x3f26_u16,
      0x3f20_u16, 0x3f1a_u16, 0x3f13_u16, 0x3f0c_u16, 0x3f06_u16,
      0x3efe_u16, 0x3ef0_u16, 0x3ee1_u16, 0x3ed2_u16, 0x3ec3_u16,
      0x3eb2_u16, 0x3ea2_u16, 0x3e91_u16, 0x3e80_u16, 0x3e5b_u16,
      0x3e36_u16, 0x3e10_u16, 0x3dd1_u16, 0x3d7c_u16, 0x3ca4_u16,
    ]

    schedule.step_count.should eq(40)
    official_bf16_time_bits.each_with_index do |expected_bits, index|
      schedule.model_timestep(
        index,
        ML::GGUF::QwenImage21ModelTimestepPrecision::BFloat16,
      ).unsafe_as(UInt32).should eq(
        expected_bits.to_u32 << 16
      )
    end
  end

  it "uses ties-to-even when rounding BF16 model timesteps" do
    precision = ML::GGUF::QwenImage21ModelTimestepPrecision::BFloat16
    halfway_above_even_lower = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 0.0_f32], [1.00390625_f32], 0.0_f32
    )
    halfway_above_odd_lower = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 0.0_f32], [1.01171875_f32], 0.0_f32
    )

    # Both expectations include the official BF16 cast of t, division by 1000,
    # and the BF16 output cast, widened back to Float32 for the native backend.
    halfway_above_even_lower.model_timestep(0, precision).unsafe_as(UInt32).should eq(0x3a830000_u32)
    halfway_above_odd_lower.model_timestep(0, precision).unsafe_as(UInt32).should eq(0x3a850000_u32)
  end

  it "passes the selected timestep precision through the denoising callback" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(40, 1024)
    observed = [] of Tuple(Int32, UInt32)

    ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.0_f32], schedule,
      timestep_precision: ML::GGUF::QwenImage21ModelTimestepPrecision::BFloat16,
    ) do |_latents, timestep, index|
      observed << {index, timestep.unsafe_as(UInt32)}
      [0.0_f32]
    end

    observed.size.should eq(40)
    observed[20].should eq({20, 0x3f20_u32 << 16})
  end

  it "parses and labels model timestep precision modes" do
    ML::GGUF::QwenImage21ModelTimestepPrecision.parse("float32").label.should eq("float32")
    ML::GGUF::QwenImage21ModelTimestepPrecision.parse("bfloat16").label.should eq("bfloat16")
    expect_raises(ArgumentError, "timestep precision must be float32 or bfloat16") do
      ML::GGUF::QwenImage21ModelTimestepPrecision.parse("float16")
    end
  end

  it "parses and labels latent state precision independently of model timestep precision" do
    ML::GGUF::QwenImage21LatentStatePrecision.parse("float32").label.should eq("float32")
    ML::GGUF::QwenImage21LatentStatePrecision.parse("bfloat16").label.should eq("bfloat16")
    expect_raises(ArgumentError, "latent state precision must be float32 or bfloat16") do
      ML::GGUF::QwenImage21LatentStatePrecision.parse("float16")
    end
  end

  it "matches the official MPS BF16 Euler product micro-fixture" do
    # The pinned MPS step keeps the FP32 dt through multiplication by the
    # BF16-rounded model output, rounds the product to BF16, adds to the
    # widened BF16 sample in Float32, then rounds the state back to BF16.
    # In particular, dt=-0.009999990463256836 is not first rounded to
    # -0.010009765625. The product is +/-0.0191650390625 and the final pair
    # is +/-0.10205078125. Torch CPU's 0-D scalar promotion pre-rounds dt and
    # instead yields +/-0.1015625 here; that backend path is not the target.
    schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 0.99_f32], [1000.0_f32], 0.5_f32,
    )
    observed_inputs = [] of Array(Float32)
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [1.00390625_f32, -1.00390625_f32, -0.12109375_f32, 0.12109375_f32], schedule,
      latent_state_precision: ML::GGUF::QwenImage21LatentStatePrecision::BFloat16,
    ) do |latents, _timestep, _index|
      observed_inputs << latents.dup
      [0.0_f32, 0.0_f32, -1.91796875_f32, 1.91796875_f32]
    end

    observed_inputs.should eq([[1.0_f32, -1.0_f32, -0.12109375_f32, 0.12109375_f32]])
    result.map(&.unsafe_as(UInt32)).should eq([
      0x3f800000_u32, 0xbf800000_u32, 0xbdd10000_u32, 0x3dd10000_u32,
    ])
    result.all? { |value| value.unsafe_as(UInt32) & 0x0000ffff_u32 == 0 }.should be_true
  end

  it "keeps positive and negative FP32 intervals until BF16 product rounding" do
    precision = ML::GGUF::QwenImage21LatentStatePrecision::BFloat16
    positive_schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [0.0_f32, 0.009999990463256836_f32], [1000.0_f32], 0.5_f32,
    )
    positive_result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.58984375_f32], positive_schedule, latent_state_precision: precision,
    ) do |_latents, _timestep, _index|
      [-1.75_f32]
    end

    negative_schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [0.0_f32, -0.009999990463256836_f32], [1000.0_f32], 0.5_f32,
    )
    negative_result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.58984375_f32], negative_schedule, latent_state_precision: precision,
    ) do |_latents, _timestep, _index|
      [-1.75_f32]
    end

    # Rounding dt to BF16 first would give 0x3f12 and 0x3f1c instead.
    positive_result.map(&.unsafe_as(UInt32)).should eq([0x3f130000_u32])
    negative_result.map(&.unsafe_as(UInt32)).should eq([0x3f1b0000_u32])
  end

  it "matches an official MPS full-step coordinate that distinguishes dt rounding" do
    # Coordinate 158 of the cached official BF16 MPS step-0 capture:
    # sample=0x3ea7, model_output=0xbf57, post_step=0x3eae, dt=-0.015080928802490234.
    # Keeping dt FP32 yields BF16 product 0x3c50 and post-state 0x3eae;
    # pre-rounding dt to BF16 yields product 0x3c4f and post-state 0x3ead.
    schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 0.9849190711975098_f32], [1000.0_f32], 0.5_f32,
    )
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.326171875_f32], schedule,
      latent_state_precision: ML::GGUF::QwenImage21LatentStatePrecision::BFloat16,
    ) do |_latents, _timestep, _index|
      [-0.83984375_f32]
    end

    result.map(&.unsafe_as(UInt32)).should eq([0x3eae0000_u32])
  end

  it "replays the official BF16 first-step latent slice bitwise" do
    x0_words = [
      0xbf79_u16, 0xbfa5_u16, 0xbff4_u16, 0xbfce_u16,
      0xbf7f_u16, 0x3f15_u16, 0x3f1a_u16, 0x3f0e_u16,
      0xbf75_u16, 0xbed8_u16, 0x3fd4_u16, 0x3e16_u16,
      0x3fea_u16, 0xbe78_u16, 0x3f41_u16, 0x3fc9_u16,
    ]
    model_output_words = [
      0xbd42_u16, 0xbfb1_u16, 0xbf92_u16, 0xbf27_u16,
      0xbee6_u16, 0xbe81_u16, 0x3e29_u16, 0x3fd5_u16,
      0xbf5f_u16, 0xbef1_u16, 0x3f21_u16, 0xbea6_u16,
      0x3ef0_u16, 0xbe24_u16, 0x3f50_u16, 0x4034_u16,
    ]
    expected_words = [
      0xbf79_u16, 0xbfa2_u16, 0xbff2_u16, 0xbfcd_u16,
      0xbf7d_u16, 0x3f16_u16, 0x3f19_u16, 0x3f08_u16,
      0xbf72_u16, 0xbed4_u16, 0x3fd3_u16, 0x3e1b_u16,
      0x3fe9_u16, 0xbe76_u16, 0x3f3e_u16, 0x3fc4_u16,
    ]
    dt = -0.015080928802490234_f32
    schedule = ML::GGUF::QwenImage21FlowMatchSchedule.new(
      [1.0_f32, 1.0_f32 + dt], [1000.0_f32], 0.5_f32,
    )
    widen_bfloat16 = ->(words : Array(UInt16)) {
      words.map { |word| (word.to_u32 << 16).unsafe_as(Float32) }
    }
    initial_latents = widen_bfloat16.call(x0_words)
    model_output = widen_bfloat16.call(model_output_words)
    observed_inputs = [] of Array(Float32)

    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      initial_latents, schedule,
      latent_state_precision: ML::GGUF::QwenImage21LatentStatePrecision::BFloat16,
    ) do |latents, _timestep, _index|
      observed_inputs << latents.dup
      model_output
    end

    observed_inputs.should eq([initial_latents])
    result.map(&.unsafe_as(UInt32)).should eq(
      expected_words.map { |word| word.to_u32 << 16 }
    )
  end

  it "keeps every observed and returned latent BF16-exact across Euler steps" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(4, 256)
    observed_inputs = [] of Array(Float32)
    observed_states = [] of Array(Float32)
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      [0.123456_f32, -0.987654_f32], schedule,
      latent_state_precision: ML::GGUF::QwenImage21LatentStatePrecision::BFloat16,
      step_latent_snapshot_observer: ->(_index : Int32, _timestep : Float32, latents : Array(Float32)) {
        observed_states << latents.dup
        nil
      },
    ) do |latents, _timestep, _index|
      observed_inputs << latents.dup
      [0.234567_f32, -0.345678_f32]
    end

    bf16_exact = ->(latents : Array(Float32)) {
      latents.all? { |value| value.finite? && (value.unsafe_as(UInt32) & 0x0000ffff_u32) == 0 }
    }
    observed_inputs.all? { |latents| bf16_exact.call(latents) }.should be_true
    observed_states.size.should eq(schedule.step_count)
    observed_states.all? { |latents| bf16_exact.call(latents) }.should be_true
    bf16_exact.call(result).should be_true
  end

  it "rejects BF16 latent state with Adams-Bashforth 2" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(3, 256)
    expect_raises(ArgumentError, "bfloat16 latent state precision requires the euler solver") do
      ML::GGUF::QwenImage21FlowMatch.denoise(
        [0.0_f32], schedule,
        solver: ML::GGUF::QwenImage21FlowMatchSolver::AdamsBashforth2,
        latent_state_precision: ML::GGUF::QwenImage21LatentStatePrecision::BFloat16,
      ) do |_latents, _timestep, _index|
        [0.0_f32]
      end
    end
  end

  it "keeps default Float32 latent-state behavior identical to explicit selection" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(4, 256)
    initial = [0.25_f32, -0.5_f32, 1.25_f32]
    predictor = ->(latents : Array(Float32), timestep : Float32, index : Int32) {
      latents.map { |value| value * 0.25_f32 + timestep * (index + 1) }
    }
    default_result = ML::GGUF::QwenImage21FlowMatch.denoise(initial, schedule) do |latents, timestep, index|
      predictor.call(latents, timestep, index)
    end
    explicit_f32_result = ML::GGUF::QwenImage21FlowMatch.denoise(
      initial, schedule,
      latent_state_precision: ML::GGUF::QwenImage21LatentStatePrecision::Float32,
    ) do |latents, timestep, index|
      predictor.call(latents, timestep, index)
    end

    explicit_f32_result.should eq(default_result)
  end

  it "rejects non-finite and overflowing BF16 latent state conversions" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(2, 256)
    precision = ML::GGUF::QwenImage21LatentStatePrecision::BFloat16
    expect_raises(ArgumentError, "BF16 latent conversion requires a finite value") do
      ML::GGUF::QwenImage21FlowMatch.denoise(
        [Float32::INFINITY], schedule, latent_state_precision: precision,
      ) do |_latents, _timestep, _index|
        [0.0_f32]
      end
    end
    expect_raises(ArgumentError, "BF16 latent conversion produced a non-finite value") do
      ML::GGUF::QwenImage21FlowMatch.denoise(
        [0.0_f32], schedule, latent_state_precision: precision,
      ) do |_latents, _timestep, _index|
        [Float32::MAX]
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

  it "reports post-update latent snapshots without exposing the live trajectory" do
    schedule = ML::GGUF::QwenImage21FlowMatch.schedule(3, 256)
    initial = [0.25_f32, -0.5_f32]
    prediction = [1.0_f32, -2.0_f32]
    expected_states = [] of Array(Float32)
    expected_state = initial.dup
    schedule.step_count.times do |index|
      expected_state = schedule.step(expected_state, prediction, index)
      expected_states << expected_state
    end

    observations = [] of Tuple(Int32, Float32, Array(Float32))
    result = ML::GGUF::QwenImage21FlowMatch.denoise(
      initial, schedule,
      step_latent_snapshot_observer: ->(index : Int32, timestep : Float32, latents : Array(Float32)) {
        observations << {index, timestep, latents.dup}
        latents.fill(99.0_f32)
        nil
      },
    ) do |_latents, _timestep, _index|
      prediction
    end

    observations.map(&.[0]).should eq([0, 1, 2])
    observations.map(&.[1]).should eq(schedule.timesteps)
    observations.map(&.[2]).should eq(expected_states)
    result.should eq(expected_state)
    initial.should eq([0.25_f32, -0.5_f32])
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
