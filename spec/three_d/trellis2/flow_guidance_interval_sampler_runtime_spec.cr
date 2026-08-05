require "spec"
require "../../spec_helper"

describe ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU do
  it "routes raw Float64 interval time inside the repeated CFG/rescale loop" do
    noise = ML::Tensor.from_array(
      [1.0_f32, -2.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    positive_condition = [:positive]
    negative_condition = [:negative]
    noise_before = noise.to_a
    positive_before = positive_condition.dup
    negative_before = negative_condition.dup
    calls = [] of String
    model_times = [] of Float32
    states = [] of UInt64

    result = ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU.sample(
      noise,
      positive_condition,
      negative_condition,
      sigma_min: 1e-5_f32,
      steps: 2_i32,
      rescale_t: 1.0_f64,
      guidance_strength: 1.7_f64,
      guidance_interval: {0.5_f64, 0.5_f64},
      guidance_rescale: 0.35_f64,
      max_result_bytes: 32_i64
    ) do |actual_x, model_t, actual_condition|
      label = if actual_condition.object_id == positive_condition.object_id
                "positive"
              else
                actual_condition.object_id.should eq(negative_condition.object_id)
                "negative"
              end
      calls << label
      states << actual_x.object_id
      model_times << model_t.to_a.first
      offset = label == "positive" ? 0.25_f32 : -0.5_f32
      ML::Tensor.from_array(
        actual_x.to_a.map { |value| (value + offset).to_f32 },
        actual_x.shape
      )
    end

    calls.should eq(["positive", "positive", "negative"])
    model_times.should eq([1000.0_f32, 500.0_f32, 500.0_f32])
    states[0].should eq(noise.object_id)
    states[1].should eq(states[2])
    result.pred_x_t.size.should eq(2)
    result.pred_x_0.size.should eq(2)
    result.samples.to_a.each { |value| value.finite?.should be_true }
    result.pred_x_t.each { |tensor| tensor.to_a.each { |value| value.finite?.should be_true } }
    result.pred_x_0.each { |tensor| tensor.to_a.each { |value| value.finite?.should be_true } }
    noise.to_a.should eq(noise_before)
    positive_condition.should eq(positive_before)
    negative_condition.should eq(negative_before)

    baseline = ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample_with_step_provider(
      noise,
      positive_condition,
      sigma_min: 1e-5_f32,
      steps: 2_i32,
      rescale_t: 1.0_f64,
      max_result_bytes: 32_i64
    ) do |actual_x, normalized_t, _t_prev, model_timesteps, actual_condition|
      ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity_with_rescale(
        actual_x,
        model_timesteps,
        actual_condition,
        negative_condition,
        sigma_min: 1e-5_f32,
        normalized_t: normalized_t,
        requested_guidance_strength: 1.7_f64,
        guidance_interval: {0.5_f64, 0.5_f64},
        guidance_rescale: 0.35_f64,
        max_result_bytes: 32_i64
      ) do |provider_x, provider_model_t, provider_condition|
        offset = provider_condition.object_id == positive_condition.object_id ? 0.25_f32 : -0.5_f32
        ML::Tensor.from_array(
          provider_x.to_a.map { |value| (value + offset).to_f32 },
          provider_x.shape
        )
      end
    end

    result.samples.to_a.should eq(baseline.samples.to_a)
    result.pred_x_t.map(&.to_a).should eq(baseline.pred_x_t.map(&.to_a))
    result.pred_x_0.map(&.to_a).should eq(baseline.pred_x_0.map(&.to_a))
  end

  it "distinguishes adjacent interval boundaries before Float32 model-time narrowing" do
    noise = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    positive_condition = [:positive]
    negative_condition = [:negative]
    invoke = ->(interval : Tuple(Float64, Float64), calls : Array(String), model_times : Array(Float32)) do
      ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU.sample(
        noise,
        positive_condition,
        negative_condition,
        sigma_min: 0.0_f32,
        steps: 2_i32,
        rescale_t: 1.0_f64,
        guidance_strength: 1.7_f64,
        guidance_interval: interval,
        guidance_rescale: 0.0_f64,
        max_result_bytes: 32_i64
      ) do |actual_x, model_t, actual_condition|
        calls << (actual_condition.object_id == positive_condition.object_id ? "positive" : "negative")
        model_times << model_t.to_a.first
        actual_x
      end
    end

    inside_calls = [] of String
    inside_times = [] of Float32
    invoke.call({0.5_f64, 0.5_f64}, inside_calls, inside_times)
    above_calls = [] of String
    above_times = [] of Float32
    invoke.call(
      {0.5000000000000001_f64, 0.5000000000000001_f64},
      above_calls,
      above_times
    )

    inside_calls.should eq(["positive", "positive", "negative"])
    above_calls.should eq(["positive", "positive"])
    inside_times.should eq([1000.0_f32, 500.0_f32, 500.0_f32])
    above_times.should eq([1000.0_f32, 500.0_f32])
  end

  it "preflights interval admission before provider work" do
    noise = ML::Tensor.from_array([1.0_f32, 2.0_f32], ML::Shape.new(1_i32, 2_i32))
    positive_condition = [:positive]
    negative_condition = [:negative]
    calls = 0

    expect_raises(ArgumentError, /guidance_interval/) do
      ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU.sample(
        noise,
        positive_condition,
        negative_condition,
        sigma_min: 0.0_f32,
        steps: 2_i32,
        guidance_strength: 1.7_f64,
        guidance_interval: {0.75_f64, 0.25_f64},
        guidance_rescale: 0.0_f64,
        max_result_bytes: 32_i64
      ) do |_actual_x, _model_t, _actual_condition|
        calls += 1
        noise
      end
    end

    calls.should eq(0)

    expect_raises(ArgumentError, /guidance_rescale/) do
      ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU.sample(
        noise,
        positive_condition,
        negative_condition,
        sigma_min: 0.0_f32,
        steps: 2_i32,
        guidance_strength: 1.7_f64,
        guidance_interval: {0.0_f64, 1.0_f64},
        guidance_rescale: Float64::NAN,
        max_result_bytes: 32_i64
      ) do |_actual_x, _model_t, _actual_condition|
        calls += 1
        noise
      end
    end

    calls.should eq(0)
  end
end
