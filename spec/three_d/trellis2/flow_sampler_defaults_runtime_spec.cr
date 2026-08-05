require "spec"
require "../../spec_helper"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# FlowEulerCfgSampler and FlowEulerGuidanceIntervalSampler both expose
# guidance_strength=3.0 as their public sampler default.

describe "TRELLIS.2 sampler guidance defaults" do
  it "binds the source-pinned numeric guidance default" do
    ML::ThreeD::Trellis2::FlowEulerSamplerCPU::DEFAULT_GUIDANCE_STRENGTH.should eq(3.0_f64)
  end

  it "uses the source-pinned default for the interval sampler" do
    noise = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    positive_condition = [:positive]
    negative_condition = [:negative]
    calls = [] of String

    ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU.sample(
      noise,
      positive_condition,
      negative_condition,
      sigma_min: 0.0_f32,
      steps: 1_i32,
      guidance_interval: {0.0_f64, 1.0_f64},
      guidance_rescale: 0.0_f64,
      max_result_bytes: 32_i64
    ) do |actual_x, _model_t, actual_condition|
      calls << if actual_condition.object_id == positive_condition.object_id
        "positive"
      else
        actual_condition.object_id.should eq(negative_condition.object_id)
        "negative"
      end
      actual_x
    end

    calls.should eq(["positive", "negative"])
  end

  it "keeps an explicit positive-only override distinct from the default" do
    noise = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    positive_condition = [:positive]
    negative_condition = [:negative]
    calls = [] of String

    ML::ThreeD::Trellis2::FlowGuidanceIntervalSamplerCPU.sample(
      noise,
      positive_condition,
      negative_condition,
      sigma_min: 0.0_f32,
      steps: 1_i32,
      guidance_strength: 1.0_f64,
      guidance_interval: {0.0_f64, 1.0_f64},
      guidance_rescale: 0.0_f64,
      max_result_bytes: 32_i64
    ) do |actual_x, _model_t, actual_condition|
      calls << if actual_condition.object_id == positive_condition.object_id
        "positive"
      else
        actual_condition.object_id.should eq(negative_condition.object_id)
        "negative"
      end
      actual_x
    end

    calls.should eq(["positive"])
  end

  it "uses the same source-pinned default for the plain CFG sampler" do
    noise = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    positive_condition = [:positive]
    negative_condition = [:negative]
    calls = [] of String

    ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceSamplerCPU.sample(
      noise,
      positive_condition,
      negative_condition,
      sigma_min: 0.0_f32,
      steps: 1_i32,
      guidance_rescale: 0.0_f64,
      max_result_bytes: 32_i64
    ) do |actual_x, _model_t, actual_condition|
      calls << if actual_condition.object_id == positive_condition.object_id
        "positive"
      else
        actual_condition.object_id.should eq(negative_condition.object_id)
        "negative"
      end
      actual_x
    end

    calls.should eq(["positive", "negative"])
  end
end
