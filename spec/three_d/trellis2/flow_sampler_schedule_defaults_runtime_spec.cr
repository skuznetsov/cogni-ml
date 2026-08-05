require "spec"
require "../../spec_helper"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# FlowEulerSampler.sample defaults rescale_t to 1.0 before its normalized
# schedule is passed through the source rescale expression.

describe "TRELLIS.2 sampler schedule defaults" do
  it "binds the source-pinned numeric rescale default" do
    ML::ThreeD::Trellis2::FlowEulerSamplerCPU::DEFAULT_RESCALE_T.should eq(1.0_f64)
  end

  it "uses the omitted default for the unscaled normalized schedule" do
    noise = ML::Tensor.from_array(
      [1.0_f32],
      ML::Shape.new(1_i32)
    )
    condition = [:condition]
    observed = [] of {Float64, Float64, Float32}

    ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample_with_step_provider(
      noise,
      condition,
      sigma_min: 0.0_f32,
      steps: 2_i32,
      max_result_bytes: 16_i64
    ) do |actual_x, t, t_prev, model_t, actual_condition|
      actual_condition.object_id.should eq(condition.object_id)
      observed << {t, t_prev, model_t.to_a.first}
      ML::Tensor.from_array([0.0_f32], actual_x.shape)
    end

    observed.should eq([
      {1.0_f64, 0.5_f64, 1000.0_f32},
      {0.5_f64, 0.0_f64, 500.0_f32},
    ])
  end

  it "keeps an explicit rescale override distinct from the default" do
    noise = ML::Tensor.from_array(
      [1.0_f32],
      ML::Shape.new(1_i32)
    )
    observed = [] of {Float64, Float64}

    ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample_with_step_provider(
      noise,
      nil,
      sigma_min: 0.0_f32,
      steps: 2_i32,
      rescale_t: 2.0_f64,
      max_result_bytes: 16_i64
    ) do |actual_x, t, t_prev, _model_t, _actual_condition|
      observed << {t, t_prev}
      ML::Tensor.from_array([0.0_f32], actual_x.shape)
    end

    observed[0][0].should eq(1.0_f64)
    observed[0][1].should be_close(2.0_f64 / 3.0_f64, 1e-15_f64)
    observed[1][0].should be_close(2.0_f64 / 3.0_f64, 1e-15_f64)
    observed[1][1].should eq(0.0_f64)
  end
end
