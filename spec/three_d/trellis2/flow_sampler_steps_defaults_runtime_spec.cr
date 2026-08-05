require "spec"
require "../../spec_helper"

# TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
# FlowEulerSampler.sample and its CFG subclasses default steps to 50.

describe "TRELLIS.2 sampler steps defaults" do
  it "binds the source-pinned numeric steps default" do
    ML::ThreeD::Trellis2::FlowEulerSamplerCPU::DEFAULT_STEPS.should eq(50_i32)
  end

  it "uses the omitted default for fifty normalized schedule pairs" do
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
      max_result_bytes: 4096_i64
    ) do |actual_x, t, t_prev, model_t, actual_condition|
      actual_condition.object_id.should eq(condition.object_id)
      observed << {t, t_prev, model_t.to_a.first}
      ML::Tensor.from_array([0.0_f32], actual_x.shape)
    end

    observed.size.should eq(50)
    observed.first[0].should eq(1.0_f64)
    observed.first[1].should be_close(0.98_f64, 1e-15_f64)
    observed.first[2].should eq(1000.0_f32)
    observed.last[0].should be_close(0.02_f64, 1e-15_f64)
    observed.last[1].should eq(0.0_f64)
    observed.last[2].should eq(20.0_f32)
  end
end
