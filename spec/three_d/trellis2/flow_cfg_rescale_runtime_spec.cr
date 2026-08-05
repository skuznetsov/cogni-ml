require "json"
require "../../spec_helper"

private def flow_cfg_rescale_runtime_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_cfg_rescale_cpu_v1.json"
  )
  JSON.parse(File.read(path))
end

private def flow_cfg_rescale_runtime_case(
  fixture : JSON::Any,
  name : String,
) : JSON::Any
  fixture["cases"].as_a.find! { |item| item["name"].as_s == name }
end

private def flow_cfg_rescale_runtime_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each do |value|
      flow_cfg_rescale_runtime_collect_f32(value, values)
    end
  else
    values << node.as_f.to_f32
  end
end

private def flow_cfg_rescale_runtime_values(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_cfg_rescale_runtime_collect_f32(node["values"], values)
  values
end

private def flow_cfg_rescale_runtime_tensor(node : JSON::Any) : ML::Tensor
  shape = ML::Shape.new(node["shape"].as_a.map(&.as_i.to_i32))
  ML::Tensor.from_array(flow_cfg_rescale_runtime_values(node), shape)
end

private def flow_cfg_rescale_runtime_assert_close(
  actual : Indexable(Float32),
  expected : Indexable(Float32),
  tolerance : Float32,
) : Nil
  actual.size.should eq(expected.size)
  actual.zip(expected).each do |left, right|
    left.should be_close(right, tolerance)
  end
end

describe ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU do
  it "preserves CFG routing and source x0-space rescale on the finite oracle" do
    fixture = flow_cfg_rescale_runtime_fixture
    normalized_t = fixture["contract"]["normalized_t"].as_f
    sigma_min = fixture["contract"]["sigma_min"].as_f.to_f32
    positive_condition = ["positive"]
    negative_condition = ["negative"]

    [
      {"mixed_zero_bypass", 0.0_f64},
      {"mixed_negative_bypass", -0.5_f64},
      {"mixed_finite_rescale", 0.35_f64},
      {"mixed_unclamped_rescale", 1.7_f64},
    ].each do |name, guidance_rescale|
      source_case = flow_cfg_rescale_runtime_case(fixture, name)
      inputs = source_case["inputs"]
      x_t = flow_cfg_rescale_runtime_tensor(inputs["x_t"])
      positive = flow_cfg_rescale_runtime_tensor(inputs["positive"])
      negative = flow_cfg_rescale_runtime_tensor(inputs["negative"])
      model_t = (1000.0_f64 * normalized_t).to_f32
      model_timesteps = ML::Tensor.from_array(
        Array.new(x_t.shape[0], model_t),
        ML::Shape.new(x_t.shape[0])
      )
      calls = [] of String
      x_before = x_t.to_a
      model_t_before = model_timesteps.to_a

      result = ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          x_t,
          model_timesteps,
          positive_condition,
          negative_condition,
          sigma_min: sigma_min,
          normalized_t: normalized_t,
          guidance_strength: source_case["guidance_strength"].as_f,
          guidance_rescale: guidance_rescale,
          max_result_bytes: 64_i64
      ) do |actual_x, actual_model_t, actual_condition|
          actual_x.object_id.should eq(x_t.object_id)
          actual_model_t.object_id.should eq(model_timesteps.object_id)
          if actual_condition.object_id == positive_condition.object_id
            calls << "positive"
            positive
          else
            actual_condition.object_id.should eq(negative_condition.object_id)
            calls << "negative"
            negative
          end
        end

      calls.should eq(["positive", "negative"])
      expected = flow_cfg_rescale_runtime_values(source_case["output"])
      tolerance = guidance_rescale > 0.0_f64 ? 1e-3_f32 : 0.0_f32
      flow_cfg_rescale_runtime_assert_close(result.to_a, expected, tolerance)
      result.object_id.should_not eq(x_t.object_id)
      result.object_id.should_not eq(positive.object_id)
      result.object_id.should_not eq(negative.object_id)
      result.shares_storage_with?(x_t).should be_false
      result.shares_storage_with?(positive).should be_false
      result.shares_storage_with?(negative).should be_false
      x_t.to_a.should eq(x_before)
      model_timesteps.to_a.should eq(model_t_before)
    end
  end

  it "keeps exact special strengths ahead of all conversion controls" do
    x_t = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )
    model_timesteps = ML::Tensor.from_array(
      [500.0_f32, 500.0_f32],
      ML::Shape.new(2_i32)
    )
    positive = ML::Tensor.from_array([5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32], x_t.shape)
    negative = ML::Tensor.from_array([9.0_f32, 10.0_f32, 11.0_f32, 12.0_f32], x_t.shape)
    positive_condition = ["positive"]
    negative_condition = ["negative"]

    [
      {1.0_f64, Float64::INFINITY, positive, "positive"},
      {0.0_f64, Float64::NAN, negative, "negative"},
      {0.0_f64, 0.35_f64, negative, "negative"},
    ].each do |strength, rescale, expected, expected_call|
      calls = [] of String
      result = ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          x_t,
          model_timesteps,
          positive_condition,
          negative_condition,
          sigma_min: Float32::NAN,
          normalized_t: Float64::NAN,
          guidance_strength: strength,
          guidance_rescale: rescale,
          max_result_bytes: 16_i64
      ) do |_actual_x, _actual_model_t, condition|
          calls << condition.first
          condition.first == "positive" ? positive : negative
        end

      calls.should eq([expected_call])
      result.object_id.should eq(expected.object_id)
    end
  end

  it "preflights active rescale policy and its two-result payload cap" do
    x_t = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 5.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )
    model_timesteps = ML::Tensor.from_array(
      [500.0_f32, 500.0_f32],
      ML::Shape.new(2_i32)
    )
    positive = ML::Tensor.from_array([0.0_f32, 1.0_f32, 1.0_f32, 3.0_f32], x_t.shape)
    negative = ML::Tensor.from_array([1.0_f32, 0.0_f32, 2.0_f32, 1.0_f32], x_t.shape)
    calls = 0
    invoke = ->(sigma_min : Float32, normalized_t : Float64, rescale : Float64, max_bytes : Int64) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          x_t,
          model_timesteps,
          ["positive"],
          ["negative"],
          sigma_min: sigma_min,
          normalized_t: normalized_t,
          guidance_strength: 1.7_f64,
          guidance_rescale: rescale,
          max_result_bytes: max_bytes
      ) do |_actual_x, _actual_model_t, condition|
          calls += 1
          condition.first == "positive" ? positive : negative
        end
    end

    bypass = invoke.call(
      Float32::NAN,
      Float64::NAN,
      0.0_f64,
      16_i64
    )
    bypass.to_a.all?(&.finite?).should be_true
    calls.should eq(2)
    calls = 0

    {Float64::NAN, Float64::INFINITY, -Float64::INFINITY}.each do |rescale|
      expect_raises(ArgumentError, /guidance_rescale/) do
        invoke.call(1e-5_f32, 0.5_f64, rescale, 32_i64)
      end
    end
    expect_raises(ArgumentError, /sigma_min/) do
      invoke.call(Float32::NAN, 0.5_f64, 0.35_f64, 32_i64)
    end
    expect_raises(ArgumentError, /normalized_t/) do
      invoke.call(1e-5_f32, Float64::NAN, 0.35_f64, 32_i64)
    end
    expect_raises(ArgumentError, /conversion denominator/) do
      invoke.call(0.0_f32, 0.0_f64, 0.35_f64, 32_i64)
    end
    expect_raises(ArgumentError, /rescale result budget requires 32 bytes/) do
      invoke.call(1e-5_f32, 0.5_f64, 0.35_f64, 31_i64)
    end
    calls.should eq(0)

    result = invoke.call(1e-5_f32, 0.5_f64, 0.35_f64, 32_i64)
    result.to_a.all?(&.finite?).should be_true
    calls.should eq(2)
  end

  it "rejects undefined sample variance and zero cfg std without an epsilon" do
    rank_one = ML::Tensor.from_array([1.0_f32, 2.0_f32], ML::Shape.new(2_i32))
    rank_one_model_t = ML::Tensor.from_array([500.0_f32, 500.0_f32], ML::Shape.new(2_i32))
    calls = 0
    expect_raises(ArgumentError, /non-batch axis/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          rank_one,
          rank_one_model_t,
          ["positive"],
          ["negative"],
          sigma_min: 1e-5_f32,
          normalized_t: 0.5_f64,
          guidance_strength: 1.7_f64,
          guidance_rescale: 0.35_f64
      ) do |actual_x, _actual_model_t, _actual_condition|
          calls += 1
          actual_x
        end
    end
    calls.should eq(0)

    single_value_samples = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32],
      ML::Shape.new(2_i32, 1_i32)
    )
    expect_raises(ArgumentError, /at least two values/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          single_value_samples,
          rank_one_model_t,
          ["positive"],
          ["negative"],
          sigma_min: 1e-5_f32,
          normalized_t: 0.5_f64,
          guidance_strength: 1.7_f64,
          guidance_rescale: 0.35_f64
      ) do |actual_x, _actual_model_t, _actual_condition|
          calls += 1
          actual_x
        end
    end
    calls.should eq(0)

    fixture = flow_cfg_rescale_runtime_fixture
    zero_std = flow_cfg_rescale_runtime_case(fixture, "mixed_zero_cfg_std")
    inputs = zero_std["inputs"]
    x_t = flow_cfg_rescale_runtime_tensor(inputs["x_t"])
    positive = flow_cfg_rescale_runtime_tensor(inputs["positive"])
    negative = flow_cfg_rescale_runtime_tensor(inputs["negative"])
    normalized_t = fixture["contract"]["normalized_t"].as_f
    model_t = (1000.0_f64 * normalized_t).to_f32
    model_timesteps = ML::Tensor.from_array(
      Array.new(x_t.shape[0], model_t),
      ML::Shape.new(x_t.shape[0])
    )
    calls = 0
    expect_raises(ArgumentError, /guided x0 standard deviation must be positive/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          x_t,
          model_timesteps,
          ["positive"],
          ["negative"],
          sigma_min: 1e-5_f32,
          normalized_t: normalized_t,
          guidance_strength: zero_std["guidance_strength"].as_f,
          guidance_rescale: zero_std["guidance_rescale"]["value"].as_f,
          max_result_bytes: 64_i64
      ) do |_actual_x, _actual_model_t, condition|
          calls += 1
          condition.first == "positive" ? positive : negative
        end
    end
    calls.should eq(2)
  end

  it "admits positive near-zero variance and keeps the final tensor independent" do
    fixture = flow_cfg_rescale_runtime_fixture
    source_case = flow_cfg_rescale_runtime_case(
      fixture,
      "mixed_near_zero_cfg_std"
    )
    inputs = source_case["inputs"]
    x_t = flow_cfg_rescale_runtime_tensor(inputs["x_t"])
    positive = flow_cfg_rescale_runtime_tensor(inputs["positive"])
    negative = flow_cfg_rescale_runtime_tensor(inputs["negative"])
    normalized_t = fixture["contract"]["normalized_t"].as_f
    model_t = (1000.0_f64 * normalized_t).to_f32
    model_timesteps = ML::Tensor.from_array(
      Array.new(x_t.shape[0], model_t),
      ML::Shape.new(x_t.shape[0])
    )

    result = ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
      .predict_velocity_with_rescale(
        x_t,
        model_timesteps,
        ["positive"],
        ["negative"],
        sigma_min: 1e-5_f32,
        normalized_t: normalized_t,
        guidance_strength: source_case["guidance_strength"].as_f,
        guidance_rescale: source_case["guidance_rescale"]["value"].as_f,
        max_result_bytes: 64_i64
    ) do |_actual_x, _actual_model_t, condition|
        condition.first == "positive" ? positive : negative
      end

    result.to_a.all?(&.finite?).should be_true
    result.to_a.any? { |value| value.abs > 100_000.0_f32 }.should be_true
    result.shares_storage_with?(x_t).should be_false
    result.shares_storage_with?(positive).should be_false
    result.shares_storage_with?(negative).should be_false
  end

  it "propagates active-rescale provider exceptions without retry" do
    x_t = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 5.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )
    model_timesteps = ML::Tensor.from_array(
      [500.0_f32, 500.0_f32],
      ML::Shape.new(2_i32)
    )
    calls = 0

    expect_raises(Exception, /negative rescale provider failed/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU
        .predict_velocity_with_rescale(
          x_t,
          model_timesteps,
          ["positive"],
          ["negative"],
          sigma_min: 1e-5_f32,
          normalized_t: 0.5_f64,
          guidance_strength: 1.7_f64,
          guidance_rescale: 0.35_f64,
          max_result_bytes: 32_i64
      ) do |actual_x, _actual_model_t, _actual_condition|
          calls += 1
          raise "negative rescale provider failed" if calls == 2
          actual_x
        end
    end
    calls.should eq(2)
  end
end
