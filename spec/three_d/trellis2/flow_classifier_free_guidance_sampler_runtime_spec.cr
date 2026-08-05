require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_CFG_RESCALE_SCHEDULE_RUNTIME_SHA256 =
  "74ed078d4b7327ad2273c1814c7ec910efd284881116ec0d452666ffd4952218"

private def flow_cfg_sampler_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_cfg_rescale_schedule_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_CFG_RESCALE_SCHEDULE_RUNTIME_SHA256
  )
  JSON.parse(payload)
end

private def flow_cfg_sampler_collect_values(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_cfg_sampler_collect_values(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_cfg_sampler_values(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_cfg_sampler_collect_values(node, values)
  values
end

private def flow_cfg_sampler_tensor(node : JSON::Any) : ML::Tensor
  shape = ML::Shape.new(node["shape"].as_a.map(&.as_i.to_i32))
  ML::Tensor.from_array(flow_cfg_sampler_values(node["values"]), shape)
end

private def flow_cfg_sampler_zero_velocity(x : ML::Tensor) : ML::Tensor
  ML::Tensor.from_array(
    Array(Float32).new(x.numel.to_i, 0.0_f32),
    x.shape
  )
end

describe ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceSamplerCPU do
  it "composes one source-pinned repeated CFG/rescale loop without losing seams" do
    fixture = flow_cfg_sampler_fixture
    config = fixture["fixture"]
    noise = flow_cfg_sampler_tensor(fixture["inputs"]["noise"])
    positive_condition = flow_cfg_sampler_tensor(fixture["inputs"]["positive_condition"])
    negative_condition = flow_cfg_sampler_tensor(fixture["inputs"]["negative_condition"])
    noise_before = noise.to_a
    positive_before = positive_condition.to_a
    negative_before = negative_condition.to_a
    model_x_scale = config["model_x_scale"].as_f.to_f32
    model_t_scale = config["model_t_scale"].as_f.to_f32
    steps = config["steps"].as_i.to_i32
    calls = [] of {String, UInt64, UInt64, Float64}
    states = [] of UInt64
    model_times = [] of Array(Float32)

    result = ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceSamplerCPU.sample(
      noise,
      positive_condition,
      negative_condition,
      sigma_min: config["sigma_min"].as_f.to_f32,
      steps: steps,
      rescale_t: config["rescale_t"].as_f,
      guidance_strength: config["guidance_strength"].as_f,
      guidance_rescale: config["guidance_rescale"].as_f
    ) do |actual_x, model_t, actual_condition|
      (
        actual_condition.object_id == positive_condition.object_id ||
          actual_condition.object_id == negative_condition.object_id
      ).should be_true
      label = if actual_condition.object_id == positive_condition.object_id
                "positive"
              else
                "negative"
              end
      calls << {label, actual_x.object_id, model_t.object_id, model_t.to_a.first.to_f64}
      states << actual_x.object_id
      model_times << model_t.to_a

      x_values = actual_x.to_a
      cond_values = actual_condition.to_a
      time_term = model_t.to_a.first * model_t_scale
      velocity = x_values.map_with_index do |value, index|
        ((value * model_x_scale) + time_term) + cond_values[index]
      end
      ML::Tensor.from_array(velocity, actual_x.shape)
    end

    result.pred_x_t.size.should eq(steps)
    result.pred_x_0.size.should eq(steps)
    calls.map(&.[0]).should eq(["positive", "negative"] * steps)
    calls.each_slice(2) do |pair|
      pair.size.should eq(2)
      pair[0][1].should eq(pair[1][1])
      pair[0][3].should eq(pair[1][3])
    end
    states.each_slice(2).with_index do |pair, index|
      pair.size.should eq(2)
      pair[0].should eq(pair[1])
      if index == 0
        pair[0].should eq(noise.object_id)
      else
        pair[0].should eq(result.pred_x_t[index - 1].object_id)
      end
    end

    source_steps = fixture["steps"].as_a
    expected_schedule = source_steps.map(&.["t"].as_f)
    model_times.each_slice(2).with_index do |pair, index|
      expected_model_t = (1000.0_f64 * expected_schedule[index]).to_f32
      pair.each { |values| values.should eq([expected_model_t, expected_model_t]) }
    end
    source_steps.each_with_index do |source_step, index|
      result.pred_x_t[index].to_a.zip(
        flow_cfg_sampler_values(source_step["pred_x_prev"]["values"])
      ).each { |actual, expected| actual.should be_close(expected, 1e-3_f32) }
      result.pred_x_0[index].to_a.zip(
        flow_cfg_sampler_values(source_step["pred_x_0"]["values"])
      ).each { |actual, expected| actual.should be_close(expected, 1e-3_f32) }
    end
    result.samples.to_a.zip(
      flow_cfg_sampler_values(fixture["expected"]["samples"]["values"])
    ).each { |actual, expected| actual.should be_close(expected, 1e-3_f32) }
    # The source fixture's exact F32 digest remains an oracle-level check; the
    # composed runtime accepts the already-admitted CFG leaf's bounded F32
    # tolerance while still checking every retained step and the final sample.
    result.samples.to_a.each { |value| value.finite?.should be_true }

    noise.to_a.should eq(noise_before)
    positive_condition.to_a.should eq(positive_before)
    negative_condition.to_a.should eq(negative_before)
  end

  it "preflights retained history before invoking the model provider" do
    tensor = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    calls = 0

    expect_raises(ArgumentError, /retained result budget/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceSamplerCPU.sample(
        tensor,
        tensor,
        tensor,
        sigma_min: 0.0_f32,
        steps: 3_i32,
        max_result_bytes: 23_i64
      ) do |_actual_x, _model_t, _actual_condition|
        calls += 1
        tensor
      end
    end
    calls.should eq(0)
  end

  it "forwards the exact normalized Float64 schedule to the step provider" do
    fixture = flow_cfg_sampler_fixture
    config = fixture["fixture"]
    noise = flow_cfg_sampler_tensor(fixture["inputs"]["noise"])
    condition = flow_cfg_sampler_tensor(fixture["inputs"]["positive_condition"])
    observed = [] of {Float64, Float64, Float32, UInt64}

    ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample_with_step_provider(
      noise,
      condition,
      sigma_min: config["sigma_min"].as_f.to_f32,
      steps: config["steps"].as_i.to_i32,
      rescale_t: config["rescale_t"].as_f,
      max_result_bytes: 192_i64
    ) do |actual_x, t, t_prev, model_t, actual_condition|
      actual_condition.object_id.should eq(condition.object_id)
      observed << {t, t_prev, model_t.to_a.first, model_t.object_id}
      flow_cfg_sampler_zero_velocity(actual_x)
    end

    expected = fixture["schedule"]["values"].as_a.map(&.as_f)
    observed.map(&.[0]).should eq(expected[0...-1])
    observed.map(&.[1]).should eq(expected[1..])
    observed.each_with_index do |entry, index|
      expected_model_t = (1000.0_f64 * expected[index]).to_f32
      entry[2].should eq(expected_model_t)
    end
  end
end
