require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_EULER_RUNTIME_FIXTURE_SHA256 =
  "84de71d38219b203cc122ac8e4c873db59fc90939cfbf37cfbc46a2cffa1d4c3"

private def flow_runtime_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_euler_schedule_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_EULER_RUNTIME_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_runtime_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_runtime_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_runtime_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_runtime_collect_f32(node, values)
  values
end

private def flow_runtime_i32(node : JSON::Any) : Array(Int32)
  node.as_a.map { |value| value.as_i.to_i32 }
end

private def flow_runtime_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_runtime_zero_velocity(x : ML::Tensor) : ML::Tensor
  ML::Tensor.from_array(
    Array(Float32).new(x.numel.to_i, 0.0_f32),
    x.shape
  )
end

describe ML::ThreeD::Trellis2::FlowEulerSamplerCPU do
  it "reproduces every source-pinned step with Float64 scheduling" do
    fixture = flow_runtime_fixture
    config = fixture["fixture"]
    shape = ML::Shape.new(flow_runtime_i32(config["shape"]))
    noise = ML::Tensor.from_array(
      flow_runtime_f32(fixture["inputs"]["noise"]["values"]),
      shape
    )
    cond = ML::Tensor.from_array(
      flow_runtime_f32(fixture["inputs"]["cond"]["values"]),
      shape
    )
    noise_before = noise.to_a
    cond_before = cond.to_a
    model_x_scale = config["model_x_scale"].as_f.to_f32
    model_t_scale = config["model_t_scale"].as_f.to_f32
    state_ids = [] of UInt64
    condition_ids = [] of UInt64
    model_times = [] of Array(Float32)

    result = ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
      noise,
      cond,
      sigma_min: config["sigma_min"].as_f.to_f32,
      steps: config["steps"].as_i.to_i32,
      rescale_t: config["rescale_t"].as_f
    ) do |actual_x, model_t, actual_cond|
      state_ids << actual_x.object_id
      condition_ids << actual_cond.object_id
      model_times << model_t.to_a

      x_values = actual_x.to_a
      cond_values = actual_cond.to_a
      time_term = model_t.to_a.first * model_t_scale
      velocity = Array(Float32).new(x_values.size)
      x_values.each_with_index do |x, index|
        velocity << (x * model_x_scale + time_term) + cond_values[index]
      end
      ML::Tensor.from_array(velocity, actual_x.shape)
    end

    source_steps = fixture["steps"].as_a
    result.pred_x_t.size.should eq(source_steps.size)
    result.pred_x_0.size.should eq(source_steps.size)
    state_ids.first.should eq(noise.object_id)
    condition_ids.should eq(Array.new(source_steps.size, cond.object_id))

    source_steps.each_with_index do |source_step, index|
      model_times[index].should eq(
        flow_runtime_f32(source_step["model_t"]["values"])
      )
      result.pred_x_t[index].to_a.should eq(
        flow_runtime_f32(source_step["pred_x_prev"]["values"])
      )
      result.pred_x_0[index].to_a.should eq(
        flow_runtime_f32(source_step["pred_x_0"]["values"])
      )
      flow_runtime_f32le_sha256(result.pred_x_t[index].to_a).should eq(
        source_step["pred_x_prev"]["f32le_sha256"].as_s
      )
      flow_runtime_f32le_sha256(result.pred_x_0[index].to_a).should eq(
        source_step["pred_x_0"]["f32le_sha256"].as_s
      )
      if index > 0
        state_ids[index].should eq(result.pred_x_t[index - 1].object_id)
      end
    end

    result.samples.object_id.should eq(result.pred_x_t.last.object_id)
    result.samples.to_a.should eq(
      flow_runtime_f32(fixture["expected"]["samples"]["values"])
    )
    flow_runtime_f32le_sha256(result.samples.to_a).should eq(
      fixture["expected"]["samples"]["f32le_sha256"].as_s
    )
    noise.to_a.should eq(noise_before)
    cond.to_a.should eq(cond_before)
    result.pred_x_t.each do |state|
      state.shares_storage_with?(noise).should be_false
    end
  end

  it "preflights schedule policy and retained payload before provider work" do
    finite = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    non_finite = ML::Tensor.from_array([Float32::NAN], ML::Shape.new(1_i32))
    cond = ["opaque"]
    calls = 0

    {0_i32, -1_i32, ML::ThreeD::Trellis2::FlowEulerSamplerCPU::MAX_STEPS + 1}.each do |steps|
      expect_raises(ArgumentError, /steps/) do
        ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
          finite,
          cond,
          sigma_min: 0.0_f32,
          steps: steps,
          rescale_t: 1.0_f64
        ) do |actual_x, _model_t, _actual_cond|
          calls += 1
          flow_runtime_zero_velocity(actual_x)
        end
      end
    end

    {0.0_f64, -1.0_f64, Float64::NAN, Float64::INFINITY}.each do |rescale_t|
      expect_raises(ArgumentError, /rescale_t/) do
        ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
          finite,
          cond,
          sigma_min: 0.0_f32,
          steps: 1,
          rescale_t: rescale_t
        ) do |actual_x, _model_t, _actual_cond|
          calls += 1
          flow_runtime_zero_velocity(actual_x)
        end
      end
    end

    expect_raises(ArgumentError, /retained result budget/) do
      ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
        non_finite,
        cond,
        sigma_min: 0.0_f32,
        steps: 3,
        rescale_t: 1.0_f64,
        max_result_bytes: 23_i64
      ) do |actual_x, _model_t, _actual_cond|
        calls += 1
        flow_runtime_zero_velocity(actual_x)
      end
    end

    expect_raises(ArgumentError, /noise values must be finite/) do
      ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
        non_finite,
        cond,
        sigma_min: 0.0_f32,
        steps: 1,
        rescale_t: 1.0_f64
      ) do |actual_x, _model_t, _actual_cond|
        calls += 1
        flow_runtime_zero_velocity(actual_x)
      end
    end

    expect_raises(ArgumentError, /sigma_min/) do
      ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
        finite,
        cond,
        sigma_min: Float32::NAN,
        steps: 1,
        rescale_t: 1.0_f64
      ) do |actual_x, _model_t, _actual_cond|
        calls += 1
        flow_runtime_zero_velocity(actual_x)
      end
    end
    calls.should eq(0)
  end

  it "admits the exact retained F32 payload boundary" do
    noise = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    calls = 0

    result = ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
      noise,
      nil,
      sigma_min: 0.0_f32,
      steps: 3,
      rescale_t: 1.0_f64,
      max_result_bytes: 24_i64
    ) do |actual_x, _model_t, _actual_cond|
      calls += 1
      flow_runtime_zero_velocity(actual_x)
    end

    calls.should eq(3)
    result.pred_x_t.size.should eq(3)
    result.pred_x_0.size.should eq(3)
    result.samples.to_a.should eq([1.0_f32])
  end

  it "stops on provider failure without retrying or entering a later step" do
    noise = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    calls = 0

    expect_raises(Exception, /second schedule call failed/) do
      ML::ThreeD::Trellis2::FlowEulerSamplerCPU.sample(
        noise,
        ["opaque"],
        sigma_min: 0.0_f32,
        steps: 3,
        rescale_t: 1.0_f64
      ) do |actual_x, _model_t, _actual_cond|
        calls += 1
        raise "second schedule call failed" if calls == 2
        flow_runtime_zero_velocity(actual_x)
      end
    end
    calls.should eq(2)
  end
end
