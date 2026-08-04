require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_CFG_RUNTIME_FIXTURE_SHA256 =
  "b1247913533ac1fc87dcd37a0ced8d13a93c9e8e4be0c5353393b101216b4861"

private def flow_cfg_runtime_fixture : JSON::Any
  path = File.join(__DIR__, "../../fixtures/trellis2/flow_euler_cfg_cpu_v1.json")
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_CFG_RUNTIME_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_cfg_runtime_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_cfg_runtime_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_cfg_runtime_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_cfg_runtime_collect_f32(node, values)
  values
end

private def flow_cfg_runtime_i32(node : JSON::Any) : Array(Int32)
  node.as_a.map { |value| value.as_i.to_i32 }
end

private def flow_cfg_runtime_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

describe ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU do
  it "reproduces source branches, identity, and mixed F32 operation order" do
    fixture = flow_cfg_runtime_fixture
    config = fixture["fixture"]
    shape = ML::Shape.new(flow_cfg_runtime_i32(config["shape"]))
    inputs = fixture["inputs"]
    x_t = ML::Tensor.from_array(
      flow_cfg_runtime_f32(inputs["x_t"]["values"]),
      shape
    )
    positive = ML::Tensor.from_array(
      flow_cfg_runtime_f32(inputs["pred_pos"]["values"]),
      shape
    )
    negative = ML::Tensor.from_array(
      flow_cfg_runtime_f32(inputs["pred_neg"]["values"]),
      shape
    )
    model_t_value = config["expected_model_t_f32"].as_f.to_f32
    model_timesteps = ML::Tensor.from_array(
      [model_t_value, model_t_value],
      ML::Shape.new(2_i32)
    )
    positive_condition = ["positive"]
    negative_condition = ["negative"]
    x_before = x_t.to_a
    model_t_before = model_timesteps.to_a
    positive_before = positive_condition.dup
    negative_before = negative_condition.dup

    fixture["cases"].as_a.each do |source_case|
      calls = [] of String
      result = ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        source_case["guidance_strength"].as_f,
        max_result_bytes: 32_i64
      ) do |actual_x, actual_model_t, actual_condition|
        actual_x.object_id.should eq(x_t.object_id)
        actual_model_t.object_id.should eq(model_timesteps.object_id)
        case actual_condition.object_id
        when positive_condition.object_id
          calls << "positive"
          positive
        when negative_condition.object_id
          calls << "negative"
          negative
        else
          raise "unexpected condition identity"
        end
      end

      calls.should eq(source_case["call_order"].as_a.map(&.as_s))
      result.to_a.should eq(
        flow_cfg_runtime_f32(source_case["output"]["values"])
      )
      flow_cfg_runtime_f32le_sha256(result.to_a).should eq(
        source_case["output"]["f32le_sha256"].as_s
      )

      case source_case["name"].as_s
      when "positive_only"
        result.object_id.should eq(positive.object_id)
      when "negative_only"
        result.object_id.should eq(negative.object_id)
      when "mixed"
        result.object_id.should_not eq(positive.object_id)
        result.object_id.should_not eq(negative.object_id)
        result.shares_storage_with?(positive).should be_false
        result.shares_storage_with?(negative).should be_false
      else
        raise "unexpected source case"
      end
    end

    x_t.to_a.should eq(x_before)
    model_timesteps.to_a.should eq(model_t_before)
    positive_condition.should eq(positive_before)
    negative_condition.should eq(negative_before)
  end

  it "composes as the admitted Euler transport provider without owning state math" do
    fixture = flow_cfg_runtime_fixture
    config = fixture["fixture"]
    shape = ML::Shape.new(flow_cfg_runtime_i32(config["shape"]))
    inputs = fixture["inputs"]
    x_t = ML::Tensor.from_array(
      flow_cfg_runtime_f32(inputs["x_t"]["values"]),
      shape
    )
    positive = ML::Tensor.from_array(
      flow_cfg_runtime_f32(inputs["pred_pos"]["values"]),
      shape
    )
    negative = ML::Tensor.from_array(
      flow_cfg_runtime_f32(inputs["pred_neg"]["values"]),
      shape
    )
    mixed_case = fixture["cases"].as_a.find do |source_case|
      source_case["name"].as_s == "mixed"
    end.not_nil!
    expected_velocity = ML::Tensor.from_array(
      flow_cfg_runtime_f32(mixed_case["output"]["values"]),
      shape
    )
    positive_condition = ["positive"]
    negative_condition = ["negative"]
    cfg_calls = [] of String

    composed = ML::ThreeD::Trellis2::FlowEulerStepCPU
      .sample_once_with_velocity_provider(
        x_t,
        positive_condition,
        sigma_min: 0.0_f32,
        t: 0.75_f32,
        t_prev: 0.5_f32
    ) do |actual_x, model_timesteps, actual_positive_condition|
        ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
          actual_x,
          model_timesteps,
          actual_positive_condition,
          negative_condition,
          mixed_case["guidance_strength"].as_f
        ) do |_cfg_x, _cfg_model_t, actual_condition|
          if actual_condition.object_id == positive_condition.object_id
            cfg_calls << "positive"
            positive
          else
            actual_condition.object_id.should eq(negative_condition.object_id)
            cfg_calls << "negative"
            negative
          end
        end
      end
    direct = ML::ThreeD::Trellis2::FlowEulerStepCPU.sample_once(
      x_t,
      expected_velocity,
      sigma_min: 0.0_f32,
      t: 0.75_f32,
      t_prev: 0.5_f32
    )

    cfg_calls.should eq(["positive", "negative"])
    composed.pred_x_prev.to_a.should eq(direct.pred_x_prev.to_a)
    composed.pred_x_0.to_a.should eq(direct.pred_x_0.to_a)
  end

  it "preflights inputs, guidance, and output capacity before provider work" do
    finite = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    non_finite = ML::Tensor.from_array(
      [Float32::NAN],
      ML::Shape.new(1_i32)
    )
    model_t = ML::Tensor.from_array([500.0_f32], ML::Shape.new(1_i32))
    non_finite_model_t = ML::Tensor.from_array(
      [Float32::INFINITY],
      ML::Shape.new(1_i32)
    )
    wrong_model_t = ML::Tensor.from_array(
      [500.0_f32, 500.0_f32],
      ML::Shape.new(2_i32)
    )
    calls = 0

    expect_raises(ArgumentError, /result budget requires 4 bytes/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        finite,
        model_t,
        ["positive"],
        ["negative"],
        1.0_f64,
        max_result_bytes: 3_i64
      ) do |actual_x, _actual_model_t, _actual_condition|
        calls += 1
        actual_x
      end
    end

    {Float64::NAN, Float64::INFINITY, Float64::MAX}.each do |guidance_strength|
      expect_raises(ArgumentError, /guidance_strength/) do
        ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
          finite,
          model_t,
          ["positive"],
          ["negative"],
          guidance_strength
        ) do |actual_x, _actual_model_t, _actual_condition|
          calls += 1
          actual_x
        end
      end
    end

    expect_raises(ArgumentError, /x_t values must be finite/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        non_finite,
        model_t,
        ["positive"],
        ["negative"],
        1.0_f64
      ) do |actual_x, _actual_model_t, _actual_condition|
        calls += 1
        actual_x
      end
    end

    expect_raises(ArgumentError, /model_timesteps.*shape/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        finite,
        wrong_model_t,
        ["positive"],
        ["negative"],
        1.0_f64
      ) do |actual_x, _actual_model_t, _actual_condition|
        calls += 1
        actual_x
      end
    end

    expect_raises(ArgumentError, /model_timesteps values must be finite/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        finite,
        non_finite_model_t,
        ["positive"],
        ["negative"],
        1.0_f64
      ) do |actual_x, _actual_model_t, _actual_condition|
        calls += 1
        actual_x
      end
    end
    calls.should eq(0)
  end

  it "validates each provider result before a later mixed-branch call" do
    x_t = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )
    model_t = ML::Tensor.from_array(
      [500.0_f32, 500.0_f32],
      ML::Shape.new(2_i32)
    )
    wrong_shape = ML::Tensor.from_array(x_t.to_a, ML::Shape.new(4_i32))
    calls = 0

    expect_raises(ArgumentError, /same shape/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_t,
        ["positive"],
        ["negative"],
        1.7_f64
      ) do |_actual_x, _actual_model_t, _actual_condition|
        calls += 1
        wrong_shape
      end
    end
    calls.should eq(1)

    calls = 0
    expect_raises(ArgumentError, /prediction values must be finite/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_t,
        ["positive"],
        ["negative"],
        1.7_f64
      ) do |_actual_x, _actual_model_t, _actual_condition|
        calls += 1
        if calls == 1
          x_t
        else
          ML::Tensor.from_array(
            [1.0_f32, Float32::NAN, 3.0_f32, 4.0_f32],
            x_t.shape
          )
        end
      end
    end
    calls.should eq(2)

    calls = 0
    expect_raises(ArgumentError, /prediction must be contiguous/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_t,
        ["positive"],
        ["negative"],
        1.0_f64
      ) do |_actual_x, _actual_model_t, _actual_condition|
        calls += 1
        x_t.transpose
      end
    end
    calls.should eq(1)
  end

  it "rejects non-finite mixed arithmetic" do
    x_t = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    model_t = ML::Tensor.from_array([500.0_f32], ML::Shape.new(1_i32))
    largest = ML::Tensor.from_array([Float32::MAX], ML::Shape.new(1_i32))
    calls = 0

    expect_raises(ArgumentError, /CFG arithmetic must produce finite outputs/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_t,
        ["positive"],
        ["negative"],
        2.0_f64
      ) do |_actual_x, _actual_model_t, _actual_condition|
        calls += 1
        largest
      end
    end
    calls.should eq(2)
  end

  it "propagates provider exceptions without retry" do
    x_t = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    model_t = ML::Tensor.from_array([500.0_f32], ML::Shape.new(1_i32))
    calls = 0

    expect_raises(Exception, /negative provider failed/) do
      ML::ThreeD::Trellis2::FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_t,
        ["positive"],
        ["negative"],
        1.7_f64
      ) do |actual_x, _actual_model_t, _actual_condition|
        calls += 1
        raise "negative provider failed" if calls == 2
        actual_x
      end
    end
    calls.should eq(2)
  end
end
