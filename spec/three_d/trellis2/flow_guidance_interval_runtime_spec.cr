require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_GUIDANCE_INTERVAL_RUNTIME_FIXTURE_SHA256 =
  "290f816befd68f498be8df4008230c6433afbb615c70f1d40f11cfba0d7d109a"

private def flow_guidance_interval_runtime_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_guidance_interval_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_GUIDANCE_INTERVAL_RUNTIME_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_guidance_interval_runtime_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each do |value|
      flow_guidance_interval_runtime_collect_f32(value, values)
    end
  else
    values << node.as_f.to_f32
  end
end

private def flow_guidance_interval_runtime_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_guidance_interval_runtime_collect_f32(node, values)
  values
end

private def flow_guidance_interval_runtime_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_guidance_interval_runtime_model_timesteps(
  normalized_t : Float64,
  batch : Int32,
) : ML::Tensor
  model_t = (1000.0_f64 * normalized_t).to_f32
  ML::Tensor.from_array(
    Array.new(batch, model_t),
    ML::Shape.new(batch)
  )
end

describe ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU do
  it "routes all source cases through caller-owned model time and CFG" do
    fixture = flow_guidance_interval_runtime_fixture
    config = fixture["fixture"]
    shape_values = config["shape"].as_a.map(&.as_i.to_i32)
    shape = ML::Shape.new(shape_values)
    batch = shape_values.first
    interval_values = config["guidance_interval"].as_a.map(&.as_f)
    guidance_interval = {interval_values[0], interval_values[1]}
    requested_strength = config["requested_guidance_strength"].as_f
    inputs = fixture["inputs"]
    x_t = ML::Tensor.from_array(
      flow_guidance_interval_runtime_f32(inputs["x_t"]["values"]),
      shape
    )
    positive = ML::Tensor.from_array(
      flow_guidance_interval_runtime_f32(inputs["pred_pos"]["values"]),
      shape
    )
    negative = ML::Tensor.from_array(
      flow_guidance_interval_runtime_f32(inputs["pred_neg"]["values"]),
      shape
    )
    positive_condition = ["positive"]
    negative_condition = ["negative"]
    x_before = x_t.to_a
    positive_before = positive_condition.dup
    negative_before = negative_condition.dup
    model_times_by_name = {} of String => Array(Float32)
    routes_by_name = {} of String => Array(String)

    fixture["cases"].as_a.each do |source_case|
      normalized_t = source_case["normalized_t"].as_f
      model_timesteps = flow_guidance_interval_runtime_model_timesteps(
        normalized_t,
        batch
      )
      model_t_before = model_timesteps.to_a
      calls = [] of String
      result = ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        normalized_t,
        requested_strength,
        guidance_interval,
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

      expected_output = flow_guidance_interval_runtime_f32(
        source_case["output"]["values"]
      )
      result.to_a.should eq(expected_output)
      flow_guidance_interval_runtime_f32le_sha256(result.to_a).should eq(
        source_case["output"]["f32le_sha256"].as_s
      )
      calls.should eq(source_case["call_order"].as_a.map(&.as_s))

      case source_case["output_identity"].as_s
      when "positive_prediction"
        result.object_id.should eq(positive.object_id)
      when "owned_mixed"
        result.object_id.should_not eq(positive.object_id)
        result.object_id.should_not eq(negative.object_id)
        result.shares_storage_with?(positive).should be_false
        result.shares_storage_with?(negative).should be_false
      else
        raise "unexpected output identity"
      end

      source_case["calls"].as_a.each_with_index do |source_call, index|
        source_model_t = flow_guidance_interval_runtime_f32(
          source_call["model_t"]["values"]
        )
        actual_model_t = model_timesteps.to_a
        actual_model_t.should eq(source_model_t)
        source_call["condition"].as_s.should eq(calls[index])
      end
      model_timesteps.to_a.should eq(model_t_before)
      model_times_by_name[source_case["name"].as_s] = model_timesteps.to_a
      routes_by_name[source_case["name"].as_s] = calls
    end

    model_times_by_name["below_lower"].should eq(
      model_times_by_name["lower_boundary"]
    )
    routes_by_name["below_lower"].should_not eq(
      routes_by_name["lower_boundary"]
    )
    model_times_by_name["upper_boundary"].should eq(
      model_times_by_name["above_upper"]
    )
    routes_by_name["upper_boundary"].should_not eq(
      routes_by_name["above_upper"]
    )
    x_t.to_a.should eq(x_before)
    positive_condition.should eq(positive_before)
    negative_condition.should eq(negative_before)
  end

  it "keeps adjacent Float64 boundaries distinct after the same F32 model time" do
    x_t = ML::Tensor.from_array([1.0_f32, 2.0_f32], ML::Shape.new(1_i32, 2_i32))
    model_timesteps = ML::Tensor.from_array(
      [250.0_f32],
      ML::Shape.new(1_i32)
    )
    positive = ML::Tensor.from_array(
      [3.0_f32, 4.0_f32],
      x_t.shape
    )
    negative = ML::Tensor.from_array(
      [7.0_f32, 8.0_f32],
      x_t.shape
    )
    positive_condition = [:positive]
    negative_condition = [:negative]
    calls_below = [] of Symbol
    calls_lower = [] of Symbol

    below = ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
      x_t,
      model_timesteps,
      positive_condition,
      negative_condition,
      0.24999999999999997_f64,
      1.7_f64,
      {0.25_f64, 0.75_f64}
    ) do |_actual_x, actual_model_t, condition|
      actual_model_t.object_id.should eq(model_timesteps.object_id)
      calls_below << condition.first
      positive
    end
    lower = ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
      x_t,
      model_timesteps,
      positive_condition,
      negative_condition,
      0.25_f64,
      1.7_f64,
      {0.25_f64, 0.75_f64}
    ) do |_actual_x, actual_model_t, condition|
      actual_model_t.object_id.should eq(model_timesteps.object_id)
      calls_lower << condition.first
      condition.first == :positive ? positive : negative
    end

    model_timesteps.to_a.should eq([250.0_f32])
    calls_below.should eq([:positive])
    calls_lower.should eq([:positive, :negative])
    below.object_id.should eq(positive.object_id)
    lower.object_id.should_not eq(positive.object_id)
  end

  it "preflights scalar and result-cap admission before provider work" do
    x_t = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    model_timesteps = ML::Tensor.from_array([500.0_f32], ML::Shape.new(1_i32))
    positive_condition = ["positive"]
    negative_condition = ["negative"]
    calls = 0
    invoke = ->(normalized_t : Float64, strength : Float64, interval : Tuple(Float64, Float64), max_bytes : Int64) do
      ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        normalized_t,
        strength,
        interval,
        max_result_bytes: max_bytes
      ) do |_actual_x, _actual_model_t, _actual_condition|
        calls += 1
        x_t
      end
    end

    {
      {Float64::NAN, 1.7_f64, {0.25_f64, 0.75_f64}},
      {Float64::INFINITY, 1.7_f64, {0.25_f64, 0.75_f64}},
      {-0.0001_f64, 1.7_f64, {0.25_f64, 0.75_f64}},
      {1.0001_f64, 1.7_f64, {0.25_f64, 0.75_f64}},
    }.each do |normalized_t, strength, interval|
      expect_raises(ArgumentError, /normalized_t/) do
        invoke.call(normalized_t, strength, interval, 4_i64)
      end
    end

    [
      {Float64::NAN, 0.5_f64},
      {-0.1_f64, 0.5_f64},
      {0.5_f64, 1.1_f64},
      {0.75_f64, 0.25_f64},
    ].each do |interval|
      expect_raises(ArgumentError, /guidance_interval/) do
        invoke.call(0.5_f64, 0.5_f64, interval, 4_i64)
      end
    end

    [
      {0.5_f64, Float64::NAN},
      {0.5_f64, Float64::INFINITY},
      {0.5_f64, Float64::MAX},
      {0.1_f64, Float32::MAX.to_f64 + 1.0e32},
      {0.5_f64, -Float32::MAX.to_f64 - 1.0e32},
    ].each do |normalized_t, strength|
      expect_raises(ArgumentError, /guidance_strength/) do
        invoke.call(normalized_t, strength, {0.25_f64, 0.75_f64}, 4_i64)
      end
    end

    expect_raises(ArgumentError, /result budget requires 4 bytes/) do
      invoke.call(0.5_f64, 1.0_f64, {0.25_f64, 0.75_f64}, 3_i64)
    end
    invoke.call(0.5_f64, 1.0_f64, {0.25_f64, 0.75_f64}, 4_i64).object_id
      .should eq(x_t.object_id)
    calls.should eq(1)
  end

  it "keeps degenerate intervals inclusive and preserves exact strength-zero branches" do
    x_t = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    model_timesteps = ML::Tensor.from_array([500.0_f32], ML::Shape.new(1_i32))
    positive = ML::Tensor.from_array([3.0_f32], x_t.shape)
    negative = ML::Tensor.from_array([7.0_f32], x_t.shape)
    positive_condition = ["positive"]
    negative_condition = ["negative"]

    inside_calls = [] of String
    inside = ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
      x_t,
      model_timesteps,
      positive_condition,
      negative_condition,
      0.5_f64,
      0.0_f64,
      {0.5_f64, 0.5_f64}
    ) do |_actual_x, _actual_model_t, condition|
      inside_calls << condition.first
      condition.first == "positive" ? positive : negative
    end

    outside_calls = [] of String
    outside = ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
      x_t,
      model_timesteps,
      positive_condition,
      negative_condition,
      0.49999999999999994_f64,
      0.0_f64,
      {0.5_f64, 0.5_f64}
    ) do |_actual_x, _actual_model_t, condition|
      outside_calls << condition.first
      condition.first == "positive" ? positive : negative
    end

    inside_calls.should eq(["negative"])
    outside_calls.should eq(["positive"])
    inside.object_id.should eq(negative.object_id)
    outside.object_id.should eq(positive.object_id)
  end

  it "propagates provider exceptions without retry or input mutation" do
    x_t = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32))
    model_timesteps = ML::Tensor.from_array([500.0_f32], ML::Shape.new(1_i32))
    positive_condition = ["positive"]
    negative_condition = ["negative"]
    x_before = x_t.to_a
    model_before = model_timesteps.to_a
    positive_before = positive_condition.dup
    negative_before = negative_condition.dup
    calls = 0

    expect_raises(Exception, /provider boom/) do
      ML::ThreeD::Trellis2::FlowGuidanceIntervalCPU.predict_velocity(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        0.5_f64,
        1.7_f64,
        {0.25_f64, 0.75_f64}
      ) do |_actual_x, _actual_model_t, _actual_condition|
        calls += 1
        raise "provider boom"
      end
    end

    calls.should eq(1)
    x_t.to_a.should eq(x_before)
    model_timesteps.to_a.should eq(model_before)
    positive_condition.should eq(positive_before)
    negative_condition.should eq(negative_before)
  end
end
