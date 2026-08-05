require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_CFG_RESCALE_SCHEDULE_FIXTURE_SHA256 =
  "74ed078d4b7327ad2273c1814c7ec910efd284881116ec0d452666ffd4952218"

private record FlowCfgScheduleReferenceStep,
  x_t : Array(Float32),
  model_t : Float32,
  pred_positive : Array(Float32),
  pred_negative : Array(Float32),
  pred_cfg : Array(Float32),
  x0_positive : Array(Float32),
  x0_cfg : Array(Float32),
  std_positive : Array(Float32),
  std_cfg : Array(Float32),
  x0_rescaled : Array(Float32),
  x0_blend : Array(Float32),
  pred_v : Array(Float32),
  pred_x_prev : Array(Float32),
  pred_x_0 : Array(Float32)

private def flow_cfg_schedule_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_cfg_rescale_schedule_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_CFG_RESCALE_SCHEDULE_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_cfg_schedule_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_cfg_schedule_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_cfg_schedule_values(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_cfg_schedule_collect_f32(node["values"], values)
  values
end

private def flow_cfg_schedule_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_cfg_schedule_f64le_sha256(values : Indexable(Float64)) : String
  bytes = Bytes.new(values.size * 8, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 8, 8])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_cfg_schedule_f32_bits(value : Float32) : UInt32
  bytes = Bytes.new(4, 0_u8)
  IO::ByteFormat::LittleEndian.encode(value, bytes)
  IO::ByteFormat::LittleEndian.decode(UInt32, bytes)
end

private def flow_cfg_schedule_f64_bits(value : Float64) : UInt64
  bytes = Bytes.new(8, 0_u8)
  IO::ByteFormat::LittleEndian.encode(value, bytes)
  IO::ByteFormat::LittleEndian.decode(UInt64, bytes)
end

private def flow_cfg_schedule_hex_u64(node : JSON::Any) : UInt64
  encoded = node.as_s
  encoded.starts_with?("0x").should be_true
  encoded[2..].to_u64(16)
end

private def flow_cfg_schedule_assert_tensor(
  node : JSON::Any,
  expected_shape : Array(Int64)? = nil,
) : Array(Float32)
  node["dtype"].as_s.should eq("float32")
  shape = node["shape"].as_a.map(&.as_i)
  shape.should_not be_empty
  shape.should eq(expected_shape) if expected_shape
  values = flow_cfg_schedule_values(node)
  shape.product.should eq(values.size.to_i64)
  node["f32le_sha256"].as_s.should eq(
    flow_cfg_schedule_f32le_sha256(values)
  )
  values
end

private def flow_cfg_schedule_should_be_close(
  actual : Indexable(Float32),
  expected : Indexable(Float32),
  tolerance : Float32 = 2e-6_f32,
) : Nil
  actual.size.should eq(expected.size)
  actual.zip(expected).each do |value, wanted|
    value.should be_close(wanted, tolerance)
  end
end

private def flow_cfg_schedule_source_mix(
  positive : Indexable(Float32),
  negative : Indexable(Float32),
  strength : Float64,
) : Array(Float32)
  positive.size.should eq(negative.size)
  positive_scale = strength.to_f32
  negative_scale = (1.0_f64 - strength).to_f32
  positive.map_with_index do |value, index|
    positive_term = positive_scale * value
    negative_term = negative_scale * negative[index]
    positive_term + negative_term
  end
end

private def flow_cfg_schedule_model(
  state : Indexable(Float32),
  model_t : Float32,
  condition : Indexable(Float32),
  x_scale : Float32,
  t_scale : Float32,
) : Array(Float32)
  state.size.should eq(condition.size)
  model_term = model_t * t_scale
  state.map_with_index do |value, index|
    x_term = value * x_scale
    (x_term + model_term) + condition[index]
  end
end

private def flow_cfg_schedule_pred_to_x0(
  state : Indexable(Float32),
  normalized_t : Float64,
  prediction : Indexable(Float32),
  sigma_min : Float32,
) : Array(Float32)
  state.size.should eq(prediction.size)
  one_minus_sigma = (1.0_f64 - sigma_min.to_f64).to_f32
  noise_scale = (
    sigma_min.to_f64 + (1.0_f64 - sigma_min.to_f64) * normalized_t
  ).to_f32
  state.map_with_index do |value, index|
    (one_minus_sigma * value) - (noise_scale * prediction[index])
  end
end

private def flow_cfg_schedule_x0_to_pred(
  state : Indexable(Float32),
  normalized_t : Float64,
  x0 : Indexable(Float32),
  sigma_min : Float32,
) : Array(Float32)
  state.size.should eq(x0.size)
  one_minus_sigma = (1.0_f64 - sigma_min.to_f64).to_f32
  noise_scale = (
    sigma_min.to_f64 + (1.0_f64 - sigma_min.to_f64) * normalized_t
  ).to_f32
  state.map_with_index do |value, index|
    ((one_minus_sigma * value) - x0[index]) / noise_scale
  end
end

private def flow_cfg_schedule_sample_std(
  values : Indexable(Float32),
  batch_size : Int32,
) : Array(Float32)
  values.size.should eq(batch_size * 4)
  Array.new(batch_size) do |batch|
    offset = batch * 4
    total = 0.0_f32
    4.times { |index| total += values[offset + index] }
    mean = total / 4.0_f32
    squared = 0.0_f32
    4.times do |index|
      delta = values[offset + index] - mean
      squared += delta * delta
    end
    Math.sqrt(squared / 3.0_f32).to_f32
  end
end

private def flow_cfg_schedule_reference(
  noise : Indexable(Float32),
  positive_condition : Indexable(Float32),
  negative_condition : Indexable(Float32),
  schedule : Indexable(Float64),
  sigma_min : Float32,
  model_x_scale : Float32,
  model_t_scale : Float32,
  guidance_strength : Float64,
  guidance_rescale : Float64,
  rescale_space : String = "x0",
) : Tuple(Array(FlowCfgScheduleReferenceStep), Array(Float32))
  noise.size.should eq(positive_condition.size)
  noise.size.should eq(negative_condition.size)
  state = noise.to_a
  steps = [] of FlowCfgScheduleReferenceStep
  schedule.each_cons_pair do |t, t_prev|
    model_t = (1000.0_f64 * t).to_f32
    pred_positive = flow_cfg_schedule_model(
      state, model_t, positive_condition, model_x_scale, model_t_scale
    )
    pred_negative = flow_cfg_schedule_model(
      state, model_t, negative_condition, model_x_scale, model_t_scale
    )
    pred_cfg = flow_cfg_schedule_source_mix(
      pred_positive, pred_negative, guidance_strength
    )
    x0_positive = flow_cfg_schedule_pred_to_x0(
      state, t, pred_positive, sigma_min
    )
    x0_cfg = flow_cfg_schedule_pred_to_x0(state, t, pred_cfg, sigma_min)

    if rescale_space == "x0"
      std_positive = flow_cfg_schedule_sample_std(x0_positive, 2)
      std_cfg = flow_cfg_schedule_sample_std(x0_cfg, 2)
      x0_rescaled = x0_cfg.map_with_index do |value, index|
        value * (std_positive[index // 4] / std_cfg[index // 4])
      end
      x0_blend = x0_rescaled.map_with_index do |value, index|
        guidance_rescale.to_f32 * value +
          (1.0_f64 - guidance_rescale).to_f32 * x0_cfg[index]
      end
      pred_v = flow_cfg_schedule_x0_to_pred(
        state, t, x0_blend, sigma_min
      )
    elsif rescale_space == "prediction"
      std_positive = flow_cfg_schedule_sample_std(pred_positive, 2)
      std_cfg = flow_cfg_schedule_sample_std(pred_cfg, 2)
      x0_rescaled = pred_cfg.map_with_index do |value, index|
        value * (std_positive[index // 4] / std_cfg[index // 4])
      end
      pred_v = x0_rescaled.map_with_index do |value, index|
        guidance_rescale.to_f32 * value +
          (1.0_f64 - guidance_rescale).to_f32 * pred_cfg[index]
      end
      x0_blend = flow_cfg_schedule_pred_to_x0(
        state, t, pred_v, sigma_min
      )
    else
      raise "unknown rescale space: #{rescale_space}"
    end

    dt = (t - t_prev).to_f32
    pred_x_prev = state.map_with_index do |value, index|
      value - dt * pred_v[index]
    end
    pred_x_0 = flow_cfg_schedule_pred_to_x0(
      state, t, pred_v, sigma_min
    )
    steps << FlowCfgScheduleReferenceStep.new(
      state,
      model_t,
      pred_positive,
      pred_negative,
      pred_cfg,
      x0_positive,
      x0_cfg,
      std_positive,
      std_cfg,
      x0_rescaled,
      x0_blend,
      pred_v,
      pred_x_prev,
      pred_x_0
    )
    state = pred_x_prev
  end
  {steps, state}
end

describe "TRELLIS.2 source-pinned repeated CFG-rescale sampler" do
  it "seals schedule order, CFG/rescale transport, and independent history" do
    fixture = flow_cfg_schedule_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/flow-cfg-rescale-schedule-oracle/v1"
    )

    provenance = fixture["provenance"]
    provenance["repository"].as_s.should eq(
      "https://github.com/microsoft/TRELLIS.2"
    )
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["device"].as_s.should eq("cpu")
    provenance["threads"].as_i.should eq(1_i64)
    provenance["dtype"].as_s.should contain("Float64 schedule")
    provenance["weights"].as_s.should eq("none")
    provenance["network"].as_s.should eq("none")
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_cfg_rescale_schedule.py"
    )
    provenance["generator_sha256"].as_s.should eq(
      "5a23b8d2a5d3d0aec364aa06ee4526f5f7761931767c9717e5355c3d7e1771da"
    )
    provenance["support"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_euler_cfg.py"
    )
    provenance["support_sha256"].as_s.should eq(
      "a51ff10617022931b234cbf455e26f696124b187c30f74dae4ba2f1446eb2837"
    )
    repo_root = File.expand_path("../../..", __DIR__)
    {"generator" => "generator_sha256", "support" => "support_sha256"}.each do |path_key, digest_key|
      source = File.read(File.join(repo_root, provenance[path_key].as_s))
      Digest::SHA256.hexdigest(source.to_slice).should eq(
        provenance[digest_key].as_s
      )
    end
    {
      "base"       => "be8530b55ea66ac58e8ab23d650f463636dd52cf7e39c7c9c66f69bf72e6a0d1",
      "cfg"        => "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee",
      "interval"   => "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2",
      "flow_euler" => "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd",
    }.each do |name, digest|
      provenance["sources"][name]["sha256"].as_s.should eq(digest)
    end

    contract = fixture["contract"]
    contract["owner"].as_s.should eq("FlowEulerSampler.sample")
    contract["entrypoint"].as_s.should end_with("FlowEulerCfgSampler.sample")
    contract["mro"].as_a.map(&.as_s).first(4).should eq(
      [
        "FlowEulerCfgSampler",
        "ClassifierFreeGuidanceSamplerMixin",
        "FlowEulerSampler",
        "Sampler",
      ]
    )
    contract["schedule"].as_s.should contain("Float64")
    contract["model_timestep"].as_s.should contain("1000*t")
    contract["cfg"].as_s.should contain("positive then negative")
    contract["rescale"].as_s.should contain("x0")
    contract["rescale"].as_s.should contain("non-batch")
    contract["euler"].as_s.should contain("normalized Float64 t")
    contract["identity"].as_s.should contain("identity")
    contract["counterfactuals"].as_s.should contain("both")
    contract["production_defaults"].as_s.should contain("not claimed")

    config = fixture["fixture"]
    config["shape"].as_a.map(&.as_i).should eq([2_i64, 2_i64, 2_i64])
    steps = config["steps"].as_i.to_i32
    rescale_t = config["rescale_t"].as_f
    sigma_min = config["sigma_min"].as_f.to_f32
    guidance_strength = config["guidance_strength"].as_f
    guidance_rescale = config["guidance_rescale"].as_f
    model_x_scale = config["model_x_scale"].as_f.to_f32
    model_t_scale = config["model_t_scale"].as_f.to_f32
    steps.should eq(3)
    rescale_t.should eq(1.7_f64)
    sigma_min.should eq(1e-5_f32)
    guidance_strength.should eq(1.7_f64)
    guidance_rescale.should eq(0.35_f64)
    model_x_scale.should eq(0.25_f32)
    model_t_scale.should eq(1.0_f32 / 1024.0_f32)

    schedule = (0..steps).map do |index|
      unscaled = 1.0_f64 - index.to_f64 / steps.to_f64
      rescale_t * unscaled /
        (1.0_f64 + (rescale_t - 1.0_f64) * unscaled)
    end
    schedule_payload = fixture["schedule"]
    schedule_payload["dtype"].as_s.should eq("float64")
    schedule_payload["values"].as_a.map(&.as_f).should eq(schedule)
    schedule_payload["f64_bits_hex"].as_a.each_with_index do |item, index|
      flow_cfg_schedule_hex_u64(item).should eq(
        flow_cfg_schedule_f64_bits(schedule[index])
      )
    end
    flow_cfg_schedule_f64le_sha256(schedule).should eq(
      "ed9fdc900020d54363b006746ad6dcc27658d1113fde648e269090bb49051622"
    )
    schedule_payload["f64le_sha256"].as_s.should eq(
      flow_cfg_schedule_f64le_sha256(schedule)
    )

    inputs = fixture["inputs"]
    inputs["all_unchanged"].as_bool.should be_true
    noise = flow_cfg_schedule_assert_tensor(
      inputs["noise"], [2_i64, 2_i64, 2_i64]
    )
    positive_condition = flow_cfg_schedule_assert_tensor(
      inputs["positive_condition"], [2_i64, 2_i64, 2_i64]
    )
    negative_condition = flow_cfg_schedule_assert_tensor(
      inputs["negative_condition"], [2_i64, 2_i64, 2_i64]
    )
    noise.should eq([1.0_f32, -2.0_f32, 0.5_f32, 3.0_f32, -0.75_f32, 1.25_f32, 2.5_f32, -3.5_f32])

    reference_steps, reference_final = flow_cfg_schedule_reference(
      noise,
      positive_condition,
      negative_condition,
      schedule,
      sigma_min,
      model_x_scale,
      model_t_scale,
      guidance_strength,
      guidance_rescale
    )
    source_steps = fixture["steps"].as_a
    source_steps.size.should eq(steps)
    source_steps.each_with_index do |source_step, index|
      reference = reference_steps[index]
      source_step["index"].as_i.should eq(index)
      source_step["x_identity"].as_s.should eq(
        index == 0 ? "initial_noise" : "prior_pred_x_prev"
      )
      source_step["cond_identity_preserved"].as_bool.should be_true
      source_step["conversion_trace"].as_a.map(&.as_s).should eq(
        [
          "pred_to_xstart:positive",
          "pred_to_xstart:cfg",
          "xstart_to_pred:blend",
        ]
      )
      source_step["conversion_times"].as_a.map(&.as_f).should eq(
        [schedule[index], schedule[index], schedule[index]]
      )
      source_step["t"].as_f.should eq(schedule[index])
      source_step["t_prev"].as_f.should eq(schedule[index + 1])
      flow_cfg_schedule_hex_u64(source_step["t_f64_bits_hex"]).should eq(
        flow_cfg_schedule_f64_bits(schedule[index])
      )
      flow_cfg_schedule_hex_u64(source_step["t_prev_f64_bits_hex"]).should eq(
        flow_cfg_schedule_f64_bits(schedule[index + 1])
      )
      dt = (schedule[index] - schedule[index + 1]).to_f32
      source_step["dt_f32_after_f64_subtract"].as_f.should eq(dt.to_f64)
      source_step["dt_f32_bits"].as_i.to_u32.should eq(
        flow_cfg_schedule_f32_bits(dt)
      )

      calls = source_step["calls"].as_a
      calls.size.should eq(2)
      calls.map { |call| call["condition"].as_s }.should eq(
        ["positive", "negative"]
      )
      calls.each_with_index do |call, call_index|
        call["index"].as_i.should eq(call_index)
        call["x_t_identity_preserved"].as_bool.should be_true
        call["condition_identity_preserved"].as_bool.should be_true
        model_t = flow_cfg_schedule_assert_tensor(
          call["model_t"], [2_i64]
        )
        model_t.should eq([reference.model_t, reference.model_t])
        condition = call_index == 0 ? positive_condition : negative_condition
        expected_prediction = call_index == 0 ? reference.pred_positive : reference.pred_negative
        actual_prediction = flow_cfg_schedule_assert_tensor(
          call["prediction"], [2_i64, 2_i64, 2_i64]
        )
        flow_cfg_schedule_should_be_close(actual_prediction, expected_prediction)
        condition.should_not be_empty
      end

      actual_x = flow_cfg_schedule_assert_tensor(
        source_step["x_t"], [2_i64, 2_i64, 2_i64]
      )
      flow_cfg_schedule_should_be_close(actual_x, reference.x_t)
      actual_velocity = flow_cfg_schedule_assert_tensor(
        source_step["pred_v"], [2_i64, 2_i64, 2_i64]
      )
      flow_cfg_schedule_should_be_close(actual_velocity, reference.pred_v)
      actual_prev = flow_cfg_schedule_assert_tensor(
        source_step["pred_x_prev"], [2_i64, 2_i64, 2_i64]
      )
      actual_x0 = flow_cfg_schedule_assert_tensor(
        source_step["pred_x_0"], [2_i64, 2_i64, 2_i64]
      )
      flow_cfg_schedule_should_be_close(actual_prev, reference.pred_x_prev)
      flow_cfg_schedule_should_be_close(actual_x0, reference.pred_x_0)

      intermediates = source_step["intermediates"]
      {
        "pred_positive" => reference.pred_positive,
        "pred_negative" => reference.pred_negative,
        "pred_cfg"      => reference.pred_cfg,
        "x0_positive"   => reference.x0_positive,
        "x0_cfg"        => reference.x0_cfg,
        "std_positive"  => reference.std_positive,
        "std_cfg"       => reference.std_cfg,
        "x0_rescaled"   => reference.x0_rescaled,
        "x0_blend"      => reference.x0_blend,
      }.each do |name, expected|
        expected_shape = (name.starts_with?("std_") ? [2_i64, 1_i64, 1_i64] : [2_i64, 2_i64, 2_i64])
        actual = flow_cfg_schedule_assert_tensor(
          intermediates[name], expected_shape
        )
        if name.starts_with?("std_")
          actual.zip(expected).each { |value, wanted| value.should be_close(wanted, 1e-5_f32) }
        else
          flow_cfg_schedule_should_be_close(actual, expected)
        end
      end
    end

    expected = fixture["expected"]
    expected["pred_x_t_count"].as_i.should eq(steps)
    expected["pred_x_0_count"].as_i.should eq(steps)
    source_final = flow_cfg_schedule_assert_tensor(
      expected["samples"], [2_i64, 2_i64, 2_i64]
    )
    flow_cfg_schedule_should_be_close(source_final, reference_final)
    source_digest = flow_cfg_schedule_f32le_sha256(source_final)
    source_digest.should eq(
      "559510f18da6f3c483de3981bed12d86146edf861dd18e63182f5a05c2accfdc"
    )
    independent = fixture["independent_scalar_reference"]
    independent["schedule_f64le_sha256"].as_s.should eq(
      flow_cfg_schedule_f64le_sha256(schedule)
    )
    independent["max_abs_error"].as_f.should be <= 2e-3_f64
    independent["samples_f32le_sha256"].as_s.should eq(
      flow_cfg_schedule_f32le_sha256(reference_final)
    )

    narrowed_schedule = schedule.map(&.to_f32.to_f64)
    _, narrowed_final = flow_cfg_schedule_reference(
      noise,
      positive_condition,
      negative_condition,
      narrowed_schedule,
      sigma_min,
      model_x_scale,
      model_t_scale,
      guidance_strength,
      guidance_rescale
    )
    endpoint_counterfactual = fixture["counterfactuals"]["endpoint_pre_narrowed_schedule"]
    endpoint_counterfactual["schedule_values"].as_a.map(&.as_f).should eq(
      narrowed_schedule
    )
    endpoint_counterfactual["differing_element_count"].as_i.should be > 0_i64
    endpoint_counterfactual["first_differing_flat_index"].as_i.should be >= 0_i64
    source_final.zip(narrowed_final).any? { |left, right| left != right }.should be_true
    flow_cfg_schedule_assert_tensor(
      endpoint_counterfactual["samples"], [2_i64, 2_i64, 2_i64]
    ).should_not eq(source_final)

    _, prediction_final = flow_cfg_schedule_reference(
      noise,
      positive_condition,
      negative_condition,
      schedule,
      sigma_min,
      model_x_scale,
      model_t_scale,
      guidance_strength,
      guidance_rescale,
      "prediction"
    )
    prediction_counterfactual = fixture["counterfactuals"]["prediction_space_rescale"]
    prediction_counterfactual["differing_element_count"].as_i.should be > 0_i64
    source_final.zip(prediction_final).any? { |left, right| left != right }.should be_true
    flow_cfg_schedule_assert_tensor(
      prediction_counterfactual["samples"], [2_i64, 2_i64, 2_i64]
    ).should_not eq(source_final)

    rejected = fixture["rejected_scope"].as_a.map(&.as_s).join(" ")
    rejected.should contain("Crystal repeated CFG-rescale sampler loop")
    rejected.should contain("RNG replay")
    rejected.should contain("GPU, or Metal")
    rejected.should contain("aggregate process")
  end
end
