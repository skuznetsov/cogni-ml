require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_EULER_SCHEDULE_FIXTURE_SHA256 =
  "84de71d38219b203cc122ac8e4c873db59fc90939cfbf37cfbc46a2cffa1d4c3"

private record FlowScheduleReferenceStep,
  x_t : Array(Float32),
  model_t : Float32,
  pred_v : Array(Float32),
  pred_x_prev : Array(Float32),
  pred_x_0 : Array(Float32)

private def flow_schedule_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_euler_schedule_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_EULER_SCHEDULE_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_schedule_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_schedule_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_schedule_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_schedule_collect_f32(node, values)
  values
end

private def flow_schedule_i32(node : JSON::Any) : Array(Int32)
  node.as_a.map { |value| value.as_i.to_i32 }
end

private def flow_schedule_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_schedule_f64le_sha256(values : Indexable(Float64)) : String
  bytes = Bytes.new(values.size * 8, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 8, 8])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_schedule_f32_bits(value : Float32) : UInt32
  bytes = Bytes.new(4, 0_u8)
  IO::ByteFormat::LittleEndian.encode(value, bytes)
  IO::ByteFormat::LittleEndian.decode(UInt32, bytes)
end

private def flow_schedule_f64_bits(value : Float64) : UInt64
  bytes = Bytes.new(8, 0_u8)
  IO::ByteFormat::LittleEndian.encode(value, bytes)
  IO::ByteFormat::LittleEndian.decode(UInt64, bytes)
end

private def flow_schedule_hex_u64(node : JSON::Any) : UInt64
  encoded = node.as_s
  encoded.starts_with?("0x").should be_true
  encoded[2..].to_u64(16)
end

private def flow_schedule_reference(
  noise : Indexable(Float32),
  cond : Indexable(Float32),
  schedule : Indexable(Float64),
  sigma_min : Float32,
  model_x_scale : Float32,
  model_t_scale : Float32,
  model_t_multiplier : Float64 = 1000.0_f64,
) : Tuple(Array(FlowScheduleReferenceStep), Array(Float32))
  noise.size.should eq(cond.size)
  state = noise.to_a
  steps = [] of FlowScheduleReferenceStep
  schedule.each_cons_pair do |t, t_prev|
    model_t = (model_t_multiplier * t).to_f32
    t_term = model_t * model_t_scale
    velocity = Array(Float32).new(state.size)
    state.each_with_index do |x, index|
      x_term = x * model_x_scale
      velocity << (x_term + t_term) + cond[index]
    end

    dt = (t - t_prev).to_f32
    one_minus_sigma = 1.0_f32 - sigma_min
    noise_scale = (
      sigma_min.to_f64 + (1.0_f64 - sigma_min.to_f64) * t
    ).to_f32
    pred_x_prev = Array(Float32).new(state.size)
    pred_x_0 = Array(Float32).new(state.size)
    state.each_with_index do |x, index|
      pred_v = velocity[index]
      pred_x_prev << x - dt * pred_v
      pred_x_0 << one_minus_sigma * x - noise_scale * pred_v
    end
    steps << FlowScheduleReferenceStep.new(
      state,
      model_t,
      velocity,
      pred_x_prev,
      pred_x_0
    )
    state = pred_x_prev
  end
  {steps, state}
end

describe "TRELLIS.2 source-pinned repeated flow Euler schedule" do
  it "seals Float64 scheduling and exposes pre-narrowed Float32 drift" do
    fixture = flow_schedule_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/flow-euler-schedule-oracle/v1"
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
    provenance["dtype"].as_s.should contain("float64 schedule")
    provenance["weights"].as_s.should eq("none")
    provenance["network"].as_s.should eq("none")
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_euler_schedule.py"
    )
    provenance["generator_sha256"].as_s.should eq(
      "a80b40137db32c99ebacd87f9e3d1866919313dd22c8868e25e5ea055688d2b4"
    )
    provenance["support_module"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_euler_step.py"
    )
    provenance["support_module_sha256"].as_s.should eq(
      "3530eb9c64b50c19783d70104f168d20cbedbac1ba946f14760b8484ca711925"
    )
    repo_root = File.expand_path("../../..", __DIR__)
    {
      "generator"      => "generator_sha256",
      "support_module" => "support_module_sha256",
    }.each do |path_key, digest_key|
      source = File.read(File.join(repo_root, provenance[path_key].as_s))
      Digest::SHA256.hexdigest(source.to_slice).should eq(
        provenance[digest_key].as_s
      )
    end

    expected_sources = {
      "base"              => "be8530b55ea66ac58e8ab23d650f463636dd52cf7e39c7c9c66f69bf72e6a0d1",
      "flow_euler"        => "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd",
      "cfg"               => "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee",
      "interval"          => "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2",
      "pipeline"          => "e2addfca672354284b23d1541a8f49228d5a727d49220fcb8512cca2cdd38ce9",
      "trainer"           => "09da19de08fc95315d3588d7ca078128e03914d692f20fa2a2ba58038c0cb563",
      "production_config" => "6128e63a7bd77db798c649d08fd05ac6b86f3fa7a0d4a405008ccf6cf29945c0",
    }
    expected_sources.each do |name, digest|
      provenance["sources"][name]["sha256"].as_s.should eq(digest)
    end

    contract = fixture["contract"]
    contract["owner"].as_s.should eq("FlowEulerSampler.sample")
    contract["entrypoint"].as_s.should end_with("FlowEulerSampler.sample")
    contract["schedule"].as_s.should contain("np.linspace")
    contract["state_scalar_boundary"].as_s.should contain("Python float64")
    contract["model_formula"].as_s.should contain("model_t*(1/1024)")
    contract["rng"].as_s.should start_with("none")
    contract["progress_display"].as_s.should contain("identity iterator")
    contract["counterfactual_effect"].as_s.should contain(
      "both model-time materialization and state coefficients"
    )
    contract["production_defaults"].as_s.should contain("not claimed")

    config = fixture["fixture"]
    flow_schedule_i32(config["shape"]).should eq([2, 2])
    steps = config["steps"].as_i.to_i32
    rescale_t = config["rescale_t"].as_f
    sigma_min = config["sigma_min"].as_f.to_f32
    model_x_scale = config["model_x_scale"].as_f.to_f32
    model_t_scale = config["model_t_scale"].as_f.to_f32
    steps.should eq(3)
    rescale_t.should eq(1.7_f64)
    sigma_min.should eq(0.125_f32)
    model_x_scale.should eq(0.25_f32)
    model_t_scale.should eq(1.0_f32 / 1024.0_f32)

    schedule_payload = fixture["schedule"]
    schedule_payload["dtype"].as_s.should eq("float64")
    schedule = (0..steps).map do |index|
      unscaled = 1.0_f64 - index.to_f64 / steps.to_f64
      rescale_t * unscaled /
        (1.0_f64 + (rescale_t - 1.0_f64) * unscaled)
    end
    fixture_schedule = schedule_payload["values"].as_a.map(&.as_f)
    fixture_schedule.should eq(schedule)
    fixture_bits = schedule_payload["f64_bits_hex"].as_a.map do |value|
      flow_schedule_hex_u64(value)
    end
    fixture_bits.should eq(schedule.map { |value| flow_schedule_f64_bits(value) })
    flow_schedule_f64le_sha256(schedule).should eq(
      "ed9fdc900020d54363b006746ad6dcc27658d1113fde648e269090bb49051622"
    )
    schedule_payload["f64le_sha256"].as_s.should eq(
      flow_schedule_f64le_sha256(schedule)
    )

    inputs = fixture["inputs"]
    inputs["noise_unchanged"].as_bool.should be_true
    inputs["cond_unchanged"].as_bool.should be_true
    noise = flow_schedule_f32(inputs["noise"]["values"])
    cond = flow_schedule_f32(inputs["cond"]["values"])
    noise.should eq([1.0_f32, -2.0_f32, 0.5_f32, 3.0_f32])
    cond.should eq([0.125_f32, -0.25_f32, 0.375_f32, -0.5_f32])

    probe = fixture["model_probe"]
    probe["call_count"].as_i.should eq(steps)
    probe["same_condition_every_call"].as_bool.should be_true
    probe["first_x_is_noise"].as_bool.should be_true
    probe["later_x_is_prior_output"].as_bool.should be_true
    probe["final_is_last_pred_x_prev"].as_bool.should be_true

    reference_steps, reference_final = flow_schedule_reference(
      noise,
      cond,
      schedule,
      sigma_min,
      model_x_scale,
      model_t_scale
    )
    source_steps = fixture["steps"].as_a
    source_steps.size.should eq(steps)
    source_steps.each_with_index do |source_step, index|
      reference = reference_steps[index]
      source_step["index"].as_i.should eq(index)
      source_step["t"].as_f.should eq(schedule[index])
      flow_schedule_hex_u64(source_step["t_f64_bits_hex"]).should eq(
        flow_schedule_f64_bits(schedule[index])
      )
      source_step["t_prev"].as_f.should eq(schedule[index + 1])
      flow_schedule_hex_u64(source_step["t_prev_f64_bits_hex"]).should eq(
        flow_schedule_f64_bits(schedule[index + 1])
      )
      source_step["dt_f32_bits"].as_i.to_u32.should eq(
        flow_schedule_f32_bits((schedule[index] - schedule[index + 1]).to_f32)
      )
      source_step["x_identity"].as_s.should eq(
        index == 0 ? "initial_noise" : "prior_pred_x_prev"
      )
      source_step["cond_identity_preserved"].as_bool.should be_true

      actual_x = flow_schedule_f32(source_step["x_t"]["values"])
      actual_model_t = flow_schedule_f32(source_step["model_t"]["values"])
      actual_velocity = flow_schedule_f32(source_step["pred_v"]["values"])
      actual_prev = flow_schedule_f32(source_step["pred_x_prev"]["values"])
      actual_x0 = flow_schedule_f32(source_step["pred_x_0"]["values"])
      actual_x.should eq(reference.x_t)
      actual_model_t.should eq([reference.model_t, reference.model_t])
      actual_velocity.should eq(reference.pred_v)
      actual_prev.should eq(reference.pred_x_prev)
      actual_x0.should eq(reference.pred_x_0)
      {
        "x_t"         => actual_x,
        "model_t"     => actual_model_t,
        "pred_v"      => actual_velocity,
        "pred_x_prev" => actual_prev,
        "pred_x_0"    => actual_x0,
      }.each do |name, values|
        source_step[name]["f32le_sha256"].as_s.should eq(
          flow_schedule_f32le_sha256(values)
        )
      end
    end

    expected = fixture["expected"]
    expected["pred_x_t_count"].as_i.should eq(steps)
    expected["pred_x_0_count"].as_i.should eq(steps)
    source_final = flow_schedule_f32(expected["samples"]["values"])
    source_final.should eq(reference_final)
    source_digest = flow_schedule_f32le_sha256(source_final)
    source_digest.should eq(
      "ff3c5d71d3946f4a2ca77dd5eae90f32e6da0da9ec0673a582f5165ed0cef973"
    )
    expected["samples"]["f32le_sha256"].as_s.should eq(source_digest)
    independent = fixture["independent_scalar_reference"]
    independent["schedule_f64le_sha256"].as_s.should eq(
      flow_schedule_f64le_sha256(schedule)
    )
    independent["max_abs_error"].as_f.should eq(0.0_f64)
    independent["samples_f32le_sha256"].as_s.should eq(source_digest)

    counterfactual = fixture["counterfactual_pre_narrowed_f32_schedule"]
    narrowed_schedule = schedule.map(&.to_f32.to_f64)
    counterfactual["schedule_values"].as_a.map(&.as_f).should eq(
      narrowed_schedule
    )
    counterfactual["schedule_f32_bits"].as_a.map(&.as_i.to_u32).should eq(
      schedule.map { |value| flow_schedule_f32_bits(value.to_f32) }
    )
    narrowed_steps, narrowed_final = flow_schedule_reference(
      noise,
      cond,
      narrowed_schedule,
      sigma_min,
      model_x_scale,
      model_t_scale
    )
    narrowed_steps.size.should eq(steps)
    recorded_narrowed = flow_schedule_f32(
      counterfactual["samples"]["values"]
    )
    recorded_narrowed.should eq(narrowed_final)
    narrowed_digest = flow_schedule_f32le_sha256(narrowed_final)
    narrowed_digest.should eq(
      "8f4b5262437c6c902290693689bae7a459e4c086fa972a0efac6b74eab67733a"
    )
    narrowed_digest.should_not eq(source_digest)
    counterfactual["separating_pair_count"].as_i.should eq(2_i64)
    counterfactual["differing_element_count"].as_i.should eq(2_i64)
    counterfactual["first_differing_flat_index"].as_i.should eq(0_i64)
    counterfactual["max_abs_difference"].as_f.should be > 0.0_f64
    counterfactual["dt_comparison"].as_a.each_with_index do |pair, index|
      source_dt = (schedule[index] - schedule[index + 1]).to_f32
      narrowed_dt = schedule[index].to_f32 - schedule[index + 1].to_f32
      pair["source_f32_bits"].as_i.to_u32.should eq(
        flow_schedule_f32_bits(source_dt)
      )
      pair["pre_narrowed_f32_bits"].as_i.to_u32.should eq(
        flow_schedule_f32_bits(narrowed_dt)
      )
      pair["differs"].as_bool.should eq(
        flow_schedule_f32_bits(source_dt) != flow_schedule_f32_bits(narrowed_dt)
      )
    end

    # This second negative control proves that the deterministic model actually
    # consumes source model-time units; recording [1000*t] alone is insufficient.
    _, wrong_model_time_final = flow_schedule_reference(
      noise,
      cond,
      schedule,
      sigma_min,
      model_x_scale,
      model_t_scale,
      model_t_multiplier: 1.0_f64
    )
    flow_schedule_f32le_sha256(wrong_model_time_final).should_not eq(source_digest)

    rejected = fixture["rejected_scope"].as_a.map(&.as_s)
    rejected.should contain("Crystal repeated-step sampler runtime or schedule API")
    rejected.should contain("CFG, guidance interval, or guidance rescale execution")
    rejected.should contain("random-noise or cross-device seed reproducibility")
    rejected.should contain(
      "aggregate process, stage, retained, RSS, native, or peak memory"
    )
    rejected.last.should contain("Metal")
  end
end
