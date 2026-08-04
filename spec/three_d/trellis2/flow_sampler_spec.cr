require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_EULER_FIXTURE_SHA256 =
  "961ab07d8f7015f3670efcc4a4396064c0003e6c7d0c8c9ae17790294c0ed1aa"

private def flow_euler_fixture : JSON::Any
  path = File.join(__DIR__, "../../fixtures/trellis2/flow_euler_step_cpu_v1.json")
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(FLOW_EULER_FIXTURE_SHA256)
  JSON.parse(payload)
end

private def flow_euler_collect_f32(node : JSON::Any, values : Array(Float32)) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_euler_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_euler_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_euler_collect_f32(node, values)
  values
end

private def flow_euler_i32(node : JSON::Any) : Array(Int32)
  node.as_a.map { |value| value.as_i.to_i32 }
end

private def flow_euler_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_euler_scalar_reference(
  x_t : Indexable(Float32),
  pred_v : Indexable(Float32),
  sigma_min : Float32,
  t : Float32,
  t_prev : Float32,
) : Tuple(Array(Float32), Array(Float32))
  x_t.size.should eq(pred_v.size)
  dt = t - t_prev
  one_minus_sigma = 1.0_f32 - sigma_min
  noise_scale = sigma_min + one_minus_sigma * t
  pred_x_prev = Array(Float32).new(x_t.size)
  pred_x_0 = Array(Float32).new(x_t.size)
  x_t.each_with_index do |x, index|
    velocity = pred_v[index]
    pred_x_prev << x - dt * velocity
    pred_x_0 << one_minus_sigma * x - noise_scale * velocity
  end
  {pred_x_prev, pred_x_0}
end

describe "TRELLIS.2 source-pinned flow Euler step" do
  it "matches an independent scalar F32 reference" do
    fixture = flow_euler_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/flow-euler-step-oracle/v1"
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
    provenance["dtype"].as_s.should eq("float32")
    provenance["weights"].as_s.should eq("none")
    provenance["network"].as_s.should eq("none")
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_euler_step.py"
    )
    provenance["generator_sha256"].as_s.should eq(
      "3530eb9c64b50c19783d70104f168d20cbedbac1ba946f14760b8484ca711925"
    )

    sources = provenance["sources"]
    sources["base"]["path"].as_s.should eq(
      "trellis2/pipelines/samplers/base.py"
    )
    sources["base"]["sha256"].as_s.should eq(
      "be8530b55ea66ac58e8ab23d650f463636dd52cf7e39c7c9c66f69bf72e6a0d1"
    )
    sources["flow_euler"]["path"].as_s.should eq(
      "trellis2/pipelines/samplers/flow_euler.py"
    )
    sources["flow_euler"]["sha256"].as_s.should eq(
      "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd"
    )
    sources["cfg"]["sha256"].as_s.should eq(
      "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee"
    )
    sources["interval"]["sha256"].as_s.should eq(
      "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2"
    )
    sources["pipeline"]["sha256"].as_s.should eq(
      "e2addfca672354284b23d1541a8f49228d5a727d49220fcb8512cca2cdd38ce9"
    )
    sources["trainer"]["sha256"].as_s.should eq(
      "09da19de08fc95315d3588d7ca078128e03914d692f20fa2a2ba58038c0cb563"
    )
    sources["production_config"]["sha256"].as_s.should eq(
      "6128e63a7bd77db798c649d08fd05ac6b86f3fa7a0d4a405008ccf6cf29945c0"
    )

    contract = fixture["contract"]
    contract["owner"].as_s.should eq("FlowEulerSampler.sample_once")
    contract["entrypoint"].as_s.should eq(
      "trellis2.pipelines.samplers.flow_euler.FlowEulerSampler.sample_once"
    )
    contract["model_timestep"].as_s.should eq(
      "float32[batch] = 1000 * normalized_t"
    )
    contract["state_timestep"].as_s.should eq(
      "fixture uses normalized scalar t in [0,1]"
    )
    contract["source_validation"].as_s.should contain("shape equality only")
    contract["arithmetic_scope"].as_s.should contain("general precision drift remains unsealed")
    contract["pred_eps_computed_but_not_returned"].as_bool.should be_true
    contract["production_sigma_min"].as_f.should eq(1e-5_f64)
    contract["initial_noise_in_fixture"].as_s.should contain(
      "pipeline RNG not executed"
    )

    inventory = fixture["inventory_not_executed"]
    inventory["schedule"].as_s.should contain("np.linspace")
    inventory["cfg"].as_s.should contain("positive-only")
    inventory["guidance_interval"].as_s.should contain("inclusive")
    inventory["guidance_rescale"].as_s.should contain("no zero-std guard")

    config = fixture["fixture"]
    shape = flow_euler_i32(config["shape"])
    shape.should eq([2, 3, 2, 2])
    sigma_min = config["sigma_min"].as_f.to_f32
    t = config["t"].as_f.to_f32
    t_prev = config["t_prev"].as_f.to_f32
    model_t = config["model_t"].as_f.to_f32
    sigma_min.should eq(0.125_f32)
    t.should eq(0.75_f32)
    t_prev.should eq(0.25_f32)
    model_t.should eq(750.0_f32)

    probe = fixture["model_probe"]
    probe["call_count"].as_i.should eq(1_i64)
    call = probe["calls"].as_a.first
    call["x_identity_preserved"].as_bool.should be_true
    call["cond_identity_preserved"].as_bool.should be_true
    flow_euler_i32(call["t_shape"]).should eq([2])
    call["t_dtype"].as_s.should eq("float32")
    call["t_device"].as_s.should eq("cpu")
    flow_euler_f32(call["t_values"]).should eq([model_t, model_t])

    x_payload = fixture["inputs"]["x_t"]
    velocity_payload = fixture["inputs"]["pred_v"]
    cond_payload = fixture["inputs"]["cond"]
    expected_prev_payload = fixture["expected"]["pred_x_prev"]
    expected_x0_payload = fixture["expected"]["pred_x_0"]
    {x_payload, velocity_payload, expected_prev_payload, expected_x0_payload}.each do |payload|
      flow_euler_i32(payload["shape"]).should eq(shape)
      payload["dtype"].as_s.should eq("float32")
    end

    x_t = flow_euler_f32(x_payload["values"])
    pred_v = flow_euler_f32(velocity_payload["values"])
    cond = flow_euler_f32(cond_payload["values"])
    expected_prev = flow_euler_f32(expected_prev_payload["values"])
    expected_x0 = flow_euler_f32(expected_x0_payload["values"])
    x_t.size.should eq(24)
    pred_v.size.should eq(24)
    cond.size.should eq(12)
    {x_t, pred_v, cond}.each { |values| values.all?(&.finite?).should be_true }
    flow_euler_f32le_sha256(x_t).should eq(x_payload["f32le_sha256"].as_s)
    flow_euler_f32le_sha256(pred_v).should eq(
      velocity_payload["f32le_sha256"].as_s
    )
    flow_euler_i32(cond_payload["shape"]).should eq([2, 2, 3])
    cond_payload["dtype"].as_s.should eq("float32")
    flow_euler_f32le_sha256(cond).should eq(cond_payload["f32le_sha256"].as_s)

    scalar_prev, scalar_x0 = flow_euler_scalar_reference(
      x_t, pred_v, sigma_min, t, t_prev
    )
    tolerance = fixture["independent_scalar_reference"]["tolerance"].as_f.to_f32
    scalar_prev.zip(expected_prev).each do |actual, expected|
      actual.should be_close(expected, tolerance)
    end
    scalar_x0.zip(expected_x0).each do |actual, expected|
      actual.should be_close(expected, tolerance)
    end
    flow_euler_f32le_sha256(scalar_prev).should eq(
      expected_prev_payload["f32le_sha256"].as_s
    )
    flow_euler_f32le_sha256(scalar_x0).should eq(
      expected_x0_payload["f32le_sha256"].as_s
    )
    fixture["independent_scalar_reference"]["max_abs_error_pred_x_prev"].as_f
      .should eq(0.0_f64)
    fixture["independent_scalar_reference"]["max_abs_error_pred_x_0"].as_f
      .should eq(0.0_f64)

    # This is the sharp boundary: the model sees 1000*t, but the state
    # transform must use normalized t.  Substituting model_t is not equivalent.
    wrong_scale = sigma_min + (1.0_f32 - sigma_min) * model_t
    wrong_x0_first = (1.0_f32 - sigma_min) * x_t.first -
                     wrong_scale * pred_v.first
    (wrong_x0_first - expected_x0.first).abs.should be > 1.0_f32

    rejected = fixture["rejected_scope"].as_a.map(&.as_s)
    rejected.should contain("CFG, guidance interval, or guidance rescale execution")
    rejected.should contain("random-noise or cross-device seed reproducibility")
    rejected.should contain("sparse-structure flow model, decoder, or complete stage")
    rejected.last.should contain("Metal")
  end
end
