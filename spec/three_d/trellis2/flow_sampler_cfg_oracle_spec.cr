require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_EULER_CFG_FIXTURE_SHA256 =
  "b1247913533ac1fc87dcd37a0ced8d13a93c9e8e4be0c5353393b101216b4861"

private def flow_cfg_fixture : JSON::Any
  path = File.join(__DIR__, "../../fixtures/trellis2/flow_euler_cfg_cpu_v1.json")
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_EULER_CFG_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_cfg_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_cfg_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_cfg_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_cfg_collect_f32(node, values)
  values
end

private def flow_cfg_i32(node : JSON::Any) : Array(Int32)
  node.as_a.map { |value| value.as_i.to_i32 }
end

private def flow_cfg_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_cfg_f32_bits(value : Float32) : UInt32
  bytes = Bytes.new(4, 0_u8)
  IO::ByteFormat::LittleEndian.encode(value, bytes)
  IO::ByteFormat::LittleEndian.decode(UInt32, bytes)
end

private def flow_cfg_source_mix(
  positive : Indexable(Float32),
  negative : Indexable(Float32),
  strength : Float64,
) : Array(Float32)
  positive.size.should eq(negative.size)
  positive_strength = strength.to_f32
  negative_strength = (1.0_f64 - strength).to_f32
  positive.map_with_index do |value, index|
    positive_term = positive_strength * value
    negative_term = negative_strength * negative[index]
    positive_term + negative_term
  end
end

private def flow_cfg_reassociated_mix(
  positive : Indexable(Float32),
  negative : Indexable(Float32),
  strength : Float64,
) : Array(Float32)
  positive.size.should eq(negative.size)
  narrowed_strength = strength.to_f32
  positive.map_with_index do |value, index|
    delta = value - negative[index]
    negative[index] + narrowed_strength * delta
  end
end

describe "TRELLIS.2 source-pinned flow Euler CFG branches" do
  it "seals special-branch calls and source-order mixed F32 arithmetic" do
    fixture = flow_cfg_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/flow-euler-cfg-oracle/v1"
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
    provenance["dtype"].as_s.should contain("float32")
    provenance["weights"].as_s.should eq("none")
    provenance["network"].as_s.should eq("none")
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_euler_cfg.py"
    )
    provenance["generator_sha256"].as_s.should eq(
      "a51ff10617022931b234cbf455e26f696124b187c30f74dae4ba2f1446eb2837"
    )
    repo_root = File.expand_path("../../..", __DIR__)
    generator = File.read(File.join(repo_root, provenance["generator"].as_s))
    Digest::SHA256.hexdigest(generator.to_slice).should eq(
      provenance["generator_sha256"].as_s
    )

    expected_sources = {
      "base"       => "be8530b55ea66ac58e8ab23d650f463636dd52cf7e39c7c9c66f69bf72e6a0d1",
      "cfg"        => "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee",
      "interval"   => "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2",
      "flow_euler" => "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd",
    }
    expected_sources.each do |name, digest|
      provenance["sources"][name]["sha256"].as_s.should eq(digest)
    end

    contract = fixture["contract"]
    contract["owner"].as_s.should eq(
      "ClassifierFreeGuidanceSamplerMixin._inference_model"
    )
    contract["entrypoint"].as_s.should end_with(
      "FlowEulerCfgSampler._inference_model"
    )
    contract["mro"].as_a.map(&.as_s).first(4).should eq(
      [
        "FlowEulerCfgSampler",
        "ClassifierFreeGuidanceSamplerMixin",
        "FlowEulerSampler",
        "Sampler",
      ]
    )
    contract["branch_1"].as_s.should contain("only the positive")
    contract["branch_0"].as_s.should contain("only the negative")
    contract["branch_mixed"].as_s.should contain("positive call, then negative")
    contract["branch_mixed"].as_s.should contain(
      "strength*positive + (1-strength)*negative"
    )
    contract["model_timestep"].as_s.should contain("1000 * normalized_t")
    contract["condition_carriers"].as_s.should contain("by identity")
    contract["guidance_rescale"].as_s.should start_with("explicitly 0.0")
    contract["arithmetic_scope"].as_s.should contain(
      "not general cross-platform parity"
    )

    config = fixture["fixture"]
    flow_cfg_i32(config["shape"]).should eq([2, 4])
    config["normalized_t"].as_f.should eq(0.6180339887498948_f64)
    model_t = config["expected_model_t_f32"].as_f.to_f32
    model_t.should eq((1000.0_f64 * config["normalized_t"].as_f).to_f32)
    strength = config["mixed_guidance_strength"].as_f
    strength.should eq(1.7_f64)

    inputs = fixture["inputs"]
    inputs["all_unchanged"].as_bool.should be_true
    x_t = flow_cfg_f32(inputs["x_t"]["values"])
    positive = flow_cfg_f32(inputs["pred_pos"]["values"])
    negative = flow_cfg_f32(inputs["pred_neg"]["values"])
    x_t.size.should eq(8)
    positive.size.should eq(x_t.size)
    negative.size.should eq(x_t.size)
    {
      "x_t"      => x_t,
      "pred_pos" => positive,
      "pred_neg" => negative,
    }.each do |name, values|
      inputs[name]["f32le_sha256"].as_s.should eq(
        flow_cfg_f32le_sha256(values)
      )
    end

    cases = fixture["cases"].as_a
    cases.size.should eq(3)
    positive_case, negative_case, mixed_case = cases
    positive_case["name"].as_s.should eq("positive_only")
    positive_case["guidance_strength"].as_f.should eq(1.0_f64)
    positive_case["call_count"].as_i.should eq(1_i64)
    positive_case["call_order"].as_a.map(&.as_s).should eq(["positive"])
    flow_cfg_f32(positive_case["output"]["values"]).should eq(positive)

    negative_case["name"].as_s.should eq("negative_only")
    negative_case["guidance_strength"].as_f.should eq(0.0_f64)
    negative_case["call_count"].as_i.should eq(1_i64)
    negative_case["call_order"].as_a.map(&.as_s).should eq(["negative"])
    flow_cfg_f32(negative_case["output"]["values"]).should eq(negative)

    mixed_case["name"].as_s.should eq("mixed")
    mixed_case["guidance_strength"].as_f.should eq(strength)
    mixed_case["call_count"].as_i.should eq(2_i64)
    mixed_case["call_order"].as_a.map(&.as_s).should eq(
      ["positive", "negative"]
    )
    cases.each do |source_case|
      calls = source_case["calls"].as_a
      source_case["call_count"].as_i.should eq(calls.size)
      source_case["call_order"].as_a.map(&.as_s).should eq(
        calls.map { |call| call["condition"].as_s }
      )
      calls.each_with_index do |call, index|
        call["index"].as_i.should eq(index)
        call["x_t_identity_preserved"].as_bool.should be_true
        expected_prediction = call["condition"].as_s == "positive" ? positive : negative
        actual_prediction = flow_cfg_f32(call["prediction"]["values"])
        actual_prediction.should eq(expected_prediction)
        call["prediction"]["f32le_sha256"].as_s.should eq(
          flow_cfg_f32le_sha256(actual_prediction)
        )
        actual_model_t = flow_cfg_f32(call["model_t"]["values"])
        actual_model_t.should eq([model_t, model_t])
        call["model_t"]["f32le_sha256"].as_s.should eq(
          flow_cfg_f32le_sha256(actual_model_t)
        )
      end
    end

    source_mixed = flow_cfg_f32(mixed_case["output"]["values"])
    reference_mixed = flow_cfg_source_mix(positive, negative, strength)
    source_mixed.should eq(reference_mixed)
    source_digest = flow_cfg_f32le_sha256(source_mixed)
    mixed_case["output"]["f32le_sha256"].as_s.should eq(source_digest)
    independent = fixture["independent_scalar_reference"]
    independent["formula"].as_s.should start_with("f32(")
    flow_cfg_f32(independent["output"]["values"]).should eq(reference_mixed)
    independent["output"]["f32le_sha256"].as_s.should eq(source_digest)
    independent["max_abs_error"].as_f.should eq(0.0_f64)

    # The numerical formula alone cannot seal the two special branches: an
    # always-two-call implementation has the same finite outputs at 0 and 1.
    flow_cfg_source_mix(positive, negative, 1.0_f64).should eq(positive)
    flow_cfg_source_mix(positive, negative, 0.0_f64).should eq(negative)
    positive_case["call_count"].as_i.should_not eq(2_i64)
    negative_case["call_count"].as_i.should_not eq(2_i64)

    counterfactual = fixture["counterfactual_reassociated_mix"]
    counterfactual["formula"].as_s.should eq(
      "negative + strength*(positive-negative)"
    )
    reassociated = flow_cfg_reassociated_mix(positive, negative, strength)
    recorded_reassociated = flow_cfg_f32(counterfactual["output"]["values"])
    recorded_reassociated.should eq(reassociated)
    counterfactual["output"]["f32le_sha256"].as_s.should eq(
      flow_cfg_f32le_sha256(reassociated)
    )
    reassociated.should_not eq(source_mixed)
    differing = source_mixed.each_index.select do |index|
      flow_cfg_f32_bits(source_mixed[index]) !=
        flow_cfg_f32_bits(reassociated[index])
    end.to_a
    differing.size.should eq(counterfactual["differing_element_count"].as_i)
    differing.first.should eq(counterfactual["first_differing_flat_index"].as_i)
    counterfactual["source_f32_bits"].as_a.each_with_index do |node, index|
      node.as_i64.to_u32.should eq(flow_cfg_f32_bits(source_mixed[index]))
    end
    counterfactual["counterfactual_f32_bits"].as_a.each_with_index do |node, index|
      node.as_i64.to_u32.should eq(flow_cfg_f32_bits(reassociated[index]))
    end
    counterfactual["max_abs_difference"].as_f.should be > 0.0_f64

    rejected = fixture["rejected_scope"].as_a.map(&.as_s)
    rejected.should contain("Crystal CFG runtime or public API")
    rejected.should contain(
      "guidance interval or standard-deviation guidance rescale execution"
    )
    rejected.should contain("Euler step or repeated sampler execution")
    rejected.should contain(
      "production defaults, pipeline, sparse-structure flow, or decoder"
    )
    rejected.last.should contain("Metal")
  end
end
