require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_GUIDANCE_INTERVAL_FIXTURE_SHA256 =
  "290f816befd68f498be8df4008230c6433afbb615c70f1d40f11cfba0d7d109a"

private def flow_guidance_interval_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../../fixtures/trellis2/flow_guidance_interval_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_GUIDANCE_INTERVAL_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_guidance_interval_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_guidance_interval_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_guidance_interval_f32(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_guidance_interval_collect_f32(node, values)
  values
end

private def flow_guidance_interval_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_guidance_interval_f64_bits_hex(value : Float64) : String
  bytes = Bytes.new(8, 0_u8)
  IO::ByteFormat::LittleEndian.encode(value, bytes)
  bits = IO::ByteFormat::LittleEndian.decode(UInt64, bytes)
  "0x#{bits.to_s(16).rjust(16, '0')}"
end

private def flow_guidance_interval_source_mix(
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

describe "TRELLIS.2 source-pinned flow guidance interval routing" do
  it "seals inclusive Float64 boundaries before F32 model-time narrowing" do
    fixture = flow_guidance_interval_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/flow-guidance-interval-oracle/v1"
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
    provenance["dtype"].as_s.should contain("Float64 normalized time")
    provenance["dtype"].as_s.should contain("Float32")
    provenance["weights"].as_s.should eq("none")
    provenance["network"].as_s.should eq("none")

    repo_root = File.expand_path("../../..", __DIR__)
    generator_path = provenance["generator"].as_s
    generator_path.should eq(
      "tools/trellis2_oracle/export_flow_guidance_interval.py"
    )
    generator = File.read(File.join(repo_root, generator_path))
    Digest::SHA256.hexdigest(generator.to_slice).should eq(
      "4e863eac8d07ef8649cad4fbeccd236a3bf5838a42a0b16c3ec6d1b4cfd25113"
    )
    provenance["generator_sha256"].as_s.should eq(
      Digest::SHA256.hexdigest(generator.to_slice)
    )
    support_path = provenance["support"].as_s
    support_path.should eq("tools/trellis2_oracle/export_flow_euler_cfg.py")
    support = File.read(File.join(repo_root, support_path))
    provenance["support_sha256"].as_s.should eq(
      Digest::SHA256.hexdigest(support.to_slice)
    )
    provenance["support_sha256"].as_s.should eq(
      "a51ff10617022931b234cbf455e26f696124b187c30f74dae4ba2f1446eb2837"
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
      "GuidanceIntervalSamplerMixin._inference_model"
    )
    contract["entrypoint"].as_s.should end_with(
      "FlowEulerGuidanceIntervalSampler._inference_model"
    )
    contract["mro"].as_a.map(&.as_s).first(5).should eq(
      [
        "FlowEulerGuidanceIntervalSampler",
        "GuidanceIntervalSamplerMixin",
        "ClassifierFreeGuidanceSamplerMixin",
        "FlowEulerSampler",
        "Sampler",
      ]
    )
    contract["predicate"].as_s.should eq(
      "guidance_interval[0] <= normalized_t <= guidance_interval[1]"
    )
    contract["outside"].as_s.should contain("strength to 1")
    contract["model_timestep"].as_s.should contain("Float64 t")
    contract["model_timestep"].as_s.should contain("Float32[batch]")
    contract["condition_carriers"].as_s.should contain("by identity")
    contract["guidance_rescale"].as_s.should start_with("explicitly 0.0")

    config = fixture["fixture"]
    config["shape"].as_a.map(&.as_i.to_i32).should eq([2, 4])
    interval = config["guidance_interval"].as_a.map(&.as_f)
    interval.should eq([0.25_f64, 0.75_f64])
    strength = config["requested_guidance_strength"].as_f
    strength.should eq(1.7_f64)
    config["case_count"].as_i.should eq(5_i64)

    inputs = fixture["inputs"]
    inputs["all_unchanged"].as_bool.should be_true
    positive = flow_guidance_interval_f32(inputs["pred_pos"]["values"])
    negative = flow_guidance_interval_f32(inputs["pred_neg"]["values"])
    positive.size.should eq(8)
    negative.size.should eq(positive.size)
    {
      "pred_pos" => positive,
      "pred_neg" => negative,
    }.each do |name, values|
      inputs[name]["f32le_sha256"].as_s.should eq(
        flow_guidance_interval_f32le_sha256(values)
      )
    end
    source_mixed = flow_guidance_interval_source_mix(
      positive,
      negative,
      strength
    )
    source_mixed.should_not eq(positive)
    source_mixed.should_not eq(negative)

    expected = {
      "below_lower" => {
        bits: "0x3fcfffffffffffff", inside: false,
        order: ["positive"], model_t: 250.0_f32,
      },
      "lower_boundary" => {
        bits: "0x3fd0000000000000", inside: true,
        order: ["positive", "negative"], model_t: 250.0_f32,
      },
      "inside" => {
        bits: "0x3fe0000000000000", inside: true,
        order: ["positive", "negative"], model_t: 500.0_f32,
      },
      "upper_boundary" => {
        bits: "0x3fe8000000000000", inside: true,
        order: ["positive", "negative"], model_t: 750.0_f32,
      },
      "above_upper" => {
        bits: "0x3fe8000000000001", inside: false,
        order: ["positive"], model_t: 750.0_f32,
      },
    }

    cases = fixture["cases"].as_a
    cases.map { |source_case| source_case["name"].as_s }.should eq(
      expected.keys
    )
    by_name = cases.to_h { |source_case| {source_case["name"].as_s, source_case} }
    cases.each do |source_case|
      name = source_case["name"].as_s
      expected_case = expected[name]
      normalized_t = source_case["normalized_t"].as_f
      source_case["normalized_t_f64_bits_hex"].as_s.should eq(
        flow_guidance_interval_f64_bits_hex(normalized_t)
      )
      source_case["normalized_t_f64_bits_hex"].as_s.should eq(
        expected_case[:bits]
      )
      independent_inside = interval[0] <= normalized_t && normalized_t <= interval[1]
      independent_inside.should eq(expected_case[:inside])
      source_case["inside_inclusive_interval"].as_bool.should eq(
        independent_inside
      )
      source_case["requested_guidance_strength"].as_f.should eq(strength)
      source_case["effective_guidance_strength"].as_f.should eq(
        independent_inside ? strength : 1.0_f64
      )
      order = source_case["call_order"].as_a.map(&.as_s)
      order.should eq(expected_case[:order])
      source_case["call_count"].as_i.should eq(order.size)
      source_case["output_identity"].as_s.should eq(
        independent_inside ? "owned_mixed" : "positive_prediction"
      )

      calls = source_case["calls"].as_a
      calls.size.should eq(order.size)
      calls.each_with_index do |call, index|
        call["index"].as_i.should eq(index)
        call["condition"].as_s.should eq(order[index])
        call["x_t_identity_preserved"].as_bool.should be_true
        call["condition_identity_preserved"].as_bool.should be_true
        model_t = flow_guidance_interval_f32(call["model_t"]["values"])
        model_t.should eq([expected_case[:model_t], expected_case[:model_t]])
        call["model_t"]["f32le_sha256"].as_s.should eq(
          flow_guidance_interval_f32le_sha256(model_t)
        )
        expected_prediction = order[index] == "positive" ? positive : negative
        prediction = flow_guidance_interval_f32(call["prediction"]["values"])
        prediction.should eq(expected_prediction)
        call["prediction"]["f32le_sha256"].as_s.should eq(
          flow_guidance_interval_f32le_sha256(prediction)
        )
      end

      output = flow_guidance_interval_f32(source_case["output"]["values"])
      output.should eq(independent_inside ? source_mixed : positive)
      source_case["output"]["f32le_sha256"].as_s.should eq(
        flow_guidance_interval_f32le_sha256(output)
      )
    end

    # Adjacent Float64 values collapse to the same F32 model timestep as each
    # exact edge. A model-time-domain comparison would therefore lose routing.
    below_model_t = by_name["below_lower"]["calls"][0]["model_t"]["values"]
    lower_model_t = by_name["lower_boundary"]["calls"][0]["model_t"]["values"]
    below_model_t.should eq(lower_model_t)
    by_name["below_lower"]["call_order"].should_not eq(
      by_name["lower_boundary"]["call_order"]
    )
    upper_model_t = by_name["upper_boundary"]["calls"][0]["model_t"]["values"]
    above_model_t = by_name["above_upper"]["calls"][0]["model_t"]["values"]
    upper_model_t.should eq(above_model_t)
    by_name["upper_boundary"]["call_order"].should_not eq(
      by_name["above_upper"]["call_order"]
    )

    independent = fixture["independent_reference"]
    independent["predicate"].as_s.should contain("normalized_t")
    independent["outside_effective_guidance_strength"].as_f.should eq(1.0_f64)
    flow_guidance_interval_f32(
      independent["inside_output"]["values"]
    ).should eq(source_mixed)

    counterfactuals = fixture["counterfactuals"]
    strict_mismatches = cases.select do |source_case|
      t = source_case["normalized_t"].as_f
      source_case["inside_inclusive_interval"].as_bool !=
        (interval[0] < t && t < interval[1])
    end.map { |source_case| source_case["name"].as_s }
    strict_mismatches.should eq(
      counterfactuals["strict_interval_mismatches"].as_a.map(&.as_s)
    )
    outside_names = cases.reject do |source_case|
      source_case["inside_inclusive_interval"].as_bool
    end.map { |source_case| source_case["name"].as_s }
    outside_names.should eq(
      counterfactuals["missing_outside_override_mismatches"].as_a.map(&.as_s)
    )
    model_domain_mismatches = cases.select do |source_case|
      t = source_case["normalized_t"].as_f * 1000.0_f64
      source_case["inside_inclusive_interval"].as_bool !=
        (interval[0] <= t && t <= interval[1])
    end.map { |source_case| source_case["name"].as_s }
    model_domain_mismatches.should eq(
      counterfactuals["model_t_domain_mismatches"].as_a.map(&.as_s)
    )
    counterfactuals["outside_negative_output_sha256"].as_s.should eq(
      flow_guidance_interval_f32le_sha256(negative)
    )
    counterfactuals["outside_negative_output_sha256"].as_s.should_not eq(
      flow_guidance_interval_f32le_sha256(positive)
    )

    rejected = fixture["rejected_scope"].as_a.map(&.as_s)
    rejected.should contain("Crystal guidance-interval runtime or public API")
    rejected.should contain("standard-deviation guidance rescale execution")
    rejected.should contain("Euler step or repeated sampler execution")
    rejected.should contain(
      "production defaults, pipeline, sparse-structure flow, or decoder"
    )
    rejected.last.should contain("Metal")
  end
end
