require "json"
require "digest/sha256"
require "../../spec_helper"

private FLOW_CFG_RESCALE_FIXTURE_SHA256 =
  "8b6269472e579ab1f1c3b5f735fa15357f5672b4d0dfb5bb2fa0726f0cd156f0"

private def flow_rescale_fixture : JSON::Any
  path = File.join(__DIR__, "../../fixtures/trellis2/flow_cfg_rescale_cpu_v1.json")
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    FLOW_CFG_RESCALE_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def flow_rescale_collect_f32(
  node : JSON::Any,
  values : Array(Float32),
) : Nil
  if nested = node.as_a?
    nested.each { |value| flow_rescale_collect_f32(value, values) }
  else
    values << node.as_f.to_f32
  end
end

private def flow_rescale_values(node : JSON::Any) : Array(Float32)
  values = [] of Float32
  flow_rescale_collect_f32(node["values"], values)
  values
end

private def flow_rescale_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_rescale_bits_sha256(node : JSON::Any) : String
  bits = node["f32_bits_hex"].as_a.map do |item|
    item.as_s.lchop("0x").to_u32(16)
  end
  bytes = Bytes.new(bits.size * 4, 0_u8)
  bits.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def flow_rescale_case(fixture : JSON::Any, name : String) : JSON::Any
  fixture["cases"].as_a.find! { |item| item["name"].as_s == name }
end

private def flow_rescale_assert_finite_tensor(node : JSON::Any) : Array(Float32)
  node["dtype"].as_s.should eq("float32")
  node["all_finite"].as_bool.should be_true
  values = flow_rescale_values(node)
  node["f32le_sha256"].as_s.should eq(flow_rescale_f32le_sha256(values))
  values
end

private def flow_rescale_source_mix(
  positive : Indexable(Float32),
  negative : Indexable(Float32),
  strength : Float64,
) : Array(Float32)
  positive_strength = strength.to_f32
  negative_strength = (1.0_f64 - strength).to_f32
  positive.map_with_index do |value, index|
    positive_term = positive_strength * value
    negative_term = negative_strength * negative[index]
    positive_term + negative_term
  end
end

private def flow_rescale_sample_std(values : Indexable(Float32)) : Array(Float32)
  values.size.should eq(8)
  Array.new(2) do |batch|
    offset = batch * 4
    sum = 0.0_f32
    4.times { |index| sum += values[offset + index] }
    mean = sum / 4.0_f32
    squared = 0.0_f32
    4.times do |index|
      delta = values[offset + index] - mean
      squared += delta * delta
    end
    Math.sqrt(squared / 3.0_f32).to_f32
  end
end

describe "TRELLIS.2 source-pinned CFG guidance rescale" do
  it "seals routing, source arithmetic, variance edges, and non-finite behavior" do
    fixture = flow_rescale_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/flow-cfg-rescale-oracle/v1"
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
    provenance["weights"].as_s.should eq("none")
    provenance["network"].as_s.should eq("none")
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_cfg_rescale.py"
    )
    provenance["generator_sha256"].as_s.should eq(
      "d0c6fed2a1eda7ed4084a3350001c27313b0fe45de49b3853c6e3c165db68be5"
    )
    provenance["support"].as_s.should eq(
      "tools/trellis2_oracle/export_flow_euler_cfg.py"
    )
    provenance["support_sha256"].as_s.should eq(
      "a51ff10617022931b234cbf455e26f696124b187c30f74dae4ba2f1446eb2837"
    )
    repo_root = File.expand_path("../../..", __DIR__)
    ["generator", "support"].each do |path_key|
      source = File.read(File.join(repo_root, provenance[path_key].as_s))
      provenance["#{path_key}_sha256"].as_s.should eq(
        Digest::SHA256.hexdigest(source.to_slice)
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
    contract["owner"].as_s.should eq(
      "ClassifierFreeGuidanceSamplerMixin._inference_model"
    )
    contract["mro"].as_a.map(&.as_s).first(4).should eq(
      [
        "FlowEulerCfgSampler",
        "ClassifierFreeGuidanceSamplerMixin",
        "FlowEulerSampler",
        "Sampler",
      ]
    )
    contract["rescale_gate"].as_s.should contain("guidance_rescale > 0")
    contract["validation"].as_s.should contain("no clamp")
    contract["validation"].as_s.should contain("epsilon")
    contract["std_axes"].as_a.map(&.as_i).should eq([1_i64, 2_i64])
    contract["std_correction"].as_i.should eq(1_i64)
    contract["std_keepdim"].as_bool.should be_true
    contract["normalized_t"].as_f.should eq(0.6180339887498948_f64)
    contract["sigma_min"].as_f.should eq(1e-5_f64)
    contract["model_timestep"].as_s.should contain("1000*normalized_t")

    fixture["cases"].as_a.size.should eq(10)
    positive = flow_rescale_case(fixture, "positive_only_ignores_rescale")
    negative = flow_rescale_case(fixture, "negative_only_ignores_rescale")
    positive["call_order"].as_a.map(&.as_s).should eq(["positive"])
    negative["call_order"].as_a.map(&.as_s).should eq(["negative"])
    positive["output_identity"].as_s.should eq("positive_prediction")
    negative["output_identity"].as_s.should eq("negative_prediction")
    positive["guidance_rescale"]["class"].as_s.should eq("positive_infinity")
    negative["guidance_rescale"]["class"].as_s.should eq("nan")
    [positive, negative].each do |source_case|
      source_case["rescale_branch_entered"].as_bool.should be_false
      source_case["conversion_trace"].as_a.should be_empty
    end

    bypass_names = [
      "mixed_zero_bypass",
      "mixed_negative_bypass",
      "mixed_nan_bypass",
    ]
    bypass = bypass_names.map { |name| flow_rescale_case(fixture, name) }
    bypass.each do |source_case|
      source_case["call_order"].as_a.map(&.as_s).should eq(
        ["positive", "negative"]
      )
      source_case["rescale_branch_entered"].as_bool.should be_false
      source_case["conversion_trace"].as_a.should be_empty
      source_case["output_identity"].as_s.should eq("owned_mixed")
      source_case["inputs"]["all_unchanged"].as_bool.should be_true
    end
    bypass.map { |source_case| source_case["output"]["f32le_sha256"].as_s }
      .uniq.size.should eq(1)
    bypass.last["guidance_rescale"]["class"].as_s.should eq("nan")
    bypass_inputs = bypass.first["inputs"]
    flow_rescale_assert_finite_tensor(bypass.first["output"]).should eq(
      flow_rescale_source_mix(
        flow_rescale_assert_finite_tensor(bypass_inputs["positive"]),
        flow_rescale_assert_finite_tensor(bypass_inputs["negative"]),
        1.7_f64
      )
    )

    conversion_trace = [
      "pred_to_xstart:positive",
      "pred_to_xstart:cfg",
      "xstart_to_pred:blend",
    ]
    rescaled_names = [
      "mixed_finite_rescale",
      "mixed_unclamped_rescale",
      "mixed_positive_infinity",
      "mixed_zero_cfg_std",
      "mixed_near_zero_cfg_std",
    ]
    rescaled_names.each do |name|
      source_case = flow_rescale_case(fixture, name)
      source_case["rescale_branch_entered"].as_bool.should be_true
      source_case["call_order"].as_a.map(&.as_s).should eq(
        ["positive", "negative"]
      )
      source_case["conversion_trace"].as_a.map(&.as_s).should eq(
        conversion_trace
      )
      source_case["output_identity"].as_s.should eq("owned_mixed")
      source_case["calls"].as_a.each do |call|
        call["x_t_identity_preserved"].as_bool.should be_true
        call["condition_identity_preserved"].as_bool.should be_true
        model_t = flow_rescale_assert_finite_tensor(call["model_t"])
        model_t.should eq([618.0339965820312_f32, 618.0339965820312_f32])
      end
    end

    finite = flow_rescale_case(fixture, "mixed_finite_rescale")
    finite["guidance_rescale"]["value"].as_f.should eq(0.35_f64)
    inputs = finite["inputs"]
    x_t = flow_rescale_assert_finite_tensor(inputs["x_t"])
    pred_positive = flow_rescale_assert_finite_tensor(inputs["positive"])
    pred_negative = flow_rescale_assert_finite_tensor(inputs["negative"])
    intermediates = finite["intermediates"]
    pred_cfg = flow_rescale_assert_finite_tensor(intermediates["pred_cfg"])
    pred_cfg.should eq(
      flow_rescale_source_mix(pred_positive, pred_negative, 1.7_f64)
    )

    x0_positive = flow_rescale_assert_finite_tensor(intermediates["x0_positive"])
    x0_cfg = flow_rescale_assert_finite_tensor(intermediates["x0_cfg"])
    std_positive = flow_rescale_assert_finite_tensor(intermediates["std_positive"])
    std_cfg = flow_rescale_assert_finite_tensor(intermediates["std_cfg"])
    std_positive.size.should eq(2)
    std_cfg.size.should eq(2)
    intermediates["std_positive"]["shape"].as_a.map(&.as_i).should eq(
      [2_i64, 1_i64, 1_i64]
    )
    scalar_std_positive = flow_rescale_sample_std(x0_positive)
    scalar_std_cfg = flow_rescale_sample_std(x0_cfg)
    independent = fixture["independent_reference"]
    flow_rescale_assert_finite_tensor(independent["std_positive"]).should eq(
      scalar_std_positive
    )
    flow_rescale_assert_finite_tensor(independent["std_cfg"]).should eq(
      scalar_std_cfg
    )
    independent["max_abs_error_vs_torch"].as_f.should be <= 1e-3_f64
    scalar_std_positive.zip(std_positive).each do |actual, expected|
      actual.should be_close(expected, 1e-3_f32)
    end
    scalar_std_cfg.zip(std_cfg).each do |actual, expected|
      actual.should be_close(expected, 1e-3_f32)
    end

    x0_rescaled = flow_rescale_assert_finite_tensor(intermediates["x0_rescaled"])
    x0_blend = flow_rescale_assert_finite_tensor(intermediates["x0_blend"])
    x0_cfg.each_with_index do |value, index|
      batch = index // 4
      expected_rescaled = value * (std_positive[batch] / std_cfg[batch])
      x0_rescaled[index].should eq(expected_rescaled)
      expected_blend = 0.35_f32 * expected_rescaled + 0.65_f32 * value
      x0_blend[index].should eq(expected_blend)
    end
    coefficient_x = (1.0_f64 - 1e-5_f64).to_f32
    coefficient_pred = (
      1e-5_f64 + (1.0_f64 - 1e-5_f64) * 0.6180339887498948_f64
    ).to_f32
    expected_x0_positive = x_t.map_with_index do |value, index|
      coefficient_x * value - coefficient_pred * pred_positive[index]
    end
    expected_x0_cfg = x_t.map_with_index do |value, index|
      coefficient_x * value - coefficient_pred * pred_cfg[index]
    end
    x0_positive.should eq(expected_x0_positive)
    x0_cfg.should eq(expected_x0_cfg)
    expected_output = x_t.map_with_index do |value, index|
      (coefficient_x * value - x0_blend[index]) / coefficient_pred
    end
    output = flow_rescale_assert_finite_tensor(finite["output"])
    output.should eq(expected_output)

    unclamped = flow_rescale_case(fixture, "mixed_unclamped_rescale")
    unclamped["guidance_rescale"]["value"].as_f.should eq(1.7_f64)
    unclamped["output"]["f32le_sha256"].as_s.should_not eq(
      fixture["counterfactuals"]["clamp_unclamped_rescale_to_one_output"]["f32le_sha256"].as_s
    )

    positive_infinity = flow_rescale_case(fixture, "mixed_positive_infinity")
    positive_infinity["guidance_rescale"]["class"].as_s.should eq(
      "positive_infinity"
    )
    positive_infinity["output"]["all_finite"].as_bool.should be_false
    positive_infinity["output"]["classes"].as_a.map(&.as_s).uniq.should eq(
      ["nan"]
    )
    flow_rescale_bits_sha256(positive_infinity["output"]).should eq(
      positive_infinity["output"]["f32le_sha256"].as_s
    )

    zero_std = flow_rescale_case(fixture, "mixed_zero_cfg_std")
    flow_rescale_assert_finite_tensor(
      zero_std["intermediates"]["std_cfg"]
    ).should eq([0.0_f32, 0.0_f32])
    zero_std["output"]["all_finite"].as_bool.should be_false
    zero_std["output"]["classes"].as_a.map(&.as_s).uniq.should eq(
      ["positive_infinity"]
    )
    flow_rescale_bits_sha256(zero_std["output"]).should eq(
      zero_std["output"]["f32le_sha256"].as_s
    )

    near_std = flow_rescale_case(fixture, "mixed_near_zero_cfg_std")
    near_values = flow_rescale_assert_finite_tensor(
      near_std["intermediates"]["std_cfg"]
    )
    near_values.all? { |value| value > 0.0_f32 && value < 1e-6_f32 }
      .should be_true
    ratios = flow_rescale_assert_finite_tensor(
      near_std["intermediates"]["std_positive"]
    ).zip(near_values).map { |numerator, denominator| numerator / denominator }
    ratios.all? { |ratio| ratio > 100_000.0_f32 }.should be_true
    near_std["output"]["all_finite"].as_bool.should be_true

    counterfactuals = fixture["counterfactuals"]
    finite["output"]["f32le_sha256"].as_s.should_not eq(
      counterfactuals["global_axes_output"]["f32le_sha256"].as_s
    )
    finite["output"]["f32le_sha256"].as_s.should_not eq(
      counterfactuals["prediction_space_output"]["f32le_sha256"].as_s
    )
    counterfactuals["population_std_positive"]["f32le_sha256"].as_s
      .should_not eq(intermediates["std_positive"]["f32le_sha256"].as_s)
    counterfactuals["population_std_cfg"]["f32le_sha256"].as_s
      .should_not eq(intermediates["std_cfg"]["f32le_sha256"].as_s)
    counterfactuals["population_ratio_note"].as_s.should contain("cancels")

    fixture["rejected_scope"].as_a.map(&.as_s).join(" ").should contain(
      "Crystal guidance-rescale runtime"
    )
    fixture["rejected_scope"].as_a.map(&.as_s).join(" ").should contain(
      "GPU, or Metal"
    )
  end
end
