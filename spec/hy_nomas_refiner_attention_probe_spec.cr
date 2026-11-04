require "spec"
require "json"
require "../examples/support/hy_nomas_refiner_attention_probe"

private def native_attention_replay_manifest : JSON::Any
  tensors = [
    {name: "q", shape: HyNomasRefinerAttentionProbe::Q_SHAPE},
    {name: "k", shape: HyNomasRefinerAttentionProbe::Q_SHAPE},
    {name: "v", shape: HyNomasRefinerAttentionProbe::Q_SHAPE},
    {name: "source_probabilities", shape: HyNomasRefinerAttentionProbe::SCORE_SHAPE},
    {name: "replay.stage.extra", shape: [1, 2]},
  ]
  artifact_bytes = tensors.sum(0_i64) do |item|
    item[:shape].reduce(1_i64) { |count, extent| count * extent } * sizeof(Float32)
  end
  offset = 0_i64
  JSON.parse(JSON.build do |json|
    json.object do
      json.field "schema", HyNomasRefinerAttentionProbe::SCHEMA
      json.field "schema_version", 1
      json.field "geometry" do
        json.object do
          json.field "batches", 1
          json.field "query_rows", 263
          json.field "key_rows", 263
          json.field "heads", 16
          json.field "head_dim", 128
        end
      end
      json.field "source_exact_qkv", true
      json.field "tensor_artifact" do
        json.object do
          json.field "basename", "native-replay.tensors.bin"
          json.field "bytes", artifact_bytes
          json.field "sha256", "b" * 64
        end
      end
      json.field "tensor_order", tensors.map(&.[:name])
      json.field "tensors" do
        json.object do
          tensors.each do |item|
            nbytes = item[:shape].reduce(1_i64) { |count, extent| count * extent } * sizeof(Float32)
            json.field item[:name] do
              json.object do
                json.field "dtype", "F32"
                json.field "shape", item[:shape]
                json.field "offset", offset
                json.field "nbytes", nbytes
                json.field "sha256", "a" * 64
              end
            end
            offset += nbytes
          end
        end
      end
      json.field "source_note", "extra ordered stage records are permitted"
    end
  end)
end

describe HyNomasRefinerAttentionProbe do
  it "admits source-exact Q/K/V and source probabilities with additional ordered replay stages" do
    packet = HyNomasRefinerAttentionProbe.validate_manifest_document!(
      native_attention_replay_manifest, "c" * 64)
    packet.tensor_order.should eq([
      "q", "k", "v", "source_probabilities", "replay.stage.extra",
    ])
    packet.tensors["q"].shape.should eq(HyNomasRefinerAttentionProbe::Q_SHAPE)
    packet.tensors["source_probabilities"].shape.should eq(
      HyNomasRefinerAttentionProbe::SCORE_SHAPE)
    packet.payload_bytes.should eq(packet.tensors.values.sum(&.nbytes))
  end

  it "rejects false or non-boolean source-exact claims and wrong input shapes" do
    false_claim = native_attention_replay_manifest
    false_claim.as_h["source_exact_qkv"] = JSON::Any.new(false)
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(false_claim, "c" * 64)
    end

    typed_claim = native_attention_replay_manifest
    typed_claim.as_h["source_exact_qkv"] = JSON::Any.new(1_i64)
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(typed_claim, "c" * 64)
    end

    wrong_q = native_attention_replay_manifest
    wrong_q.as_h["tensors"].as_h["q"].as_h["shape"] = JSON.parse("[1,263,128,16]")
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(wrong_q, "c" * 64)
    end

    wrong_probabilities = native_attention_replay_manifest
    wrong_probabilities.as_h["tensors"].as_h["source_probabilities"].as_h["shape"] =
      JSON.parse("[1,263,16,256]")
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(wrong_probabilities, "c" * 64)
    end
  end

  it "requires a positive budget within 512 MiB and admits the exact six-stage setup" do
    required = HyNomasRefinerAttentionProbe.required_named_bytes
    required.should be > 0
    required.should be < HyNomasRefinerAttentionProbe::MAX_NAMED_BYTES
    HyNomasRefinerAttentionProbe.validate_budget!(required).should eq(required)
    HyNomasRefinerAttentionProbe.validate_budget!(HyNomasRefinerAttentionProbe::MAX_NAMED_BYTES).should eq(required)
    {0_i64, -1_i64, HyNomasRefinerAttentionProbe::MAX_NAMED_BYTES + 1}.each do |budget|
      expect_raises(ArgumentError) { HyNomasRefinerAttentionProbe.validate_budget!(budget) }
    end
    expect_raises(ArgumentError) { HyNomasRefinerAttentionProbe.validate_budget!(required - 1) }
  end

  it "rejects omitted, duplicate, reordered or overlapping payload records and unsafe basenames" do
    omitted = native_attention_replay_manifest
    omitted.as_h["tensor_order"] = JSON::Any.new(omitted["tensor_order"].as_a[0...-1])
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(omitted, "c" * 64)
    end

    duplicate = native_attention_replay_manifest
    order = duplicate["tensor_order"].as_a.dup
    order[1] = order[0]
    duplicate.as_h["tensor_order"] = JSON::Any.new(order)
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(duplicate, "c" * 64)
    end

    reordered = native_attention_replay_manifest
    order = reordered["tensor_order"].as_a.dup
    order[0], order[1] = order[1], order[0]
    reordered.as_h["tensor_order"] = JSON::Any.new(order)
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(reordered, "c" * 64)
    end

    overlap = native_attention_replay_manifest
    overlap.as_h["tensors"].as_h["k"].as_h["offset"] = JSON::Any.new(0_i64)
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(overlap, "c" * 64)
    end

    traversal = native_attention_replay_manifest
    traversal.as_h["tensor_artifact"].as_h["basename"] = JSON::Any.new("../escape.tensors.bin")
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(traversal, "c" * 64)
    end

    length_mismatch = native_attention_replay_manifest
    expected_bytes = length_mismatch["tensor_artifact"]["bytes"].as_i64
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(
        length_mismatch, "c" * 64, expected_bytes + 1)
    end

    oversized = native_attention_replay_manifest
    oversized.as_h["tensor_artifact"].as_h["bytes"] =
      JSON::Any.new(HyNomasRefinerAttentionProbe::MAX_NAMED_BYTES + 1)
    expect_raises(ArgumentError) do
      HyNomasRefinerAttentionProbe.validate_manifest_document!(oversized, "c" * 64)
    end
  end

  it "rejects existing output files before native setup" do
    File.tempfile("hy-nomas-attention-output", ".json") do |file|
      manifest, payload = HyNomasRefinerAttentionProbe.checked_output_paths!(file.path)
      expect_raises(ArgumentError) do
        HyNomasRefinerAttentionProbe.ensure_fresh_outputs!(manifest, payload)
      end
    end
  end
end
