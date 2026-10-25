require "spec"
require "../examples/support/hy_nomas_conditioned_four_probe"

private def conditioned_four_document : JSON::Any
  cursor = 0_i64
  tensors = {} of String => JSON::Any
  HyNomasConditionedFourProbe.shapes.keys.sort.each do |name|
    shape = HyNomasConditionedFourProbe.shapes[name]
    bytes = shape.reduce(1_i64) { |a, b| a * b } * 4
    tensors[name] = JSON.parse({dtype: "F32", shape: shape, offset_bytes: cursor, nbytes: bytes}.to_json)
    cursor += bytes
  end
  blocks = {} of String => JSON::Any
  HyNomasFourBlockProbe::SOURCE_PINS.each_with_index do |pin, i|
    blocks["block#{i}"] = JSON.parse({block_index: i, file_sha256: pin[:file_sha256],
                                      manifest_sha256: pin[:manifest_sha256],
                                      tensor_prefix: "double_blocks.#{i}.", tensor_count: 32}.to_json)
  end
  JSON.parse({schema: HyNomasConditionedFourProbe::SCHEMA, schema_version: 1,
              upstream: {revision: HyNomasPretrainedSerialProbe::PINNED_UPSTREAM_REVISION},
              geometry: {batches: 1, video_tokens: 5280, text_tokens: 1280, dim: 2048,
                         heads: 16, head_dim: 128, ffn_dim: 8192, query_chunk: 256,
                         attention_segments: 1, chunk_size_tokens: 0, rope_grid: [6, 22, 40],
                         rope_axis_dims: [16, 56, 56], rope_theta: 256.0, valid_text_tokens: 23,
                         padded_text_tokens: 1257, valid_joint_tokens: 5303, refiner_input_dim: 3584, refiner_depth: 2},
              tolerance: {atol: 2e-5, rtol: 2e-5},
              precision: {compute: "F32", parameter_storage: "F32", attention: "PyTorch MPS SDPA F32", parameter_values: "BF16-rounded values"},
              producer: {device: "mps", activation_readback: "one packed final F32 tensor sidecar readback",
                         conditioning_invocations: 1, refiner_rows: 1280, refiner_attention_calls: 2,
                         block_invocations: ["block0", "block1", "block2", "block3"], captures_per_block: ["img.output", "txt.output"],
                         attention_adapter_calls: 4, attention_query_chunks_per_block: 21,
                         attention_query_chunk_calls: 84, terminal_tensor_count: 13,
                         padding_poison_control_calls: 1, padding_poison_control_query_chunks: 21,
                         padding_poison_control_passed: true, block_handoff: {same_mps_tensor_objects: true}, terminal_sidecar_bytes: cursor},
              qwen_source: {reference_sha256: "a" * 64, sidecar_sha256: "b" * 64, state_index: 26, crop: [108, 1388]},
              text_source: {sha256: HyNomasConditionedFourProbe::TEXT_SHA, manifest_sha256: HyNomasConditionedFourProbe::TEXT_MANIFEST_SHA,
                            nbytes: 583_316_840_i64, tensor_count: 42},
              mask: Array(Int32).new(1280) { |i| i < 23 ? 1 : 0 }, timestep: 500.0,
              padding_policy: {kind: "unpad_zero_tokens", valid_joint_prefix: 5303, valid_text_rows: 23, padded_text_rows: 1257, refiner_rows: 1280},
              source: {block_order: ["block0", "block1", "block2", "block3"], blocks: blocks},
              tensor_file: {name: "conditioned.tensors.f32le", encoding: "f32-le", sha256: "c" * 64, nbytes: cursor}, tensors: tensors}.to_json)
end

describe HyNomasConditionedFourProbe do
  it "admits the full padded tensor extent with only valid-prefix attention before Metal" do
    before = ML::MetalBuffer.hy_text_probe_allocations
    document = conditioned_four_document
    HyNomasConditionedFourProbe.validate_document!(document, "a" * 64)
    document["tensors"].as_h.size.should eq(13)
    document["tensor_file"]["nbytes"].as_i64.should eq(274_112_512_i64)
    ML::MetalBuffer.hy_text_probe_allocations.should eq(before)
  end

  it "rejects a shortened tensor, missing capture, changed mask, tolerance, source or execution receipt" do
    before = ML::MetalBuffer.hy_text_probe_allocations
    mutations = [
      {"tensors", "block3.txt.output", "shape", "[1,23,2048]"},
      {"source", "blocks", "block3", "null"},
      {"qwen_source", "crop", "[108,1387]"},
      {"padding_policy", "valid_joint_prefix", "6560"},
      {"tolerance", "atol", "0.001"},
      {"producer", "attention_query_chunks_per_block", "26"},
      {"producer", "padding_poison_control_passed", "false"},
    ]
    mutations.each do |mutation|
      document = conditioned_four_document
      node = document
      mutation.to_a[0...-2].each { |key| node = node[key] }
      node.as_h[mutation[-2]] = JSON.parse(mutation[-1])
      # Malformed JSON types and unsupported values both fail admission.
      expect_raises(Exception) { HyNomasConditionedFourProbe.validate_document!(document, "a" * 64) }
    end
    document = conditioned_four_document
    document["mask"].as_a[1279] = JSON.parse("1")
    expect_raises(ArgumentError) { HyNomasConditionedFourProbe.validate_document!(document, "a" * 64) }
    document = conditioned_four_document
    document["tensors"].as_h.delete("block3.img.output")
    expect_raises(ArgumentError) { HyNomasConditionedFourProbe.validate_document!(document, "a" * 64) }
    ML::MetalBuffer.hy_text_probe_allocations.should eq(before)
  end

  it "requires finite complete failures at the injected boundary for a negative control" do
    rows = HyNomasConditionedFourProbe.shapes.keys.reject(&.starts_with?("input.")).sort.map do |name|
      {name: name, elements: 1, failures: name.starts_with?("block3.") ? 1 : 0,
       raw_bit_differences: 0, max_abs: 1.0.as(Float64?), finite: true,
       native_sha256: "d" * 64, first_request_sha256_match: nil.as(Bool?)}
    end
    HyNomasConditionedFourProbe.negative_detected?("block3-as-block2", rows).should be_true
    HyNomasConditionedFourProbe.negative_detected?("padding-as-valid", rows).should be_false
    HyNomasConditionedFourProbe.negative_detected?("block3-as-block2", rows[0...-1]).should be_false
    rows[0] = rows[0].merge(finite: false)
    HyNomasConditionedFourProbe.negative_detected?("block3-as-block2", rows).should be_false
  end
end
