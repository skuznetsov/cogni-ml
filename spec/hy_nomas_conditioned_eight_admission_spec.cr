require "spec"
require "../examples/support/hy_nomas_conditioned_eight_probe"

private def conditioned_eight_document : JSON::Any
  cursor = 0_i64
  tensors = {} of String => JSON::Any
  HyNomasConditionedEightProbe.shapes.keys.sort.each do |name|
    shape = HyNomasConditionedEightProbe.shapes[name]
    bytes = shape.reduce(1_i64) { |a, b| a * b } * 4
    tensors[name] = JSON.parse({dtype: "F32", shape: shape, offset_bytes: cursor, nbytes: bytes}.to_json)
    cursor += bytes
  end
  blocks = {} of String => JSON::Any
  HyNomasConditionedEightProbe::SOURCE_PINS.each_with_index do |pin, index|
    blocks["block#{index}"] = JSON.parse({block_index: index, file_sha256: pin[:file_sha256],
                                          manifest_sha256: pin[:manifest_sha256],
                                          tensor_prefix: "double_blocks.#{index}.", tensor_count: 32}.to_json)
  end
  block_order = (0...8).map { |index| "block#{index}" }
  JSON.parse({schema: HyNomasConditionedEightProbe::SCHEMA, schema_version: 1,
              upstream: {revision: HyNomasPretrainedSerialProbe::PINNED_UPSTREAM_REVISION,
                         source_sha256: HyNomasConditionedEightProbe::UPSTREAM_SOURCE_SHA256},
              geometry: {batches: 1, video_tokens: 5280, text_tokens: 1280, dim: 2048,
                         heads: 16, head_dim: 128, ffn_dim: 8192, query_chunk: 256,
                         attention_segments: 1, chunk_size_tokens: 0, rope_grid: [6, 22, 40],
                         rope_axis_dims: [16, 56, 56], rope_theta: 256.0, valid_text_tokens: 23,
                         padded_text_tokens: 1257, valid_joint_tokens: 5303, refiner_input_dim: 3584, refiner_depth: 2},
              tolerance: {atol: 2e-5, rtol: 2e-5},
              precision: {source_weight_dtype: "F32", parameter_values: "BF16-rounded values",
                          parameter_storage: "F32", compute: "F32", attention: "PyTorch MPS SDPA F32",
                          weight_rounding: "source F32 values converted F32->BF16->F32 on MPS",
                          native_bf16_parameter_storage: false, native_bf16_compute: false,
                          cuda_bf16_parity: false},
              input_generation: {kind: "deterministic synthetic MPS F32 video input plus pinned upstream ND RoPE",
                                 video_formula: "x[i] = 0.1*sin(float32(i)*0.017+phase) + 0.05*cos(float32(i)*0.0031+phase*0.7)",
                                 video_phase: 0.11,
                                 rope: {implementation: "pinned upstream posemb_layers.get_nd_rotary_pos_embed",
                                        grid: [6, 22, 40], grid_axis_order: ["temporal", "height", "width"],
                                        axis_dims: [16, 56, 56], theta: 256.0, use_real: true, device: "mps"}},
              producer: {device: "mps", activation_readback: "one packed final F32 tensor sidecar readback",
                         conditioning_invocations: 1, refiner_rows: 1280, refiner_attention_calls: 2,
                         attention_adapter: "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256",
                         block_invocations: block_order, captures_per_block: ["img.output", "txt.output"],
                         attention_adapter_calls: 8, attention_query_chunks_per_block: 21,
                         attention_query_chunk_calls: 168, terminal_tensor_count: 21,
                         max_attention_query_chunk: 256,
                         padding_poison_control_calls: 1, padding_poison_control_query_chunks: 21,
                         padding_poison_control_passed: true,
                         block_handoff: {img: "each block output passed directly as next block input",
                                         txt: "each block output passed directly as next block input",
                                         same_mps_tensor_objects: true}, terminal_sidecar_bytes: cursor},
              qwen_source: {reference_sha256: "a" * 64, sidecar_sha256: "b" * 64, state_index: 26, crop: [108, 1388]},
              text_source: {sha256: HyNomasConditionedEightProbe::TEXT_SHA, manifest_sha256: HyNomasConditionedEightProbe::TEXT_MANIFEST_SHA,
                            nbytes: 583_316_840_i64, tensor_count: 42},
              mask: Array(Int32).new(1280) { |i| i < 23 ? 1 : 0 }, timestep: 500.0,
              padding_policy: {kind: "unpad_zero_tokens", valid_joint_prefix: 5303, valid_text_rows: 23,
                               padded_text_rows: 1257, refiner_rows: 1280,
                               padded_query_outputs: "attention result explicitly zero-filled"},
              source: {block_order: block_order, blocks: blocks},
              tensor_file: {name: "conditioned-eight.tensors.f32le", encoding: "f32-le", sha256: "c" * 64, nbytes: cursor}, tensors: tensors}.to_json)
end

describe HyNomasConditionedEightProbe do
  it "admits exactly twenty-one full-J tensors and eight ordered block pins before Metal" do
    before = ML::MetalBuffer.hy_text_probe_allocations
    document = conditioned_eight_document
    HyNomasConditionedEightProbe.validate_document!(document, "a" * 64)
    document["tensors"].as_h.size.should eq(21)
    document["tensor_file"]["nbytes"].as_i64.should eq(489_070_592_i64)
    document["source"]["block_order"].as_a.size.should eq(8)
    ML::MetalBuffer.hy_text_probe_allocations.should eq(before)
  end

  it "rejects tensor, source, geometry, tolerance, precision, and execution-receipt drift" do
    before = ML::MetalBuffer.hy_text_probe_allocations
    mutations = [
      {path: ["tensors", "block7.txt.output", "shape"], value: "[1,23,2048]"},
      {path: ["source", "block_order"], value: "[\"block0\",\"block1\",\"block2\",\"block3\"]"},
      {path: ["source", "blocks", "block7", "file_sha256"], value: "\"" + ("0" * 64) + "\""},
      {path: ["upstream", "source_sha256", "hymm/models/dit/modules/embed_layers.py"], value: "\"" + ("0" * 64) + "\""},
      {path: ["input_generation", "rope", "grid_axis_order"], value: "[\"width\",\"height\",\"temporal\"]"},
      {path: ["producer", "attention_adapter"], value: "\"different attention implementation\""},
      {path: ["producer", "block_handoff", "same_mps_tensor_objects"], value: "false"},
      {path: ["padding_policy", "padded_query_outputs"], value: "\"undefined\""},
      {path: ["qwen_source", "crop"], value: "[108,1387]"},
      {path: ["padding_policy", "valid_joint_prefix"], value: "6560"},
      {path: ["tolerance", "atol"], value: "0.001"},
      {path: ["producer", "attention_adapter_calls"], value: "4"},
      {path: ["producer", "attention_query_chunk_calls"], value: "84"},
      {path: ["producer", "terminal_tensor_count"], value: "13"},
      {path: ["precision", "parameter_values"], value: "\"unrounded F32 values\""},
      {path: ["producer", "padding_poison_control_passed"], value: "false"},
    ]
    mutations.each do |mutation|
      document = conditioned_eight_document
      node = document
      mutation[:path][0...-1].each { |key| node = node[key] }
      node.as_h[mutation[:path].last] = JSON.parse(mutation[:value])
      expect_raises(Exception) { HyNomasConditionedEightProbe.validate_document!(document, "a" * 64) }
    end
    document = conditioned_eight_document
    document["mask"].as_a[1279] = JSON.parse("1")
    expect_raises(ArgumentError) { HyNomasConditionedEightProbe.validate_document!(document, "a" * 64) }
    document = conditioned_eight_document
    document["tensors"].as_h.delete("block7.img.output")
    expect_raises(ArgumentError) { HyNomasConditionedEightProbe.validate_document!(document, "a" * 64) }
    document = conditioned_eight_document
    document["tensors"].as_h["unexpected"] = JSON.parse("{}")
    expect_raises(Exception) { HyNomasConditionedEightProbe.validate_document!(document, "a" * 64) }
    ML::MetalBuffer.hy_text_probe_allocations.should eq(before)
  end

  it "detects wrong block substitution at the first new boundary and at block seven" do
    [4, 7].each do |fault|
      rows = HyNomasConditionedEightProbe.shapes.keys.reject(&.starts_with?("input.")).sort.map do |name|
        {name: name, elements: 1, failures: name.starts_with?("block#{fault}.") ? 1 : 0,
         raw_bit_differences: 0, max_abs: 1.0.as(Float64?), finite: true,
         native_sha256: "d" * 64, first_request_sha256_match: nil.as(Bool?)}
      end
      mode = fault == 4 ? "block4-as-block3" : "block7-as-block6"
      HyNomasConditionedEightProbe.negative_detected?(mode, rows).should be_true
      HyNomasConditionedEightProbe.negative_detected?("padding-as-valid", rows).should be_false
      HyNomasConditionedEightProbe.negative_detected?(mode, rows[0...-1]).should be_false
      rows[0] = rows[0].merge(finite: false)
      HyNomasConditionedEightProbe.negative_detected?(mode, rows).should be_false
    end
  end
end
