require "spec"
require "../examples/support/hy_nomas_text_probe"

private def hy_text_probe_document : JSON::Any
  JSON.parse({
    schema: "hy-nomas-text-conditioning-reference-v1", schema_version: 1,
    upstream_revision: "911cea6458a89c2663513ff7099561ec190dd491",
    geometry: {batches: 1, text_tokens: 1280, valid_tokens: 23, input_dim: 3584,
               dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192, video_tokens: 6,
               segments: 1, chunk_size_tokens: 0, text_seg_pos_emb_num: 0},
    precision: {resident_weights: "BF16", compute: "F32", attention: "f32"},
    qwen_source: {reference_sha256: "a" * 64, sidecar_sha256: "b" * 64, state_index: 26, crop: [108, 1388]},
    timestep: 500.0, mask: Array(Int32).new(1280) { |i| i < 23 ? 1 : 0 },
    compaction: {passed: true},
  }.to_json)
end

describe HyNomasTextProbe do
  it "admits only the qualified single-prefix control and authenticated Qwen source" do
    HyNomasTextProbe.validate_document!(hy_text_probe_document, "a" * 64)
    expect_raises(ArgumentError, /Qwen source/) do
      HyNomasTextProbe.validate_document!(hy_text_probe_document, "c" * 64)
    end
    ["batches", "segments", "chunk_size_tokens", "text_seg_pos_emb_num"].each do |field|
      document = hy_text_probe_document
      document["geometry"].as_h[field] = JSON::Any.new(2_i64)
      expect_raises(ArgumentError, /geometry/) { HyNomasTextProbe.validate_document!(document, "a" * 64) }
    end
  end

  it "rejects padding holes, forced-token-zero changes and failed compaction evidence" do
    [0, 24].each do |index|
      document = hy_text_probe_document
      document["mask"].as_a[index] = JSON::Any.new(index == 0 ? 0_i64 : 1_i64)
      expect_raises(ArgumentError, /mask/) { HyNomasTextProbe.validate_document!(document, "a" * 64) }
    end
    document = hy_text_probe_document
    document["compaction"].as_h["passed"] = JSON::Any.new(false)
    expect_raises(ArgumentError, /compaction/) { HyNomasTextProbe.validate_document!(document, "a" * 64) }
  end

  it "cross-binds the live producer sidecar, crop and mask before allocation" do
    document = hy_text_probe_document
    qwen = JSON.parse({
      capture: {hidden_states: {dtype: "bfloat16", shape: [27, 1, 1388, 3584],
                                storage: {offset_bytes: 0, sidecar: "bf16"}}},
      sidecars:        {bf16: {sha256: "b" * 64}},
      hy_nomas_prompt: {crop_start: 108, cropped_attention_mask: document["mask"]},
    }.to_json)
    HyNomasTextProbe.validate_qwen_binding!(document, qwen)
    changed = JSON.parse(qwen.to_json)
    changed["sidecars"]["bf16"].as_h["sha256"] = JSON::Any.new("c" * 64)
    expect_raises(ArgumentError, /binding/) { HyNomasTextProbe.validate_qwen_binding!(document, changed) }
    changed = JSON.parse(qwen.to_json)
    changed["hy_nomas_prompt"].as_h["crop_start"] = JSON::Any.new(107_i64)
    expect_raises(ArgumentError, /binding/) { HyNomasTextProbe.validate_qwen_binding!(document, changed) }
    changed = JSON.parse(qwen.to_json)
    changed["hy_nomas_prompt"]["cropped_attention_mask"].as_a[22] = JSON::Any.new(0_i64)
    expect_raises(ArgumentError, /binding/) { HyNomasTextProbe.validate_qwen_binding!(document, changed) }
  end

  it "does not promote the F32 control to a different precision policy" do
    ["compute", "attention", "resident_weights"].each do |field|
      document = hy_text_probe_document
      document["precision"].as_h[field] = JSON::Any.new("unsupported")
      expect_raises(ArgumentError, /precision/) { HyNomasTextProbe.validate_document!(document, "a" * 64) }
    end
  end

  it "rejects aliases, gaps, short sidecars, wrong dtypes and oversized shapes" do
    tensors = JSON.parse({
      a: {dtype: "F32", shape: [2, 3], offset_bytes: 0, nbytes: 24},
      b: {dtype: "F32", shape: [2], offset_bytes: 24, nbytes: 8},
    }.to_json)
    HyNomasTextProbe.validate_tensor_layout!(tensors, 32_i64)
    expect_raises(ArgumentError) { HyNomasTextProbe.validate_tensor_layout!(tensors, 31_i64) }
    [0_i64, 28_i64].each do |offset|
      changed = JSON.parse(tensors.to_json)
      changed["b"].as_h["offset_bytes"] = JSON::Any.new(offset)
      expect_raises(ArgumentError) { HyNomasTextProbe.validate_tensor_layout!(changed, 32_i64) }
    end
    tensors["a"].as_h["dtype"] = JSON::Any.new("BF16")
    expect_raises(ArgumentError) { HyNomasTextProbe.validate_tensor_layout!(tensors, 32_i64) }
    oversized = JSON.parse({a: {dtype: "F32", shape: [2147483647, 2147483647], offset_bytes: 0, nbytes: 4}}.to_json)
    expect_raises(ArgumentError) { HyNomasTextProbe.validate_tensor_layout!(oversized, 4_i64) }
  end

  it "rejects one changed value and every nonfinite value at the unchanged gate" do
    reference = [0.0_f32, 1.0_f32, -1.0_f32]
    result = HyNomasTextProbe.compare_values(reference, reference)
    result[:failures].should eq(0)
    result[:raw_bit_differences].should eq(0)
    result[:first_mismatch].should be_nil
    changed = reference.dup
    changed[0] = 3e-5_f32
    changed_result = HyNomasTextProbe.compare_values(changed, reference)
    changed_result[:failures].should eq(1)
    changed_result[:first_mismatch].should eq(0)
    changed_result[:first_actual].should eq(3e-5_f32)
    changed_result[:first_expected].should eq(0.0_f32)
    [Float32::NAN, Float32::INFINITY, -Float32::INFINITY].each do |bad|
      changed[0] = bad
      HyNomasTextProbe.compare_values(changed, reference)[:failures].should eq(1)
    end
  end
end
