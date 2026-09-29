require "./spec_helper"
require "digest/sha256"
require "json"
require "../src/ml/gguf/qwen3vl_text_weights"
require "../src/ml/gguf/qwen3vl_text_block"

private QWEN3VL_QNORM_TRACE_SHA256 = "d856c64eafca40294e6de8cda3261ab0bb3766bf2c8edc23672de941ca3b8fff"
private QWEN3VL_QNORM_INPUT_SHA256 = {
  "q_proj" => "d770317fb1bfdf92cca3ab3e0ece3be4e0dbebd82a88da27e7d3f16145dbae93",
  "k_proj" => "773ef1a1e9600ecc8ccb6d4fb46bd93399db03df35372a750fc9ff174aa350b1",
}
private QWEN3VL_QNORM_OUTPUT_SHA256 = {
  "q_norm" => "95f780c845178dc874aa86667b907f4915a18ef970ce8188346031d6eb779373",
  "k_norm" => "6c138005385095ae62161b20d44bbcabc87b6a978d7c8af26562ec2c321eb01d",
}
private QWEN3VL_QNORM_WEIGHT_SHA256 = {
  "q_norm" => "67f20b455b84de4912c8fbbc8cc10a86a9d6d1db8d9bb5e374ad9ca14f5f2087",
  "k_norm" => "a6a4d148e39c4703670c187ef3f7e3c3f1a6d738c770a29df3537ff3f891937a",
}

private def qwen3vl_qnorm_fixture_bytes(path : String, element_count : Int32) : Bytes
  bytes = File.read(path).to_slice.dup
  raise "BF16 sidecar shape mismatch: #{path}" unless bytes.size == element_count * 2
  bytes
end

private def qwen3vl_qnorm_fixture_values(bytes : Bytes) : Array(Float32)
  raise "BF16 sidecar byte count must be even" unless bytes.size.even?
  Array(Float32).new(bytes.size // 2) do |index|
    offset = index * 2
    bits = bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
    (bits << 16).unsafe_as(Float32)
  end
end

private def qwen3vl_qnorm_fixture_bf16_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    bits = value.unsafe_as(UInt32) >> 16
    bytes[index * 2] = (bits & 0xff_u32).to_u8
    bytes[index * 2 + 1] = ((bits >> 8) & 0xff_u32).to_u8
  end
  bytes
end

private def qwen3vl_qnorm_fixture_mismatches(actual : Bytes, expected : Bytes) : Int32
  raise "BF16 sidecar byte lengths differ" unless actual.size == expected.size
  mismatch_count = 0_i32
  (actual.size // 2).times do |index|
    offset = index * 2
    mismatch_count += 1 if actual[offset] != expected[offset] || actual[offset + 1] != expected[offset + 1]
  end
  mismatch_count
end

private def qwen3vl_qnorm_shape(metadata : JSON::Any, expected : Array(Int64)) : Nil
  metadata["dtype"].as_s.should eq("bfloat16-le")
  metadata["shape"].as_a.map(&.as_i).should eq(expected)
end

if fixture_dir = ENV["QWEN3VL_QNORM_FIXTURE_DIR"]?
  if encoder_dir = ENV["QWEN3VL_QNORM_TEXT_ENCODER_DIR"]?
    describe "Qwen3-VL production Q/K RMSNorm on pinned projection sidecars" do
      it "matches complete official norm sidecars and preserves generic-profile control" do
        trace_path = File.join(fixture_dir, "trace.json")
        Digest::SHA256.hexdigest(File.read(trace_path)).should eq(QWEN3VL_QNORM_TRACE_SHA256)
        trace = JSON.parse(File.read(trace_path))
        trace["fixture"]["revision_sha"].as_s.should eq(
          ML::GGUF::Qwen3VLTextWeights::EXPECTED_MODEL_REVISION
        )
        trace["runtime"]["device"].as_s.should eq("cpu")
        trace["runtime"]["model_dtype"].as_s.should eq("torch.bfloat16")
        official = trace["official_stage_sidecars"]

        q_proj_meta = official["layers.0.self_attn.q_proj"]
        k_proj_meta = official["layers.0.self_attn.k_proj"]
        q_norm_meta = official["layers.0.self_attn.q_norm"]
        k_norm_meta = official["layers.0.self_attn.k_norm"]
        qwen3vl_qnorm_shape(q_proj_meta, [244_i64, 4096_i64])
        qwen3vl_qnorm_shape(k_proj_meta, [244_i64, 1024_i64])
        qwen3vl_qnorm_shape(q_norm_meta, [244_i64, 4096_i64])
        qwen3vl_qnorm_shape(k_norm_meta, [244_i64, 1024_i64])

        q_proj_bytes = qwen3vl_qnorm_fixture_bytes(
          File.join(fixture_dir, q_proj_meta["filename"].as_s), 244 * 4096
        )
        k_proj_bytes = qwen3vl_qnorm_fixture_bytes(
          File.join(fixture_dir, k_proj_meta["filename"].as_s), 244 * 1024
        )
        q_norm_expected = qwen3vl_qnorm_fixture_bytes(
          File.join(fixture_dir, q_norm_meta["filename"].as_s), 244 * 4096
        )
        k_norm_expected = qwen3vl_qnorm_fixture_bytes(
          File.join(fixture_dir, k_norm_meta["filename"].as_s), 244 * 1024
        )

        Digest::SHA256.hexdigest(q_proj_bytes).should eq(QWEN3VL_QNORM_INPUT_SHA256["q_proj"])
        Digest::SHA256.hexdigest(k_proj_bytes).should eq(QWEN3VL_QNORM_INPUT_SHA256["k_proj"])
        Digest::SHA256.hexdigest(q_norm_expected).should eq(QWEN3VL_QNORM_OUTPUT_SHA256["q_norm"])
        Digest::SHA256.hexdigest(k_norm_expected).should eq(QWEN3VL_QNORM_OUTPUT_SHA256["k_norm"])
        Digest::SHA256.hexdigest(q_proj_bytes).should eq(q_proj_meta["sha256"].as_s)
        Digest::SHA256.hexdigest(k_proj_bytes).should eq(k_proj_meta["sha256"].as_s)
        Digest::SHA256.hexdigest(q_norm_expected).should eq(q_norm_meta["sha256"].as_s)
        Digest::SHA256.hexdigest(k_norm_expected).should eq(k_norm_meta["sha256"].as_s)

        weights = ML::GGUF::Qwen3VLTextWeights.from_directory(encoder_dir)
        begin
          q_norm_weight = weights.layer0_tensor_f32(
            "model.language_model.layers.0.self_attn.q_norm.weight"
          )
          k_norm_weight = weights.layer0_tensor_f32(
            "model.language_model.layers.0.self_attn.k_norm.weight"
          )
          Digest::SHA256.hexdigest(qwen3vl_qnorm_fixture_bf16_bytes(q_norm_weight)).should eq(
            QWEN3VL_QNORM_WEIGHT_SHA256["q_norm"]
          )
          Digest::SHA256.hexdigest(qwen3vl_qnorm_fixture_bf16_bytes(k_norm_weight)).should eq(
            QWEN3VL_QNORM_WEIGHT_SHA256["k_norm"]
          )

          config = ML::GGUF::Qwen3VLTextBlockConfig.new(
            hidden_dim: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 1,
            qkv_projection_backend: ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend::Torch26Arm64Bf16,
          )
          generic_config = ML::GGUF::Qwen3VLTextBlockConfig.new(
            hidden_dim: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 1,
          )
          profiled = ML::GGUF::Qwen3VLTextBlock.normalize_qk(
            qwen3vl_qnorm_fixture_values(q_proj_bytes),
            qwen3vl_qnorm_fixture_values(k_proj_bytes),
            244,
            q_norm_weight,
            k_norm_weight,
            config,
          )
          generic = ML::GGUF::Qwen3VLTextBlock.normalize_qk(
            qwen3vl_qnorm_fixture_values(q_proj_bytes),
            qwen3vl_qnorm_fixture_values(k_proj_bytes),
            244,
            q_norm_weight,
            k_norm_weight,
            generic_config,
          )
          q_profiled_bytes = qwen3vl_qnorm_fixture_bf16_bytes(profiled[:q])
          k_profiled_bytes = qwen3vl_qnorm_fixture_bf16_bytes(profiled[:k])
          q_generic_bytes = qwen3vl_qnorm_fixture_bf16_bytes(generic[:q])
          k_generic_bytes = qwen3vl_qnorm_fixture_bf16_bytes(generic[:k])
          profile_mismatches = {
            "q_norm" => qwen3vl_qnorm_fixture_mismatches(q_profiled_bytes, q_norm_expected),
            "k_norm" => qwen3vl_qnorm_fixture_mismatches(k_profiled_bytes, k_norm_expected),
          }
          generic_mismatches = {
            "q_norm" => qwen3vl_qnorm_fixture_mismatches(q_generic_bytes, q_norm_expected),
            "k_norm" => qwen3vl_qnorm_fixture_mismatches(k_generic_bytes, k_norm_expected),
          }

          profile_mismatches.should eq({"q_norm" => 0, "k_norm" => 0}),
            "profiled complete-sidecar BF16 mismatches=#{profile_mismatches}"
          generic_mismatches.should eq({"q_norm" => 5, "k_norm" => 0}),
            "generic complete-sidecar BF16 mismatches=#{generic_mismatches}"
        ensure
          weights.close
        end
      end
    end
  end
end
