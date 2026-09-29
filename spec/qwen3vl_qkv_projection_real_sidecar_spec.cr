require "./spec_helper"
require "digest/sha256"
require "json"
require "../src/ml/gguf/qwen3vl_text_weights"
require "../src/ml/gguf/qwen3vl_text_block"

private QWEN3VL_QKV_INPUT_SHA256    = "39f0dbf28df1975ddb924698b8fb2adc639a56ad440aa691b5d793ebf6573bb5"
private QWEN3VL_QKV_OFFICIAL_SHA256 = {
  "q" => "d770317fb1bfdf92cca3ab3e0ece3be4e0dbebd82a88da27e7d3f16145dbae93",
  "k" => "773ef1a1e9600ecc8ccb6d4fb46bd93399db03df35372a750fc9ff174aa350b1",
  "v" => "fec0f93ff097003cceef0606d13e590dbc5f40b279d8c98838e23b87b198394e",
}

private def qwen3vl_qkv_sidecar_sha256(bytes : Bytes) : String
  Digest::SHA256.hexdigest(bytes)
end

private def qwen3vl_qkv_read_bf16(path : String, rows : Int32, columns : Int32) : Array(Float32)
  bytes = File.read(path).to_slice
  raise "BF16 sidecar shape mismatch: #{path}" unless bytes.size == rows * columns * 2
  Array(Float32).new(rows * columns) do |index|
    offset = index * 2
    bits = bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
    (bits << 16).unsafe_as(Float32)
  end
end

private def qwen3vl_qkv_bf16_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    bits = value.unsafe_as(UInt32) >> 16
    bytes[index * 2] = (bits & 0xff_u32).to_u8
    bytes[index * 2 + 1] = ((bits >> 8) & 0xff_u32).to_u8
  end
  bytes
end

private def qwen3vl_qkv_mismatch_count(actual : Bytes, expected : Bytes) : Int32
  raise "BF16 sidecar byte lengths differ" unless actual.size == expected.size
  mismatches = 0_i32
  (actual.size // 2).times do |index|
    offset = index * 2
    mismatches += 1 if actual[offset] != expected[offset] || actual[offset + 1] != expected[offset + 1]
  end
  mismatches
end

if fixture_dir = ENV["QWEN3VL_QKV_PROJECTION_FIXTURE_DIR"]?
  if encoder_dir = ENV["QWEN3VL_TEXT_ENCODER_DIR"]?
    describe "optional real Qwen3-VL Q/K/V projection arithmetic" do
      it "matches official BF16 layer-zero projection sidecars from the identical normalized input" do
        trace_path = File.join(fixture_dir, "trace.json")
        trace = JSON.parse(File.read(trace_path))
        trace["fixture"]["revision_sha"].as_s.should eq(
          ML::GGUF::Qwen3VLTextWeights::EXPECTED_MODEL_REVISION
        )
        trace["runtime"]["device"].as_s.should eq("cpu")
        trace["runtime"]["model_dtype"].as_s.should eq("torch.bfloat16")

        official_stages = trace["official_stage_sidecars"]
        input_meta = official_stages["layers.0.input_layernorm"]
        input_meta["dtype"].as_s.should eq("bfloat16-le")
        input_meta["shape"].as_a.map(&.as_i).should eq([244, 4096])
        input_bytes = File.read(File.join(fixture_dir, input_meta["filename"].as_s)).to_slice.dup
        qwen3vl_qkv_sidecar_sha256(input_bytes).should eq(QWEN3VL_QKV_INPUT_SHA256)
        qwen3vl_qkv_sidecar_sha256(input_bytes).should eq(input_meta["sha256"].as_s)
        input = qwen3vl_qkv_read_bf16(
          File.join(fixture_dir, input_meta["filename"].as_s), 244, 4096
        )

        weights = ML::GGUF::Qwen3VLTextWeights.from_directory(encoder_dir)
        begin
          layer = weights.block_weights(0)
          config = ML::GGUF::Qwen3VLTextBlockConfig.new(
            hidden_dim: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 1,
            projection_backend: ML::GGUF::Qwen3VLTextBlockConfig::ProjectionBackend::Scalar,
            qkv_projection_backend: ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend::Torch26Arm64Bf16,
          )
          config.projection_backend.should eq(
            ML::GGUF::Qwen3VLTextBlockConfig::ProjectionBackend::Scalar
          )
          qkv = ML::GGUF::Qwen3VLTextBlock.project_qkv(input, 244, layer, config)
          projections = {
            {"q", layer.q_proj, 4096_i32, "196ea55e2ad0ab4967155588a5b549beade1ddf65d793931b17f74edc73993e4", qkv[:q]},
            {"k", layer.k_proj, 1024_i32, "9c56b3b01d2e88690fdb1bc424ae4e967462af4a7098982a2f76751c143cf149", qkv[:k]},
            {"v", layer.v_proj, 1024_i32, "935e78eb863f7a986ab5db1de2602e447f34305a2f614221e487b61d7ab7b61c", qkv[:v]},
          }
          observed = {} of String => Int32
          projections.each do |name, weight, output_dim, weight_sha, values|
            qwen3vl_qkv_sidecar_sha256(qwen3vl_qkv_bf16_bytes(weight)).should eq(weight_sha)
            stage_meta = official_stages["layers.0.self_attn.#{name}_proj"]
            stage_meta["dtype"].as_s.should eq("bfloat16-le")
            stage_meta["shape"].as_a.map(&.as_i).should eq([244, output_dim.to_i])
            expected = File.read(File.join(fixture_dir, stage_meta["filename"].as_s)).to_slice.dup
            qwen3vl_qkv_sidecar_sha256(expected).should eq(QWEN3VL_QKV_OFFICIAL_SHA256[name])
            qwen3vl_qkv_sidecar_sha256(expected).should eq(stage_meta["sha256"].as_s)
            actual = qwen3vl_qkv_bf16_bytes(values)
            observed[name] = qwen3vl_qkv_mismatch_count(actual, expected)
          end

          seed_meta = official_stages["layers.0.self_attn.q_proj"]
          seed = File.read(File.join(fixture_dir, seed_meta["filename"].as_s)).to_slice.dup
          seed[0] = seed[0] ^ 1_u8
          qwen3vl_qkv_mismatch_count(
            seed, File.read(File.join(fixture_dir, seed_meta["filename"].as_s)).to_slice
          ).should eq(1)

          observed.should eq({"q" => 0, "k" => 0, "v" => 0}),
            "Q/K/V BF16 mismatches=#{observed}"
        ensure
          weights.close
        end
      end
    end
  end
end
