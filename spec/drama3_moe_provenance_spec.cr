require "spec"
require "digest/sha256"
require "file_utils"
require "../examples/support/drama3_moe_provenance"

include ML::Audio::Drama3

private def drama3_moe_provenance_fixture(root : String) : Tuple(String, WeightStore, Hash(String, JSON::Any))
  Dir.mkdir_p(root)
  config_path = File.join(root, "config.json")
  index_path = File.join(root, "model.safetensors.index.json")
  File.write(config_path, "{\"model_type\":\"spec\"}")

  # Put the lexically later tensor in the lexically earlier shard. The upstream
  # contract sorts shard first, then tensor name, so this distinguishes the two.
  index = {
    "weight_map" => {
      "backbone.layers.39.mlp.z_tensor" => "a-shard.safetensors",
      "backbone.layers.39.mlp.a_tensor" => "z-shard.safetensors",
    },
  }
  File.write(index_path, index.to_json)

  z_path = File.join(root, "z-raw.bin")
  z_io = IO::Memory.new
  z_io.write_bytes(0x04030201_u32, IO::ByteFormat::LittleEndian)
  z_io.write(Bytes.new(8, 0_u8))
  File.write(z_path, z_io.to_slice)
  z_owner = MappedFile.new(z_path)

  a_path = File.join(root, "a-raw.bin")
  a_io = IO::Memory.new
  a_io.write_bytes(1.5_f32, IO::ByteFormat::LittleEndian)
  a_io.write(Bytes.new(8, 0_u8))
  File.write(a_path, a_io.to_slice)
  a_owner = MappedFile.new(a_path)

  tensors = {
    "backbone.layers.39.mlp.z_tensor" => Tensor.new([1], "U32", z_owner.slice(0_i64, 4_i64), z_owner),
    "backbone.layers.39.mlp.a_tensor" => Tensor.new([1], "F32", a_owner.slice(0_i64, 4_i64), a_owner),
  }
  store = WeightStore.new(tensors)
  expected = JSON.parse({
    "config.json"                  => Digest::SHA256.hexdigest(File.read(config_path)),
    "model.safetensors.index.json" => Digest::SHA256.hexdigest(File.read(index_path)),
    "target_moe_weights"           => "b46e29a00498381cb9f30ddc76bbfa47d36b1b515b4a5cbc2bbf177716f25a46",
  }.to_json).as_h
  {root, store, expected}
end

describe Drama3MoeProvenance do
  it "matches the upstream raw tensor encoding and shard/name ordering" do
    root = File.tempname("drama3-moe-provenance")
    begin
      model_path, store, expected = drama3_moe_provenance_fixture(root)
      record = Drama3MoeProvenance.verify!(model_path, store, expected)

      record[:target_layer].should eq(39_i32)
      record[:target_moe_weights_sha256].should eq(expected["target_moe_weights"].as_s)
      record[:tensor_count].should eq(2_i32)
      record[:model_path].should eq(File.expand_path(model_path))
      record[:config_sha256].should eq(expected["config.json"].as_s)
      record[:index_sha256].should eq(expected["model.safetensors.index.json"].as_s)
      record[:verified_before_gpu].should be_true
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "rejects a fixture digest that does not match the mapped WeightStore bytes" do
    root = File.tempname("drama3-moe-provenance-mismatch")
    begin
      model_path, store, expected = drama3_moe_provenance_fixture(root)
      bad_expected = JSON.parse({
        "config.json"                  => expected["config.json"].as_s,
        "model.safetensors.index.json" => expected["model.safetensors.index.json"].as_s,
        "target_moe_weights"           => "0" * 64,
      }.to_json).as_h

      expect_raises(ArgumentError, /target MoE weight digest mismatch/) do
        Drama3MoeProvenance.verify!(model_path, store, bad_expected)
      end
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "rejects config bytes that differ from the fixture source hash" do
    root = File.tempname("drama3-moe-provenance-config")
    begin
      model_path, store, expected = drama3_moe_provenance_fixture(root)
      File.write(File.join(root, "config.json"), "{\"model_type\":\"changed\"}")

      expect_raises(ArgumentError, /model config.json digest mismatch/) do
        Drama3MoeProvenance.verify!(model_path, store, expected)
      end
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end
end
