require "json"

require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_payload_certificate : ML::Vision::DinoV3::ConfigCertificate
  source = File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json")
  )
  ML::Vision::DinoV3::ConfigCertificate.parse(
    source,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

private def dino_v3_payload_manifest : ML::Vision::DinoV3::CheckpointManifest
  ML::Vision::DinoV3::CheckpointManifest.parse(
    File.read(
      File.join(__DIR__, "../../fixtures/trellis2/dino_v3_checkpoint_manifest_v1.json")
    ),
    certificate: dino_v3_payload_certificate
  )
end

private def unique_dino_v3_payload_checkpoint_path : Tuple(String, String)
  root = File.tempname("pic3d-dino-v3-payload")
  Dir.mkdir(root)
  {root, File.join(root, "model.safetensors")}
end

private def write_dino_v3_payload_checkpoint(path : String) : Nil
  header = %({"synthetic.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})
  File.open(path, "wb") do |file|
    file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    file.write(header.to_slice)
    file.write(Bytes[0, 0, 128, 63])
    file.truncate(ML::Vision::DinoV3::CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH)
  end
end

describe ML::Vision::DinoV3::CheckpointPayloadReader do
  it "reads exactly one registered payload range" do
    root, path = unique_dino_v3_payload_checkpoint_path
    begin
      write_dino_v3_payload_checkpoint(path)
      inventory = ML::Vision::DinoV3::CheckpointInventory.load(
        path,
        manifest: dino_v3_payload_manifest
      )
      reader = ML::Vision::DinoV3::CheckpointPayloadReader.new(inventory)

      bytes = reader.read_tensor("synthetic.weight")

      bytes.should eq(Bytes[0, 0, 128, 63])
      IO::ByteFormat::LittleEndian.decode(Float32, bytes).should eq(1.0_f32)
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "rejects unknown names and reads above the configured byte budget" do
    root, path = unique_dino_v3_payload_checkpoint_path
    begin
      write_dino_v3_payload_checkpoint(path)
      inventory = ML::Vision::DinoV3::CheckpointInventory.load(
        path,
        manifest: dino_v3_payload_manifest
      )
      reader = ML::Vision::DinoV3::CheckpointPayloadReader.new(
        inventory,
        max_read_bytes: 4_i64
      )

      expect_raises(
        ML::Vision::DinoV3::CheckpointPayloadError,
        /unknown DINOv3 checkpoint tensor/
      ) do
        reader.read_tensor("missing.weight")
      end
      expect_raises(
        ML::Vision::DinoV3::CheckpointPayloadError,
        /byte budget/
      ) do
        reader.read_tensor("synthetic.weight", max_bytes: 3_i64)
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "rejects a descriptor not owned by the inspected inventory and stale files" do
    root, path = unique_dino_v3_payload_checkpoint_path
    begin
      write_dino_v3_payload_checkpoint(path)
      inventory = ML::Vision::DinoV3::CheckpointInventory.load(
        path,
        manifest: dino_v3_payload_manifest
      )
      reader = ML::Vision::DinoV3::CheckpointPayloadReader.new(inventory)
      spoofed = ML::Vision::DinoV3::CheckpointTensor.new(
        "synthetic.weight",
        "F32",
        [1_i64],
        4_i64,
        8_i64
      )

      expect_raises(
        ML::Vision::DinoV3::CheckpointPayloadError,
        /does not belong/
      ) do
        reader.read_tensor(spoofed)
      end

      File.open(path, "r+b") { |file| file.truncate(8) }
      expect_raises(
        ML::Vision::DinoV3::CheckpointPayloadError,
        /changed after inventory/
      ) do
        reader.read_tensor("synthetic.weight")
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end
end
