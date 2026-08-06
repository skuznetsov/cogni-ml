require "json"

require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_checkpoint_certificate : ML::Vision::DinoV3::ConfigCertificate
  source = File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json")
  )
  ML::Vision::DinoV3::ConfigCertificate.parse(
    source,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

private def dino_v3_checkpoint_manifest_fixture : String
  File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_checkpoint_manifest_v1.json")
  )
end

private def dino_v3_checkpoint_source_evidence_fixture : JSON::Any
  JSON.parse(
    File.read(
      File.join(
        __DIR__,
        "../../fixtures/trellis2/dino_v3_checkpoint_source_evidence_v1.json"
      )
    )
  )
end

private def mutate_dino_v3_checkpoint_manifest(& : Hash(String, JSON::Any) ->)
  root = JSON.parse(dino_v3_checkpoint_manifest_fixture).as_h
  yield root
  root.to_json
end

private def write_synthetic_dino_v3_checkpoint(
  path : String,
  *,
  truncate : Bool = true,
  header_json : String? = nil,
)
  header = header_json || JSON.build do |json|
    json.object do
      json.field("__metadata__") do
        json.object do
          json.field("format", "synthetic-test")
        end
      end
      json.field("synthetic.weight") do
        json.object do
          json.field("dtype", "F32")
          json.field("shape") { json.array { json.number(1) } }
          json.field("data_offsets") do
            json.array do
              json.number(0)
              json.number(4)
            end
          end
        end
      end
    end
  end

  File.open(path, "wb") do |file|
    file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    file.write(header.to_slice)
    file.write(Bytes[0, 0, 128, 63])
    file.truncate(ML::Vision::DinoV3::CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH) if truncate
  end
end

private def unique_dino_v3_checkpoint_path : Tuple(String, String)
  root = File.tempname("pic3d-dino-v3")
  Dir.mkdir(root)
  {root, File.join(root, "model.safetensors")}
end

private class SpoofedDinoV3Certificate < ML::Vision::DinoV3::ConfigCertificate
  def source_model : String
    ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL
  end

  def source_revision : String
    ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  end

  def config_sha256 : String
    ML::Vision::DinoV3::ConfigCertificate::PINNED_CONFIG_SHA256
  end

  def access_mode : String
    "spoofed-access"
  end

  def license_name : String
    "spoofed-license"
  end

  def license_link : String
    "https://example.invalid/spoofed-license"
  end
end

describe ML::Vision::DinoV3::CheckpointManifest do
  it "binds declared checkpoint metadata to the pinned DINOv3 revision" do
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      dino_v3_checkpoint_manifest_fixture,
      certificate: dino_v3_checkpoint_certificate
    )

    manifest.schema.should eq(
      "cogni-ml/vision/dino-v3/checkpoint-manifest/v1"
    )
    manifest.model.should eq(
      "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    manifest.revision.should eq(
      "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    )
    manifest.access_mode.should eq("manual")
    manifest.license_name.should eq("dinov3-license")
    manifest.license_link.should eq(
      "https://ai.meta.com/resources/models-and-libraries/dinov3-license"
    )
    manifest.config_path.should eq("config.json")
    manifest.config_byte_length.should eq(745_i64)
    manifest.config_sha256.should eq(
      "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
    )
    manifest.weights_path.should eq("model.safetensors")
    manifest.weights_format.should eq("safetensors")
    manifest.weights_byte_length.should eq(1_212_559_808_i64)
    manifest.weights_sha256.should eq(
      "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
    )

    evidence = dino_v3_checkpoint_source_evidence_fixture
    evidence["schema"].as_s.should eq(
      "cogni-ml/vision/dino-v3/checkpoint-source-evidence/v1"
    )
    evidence["retrieved_on"].as_s.should eq("2026-08-06")
    evidence["source"]["model"].as_s.should eq(manifest.model)
    evidence["source"]["revision"].as_s.should eq(manifest.revision)
    evidence["files"]["model.safetensors"]["byte_length"].as_i64.should eq(
      manifest.weights_byte_length
    )
    evidence["files"]["model.safetensors"]["lfs_sha256"].as_s.should eq(
      manifest.weights_sha256
    )
    manifest.config_url.should eq(
      "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/" \
      "ea8dc2863c51be0a264bab82070e3e8836b02d51/config.json"
    )
    manifest.weights_url.should eq(
      "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/" \
      "ea8dc2863c51be0a264bab82070e3e8836b02d51/model.safetensors"
    )
  end

  it "rejects model, revision, license, and access drift" do
    {
      {"model", "model"},
      {"revision", "revision"},
      {"access_mode", "access mode"},
    }.each do |field, label|
      mutated = mutate_dino_v3_checkpoint_manifest do |root|
        root[field] = JSON::Any.new("spoof")
      end
      expect_raises(
        ML::Vision::DinoV3::CheckpointManifestError,
        /#{label}/
      ) do
        ML::Vision::DinoV3::CheckpointManifest.parse(
          mutated,
          certificate: dino_v3_checkpoint_certificate
        )
      end
    end

    license_drift = mutate_dino_v3_checkpoint_manifest do |root|
      root["license"].as_h["name"] = JSON::Any.new("spoof")
    end
    expect_raises(
      ML::Vision::DinoV3::CheckpointManifestError,
      /license/
    ) do
      ML::Vision::DinoV3::CheckpointManifest.parse(
        license_drift,
        certificate: dino_v3_checkpoint_certificate
      )
    end
  end

  it "rejects config and weight identity drift before any loader can run" do
    mutations = [
      {"config", "sha256", "config digest"},
      {"config", "byte_length", "config byte length"},
      {"weights", "path", "weights path"},
      {"weights", "format", "weights format"},
      {"weights", "byte_length", "weights byte length"},
      {"weights", "sha256", "weights digest"},
    ]

    mutations.each do |section, field, label|
      mutated = mutate_dino_v3_checkpoint_manifest do |root|
        value = if field == "byte_length"
                  JSON::Any.new(1_i64)
                else
                  JSON::Any.new("spoof")
                end
        root[section].as_h[field] = value
      end
      expect_raises(
        ML::Vision::DinoV3::CheckpointManifestError,
        /#{label}/
      ) do
        ML::Vision::DinoV3::CheckpointManifest.parse(
          mutated,
          certificate: dino_v3_checkpoint_certificate
        )
      end
    end
  end

  it "rejects unknown and missing keys instead of widening the contract" do
    unknown = mutate_dino_v3_checkpoint_manifest do |root|
      root["surprise"] = JSON::Any.new(true)
    end
    expect_raises(
      ML::Vision::DinoV3::CheckpointManifestError,
      /unknown checkpoint manifest key/
    ) do
      ML::Vision::DinoV3::CheckpointManifest.parse(
        unknown,
        certificate: dino_v3_checkpoint_certificate
      )
    end

    missing = mutate_dino_v3_checkpoint_manifest do |root|
      root["weights"].as_h.delete("sha256")
    end
    expect_raises(
      ML::Vision::DinoV3::CheckpointManifestError,
      /missing weights key/
    ) do
      ML::Vision::DinoV3::CheckpointManifest.parse(
        missing,
        certificate: dino_v3_checkpoint_certificate
      )
    end
  end

  it "rejects a certificate subtype with spoofed semantic provenance" do
    expect_raises(
      ML::Vision::DinoV3::CheckpointManifestError,
      /pinned DINOv3 config certificate/
    ) do
      ML::Vision::DinoV3::CheckpointManifest.parse(
        dino_v3_checkpoint_manifest_fixture,
        certificate: SpoofedDinoV3Certificate.allocate
      )
    end
  end

  it "loads only a bounded safetensors header from the pinned materialized file" do
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      dino_v3_checkpoint_manifest_fixture,
      certificate: dino_v3_checkpoint_certificate
    )
    root, path = unique_dino_v3_checkpoint_path

    begin
      write_synthetic_dino_v3_checkpoint(path)
      inventory = ML::Vision::DinoV3::CheckpointInventory.load(
        path,
        manifest: manifest
      )

      inventory.file_byte_length.should eq(
        ML::Vision::DinoV3::CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH
      )
      inventory.tensors.size.should eq(1)
      inventory.tensors[0].name.should eq("synthetic.weight")
      inventory.tensors[0].dtype.should eq("F32")
      inventory.tensors[0].shape.should eq([1_i64])
      inventory.tensors[0].data_bytes.should eq(4_i64)
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "rejects materialized files whose header or offsets escape the file" do
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      dino_v3_checkpoint_manifest_fixture,
      certificate: dino_v3_checkpoint_certificate
    )
    root, path = unique_dino_v3_checkpoint_path

    begin
      write_synthetic_dino_v3_checkpoint(path, truncate: false)
      expect_raises(
        ML::Vision::DinoV3::CheckpointInventoryError,
        /file size/
      ) do
        ML::Vision::DinoV3::CheckpointInventory.load(path, manifest: manifest)
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "rejects overlapping tensor ranges before payload access" do
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      dino_v3_checkpoint_manifest_fixture,
      certificate: dino_v3_checkpoint_certificate
    )
    root, path = unique_dino_v3_checkpoint_path
    header = %({"first":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"second":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}})

    begin
      write_synthetic_dino_v3_checkpoint(path, header_json: header)
      expect_raises(
        ML::Vision::DinoV3::CheckpointInventoryError,
        /ranges overlap/
      ) do
        ML::Vision::DinoV3::CheckpointInventory.load(path, manifest: manifest)
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "rejects tensor ranges whose length disagrees with dtype and shape" do
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      dino_v3_checkpoint_manifest_fixture,
      certificate: dino_v3_checkpoint_certificate
    )
    root, path = unique_dino_v3_checkpoint_path
    header = %({"wrong_shape":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})

    begin
      write_synthetic_dino_v3_checkpoint(path, header_json: header)
      expect_raises(
        ML::Vision::DinoV3::CheckpointInventoryError,
        /does not match dtype and shape/
      ) do
        ML::Vision::DinoV3::CheckpointInventory.load(path, manifest: manifest)
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end
end
