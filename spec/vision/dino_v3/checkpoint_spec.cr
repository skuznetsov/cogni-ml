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

private def mutate_dino_v3_checkpoint_manifest(& : Hash(String, JSON::Any) ->)
  root = JSON.parse(dino_v3_checkpoint_manifest_fixture).as_h
  yield root
  root.to_json
end

describe ML::Vision::DinoV3::CheckpointManifest do
  it "binds the gated config and weight artifacts to the pinned DINOv3 revision" do
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
end
