require "json"
require "digest/sha256"
require "../../../src/ml/three_d/trellis2/image_conditioner"
require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def trellis2_dino_v3_runtime_adapter : ML::Vision::DinoV3::RuntimeAdapter
  certificate_path = File.join(
    __DIR__,
    "../../fixtures/trellis2/dino_v3_config_certificate_v1.json"
  )
  certificate = ML::Vision::DinoV3::ConfigCertificate.parse(
    File.read(certificate_path),
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
  fixture = JSON.parse(
    File.read(File.join(__DIR__, "../../fixtures/trellis2/dino_v3_embeddings_cpu_v1.json"))
  )
  provenance = fixture["provenance"]
  compatibility = fixture["compatibility"]
  ML::Vision::DinoV3::RuntimeAdapter.new(
    certificate,
    trellis_source_revision: provenance["commit"].as_s,
    transformers_version: provenance["transformers_version"].as_s,
    transformers_modeling_sha256: provenance["transformers_modeling_sha256"].as_s,
    transformers_config_sha256: provenance["transformers_config_sha256"].as_s,
    source_runtime_path: compatibility["pinned_upstream_layer_path"].as_s,
    instance_runtime_path: compatibility["transformers_5_8_1_layer_path"].as_s,
    source_path_available: compatibility["pinned_upstream_path_available_on_instance"].as_bool,
    instance_path_available: compatibility["transformers_5_8_1_path_available_on_instance"].as_bool
  )
end

private def trellis2_conditioner_recipe(width : Int32, height : Int32) : Bytes
  pixels = Bytes.new(width * height * 3, 0_u8)
  height.times do |y|
    width.times do |x|
      offset = (y * width + x) * 3
      pixels[offset] = ((17 * x + 13 * y + 3) % 256).to_u8
      pixels[offset + 1] = ((5 * x + 29 * y + 7) % 256).to_u8
      pixels[offset + 2] = ((x * x + 3 * y + 11) % 256).to_u8
    end
  end
  pixels
end

private def trellis2_f32le_sha256(values : Array(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

describe ML::ThreeD::Trellis2::DinoV3ImageConditionerCPU do
  it "matches pinned Pillow resize and PyTorch normalization for 512 and 1024" do
    fixture_path = File.join(
      __DIR__,
      "../../fixtures/trellis2/dino_v3_image_conditioner_cpu_v1.json"
    )
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/dino-v3-image-conditioner-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["provenance"]["weights"].as_s.should eq("none")
    fixture["provenance"]["device"].as_s.should eq("cpu")
    fixture["provenance"]["pillow_license"].as_s.should eq("MIT-CMU")
    fixture["provenance"]["pillow_version"].as_s.should eq("12.2.0")
    fixture["provenance"]["pillow_resample"].as_s.should contain(
      "/Pillow/blob/12.2.0/"
    )
    runtime_adapter = trellis2_dino_v3_runtime_adapter
    conditioner = ML::ThreeD::Trellis2::DinoV3ImageConditionerCPU.new(runtime_adapter)

    fixture["cases"].as_a.each do |test_case|
      name = test_case["name"].as_s
      input = test_case["input"]
      target = test_case["target"].as_i.to_i32
      width = input["width"].as_i.to_i32
      height = input["height"].as_i.to_i32
      pixels = trellis2_conditioner_recipe(width, height)
      source_sha256 = Digest::SHA256.hexdigest(pixels)
      source_sha256.should eq(input["rgb_sha256"].as_s), "#{name} source recipe"

      result = conditioner.prepare(
        ML::ThreeD::Trellis2::RGBImage.new(width, height, pixels),
        target
      )
      expected = test_case["expected"]
      result.source_revision.should eq(
        "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
      )
      result.runtime_adapter.object_id.should eq(runtime_adapter.object_id)
      result.pillow_revision.should eq(fixture["provenance"]["pillow_version"].as_s)
      result.resized.width.should eq(target)
      result.resized.height.should eq(target)
      Digest::SHA256.hexdigest(result.resized.pixels).should eq(
        expected["resized_rgb_sha256"].as_s
      ), "#{name} resized RGB"
      result.tensor.shape.to_a.should eq(
        expected["normalized_shape"].as_a.map(&.as_i.to_i32)
      ), "#{name} normalized shape"
      tensor_values = result.tensor.to_a
      trellis2_f32le_sha256(tensor_values).should eq(
        expected["normalized_f32le_sha256"].as_s
      ), "#{name} normalized bytes"
      expected["probes"].as_a.each do |probe|
        index = probe["index"].as_a.map(&.as_i.to_i32)
        offset = ((index[0] * 3 + index[1]) * target + index[2]) * target + index[3]
        tensor_values[offset].should be_close(
          probe["value"].as_f.to_f32,
          1.0e-6_f32
        ), "#{name} probe #{index}"
      end
      Digest::SHA256.hexdigest(pixels).should eq(source_sha256), "#{name} source immutability"
    end
  end

  it "rejects unsupported target, geometry, and source size before resampling" do
    conditioner = ML::ThreeD::Trellis2::DinoV3ImageConditionerCPU.new(
      trellis2_dino_v3_runtime_adapter
    )
    rectangle = ML::ThreeD::Trellis2::RGBImage.new(
      3,
      2,
      trellis2_conditioner_recipe(3, 2)
    )
    expect_raises(ML::ThreeD::Trellis2::DinoV3ImageConditionerError, /square/) do
      conditioner.prepare(rectangle, 512)
    end

    square = ML::ThreeD::Trellis2::RGBImage.new(
      3,
      3,
      trellis2_conditioner_recipe(3, 3)
    )
    expect_raises(ML::ThreeD::Trellis2::DinoV3ImageConditionerError, /512 or 1024/) do
      conditioner.prepare(square, 256)
    end

    oversize = ML::ThreeD::Trellis2::RGBImage.new(
      1025,
      1025,
      trellis2_conditioner_recipe(1025, 1025)
    )
    expect_raises(ML::ThreeD::Trellis2::DinoV3ImageConditionerError, /1024/) do
      conditioner.prepare(oversize, 512)
    end
  end
end
