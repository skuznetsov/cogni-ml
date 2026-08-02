require "json"
require "digest/sha256"
require "../../../src/ml/three_d/trellis2/image_preprocess"
require "../../spec_helper"

private class FixedTrellis2BackgroundRemover < ML::ThreeD::Trellis2::BackgroundRemoverCPU
  getter calls : Int32 = 0

  def remove(image : ML::ThreeD::Trellis2::RGBImage) : ML::ThreeD::Trellis2::RGBAImage
    @calls += 1
    image.width.should eq(5)
    image.height.should eq(5)
    image.pixel(2, 2).should eq({90_u8, 80_u8, 70_u8})

    pixels = Bytes.new(5 * 5 * 4, 0_u8)
    1.upto(3) do |y|
      1.upto(3) do |x|
        offset = (y * 5 + x) * 4
        pixels[offset] = 90_u8
        pixels[offset + 1] = 80_u8
        pixels[offset + 2] = 70_u8
        pixels[offset + 3] = 255_u8
      end
    end
    ML::ThreeD::Trellis2::RGBAImage.new(5, 5, pixels)
  end
end

private def trellis2_fixture_bytes(payload : JSON::Any) : Bytes
  values = payload.as_a
  Bytes.new(values.size) { |index| values[index].as_i.to_u8 }
end

describe ML::ThreeD::Trellis2::Trellis2ImagePreprocessorCPU do
  it "matches pinned NumPy and Pillow execution for crop, padding, and premultiply" do
    fixture_path = File.join(
      __DIR__,
      "../../fixtures/trellis2/dino_v3_source_preprocess_cpu_v1.json"
    )
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/dino-v3-source-preprocess-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["provenance"]["weights"].as_s.should eq("none")
    fixture["provenance"]["device"].as_s.should eq("cpu")

    fixture["cases"].as_a.each do |test_case|
      name = test_case["name"].as_s
      input = test_case["input"]
      expected = test_case["expected"]
      result = ML::ThreeD::Trellis2::Trellis2ImagePreprocessorCPU.new.preprocess(
        ML::ThreeD::Trellis2::RGBAImage.new(
          input["width"].as_i.to_i32,
          input["height"].as_i.to_i32,
          trellis2_fixture_bytes(input["rgba"])
        )
      )
      rounded_crop = expected["rounded_crop"].as_a.map(&.as_i.to_i32)

      result.source_revision.should eq(
        "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
      ), "#{name} revision"
      result.background_removed.should be_false, "#{name} background path"
      result.crop.should eq(
        ML::ThreeD::Trellis2::CropBox.new(
          rounded_crop[0],
          rounded_crop[1],
          rounded_crop[2],
          rounded_crop[3]
        )
      ), "#{name} crop"
      result.image.width.should eq(expected["width"].as_i), "#{name} width"
      result.image.height.should eq(expected["height"].as_i), "#{name} height"
      result.image.pixels.should eq(
        trellis2_fixture_bytes(expected["rgb"])
      ), "#{name} pixels"
    end

    sweep = fixture["premultiply_sweep"]
    sweep["recipe"].as_s.should eq(
      "rgba(x,y) = [x, 255-x, (x+y)%256, y] on 256x256"
    )
    sweep_pixels = Bytes.new(256 * 256 * 4, 0_u8)
    256.times do |y|
      256.times do |x|
        offset = (y * 256 + x) * 4
        sweep_pixels[offset] = x.to_u8
        sweep_pixels[offset + 1] = (255 - x).to_u8
        sweep_pixels[offset + 2] = ((x + y) % 256).to_u8
        sweep_pixels[offset + 3] = y.to_u8
      end
    end
    input_sha256 = Digest::SHA256.hexdigest(sweep_pixels)
    sweep_result = ML::ThreeD::Trellis2::Trellis2ImagePreprocessorCPU.new.preprocess(
      ML::ThreeD::Trellis2::RGBAImage.new(256, 256, sweep_pixels)
    )
    rounded_sweep_crop = sweep["rounded_crop"].as_a.map(&.as_i.to_i32)
    sweep_result.crop.should eq(
      ML::ThreeD::Trellis2::CropBox.new(
        rounded_sweep_crop[0],
        rounded_sweep_crop[1],
        rounded_sweep_crop[2],
        rounded_sweep_crop[3]
      )
    )
    sweep_result.image.width.should eq(sweep["width"].as_i)
    sweep_result.image.height.should eq(sweep["height"].as_i)
    Digest::SHA256.hexdigest(sweep_result.image.pixels).should eq(
      sweep["rgb_sha256"].as_s
    )
    Digest::SHA256.hexdigest(sweep_pixels).should eq(input_sha256), "source immutability"
  end

  it "routes fully opaque inputs through the explicit background-removal boundary" do
    pixels = Bytes.new(5 * 5 * 4, 255_u8)
    offset = (2 * 5 + 2) * 4
    pixels[offset] = 90_u8
    pixels[offset + 1] = 80_u8
    pixels[offset + 2] = 70_u8
    remover = FixedTrellis2BackgroundRemover.new

    result = ML::ThreeD::Trellis2::Trellis2ImagePreprocessorCPU.new(remover).preprocess(
      ML::ThreeD::Trellis2::RGBAImage.new(5, 5, pixels)
    )

    remover.calls.should eq(1)
    result.background_removed.should be_true
    result.crop.should eq(ML::ThreeD::Trellis2::CropBox.new(1, 1, 3, 3))
    result.image.pixels.to_a.should eq([
      90_u8, 80_u8, 70_u8, 90_u8, 80_u8, 70_u8,
      90_u8, 80_u8, 70_u8, 90_u8, 80_u8, 70_u8,
    ])
  end

  it "rejects absent, threshold-only, and degenerate foregrounds with typed errors" do
    preprocessor = ML::ThreeD::Trellis2::Trellis2ImagePreprocessorCPU.new

    expect_raises(ML::ThreeD::Trellis2::BackgroundRemovalRequiredError) do
      preprocessor.preprocess(
        ML::ThreeD::Trellis2::RGBAImage.new(2, 2, Bytes.new(2 * 2 * 4, 255_u8))
      )
    end

    threshold_pixels = Bytes.new(2 * 2 * 4, 0_u8)
    threshold_pixels[3] = 204_u8
    expect_raises(ML::ThreeD::Trellis2::EmptyForegroundError) do
      preprocessor.preprocess(
        ML::ThreeD::Trellis2::RGBAImage.new(2, 2, threshold_pixels)
      )
    end

    one_pixel = Bytes.new(3 * 3 * 4, 0_u8)
    one_pixel[(1 * 3 + 1) * 4 + 3] = 255_u8
    expect_raises(ML::ThreeD::Trellis2::DegenerateForegroundError) do
      preprocessor.preprocess(
        ML::ThreeD::Trellis2::RGBAImage.new(3, 3, one_pixel)
      )
    end
  end

  it "rejects unimplemented oversize resampling and malformed image buffers before allocation" do
    expect_raises(ArgumentError, /pixel buffer/) do
      ML::ThreeD::Trellis2::RGBAImage.new(2, 2, Bytes.new(15, 0_u8))
    end
    expect_raises(ArgumentError, /positive/) do
      ML::ThreeD::Trellis2::RGBAImage.new(0, 2, Bytes.empty)
    end

    pixels = Bytes.new(1025 * 2 * 4, 0_u8)
    pixels[3] = 255_u8
    expect_raises(ML::ThreeD::Trellis2::ImageResamplingRequiredError) do
      ML::ThreeD::Trellis2::Trellis2ImagePreprocessorCPU.new.preprocess(
        ML::ThreeD::Trellis2::RGBAImage.new(1025, 2, pixels)
      )
    end
  end
end
