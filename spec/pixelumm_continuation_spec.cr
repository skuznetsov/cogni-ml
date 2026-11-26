require "spec"
require "../src/ml/video/pixelumm_continuation"

alias PixelUMMContinuation = ML::Video::PixelUMMContinuation

describe PixelUMMContinuation::Config do
  it "derives joined decoder inputs and current-only display output geometry" do
    config = PixelUMMContinuation::Config.new(previous_latent_frames: 30)

    config.current_latent_frames.should eq(30)
    config.decoder_config.latent_shape.should eq([1, 32, 35, 9, 16])
    config.decoder_config.noise_shape.should eq([1, 3, 140, 288, 512])
    config.current_config.latent_shape.should eq([1, 32, 30, 9, 16])
    config.current_config.output_shape.should eq([1, 3, 120, 288, 512])
    config.previous_latent_shape.should eq([1, 32, 30, 9, 16])
    config.previous_noise_shape.should eq([1, 3, 120, 288, 512])
    config.current_latent_shape.should eq([1, 32, 30, 9, 16])
    config.current_noise_shape.should eq([1, 3, 120, 288, 512])
    config.named_bytes_by_name.should eq({
      "latent_f32" => 645_120_i64,
      "noise_f32"  => 247_726_080_i64,
      "output_f32" => 212_336_640_i64,
    })
    config.named_bytes.should eq(460_707_840_i64)
  end

  it "admits long previous histories when their tensor extents remain representable" do
    config = PixelUMMContinuation::Config.new(
      previous_latent_frames: 185, current_latent_frames: 10,
      latent_height: 1, latent_width: 16)

    config.previous_latent_shape.should eq([1, 32, 185, 1, 16])
    config.previous_noise_shape.should eq([1, 3, 740, 32, 512])
    config.decoder_config.total_latent_frames.should eq(15)
  end

  it "maps joined latent and noise elements through full NCTHW channel strides" do
    config = PixelUMMContinuation::Config.new(
      previous_latent_frames: 20, current_latent_frames: 10,
      latent_height: 9, latent_width: 16)
    latent_spatial = 9_i64 * 16
    latent_join_channel_stride = 15_i64 * latent_spatial
    latent_previous_channel_stride = 20_i64 * latent_spatial
    latent_current_channel_stride = 10_i64 * latent_spatial

    config.latent_join_source(2_i32 * latent_join_channel_stride.to_i32 + 4 * latent_spatial.to_i32 + 7)
      .should eq({:previous, 2_i64 * latent_previous_channel_stride + 19 * latent_spatial + 7})
    config.latent_join_source(2_i32 * latent_join_channel_stride.to_i32 + 5 * latent_spatial.to_i32 + 7)
      .should eq({:current, 2_i64 * latent_current_channel_stride + 7})

    pixel_spatial = 288_i64 * 512
    noise_join_channel_stride = 60_i64 * pixel_spatial
    noise_previous_channel_stride = 80_i64 * pixel_spatial
    noise_current_channel_stride = 40_i64 * pixel_spatial
    config.noise_join_source(1_i32 * noise_join_channel_stride.to_i32 + 19 * pixel_spatial.to_i32 + 11)
      .should eq({:previous, noise_previous_channel_stride + 79 * pixel_spatial + 11})
    config.noise_join_source(1_i32 * noise_join_channel_stride.to_i32 + 20 * pixel_spatial.to_i32 + 11)
      .should eq({:current, noise_current_channel_stride + 11})
  end

  it "maps cropped output elements to the full decoder display buffer with channel strides" do
    config = PixelUMMContinuation::Config.new(
      previous_latent_frames: 20, current_latent_frames: 10)
    spatial = config.current_config.pixel_height.to_i64 * config.current_config.pixel_width
    current_frames = config.current_config.pixel_frames
    full_frames = config.decoder_config.pixel_frames

    config.output_source_index(2_i32 * current_frames * spatial + 3 * spatial + 17)
      .should eq(2_i64 * full_frames * spatial + 23 * spatial + 17)
    config.output_source_index(config.output_elements.to_i32 - 1)
      .should eq(config.decoder_config.output_elements - 1)
  end

  it "rejects invalid previous and current timeline geometry before allocation" do
    [-1, 0, 4, 6, 19].each do |previous_frames|
      expect_raises(ArgumentError) do
        PixelUMMContinuation::Config.new(previous_latent_frames: previous_frames)
      end
    end

    [5, 11, 31, 176, 180].each do |current_frames|
      expect_raises(ArgumentError) do
        PixelUMMContinuation::Config.new(previous_latent_frames: 20,
          current_latent_frames: current_frames)
      end
    end

    expect_raises(ArgumentError) do
      PixelUMMContinuation::Config.new(previous_latent_frames: 20,
        latent_height: 0, latent_width: 16)
    end
  end

  it "rejects flattened index helpers outside each joined or cropped output" do
    config = PixelUMMContinuation::Config.new(previous_latent_frames: 20,
      current_latent_frames: 10)

    expect_raises(ArgumentError, /out of range/) do
      config.latent_join_source(config.latent_elements.to_i32)
    end
    expect_raises(ArgumentError, /out of range/) do
      config.noise_join_source(config.noise_elements.to_i32)
    end
    expect_raises(ArgumentError, /out of range/) do
      config.output_source_index(config.output_elements.to_i32)
    end
  end
end
