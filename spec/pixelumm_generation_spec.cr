require "spec"
require "../src/ml/video/pixelumm_generation"

describe ML::Video::PixelUMMGeneration::Config do
  it "keeps the fixed 512x288 local-attention window for the default timeline" do
    config = ML::Video::PixelUMMGeneration::Config.new

    config.output_shape.should eq([1, 3, 120, 288, 512])
    config.window_output_shape.should eq([1, 3, 40, 288, 512])
    config.output_elements.should eq(53_084_160_i64)
    config.window_output_elements.should eq(17_694_720_i64)
    config.decoder_config.layout.local_attention?.should be_true
    config.decoder_config.layout.tile_count.should eq(12)
    config.block_input_bytes.should eq(47_185_920_i64)
  end

  it "derives the opt-in 768x416 five-second decoder geometry from latent cells" do
    config = ML::Video::PixelUMMGeneration::Config.new(
      total_latent_frames: 30, latent_height: 13, latent_width: 24)

    config.latent_shape.should eq([1, 32, 30, 13, 24])
    config.noise_shape.should eq([1, 3, 120, 416, 768])
    config.output_shape.should eq(config.noise_shape)
    config.window_output_shape.should eq([1, 3, 40, 416, 768])
    config.decoder_config.height.should eq(26)
    config.decoder_config.width.should eq(48)
    config.decoder_config.tokens.should eq(12_480)
    config.decoder_config.layout.local_attention?.should be_true
    config.decoder_config.layout.tile_count.should eq(24)
    config.block_input_bytes.should eq(102_236_160_i64)
    config.window_starts.should eq([0, 5, 10, 15, 20])
  end

  it "preserves default and candidate first/last flattened stem mappings" do
    {
      ML::Video::PixelUMMGeneration::Config.new,
      ML::Video::PixelUMMGeneration::Config.new(
        total_latent_frames: 30, latent_height: 13, latent_width: 24),
    }.each do |config|
      patch_last = config.token_count * ML::Video::PixelUMMGeneration::EMBED_INPUT_DIM - 1
      condition_last = config.latent_token_count * ML::Video::PixelUMMGeneration::LATENT_CHANNELS - 1
      projection_last_row = config.token_count - 1

      config.patch_source_index(0, 0).should eq(0_i64)
      config.patch_source_index(20, patch_last).should eq(config.output_elements - 1)
      config.condition_source_index(0, 0).should eq(0_i64)
      config.condition_source_index(20, condition_last).should eq(config.latent_elements - 1)
      config.condition_projection_index(projection_last_row,
        ML::Video::PixelUMMGeneration::HIDDEN - 1).should eq(
        config.latent_token_count.to_i64 * ML::Video::PixelUMMGeneration::CONDITION_OUTPUT_DIM - 1)

      expect_raises(ArgumentError, /configured window start/) do
        config.patch_source_index(1, 0)
      end
      expect_raises(ArgumentError, /out of range/) do
        config.patch_source_index(20, patch_last + 1)
      end
      expect_raises(ArgumentError, /coordinate is out of range/) do
        config.condition_projection_index(projection_last_row, ML::Video::PixelUMMGeneration::HIDDEN)
      end
    end
  end

  it "rejects nonpositive latent spatial dimensions before building a plan" do
    expect_raises(ArgumentError, /latent height and width must be positive/) do
      ML::Video::PixelUMMGeneration::Config.new(
        total_latent_frames: 30, latent_height: 0, latent_width: 24)
    end
  end

  it "accounts the exact owned stem and assembly buffers without constructing Metal plans" do
    config = ML::Video::PixelUMMGeneration::Config.new
    stem = config.stem_named_bytes_by_name
    assembly = config.assembly_named_bytes_by_name

    stem.keys.sort.should eq(%w(
      condition_f32 embed_bias_bf16 embed_weight_bf16 mps_weight_stage_f32
      output_bf16 patch_f32 projection_f32 tile_major_to_original cond_bias_bf16
      cond_weight_bf16
    ).sort)
    stem["patch_f32"].should eq(70_778_880_i64)
    stem["projection_f32"].should eq(94_371_840_i64)
    stem["output_bf16"].should eq(47_185_920_i64)
    stem["mps_weight_stage_f32"].should eq(50_331_648_i64)
    stem.values.sum.should eq(289_131_008_i64)

    assembly.should eq({
      "output_f32"         => 212_336_640_i64,
      "blend_weights_bf16" => 40_i64,
    })
    assembly.values.sum.should eq(212_336_680_i64)
  end

  it "keeps one global 60-latent timeline and fixed-size windows" do
    config = ML::Video::PixelUMMGeneration::Config.new(total_latent_frames: 60)

    config.total_latent_frames.should eq(60)
    config.pixel_frames.should eq(240)
    config.latent_shape.should eq([1, 32, 60, 9, 16])
    config.noise_shape.should eq([1, 3, 240, 288, 512])
    config.output_shape.should eq(config.noise_shape)
    config.window_starts.should eq((0..50).step(5).to_a)
    config.window_starts.includes?(25).should be_true # window [25, 35) crosses the 30-latent seam
    config.window_count.should eq(11)
    config.window_output_shape.should eq([1, 3, 40, 288, 512])
    config.window_output_elements.should eq(17_694_720_i64)

    coverage = config.pixel_frame_coverage_counts
    coverage.size.should eq(240)
    coverage.all? { |count| count > 0 && count <= 2 }.should be_true
    coverage.count(1).should eq(40)
    coverage.count(2).should eq(200)

    config.input_named_bytes_by_name.should eq({
      "latent_f32" => 1_105_920_i64,
      "noise_f32"  => 424_673_280_i64,
    })
    config.assembly_named_bytes_by_name["output_f32"].should eq(424_673_280_i64)
    config.stem_named_bytes_by_name.values.sum.should eq(289_131_008_i64)
  end

  it "covers all 720 output frames with the 35 global windows" do
    config = ML::Video::PixelUMMGeneration::Config.new(total_latent_frames: 180)

    config.output_shape.should eq([1, 3, 720, 288, 512])
    config.window_count.should eq(35)
    config.window_starts.first.should eq(0)
    config.window_starts.last.should eq(170)
    [25, 55, 85, 115, 145].each do |seam_start|
      config.window_starts.includes?(seam_start).should be_true
    end

    coverage = config.pixel_frame_coverage_counts
    coverage.size.should eq(720)
    coverage.all? { |count| count == 1 || count == 2 }.should be_true
    coverage.count(1).should eq(40)
    coverage.count(2).should eq(680)
    config.input_named_bytes_by_name.should eq({
      "latent_f32" => 3_317_760_i64,
      "noise_f32"  => 1_274_019_840_i64,
    })
    config.assembly_named_bytes_by_name["output_f32"].should eq(1_274_019_840_i64)
  end

  it "preserves the five-window default and rejects unaligned or oversized timelines" do
    config = ML::Video::PixelUMMGeneration::Config.new
    config.total_latent_frames.should eq(30)
    config.window_starts.should eq([0, 5, 10, 15, 20])
    config.window_start(-1).should be_nil
    config.window_start(config.window_count).should be_nil
    config.output_shape.should eq([1, 3, 120, 288, 512])
    config.noise_shape.should eq([1, 3, 120, 288, 512])
    config.assembly_named_bytes_by_name["output_f32"].should eq(212_336_640_i64)

    minimum = ML::Video::PixelUMMGeneration::Config.new(total_latent_frames: 10)
    minimum.window_starts.should eq([0])
    minimum.pixel_frame_coverage_counts.should eq(Array(Int32).new(40, 1))

    [9, 11, 31, 181].each do |frames|
      expect_raises(ArgumentError, /align to the window stride/) do
        ML::Video::PixelUMMGeneration::Config.new(total_latent_frames: frames)
      end
    end
  end
end
