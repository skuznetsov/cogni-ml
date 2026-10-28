require "spec"
require "../src/ml/video/pixelumm_generation"

describe ML::Video::PixelUMMGeneration::Config do
  it "admits only the fixed 512x288, thirty-latent local-attention preview" do
    config = ML::Video::PixelUMMGeneration::Config.new

    config.output_shape.should eq([1, 3, 120, 288, 512])
    config.window_output_shape.should eq([1, 3, 40, 288, 512])
    config.output_elements.should eq(53_084_160_i64)
    config.window_output_elements.should eq(17_694_720_i64)
    config.decoder_config.layout.local_attention?.should be_true
    config.decoder_config.layout.tile_count.should eq(12)
    config.block_input_bytes.should eq(47_185_920_i64)
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
end
