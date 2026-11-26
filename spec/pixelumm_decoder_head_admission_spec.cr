require "spec"
require "../src/ml/video/pixelumm_decoder_head_mps"

private def pixelumm_head_weight_bytes : Hash(String, Int64)
  {
    "final_norm_weight"  => 4_096_i64 * 2,
    "head_norm_weight"   => 4_096_i64 * 2,
    "head_linear_weight" => 3_072_i64 * 4_096 * 2,
    "head_linear_bias"   => 3_072_i64 * 2,
  }
end

private def pixelumm_head_owned_bytes(config : ML::Video::PixelUMMDecoderHeadMPS::Config) : Int64
  j = config.tokens.to_i64
  d = config.model_dim.to_i64
  out = config.output_dim.to_i64
  5_i64 * j * d * 2 + 2_i64 * j * out * 2 + 2_i64 * j * 4 + j * 4 + 4
end

describe ML::Video::PixelUMMDecoderHeadMPS::Config do
  it "pins the production output shape and post-block token geometry" do
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(
      temporal: 6, height: 44, width: 80)

    config.tokens.should eq(21_120)
    config.model_dim.should eq(4_096)
    config.output_dim.should eq(3_072)
    config.output_shape.should eq([1, 3, 24, 704, 1_280])
    config.output_elements.should eq(64_880_640_i64)
    config.output_bytes.should eq(129_761_280_i64)
  end

  it "uses the decoder block inverse permutation for restoration" do
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(
      temporal: 2, height: 32, width: 32)
    permutation = config.layout.permutation
    inverse = config.restoration_indices

    permutation.size.should eq(config.tokens)
    inverse.size.should eq(config.tokens)
    permutation.should_not eq((0...config.tokens).to_a)
    inverse.each_with_index do |tile_slot, original_slot|
      permutation[tile_slot].should eq(original_slot.to_i32)
    end
  end

  it "rejects invalid geometry before any Metal buffer is created" do
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: 0, height: 44, width: 80)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: 6, height: -1, width: 80)
    end
    expect_raises(ArgumentError) do
      ML::Video::PixelUMMDecoderHeadMPS::Config.new(
        temporal: 6, height: 44, width: 80, tile_tokens: 16)
    end
  end
end

describe ML::Video::PixelUMMDecoderHeadMPS::PoisonPattern do
  it "fills every slab and the tail with the BF16 NaN byte pattern" do
    slab = ML::Video::PixelUMMDecoderHeadMPS::PoisonPattern.nan_slab
    slab_size = ML::Video::PixelUMMDecoderHeadMPS::PoisonPattern::SLAB_BYTES
    destination_size = slab_size * 2 + 12_346
    destination = Bytes.new(destination_size, 0x00_u8)

    ML::Video::PixelUMMDecoderHeadMPS::PoisonPattern.fill!(destination, slab)

    expected = Bytes.new(destination_size, 0x00_u8)
    destination_size.times do |index|
      expected[index] = index.even? ? 0xc0_u8 : 0x7f_u8
    end
    destination.should eq(expected)
    [0, slab_size - 1, slab_size, slab_size + 1,
     slab_size * 2 - 1, slab_size * 2, destination_size - 1].each do |index|
      destination[index].should eq(index.even? ? 0xc0_u8 : 0x7f_u8)
    end
  end
end

describe ML::Video::PixelUMMDecoderHeadMPS::Plan do
  it "admits exact BF16 bindings and a budget that includes all bounded scratch" do
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(
      temporal: 6, height: 44, width: 80)
    weights = pixelumm_head_weight_bytes
    input_bytes = config.tokens.to_i64 * config.model_dim * 2
    captures = ML::Video::PixelUMMDecoderHeadMPS::Plan::CAPTURE_NAMES.to_a
    required = pixelumm_head_owned_bytes(config)

    ML::Video::PixelUMMDecoderHeadMPS::Plan.owned_bytes(config)
      .should eq(required)
    ML::Video::PixelUMMDecoderHeadMPS::Plan.owned_bytes(config, captures)
      .should eq(required)
    ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
      input_bytes: input_bytes, weights_bytes: weights,
      max_named_bytes: required, captures: captures)
  end

  it "rejects a one-byte-short input, wrong weight extents, and insufficient budget" do
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(
      temporal: 2, height: 16, width: 24)
    weights = pixelumm_head_weight_bytes
    input_bytes = config.tokens.to_i64 * config.model_dim * 2
    required = pixelumm_head_owned_bytes(config)

    expect_raises(ArgumentError, /input extent/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes - 1, weights_bytes: weights,
        max_named_bytes: required)
    end

    short_weight = weights.dup
    short_weight["head_linear_bias"] -= 1
    expect_raises(ArgumentError, /head_linear_bias/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, weights_bytes: short_weight,
        max_named_bytes: required)
    end

    expect_raises(ArgumentError, /budget/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, weights_bytes: weights,
        max_named_bytes: required - 1)
    end
  end

  it "rejects missing, extra, duplicate, or unknown head bindings and captures" do
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(
      temporal: 2, height: 16, width: 24)
    weights = pixelumm_head_weight_bytes
    input_bytes = config.tokens.to_i64 * config.model_dim * 2
    required = pixelumm_head_owned_bytes(config)

    missing = weights.dup
    missing.delete("head_norm_weight")
    expect_raises(ArgumentError, /exactly four/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, weights_bytes: missing,
        max_named_bytes: required)
    end

    extra = weights.dup
    extra["head_linear_extra"] = 2_i64
    expect_raises(ArgumentError, /exactly four/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, weights_bytes: extra,
        max_named_bytes: required)
    end

    expect_raises(ArgumentError, /unique/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, weights_bytes: weights,
        max_named_bytes: required,
        captures: ["restored_thw", "restored_thw"])
    end

    expect_raises(ArgumentError, /unknown/) do
      ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
        input_bytes: input_bytes, weights_bytes: weights,
        max_named_bytes: required, captures: ["pre_patch_linear"])
    end
  end
end
