require "spec"
require "../examples/support/pixelumm_decoder_probe"

describe PixelUMMDecoderProbe do
  it "admits the derived unique full-J aggregate without counting the borrowed head input twice" do
    block_named = 22_958_105_572_i64
    head_weights = 25_188_352_i64
    config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: 6, height: 44, width: 80)
    head_owned = ML::Video::PixelUMMDecoderHeadMPS::Plan.owned_bytes(config, PixelUMMHeadProbe::CAPTURES)

    head_owned.should eq(1_124_851_204_i64)
    required = PixelUMMDecoderProbe.aggregate_named_bytes(
      block_named: block_named, head_weight_named: head_weights, head_owned_named: head_owned)
    required.should eq(24_108_145_128_i64)
    PixelUMMDecoderProbe.validate_aggregate_budget!(required, required).should eq(required)
    expect_raises(ArgumentError, /aggregate/) do
      PixelUMMDecoderProbe.validate_aggregate_budget!(required, required - 1)
    end
  end

  it "requires the head fixture input to be exactly the authenticated block-35 output" do
    digest = "a" * 64
    PixelUMMDecoderProbe.validate_pair!(digest, digest)
    expect_raises(ArgumentError, /block35/) do
      PixelUMMDecoderProbe.validate_pair!(digest, "b" * 64)
    end
  end

  it "rejects head encoding until all block encoding returned on the same open chain" do
    PixelUMMDecoderProbe.validate_head_order!(blocks_encoded: true, same_chain: true, chain_open: true)
    {
      {blocks_encoded: false, same_chain: true, chain_open: true},
      {blocks_encoded: true, same_chain: false, chain_open: true},
      {blocks_encoded: true, same_chain: true, chain_open: false},
    }.each do |case_values|
      expect_raises(ArgumentError, /head.*after.*blocks/) do
        PixelUMMDecoderProbe.validate_head_order!(**case_values)
      end
    end
  end
end
