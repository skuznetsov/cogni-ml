require "spec"
require "../examples/support/pixelumm_tiled_gate_up_probe"
require "../examples/support/pixelumm_decoder_probe"

describe PixelUMMTiledGateUpProbe do
  it "covers the full row extent once, including a partial final tile" do
    ranges = PixelUMMTiledGateUpProbe.row_ranges(21120, 4096)
    ranges.should eq([{0, 4096}, {4096, 4096}, {8192, 4096}, {12288, 4096}, {16384, 4096}, {20480, 640}])
    ranges.sum(&.[1]).should eq(21120)
    PixelUMMTiledGateUpProbe.row_ranges(2048, 768).should eq([{0, 768}, {768, 768}, {1536, 512}])
  end

  it "refuses empty, unsupported, full-width and oversized tiles" do
    [{21120, 0}, {21120, -1}, {21120, 21120}, {21120, 22000}, {5760, 1024}].each do |tokens, tile|
      expect_raises(ArgumentError) { PixelUMMTiledGateUpProbe.validate_rows!(tokens, tile) }
    end
  end

  it "crosses both tiled FFN orders with fixed input, observers and poison policy" do
    normal = PixelUMMDecoderProbe.tiled_gate_up_schedule(false)
    reverse = PixelUMMDecoderProbe.tiled_gate_up_schedule(true)
    normal.map(&.tiled_gate_up).should eq([true, false, false, true, true, false])
    reverse.map(&.tiled_gate_up).should eq([false, true, true, false, false, true])
    {normal, reverse}.each do |arms|
      arms.size.should eq(6)
      arms.count(&.poisoned).should eq(1)
      arms.first.poisoned.should be_true
      arms.any? { |arm| arm.profile || arm.alternate || arm.shared_gate_up || arm.combined_attention }.should be_false
      arms[2, 4].map(&.kind).should eq(["tiled_timed_0", "tiled_timed_1", "tiled_timed_2", "tiled_timed_3"])
    end
  end
end
