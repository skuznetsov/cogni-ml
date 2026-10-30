require "spec"
require "../examples/support/pixelumm_window_batch_probe"

describe PixelUMMWindowBatchProbe do
  it "requires both saved upstream attention and block output before allocation" do
    PixelUMMWindowBatchProbe.validate_capture_inventory!(["attention", "output"])
    expect_raises(ArgumentError, "exactly attention and output") do
      PixelUMMWindowBatchProbe.validate_capture_inventory!(["output"])
    end
    expect_raises(ArgumentError) do
      PixelUMMWindowBatchProbe.validate_capture_inventory!(["output", "attention"])
    end
  end

  it "pairs all full-J windows without changing either reduction geometry" do
    layout = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80).layout
    pairs = PixelUMMWindowBatchProbe.pairs(layout)
    pairs.size.should eq(30)
    pairs.count { |p| {p.queries, p.keys} == {384, 3456} }.should eq(20)
    pairs.count { |p| {p.queries, p.keys} == {384, 2880} }.should eq(5)
    pairs.count { |p| {p.queries, p.keys} == {192, 2880} }.should eq(5)
    pairs.flat_map { |p| layout.query_indices(p.first) + layout.query_indices(p.second) }.sort.should eq((0...21120).to_a)
    PixelUMMWindowBatchProbe::EXTRA_PANEL_BYTES.should eq(931135488_i64)
    PixelUMMWindowBatchProbe.expected_elements(1, layout).should eq(2229534720_u32)
    PixelUMMWindowBatchProbe.expected_elements(2, layout).should eq(86507520_u32)
    PixelUMMWindowBatchProbe.expected_elements(3, layout).should eq(84934656_u32)
    expect_raises(ArgumentError) { PixelUMMWindowBatchProbe.expected_elements(0, layout) }
  end

  it "retains distinct contexts even when both windows have the same shape" do
    layout = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80).layout
    pair = PixelUMMWindowBatchProbe.pairs(layout).find do |p|
      layout.key_indices(p.first) != layout.key_indices(p.second)
    end.not_nil!
    PixelUMMWindowBatchProbe.validate_lane_context!(layout, pair.first,
      layout.query_indices(pair.first), layout.key_indices(pair.first))
    expect_raises(ArgumentError, "ordered query/key context") do
      PixelUMMWindowBatchProbe.validate_lane_context!(layout, pair.first,
        layout.query_indices(pair.first), layout.key_indices(pair.second))
    end
    expect_raises(ArgumentError, "ordered query/key context") do
      PixelUMMWindowBatchProbe.validate_lane_context!(layout, pair.first,
        layout.query_indices(pair.first).reverse, layout.key_indices(pair.first))
    end
  end

  it "rejects duplicate coverage and mixing partial border windows with full windows" do
    layout = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80).layout
    pairs = PixelUMMWindowBatchProbe.pairs(layout)
    duplicate = pairs.dup
    duplicate[-1] = duplicate.first
    expect_raises(ArgumentError, "exactly once") { PixelUMMWindowBatchProbe.validate_pairs!(layout, duplicate) }
    wrong = pairs.dup
    p = wrong.last
    wrong[-1] = PixelUMMWindowBatchProbe::Pair.new(p.first, p.second, p.queries + 1, p.keys)
    expect_raises(ArgumentError, "reduction geometry") { PixelUMMWindowBatchProbe.validate_pairs!(layout, wrong) }
  end
end
