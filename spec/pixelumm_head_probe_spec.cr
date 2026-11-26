require "spec"
require "../examples/support/pixelumm_head_probe"

describe PixelUMMHeadProbe do
  it "detects wrong token or tubelet order at the unchanged tolerance" do
    values = Bytes[0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]
    reordered = Bytes[0x00, 0x40, 0x80, 0x3f, 0x80, 0x40, 0x40, 0x40]
    exact = PixelUMMHeadProbe.compare_bytes("layout", values, values)
    exact[:failures].should eq(0)
    exact[:different_words].should eq(0)
    wrong = PixelUMMHeadProbe.compare_bytes("layout", values, reordered)
    wrong[:failures].should eq(4)
    wrong[:first_failure_index].should eq(0)
  end

  it "rejects non-finite output and mismatched extents" do
    finite = Bytes[0x80, 0x3f]
    nan = Bytes[0xc0, 0x7f]
    result = PixelUMMHeadProbe.compare_bytes("finite", finite, nan)
    result[:finite].should be_false
    result[:failures].should eq(1)
    expect_raises(ArgumentError) { PixelUMMHeadProbe.compare_bytes("shape", finite, Bytes[0]) }
  end

  it "keeps the exact full-J output and head weight extents" do
    shapes = PixelUMMHeadProbe.capture_shapes([6, 44, 80])
    shapes["output_thw"].should eq([1, 3, 24, 704, 1280])
    PixelUMMHeadProbe.weight_shapes.values.sum { |shape| shape.reduce(2_i64) { |a, b| a * b } }.should eq(25_188_352_i64)
  end
end
