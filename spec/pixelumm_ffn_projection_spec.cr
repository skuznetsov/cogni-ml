require "spec"
require "../examples/support/pixelumm_ffn_projection_probe"

private def pixelumm_ffn_words(values : Array(UInt16)) : Bytes
  bytes = Bytes.new(values.size * 2)
  values.each_with_index { |value, i| IO::ByteFormat::LittleEndian.encode(value, bytes[i * 2, 2]) }
  bytes
end

describe PixelUMMFFNProjectionProbe do
  it "checks every finite BF16 word against the unchanged absolute and relative tolerance" do
    expected = pixelumm_ffn_words([0_u16, 0x3f80_u16, 0x4000_u16, 0x3f80_u16])
    actual = pixelumm_ffn_words([0x3780_u16, 0x3f81_u16, 0x4000_u16, 0x3f80_u16])
    metric = PixelUMMFFNProjectionProbe.compare(expected, actual)
    # 2^-16 is inside the absolute allowance; one BF16 step at one is not.
    metric.should eq({failures: 1_u32, different_words: 2_u32, nonfinite: 0_u32, elements: 4_u32})
  end

  it "rejects matching NaNs and infinities rather than passing equal payloads" do
    values = pixelumm_ffn_words([0x7fc0_u16, 0x7f80_u16, 0xff80_u16])
    PixelUMMFFNProjectionProbe.compare(values, values).should eq(
      {failures: 3_u32, different_words: 0_u32, nonfinite: 3_u32, elements: 3_u32})
  end

  it "preserves numerical equality of signed zero while reporting changed words" do
    PixelUMMFFNProjectionProbe.compare(pixelumm_ffn_words([0_u16]), pixelumm_ffn_words([0x8000_u16])).should eq(
      {failures: 0_u32, different_words: 1_u32, nonfinite: 0_u32, elements: 1_u32})
  end

  it "rejects a partial, empty or mismatched comparison extent" do
    expect_raises(ArgumentError) { PixelUMMFFNProjectionProbe.compare(Bytes.empty, Bytes.empty) }
    expect_raises(ArgumentError) { PixelUMMFFNProjectionProbe.compare(Bytes.new(1), Bytes.new(1)) }
    expect_raises(ArgumentError) { PixelUMMFFNProjectionProbe.compare(Bytes.new(2), Bytes.new(4)) }
  end
end
