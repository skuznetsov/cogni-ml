require "spec"
require "../examples/support/hy_nomas_t0_rgb_history"

describe HyNomasT0RgbHistory do
  it "admits only moving and repeat-last source modes with the pinned frame map" do
    HyNomasT0RgbHistory.validate_context_mode!("moving").should eq("moving")
    HyNomasT0RgbHistory.validate_context_mode!("repeat-last").should eq("repeat-last")
    120.times do |frame|
      HyNomasT0RgbHistory.source_frame("moving", frame).should eq(frame)
      HyNomasT0RgbHistory.source_frame("repeat-last", frame).should eq(119)
    end
    expect_raises(ArgumentError, /context mode/) do
      HyNomasT0RgbHistory.validate_context_mode!("first-only")
    end
    expect_raises(ArgumentError, /output frame/) do
      HyNomasT0RgbHistory.source_frame("moving", 120)
    end
  end

  it "requires the exact pinned F32 planar source geometry and byte extent" do
    HyNomasT0RgbHistory.validate_source_geometry!("F32LE",
      [1, 3, 360, 288, 512], HyNomasT0RgbHistory::SOURCE_BYTES).should be_true
    expect_raises(ArgumentError, /dtype/) do
      HyNomasT0RgbHistory.validate_source_geometry!("BF16LE",
        [1, 3, 360, 288, 512], HyNomasT0RgbHistory::SOURCE_BYTES)
    end
    expect_raises(ArgumentError, /shape/) do
      HyNomasT0RgbHistory.validate_source_geometry!("F32LE",
        [1, 3, 120, 288, 512], HyNomasT0RgbHistory::SOURCE_BYTES)
    end
    expect_raises(ArgumentError, /extent/) do
      HyNomasT0RgbHistory.validate_source_geometry!("F32LE",
        [1, 3, 360, 288, 512], HyNomasT0RgbHistory::SOURCE_BYTES - 4)
    end
  end

  it "rejects a substitute source path or mismatched authenticated digest" do
    HyNomasT0RgbHistory.validate_source_binding!(
      HyNomasT0RgbHistory::SOURCE_PIXELS_PATH,
      HyNomasT0RgbHistory::SOURCE_PIXELS_PATH,
      HyNomasT0RgbHistory::SOURCE_BYTES,
      HyNomasT0RgbHistory::SOURCE_PIXELS_SHA256,
      HyNomasT0RgbHistory::SOURCE_BYTES,
      HyNomasT0RgbHistory::SOURCE_PIXELS_SHA256).should be_true
    expect_raises(ArgumentError, /pinned source path/) do
      HyNomasT0RgbHistory.validate_source_binding!(
        "/tmp/other-pixels.f32le", HyNomasT0RgbHistory::SOURCE_PIXELS_PATH,
        HyNomasT0RgbHistory::SOURCE_BYTES,
        HyNomasT0RgbHistory::SOURCE_PIXELS_SHA256,
        HyNomasT0RgbHistory::SOURCE_BYTES,
        HyNomasT0RgbHistory::SOURCE_PIXELS_SHA256)
    end
    expect_raises(ArgumentError, /SHA-256/) do
      HyNomasT0RgbHistory.validate_source_binding!(
        HyNomasT0RgbHistory::SOURCE_PIXELS_PATH,
        HyNomasT0RgbHistory::SOURCE_PIXELS_PATH,
        HyNomasT0RgbHistory::SOURCE_BYTES,
        HyNomasT0RgbHistory::SOURCE_PIXELS_SHA256,
        HyNomasT0RgbHistory::SOURCE_BYTES,
        "a" * 64)
    end
  end

  it "rejects named plans above the caller's budget" do
    HyNomasT0RgbHistory.validate_named_budget!(100_i64, 100_i64)
    expect_raises(ArgumentError, /named plan exceeds budget/) do
      HyNomasT0RgbHistory.validate_named_budget!(101_i64, 100_i64)
    end
  end

  it "requires zero preterminal counters and exactly one terminal wait" do
    HyNomasT0RgbHistory.validate_preterminal_metrics!(0_i64, 0_i64, 0_i64)
    [[1_i64, 0_i64, 0_i64], [0_i64, 1_i64, 0_i64], [0_i64, 0_i64, 1_i64]].each do |counts|
      expect_raises(ArgumentError, /preterminal/) do
        HyNomasT0RgbHistory.validate_preterminal_metrics!(counts[0], counts[1], counts[2])
      end
    end
    HyNomasT0RgbHistory.validate_encode_metrics!(0_i64, 0_i64, 0_i64, 1_i64)
    [[1_i64, 0_i64, 0_i64, 1_i64], [0_i64, 1_i64, 0_i64, 1_i64],
     [0_i64, 0_i64, 1_i64, 1_i64], [0_i64, 0_i64, 0_i64, 2_i64]].each do |counts|
      expect_raises(ArgumentError) do
        HyNomasT0RgbHistory.validate_encode_metrics!(counts[0], counts[1], counts[2], counts[3])
      end
    end
  end

  it "requires the supplied normalized clip digest and exact GPU-byte match" do
    expected = "a" * 64
    HyNomasT0RgbHistory.validate_expected_clip_sha256!(expected).should eq(expected)
    HyNomasT0RgbHistory.validate_actual_clip_sha256!(expected, expected).should eq(expected)
    expect_raises(ArgumentError, /lowercase hex/) do
      HyNomasT0RgbHistory.validate_expected_clip_sha256!("A" * 64)
    end
    expect_raises(ArgumentError, /differs from/) do
      HyNomasT0RgbHistory.validate_actual_clip_sha256!("b" * 64, expected)
    end
  end

  it "expands BF16 endpoint values into checked little-endian F32 bytes" do
    HyNomasT0RgbHistory.bf16_f32_le_bytes(0x3f80_u16).should eq(
      {0x00_u8, 0x00_u8, 0x80_u8, 0x3f_u8})
  end
end
