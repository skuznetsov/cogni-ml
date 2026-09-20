require "./spec_helper"
require "../src/ml/gguf/qwen35_mmap_views"
require "../src/ml/gguf/qwen35_weights"

describe ML::GGUF::Qwen35MmapWeightView do
  it "bounds a dense command group to containing VM pages" do
    view = ML::GGUF::Qwen35MmapWeightView.for_spans(
      0x10000_u64,
      0x40000_u64,
      [
        ML::GGUF::Qwen35MmapWeightSpan.new(0x12345_u64, 0x5000_i64),
        ML::GGUF::Qwen35MmapWeightSpan.new(0x18000_u64, 0x7000_i64),
      ],
      0x4000_i64,
      0.50_f64,
    )

    view.address.should eq(0x10000_u64)
    view.length.should eq(0x10000_i64)
    view.tensor_bytes.should eq(0xc000_i64)
    view.contains?(ML::GGUF::Qwen35MmapWeightSpan.new(0x18000_u64, 0x7000_i64)).should be_true
  end

  it "rejects a sparse group that would recreate a whole-file view" do
    expect_raises(ArgumentError, /density/) do
      ML::GGUF::Qwen35MmapWeightView.for_spans(
        0x10000_u64,
        0x100000_u64,
        [
          ML::GGUF::Qwen35MmapWeightSpan.new(0x10000_u64, 0x4000_i64),
          ML::GGUF::Qwen35MmapWeightSpan.new(0x10c000_u64, 0x4000_i64),
        ],
        0x4000_i64,
      )
    end
  end

  it "rejects invalid and out-of-map geometry" do
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35MmapWeightView.for_spans(
        0x10001_u64,
        0x10000_u64,
        [ML::GGUF::Qwen35MmapWeightSpan.new(0x14000_u64, 0x4000_i64)],
        0x4000_i64,
      )
    end

    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35MmapWeightView.for_spans(
        0x10000_u64,
        0x10000_u64,
        [ML::GGUF::Qwen35MmapWeightSpan.new(0x1c000_u64, 0x8000_i64)],
        0x4000_i64,
      )
    end
  end

  it "rejects overlapping spans instead of inflating density" do
    expect_raises(ArgumentError, /overlap/) do
      ML::GGUF::Qwen35MmapWeightView.for_spans(
        0x10000_u64,
        0x10000_u64,
        [
          ML::GGUF::Qwen35MmapWeightSpan.new(0x10000_u64, 0x8000_i64),
          ML::GGUF::Qwen35MmapWeightSpan.new(0x14000_u64, 0x8000_i64),
        ],
        0x4000_i64,
      )
    end
  end

  it "rejects overflowing owner geometry and oversized views" do
    expect_raises(ArgumentError, /overflows/) do
      ML::GGUF::Qwen35MmapWeightView.for_spans(
        UInt64::MAX - 0x3fff_u64,
        0x8000_u64,
        [ML::GGUF::Qwen35MmapWeightSpan.new(UInt64::MAX - 0x3fff_u64, 0x4000_i64)],
        0x4000_i64,
      )
    end

    expect_raises(ArgumentError, /too large/) do
      ML::GGUF::Qwen35MmapWeightView.for_spans(
        0_u64,
        0x8000000000000000_u64,
        [ML::GGUF::Qwen35MmapWeightSpan.new(0_u64, Int64::MAX)],
        0x4000_i64,
      )
    end
  end
end

describe ML::GGUF::Qwen35Weights do
  it "keeps coarse views explicitly fail-closed" do
    ML::GGUF::Qwen35Weights.coarse_mmap_views_enabled?(nil).should be_false
    ML::GGUF::Qwen35Weights.coarse_mmap_views_enabled?("0").should be_false
    ML::GGUF::Qwen35Weights.coarse_mmap_views_enabled?("1").should be_true

    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35Weights.coarse_mmap_views_enabled?("yes")
    end
  end
end
