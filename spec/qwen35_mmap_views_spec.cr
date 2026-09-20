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
    ML::GGUF::Qwen35Weights.coarse_mmap_views_enabled?("2").should be_true

    ML::GGUF::Qwen35Weights.coarse_mmap_view_merge(nil).should eq(0)
    ML::GGUF::Qwen35Weights.coarse_mmap_view_merge("0").should eq(0)
    ML::GGUF::Qwen35Weights.coarse_mmap_view_merge("1").should eq(1)
    ML::GGUF::Qwen35Weights.coarse_mmap_view_merge("2").should eq(2)

    ["", " ", "01", "3", "yes"].each do |configured|
      expect_raises(ArgumentError) do
        ML::GGUF::Qwen35Weights.coarse_mmap_views_enabled?(configured)
      end
    end

    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35Weights.coarse_mmap_view_merge("3")
    end
  end

  it "pairs adjacent command groups without crossing the output-head boundary" do
    full_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

    original = ML::GGUF::Qwen35Weights.coarse_mmap_layer_ranges(full_layers, 64, 1)
    original.size.should eq(16)
    original.first.should eq({0, 6})
    original.last.should eq({63, 63})

    paired = ML::GGUF::Qwen35Weights.coarse_mmap_layer_ranges(full_layers, 64, 2)
    paired.should eq([
      {0, 10}, {11, 18}, {19, 26}, {27, 34},
      {35, 42}, {43, 50}, {51, 58}, {59, 63},
    ])
    paired.each_cons_pair { |left, right| (left[1] + 1).should eq(right[0]) }
    paired.sum { |first, last| last - first + 1 }.should eq(64)
    # The output head is deliberately not part of these layer ranges. Runtime
    # registration appends its ninth view after the eight paired layer views.
  end

  it "rejects malformed layer geometry before constructing Metal views" do
    expect_raises(ArgumentError, /strictly increasing/) do
      ML::GGUF::Qwen35Weights.coarse_mmap_layer_ranges([3, 3], 8, 2)
    end

    expect_raises(ArgumentError, /outside/) do
      ML::GGUF::Qwen35Weights.coarse_mmap_layer_ranges([3, 8], 8, 2)
    end

    expect_raises(ArgumentError, /merge/) do
      ML::GGUF::Qwen35Weights.coarse_mmap_layer_ranges([3, 7], 8, 3)
    end

    expect_raises(ArgumentError, /even/) do
      ML::GGUF::Qwen35Weights.coarse_mmap_layer_ranges([1, 3, 5], 8, 2)
    end
  end
end
