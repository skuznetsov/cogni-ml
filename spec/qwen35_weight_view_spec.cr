require "./spec_helper"
require "../src/ml/gguf/qwen35_weight_view"

describe ML::GGUF::Qwen35WeightView do
  it "bounds an unaligned tensor slice to containing VM pages" do
    view = ML::GGUF::Qwen35WeightView.for_slice(
      0x10000_u64,
      0x10000_i64,
      0x12345_u64,
      0x5000_i64,
      0x4000_i64,
    ).not_nil!

    view.address.should eq(0x10000_u64)
    view.length.should eq(0x8000_i64)
    view.offset.should eq(0x2345_i64)
  end

  it "keeps an aligned tensor to its exact page span" do
    view = ML::GGUF::Qwen35WeightView.for_slice(
      0x10000_u64,
      0x10000_i64,
      0x14000_u64,
      0x4000_i64,
      0x4000_i64,
    ).not_nil!

    view.address.should eq(0x14000_u64)
    view.length.should eq(0x4000_i64)
    view.offset.should eq(0_i64)
  end

  it "rejects slices outside the registered aligned region" do
    ML::GGUF::Qwen35WeightView.for_slice(
      0x10000_u64,
      0xc000_i64,
      0x1b000_u64,
      0x2000_i64,
      0x4000_i64,
    ).should be_nil

    ML::GGUF::Qwen35WeightView.for_slice(
      0x10000_u64,
      0xc000_i64,
      0xffff_u64,
      1_i64,
      0x4000_i64,
    ).should be_nil
  end

  it "rejects invalid geometry instead of rounding unsafely" do
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35WeightView.for_slice(0x10000_u64, 0_i64, 0x10000_u64, 1_i64, 0x4000_i64)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35WeightView.for_slice(0x10000_u64, 0x4000_i64, 0x10000_u64, 0_i64, 0x4000_i64)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35WeightView.for_slice(0x10000_u64, 0x4000_i64, 0x10000_u64, 1_i64, 0x3000_i64)
    end
  end
end

describe ML::GGUF::Qwen35WeightViewPlan do
  it "coalesces overlapping page views and returns containing slots" do
    plan = ML::GGUF::Qwen35WeightViewPlan.for_spans(
      0x10000_u64,
      0x20000_i64,
      [
        ML::GGUF::Qwen35WeightSpan.new(0x12345_u64, 0x5000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x17000_u64, 0x6000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x28000_u64, 0x1000_i64),
      ],
      0x4000_i64,
    )

    plan.runs.should eq([
      ML::GGUF::Qwen35WeightRun.new(0x10000_u64, 0x10000_i64),
      ML::GGUF::Qwen35WeightRun.new(0x28000_u64, 0x4000_i64),
    ])
    plan.slots.should eq([
      ML::GGUF::Qwen35WeightSlot.new(0, 0x2345_i64),
      ML::GGUF::Qwen35WeightSlot.new(0, 0x7000_i64),
      ML::GGUF::Qwen35WeightSlot.new(1, 0_i64),
    ])
  end

  it "is independent of input order without overlapping output runs" do
    plan = ML::GGUF::Qwen35WeightViewPlan.for_spans(
      0x10000_u64,
      0x20000_i64,
      [
        ML::GGUF::Qwen35WeightSpan.new(0x28000_u64, 0x1000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x12345_u64, 0x5000_i64),
      ],
      0x4000_i64,
    )

    plan.runs.each_cons_pair do |left, right|
      (left.address + left.length.to_u64).should be <= right.address
    end
    plan.slots[0].run_index.should eq(1)
    plan.slots[1].run_index.should eq(0)
  end

  it "coalesces nearby page runs up to a bounded target" do
    plan = ML::GGUF::Qwen35WeightViewPlan.for_spans(
      0x10000_u64,
      0x20000_i64,
      [
        ML::GGUF::Qwen35WeightSpan.new(0x10000_u64, 0x4000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x14000_u64, 0x4000_i64),
      ],
      0x4000_i64,
      0x8000_i64,
    )

    plan.runs.should eq([
      ML::GGUF::Qwen35WeightRun.new(0x10000_u64, 0x8000_i64),
    ])
    plan.slots.should eq([
      ML::GGUF::Qwen35WeightSlot.new(0, 0_i64),
      ML::GGUF::Qwen35WeightSlot.new(0, 0x4000_i64),
    ])
  end

  it "stops before the coalescing target and keeps an oversized tensor intact" do
    plan = ML::GGUF::Qwen35WeightViewPlan.for_spans(
      0x10000_u64,
      0x40000_i64,
      [
        ML::GGUF::Qwen35WeightSpan.new(0x10000_u64, 0x4000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x18000_u64, 0x4000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x20000_u64, 0x4000_i64),
        ML::GGUF::Qwen35WeightSpan.new(0x28000_u64, 0x10000_i64),
      ],
      0x4000_i64,
      0xc000_i64,
    )

    plan.runs.should eq([
      ML::GGUF::Qwen35WeightRun.new(0x10000_u64, 0xc000_i64),
      ML::GGUF::Qwen35WeightRun.new(0x20000_u64, 0x4000_i64),
      ML::GGUF::Qwen35WeightRun.new(0x28000_u64, 0x10000_i64),
    ])
    plan.slots.map(&.run_index).should eq([0, 0, 1, 2])
  end

  it "rejects a non-page-aligned coalescing target" do
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen35WeightViewPlan.for_spans(
        0x10000_u64,
        0x20000_i64,
        [ML::GGUF::Qwen35WeightSpan.new(0x10000_u64, 0x4000_i64)],
        0x4000_i64,
        0x5000_i64,
      )
    end
  end
end
