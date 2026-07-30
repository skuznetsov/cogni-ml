require "../spec_helper"

describe ML::Shape do
  it "rejects negative dimensions at construction" do
    expect_raises(ArgumentError, /dimension 0.*non-negative/) do
      ML::Shape.new(-2_i32, 3_i32)
    end
  end

  it "rejects an unrepresentable element count at construction" do
    expect_raises(ArgumentError, /element count overflow/) do
      ML::Shape.new(46_341_i32, 46_341_i32)
    end
  end

  it "preserves representable zero-element shapes" do
    shape = ML::Shape.new(2_i32, 0_i32, 3_i32)

    shape.numel.should eq(0)
    ML::Strides.new(shape).to_a.should eq([0, 3, 1])

    trailing_zero = ML::Shape.new(Int32::MAX, Int32::MAX, 0_i32)
    trailing_zero.numel.should eq(0)
    ML::Strides.new(trailing_zero).to_a.should eq([0, 0, 1])
  end

  it "keeps the exact Int32 element-count boundary representable" do
    shape = ML::Shape.new(Int32::MAX)

    shape.numel.should eq(Int32::MAX)
    ML::Strides.new(shape).to_a.should eq([1])
  end
end

describe ML::Strides do
  it "rejects an unrepresentable contiguous stride before tensor allocation" do
    shape = ML::Shape.new(0_i32, Int32::MAX, Int32::MAX)

    expect_raises(ArgumentError, /contiguous stride overflow/) do
      ML::Strides.new(shape)
    end
  end
end

describe ML::ShapeOps do
  it "reports a typed flatten overflow instead of leaking OverflowError" do
    shape = ML::Shape.new(0_i32, Int32::MAX, Int32::MAX)

    expect_raises(ArgumentError, /element count overflow/) do
      ML::ShapeOps.flatten_shape(shape, 1_i32, 2_i32)
    end
  end

  it "normalizes valid negative flatten dimensions" do
    shape = ML::Shape.new(2_i32, 3_i32, 4_i32)

    ML::ShapeOps.flatten_shape(shape, -2_i32, -1_i32)
      .should eq(ML::Shape.new(2_i32, 12_i32))
  end

  it "rejects flatten dimensions outside the shape rank" do
    shape = ML::Shape.new(2_i32, 3_i32, 4_i32)

    expect_raises(ArgumentError, /Invalid flatten dims/) do
      ML::ShapeOps.flatten_shape(shape, 3_i32, 3_i32)
    end
    expect_raises(ArgumentError, /Invalid flatten dims/) do
      ML::ShapeOps.flatten_shape(shape, -4_i32, -1_i32)
    end
  end
end
