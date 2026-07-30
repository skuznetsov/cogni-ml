require "../spec_helper"

describe ML::Tensor::CPUReadView do
  it "borrows contiguous CPU storage without materialization" do
    tensor = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )

    tensor.layout_class.should eq(ML::Tensor::LayoutClass::Contiguous)
    read = tensor.cpu_read

    read.source_layout.should eq(ML::Tensor::LayoutClass::Contiguous)
    read.source_device.should eq(ML::Tensor::Device::CPU)
    read.source_shape.should eq(ML::Shape.new(2_i32, 2_i32))
    read.source_strides.to_a.should eq([2_i32, 1_i32])
    read.borrowed?.should be_true
    read.materialized?.should be_false
    read.materialized_bytes.should eq(0_i64)
    read.to_a.should eq([1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])

    tensor[0_i32, 0_i32] = 9.0_f32
    read[0].should eq(9.0_f32)
  end

  it "materializes one logical snapshot for a dense strided CPU view" do
    base = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
      ML::Shape.new(2_i32, 3_i32)
    )
    view = base.transpose

    view.layout_class.should eq(ML::Tensor::LayoutClass::DenseStrided)
    read = view.cpu_read

    read.source_layout.should eq(ML::Tensor::LayoutClass::DenseStrided)
    read.source_device.should eq(ML::Tensor::Device::CPU)
    read.source_shape.should eq(ML::Shape.new(3_i32, 2_i32))
    read.source_strides.to_a.should eq([1_i32, 3_i32])
    read.borrowed?.should be_false
    read.materialized?.should be_true
    read.materialized_bytes.should eq(24_i64)
    read.to_a.should eq([
      1.0_f32, 4.0_f32,
      2.0_f32, 5.0_f32,
      3.0_f32, 6.0_f32,
    ])

    base[0_i32, 0_i32] = 99.0_f32
    read[0].should eq(1.0_f32)
  end

  it "treats singleton-axis stride differences as contiguous" do
    tensor = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32],
      ML::Shape.new(2_i32, 1_i32)
    )
    view = tensor.transpose

    view.shape.should eq(ML::Shape.new(1_i32, 2_i32))
    view.strides.to_a.should eq([1_i32, 1_i32])
    view.layout_class.should eq(ML::Tensor::LayoutClass::Contiguous)
    read = view.cpu_read
    read.borrowed?.should be_true
    read.materialized_bytes.should eq(0_i64)
    read.to_a.should eq([1.0_f32, 2.0_f32])
  end
end
