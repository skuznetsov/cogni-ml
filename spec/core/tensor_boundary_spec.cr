require "../spec_helper"

module TensorBoundarySpec
  class ProtectedConstructorProbe < ML::Tensor
    def self.cpu(dtype : ML::DType, data : Array(Float32)?) : ML::Tensor
      shape = ML::Shape.new(1_i32)
      new(
        shape,
        ML::Strides.new(shape),
        dtype,
        ML::Tensor::Device::CPU,
        nil,
        data
      )
    end

    def self.gpu_without_buffer(dtype : ML::DType) : ML::Tensor
      shape = ML::Shape.new(1_i32)
      new(
        shape,
        ML::Strides.new(shape),
        dtype,
        ML::Tensor::Device::GPU,
        nil,
        nil
      )
    end

    def self.with_strides(strides : ML::Strides) : ML::Tensor
      shape = ML::Shape.new(1_i32)
      new(
        shape,
        strides,
        ML::DType::F32,
        ML::Tensor::Device::CPU,
        nil,
        [0.0_f32]
      )
    end

    def self.with_layout(shape : ML::Shape, strides : ML::Strides) : ML::Tensor
      new(
        shape,
        strides,
        ML::DType::F32,
        ML::Tensor::Device::CPU,
        nil,
        Array(Float32).new(shape.numel, 0.0_f32)
      )
    end
  end
end

describe ML::Tensor do
  it "accepts a coherent F32 protected-constructor value" do
    tensor = TensorBoundarySpec::ProtectedConstructorProbe.cpu(
      ML::DType::F32,
      [1.0_f32]
    )

    tensor.dtype.should eq(ML::DType::F32)
    tensor.to_a.should eq([1.0_f32])
  end

  it "rejects non-F32 through the protected constructor" do
    {ML::DType::F16, ML::DType::BF16}.each do |dtype|
      expect_raises(ArgumentError, /Only F32/) do
        TensorBoundarySpec::ProtectedConstructorProbe.cpu(
          dtype,
          [1.0_f32]
        )
      end
    end
  end

  it "rejects missing or mismatched CPU storage at construction" do
    expect_raises(ArgumentError, /CPU tensor requires data/) do
      TensorBoundarySpec::ProtectedConstructorProbe.cpu(
        ML::DType::F32,
        nil
      )
    end
    expect_raises(ArgumentError, /CPU tensor data length/) do
      TensorBoundarySpec::ProtectedConstructorProbe.cpu(
        ML::DType::F32,
        [] of Float32
      )
    end
  end

  it "rejects missing GPU storage before dispatch" do
    expect_raises(ArgumentError, /GPU tensor requires a buffer/) do
      TensorBoundarySpec::ProtectedConstructorProbe.gpu_without_buffer(
        ML::DType::F32
      )
    end
  end

  it "rejects zero-element GPU allocation before buffer creation" do
    expect_raises(ArgumentError, /zero-element GPU tensor/) do
      ML::Tensor.new(
        ML::Shape.new(0_i32),
        ML::DType::F32,
        ML::Tensor::Device::GPU
      )
    end
  end

  it "preserves zero-element CPU tensors as an explicit empty value" do
    tensor = ML::Tensor.new(
      ML::Shape.new(0_i32),
      ML::DType::F32,
      ML::Tensor::Device::CPU
    )

    tensor.numel.should eq(0)
    tensor.safe_cpu_data.should be_empty
    expect_raises(IndexError) { tensor[0_i32] }
  end

  it "rejects zero-element mutating GPU transfer before backend access" do
    tensor = ML::Tensor.new(
      ML::Shape.new(0_i32),
      ML::DType::F32,
      ML::Tensor::Device::CPU
    )

    expect_raises(ArgumentError, /zero-element GPU tensor/) do
      tensor.to_gpu!
    end
    tensor.on_cpu?.should be_true
    tensor.safe_cpu_data.should be_empty
  end

  it "rejects a stride rank mismatch at construction" do
    expect_raises(ArgumentError, /stride rank/) do
      TensorBoundarySpec::ProtectedConstructorProbe.with_strides(
        ML::Strides.new([1_i32, 1_i32])
      )
    end
  end

  it "rejects negative or overlapping dense storage layouts" do
    shape = ML::Shape.new(2_i32, 2_i32)

    {
      ML::Strides.new([-1_i32, 1_i32]),
      ML::Strides.new([3_i32, 0_i32]),
    }.each do |strides|
      expect_raises(ArgumentError, /dense non-overlapping/) do
        TensorBoundarySpec::ProtectedConstructorProbe.with_layout(
          shape,
          strides
        )
      end
    end
  end

  it "keeps reshape views coherent through the guarded constructor" do
    tensor = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )

    reshaped = tensor.reshape(4_i32)
    reshaped.shape.should eq(ML::Shape.new(4_i32))
    reshaped.to_a.should eq(tensor.to_a)
  end

  it "accepts a dense transposed layout without flattening its strides" do
    tensor = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
      ML::Shape.new(2_i32, 3_i32)
    )

    transposed = tensor.transpose
    transposed.shape.should eq(ML::Shape.new(3_i32, 2_i32))
    transposed.strides.to_a.should eq([1, 3])
    transposed[0_i32, 1_i32].should eq(4.0_f32)
  end

  it "checks every Tensor index before flat-index arithmetic" do
    tensor = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
      ML::Shape.new(2_i32, 2_i32)
    )

    expect_raises(IndexError, /index -1.*dimension 0/) do
      tensor[-1_i32, 0_i32]
    end
    expect_raises(IndexError, /index 2.*dimension 0/) do
      tensor[2_i32, 0_i32] = 0.0_f32
    end
  end
end
