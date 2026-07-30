require "../spec_helper"

describe ML::DType do
  it "preserves existing enum values while appending BF16" do
    ML::DType::F32.value.should eq(0)
    ML::DType::F16.value.should eq(1)
    ML::DType::I32.value.should eq(2)
    ML::DType::I64.value.should eq(3)
    ML::DType::U8.value.should eq(4)
    ML::DType::BF16.value.should eq(5)
  end

  it "reports exact byte widths for all floating dtypes" do
    ML::DType::F32.byte_size.should eq(4)
    ML::DType::F16.byte_size.should eq(2)
    ML::DType::BF16.byte_size.should eq(2)

    ML::DType::F32.floating?.should be_true
    ML::DType::F16.floating?.should be_true
    ML::DType::BF16.floating?.should be_true
    ML::DType::I32.floating?.should be_false
  end
end

describe ML::Tensor do
  it "keeps F16 and BF16 outside the F32-only Tensor boundary" do
    shape = ML::Shape.new([1_i32])

    expect_raises(ArgumentError, /Only F32/) do
      ML::Tensor.new(shape, ML::DType::F16)
    end
    expect_raises(ArgumentError, /Only F32/) do
      ML::Tensor.new(shape, ML::DType::BF16)
    end
  end
end

describe ML::FloatingStorage do
  it "allocates exactly dtype-sized zeroed CPU storage" do
    {
      ML::DType::F32  => 12,
      ML::DType::F16  => 6,
      ML::DType::BF16 => 6,
    }.each do |dtype, expected_bytes|
      storage = ML::FloatingStorage.zeros(3, dtype)

      storage.dtype.should eq(dtype)
      storage.numel.should eq(3)
      storage.byte_size.should eq(expected_bytes)
      storage.to_bytes.should eq(Bytes.new(expected_bytes, 0_u8))
    end
  end

  it "round-trips F16 and BF16 payload bits without conversion" do
    {
      ML::DType::F16  => [0x0000_u16, 0x3c00_u16, 0x7e01_u16],
      ML::DType::BF16 => [0x8000_u16, 0x3f80_u16, 0x7fc1_u16],
    }.each do |dtype, words|
      storage = ML::FloatingStorage.zeros(words.size.to_i32, dtype)
      words.each_with_index do |word, index|
        storage.write_u16_bits(index.to_i32, word)
      end

      words.each_with_index do |word, index|
        storage.read_u16_bits(index.to_i32).should eq(word)
      end
      restored = ML::FloatingStorage.from_bytes(
        storage.to_bytes,
        words.size.to_i32,
        dtype
      )
      restored.to_bytes.should eq(storage.to_bytes)
    end
  end

  it "preserves exact F32 bits including signed zero and NaN payloads" do
    bits = [0x00000000_u32, 0x80000000_u32, 0x7fc01234_u32]
    bytes = Bytes.new(bits.size * 4)
    bits.each_with_index do |value, index|
      IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
    end

    storage = ML::FloatingStorage.from_bytes(
      bytes,
      bits.size.to_i32,
      ML::DType::F32
    )
    bits.each_with_index do |value, index|
      storage.read_f32(index.to_i32).unsafe_as(UInt32).should eq(value)
    end
    storage.to_bytes.should eq(bytes)

    storage.write_f32(0, -1.5_f32)
    storage.read_f32(0).should eq(-1.5_f32)
  end

  it "copies imported, exported, and cloned storage" do
    source = Bytes[0x00, 0x3c, 0x00, 0x40]
    storage = ML::FloatingStorage.from_bytes(
      source,
      2,
      ML::DType::F16
    )
    source[0] = 0xff
    storage.to_bytes.should eq(Bytes[0x00, 0x3c, 0x00, 0x40])

    exported = storage.to_bytes
    exported[1] = 0xff
    storage.read_u16_bits(0).should eq(0x3c00_u16)

    cloned = storage.clone
    cloned.write_u16_bits(0, 0x0000_u16)
    storage.read_u16_bits(0).should eq(0x3c00_u16)
  end

  it "supports zero elements without inventing bytes" do
    storage = ML::FloatingStorage.zeros(0, ML::DType::BF16)

    storage.numel.should eq(0)
    storage.byte_size.should eq(0)
    storage.to_bytes.should be_empty
  end

  it "rejects invalid dtype, size, access mode, and index before allocation" do
    expect_raises(ML::FloatingStorageError, /floating dtype/) do
      ML::FloatingStorage.zeros(1, ML::DType::I32)
    end
    expect_raises(ML::FloatingStorageError, /non-negative/) do
      ML::FloatingStorage.zeros(-1, ML::DType::F16)
    end
    expect_raises(ML::FloatingStorageError, /byte size overflow/) do
      ML::FloatingStorage.zeros(Int32::MAX, ML::DType::F32)
    end
    expect_raises(ML::FloatingStorageError, /byte limit/) do
      ML::FloatingStorage.zeros(
        ML::FloatingStorage::MAX_BYTES // 2 + 1,
        ML::DType::F16
      )
    end
    expect_raises(ML::FloatingStorageError, /byte length/) do
      ML::FloatingStorage.from_bytes(
        Bytes[0_u8],
        1,
        ML::DType::F16
      )
    end

    f32 = ML::FloatingStorage.zeros(1, ML::DType::F32)
    f16 = ML::FloatingStorage.zeros(1, ML::DType::F16)
    expect_raises(ML::FloatingStorageError, /F16 or BF16/) do
      f32.read_u16_bits(0)
    end
    expect_raises(ML::FloatingStorageError, /F32/) do
      f16.read_f32(0)
    end
    expect_raises(IndexError) do
      f16.read_u16_bits(1)
    end
  end
end
