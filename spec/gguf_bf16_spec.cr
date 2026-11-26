require "spec"
require "../src/ml/gguf/quant_matmul"

describe ML::GGUF::TensorType::BF16 do
  it "reads BF16 tensor sizes and preserves the Float32 representation" do
    type = ML::GGUF::TensorType.from_value(30_u32)
    type.should eq(ML::GGUF::TensorType::BF16)
    type.block_bytes.should eq(2)
    type.block_elements.should eq(1)
    info = ML::GGUF::TensorInfo.new("bf16", [3_i64, 2_i64], type, 0_u64)
    info.data_bytes.should eq(12)

    weights = Bytes[0x80, 0x3f, 0x20, 0xc0, 0x00, 0x80, 0x01, 0x00]
    values = ML::GGUF::Dequant.dequantize(weights, type, 4)
    values.map { |value| value.unsafe_as(UInt32) }.should eq([
      0x3f800000_u32, 0xc0200000_u32, 0x80000000_u32, 0x00010000_u32,
    ])
  end

  it "multiplies BF16 weights with a Float32 bias" do
    weights = Bytes[0x80, 0x3f, 0x20, 0xc0]
    ML::GGUF::QuantMatmul.matmul_add(
      [2_f32, 4_f32], 1, 2, weights, ML::GGUF::TensorType::BF16, 1, [0.5_f32],
    ).should eq([-7.5_f32])
  end
end
