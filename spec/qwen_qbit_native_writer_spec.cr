require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_native_writer"

private def qbit_write_f32(bytes : Bytes, offset : Int32, value : Float32) : Nil
  bits = value.unsafe_as(UInt32)
  4.times { |i| bytes[offset + i] = ((bits >> (i * 8)) & 0xff).to_u8 }
end

private def qbit_p7_fixture : ML::GGUF::QwenQBitGaussianCodec::Encoded
  # Two already-transposed 8-code tiles. The seven bytes after each header are
  # the retained QBit streams; the Native writer must add a zero LSB stream.
  payload = Bytes.new(30, 0_u8)
  qbit_write_f32(payload, 0, 1.5_f32)
  qbit_write_f32(payload, 4, 0.25_f32)
  payload[8, 7].copy_from(Bytes[0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xcc])
  qbit_write_f32(payload, 15, -2.5_f32)
  qbit_write_f32(payload, 19, 0.5_f32)
  payload[23, 7].copy_from(Bytes[0xff, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x33])
  ML::GGUF::QwenQBitGaussianCodec::Encoded.new(16, 8, 7, payload)
end

describe ML::GGUF::QwenQBitNativeWriter do
  writer = ML::GGUF::QwenQBitNativeWriter

  it "writes revision-zero Native columns with plane-major QBit payloads" do
    record = ML::GGUF::QwenQBitNativeWriter::Record.new(42_u64, 3_i32, 4_u8, qbit_p7_fixture)
    bytes = writer.encode([record])

    # ClickHouse-generated reference for the same two rows. This fixture checks
    # the full block framing and that no Array -> QBit transpose is performed.
    expected_hex = "08020863616368655f69640655496e7436342a000000000000002a00000000000000056c6179657205496e7433320300000003000000046b696e640555496e743804040474696c650655496e74333200000000010000000b76616c75655f636f756e740655496e74313608000800046d65616e07466c6f617433320000c03f000020c0057369676d6107466c6f617433320000803e0000003f05636f6465730d5142697428496e74382c20382900ff00ff00ff00ff00fff00fcc330000"
    bytes.hexstring.should eq(expected_hex)
  end

  it "fails closed on a precision that cannot populate a p7 Native contract" do
    encoded = ML::GGUF::QwenQBitGaussianCodec::Encoded.new(8, 8, 6, Bytes.new(14, 0_u8))
    expect_raises(ArgumentError, /precision/) do
      writer.encode([ML::GGUF::QwenQBitNativeWriter::Record.new(1_u64, 0_i32, 0_u8, encoded)])
    end
  end
end
