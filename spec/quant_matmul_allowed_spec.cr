require "./spec_helper"
require "../src/ml/gguf/quant_matmul"

private def f32_bytes(values : Array(Float32)) : Bytes
  io = IO::Memory.new
  values.each { |v| io.write_bytes(v, IO::ByteFormat::LittleEndian) }
  io.to_slice
end

private def q8_0_bytes(rows : Array(Array(Int8))) : Bytes
  io = IO::Memory.new
  rows.each do |row|
    raise "Q8_0 fixture row must have 32 values" unless row.size == 32

    io.write_bytes(0x3c00_u16, IO::ByteFormat::LittleEndian) # fp16(1.0)
    row.each { |v| io.write_byte(v.to_u8!) }
  end
  io.to_slice
end

private def q6_k_single_value_rows(values : Array(Int32)) : Bytes
  io = IO::Memory.new
  values.each do |value|
    raise "Q6_K fixture value out of encodable range" unless value >= -32 && value <= 31

    q = value + 32
    ql = Bytes.new(128, 0_u8)
    qh = Bytes.new(64, 0_u8)
    scales = Bytes.new(16, 0_u8)
    ql[0] = (q & 0x0F).to_u8
    qh[0] = (((q >> 4) & 0x03) << 0).to_u8
    scales[0] = 1_u8 # first 16 q1 lanes use scale 1

    io.write(ql)
    io.write(qh)
    io.write(scales)
    io.write_bytes(0x3c00_u16, IO::ByteFormat::LittleEndian) # fp16(1.0)
  end
  io.to_slice
end

private def restricted_top1(logits : Array(Float32), allowed : Array(Int32)) : {Int32, Float32}
  best_id = allowed[0]
  best_logit = logits[best_id]
  allowed.each do |id|
    if logits[id] > best_logit
      best_id = id
      best_logit = logits[id]
    end
  end
  {best_id, best_logit}
end

describe ML::GGUF::QuantMatmul do
  describe ".row_bytes" do
    it "reports row stride for scalar and block quantized tensor types" do
      ML::GGUF::QuantMatmul.row_bytes(ML::GGUF::TensorType::F32, 3).should eq(12)
      ML::GGUF::QuantMatmul.row_bytes(ML::GGUF::TensorType::F16, 3).should eq(6)
      ML::GGUF::QuantMatmul.row_bytes(ML::GGUF::TensorType::Q4_K, 256).should eq(144)
      ML::GGUF::QuantMatmul.row_bytes(ML::GGUF::TensorType::Q4_K, 257).should eq(288)
      ML::GGUF::QuantMatmul.row_bytes(ML::GGUF::TensorType::Q6_K, 256).should eq(210)
    end
  end

  describe ".top1_allowed" do
    it "matches full F32 matmul restricted to allowed row ids" do
      x = [1.0_f32, -2.0_f32, 0.5_f32]
      weights = [
        0.25_f32, 0.50_f32, -1.0_f32, # row 0 -> -1.25
        1.50_f32, -0.25_f32, 0.0_f32, # row 1 -> 2.00
        -1.0_f32, -2.0_f32, 2.0_f32, # row 2 -> 4.00
        0.0_f32, 0.25_f32, 8.0_f32,  # row 3 -> 3.50
      ]
      raw = f32_bytes(weights)
      out_dim = 4
      full = ML::GGUF::QuantMatmul.matmul_add(
        x, 1, 3, raw, ML::GGUF::TensorType::F32, out_dim, Array.new(out_dim, 0.0_f32)
      )

      allowed = [0, 3, 1]
      expected_id, expected_logit = restricted_top1(full, allowed)
      actual_id, actual_logit = ML::GGUF::QuantMatmul.top1_allowed(
        x, 3, raw, ML::GGUF::TensorType::F32, out_dim, allowed
      )

      actual_id.should eq(expected_id)
      actual_logit.should be_close(expected_logit, 1.0e-6_f32)
    end

    it "uses the lower token id as the deterministic tie breaker" do
      x = [1.0_f32, 1.0_f32]
      raw = f32_bytes([
        1.0_f32, 0.0_f32, # row 0 -> 1.0
        0.0_f32, 1.0_f32, # row 1 -> 1.0
      ])

      id, logit = ML::GGUF::QuantMatmul.top1_allowed(
        x, 2, raw, ML::GGUF::TensorType::F32, 2, [1, 0]
      )

      id.should eq(0)
      logit.should be_close(1.0_f32, 1.0e-6_f32)
    end

    it "slices quantized Q8_0 rows by block stride" do
      x = Array.new(32, 1.0_f32)
      row0 = Array.new(32, 0_i8)
      row1 = Array.new(32, 0_i8)
      row0[0] = 1_i8
      row1[0] = 2_i8
      raw = q8_0_bytes([row0, row1])

      id, logit = ML::GGUF::QuantMatmul.top1_allowed(
        x, 32, raw, ML::GGUF::TensorType::Q8_0, 2, [0, 1]
      )

      id.should eq(1)
      logit.should be_close(2.0_f32, 1.0e-6_f32)
    end

    it "slices quantized Q6_K rows by K-block stride" do
      x = Array.new(256, 0.0_f32)
      x[0] = 1.0_f32
      raw = q6_k_single_value_rows([1, 3, 2])

      id, logit = ML::GGUF::QuantMatmul.top1_allowed(
        x, 256, raw, ML::GGUF::TensorType::Q6_K, 3, [0, 2, 1]
      )

      id.should eq(1)
      logit.should be_close(3.0_f32, 1.0e-6_f32)
    end

    it "validates allowed row ids and activation length" do
      raw = f32_bytes([
        1.0_f32, 0.0_f32,
        0.0_f32, 1.0_f32,
      ])

      expect_raises(ArgumentError, /allowed_ids/) do
        ML::GGUF::QuantMatmul.top1_allowed([1.0_f32, 2.0_f32], 2, raw, ML::GGUF::TensorType::F32, 2, [] of Int32)
      end

      expect_raises(ArgumentError, /out of range/) do
        ML::GGUF::QuantMatmul.top1_allowed([1.0_f32, 2.0_f32], 2, raw, ML::GGUF::TensorType::F32, 2, [2])
      end

      expect_raises(ArgumentError, /x size/) do
        ML::GGUF::QuantMatmul.top1_allowed([1.0_f32], 2, raw, ML::GGUF::TensorType::F32, 2, [0])
      end
    end
  end
end
