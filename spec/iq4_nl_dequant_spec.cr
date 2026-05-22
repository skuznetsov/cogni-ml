require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/dequant"
require "../src/ml/gguf/quant_matmul"

private IQ4_NL_KVALUES = [
  -127_f32, -104_f32, -83_f32, -65_f32,
   -49_f32,  -35_f32, -22_f32, -10_f32,
     1_f32,   13_f32,  25_f32,  38_f32,
    53_f32,   69_f32,  89_f32, 113_f32,
]

private def append_f16_le(bytes : Array(UInt8), value : UInt16)
  bytes << (value & 0xFF).to_u8
  bytes << ((value >> 8) & 0xFF).to_u8
end

private def bytes_of(raw : Array(UInt8)) : Bytes
  Bytes.new(raw.size) { |i| raw[i] }
end

describe ML::GGUF::TensorType do
  it "describes IQ4_NL block geometry" do
    type = ML::GGUF::TensorType::IQ4_NL
    type.value.should eq(20)
    type.name.should eq("IQ4_NL")
    type.block_elements.should eq(32)
    type.block_bytes.should eq(18)
  end
end

describe ML::GGUF::Dequant do
  describe "dequantize_iq4_nl" do
    it "matches llama.cpp nibble order for a full synthetic block" do
      raw = [] of UInt8
      append_f16_le(raw, 0x3C00_u16) # 1.0
      16.times do |j|
        low = j
        high = 15 - j
        raw << (low | (high << 4)).to_u8
      end

      out = ML::GGUF::Dequant.dequantize(
        bytes_of(raw),
        ML::GGUF::TensorType::IQ4_NL,
        32
      )

      out[0, 16].should eq(IQ4_NL_KVALUES)
      out[16, 16].should eq(IQ4_NL_KVALUES.reverse)
    end

    it "handles partial tail blocks without writing past n" do
      raw = [] of UInt8
      append_f16_le(raw, 0x3C00_u16) # 1.0
      16.times { |j| raw << (j | ((15 - j) << 4)).to_u8 }
      append_f16_le(raw, 0xB800_u16) # -0.5
      16.times { raw << 0x08_u8 }    # low=8 -> 1, high=0 -> -127

      out = ML::GGUF::Dequant.dequantize_iq4_nl(bytes_of(raw), 40)

      out.size.should eq(40)
      out[0].should eq(-127_f32)
      out[15].should eq(113_f32)
      out[16].should eq(113_f32)
      out[31].should eq(-127_f32)
      out[32, 8].should eq(Array.new(8, -0.5_f32))
    end
  end
end

describe ML::GGUF::QuantMatmul do
  it "matches bulk IQ4_NL dequant plus dense matmul on synthetic rows" do
    in_dim = 32
    out_dim = 2
    rows = 2
    raw = [] of UInt8

    append_f16_le(raw, 0x3C00_u16) # row 0 scale: 1.0
    16.times { |j| raw << (j | ((15 - j) << 4)).to_u8 }
    append_f16_le(raw, 0x3800_u16) # row 1 scale: 0.5
    16.times { |j| raw << ((15 - j) | (j << 4)).to_u8 }

    x = Array(Float32).new(rows * in_dim) do |i|
      (((i * 17) % 23) - 11).to_f32 / 7.0_f32
    end
    bias = [0.25_f32, -0.75_f32]

    mine = ML::GGUF::QuantMatmul.matmul_add(
      x, rows, in_dim, bytes_of(raw), ML::GGUF::TensorType::IQ4_NL, out_dim, bias
    )

    dense_w = ML::GGUF::Dequant.dequantize_iq4_nl(bytes_of(raw), in_dim * out_dim)
    ref = Array(Float32).new(rows * out_dim, 0.0_f32)
    rows.times do |r|
      out_dim.times do |o|
        sum = bias[o].to_f64
        in_dim.times do |j|
          sum += x[r * in_dim + j].to_f64 * dense_w[o * in_dim + j].to_f64
        end
        ref[r * out_dim + o] = sum.to_f32
      end
    end

    mine.should eq(ref)
  end
end
