require "./spec_helper"
require "../src/ml/gguf/qwen_image21_block"

private def f32_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * 4)
  raw.to_unsafe.copy_from(values.to_unsafe.as(Pointer(UInt8)), raw.size)
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
end

private def bf16_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    encoded = (value.unsafe_as(UInt32) >> 16).to_u16
    raw[index * 2] = (encoded & 0xff).to_u8
    raw[index * 2 + 1] = (encoded >> 8).to_u8
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::BF16, out_dim, in_dim)
end

describe ML::GGUF::QuantMatmul do
  it "multiplies BF16 matrices without bulk dequantization" do
    weight = bf16_weight([
      1.0_f32, -2.0_f32, 0.5_f32,
      -0.25_f32, 4.0_f32, 2.0_f32,
    ], 2, 3)
    result = ML::GGUF::F32Backend.new.matmul(
      [2.0_f32, -1.0_f32, 0.5_f32], 1, weight, [0.25_f32, -0.5_f32]
    )
    result.should eq([4.5_f32, -4.0_f32])
  end
end

describe ML::GGUF::QwenImage21BlockCPU do
  it "matches the upstream PyTorch equations for a mixed text/image block" do
    config = ML::GGUF::QwenImage21BlockConfig.new(
      hidden_dim: 6,
      heads: 1,
      head_dim: 6,
      intermediate_dim: 5,
      axes_dims: StaticArray[2, 2, 2],
    )

    matrix = ->(out_dim : Int32, in_dim : Int32, phase : Int32) do
      Array(Float32).new(out_dim * in_dim) do |index|
        (((index * 17 + phase * 13) % 29) - 14).to_f32 / 37.0_f32
      end
    end
    weights = ML::GGUF::QwenImage21BlockWeights.new(
      f32_weight(matrix.call(6, 6, 1), 6, 6),
      f32_weight(matrix.call(6, 6, 2), 6, 6),
      f32_weight(matrix.call(6, 6, 3), 6, 6),
      f32_weight(matrix.call(6, 6, 4), 6, 6),
      [1.0_f32, 0.9_f32, 1.1_f32, 0.8_f32, 1.2_f32, 0.95_f32],
      [0.85_f32, 1.05_f32, 0.9_f32, 1.15_f32, 0.8_f32, 1.1_f32],
      f32_weight(matrix.call(10, 6, 5), 10, 6),
      f32_weight(matrix.call(6, 5, 6), 6, 5),
    )

    hidden = Array(Float32).new(4 * 6) { |i| (((i * 11) % 23) - 11).to_f32 / 19.0_f32 }
    modulation = Array(Float32).new(4 * 4 * 6) { |i| (((i * 7) % 31) - 15).to_f32 / 101.0_f32 }
    positions = [
      StaticArray[0, 0, 0],
      StaticArray[1, 1, 1],
      StaticArray[2, -1, 0],
      StaticArray[2, 0, 0],
    ]
    image_ids = [-1, -1, 0, 0]

    result = ML::GGUF::QwenImage21BlockCPU.forward(
      hidden, 4, modulation, positions, image_ids, weights, config
    )

    # Golden values are produced by spec/support/qwen_image21_torch_reference.py.
    expected = [
      -0.57213414_f32, 0.02648832_f32, 0.52566266_f32, -0.09056844_f32, 0.55803943_f32, -0.10141043_f32,
      0.46424446_f32, -0.19664985_f32, 0.45106608_f32, -0.20789142_f32, 0.33902454_f32, -0.24440029_f32,
      0.32813126_f32, -0.31573537_f32, 0.24351224_f32, -0.34689835_f32, 0.26206315_f32, -0.42857623_f32,
      0.15305819_f32, -0.45203796_f32, 0.15367654_f32, -0.52929974_f32, 0.05194872_f32, -0.56617659_f32,
    ]
    result.zip(expected).each do |actual, reference|
      actual.should be_close(reference, 2e-5_f32)
    end
  end
end
