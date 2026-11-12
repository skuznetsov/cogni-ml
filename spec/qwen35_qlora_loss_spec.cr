require "spec"
require "../src/ml/gguf/qwen35_qlora"

describe ML::GGUF::Qwen35QLoRA::CompletionLoss do
  it "differentiates masked completion loss through the frozen normalized output head" do
    head = [0.31_f32, -0.27_f32, -0.51_f32, 0.4_f32, 0.6_f32, 0.32_f32]
    bytes = Bytes.new(head.size * sizeof(Float32))
    head.each_with_index do |v, i|
      IO::ByteFormat::LittleEndian.encode(v, bytes[i * sizeof(Float32), sizeof(Float32)])
    end
    weight = ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, 3, 2)
    loss = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(weight, [0.7_f32, 1.2_f32], 1e-6_f32)
    hidden = [0.6_f32, -0.24_f32, 0.33_f32, -0.39_f32, -0.2_f32, 0.8_f32]
    labels = [2, 0]
    value, grad = loss.calculate(hidden, 1, labels, backward: true)
    value.should be > 0.0
    grad[0, 2].should eq([0_f32, 0_f32])
    h = 0.001_f32
    (2...hidden.size).each do |i|
      plus = hidden.dup
      minus = hidden.dup
      plus[i] += h
      minus[i] -= h
      upper = loss.calculate(plus, 1, labels, backward: false)[0]
      lower = loss.calculate(minus, 1, labels, backward: false)[0]
      grad[i].should be_close((upper - lower) / (2 * h), 0.0002)
    end
  end
end
