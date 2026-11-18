require "spec"
require "../src/ml/gguf/qwen35_qlora"

private def qwen35_choice_weight(rows : Array(Float32)) : ML::GGUF::QuantWeight
  bytes = Bytes.new(rows.size * sizeof(Float32))
  rows.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * sizeof(Float32), sizeof(Float32)])
  end
  ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, rows.size // 2, 2)
end

private def qwen35_choice_q6_weight : ML::GGUF::QuantWeight
  vocab = 4
  width = 512
  block_bytes = 210
  raw = Bytes.new(vocab * (width // 256) * block_bytes, 0_u8)
  d_values = [0x2400_u16, 0x2800_u16, 0x2c00_u16, 0x3000_u16,
              0x2400_u16, 0x2800_u16, 0x2c00_u16, 0x3000_u16]
  (vocab * 2).times do |block|
    offset = block * block_bytes
    raw[offset, 128].fill(0x44_u8)
    raw[offset + 128, 64].fill(0xaa_u8)
    16.times do |scale|
      raw[offset + 192 + scale] = (1 + ((block + scale // 4) % 3)).to_u8
    end
    IO::ByteFormat::LittleEndian.encode(d_values[block], raw[offset + 208, 2])
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::Q6_K, vocab, width)
end

private def qwen35_choice_expected_loss(
  hidden : Array(Float32), norm : Array(Float32), eps : Float32,
  false_row : Array(Float32), true_row : Array(Float32), target : Int32,
) : Float64
  square_mean = (hidden[0] * hidden[0] + hidden[1] * hidden[1]) / 2_f32
  inv_rms = 1_f32 / Math.sqrt(square_mean + eps)
  normalized = [
    hidden[0] * inv_rms * norm[0],
    hidden[1] * inv_rms * norm[1],
  ]
  false_logit = normalized[0] * false_row[0] + normalized[1] * false_row[1]
  true_logit = normalized[0] * true_row[0] + normalized[1] * true_row[1]
  max_logit = Math.max(false_logit, true_logit)
  logsumexp = max_logit.to_f64 + Math.log(Math.exp((false_logit - max_logit).to_f64) +
                                          Math.exp((true_logit - max_logit).to_f64))
  chosen = (target == 1 ? false_logit : true_logit).to_f64
  logsumexp - chosen
end

describe ML::GGUF::Qwen35QLoRA::CompletionLoss do
  it "uses only the selected pair for analytic loss and RMS-normalized gradients" do
    # The selected classes are IDs 1 and 3. ID 2 would dominate a full-vocabulary
    # softmax, so matching the two-logit calculation proves it is excluded.
    weight = qwen35_choice_weight([
      0.13_f32, 0.4_f32,
      0.3_f32, -0.7_f32,
      1200_f32, -900_f32,
      -0.4_f32, 0.23_f32,
    ])
    norm = [0.7_f32, 1.3_f32]
    eps = 1e-4_f32
    loss = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(weight, norm, eps, choices: {1, 3})
    hidden = [12_f32, -14_f32, 0.55_f32, -0.92_f32]

    {1, 3}.each do |target|
      value, _ = loss.calculate(hidden, 1, [target], backward: false)
      expected = qwen35_choice_expected_loss(
        [hidden[2], hidden[3]], norm, eps,
        [0.3_f32, -0.7_f32], [-0.4_f32, 0.23_f32], target)
      value.should be_close(expected, 0.00001)
    end

    selected_loss, grad = loss.calculate(hidden, 1, [3], backward: true)
    full_vocab = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(weight, norm, eps)
    full_vocab_loss = full_vocab.calculate(hidden, 1, [3], backward: false)[0]
    full_vocab_loss.should be > selected_loss + 1000.0
    grad[0, 2].should eq([0_f32, 0_f32])
    step = 0.001_f32
    (2...4).each do |index|
      plus = hidden.dup
      minus = hidden.dup
      plus[index] += step
      minus[index] -= step
      upper = loss.calculate(plus, 1, [3], backward: false)[0]
      lower = loss.calculate(minus, 1, [3], backward: false)[0]
      grad[index].should be_close((upper - lower) / (2 * step), 0.0002)
    end

    changed_unrelated = qwen35_choice_weight([
      -2500_f32, 1700_f32,
      0.3_f32, -0.7_f32,
      -8000_f32, 2300_f32,
      -0.4_f32, 0.23_f32,
    ])
    changed_loss = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(changed_unrelated, norm, eps, choices: {1, 3})
    altered_value, altered_grad = changed_loss.calculate(hidden, 1, [3], backward: true)
    altered_value.should eq(selected_loss)
    altered_grad.should eq(grad)
  end

  it "rejects a target outside the selected pair" do
    weight = qwen35_choice_weight([
      0.13_f32, 0.4_f32,
      0.3_f32, -0.7_f32,
      1200_f32, -900_f32,
      -0.4_f32, 0.23_f32,
    ])
    loss = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(weight, [0.7_f32, 1.3_f32], 1e-4_f32, choices: {1, 3})

    expect_raises(ArgumentError) do
      loss.calculate([0.55_f32, -0.92_f32], 0, [2], backward: false)
    end
  end

  it "reads selected Q6_K rows across distinct two-block row offsets" do
    weight = qwen35_choice_q6_weight
    width = 512
    dense = ML::GGUF::Dequant.dequantize(weight.raw, weight.type, 4 * width)
    dense[width].should_not eq(dense[width + 256])
    dense[3 * width].should_not eq(dense[3 * width + 256])

    hidden = Array(Float32).new(width) { |i| i.even? ? 0.1_f32 : -0.099_f32 }
    norm = Array(Float32).new(width, 1_f32)
    eps = 1e-5_f32
    sum_squares = hidden.sum { |value| value * value }
    inv_rms = 1_f32 / Math.sqrt(sum_squares / width + eps)
    false_logit = 0_f32
    true_logit = 0_f32
    width.times do |i|
      normalized = hidden[i] * inv_rms
      false_logit += normalized * dense[width + i]
      true_logit += normalized * dense[3 * width + i]
    end
    max_logit = Math.max(false_logit, true_logit)
    logsumexp = max_logit.to_f64 + Math.log(Math.exp((false_logit - max_logit).to_f64) +
                                            Math.exp((true_logit - max_logit).to_f64))
    loss = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(weight, norm, eps, choices: {1, 3})

    {1, 3}.each do |target|
      chosen = (target == 1 ? false_logit : true_logit).to_f64
      expected = logsumexp - chosen
      actual = loss.calculate(hidden, 0, [target], backward: false)[0]
      actual.should be_close(expected, 0.001)
    end
  end
end
