require "spec"
require "../src/ml/gguf/qwen35_qlora"

private def qwen35_paired_head(rows : Array(Float32)) : ML::GGUF::QuantWeight
  bytes = Bytes.new(rows.size * sizeof(Float32))
  rows.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * sizeof(Float32), sizeof(Float32)])
  end
  ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, rows.size // 2, 2)
end

describe ML::GGUF::Qwen35QLoRA::PairedObjective do
  it "keeps the weaker literal active after equal direct and negated margin gains" do
    objective = ML::GGUF::Qwen35QLoRA::PairedObjective
    weight = 0.8_f32
    floor = 0.5_f32

    before = objective.calculate(1.25_f64, 0.1_f32, -0.3_f32, weight, floor)
    after = objective.calculate(1.25_f64, 0.3_f32, -0.1_f32, weight, floor)

    before[:pair].should be_close(0.8, 1e-6)
    after[:pair].should be_close(0.6, 1e-6)
    after[:pair].should be > 0.0
    after[:loss].should be_close(1.73, 1e-6)
    after[:anchor_scale].should eq(0_f32)
    after[:mate_scale].should eq(-weight)
  end

  it "splits a tied hinge derivative and is inactive once both margins pass" do
    objective = ML::GGUF::Qwen35QLoRA::PairedObjective
    tied = objective.calculate(2.0_f64, 0.2_f32, 0.2_f32, 0.8_f32, 0.5_f32)
    tied[:pair].should be_close(0.3, 1e-6)
    tied[:loss].should be_close(2.24, 1e-6)
    tied[:anchor_scale].should eq(-0.4_f32)
    tied[:mate_scale].should eq(-0.4_f32)

    passed = objective.calculate(2.0_f64, 0.5_f32, 0.9_f32, 0.8_f32, 0.5_f32)
    passed[:pair].should eq(0.0)
    passed[:loss].should eq(2.0)
    passed[:anchor_scale].should eq(0_f32)
    passed[:mate_scale].should eq(0_f32)
  end

  it "matches RMS-normalized hidden derivatives through the weaker two-choice margin" do
    objective = ML::GGUF::Qwen35QLoRA::PairedObjective
    hidden = [0.8_f32, -0.4_f32, 1.1_f32]
    norm = [1.0_f32, 0.8_f32, 1.2_f32]
    correct_head = [-0.7_f32, 0.1_f32, -0.3_f32]
    other_head = [0.8_f32, -0.2_f32, 0.3_f32]
    eps = 1e-5_f32
    weight = 0.7_f32
    floor = 0.5_f32
    step = 0.001_f32

    result = objective.two_choice_margin_and_hidden_gradient(hidden, norm, correct_head, other_head, eps)
    result[:margin].should be < 0_f32
    base = objective.calculate(0.4_f64, 1.0_f32, result[:margin], weight, floor)
    base[:mate_scale].should eq(-weight)

    hidden.each_index do |index|
      plus = hidden.dup
      minus = hidden.dup
      plus[index] += step
      minus[index] -= step
      upper_margin = objective.two_choice_margin_and_hidden_gradient(plus, norm, correct_head, other_head, eps, backward: false)[:margin]
      lower_margin = objective.two_choice_margin_and_hidden_gradient(minus, norm, correct_head, other_head, eps, backward: false)[:margin]
      upper = objective.calculate(0.4_f64, 1.0_f32, upper_margin, weight, floor)[:loss]
      lower = objective.calculate(0.4_f64, 1.0_f32, lower_margin, weight, floor)[:loss]
      analytic = base[:mate_scale] * result[:gradient][index]
      analytic.should be_close((upper - lower).to_f32 / (2_f32 * step), 0.002)
    end
  end

  it "maps the first literal target to its selected output row" do
    rows = [
      200_f32, -300_f32,
      -0.4_f32, 0.3_f32,
      700_f32, 800_f32,
      0.2_f32, -0.7_f32,
    ]
    norm = [0.9_f32, 1.1_f32]
    eps = 1e-4_f32
    hidden = [1000_f32, -900_f32, 0.7_f32, -1.2_f32, -600_f32, 800_f32]
    # Token 2 is deliberately a different, unselected continuation target.
    example = ML::GGUF::Qwen35QLoRA::Example.new(hidden, [1, 2], 1)
    head = ML::GGUF::Qwen35QLoRA::CompletionLoss.new(qwen35_paired_head(rows), norm, eps, choices: {1, 3})

    first = head.choice_margin_and_hidden_gradient(example.input, example.first, example.targets[0])
    expected = ML::GGUF::Qwen35QLoRA::PairedObjective.two_choice_margin_and_hidden_gradient(
      hidden[2, 2], norm, rows[2, 2], rows[6, 2], eps)

    first[:margin].should be_close(expected[:margin], 1e-6)
    first[:gradient].should eq(expected[:gradient])
    expect_raises(ArgumentError) do
      head.choice_margin_and_hidden_gradient(example.input, example.first, example.targets[1])
    end
  end
end
