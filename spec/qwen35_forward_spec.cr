require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_FWD = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35CPU, "full decoder forward" do
  pending!("9B model not present") unless File.exists?(QWEN_9B_FWD)

  it "produces finite logits at pos=0 for token 0" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)

    t0 = Time.instant
    logits = ML::GGUF::Qwen35CPU.forward(w, 0, 0, state)
    dt = Time.instant - t0
    puts "  [qwen35_forward] first-token latency: #{dt.total_milliseconds.round(1)} ms"

    logits.size.should eq(w.output.out_dim)  # vocab_size (≈248k)
    logits.all? { |v| v.finite? }.should be_true

    # Logits should have some spread (not all identical)
    maxv = logits.max
    minv = logits.min
    (maxv - minv).should be > 1.0_f32

    top = logits.index(maxv).not_nil!
    puts "  [qwen35_forward] top token id=#{top}, logit=#{maxv}"
  end

  it "produces different logits for different token inputs" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    state_a = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    state_b = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)

    logits_a = ML::GGUF::Qwen35CPU.forward(w, 100, 0, state_a)
    logits_b = ML::GGUF::Qwen35CPU.forward(w, 5000, 0, state_b)

    # Top-1 should almost certainly differ between input 100 and input 5000
    top_a = logits_a.index(logits_a.max).not_nil!
    top_b = logits_b.index(logits_b.max).not_nil!
    top_a.should_not eq(top_b)
  end
end
