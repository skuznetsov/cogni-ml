require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"

QWEN_9B_TOP2 = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

private def spec_top2_from_logits(logits : Array(Float32)) : {Int32, Float32, Int32, Float32}
  best = -Float32::INFINITY
  second = -Float32::INFINITY
  best_id = 0_i32
  second_id = 0_i32
  logits.each_with_index do |v, id|
    id32 = id.to_i32
    if v > best || (v == best && id32 < best_id)
      second = best
      second_id = best_id
      best = v
      best_id = id32
    elsif id32 != best_id && (v > second || (v == second && id32 < second_id))
      second = v
      second_id = id32
    end
  end
  {best_id, best, second_id, second}
end

describe ML::GGUF::Qwen35CPU do
  it "matches full logits for Metal decode top2" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_TOP2)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_TOP2)
    hp = weights.hparams
    prefix = [1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32, 7_i32, 8_i32]
    next_token = 42_i32

    base = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(base, hp)
    ML::GGUF::Qwen35CPU.prefill_tokens(weights, prefix, 0, base)

    full_state = base.fork
    top2_state = base.fork

    expected = spec_top2_from_logits(ML::GGUF::Qwen35CPU.forward(weights, next_token, prefix.size, full_state))
    actual = ML::GGUF::Qwen35CPU.forward_top2(weights, next_token, prefix.size, top2_state)

    actual[0].should eq(expected[0])
    actual[2].should eq(expected[2])
    actual[1].should be_close(expected[1], 1.0e-3)
    actual[3].should be_close(expected[3], 1.0e-3)
  end
end
