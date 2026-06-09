require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_08B_ALLOWED = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
QWEN_2B_ALLOWED  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf"

private def spec_restricted_top1(logits : Array(Float32), allowed : Array(Int32)) : {Int32, Float32}
  best_id = allowed[0]
  best = logits[best_id]
  allowed.each do |id|
    value = logits[id]
    if value > best || (value == best && id < best_id)
      best = value
      best_id = id
    end
  end
  {best_id, best}
end

private def spec_restore_env(name : String, old : String?)
  if old
    ENV[name] = old
  else
    ENV.delete(name)
  end
end

describe ML::GGUF::Qwen35CPU, "allowed lm-head route" do
  it "matches full logits and preserves decode state after Metal hidden readback" do
    pending!("0.8B model not present") unless File.exists?(QWEN_08B_ALLOWED)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_08B_ALLOWED)
    weights.output.type.q8_0?.should be_true
    hp = weights.hparams

    old_wave = ENV["QWEN35_DECODE_WAVE_OFF"]?
    old_head = ENV["QWEN35_HEAD_TOP1_FUSED"]?
    ENV.delete("QWEN35_DECODE_WAVE_OFF")
    ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
    begin
      full_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      full_logits = ML::GGUF::Qwen35CPU.forward(weights, 0, 0, full_state)
      full_top = full_logits.index(full_logits.max).not_nil!.to_i32

      allowed = [0_i32, 1_i32, 2_i32, 198_i32, 606_i32].reject { |id| id == full_top }
      expected_id, expected_logit = spec_restricted_top1(full_logits, allowed)

      allowed_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      actual_id, actual_logit = ML::GGUF::Qwen35CPU.forward_top1_allowed(weights, 0, 0, allowed_state, allowed)
      actual_id.should eq(expected_id)
      actual_logit.should be_close(expected_logit, 1.0e-3_f32)

      full_next = ML::GGUF::Qwen35CPU.forward_top1(weights, 100, 1, full_state)
      allowed_next = ML::GGUF::Qwen35CPU.forward_top1(weights, 100, 1, allowed_state)
      allowed_next[0].should eq(full_next[0])
      allowed_next[1].should be_close(full_next[1], 1.0e-3_f32)
    ensure
      spec_restore_env("QWEN35_DECODE_WAVE_OFF", old_wave)
      spec_restore_env("QWEN35_HEAD_TOP1_FUSED", old_head)
    end
  end

  it "honors the CPU threshold on a Q6_K output head" do
    pending!("2B model not present") unless File.exists?(QWEN_2B_ALLOWED)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_2B_ALLOWED)
    weights.output.type.q6_k?.should be_true
    hp = weights.hparams

    old_wave = ENV["QWEN35_DECODE_WAVE_OFF"]?
    old_head = ENV["QWEN35_HEAD_TOP1_FUSED"]?
    old_cpu_max = ENV["QWEN35_ALLOWED_HEAD_CPU_MAX"]?
    ENV.delete("QWEN35_DECODE_WAVE_OFF")
    ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
    ENV["QWEN35_ALLOWED_HEAD_CPU_MAX"] = "16"
    begin
      full_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      full_logits = ML::GGUF::Qwen35CPU.forward(weights, 0, 0, full_state)
      full_top = full_logits.index(full_logits.max).not_nil!.to_i32

      allowed = [0_i32, 1_i32, 2_i32, 198_i32, 606_i32].reject { |id| id == full_top }
      expected_id, expected_logit = spec_restricted_top1(full_logits, allowed)

      allowed_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      actual_id, actual_logit = ML::GGUF::Qwen35CPU.forward_top1_allowed(weights, 0, 0, allowed_state, allowed)
      actual_id.should eq(expected_id)
      actual_logit.should be_close(expected_logit, 1.0e-3_f32)

      full_next = ML::GGUF::Qwen35CPU.forward_top1(weights, 100, 1, full_state)
      allowed_next = ML::GGUF::Qwen35CPU.forward_top1(weights, 100, 1, allowed_state)
      allowed_next[0].should eq(full_next[0])
      allowed_next[1].should be_close(full_next[1], 1.0e-3_f32)
    ensure
      spec_restore_env("QWEN35_DECODE_WAVE_OFF", old_wave)
      spec_restore_env("QWEN35_HEAD_TOP1_FUSED", old_head)
      spec_restore_env("QWEN35_ALLOWED_HEAD_CPU_MAX", old_cpu_max)
    end
  end
end
