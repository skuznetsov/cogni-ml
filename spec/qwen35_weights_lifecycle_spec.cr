require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen35_weights"

describe "Qwen35 weight mmap lifecycle" do
  it "keeps unregister_mmap idempotent for an unknown base" do
    base = Pointer(UInt8).new(0x1000_u64)

    ML::GGUF::Qwen35Metal.unregister_mmap(base).should be_false
    ML::GGUF::Qwen35Metal.unregister_mmap(base).should be_false
  end

  it "closes mmap-backed weights idempotently" do
    model_path = ENV["QWEN35_LIFECYCLE_MODEL"]? ||
                 "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    pending!("Qwen lifecycle model not present") unless File.exists?(model_path)

    {% unless flag?(:cpu_only) %}
      ML::GGUF::Qwen35Metal.available?.should be_true
      gguf = ML::GGUF::GGUFFile.new(model_path)
      begin
        base, size = gguf.mmap_region.not_nil!
        ML::GGUF::Qwen35Metal.register_mmap(base, size)
        ML::GGUF::Qwen35Metal.unregister_mmap(base).should be_true
        ML::GGUF::Qwen35Metal.unregister_mmap(base).should be_false
      ensure
        gguf.close
      end
    {% end %}

    weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
    closed = Channel(Nil).new(4)
    4.times do
      spawn do
        weights.close
        closed.send(nil)
      end
    end
    4.times { closed.receive }
    weights.close
    weights.finalize
  end

  it "fails closed instead of replacing a strict mmap owner" do
    {% if flag?(:cpu_only) %}
      pending!("Metal mmap registration is unavailable in cpu_only builds")
    {% else %}
      model_path = ENV["QWEN35_LIFECYCLE_MODEL"]? ||
                   "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
      pending!("Qwen lifecycle model not present") unless File.exists?(model_path)
      ML::GGUF::Qwen35Metal.available?.should be_true

      first = ML::GGUF::GGUFFile.new(model_path)
      second = ML::GGUF::GGUFFile.new(model_path)
      begin
        first_base, first_size = first.mmap_region.not_nil!
        second_base, second_size = second.mmap_region.not_nil!
        first_base.address.should_not eq(second_base.address)

        page = 16_384_u64
        first_aligned = (first_size // page) * page
        strict_view = ML::GGUF::Qwen35MmapWeightView.new(
          first_base.address,
          first_aligned.to_i64,
          first_aligned.to_i64,
        )
        ML::GGUF::Qwen35Metal.register_mmap_views(first_base, first_size, [strict_view], strict: true)
        expect_raises(Exception, /strict Metal mmap registration/) do
          ML::GGUF::Qwen35Metal.register_mmap(second_base, second_size)
        end
        ML::GGUF::Qwen35Metal.unregister_mmap(first_base).should be_true
        ML::GGUF::Qwen35Metal.register_mmap(second_base, second_size)
        ML::GGUF::Qwen35Metal.unregister_mmap(second_base).should be_true
      ensure
        first.close
        second.close
      end
    {% end %}
  end
end
