require "./spec_helper"
require "../src/ml/gguf/qwen3vl_text_reference"
require "../src/ml/gguf/qwen3vl_text_weights"
require "../src/ml/gguf/qwen3vl_text_block"

private def qwen3vl_ordered_bf16_bits(value : Float32) : Int32
  bits = (value.unsafe_as(UInt32) >> 16).to_i32
  (bits & 0x8000) == 0 ? (bits | 0x8000) : (~bits & 0xffff)
end

private def qwen3vl_bf16_bits(value : Float32) : UInt16
  (value.unsafe_as(UInt32) >> 16).to_u16
end

private def qwen3vl_bf16_mismatch?(left : Float32, right : Float32) : Bool
  qwen3vl_bf16_bits(left) != qwen3vl_bf16_bits(right)
end

describe "Qwen3VL BF16 exact mismatch guard" do
  it "distinguishes positive and negative zero encodings" do
    positive_zero = 0x0000_0000_u32.unsafe_as(Float32)
    negative_zero = 0x8000_0000_u32.unsafe_as(Float32)

    qwen3vl_bf16_bits(positive_zero).should_not eq(qwen3vl_bf16_bits(negative_zero))
    qwen3vl_bf16_mismatch?(positive_zero, negative_zero).should be_true
  end
end

private def qwen3vl_write_layer0_bf16(path : String, values : Array(Float32)) : Nil
  File.open(path, "wb") do |io|
    values.each do |value|
      io.write_bytes((value.unsafe_as(UInt32) >> 16).to_u16, IO::ByteFormat::LittleEndian)
    end
  end
end

# The official checkpoint and BF16 reference are intentionally external to the
# repository. Set both paths to run this numerical gate against the pinned
# Qwen-Image 2.1 revision. A two-token prefix isolates the first decoder block:
# in a causal model its output cannot depend on subsequent prompt tokens, while
# the second token exercises both keys in the attention softmax.
if ENV["QWEN3VL_TEXT_ENCODER_DIR"]? && ENV["QWEN3VL_TEXT_REFERENCE_DIR"]?
  describe "optional real Qwen3-VL first-layer parity" do
    it "matches the official first two tokens after decoder layer zero" do
      encoder_dir = ENV["QWEN3VL_TEXT_ENCODER_DIR"].not_nil!
      fixture_dir = ENV["QWEN3VL_TEXT_REFERENCE_DIR"].not_nil!
      reference = ML::GGUF::Qwen3VLTextReference.load(
        File.join(fixture_dir, "qwen_image21_text_reference.json")
      )
      reference.model_revision.should eq(ML::GGUF::Qwen3VLTextWeights::EXPECTED_MODEL_REVISION)
      reference.prompt.should eq("red cube")
      manifest = JSON.parse(File.read(File.join(fixture_dir, "qwen_image21_text_reference.json")))
      manifest["payload_sha256"].as_s.should eq(
        "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07"
      )
      prefix_tokens = 2
      reference.attention_mask[0, prefix_tokens].should eq([true, true])

      hidden_dim = ML::GGUF::Qwen3VLTextWeights::HIDDEN_SIZE
      scalar_count = prefix_tokens * hidden_dim
      first_input = reference.hidden_state(0)[0, scalar_count]
      expected = reference.hidden_state(1)[0, scalar_count]
      config = ML::GGUF::Qwen3VLTextBlockConfig.new(
        hidden_dim: hidden_dim,
        heads: ML::GGUF::Qwen3VLTextWeights::NUM_ATTENTION_HEADS,
        kv_heads: ML::GGUF::Qwen3VLTextWeights::NUM_KEY_VALUE_HEADS,
        head_dim: ML::GGUF::Qwen3VLTextWeights::HEAD_DIM,
        intermediate_dim: ML::GGUF::Qwen3VLTextWeights::INTERMEDIATE_SIZE,
        attention_arithmetic: ML::GGUF::Qwen3VLTextBlockConfig::AttentionArithmetic::SdpaF32,
      )

      weights = ML::GGUF::Qwen3VLTextWeights.from_directory(encoder_dir)
      begin
        trace_dir = ENV["QWEN3VL_TEXT_LAYER0_TRACE_DIR"]?
        trace = trace_dir ? {} of String => Array(Float32) : nil
        actual = ML::GGUF::Qwen3VLTextBlock.forward(
          first_input, [true, true], weights.block_weights(0), config, trace: trace
        )
        actual.size.should eq(scalar_count)
        if trace_dir && trace
          raise ArgumentError.new("trace output directory already exists: #{trace_dir}") if Dir.exists?(trace_dir)
          Dir.mkdir_p(trace_dir)
          trace.each do |stage, values|
            qwen3vl_write_layer0_bf16(File.join(trace_dir, "#{stage}.bf16le"), values)
          end
        end
        if actual_output_path = ENV["QWEN3VL_TEXT_LAYER0_ACTUAL_BF16_OUT"]?
          raise ArgumentError.new("BF16 output already exists: #{actual_output_path}") if File.exists?(actual_output_path)
          qwen3vl_write_layer0_bf16(actual_output_path, actual)
        end
        mismatches = 0
        first_token_mismatches = 0
        one_ulp_mismatches = 0
        max_abs_error = 0.0_f32
        sum_squared_error = 0.0_f64
        sum_squared_expected = 0.0_f64
        actual.each_with_index do |value, index|
          value.finite?.should be_true
          expected[index].finite?.should be_true
          error = (value - expected[index]).abs
          max_abs_error = error if error > max_abs_error
          error64 = error.to_f64
          expected64 = expected[index].to_f64
          sum_squared_error += error64 * error64
          sum_squared_expected += expected64 * expected64
          if qwen3vl_bf16_mismatch?(value, expected[index])
            mismatches += 1
            first_token_mismatches += 1 if index < hidden_dim
            ulp_error = (qwen3vl_ordered_bf16_bits(value) - qwen3vl_ordered_bf16_bits(expected[index])).abs
            one_ulp_mismatches += 1 if ulp_error == 1
          end
        end
        rmse = Math.sqrt(sum_squared_error / scalar_count)
        relative_rms = Math.sqrt(sum_squared_error / (sum_squared_expected + 1e-30))
        mismatches.should eq(0),
          "two-token layer-0 BF16 mismatch: #{mismatches}/#{scalar_count} values, " +
          "first_token_mismatches=#{first_token_mismatches}, one_ulp_mismatches=#{one_ulp_mismatches}, " +
          "max_abs_error=#{max_abs_error}, rmse=#{rmse}, relative_rms=#{relative_rms}"
      ensure
        weights.close
      end
    end
  end
end
