#!/usr/bin/env crystal

require "json"
require "option_parser"
require "../src/ml/gguf/qwen3vl_text_reference"
require "../src/ml/gguf/qwen3vl_text_weights"
require "../src/ml/gguf/qwen3vl_text_block"

private PINNED_PROMPT         = "red cube"
private PINNED_PAYLOAD_SHA256 = "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07"
private TOKEN_COUNT           = 2

private def qwen3vl_prefix(values : Array(Float32), scalar_count : Int32) : Array(Float32)
  raise ArgumentError.new("reference state is shorter than the requested two-token prefix") if values.size < scalar_count
  values[0, scalar_count]
end

private def qwen3vl_layer_pair(
  weights : ML::GGUF::Qwen3VLTextWeights,
  layer_index : Int32,
  isolated_input : Array(Float32),
  composed_input : Array(Float32),
  config : ML::GGUF::Qwen3VLTextBlockConfig,
) : {isolated: Array(Float32), composed: Array(Float32)}
  block_weights = weights.block_weights(layer_index)
  {
    isolated: ML::GGUF::Qwen3VLTextBlock.forward(isolated_input, [true, true], block_weights, config),
    composed: ML::GGUF::Qwen3VLTextBlock.forward(composed_input, [true, true], block_weights, config),
  }
end

private def qwen3vl_bf16_bits(value : Float32) : UInt16
  (value.unsafe_as(UInt32) >> 16).to_u16
end

private def qwen3vl_metrics(actual : Array(Float32), expected : Array(Float32), hidden_dim : Int32)
  raise ArgumentError.new("actual and official state sizes differ") unless actual.size == expected.size
  mismatches = 0
  token0 = 0
  token1 = 0
  max_abs = 0.0_f64
  sum_squared_error = 0.0_f64
  sum_squared_expected = 0.0_f64

  actual.each_with_index do |value, index|
    reference = expected[index]
    raise ArgumentError.new("non-finite comparison value at index #{index}") unless value.finite? && reference.finite?
    if qwen3vl_bf16_bits(value) != qwen3vl_bf16_bits(reference)
      mismatches += 1
      token0 += 1 if index < hidden_dim
      token1 += 1 if index >= hidden_dim
    end
    error = (value.to_f64 - reference.to_f64).abs
    max_abs = error if error > max_abs
    sum_squared_error += error * error
    expected64 = reference.to_f64
    sum_squared_expected += expected64 * expected64
  end

  {
    mismatches: mismatches,
    token0:     token0,
    token1:     token1,
    max_abs:    max_abs,
    rel_rms:    Math.sqrt(sum_squared_error / (sum_squared_expected + 1e-30)),
  }
end

layers = 2
encoder_dir = ENV["QWEN3VL_TEXT_ENCODER_DIR"]?
reference_dir = ENV["QWEN3VL_TEXT_REFERENCE_DIR"]?

OptionParser.parse do |parser|
  parser.banner = "Usage: crystal run scripts/qwen3vl_text_layer_sweep.cr -- [--layers N] --text-encoder-dir DIR --reference-dir DIR"
  parser.on("--layers=N", "Number of leading decoder layers to sweep (1..36, default 2)") { |value| layers = value.to_i? || abort("--layers must be an integer") }
  parser.on("--text-encoder-dir=DIR", "Local Qwen3-VL text encoder checkpoint directory") { |value| encoder_dir = value }
  parser.on("--reference-dir=DIR", "Directory containing the pinned text reference fixture") { |value| reference_dir = value }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

abort("unexpected arguments: #{ARGV.join(" ")}") unless ARGV.empty?
abort("--layers must be in 1..36") unless 1 <= layers <= ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS
abort("provide --text-encoder-dir or QWEN3VL_TEXT_ENCODER_DIR") unless encoder_dir
abort("provide --reference-dir or QWEN3VL_TEXT_REFERENCE_DIR") unless reference_dir
encoder_path = encoder_dir.not_nil!
reference_path = reference_dir.not_nil!

begin
  manifest_path = File.join(reference_path, "qwen_image21_text_reference.json")
  reference = ML::GGUF::Qwen3VLTextReference.load(manifest_path)
  manifest = JSON.parse(File.read(manifest_path))
  payload_sha256 = manifest["payload_sha256"].as_s.downcase
  raise ArgumentError.new("fixture payload SHA-256 is not the pinned red-cube artifact") unless payload_sha256 == PINNED_PAYLOAD_SHA256
  raise ArgumentError.new("fixture model revision is not the pinned Qwen-Image 2.1 revision") unless reference.model_revision == ML::GGUF::Qwen3VLTextWeights::EXPECTED_MODEL_REVISION
  raise ArgumentError.new("fixture prompt must be #{PINNED_PROMPT.inspect}") unless reference.prompt == PINNED_PROMPT
  raise ArgumentError.new("fixture must contain 37 hidden states") unless reference.hidden_state_count == ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS + 1
  raise ArgumentError.new("fixture must have two leading attended tokens") unless reference.attention_mask.size >= TOKEN_COUNT && reference.attention_mask[0, TOKEN_COUNT] == [true, true]

  hidden_dim = ML::GGUF::Qwen3VLTextWeights::HIDDEN_SIZE
  scalar_count = TOKEN_COUNT * hidden_dim
  config = ML::GGUF::Qwen3VLTextBlockConfig.new(
    hidden_dim: hidden_dim,
    heads: ML::GGUF::Qwen3VLTextWeights::NUM_ATTENTION_HEADS,
    kv_heads: ML::GGUF::Qwen3VLTextWeights::NUM_KEY_VALUE_HEADS,
    head_dim: ML::GGUF::Qwen3VLTextWeights::HEAD_DIM,
    intermediate_dim: ML::GGUF::Qwen3VLTextWeights::INTERMEDIATE_SIZE,
    attention_arithmetic: ML::GGUF::Qwen3VLTextBlockConfig::AttentionArithmetic::SdpaF32,
  )

  weights = ML::GGUF::Qwen3VLTextWeights.from_directory(encoder_path)
  begin
    composed_input = qwen3vl_prefix(reference.hidden_state(0), scalar_count)
    puts "model_revision=#{reference.model_revision} prompt=#{reference.prompt.inspect} payload_sha256=#{payload_sha256} layers=#{layers} tokens=#{TOKEN_COUNT} attention_arithmetic=SdpaF32"
    layers.times do |layer_index|
      isolated_input = qwen3vl_prefix(reference.hidden_state(layer_index), scalar_count)
      expected = qwen3vl_prefix(reference.hidden_state(layer_index + 1), scalar_count)
      result = qwen3vl_layer_pair(weights, layer_index, isolated_input, composed_input, config)
      GC.collect

      {"isolated" => result[:isolated], "composed" => result[:composed]}.each do |mode, actual|
        metrics = qwen3vl_metrics(actual, expected, hidden_dim)
        puts "#{mode} layer=#{layer_index} bf16_mismatches=#{metrics[:mismatches]}/#{scalar_count} " \
             "token0_mismatches=#{metrics[:token0]} token1_mismatches=#{metrics[:token1]} " \
             "max_abs=#{metrics[:max_abs]} rel_RMS=#{metrics[:rel_rms]}"
      end
      composed_input = result[:composed]
    end
  ensure
    weights.close
  end
rescue ex : Exception
  STDERR.puts "error: #{ex.message}"
  exit 2
end
