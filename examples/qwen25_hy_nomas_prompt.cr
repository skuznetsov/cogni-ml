require "json"
require "option_parser"
require "../src/ml/gguf/qwen25_hy_nomas_prompt"

private def int_array_json(values : Array(Int32)) : JSON::Any
  JSON::Any.new(values.map { |value| JSON::Any.new(value.to_i64) })
end

private def float_array_json(values : Array(Float32)) : JSON::Any
  JSON::Any.new(values.map { |value| JSON::Any.new(value.to_f64) })
end

private def compare_reference!(prepared : ML::GGUF::Qwen25HyNomasPrompt::Prepared,
                               reference_path : String) : Nil
  raw = JSON.parse(File.read(reference_path))
  metadata = raw.as_h["hy_nomas_prompt"]? || raw
  compare = ->(key : String, actual : JSON::Any) do
    expected = metadata.as_h[key]?
    raise ArgumentError.new("prompt reference is missing #{key}") unless expected
    raise ArgumentError.new("native prompt differs from reference field #{key}") unless expected == actual
  end

  compare.call("prompt", JSON::Any.new(prepared.prompt))
  compare.call("prompt_sha256", JSON::Any.new(prepared.prompt_sha256))
  compare.call("rendered_prompt", JSON::Any.new(prepared.rendered_prompt))
  compare.call("rendered_prompt_sha256", JSON::Any.new(prepared.rendered_prompt_sha256))
  compare.call("crop_start", JSON::Any.new(prepared.crop_start.to_i64))
  compare.call("raw_length", JSON::Any.new(prepared.raw_length.to_i64))
  compare.call("text_length", JSON::Any.new(prepared.text_length.to_i64))
  compare.call("token_ids", int_array_json(prepared.tokens))
  compare.call("attention_mask", int_array_json(prepared.attention_mask))
  compare.call("cropped_token_ids", int_array_json(prepared.cropped_tokens))
  compare.call("cropped_attention_mask", int_array_json(prepared.cropped_attention_mask))
  compare.call("marker_token_ids", int_array_json(prepared.marker_token_ids))
end

private def report_json(prepared : ML::GGUF::Qwen25HyNomasPrompt::Prepared) : String
  tokens = prepared.tokens
  mask = prepared.attention_mask
  cropped_tokens = prepared.cropped_tokens
  cropped_mask = prepared.cropped_attention_mask
  positions = prepared.positions
  position_axes = 3.times.map { float_array_json(positions) }.to_a
  official_positions = Array(Int32).new(prepared.raw_length) { |index| index.to_i32 }
  official_axes = 3.times.map { int_array_json(official_positions) }.to_a
  messages = prepared.messages.map do |message|
    JSON::Any.new({
      "role"    => JSON::Any.new(message[:role]),
      "content" => JSON::Any.new(message[:content]),
    } of String => JSON::Any)
  end
  tokenizer_assets = JSON::Any.new({
    "tokenizer.json"        => JSON::Any.new(prepared.tokenizer_json_sha256),
    "tokenizer_config.json" => JSON::Any.new(prepared.tokenizer_config_sha256),
  } of String => JSON::Any)
  forbidden = JSON::Any.new(ML::GGUF::Qwen25HyNomasPrompt::FORBIDDEN_MULTIMODAL_IDS.transform_values do |id|
    JSON::Any.new(id.to_i64)
  end)

  payload = {
    "schema"                          => JSON::Any.new("cogni_ml.hy_nomas_qwen25_video_prompt_reference"),
    "schema_version"                  => JSON::Any.new(1_i64),
    "model_id"                        => JSON::Any.new(ML::GGUF::Qwen25HyNomasPrompt::MODEL_ID),
    "model_revision"                  => JSON::Any.new(ML::GGUF::Qwen25HyNomasPrompt::MODEL_REVISION),
    "template_name"                   => JSON::Any.new(prepared.template_name),
    "template_sha256"                 => JSON::Any.new(ML::GGUF::Qwen25HyNomasPrompt::TEMPLATE_SHA256),
    "prompt"                          => JSON::Any.new(prepared.prompt),
    "prompt_sha256"                   => JSON::Any.new(prepared.prompt_sha256),
    "rendered_prompt"                 => JSON::Any.new(prepared.rendered_prompt),
    "rendered_prompt_sha256"          => JSON::Any.new(prepared.rendered_prompt_sha256),
    "messages"                        => JSON::Any.new(messages),
    "marker"                          => JSON::Any.new(ML::GGUF::Qwen25HyNomasPrompt::USER_MARKER),
    "marker_token_ids"                => int_array_json(prepared.marker_token_ids),
    "crop_start"                      => JSON::Any.new(prepared.crop_start.to_i64),
    "raw_length"                      => JSON::Any.new(prepared.raw_length.to_i64),
    "text_length"                     => JSON::Any.new(prepared.text_length.to_i64),
    "token_ids"                       => int_array_json(tokens),
    "attention_mask"                  => int_array_json(mask),
    "cropped_token_ids"               => int_array_json(cropped_tokens),
    "cropped_attention_mask"          => int_array_json(cropped_mask),
    "position_offset"                 => JSON::Any.new(0_i64),
    "position_ids_f32"                => JSON::Any.new(position_axes),
    "position_ids_for_official_model" => JSON::Any.new(official_axes),
    "multimodal_position_shape"       => JSON::Any.new([JSON::Any.new(3_i64), JSON::Any.new(1_i64), JSON::Any.new(prepared.raw_length.to_i64)]),
    "tokenizer_asset_sha256s"         => tokenizer_assets,
    "forbidden_multimodal_token_ids"  => forbidden,
    "text_only"                       => JSON::Any.new(true),
  } of String => JSON::Any
  JSON::Any.new(payload).to_json
end

checkpoint_root = ""
prompt_file = ""
reference_path = ""
output_path = ""

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen25_hy_nomas_prompt --checkpoint-root PATH --prompt-file PATH [--reference PATH] [--output PATH]"
  parser.on("--checkpoint-root PATH", "Pinned local Qwen2.5-VL tokenizer directory") { |value| checkpoint_root = value }
  parser.on("--prompt-file PATH", "UTF-8 prompt text file (maximum 64 KiB)") { |value| prompt_file = value }
  parser.on("--reference PATH", "Optional upstream prompt metadata JSON for exact CPU comparison") { |value| reference_path = value }
  parser.on("--output PATH", "Write JSON report to a new path instead of stdout") { |value| output_path = value }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit(0)
  end
end

begin
  raise ArgumentError.new("--checkpoint-root is required") if checkpoint_root.empty?
  raise ArgumentError.new("--prompt-file is required") if prompt_file.empty?
  prompt = File.read(prompt_file)
  raise ArgumentError.new("prompt file exceeds 64 KiB") if prompt.bytesize > ML::GGUF::Qwen25HyNomasPrompt::MAX_PROMPT_BYTES
  raise ArgumentError.new("prompt file is not valid UTF-8") unless prompt.valid_encoding?

  prepared = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root).prepare(prompt)
  compare_reference!(prepared, reference_path) unless reference_path.empty?
  report = report_json(prepared)
  if output_path.empty?
    puts report
  else
    raise ArgumentError.new("report path already exists") if File.exists?(output_path)
    File.write(output_path, report)
  end
rescue ex
  STDERR.puts "qwen25_hy_nomas_prompt: #{ex.message}"
  exit(1)
end
