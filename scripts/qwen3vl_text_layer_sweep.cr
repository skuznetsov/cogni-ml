#!/usr/bin/env crystal

require "json"
require "digest/sha256"
require "option_parser"
require "../src/ml/gguf/qwen3vl_text_reference"
require "../src/ml/gguf/qwen3vl_text_weights"
require "../src/ml/gguf/qwen3vl_text_block"

private PINNED_PROMPT         = "red cube"
private PINNED_PAYLOAD_SHA256 = "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07"
private TOKEN_COUNT           = 2
private PREFIX_DROP_COUNT     = 14
private MAX_RAW_TOKEN_COUNT   = 256
private MAX_REFERENCE_BYTES   = MAX_RAW_TOKEN_COUNT.to_i64 * ML::GGUF::Qwen3VLTextWeights::HIDDEN_SIZE * 4_i64 * 38_i64 + MAX_RAW_TOKEN_COUNT.to_i64 * 24_i64
private SHA256_RE             = /\A[0-9a-fA-F]{64}\z/
private PINNED_DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"

private def qwen3vl_prefix(values : Array(Float32), scalar_count : Int32) : Array(Float32)
  raise ArgumentError.new("reference state is shorter than the requested token prefix") if values.size < scalar_count
  values[0, scalar_count]
end

private def qwen3vl_select_rows(values : Array(Float32), rows : Array(Int32), hidden_dim : Int32) : Array(Float32)
  raise ArgumentError.new("reference state shape is not a whole number of rows") unless values.size.divisible_by?(hidden_dim)
  row_count = values.size // hidden_dim
  selected = Array(Float32).new(rows.size * hidden_dim)
  rows.each do |row|
    raise ArgumentError.new("selected row is outside the reference state") unless 0 <= row < row_count
    start = row * hidden_dim
    hidden_dim.times { |column| selected << values[start + column] }
  end
  selected
end

private def qwen3vl_retained_rows(mask : Array(Bool), drop_idx : Int32) : Array(Int32)
  selected = [] of Int32
  attended_index = 0
  mask.each_with_index do |attended, row|
    next unless attended
    selected << row.to_i32 if attended_index >= drop_idx
    attended_index += 1
  end
  selected
end

private def qwen3vl_layer_pair(
  weights : ML::GGUF::Qwen3VLTextWeights,
  layer_index : Int32,
  isolated_input : Array(Float32),
  composed_input : Array(Float32),
  attention_mask : Array(Bool),
  config : ML::GGUF::Qwen3VLTextBlockConfig,
) : {isolated: Array(Float32), composed: Array(Float32)}
  block_weights = weights.block_weights(layer_index)
  {
    isolated: ML::GGUF::Qwen3VLTextBlock.forward(isolated_input, attention_mask, block_weights, config),
    composed: ML::GGUF::Qwen3VLTextBlock.forward(composed_input, attention_mask, block_weights, config),
  }
end

private def qwen3vl_bf16_bits(value : Float32) : UInt16
  (value.unsafe_as(UInt32) >> 16).to_u16
end

private def qwen3vl_metrics(actual : Array(Float32), expected : Array(Float32), hidden_dim : Int32)
  raise ArgumentError.new("actual and official state sizes differ") unless actual.size == expected.size
  raise ArgumentError.new("state size is not divisible by the hidden dimension") unless actual.size.divisible_by?(hidden_dim)
  row_mismatches = Array(Int32).new(actual.size // hidden_dim, 0)
  mismatches = 0
  max_abs = 0.0_f64
  sum_squared_error = 0.0_f64
  sum_squared_expected = 0.0_f64

  actual.each_with_index do |value, index|
    reference = expected[index]
    raise ArgumentError.new("non-finite comparison value at index #{index}") unless value.finite? && reference.finite?
    if qwen3vl_bf16_bits(value) != qwen3vl_bf16_bits(reference)
      mismatches += 1
      row_mismatches[index // hidden_dim] += 1
    end
    error = (value.to_f64 - reference.to_f64).abs
    max_abs = error if error > max_abs
    sum_squared_error += error * error
    expected64 = reference.to_f64
    sum_squared_expected += expected64 * expected64
  end

  {
    mismatches:         mismatches,
    row_mismatches:     row_mismatches,
    rows_with_mismatch: row_mismatches.count { |count| count > 0 },
    max_abs:            max_abs,
    rel_rms:            Math.sqrt(sum_squared_error / (sum_squared_expected + 1e-30)),
  }
end

private def qwen3vl_bf16_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    raise ArgumentError.new("non-finite trace value at index #{index}") unless value.finite?
    bits = qwen3vl_bf16_bits(value)
    bytes[index * 2] = (bits & 0xff).to_u8
    bytes[index * 2 + 1] = (bits >> 8).to_u8
  end
  bytes
end

private def qwen3vl_link_new_file(path : String, bytes : Bytes) : Nil
  temp = File.tempfile("qwen3vl-trace", ".tmp", dir: File.dirname(path))
  begin
    temp.write(bytes)
    temp.close
    File.link(temp.path, path)
  ensure
    temp.delete if File.exists?(temp.path)
  end
end

private def qwen3vl_write_layer0_trace(
  directory : String,
  trace : Hash(String, Array(Float32)),
  token_count : Int32,
  prompt : String,
  model_revision : String,
  fixture_payload_sha256 : String,
) : Nil
  raise ArgumentError.new("native layer-0 trace is empty") if trace.empty?
  Dir.mkdir(directory)
  stages = trace.keys.sort
  entries = {} of String => {filename: String, shape: Array(Int32), nbytes: Int32, sha256: String}
  stages.each do |stage|
    raise ArgumentError.new("invalid native layer-0 trace stage name: #{stage}") unless stage.matches?(/\A[a-zA-Z0-9_.]+\z/)
    values = trace[stage]
    raise ArgumentError.new("native layer-0 trace stage has invalid token count: #{stage}") unless values.size.divisible_by?(token_count)
    filename = "#{stage}.bf16le"
    bytes = qwen3vl_bf16_bytes(values)
    qwen3vl_link_new_file(File.join(directory, filename), bytes)
    entries[stage] = {
      filename: filename,
      shape:    [token_count, values.size // token_count],
      nbytes:   bytes.size,
      sha256:   Digest::SHA256.hexdigest(bytes),
    }
  end
  manifest = JSON.build do |json|
    json.object do
      json.field "schema", "qwen3vl-native-layer0-trace"
      json.field "schema_version", 1
      json.field "prompt", prompt
      json.field "model_revision", model_revision
      json.field "fixture_payload_sha256", fixture_payload_sha256
      json.field "dtype", "bfloat16-le"
      json.field "stages" do
        json.object do
          stages.each do |stage|
            entry = entries[stage]
            json.field stage do
              json.object do
                json.field "filename", entry[:filename]
                json.field "shape", entry[:shape]
                json.field "nbytes", entry[:nbytes]
                json.field "sha256", entry[:sha256]
              end
            end
          end
        end
      end
    end
  end
  qwen3vl_link_new_file(File.join(directory, "trace.json"), manifest.to_slice)
end

private def qwen3vl_write_retained_bf16(
  path : String,
  values : Array(Float32),
  prompt : String,
  model_revision : String,
  fixture_payload_sha256 : String,
  fixture_manifest_sha256 : String,
  drop_idx : Int32,
  row_count : Int32,
  hidden_dim : Int32,
) : {sha256: String, nbytes: Int32, manifest_path: String}
  manifest_path = "#{path}.json"
  raise ArgumentError.new("retained BF16 output already exists: #{path}") if File.exists?(path)
  raise ArgumentError.new("retained BF16 output manifest already exists: #{manifest_path}") if File.exists?(manifest_path)
  bytes = qwen3vl_bf16_bytes(values)
  sha256 = Digest::SHA256.hexdigest(bytes)
  manifest = JSON.build do |json|
    json.object do
      json.field "schema", "qwen3vl-retained-embeddings"
      json.field "schema_version", 1
      json.field "payload_file", File.basename(path)
      json.field "prompt", prompt
      json.field "model_revision", model_revision
      json.field "fixture_payload_sha256", fixture_payload_sha256
      json.field "fixture_manifest_sha256", fixture_manifest_sha256
      json.field "drop_idx", drop_idx
      json.field "shape", [row_count, hidden_dim]
      json.field "dtype", "bfloat16-le"
      json.field "nbytes", bytes.size
      json.field "sha256", sha256
    end
  end
  output_dir = File.dirname(path)
  raise ArgumentError.new("retained BF16 output directory does not exist: #{output_dir}") unless Dir.exists?(output_dir)
  payload_temp = File.tempfile("qwen3vl-retained", ".bf16le.tmp", dir: output_dir)
  manifest_temp = File.tempfile("qwen3vl-retained", ".json.tmp", dir: output_dir)
  begin
    payload_temp.write(bytes)
    payload_temp.close
    manifest_temp.print(manifest)
    manifest_temp.close
    # Hard-link publication is atomic and fails if the destination already exists.
    File.link(payload_temp.path, path)
    begin
      File.link(manifest_temp.path, manifest_path)
    rescue ex
      File.delete(path) if File.same?(payload_temp.path, path)
      raise ex
    end
  ensure
    payload_temp.delete if File.exists?(payload_temp.path)
    manifest_temp.delete if File.exists?(manifest_temp.path)
  end
  {sha256: sha256, nbytes: bytes.size, manifest_path: manifest_path}
end

layers = 2
full_prompt = false
composed_only = false
reference_sha256 : String? = nil
projection_backend = ML::GGUF::Qwen3VLTextBlockConfig::ProjectionBackend::Scalar
projection_backend_label = "scalar"
encoder_dir = ENV["QWEN3VL_TEXT_ENCODER_DIR"]?
reference_dir = ENV["QWEN3VL_TEXT_REFERENCE_DIR"]?
retained_bf16_out : String? = nil
layer0_trace_dir : String? = nil

OptionParser.parse do |parser|
  parser.banner = "Usage: crystal run scripts/qwen3vl_text_layer_sweep.cr -- [--layers N] [--full-prompt [--composed-only]] [--reference-sha256 SHA256] [--projection-backend scalar|accelerate] [--retained-bf16-out PATH] [--layer0-trace-dir DIR] --text-encoder-dir DIR --reference-dir DIR"
  parser.on("--layers=N", "Number of leading decoder layers to sweep (1..36, default 2)") { |value| layers = value.to_i? || abort("--layers must be an integer") }
  parser.on("--full-prompt", "Use all raw prompt tokens (default uses the first two)") { full_prompt = true }
  parser.on("--composed-only", "Skip isolated block runs; requires --full-prompt") { composed_only = true }
  parser.on("--reference-sha256=SHA256", "Opt in to a different text reference only when its exact payload SHA-256 matches") { |value| reference_sha256 = value }
  parser.on("--projection-backend=BACKEND", "Text projection backend (scalar default; accelerate is diagnostic opt-in)") do |value|
    case value.downcase
    when "scalar"
      projection_backend = ML::GGUF::Qwen3VLTextBlockConfig::ProjectionBackend::Scalar
      projection_backend_label = "scalar"
    when "accelerate"
      projection_backend = ML::GGUF::Qwen3VLTextBlockConfig::ProjectionBackend::Accelerate
      projection_backend_label = "accelerate"
    else
      abort("--projection-backend must be scalar or accelerate")
    end
  end
  parser.on("--retained-bf16-out=PATH", "Write final retained BF16 rows (requires --full-prompt and --layers=36)") { |value| retained_bf16_out = value }
  parser.on("--layer0-trace-dir=DIR", "Write all raw-token layer-0 BF16 stages (requires --full-prompt --composed-only --layers=1)") { |value| layer0_trace_dir = value }
  parser.on("--text-encoder-dir=DIR", "Local Qwen3-VL text encoder checkpoint directory") { |value| encoder_dir = value }
  parser.on("--reference-dir=DIR", "Directory containing the checksummed text reference fixture") { |value| reference_dir = value }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

abort("unexpected arguments: #{ARGV.join(" ")}") unless ARGV.empty?
abort("--layers must be in 1..36") unless 1 <= layers <= ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS
abort("--composed-only requires --full-prompt") if composed_only && !full_prompt
if sha256 = reference_sha256
  abort("--reference-sha256 must be a 64-character SHA-256") unless SHA256_RE.matches?(sha256)
  abort("--reference-sha256 requires --full-prompt") unless full_prompt
end
if retained_bf16_out && (!full_prompt || layers != ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS)
  abort("--retained-bf16-out requires --full-prompt and --layers=36")
end
if layer0_trace_dir && (!full_prompt || !composed_only || layers != 1)
  abort("--layer0-trace-dir requires --full-prompt --composed-only --layers=1")
end
if trace_dir = layer0_trace_dir
  abort("native layer-0 trace output already exists: #{trace_dir}") if File.exists?(trace_dir) || Dir.exists?(trace_dir)
  abort("native layer-0 trace parent does not exist: #{File.dirname(trace_dir)}") unless Dir.exists?(File.dirname(trace_dir))
end
if output_path = retained_bf16_out
  manifest_path = "#{output_path}.json"
  abort("retained BF16 output already exists: #{output_path}") if File.exists?(output_path)
  abort("retained BF16 output manifest already exists: #{manifest_path}") if File.exists?(manifest_path)
  output_dir = File.dirname(output_path)
  abort("retained BF16 output directory does not exist: #{output_dir}") unless Dir.exists?(output_dir)
end
abort("provide --text-encoder-dir or QWEN3VL_TEXT_ENCODER_DIR") unless encoder_dir
abort("provide --reference-dir or QWEN3VL_TEXT_REFERENCE_DIR") unless reference_dir
encoder_path = encoder_dir.not_nil!
reference_path = reference_dir.not_nil!

begin
  manifest_path = File.join(reference_path, "qwen_image21_text_reference.json")
  raise ArgumentError.new("text reference manifest must not be a symlink") if File.symlink?(manifest_path)
  raise ArgumentError.new("text reference manifest is missing or is not a regular file") unless File.file?(manifest_path)
  raise ArgumentError.new("text reference manifest exceeds the 2 MiB guard") if File.info(manifest_path).size > 2_i64 * 1024 * 1024
  manifest_json = File.read(manifest_path)
  reference_manifest_sha256 = Digest::SHA256.hexdigest(manifest_json)
  manifest = JSON.parse(manifest_json)
  model_metadata = manifest["model"]
  unless model_metadata["repo"].as_s == "Qwen/Qwen-Image-2.1" &&
         model_metadata["pipeline_class"].as_s == "QwenImage21Pipeline" &&
         model_metadata["text_encoder_class"].as_s == "Qwen3VLForConditionalGeneration" &&
         model_metadata["processor_class"].as_s == "Qwen3VLProcessor" &&
         {"local_cache_metadata", "argument", "argument_and_local_cache_metadata"}.includes?(model_metadata["revision_source"].as_s)
    raise ArgumentError.new("text reference must identify the official pinned Qwen-Image 2.1 pipeline/model classes")
  end
  tokenization_metadata = manifest["tokenization"]
  sequence_metadata = manifest["sequence"]
  embedding_metadata = manifest["embedding"]
  runtime_metadata = manifest["runtime"]
  expected_template = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n" \
                     "<|im_start|>user\n#{manifest["prompt"].as_s}<|im_end|>\n" \
                     "<|im_start|>assistant\n"
  processor_kwargs = tokenization_metadata["processor_kwargs"]
  unless tokenization_metadata["raw_template_text"].as_s == expected_template &&
         tokenization_metadata["tokenizer_truncation"].as_bool == false &&
         processor_kwargs["padding"].as_bool &&
         processor_kwargs["padding_side"].as_s == "left" &&
         processor_kwargs["return_tensors"].as_s == "pt" &&
         processor_kwargs.as_h.size == 3
    raise ArgumentError.new("text reference prompt template or processor/truncation metadata differs from the official capture contract")
  end
  unless sequence_metadata["max_sequence_length_semantics"].as_s == "post-drop validation guard only; no processor truncation" &&
         embedding_metadata["source"].as_s == "QwenImage21Pipeline._get_qwen_prompt_embeds" &&
         embedding_metadata["rmsnorm_hook_observed_and_verified"].as_bool &&
         embedding_metadata["source_dtype"].as_s == "bfloat16" &&
         embedding_metadata["expected_hidden_state_count"].as_i64 == ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS + 1 &&
         embedding_metadata["expected_decoder_layer_count"].as_i64 == ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS &&
         embedding_metadata["expected_hidden_state_count_source"].as_s == "loaded text_encoder.config.text_config.num_hidden_layers + 1" &&
         runtime_metadata["source_dtype"].as_s == "bfloat16" &&
         runtime_metadata["device"].as_s == "cpu" &&
         runtime_metadata["diffusers_commit"].as_s == PINNED_DIFFUSERS_COMMIT &&
         runtime_metadata["official_source_file"].as_s == "diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py"
    raise ArgumentError.new("text reference capture source, RMSNorm, sequence guard, or runtime provenance differs from the pinned official contract")
  end
  payload_name = manifest["payload_file"].as_s
  raise ArgumentError.new("text reference payload_file must name qwen_image21_text_reference.bin") unless payload_name == "qwen_image21_text_reference.bin"
  payload_path = File.join(reference_path, payload_name)
  raise ArgumentError.new("text reference payload must not be a symlink") if File.symlink?(payload_path)
  raise ArgumentError.new("text reference payload is missing or is not a regular file") unless File.file?(payload_path)
  payload_sha256 = manifest["payload_sha256"].as_s.downcase
  raise ArgumentError.new("text reference payload SHA-256 is invalid") unless SHA256_RE.matches?(payload_sha256)
  payload_nbytes = manifest["payload_nbytes"].as_i64
  raise ArgumentError.new("text reference payload length is invalid") unless payload_nbytes > 0 && File.info(payload_path).size == payload_nbytes
  raise ArgumentError.new("text reference payload exceeds the #{MAX_REFERENCE_BYTES}-byte memory guard") if payload_nbytes > MAX_REFERENCE_BYTES
  sequence_manifest = manifest["sequence"]
  raw_shape = sequence_manifest["raw_input_shape"].as_a.map(&.as_i64)
  raise ArgumentError.new("text reference raw input shape must be [1, raw_sequence_length]") unless raw_shape.size == 2 && raw_shape[0] == 1 && raw_shape[1] > 0
  raise ArgumentError.new("text reference raw token count exceeds #{MAX_RAW_TOKEN_COUNT}") if raw_shape[1] > MAX_RAW_TOKEN_COUNT
  if expected_sha256 = reference_sha256
    raise ArgumentError.new("reference SHA-256 does not match --reference-sha256") unless payload_sha256 == expected_sha256.downcase
  else
    raise ArgumentError.new("fixture payload SHA-256 is not the pinned red-cube artifact") unless payload_sha256 == PINNED_PAYLOAD_SHA256
  end
  reference = ML::GGUF::Qwen3VLTextReference.load(manifest_path)
  raise ArgumentError.new("fixture model revision is not the pinned Qwen-Image 2.1 revision") unless reference.model_revision == ML::GGUF::Qwen3VLTextWeights::EXPECTED_MODEL_REVISION
  if reference_sha256.nil?
    raise ArgumentError.new("fixture prompt must be #{PINNED_PROMPT.inspect}") unless reference.prompt == PINNED_PROMPT
  end
  raise ArgumentError.new("fixture must contain 37 hidden states") unless reference.hidden_state_count == ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS + 1
  raise ArgumentError.new("reference retained drop_idx must equal the Qwen prefix drop count #{PREFIX_DROP_COUNT}") unless reference.drop_idx == PREFIX_DROP_COUNT
  raise ArgumentError.new("reference retained length exceeds max_sequence_length post-drop guard") if reference.actual_sequence_length > reference.max_sequence_length
  raise ArgumentError.new("reference mask shape differs from the raw input shape") unless reference.attention_mask.size == reference.raw_sequence_length
  raise ArgumentError.new("fixture actual length does not agree with mask and drop_idx") unless reference.attention_mask.count(true) - reference.drop_idx == reference.actual_sequence_length
  raise ArgumentError.new("fixture must contain pre-final-RMSNorm embeddings") unless manifest["embedding"]["pre_final_rmsnorm"].as_bool

  hidden_dim = ML::GGUF::Qwen3VLTextWeights::HIDDEN_SIZE
  raise ArgumentError.new("fixture pre-final embedding shape differs from the retained row count") unless reference.pre_final_rmsnorm_embeddings.size == reference.actual_sequence_length * hidden_dim
  reference_embedding_shape = manifest["embedding"]["shape"].as_a.map(&.as_i64)
  raise ArgumentError.new("reference pre-final embedding shape does not match the retained rows") unless reference_embedding_shape == [1_i64, reference.actual_sequence_length.to_i64, hidden_dim.to_i64]
  raise ArgumentError.new("reference raw input shape differs from the loaded token count") unless raw_shape == [1_i64, reference.raw_sequence_length.to_i64]
  if full_prompt
    if reference_sha256.nil?
      raise ArgumentError.new("pinned full prompt must have raw shape [1, 24]") unless reference.raw_sequence_length == 24
      raise ArgumentError.new("pinned full prompt must retain 10 rows after dropping 14 attended rows") unless reference.actual_sequence_length == 10 && reference.drop_idx == PREFIX_DROP_COUNT
      raise ArgumentError.new("pinned full prompt must attend all 24 raw rows") unless reference.attention_mask.all? { |attended| attended }
      raw_shape = manifest["sequence"]["raw_input_shape"].as_a.map(&.as_i64)
      embedding_shape = manifest["embedding"]["shape"].as_a.map(&.as_i64)
      raise ArgumentError.new("fixture raw input shape must be [1, 24]") unless raw_shape == [1_i64, 24_i64]
      raise ArgumentError.new("fixture pre-final embedding shape must be [1, 10, 4096]") unless embedding_shape == [1_i64, 10_i64, hidden_dim.to_i64]
    end
  else
    raise ArgumentError.new("--reference-sha256 requires --full-prompt") unless reference_sha256.nil?
    raise ArgumentError.new("fixture must have two leading attended tokens") unless reference.attention_mask.size >= TOKEN_COUNT && reference.attention_mask[0, TOKEN_COUNT] == [true, true]
  end

  token_count = full_prompt ? reference.raw_sequence_length : TOKEN_COUNT
  scalar_count = token_count * hidden_dim
  attention_mask = reference.attention_mask[0, token_count]
  retained_raw_rows = qwen3vl_retained_rows(reference.attention_mask, reference.drop_idx)
  metric_rows = full_prompt ? retained_raw_rows : Array(Int32).new(token_count) { |index| index.to_i32 }
  raise ArgumentError.new("retained row selection does not match actual sequence length") unless retained_raw_rows.size == reference.actual_sequence_length
  if full_prompt
    fixture_final_retained = qwen3vl_select_rows(
      reference.hidden_state(ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS), retained_raw_rows, hidden_dim
    )
    fixture_final_metrics = qwen3vl_metrics(
      fixture_final_retained, reference.pre_final_rmsnorm_embeddings, hidden_dim
    )
    unless fixture_final_metrics[:mismatches] == 0
      raise ArgumentError.new("fixture hidden_state_036 retained rows differ from pre-final-RMSNorm embeddings: " \
                              "#{fixture_final_metrics[:mismatches]}/#{fixture_final_retained.size} BF16 mismatches")
    end
    puts "fixture_guard mapping=hidden_state_036_attention_mask_then_drop pre_final_rmsnorm=true " \
         "final_norm_applied=false rows=#{retained_raw_rows.size} shape=#{retained_raw_rows.size}x#{hidden_dim} " \
         "bf16_mismatches=#{fixture_final_metrics[:mismatches]}/#{fixture_final_retained.size}"
  end
  config = ML::GGUF::Qwen3VLTextBlockConfig.new(
    hidden_dim: hidden_dim,
    heads: ML::GGUF::Qwen3VLTextWeights::NUM_ATTENTION_HEADS,
    kv_heads: ML::GGUF::Qwen3VLTextWeights::NUM_KEY_VALUE_HEADS,
    head_dim: ML::GGUF::Qwen3VLTextWeights::HEAD_DIM,
    intermediate_dim: ML::GGUF::Qwen3VLTextWeights::INTERMEDIATE_SIZE,
    attention_arithmetic: ML::GGUF::Qwen3VLTextBlockConfig::AttentionArithmetic::SdpaF32,
    projection_backend: projection_backend,
  )

  puts "reference_validation=passed model_revision=#{reference.model_revision} prompt=#{reference.prompt.inspect} payload_sha256=#{payload_sha256} manifest_sha256=#{reference_manifest_sha256} raw_tokens=#{reference.raw_sequence_length} retained_tokens=#{reference.actual_sequence_length} drop_idx=#{reference.drop_idx} retained_rows=#{retained_raw_rows.inspect} layers=#{layers} tokens=#{token_count} attention_mask=#{attention_mask.count(true)}/#{token_count} attention_arithmetic=SdpaF32 projection_backend=#{projection_backend_label} composed_only=#{composed_only}"
  weights = ML::GGUF::Qwen3VLTextWeights.from_directory(encoder_path)
  begin
    composed_input = qwen3vl_prefix(reference.hidden_state(0), scalar_count)
    layers.times do |layer_index|
      isolated_input = qwen3vl_prefix(reference.hidden_state(layer_index), scalar_count)
      expected = qwen3vl_prefix(reference.hidden_state(layer_index + 1), scalar_count)
      if composed_only
        trace = layer0_trace_dir ? {} of String => Array(Float32) : nil
        actual = ML::GGUF::Qwen3VLTextBlock.forward(composed_input, attention_mask, weights.block_weights(layer_index), config, trace: trace)
        if trace_dir = layer0_trace_dir
          qwen3vl_write_layer0_trace(trace_dir, trace.not_nil!, token_count, reference.prompt, reference.model_revision, payload_sha256)
          puts "native_layer0_trace=#{trace_dir} stages=#{trace.not_nil!.size} dtype=bfloat16-le"
        end
        metrics = qwen3vl_metrics(actual, expected, hidden_dim)
        if full_prompt
          retained_metrics = qwen3vl_metrics(
            qwen3vl_select_rows(actual, metric_rows, hidden_dim),
            qwen3vl_select_rows(expected, metric_rows, hidden_dim), hidden_dim
          )
          puts "composed layer=#{layer_index} bf16_mismatches=#{metrics[:mismatches]}/#{scalar_count} rows_with_mismatch=#{metrics[:rows_with_mismatch]}/#{token_count} row_bf16_mismatches=#{metrics[:row_mismatches].inspect} max_abs=#{metrics[:max_abs]} rel_RMS=#{metrics[:rel_rms]} " \
               "retained_bf16_mismatches=#{retained_metrics[:mismatches]}/#{metric_rows.size * hidden_dim} retained_rows_with_mismatch=#{retained_metrics[:rows_with_mismatch]}/#{metric_rows.size} retained_max_abs=#{retained_metrics[:max_abs]} retained_rel_RMS=#{retained_metrics[:rel_rms]}"
        else
          puts "composed layer=#{layer_index} bf16_mismatches=#{metrics[:mismatches]}/#{scalar_count} token0_mismatches=#{metrics[:row_mismatches][0]} token1_mismatches=#{metrics[:row_mismatches][1]} max_abs=#{metrics[:max_abs]} rel_RMS=#{metrics[:rel_rms]}"
        end
        composed_input = actual
      else
        result = qwen3vl_layer_pair(weights, layer_index, isolated_input, composed_input, attention_mask, config)
        {"isolated" => result[:isolated], "composed" => result[:composed]}.each do |mode, actual|
          metrics = qwen3vl_metrics(actual, expected, hidden_dim)
          if full_prompt
            retained_metrics = qwen3vl_metrics(
              qwen3vl_select_rows(actual, metric_rows, hidden_dim),
              qwen3vl_select_rows(expected, metric_rows, hidden_dim), hidden_dim
            )
            puts "#{mode} layer=#{layer_index} bf16_mismatches=#{metrics[:mismatches]}/#{scalar_count} rows_with_mismatch=#{metrics[:rows_with_mismatch]}/#{token_count} row_bf16_mismatches=#{metrics[:row_mismatches].inspect} max_abs=#{metrics[:max_abs]} rel_RMS=#{metrics[:rel_rms]} " \
                 "retained_bf16_mismatches=#{retained_metrics[:mismatches]}/#{metric_rows.size * hidden_dim} retained_rows_with_mismatch=#{retained_metrics[:rows_with_mismatch]}/#{metric_rows.size} retained_max_abs=#{retained_metrics[:max_abs]} retained_rel_RMS=#{retained_metrics[:rel_rms]}"
          else
            puts "#{mode} layer=#{layer_index} bf16_mismatches=#{metrics[:mismatches]}/#{scalar_count} " \
                 "token0_mismatches=#{metrics[:row_mismatches][0]} token1_mismatches=#{metrics[:row_mismatches][1]} " \
                 "max_abs=#{metrics[:max_abs]} rel_RMS=#{metrics[:rel_rms]}"
          end
        end
        composed_input = result[:composed]
      end
      GC.collect
    end

    if full_prompt
      if layers == ML::GGUF::Qwen3VLTextWeights::NUM_LAYERS
        retained_actual = qwen3vl_select_rows(composed_input, retained_raw_rows, hidden_dim)
        final_metrics = qwen3vl_metrics(retained_actual, reference.pre_final_rmsnorm_embeddings, hidden_dim)
        puts "final_retained mapping=hidden_state_036_attention_mask_then_drop pre_final_rmsnorm=true final_norm_applied=false rows=#{retained_raw_rows.size} shape=#{retained_raw_rows.size}x#{hidden_dim} bf16_mismatches=#{final_metrics[:mismatches]}/#{retained_actual.size} rows_with_mismatch=#{final_metrics[:rows_with_mismatch]}/#{retained_raw_rows.size} max_abs=#{final_metrics[:max_abs]} rel_RMS=#{final_metrics[:rel_rms]}"
        if output_path = retained_bf16_out
          artifact = qwen3vl_write_retained_bf16(
            output_path, retained_actual, reference.prompt, reference.model_revision,
            payload_sha256, reference_manifest_sha256, reference.drop_idx,
            retained_raw_rows.size.to_i32, hidden_dim
          )
          puts "retained_bf16_out=#{output_path} dtype=bfloat16-le shape=#{retained_raw_rows.size}x#{hidden_dim} nbytes=#{artifact[:nbytes]} sha256=#{artifact[:sha256]} manifest=#{artifact[:manifest_path]}"
        end
      else
        puts "final_retained comparison=skipped reason=requires_36_layers completed_layers=#{layers}"
      end
    end
  ensure
    weights.close
  end
rescue ex : Exception
  STDERR.puts "error: #{ex.message}"
  exit 2
end
