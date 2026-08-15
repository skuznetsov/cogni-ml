# Default-off quality, size, and restore falsifier for QBit-style Gaussian
# compression of recurrent Qwen cache state. P7 restores directly from the
# plane-major ClickHouse Native block into prepared Metal state; other
# precisions retain the scalar CPU reference path.

require "option_parser"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_native_block"
require "../src/ml/gguf/qwen_qbit_native_writer"
require "../src/ml/gguf/qwen_qbit_state_snapshot"

DEFAULT_MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
prompt = "The capital of France is"
n_gen = 16
max_seq = 64
block_size = 1024
precisions = [8, 7, 6] of Int32
prepare_state = true
chat_mode = false
native_out : String? = nil
native_in : String? = nil
kv_out : String? = nil
int8_out : String? = nil
cache_id : UInt64? = nil
native_max_mib = 256
restore_repeats = 5

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_cache_probe [options] [prompt]"
  parser.on("--model PATH", "Target Qwen GGUF path") { |value| model_path = value }
  parser.on("--gen N", "Continuation tokens including the cached first token (default: 16)") { |value| n_gen = value.to_i }
  parser.on("--max-seq N", "Decode state sequence capacity (default: 64)") { |value| max_seq = value.to_i }
  parser.on("--block-size N", "Affine QBit block size (default: 1024)") { |value| block_size = value.to_i }
  parser.on("--precisions LIST", "Comma-separated retained planes (default: 8,7,6)") do |value|
    precisions = value.split(',').map(&.to_i32)
  end
  parser.on("--chat", "Render the prompt through the embedded Qwen chat template") { chat_mode = true }
  parser.on("--native-out PATH", "Write recurrent p7 tiles as a revision-zero ClickHouse Native block") { |value| native_out = value }
  parser.on("--native-in PATH", "Restore p7 tiles from ordered ClickHouse Native response blocks") { |value| native_in = value }
  parser.on("--kv-out PATH", "Write an exact raw-KV-only diagnostic artifact") { |value| kv_out = value }
  parser.on("--int8-out PATH", "Write the complete recurrent-INT8 comparison artifact") { |value| int8_out = value }
  parser.on("--cache-id ID", "Expected UInt64 Native cache identity (required with --native-in)") { |value| cache_id = value.to_u64 }
  parser.on("--native-max-mib N", "Maximum accepted Native response size (default: 256, hard max: 1024)") { |value| native_max_mib = value.to_i }
  parser.on("--restore-repeats N", "Timed restores after one cold restore (default: 5)") { |value| restore_repeats = value.to_i }
  parser.on("--no-prepare-state", "Do not eagerly prepare Metal state") { prepare_state = false }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end
prompt = ARGV.join(" ") unless ARGV.empty?
model_prompt = chat_mode ? ML::GGUF::Qwen35Chat.render_user_prompt(prompt) : prompt

raise "--gen must be at least 2" unless n_gen >= 2
raise "--max-seq must be positive" unless max_seq > 0
raise "--block-size must be positive" unless block_size > 0
raise "--precisions cannot be empty" if precisions.empty?
raise "--restore-repeats must be positive" unless restore_repeats > 0
raise "--native-in requires p7 in --precisions" if native_in && !precisions.includes?(7)
raise "--native-in requires prepared Metal state" if native_in && !prepare_state
raise "--native-in file does not exist" if (path = native_in) && !File.file?(path)
raise "--native-in requires an explicit --cache-id" if native_in && cache_id.nil?
raise "--native-max-mib must be within 1..1024" unless native_max_mib.in?(1..1024)
output_paths = [native_out, kv_out, int8_out].compact
raise "diagnostic output paths must be distinct" unless output_paths.uniq.size == output_paths.size
native_limit_bytes = native_max_mib.to_i64 * 1024 * 1024
if path = native_in
  native_input_size = File.size(path)
  unless native_input_size > 0 && native_input_size <= native_limit_bytes
    raise "--native-in size is outside 1..#{native_limit_bytes} bytes"
  end
end
effective_cache_id = cache_id.nil? ? 0_u64 : cache_id.not_nil!
precisions.each do |precision|
  unless precision >= ML::GGUF::QwenQBitGaussianCodec::MIN_PRECISION && precision <= ML::GGUF::QwenQBitGaussianCodec::MAX_PRECISION
    raise "unsupported precision: #{precision}"
  end
end

def floats_from(bytes : Bytes) : Array(Float32)
  raise "state record is not Float32-aligned" unless bytes.size % sizeof(Float32) == 0
  values = Array(Float32).new(bytes.size // sizeof(Float32), 0.0_f32)
  Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size).copy_from(bytes)
  values
end

def bytes_from(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  bytes.copy_from(Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size))
  bytes
end

def read_bytes(path : String, max_bytes : Int64) : Bytes
  size = File.size(path)
  unless size > 0 && size <= max_bytes && size <= Int32::MAX
    raise "Native response size changed outside the admitted limit"
  end
  bytes = Bytes.new(size.to_i32)
  File.open(path, "r") { |file| file.read_fully(bytes) }
  bytes
end

def recurrent_record?(kind : ML::GGUF::Qwen35StateSnapshot::RecordKind) : Bool
  case kind
  in ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState,
     ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState
    true
  in ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
     ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache
    false
  end
end

def qbit_snapshot(snapshot : ML::GGUF::Qwen35StateSnapshot::Snapshot,
                  block_size : Int32,
                  precision : Int32) : {ML::GGUF::QwenQBitStateSnapshot::Snapshot, Int64, Float64, Float64}
  started = Time.instant
  encoded = ML::GGUF::QwenQBitStateSnapshot.encode(snapshot, block_size: block_size, precision: precision)
  encode_ms = (Time.instant - started).total_milliseconds

  # Retain the scalar reference as a falsifier, but do not use it for the
  # measured cache-hit restore path.
  started = Time.instant
  ML::GGUF::QwenQBitStateSnapshot.decode(encoded)
  decode_ms = (Time.instant - started).total_milliseconds
  {encoded, encoded.payload_byte_size, encode_ms, decode_ms}
end

def continuation(weights : ML::GGUF::Qwen35Weights,
                 state : ML::GGUF::Qwen35CPU::State,
                 first_token : Int32,
                 first_logit : Float32,
                 position : Int32,
                 count : Int32) : {Array(Int32), Array(Float32)}
  ids = [first_token]
  logits = [first_logit]
  while ids.size < count
    token, logit = ML::GGUF::Qwen35CPU.forward_top1(weights, ids[-1], position, state)
    ids << token
    logits << logit
    position += 1
  end
  {ids, logits}
end

def teacher_forced_predictions(weights : ML::GGUF::Qwen35Weights,
                               state : ML::GGUF::Qwen35CPU::State,
                               exact_ids : Array(Int32),
                               position : Int32) : {Array(Int32), Array(Float32)}
  predicted = [] of Int32
  logits = [] of Float32
  (exact_ids.size - 1).times do |i|
    token, logit = ML::GGUF::Qwen35CPU.forward_top1(weights, exact_ids[i], position, state)
    predicted << token
    logits << logit
    position += 1
  end
  {predicted, logits}
end

def common_prefix(a : Array(Int32), b : Array(Int32)) : Int32
  limit = Math.min(a.size, b.size)
  i = 0
  while i < limit && a[i] == b[i]
    i += 1
  end
  i.to_i32
end

def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  middle = sorted.size // 2
  return sorted[middle] if sorted.size.odd?
  (sorted[middle - 1] + sorted[middle]) / 2.0
end

def release_metal_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
    layer.k_cache_buf = nil
    layer.v_cache_buf = nil
    layer.conv_state_buf = nil
    layer.ssm_state_buf = nil
  end
end

startup_started = Time.instant
gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
startup_ms = (Time.instant - startup_started).total_milliseconds

tokens = tokenizer.encode(model_prompt)
raise "prompt encoded to zero tokens" if tokens.empty?
raise "prompt plus continuation exceeds max_seq" if tokens.size + n_gen >= max_seq

state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams) if prepare_state
prefill_started = Time.instant
first_token, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, tokens, 0, state)
prefill_ms = (Time.instant - prefill_started).total_milliseconds
snapshot_started = Time.instant
snapshot = ML::GGUF::Qwen35StateSnapshot.capture(state)
snapshot_ms = (Time.instant - snapshot_started).total_milliseconds
release_metal_state!(state)

exact_state = ML::GGUF::Qwen35StateSnapshot.restore(snapshot, weights.hparams)
exact_ids, exact_logits = continuation(weights, exact_state, first_token, first_logit, tokens.size.to_i32, n_gen)
release_metal_state!(exact_state)

recurrent_raw_bytes = snapshot.records.sum(0_i64) { |record| recurrent_record?(record.kind) ? record.bytes.size.to_i64 : 0_i64 }
kv_raw_bytes = snapshot.byte_size - recurrent_raw_bytes
kv_artifact_bytes = 0_i64
if path = kv_out
  kv_snapshot = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
    snapshot.max_seq,
    snapshot.layer_count,
    snapshot.positions.dup,
    snapshot.records.reject { |record| recurrent_record?(record.kind) },
  )
  kv_artifact = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(kv_snapshot)
  File.open(path, "w") { |file| file.write(kv_artifact) }
  kv_artifact_bytes = kv_artifact.size.to_i64
end
baseline_artifacts = [
  {"recurrent-bf16", nil},
  {"recurrent-int8", block_size},
].map do |codec, codec_block|
  started = Time.instant
  bytes = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(
    snapshot,
    artifact_codec: codec,
    artifact_codec_block: codec_block,
  )
  if codec == "recurrent-int8" && (path = int8_out)
    File.open(path, "w") { |file| file.write(bytes) }
  end
  encode_ms = (Time.instant - started).total_milliseconds
  encoded = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(
    bytes,
    expected_codec: codec,
    expected_codec_block: codec_block,
    copy_payloads: false,
  )
  target = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(target, weights.hparams) if prepare_state
  started = Time.instant
  ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(encoded, weights.hparams, target, prefer_metal: prepare_state)
  cold_restore_ms = (Time.instant - started).total_milliseconds
  restore_samples = Array(Float64).new(restore_repeats)
  restore_repeats.times do
    started = Time.instant
    ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(encoded, weights.hparams, target, prefer_metal: prepare_state)
    restore_samples << (Time.instant - started).total_milliseconds
  end
  restore_median_ms = median(restore_samples)
  release_metal_state!(target)
  {codec, bytes.size.to_i64, encode_ms, cold_restore_ms, restore_median_ms}
end
# Artifact bytes and decoded views are no longer part of the benchmark working
# set. Reclaim them before QBit states are prepared on the unified-memory host.
GC.collect

puts "qwen35_qbit_cache_probe"
puts "  model=#{model_path}"
puts "  prompt=#{prompt.inspect} chat=#{chat_mode} prompt_tokens=#{tokens.size} gen=#{n_gen} max_seq=#{max_seq} block_size=#{block_size} restore_repeats=#{restore_repeats}"
puts "  startup_ms=#{startup_ms.round(3)} prefill_ms=#{prefill_ms.round(3)} snapshot_ms=#{snapshot_ms.round(3)}"
puts "  raw_total_bytes=#{snapshot.byte_size} recurrent_raw_bytes=#{recurrent_raw_bytes} kv_raw_bytes=#{kv_raw_bytes}"
puts "  kv_artifact_bytes=#{kv_artifact_bytes}" if kv_artifact_bytes > 0
if ML::Metal::Device.available?
  device = ML::Metal::Device.instance
  puts "  metal_allocated_bytes=#{device.current_allocated_size} metal_recommended_working_set_bytes=#{device.recommended_working_set_size}"
end
baseline_artifacts.each do |codec, bytes, encode_ms, cold_restore_ms, restore_median_ms|
  puts "  baseline_codec=#{codec} artifact_bytes=#{bytes} ratio=#{(bytes.to_f64 / snapshot.byte_size).round(6)} encode_ms=#{encode_ms.round(3)} prepared_restore_cold_ms=#{cold_restore_ms.round(3)} prepared_restore_median_ms=#{restore_median_ms.round(3)}"
end
puts "  exact_ids=#{exact_ids.join(',')}"

precisions.each do |precision|
  quantized, payload_bytes, encode_ms, decode_ms = qbit_snapshot(snapshot, block_size, precision)
  # `qbit_snapshot` materializes a scalar reference solely as a falsifier.
  # Reclaim it before measuring the device restore path.
  GC.collect
  native_bytes = 0_i64
  native_encode_ms = 0.0_f64
  native_read_ms = 0.0_f64
  native_parse_ms = 0.0_f64
  native_response_bytes = 0_i64
  native_block_count = 0
  parsed_native : ML::GGUF::QwenQBitNativeBlock::Parsed? = nil
  parsed_native_stream : ML::GGUF::QwenQBitNativeBlock::Stream? = nil
  if precision == 7
    native_started = Time.instant
    native_block = ML::GGUF::QwenQBitStateSnapshot.encode_native_recurrent(quantized, effective_cache_id)
    native_encode_ms = (Time.instant - native_started).total_milliseconds
    native_bytes = native_block.size.to_i64
    if path = native_out
      File.open(path, "w") { |file| file.write(native_block) }
    end
    if path = native_in
      native_started = Time.instant
      response = read_bytes(path, native_limit_bytes)
      native_read_ms = (Time.instant - native_started).total_milliseconds
      native_response_bytes = response.size.to_i64
      native_started = Time.instant
      parsed_native_stream = ML::GGUF::QwenQBitNativeBlock.parse_stream(response)
      native_block_count = parsed_native_stream.not_nil!.blocks.size
    else
      native_started = Time.instant
      parsed_native = ML::GGUF::QwenQBitNativeBlock.parse(native_block)
      native_response_bytes = native_bytes
      native_block_count = 1
    end
    native_parse_ms = (Time.instant - native_started).total_milliseconds
  end

  direct_metal = prepare_state && precision == 7
  free_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(free_state, weights.hparams) if direct_metal
  restore_started = Time.instant
  if direct_metal
    if stream = parsed_native_stream
      ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(quantized, stream, cache_id.not_nil!, weights.hparams, free_state)
    else
      ML::GGUF::QwenQBitStateSnapshot.restore_native_into(quantized, parsed_native.not_nil!, 0_u64, weights.hparams, free_state)
    end
  else
    ML::GGUF::QwenQBitStateSnapshot.restore_into(quantized, weights.hparams, free_state, prefer_metal: false)
  end
  restore_ms = (Time.instant - restore_started).total_milliseconds
  restore_samples = Array(Float64).new(restore_repeats)
  restore_repeats.times do
    restore_started = Time.instant
    if direct_metal
      if stream = parsed_native_stream
        ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(quantized, stream, cache_id.not_nil!, weights.hparams, free_state)
      else
        ML::GGUF::QwenQBitStateSnapshot.restore_native_into(quantized, parsed_native.not_nil!, 0_u64, weights.hparams, free_state)
      end
    else
      ML::GGUF::QwenQBitStateSnapshot.restore_into(quantized, weights.hparams, free_state, prefer_metal: false)
    end
    restore_samples << (Time.instant - restore_started).total_milliseconds
  end
  restore_median_ms = median(restore_samples)
  free_ids, _free_logits = continuation(weights, free_state, first_token, first_logit, tokens.size.to_i32, n_gen)

  forced_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(forced_state, weights.hparams) if direct_metal
  forced_restore_started = Time.instant
  if direct_metal
    if stream = parsed_native_stream
      ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(quantized, stream, cache_id.not_nil!, weights.hparams, forced_state)
    else
      ML::GGUF::QwenQBitStateSnapshot.restore_native_into(quantized, parsed_native.not_nil!, 0_u64, weights.hparams, forced_state)
    end
  else
    ML::GGUF::QwenQBitStateSnapshot.restore_into(quantized, weights.hparams, forced_state, prefer_metal: false)
  end
  forced_restore_ms = (Time.instant - forced_restore_started).total_milliseconds
  forced_ids, forced_logits = teacher_forced_predictions(weights, forced_state, exact_ids, tokens.size.to_i32)
  expected_forced = exact_ids[1, exact_ids.size - 1]
  forced_matches = forced_ids.each_with_index.count { |id, i| id == expected_forced[i] }
  matched_logit_deltas = forced_ids.each_with_index.compact_map do |id, i|
    id == expected_forced[i] ? (forced_logits[i] - exact_logits[i + 1]).abs.to_f64 : nil
  end
  max_matched_logit_delta = matched_logit_deltas.max? || Float64::NAN
  prefix = common_prefix(exact_ids, free_ids)
  first_divergence = prefix == exact_ids.size ? -1 : prefix
  ratio = payload_bytes.to_f64 / snapshot.byte_size

  restore_route = if direct_metal
                    parsed_native_stream ? "native-stream-metal-direct" : "native-metal-direct"
                  else
                    "cpu-reference"
                  end
  puts "  p#{precision} payload_bytes=#{payload_bytes} ratio=#{ratio.round(6)} encode_ms=#{encode_ms.round(3)} cpu_decode_ms=#{decode_ms.round(3)} native_cache_id=#{effective_cache_id} native_recurrent_bytes=#{native_bytes} native_response_bytes=#{native_response_bytes} native_block_count=#{native_block_count} native_encode_ms=#{native_encode_ms.round(3)} native_read_ms=#{native_read_ms.round(3)} native_parse_ms=#{native_parse_ms.round(3)} restore_route=#{restore_route} restore_cold_ms=#{restore_ms.round(3)} restore_median_ms=#{restore_median_ms.round(3)} forced_restore_ms=#{forced_restore_ms.round(3)} free_prefix=#{prefix}/#{exact_ids.size} first_divergence=#{first_divergence} forced_top1=#{forced_matches}/#{expected_forced.size} max_matched_top1_logit_delta=#{max_matched_logit_delta.round(6)}"
  if direct_metal
    puts "    metal_allocated_bytes=#{ML::Metal::Device.instance.current_allocated_size}"
  end
  puts "    free_ids=#{free_ids.join(',')}"
  release_metal_state!(free_state)
  release_metal_state!(forced_state)
  GC.collect
end
