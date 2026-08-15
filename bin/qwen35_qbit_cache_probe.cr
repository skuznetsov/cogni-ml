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
kv_in : String? = nil
int8_in : String? = nil
cache_id : UInt64? = nil
native_max_mib = 256
artifact_max_mib = 256
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
  parser.on("--kv-in PATH", "Attach exact KV from a raw KV-only artifact to --native-in") { |value| kv_in = value }
  parser.on("--int8-in PATH", "Measure a complete recurrent-INT8 artifact corridor") { |value| int8_in = value }
  parser.on("--cache-id ID", "Expected UInt64 Native cache identity (required with --native-in)") { |value| cache_id = value.to_u64 }
  parser.on("--native-max-mib N", "Maximum accepted Native response size (default: 256, hard max: 1024)") { |value| native_max_mib = value.to_i }
  parser.on("--artifact-max-mib N", "Maximum accepted KV/INT8 artifact size (default: 256, hard max: 1024)") { |value| artifact_max_mib = value.to_i }
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
raise "--kv-in requires --native-in" if kv_in && native_in.nil?
raise "--kv-in file does not exist" if (path = kv_in) && !File.file?(path)
raise "--int8-in requires prepared Metal state" if int8_in && !prepare_state
raise "--int8-in file does not exist" if (path = int8_in) && !File.file?(path)
raise "--native-max-mib must be within 1..1024" unless native_max_mib.in?(1..1024)
raise "--artifact-max-mib must be within 1..1024" unless artifact_max_mib.in?(1..1024)
output_paths = [native_out, kv_out, int8_out].compact
raise "diagnostic output paths must be distinct" unless output_paths.uniq.size == output_paths.size
input_paths = [native_in, kv_in, int8_in].compact
raise "diagnostic input paths must be distinct" unless input_paths.uniq.size == input_paths.size
raise "diagnostic input and output paths must not overlap" unless (input_paths & output_paths).empty?
native_limit_bytes = native_max_mib.to_i64 * 1024 * 1024
artifact_limit_bytes = artifact_max_mib.to_i64 * 1024 * 1024
if path = native_in
  native_input_size = File.size(path)
  unless native_input_size > 0 && native_input_size <= native_limit_bytes
    raise "--native-in size is outside 1..#{native_limit_bytes} bytes"
  end
end
[kv_in, int8_in].compact.each do |path|
  artifact_input_size = File.size(path)
  unless artifact_input_size > 0 && artifact_input_size <= artifact_limit_bytes
    raise "diagnostic artifact size is outside 1..#{artifact_limit_bytes} bytes"
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

def read_bytes(path : String, max_bytes : Int64, label : String) : Bytes
  size = File.size(path)
  unless size > 0 && size <= max_bytes && size <= Int32::MAX
    raise "#{label} size changed outside the admitted limit"
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

def validate_complete_int8_artifact!(artifact : ML::GGUF::Qwen35StateSnapshot::EncodedSnapshot,
                                     expected : ML::GGUF::Qwen35StateSnapshot::Snapshot) : Nil
  raise "INT8 artifact max_seq mismatch" unless artifact.max_seq == expected.max_seq
  raise "INT8 artifact layer count mismatch" unless artifact.layer_count == expected.layer_count
  raise "INT8 artifact positions mismatch" unless artifact.positions == expected.positions

  actual_records = Hash({Int32, UInt8}, ML::GGUF::Qwen35StateSnapshot::EncodedRecord).new
  artifact.records.each do |record|
    key = {record.layer, record.kind.value}
    raise "duplicate INT8 artifact record" if actual_records.has_key?(key)
    actual_records[key] = record
  end
  expected_keys = expected.records.map { |record| {record.layer, record.kind.value} }.to_set
  raise "INT8 artifact record set mismatch" unless actual_records.keys.to_set == expected_keys

  expected.records.each do |record|
    actual = actual_records[{record.layer, record.kind.value}]
    raise "INT8 artifact storage mode mismatch" unless actual.storage_mode == record.storage_mode
    raise "INT8 artifact byte size mismatch" unless actual.original_byte_size == record.bytes.size
    expected_codec = recurrent_record?(record.kind) ? ML::GGUF::Qwen35StateSnapshot::RecordCodec::BlockI8 : ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32
    raise "INT8 artifact record codec mismatch" unless actual.codec == expected_codec
  end
end

record CacheHitCorridorSample,
  primary_read_ms : Float64,
  secondary_read_ms : Float64,
  parse_validate_ms : Float64,
  restore_ms : Float64,
  forward_ms : Float64,
  total_ms : Float64,
  post_id : Int32,
  logit_delta : Float64

def measure_qbit_cache_hit(native_path : String,
                           kv_path : String,
                           native_limit_bytes : Int64,
                           artifact_limit_bytes : Int64,
                           template : ML::GGUF::QwenQBitStateSnapshot::Snapshot,
                           cache_id : UInt64,
                           weights : ML::GGUF::Qwen35Weights,
                           state : ML::GGUF::Qwen35CPU::State,
                           first_token : Int32,
                           position : Int32,
                           expected_id : Int32,
                           expected_logit : Float32) : CacheHitCorridorSample
  total_started = Time.instant
  phase_started = Time.instant
  native_response = read_bytes(native_path, native_limit_bytes, "Native response")
  native_read_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  kv_response = read_bytes(kv_path, artifact_limit_bytes, "exact KV artifact")
  kv_read_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  native_stream = ML::GGUF::QwenQBitNativeBlock.parse_stream(native_response)
  kv_artifact = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(kv_response, copy_payloads: false)
  restore_snapshot = ML::GGUF::QwenQBitStateSnapshot.with_exact_artifact(template, kv_artifact)
  parse_validate_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(restore_snapshot, native_stream, cache_id, weights.hparams, state)
  restore_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  post_id, post_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, first_token, position, state)
  forward_ms = (Time.instant - phase_started).total_milliseconds
  total_ms = (Time.instant - total_started).total_milliseconds
  raise "QBit corridor first post-restore token mismatch" unless post_id == expected_id

  CacheHitCorridorSample.new(
    native_read_ms,
    kv_read_ms,
    parse_validate_ms,
    restore_ms,
    forward_ms,
    total_ms,
    post_id,
    (post_logit - expected_logit).abs.to_f64,
  )
end

def measure_int8_cache_hit(path : String,
                           artifact_limit_bytes : Int64,
                           block_size : Int32,
                           expected_snapshot : ML::GGUF::Qwen35StateSnapshot::Snapshot,
                           weights : ML::GGUF::Qwen35Weights,
                           state : ML::GGUF::Qwen35CPU::State,
                           first_token : Int32,
                           position : Int32,
                           expected_id : Int32,
                           expected_logit : Float32) : CacheHitCorridorSample
  total_started = Time.instant
  phase_started = Time.instant
  artifact_bytes = read_bytes(path, artifact_limit_bytes, "INT8 artifact")
  read_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  artifact = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(
    artifact_bytes,
    expected_codec: "recurrent-int8",
    expected_codec_block: block_size,
    copy_payloads: false,
  )
  validate_complete_int8_artifact!(artifact, expected_snapshot)
  parse_validate_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(artifact, weights.hparams, state, prefer_metal: true)
  restore_ms = (Time.instant - phase_started).total_milliseconds

  phase_started = Time.instant
  post_id, post_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, first_token, position, state)
  forward_ms = (Time.instant - phase_started).total_milliseconds
  total_ms = (Time.instant - total_started).total_milliseconds
  raise "INT8 corridor first post-restore token mismatch" unless post_id == expected_id

  CacheHitCorridorSample.new(
    read_ms,
    0.0_f64,
    parse_validate_ms,
    restore_ms,
    forward_ms,
    total_ms,
    post_id,
    (post_logit - expected_logit).abs.to_f64,
  )
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

def print_qbit_corridor(samples : Array(CacheHitCorridorSample), expected_id : Int32) : Float64
  total_median = median(samples.map(&.total_ms))
  puts "  qbit_complete_corridor repeats=#{samples.size} native_read_median_ms=#{median(samples.map(&.primary_read_ms)).round(3)} kv_read_median_ms=#{median(samples.map(&.secondary_read_ms)).round(3)} parse_validate_median_ms=#{median(samples.map(&.parse_validate_ms)).round(3)} restore_median_ms=#{median(samples.map(&.restore_ms)).round(3)} first_post_restore_forward_median_ms=#{median(samples.map(&.forward_ms)).round(3)} total_median_ms=#{total_median.round(3)} first_post_restore_id=#{expected_id} max_logit_delta=#{samples.max_of(&.logit_delta).round(6)} prepared_state=true"
  total_median
end

def print_int8_corridor(samples : Array(CacheHitCorridorSample), expected_id : Int32) : Float64
  total_median = median(samples.map(&.total_ms))
  puts "  int8_complete_corridor repeats=#{samples.size} read_median_ms=#{median(samples.map(&.primary_read_ms)).round(3)} parse_validate_median_ms=#{median(samples.map(&.parse_validate_ms)).round(3)} restore_median_ms=#{median(samples.map(&.restore_ms)).round(3)} first_post_restore_forward_median_ms=#{median(samples.map(&.forward_ms)).round(3)} total_median_ms=#{total_median.round(3)} first_post_restore_id=#{expected_id} max_logit_delta=#{samples.max_of(&.logit_delta).round(6)} prepared_state=true"
  total_median
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

qbit_corridor_template : ML::GGUF::QwenQBitStateSnapshot::Snapshot? = nil
precisions.each do |precision|
  quantized, payload_bytes, encode_ms, decode_ms = qbit_snapshot(snapshot, block_size, precision)
  qbit_corridor_template = quantized if precision == 7
  # `qbit_snapshot` materializes a scalar reference solely as a falsifier.
  # Reclaim it before measuring the device restore path.
  GC.collect
  native_bytes = 0_i64
  native_encode_ms = 0.0_f64
  native_read_ms = 0.0_f64
  native_parse_ms = 0.0_f64
  native_response_bytes = 0_i64
  native_block_count = 0
  kv_response_bytes = 0_i64
  kv_read_ms = 0.0_f64
  kv_parse_ms = 0.0_f64
  parsed_native : ML::GGUF::QwenQBitNativeBlock::Parsed? = nil
  parsed_native_stream : ML::GGUF::QwenQBitNativeBlock::Stream? = nil
  restore_snapshot = quantized
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
      response = read_bytes(path, native_limit_bytes, "Native response")
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
    if path = kv_in
      kv_started = Time.instant
      kv_response = read_bytes(path, artifact_limit_bytes, "exact KV artifact")
      kv_read_ms = (Time.instant - kv_started).total_milliseconds
      kv_response_bytes = kv_response.size.to_i64
      kv_started = Time.instant
      kv_artifact = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(kv_response, copy_payloads: false)
      restore_snapshot = ML::GGUF::QwenQBitStateSnapshot.with_exact_artifact(quantized, kv_artifact)
      kv_parse_ms = (Time.instant - kv_started).total_milliseconds
    end
  end

  direct_metal = prepare_state && precision == 7
  free_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(free_state, weights.hparams) if direct_metal
  restore_started = Time.instant
  if direct_metal
    if stream = parsed_native_stream
      ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(restore_snapshot, stream, cache_id.not_nil!, weights.hparams, free_state)
    else
      ML::GGUF::QwenQBitStateSnapshot.restore_native_into(restore_snapshot, parsed_native.not_nil!, 0_u64, weights.hparams, free_state)
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
        ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(restore_snapshot, stream, cache_id.not_nil!, weights.hparams, free_state)
      else
        ML::GGUF::QwenQBitStateSnapshot.restore_native_into(restore_snapshot, parsed_native.not_nil!, 0_u64, weights.hparams, free_state)
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
      ML::GGUF::QwenQBitStateSnapshot.restore_native_stream_into(restore_snapshot, stream, cache_id.not_nil!, weights.hparams, forced_state)
    else
      ML::GGUF::QwenQBitStateSnapshot.restore_native_into(restore_snapshot, parsed_native.not_nil!, 0_u64, weights.hparams, forced_state)
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
  puts "  p#{precision} payload_bytes=#{payload_bytes} ratio=#{ratio.round(6)} encode_ms=#{encode_ms.round(3)} cpu_decode_ms=#{decode_ms.round(3)} native_cache_id=#{effective_cache_id} native_recurrent_bytes=#{native_bytes} native_response_bytes=#{native_response_bytes} native_block_count=#{native_block_count} native_encode_ms=#{native_encode_ms.round(3)} native_read_ms=#{native_read_ms.round(3)} native_parse_ms=#{native_parse_ms.round(3)} kv_response_bytes=#{kv_response_bytes} kv_read_ms=#{kv_read_ms.round(3)} kv_parse_ms=#{kv_parse_ms.round(3)} restore_route=#{restore_route} restore_cold_ms=#{restore_ms.round(3)} restore_median_ms=#{restore_median_ms.round(3)} forced_restore_ms=#{forced_restore_ms.round(3)} free_prefix=#{prefix}/#{exact_ids.size} first_divergence=#{first_divergence} forced_top1=#{forced_matches}/#{expected_forced.size} max_matched_top1_logit_delta=#{max_matched_logit_delta.round(6)}"
  if direct_metal
    puts "    metal_allocated_bytes=#{ML::Metal::Device.instance.current_allocated_size}"
  end
  puts "    free_ids=#{free_ids.join(',')}"
  release_metal_state!(free_state)
  release_metal_state!(forced_state)
  GC.collect
end

if native_in && kv_in && int8_in
  qbit_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  int8_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(qbit_state, weights.hparams)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(int8_state, weights.hparams)
  qbit_samples = [] of CacheHitCorridorSample
  int8_samples = [] of CacheHitCorridorSample
  paired_total_deltas = [] of Float64
  paired_ready_deltas = [] of Float64

  restore_repeats.times do |index|
    qbit_sample : CacheHitCorridorSample
    int8_sample : CacheHitCorridorSample
    if index.even?
      qbit_sample = measure_qbit_cache_hit(native_in.not_nil!, kv_in.not_nil!, native_limit_bytes, artifact_limit_bytes, qbit_corridor_template.not_nil!, cache_id.not_nil!, weights, qbit_state, first_token, tokens.size.to_i32, exact_ids[1], exact_logits[1])
      int8_sample = measure_int8_cache_hit(int8_in.not_nil!, artifact_limit_bytes, block_size, snapshot, weights, int8_state, first_token, tokens.size.to_i32, exact_ids[1], exact_logits[1])
    else
      int8_sample = measure_int8_cache_hit(int8_in.not_nil!, artifact_limit_bytes, block_size, snapshot, weights, int8_state, first_token, tokens.size.to_i32, exact_ids[1], exact_logits[1])
      qbit_sample = measure_qbit_cache_hit(native_in.not_nil!, kv_in.not_nil!, native_limit_bytes, artifact_limit_bytes, qbit_corridor_template.not_nil!, cache_id.not_nil!, weights, qbit_state, first_token, tokens.size.to_i32, exact_ids[1], exact_logits[1])
    end
    qbit_samples << qbit_sample
    int8_samples << int8_sample
    paired_total_deltas << qbit_sample.total_ms - int8_sample.total_ms
    qbit_ready_ms = qbit_sample.primary_read_ms + qbit_sample.secondary_read_ms + qbit_sample.parse_validate_ms + qbit_sample.restore_ms
    int8_ready_ms = int8_sample.primary_read_ms + int8_sample.parse_validate_ms + int8_sample.restore_ms
    paired_ready_deltas << qbit_ready_ms - int8_ready_ms
  end

  qbit_total_median = print_qbit_corridor(qbit_samples, exact_ids[1])
  int8_total_median = print_int8_corridor(int8_samples, exact_ids[1])
  puts "  paired_complete_corridor order=ABBA qbit_minus_int8_ready_median_ms=#{median(paired_ready_deltas).round(3)} qbit_minus_int8_total_median_ms=#{median(paired_total_deltas).round(3)} qbit_vs_int8_total_median_ratio=#{(qbit_total_median / int8_total_median).round(6)}"
  release_metal_state!(qbit_state)
  release_metal_state!(int8_state)
  GC.collect
elsif native_in && kv_in
  qbit_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(qbit_state, weights.hparams)
  qbit_samples = Array(CacheHitCorridorSample).new(restore_repeats)
  restore_repeats.times do
    qbit_samples << measure_qbit_cache_hit(native_in.not_nil!, kv_in.not_nil!, native_limit_bytes, artifact_limit_bytes, qbit_corridor_template.not_nil!, cache_id.not_nil!, weights, qbit_state, first_token, tokens.size.to_i32, exact_ids[1], exact_logits[1])
  end
  print_qbit_corridor(qbit_samples, exact_ids[1])
  release_metal_state!(qbit_state)
  GC.collect
elsif int8_in
  int8_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: snapshot.max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(int8_state, weights.hparams)
  int8_samples = Array(CacheHitCorridorSample).new(restore_repeats)
  restore_repeats.times do
    int8_samples << measure_int8_cache_hit(int8_in.not_nil!, artifact_limit_bytes, block_size, snapshot, weights, int8_state, first_token, tokens.size.to_i32, exact_ids[1], exact_logits[1])
  end
  print_int8_corridor(int8_samples, exact_ids[1])
  release_metal_state!(int8_state)
  GC.collect
end
