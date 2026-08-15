# Default-off quality and size falsifier for QBit-style Gaussian compression of
# recurrent Qwen cache state. This probe deliberately decodes on CPU and then
# restores Float32 state; it does not claim production restore latency.

require "option_parser"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"

DEFAULT_MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
prompt = "The capital of France is"
n_gen = 16
max_seq = 64
block_size = 1024
precisions = [8, 7, 6] of Int32
prepare_state = true
chat_mode = false

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
                  precision : Int32) : {ML::GGUF::Qwen35StateSnapshot::Snapshot, Int64, Float64, Float64}
  codec = ML::GGUF::QwenQBitGaussianCodec
  payload_bytes = 0_i64
  encode_ms = 0.0_f64
  decode_ms = 0.0_f64
  records = snapshot.records.map do |record|
    unless recurrent_record?(record.kind)
      payload_bytes += record.bytes.size
      next record
    end

    values = floats_from(record.bytes)
    started = Time.instant
    encoded = codec.encode(values, block_size: block_size, precision: precision)
    encode_ms += (Time.instant - started).total_milliseconds
    payload_bytes += encoded.payload.size

    started = Time.instant
    decoded = codec.decode(encoded)
    decode_ms += (Time.instant - started).total_milliseconds
    ML::GGUF::Qwen35StateSnapshot::Record.new(
      record.layer,
      record.kind,
      bytes_from(decoded),
      record.storage_mode,
    )
  end

  {
    ML::GGUF::Qwen35StateSnapshot::Snapshot.new(snapshot.max_seq, snapshot.layer_count, snapshot.positions, records),
    payload_bytes,
    encode_ms,
    decode_ms,
  }
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

exact_state = ML::GGUF::Qwen35StateSnapshot.restore(snapshot, weights.hparams)
exact_ids, exact_logits = continuation(weights, exact_state, first_token, first_logit, tokens.size.to_i32, n_gen)

recurrent_raw_bytes = snapshot.records.sum(0_i64) { |record| recurrent_record?(record.kind) ? record.bytes.size.to_i64 : 0_i64 }
kv_raw_bytes = snapshot.byte_size - recurrent_raw_bytes
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
  {codec, bytes.size.to_i64, (Time.instant - started).total_milliseconds}
end

puts "qwen35_qbit_cache_probe"
puts "  model=#{model_path}"
puts "  prompt=#{prompt.inspect} chat=#{chat_mode} prompt_tokens=#{tokens.size} gen=#{n_gen} max_seq=#{max_seq} block_size=#{block_size}"
puts "  startup_ms=#{startup_ms.round(3)} prefill_ms=#{prefill_ms.round(3)} snapshot_ms=#{snapshot_ms.round(3)}"
puts "  raw_total_bytes=#{snapshot.byte_size} recurrent_raw_bytes=#{recurrent_raw_bytes} kv_raw_bytes=#{kv_raw_bytes}"
baseline_artifacts.each do |codec, bytes, encode_ms|
  puts "  baseline_codec=#{codec} artifact_bytes=#{bytes} ratio=#{(bytes.to_f64 / snapshot.byte_size).round(6)} encode_ms=#{encode_ms.round(3)}"
end
puts "  exact_ids=#{exact_ids.join(',')}"

precisions.each do |precision|
  quantized, payload_bytes, encode_ms, decode_ms = qbit_snapshot(snapshot, block_size, precision)

  restore_started = Time.instant
  free_state = ML::GGUF::Qwen35StateSnapshot.restore(quantized, weights.hparams)
  restore_ms = (Time.instant - restore_started).total_milliseconds
  free_ids, _free_logits = continuation(weights, free_state, first_token, first_logit, tokens.size.to_i32, n_gen)

  forced_state = ML::GGUF::Qwen35StateSnapshot.restore(quantized, weights.hparams)
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

  puts "  p#{precision} payload_bytes=#{payload_bytes} ratio=#{ratio.round(6)} encode_ms=#{encode_ms.round(3)} decode_ms=#{decode_ms.round(3)} raw_restore_ms=#{restore_ms.round(3)} free_prefix=#{prefix}/#{exact_ids.size} first_divergence=#{first_divergence} forced_top1=#{forced_matches}/#{expected_forced.size} max_matched_top1_logit_delta=#{max_matched_logit_delta.round(6)}"
  puts "    free_ids=#{free_ids.join(',')}"
end
