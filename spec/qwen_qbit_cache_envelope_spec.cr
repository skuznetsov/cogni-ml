require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_cache_envelope"

private def qbit_envelope_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  bytes.copy_from(Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size))
  bytes
end

private def qbit_envelope_context(template : String = "{{ messages }}") : ML::GGUF::QwenQBitCacheEnvelope::Context
  tokens = [11_i32, 22_i32, 33_i32, 44_i32]
  state_abi = ML::GGUF::QwenQBitCacheEnvelope::StateABI.new(
    layer_count: 2,
    full_attention_interval: 2,
    kv_record_byte_size: 3_i64 * sizeof(Float32),
    conv_record_byte_size: 13_i64 * sizeof(Float32),
    ssm_record_byte_size: 2_i64 * sizeof(Float32),
  )
  ML::GGUF::QwenQBitCacheEnvelope::Context.new(
    model_id: "model-a",
    tokenizer_id: "tokenizer-a",
    template_id: ML::GGUF::QwenQBitCacheEnvelope.template_id(template),
    prompt_hash: ML::GGUF::Qwen35PromptCache.prompt_hash(tokens[0, 3], "rendered prompt"),
    token_hash: ML::GGUF::Qwen35PromptCache.token_hash(tokens, 3),
    prefix_len: 3,
    max_seq: 16,
    layer_count: 2,
    qbit_block_size: 8,
    qbit_precision: 7,
    validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
    validation_steps: 1,
    validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(tokens),
    next_token_id: tokens.last,
    state_abi: state_abi,
  )
end

private def qbit_envelope_artifacts(context : ML::GGUF::QwenQBitCacheEnvelope::Context) : {Bytes, Bytes}
  codec = ML::GGUF::QwenQBitGaussianCodec
  cache_id = ML::GGUF::QwenQBitCacheEnvelope.cache_id(context)
  native = ML::GGUF::QwenQBitNativeWriter.encode([
    ML::GGUF::QwenQBitNativeWriter::Record.new(
      cache_id,
      0_i32,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState.value,
      codec.encode(Array(Float32).new(13) { |i| (i - 6).to_f32 / 3.0_f32 }, 8, 7),
    ),
    ML::GGUF::QwenQBitNativeWriter::Record.new(
      cache_id,
      0_i32,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState.value,
      codec.encode([1.0_f32, -1.0_f32], 8, 7),
    ),
  ])
  exact = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
    context.max_seq,
    context.layer_count,
    [context.prefix_len, context.prefix_len],
    [
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
        qbit_envelope_bytes([1.0_f32, 2.0_f32, 3.0_f32]),
        ML::StorageMode::Shared,
      ),
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache,
        qbit_envelope_bytes([4.0_f32, 5.0_f32, 6.0_f32]),
        ML::StorageMode::Shared,
      ),
    ],
  )
  {native, ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(exact)}
end

describe ML::GGUF::QwenQBitCacheEnvelope do
  envelope = ML::GGUF::QwenQBitCacheEnvelope

  it "separates missing, empty, and changed chat templates" do
    envelope.template_id(nil).should_not eq(envelope.template_id(""))
    envelope.template_id("template-a").should_not eq(envelope.template_id("template-b"))
    envelope.template_id("template-a").should eq(envelope.template_id("template-a"))
  end

  it "builds a deterministic versioned envelope and admits its exact artifacts" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 123_i64)

    entry.schema_id.should eq(ML::GGUF::QwenQBitCacheEnvelope::SCHEMA_ID)
    entry.cache_id.should eq(envelope.cache_id(context))
    entry.certificate_id.should eq(envelope.certificate_id(entry))

    round_trip = ML::GGUF::QwenQBitCacheEnvelope::Entry.from_json(entry.to_json)
    admitted = envelope.admit(round_trip, context, native, kv)
    admitted.entry.certificate_id.should eq(entry.certificate_id)
    admitted.native_stream.record_spans.size.should eq(2)
    admitted.exact_artifact.records.size.should eq(2)
  end

  it "rejects a self-consistent but incomplete model state" do
    context = qbit_envelope_context
    _native, kv = qbit_envelope_artifacts(context)
    cache_id = envelope.cache_id(context)
    incomplete = ML::GGUF::QwenQBitNativeWriter.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(
        cache_id,
        0_i32,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState.value,
        ML::GGUF::QwenQBitGaussianCodec.encode(Array(Float32).new(13, 1.0_f32), 8, 7),
      ),
    ])

    expect_raises(ArgumentError, /record set/) { envelope.build(context, incomplete, kv) }
  end

  it "fails closed on context, logical QBit content, and exact KV changes" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv)

    wrong_template = qbit_envelope_context("different template")
    expect_raises(ArgumentError, /template/) { envelope.admit(entry, wrong_template, native, kv) }

    changed_native = native.dup
    parsed = ML::GGUF::QwenQBitNativeBlock.parse_stream(changed_native)
    changed_native[parsed.blocks.first.codes_offset] ^= 0x80_u8
    expect_raises(ArgumentError, /logical checksum/) { envelope.admit(entry, context, changed_native, kv) }

    changed_kv = kv.dup
    changed_kv[changed_kv.size - 1] ^= 0x01_u8
    expect_raises(ArgumentError, /KV checksum/) { envelope.admit(entry, context, native, changed_kv) }
  end

  it "rejects a structurally valid response under another derived cache identity" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv)
    encoded = ML::GGUF::QwenQBitGaussianCodec.encode([1.0_f32, 2.0_f32], 8, 7)
    wrong_native = ML::GGUF::QwenQBitNativeWriter.encode([
      ML::GGUF::QwenQBitNativeWriter::Record.new(
        entry.cache_id &+ 1_u64,
        0_i32,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState.value,
        encoded,
      ),
    ])

    expect_raises(ArgumentError, /cache identity/) { envelope.admit(entry, context, wrong_native, kv) }
  end

  it "rejects a manifest whose covered admission fields were changed" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv)
    entry.recurrent_record_count += 1

    expect_raises(ArgumentError, /certificate/) { envelope.admit(entry, context, native, kv) }
  end
end
