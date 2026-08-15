require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_cache_envelope"
require "../src/ml/gguf/qwen35_qbit_runtime_cache"

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

  it "looks up by request-known identity without requiring cached outcomes" do
    context = qbit_envelope_context
    lookup = envelope.lookup_context(context)
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 123_i64)
    different_outcome = ML::GGUF::QwenQBitCacheEnvelope::Context.new(
      model_id: context.model_id,
      tokenizer_id: context.tokenizer_id,
      template_id: context.template_id,
      prompt_hash: context.prompt_hash,
      token_hash: context.token_hash,
      prefix_len: context.prefix_len,
      max_seq: context.max_seq,
      layer_count: context.layer_count,
      qbit_block_size: context.qbit_block_size,
      qbit_precision: context.qbit_precision,
      validation_kind: context.validation_kind,
      validation_steps: 2,
      validation_hash: "0" * 64,
      next_token_id: context.next_token_id + 1,
      state_abi: context.state_abi,
    )

    envelope.lookup_key(lookup).should eq(envelope.lookup_key(context))
    envelope.lookup_key(different_outcome).should eq(envelope.lookup_key(context))
    envelope.cache_id(lookup).should eq(envelope.cache_id(context))
    envelope.validate_manifest!(entry, lookup)
    admitted = envelope.admit(entry, lookup, native, kv)
    admitted.entry.next_token_id.should eq(context.next_token_id)
  end

  it "derives a prompt-independent prefix scope and strictly admits a prefix candidate" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 123_i64)
    prefix = envelope.prefix_context(context)
    different_prompt = ML::GGUF::QwenQBitCacheEnvelope::LookupContext.new(
      model_id: context.model_id,
      tokenizer_id: context.tokenizer_id,
      template_id: context.template_id,
      prompt_hash: "1" * 64,
      token_hash: "2" * 64,
      prefix_len: 2,
      max_seq: context.max_seq,
      layer_count: context.layer_count,
      qbit_block_size: context.qbit_block_size,
      qbit_precision: context.qbit_precision,
      state_abi: context.state_abi,
    )

    envelope.prefix_scope_key(prefix).should eq(
      envelope.prefix_scope_key(envelope.prefix_context(different_prompt))
    )
    candidate_lookup = envelope.validate_prefix_manifest!(
      entry,
      prefix,
      context.token_hash,
      context.prefix_len,
    )
    envelope.lookup_key(candidate_lookup).should eq(envelope.lookup_key(context))
    envelope.admit(entry, candidate_lookup, native, kv).entry.certificate_id.should eq(entry.certificate_id)

    expect_raises(ArgumentError, /token hash/) do
      envelope.validate_prefix_manifest!(entry, prefix, "3" * 64, context.prefix_len)
    end
    expect_raises(ArgumentError, /prefix length/) do
      envelope.validate_prefix_manifest!(entry, prefix, context.token_hash, context.prefix_len - 1)
    end

    wrong_max_seq = ML::GGUF::QwenQBitCacheEnvelope::PrefixContext.new(
      model_id: prefix.model_id,
      tokenizer_id: prefix.tokenizer_id,
      template_id: prefix.template_id,
      max_seq: prefix.max_seq + 1,
      layer_count: prefix.layer_count,
      qbit_block_size: prefix.qbit_block_size,
      qbit_precision: prefix.qbit_precision,
      state_abi: prefix.state_abi,
    )
    envelope.prefix_scope_key(wrong_max_seq).should_not eq(envelope.prefix_scope_key(prefix))
    expect_raises(ArgumentError, /max_seq/) do
      envelope.validate_prefix_manifest!(entry, wrong_max_seq, context.token_hash, context.prefix_len)
    end
  end

  it "binds the cached outcome to prompt tokens before runtime restore" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 123_i64)

    ML::GGUF::Qwen35QBitRuntimeCache.validate_cached_outcome!(
      entry,
      [11_i32, 22_i32, 33_i32],
      128,
    )
    entry.next_token_id = 45
    expect_raises(ArgumentError, /validation hash/) do
      ML::GGUF::Qwen35QBitRuntimeCache.validate_cached_outcome!(
        entry,
        [11_i32, 22_i32, 33_i32],
        128,
      )
    end
  end

  it "plans exact-hit reuse or bounded suffix replay from an admitted prefix" do
    context = qbit_envelope_context
    native, kv = qbit_envelope_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 123_i64)

    exact = ML::GGUF::Qwen35QBitRuntimeCache.replay_plan(
      entry,
      [11_i32, 22_i32, 33_i32],
      128,
    )
    exact.prefix_len.should eq(3)
    exact.replayed_tokens.should eq(0)
    exact.cached_next_token?.should be_true

    prefix = ML::GGUF::Qwen35QBitRuntimeCache.replay_plan(
      entry,
      [11_i32, 22_i32, 33_i32, 55_i32],
      128,
    )
    prefix.prefix_len.should eq(3)
    prefix.replayed_tokens.should eq(1)
    prefix.cached_next_token?.should be_false

    expect_raises(ArgumentError, /token hash/) do
      ML::GGUF::Qwen35QBitRuntimeCache.replay_plan(
        entry,
        [11_i32, 22_i32, 99_i32, 55_i32],
        128,
      )
    end
  end

  it "rejects unsafe runtime cache limits before model allocation" do
    runtime_cache = ML::GGUF::Qwen35QBitRuntimeCache
    runtime_cache.validate_options!(24.hours, 0_i64)
    runtime_cache.validate_options!(24.hours, ML::GGUF::Qwen35QBitRuntimeCache::MAX_WRITE_BACK_SOURCE_BYTES)
    runtime_cache.validate_options!(24.hours, 1_i64, true)

    expect_raises(ArgumentError, /TTL/) do
      runtime_cache.validate_options!(0.seconds, 0_i64)
    end
    expect_raises(ArgumentError, /256MiB/) do
      runtime_cache.validate_options!(
        24.hours,
        ML::GGUF::Qwen35QBitRuntimeCache::MAX_WRITE_BACK_SOURCE_BYTES + 1_i64,
      )
    end
    expect_raises(ArgumentError, /require write-back/) do
      runtime_cache.validate_options!(24.hours, 0_i64, true)
    end
  end

  it "canonicalizes exact KV positions to the cached prompt boundary" do
    source = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      16,
      2,
      [0_i32, 0_i32],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(
          0,
          ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState,
          qbit_envelope_bytes([1.0_f32]),
          ML::StorageMode::Shared,
        ),
        ML::GGUF::Qwen35StateSnapshot::Record.new(
          1,
          ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
          qbit_envelope_bytes([2.0_f32]),
          ML::StorageMode::Shared,
        ),
        ML::GGUF::Qwen35StateSnapshot::Record.new(
          1,
          ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache,
          qbit_envelope_bytes([3.0_f32]),
          ML::StorageMode::Shared,
        ),
      ],
    )

    exact = ML::GGUF::Qwen35QBitRuntimeCache.exact_kv_snapshot(source, 3)
    exact.positions.should eq([3_i32, 3_i32])
    exact.records.map(&.kind).should eq([
      ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache,
    ])

    expect_raises(ArgumentError, /outside snapshot capacity/) do
      ML::GGUF::Qwen35QBitRuntimeCache.exact_kv_snapshot(source, 0)
    end
    expect_raises(ArgumentError, /outside snapshot capacity/) do
      ML::GGUF::Qwen35QBitRuntimeCache.exact_kv_snapshot(source, 17)
    end
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
