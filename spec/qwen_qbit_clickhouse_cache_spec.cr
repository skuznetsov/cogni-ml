require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_clickhouse_cache"

private def qbit_ch_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  bytes.copy_from(Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size))
  bytes
end

private def qbit_ch_context(template : String = "{{ messages }}",
                            kv_record_byte_size : Int64 = 3_i64 * sizeof(Float32),
                            kv_artifact_codec : String = ML::GGUF::QwenQBitCacheEnvelope::EXACT_ARTIFACT_CODEC) : ML::GGUF::QwenQBitCacheEnvelope::Context
  tokens = [11_i32, 22_i32, 33_i32, 44_i32]
  state_abi = ML::GGUF::QwenQBitCacheEnvelope::StateABI.new(
    layer_count: 2,
    full_attention_interval: 2,
    kv_record_byte_size: kv_record_byte_size,
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
    qbit_block_size: 1024,
    qbit_precision: 7,
    validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
    validation_steps: 1,
    validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(tokens),
    next_token_id: tokens.last,
    state_abi: state_abi,
    kv_artifact_codec: kv_artifact_codec,
  )
end

private def qbit_ch_adaptive_artifact(context : ML::GGUF::QwenQBitCacheEnvelope::Context) : Bytes
  row_values = ML::GGUF::QwenQBitAdaptiveKV::ROW_VALUES
  live_values = context.prefix_len * row_values
  encoded = ML::GGUF::QwenQBitAdaptiveKV.encode(
    Array(Float32).new(live_values) { |i| (i - live_values // 2).to_f32 / 97.0_f32 },
    Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(context.prefix_len) do |i|
      i.even? ? ML::GGUF::QwenQBitAdaptiveKV::Tier::P4 : ML::GGUF::QwenQBitAdaptiveKV::Tier::P5
    end,
  )
  records = [
    ML::GGUF::Qwen35StateSnapshot::EncodedRecord.new(
      1,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
      ML::StorageMode::Shared,
      ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
      context.state_abi.kv_record_byte_size.to_i32,
      encoded.payload,
    ),
    ML::GGUF::Qwen35StateSnapshot::EncodedRecord.new(
      1,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache,
      ML::StorageMode::Shared,
      ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
      context.state_abi.kv_record_byte_size.to_i32,
      encoded.payload,
    ),
  ]
  snapshot = ML::GGUF::Qwen35StateSnapshot::EncodedSnapshot.new(
    context.max_seq,
    context.layer_count,
    Array(Int32).new(context.layer_count, context.prefix_len),
    records,
    ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
    0_i32,
    artifact_version: ML::GGUF::Qwen35StateSnapshot::ARTIFACT_VERSION_V3,
  )
  ML::GGUF::Qwen35StateSnapshot.encode_preencoded_artifact_bytes(snapshot)
end

private def qbit_ch_live_kv_artifact(context : ML::GGUF::QwenQBitCacheEnvelope::Context) : Bytes
  records = [
    ML::GGUF::Qwen35StateSnapshot::EncodedRecord.new(
      1,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
      ML::StorageMode::Shared,
      ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
      context.state_abi.kv_record_byte_size.to_i32,
      qbit_ch_bytes([1.0_f32, 2.0_f32, 3.0_f32]),
    ),
    ML::GGUF::Qwen35StateSnapshot::EncodedRecord.new(
      1,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache,
      ML::StorageMode::Shared,
      ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
      context.state_abi.kv_record_byte_size.to_i32,
      qbit_ch_bytes([4.0_f32, 5.0_f32, 6.0_f32]),
    ),
  ]
  exact = ML::GGUF::Qwen35StateSnapshot::EncodedSnapshot.new(
    context.max_seq,
    context.layer_count,
    [context.prefix_len, context.prefix_len],
    records,
    ML::GGUF::Qwen35StateSnapshot::RecordCodec::RawF32,
    0,
    artifact_version: ML::GGUF::Qwen35StateSnapshot::ARTIFACT_VERSION_V3,
  )
  ML::GGUF::Qwen35StateSnapshot.encode_preencoded_artifact_bytes(exact)
end

private def qbit_ch_artifacts(context : ML::GGUF::QwenQBitCacheEnvelope::Context) : {Bytes, Bytes}
  codec = ML::GGUF::QwenQBitGaussianCodec
  cache_id = ML::GGUF::QwenQBitCacheEnvelope.cache_id(context)
  native = ML::GGUF::QwenQBitNativeWriter.encode([
    ML::GGUF::QwenQBitNativeWriter::Record.new(
      cache_id,
      0_i32,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState.value,
      codec.encode(Array(Float32).new(13) { |i| (i - 6).to_f32 / 3.0_f32 }, 1024, 7),
    ),
    ML::GGUF::QwenQBitNativeWriter::Record.new(
      cache_id,
      0_i32,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState.value,
      codec.encode([1.0_f32, -1.0_f32], 1024, 7),
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
        qbit_ch_bytes([1.0_f32, 2.0_f32, 3.0_f32]),
        ML::StorageMode::Shared,
      ),
      ML::GGUF::Qwen35StateSnapshot::Record.new(
        1,
        ML::GGUF::Qwen35StateSnapshot::RecordKind::VCache,
        qbit_ch_bytes([4.0_f32, 5.0_f32, 6.0_f32]),
        ML::StorageMode::Shared,
      ),
    ],
  )
  {native, ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(exact)}
end

private def qbit_ch_stream_body(context : ML::GGUF::QwenQBitCacheEnvelope::Context) : ML::GGUF::QwenQBitStateSnapshot::NativeRecurrentBody
  records = [
    ML::GGUF::Qwen35StateSnapshot::Record.new(
      0,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::ConvState,
      qbit_ch_bytes(Array(Float32).new(13) { |i| (i - 6).to_f32 / 3.0_f32 }),
      ML::StorageMode::Shared,
    ),
    ML::GGUF::Qwen35StateSnapshot::Record.new(
      0,
      ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState,
      qbit_ch_bytes([1.0_f32, -1.0_f32]),
      ML::StorageMode::Shared,
    ),
  ]
  index = 0
  source = -> : ML::GGUF::Qwen35StateSnapshot::Record? do
    record = records[index]?
    index += 1 if record
    record
  end
  ML::GGUF::QwenQBitStateSnapshot::NativeRecurrentBody.new(
    source,
    cache_id: ML::GGUF::QwenQBitCacheEnvelope.cache_id(context),
    block_size: context.qbit_block_size,
    precision: context.qbit_precision,
  )
end

private def qbit_ch_kv_stream_body(kv : Bytes) : ML::GGUF::Qwen35StateSnapshot::PreencodedArtifactBody
  exact = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(kv)
  records = exact.records.dup
  source = -> : ML::GGUF::Qwen35StateSnapshot::EncodedRecord? { records.shift? }
  ML::GGUF::Qwen35StateSnapshot::PreencodedArtifactBody.new(
    exact.max_seq,
    exact.layer_count,
    exact.positions,
    exact.records.size.to_i32,
    source,
  )
end

private class QBitCHMemoryTransport < ML::GGUF::QwenQBitClickHouseCache::Transport
  record Request, query : String, body : Bytes, max_response_bytes : Int64

  getter requests = [] of Request
  getter responses = [] of Bytes
  getter stream_read_sizes = [] of Int32

  def queue(response : Bytes) : Nil
    @responses << response.dup
  end

  def post(query : String, body : Bytes, max_response_bytes : Int64) : Bytes
    @requests << Request.new(query, body.dup, max_response_bytes)
    @responses.shift?.try(&.dup) || Bytes.empty
  end

  def post_stream(query : String,
                  body : IO,
                  max_body_bytes : Int64,
                  max_response_bytes : Int64) : Bytes
    output = IO::Memory.new
    bounded = ML::GGUF::QwenQBitClickHouseCache::BoundedRequestBody.new(body, max_body_bytes)
    chunk = Bytes.new(3)
    while (read = bounded.read(chunk)) > 0
      @stream_read_sizes << read
      output.write(chunk[0, read])
    end
    @requests << Request.new(query, output.to_slice.dup, max_response_bytes)
    @responses.shift?.try(&.dup) || Bytes.empty
  end
end

private class QBitCHStreamingReadTransport < QBitCHMemoryTransport
  getter response_stream_calls = 0
  getter named_spool_seen = false

  def post_into(query : String,
                body : Bytes,
                max_response_bytes : Int64,
                output : IO) : Int64
    @response_stream_calls += 1
    @requests << Request.new(query, body.dup, max_response_bytes)
    if file = output.as?(File)
      @named_spool_seen ||= File.exists?(file.path)
    end
    response = @responses.shift?.try(&.dup) || Bytes.empty
    written = 0_i64
    offset = 0
    while offset < response.size
      chunk = response[offset, Math.min(3, response.size - offset)]
      output.write(chunk)
      written += chunk.size
      offset += chunk.size
    end
    written
  end
end

private class QBitCHFailingTransport < ML::GGUF::QwenQBitClickHouseCache::Transport
  record Request, query : String, body : Bytes, max_response_bytes : Int64

  getter requests = [] of Request

  def initialize(@fail_on_request : Int32)
  end

  def post(query : String, body : Bytes, max_response_bytes : Int64) : Bytes
    @requests << Request.new(query, body.dup, max_response_bytes)
    if @requests.size == @fail_on_request
      raise IO::Error.new("injected ClickHouse insert failure")
    end
    Bytes.empty
  end

  def post_stream(query : String,
                  body : IO,
                  max_body_bytes : Int64,
                  max_response_bytes : Int64) : Bytes
    @requests << Request.new(query, Bytes.empty, max_response_bytes)
    if @requests.size == @fail_on_request
      raise IO::Error.new("injected ClickHouse insert failure")
    end
    bounded = ML::GGUF::QwenQBitClickHouseCache::BoundedRequestBody.new(body, max_body_bytes)
    IO.copy(bounded, IO::Memory.new)
    Bytes.empty
  end
end

describe ML::GGUF::QwenQBitClickHouseCache do
  cache = ML::GGUF::QwenQBitClickHouseCache
  envelope = ML::GGUF::QwenQBitCacheEnvelope

  it "defines generation-scoped tables with manifest-last visibility" do
    transport = QBitCHMemoryTransport.new
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test")
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, transport, -> { "a" * 64 })
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)

    store.create_schema
    saved = store.save(context, native, kv, ttl: 30.minutes, created_at_unix: 100_i64)

    transport.requests.size.should eq(9)
    transport.requests[0].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_recurrent")
    transport.requests[1].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_kv")
    transport.requests[2].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_manifest")
    transport.requests[3].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_prefix_index")
    transport.requests[4].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_checkpoints")
    transport.requests[5].query.should contain("INSERT INTO qwen_cache_test_recurrent")
    transport.requests[6].query.should contain("INSERT INTO qwen_cache_test_kv")
    transport.requests[7].query.should contain("INSERT INTO qwen_cache_test_manifest")
    transport.requests[8].query.should contain("INSERT INTO qwen_cache_test_prefix_index")
    transport.requests[5].query.should contain(envelope.lookup_key(context))
    transport.requests[5].query.should contain(saved.generation_id)
    String.new(transport.requests[7].body).should eq(saved.entry.to_json)
    String.new(transport.requests[8].body).should eq(saved.entry.to_json)
    saved.expires_at_unix.should eq(1_900_i64)
  end

  it "streams recurrent Native blocks before manifest publication" do
    transport = QBitCHMemoryTransport.new
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test")
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, transport, -> { "a" * 64 })
    context = qbit_ch_context
    _native, kv = qbit_ch_artifacts(context)
    body = qbit_ch_stream_body(context)

    saved = store.save_streaming(context, body, kv, ttl: 30.minutes, created_at_unix: 100_i64)

    transport.requests.size.should eq(4)
    transport.requests[0].query.should contain("_recurrent")
    transport.requests[1].query.should contain("_kv")
    transport.requests[2].query.should contain("_manifest")
    transport.requests[3].query.should contain("_prefix_index")
    transport.stream_read_sizes.should_not be_empty
    transport.stream_read_sizes.max.should be <= 3
    body.summary.logical_sha256.should eq(saved.entry.recurrent_logical_sha256)
    String.new(transport.requests[2].body).should eq(saved.entry.to_json)
    legacy = envelope.build(context, transport.requests[0].body, kv, created_at_unix: 100_i64)
    saved.entry.to_json.should eq(legacy.to_json)
  end

  it "streams compact KV before recurrent state and preserves the legacy envelope" do
    transport = QBitCHMemoryTransport.new
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test")
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, transport, -> { "a" * 64 })
    context = qbit_ch_context(kv_record_byte_size: 16_i64 * sizeof(Float32))
    kv = qbit_ch_live_kv_artifact(context)
    recurrent_body = qbit_ch_stream_body(context)
    kv_body = qbit_ch_kv_stream_body(kv)

    saved = store.save_streaming(context, recurrent_body, kv_body, ttl: 30.minutes, created_at_unix: 100_i64)

    transport.requests.size.should eq(4)
    transport.requests[0].query.should contain("_kv")
    transport.requests[1].query.should contain("_recurrent")
    transport.requests[2].query.should contain("_manifest")
    transport.requests[3].query.should contain("_prefix_index")
    transport.requests[0].body.should eq(kv)
    kv_body.summary.sha256.should eq(saved.entry.kv_artifact_sha256)
    recurrent_body.summary.logical_sha256.should eq(saved.entry.recurrent_logical_sha256)
    legacy = envelope.build(context, transport.requests[1].body, kv, created_at_unix: 100_i64)
    saved.entry.to_json.should eq(legacy.to_json)
  end

  it "does not consume recurrent state or publish a manifest when streamed KV is incomplete" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "a" * 64 },
    )
    context = qbit_ch_context
    recurrent_body = qbit_ch_stream_body(context)
    kv_body = ML::GGUF::Qwen35StateSnapshot::PreencodedArtifactBody.new(
      context.max_seq,
      context.layer_count,
      [context.prefix_len, context.prefix_len],
      1,
      -> : ML::GGUF::Qwen35StateSnapshot::EncodedRecord? { nil },
    )

    expect_raises(ArgumentError, /record count mismatch/) do
      store.save_streaming(context, recurrent_body, kv_body, ttl: 30.minutes, created_at_unix: 100_i64)
    end

    recurrent_body.record_count.should eq(0)
    transport.requests.none? { |request| request.query.includes?("_manifest") }.should be_true
  end

  it "does not publish a manifest when recurrent upload fails after streamed KV" do
    transport = QBitCHFailingTransport.new(fail_on_request: 2)
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "a" * 64 },
    )
    context = qbit_ch_context(kv_record_byte_size: 16_i64 * sizeof(Float32))
    kv = qbit_ch_live_kv_artifact(context)

    expect_raises(IO::Error, /injected ClickHouse/) do
      store.save_streaming(
        context,
        qbit_ch_stream_body(context),
        qbit_ch_kv_stream_body(kv),
        ttl: 30.minutes,
        created_at_unix: 100_i64,
      )
    end

    transport.requests.size.should eq(2)
    transport.requests[0].query.should contain("_kv")
    transport.requests[1].query.should contain("_recurrent")
    transport.requests.none? { |request| request.query.includes?("_manifest") }.should be_true
  end

  it "rejects an invalid exact KV artifact before consuming or uploading recurrent state" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "a" * 64 },
    )
    context = qbit_ch_context
    _native, valid_kv = qbit_ch_artifacts(context)
    invalid_kv = valid_kv.dup
    invalid_kv[8] = (context.max_seq + 1).to_u8
    body = qbit_ch_stream_body(context)

    expect_raises(ArgumentError, /max_seq mismatch/) do
      store.save_streaming(context, body, invalid_kv, ttl: 30.minutes, created_at_unix: 100_i64)
    end

    transport.requests.should be_empty
    body.record_count.should eq(0)
    body.byte_size.should eq(0)
  end

  it "rejects an incomplete exact KV record set before recurrent upload" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "a" * 64 },
    )
    context = qbit_ch_context
    incomplete = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      context.max_seq,
      context.layer_count,
      [context.prefix_len, context.prefix_len],
      [
        ML::GGUF::Qwen35StateSnapshot::Record.new(
          1,
          ML::GGUF::Qwen35StateSnapshot::RecordKind::KCache,
          qbit_ch_bytes([1.0_f32, 2.0_f32, 3.0_f32]),
          ML::StorageMode::Shared,
        ),
      ],
    )
    kv = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(incomplete)
    body = qbit_ch_stream_body(context)

    expect_raises(ArgumentError, /record set mismatch/) do
      store.save_streaming(context, body, kv, ttl: 30.minutes, created_at_unix: 100_i64)
    end

    transport.requests.should be_empty
    body.record_count.should eq(0)
  end

  it "does not publish a streamed manifest when recurrent insertion fails" do
    transport = QBitCHFailingTransport.new(fail_on_request: 1)
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "a" * 64 },
    )
    context = qbit_ch_context
    _native, kv = qbit_ch_artifacts(context)

    expect_raises(IO::Error, /injected ClickHouse/) do
      store.save_streaming(
        context,
        qbit_ch_stream_body(context),
        kv,
        ttl: 30.minutes,
        created_at_unix: 100_i64,
      )
    end

    transport.requests.size.should eq(1)
    transport.requests.first.query.should contain("_recurrent")
    transport.requests.none? { |request| request.query.includes?("_manifest") }.should be_true
  end

  it "returns a strict admission for one committed generation" do
    transport = QBitCHMemoryTransport.new
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test")
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, transport)
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "b" * 64
    transport.queue((generation + entry.to_json).to_slice)
    transport.queue(native)
    transport.queue(kv)

    lookup = envelope.lookup_context(context)
    admitted = store.lookup(lookup).not_nil!

    admitted.entry.certificate_id.should eq(entry.certificate_id)
    transport.requests.size.should eq(3)
    transport.requests[0].query.should contain(envelope.lookup_key(lookup))
    transport.requests[1].query.should contain(generation)
    transport.requests[2].query.should contain(generation)
  end

  it "finds the longest indexed token prefix and revalidates its full identity" do
    transport = QBitCHMemoryTransport.new
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test")
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, transport)
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "f" * 64
    lookup_key = envelope.lookup_key(context)
    transport.queue((generation + lookup_key + entry.to_json).to_slice)
    transport.queue(native)
    transport.queue(kv)

    tokens = [11_i32, 22_i32, 33_i32, 55_i32]
    admitted = store.lookup_longest_prefix(envelope.prefix_context(context), tokens).not_nil!

    admitted.entry.prefix_len.should eq(3)
    transport.requests.size.should eq(3)
    transport.requests[0].query.should contain(envelope.prefix_scope_key(envelope.prefix_context(context)))
    transport.requests[0].query.should contain(ML::GGUF::Qwen35PromptCache.token_hash(tokens, 3))
    transport.requests[0].query.should contain("ORDER BY prefix_len DESC")
    transport.requests[1].query.should contain(generation)
    transport.requests[2].query.should contain(generation)
  end

  it "rejects a prefix-index row whose claimed token prefix does not match the request" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "9" * 64
    lookup_key = envelope.lookup_key(context)
    transport.queue((generation + lookup_key + entry.to_json).to_slice)

    expect_raises(ArgumentError, /token hash/) do
      store.lookup_longest_prefix(envelope.prefix_context(context), [11_i32, 22_i32, 99_i32, 55_i32])
    end
    transport.requests.size.should eq(1)
  end

  it "does not publish a manifest when an artifact insert fails" do
    transport = QBitCHFailingTransport.new(fail_on_request: 2)
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "d" * 64 },
    )
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)

    expect_raises(IO::Error, /injected ClickHouse/) do
      store.save(context, native, kv, ttl: 30.minutes, created_at_unix: 100_i64)
    end

    transport.requests.size.should eq(2)
    transport.requests[0].query.should contain("_recurrent")
    transport.requests[1].query.should contain("_kv")
    transport.requests.none? { |request| request.query.includes?("_manifest") }.should be_true
  end

  it "treats an absent manifest as a cache miss without reading artifacts" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )
    transport.queue(Bytes.empty)

    store.lookup(qbit_ch_context).should be_nil
    transport.requests.size.should eq(1)
  end

  it "rejects oversized and malformed manifest responses before artifact reads" do
    context = qbit_ch_context

    oversized_transport = QBitCHMemoryTransport.new
    oversized_config = ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test", max_envelope_bytes: 1024)
    oversized_store = ML::GGUF::QwenQBitClickHouseCache::Store.new(oversized_config, oversized_transport)
    oversized_transport.queue(Bytes.new(1024 + 65, 0_u8))
    expect_raises(ArgumentError, /response exceeds/) { oversized_store.lookup(context) }
    oversized_transport.requests.size.should eq(1)

    malformed_transport = QBitCHMemoryTransport.new
    malformed_store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      malformed_transport,
    )
    malformed_transport.queue("not-a-generation".to_slice)
    expect_raises(ArgumentError, /manifest response/) { malformed_store.lookup(context) }
    malformed_transport.requests.size.should eq(1)
  end

  it "enforces the combined artifact budget before a KV response is read" do
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "e" * 64
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(
      table_prefix: "qwen_cache_test",
      max_total_artifact_bytes: native.size.to_i64 + kv.size - 1,
    )

    save_transport = QBitCHMemoryTransport.new
    save_store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, save_transport)
    expect_raises(ArgumentError, /combined artifact/) do
      save_store.save(context, native, kv, ttl: 30.minutes, created_at_unix: 100_i64)
    end
    save_transport.requests.should be_empty

    lookup_transport = QBitCHMemoryTransport.new
    lookup_transport.queue((generation + entry.to_json).to_slice)
    lookup_transport.queue(native)
    lookup_transport.queue(kv)
    lookup_store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, lookup_transport)
    expect_raises(ArgumentError, /combined artifact/) { lookup_store.lookup(context) }
    lookup_transport.requests.size.should eq(2)
  end

  it "reuses a byte-bounded immutable admission after rechecking the manifest" do
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "c" * 64
    manifest = (generation + entry.to_json).to_slice
    transport = QBitCHMemoryTransport.new
    transport.queue(manifest)
    transport.queue(native)
    transport.queue(kv)
    transport.queue(manifest)
    config = ML::GGUF::QwenQBitClickHouseCache::Config.new(
      table_prefix: "qwen_cache_test",
      resident_admission_bytes: native.size.to_i64 + kv.size,
    )
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config, transport)

    first = store.lookup(context).not_nil!
    second = store.lookup(context).not_nil!

    first.same?(second).should be_true
    transport.requests.size.should eq(4)
  end

  it "streams cold artifact responses instead of materializing transport Bytes" do
    context = qbit_ch_context(
      kv_record_byte_size: 16_i64 * ML::GGUF::QwenQBitAdaptiveKV::ROW_VALUES * sizeof(Float32),
      kv_artifact_codec: "qkv-adaptive-qbit-v1|1=0",
    )
    native, _raw_kv = qbit_ch_artifacts(context)
    kv = qbit_ch_adaptive_artifact(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "b" * 64
    transport = QBitCHStreamingReadTransport.new
    transport.queue((generation + entry.to_json).to_slice)
    transport.queue(native)
    transport.queue(kv)
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )

    admission = store.lookup(context).not_nil!
    GC.collect

    admission.entry.certificate_id.should eq(entry.certificate_id)
    admission.exact_artifact.records.first.payload.should_not be_empty
    transport.response_stream_calls.should eq(2)
    transport.named_spool_seen.should be_false
    transport.requests.size.should eq(3)
  end

  it "keeps raw artifact reads on the existing transport path" do
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "c" * 64
    transport = QBitCHStreamingReadTransport.new
    transport.queue((generation + entry.to_json).to_slice)
    transport.queue(native)
    transport.queue(kv)
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )

    store.lookup(context).should_not be_nil

    transport.response_stream_calls.should eq(0)
    transport.requests.size.should eq(3)
  end

  it "rejects a truncated streamed adaptive artifact after both bounded reads" do
    context = qbit_ch_context(
      kv_record_byte_size: 16_i64 * ML::GGUF::QwenQBitAdaptiveKV::ROW_VALUES * sizeof(Float32),
      kv_artifact_codec: "qkv-adaptive-qbit-v1|1=0",
    )
    native, _raw_kv = qbit_ch_artifacts(context)
    kv = qbit_ch_adaptive_artifact(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "d" * 64
    transport = QBitCHStreamingReadTransport.new
    transport.queue((generation + entry.to_json).to_slice)
    transport.queue(native)
    transport.queue(kv[0, kv.size - 1])
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )

    expect_raises(ArgumentError, /byte-size mismatch/) do
      store.lookup(context)
    end

    transport.response_stream_calls.should eq(2)
    transport.requests.size.should eq(3)
  end

  it "rejects an adaptive combined budget before streaming the KV artifact" do
    context = qbit_ch_context(
      kv_record_byte_size: 16_i64 * ML::GGUF::QwenQBitAdaptiveKV::ROW_VALUES * sizeof(Float32),
      kv_artifact_codec: "qkv-adaptive-qbit-v1|1=0",
    )
    native, _raw_kv = qbit_ch_artifacts(context)
    kv = qbit_ch_adaptive_artifact(context)
    entry = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "e" * 64
    transport = QBitCHStreamingReadTransport.new
    transport.queue((generation + entry.to_json).to_slice)
    transport.queue(native)
    transport.queue(kv)
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(
        table_prefix: "qwen_cache_test",
        max_total_artifact_bytes: native.size.to_i64 + kv.size - 1,
      ),
      transport,
    )

    expect_raises(ArgumentError, /combined artifact/) do
      store.lookup(context)
    end

    transport.response_stream_calls.should eq(1)
    transport.requests.size.should eq(2)
  end

  it "bounds streamed HTTP response reads" do
    ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.read_bounded(IO::Memory.new("1234"), 4).should eq("1234".to_slice)
    expect_raises(ArgumentError, /response exceeds/) do
      ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.read_bounded(IO::Memory.new("12345"), 4)
    end

    output = IO::Memory.new
    ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.copy_bounded(
      IO::Memory.new("1234"),
      output,
      4,
    ).should eq(4)
    output.to_slice.should eq("1234".to_slice)
    expect_raises(ArgumentError, /response exceeds/) do
      ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.copy_bounded(
        IO::Memory.new("12345"),
        IO::Memory.new,
        4,
      )
    end

    expect_raises(ArgumentError, /length mismatch/) do
      ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.copy_bounded(
        HTTP::FixedLengthContent.new(IO::Memory.new("123"), 4),
        IO::Memory.new,
        4,
      )
    end
    expect_raises(ArgumentError, /response exceeds/) do
      ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.copy_bounded(
        HTTP::FixedLengthContent.new(IO::Memory.new("12345"), 5),
        IO::Memory.new,
        4,
      )
    end
  end

  it "rejects a request body beyond its byte budget" do
    bounded = ML::GGUF::QwenQBitClickHouseCache::BoundedRequestBody.new(IO::Memory.new("12345"), 4)
    chunk = Bytes.new(4)
    bounded.read(chunk).should eq(4)
    expect_raises(ArgumentError, /request exceeds/) { bounded.read(Bytes.new(1)) }
  end

  it "rejects unsafe SQL identifiers and invalid generations before transport" do
    expect_raises(ArgumentError, /safe identifier/) do
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "cache; DROP TABLE cache")
    end

    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
      -> { "not-hex" },
    )
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    expect_raises(ArgumentError, /generation identity/) do
      store.save(context, native, kv, ttl: 30.minutes, created_at_unix: 100_i64)
    end
    transport.requests.should be_empty
  end

  it "bounds the number of prefix hashes before constructing a ClickHouse query" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )
    context = qbit_ch_context
    prefix = ML::GGUF::QwenQBitCacheEnvelope::PrefixContext.new(
      model_id: context.model_id,
      tokenizer_id: context.tokenizer_id,
      template_id: context.template_id,
      max_seq: ML::GGUF::QwenQBitClickHouseCache::MAX_PREFIX_CANDIDATES + 1,
      layer_count: context.layer_count,
      qbit_block_size: context.qbit_block_size,
      qbit_precision: context.qbit_precision,
      state_abi: context.state_abi,
    )
    tokens = Array(Int32).new(ML::GGUF::QwenQBitClickHouseCache::MAX_PREFIX_CANDIDATES + 1, 1_i32)

    expect_raises(ArgumentError, /candidate limit/) do
      store.lookup_longest_prefix(prefix, tokens)
    end
    transport.requests.should be_empty
  end

  it "stores and retrieves a session checkpoint by id or longest transcript prefix" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    anchor = envelope.build(context, native, kv, created_at_unix: 100_i64)
    tokens = [11_i32, 22_i32, 33_i32]
    checkpoint = ML::GGUF::QwenQBitSessionCheckpoint.build_anchor(
      session_id: "session-a",
      checkpoint_id: "7" * 64,
      parent_checkpoint_id: nil,
      anchor_cache_id: anchor.cache_id,
      anchor_lookup_key: envelope.lookup_key(context),
      anchor_generation_id: "8" * 64,
      anchor_certificate_id: anchor.certificate_id,
      token_ids: tokens,
      boundary_text: "rendered-boundary-a<|im_end|>\n",
      created_at_unix: 100_i64,
      expires_at_unix: 1_000_i64,
    )

    store.save_checkpoint(checkpoint)
    transport.requests.last.query.should contain("INSERT INTO qwen_cache_test_checkpoints")
    String.new(transport.requests.last.body).should eq(checkpoint.to_json)

    transport.queue(checkpoint.to_json.to_slice)
    rendered = "rendered-boundary-a<|im_end|>\ncontinuation"
    continuation_tokens = tokens + [44_i32]
    explicit = store.lookup_checkpoint("session-a", checkpoint.checkpoint_id, rendered, continuation_tokens).not_nil!
    explicit.checkpoint_id.should eq(checkpoint.checkpoint_id)
    transport.requests.last.query.should contain(checkpoint.checkpoint_id)

    transport.queue(checkpoint.to_json.to_slice)
    latest = store.lookup_latest_checkpoint("session-a", rendered, continuation_tokens).not_nil!
    latest.checkpoint_id.should eq(checkpoint.checkpoint_id)
    transport.requests.last.query.should contain(checkpoint.boundary_text_hash)
    transport.requests.last.query.should contain("ORDER BY boundary_text_bytes DESC")

    transport.queue(checkpoint.to_json.to_slice)
    expect_raises(ArgumentError, /child token hash/) do
      store.lookup_checkpoint(
        "session-a",
        checkpoint.checkpoint_id,
        rendered,
        [11_i32, 99_i32, 33_i32, 44_i32],
      )
    end
  end

  it "restores only the immutable anchor generation named by a validated checkpoint" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )
    context = qbit_ch_context
    native, kv = qbit_ch_artifacts(context)
    anchor = envelope.build(context, native, kv, created_at_unix: 100_i64)
    generation = "8" * 64
    tokens = [11_i32, 22_i32, 33_i32]
    checkpoint = ML::GGUF::QwenQBitSessionCheckpoint.build_anchor(
      session_id: "session-a",
      checkpoint_id: "7" * 64,
      parent_checkpoint_id: nil,
      anchor_cache_id: anchor.cache_id,
      anchor_lookup_key: envelope.lookup_key(context),
      anchor_generation_id: generation,
      anchor_certificate_id: anchor.certificate_id,
      token_ids: tokens,
      boundary_text: "rendered-boundary-a<|im_end|>\n",
      created_at_unix: 100_i64,
      expires_at_unix: 1_000_i64,
    )
    transport.queue(anchor.to_json.to_slice)
    transport.queue(native)
    transport.queue(kv)

    admitted = store.lookup_checkpoint_anchor(
      checkpoint,
      envelope.prefix_context(context),
    ).not_nil!

    admitted.entry.certificate_id.should eq(anchor.certificate_id)
    transport.requests.size.should eq(3)
    transport.requests[0].query.should contain(generation)
    transport.requests[1].query.should contain(generation)
    transport.requests[2].query.should contain(generation)
  end

  it "bounds completed-message checkpoint candidates before constructing SQL" do
    transport = QBitCHMemoryTransport.new
    store = ML::GGUF::QwenQBitClickHouseCache::Store.new(
      ML::GGUF::QwenQBitClickHouseCache::Config.new(table_prefix: "qwen_cache_test"),
      transport,
    )
    rendered = "<|im_end|>\n" * (ML::GGUF::QwenQBitSessionCheckpoint::MAX_BOUNDARY_CANDIDATES + 1)

    expect_raises(ArgumentError, /candidate limit/) do
      store.lookup_latest_checkpoint("session-a", rendered, [1_i32])
    end
    transport.requests.should be_empty
  end
end
