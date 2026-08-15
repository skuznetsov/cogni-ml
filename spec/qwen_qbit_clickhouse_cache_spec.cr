require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_clickhouse_cache"

private def qbit_ch_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  bytes.copy_from(Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size))
  bytes
end

private def qbit_ch_context(template : String = "{{ messages }}") : ML::GGUF::QwenQBitCacheEnvelope::Context
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
    qbit_block_size: 1024,
    qbit_precision: 7,
    validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
    validation_steps: 1,
    validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(tokens),
    next_token_id: tokens.last,
    state_abi: state_abi,
  )
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

private class QBitCHMemoryTransport < ML::GGUF::QwenQBitClickHouseCache::Transport
  record Request, query : String, body : Bytes, max_response_bytes : Int64

  getter requests = [] of Request
  getter responses = [] of Bytes

  def queue(response : Bytes) : Nil
    @responses << response.dup
  end

  def post(query : String, body : Bytes, max_response_bytes : Int64) : Bytes
    @requests << Request.new(query, body.dup, max_response_bytes)
    @responses.shift?.try(&.dup) || Bytes.empty
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

    transport.requests.size.should eq(6)
    transport.requests[0].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_recurrent")
    transport.requests[1].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_kv")
    transport.requests[2].query.should contain("CREATE TABLE IF NOT EXISTS qwen_cache_test_manifest")
    transport.requests[3].query.should contain("INSERT INTO qwen_cache_test_recurrent")
    transport.requests[4].query.should contain("INSERT INTO qwen_cache_test_kv")
    transport.requests[5].query.should contain("INSERT INTO qwen_cache_test_manifest")
    transport.requests[3].query.should contain(envelope.lookup_key(context))
    transport.requests[3].query.should contain(saved.generation_id)
    String.new(transport.requests[5].body).should eq(saved.entry.to_json)
    saved.expires_at_unix.should eq(1_900_i64)
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

  it "bounds streamed HTTP response reads" do
    ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.read_bounded(IO::Memory.new("1234"), 4).should eq("1234".to_slice)
    expect_raises(ArgumentError, /response exceeds/) do
      ML::GGUF::QwenQBitClickHouseCache::HTTPTransport.read_bounded(IO::Memory.new("12345"), 4)
    end
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
end
