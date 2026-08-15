require "./qwen_qbit_clickhouse_cache"
require "./qwen_qbit_state_snapshot"

module ML::GGUF
  # Prefix-aware adapter between Qwen35NativeRuntime and the bounded ClickHouse
  # QBit store. The store restores only an admitted state boundary; the runtime
  # deterministically replays any request suffix with the live model.
  class Qwen35QBitRuntimeCache
    BLOCK_SIZE      = QwenQBitClickHouseCache::QBIT_BLOCK_SIZE
    PRECISION       = QwenQBitCacheEnvelope::REQUIRED_PRECISION
    MAX_TTL_SECONDS = 365_i64 * 24 * 60 * 60
    # Runtime write-back is deliberately stricter than the storage envelope.
    # Qwen3.8 27B at the currently exercised max_seq fits below this bound,
    # while an accidentally much larger capture fails before allocating a
    # host snapshot or compressed stream.
    MAX_WRITE_BACK_SOURCE_BYTES = 256_i64 * 1024 * 1024

    record ReplayPlan,
      prefix_len : Int32,
      replayed_tokens : Int32,
      cached_next_token : Bool do
      def cached_next_token? : Bool
        cached_next_token
      end
    end

    getter write_back_max_source_bytes : Int64

    def initialize(@store : QwenQBitClickHouseCache::Store,
                   @model_id : String,
                   @tokenizer_id : String,
                   @template_id : String,
                   @ttl : Time::Span = 24.hours,
                   @write_back_max_source_bytes : Int64 = 0_i64)
      raise ArgumentError.new("QBit runtime model identity must not be empty") if @model_id.empty?
      raise ArgumentError.new("QBit runtime tokenizer identity must not be empty") if @tokenizer_id.empty?
      unless @template_id.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("QBit runtime template identity is invalid")
      end
      self.class.validate_options!(@ttl, @write_back_max_source_bytes)
    end

    def write_back? : Bool
      @write_back_max_source_bytes > 0
    end

    def self.validate_options!(ttl : Time::Span, write_back_max_source_bytes : Int64) : Nil
      ttl_seconds = ttl.total_seconds.to_i64
      unless ttl_seconds > 0 && ttl_seconds <= MAX_TTL_SECONDS
        raise ArgumentError.new("QBit runtime TTL must be within 1 second..365 days")
      end
      unless write_back_max_source_bytes >= 0 && write_back_max_source_bytes <= MAX_WRITE_BACK_SOURCE_BYTES
        raise ArgumentError.new("QBit runtime write-back source limit is outside 0..256MiB")
      end
    end

    def lookup(prompt_text : String,
               prompt_ids : Array(Int32),
               max_seq : Int32,
               state_abi : QwenQBitCacheEnvelope::StateABI,
               vocab_size : Int32) : QwenQBitCacheEnvelope::Admission?
      lookup = lookup_context(prompt_text, prompt_ids, max_seq, state_abi)
      admission = @store.lookup(lookup)
      if admitted = admission
        self.class.validate_cached_outcome!(admitted.entry, prompt_ids, vocab_size)
      end
      admission
    end

    def lookup_longest_prefix(prompt_ids : Array(Int32),
                              max_seq : Int32,
                              state_abi : QwenQBitCacheEnvelope::StateABI,
                              vocab_size : Int32) : QwenQBitCacheEnvelope::Admission?
      admission = @store.lookup_longest_prefix(prefix_context(max_seq, state_abi), prompt_ids)
      if admitted = admission
        self.class.replay_plan(admitted.entry, prompt_ids, vocab_size)
      end
      admission
    end

    def restore(admission : QwenQBitCacheEnvelope::Admission,
                hp : Qwen35Hparams,
                state : Qwen35CPU::State) : Nil
      QwenQBitStateSnapshot.restore_admitted_native_stream_into(
        admission.native_stream,
        admission.exact_artifact,
        admission.entry.cache_id,
        hp,
        state,
      )
    end

    def save(prompt_text : String,
             prompt_ids : Array(Int32),
             next_token_id : Int32,
             hp : Qwen35Hparams,
             state : Qwen35CPU::State) : QwenQBitClickHouseCache::Saved
      raise ArgumentError.new("QBit runtime write-back is disabled") unless write_back?
      state_abi = QwenQBitCacheEnvelope.state_abi(hp, state.max_seq)
      source_bytes = source_state_byte_size(state_abi)
      if source_bytes > @write_back_max_source_bytes
        raise ArgumentError.new(
          "QBit runtime source state exceeds write-back limit: #{source_bytes} > #{@write_back_max_source_bytes}"
        )
      end

      lookup = lookup_context(prompt_text, prompt_ids, state.max_seq, state_abi)
      validation_hash = self.class.validation_hash(prompt_ids, next_token_id)
      context = QwenQBitCacheEnvelope::Context.new(
        model_id: lookup.model_id,
        tokenizer_id: lookup.tokenizer_id,
        template_id: lookup.template_id,
        prompt_hash: lookup.prompt_hash,
        token_hash: lookup.token_hash,
        prefix_len: lookup.prefix_len,
        max_seq: lookup.max_seq,
        layer_count: lookup.layer_count,
        qbit_block_size: lookup.qbit_block_size,
        qbit_precision: lookup.qbit_precision,
        validation_kind: Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        validation_steps: 1,
        validation_hash: validation_hash,
        next_token_id: next_token_id,
        state_abi: lookup.state_abi,
        state_runtime_id: lookup.state_runtime_id,
      )

      snapshot = Qwen35StateSnapshot.capture(state)
      encoded = QwenQBitStateSnapshot.encode(snapshot, block_size: BLOCK_SIZE, precision: PRECISION)
      recurrent_native = QwenQBitStateSnapshot.encode_native_recurrent(
        encoded,
        QwenQBitCacheEnvelope.cache_id(context),
      )
      kv_snapshot = self.class.exact_kv_snapshot(snapshot, prompt_ids.size.to_i32)
      kv_artifact = Qwen35StateSnapshot.encode_artifact_bytes(kv_snapshot)
      @store.save(context, recurrent_native, kv_artifact, ttl: @ttl)
    end

    # Qwen35 State#position is currently advisory and is not bumped by every
    # fused prefill route. The cache boundary is request-known, so persist the
    # canonical prompt length for every layer instead of serializing stale
    # per-layer zeros into an otherwise valid exact-KV artifact.
    def self.exact_kv_snapshot(snapshot : Qwen35StateSnapshot::Snapshot,
                               prefix_len : Int32) : Qwen35StateSnapshot::Snapshot
      unless prefix_len > 0 && prefix_len <= snapshot.max_seq
        raise ArgumentError.new("QBit exact KV prefix length is outside snapshot capacity")
      end
      Qwen35StateSnapshot::Snapshot.new(
        snapshot.max_seq,
        snapshot.layer_count,
        Array(Int32).new(snapshot.layer_count, prefix_len),
        snapshot.records.select { |record| record.kind.k_cache? || record.kind.v_cache? },
      )
    end

    def self.validation_hash(prompt_ids : Array(Int32), next_token_id : Int32) : String
      Qwen35PromptCache.token_hash_concat(prompt_ids, [next_token_id])
    end

    def self.validate_cached_outcome!(entry : QwenQBitCacheEnvelope::Entry,
                                      prompt_ids : Array(Int32),
                                      vocab_size : Int32) : Nil
      raise ArgumentError.new("QBit cached vocabulary size must be positive") unless vocab_size > 0
      unless entry.validation_kind == Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND && entry.validation_steps == 1
        raise ArgumentError.new("QBit cached outcome is not a one-step exact validation")
      end
      unless entry.next_token_id >= 0 && entry.next_token_id < vocab_size
        raise ArgumentError.new("QBit cached next token is outside the model vocabulary")
      end
      unless entry.validation_hash == validation_hash(prompt_ids, entry.next_token_id)
        raise ArgumentError.new("QBit cached outcome validation hash mismatch")
      end
    end

    def self.replay_plan(entry : QwenQBitCacheEnvelope::Entry,
                         prompt_ids : Array(Int32),
                         vocab_size : Int32) : ReplayPlan
      prefix_len = entry.prefix_len
      unless prefix_len > 0 && prefix_len <= prompt_ids.size
        raise ArgumentError.new("QBit cached prefix length is outside the request")
      end
      prefix_ids = prompt_ids[0, prefix_len]
      unless entry.token_hash == Qwen35PromptCache.token_hash(prefix_ids)
        raise ArgumentError.new("QBit cached token hash mismatch")
      end
      validate_cached_outcome!(entry, prefix_ids, vocab_size)
      replayed_tokens = prompt_ids.size.to_i32 - prefix_len
      ReplayPlan.new(prefix_len, replayed_tokens, replayed_tokens == 0)
    end

    private def lookup_context(prompt_text : String,
                               prompt_ids : Array(Int32),
                               max_seq : Int32,
                               state_abi : QwenQBitCacheEnvelope::StateABI) : QwenQBitCacheEnvelope::LookupContext
      raise ArgumentError.new("QBit runtime prompt is empty") if prompt_ids.empty?
      QwenQBitCacheEnvelope::LookupContext.new(
        model_id: @model_id,
        tokenizer_id: @tokenizer_id,
        template_id: @template_id,
        prompt_hash: Qwen35PromptCache.prompt_hash(prompt_ids, prompt_text),
        token_hash: Qwen35PromptCache.token_hash(prompt_ids),
        prefix_len: prompt_ids.size.to_i32,
        max_seq: max_seq,
        layer_count: state_abi.layer_count,
        qbit_block_size: BLOCK_SIZE,
        qbit_precision: PRECISION,
        state_abi: state_abi,
      )
    end

    private def prefix_context(max_seq : Int32,
                               state_abi : QwenQBitCacheEnvelope::StateABI) : QwenQBitCacheEnvelope::PrefixContext
      QwenQBitCacheEnvelope::PrefixContext.new(
        model_id: @model_id,
        tokenizer_id: @tokenizer_id,
        template_id: @template_id,
        max_seq: max_seq,
        layer_count: state_abi.layer_count,
        qbit_block_size: BLOCK_SIZE,
        qbit_precision: PRECISION,
        state_abi: state_abi,
      )
    end

    private def source_state_byte_size(state_abi : QwenQBitCacheEnvelope::StateABI) : Int64
      full_layers = (0...state_abi.layer_count).count { |layer| state_abi.full_attention?(layer) }.to_i64
      recurrent_layers = state_abi.layer_count.to_i64 - full_layers
      full_layers * 2_i64 * state_abi.kv_record_byte_size +
        recurrent_layers * (state_abi.conv_record_byte_size + state_abi.ssm_record_byte_size)
    end
  end
end
