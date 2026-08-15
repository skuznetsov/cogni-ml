require "digest/sha256"
require "json"
require "./qwen35_prompt_cache"
require "./qwen_qbit_native_block"

module ML::GGUF
  # Versioned compatibility envelope for split QBit cache artifacts.
  #
  # The manifest is not an authentication token. `Admission` is a process-local
  # validation certificate created only after the expected prompt context, the
  # block-framing-independent recurrent digest, the exact KV artifact, and the
  # combined state layout have all been checked.
  module QwenQBitCacheEnvelope
    extend self

    SCHEMA_ID            = "cogni-ml/qwen-qbit-cache-envelope-v1"
    NATIVE_LAYOUT_ID     = "clickhouse-native-qbit-int8-p7-v0"
    EXACT_ARTIFACT_CODEC = "qkv-raw-v1"
    LOGICAL_HASH_KIND    = "qwen-qbit-native-logical-v1"
    REQUIRED_PRECISION   = 7

    class StateABI
      getter layer_count : Int32
      getter full_attention_interval : Int32
      getter kv_record_byte_size : Int64
      getter conv_record_byte_size : Int64
      getter ssm_record_byte_size : Int64

      def initialize(@layer_count : Int32,
                     @full_attention_interval : Int32,
                     @kv_record_byte_size : Int64,
                     @conv_record_byte_size : Int64,
                     @ssm_record_byte_size : Int64)
      end

      def full_attention?(layer : Int32) : Bool
        (layer + 1) % @full_attention_interval == 0
      end
    end

    # Cache identity available before model inference. Cached validation and
    # next-token fields deliberately live only in `Context`/`Entry`: requiring
    # them here would make a real cold lookup impossible.
    class LookupContext
      getter state_runtime_id : String
      getter model_id : String
      getter tokenizer_id : String
      getter template_id : String
      getter prompt_hash : String
      getter token_hash : String
      getter prefix_len : Int32
      getter max_seq : Int32
      getter layer_count : Int32
      getter qbit_block_size : Int32
      getter qbit_precision : Int32
      getter state_abi : StateABI

      def initialize(@model_id : String,
                     @tokenizer_id : String,
                     @template_id : String,
                     @prompt_hash : String,
                     @token_hash : String,
                     @prefix_len : Int32,
                     @max_seq : Int32,
                     @layer_count : Int32,
                     @qbit_block_size : Int32,
                     @qbit_precision : Int32,
                     @state_abi : StateABI,
                     @state_runtime_id : String = Qwen35PromptCache::RUNTIME_ID)
      end
    end

    # Request-known identity shared by every token prefix of one runtime/model
    # configuration. Prompt text and token hashes are intentionally excluded:
    # the ClickHouse index uses this scope only to find candidates, while
    # `validate_prefix_manifest!` performs the actual admission check against
    # the caller's token prefix.
    class PrefixContext
      getter state_runtime_id : String
      getter model_id : String
      getter tokenizer_id : String
      getter template_id : String
      getter max_seq : Int32
      getter layer_count : Int32
      getter qbit_block_size : Int32
      getter qbit_precision : Int32
      getter state_abi : StateABI

      def initialize(@model_id : String,
                     @tokenizer_id : String,
                     @template_id : String,
                     @max_seq : Int32,
                     @layer_count : Int32,
                     @qbit_block_size : Int32,
                     @qbit_precision : Int32,
                     @state_abi : StateABI,
                     @state_runtime_id : String = Qwen35PromptCache::RUNTIME_ID)
      end
    end

    class Context < LookupContext
      getter validation_kind : String
      getter validation_steps : Int32
      getter validation_hash : String
      getter next_token_id : Int32

      def initialize(model_id : String,
                     tokenizer_id : String,
                     template_id : String,
                     prompt_hash : String,
                     token_hash : String,
                     prefix_len : Int32,
                     max_seq : Int32,
                     layer_count : Int32,
                     qbit_block_size : Int32,
                     qbit_precision : Int32,
                     @validation_kind : String,
                     @validation_steps : Int32,
                     @validation_hash : String,
                     @next_token_id : Int32,
                     state_abi : StateABI,
                     state_runtime_id : String = Qwen35PromptCache::RUNTIME_ID)
        super(
          model_id,
          tokenizer_id,
          template_id,
          prompt_hash,
          token_hash,
          prefix_len,
          max_seq,
          layer_count,
          qbit_block_size,
          qbit_precision,
          state_abi,
          state_runtime_id,
        )
      end
    end

    class Entry
      include JSON::Serializable

      property schema_id : String
      property state_runtime_id : String
      property native_layout_id : String
      property exact_artifact_codec : String
      property logical_hash_kind : String
      property state_abi_id : String
      property model_id : String
      property tokenizer_id : String
      property template_id : String
      property prompt_hash : String
      property token_hash : String
      property prefix_len : Int32
      property max_seq : Int32
      property layer_count : Int32
      property cache_id : UInt64
      property qbit_block_size : Int32
      property qbit_precision : Int32
      property recurrent_record_count : Int32
      property recurrent_tile_count : Int32
      property recurrent_value_count : Int64
      property recurrent_logical_sha256 : String
      property kv_artifact_sha256 : String
      property kv_artifact_byte_size : Int64
      property state_layout_sha256 : String
      property validation_kind : String
      property validation_steps : Int32
      property validation_hash : String
      property next_token_id : Int32
      property certificate_id : String
      property created_at_unix : Int64

      def initialize(@schema_id : String,
                     @state_runtime_id : String,
                     @native_layout_id : String,
                     @exact_artifact_codec : String,
                     @logical_hash_kind : String,
                     @state_abi_id : String,
                     @model_id : String,
                     @tokenizer_id : String,
                     @template_id : String,
                     @prompt_hash : String,
                     @token_hash : String,
                     @prefix_len : Int32,
                     @max_seq : Int32,
                     @layer_count : Int32,
                     @cache_id : UInt64,
                     @qbit_block_size : Int32,
                     @qbit_precision : Int32,
                     @recurrent_record_count : Int32,
                     @recurrent_tile_count : Int32,
                     @recurrent_value_count : Int64,
                     @recurrent_logical_sha256 : String,
                     @kv_artifact_sha256 : String,
                     @kv_artifact_byte_size : Int64,
                     @state_layout_sha256 : String,
                     @validation_kind : String,
                     @validation_steps : Int32,
                     @validation_hash : String,
                     @next_token_id : Int32,
                     @certificate_id : String,
                     @created_at_unix : Int64)
      end
    end

    # Retains the validated zero-copy views. Callers must treat their backing
    # byte slices as immutable for the lifetime of this certificate.
    class Admission
      getter entry : Entry
      getter native_stream : QwenQBitNativeBlock::Stream
      getter exact_artifact : Qwen35StateSnapshot::EncodedSnapshot

      def initialize(@entry : Entry,
                     @native_stream : QwenQBitNativeBlock::Stream,
                     @exact_artifact : Qwen35StateSnapshot::EncodedSnapshot)
      end
    end

    private record LayoutRecord,
      layer : Int32,
      kind : UInt8,
      original_byte_size : Int64,
      storage_mode : UInt8

    def template_id(chat_template : String?) : String
      marker = chat_template.nil? ? "missing" : "present"
      Digest::SHA256.hexdigest("qwen35-chat-template-v1\0#{marker}\0#{chat_template || ""}")
    end

    def lookup_context(context : Context) : LookupContext
      LookupContext.new(
        context.model_id,
        context.tokenizer_id,
        context.template_id,
        context.prompt_hash,
        context.token_hash,
        context.prefix_len,
        context.max_seq,
        context.layer_count,
        context.qbit_block_size,
        context.qbit_precision,
        context.state_abi,
        context.state_runtime_id,
      )
    end

    def prefix_context(context : LookupContext) : PrefixContext
      PrefixContext.new(
        context.model_id,
        context.tokenizer_id,
        context.template_id,
        context.max_seq,
        context.layer_count,
        context.qbit_block_size,
        context.qbit_precision,
        context.state_abi,
        context.state_runtime_id,
      )
    end

    def prefix_scope_key(context : PrefixContext) : String
      validate_prefix_context!(context)
      io = IO::Memory.new
      write_string(io, "qwen-qbit-prefix-scope-v1")
      write_string(io, context.state_runtime_id)
      write_string(io, context.model_id)
      write_string(io, context.tokenizer_id)
      write_string(io, context.template_id)
      write_string(io, state_abi_id(context.state_abi))
      io.write_bytes(context.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.qbit_block_size.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.qbit_precision.to_u32, IO::ByteFormat::LittleEndian)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def lookup_key(context : LookupContext) : String
      validate_lookup_context!(context)
      io = IO::Memory.new
      write_string(io, "qwen-qbit-cache-key-v1")
      write_string(io, context.state_runtime_id)
      write_string(io, context.model_id)
      write_string(io, context.tokenizer_id)
      write_string(io, context.template_id)
      write_string(io, context.prompt_hash)
      write_string(io, context.token_hash)
      write_string(io, state_abi_id(context.state_abi))
      io.write_bytes(context.prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.qbit_block_size.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.qbit_precision.to_u32, IO::ByteFormat::LittleEndian)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    # UInt64 is the narrow ClickHouse row key. The full lookup key and all
    # context fields remain in the envelope and are checked after lookup.
    def cache_id(context : LookupContext) : UInt64
      lookup_key(context)[0, 16].to_u64(16)
    end

    def build(context : Context,
              recurrent_native : Bytes,
              kv_artifact : Bytes,
              created_at_unix : Int64 = Time.utc.to_unix) : Entry
      validate_context!(context)
      stream = QwenQBitNativeBlock.parse_stream(recurrent_native)
      exact = Qwen35StateSnapshot.decode_artifact_encoded_bytes(kv_artifact, copy_payloads: false)
      validate_artifact_shapes!(context, stream, exact)

      entry = Entry.new(
        schema_id: SCHEMA_ID,
        state_runtime_id: context.state_runtime_id,
        native_layout_id: NATIVE_LAYOUT_ID,
        exact_artifact_codec: EXACT_ARTIFACT_CODEC,
        logical_hash_kind: LOGICAL_HASH_KIND,
        state_abi_id: state_abi_id(context.state_abi),
        model_id: context.model_id,
        tokenizer_id: context.tokenizer_id,
        template_id: context.template_id,
        prompt_hash: context.prompt_hash,
        token_hash: context.token_hash,
        prefix_len: context.prefix_len,
        max_seq: context.max_seq,
        layer_count: context.layer_count,
        cache_id: cache_id(context),
        qbit_block_size: context.qbit_block_size,
        qbit_precision: context.qbit_precision,
        recurrent_record_count: stream.record_spans.size.to_i32,
        recurrent_tile_count: stream.row_count,
        recurrent_value_count: stream.record_spans.sum(0_i64, &.value_count.to_i64),
        recurrent_logical_sha256: QwenQBitNativeBlock.logical_sha256(stream),
        kv_artifact_sha256: Digest::SHA256.hexdigest(kv_artifact),
        kv_artifact_byte_size: kv_artifact.size.to_i64,
        state_layout_sha256: state_layout_sha256(context, stream, exact),
        validation_kind: context.validation_kind,
        validation_steps: context.validation_steps,
        validation_hash: context.validation_hash,
        next_token_id: context.next_token_id,
        certificate_id: "",
        created_at_unix: created_at_unix,
      )
      entry.certificate_id = certificate_id(entry)
      validate_entry!(entry, context)
      entry
    end

    def admit(entry : Entry,
              context : Context,
              recurrent_native : Bytes,
              kv_artifact : Bytes) : Admission
      validate_entry!(entry, context)
      admit_artifacts(entry, context, recurrent_native, kv_artifact)
    end

    def admit(entry : Entry,
              context : LookupContext,
              recurrent_native : Bytes,
              kv_artifact : Bytes) : Admission
      validate_entry_identity!(entry, context)
      admit_artifacts(entry, context, recurrent_native, kv_artifact)
    end

    private def admit_artifacts(entry : Entry,
                                context : LookupContext,
                                recurrent_native : Bytes,
                                kv_artifact : Bytes) : Admission
      raise ArgumentError.new("QBit exact KV byte-size mismatch") unless kv_artifact.size.to_i64 == entry.kv_artifact_byte_size
      unless Digest::SHA256.hexdigest(kv_artifact) == entry.kv_artifact_sha256
        raise ArgumentError.new("QBit exact KV checksum mismatch")
      end

      stream = QwenQBitNativeBlock.parse_stream(recurrent_native)
      validate_stream_identity!(context, stream)
      unless QwenQBitNativeBlock.logical_sha256(stream) == entry.recurrent_logical_sha256
        raise ArgumentError.new("QBit recurrent logical checksum mismatch")
      end
      exact = Qwen35StateSnapshot.decode_artifact_encoded_bytes(kv_artifact, copy_payloads: false)
      validate_artifact_shapes!(context, stream, exact)
      raise ArgumentError.new("QBit recurrent record count mismatch") unless stream.record_spans.size == entry.recurrent_record_count
      raise ArgumentError.new("QBit recurrent tile count mismatch") unless stream.row_count == entry.recurrent_tile_count
      values = stream.record_spans.sum(0_i64, &.value_count.to_i64)
      raise ArgumentError.new("QBit recurrent value count mismatch") unless values == entry.recurrent_value_count
      unless state_layout_sha256(context, stream, exact) == entry.state_layout_sha256
        raise ArgumentError.new("QBit state layout checksum mismatch")
      end
      Admission.new(entry, stream, exact)
    end

    # Validate the manifest and its full lookup context before artifact reads.
    # Artifact content still requires `admit` unless a previously issued
    # process-local Admission for the same immutable generation is retained.
    def validate_manifest!(entry : Entry, context : Context) : Nil
      validate_entry!(entry, context)
    end

    def validate_manifest!(entry : Entry, context : LookupContext) : Nil
      validate_entry_identity!(entry, context)
    end

    # Reconstruct the entry's full lookup identity only after checking every
    # request-known scope field plus the actual caller-derived token hash and
    # prefix length. The returned context is safe to use for artifact admission.
    def validate_prefix_manifest!(entry : Entry,
                                  context : PrefixContext,
                                  token_hash : String,
                                  prefix_len : Int32) : LookupContext
      validate_prefix_context!(context)
      raise ArgumentError.new("QBit token hash is invalid") unless sha256?(token_hash)
      unless prefix_len > 0 && prefix_len <= context.max_seq
        raise ArgumentError.new("QBit prefix length is invalid")
      end
      lookup = LookupContext.new(
        model_id: context.model_id,
        tokenizer_id: context.tokenizer_id,
        template_id: context.template_id,
        prompt_hash: entry.prompt_hash,
        token_hash: token_hash,
        prefix_len: prefix_len,
        max_seq: context.max_seq,
        layer_count: context.layer_count,
        qbit_block_size: context.qbit_block_size,
        qbit_precision: context.qbit_precision,
        state_abi: context.state_abi,
        state_runtime_id: context.state_runtime_id,
      )
      validate_entry_identity!(entry, lookup)
      lookup
    end

    def certificate_id(entry : Entry) : String
      io = IO::Memory.new
      write_string(io, "qwen-qbit-cache-certificate-v1")
      write_string(io, entry.schema_id)
      write_string(io, entry.state_runtime_id)
      write_string(io, entry.native_layout_id)
      write_string(io, entry.exact_artifact_codec)
      write_string(io, entry.logical_hash_kind)
      write_string(io, entry.state_abi_id)
      write_string(io, entry.model_id)
      write_string(io, entry.tokenizer_id)
      write_string(io, entry.template_id)
      write_string(io, entry.prompt_hash)
      write_string(io, entry.token_hash)
      io.write_bytes(entry.prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.cache_id, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.qbit_block_size.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.qbit_precision.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.recurrent_record_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.recurrent_tile_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.recurrent_value_count.to_u64, IO::ByteFormat::LittleEndian)
      write_string(io, entry.recurrent_logical_sha256)
      write_string(io, entry.kv_artifact_sha256)
      io.write_bytes(entry.kv_artifact_byte_size.to_u64, IO::ByteFormat::LittleEndian)
      write_string(io, entry.state_layout_sha256)
      write_string(io, entry.validation_kind)
      io.write_bytes(entry.validation_steps.to_u32, IO::ByteFormat::LittleEndian)
      write_string(io, entry.validation_hash)
      io.write_bytes(entry.next_token_id.unsafe_as(UInt32), IO::ByteFormat::LittleEndian)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    private def validate_lookup_context!(context : LookupContext) : Nil
      raise ArgumentError.new("QBit state runtime mismatch") unless context.state_runtime_id == Qwen35PromptCache::RUNTIME_ID
      raise ArgumentError.new("QBit model identity is empty") if context.model_id.empty?
      raise ArgumentError.new("QBit tokenizer identity is empty") if context.tokenizer_id.empty?
      raise ArgumentError.new("QBit template identity is invalid") unless sha256?(context.template_id)
      raise ArgumentError.new("QBit prompt hash is invalid") unless sha256?(context.prompt_hash)
      raise ArgumentError.new("QBit token hash is invalid") unless sha256?(context.token_hash)
      raise ArgumentError.new("QBit prefix length is invalid") unless context.prefix_len > 0 && context.prefix_len <= context.max_seq
      raise ArgumentError.new("QBit layer count is invalid") unless context.layer_count > 0
      abi = context.state_abi
      raise ArgumentError.new("QBit state ABI layer count mismatch") unless abi.layer_count == context.layer_count
      raise ArgumentError.new("QBit state ABI attention interval is invalid") unless abi.full_attention_interval > 0
      unless abi.kv_record_byte_size > 0 && abi.conv_record_byte_size > 0 && abi.ssm_record_byte_size > 0
        raise ArgumentError.new("QBit state ABI record size is invalid")
      end
      unless context.qbit_block_size > 0 && context.qbit_block_size <= UInt16::MAX && context.qbit_block_size % 8 == 0
        raise ArgumentError.new("QBit block size is invalid")
      end
      raise ArgumentError.new("QBit precision is unsupported") unless context.qbit_precision == REQUIRED_PRECISION
    end

    private def validate_prefix_context!(context : PrefixContext) : Nil
      raise ArgumentError.new("QBit state runtime mismatch") unless context.state_runtime_id == Qwen35PromptCache::RUNTIME_ID
      raise ArgumentError.new("QBit model identity is empty") if context.model_id.empty?
      raise ArgumentError.new("QBit tokenizer identity is empty") if context.tokenizer_id.empty?
      raise ArgumentError.new("QBit template identity is invalid") unless sha256?(context.template_id)
      raise ArgumentError.new("QBit max_seq is invalid") unless context.max_seq > 0
      raise ArgumentError.new("QBit layer count is invalid") unless context.layer_count > 0
      abi = context.state_abi
      raise ArgumentError.new("QBit state ABI layer count mismatch") unless abi.layer_count == context.layer_count
      raise ArgumentError.new("QBit state ABI attention interval is invalid") unless abi.full_attention_interval > 0
      unless abi.kv_record_byte_size > 0 && abi.conv_record_byte_size > 0 && abi.ssm_record_byte_size > 0
        raise ArgumentError.new("QBit state ABI record size is invalid")
      end
      unless context.qbit_block_size > 0 && context.qbit_block_size <= UInt16::MAX && context.qbit_block_size % 8 == 0
        raise ArgumentError.new("QBit block size is invalid")
      end
      raise ArgumentError.new("QBit precision is unsupported") unless context.qbit_precision == REQUIRED_PRECISION
    end

    private def validate_context!(context : Context) : Nil
      validate_lookup_context!(context)
      unless context.validation_kind == Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND
        raise ArgumentError.new("QBit validation kind is unsupported")
      end
      raise ArgumentError.new("QBit validation span is empty") unless context.validation_steps > 0
      raise ArgumentError.new("QBit validation hash is invalid") unless sha256?(context.validation_hash)
      raise ArgumentError.new("QBit next token is invalid") if context.next_token_id < 0
    end

    private def validate_entry!(entry : Entry, context : Context) : Nil
      validate_entry_identity!(entry, context)
      validate_context!(context)
      raise ArgumentError.new("QBit validation kind mismatch") unless entry.validation_kind == context.validation_kind
      raise ArgumentError.new("QBit validation steps mismatch") unless entry.validation_steps == context.validation_steps
      raise ArgumentError.new("QBit validation hash mismatch") unless entry.validation_hash == context.validation_hash
      raise ArgumentError.new("QBit next token mismatch") unless entry.next_token_id == context.next_token_id
    end

    private def validate_entry_identity!(entry : Entry, context : LookupContext) : Nil
      validate_lookup_context!(context)
      raise ArgumentError.new("QBit envelope schema mismatch") unless entry.schema_id == SCHEMA_ID
      raise ArgumentError.new("QBit state runtime mismatch") unless entry.state_runtime_id == context.state_runtime_id
      raise ArgumentError.new("QBit Native layout mismatch") unless entry.native_layout_id == NATIVE_LAYOUT_ID
      raise ArgumentError.new("QBit exact artifact codec mismatch") unless entry.exact_artifact_codec == EXACT_ARTIFACT_CODEC
      raise ArgumentError.new("QBit logical hash kind mismatch") unless entry.logical_hash_kind == LOGICAL_HASH_KIND
      raise ArgumentError.new("QBit state ABI mismatch") unless entry.state_abi_id == state_abi_id(context.state_abi)
      raise ArgumentError.new("QBit model identity mismatch") unless entry.model_id == context.model_id
      raise ArgumentError.new("QBit tokenizer identity mismatch") unless entry.tokenizer_id == context.tokenizer_id
      raise ArgumentError.new("QBit template identity mismatch") unless entry.template_id == context.template_id
      raise ArgumentError.new("QBit prompt hash mismatch") unless entry.prompt_hash == context.prompt_hash
      raise ArgumentError.new("QBit token hash mismatch") unless entry.token_hash == context.token_hash
      raise ArgumentError.new("QBit prefix length mismatch") unless entry.prefix_len == context.prefix_len
      raise ArgumentError.new("QBit max_seq mismatch") unless entry.max_seq == context.max_seq
      raise ArgumentError.new("QBit layer count mismatch") unless entry.layer_count == context.layer_count
      raise ArgumentError.new("QBit cache identity mismatch") unless entry.cache_id == cache_id(context)
      raise ArgumentError.new("QBit block size mismatch") unless entry.qbit_block_size == context.qbit_block_size
      raise ArgumentError.new("QBit precision mismatch") unless entry.qbit_precision == context.qbit_precision
      raise ArgumentError.new("QBit recurrent record count is invalid") unless entry.recurrent_record_count > 0
      raise ArgumentError.new("QBit recurrent tile count is invalid") unless entry.recurrent_tile_count > 0
      raise ArgumentError.new("QBit recurrent value count is invalid") unless entry.recurrent_value_count > 0
      raise ArgumentError.new("QBit recurrent logical checksum is invalid") unless sha256?(entry.recurrent_logical_sha256)
      raise ArgumentError.new("QBit exact KV checksum is invalid") unless sha256?(entry.kv_artifact_sha256)
      raise ArgumentError.new("QBit exact KV byte size is invalid") unless entry.kv_artifact_byte_size > 0
      raise ArgumentError.new("QBit state layout checksum is invalid") unless sha256?(entry.state_layout_sha256)
      unless entry.validation_kind == Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND
        raise ArgumentError.new("QBit validation kind is unsupported")
      end
      raise ArgumentError.new("QBit validation span is empty") unless entry.validation_steps > 0
      raise ArgumentError.new("QBit validation hash is invalid") unless sha256?(entry.validation_hash)
      raise ArgumentError.new("QBit next token is invalid") if entry.next_token_id < 0
      raise ArgumentError.new("QBit envelope certificate mismatch") unless entry.certificate_id == certificate_id(entry)
    end

    private def validate_artifact_shapes!(context : LookupContext,
                                          stream : QwenQBitNativeBlock::Stream,
                                          exact : Qwen35StateSnapshot::EncodedSnapshot) : Nil
      validate_stream_identity!(context, stream)
      recurrent_keys = Set({Int32, UInt8}).new
      stream.record_spans.each do |span|
        raise ArgumentError.new("QBit recurrent layer is out of range") unless span.layer >= 0 && span.layer < context.layer_count
        kind = Qwen35StateSnapshot::RecordKind.from_value?(span.kind)
        unless kind && (kind.conv_state? || kind.ssm_state?)
          raise ArgumentError.new("QBit Native stream contains a non-recurrent record")
        end
        raise ArgumentError.new("duplicate QBit recurrent record") unless recurrent_keys.add?({span.layer, span.kind})
      end

      raise ArgumentError.new("QBit exact artifact max_seq mismatch") unless exact.max_seq == context.max_seq
      raise ArgumentError.new("QBit exact artifact layer count mismatch") unless exact.layer_count == context.layer_count
      unless exact.positions.size == context.layer_count && exact.positions.all? { |position| position == context.prefix_len }
        raise ArgumentError.new("QBit exact artifact positions mismatch")
      end
      raise ArgumentError.new("QBit exact artifact must use the raw codec") unless exact.codec.raw_f32?
      raise ArgumentError.new("QBit exact artifact must contain KV records") if exact.records.empty?
      exact_keys = Set({Int32, UInt8}).new
      exact.records.each do |record|
        raise ArgumentError.new("QBit exact artifact layer is out of range") unless record.layer >= 0 && record.layer < context.layer_count
        unless record.kind.k_cache? || record.kind.v_cache?
          raise ArgumentError.new("QBit exact artifact must be KV-only")
        end
        raise ArgumentError.new("QBit exact artifact record must stay raw") unless record.codec.raw_f32?
        raise ArgumentError.new("QBit exact artifact record is truncated") unless record.payload.size == record.original_byte_size
        raise ArgumentError.new("duplicate QBit exact artifact record") unless exact_keys.add?({record.layer, record.kind.value})
      end
      validate_complete_record_set!(context, stream, exact)
    end

    private def validate_complete_record_set!(context : LookupContext,
                                              stream : QwenQBitNativeBlock::Stream,
                                              exact : Qwen35StateSnapshot::EncodedSnapshot) : Nil
      actual = {} of {Int32, UInt8} => Int64
      stream.record_spans.each do |span|
        actual[{span.layer, span.kind}] = span.value_count.to_i64 * sizeof(Float32)
      end
      exact.records.each do |record|
        actual[{record.layer, record.kind.value}] = record.original_byte_size.to_i64
      end

      expected = {} of {Int32, UInt8} => Int64
      abi = context.state_abi
      context.layer_count.times do |layer|
        if abi.full_attention?(layer)
          expected[{layer, Qwen35StateSnapshot::RecordKind::KCache.value}] = abi.kv_record_byte_size
          expected[{layer, Qwen35StateSnapshot::RecordKind::VCache.value}] = abi.kv_record_byte_size
        else
          expected[{layer, Qwen35StateSnapshot::RecordKind::ConvState.value}] = abi.conv_record_byte_size
          expected[{layer, Qwen35StateSnapshot::RecordKind::SsmState.value}] = abi.ssm_record_byte_size
        end
      end
      raise ArgumentError.new("QBit state record set mismatch") unless actual == expected
    end

    def state_abi(hp : Qwen35Hparams, max_seq : Int32) : StateABI
      raise ArgumentError.new("QBit state ABI max_seq must be positive") unless max_seq > 0
      kv_values = max_seq.to_i64 * hp.head_dim * hp.n_head_kv
      qkv_dim = 2_i64 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size
      conv_values = (hp.ssm_conv_kernel - 1).to_i64 * qkv_dim
      ssm_values = hp.ssm_time_step_rank.to_i64 * hp.ssm_state_size * hp.ssm_state_size
      StateABI.new(
        hp.n_layer,
        hp.full_attention_interval,
        kv_values * sizeof(Float32),
        conv_values * sizeof(Float32),
        ssm_values * sizeof(Float32),
      )
    end

    def state_abi_id(abi : StateABI) : String
      io = IO::Memory.new
      write_string(io, "qwen35-state-abi-v1")
      io.write_bytes(abi.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(abi.full_attention_interval.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(abi.kv_record_byte_size.to_u64, IO::ByteFormat::LittleEndian)
      io.write_bytes(abi.conv_record_byte_size.to_u64, IO::ByteFormat::LittleEndian)
      io.write_bytes(abi.ssm_record_byte_size.to_u64, IO::ByteFormat::LittleEndian)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    private def validate_stream_identity!(context : LookupContext,
                                          stream : QwenQBitNativeBlock::Stream) : Nil
      raise ArgumentError.new("QBit Native block size mismatch") unless stream.block_size == context.qbit_block_size
      unless stream.record_spans.all? { |span| span.cache_id == cache_id(context) }
        raise ArgumentError.new("unexpected QBit Native cache identity")
      end
    end

    private def state_layout_sha256(context : LookupContext,
                                    stream : QwenQBitNativeBlock::Stream,
                                    exact : Qwen35StateSnapshot::EncodedSnapshot) : String
      records = [] of LayoutRecord
      stream.record_spans.each do |span|
        records << LayoutRecord.new(span.layer, span.kind, span.value_count.to_i64 * sizeof(Float32), 0xff_u8)
      end
      exact.records.each do |record|
        records << LayoutRecord.new(record.layer, record.kind.value, record.original_byte_size.to_i64, record.storage_mode.value.to_u8)
      end
      records.sort_by! { |record| {record.layer, record.kind} }

      io = IO::Memory.new
      write_string(io, "qwen-qbit-state-layout-v1")
      io.write_bytes(context.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(context.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      exact.positions.each { |position| io.write_bytes(position.unsafe_as(UInt32), IO::ByteFormat::LittleEndian) }
      io.write_bytes(records.size.to_u32, IO::ByteFormat::LittleEndian)
      records.each do |record|
        io.write_bytes(record.layer.unsafe_as(UInt32), IO::ByteFormat::LittleEndian)
        io.write_byte(record.kind)
        io.write_byte(record.storage_mode)
        io.write_bytes(0_u16, IO::ByteFormat::LittleEndian)
        io.write_bytes(record.original_byte_size.to_u64, IO::ByteFormat::LittleEndian)
      end
      Digest::SHA256.hexdigest(io.to_slice)
    end

    private def write_string(io : IO, value : String) : Nil
      io.write_bytes(value.bytesize.to_u32, IO::ByteFormat::LittleEndian)
      io.write(value.to_slice)
    end

    private def sha256?(value : String) : Bool
      value.matches?(/\A[0-9a-f]{64}\z/)
    end
  end
end
