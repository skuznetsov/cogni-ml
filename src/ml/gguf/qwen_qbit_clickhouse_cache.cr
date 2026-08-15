require "http/client"
require "random/secure"
require "uri"
require "uri/params"
require "./qwen_qbit_cache_envelope"
require "./qwen_qbit_session_checkpoint"

module ML::GGUF
  # Bounded ClickHouse HTTP storage for split recurrent-QBit/exact-KV state.
  #
  # Every write uses a random generation and publishes its exact manifest only
  # after both artifacts. The optional prefix index is published after that
  # manifest, so every visible prefix row names an already committed exact
  # generation. Background merges and TTL are never admission authorities.
  module QwenQBitClickHouseCache
    extend self

    QBIT_BLOCK_SIZE       =  1024
    GENERATION_HEX_SIZE   =    64
    LOOKUP_HEX_SIZE       =    64
    MAX_PREFIX_CANDIDATES = 8_192
    EMPTY_BODY            = Bytes.empty

    class Config
      getter endpoint : URI
      getter database : String
      getter table_prefix : String
      getter username : String?
      getter password : String?
      getter connect_timeout : Time::Span
      getter read_timeout : Time::Span
      getter write_timeout : Time::Span
      getter max_recurrent_bytes : Int64
      getter max_kv_bytes : Int64
      getter max_total_artifact_bytes : Int64
      getter max_envelope_bytes : Int64
      getter resident_admission_bytes : Int64

      def initialize(endpoint : String = "http://127.0.0.1:8123",
                     @database : String = "default",
                     @table_prefix : String = "qwen_qbit_cache",
                     @username : String? = nil,
                     @password : String? = nil,
                     @connect_timeout : Time::Span = 2.seconds,
                     @read_timeout : Time::Span = 30.seconds,
                     @write_timeout : Time::Span = 30.seconds,
                     @max_recurrent_bytes : Int64 = 128_i64 * 1024 * 1024,
                     @max_kv_bytes : Int64 = 128_i64 * 1024 * 1024,
                     @max_total_artifact_bytes : Int64 = 192_i64 * 1024 * 1024,
                     @max_envelope_bytes : Int64 = 64_i64 * 1024,
                     @resident_admission_bytes : Int64 = 0_i64)
        @endpoint = URI.parse(endpoint)
        validate!
      end

      private def validate! : Nil
        unless @endpoint.scheme.in?("http", "https") && @endpoint.host && !@endpoint.host.not_nil!.empty?
          raise ArgumentError.new("QBit ClickHouse endpoint must be HTTP or HTTPS with a host")
        end
        if @endpoint.query || @endpoint.fragment || @endpoint.user || @endpoint.password
          raise ArgumentError.new("QBit ClickHouse endpoint must not contain query, fragment, or credentials")
        end
        validate_identifier!(@database, "database")
        validate_identifier!(@table_prefix, "table prefix")
        unless @connect_timeout > Time::Span.zero && @read_timeout > Time::Span.zero && @write_timeout > Time::Span.zero
          raise ArgumentError.new("QBit ClickHouse timeouts must be positive")
        end
        validate_limit!(@max_recurrent_bytes, "recurrent response", 512_i64 * 1024 * 1024)
        validate_limit!(@max_kv_bytes, "KV response", 512_i64 * 1024 * 1024)
        validate_limit!(@max_total_artifact_bytes, "combined artifact", 512_i64 * 1024 * 1024)
        validate_limit!(@max_envelope_bytes, "envelope response", 1024_i64 * 1024)
        unless @resident_admission_bytes >= 0 && @resident_admission_bytes <= 512_i64 * 1024 * 1024
          raise ArgumentError.new("QBit resident admission limit is outside 0..512MiB")
        end
      end

      private def validate_identifier!(value : String, label : String) : Nil
        unless value.matches?(/\A[A-Za-z_][A-Za-z0-9_]*\z/)
          raise ArgumentError.new("QBit ClickHouse #{label} is not a safe identifier")
        end
      end

      private def validate_limit!(value : Int64, label : String, maximum : Int64) : Nil
        unless value > 0 && value <= maximum
          raise ArgumentError.new("QBit ClickHouse #{label} limit is outside 1..#{maximum}")
        end
      end
    end

    abstract class Transport
      abstract def post(query : String, body : Bytes, max_response_bytes : Int64) : Bytes
    end

    class HTTPTransport < Transport
      def initialize(@config : Config)
      end

      def post(query : String, body : Bytes, max_response_bytes : Int64) : Bytes
        raise ArgumentError.new("QBit ClickHouse response limit must be positive") unless max_response_bytes > 0
        headers = HTTP::Headers{
          "Accept-Encoding" => "identity",
          "Content-Type"    => "application/octet-stream",
        }
        if username = @config.username
          headers["X-ClickHouse-User"] = username
          headers["X-ClickHouse-Key"] = @config.password || ""
        end

        response_bytes = Bytes.empty
        HTTP::Client.new(@config.endpoint) do |client|
          client.connect_timeout = @config.connect_timeout
          client.read_timeout = @config.read_timeout
          client.write_timeout = @config.write_timeout
          client.exec("POST", request_path(query), headers, body) do |response|
            unless response.status.success?
              error = begin
                String.new(self.class.read_bounded(response.body_io, 64_i64 * 1024))
              rescue ArgumentError
                "response body exceeded 64KiB"
              end
              raise IO::Error.new("ClickHouse HTTP #{response.status_code}: #{error}")
            end
            response_bytes = self.class.read_bounded(response.body_io, max_response_bytes)
          end
        end
        response_bytes
      end

      def self.read_bounded(io : IO, max_bytes : Int64) : Bytes
        raise ArgumentError.new("QBit ClickHouse response limit must be positive") unless max_bytes > 0
        output = IO::Memory.new(Math.min(max_bytes, 64_i64 * 1024).to_i)
        buffer = Bytes.new(64 * 1024)
        total = 0_i64
        while (read = io.read(buffer)) > 0
          total += read
          if total > max_bytes
            raise ArgumentError.new("QBit ClickHouse response exceeds #{max_bytes} bytes")
          end
          output.write(buffer[0, read])
        end
        output.to_slice.dup
      end

      private def request_path(query : String) : String
        path = @config.endpoint.path
        path = "/" if path.empty?
        params = URI::Params.encode({
          "query"             => query,
          "database"          => @config.database,
          "wait_end_of_query" => "1",
          "async_insert"      => "0",
        })
        "#{path}?#{params}"
      end
    end

    record Saved,
      entry : QwenQBitCacheEnvelope::Entry,
      generation_id : String,
      expires_at_unix : Int64

    private record ResidentAdmission,
      admission : QwenQBitCacheEnvelope::Admission,
      byte_size : Int64

    class Store
      @transport : Transport
      @generation_id_factory : Proc(String)
      @resident = {} of String => ResidentAdmission
      @resident_order = [] of String
      @resident_bytes = 0_i64
      @resident_mutex = Mutex.new

      def initialize(@config : Config,
                     transport : Transport? = nil,
                     generation_id_factory : Proc(String)? = nil)
        @transport = transport || HTTPTransport.new(@config)
        @generation_id_factory = generation_id_factory || -> { Random::Secure.hex(32) }
      end

      def create_schema : Nil
        schema_queries.each do |query|
          @transport.post(query, EMPTY_BODY, @config.max_envelope_bytes)
        end
      end

      def save(context : QwenQBitCacheEnvelope::Context,
               recurrent_native : Bytes,
               kv_artifact : Bytes,
               ttl : Time::Span,
               created_at_unix : Int64 = Time.utc.to_unix) : Saved
        validate_context_layout!(context)
        validate_input_size!(recurrent_native, @config.max_recurrent_bytes, "recurrent")
        validate_input_size!(kv_artifact, @config.max_kv_bytes, "KV")
        validate_combined_size!(recurrent_native.size.to_i64, kv_artifact.size.to_i64)
        ttl_seconds = ttl.total_seconds.to_i64
        unless ttl_seconds > 0 && ttl_seconds <= 365_i64 * 24 * 60 * 60
          raise ArgumentError.new("QBit ClickHouse TTL must be within 1 second..365 days")
        end
        unless created_at_unix > 0 && created_at_unix <= Int64::MAX - ttl_seconds
          raise ArgumentError.new("QBit ClickHouse creation time is invalid")
        end

        entry = QwenQBitCacheEnvelope.build(context, recurrent_native, kv_artifact, created_at_unix)
        envelope_json = entry.to_json
        if envelope_json.bytesize.to_i64 > @config.max_envelope_bytes
          raise ArgumentError.new("QBit cache envelope exceeds #{@config.max_envelope_bytes} bytes")
        end
        generation_id = @generation_id_factory.call
        validate_hex_id!(generation_id, GENERATION_HEX_SIZE, "generation")
        lookup_key = QwenQBitCacheEnvelope.lookup_key(context)
        expires_at_unix = created_at_unix + ttl_seconds

        # Publish artifacts first. The exact manifest is their commit marker;
        # the prefix row is a secondary, fail-closed discoverability index and
        # is visible only after the exact generation is committed.
        @transport.post(
          recurrent_insert_query(lookup_key, generation_id, expires_at_unix),
          recurrent_native,
          @config.max_envelope_bytes,
        )
        @transport.post(
          blob_insert_query(kv_table, "payload", entry.cache_id, lookup_key, generation_id, expires_at_unix),
          kv_artifact,
          @config.max_envelope_bytes,
        )
        @transport.post(
          manifest_insert_query(entry, lookup_key, generation_id, expires_at_unix),
          envelope_json.to_slice,
          @config.max_envelope_bytes,
        )
        @transport.post(
          prefix_insert_query(entry, context, lookup_key, generation_id, expires_at_unix),
          envelope_json.to_slice,
          @config.max_envelope_bytes,
        )
        Saved.new(entry, generation_id, expires_at_unix)
      end

      def lookup(context : QwenQBitCacheEnvelope::Context) : QwenQBitCacheEnvelope::Admission?
        lookup_internal(context, context)
      end

      def lookup(context : QwenQBitCacheEnvelope::LookupContext) : QwenQBitCacheEnvelope::Admission?
        lookup_internal(context, nil)
      end

      def lookup_longest_prefix(context : QwenQBitCacheEnvelope::PrefixContext,
                                token_ids : Array(Int32)) : QwenQBitCacheEnvelope::Admission?
        validate_prefix_request!(context, token_ids)
        response_limit = @config.max_envelope_bytes + GENERATION_HEX_SIZE + LOOKUP_HEX_SIZE
        indexed = @transport.post(
          prefix_lookup_query(context, token_ids),
          EMPTY_BODY,
          response_limit,
        )
        return nil if indexed.empty?
        enforce_response_limit!(indexed, response_limit)
        header_size = GENERATION_HEX_SIZE + LOOKUP_HEX_SIZE
        unless indexed.size > header_size
          raise ArgumentError.new("malformed QBit ClickHouse prefix response")
        end

        generation_id = String.new(indexed[0, GENERATION_HEX_SIZE])
        validate_hex_id!(generation_id, GENERATION_HEX_SIZE, "generation")
        indexed_lookup_key = String.new(indexed[GENERATION_HEX_SIZE, LOOKUP_HEX_SIZE])
        validate_hex_id!(indexed_lookup_key, LOOKUP_HEX_SIZE, "lookup")
        entry = begin
          QwenQBitCacheEnvelope::Entry.from_json(
            String.new(indexed[header_size, indexed.size - header_size])
          )
        rescue ex : JSON::ParseException
          raise ArgumentError.new("malformed QBit ClickHouse prefix response: #{ex.message}")
        end
        unless entry.prefix_len > 0 && entry.prefix_len <= token_ids.size
          raise ArgumentError.new("QBit prefix length is outside the request")
        end
        token_hash = Qwen35PromptCache.token_hash(token_ids, entry.prefix_len)
        lookup = QwenQBitCacheEnvelope.validate_prefix_manifest!(
          entry,
          context,
          token_hash,
          entry.prefix_len,
        )
        unless indexed_lookup_key == QwenQBitCacheEnvelope.lookup_key(lookup)
          raise ArgumentError.new("QBit prefix lookup key mismatch")
        end
        load_admission(entry, lookup, indexed_lookup_key, generation_id, nil)
      end

      def save_checkpoint(entry : QwenQBitSessionCheckpoint::Entry) : QwenQBitSessionCheckpoint::Entry
        QwenQBitSessionCheckpoint.validate_certificate!(entry)
        envelope_json = entry.to_json
        if envelope_json.bytesize.to_i64 > @config.max_envelope_bytes
          raise ArgumentError.new("QBit checkpoint envelope exceeds #{@config.max_envelope_bytes} bytes")
        end
        @transport.post(
          checkpoint_insert_query(entry),
          envelope_json.to_slice,
          @config.max_envelope_bytes,
        )
        entry
      end

      def lookup_checkpoint(session_id : String,
                            checkpoint_id : String,
                            rendered : String,
                            token_ids : Array(Int32)) : QwenQBitSessionCheckpoint::Entry?
        session_hash = QwenQBitSessionCheckpoint.session_hash(session_id)
        validate_hex_id!(checkpoint_id, QwenQBitSessionCheckpoint::HEX_ID_SIZE, "checkpoint")
        response = @transport.post(
          checkpoint_lookup_query(session_hash, checkpoint_id),
          EMPTY_BODY,
          @config.max_envelope_bytes,
        )
        parse_checkpoint_response(response, session_id, rendered, token_ids, checkpoint_id)
      end

      def lookup_latest_checkpoint(session_id : String,
                                   rendered : String,
                                   token_ids : Array(Int32)) : QwenQBitSessionCheckpoint::Entry?
        boundary_hashes = checkpoint_boundary_hashes(rendered)
        return nil if boundary_hashes.empty?
        session_hash = QwenQBitSessionCheckpoint.session_hash(session_id)
        response = @transport.post(
          latest_checkpoint_lookup_query(session_hash, boundary_hashes),
          EMPTY_BODY,
          @config.max_envelope_bytes,
        )
        parse_checkpoint_response(response, session_id, rendered, token_ids, nil)
      end

      def lookup_checkpoint_anchor(checkpoint : QwenQBitSessionCheckpoint::Entry,
                                   context : QwenQBitCacheEnvelope::PrefixContext) : QwenQBitCacheEnvelope::Admission?
        QwenQBitSessionCheckpoint.validate_certificate!(checkpoint)
        anchor_token_hash = Qwen35PromptCache.token_hash(checkpoint.anchor_token_ids)
        unless anchor_token_hash == checkpoint.anchor_token_hash
          raise ArgumentError.new("QBit checkpoint anchor token hash mismatch")
        end
        manifest = @transport.post(
          checkpoint_anchor_manifest_query(checkpoint),
          EMPTY_BODY,
          @config.max_envelope_bytes,
        )
        return nil if manifest.empty?
        enforce_response_limit!(manifest, @config.max_envelope_bytes)
        entry = begin
          QwenQBitCacheEnvelope::Entry.from_json(String.new(manifest))
        rescue ex : JSON::ParseException
          raise ArgumentError.new("malformed QBit checkpoint anchor response: #{ex.message}")
        end
        unless entry.cache_id == checkpoint.anchor_cache_id &&
               entry.certificate_id == checkpoint.anchor_certificate_id
          raise ArgumentError.new("QBit checkpoint anchor reference mismatch")
        end
        lookup = QwenQBitCacheEnvelope.validate_prefix_manifest!(
          entry,
          context,
          anchor_token_hash,
          checkpoint.anchor_prefix_len,
        )
        unless QwenQBitCacheEnvelope.lookup_key(lookup) == checkpoint.anchor_lookup_key
          raise ArgumentError.new("QBit checkpoint anchor lookup key mismatch")
        end
        load_admission(
          entry,
          lookup,
          checkpoint.anchor_lookup_key,
          checkpoint.anchor_generation_id,
          nil,
        )
      end

      private def lookup_internal(context : QwenQBitCacheEnvelope::LookupContext,
                                  expected : QwenQBitCacheEnvelope::Context?) : QwenQBitCacheEnvelope::Admission?
        validate_context_layout!(context)
        lookup_key = QwenQBitCacheEnvelope.lookup_key(context)
        manifest = @transport.post(
          manifest_lookup_query(QwenQBitCacheEnvelope.cache_id(context), lookup_key),
          EMPTY_BODY,
          @config.max_envelope_bytes + GENERATION_HEX_SIZE,
        )
        return nil if manifest.empty?
        if manifest.size.to_i64 > @config.max_envelope_bytes + GENERATION_HEX_SIZE
          raise ArgumentError.new("QBit ClickHouse response exceeds #{@config.max_envelope_bytes + GENERATION_HEX_SIZE} bytes")
        end
        unless manifest.size > GENERATION_HEX_SIZE
          raise ArgumentError.new("malformed QBit ClickHouse manifest response")
        end

        generation_id = String.new(manifest[0, GENERATION_HEX_SIZE])
        validate_hex_id!(generation_id, GENERATION_HEX_SIZE, "generation")
        entry = begin
          QwenQBitCacheEnvelope::Entry.from_json(String.new(manifest[GENERATION_HEX_SIZE, manifest.size - GENERATION_HEX_SIZE]))
        rescue ex : JSON::ParseException
          raise ArgumentError.new("malformed QBit ClickHouse manifest response: #{ex.message}")
        end
        if full_context = expected
          QwenQBitCacheEnvelope.validate_manifest!(entry, full_context)
        else
          QwenQBitCacheEnvelope.validate_manifest!(entry, context)
        end

        load_admission(entry, context, lookup_key, generation_id, expected)
      end

      private def schema_queries : Array(String)
        [
          <<-SQL,
          CREATE TABLE IF NOT EXISTS #{recurrent_table} (
            cache_id UInt64,
            lookup_key FixedString(64),
            generation_id FixedString(64),
            expires_at_unix Int64,
            layer Int32,
            kind UInt8,
            tile UInt32,
            value_count UInt16,
            mean Float32,
            sigma Float32,
            codes QBit(Int8, #{QBIT_BLOCK_SIZE})
          ) ENGINE=MergeTree
          ORDER BY (cache_id, lookup_key, generation_id, layer, kind, tile)
          TTL toDateTime(expires_at_unix)
          SQL
          <<-SQL,
          CREATE TABLE IF NOT EXISTS #{kv_table} (
            cache_id UInt64,
            lookup_key FixedString(64),
            generation_id FixedString(64),
            expires_at_unix Int64,
            payload String CODEC(LZ4)
          ) ENGINE=MergeTree
          ORDER BY (cache_id, lookup_key, generation_id)
          TTL toDateTime(expires_at_unix)
          SQL
          <<-SQL,
          CREATE TABLE IF NOT EXISTS #{manifest_table} (
            cache_id UInt64,
            lookup_key FixedString(64),
            generation_id FixedString(64),
            created_at_unix Int64,
            expires_at_unix Int64,
            envelope String CODEC(ZSTD(1))
          ) ENGINE=MergeTree
          ORDER BY (cache_id, lookup_key, created_at_unix, generation_id)
          TTL toDateTime(expires_at_unix)
          SQL
          <<-SQL,
          CREATE TABLE IF NOT EXISTS #{prefix_index_table} (
            scope_key FixedString(64),
            token_hash FixedString(64),
            prefix_len UInt32,
            cache_id UInt64,
            lookup_key FixedString(64),
            generation_id FixedString(64),
            created_at_unix Int64,
            expires_at_unix Int64,
            envelope String CODEC(ZSTD(1))
          ) ENGINE=MergeTree
          ORDER BY (scope_key, token_hash, prefix_len, created_at_unix, generation_id)
          TTL toDateTime(expires_at_unix)
          SQL
          <<-SQL,
          CREATE TABLE IF NOT EXISTS #{checkpoint_table} (
            session_hash FixedString(64),
            checkpoint_id FixedString(64),
            parent_checkpoint_id String,
            child_token_hash FixedString(64),
            child_prefix_len UInt32,
            boundary_text_hash FixedString(64),
            boundary_text_bytes UInt32,
            created_at_unix Int64,
            expires_at_unix Int64,
            envelope String CODEC(ZSTD(1))
          ) ENGINE=MergeTree
          ORDER BY (session_hash, checkpoint_id, boundary_text_hash, boundary_text_bytes, created_at_unix)
          TTL toDateTime(expires_at_unix)
          SQL
        ]
      end

      private def recurrent_insert_query(lookup_key : String,
                                         generation_id : String,
                                         expires_at_unix : Int64) : String
        <<-SQL
        INSERT INTO #{recurrent_table}
        SELECT cache_id, toFixedString('#{lookup_key}', 64), toFixedString('#{generation_id}', 64),
               toInt64(#{expires_at_unix}), layer, kind, tile, value_count, mean, sigma, codes
        FROM input('cache_id UInt64, layer Int32, kind UInt8, tile UInt32, value_count UInt16, mean Float32, sigma Float32, codes QBit(Int8, #{QBIT_BLOCK_SIZE})')
        FORMAT Native
        SQL
      end

      private def blob_insert_query(table : String,
                                    column : String,
                                    cache_id : UInt64,
                                    lookup_key : String,
                                    generation_id : String,
                                    expires_at_unix : Int64) : String
        <<-SQL
        INSERT INTO #{table}
        SELECT toUInt64(#{cache_id}), toFixedString('#{lookup_key}', 64), toFixedString('#{generation_id}', 64),
               toInt64(#{expires_at_unix}), #{column}
        FROM input('#{column} String')
        FORMAT RawBLOB
        SQL
      end

      private def manifest_insert_query(entry : QwenQBitCacheEnvelope::Entry,
                                        lookup_key : String,
                                        generation_id : String,
                                        expires_at_unix : Int64) : String
        <<-SQL
        INSERT INTO #{manifest_table}
        SELECT toUInt64(#{entry.cache_id}), toFixedString('#{lookup_key}', 64), toFixedString('#{generation_id}', 64),
               toInt64(#{entry.created_at_unix}), toInt64(#{expires_at_unix}), envelope
        FROM input('envelope String')
        FORMAT RawBLOB
        SQL
      end

      private def manifest_lookup_query(cache_id : UInt64, lookup_key : String) : String
        <<-SQL
        SELECT concat(generation_id, envelope)
        FROM #{manifest_table}
        WHERE cache_id = toUInt64(#{cache_id})
          AND lookup_key = toFixedString('#{lookup_key}', 64)
          AND expires_at_unix > toUnixTimestamp(now())
        ORDER BY created_at_unix DESC, generation_id DESC
        LIMIT 1
        FORMAT RawBLOB
        SQL
      end

      private def prefix_insert_query(entry : QwenQBitCacheEnvelope::Entry,
                                      context : QwenQBitCacheEnvelope::LookupContext,
                                      lookup_key : String,
                                      generation_id : String,
                                      expires_at_unix : Int64) : String
        scope_key = QwenQBitCacheEnvelope.prefix_scope_key(QwenQBitCacheEnvelope.prefix_context(context))
        <<-SQL
        INSERT INTO #{prefix_index_table}
        SELECT toFixedString('#{scope_key}', 64), toFixedString('#{entry.token_hash}', 64),
               toUInt32(#{entry.prefix_len}), toUInt64(#{entry.cache_id}),
               toFixedString('#{lookup_key}', 64), toFixedString('#{generation_id}', 64),
               toInt64(#{entry.created_at_unix}), toInt64(#{expires_at_unix}), envelope
        FROM input('envelope String')
        FORMAT RawBLOB
        SQL
      end

      private def prefix_lookup_query(context : QwenQBitCacheEnvelope::PrefixContext,
                                      token_ids : Array(Int32)) : String
        scope_key = QwenQBitCacheEnvelope.prefix_scope_key(context)
        token_hashes = (1..token_ids.size).map do |prefix_len|
          "toFixedString('#{Qwen35PromptCache.token_hash(token_ids, prefix_len)}', 64)"
        end.join(", ")
        <<-SQL
        SELECT concat(generation_id, lookup_key, envelope)
        FROM #{prefix_index_table}
        WHERE scope_key = toFixedString('#{scope_key}', 64)
          AND token_hash IN (#{token_hashes})
          AND prefix_len <= toUInt32(#{token_ids.size})
          AND expires_at_unix > toUnixTimestamp(now())
        ORDER BY prefix_len DESC, created_at_unix DESC, generation_id DESC
        LIMIT 1
        FORMAT RawBLOB
        SQL
      end

      private def checkpoint_insert_query(entry : QwenQBitSessionCheckpoint::Entry) : String
        parent = entry.parent_checkpoint_id || ""
        <<-SQL
        INSERT INTO #{checkpoint_table}
        SELECT toFixedString('#{entry.session_hash}', 64), toFixedString('#{entry.checkpoint_id}', 64),
               '#{parent}', toFixedString('#{entry.child_token_hash}', 64),
               toUInt32(#{entry.child_prefix_len}), toFixedString('#{entry.boundary_text_hash}', 64),
               toUInt32(#{entry.boundary_text_bytes}), toInt64(#{entry.created_at_unix}),
               toInt64(#{entry.expires_at_unix}), envelope
        FROM input('envelope String')
        FORMAT RawBLOB
        SQL
      end

      private def checkpoint_lookup_query(session_hash : String, checkpoint_id : String) : String
        <<-SQL
        SELECT envelope
        FROM #{checkpoint_table}
        WHERE session_hash = toFixedString('#{session_hash}', 64)
          AND checkpoint_id = toFixedString('#{checkpoint_id}', 64)
          AND expires_at_unix > toUnixTimestamp(now())
        ORDER BY created_at_unix DESC
        LIMIT 1
        FORMAT RawBLOB
        SQL
      end

      private def latest_checkpoint_lookup_query(session_hash : String,
                                                 boundary_hashes : Array(String)) : String
        hashes = boundary_hashes.map { |hash| "toFixedString('#{hash}', 64)" }.join(", ")
        <<-SQL
        SELECT envelope
        FROM #{checkpoint_table}
        WHERE session_hash = toFixedString('#{session_hash}', 64)
          AND boundary_text_hash IN (#{hashes})
          AND expires_at_unix > toUnixTimestamp(now())
        ORDER BY boundary_text_bytes DESC, created_at_unix DESC, checkpoint_id DESC
        LIMIT 1
        FORMAT RawBLOB
        SQL
      end

      private def checkpoint_anchor_manifest_query(entry : QwenQBitSessionCheckpoint::Entry) : String
        <<-SQL
        SELECT envelope
        FROM #{manifest_table}
        WHERE cache_id = toUInt64(#{entry.anchor_cache_id})
          AND lookup_key = toFixedString('#{entry.anchor_lookup_key}', 64)
          AND generation_id = toFixedString('#{entry.anchor_generation_id}', 64)
          AND expires_at_unix > toUnixTimestamp(now())
        LIMIT 1
        FORMAT RawBLOB
        SQL
      end

      private def recurrent_lookup_query(cache_id : UInt64,
                                         lookup_key : String,
                                         generation_id : String) : String
        <<-SQL
        SELECT cache_id, layer, kind, tile, value_count, mean, sigma, codes
        FROM #{recurrent_table}
        WHERE cache_id = toUInt64(#{cache_id})
          AND lookup_key = toFixedString('#{lookup_key}', 64)
          AND generation_id = toFixedString('#{generation_id}', 64)
          AND expires_at_unix > toUnixTimestamp(now())
        ORDER BY layer, kind, tile
        FORMAT Native
        SQL
      end

      private def kv_lookup_query(cache_id : UInt64,
                                  lookup_key : String,
                                  generation_id : String) : String
        <<-SQL
        SELECT payload
        FROM #{kv_table}
        WHERE cache_id = toUInt64(#{cache_id})
          AND lookup_key = toFixedString('#{lookup_key}', 64)
          AND generation_id = toFixedString('#{generation_id}', 64)
          AND expires_at_unix > toUnixTimestamp(now())
        LIMIT 2
        FORMAT RawBLOB
        SQL
      end

      private def recurrent_table : String
        "#{@config.table_prefix}_recurrent"
      end

      private def kv_table : String
        "#{@config.table_prefix}_kv"
      end

      private def manifest_table : String
        "#{@config.table_prefix}_manifest"
      end

      private def prefix_index_table : String
        "#{@config.table_prefix}_prefix_index"
      end

      private def checkpoint_table : String
        "#{@config.table_prefix}_checkpoints"
      end

      private def validate_context_layout!(context : QwenQBitCacheEnvelope::LookupContext) : Nil
        unless context.qbit_block_size == QBIT_BLOCK_SIZE
          raise ArgumentError.new("QBit ClickHouse cache requires block size #{QBIT_BLOCK_SIZE}")
        end
      end

      private def validate_prefix_request!(context : QwenQBitCacheEnvelope::PrefixContext,
                                           token_ids : Array(Int32)) : Nil
        # Validate the complete scope before constructing SQL from it.
        QwenQBitCacheEnvelope.prefix_scope_key(context)
        raise ArgumentError.new("QBit prefix request is empty") if token_ids.empty?
        if token_ids.size > context.max_seq
          raise ArgumentError.new("QBit prefix request exceeds max_seq")
        end
        if token_ids.size > MAX_PREFIX_CANDIDATES
          raise ArgumentError.new("QBit prefix candidate limit exceeded")
        end
        unless context.qbit_block_size == QBIT_BLOCK_SIZE
          raise ArgumentError.new("QBit ClickHouse cache requires block size #{QBIT_BLOCK_SIZE}")
        end
      end

      private def checkpoint_boundary_hashes(rendered : String) : Array(String)
        raise ArgumentError.new("QBit checkpoint transcript is empty") if rendered.empty?
        if rendered.bytesize > QwenQBitSessionCheckpoint::MAX_BOUNDARY_BYTES
          raise ArgumentError.new("QBit checkpoint transcript exceeds the text boundary limit")
        end
        hashes = [] of String
        cursor = 0
        marker = "<|im_end|>"
        while offset = rendered.index(marker, cursor)
          boundary_end = offset + marker.bytesize
          boundary_end += 1 if rendered.byte_at?(boundary_end) == '\n'.ord
          prefix = rendered.byte_slice(0, boundary_end)
          hashes << QwenQBitSessionCheckpoint.boundary_hash(prefix) if prefix && !prefix.empty?
          if hashes.size > QwenQBitSessionCheckpoint::MAX_BOUNDARY_CANDIDATES
            raise ArgumentError.new("QBit checkpoint text boundary candidate limit exceeded")
          end
          cursor = boundary_end
        end
        hashes.uniq!
        hashes
      end

      private def parse_checkpoint_response(response : Bytes,
                                            session_id : String,
                                            rendered : String,
                                            token_ids : Array(Int32),
                                            checkpoint_id : String?) : QwenQBitSessionCheckpoint::Entry?
        return nil if response.empty?
        enforce_response_limit!(response, @config.max_envelope_bytes)
        entry = begin
          QwenQBitSessionCheckpoint::Entry.from_json(String.new(response))
        rescue ex : JSON::ParseException
          raise ArgumentError.new("malformed QBit checkpoint response: #{ex.message}")
        end
        QwenQBitSessionCheckpoint.validate_boundary!(
          entry,
          session_id,
          rendered,
          checkpoint_id: checkpoint_id,
        )
        QwenQBitSessionCheckpoint.validate!(
          entry,
          session_id,
          token_ids,
          checkpoint_id: checkpoint_id,
        )
        entry
      end

      private def load_admission(entry : QwenQBitCacheEnvelope::Entry,
                                 context : QwenQBitCacheEnvelope::LookupContext,
                                 lookup_key : String,
                                 generation_id : String,
                                 expected : QwenQBitCacheEnvelope::Context?) : QwenQBitCacheEnvelope::Admission
        if admission = resident_admission(generation_id, entry.certificate_id)
          return admission
        end

        recurrent_native = @transport.post(
          recurrent_lookup_query(entry.cache_id, lookup_key, generation_id),
          EMPTY_BODY,
          @config.max_recurrent_bytes,
        )
        enforce_response_limit!(recurrent_native, @config.max_recurrent_bytes)
        # The envelope authenticates the exact-KV byte count. Reject a combined
        # response that cannot fit the configured budget before allocating the
        # second large HTTP response.
        validate_combined_size!(recurrent_native.size.to_i64, entry.kv_artifact_byte_size)
        kv_artifact = @transport.post(
          kv_lookup_query(entry.cache_id, lookup_key, generation_id),
          EMPTY_BODY,
          @config.max_kv_bytes,
        )
        enforce_response_limit!(kv_artifact, @config.max_kv_bytes)
        validate_combined_size!(recurrent_native.size.to_i64, kv_artifact.size.to_i64)
        admission = if full_context = expected
                      QwenQBitCacheEnvelope.admit(entry, full_context, recurrent_native, kv_artifact)
                    else
                      QwenQBitCacheEnvelope.admit(entry, context, recurrent_native, kv_artifact)
                    end
        remember_admission(generation_id, admission, recurrent_native.size.to_i64 + kv_artifact.size)
        admission
      end

      private def validate_input_size!(bytes : Bytes, limit : Int64, label : String) : Nil
        unless bytes.size > 0 && bytes.size.to_i64 <= limit
          raise ArgumentError.new("QBit #{label} input is outside 1..#{limit} bytes")
        end
      end

      private def enforce_response_limit!(bytes : Bytes, limit : Int64) : Nil
        if bytes.size.to_i64 > limit
          raise ArgumentError.new("QBit ClickHouse response exceeds #{limit} bytes")
        end
      end

      private def validate_combined_size!(recurrent_bytes : Int64, kv_bytes : Int64) : Nil
        total = recurrent_bytes + kv_bytes
        if total > @config.max_total_artifact_bytes
          raise ArgumentError.new("QBit combined artifact exceeds #{@config.max_total_artifact_bytes} bytes")
        end
      end

      private def validate_hex_id!(value : String, size : Int32, label : String) : Nil
        unless value.bytesize == size && value.matches?(/\A[0-9a-f]+\z/)
          raise ArgumentError.new("QBit ClickHouse #{label} identity is invalid")
        end
      end

      private def resident_admission(generation_id : String,
                                     certificate_id : String) : QwenQBitCacheEnvelope::Admission?
        return nil if @config.resident_admission_bytes == 0
        @resident_mutex.synchronize do
          if resident = @resident[generation_id]?
            if resident.admission.entry.certificate_id == certificate_id
              @resident_order.delete(generation_id)
              @resident_order << generation_id
              return resident.admission
            end
            remove_resident!(generation_id)
          end
        end
        nil
      end

      private def remember_admission(generation_id : String,
                                     admission : QwenQBitCacheEnvelope::Admission,
                                     byte_size : Int64) : Nil
        limit = @config.resident_admission_bytes
        return if limit == 0 || byte_size > limit
        @resident_mutex.synchronize do
          remove_resident!(generation_id)
          while @resident_bytes + byte_size > limit && (oldest = @resident_order.first?)
            remove_resident!(oldest)
          end
          @resident[generation_id] = ResidentAdmission.new(admission, byte_size)
          @resident_order << generation_id
          @resident_bytes += byte_size
        end
      end

      private def remove_resident!(generation_id : String) : Nil
        if resident = @resident.delete(generation_id)
          @resident_bytes -= resident.byte_size
        end
        @resident_order.delete(generation_id)
      end
    end
  end
end
