require "digest/sha256"
require "file_utils"
require "json"
require "./qwen35_state_snapshot"

module ML::GGUF
  # Exact prompt-prefix cache for Qwen 3.5/3.6 decode state.
  #
  # The cache is deliberately conservative: it only restores artifacts whose
  # runtime/model/tokenizer/prompt metadata match and whose .qkv SHA-256 verifies.
  module Qwen35PromptCache
    extend self

    RUNTIME_ID = "cogni-ml/qwen35-state-v1"

    class Entry
      include JSON::Serializable

      property runtime_id : String
      property session_id : String
      property turn_id : String?
      property model_id : String
      property tokenizer_id : String
      property prompt_hash : String
      property token_hash : String? = nil
      property prefix_len : Int32
      property max_seq : Int32
      property layer_count : Int32
      property artifact_path : String
      property artifact_sha256 : String
      property artifact_byte_size : Int64
      property state_byte_size : Int64
      property created_at_unix : Int64
      property prompt_preview : String?

      def initialize(@runtime_id : String,
                     @session_id : String,
                     @turn_id : String?,
                     @model_id : String,
                     @tokenizer_id : String,
                     @prompt_hash : String,
                     @prefix_len : Int32,
                     @max_seq : Int32,
                     @layer_count : Int32,
                     @artifact_path : String,
                     @artifact_sha256 : String,
                     @artifact_byte_size : Int64,
                     @state_byte_size : Int64,
                     @created_at_unix : Int64,
                     @prompt_preview : String?,
                     @token_hash : String? = nil)
      end
    end

    record ReplayResult,
      state : Qwen35CPU::State,
      reused_prefix_len : Int32,
      replayed_tokens : Int32,
      entry : Entry,
      next_token_id : Int32?,
      next_token_logit : Float32?

    class Store
      getter root : String
      getter manifest_path : String

      def initialize(@root : String = Qwen35PromptCache.default_root)
        @manifest_path = File.join(@root, "manifest.jsonl")
      end

      def save(session_id : String,
               model_id : String,
               tokenizer_id : String,
               prompt_text : String,
               token_ids : Array(Int32),
               state : Qwen35CPU::State,
               turn_id : String? = nil,
               prompt_preview : String? = nil) : Entry
        snapshot = Qwen35StateSnapshot.capture(state)
        prompt_hash = Qwen35PromptCache.prompt_hash(token_ids, prompt_text)
        token_hash = Qwen35PromptCache.token_hash(token_ids)
        artifact_path = artifact_path(model_id, tokenizer_id, prompt_hash, token_ids.size)
        artifact = Qwen35StateSnapshot.write_artifact(snapshot, artifact_path)

        entry = Entry.new(
          runtime_id: RUNTIME_ID,
          session_id: session_id,
          turn_id: turn_id,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          prompt_hash: prompt_hash,
          prefix_len: token_ids.size.to_i32,
          max_seq: snapshot.max_seq,
          layer_count: snapshot.layer_count,
          artifact_path: artifact.path,
          artifact_sha256: artifact.sha256,
          artifact_byte_size: artifact.byte_size,
          state_byte_size: snapshot.byte_size,
          created_at_unix: Time.utc.to_unix,
          prompt_preview: prompt_preview,
          token_hash: token_hash,
        )
        append_manifest(entry)
        entry
      end

      def lookup_exact(model_id : String,
                       tokenizer_id : String,
                       prompt_hash : String,
                       prefix_len : Int32) : Entry?
        entries.reverse_each.find do |entry|
          compatible?(entry, model_id, tokenizer_id, prompt_hash, prefix_len) &&
            File.exists?(entry.artifact_path)
        end
      end

      def lookup_prompt(model_id : String,
                        tokenizer_id : String,
                        prompt_text : String,
                        token_ids : Array(Int32)) : Entry?
        lookup_exact(
          model_id,
          tokenizer_id,
          Qwen35PromptCache.prompt_hash(token_ids, prompt_text),
          token_ids.size.to_i32,
        )
      end

      def lookup_longest_prefix(model_id : String,
                                tokenizer_id : String,
                                token_ids : Array(Int32),
                                min_prefix_len : Int32 = 1,
                                max_prefix_len : Int32 = token_ids.size) : Entry?
        raise ArgumentError.new("min_prefix_len must be non-negative") if min_prefix_len < 0
        raise ArgumentError.new("max_prefix_len out of range: #{max_prefix_len}") if max_prefix_len < 0 || max_prefix_len > token_ids.size

        hash_by_len = {} of Int32 => String
        candidates = entries.select do |entry|
          next false unless entry.runtime_id == RUNTIME_ID
          next false unless entry.model_id == model_id
          next false unless entry.tokenizer_id == tokenizer_id
          next false if entry.prefix_len < min_prefix_len
          next false if entry.prefix_len > max_prefix_len
          next false if entry.prefix_len > token_ids.size
          next false unless stored_token_hash = entry.token_hash
          next false unless File.exists?(entry.artifact_path)

          expected = hash_by_len[entry.prefix_len]?
          unless expected
            expected = Qwen35PromptCache.token_hash(token_ids, entry.prefix_len)
            hash_by_len[entry.prefix_len] = expected
          end
          stored_token_hash == expected
        end
        candidates.max_by? { |entry| {entry.prefix_len, entry.created_at_unix} }
      end

      def lookup_session(session_id : String,
                         turn_id : String? = nil,
                         prefix_len : Int32? = nil) : Entry?
        candidates = entries.select do |entry|
          next false unless entry.runtime_id == RUNTIME_ID
          next false unless entry.session_id == session_id
          next false if turn_id && entry.turn_id != turn_id
          next false if prefix_len && entry.prefix_len != prefix_len

          File.exists?(entry.artifact_path)
        end
        candidates.max_by? { |entry| {entry.created_at_unix, entry.prefix_len} }
      end

      def restore(entry : Entry,
                  hp : Qwen35Hparams,
                  prefer_metal : Bool = Qwen35Metal.available?) : Qwen35CPU::State
        raise ArgumentError.new("unsupported Qwen prompt-cache runtime: #{entry.runtime_id}") unless entry.runtime_id == RUNTIME_ID

        snapshot = Qwen35StateSnapshot.read_artifact(entry.artifact_path, expected_sha256: entry.artifact_sha256)
        raise ArgumentError.new("prompt-cache max_seq mismatch") unless snapshot.max_seq == entry.max_seq
        raise ArgumentError.new("prompt-cache layer count mismatch") unless snapshot.layer_count == entry.layer_count
        Qwen35StateSnapshot.restore(snapshot, hp, prefer_metal: prefer_metal)
      end

      def restore_and_replay_suffix(entry : Entry,
                                    weights : Qwen35Weights,
                                    token_ids : Array(Int32),
                                    prefer_metal : Bool = Qwen35Metal.available?) : ReplayResult
        raise ArgumentError.new("cache prefix longer than prompt: prefix=#{entry.prefix_len}, prompt=#{token_ids.size}") if entry.prefix_len > token_ids.size

        state = restore(entry, weights.hparams, prefer_metal: prefer_metal)
        next_token_id = nil.as(Int32?)
        next_token_logit = nil.as(Float32?)
        (entry.prefix_len...token_ids.size).each do |pos|
          top, logit = Qwen35CPU.forward_top1(weights, token_ids[pos], pos.to_i32, state)
          next_token_id = top
          next_token_logit = logit
        end
        ReplayResult.new(state, entry.prefix_len, token_ids.size - entry.prefix_len, entry, next_token_id, next_token_logit)
      end

      def entries : Array(Entry)
        return [] of Entry unless File.exists?(@manifest_path)

        out = [] of Entry
        File.each_line(@manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            out << Entry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt manifest line must not produce a cache hit.
          end
        end
        out
      end

      private def append_manifest(entry : Entry) : Nil
        FileUtils.mkdir_p(@root)
        File.open(@manifest_path, "a") do |file|
          entry.to_json(file)
          file << '\n'
        end
      end

      private def compatible?(entry : Entry,
                              model_id : String,
                              tokenizer_id : String,
                              prompt_hash : String,
                              prefix_len : Int32) : Bool
        entry.runtime_id == RUNTIME_ID &&
          entry.model_id == model_id &&
          entry.tokenizer_id == tokenizer_id &&
          entry.prompt_hash == prompt_hash.downcase &&
          entry.prefix_len == prefix_len
      end

      private def artifact_path(model_id : String,
                                tokenizer_id : String,
                                prompt_hash : String,
                                prefix_len : Int32) : String
        bucket = Qwen35PromptCache.short_hash("#{model_id}\0#{tokenizer_id}")
        File.join(@root, "artifacts", bucket, "#{prefix_len}-#{prompt_hash.downcase}.qkv")
      end
    end

    def default_root : String
      if home = ENV["HOME"]?
        File.join(home, ".cache", "cogni-ml", "qwen35-kv-cache")
      else
        File.join(Dir.current, ".qwen35-kv-cache")
      end
    end

    def prompt_hash(token_ids : Array(Int32), prompt_text : String = "") : String
      io = IO::Memory.new
      io.write("qwen35-prompt-v1\0".to_slice)
      io.write_bytes(token_ids.size.to_u32, IO::ByteFormat::LittleEndian)
      token_ids.each do |token_id|
        io.write_bytes(token_id, IO::ByteFormat::LittleEndian)
      end
      io.write(prompt_text.to_slice)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def token_hash(token_ids : Array(Int32), prefix_len : Int32 = token_ids.size) : String
      raise ArgumentError.new("prefix_len out of range: #{prefix_len}") if prefix_len < 0 || prefix_len > token_ids.size

      io = IO::Memory.new
      io.write("qwen35-token-v1\0".to_slice)
      io.write_bytes(prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      prefix_len.times do |i|
        io.write_bytes(token_ids[i], IO::ByteFormat::LittleEndian)
      end
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def short_hash(value : String) : String
      Digest::SHA256.hexdigest(value)[0, 16]
    end

    def pg_sorted_heap_schema_sql(table_name : String = "qwen35_prompt_cache",
                                  table_am : String = "sorted_heap") : String
      table = pg_identifier(table_name)
      am = pg_identifier(table_am)
      <<-SQL
      CREATE EXTENSION IF NOT EXISTS pg_sorted_heap;

      CREATE TABLE IF NOT EXISTS #{table} (
          cache_id           bigserial PRIMARY KEY,
          runtime_id         text NOT NULL,
          session_id         text NOT NULL,
          turn_id            text,
          model_id           text NOT NULL,
          tokenizer_id       text NOT NULL,
          prompt_hash        text NOT NULL,
          token_hash         text NOT NULL,
          prefix_len         integer NOT NULL CHECK (prefix_len >= 0),
          max_seq            integer NOT NULL CHECK (max_seq >= prefix_len),
          layer_count        integer NOT NULL CHECK (layer_count > 0),
          artifact_path      text NOT NULL,
          artifact_sha256    text NOT NULL CHECK (length(artifact_sha256) = 64),
          artifact_byte_size bigint NOT NULL CHECK (artifact_byte_size >= 0),
          state_byte_size    bigint NOT NULL CHECK (state_byte_size >= 0),
          created_at_unix    bigint NOT NULL,
          prompt_preview     text
      ) USING #{am};

      CREATE UNIQUE INDEX IF NOT EXISTS #{table}_exact_idx
          ON #{table} (model_id, tokenizer_id, prompt_hash, prefix_len);

      CREATE INDEX IF NOT EXISTS #{table}_session_idx
          ON #{table} (session_id, turn_id, prefix_len, created_at_unix DESC);

      CREATE INDEX IF NOT EXISTS #{table}_prefix_idx
          ON #{table} (model_id, tokenizer_id, token_hash, prefix_len DESC);
      SQL
    end

    def pg_insert_sql(table_name : String = "qwen35_prompt_cache") : String
      table = pg_identifier(table_name)
      <<-SQL
      INSERT INTO #{table} (
          runtime_id, session_id, turn_id, model_id, tokenizer_id,
          prompt_hash, token_hash, prefix_len, max_seq, layer_count,
          artifact_path, artifact_sha256, artifact_byte_size, state_byte_size,
          created_at_unix, prompt_preview
      ) VALUES (
          $1, $2, $3, $4, $5,
          $6, $7, $8, $9, $10,
          $11, $12, $13, $14,
          $15, $16
      )
      ON CONFLICT (model_id, tokenizer_id, prompt_hash, prefix_len)
      DO UPDATE SET
          runtime_id = EXCLUDED.runtime_id,
          session_id = EXCLUDED.session_id,
          turn_id = EXCLUDED.turn_id,
          token_hash = EXCLUDED.token_hash,
          max_seq = EXCLUDED.max_seq,
          layer_count = EXCLUDED.layer_count,
          artifact_path = EXCLUDED.artifact_path,
          artifact_sha256 = EXCLUDED.artifact_sha256,
          artifact_byte_size = EXCLUDED.artifact_byte_size,
          state_byte_size = EXCLUDED.state_byte_size,
          created_at_unix = EXCLUDED.created_at_unix,
          prompt_preview = EXCLUDED.prompt_preview;
      SQL
    end

    def pg_insert_values(entry : Entry) : Array(String | Int32 | Int64 | Nil)
      [
        entry.runtime_id,
        entry.session_id,
        entry.turn_id,
        entry.model_id,
        entry.tokenizer_id,
        entry.prompt_hash,
        entry.token_hash,
        entry.prefix_len,
        entry.max_seq,
        entry.layer_count,
        entry.artifact_path,
        entry.artifact_sha256,
        entry.artifact_byte_size,
        entry.state_byte_size,
        entry.created_at_unix,
        entry.prompt_preview,
      ] of String | Int32 | Int64 | Nil
    end

    private def pg_identifier(name : String) : String
      raise ArgumentError.new("unsafe PostgreSQL identifier: #{name.inspect}") unless name.matches?(/\A[a-zA-Z_][a-zA-Z0-9_]*\z/)

      name
    end
  end
end
