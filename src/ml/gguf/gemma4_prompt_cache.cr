require "digest/sha256"
require "file_utils"
require "json"
require "./gemma4_state_snapshot"

module ML::GGUF
  # Minimal exact prompt-prefix cache for Gemma 4 resident K/V state.
  module Gemma4PromptCache
    extend self

    RUNTIME_ID = "cogni-ml/gemma4-state-v1"

    class Entry
      include JSON::Serializable

      property runtime_id : String
      property session_id : String
      property turn_id : String?
      property model_id : String
      property tokenizer_id : String
      property prompt_hash : String
      property token_hash : String
      property prefix_len : Int32
      property max_seq : Int32
      property layer_count : Int32
      property kv_dims : Array(Int32)
      property artifact_path : String
      property artifact_sha256 : String
      property artifact_byte_size : Int64
      property state_byte_size : Int64
      property created_at_unix : Int64
      property prompt_preview : String?
      property next_token_id : Int32? = nil

      def initialize(@runtime_id : String,
                     @session_id : String,
                     @turn_id : String?,
                     @model_id : String,
                     @tokenizer_id : String,
                     @prompt_hash : String,
                     @token_hash : String,
                     @prefix_len : Int32,
                     @max_seq : Int32,
                     @layer_count : Int32,
                     @kv_dims : Array(Int32),
                     @artifact_path : String,
                     @artifact_sha256 : String,
                     @artifact_byte_size : Int64,
                     @state_byte_size : Int64,
                     @created_at_unix : Int64,
                     @prompt_preview : String?,
                     @next_token_id : Int32? = nil)
      end
    end

    record ManifestFingerprint, size : Int64, mtime_sec : Int64, mtime_nsec : Int32
    record ArtifactFingerprint, sha256 : String, expected_size : Int64, file_size : Int64, mtime_sec : Int64, mtime_nsec : Int32

    class Store
      getter root : String
      getter snapshot_cache_byte_limit : Int64
      getter snapshot_cache_entry_limit : Int32
      getter snapshot_cache_bytes : Int64
      getter snapshot_cache_hits : Int64
      getter snapshot_cache_misses : Int64

      @entry_manifest_fingerprint : ManifestFingerprint?
      @entry_manifest_cache : Array(Entry)
      @entry_exact_index : Hash(Tuple(String, String, String, Int32), Array(Entry))
      @entry_prefix_index : Hash(Tuple(String, String, Int32, String), Array(Entry))
      @snapshot_cache : Hash(String, CachedSnapshot)

      private class CachedSnapshot
        property fingerprint : ArtifactFingerprint
        property snapshot : Gemma4StateSnapshot::Snapshot
        property byte_size : Int64
        property last_used : UInt64

        def initialize(@fingerprint : ArtifactFingerprint,
                       @snapshot : Gemma4StateSnapshot::Snapshot,
                       @byte_size : Int64,
                       @last_used : UInt64)
        end
      end

      def initialize(@root : String,
                     @snapshot_cache_byte_limit : Int64 = 0_i64,
                     @snapshot_cache_entry_limit : Int32 = 0)
        raise ArgumentError.new("snapshot_cache_byte_limit must be non-negative") if @snapshot_cache_byte_limit < 0
        raise ArgumentError.new("snapshot_cache_entry_limit must be non-negative") if @snapshot_cache_entry_limit < 0

        FileUtils.mkdir_p(File.join(@root, "artifacts"))
        @manifest_path = File.join(@root, "manifest.jsonl")
        @entry_manifest_fingerprint = nil
        @entry_manifest_cache = [] of Entry
        @entry_exact_index = {} of Tuple(String, String, String, Int32) => Array(Entry)
        @entry_prefix_index = {} of Tuple(String, String, Int32, String) => Array(Entry)
        @snapshot_cache = {} of String => CachedSnapshot
        @snapshot_cache_bytes = 0_i64
        @snapshot_cache_hits = 0_i64
        @snapshot_cache_misses = 0_i64
        @snapshot_cache_clock = 0_u64
      end

      def save_resident_state(state : Gemma4Metal::ResidentState,
                              token_ids : Array(Int32),
                              model_id : String,
                              tokenizer_id : String,
                              prompt_text : String = "",
                              session_id : String = "default",
                              turn_id : String? = nil,
                              prompt_preview : String? = nil,
                              next_token_id : Int32? = nil) : Entry
        snapshot = Gemma4StateSnapshot.capture(state, prefix_len: token_ids.size.to_i32)
        prompt_hash = Gemma4PromptCache.prompt_hash(token_ids, prompt_text)
        token_hash = Gemma4PromptCache.token_hash(token_ids)
        artifact_path = artifact_path(model_id, tokenizer_id, prompt_hash, token_ids.size.to_i32)
        artifact = Gemma4StateSnapshot.write_artifact(snapshot, artifact_path)
        kv_dims = state.layers.map(&.kv_dim)

        entry = Entry.new(
          runtime_id: RUNTIME_ID,
          session_id: session_id,
          turn_id: turn_id,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          prompt_hash: prompt_hash,
          token_hash: token_hash,
          prefix_len: token_ids.size.to_i32,
          max_seq: snapshot.max_seq,
          layer_count: snapshot.layer_count,
          kv_dims: kv_dims,
          artifact_path: artifact.path,
          artifact_sha256: artifact.sha256,
          artifact_byte_size: artifact.byte_size,
          state_byte_size: snapshot.byte_size,
          created_at_unix: Time.utc.to_unix,
          prompt_preview: prompt_preview,
          next_token_id: next_token_id,
        )
        append_manifest(entry)
        entry
      end

      def lookup_exact(model_id : String,
                       tokenizer_id : String,
                       prompt_hash : String,
                       prefix_len : Int32) : Entry?
        ensure_entry_indices
        candidates = @entry_exact_index[{model_id, tokenizer_id, prompt_hash.downcase, prefix_len}]?
        return nil unless candidates

        if hit = candidates.reverse_each.find { |entry| usable_entry?(entry) }
          clone_entry(hit)
        end
      end

      def lookup_prompt(model_id : String,
                        tokenizer_id : String,
                        prompt_text : String,
                        token_ids : Array(Int32)) : Entry?
        lookup_exact(model_id, tokenizer_id, Gemma4PromptCache.prompt_hash(token_ids, prompt_text), token_ids.size.to_i32)
      end

      def lookup_longest_prefix(model_id : String,
                                tokenizer_id : String,
                                token_ids : Array(Int32),
                                min_prefix_len : Int32 = 1,
                                max_prefix_len : Int32 = token_ids.size) : Entry?
        raise ArgumentError.new("min_prefix_len must be non-negative") if min_prefix_len < 0
        raise ArgumentError.new("max_prefix_len out of range: #{max_prefix_len}") if max_prefix_len < 0 || max_prefix_len > token_ids.size

        ensure_entry_indices
        upper = max_prefix_len < token_ids.size ? max_prefix_len : token_ids.size
        upper.downto(min_prefix_len) do |prefix_len|
          expected = Gemma4PromptCache.token_hash(token_ids, prefix_len)
          candidates = @entry_prefix_index[{model_id, tokenizer_id, prefix_len, expected}]?
          next unless candidates

          valid_candidates = candidates.select { |entry| usable_entry?(entry) }
          if hit = valid_candidates.max_by? { |entry| entry.created_at_unix }
            return clone_entry(hit)
          end
        end
        nil
      end

      def lookup_session(session_id : String,
                         turn_id : String? = nil,
                         prefix_len : Int32? = nil) : Entry?
        candidates = manifest_entries.select do |entry|
          next false unless entry.runtime_id == RUNTIME_ID
          next false unless entry.session_id == session_id
          next false if turn_id && entry.turn_id != turn_id
          next false if prefix_len && entry.prefix_len != prefix_len

          usable_entry?(entry)
        end
        if hit = candidates.max_by? { |entry| {entry.created_at_unix, entry.prefix_len} }
          clone_entry(hit)
        end
      end

      def restore(entry : Entry, reuse_state : Gemma4Metal::ResidentState? = nil) : Gemma4Metal::ResidentState
        raise ArgumentError.new("unsupported Gemma prompt-cache runtime: #{entry.runtime_id}") unless entry.runtime_id == RUNTIME_ID
        Gemma4PromptCache.validate_restorable_artifact!(entry)

        fingerprint = artifact_fingerprint(entry)
        snapshot = cached_snapshot(entry, fingerprint) || begin
          loaded = Gemma4StateSnapshot.read_artifact(entry.artifact_path, expected_sha256: entry.artifact_sha256)
          ensure_artifact_unchanged!(entry, fingerprint)
          remember_snapshot(entry, fingerprint, loaded)
          loaded
        end
        validate_snapshot_for_entry!(snapshot, entry)

        if state = reuse_state
          Gemma4StateSnapshot.restore_into(snapshot, state)
          state
        else
          Gemma4StateSnapshot.restore(snapshot, entry.kv_dims)
        end
      end

      def entries : Array(Entry)
        clone_entries(manifest_entries)
      end

      def snapshot_cache_enabled? : Bool
        @snapshot_cache_byte_limit > 0 && @snapshot_cache_entry_limit > 0
      end

      def clear_snapshot_cache : Nil
        @snapshot_cache.clear
        @snapshot_cache_bytes = 0_i64
        @snapshot_cache_hits = 0_i64
        @snapshot_cache_misses = 0_i64
      end

      private def append_manifest(entry : Entry) : Nil
        File.open(@manifest_path, "a") do |file|
          file.puts(entry.to_json)
        end
        @entry_manifest_fingerprint = nil
      end

      private def manifest_entries : Array(Entry)
        fingerprint = manifest_fingerprint(@manifest_path)
        unless fingerprint
          @entry_manifest_fingerprint = nil
          @entry_manifest_cache = [] of Entry
          return [] of Entry
        end
        return @entry_manifest_cache if @entry_manifest_fingerprint == fingerprint

        parsed = [] of Entry
        File.each_line(@manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            parsed << Entry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt manifest line must not produce a cache hit.
          end
        end
        @entry_manifest_fingerprint = fingerprint
        @entry_manifest_cache = parsed
      end

      private def ensure_entry_indices : Nil
        entries = manifest_entries
        @entry_exact_index.clear
        @entry_prefix_index.clear
        entries.each do |entry|
          next unless entry.runtime_id == RUNTIME_ID

          exact_key = {entry.model_id, entry.tokenizer_id, entry.prompt_hash.downcase, entry.prefix_len}
          (@entry_exact_index[exact_key] ||= [] of Entry) << entry
          prefix_key = {entry.model_id, entry.tokenizer_id, entry.prefix_len, entry.token_hash}
          (@entry_prefix_index[prefix_key] ||= [] of Entry) << entry
        end
      end

      private def usable_entry?(entry : Entry) : Bool
        File.exists?(entry.artifact_path) &&
          Gemma4PromptCache.artifact_trust_metadata_valid?(entry)
      end

      private def artifact_path(model_id : String,
                                tokenizer_id : String,
                                prompt_hash : String,
                                prefix_len : Int32) : String
        bucket = Gemma4PromptCache.short_hash("#{model_id}\0#{tokenizer_id}")
        File.join(@root, "artifacts", bucket, "#{prefix_len}-#{prompt_hash.downcase}.gkv")
      end

      private def artifact_fingerprint(entry : Entry) : ArtifactFingerprint
        info = File.info(entry.artifact_path)
        file_size = info.size
        raise ArgumentError.new("prompt-cache artifact byte-size mismatch") unless file_size == entry.artifact_byte_size

        mtime = info.modification_time
        ArtifactFingerprint.new(
          entry.artifact_sha256.downcase,
          entry.artifact_byte_size,
          file_size,
          mtime.to_unix,
          mtime.nanosecond,
        )
      end

      private def ensure_artifact_unchanged!(entry : Entry, fingerprint : ArtifactFingerprint) : Nil
        return if artifact_fingerprint(entry) == fingerprint

        raise ArgumentError.new("prompt-cache artifact changed during restore")
      end

      private def cached_snapshot(entry : Entry, fingerprint : ArtifactFingerprint) : Gemma4StateSnapshot::Snapshot?
        return nil unless snapshot_cache_enabled?

        key = snapshot_cache_key(entry)
        cached = @snapshot_cache[key]?
        unless cached
          @snapshot_cache_misses += 1
          return nil
        end

        unless cached.fingerprint == fingerprint
          drop_cached_snapshot(key)
          @snapshot_cache_misses += 1
          return nil
        end

        @snapshot_cache_clock += 1
        cached.last_used = @snapshot_cache_clock
        @snapshot_cache_hits += 1
        cached.snapshot
      end

      private def remember_snapshot(entry : Entry,
                                    fingerprint : ArtifactFingerprint,
                                    snapshot : Gemma4StateSnapshot::Snapshot) : Nil
        return unless snapshot_cache_enabled?

        byte_size = snapshot.byte_size
        return if byte_size > @snapshot_cache_byte_limit

        key = snapshot_cache_key(entry)
        drop_cached_snapshot(key)
        @snapshot_cache_clock += 1
        @snapshot_cache[key] = CachedSnapshot.new(fingerprint, snapshot, byte_size, @snapshot_cache_clock)
        @snapshot_cache_bytes += byte_size
        evict_snapshot_cache
      end

      private def evict_snapshot_cache : Nil
        while @snapshot_cache.size > @snapshot_cache_entry_limit || @snapshot_cache_bytes > @snapshot_cache_byte_limit
          victim_key = nil.as(String?)
          victim_last_used = UInt64::MAX
          @snapshot_cache.each do |key, cached|
            if cached.last_used < victim_last_used
              victim_key = key
              victim_last_used = cached.last_used
            end
          end
          break unless key = victim_key

          drop_cached_snapshot(key)
        end
      end

      private def drop_cached_snapshot(key : String) : Nil
        if cached = @snapshot_cache.delete(key)
          @snapshot_cache_bytes -= cached.byte_size
        end
      end

      private def snapshot_cache_key(entry : Entry) : String
        "#{entry.artifact_path}\0#{entry.artifact_sha256.downcase}\0#{entry.artifact_byte_size}"
      end

      private def validate_snapshot_for_entry!(snapshot : Gemma4StateSnapshot::Snapshot, entry : Entry) : Nil
        raise ArgumentError.new("prompt-cache max_seq mismatch") unless snapshot.max_seq == entry.max_seq
        raise ArgumentError.new("prompt-cache prefix_len mismatch") unless snapshot.prefix_len == entry.prefix_len
        raise ArgumentError.new("prompt-cache layer count mismatch") unless snapshot.layer_count == entry.layer_count
      end

      private def manifest_fingerprint(path : String) : ManifestFingerprint?
        return nil unless File.exists?(path)

        info = File.info(path)
        mtime = info.modification_time
        ManifestFingerprint.new(info.size, mtime.to_unix, mtime.nanosecond)
      end

      private def clone_entries(entries : Array(Entry)) : Array(Entry)
        entries.map { |entry| clone_entry(entry) }
      end

      private def clone_entry(entry : Entry) : Entry
        Entry.new(
          runtime_id: entry.runtime_id,
          session_id: entry.session_id,
          turn_id: entry.turn_id,
          model_id: entry.model_id,
          tokenizer_id: entry.tokenizer_id,
          prompt_hash: entry.prompt_hash,
          token_hash: entry.token_hash,
          prefix_len: entry.prefix_len,
          max_seq: entry.max_seq,
          layer_count: entry.layer_count,
          kv_dims: entry.kv_dims.dup,
          artifact_path: entry.artifact_path,
          artifact_sha256: entry.artifact_sha256,
          artifact_byte_size: entry.artifact_byte_size,
          state_byte_size: entry.state_byte_size,
          created_at_unix: entry.created_at_unix,
          prompt_preview: entry.prompt_preview,
          next_token_id: entry.next_token_id,
        )
      end
    end

    def prompt_hash(token_ids : Array(Int32), prompt_text : String = "") : String
      io = IO::Memory.new
      io.write("gemma4-prompt-v1\0".to_slice)
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
      io.write("gemma4-token-v1\0".to_slice)
      io.write_bytes(prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      prefix_len.times do |i|
        io.write_bytes(token_ids[i], IO::ByteFormat::LittleEndian)
      end
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def artifact_trust_metadata_valid?(entry : Entry) : Bool
      return false unless entry.runtime_id == RUNTIME_ID
      return false unless entry.prefix_len >= 0 && entry.prefix_len <= entry.max_seq
      return false unless entry.layer_count > 0 && entry.kv_dims.size == entry.layer_count
      return false unless entry.kv_dims.all? { |dim| dim > 0 }
      return false unless entry.artifact_sha256.size == 64
      return false unless entry.artifact_byte_size >= 0 && entry.state_byte_size >= 0
      return false unless entry.prompt_hash.size == 64 && entry.token_hash.size == 64
      return false if entry.next_token_id.try { |token_id| token_id < 0 }

      true
    end

    def validate_restorable_artifact!(entry : Entry) : Nil
      raise ArgumentError.new("invalid Gemma prompt-cache artifact metadata") unless artifact_trust_metadata_valid?(entry)
    end

    def short_hash(value : String) : String
      Digest::SHA256.hexdigest(value)[0, 16]
    end
  end
end
