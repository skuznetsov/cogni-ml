require "digest/sha256"
require "file_utils"
require "json"
require "./gemma4_state_snapshot"

module ML::GGUF
  # Minimal exact prompt-prefix cache for Gemma 4 resident K/V state.
  module Gemma4PromptCache
    extend self

    RUNTIME_ID                       = "cogni-ml/gemma4-state-v1"
    OUTPUT_FAST_FORWARD_RUNTIME_ID   = "cogni-ml/gemma4-output-fast-forward-v1"
    EXACT_KNOWN_SPAN_VALIDATION_KIND = "exact-known-span-v1"

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
      property artifact_validation_kind : String? = nil
      property artifact_validation_steps : Int32? = nil
      property artifact_validation_hash : String? = nil

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
                     @next_token_id : Int32? = nil,
                     @artifact_validation_kind : String? = nil,
                     @artifact_validation_steps : Int32? = nil,
                     @artifact_validation_hash : String? = nil)
      end
    end

    class OutputFastForwardEntry
      include JSON::Serializable

      property runtime_id : String
      property session_id : String
      property turn_id : String?
      property model_id : String
      property tokenizer_id : String
      property prompt_text_hash : String
      property prompt_token_hash : String
      property prompt_token_count : Int32
      property prompt_token_ids : Array(Int32)
      property output_token_hash : String
      property output_token_count : Int32
      property output_token_ids : Array(Int32)
      property full_history_hash : String
      property generated_text : String
      property generated_text_hash : String
      property artifact_validation_kind : String
      property artifact_validation_steps : Int32
      property artifact_validation_hash : String
      property artifact_prefix_len : Int32
      property artifact_token_hash : String
      property artifact_next_token_id : Int32
      property terminal_token_id : Int32? = nil
      property created_at_unix : Int64

      def initialize(@runtime_id : String,
                     @session_id : String,
                     @turn_id : String?,
                     @model_id : String,
                     @tokenizer_id : String,
                     @prompt_text_hash : String,
                     @prompt_token_hash : String,
                     @prompt_token_count : Int32,
                     @prompt_token_ids : Array(Int32),
                     @output_token_hash : String,
                     @output_token_count : Int32,
                     @output_token_ids : Array(Int32),
                     @full_history_hash : String,
                     @generated_text : String,
                     @generated_text_hash : String,
                     @artifact_validation_kind : String,
                     @artifact_validation_steps : Int32,
                     @artifact_validation_hash : String,
                     @artifact_prefix_len : Int32,
                     @artifact_token_hash : String,
                     @artifact_next_token_id : Int32,
                     @created_at_unix : Int64,
                     @terminal_token_id : Int32? = nil)
      end
    end

    record ManifestFingerprint, size : Int64, mtime_sec : Int64, mtime_nsec : Int32
    record ArtifactFingerprint, sha256 : String, expected_size : Int64, file_size : Int64, mtime_sec : Int64, mtime_nsec : Int32

    class Store
      getter root : String
      getter snapshot_cache_byte_limit : Int64
      getter snapshot_cache_requested_byte_limit : Int64
      getter snapshot_cache_min_free_bytes : Int64
      getter snapshot_cache_entry_limit : Int32
      getter snapshot_cache_bytes : Int64
      getter snapshot_cache_hits : Int64
      getter snapshot_cache_misses : Int64
      getter output_fast_forward_dir : String

      @entry_manifest_fingerprint : ManifestFingerprint?
      @entry_manifest_cache : Array(Entry)
      @entry_exact_index : Hash(Tuple(String, String, String, Int32), Array(Entry))
      @entry_prefix_index : Hash(Tuple(String, String, Int32, String), Array(Entry))
      @snapshot_cache : Hash(String, CachedSnapshot)
      @output_fast_forward_cache : Hash(String, Tuple(ManifestFingerprint, OutputFastForwardEntry))

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
                     snapshot_cache_byte_limit : Int64 = 0_i64,
                     @snapshot_cache_entry_limit : Int32 = 0,
                     @snapshot_cache_min_free_bytes : Int64 = 0_i64)
        raise ArgumentError.new("snapshot_cache_byte_limit must be non-negative") if snapshot_cache_byte_limit < 0
        raise ArgumentError.new("snapshot_cache_entry_limit must be non-negative") if @snapshot_cache_entry_limit < 0
        raise ArgumentError.new("snapshot_cache_min_free_bytes must be non-negative") if @snapshot_cache_min_free_bytes < 0

        FileUtils.mkdir_p(File.join(@root, "artifacts"))
        @snapshot_cache_requested_byte_limit = snapshot_cache_byte_limit
        @snapshot_cache_byte_limit = Gemma4PromptCache.safe_snapshot_cache_byte_limit(
          snapshot_cache_byte_limit,
          @snapshot_cache_min_free_bytes,
        )
        @manifest_path = File.join(@root, "manifest.jsonl")
        @output_fast_forward_dir = File.join(@root, "output_fast_forward")
        @entry_manifest_fingerprint = nil
        @entry_manifest_cache = [] of Entry
        @entry_exact_index = {} of Tuple(String, String, String, Int32) => Array(Entry)
        @entry_prefix_index = {} of Tuple(String, String, Int32, String) => Array(Entry)
        @snapshot_cache = {} of String => CachedSnapshot
        @output_fast_forward_cache = {} of String => Tuple(ManifestFingerprint, OutputFastForwardEntry)
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
                              next_token_id : Int32? = nil,
                              artifact_validation_kind : String? = nil,
                              artifact_validation_steps : Int32? = nil,
                              artifact_validation_hash : String? = nil) : Entry
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
          artifact_validation_kind: artifact_validation_kind,
          artifact_validation_steps: artifact_validation_steps,
          artifact_validation_hash: artifact_validation_hash,
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

      def save_output_fast_forward(session_id : String,
                                   model_id : String,
                                   tokenizer_id : String,
                                   prompt_text : String,
                                   prompt_token_ids : Array(Int32),
                                   output_token_ids : Array(Int32),
                                   generated_text : String,
                                   exact_entry : Entry,
                                   terminal_token_id : Int32? = nil,
                                   turn_id : String? = nil) : OutputFastForwardEntry
        raise ArgumentError.new("output_token_ids must not be empty") if output_token_ids.empty?
        raise ArgumentError.new("terminal_token_id must match final output token") if terminal_token_id && terminal_token_id != output_token_ids[-1]
        tmp = nil.as(String?)

        full_history_len = prompt_token_ids.size + output_token_ids.size
        full_history_hash = Gemma4PromptCache.token_hash_concat(prompt_token_ids, output_token_ids)
        artifact_steps = exact_entry.artifact_validation_steps
        artifact_hash = exact_entry.artifact_validation_hash
        artifact_next = exact_entry.next_token_id
        raise ArgumentError.new("exact_entry is not an exact-known-span artifact") unless exact_entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
        raise ArgumentError.new("exact_entry validation steps mismatch") unless artifact_steps == output_token_ids.size
        raise ArgumentError.new("exact_entry validation hash mismatch") unless artifact_hash == full_history_hash
        raise ArgumentError.new("exact_entry prefix mismatch") unless exact_entry.prefix_len == full_history_len - 1
        raise ArgumentError.new("exact_entry prefix token hash mismatch") unless exact_entry.token_hash == Gemma4PromptCache.token_hash_concat(prompt_token_ids, output_token_ids, exact_entry.prefix_len)
        raise ArgumentError.new("exact_entry next token mismatch") unless artifact_next == output_token_ids[-1]

        entry = OutputFastForwardEntry.new(
          runtime_id: OUTPUT_FAST_FORWARD_RUNTIME_ID,
          session_id: session_id,
          turn_id: turn_id,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          prompt_text_hash: Gemma4PromptCache.prompt_text_hash(prompt_text),
          prompt_token_hash: Gemma4PromptCache.token_hash(prompt_token_ids),
          prompt_token_count: prompt_token_ids.size.to_i32,
          prompt_token_ids: prompt_token_ids.dup,
          output_token_hash: Gemma4PromptCache.token_hash(output_token_ids),
          output_token_count: output_token_ids.size.to_i32,
          output_token_ids: output_token_ids.dup,
          full_history_hash: full_history_hash,
          generated_text: generated_text,
          generated_text_hash: Gemma4PromptCache.generated_text_hash(generated_text),
          artifact_validation_kind: exact_entry.artifact_validation_kind.not_nil!,
          artifact_validation_steps: artifact_steps.not_nil!,
          artifact_validation_hash: artifact_hash.not_nil!,
          artifact_prefix_len: exact_entry.prefix_len,
          artifact_token_hash: exact_entry.token_hash,
          artifact_next_token_id: artifact_next.not_nil!,
          created_at_unix: Time.utc.to_unix,
          terminal_token_id: terminal_token_id,
        )
        path = output_fast_forward_path(model_id, session_id, turn_id, entry.prompt_text_hash, output_token_ids.size)
        FileUtils.mkdir_p(File.dirname(path))
        tmp = "#{path}.tmp-#{Process.pid}-#{Random::Secure.hex(4)}"
        File.open(tmp, "w") do |file|
          entry.to_json(file)
          file << '\n'
        end
        File.rename(tmp, path)
        @output_fast_forward_cache.delete(path)
        clone_output_fast_forward_entry(entry)
      ensure
        File.delete(tmp) if tmp && File.exists?(tmp)
      end

      def lookup_output_fast_forward(model_id : String,
                                     session_id : String,
                                     prompt_text : String,
                                     output_token_count : Int32,
                                     tokenizer_id : String? = nil,
                                     turn_id : String? = nil) : OutputFastForwardEntry?
        return nil if output_token_count <= 0

        prompt_text_hash = Gemma4PromptCache.prompt_text_hash(prompt_text)
        path = output_fast_forward_path(model_id, session_id, turn_id, prompt_text_hash, output_token_count)
        unless fingerprint = manifest_fingerprint(path)
          @output_fast_forward_cache.delete(path)
          return nil
        end

        cached = @output_fast_forward_cache[path]?
        entry = if cached && cached[0] == fingerprint
                  cached[1]
                else
                  begin
                    parsed = OutputFastForwardEntry.from_json(File.read(path))
                    @output_fast_forward_cache[path] = {fingerprint, parsed}
                    parsed
                  rescue JSON::ParseException | KeyError
                    @output_fast_forward_cache.delete(path)
                    return nil
                  end
                end
        return nil unless Gemma4PromptCache.output_fast_forward_entry_valid?(
                            entry,
                            model_id,
                            session_id,
                            prompt_text,
                            output_token_count,
                            tokenizer_id: tokenizer_id,
                            turn_id: turn_id,
                          )

        clone_output_fast_forward_entry(entry)
      end

      def lookup_output_fast_forward_at_most(model_id : String,
                                             session_id : String,
                                             prompt_text : String,
                                             max_output_token_count : Int32,
                                             terminal_token_id : Int32? = nil,
                                             tokenizer_id : String? = nil,
                                             turn_id : String? = nil) : OutputFastForwardEntry?
        return nil if max_output_token_count <= 0

        if exact = lookup_output_fast_forward(model_id, session_id, prompt_text, max_output_token_count, tokenizer_id: tokenizer_id, turn_id: turn_id)
          return exact
        end
        return nil unless eos = terminal_token_id

        (max_output_token_count - 1).downto(1) do |count|
          next unless hit = lookup_output_fast_forward(model_id, session_id, prompt_text, count, tokenizer_id: tokenizer_id, turn_id: turn_id)
          return hit if hit.output_token_ids.last? == eos
        end

        nil
      end

      def lookup_terminal_output_fast_forward_at_most(model_id : String,
                                                      session_id : String,
                                                      prompt_text : String,
                                                      max_output_token_count : Int32,
                                                      tokenizer_id : String? = nil,
                                                      turn_id : String? = nil) : OutputFastForwardEntry?
        return nil if max_output_token_count <= 0

        if exact = lookup_output_fast_forward(model_id, session_id, prompt_text, max_output_token_count, tokenizer_id: tokenizer_id, turn_id: turn_id)
          return exact
        end

        (max_output_token_count - 1).downto(1) do |count|
          next unless hit = lookup_output_fast_forward(model_id, session_id, prompt_text, count, tokenizer_id: tokenizer_id, turn_id: turn_id)
          terminal = hit.terminal_token_id
          return hit if terminal && hit.output_token_ids.last? == terminal
        end

        nil
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
          artifact_validation_kind: entry.artifact_validation_kind,
          artifact_validation_steps: entry.artifact_validation_steps,
          artifact_validation_hash: entry.artifact_validation_hash,
        )
      end

      private def clone_output_fast_forward_entry(entry : OutputFastForwardEntry) : OutputFastForwardEntry
        OutputFastForwardEntry.new(
          runtime_id: entry.runtime_id,
          session_id: entry.session_id,
          turn_id: entry.turn_id,
          model_id: entry.model_id,
          tokenizer_id: entry.tokenizer_id,
          prompt_text_hash: entry.prompt_text_hash,
          prompt_token_hash: entry.prompt_token_hash,
          prompt_token_count: entry.prompt_token_count,
          prompt_token_ids: entry.prompt_token_ids.dup,
          output_token_hash: entry.output_token_hash,
          output_token_count: entry.output_token_count,
          output_token_ids: entry.output_token_ids.dup,
          full_history_hash: entry.full_history_hash,
          generated_text: entry.generated_text,
          generated_text_hash: entry.generated_text_hash,
          artifact_validation_kind: entry.artifact_validation_kind,
          artifact_validation_steps: entry.artifact_validation_steps,
          artifact_validation_hash: entry.artifact_validation_hash,
          artifact_prefix_len: entry.artifact_prefix_len,
          artifact_token_hash: entry.artifact_token_hash,
          artifact_next_token_id: entry.artifact_next_token_id,
          created_at_unix: entry.created_at_unix,
          terminal_token_id: entry.terminal_token_id,
        )
      end

      private def output_fast_forward_path(model_id : String,
                                           session_id : String,
                                           turn_id : String?,
                                           prompt_text_hash : String,
                                           output_token_count : Int32) : String
        key = Gemma4PromptCache.output_fast_forward_key(model_id, session_id, turn_id, prompt_text_hash, output_token_count)
        File.join(@output_fast_forward_dir, key[0, 2], key[2, 2], "#{key}.json")
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

    def token_hash_concat(left_ids : Array(Int32),
                          right_ids : Array(Int32),
                          prefix_len : Int32 = left_ids.size + right_ids.size) : String
      total = left_ids.size + right_ids.size
      raise ArgumentError.new("prefix_len out of range: #{prefix_len}") if prefix_len < 0 || prefix_len > total

      io = IO::Memory.new
      io.write("gemma4-token-v1\0".to_slice)
      io.write_bytes(prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      left_count = prefix_len < left_ids.size ? prefix_len : left_ids.size
      left_count.times do |i|
        io.write_bytes(left_ids[i], IO::ByteFormat::LittleEndian)
      end
      right_count = prefix_len - left_count
      right_count.times do |i|
        io.write_bytes(right_ids[i], IO::ByteFormat::LittleEndian)
      end
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def prompt_text_hash(prompt_text : String) : String
      Digest::SHA256.hexdigest("gemma4-prompt-text-v1\0#{prompt_text}")
    end

    def generated_text_hash(generated_text : String) : String
      Digest::SHA256.hexdigest("gemma4-generated-text-v1\0#{generated_text}")
    end

    def output_fast_forward_key(model_id : String,
                                session_id : String,
                                turn_id : String?,
                                prompt_text_hash : String,
                                output_token_count : Int32) : String
      io = IO::Memory.new
      io.write("gemma4-output-fast-forward-key-v1\0".to_slice)
      io.write(model_id.to_slice)
      io.write_byte(0_u8)
      io.write(session_id.to_slice)
      io.write_byte(0_u8)
      io.write((turn_id || "").to_slice)
      io.write_byte(0_u8)
      io.write(prompt_text_hash.to_slice)
      io.write_byte(0_u8)
      io.write_bytes(output_token_count.to_u32, IO::ByteFormat::LittleEndian)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def output_fast_forward_entry_valid?(entry : OutputFastForwardEntry,
                                         model_id : String,
                                         session_id : String,
                                         prompt_text : String,
                                         expected_output_tokens : Int32,
                                         tokenizer_id : String? = nil,
                                         turn_id : String? = nil) : Bool
      return false unless expected_output_tokens > 0
      return false unless entry.runtime_id == OUTPUT_FAST_FORWARD_RUNTIME_ID
      return false unless entry.model_id == model_id
      return false unless entry.session_id == session_id
      return false if tokenizer_id && entry.tokenizer_id != tokenizer_id
      return false if turn_id && entry.turn_id != turn_id
      return false unless entry.prompt_text_hash == prompt_text_hash(prompt_text)
      return false unless entry.prompt_token_count == entry.prompt_token_ids.size
      return false unless entry.prompt_token_hash == token_hash(entry.prompt_token_ids)
      return false unless entry.output_token_count == expected_output_tokens
      return false unless entry.output_token_count == entry.output_token_ids.size
      return false unless entry.output_token_hash == token_hash(entry.output_token_ids)
      return false unless entry.generated_text_hash == generated_text_hash(entry.generated_text)

      full_history_len = entry.prompt_token_ids.size + entry.output_token_ids.size
      full_hash = token_hash_concat(entry.prompt_token_ids, entry.output_token_ids)
      return false unless entry.full_history_hash == full_hash
      return false unless entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
      return false unless entry.artifact_validation_steps == entry.output_token_count
      return false unless entry.artifact_validation_hash == full_hash
      return false unless entry.artifact_prefix_len == full_history_len - 1
      return false unless entry.artifact_token_hash == token_hash_concat(entry.prompt_token_ids, entry.output_token_ids, entry.artifact_prefix_len)
      return false unless entry.artifact_next_token_id == entry.output_token_ids[-1]
      return false if entry.terminal_token_id && entry.terminal_token_id != entry.output_token_ids[-1]

      true
    end

    def exact_known_span_entry_valid?(entry : Entry,
                                      full_history : Array(Int32),
                                      emitted_steps : Int32,
                                      full_history_len : Int32 = full_history.size) : Bool
      return false if full_history_len <= 0
      return false if full_history_len > full_history.size
      return false unless emitted_steps > 0
      return false unless full_history_len >= emitted_steps
      return false unless entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
      return false unless entry.artifact_validation_steps == emitted_steps
      return false unless entry.artifact_validation_hash == token_hash(full_history, full_history_len)
      return false unless entry.prefix_len == full_history_len - 1
      return false unless entry.token_hash == token_hash(full_history, entry.prefix_len)
      return false unless entry.next_token_id == full_history[full_history_len - 1]

      true
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
      validation_fields = {
        entry.artifact_validation_kind,
        entry.artifact_validation_steps,
        entry.artifact_validation_hash,
      }.count { |field| !field.nil? }
      if validation_fields > 0
        return false unless entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
        return false unless entry.artifact_validation_steps.try { |steps| steps > 0 }
        return false unless entry.artifact_validation_hash.try { |hash| hash.size == 64 }
        return false unless entry.next_token_id
      end

      true
    end

    def validate_restorable_artifact!(entry : Entry) : Nil
      raise ArgumentError.new("invalid Gemma prompt-cache artifact metadata") unless artifact_trust_metadata_valid?(entry)
    end

    def safe_snapshot_cache_byte_limit(requested_bytes : Int64,
                                       min_free_bytes : Int64,
                                       available_bytes : Int64? = available_memory_bytes) : Int64
      clamp_snapshot_cache_byte_limit(requested_bytes, min_free_bytes, available_bytes)
    end

    def clamp_snapshot_cache_byte_limit(requested_bytes : Int64,
                                        min_free_bytes : Int64,
                                        available_bytes : Int64?) : Int64
      raise ArgumentError.new("requested_bytes must be non-negative") if requested_bytes < 0
      raise ArgumentError.new("min_free_bytes must be non-negative") if min_free_bytes < 0
      return requested_bytes if requested_bytes == 0 || min_free_bytes == 0
      return requested_bytes unless available = available_bytes
      return 0_i64 if available <= min_free_bytes

      allowed = available - min_free_bytes
      requested_bytes < allowed ? requested_bytes : allowed
    end

    def available_memory_bytes : Int64?
      available_memory_bytes_linux || available_memory_bytes_macos
    end

    private def available_memory_bytes_linux : Int64?
      path = "/proc/meminfo"
      return nil unless File.exists?(path)

      File.each_line(path) do |line|
        next unless line.starts_with?("MemAvailable:")

        parts = line.split
        return nil if parts.size < 2
        kb = parts[1].to_i64?
        return nil unless kb
        return kb * 1024_i64
      end
      nil
    end

    private def available_memory_bytes_macos : Int64?
      return nil unless {{ flag?(:darwin) }}

      output = IO::Memory.new
      status = Process.run("vm_stat", output: output, error: Process::Redirect::Close)
      return nil unless status.success?

      page_size = 4096_i64
      first_line = output.to_s.lines.first?
      if first_line
        if match = first_line.match(/page size of (\d+) bytes/)
          page_size = match[1].to_i64
        end
      end

      available_pages = 0_i64
      output.to_s.each_line do |line|
        case line
        when /^Pages free:\s+([0-9.]+)\./
          available_pages += $1.gsub(".", "").to_i64
        when /^Pages inactive:\s+([0-9.]+)\./
          available_pages += $1.gsub(".", "").to_i64
        when /^Pages speculative:\s+([0-9.]+)\./
          available_pages += $1.gsub(".", "").to_i64
        end
      end
      return nil if available_pages <= 0

      available_pages * page_size
    rescue
      nil
    end

    def short_hash(value : String) : String
      Digest::SHA256.hexdigest(value)[0, 16]
    end
  end
end
