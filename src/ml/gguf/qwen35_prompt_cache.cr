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

    RUNTIME_ID                       = "cogni-ml/qwen35-state-v1"
    RAW_ARTIFACT_CODECS              = {nil, "", "raw", "raw-fp32", "qkv-raw"}
    COMPRESSED_ARTIFACT_CODECS       = {"recurrent-bf16", "recurrent-int8"}
    SOURCE_HISTORY_RUNTIME_ID        = "cogni-ml/qwen35-source-history-v1"
    TOKENIZED_PROMPT_RUNTIME_ID      = "cogni-ml/qwen35-tokenized-prompt-v1"
    OUTPUT_FAST_FORWARD_RUNTIME_ID   = "cogni-ml/qwen35-output-fast-forward-v1"
    EXACT_KNOWN_SPAN_VALIDATION_KIND = "exact-known-span-v1"

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
      property artifact_codec : String? = nil
      property artifact_codec_block : Int32? = nil
      property artifact_validation_kind : String? = nil
      property artifact_validation_steps : Int32? = nil
      property artifact_validation_hash : String? = nil
      property next_token_id : Int32? = nil
      property next_token_logit : Float32? = nil
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
                     @token_hash : String? = nil,
                     @artifact_codec : String? = nil,
                     @artifact_codec_block : Int32? = nil,
                     @artifact_validation_kind : String? = nil,
                     @artifact_validation_steps : Int32? = nil,
                     @artifact_validation_hash : String? = nil,
                     @next_token_id : Int32? = nil,
                     @next_token_logit : Float32? = nil)
      end
    end

    class SourceHistoryEntry
      include JSON::Serializable

      property runtime_id : String
      property session_id : String
      property turn_id : String?
      property model_id : String
      property tokenizer_id : String
      property token_hash : String
      property token_count : Int32
      property token_ids : Array(Int32)
      property generated_token_count : Int32? = nil
      property generated_text : String? = nil
      property generated_text_hash : String? = nil
      property created_at_unix : Int64

      def initialize(@runtime_id : String,
                     @session_id : String,
                     @turn_id : String?,
                     @model_id : String,
                     @tokenizer_id : String,
                     @token_hash : String,
                     @token_count : Int32,
                     @token_ids : Array(Int32),
                     @created_at_unix : Int64,
                     @generated_token_count : Int32? = nil,
                     @generated_text : String? = nil,
                     @generated_text_hash : String? = nil)
      end
    end

    class TokenizedPromptEntry
      include JSON::Serializable

      property runtime_id : String
      property model_id : String
      property tokenizer_id : String
      property prompt_text_hash : String
      property token_hash : String
      property token_count : Int32
      property token_ids : Array(Int32)
      property created_at_unix : Int64

      def initialize(@runtime_id : String,
                     @model_id : String,
                     @tokenizer_id : String,
                     @prompt_text_hash : String,
                     @token_hash : String,
                     @token_count : Int32,
                     @token_ids : Array(Int32),
                     @created_at_unix : Int64)
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
                     @created_at_unix : Int64)
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
      getter source_history_manifest_path : String
      getter tokenized_prompt_manifest_path : String
      getter output_fast_forward_dir : String

      private record ArtifactFingerprint,
        sha256 : String,
        artifact_byte_size : Int64,
        file_byte_size : Int64,
        mtime_unix : Int64,
        mtime_nanosecond : Int32

      def initialize(@root : String = Qwen35PromptCache.default_root,
                     resident_state_cache_entries : Int32? = nil)
        @manifest_path = File.join(@root, "manifest.jsonl")
        @source_history_manifest_path = File.join(@root, "source_history.jsonl")
        @tokenized_prompt_manifest_path = File.join(@root, "tokenized_prompts.jsonl")
        @output_fast_forward_dir = File.join(@root, "output_fast_forward")
        @resident_state_cache_limit = resident_state_cache_entries || (ENV["QWEN35_PROMPT_CACHE_RESIDENT_STATES"]? || "0").to_i
        raise ArgumentError.new("resident_state_cache_entries must be non-negative") if @resident_state_cache_limit < 0
        @resident_state_cache = {} of String => Qwen35CPU::State
        @resident_state_cache_order = [] of String
        @validated_artifacts = {} of String => ArtifactFingerprint
      end

      def save(session_id : String,
               model_id : String,
               tokenizer_id : String,
               prompt_text : String,
               token_ids : Array(Int32),
               state : Qwen35CPU::State,
               turn_id : String? = nil,
               prompt_preview : String? = nil,
               artifact_codec : String? = nil,
               artifact_codec_block : Int32? = nil,
               artifact_validation_kind : String? = nil,
               artifact_validation_steps : Int32? = nil,
               artifact_validation_hash : String? = nil,
               next_token_id : Int32? = nil,
               next_token_logit : Float32? = nil) : Entry
        snapshot = Qwen35StateSnapshot.capture(state)
        prompt_hash = Qwen35PromptCache.prompt_hash(token_ids, prompt_text)
        token_hash = Qwen35PromptCache.token_hash(token_ids)
        artifact_path = artifact_path(model_id, tokenizer_id, prompt_hash, token_ids.size)
        artifact = Qwen35StateSnapshot.write_artifact(
          snapshot,
          artifact_path,
          artifact_codec: artifact_codec,
          artifact_codec_block: artifact_codec_block,
        )

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
          artifact_codec: artifact_codec,
          artifact_codec_block: artifact_codec_block,
          artifact_validation_kind: artifact_validation_kind,
          artifact_validation_steps: artifact_validation_steps,
          artifact_validation_hash: artifact_validation_hash,
          next_token_id: next_token_id,
          next_token_logit: next_token_logit,
        )
        append_manifest(entry)
        remember_validated_artifact(entry)
        entry
      end

      def lookup_exact(model_id : String,
                       tokenizer_id : String,
                       prompt_hash : String,
                       prefix_len : Int32) : Entry?
        entries.reverse_each.find do |entry|
          compatible?(entry, model_id, tokenizer_id, prompt_hash, prefix_len) &&
            usable_entry?(entry)
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
          next false unless usable_entry?(entry)

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

          usable_entry?(entry)
        end
        candidates.max_by? { |entry| {entry.created_at_unix, entry.prefix_len} }
      end

      def restore(entry : Entry,
                  hp : Qwen35Hparams,
                  prefer_metal : Bool = Qwen35Metal.available?,
                  reuse_state : Qwen35CPU::State? = nil) : Qwen35CPU::State
        raise ArgumentError.new("unsupported Qwen prompt-cache runtime: #{entry.runtime_id}") unless entry.runtime_id == RUNTIME_ID
        Qwen35PromptCache.validate_restorable_artifact!(entry)
        if state = restore_resident_state(entry, hp, prefer_metal, reuse_state)
          return state
        end

        codec = entry.artifact_codec.try(&.downcase)
        if prefer_metal && Qwen35Metal.available? && COMPRESSED_ARTIFACT_CODECS.includes?(codec)
          if codec == "recurrent-int8" && ENV["QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE"]? != "1"
            raise ArgumentError.new("Metal recurrent-int8 prompt-cache restore requires QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE=1")
          end

          fingerprint = artifact_fingerprint(entry)
          expected_sha256 = expected_artifact_sha256(entry, fingerprint)
          mapped = Qwen35StateSnapshot.read_artifact_encoded_mmap(
            entry.artifact_path,
            expected_sha256: expected_sha256,
            expected_codec: entry.artifact_codec,
            expected_codec_block: entry.artifact_codec_block,
          )
          begin
            encoded = mapped.encoded
            raise ArgumentError.new("prompt-cache max_seq mismatch") unless encoded.max_seq == entry.max_seq
            raise ArgumentError.new("prompt-cache layer count mismatch") unless encoded.layer_count == entry.layer_count
            state = reuse_state
            state = nil if state && state.max_seq != entry.max_seq
            state ||= Qwen35CPU::State.new(hp, max_seq: entry.max_seq)
            Qwen35StateSnapshot.restore_encoded_into(encoded, hp, state, prefer_metal: true)
            ensure_artifact_unchanged!(entry, fingerprint)
            remember_resident_state(entry, prefer_metal, state)
            remember_validated_artifact(entry, fingerprint) if expected_sha256
            return state
          ensure
            mapped.close
          end
        end

        fingerprint = artifact_fingerprint(entry)
        expected_sha256 = expected_artifact_sha256(entry, fingerprint)
        snapshot = Qwen35StateSnapshot.read_artifact(
          entry.artifact_path,
          expected_sha256: expected_sha256,
          expected_codec: entry.artifact_codec,
          expected_codec_block: entry.artifact_codec_block,
        )
        ensure_artifact_unchanged!(entry, fingerprint)
        remember_validated_artifact(entry, fingerprint) if expected_sha256
        raise ArgumentError.new("prompt-cache max_seq mismatch") unless snapshot.max_seq == entry.max_seq
        raise ArgumentError.new("prompt-cache layer count mismatch") unless snapshot.layer_count == entry.layer_count
        if state = reuse_state
          Qwen35StateSnapshot.restore_into(snapshot, hp, state, prefer_metal: prefer_metal)
          remember_resident_state(entry, prefer_metal, state)
          state
        else
          state = Qwen35StateSnapshot.restore(snapshot, hp, prefer_metal: prefer_metal)
          remember_resident_state(entry, prefer_metal, state)
          state
        end
      end

      def restore_and_replay_suffix(entry : Entry,
                                    weights : Qwen35Weights,
                                    token_ids : Array(Int32),
                                    prefer_metal : Bool = Qwen35Metal.available?,
                                    reuse_state : Qwen35CPU::State? = nil) : ReplayResult
        raise ArgumentError.new("cache prefix longer than prompt: prefix=#{entry.prefix_len}, prompt=#{token_ids.size}") if entry.prefix_len > token_ids.size

        reusable = reuse_state
        reusable = nil if reusable && reusable.max_seq != entry.max_seq
        state = restore(entry, weights.hparams, prefer_metal: prefer_metal, reuse_state: reusable)
        next_token_id = nil.as(Int32?)
        next_token_logit = nil.as(Float32?)

        suffix_start = entry.prefix_len
        final_pos = token_ids.size - 1
        if suffix_start <= final_pos
          top, logit = Qwen35CPU.prefill_tokens_top1(weights, token_ids[suffix_start..final_pos], suffix_start.to_i32, state)
          next_token_id = top
          next_token_logit = logit
        elsif suffix_start == token_ids.size
          next_token_id = entry.next_token_id
          next_token_logit = entry.next_token_logit
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

      def save_source_history(session_id : String,
                              model_id : String,
                              tokenizer_id : String,
                              token_ids : Array(Int32),
                              generated_token_count : Int32? = nil,
                              generated_text : String? = nil,
                              turn_id : String? = nil) : SourceHistoryEntry
        generated_text_hash = generated_text ? Qwen35PromptCache.generated_text_hash(generated_text) : nil
        entry = SourceHistoryEntry.new(
          runtime_id: SOURCE_HISTORY_RUNTIME_ID,
          session_id: session_id,
          turn_id: turn_id,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          token_hash: Qwen35PromptCache.token_hash(token_ids),
          token_count: token_ids.size.to_i32,
          token_ids: token_ids.dup,
          created_at_unix: Time.utc.to_unix,
          generated_token_count: generated_token_count,
          generated_text: generated_text,
          generated_text_hash: generated_text_hash,
        )
        FileUtils.mkdir_p(@root)
        File.open(@source_history_manifest_path, "a") do |file|
          entry.to_json(file)
          file << '\n'
        end
        entry
      end

      def lookup_source_history(session_id : String,
                                model_id : String,
                                tokenizer_id : String,
                                turn_id : String? = nil) : SourceHistoryEntry?
        source_history_entries.select do |entry|
          next false unless entry.runtime_id == SOURCE_HISTORY_RUNTIME_ID
          next false unless entry.session_id == session_id
          next false unless entry.model_id == model_id
          next false unless entry.tokenizer_id == tokenizer_id
          next false if turn_id && entry.turn_id != turn_id

          entry.token_count == entry.token_ids.size &&
            entry.token_hash == Qwen35PromptCache.token_hash(entry.token_ids)
        end.max_by? { |entry| {entry.created_at_unix, entry.token_count} }
      end

      def source_history_entries : Array(SourceHistoryEntry)
        return [] of SourceHistoryEntry unless File.exists?(@source_history_manifest_path)

        out = [] of SourceHistoryEntry
        File.each_line(@source_history_manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            out << SourceHistoryEntry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt source-history line must not produce a replay source.
          end
        end
        out
      end

      def save_tokenized_prompt(model_id : String,
                                tokenizer_id : String,
                                prompt_text : String,
                                token_ids : Array(Int32)) : TokenizedPromptEntry
        entry = TokenizedPromptEntry.new(
          runtime_id: TOKENIZED_PROMPT_RUNTIME_ID,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          prompt_text_hash: Qwen35PromptCache.prompt_text_hash(prompt_text),
          token_hash: Qwen35PromptCache.token_hash(token_ids),
          token_count: token_ids.size.to_i32,
          token_ids: token_ids.dup,
          created_at_unix: Time.utc.to_unix,
        )
        FileUtils.mkdir_p(@root)
        File.open(@tokenized_prompt_manifest_path, "a") do |file|
          entry.to_json(file)
          file << '\n'
        end
        entry
      end

      def lookup_tokenized_prompt(model_id : String,
                                  tokenizer_id : String,
                                  prompt_text : String) : TokenizedPromptEntry?
        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
        tokenized_prompt_entries.select do |entry|
          next false unless entry.runtime_id == TOKENIZED_PROMPT_RUNTIME_ID
          next false unless entry.model_id == model_id
          next false unless entry.tokenizer_id == tokenizer_id
          next false unless entry.prompt_text_hash == prompt_text_hash

          entry.token_count == entry.token_ids.size &&
            entry.token_hash == Qwen35PromptCache.token_hash(entry.token_ids)
        end.max_by? { |entry| {entry.created_at_unix, entry.token_count} }
      end

      def lookup_tokenized_prompt_for_model(model_id : String,
                                            prompt_text : String) : TokenizedPromptEntry?
        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
        tokenized_prompt_entries.select do |entry|
          next false unless entry.runtime_id == TOKENIZED_PROMPT_RUNTIME_ID
          next false unless entry.model_id == model_id
          next false unless entry.prompt_text_hash == prompt_text_hash

          entry.token_count == entry.token_ids.size &&
            entry.token_hash == Qwen35PromptCache.token_hash(entry.token_ids)
        end.max_by? { |entry| {entry.created_at_unix, entry.token_count} }
      end

      def tokenized_prompt_entries : Array(TokenizedPromptEntry)
        return [] of TokenizedPromptEntry unless File.exists?(@tokenized_prompt_manifest_path)

        out = [] of TokenizedPromptEntry
        File.each_line(@tokenized_prompt_manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            out << TokenizedPromptEntry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt tokenized-prompt line must not produce a token cache hit.
          end
        end
        out
      end

      def save_output_fast_forward(session_id : String,
                                   model_id : String,
                                   tokenizer_id : String,
                                   prompt_text : String,
                                   prompt_token_ids : Array(Int32),
                                   output_token_ids : Array(Int32),
                                   generated_text : String,
                                   exact_entry : Entry,
                                   turn_id : String? = nil) : OutputFastForwardEntry
        raise ArgumentError.new("output_token_ids must not be empty") if output_token_ids.empty?
        tmp = nil.as(String?)

        full_history = prompt_token_ids.dup
        full_history.concat(output_token_ids)
        full_history_hash = Qwen35PromptCache.token_hash(full_history)
        artifact_steps = exact_entry.artifact_validation_steps
        artifact_hash = exact_entry.artifact_validation_hash
        artifact_next = exact_entry.next_token_id
        raise ArgumentError.new("exact_entry is not an exact-known-span artifact") unless exact_entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
        raise ArgumentError.new("exact_entry validation steps mismatch") unless artifact_steps == output_token_ids.size
        raise ArgumentError.new("exact_entry validation hash mismatch") unless artifact_hash == full_history_hash
        raise ArgumentError.new("exact_entry prefix mismatch") unless exact_entry.prefix_len == full_history.size - 1
        raise ArgumentError.new("exact_entry prefix token hash mismatch") unless exact_entry.token_hash == Qwen35PromptCache.token_hash(full_history, exact_entry.prefix_len)
        raise ArgumentError.new("exact_entry next token mismatch") unless artifact_next == output_token_ids[-1]

        entry = OutputFastForwardEntry.new(
          runtime_id: OUTPUT_FAST_FORWARD_RUNTIME_ID,
          session_id: session_id,
          turn_id: turn_id,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          prompt_text_hash: Qwen35PromptCache.prompt_text_hash(prompt_text),
          prompt_token_hash: Qwen35PromptCache.token_hash(prompt_token_ids),
          prompt_token_count: prompt_token_ids.size.to_i32,
          prompt_token_ids: prompt_token_ids.dup,
          output_token_hash: Qwen35PromptCache.token_hash(output_token_ids),
          output_token_count: output_token_ids.size.to_i32,
          output_token_ids: output_token_ids.dup,
          full_history_hash: full_history_hash,
          generated_text: generated_text,
          generated_text_hash: Qwen35PromptCache.generated_text_hash(generated_text),
          artifact_validation_kind: exact_entry.artifact_validation_kind.not_nil!,
          artifact_validation_steps: artifact_steps.not_nil!,
          artifact_validation_hash: artifact_hash.not_nil!,
          artifact_prefix_len: exact_entry.prefix_len,
          artifact_token_hash: exact_entry.token_hash.not_nil!,
          artifact_next_token_id: artifact_next.not_nil!,
          created_at_unix: Time.utc.to_unix,
        )
        path = output_fast_forward_path(model_id, session_id, turn_id, entry.prompt_text_hash, output_token_ids.size)
        FileUtils.mkdir_p(File.dirname(path))
        tmp = "#{path}.tmp-#{Process.pid}-#{Random::Secure.hex(4)}"
        File.open(tmp, "w") do |file|
          entry.to_json(file)
          file << '\n'
        end
        File.rename(tmp, path)
        entry
      ensure
        File.delete(tmp) if tmp && File.exists?(tmp)
      end

      def lookup_output_fast_forward(model_id : String,
                                     session_id : String,
                                     prompt_text : String,
                                     output_token_count : Int32,
                                     turn_id : String? = nil) : OutputFastForwardEntry?
        return nil if output_token_count <= 0

        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
        path = output_fast_forward_path(model_id, session_id, turn_id, prompt_text_hash, output_token_count)
        return nil unless File.exists?(path)

        begin
          entry = OutputFastForwardEntry.from_json(File.read(path))
        rescue JSON::ParseException | KeyError
          return nil
        end
        return nil unless Qwen35PromptCache.output_fast_forward_entry_valid?(
                            entry,
                            model_id,
                            session_id,
                            prompt_text,
                            output_token_count,
                            turn_id: turn_id,
                          )

        entry
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

      private def usable_entry?(entry : Entry) : Bool
        File.exists?(entry.artifact_path) &&
          Qwen35PromptCache.artifact_trust_metadata_valid?(entry)
      end

      private def artifact_path(model_id : String,
                                tokenizer_id : String,
                                prompt_hash : String,
                                prefix_len : Int32) : String
        bucket = Qwen35PromptCache.short_hash("#{model_id}\0#{tokenizer_id}")
        File.join(@root, "artifacts", bucket, "#{prefix_len}-#{prompt_hash.downcase}.qkv")
      end

      private def restore_resident_state(entry : Entry,
                                         hp : Qwen35Hparams,
                                         prefer_metal : Bool,
                                         reuse_state : Qwen35CPU::State?) : Qwen35CPU::State?
        return nil if @resident_state_cache_limit <= 0
        cached = @resident_state_cache[resident_state_cache_key(entry, prefer_metal)]?
        return nil unless cached

        touch_resident_state(entry, prefer_metal)
        if state = reuse_state
          return nil unless state.max_seq == cached.max_seq

          if prefer_metal && Qwen35Metal.available?
            Qwen35CPU.prepare_state_metal!(state, hp, clear: false)
            Qwen35CPU.copy_state_metal_used!(state, cached, hp, used_tokens: entry.prefix_len)
          else
            state.copy_from!(cached)
          end
          state
        else
          cached.fork
        end
      end

      private def remember_resident_state(entry : Entry, prefer_metal : Bool, state : Qwen35CPU::State) : Nil
        return if @resident_state_cache_limit <= 0

        key = resident_state_cache_key(entry, prefer_metal)
        @resident_state_cache[key] = state.fork
        @resident_state_cache_order.delete(key)
        @resident_state_cache_order << key
        while @resident_state_cache_order.size > @resident_state_cache_limit
          evicted = @resident_state_cache_order.shift
          @resident_state_cache.delete(evicted)
        end
      end

      private def touch_resident_state(entry : Entry, prefer_metal : Bool) : Nil
        key = resident_state_cache_key(entry, prefer_metal)
        @resident_state_cache_order.delete(key)
        @resident_state_cache_order << key
      end

      private def resident_state_cache_key(entry : Entry, prefer_metal : Bool) : String
        backend = prefer_metal ? "metal" : "cpu"
        "#{backend}:#{entry.artifact_sha256}:#{entry.max_seq}:#{entry.layer_count}"
      end

      private def artifact_validation_key(entry : Entry) : String
        "#{entry.artifact_path}:#{entry.artifact_sha256.downcase}:#{entry.artifact_byte_size}"
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

      private def expected_artifact_sha256(entry : Entry, fingerprint : ArtifactFingerprint) : String?
        cached = @validated_artifacts[artifact_validation_key(entry)]?
        cached == fingerprint ? nil : entry.artifact_sha256
      end

      private def ensure_artifact_unchanged!(entry : Entry, fingerprint : ArtifactFingerprint) : Nil
        return if artifact_fingerprint(entry) == fingerprint

        raise ArgumentError.new("prompt-cache artifact changed during restore")
      end

      private def remember_validated_artifact(entry : Entry) : Nil
        remember_validated_artifact(entry, artifact_fingerprint(entry))
      end

      private def remember_validated_artifact(entry : Entry, fingerprint : ArtifactFingerprint) : Nil
        @validated_artifacts[artifact_validation_key(entry)] = fingerprint
      end

      private def output_fast_forward_path(model_id : String,
                                           session_id : String,
                                           turn_id : String?,
                                           prompt_text_hash : String,
                                           output_token_count : Int32) : String
        key = Qwen35PromptCache.output_fast_forward_key(model_id, session_id, turn_id, prompt_text_hash, output_token_count)
        File.join(@output_fast_forward_dir, key[0, 2], key[2, 2], "#{key}.json")
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

    def prompt_text_hash(prompt_text : String) : String
      Digest::SHA256.hexdigest("qwen35-prompt-text-v1\0#{prompt_text}")
    end

    def generated_text_hash(generated_text : String) : String
      Digest::SHA256.hexdigest("qwen35-generated-text-v1\0#{generated_text}")
    end

    def output_fast_forward_key(model_id : String,
                                session_id : String,
                                turn_id : String?,
                                prompt_text_hash : String,
                                output_token_count : Int32) : String
      io = IO::Memory.new
      io.write("qwen35-output-fast-forward-key-v1\0".to_slice)
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

    def generated_text_metadata_valid?(entry : SourceHistoryEntry,
                                       expected_generated_tokens : Int32) : Bool
      return false unless expected_generated_tokens > 0
      return false unless entry.generated_token_count == expected_generated_tokens
      text = entry.generated_text
      hash = entry.generated_text_hash
      return false unless text && hash

      generated_text_hash(text) == hash
    end

    def output_fast_forward_entry_valid?(entry : OutputFastForwardEntry,
                                         model_id : String,
                                         session_id : String,
                                         prompt_text : String,
                                         expected_output_tokens : Int32,
                                         turn_id : String? = nil) : Bool
      return false unless expected_output_tokens > 0
      return false unless entry.runtime_id == OUTPUT_FAST_FORWARD_RUNTIME_ID
      return false unless entry.model_id == model_id
      return false unless entry.session_id == session_id
      return false if turn_id && entry.turn_id != turn_id
      return false unless entry.prompt_text_hash == prompt_text_hash(prompt_text)
      return false unless entry.prompt_token_count == entry.prompt_token_ids.size
      return false unless entry.prompt_token_hash == token_hash(entry.prompt_token_ids)
      return false unless entry.output_token_count == expected_output_tokens
      return false unless entry.output_token_count == entry.output_token_ids.size
      return false unless entry.output_token_hash == token_hash(entry.output_token_ids)
      return false unless entry.generated_text_hash == generated_text_hash(entry.generated_text)

      full_history = entry.prompt_token_ids.dup
      full_history.concat(entry.output_token_ids)
      full_hash = token_hash(full_history)
      return false unless entry.full_history_hash == full_hash
      return false unless entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
      return false unless entry.artifact_validation_steps == entry.output_token_count
      return false unless entry.artifact_validation_hash == full_hash
      return false unless entry.artifact_prefix_len == full_history.size - 1
      return false unless entry.artifact_token_hash == token_hash(full_history, entry.artifact_prefix_len)
      return false unless entry.artifact_next_token_id == entry.output_token_ids[-1]

      true
    end

    def source_history_prefix_match?(source_history : Array(Int32),
                                     prefix_ids : Array(Int32),
                                     replay_start : Int32) : Bool
      return false if replay_start < prefix_ids.size
      source_prefix_start = replay_start - prefix_ids.size
      return false if source_prefix_start < 0
      return false if source_prefix_start + prefix_ids.size > source_history.size

      source_history[source_prefix_start, prefix_ids.size] == prefix_ids
    end

    def exact_known_span_entry_valid?(entry : Entry,
                                      full_history : Array(Int32),
                                      emitted_steps : Int32) : Bool
      return false if full_history.empty?
      return false unless emitted_steps > 0
      return false unless full_history.size >= emitted_steps
      return false unless entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
      return false unless entry.artifact_validation_steps == emitted_steps
      return false unless entry.artifact_validation_hash == token_hash(full_history)
      return false unless entry.prefix_len == full_history.size - 1
      return false unless entry.token_hash == token_hash(full_history, entry.prefix_len)
      return false unless entry.next_token_id == full_history[-1]

      true
    end

    def artifact_trust_metadata_valid?(entry : Entry) : Bool
      codec = normalized_artifact_codec(entry)
      return true if RAW_ARTIFACT_CODECS.includes?(codec)
      return false unless COMPRESSED_ARTIFACT_CODECS.includes?(codec)
      return false unless entry.artifact_validation_kind.try(&.empty?) == false
      return false unless entry.artifact_validation_hash.try(&.empty?) == false
      steps = entry.artifact_validation_steps
      return false unless steps && steps > 0

      if codec == "recurrent-int8"
        block = entry.artifact_codec_block
        return false unless block && block > 0
      end
      true
    end

    def validate_restorable_artifact!(entry : Entry) : Nil
      raise ArgumentError.new("prompt-cache artifact has invalid codec validation metadata") unless artifact_trust_metadata_valid?(entry)
      codec = normalized_artifact_codec(entry)
      return if RAW_ARTIFACT_CODECS.includes?(codec)
      return if COMPRESSED_ARTIFACT_CODECS.includes?(codec)

      raise ArgumentError.new("unsupported prompt-cache artifact codec: #{codec.inspect}")
    end

    private def normalized_artifact_codec(entry : Entry) : String?
      entry.artifact_codec.try(&.downcase)
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
          artifact_codec     text,
          artifact_codec_block integer CHECK (artifact_codec_block IS NULL OR artifact_codec_block > 0),
          artifact_validation_kind text,
          artifact_validation_steps integer CHECK (artifact_validation_steps IS NULL OR artifact_validation_steps >= 0),
          artifact_validation_hash text,
          next_token_id      integer,
          next_token_logit   real,
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
          artifact_codec, artifact_codec_block, artifact_validation_kind,
          artifact_validation_steps, artifact_validation_hash,
          next_token_id, next_token_logit,
          created_at_unix, prompt_preview
      ) VALUES (
          $1, $2, $3, $4, $5,
          $6, $7, $8, $9, $10,
          $11, $12, $13, $14,
          $15, $16, $17,
          $18, $19,
          $20, $21,
          $22, $23
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
          artifact_codec = EXCLUDED.artifact_codec,
          artifact_codec_block = EXCLUDED.artifact_codec_block,
          artifact_validation_kind = EXCLUDED.artifact_validation_kind,
          artifact_validation_steps = EXCLUDED.artifact_validation_steps,
          artifact_validation_hash = EXCLUDED.artifact_validation_hash,
          next_token_id = EXCLUDED.next_token_id,
          next_token_logit = EXCLUDED.next_token_logit,
          created_at_unix = EXCLUDED.created_at_unix,
          prompt_preview = EXCLUDED.prompt_preview;
      SQL
    end

    def pg_insert_values(entry : Entry) : Array(String | Int32 | Int64 | Float32 | Nil)
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
        entry.artifact_codec,
        entry.artifact_codec_block,
        entry.artifact_validation_kind,
        entry.artifact_validation_steps,
        entry.artifact_validation_hash,
        entry.next_token_id,
        entry.next_token_logit,
        entry.created_at_unix,
        entry.prompt_preview,
      ] of String | Int32 | Int64 | Float32 | Nil
    end

    private def pg_identifier(name : String) : String
      raise ArgumentError.new("unsafe PostgreSQL identifier: #{name.inspect}") unless name.matches?(/\A[a-zA-Z_][a-zA-Z0-9_]*\z/)

      name
    end
  end
end
