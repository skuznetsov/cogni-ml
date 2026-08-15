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
    PROPOSAL_ROUTE_RUNTIME_ID        = "cogni-ml/qwen35-proposal-route-v1"
    EXACT_KNOWN_SPAN_VALIDATION_KIND = "exact-known-span-v1"
    PROPOSAL_ROUTE_BASELINE          = "baseline"
    PROPOSAL_ROUTE_PCA_UPDOWN        = "pca_updown"

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
      property artifact_live_kv_tokens : Int32? = nil
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
                     @artifact_live_kv_tokens : Int32? = nil,
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

    class ProposalRouteEntry
      include JSON::Serializable

      property runtime_id : String
      property model_id : String
      property tokenizer_id : String
      property prompt_text_hash : String
      property prompt_token_hash : String
      property prompt_token_count : Int32
      property route : String
      property route_rank : Int32? = nil
      property route_layers : Array(Int32)
      property route_key_hash : String? = nil
      property route_key_preview : String? = nil
      property trigger : String? = nil
      property evidence : String? = nil
      property created_at_unix : Int64

      def initialize(@runtime_id : String,
                     @model_id : String,
                     @tokenizer_id : String,
                     @prompt_text_hash : String,
                     @prompt_token_hash : String,
                     @prompt_token_count : Int32,
                     @route : String,
                     @route_layers : Array(Int32),
                     @created_at_unix : Int64,
                     @route_rank : Int32? = nil,
                     @route_key_hash : String? = nil,
                     @route_key_preview : String? = nil,
                     @trigger : String? = nil,
                     @evidence : String? = nil)
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
      getter proposal_route_manifest_path : String
      getter output_fast_forward_dir : String

      private record ArtifactFingerprint,
        sha256 : String,
        artifact_byte_size : Int64,
        file_byte_size : Int64,
        mtime_unix : Int64,
        mtime_nanosecond : Int32

      private record ManifestFingerprint,
        file_byte_size : Int64,
        mtime_unix : Int64,
        mtime_nanosecond : Int32

      def initialize(@root : String = Qwen35PromptCache.default_root,
                     resident_state_cache_entries : Int32? = nil)
        @manifest_path = File.join(@root, "manifest.jsonl")
        @source_history_manifest_path = File.join(@root, "source_history.jsonl")
        @tokenized_prompt_manifest_path = File.join(@root, "tokenized_prompts.jsonl")
        @proposal_route_manifest_path = File.join(@root, "proposal_routes.jsonl")
        @output_fast_forward_dir = File.join(@root, "output_fast_forward")
        @resident_state_cache_limit = resident_state_cache_entries || (ENV["QWEN35_PROMPT_CACHE_RESIDENT_STATES"]? || "0").to_i
        raise ArgumentError.new("resident_state_cache_entries must be non-negative") if @resident_state_cache_limit < 0
        @resident_state_cache = {} of String => Qwen35CPU::State
        @resident_state_cache_order = [] of String
        @validated_artifacts = {} of String => ArtifactFingerprint
        @entry_manifest_fingerprint = nil.as(ManifestFingerprint?)
        @entry_manifest_cache = [] of Entry
        @source_history_manifest_fingerprint = nil.as(ManifestFingerprint?)
        @source_history_manifest_cache = [] of SourceHistoryEntry
        @tokenized_prompt_manifest_fingerprint = nil.as(ManifestFingerprint?)
        @tokenized_prompt_manifest_cache = [] of TokenizedPromptEntry
        @proposal_route_manifest_fingerprint = nil.as(ManifestFingerprint?)
        @proposal_route_manifest_cache = [] of ProposalRouteEntry
        @entry_index_fingerprint = nil.as(ManifestFingerprint?)
        @entry_exact_index = {} of Tuple(String, String, String, Int32) => Array(Entry)
        @entry_prefix_index = {} of Tuple(String, String, Int32, String) => Array(Entry)
        @source_history_index_fingerprint = nil.as(ManifestFingerprint?)
        @source_history_base_index = {} of Tuple(String, String, String) => Array(SourceHistoryEntry)
        @source_history_turn_index = {} of Tuple(String, String, String, String) => Array(SourceHistoryEntry)
        @tokenized_prompt_index_fingerprint = nil.as(ManifestFingerprint?)
        @tokenized_prompt_exact_index = {} of Tuple(String, String, String) => Array(TokenizedPromptEntry)
        @tokenized_prompt_model_index = {} of Tuple(String, String) => Array(TokenizedPromptEntry)
        @proposal_route_index_fingerprint = nil.as(ManifestFingerprint?)
        @proposal_route_prompt_index = {} of Tuple(String, String, String, String) => Array(ProposalRouteEntry)
        @proposal_route_key_index = {} of Tuple(String, String, String) => Array(ProposalRouteEntry)
        @output_fast_forward_cache = {} of String => Tuple(ManifestFingerprint, OutputFastForwardEntry)
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
               artifact_live_kv_tokens : Int32? = nil,
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
          artifact_live_kv_tokens: artifact_live_kv_tokens,
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
          artifact_live_kv_tokens: artifact_live_kv_tokens,
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
        ensure_entry_indices
        key = {model_id, tokenizer_id, prompt_hash.downcase, prefix_len}
        candidates = @entry_exact_index[key]?
        return nil unless candidates

        if hit = candidates.reverse_each.find { |entry| usable_entry?(entry) }
          clone_entry(hit)
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
                                max_prefix_len : Int32 = token_ids.size,
                                required_max_seq : Int32? = nil,
                                maximum_max_seq : Int32? = nil) : Entry?
        raise ArgumentError.new("min_prefix_len must be non-negative") if min_prefix_len < 0
        raise ArgumentError.new("max_prefix_len out of range: #{max_prefix_len}") if max_prefix_len < 0 || max_prefix_len > token_ids.size
        if required = required_max_seq
          raise ArgumentError.new("required_max_seq must be positive") unless required > 0
        end
        if maximum = maximum_max_seq
          raise ArgumentError.new("maximum_max_seq must be positive") unless maximum > 0
        end
        if required = required_max_seq
          if maximum = maximum_max_seq
            if required > maximum
              raise ArgumentError.new("required_max_seq #{required} exceeds maximum_max_seq #{maximum}")
            end
          end
        end

        ensure_entry_indices
        upper = max_prefix_len < token_ids.size ? max_prefix_len : token_ids.size
        upper.downto(min_prefix_len) do |prefix_len|
          expected = Qwen35PromptCache.token_hash(token_ids, prefix_len)
          candidates = @entry_prefix_index[{model_id, tokenizer_id, prefix_len, expected}]?
          next unless candidates

          valid_candidates = candidates.select do |entry|
            usable_entry?(entry) &&
              (!required_max_seq || entry.max_seq >= required_max_seq.not_nil!) &&
              (!maximum_max_seq || entry.max_seq <= maximum_max_seq.not_nil!)
          end
          if hit = valid_candidates.max_by? { |entry| entry.created_at_unix }
            return clone_entry(hit)
          end
        end
        nil
      end

      def lookup_token_prefix(model_id : String,
                              tokenizer_id : String,
                              token_ids : Array(Int32),
                              prefix_len : Int32) : Entry?
        raise ArgumentError.new("prefix_len out of range: #{prefix_len}") if prefix_len < 0 || prefix_len > token_ids.size

        ensure_entry_indices
        expected = Qwen35PromptCache.token_hash(token_ids, prefix_len)
        candidates = @entry_prefix_index[{model_id, tokenizer_id, prefix_len, expected}]?
        return nil unless candidates

        valid_candidates = candidates.select { |entry| usable_entry?(entry) }
        if hit = valid_candidates.max_by? { |entry| entry.created_at_unix }
          clone_entry(hit)
        end
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
        clone_entries(manifest_entries)
      end

      private def manifest_entries : Array(Entry)
        fingerprint = manifest_fingerprint(@manifest_path)
        unless fingerprint
          @entry_manifest_fingerprint = nil
          @entry_manifest_cache = [] of Entry
          return [] of Entry
        end
        if @entry_manifest_fingerprint == fingerprint
          return @entry_manifest_cache
        end

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
        parsed
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
        @source_history_manifest_fingerprint = nil
        entry
      end

      def lookup_source_history(session_id : String,
                                model_id : String,
                                tokenizer_id : String,
                                turn_id : String? = nil) : SourceHistoryEntry?
        ensure_source_history_indices
        candidates = if turn = turn_id
                       @source_history_turn_index[{session_id, model_id, tokenizer_id, turn}]?
                     else
                       @source_history_base_index[{session_id, model_id, tokenizer_id}]?
                     end
        return nil unless candidates

        valid_candidates = candidates.select { |entry| source_history_entry_valid?(entry) }
        if hit = valid_candidates.max_by? { |entry| {entry.created_at_unix, entry.token_count} }
          clone_source_history_entry(hit)
        end
      end

      def source_history_entries : Array(SourceHistoryEntry)
        clone_source_history_entries(source_history_manifest_entries)
      end

      private def source_history_manifest_entries : Array(SourceHistoryEntry)
        fingerprint = manifest_fingerprint(@source_history_manifest_path)
        unless fingerprint
          @source_history_manifest_fingerprint = nil
          @source_history_manifest_cache = [] of SourceHistoryEntry
          return [] of SourceHistoryEntry
        end
        if @source_history_manifest_fingerprint == fingerprint
          return @source_history_manifest_cache
        end

        parsed = [] of SourceHistoryEntry
        File.each_line(@source_history_manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            parsed << SourceHistoryEntry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt source-history line must not produce a replay source.
          end
        end
        @source_history_manifest_fingerprint = fingerprint
        @source_history_manifest_cache = parsed
        parsed
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
        @tokenized_prompt_manifest_fingerprint = nil
        entry
      end

      def lookup_tokenized_prompt(model_id : String,
                                  tokenizer_id : String,
                                  prompt_text : String) : TokenizedPromptEntry?
        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
        ensure_tokenized_prompt_indices
        candidates = @tokenized_prompt_exact_index[{model_id, tokenizer_id, prompt_text_hash}]?
        return nil unless candidates

        valid_candidates = candidates.select { |entry| tokenized_prompt_entry_valid?(entry) }
        if hit = valid_candidates.max_by? { |entry| {entry.created_at_unix, entry.token_count} }
          clone_tokenized_prompt_entry(hit)
        end
      end

      def lookup_tokenized_prompt_for_model(model_id : String,
                                            prompt_text : String) : TokenizedPromptEntry?
        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
        ensure_tokenized_prompt_indices
        candidates = @tokenized_prompt_model_index[{model_id, prompt_text_hash}]?
        return nil unless candidates

        valid_candidates = candidates.select { |entry| tokenized_prompt_entry_valid?(entry) }
        if hit = valid_candidates.max_by? { |entry| {entry.created_at_unix, entry.token_count} }
          clone_tokenized_prompt_entry(hit)
        end
      end

      def tokenized_prompt_entries : Array(TokenizedPromptEntry)
        clone_tokenized_prompt_entries(tokenized_prompt_manifest_entries)
      end

      def save_proposal_route(model_id : String,
                              tokenizer_id : String,
                              prompt_text : String,
                              token_ids : Array(Int32),
                              route : String,
                              route_rank : Int32? = nil,
                              route_layers : Array(Int32) = [] of Int32,
                              route_key : String? = nil,
                              trigger : String? = nil,
                              evidence : String? = nil) : ProposalRouteEntry
        validate_proposal_route!(route, route_rank, route_layers)
        entry = ProposalRouteEntry.new(
          runtime_id: PROPOSAL_ROUTE_RUNTIME_ID,
          model_id: model_id,
          tokenizer_id: tokenizer_id,
          prompt_text_hash: Qwen35PromptCache.prompt_text_hash(prompt_text),
          prompt_token_hash: Qwen35PromptCache.token_hash(token_ids),
          prompt_token_count: token_ids.size.to_i32,
          route: route,
          route_rank: route_rank,
          route_layers: route_layers.uniq.sort,
          route_key_hash: route_key.try { |key| Qwen35PromptCache.proposal_route_key_hash(key) },
          route_key_preview: route_key.try { |key| key[0, Math.min(key.size, 80)] },
          trigger: trigger,
          evidence: evidence,
          created_at_unix: Time.utc.to_unix,
        )
        FileUtils.mkdir_p(@root)
        File.open(@proposal_route_manifest_path, "a") do |file|
          entry.to_json(file)
          file << '\n'
        end
        @proposal_route_manifest_fingerprint = nil
        clone_proposal_route_entry(entry)
      end

      def lookup_proposal_route(model_id : String,
                                tokenizer_id : String,
                                prompt_text : String,
                                token_ids : Array(Int32)) : ProposalRouteEntry?
        ensure_proposal_route_indices
        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
        token_hash = Qwen35PromptCache.token_hash(token_ids)
        candidates = @proposal_route_prompt_index[{model_id, tokenizer_id, prompt_text_hash, token_hash}]?
        return nil unless candidates

        valid_candidates = candidates.select do |entry|
          Qwen35PromptCache.proposal_route_entry_valid?(entry, model_id, tokenizer_id, prompt_text, token_ids)
        end
        if hit = valid_candidates.max_by? { |entry| entry.created_at_unix }
          clone_proposal_route_entry(hit)
        end
      end

      def lookup_proposal_route_key(model_id : String,
                                    tokenizer_id : String,
                                    route_key : String) : ProposalRouteEntry?
        ensure_proposal_route_indices
        route_key_hash = Qwen35PromptCache.proposal_route_key_hash(route_key)
        candidates = @proposal_route_key_index[{model_id, tokenizer_id, route_key_hash}]?
        return nil unless candidates

        valid_candidates = candidates.select do |entry|
          Qwen35PromptCache.proposal_route_entry_valid?(entry, model_id, tokenizer_id) &&
            entry.route_key_hash == route_key_hash
        end
        if hit = valid_candidates.max_by? { |entry| entry.created_at_unix }
          clone_proposal_route_entry(hit)
        end
      end

      def proposal_route_entries : Array(ProposalRouteEntry)
        clone_proposal_route_entries(proposal_route_manifest_entries)
      end

      private def tokenized_prompt_manifest_entries : Array(TokenizedPromptEntry)
        fingerprint = manifest_fingerprint(@tokenized_prompt_manifest_path)
        unless fingerprint
          @tokenized_prompt_manifest_fingerprint = nil
          @tokenized_prompt_manifest_cache = [] of TokenizedPromptEntry
          return [] of TokenizedPromptEntry
        end
        if @tokenized_prompt_manifest_fingerprint == fingerprint
          return @tokenized_prompt_manifest_cache
        end

        parsed = [] of TokenizedPromptEntry
        File.each_line(@tokenized_prompt_manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            parsed << TokenizedPromptEntry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt tokenized-prompt line must not produce a token cache hit.
          end
        end
        @tokenized_prompt_manifest_fingerprint = fingerprint
        @tokenized_prompt_manifest_cache = parsed
        parsed
      end

      private def proposal_route_manifest_entries : Array(ProposalRouteEntry)
        fingerprint = manifest_fingerprint(@proposal_route_manifest_path)
        unless fingerprint
          @proposal_route_manifest_fingerprint = nil
          @proposal_route_manifest_cache = [] of ProposalRouteEntry
          return [] of ProposalRouteEntry
        end
        if @proposal_route_manifest_fingerprint == fingerprint
          return @proposal_route_manifest_cache
        end

        parsed = [] of ProposalRouteEntry
        File.each_line(@proposal_route_manifest_path) do |line|
          stripped = line.strip
          next if stripped.empty?

          begin
            parsed << ProposalRouteEntry.from_json(stripped)
          rescue JSON::ParseException | KeyError
            # A corrupt proposal-route line must not enable an approximate drafter.
          end
        end
        @proposal_route_manifest_fingerprint = fingerprint
        @proposal_route_manifest_cache = parsed
        parsed
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
        full_history_hash = Qwen35PromptCache.token_hash_concat(prompt_token_ids, output_token_ids)
        artifact_steps = exact_entry.artifact_validation_steps
        artifact_hash = exact_entry.artifact_validation_hash
        artifact_next = exact_entry.next_token_id
        raise ArgumentError.new("exact_entry is not an exact-known-span artifact") unless exact_entry.artifact_validation_kind == EXACT_KNOWN_SPAN_VALIDATION_KIND
        raise ArgumentError.new("exact_entry validation steps mismatch") unless artifact_steps == output_token_ids.size
        raise ArgumentError.new("exact_entry validation hash mismatch") unless artifact_hash == full_history_hash
        raise ArgumentError.new("exact_entry prefix mismatch") unless exact_entry.prefix_len == full_history_len - 1
        raise ArgumentError.new("exact_entry prefix token hash mismatch") unless exact_entry.token_hash == Qwen35PromptCache.token_hash_concat(prompt_token_ids, output_token_ids, exact_entry.prefix_len)
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
        entry
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

        prompt_text_hash = Qwen35PromptCache.prompt_text_hash(prompt_text)
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
        valid = Qwen35PromptCache.output_fast_forward_entry_valid?(
          entry,
          model_id,
          session_id,
          prompt_text,
          output_token_count,
          tokenizer_id: tokenizer_id,
          turn_id: turn_id,
        )
        unless valid
          return nil
        end

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
        FileUtils.mkdir_p(@root)
        File.open(@manifest_path, "a") do |file|
          entry.to_json(file)
          file << '\n'
        end
        @entry_manifest_fingerprint = nil
      end

      private def ensure_entry_indices : Nil
        manifest_entries
        fingerprint = @entry_manifest_fingerprint
        return if @entry_index_fingerprint == fingerprint

        exact_index = {} of Tuple(String, String, String, Int32) => Array(Entry)
        prefix_index = {} of Tuple(String, String, Int32, String) => Array(Entry)
        @entry_manifest_cache.each do |entry|
          next unless entry.runtime_id == RUNTIME_ID

          exact_key = {entry.model_id, entry.tokenizer_id, entry.prompt_hash.downcase, entry.prefix_len}
          (exact_index[exact_key] ||= [] of Entry) << entry
          if token_hash = entry.token_hash
            prefix_key = {entry.model_id, entry.tokenizer_id, entry.prefix_len, token_hash}
            (prefix_index[prefix_key] ||= [] of Entry) << entry
          end
        end
        @entry_exact_index = exact_index
        @entry_prefix_index = prefix_index
        @entry_index_fingerprint = fingerprint
      end

      private def ensure_source_history_indices : Nil
        source_history_manifest_entries
        fingerprint = @source_history_manifest_fingerprint
        return if @source_history_index_fingerprint == fingerprint

        base_index = {} of Tuple(String, String, String) => Array(SourceHistoryEntry)
        turn_index = {} of Tuple(String, String, String, String) => Array(SourceHistoryEntry)
        @source_history_manifest_cache.each do |entry|
          next unless entry.runtime_id == SOURCE_HISTORY_RUNTIME_ID

          base_key = {entry.session_id, entry.model_id, entry.tokenizer_id}
          (base_index[base_key] ||= [] of SourceHistoryEntry) << entry
          if turn = entry.turn_id
            turn_key = {entry.session_id, entry.model_id, entry.tokenizer_id, turn}
            (turn_index[turn_key] ||= [] of SourceHistoryEntry) << entry
          end
        end
        @source_history_base_index = base_index
        @source_history_turn_index = turn_index
        @source_history_index_fingerprint = fingerprint
      end

      private def ensure_tokenized_prompt_indices : Nil
        tokenized_prompt_manifest_entries
        fingerprint = @tokenized_prompt_manifest_fingerprint
        return if @tokenized_prompt_index_fingerprint == fingerprint

        exact_index = {} of Tuple(String, String, String) => Array(TokenizedPromptEntry)
        model_index = {} of Tuple(String, String) => Array(TokenizedPromptEntry)
        @tokenized_prompt_manifest_cache.each do |entry|
          next unless entry.runtime_id == TOKENIZED_PROMPT_RUNTIME_ID

          exact_key = {entry.model_id, entry.tokenizer_id, entry.prompt_text_hash}
          (exact_index[exact_key] ||= [] of TokenizedPromptEntry) << entry
          model_key = {entry.model_id, entry.prompt_text_hash}
          (model_index[model_key] ||= [] of TokenizedPromptEntry) << entry
        end
        @tokenized_prompt_exact_index = exact_index
        @tokenized_prompt_model_index = model_index
        @tokenized_prompt_index_fingerprint = fingerprint
      end

      private def ensure_proposal_route_indices : Nil
        proposal_route_manifest_entries
        fingerprint = @proposal_route_manifest_fingerprint
        return if @proposal_route_index_fingerprint == fingerprint

        prompt_index = {} of Tuple(String, String, String, String) => Array(ProposalRouteEntry)
        key_index = {} of Tuple(String, String, String) => Array(ProposalRouteEntry)
        @proposal_route_manifest_cache.each do |entry|
          next unless entry.runtime_id == PROPOSAL_ROUTE_RUNTIME_ID

          prompt_key = {entry.model_id, entry.tokenizer_id, entry.prompt_text_hash, entry.prompt_token_hash}
          (prompt_index[prompt_key] ||= [] of ProposalRouteEntry) << entry
          if route_key_hash = entry.route_key_hash
            key = {entry.model_id, entry.tokenizer_id, route_key_hash}
            (key_index[key] ||= [] of ProposalRouteEntry) << entry
          end
        end
        @proposal_route_prompt_index = prompt_index
        @proposal_route_key_index = key_index
        @proposal_route_index_fingerprint = fingerprint
      end

      private def source_history_entry_valid?(entry : SourceHistoryEntry) : Bool
        entry.runtime_id == SOURCE_HISTORY_RUNTIME_ID &&
          entry.token_count == entry.token_ids.size &&
          entry.token_hash == Qwen35PromptCache.token_hash(entry.token_ids)
      end

      private def tokenized_prompt_entry_valid?(entry : TokenizedPromptEntry) : Bool
        entry.runtime_id == TOKENIZED_PROMPT_RUNTIME_ID &&
          entry.token_count == entry.token_ids.size &&
          entry.token_hash == Qwen35PromptCache.token_hash(entry.token_ids)
      end

      private def validate_proposal_route!(route : String, route_rank : Int32?, route_layers : Array(Int32)) : Nil
        unless Qwen35PromptCache.valid_proposal_route_name?(route)
          raise ArgumentError.new("unsupported proposal route #{route.inspect}")
        end
        if route == PROPOSAL_ROUTE_BASELINE && route_rank
          raise ArgumentError.new("baseline proposal route must not set route_rank")
        end
        if route == PROPOSAL_ROUTE_PCA_UPDOWN
          raise ArgumentError.new("pca_updown proposal route requires positive route_rank") unless route_rank && route_rank.not_nil! > 0
        end
        raise ArgumentError.new("proposal route layers must be non-negative") if route_layers.any? { |layer| layer < 0 }
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
          prefix_len: entry.prefix_len,
          max_seq: entry.max_seq,
          layer_count: entry.layer_count,
          artifact_path: entry.artifact_path,
          artifact_sha256: entry.artifact_sha256,
          artifact_byte_size: entry.artifact_byte_size,
          state_byte_size: entry.state_byte_size,
          created_at_unix: entry.created_at_unix,
          prompt_preview: entry.prompt_preview,
          token_hash: entry.token_hash,
          artifact_codec: entry.artifact_codec,
          artifact_codec_block: entry.artifact_codec_block,
          artifact_live_kv_tokens: entry.artifact_live_kv_tokens,
          artifact_validation_kind: entry.artifact_validation_kind,
          artifact_validation_steps: entry.artifact_validation_steps,
          artifact_validation_hash: entry.artifact_validation_hash,
          next_token_id: entry.next_token_id,
          next_token_logit: entry.next_token_logit,
        )
      end

      private def clone_source_history_entries(entries : Array(SourceHistoryEntry)) : Array(SourceHistoryEntry)
        entries.map { |entry| clone_source_history_entry(entry) }
      end

      private def clone_source_history_entry(entry : SourceHistoryEntry) : SourceHistoryEntry
        SourceHistoryEntry.new(
          runtime_id: entry.runtime_id,
          session_id: entry.session_id,
          turn_id: entry.turn_id,
          model_id: entry.model_id,
          tokenizer_id: entry.tokenizer_id,
          token_hash: entry.token_hash,
          token_count: entry.token_count,
          token_ids: entry.token_ids.dup,
          created_at_unix: entry.created_at_unix,
          generated_token_count: entry.generated_token_count,
          generated_text: entry.generated_text,
          generated_text_hash: entry.generated_text_hash,
        )
      end

      private def clone_tokenized_prompt_entries(entries : Array(TokenizedPromptEntry)) : Array(TokenizedPromptEntry)
        entries.map { |entry| clone_tokenized_prompt_entry(entry) }
      end

      private def clone_tokenized_prompt_entry(entry : TokenizedPromptEntry) : TokenizedPromptEntry
        TokenizedPromptEntry.new(
          runtime_id: entry.runtime_id,
          model_id: entry.model_id,
          tokenizer_id: entry.tokenizer_id,
          prompt_text_hash: entry.prompt_text_hash,
          token_hash: entry.token_hash,
          token_count: entry.token_count,
          token_ids: entry.token_ids.dup,
          created_at_unix: entry.created_at_unix,
        )
      end

      private def clone_proposal_route_entries(entries : Array(ProposalRouteEntry)) : Array(ProposalRouteEntry)
        entries.map { |entry| clone_proposal_route_entry(entry) }
      end

      private def clone_proposal_route_entry(entry : ProposalRouteEntry) : ProposalRouteEntry
        ProposalRouteEntry.new(
          runtime_id: entry.runtime_id,
          model_id: entry.model_id,
          tokenizer_id: entry.tokenizer_id,
          prompt_text_hash: entry.prompt_text_hash,
          prompt_token_hash: entry.prompt_token_hash,
          prompt_token_count: entry.prompt_token_count,
          route: entry.route,
          route_rank: entry.route_rank,
          route_layers: entry.route_layers.dup,
          route_key_hash: entry.route_key_hash,
          route_key_preview: entry.route_key_preview,
          trigger: entry.trigger,
          evidence: entry.evidence,
          created_at_unix: entry.created_at_unix,
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

      private def manifest_fingerprint(path : String) : ManifestFingerprint?
        return nil unless File.exists?(path)

        info = File.info(path)
        mtime = info.modification_time
        ManifestFingerprint.new(info.size, mtime.to_unix, mtime.nanosecond)
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

    def token_hash_concat(left_ids : Array(Int32),
                          right_ids : Array(Int32),
                          prefix_len : Int32 = left_ids.size + right_ids.size) : String
      total = left_ids.size + right_ids.size
      raise ArgumentError.new("prefix_len out of range: #{prefix_len}") if prefix_len < 0 || prefix_len > total

      io = IO::Memory.new
      io.write("qwen35-token-v1\0".to_slice)
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
      Digest::SHA256.hexdigest("qwen35-prompt-text-v1\0#{prompt_text}")
    end

    def proposal_route_key_hash(route_key : String) : String
      Digest::SHA256.hexdigest("qwen35-proposal-route-key-v1\0#{route_key}")
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

    def valid_proposal_route_name?(route : String) : Bool
      route == PROPOSAL_ROUTE_BASELINE || route == PROPOSAL_ROUTE_PCA_UPDOWN
    end

    def proposal_route_entry_valid?(entry : ProposalRouteEntry,
                                    model_id : String,
                                    tokenizer_id : String,
                                    prompt_text : String? = nil,
                                    token_ids : Array(Int32)? = nil) : Bool
      return false unless entry.runtime_id == PROPOSAL_ROUTE_RUNTIME_ID
      return false unless entry.model_id == model_id
      return false unless entry.tokenizer_id == tokenizer_id
      return false unless valid_proposal_route_name?(entry.route)
      return false unless entry.prompt_token_count > 0
      return false if entry.route_layers.any? { |layer| layer < 0 }
      if entry.route == PROPOSAL_ROUTE_BASELINE
        return false if entry.route_rank
      elsif entry.route == PROPOSAL_ROUTE_PCA_UPDOWN
        rank = entry.route_rank
        return false unless rank && rank > 0
      end
      if prompt = prompt_text
        return false unless entry.prompt_text_hash == prompt_text_hash(prompt)
      end
      if ids = token_ids
        return false unless entry.prompt_token_count == ids.size
        return false unless entry.prompt_token_hash == token_hash(ids)
      end

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
      codec = normalized_artifact_codec(entry)
      return live_kv_metadata_valid?(entry) if RAW_ARTIFACT_CODECS.includes?(codec)
      return false unless COMPRESSED_ARTIFACT_CODECS.includes?(codec)
      return false unless entry.artifact_validation_kind.try(&.empty?) == false
      return false unless entry.artifact_validation_hash.try(&.empty?) == false
      steps = entry.artifact_validation_steps
      return false unless steps && steps > 0

      if codec == "recurrent-int8"
        block = entry.artifact_codec_block
        return false unless block && block > 0
      end
      return false unless live_kv_metadata_valid?(entry)
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

    private def live_kv_metadata_valid?(entry : Entry) : Bool
      live = entry.artifact_live_kv_tokens
      return true unless live
      return false unless live > 0
      return false if live > entry.max_seq
      return false if live > entry.prefix_len
      return false unless entry.token_hash.try(&.empty?) == false

      true
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
          artifact_live_kv_tokens integer CHECK (artifact_live_kv_tokens IS NULL OR artifact_live_kv_tokens > 0),
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
          artifact_codec, artifact_codec_block, artifact_live_kv_tokens,
          artifact_validation_kind,
          artifact_validation_steps, artifact_validation_hash,
          next_token_id, next_token_logit,
          created_at_unix, prompt_preview
      ) VALUES (
          $1, $2, $3, $4, $5,
          $6, $7, $8, $9, $10,
          $11, $12, $13, $14,
          $15, $16, $17,
          $18,
          $19, $20,
          $21, $22,
          $23, $24
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
          artifact_live_kv_tokens = EXCLUDED.artifact_live_kv_tokens,
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
        entry.artifact_live_kv_tokens,
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
