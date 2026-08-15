require "digest/sha256"
require "json"
require "./qwen35_prompt_cache"

module ML::GGUF
  # Immutable metadata for rollback-capable session boundaries.
  #
  # A depth-zero checkpoint names one full QBit anchor. Descendants keep that
  # anchor immutable and store only the exact cumulative token suffix needed to
  # reach their boundary. The caller's transcript remains authoritative: every
  # lookup re-hashes and compares its token prefix before an anchor is restored.
  module QwenQBitSessionCheckpoint
    extend self

    SCHEMA_ID               = "cogni-ml/qwen-qbit-session-checkpoint-v1"
    MAX_SESSION_BYTES       = 1024
    MAX_DELTA_DEPTH         =    8
    MAX_REPLAY_TOKENS       =  512
    MAX_BOUNDARY_BYTES      = 1024 * 1024
    MAX_BOUNDARY_CANDIDATES = 1024
    HEX_ID_SIZE             =   64

    class Entry
      include JSON::Serializable

      property schema_id : String
      property session_hash : String
      property checkpoint_id : String
      property parent_checkpoint_id : String?
      property anchor_cache_id : UInt64
      property anchor_lookup_key : String
      property anchor_generation_id : String
      property anchor_certificate_id : String
      property anchor_prefix_len : Int32
      property anchor_token_hash : String
      property anchor_token_ids : Array(Int32)
      property depth : Int32
      property cumulative_token_ids : Array(Int32)
      property delta_token_hash : String
      property child_prefix_len : Int32
      property child_token_hash : String
      property boundary_text_bytes : Int32
      property boundary_text_hash : String
      property created_at_unix : Int64
      property expires_at_unix : Int64
      property certificate_id : String

      def initialize(@schema_id : String,
                     @session_hash : String,
                     @checkpoint_id : String,
                     @parent_checkpoint_id : String?,
                     @anchor_cache_id : UInt64,
                     @anchor_lookup_key : String,
                     @anchor_generation_id : String,
                     @anchor_certificate_id : String,
                     @anchor_prefix_len : Int32,
                     @anchor_token_hash : String,
                     @anchor_token_ids : Array(Int32),
                     @depth : Int32,
                     @cumulative_token_ids : Array(Int32),
                     @delta_token_hash : String,
                     @child_prefix_len : Int32,
                     @child_token_hash : String,
                     @boundary_text_bytes : Int32,
                     @boundary_text_hash : String,
                     @created_at_unix : Int64,
                     @expires_at_unix : Int64,
                     @certificate_id : String)
      end
    end

    def session_hash(session_id : String) : String
      validate_session_id!(session_id)
      Digest::SHA256.hexdigest("qwen-qbit-session-v1\0#{session_id}")
    end

    def build_anchor(*,
                     session_id : String,
                     checkpoint_id : String,
                     parent_checkpoint_id : String?,
                     anchor_cache_id : UInt64,
                     anchor_lookup_key : String,
                     anchor_generation_id : String,
                     anchor_certificate_id : String,
                     token_ids : Array(Int32),
                     boundary_text : String,
                     expires_at_unix : Int64,
                     created_at_unix : Int64 = Time.utc.to_unix) : Entry
      raise ArgumentError.new("QBit checkpoint anchor token sequence is empty") if token_ids.empty?
      token_hash = Qwen35PromptCache.token_hash(token_ids)
      entry = Entry.new(
        schema_id: SCHEMA_ID,
        session_hash: session_hash(session_id),
        checkpoint_id: checkpoint_id,
        parent_checkpoint_id: parent_checkpoint_id,
        anchor_cache_id: anchor_cache_id,
        anchor_lookup_key: anchor_lookup_key,
        anchor_generation_id: anchor_generation_id,
        anchor_certificate_id: anchor_certificate_id,
        anchor_prefix_len: token_ids.size.to_i32,
        anchor_token_hash: token_hash,
        anchor_token_ids: token_ids,
        depth: 0,
        cumulative_token_ids: [] of Int32,
        delta_token_hash: Qwen35PromptCache.token_hash([] of Int32),
        child_prefix_len: token_ids.size.to_i32,
        child_token_hash: token_hash,
        boundary_text_bytes: boundary_text.bytesize.to_i32,
        boundary_text_hash: boundary_hash(boundary_text),
        created_at_unix: created_at_unix,
        expires_at_unix: expires_at_unix,
        certificate_id: "",
      )
      entry.certificate_id = certificate_id(entry)
      validate!(entry, session_id, token_ids, checkpoint_id: checkpoint_id)
      entry
    end

    def build_delta(*,
                    session_id : String,
                    checkpoint_id : String,
                    parent : Entry,
                    token_ids : Array(Int32),
                    boundary_text : String,
                    created_at_unix : Int64 = Time.utc.to_unix) : Entry
      validate!(parent, session_id, token_ids, checkpoint_id: parent.checkpoint_id)
      unless delta_admissible?(parent, token_ids)
        raise ArgumentError.new("QBit checkpoint delta exceeds replay depth or token bounds")
      end
      if created_at_unix < parent.created_at_unix || created_at_unix >= parent.expires_at_unix
        raise ArgumentError.new("QBit checkpoint delta time is outside its anchor lifetime")
      end

      cumulative = token_ids[parent.anchor_prefix_len, token_ids.size - parent.anchor_prefix_len]
      entry = Entry.new(
        schema_id: SCHEMA_ID,
        session_hash: parent.session_hash,
        checkpoint_id: checkpoint_id,
        parent_checkpoint_id: parent.checkpoint_id,
        anchor_cache_id: parent.anchor_cache_id,
        anchor_lookup_key: parent.anchor_lookup_key,
        anchor_generation_id: parent.anchor_generation_id,
        anchor_certificate_id: parent.anchor_certificate_id,
        anchor_prefix_len: parent.anchor_prefix_len,
        anchor_token_hash: parent.anchor_token_hash,
        anchor_token_ids: parent.anchor_token_ids,
        depth: parent.depth + 1,
        cumulative_token_ids: cumulative,
        delta_token_hash: Qwen35PromptCache.token_hash(cumulative),
        child_prefix_len: token_ids.size.to_i32,
        child_token_hash: Qwen35PromptCache.token_hash(token_ids),
        boundary_text_bytes: boundary_text.bytesize.to_i32,
        boundary_text_hash: boundary_hash(boundary_text),
        created_at_unix: created_at_unix,
        expires_at_unix: parent.expires_at_unix,
        certificate_id: "",
      )
      entry.certificate_id = certificate_id(entry)
      validate!(entry, session_id, token_ids, checkpoint_id: checkpoint_id)
      entry
    end

    def delta_admissible?(parent : Entry, token_ids : Array(Int32)) : Bool
      return false if parent.depth >= MAX_DELTA_DEPTH
      return false if token_ids.size <= parent.child_prefix_len
      cumulative_size = token_ids.size - parent.anchor_prefix_len
      return false if cumulative_size > MAX_REPLAY_TOKENS
      return false if parent.child_prefix_len > token_ids.size
      Qwen35PromptCache.token_hash(token_ids, parent.child_prefix_len) == parent.child_token_hash
    end

    def validate!(entry : Entry,
                  session_id : String,
                  token_ids : Array(Int32),
                  checkpoint_id : String? = nil) : Nil
      validate_entry_shape!(entry)
      raise ArgumentError.new("QBit checkpoint session mismatch") unless entry.session_hash == session_hash(session_id)
      if expected_id = checkpoint_id
        raise ArgumentError.new("QBit checkpoint identity mismatch") unless entry.checkpoint_id == expected_id
      end
      unless entry.child_prefix_len <= token_ids.size
        raise ArgumentError.new("QBit checkpoint child prefix is outside the caller transcript")
      end
      child_tokens = token_ids[0, entry.child_prefix_len]
      unless Qwen35PromptCache.token_hash(child_tokens) == entry.child_token_hash
        raise ArgumentError.new("QBit checkpoint child token hash mismatch")
      end
      anchor_tokens = child_tokens[0, entry.anchor_prefix_len]
      unless Qwen35PromptCache.token_hash(anchor_tokens) == entry.anchor_token_hash
        raise ArgumentError.new("QBit checkpoint anchor token hash mismatch")
      end
      actual_delta = child_tokens[entry.anchor_prefix_len, entry.child_prefix_len - entry.anchor_prefix_len]
      unless actual_delta == entry.cumulative_token_ids
        raise ArgumentError.new("QBit checkpoint delta tokens mismatch")
      end
      unless entry.delta_token_hash == Qwen35PromptCache.token_hash(entry.cumulative_token_ids)
        raise ArgumentError.new("QBit checkpoint delta token hash mismatch")
      end
      raise ArgumentError.new("QBit checkpoint certificate mismatch") unless entry.certificate_id == certificate_id(entry)
    end

    def validate_boundary!(entry : Entry,
                           session_id : String,
                           rendered : String,
                           checkpoint_id : String? = nil) : Nil
      validate_entry_shape!(entry)
      raise ArgumentError.new("QBit checkpoint session mismatch") unless entry.session_hash == session_hash(session_id)
      if expected_id = checkpoint_id
        raise ArgumentError.new("QBit checkpoint identity mismatch") unless entry.checkpoint_id == expected_id
      end
      if entry.boundary_text_bytes > rendered.bytesize
        raise ArgumentError.new("QBit checkpoint text boundary is outside the caller transcript")
      end
      boundary = rendered.byte_slice(0, entry.boundary_text_bytes)
      unless boundary && boundary_hash(boundary) == entry.boundary_text_hash
        raise ArgumentError.new("QBit checkpoint text boundary mismatch")
      end
      raise ArgumentError.new("QBit checkpoint certificate mismatch") unless entry.certificate_id == certificate_id(entry)
    end

    def canonical_token_ids(entry : Entry) : Array(Int32)
      validate_certificate!(entry)
      tokens = entry.anchor_token_ids + entry.cumulative_token_ids
      unless tokens.size == entry.child_prefix_len && Qwen35PromptCache.token_hash(tokens) == entry.child_token_hash
        raise ArgumentError.new("QBit checkpoint canonical token history mismatch")
      end
      tokens
    end

    def boundary_hash(text : String) : String
      Digest::SHA256.hexdigest(text)
    end

    def validate_certificate!(entry : Entry) : Nil
      validate_entry_shape!(entry)
      raise ArgumentError.new("QBit checkpoint certificate mismatch") unless entry.certificate_id == certificate_id(entry)
    end

    def certificate_id(entry : Entry) : String
      io = IO::Memory.new
      write_string(io, "qwen-qbit-session-certificate-v1")
      write_string(io, entry.schema_id)
      write_string(io, entry.session_hash)
      write_string(io, entry.checkpoint_id)
      write_string(io, entry.parent_checkpoint_id || "")
      io.write_bytes(entry.anchor_cache_id, IO::ByteFormat::LittleEndian)
      write_string(io, entry.anchor_lookup_key)
      write_string(io, entry.anchor_generation_id)
      write_string(io, entry.anchor_certificate_id)
      io.write_bytes(entry.anchor_prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      write_string(io, entry.anchor_token_hash)
      io.write_bytes(entry.anchor_token_ids.size.to_u32, IO::ByteFormat::LittleEndian)
      entry.anchor_token_ids.each do |token_id|
        io.write_bytes(token_id, IO::ByteFormat::LittleEndian)
      end
      io.write_bytes(entry.depth.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.cumulative_token_ids.size.to_u32, IO::ByteFormat::LittleEndian)
      entry.cumulative_token_ids.each do |token_id|
        io.write_bytes(token_id, IO::ByteFormat::LittleEndian)
      end
      write_string(io, entry.delta_token_hash)
      io.write_bytes(entry.child_prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      write_string(io, entry.child_token_hash)
      io.write_bytes(entry.boundary_text_bytes.to_u32, IO::ByteFormat::LittleEndian)
      write_string(io, entry.boundary_text_hash)
      io.write_bytes(entry.created_at_unix, IO::ByteFormat::LittleEndian)
      io.write_bytes(entry.expires_at_unix, IO::ByteFormat::LittleEndian)
      Digest::SHA256.hexdigest(io.to_slice)
    end

    private def validate_entry_shape!(entry : Entry) : Nil
      raise ArgumentError.new("QBit checkpoint schema mismatch") unless entry.schema_id == SCHEMA_ID
      validate_hex_id!(entry.session_hash, "session hash")
      validate_hex_id!(entry.checkpoint_id, "checkpoint")
      if parent = entry.parent_checkpoint_id
        validate_hex_id!(parent, "parent checkpoint")
      end
      validate_hex_id!(entry.anchor_lookup_key, "anchor lookup")
      validate_hex_id!(entry.anchor_generation_id, "anchor generation")
      validate_hex_id!(entry.anchor_certificate_id, "anchor certificate")
      unless entry.anchor_prefix_len > 0 && entry.anchor_prefix_len <= entry.child_prefix_len
        raise ArgumentError.new("QBit checkpoint anchor prefix length is invalid")
      end
      validate_hex_id!(entry.anchor_token_hash, "anchor token hash")
      unless entry.anchor_token_ids.size == entry.anchor_prefix_len &&
             Qwen35PromptCache.token_hash(entry.anchor_token_ids) == entry.anchor_token_hash
        raise ArgumentError.new("QBit checkpoint anchor token history mismatch")
      end
      raise ArgumentError.new("QBit checkpoint anchor contains a negative token") if entry.anchor_token_ids.any? { |id| id < 0 }
      unless entry.depth >= 0 && entry.depth <= MAX_DELTA_DEPTH
        raise ArgumentError.new("QBit checkpoint depth is invalid")
      end
      if entry.depth == 0
        unless entry.cumulative_token_ids.empty? && entry.child_prefix_len == entry.anchor_prefix_len
          raise ArgumentError.new("QBit checkpoint anchor contains a delta")
        end
      else
        raise ArgumentError.new("QBit checkpoint delta has no parent") unless entry.parent_checkpoint_id
        unless entry.cumulative_token_ids.size == entry.child_prefix_len - entry.anchor_prefix_len
          raise ArgumentError.new("QBit checkpoint delta length mismatch")
        end
      end
      if entry.cumulative_token_ids.size > MAX_REPLAY_TOKENS
        raise ArgumentError.new("QBit checkpoint replay token limit exceeded")
      end
      raise ArgumentError.new("QBit checkpoint contains a negative token") if entry.cumulative_token_ids.any? { |id| id < 0 }
      validate_hex_id!(entry.delta_token_hash, "delta token hash")
      validate_hex_id!(entry.child_token_hash, "child token hash")
      unless entry.boundary_text_bytes > 0 && entry.boundary_text_bytes <= MAX_BOUNDARY_BYTES
        raise ArgumentError.new("QBit checkpoint text boundary size is invalid")
      end
      validate_hex_id!(entry.boundary_text_hash, "text boundary hash")
      unless entry.created_at_unix > 0 && entry.expires_at_unix > entry.created_at_unix
        raise ArgumentError.new("QBit checkpoint lifetime is invalid")
      end
      validate_hex_id!(entry.certificate_id, "certificate")
    end

    private def validate_session_id!(session_id : String) : Nil
      size = session_id.bytesize
      unless size > 0 && size <= MAX_SESSION_BYTES
        raise ArgumentError.new("QBit session identity is outside 1..#{MAX_SESSION_BYTES} bytes")
      end
    end

    private def validate_hex_id!(value : String, label : String) : Nil
      unless value.bytesize == HEX_ID_SIZE && value.matches?(/\A[0-9a-f]+\z/)
        raise ArgumentError.new("QBit checkpoint #{label} is invalid")
      end
    end

    private def write_string(io : IO, value : String) : Nil
      io.write_bytes(value.bytesize.to_u32, IO::ByteFormat::LittleEndian)
      io.write(value.to_slice)
    end
  end
end
