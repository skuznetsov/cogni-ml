require "digest/sha256"
require "./reader"
require "./gpt2_bpe"

module ML::GGUF
  # Native byte-level GPT-4o BPE tokenizer for Inkling 2 Mini GGUFs.
  #
  # The pre-tokenizer follows the pinned llama.cpp GPT4O regex used for
  # tokenizer.ggml.pre=inkling2. Special GGML token types are split before
  # pre-tokenization, and encoding never adds BOS (Inkling2 has no BOS token).
  class Inkling2Tokenizer
    ENCODING_REVISION = "1-gpt4o-reference"
    EOS_TOKEN_ID      = 200006
    PAD_TOKEN_ID      = 199999

    # The model template's renderer output is admitted by digest so a changed
    # template cannot silently keep using this small, intentionally scoped
    # single-user-text renderer.
    SUPPORTED_CHAT_TEMPLATE_SHA256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"

    getter vocab : Array(String)
    getter token_types : Array(Int32)
    getter eos_id : Int32
    getter pad_id : Int32
    getter chat_template : String?

    private alias NativeFragment = String | Int32

    private INKLING2_GPT4O_PRETOKENIZER = /[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n\/]*|\s*[\r\n]+|\s+(?!\S)|\s+/

    @token_to_id : Hash(String, Int32)
    @bpe_ranks : Hash(Tuple(String, String), Int32)
    @special_tokens : Array(Tuple(String, Int32))

    def initialize(@vocab : Array(String), *,
                   merges : Array(String) = [] of String,
                   @token_types : Array(Int32),
                   @eos_id : Int32,
                   @pad_id : Int32,
                   @chat_template : String? = nil)
      raise ArgumentError.new("token type count must match vocabulary size") unless @token_types.size == @vocab.size
      raise ArgumentError.new("eos token id #{@eos_id} is out of range") unless @eos_id >= 0 && @eos_id < @vocab.size
      raise ArgumentError.new("padding token id #{@pad_id} is out of range") unless @pad_id >= 0 && @pad_id < @vocab.size

      @token_to_id = {} of String => Int32
      @vocab.each_with_index { |piece, id| @token_to_id[piece] = id.to_i32 }

      @bpe_ranks = {} of Tuple(String, String) => Int32
      merges.each_with_index do |merge, rank|
        parts = merge.split(' ', limit: 2)
        next unless parts.size == 2

        @bpe_ranks[{parts[0], parts[1]}] = rank.to_i32
      end

      @special_tokens = [] of Tuple(String, Int32)
      @vocab.each_with_index do |piece, id|
        type = @token_types[id]
        if special_type?(type) && !piece.empty?
          @special_tokens << {piece, id.to_i32}
        end
      end
      @special_tokens.sort! do |left, right|
        by_length = right[0].bytesize <=> left[0].bytesize
        by_length == 0 ? (left[1] <=> right[1]) : by_length
      end
    end

    def self.from_gguf(gguf : GGUFFile) : Inkling2Tokenizer
      pretokenizer = gguf.get_string("tokenizer.ggml.pre") || ""
      raise ArgumentError.new("inkling2_tokenizer: expected tokenizer.ggml.pre=inkling2, got #{pretokenizer.inspect}") unless pretokenizer == "inkling2"

      model = gguf.get_string("tokenizer.ggml.model") || ""
      raise ArgumentError.new("inkling2_tokenizer: expected tokenizer.ggml.model=gpt2, got #{model.inspect}") unless model == "gpt2"

      tokens_raw = gguf.metadata["tokenizer.ggml.tokens"]?
      raise ArgumentError.new("inkling2_tokenizer: missing tokenizer.ggml.tokens") unless tokens_raw.is_a?(Array)
      vocab = tokens_raw.map { |token| token.as(String) }

      merges_raw = gguf.metadata["tokenizer.ggml.merges"]?
      raise ArgumentError.new("inkling2_tokenizer: missing tokenizer.ggml.merges") unless merges_raw.is_a?(Array)
      merges = merges_raw.map { |merge| merge.as(String) }

      types_raw = gguf.metadata["tokenizer.ggml.token_type"]?
      raise ArgumentError.new("inkling2_tokenizer: missing tokenizer.ggml.token_type") unless types_raw.is_a?(Array)
      token_types = types_raw.map { |value| metadata_integer(value).to_i32 }
      raise ArgumentError.new("inkling2_tokenizer: token type count does not match vocabulary") unless token_types.size == vocab.size

      eos_id = gguf.get_int("tokenizer.ggml.eos_token_id").try(&.to_i32) ||
               raise ArgumentError.new("inkling2_tokenizer: missing tokenizer.ggml.eos_token_id")
      pad_id = gguf.get_int("tokenizer.ggml.padding_token_id").try(&.to_i32) || eos_id

      if add_bos = gguf.metadata["tokenizer.ggml.add_bos_token"]?
        should_add_bos = case add_bos
                         when Bool                                                     then add_bos
                         when Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64 then add_bos != 0
                         else                                                               false
                         end
        raise ArgumentError.new("inkling2_tokenizer: Inkling2 requires add_bos_token=false") if should_add_bos
      end

      chat_template = gguf.get_string("tokenizer.chat_template")
      new(vocab, merges: merges, token_types: token_types, eos_id: eos_id, pad_id: pad_id, chat_template: chat_template)
    end

    # Encode UTF-8 text to token IDs. GGML special token types 2/3/4 are parsed
    # as control tokens just like llama.cpp's tokenizer special-token pass.
    def encode(text : String) : Array(Int32)
      ids = [] of Int32
      partition_special_tokens(text).each do |fragment|
        case fragment
        when Int32
          ids << fragment
        when String
          encode_text_fragment(fragment, ids)
        end
      end
      ids
    end

    # Decode byte-level tokens. Control tokens are omitted by default so
    # generated Inkling message framing does not leak into user-facing text.
    def decode(ids : Array(Int32), *, skip_special : Bool = true) : String
      bytes = [] of UInt8
      ids.each do |id|
        raise ArgumentError.new("decode: token id #{id} out of range (vocab=#{@vocab.size})") if id < 0 || id >= @vocab.size

        piece = @vocab[id]
        if special_type?(@token_types[id])
          piece.to_slice.each { |byte| bytes << byte } unless skip_special
        else
          GPT2BPE.decode_piece(piece, bytes)
        end
      end

      String.new(Slice.new(bytes.to_unsafe, bytes.size))
    end

    def decode_single(id : Int32, *, skip_special : Bool = true) : String
      decode([id], skip_special: skip_special)
    end

    def chat_template_supported? : Bool
      template = @chat_template
      return false unless template

      Digest::SHA256.hexdigest(template) == SUPPORTED_CHAT_TEMPLATE_SHA256
    end

    # Render the actual GGUF template's supported simple case: a default
    # effort-control message followed by one user text message. The pinned
    # template has no add_generation_prompt branch, so llama.cpp computes an
    # empty generation suffix; the model predicts its own assistant header.
    def render_user_prompt(text : String) : String
      unless chat_template_supported?
        raise ArgumentError.new("unsupported Inkling chat template; use --raw or provide the pinned Inkling 2 Mini template")
      end

      "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" +
        "<|message_user|><|content_text|>#{text}<|end_message|>"
    end

    private def encode_text_fragment(text : String, ids : Array(Int32)) : Nil
      pretokenized = [] of String
      text.scan(INKLING2_GPT4O_PRETOKENIZER) { |match| pretokenized << match[0] }
      unless pretokenized.join == text
        raise "inkling2 tokenize: GPT4O pretokenizer failed to cover the input fragment"
      end

      pretokenized.each do |word|
        encoded_word = GPT2BPE.encode_bytes(word)
        GPT2BPE.bpe(encoded_word, @token_to_id, @bpe_ranks).each do |piece|
          id = @token_to_id[piece]?
          raise "inkling2 encode: missing token piece #{piece.inspect}" unless id
          ids << id
        end
      end
    end

    private def partition_special_tokens(text : String) : Array(NativeFragment)
      fragments = [text] of NativeFragment
      @special_tokens.each do |special_text, special_id|
        partitioned = [] of NativeFragment
        fragments.each do |fragment|
          if fragment.is_a?(Int32)
            partitioned << fragment
            next
          end

          offset = 0
          matched = false
          while match_offset = fragment.byte_index(special_text, offset)
            matched = true
            if match_offset > offset
              partitioned << fragment.byte_slice(offset, match_offset - offset).not_nil!
            end
            partitioned << special_id
            offset = match_offset + special_text.bytesize
          end

          if matched
            if offset < fragment.bytesize
              partitioned << fragment.byte_slice(offset, fragment.bytesize - offset).not_nil!
            end
          else
            partitioned << fragment
          end
        end
        fragments = partitioned
      end
      fragments
    end

    private def special_type?(type : Int32) : Bool
      type == 2 || type == 3 || type == 4
    end

    private def self.metadata_integer(value : Value) : Int64
      case value
      when Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64
        value.to_i64
      else
        raise ArgumentError.new("inkling2_tokenizer: tokenizer.ggml.token_type contains a non-integer value")
      end
    end
  end
end
