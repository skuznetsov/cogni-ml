require "./reader"

# Qwen 3.5 / 3.6 / 3.8 tokenizer (GPT-2-style BPE).
#
# Current scope: native decoder + native Qwen3.5 BPE encoder.
# - Decoder is native Crystal: uses tokenizer.ggml.tokens[] from GGUF, handles
#   Ġ-for-space and Ċ-for-newline conventions.
# - Encoder is native Crystal using tokenizer.ggml.merges[] and the Qwen3.5
#   pre-tokenizer. Set `QWEN35_NATIVE_TOKENIZER_OFF=1` to use the older
#   llama.cpp `llama-tokenize` bootstrap path for A/B and debugging.
module ML::GGUF
  class Qwen35Tokenizer
    # Bump whenever the same GGUF metadata can encode a prompt differently so
    # prompt/token caches cannot silently reuse IDs from an older algorithm.
    ENCODING_REVISION = "2-special-partition"

    getter vocab : Array(String)
    getter eos_id : Int32
    getter pad_id : Int32
    getter? add_bos : Bool
    getter chat_template : String?
    getter llama_tokenize_bin : String
    getter model_path : String
    getter token_to_id : Hash(String, Int32)
    getter bpe_ranks : Hash(Tuple(String, String), Int32)
    getter token_types : Array(Int32)

    private alias NativeFragment = String | Int32

    @special_tokens : Array(Tuple(String, Int32))

    def initialize(@vocab : Array(String), @eos_id : Int32, @pad_id : Int32,
                   @add_bos : Bool, @model_path : String,
                   @llama_tokenize_bin : String = "",
                   @chat_template : String? = nil,
                   @token_to_id : Hash(String, Int32) = {} of String => Int32,
                   @bpe_ranks : Hash(Tuple(String, String), Int32) = {} of Tuple(String, String) => Int32,
                   @token_types : Array(Int32) = [] of Int32)
      @special_tokens = [] of Tuple(String, Int32)
      @vocab.each_with_index do |piece, id|
        token_type = @token_types[id]? || 0
        # GGML token types: 2=unknown, 3=control, 4=user-defined. llama.cpp
        # partitions these pieces before running the model pre-tokenizer.
        if token_type == 2 || token_type == 3 || token_type == 4
          @special_tokens << {piece, id.to_i32} unless piece.empty?
        end
      end
      # Match llama.cpp's longest-first special-token partitioning so an
      # overlapping short token cannot consume the prefix of a longer token.
      @special_tokens.sort! do |a, b|
        by_length = b[0].bytesize <=> a[0].bytesize
        by_length == 0 ? (a[1] <=> b[1]) : by_length
      end
    end

    def self.from_gguf(g : GGUFFile, model_path : String,
                       llama_tokenize_bin : String = "") : Qwen35Tokenizer
      tokens_raw = g.metadata["tokenizer.ggml.tokens"]?
      raise "qwen35_tokenizer: missing tokenizer.ggml.tokens" unless tokens_raw.is_a?(Array)
      vocab = tokens_raw.map { |t| t.as(String) }

      eos = g.get_int("tokenizer.ggml.eos_token_id").try(&.to_i32) || 248046
      pad = g.get_int("tokenizer.ggml.padding_token_id").try(&.to_i32) || eos
      add_bos_raw = g.metadata["tokenizer.ggml.add_bos_token"]?
      add_bos = case add_bos_raw
                when Bool then add_bos_raw
                when Int  then add_bos_raw != 0
                else           false
                end

      token_to_id = {} of String => Int32
      vocab.each_with_index { |piece, id| token_to_id[piece] = id.to_i32 }

      bpe_ranks = {} of Tuple(String, String) => Int32
      if merges_raw = g.metadata["tokenizer.ggml.merges"]?
        if merges_raw.is_a?(Array)
          merges_raw.each_with_index do |merge, rank|
            parts = merge.as(String).split(' ', limit: 2)
            next unless parts.size == 2

            bpe_ranks[{parts[0], parts[1]}] = rank.to_i32
          end
        end
      end

      types_raw = g.metadata["tokenizer.ggml.token_type"]?
      token_types = if types_raw.is_a?(Array)
                      types_raw.map do |value|
                        case value
                        when Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64
                          value.to_i32
                        else
                          0
                        end
                      end
                    else
                      [] of Int32
                    end

      chat_template = g.metadata["tokenizer.chat_template"]?.as?(String)

      new(vocab, eos, pad, add_bos, model_path, llama_tokenize_bin, chat_template, token_to_id, bpe_ranks, token_types)
    end

    # Decode a list of token ids back into a UTF-8 string.
    #
    # GPT-2 BPE convention: the tokenizer encodes bytes through a mapping that
    # replaces non-printable / whitespace bytes with printable Unicode. The
    # inverse mapping is used here to recover the original bytes.
    def decode(ids : Array(Int32)) : String
      bytes_out = Bytes.new(0)
      buf = [] of UInt8
      ids.each do |id|
        raise "decode: token id #{id} out of range (vocab=#{@vocab.size})" if id < 0 || id >= @vocab.size
        piece = @vocab[id]
        piece.each_char { |ch| byte = self.class.gpt2_byte_for(ch); buf << byte if byte }
      end
      String.new(Slice.new(buf.to_unsafe, buf.size))
    end

    # Decode a single token id (useful for streaming).
    def decode_single(id : Int32) : String
      decode([id])
    end

    # Encode text to token ids using the bootstrap external tokenizer.
    # Requires `llama_tokenize_bin` to be set to a valid llama-tokenize executable
    # (from llama.cpp build). Falls back to raising.
    def encode(text : String, *, add_bos_override : Bool? = nil) : Array(Int32)
      should_add_bos = (add_bos_override == true) || (add_bos_override.nil? && @add_bos)
      if ENV["QWEN35_NATIVE_TOKENIZER_OFF"]? != "1" && !should_add_bos && native_encoder_available?
        return encode_native(text)
      end

      encode_external(text, add_bos_override: add_bos_override)
    end

    def native_encoder_available? : Bool
      !@token_to_id.empty? && !@bpe_ranks.empty?
    end

    private QWEN35_PRETOKENIZER = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/

    private def encode_native(text : String) : Array(Int32)
      ids = [] of Int32
      partition_special_tokens(text).each do |fragment|
        case fragment
        when Int32
          ids << fragment
        when String
          encode_native_text(fragment, ids)
        end
      end
      ids
    end

    private def encode_native_text(text : String, ids : Array(Int32)) : Nil
      text.scan(QWEN35_PRETOKENIZER) do |match|
        encoded_piece = encode_piece_bytes(match[0])
        bpe(encoded_piece).each do |piece|
          id = @token_to_id[piece]?
          raise "native encode: missing token piece #{piece.inspect}" unless id

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

    private def encode_piece_bytes(piece : String) : String
      String.build do |io|
        piece.to_slice.each do |byte|
          io << self.class.gpt2_char_for(byte)
        end
      end
    end

    private def bpe(piece : String) : Array(String)
      return [piece] if @token_to_id.has_key?(piece)

      word = piece.chars.map(&.to_s)
      return word if word.size <= 1

      loop do
        best_rank = Int32::MAX
        best_pair = nil.as(Tuple(String, String)?)
        (word.size - 1).times do |i|
          pair = {word[i], word[i + 1]}
          if rank = @bpe_ranks[pair]?
            if rank < best_rank
              best_rank = rank
              best_pair = pair
            end
          end
        end

        pair = best_pair
        break unless pair

        merged = [] of String
        i = 0
        while i < word.size
          if i < word.size - 1 && word[i] == pair[0] && word[i + 1] == pair[1]
            merged << word[i] + word[i + 1]
            i += 2
          else
            merged << word[i]
            i += 1
          end
        end
        word = merged
        break if word.size == 1
      end

      word
    end

    private def encode_external(text : String, *, add_bos_override : Bool? = nil) : Array(Int32)
      bin = @llama_tokenize_bin
      raise "encode: llama_tokenize_bin not configured (pass to from_gguf)" if bin.empty?
      raise "encode: llama_tokenize_bin not found at #{bin}" unless File.exists?(bin)

      args = ["-m", @model_path, "-p", text, "--ids", "--log-disable"]
      args << "--no-bos" if (add_bos_override == false) || (add_bos_override.nil? && !@add_bos)

      stdout = IO::Memory.new
      stderr = IO::Memory.new
      status = Process.run(bin, args, output: stdout, error: stderr)
      raise "encode: llama-tokenize exited #{status.exit_code}: #{stderr.to_s}" unless status.success?

      # Output is "[1, 2, 3]" on a single line. Strip brackets and parse.
      line = stdout.to_s.strip
      line = line[1..-2] if line.starts_with?('[') && line.ends_with?(']')
      line.split(',').map { |s| s.strip.to_i32 }
    end

    # --- GPT-2 byte ↔ printable unicode mapping -------------------------------
    # See HuggingFace tokenizers / OpenAI's original GPT-2 bpe.py bytes_to_unicode().
    # Characters in ranges 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map directly.
    # The remaining 68 bytes get shifted into [0x100..0x143].

    @@gpt2_byte_encoder : Hash(UInt8, Char)?
    @@gpt2_byte_decoder : Hash(Char, UInt8)?

    def self.build_gpt2_maps
      bs = [] of UInt8
      cs = [] of Int32
      (0x21_u8..0x7E_u8).each { |b| bs << b; cs << b.to_i32 }
      (0xA1_u8..0xAC_u8).each { |b| bs << b; cs << b.to_i32 }
      (0xAE_u8..0xFF_u8).each { |b| bs << b; cs << b.to_i32 }

      n = 0
      (0..255).each do |b|
        ub = b.to_u8
        unless bs.includes?(ub)
          bs << ub
          cs << (256 + n)
          n += 1
        end
      end

      enc = Hash(UInt8, Char).new
      dec = Hash(Char, UInt8).new
      bs.each_with_index do |b, i|
        ch = cs[i].chr
        enc[b] = ch
        dec[ch] = b
      end
      @@gpt2_byte_encoder = enc
      @@gpt2_byte_decoder = dec
    end

    def self.gpt2_byte_for(ch : Char) : UInt8?
      build_gpt2_maps if @@gpt2_byte_decoder.nil?
      @@gpt2_byte_decoder.not_nil![ch]?
    end

    def self.gpt2_char_for(byte : UInt8) : Char
      build_gpt2_maps if @@gpt2_byte_encoder.nil?
      @@gpt2_byte_encoder.not_nil![byte]
    end
  end
end
