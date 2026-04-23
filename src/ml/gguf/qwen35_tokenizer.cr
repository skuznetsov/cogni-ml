require "./reader"

# Qwen 3.5 / 3.6 tokenizer (GPT-2-style BPE).
#
# Current scope: DECODER + bootstrap ENCODER.
# - Decoder is native Crystal: uses tokenizer.ggml.tokens[] from GGUF, handles
#   Ġ-for-space and Ċ-for-newline conventions.
# - Encoder shells out to llama.cpp's `llama-tokenize` binary (bootstrap only);
#   a native pure-Crystal BPE encoder is deferred (Phase 1b.7 goal, but the
#   decoder is the critical piece for end-to-end verification).
module ML::GGUF
  class Qwen35Tokenizer
    getter vocab : Array(String)
    getter eos_id : Int32
    getter pad_id : Int32
    getter? add_bos : Bool
    getter llama_tokenize_bin : String
    getter model_path : String

    def initialize(@vocab : Array(String), @eos_id : Int32, @pad_id : Int32,
                   @add_bos : Bool, @model_path : String,
                   @llama_tokenize_bin : String = "")
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

      new(vocab, eos, pad, add_bos, model_path, llama_tokenize_bin)
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
