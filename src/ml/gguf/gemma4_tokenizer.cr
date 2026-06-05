require "./reader"

module ML::GGUF
  # Minimal Gemma4 tokenizer bridge.
  #
  # Encoding intentionally delegates to llama.cpp's tokenizer oracle. Gemma4 is
  # a non-byte-encoded BPE variant with model-specific pre-tokenization, so a
  # partial native encoder would be too easy to trust accidentally.
  #
  # Decoding is conservative and intended for qualitative probe output: normal
  # pieces are concatenated, SentencePiece-style "▁" is rendered as a space,
  # byte-fallback pieces like <0x0A> are decoded, and control tokens are skipped.
  class Gemma4Tokenizer
    getter vocab : Array(String)
    getter token_types : Array(Int32)
    getter bos_id : Int32
    getter eos_id : Int32
    getter pad_id : Int32
    getter unk_id : Int32
    getter model_path : String
    getter llama_tokenize_bin : String

    def initialize(@vocab : Array(String),
                   @token_types : Array(Int32),
                   @bos_id : Int32,
                   @eos_id : Int32,
                   @pad_id : Int32,
                   @unk_id : Int32,
                   @model_path : String,
                   @llama_tokenize_bin : String)
    end

    def self.from_gguf(g : GGUFFile, model_path : String,
                       llama_tokenize_bin : String) : Gemma4Tokenizer
      model = g.get_string("tokenizer.ggml.model") || ""
      raise "gemma4_tokenizer: expected tokenizer.ggml.model=gemma4, got #{model.inspect}" unless model == "gemma4"

      tokens_raw = g.metadata["tokenizer.ggml.tokens"]?
      raise "gemma4_tokenizer: missing tokenizer.ggml.tokens" unless tokens_raw.is_a?(Array)
      vocab = tokens_raw.map { |t| t.as(String) }

      types_raw = g.metadata["tokenizer.ggml.token_type"]?
      token_types = if types_raw.is_a?(Array)
                      types_raw.map { |v| v.as?(Int32) || v.as?(UInt32).try(&.to_i32) || 0 }
                    else
                      Array(Int32).new(vocab.size, 0)
                    end

      bos = g.get_int("tokenizer.ggml.bos_token_id").try(&.to_i32) || 2
      eos = g.get_int("tokenizer.ggml.eos_token_id").try(&.to_i32) || 1
      pad = g.get_int("tokenizer.ggml.padding_token_id").try(&.to_i32) || 0
      unk = g.get_int("tokenizer.ggml.unknown_token_id").try(&.to_i32) || 3

      new(vocab, token_types, bos, eos, pad, unk, model_path, llama_tokenize_bin)
    end

    def encode(text : String, *, add_bos : Bool = true, parse_special : Bool = true) : Array(Int32)
      raise "encode: llama_tokenize_bin not configured" if @llama_tokenize_bin.empty?
      raise "encode: llama_tokenize_bin not found at #{@llama_tokenize_bin}" unless File.exists?(@llama_tokenize_bin)
      raise "encode: model not found at #{@model_path}" unless File.exists?(@model_path)

      stdout = IO::Memory.new
      stderr = IO::Memory.new
      args = ["-m", @model_path, "--stdin", "--ids", "--log-disable"]
      args << "--no-bos" unless add_bos
      args << "--no-parse-special" unless parse_special
      status = Process.run(@llama_tokenize_bin, args, input: IO::Memory.new(text), output: stdout, error: stderr)
      raise "encode: llama-tokenize exited #{status.exit_code}: #{stderr.to_s}" unless status.success?

      parse_id_list(stdout.to_s)
    end

    def decode(ids : Array(Int32), *, skip_special : Bool = true) : String
      bytes = [] of UInt8
      ids.each do |id|
        next if skip_special && special_id?(id)
        raise "decode: token id #{id} out of range (vocab=#{@vocab.size})" if id < 0 || id >= @vocab.size

        piece = @vocab[id]
        next if skip_special && control_piece?(id, piece)

        if byte = byte_fallback(piece)
          bytes << byte
          next
        end

        piece = piece.gsub("▁", " ")
        piece.to_slice.each { |b| bytes << b }
      end

      clean_spaces(String.new(Slice.new(bytes.to_unsafe, bytes.size)))
    end

    def decode_single(id : Int32) : String
      decode([id])
    end

    private def parse_id_list(output : String) : Array(Int32)
      line = output.lines.find { |l| l.strip.starts_with?('[') } || output.strip
      line = line.strip
      line = line[1..-2] if line.starts_with?('[') && line.ends_with?(']')
      return [] of Int32 if line.empty?
      line.split(',').reject(&.strip.empty?).map { |s| s.strip.to_i32 }
    end

    private def special_id?(id : Int32) : Bool
      id == @bos_id || id == @eos_id || id == @pad_id || id == @unk_id
    end

    private def control_piece?(id : Int32, piece : String) : Bool
      return true if @token_types[id]? == 3
      piece.starts_with?('<') && piece.ends_with?('>')
    end

    private def byte_fallback(piece : String) : UInt8?
      return nil unless piece.size == 6 && piece.starts_with?("<0x") && piece.ends_with?('>')
      hex = piece[3, 2]
      hex.to_i(16).to_u8
    rescue
      nil
    end

    private def clean_spaces(text : String) : String
      text
        .gsub(" ?", "?")
        .gsub(" !", "!")
        .gsub(" .", ".")
        .gsub(" ,", ",")
    end
  end
end
