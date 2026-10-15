require "./native"
require "../../gguf/qwen35_tokenizer"

module ML::Audio::Drama3
  # Reuse the native Qwen byte-BPE implementation, with the checkpoint's NFC
  # normalization and explicit special token IDs. No external tokenizer process.
  class Tokenizer
    @bpe : ML::GGUF::Qwen35Tokenizer

    def initialize(path : String, eos : Int32, pad : Int32)
      root = JSON.parse(File.read(path))
      raise ArgumentError.new("unsupported tokenizer normalizer") unless root["normalizer"]["type"].as_s == "NFC"
      model = root["model"]
      raise ArgumentError.new("unsupported tokenizer model") unless model["type"].as_s == "BPE" && model["dropout"]?.try(&.raw).nil?
      vocab_hash = model["vocab"].as_h
      added = root["added_tokens"].as_a
      max_id = (vocab_hash.values.map(&.as_i) + added.map { |entry| entry["id"].as_i }).max
      vocab = Array(String).new(max_id + 1, "")
      types = Array(Int32).new(max_id + 1, 0)
      vocab_hash.each { |piece, id| vocab[id.as_i] = piece }
      added.each do |entry|
        id = entry["id"].as_i
        raise ArgumentError.new("unsupported added token flags") if entry["single_word"].as_bool || entry["lstrip"].as_bool || entry["rstrip"].as_bool
        vocab[id] = entry["content"].as_s
        types[id] = 3
      end
      raise ArgumentError.new("non-contiguous tokenizer vocabulary") if vocab.any?(&.empty?)
      ranks = {} of Tuple(String, String) => Int32
      model["merges"].as_a.each_with_index do |merge, i|
        pair = merge.as_a.map(&.as_s)
        raise ArgumentError.new("malformed BPE merge") unless pair.size == 2
        ranks[{pair[0], pair[1]}] = i
      end
      mapping = {} of String => Int32
      vocab.each_with_index { |v, i| mapping[v] = i }
      @bpe = ML::GGUF::Qwen35Tokenizer.new(vocab, eos, pad, false, path, token_to_id: mapping, bpe_ranks: ranks, token_types: types)
    end

    def token_id(piece : String) : Int32
      @bpe.token_to_id[piece]? || raise ArgumentError.new("missing required token #{piece}")
    end

    def encode(text : String) : Array(Int32)
      raise ArgumentError.new("native tokenizer disabled by QWEN35_NATIVE_TOKENIZER_OFF") if ENV["QWEN35_NATIVE_TOKENIZER_OFF"]? == "1"
      @bpe.encode(text.unicode_normalize(:nfc))
    end
  end

  module Prompt
    extend self
    STRUCTURAL            = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]
    REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
    SPEAKER_ZERO          = "<|speaker:0|>"
    MESSAGE_END           = "<|im_end|>\n"

    # Host-built text rows and the exact insertion point for reference codes.
    # Audio rows are filled by PromptResident directly from the encoder buffer.
    class ReferencePlan
      getter prefix_rows : Array(Array(Int32))
      getter suffix_rows : Array(Array(Int32))
      getter audio_row_start : Int32
      getter audio_row_count : Int32
      getter row_width : Int32
      getter codebooks : Int32
      getter semantic_begin : Int32
      getter codebook_size : Int32

      def initialize(@prefix_rows : Array(Array(Int32)), @suffix_rows : Array(Array(Int32)),
                     @audio_row_count : Int32, @row_width : Int32, @codebooks : Int32,
                     @semantic_begin : Int32, @codebook_size : Int32)
        @audio_row_start = @prefix_rows.size
        raise ArgumentError.new("invalid reference prompt plan") unless @audio_row_count > 0 &&
                                                                        @codebooks > 0 && @codebooks < Int32::MAX && @row_width == @codebooks + 1 &&
                                                                        @codebook_size > 0 && @semantic_begin >= 0 &&
                                                                        @semantic_begin.to_i64 + @codebook_size - 1 <= 16_777_215_i64
        raise ArgumentError.new("prompt exceeds 8192 rows") if @prefix_rows.size.to_i64 + @audio_row_count + @suffix_rows.size > 8192
      end

      def row_count : Int32
        @prefix_rows.size + @audio_row_count + @suffix_rows.size
      end
    end

    def build(text : String, tokenizer : Tokenizer, codebooks : Int32 = 10) : Array(Array(Int32))
      raise ArgumentError.new("text must not be empty") if text.strip.empty?
      raise ArgumentError.new("text contains a structural prompt token") if STRUCTURAL.any? { |token| text.includes?(token) }
      raise ArgumentError.new("initial native slice supports only complete speaker:0 tags") if text.gsub("<|speaker:0|>", "").includes?("<|speaker")
      tagged = text.lstrip.starts_with?("<|speaker:0|>") ? text : "<|speaker:0|> #{text}"
      # Segment boundaries are semantically relevant to BPE: never concatenate
      # these strings before tokenization.
      segments = ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                  "<|im_start|>user\n", tagged, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
      ids = segments.flat_map { |s| tokenizer.encode(s) }
      raise ArgumentError.new("prompt exceeds 8192 rows") if ids.size > 8192
      ids.map { |id| [id] + Array(Int32).new(codebooks, 0) }
    end

    # Native one-reference prompt matching the pinned MLX builder. Every text
    # literal is tokenized independently; the reference code span is inserted
    # between the `Speech:` prefix and the message end.
    def reference_plan(text : String, ref_text : String, ref_frames : Int32, tokenizer : Tokenizer,
                       codebooks : Int32, semantic_begin : Int32, codebook_size : Int32) : ReferencePlan
      validate_reference_text!(text, "text")
      validate_reference_text!(ref_text, "reference 1")
      raise ArgumentError.new("reference codes must have at least one frame") unless ref_frames > 0
      raise ArgumentError.new("invalid reference codebook count") unless codebooks > 0 && codebooks < Int32::MAX
      raise ArgumentError.new("invalid reference codebook size") unless codebook_size > 0
      raise ArgumentError.new("invalid semantic token band") unless semantic_begin >= 0 &&
                                                                    semantic_begin.to_i64 + codebook_size - 1 <= 16_777_215_i64

      tagged_reference = with_speaker_zero(ref_text)
      body = unicode_lstrip(tagged_reference)
      leading_bytes = tagged_reference.bytesize - body.bytesize
      leading = tagged_reference.byte_slice(0, leading_bytes) || ""
      after_tag = body.byte_slice(SPEAKER_ZERO.bytesize, body.bytesize - SPEAKER_ZERO.bytesize) || ""
      canonical_reference = SPEAKER_ZERO + leading + after_tag

      prefix_segments = [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION,
        "\n\n[Reference 1]\nText: ",
        canonical_reference,
        "\nSpeech: ",
      ]
      suffix_segments = [
        MESSAGE_END,
        "<|im_start|>user\n",
        with_speaker_zero(text),
        MESSAGE_END,
        "<|im_start|>assistant\n<|voice|>",
      ]
      prefix_rows = tokenize_rows(prefix_segments, tokenizer, codebooks)
      suffix_rows = tokenize_rows(suffix_segments, tokenizer, codebooks)
      rows = prefix_rows.size.to_i64 + ref_frames + suffix_rows.size
      raise ArgumentError.new("prompt exceeds 8192 rows") if rows > 8192

      ReferencePlan.new(prefix_rows, suffix_rows, ref_frames, codebooks + 1, codebooks,
        semantic_begin, codebook_size)
    end

    private def validate_reference_text!(text : String, location : String) : Nil
      raise ArgumentError.new("#{location}: text must not be empty") if text.each_char.all? { |char| unicode_white_space?(char) }
      raise ArgumentError.new("#{location}: structural prompt tokens are written by the prompt builder") if STRUCTURAL.any? { |token| text.includes?(token) }

      offset = 0
      while start = text.byte_index("<|speaker", offset)
        tail = text.byte_slice(start, text.bytesize - start) || ""
        unless tail.starts_with?(SPEAKER_ZERO)
          raise ArgumentError.new("#{location}: this native slice supports only complete speaker:0 tags")
        end
        offset = start + SPEAKER_ZERO.bytesize
      end
    end

    private def tokenize_rows(segments : Array(String), tokenizer : Tokenizer, codebooks : Int32) : Array(Array(Int32))
      rows = [] of Array(Int32)
      segments.each do |segment|
        tokenizer.encode(segment).each do |id|
          rows << [id] + Array(Int32).new(codebooks, 0)
        end
      end
      rows
    end

    private def with_speaker_zero(text : String) : String
      unicode_lstrip(text).starts_with?(SPEAKER_ZERO) ? text : "#{SPEAKER_ZERO} #{text}"
    end

    private def unicode_lstrip(text : String) : String
      chars = text.chars
      offset = 0
      while offset < chars.size && unicode_white_space?(chars[offset])
        offset += 1
      end
      String.build do |io|
        chars.each_with_index { |char, index| io << char if index >= offset }
      end
    end

    private def unicode_white_space?(char : Char) : Bool
      code = char.ord
      code.in?(0x09..0x0D) || code == 0x20 || code == 0x85 || code == 0xA0 ||
        code == 0x1680 || code.in?(0x2000..0x200A) || code.in?(0x2028..0x2029) ||
        code == 0x202F || code == 0x205F || code == 0x3000
    end
  end
end
