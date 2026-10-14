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
    STRUCTURAL = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]

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
  end
end
