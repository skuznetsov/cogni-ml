require "./spec_helper"
require "../src/ml/gguf/qwen35_tokenizer"

QWEN_9B_TOK = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

describe ML::GGUF::Qwen35Tokenizer do
  it "GPT-2 byte ↔ char map is lossless roundtrip over all 256 bytes" do
    ML::GGUF::Qwen35Tokenizer.build_gpt2_maps
    256.times do |b|
      ub = b.to_u8
      ch = ML::GGUF::Qwen35Tokenizer.gpt2_char_for(ub)
      back = ML::GGUF::Qwen35Tokenizer.gpt2_byte_for(ch).not_nil!
      back.should eq(ub)
    end
  end

  it "decodes tokens from Qwen 3.5 9B GGUF to 'Hello, world'" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_TOK)
    g = ML::GGUF::GGUFFile.new(QWEN_9B_TOK)
    tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, QWEN_9B_TOK)
    g.close

    # From: llama-tokenize -p "Hello, world" → [9419, 11, 1814]
    text = tok.decode([9419, 11, 1814])
    text.should eq("Hello, world")
  end

  it "round-trips via llama-tokenize bootstrap encoder" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_TOK)
    pending!("llama-tokenize not built") unless File.exists?(LLAMA_TOKENIZE_BIN)

    g = ML::GGUF::GGUFFile.new(QWEN_9B_TOK)
    tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, QWEN_9B_TOK, LLAMA_TOKENIZE_BIN)
    g.close

    ids = tok.encode("Hello, world")
    ids.should eq([9419, 11, 1814])
    tok.decode(ids).should eq("Hello, world")

    # Round-trip on a richer string
    sample = "The capital of France is Paris."
    ids2 = tok.encode(sample)
    tok.decode(ids2).should eq(sample)
  end
end
