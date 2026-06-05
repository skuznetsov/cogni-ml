require "./spec_helper"
require "../src/ml/gguf/gemma4_tokenizer"

GEMMA4_TOK_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
GEMMA4_LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

describe ML::GGUF::Gemma4Tokenizer do
  it "oracle-encodes representative Gemma4 prompts and conservatively decodes them" do
    pending!("Gemma4 model not present") unless File.exists?(GEMMA4_TOK_MODEL)
    pending!("llama-tokenize not built") unless File.exists?(GEMMA4_LLAMA_TOKENIZE_BIN)

    g = ML::GGUF::GGUFFile.new(GEMMA4_TOK_MODEL)
    tok = ML::GGUF::Gemma4Tokenizer.from_gguf(g, GEMMA4_TOK_MODEL, GEMMA4_LLAMA_TOKENIZE_BIN)
    g.close

    samples = [
      "Hello, world",
      "def add(a, b):\n    return a + b\n",
      "Explain why cached session state changes prefill speed.",
    ]

    samples.each do |sample|
      ids = tok.encode(sample)
      ids.should_not be_empty
      decoded = tok.decode(ids)
      decoded.should contain(sample[0, Math.min(sample.size, 12)].strip)
    end
  end

  it "decodes Gemma4 byte fallback pieces" do
    vocab = ["<pad>", "<eos>", "<bos>", "<unk>", "<0x0A>", "hello"]
    types = [3, 3, 3, 2, 6, 1]
    tok = ML::GGUF::Gemma4Tokenizer.new(vocab, types, 2, 1, 0, 3, "model.gguf", "llama-tokenize")

    tok.decode([5, 4], skip_special: false).should eq("hello\n")
  end
end
