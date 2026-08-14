require "./spec_helper"
require "../src/ml/gguf/qwen35_tokenizer"

QWEN_9B_TOK        = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN_38_27B_TOK    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
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

  it "native BPE encoder matches llama-tokenize on representative Qwen3.5 prompts" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_TOK)
    pending!("llama-tokenize not built") unless File.exists?(LLAMA_TOKENIZE_BIN)

    g = ML::GGUF::GGUFFile.new(QWEN_9B_TOK)
    tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, QWEN_9B_TOK, LLAMA_TOKENIZE_BIN)
    g.close

    tok.native_encoder_available?.should be_true

    samples = [
      "Hello, world",
      "The capital of France is Paris.",
      "alpha beta gamma delta alpha beta gamma delta",
      "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
      "YAML:\n- host: api-1\n- host: api-2\n",
      "Unicode: café, naïve, Привет, こんにちは",
      "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    ]

    old = ENV["QWEN35_NATIVE_TOKENIZER_OFF"]?
    samples.each do |sample|
      ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
      native = tok.encode(sample)
      ENV["QWEN35_NATIVE_TOKENIZER_OFF"] = "1"
      external = tok.encode(sample)
      native.should eq(external)
      tok.decode(native).should eq(sample)
    end
  ensure
    if old
      ENV["QWEN35_NATIVE_TOKENIZER_OFF"] = old
    else
      ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
    end
  end

  it "partitions GGML special token types before native BPE" do
    vocab = ["a", "b", "ab", "<x>", "<x>long", "<unk>"]
    token_to_id = {} of String => Int32
    vocab.each_with_index { |piece, id| token_to_id[piece] = id.to_i32 }
    tok = ML::GGUF::Qwen35Tokenizer.new(
      vocab,
      eos_id: 3,
      pad_id: 3,
      add_bos: false,
      model_path: "fake.gguf",
      token_to_id: token_to_id,
      bpe_ranks: { {"a", "b"} => 0 },
      token_types: [1, 1, 1, 3, 4, 2],
    )

    text = "ab<x>ab<x>long<unk>ab"
    ids = tok.encode(text, add_bos_override: false)
    ids.should eq([2, 3, 2, 4, 5, 2])
    tok.decode(ids).should eq(text)
  end

  it "matches llama-tokenize for Qwen3.8 chat control tokens" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_27B_TOK)
    pending!("llama-tokenize not built") unless File.exists?(LLAMA_TOKENIZE_BIN)

    g = ML::GGUF::GGUFFile.new(QWEN_38_27B_TOK)
    tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, QWEN_38_27B_TOK, LLAMA_TOKENIZE_BIN)
    g.close

    rendered = "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
    old = ENV["QWEN35_NATIVE_TOKENIZER_OFF"]?

    ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
    native = tok.encode(rendered, add_bos_override: false)
    ENV["QWEN35_NATIVE_TOKENIZER_OFF"] = "1"
    external = tok.encode(rendered, add_bos_override: false)

    external.should eq([248045, 846, 198, 17, 10, 17, 30, 248046, 198, 248045, 74455, 198])
    native.should eq(external)

    mixed = "é<|im_start|>user\nkeep <|im_startX|> literal <think>x</think><|im_end|>"
    ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
    native_mixed = tok.encode(mixed, add_bos_override: false)
    ENV["QWEN35_NATIVE_TOKENIZER_OFF"] = "1"
    external_mixed = tok.encode(mixed, add_bos_override: false)
    native_mixed.should eq(external_mixed)

    special_pieces = [] of String
    tok.vocab.each_with_index do |piece, id|
      token_type = tok.token_types[id]? || 0
      special_pieces << piece if token_type == 2 || token_type == 3 || token_type == 4
    end
    special_pieces.should_not be_empty
    all_specials = special_pieces.join(" alpha ")
    ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
    native_all = tok.encode(all_specials, add_bos_override: false)
    ENV["QWEN35_NATIVE_TOKENIZER_OFF"] = "1"
    external_all = tok.encode(all_specials, add_bos_override: false)
    native_all.should eq(external_all)
  ensure
    if old
      ENV["QWEN35_NATIVE_TOKENIZER_OFF"] = old
    else
      ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
    end
  end
end
