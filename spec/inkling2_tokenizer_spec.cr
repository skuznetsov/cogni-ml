require "./spec_helper"
require "../src/ml/gguf/inkling2_tokenizer"

INKLING2_TOKENIZER_MODEL = ENV["INKLING2_TOKENIZER_TEST_MODEL"]? || begin
  complete = "/opt/mldata_ssd/cogni-ml/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf"
  partial = "#{complete}.part"
  File.exists?(complete) ? complete : partial
end
INKLING2_LLAMA_TOKENIZE = "/private/tmp/cogni-inkling-reference.dXTu80/build/bin/llama-tokenize"

private def inkling2_byte_vocab(extra : Array(String) = [] of String) : Array(String)
  vocab = Array(String).new(256) { |byte| ML::GGUF::GPT2BPE.char_for(byte.to_u8).to_s }
  vocab.concat(extra)
  vocab
end

describe ML::GGUF::Inkling2Tokenizer do
  it "round-trips Unicode, combining marks, tabs, and CRLF through byte BPE" do
    vocab = inkling2_byte_vocab
    types = Array(Int32).new(vocab.size, 1)
    tok = ML::GGUF::Inkling2Tokenizer.new(vocab, token_types: types, eos_id: 0, pad_id: 1)

    text = "ASCII café e\u0301 Привет こんにちは 👋\tend\r\n"
    ids = tok.encode(text)
    ids.should eq(text.to_slice.map(&.to_i32).to_a)
    tok.decode(ids).should eq(text)
  end

  it "uses the GPT4O one-to-three digit boundary" do
    vocab = inkling2_byte_vocab(["123", "1234"] of String)
    types = Array(Int32).new(vocab.size, 1)
    tok = ML::GGUF::Inkling2Tokenizer.new(vocab, token_types: types, eos_id: 0, pad_id: 1)

    # GPT4O splits 1234 into "123" and "4"; an unbounded number run would
    # select the deliberately present "1234" token instead.
    tok.encode("1234").should eq([256, '4'.ord.to_i32])
  end

  it "merges the lowest-ranked pair repeatedly, independent of a later pair" do
    vocab = inkling2_byte_vocab(["bc", "abc"] of String)
    types = Array(Int32).new(vocab.size, 1)
    tok = ML::GGUF::Inkling2Tokenizer.new(
      vocab,
      merges: ["b c", "a b", "a bc"],
      token_types: types,
      eos_id: 0,
      pad_id: 1,
    )

    tok.encode("abc").should eq([257])
  end

  it "partitions longest special tokens and decodes them only when requested" do
    vocab = inkling2_byte_vocab(["<|z|>", "<|z|>long"] of String)
    types = Array(Int32).new(vocab.size, 1)
    types[256] = 3
    types[257] = 3
    tok = ML::GGUF::Inkling2Tokenizer.new(vocab, token_types: types, eos_id: 0, pad_id: 1)

    text = "é<|z|>X<|z|>long"
    ids = tok.encode(text)
    ids.should eq([195, 169, 256, 'X'.ord.to_i32, 257])
    tok.decode(ids).should eq("éX")
    tok.decode(ids, skip_special: false).should eq(text)
  end

  it "renders the simple chat path from the model's actual template" do
    pending!("Inkling 2 Mini GGUF metadata not present") unless File.exists?(INKLING2_TOKENIZER_MODEL)

    gguf = ML::GGUF::GGUFFile.new(INKLING2_TOKENIZER_MODEL, mmap_tensors: false)
    tok = ML::GGUF::Inkling2Tokenizer.from_gguf(gguf)
    tok.chat_template_supported?.should be_true
    tok.eos_id.should eq(200006)
    tok.decode([tok.eos_id]).should eq("")
    tok.decode([tok.eos_id], skip_special: false).should eq("<|content_model_end_sampling|>")
    tok.render_user_prompt("2+2?").should eq(
      "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" +
      "<|message_user|><|content_text|>2+2?<|end_message|>"
    )
    tok.render_user_prompt("2+2?").should_not contain("<|message_model|>")
  ensure
    gguf.try(&.close)
  end

  it "matches the pinned llama-tokenize oracle on Unicode and boundary cases" do
    complete_model = "/opt/mldata_ssd/cogni-ml/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf"
    pending!("complete Inkling 2 Mini GGUF not present") unless File.exists?(complete_model)
    pending!("pinned llama-tokenize not built") unless File.exists?(INKLING2_LLAMA_TOKENIZE)

    samples = [
      "Hello, world",
      "UPPER lower 1234 1,2345",
      "can't I'LL they'd\n\r\n",
      "café e\u0301 Привет こんにちは 👋",
      " trailing and trailing \t",
      "<|message_user|>literal control boundary",
    ]
    text = samples.join("<|end_message|>")

    gguf = ML::GGUF::GGUFFile.new(complete_model, mmap_tensors: false)
    tok = ML::GGUF::Inkling2Tokenizer.from_gguf(gguf)
    chat_prompt = tok.render_user_prompt("2+2?")
    text = text + "<|end_message|>" + chat_prompt
    native = tok.encode(text)

    stdout = IO::Memory.new
    stderr = IO::Memory.new
    status = Process.run(
      INKLING2_LLAMA_TOKENIZE,
      ["-m", complete_model, "-p", text, "--ids", "--no-bos", "--log-disable"],
      output: stdout,
      error: stderr,
    )
    status.success?.should be_true, stderr.to_s

    oracle = stdout.to_s.strip[1..-2].split(',').reject(&.strip.empty?).map { |id| id.strip.to_i32 }
    native.should eq(oracle)
  ensure
    gguf.try(&.close)
  end
end
