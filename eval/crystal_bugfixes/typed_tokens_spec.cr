require "spec"
require "../src/ml/gguf/qwen35_tokenizer"

describe ML::GGUF::Qwen35Tokenizer do
  it "partitions GGML special token types before native BPE" do
    path = File.tempname("typed-tokens", ".gguf")
    File.open(path, "wb") do |file|
      file << "GGUF"
      file.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
      file.write_bytes(0_i64, IO::ByteFormat::LittleEndian)
      file.write_bytes(0_i64, IO::ByteFormat::LittleEndian)
    end

    gguf = ML::GGUF::GGUFFile.new(path)
    begin
      # Exercise the same public loader on both revisions without a model file.
      gguf.metadata["tokenizer.ggml.tokens"] =
        ["a", "b", "ab", "<x>", "<x>long", "<unk>", "<", "x", ">", "l", "o", "n", "g", "u", "k"] of ML::GGUF::Value
      gguf.metadata["tokenizer.ggml.merges"] = ["a b"] of ML::GGUF::Value
      gguf.metadata["tokenizer.ggml.token_type"] =
        [1, 1, 1, 3, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1] of ML::GGUF::Value

      tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, path)
      text = "ab<x>ab<x>long<unk>ab"
      ids = tokenizer.encode(text, add_bos_override: false)
      ids.should eq([2, 3, 2, 4, 5, 2])
      tokenizer.decode(ids).should eq(text)
    ensure
      gguf.close
      File.delete(path)
    end
  end
end
