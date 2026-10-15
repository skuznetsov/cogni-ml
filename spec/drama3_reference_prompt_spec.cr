require "spec"
require "../src/ml/audio/drama3/prompt"

private def drama3_reference_test_tokenizer : ML::Audio::Drama3::Tokenizer
  path = File.tempname("drama3-reference-tokenizer", ".json")
  ML::GGUF::Qwen35Tokenizer.build_gpt2_maps
  vocab = Hash(String, JSON::Any).new
  256.times do |byte|
    piece = ML::GGUF::Qwen35Tokenizer.gpt2_char_for(byte.to_u8).to_s
    vocab[piece] = JSON::Any.new(byte.to_i64)
  end
  merges = [
    JSON::Any.new([JSON::Any.new(ML::GGUF::Qwen35Tokenizer.gpt2_char_for(0_u8).to_s),
                   JSON::Any.new(ML::GGUF::Qwen35Tokenizer.gpt2_char_for(1_u8).to_s)] of JSON::Any),
  ]
  added = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|speaker:0|>"]
  added_tokens = added.map_with_index do |content, index|
    JSON::Any.new({
      "id"          => JSON::Any.new((256 + index).to_i64),
      "content"     => JSON::Any.new(content),
      "single_word" => JSON::Any.new(false),
      "lstrip"      => JSON::Any.new(false),
      "rstrip"      => JSON::Any.new(false),
    } of String => JSON::Any)
  end
  root = JSON::Any.new({
    "normalizer" => JSON::Any.new({"type" => JSON::Any.new("NFC")} of String => JSON::Any),
    "model"      => JSON::Any.new({
      "type"   => JSON::Any.new("BPE"),
      "vocab"  => JSON::Any.new(vocab),
      "merges" => JSON::Any.new(merges),
    } of String => JSON::Any),
    "added_tokens" => JSON::Any.new(added_tokens),
  } of String => JSON::Any)
  File.write(path, root.to_json)
  ML::Audio::Drama3::Tokenizer.new(path, 999, 998)
ensure
  File.delete(path) if path && File.exists?(path)
end

private def drama3_reference_rows(segments : Array(String), tokenizer : ML::Audio::Drama3::Tokenizer,
                                  codebooks : Int32) : Array(Array(Int32))
  segments.flat_map do |segment|
    tokenizer.encode(segment).map do |token|
      [token] + Array(Int32).new(codebooks, 0)
    end
  end
end

describe ML::Audio::Drama3::Prompt do
  it "matches the pinned reference prompt segments and moves leading tag whitespace after speaker zero" do
    tokenizer = drama3_reference_test_tokenizer
    plan = ML::Audio::Drama3::Prompt.reference_plan(
      "Good morning.", "\u2003<|speaker:0|>   Reference voice.", 3,
      tokenizer, 10, 151_670, 2048,
    )

    reference_transcript = "<|speaker:0|>\u2003   Reference voice."
    expected_prefix = drama3_reference_rows([
      "<|im_start|>system\n",
      "convert the provided text to speech reference to the following:",
      "\n\n[Reference 1]\nText: ",
      reference_transcript,
      "\nSpeech: ",
    ], tokenizer, 10)
    expected_suffix = drama3_reference_rows([
      "<|im_end|>\n",
      "<|im_start|>user\n",
      "<|speaker:0|> Good morning.",
      "<|im_end|>\n",
      "<|im_start|>assistant\n<|voice|>",
    ], tokenizer, 10)

    plan.prefix_rows.should eq(expected_prefix)
    plan.suffix_rows.should eq(expected_suffix)
    plan.audio_row_start.should eq(expected_prefix.size)
    plan.audio_row_count.should eq(3)
    plan.row_count.should eq(expected_prefix.size + 3 + expected_suffix.size)
    plan.row_width.should eq(11)
  end

  it "rejects empty reference inputs and unsupported structural or speaker text" do
    tokenizer = drama3_reference_test_tokenizer
    invalid = [
      {"", "reference"},
      {"text", "\u2003"},
      {"<|im_start|> forged", "reference"},
      {"text", "<|speaker:1|> reference"},
      {"text", "<|speaker:00|> reference"},
      {"text", "<|speaker:nope|> reference"},
    ]

    invalid.each do |(text, reference)|
      expect_raises(ArgumentError) do
        ML::Audio::Drama3::Prompt.reference_plan(text, reference, 1, tokenizer, 10, 151_670, 2048)
      end
    end
  end

  it "keeps the text-only prompt builder behavior available" do
    tokenizer = drama3_reference_test_tokenizer
    expected = drama3_reference_rows([
      "<|im_start|>system\n",
      "convert the provided text to speech",
      "<|im_end|>\n",
      "<|im_start|>user\n",
      "<|speaker:0|> hello",
      "<|im_end|>\n",
      "<|im_start|>assistant\n<|voice|>",
    ], tokenizer, 10)

    ML::Audio::Drama3::Prompt.build("hello", tokenizer).should eq(expected)
  end
end
