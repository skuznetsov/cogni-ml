require "spec"
require "digest/sha256"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen25_hy_nomas_prompt"

describe ML::GGUF::Qwen35Tokenizer do
  it "accepts a per-checkpoint pretokenizer without changing its Qwen35 default" do
    ML::GGUF::Qwen35Tokenizer.build_gpt2_maps
    accent_bytes = "\u0301".to_slice.map { |byte| ML::GGUF::Qwen35Tokenizer.gpt2_char_for(byte) }.join
    merged_once = "a" + accent_bytes.chars[0].to_s
    merged = merged_once + accent_bytes.chars[1].to_s
    vocab = ["a", accent_bytes, merged]
    token_to_id = vocab.each_with_index.to_h { |piece, index| {piece, index.to_i32} }
    ranks = {
      {"a", accent_bytes.chars[0].to_s}         => 0_i32,
      {merged_once, accent_bytes.chars[1].to_s} => 1_i32,
    }

    qwen35 = ML::GGUF::Qwen35Tokenizer.new(
      vocab, 0, 0, false, "fake.gguf", token_to_id: token_to_id, bpe_ranks: ranks)
    qwen25 = ML::GGUF::Qwen35Tokenizer.new(
      vocab, 0, 0, false, "fake.json", token_to_id: token_to_id, bpe_ranks: ranks,
      pretokenizer: /\p{L}+|\p{M}+|[^\s\p{L}\p{N}\p{M}]+|\s+/,
    )

    qwen35.encode("a\u0301").should eq([2])
    qwen25.encode("a\u0301").should eq([0, 1])
  end
end

describe ML::GGUF::Qwen25HyNomasPrompt do
  it "rejects a changed tokenizer asset before parsing it" do
    root = File.join(Dir.tempdir, "qwen25-prompt-assets-#{Random.rand(1_000_000_000)}")
    Dir.mkdir(root)
    begin
      File.write(File.join(root, "tokenizer.json"), "{}\n")
      expect_raises(ArgumentError, /digest does not match/) do
        ML::GGUF::Qwen25HyNomasPrompt.new(root)
      end
    ensure
      File.delete(File.join(root, "tokenizer.json")) if File.exists?(File.join(root, "tokenizer.json"))
      Dir.delete(root)
    end
  end

  it "rejects tokenizer assets linked outside the checkpoint root" do
    root = File.join(Dir.tempdir, "qwen25-prompt-link-#{Random.rand(1_000_000_000)}")
    outside = File.join(Dir.tempdir, "qwen25-prompt-outside-#{Random.rand(1_000_000_000)}")
    Dir.mkdir(root)
    File.write(outside, "{}\n")
    begin
      File.symlink(outside, File.join(root, "tokenizer.json"))
      expect_raises(ArgumentError, /regular file directly inside/) do
        ML::GGUF::Qwen25HyNomasPrompt.new(root)
      end
    ensure
      File.delete(File.join(root, "tokenizer.json")) if File.exists?(File.join(root, "tokenizer.json")) || File.symlink?(File.join(root, "tokenizer.json"))
      File.delete(outside) if File.exists?(outside)
      Dir.delete(root)
    end
  end
end

# Real-tokenizer cases are opt-in because they read the pinned local tokenizer
# JSON files. They never load model weights or invoke Metal.
if checkpoint_root = ENV["QWEN25_PROMPT_CHECKPOINT_ROOT"]?
  describe "Qwen2.5 Hy-Nomas prompt preparation with pinned tokenizer assets" do
    it "matches the natural prompt crop and token/mask snapshots" do
      prompt = "A yellow tram turns through a wet downtown intersection at dusk as pedestrians wait under umbrellas."
      prepared = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root).prepare(prompt)

      prepared.crop_start.should eq(108_i32)
      prepared.raw_length.should eq(1388_i32)
      prepared.text_length.should eq(1280_i32)
      prepared.marker_token_ids.should eq([151644_i32, 872_i32, 198_i32])
      prepared.attention_mask.count(1_i32).should eq(131)
      prepared.cropped_attention_mask.count(1_i32).should eq(23)
      Digest::SHA256.hexdigest(prepared.prompt).should eq("91246a3eaf20efb68e83f9c6fe1e91afaedd8013a77971ebd74546fe2ed1e5a9")
      Digest::SHA256.hexdigest(prepared.rendered_prompt).should eq("ab7928a9196c40f714755cfd44cdcd95a9eb8b6f2939bceb1548d52168db688d")
      Digest::SHA256.hexdigest(prepared.tokens.to_json).should eq("1b38d85972055a0d257dcaea32432c79d5a945ed89f732fd3a6b078b86b29394")
      Digest::SHA256.hexdigest(prepared.attention_mask.to_json).should eq("8fe94ab8df0594673964cee38d517a43efcb1e1678635d392239286af2ce467a")
      Digest::SHA256.hexdigest(prepared.cropped_tokens.to_json).should eq("71fe7a2ee924e5205b825aad0f7f7b5a12bb7d247bc1a5a6aea0eab9ea7c0156")
      Digest::SHA256.hexdigest(prepared.cropped_attention_mask.to_json).should eq("eb53ab5d848e7635612786fea4aeeecc557c32262ffa763ae57ac19e2831e330")
      prepared.positions.should eq(Array(Float32).new(1388) { |index| index.to_f32 })
    end

    it "normalizes decomposed Unicode, preserves newlines, and pads on the right" do
      tokenizer = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root)
      composed = tokenizer.prepare("café")
      decomposed = tokenizer.prepare("cafe\u0301")
      composed.tokens.should eq(decomposed.tokens)

      with_newline = tokenizer.prepare("rain\nfalls")
      with_newline.raw_length.should eq(1388_i32)
      with_newline.tokens.size.should eq(1388)
      with_newline.attention_mask.size.should eq(1388)
      first_padding = with_newline.attention_mask.index(0_i32).not_nil!
      with_newline.attention_mask[first_padding..].should eq(Array(Int32).new(1388 - first_padding, 0_i32))
      with_newline.attention_mask[0...first_padding].should eq(Array(Int32).new(first_padding, 1_i32))
      with_newline.positions.last.should eq(1387_f32)
    end

    it "truncates at the raw width and rejects multimodal tokens and invalid prompt bytes" do
      tokenizer = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root)
      long_prompt = tokenizer.prepare("tram " * 4000)
      long_prompt.tokens.size.should eq(1388)
      long_prompt.attention_mask.should eq(Array(Int32).new(1388, 1_i32))
      long_prompt.cropped_tokens.size.should eq(1280)

      expect_raises(ArgumentError, /multimodal token/) { tokenizer.prepare("look <|video_pad|>") }
      expect_raises(ArgumentError, /must not contain NUL/) { tokenizer.prepare("bad\0prompt") }
      expect_raises(ArgumentError, /exceeds 64 KiB/) { tokenizer.prepare("x" * 65_537) }
    end

    it "maps an empty prompt to the upstream single-space user content" do
      prepared = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root).prepare("")
      prepared.prompt.should eq("")
      prepared.messages[1][:content].should eq(" ")
      prepared.rendered_prompt.should contain("<|im_start|>user\n <|im_end|>")
    end
  end
end
