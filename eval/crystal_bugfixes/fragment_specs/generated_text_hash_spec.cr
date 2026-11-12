require "spec"
require "../src/ml/gguf/qwen35_prompt_cache"

describe ML::GGUF::Qwen35PromptCache::Store do
  it "persists a digest of generated text in source history" do
    root = File.tempname("qwen-source-history")
    store = ML::GGUF::Qwen35PromptCache::Store.new(root)
    store.save_source_history(
      session_id: "session", turn_id: "turn", model_id: "model",
      tokenizer_id: "tokenizer", token_ids: [1_i32, 2_i32],
      generated_token_count: 1, generated_text: " answer",
    )
    entry = store.lookup_source_history("session", "model", "tokenizer", turn_id: "turn").not_nil!
    expected = Digest::SHA256.hexdigest("qwen35-generated-text-v1\0 answer")
    JSON.parse(entry.to_json)["generated_text_hash"]?.try(&.as_s).should eq(expected)
  ensure
    FileUtils.rm_rf(root) if root
  end
end
