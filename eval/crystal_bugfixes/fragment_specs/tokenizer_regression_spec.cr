require "spec"
require "../src/ml/gguf/qwen35_prompt_cache"

describe "Qwen35 direct-output cache tokenizer validation" do
  it "rejects an otherwise valid entry for a different tokenizer" do
    prompt = "prompt"
    prompt_ids = [1_i32]
    output_ids = [2_i32]
    full_hash = ML::GGUF::Qwen35PromptCache.token_hash_concat(prompt_ids, output_ids)
    entry = ML::GGUF::Qwen35PromptCache::OutputFastForwardEntry.new(
      runtime_id: ML::GGUF::Qwen35PromptCache::OUTPUT_FAST_FORWARD_RUNTIME_ID,
      session_id: "session",
      turn_id: "tok-b",
      model_id: "model",
      tokenizer_id: "tok-a",
      prompt_text_hash: ML::GGUF::Qwen35PromptCache.prompt_text_hash(prompt),
      prompt_token_hash: ML::GGUF::Qwen35PromptCache.token_hash(prompt_ids),
      prompt_token_count: prompt_ids.size,
      prompt_token_ids: prompt_ids,
      output_token_hash: ML::GGUF::Qwen35PromptCache.token_hash(output_ids),
      output_token_count: output_ids.size,
      output_token_ids: output_ids,
      full_history_hash: full_hash,
      generated_text: "x",
      generated_text_hash: ML::GGUF::Qwen35PromptCache.generated_text_hash("x"),
      artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
      artifact_validation_steps: output_ids.size,
      artifact_validation_hash: full_hash,
      artifact_prefix_len: 1,
      artifact_token_hash: ML::GGUF::Qwen35PromptCache.token_hash_concat(prompt_ids, output_ids, 1),
      artifact_next_token_id: output_ids.last,
      created_at_unix: Time.utc.to_unix,
    )

    ML::GGUF::Qwen35PromptCache.output_fast_forward_entry_valid?(
      entry, "model", "session", prompt, 1, "tok-b",
    ).should be_false
  end
end
