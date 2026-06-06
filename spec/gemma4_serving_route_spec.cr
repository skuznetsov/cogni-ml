require "./spec_helper"
require "file_utils"
require "../src/ml/gguf/gemma4_serving_route"

private def gemma4_serving_exact_entry(model_id : String, tokenizer_id : String, full_history : Array(Int32), output_ids : Array(Int32))
  ML::GGUF::Gemma4PromptCache::Entry.new(
    runtime_id: ML::GGUF::Gemma4PromptCache::RUNTIME_ID,
    session_id: "s1",
    turn_id: nil,
    model_id: model_id,
    tokenizer_id: tokenizer_id,
    prompt_hash: "unused",
    token_hash: ML::GGUF::Gemma4PromptCache.token_hash(full_history, full_history.size - 1),
    prefix_len: full_history.size - 1,
    max_seq: 16,
    layer_count: 1,
    kv_dims: [2_i32],
    artifact_path: "artifact.gkv",
    artifact_sha256: "0" * 64,
    artifact_byte_size: 0_i64,
    state_byte_size: 0_i64,
    created_at_unix: Time.utc.to_unix,
    prompt_preview: nil,
    next_token_id: output_ids[-1],
    artifact_validation_kind: ML::GGUF::Gemma4PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
    artifact_validation_steps: output_ids.size,
    artifact_validation_hash: ML::GGUF::Gemma4PromptCache.token_hash(full_history),
  )
end

describe ML::GGUF::Gemma4ServingRoute do
  it "serves direct output certificates before metadata fallback" do
    dir = File.tempname("gemma4-serving-route")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      prompt_ids = [10_i32, 20_i32]
      output_ids = [30_i32, 40_i32]
      full_history = prompt_ids + output_ids
      exact = gemma4_serving_exact_entry("model-a", "tok-a", full_history, output_ids)
      store.save_output_fast_forward(
        session_id: "s1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: "prompt text",
        prompt_token_ids: prompt_ids,
        output_token_ids: output_ids,
        generated_text: "cached output",
        exact_entry: exact,
      )

      result = ML::GGUF::Gemma4ServingRoute.serve_exact_cached_span(
        store,
        "model-a",
        "s1",
        "prompt text",
        output_ids,
        exact,
        full_history,
      )
      result.route.should eq(ML::GGUF::Gemma4ServingRoute::DIRECT_OUTPUT)
      result.output_token_ids.should eq(output_ids)
      result.prompt_token_count.should eq(prompt_ids.size)
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "falls back to exact metadata when no direct output file exists" do
    dir = File.tempname("gemma4-serving-route-fallback")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      prompt_ids = [10_i32, 20_i32]
      output_ids = [30_i32, 40_i32]
      full_history = prompt_ids + output_ids
      exact = gemma4_serving_exact_entry("model-a", "tok-a", full_history, output_ids)

      result = ML::GGUF::Gemma4ServingRoute.serve_exact_cached_span(
        store,
        "model-a",
        "s1",
        "prompt text",
        output_ids,
        exact,
        full_history,
      )
      result.route.should eq(ML::GGUF::Gemma4ServingRoute::EXACT_METADATA_FALLBACK)
      result.output_token_ids.should eq(output_ids)
      result.prompt_token_count.should eq(prompt_ids.size)
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "fails closed on mismatched tokens or unsupported continuation" do
    dir = File.tempname("gemma4-serving-route-fail-closed")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      prompt_ids = [10_i32, 20_i32]
      output_ids = [30_i32, 40_i32]
      full_history = prompt_ids + output_ids
      exact = gemma4_serving_exact_entry("model-a", "tok-a", full_history, output_ids)

      expect_raises(Exception, /output mismatch/) do
        ML::GGUF::Gemma4ServingRoute.serve_exact_cached_span(
          store,
          "model-a",
          "s1",
          "prompt text",
          [30_i32, 99_i32],
          exact,
          full_history,
        )
      end

      expect_raises(Exception, /continuation state route/) do
        ML::GGUF::Gemma4ServingRoute.serve_exact_cached_span(
          store,
          "model-a",
          "s1",
          "prompt text",
          output_ids,
          exact,
          full_history,
          continuation_required: true,
        )
      end
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end
end
