require "./spec_helper"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_PROMPT_CACHE = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35PromptCache do
  it "computes deterministic exact prompt hashes" do
    a = ML::GGUF::Qwen35PromptCache.prompt_hash([1_i32, 2_i32, 3_i32], "abc")
    b = ML::GGUF::Qwen35PromptCache.prompt_hash([1_i32, 2_i32, 3_i32], "abc")
    c = ML::GGUF::Qwen35PromptCache.prompt_hash([1_i32, 2_i32, 4_i32], "abc")
    d = ML::GGUF::Qwen35PromptCache.prompt_hash([1_i32, 2_i32, 3_i32], "abcd")

    a.should eq(b)
    a.size.should eq(64)
    a.should_not eq(c)
    a.should_not eq(d)

    token_full = ML::GGUF::Qwen35PromptCache.token_hash([1_i32, 2_i32, 3_i32])
    token_prefix = ML::GGUF::Qwen35PromptCache.token_hash([1_i32, 2_i32, 3_i32], 2)
    token_prefix.should eq(ML::GGUF::Qwen35PromptCache.token_hash([1_i32, 2_i32], 2))
    token_full.should_not eq(token_prefix)
  end

  it "indexes entries by exact prompt hash and session" do
    root = File.tempname("qwen35-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      snapshot = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
        max_seq: 8,
        layer_count: 1,
        positions: [0_i32],
        records: [] of ML::GGUF::Qwen35StateSnapshot::Record,
      )
      artifact = ML::GGUF::Qwen35StateSnapshot.write_artifact(
        snapshot,
        File.join(root, "manual.qkv"),
      )
      prompt_hash = ML::GGUF::Qwen35PromptCache.prompt_hash([42_i32], "x")
      token_hash = ML::GGUF::Qwen35PromptCache.token_hash([42_i32])
      entry = ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: "t1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_hash: prompt_hash,
        prefix_len: 1,
        max_seq: snapshot.max_seq,
        layer_count: snapshot.layer_count,
        artifact_path: artifact.path,
        artifact_sha256: artifact.sha256,
        artifact_byte_size: artifact.byte_size,
        state_byte_size: snapshot.byte_size,
        created_at_unix: Time.utc.to_unix,
        prompt_preview: nil,
        token_hash: token_hash,
      )
      File.open(store.manifest_path, "w") do |file|
        file.puts("{bad json")
        entry.to_json(file)
        file << '\n'
      end

      store.lookup_exact("model-a", "tok-a", prompt_hash, 1).should_not be_nil
      store.lookup_prompt("model-a", "tok-a", "x", [42_i32]).should_not be_nil
      store.lookup_longest_prefix("model-a", "tok-a", [42_i32, 99_i32]).try(&.prefix_len).should eq(1)
      store.lookup_session("s1", turn_id: "t1").should_not be_nil
      store.lookup_exact("model-b", "tok-a", prompt_hash, 1).should be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "generates pg_sorted_heap metadata SQL without accepting unsafe identifiers" do
    sql = ML::GGUF::Qwen35PromptCache.pg_sorted_heap_schema_sql
    sql.should contain("CREATE EXTENSION IF NOT EXISTS pg_sorted_heap")
    sql.should contain("USING sorted_heap")
    sql.should contain("qwen35_prompt_cache_exact_idx")
    sql.should contain("qwen35_prompt_cache_prefix_idx")

    legacy = ML::GGUF::Qwen35PromptCache.pg_sorted_heap_schema_sql(table_am: "clustered_heap")
    legacy.should contain("USING clustered_heap")

    insert_sql = ML::GGUF::Qwen35PromptCache.pg_insert_sql
    insert_sql.should contain("ON CONFLICT (model_id, tokenizer_id, prompt_hash, prefix_len)")
    ML::GGUF::Qwen35PromptCache.pg_insert_values(
      ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s",
        turn_id: nil,
        model_id: "m",
        tokenizer_id: "t",
        prompt_hash: "p",
        prefix_len: 0,
        max_seq: 1,
        layer_count: 1,
        artifact_path: "a",
        artifact_sha256: "0" * 64,
        artifact_byte_size: 0_i64,
        state_byte_size: 0_i64,
        created_at_unix: 1_i64,
        prompt_preview: nil,
        token_hash: "h",
      )
    ).size.should eq(16)

    expect_raises(ArgumentError, /unsafe PostgreSQL identifier/) do
      ML::GGUF::Qwen35PromptCache.pg_sorted_heap_schema_sql(table_name: "cache; drop table x")
    end
  end

  pending!("9B model not present") unless File.exists?(QWEN_9B_PROMPT_CACHE)

  it "saves and restores a prompt-prefill state from an exact cache hit" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"
    prompt_text = "The capital of France is"

    root = File.tempname("qwen35-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      prompt.each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
      end

      saved = store.save(
        session_id: "session-a",
        turn_id: "turn-a",
        model_id: "qwen35-9b-q4km-test",
        tokenizer_id: "qwen35-tokenizer-test",
        prompt_text: prompt_text,
        token_ids: prompt,
        state: live,
      )
      hit = store.lookup_prompt(
        "qwen35-9b-q4km-test",
        "qwen35-tokenizer-test",
        prompt_text,
        prompt,
      ).not_nil!
      hit.artifact_sha256.should eq(saved.artifact_sha256)

      restored = store.restore(hit, hp)
      live_top, live_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, live)
      restored_top, restored_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, restored)

      restored_top.should eq(live_top)
      restored_logit.should be_close(live_logit, 1e-4_f32)
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "restores the longest cached prefix and replays only the suffix" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"
    prefix_len = 3

    root = File.tempname("qwen35-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      prefix_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      prompt.first(prefix_len).each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, prefix_state)
      end
      store.save(
        session_id: "session-prefix",
        model_id: "qwen35-9b-q4km-test",
        tokenizer_id: "qwen35-tokenizer-test",
        prompt_text: "The capital",
        token_ids: prompt.first(prefix_len),
        state: prefix_state,
      )

      live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      prompt.each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
      end

      hit = store.lookup_longest_prefix(
        "qwen35-9b-q4km-test",
        "qwen35-tokenizer-test",
        prompt,
      ).not_nil!
      replay = store.restore_and_replay_suffix(hit, w, prompt)
      replay.reused_prefix_len.should eq(prefix_len)
      replay.replayed_tokens.should eq(prompt.size - prefix_len)
      replay.next_token_id.should_not be_nil
      replay.next_token_logit.should_not be_nil

      live_top, live_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, live)
      replay_top, replay_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, replay.state)

      replay_top.should eq(live_top)
      replay_logit.should be_close(live_logit, 1e-4_f32)
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end
end
