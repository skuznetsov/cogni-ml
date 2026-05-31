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
    ML::GGUF::Qwen35PromptCache.token_hash_concat([1_i32, 2_i32], [3_i32, 4_i32])
      .should eq(ML::GGUF::Qwen35PromptCache.token_hash([1_i32, 2_i32, 3_i32, 4_i32]))
    ML::GGUF::Qwen35PromptCache.token_hash_concat([1_i32, 2_i32], [3_i32, 4_i32], 3)
      .should eq(ML::GGUF::Qwen35PromptCache.token_hash([1_i32, 2_i32, 3_i32, 4_i32], 3))
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
      store.lookup_token_prefix("model-a", "tok-a", [42_i32, 99_i32], 1).try(&.prefix_len).should eq(1)
      store.lookup_session("s1", turn_id: "t1").should_not be_nil
      store.lookup_exact("model-b", "tok-a", prompt_hash, 1).should be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "stores opt-in source histories with hash validation" do
    root = File.tempname("qwen35-source-history")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      entry = store.save_source_history(
        session_id: "s1",
        turn_id: "t1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        token_ids: [10_i32, 20_i32, 30_i32, 40_i32],
        generated_token_count: 2,
        generated_text: " generated text",
      )

      entry.token_hash.should eq(ML::GGUF::Qwen35PromptCache.token_hash([10_i32, 20_i32, 30_i32, 40_i32]))
      entry.generated_token_count.should eq(2)
      entry.generated_text.should eq(" generated text")
      entry.generated_text_hash.should eq(ML::GGUF::Qwen35PromptCache.generated_text_hash(" generated text"))
      hit = store.lookup_source_history("s1", "model-a", "tok-a", turn_id: "t1")
      hit.should_not be_nil
      hit.not_nil!.token_ids.should eq([10_i32, 20_i32, 30_i32, 40_i32])
      hit.not_nil!.generated_text.should eq(" generated text")
      ML::GGUF::Qwen35PromptCache.generated_text_metadata_valid?(hit.not_nil!, 2).should be_true
      ML::GGUF::Qwen35PromptCache.generated_text_metadata_valid?(hit.not_nil!, 1).should be_false
      hit.not_nil!.generated_text_hash = "bad"
      ML::GGUF::Qwen35PromptCache.generated_text_metadata_valid?(hit.not_nil!, 2).should be_false
      ML::GGUF::Qwen35PromptCache.source_history_prefix_match?(hit.not_nil!.token_ids, [10_i32, 20_i32], 2).should be_true
      ML::GGUF::Qwen35PromptCache.source_history_prefix_match?(hit.not_nil!.token_ids, [20_i32, 30_i32], 2).should be_false

      File.open(store.source_history_manifest_path, "a") do |file|
        file.puts("{bad json")
      end
      store.lookup_source_history("s1", "model-a", "tok-a").should_not be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "stores tokenized prompts with text-hash and token-hash validation" do
    root = File.tempname("qwen35-tokenized-prompt")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      saved = store.save_tokenized_prompt(
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: "Hello, world",
        token_ids: [9419_i32, 11_i32, 1814_i32],
      )

      saved.prompt_text_hash.should eq(ML::GGUF::Qwen35PromptCache.prompt_text_hash("Hello, world"))
      saved.token_hash.should eq(ML::GGUF::Qwen35PromptCache.token_hash([9419_i32, 11_i32, 1814_i32]))

      hit = store.lookup_tokenized_prompt("model-a", "tok-a", "Hello, world")
      hit.should_not be_nil
      hit.not_nil!.token_ids.should eq([9419_i32, 11_i32, 1814_i32])
      model_hit = store.lookup_tokenized_prompt_for_model("model-a", "Hello, world")
      model_hit.should_not be_nil
      model_hit.not_nil!.tokenizer_id.should eq("tok-a")
      store.lookup_tokenized_prompt("model-a", "tok-a", "Hello, worlds").should be_nil
      store.lookup_tokenized_prompt("model-a", "tok-b", "Hello, world").should be_nil
      store.lookup_tokenized_prompt_for_model("model-b", "Hello, world").should be_nil

      File.open(store.tokenized_prompt_manifest_path, "a") do |file|
        file.puts("{bad json")
      end
      store.lookup_tokenized_prompt("model-a", "tok-a", "Hello, world").should_not be_nil
      store.lookup_tokenized_prompt_for_model("model-a", "Hello, world").should_not be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "stores proposal-route memory with fail-closed prompt and route-key validation" do
    root = File.tempname("qwen35-proposal-route")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      prompt = "Write a Crystal square function"
      token_ids = [10_i32, 20_i32, 30_i32]

      saved = store.save_proposal_route(
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: prompt,
        token_ids: token_ids,
        route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN,
        route_rank: 4,
        route_layers: [4, 0, 2, 2],
        route_key: "code_square",
        trigger: "first-margin<=0.10",
        evidence: "accept=100 overlap_ms=697.708",
      )

      saved.route.should eq(ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN)
      saved.route_rank.should eq(4)
      saved.route_layers.should eq([0, 2, 4])
      saved.route_key_hash.should eq(ML::GGUF::Qwen35PromptCache.proposal_route_key_hash("code_square"))

      hit = store.lookup_proposal_route("model-a", "tok-a", prompt, token_ids)
      hit.should_not be_nil
      hit.not_nil!.route.should eq(ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN)
      hit.not_nil!.route_layers << 99
      store.lookup_proposal_route("model-a", "tok-a", prompt, token_ids).not_nil!.route_layers.should eq([0, 2, 4])

      key_hit = store.lookup_proposal_route_key("model-a", "tok-a", "code_square")
      key_hit.should_not be_nil
      key_hit.not_nil!.route_rank.should eq(4)

      store.lookup_proposal_route("model-a", "tok-a", "#{prompt}!", token_ids).should be_nil
      store.lookup_proposal_route("model-a", "tok-b", prompt, token_ids).should be_nil
      store.lookup_proposal_route_key("model-a", "tok-a", "code_clamp").should be_nil
      ML::GGUF::Qwen35PromptCache.proposal_route_entry_valid?(
        saved,
        "model-a",
        "tok-a",
        prompt,
        token_ids,
      ).should be_true

      File.open(store.proposal_route_manifest_path, "a") do |file|
        file.puts("{bad json")
      end
      store.lookup_proposal_route("model-a", "tok-a", prompt, token_ids).should_not be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "rejects malformed proposal-route policies" do
    root = File.tempname("qwen35-proposal-route-bad")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      expect_raises(ArgumentError, /requires positive route_rank/) do
        store.save_proposal_route(
          model_id: "model-a",
          tokenizer_id: "tok-a",
          prompt_text: "prompt",
          token_ids: [1_i32],
          route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN,
        )
      end
      expect_raises(ArgumentError, /must not set route_rank/) do
        store.save_proposal_route(
          model_id: "model-a",
          tokenizer_id: "tok-a",
          prompt_text: "prompt",
          token_ids: [1_i32],
          route: ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_BASELINE,
          route_rank: 4,
        )
      end
      expect_raises(ArgumentError, /unsupported proposal route/) do
        store.save_proposal_route(
          model_id: "model-a",
          tokenizer_id: "tok-a",
          prompt_text: "prompt",
          token_ids: [1_i32],
          route: "unknown",
        )
      end
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "reuses parsed manifests without exposing mutable cache state" do
    root = File.tempname("qwen35-manifest-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      first = store.save_tokenized_prompt(
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: "first",
        token_ids: [1_i32],
      )

      store.lookup_tokenized_prompt("model-a", "tok-a", "first").try(&.token_ids).should eq([1_i32])
      cached = store.tokenized_prompt_entries
      cached.size.should eq(1)
      cached[0].token_ids << 99_i32
      cached.clear
      store.tokenized_prompt_entries[0].token_ids.should eq([1_i32])

      second = ML::GGUF::Qwen35PromptCache::TokenizedPromptEntry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::TOKENIZED_PROMPT_RUNTIME_ID,
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text_hash: ML::GGUF::Qwen35PromptCache.prompt_text_hash("second"),
        token_hash: ML::GGUF::Qwen35PromptCache.token_hash([2_i32]),
        token_count: 1,
        token_ids: [2_i32],
        created_at_unix: first.created_at_unix + 1,
      )
      File.open(store.tokenized_prompt_manifest_path, "a") do |file|
        second.to_json(file)
        file << '\n'
      end

      store.lookup_tokenized_prompt("model-a", "tok-a", "second").try(&.token_ids).should eq([2_i32])
      store.tokenized_prompt_entries.size.should eq(2)
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "stores direct output fast-forward certificates with hash validation" do
    root = File.tempname("qwen35-output-fast-forward")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      prompt_ids = [10_i32, 20_i32]
      output_ids = [30_i32, 40_i32, 50_i32]
      full_history = prompt_ids + output_ids
      exact_entry = ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: "t1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_hash: "unused",
        prefix_len: full_history.size - 1,
        max_seq: 16,
        layer_count: 1,
        artifact_path: "artifact.qkv",
        artifact_sha256: "0" * 64,
        artifact_byte_size: 0_i64,
        state_byte_size: 0_i64,
        created_at_unix: Time.utc.to_unix,
        prompt_preview: nil,
        token_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history, full_history.size - 1),
        artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: output_ids.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history),
        next_token_id: output_ids[-1],
      )

      saved = store.save_output_fast_forward(
        session_id: "s1",
        turn_id: "t1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: "prompt text",
        prompt_token_ids: prompt_ids,
        output_token_ids: output_ids,
        generated_text: " generated text",
        exact_entry: exact_entry,
      )

      ML::GGUF::Qwen35PromptCache.output_fast_forward_entry_valid?(
        saved,
        "model-a",
        "s1",
        "prompt text",
        output_ids.size,
        tokenizer_id: "tok-a",
        turn_id: "t1",
      ).should be_true
      ML::GGUF::Qwen35PromptCache.output_fast_forward_entry_valid?(
        saved,
        "model-a",
        "s1",
        "prompt text",
        output_ids.size,
        tokenizer_id: "tok-b",
        turn_id: "t1",
      ).should be_false
      hit = store.lookup_output_fast_forward("model-a", "s1", "prompt text", output_ids.size, turn_id: "t1")
      hit.should_not be_nil
      hit.not_nil!.prompt_token_ids.should eq(prompt_ids)
      hit.not_nil!.output_token_ids.should eq(output_ids)
      hit.not_nil!.generated_text.should eq(" generated text")
      store.lookup_output_fast_forward("model-a", "s1", "prompt text", output_ids.size, tokenizer_id: "tok-a", turn_id: "t1").should_not be_nil
      store.lookup_output_fast_forward("model-a", "s1", "prompt text", output_ids.size, tokenizer_id: "tok-b", turn_id: "t1").should be_nil
      hit.not_nil!.output_token_ids << 999_i32
      store.lookup_output_fast_forward("model-a", "s1", "prompt text", output_ids.size, turn_id: "t1").try(&.output_token_ids).should eq(output_ids)

      saved.generated_text_hash = "bad"
      ML::GGUF::Qwen35PromptCache.output_fast_forward_entry_valid?(
        saved,
        "model-a",
        "s1",
        "prompt text",
        output_ids.size,
        turn_id: "t1",
      ).should be_false
      store.lookup_output_fast_forward("model-a", "s1", "other prompt", output_ids.size, turn_id: "t1").should be_nil

      replacement_ids = [60_i32, 70_i32, 80_i32]
      replacement_history = prompt_ids + replacement_ids
      replacement_entry = ML::GGUF::Qwen35PromptCache::OutputFastForwardEntry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::OUTPUT_FAST_FORWARD_RUNTIME_ID,
        session_id: "s1",
        turn_id: "t1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text_hash: ML::GGUF::Qwen35PromptCache.prompt_text_hash("prompt text"),
        prompt_token_hash: ML::GGUF::Qwen35PromptCache.token_hash(prompt_ids),
        prompt_token_count: prompt_ids.size,
        prompt_token_ids: prompt_ids,
        output_token_hash: ML::GGUF::Qwen35PromptCache.token_hash(replacement_ids),
        output_token_count: replacement_ids.size,
        output_token_ids: replacement_ids,
        full_history_hash: ML::GGUF::Qwen35PromptCache.token_hash(replacement_history),
        generated_text: " replacement text",
        generated_text_hash: ML::GGUF::Qwen35PromptCache.generated_text_hash(" replacement text"),
        artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: replacement_ids.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(replacement_history),
        artifact_prefix_len: replacement_history.size - 1,
        artifact_token_hash: ML::GGUF::Qwen35PromptCache.token_hash(replacement_history, replacement_history.size - 1),
        artifact_next_token_id: replacement_ids[-1],
        created_at_unix: Time.utc.to_unix + 1,
      )
      key = ML::GGUF::Qwen35PromptCache.output_fast_forward_key(
        "model-a",
        "s1",
        "t1",
        ML::GGUF::Qwen35PromptCache.prompt_text_hash("prompt text"),
        replacement_ids.size,
      )
      bucket = File.join(store.output_fast_forward_dir, key[0, 2], key[2, 2])
      FileUtils.mkdir_p(bucket)
      File.open(File.join(bucket, "#{key}.json"), "w") do |file|
        replacement_entry.to_json(file)
        file << '\n'
      end
      store.lookup_output_fast_forward("model-a", "s1", "prompt text", replacement_ids.size, turn_id: "t1").try(&.output_token_ids).should eq(replacement_ids)

      exact_entry.artifact_validation_steps = output_ids.size - 1
      expect_raises(ArgumentError, /validation steps mismatch/) do
        store.save_output_fast_forward(
          session_id: "s1",
          turn_id: "t1",
          model_id: "model-a",
          tokenizer_id: "tok-a",
          prompt_text: "prompt text",
          prompt_token_ids: prompt_ids,
          output_token_ids: output_ids,
          generated_text: " generated text",
          exact_entry: exact_entry,
        )
      end
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "finds shorter terminal direct output certificates with an EOS guard" do
    root = File.tempname("qwen35-output-fast-forward-at-most")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      prompt_ids = [10_i32, 20_i32]
      terminal_output = [30_i32, 40_i32, 50_i32]
      nonterminal_output = [31_i32, 41_i32]
      terminal_history = prompt_ids + terminal_output
      nonterminal_history = prompt_ids + nonterminal_output
      terminal_entry = ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: nil,
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_hash: "unused",
        prefix_len: terminal_history.size - 1,
        max_seq: 16,
        layer_count: 1,
        artifact_path: "terminal.qkv",
        artifact_sha256: "0" * 64,
        artifact_byte_size: 0_i64,
        state_byte_size: 0_i64,
        created_at_unix: Time.utc.to_unix,
        prompt_preview: nil,
        token_hash: ML::GGUF::Qwen35PromptCache.token_hash(terminal_history, terminal_history.size - 1),
        artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: terminal_output.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(terminal_history),
        next_token_id: terminal_output[-1],
      )
      nonterminal_entry = ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: nil,
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_hash: "unused",
        prefix_len: nonterminal_history.size - 1,
        max_seq: 16,
        layer_count: 1,
        artifact_path: "nonterminal.qkv",
        artifact_sha256: "0" * 64,
        artifact_byte_size: 0_i64,
        state_byte_size: 0_i64,
        created_at_unix: Time.utc.to_unix,
        prompt_preview: nil,
        token_hash: ML::GGUF::Qwen35PromptCache.token_hash(nonterminal_history, nonterminal_history.size - 1),
        artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: nonterminal_output.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(nonterminal_history),
        next_token_id: nonterminal_output[-1],
      )

      store.save_output_fast_forward(
        session_id: "s1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: "terminal prompt",
        prompt_token_ids: prompt_ids,
        output_token_ids: terminal_output,
        generated_text: " terminal",
        exact_entry: terminal_entry,
        terminal_token_id: 50_i32,
      )
      store.save_output_fast_forward(
        session_id: "s1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_text: "nonterminal prompt",
        prompt_token_ids: prompt_ids,
        output_token_ids: nonterminal_output,
        generated_text: " nonterminal",
        exact_entry: nonterminal_entry,
      )

      store.lookup_output_fast_forward_at_most("model-a", "s1", "terminal prompt", 5, terminal_token_id: 50_i32).try(&.output_token_ids).should eq(terminal_output)
      store.lookup_output_fast_forward_at_most("model-a", "s1", "terminal prompt", 5).should be_nil
      store.lookup_terminal_output_fast_forward_at_most("model-a", "s1", "terminal prompt", 5).try(&.output_token_ids).should eq(terminal_output)
      store.lookup_terminal_output_fast_forward_at_most("model-a", "s1", "terminal prompt", 5, tokenizer_id: "tok-a").try(&.output_token_ids).should eq(terminal_output)
      store.lookup_terminal_output_fast_forward_at_most("model-a", "s1", "terminal prompt", 5, tokenizer_id: "tok-b").should be_nil
      store.lookup_output_fast_forward_at_most("model-a", "s1", "nonterminal prompt", 5, terminal_token_id: 50_i32).should be_nil
      store.lookup_terminal_output_fast_forward_at_most("model-a", "s1", "nonterminal prompt", 5).should be_nil
      store.lookup_output_fast_forward_at_most("model-a", "s1", "nonterminal prompt", 2, terminal_token_id: 50_i32).try(&.output_token_ids).should eq(nonterminal_output)
      store.lookup_output_fast_forward_at_most("model-a", "s1", "nonterminal prompt", 2, terminal_token_id: 50_i32, tokenizer_id: "tok-a").try(&.output_token_ids).should eq(nonterminal_output)
      store.lookup_output_fast_forward_at_most("model-a", "s1", "nonterminal prompt", 2, terminal_token_id: 50_i32, tokenizer_id: "tok-b").should be_nil
      store.lookup_terminal_output_fast_forward_at_most("model-a", "s1", "nonterminal prompt", 2).try(&.output_token_ids).should eq(nonterminal_output)

      expect_raises(ArgumentError, /terminal_token_id/) do
        store.save_output_fast_forward(
          session_id: "s1",
          model_id: "model-a",
          tokenizer_id: "tok-a",
          prompt_text: "bad terminal prompt",
          prompt_token_ids: prompt_ids,
          output_token_ids: terminal_output,
          generated_text: " terminal",
          exact_entry: terminal_entry,
          terminal_token_id: 999_i32,
        )
      end
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
    sql.should contain("artifact_codec")
    sql.should contain("artifact_live_kv_tokens")
    sql.should contain("artifact_validation_hash")

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
        artifact_codec: "recurrent-int8",
        artifact_codec_block: 8,
        artifact_live_kv_tokens: 1,
        artifact_validation_kind: "free-run-top1",
        artifact_validation_steps: 32,
        artifact_validation_hash: "v",
      )
    ).size.should eq(24)

    expect_raises(ArgumentError, /unsafe PostgreSQL identifier/) do
      ML::GGUF::Qwen35PromptCache.pg_sorted_heap_schema_sql(table_name: "cache; drop table x")
    end
  end

  it "keeps codec validation metadata optional and JSON-compatible" do
    entry = ML::GGUF::Qwen35PromptCache::Entry.new(
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
      artifact_codec: "recurrent-int8",
      artifact_codec_block: 8,
      artifact_live_kv_tokens: 0,
      artifact_validation_kind: "free-run-top1",
      artifact_validation_steps: 32,
      artifact_validation_hash: "hash",
    )

    parsed = ML::GGUF::Qwen35PromptCache::Entry.from_json(entry.to_json)
    parsed.artifact_codec.should eq("recurrent-int8")
    parsed.artifact_codec_block.should eq(8)
    parsed.artifact_live_kv_tokens.should eq(0)
    parsed.artifact_validation_kind.should eq("free-run-top1")
    parsed.artifact_validation_steps.should eq(32)
    parsed.artifact_validation_hash.should eq("hash")

    legacy_json = %({"runtime_id":"#{ML::GGUF::Qwen35PromptCache::RUNTIME_ID}","session_id":"s","turn_id":null,"model_id":"m","tokenizer_id":"t","prompt_hash":"p","token_hash":"h","prefix_len":0,"max_seq":1,"layer_count":1,"artifact_path":"a","artifact_sha256":"#{"0" * 64}","artifact_byte_size":0,"state_byte_size":0,"created_at_unix":1,"prompt_preview":null})
    legacy = ML::GGUF::Qwen35PromptCache::Entry.from_json(legacy_json)
    legacy.artifact_codec.should be_nil
    legacy.artifact_live_kv_tokens.should be_nil
    legacy.artifact_validation_hash.should be_nil
  end

  it "requires valid live-KV metadata before trusting live-KV artifacts" do
    entry = ML::GGUF::Qwen35PromptCache::Entry.new(
      runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
      session_id: "s",
      turn_id: nil,
      model_id: "m",
      tokenizer_id: "t",
      prompt_hash: "p",
      prefix_len: 4,
      max_seq: 8,
      layer_count: 1,
      artifact_path: "a",
      artifact_sha256: "0" * 64,
      artifact_byte_size: 0_i64,
      state_byte_size: 0_i64,
      created_at_unix: 1_i64,
      prompt_preview: nil,
      token_hash: "h",
      artifact_live_kv_tokens: 4,
    )

    ML::GGUF::Qwen35PromptCache.artifact_trust_metadata_valid?(entry).should be_true
    entry.artifact_live_kv_tokens = 5
    ML::GGUF::Qwen35PromptCache.artifact_trust_metadata_valid?(entry).should be_false
    entry.artifact_live_kv_tokens = 4
    entry.token_hash = nil
    ML::GGUF::Qwen35PromptCache.artifact_trust_metadata_valid?(entry).should be_false
  end

  it "validates exact known-span fast-forward metadata strictly" do
    full_history = [10_i32, 20_i32, 30_i32, 40_i32]
    entry = ML::GGUF::Qwen35PromptCache::Entry.new(
      runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
      session_id: "s",
      turn_id: nil,
      model_id: "m",
      tokenizer_id: "t",
      prompt_hash: "p",
      prefix_len: full_history.size - 1,
      max_seq: 16,
      layer_count: 1,
      artifact_path: "a",
      artifact_sha256: "0" * 64,
      artifact_byte_size: 0_i64,
      state_byte_size: 0_i64,
      created_at_unix: 1_i64,
      prompt_preview: nil,
      token_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history, full_history.size - 1),
      artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
      artifact_validation_steps: 2,
      artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history),
      next_token_id: full_history[-1],
    )

    ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_true

    entry.artifact_validation_hash = ML::GGUF::Qwen35PromptCache.token_hash(full_history[0, 3])
    ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
    entry.artifact_validation_hash = ML::GGUF::Qwen35PromptCache.token_hash(full_history)

    entry.next_token_id = 999_i32
    ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
    entry.next_token_id = full_history[-1]

    entry.prefix_len = full_history.size - 2
    ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
    entry.prefix_len = full_history.size - 1

    entry.artifact_validation_steps = 1
    ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
  end

  it "skips cache entries with incomplete compressed-artifact validation metadata" do
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
      artifact = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, File.join(root, "manual.qkv"))
      prompt_hash = ML::GGUF::Qwen35PromptCache.prompt_hash([42_i32], "x")
      token_hash = ML::GGUF::Qwen35PromptCache.token_hash([42_i32])
      raw = ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s",
        turn_id: nil,
        model_id: "m",
        tokenizer_id: "t",
        prompt_hash: prompt_hash,
        prefix_len: 1,
        max_seq: snapshot.max_seq,
        layer_count: snapshot.layer_count,
        artifact_path: artifact.path,
        artifact_sha256: artifact.sha256,
        artifact_byte_size: artifact.byte_size,
        state_byte_size: snapshot.byte_size,
        created_at_unix: 1_i64,
        prompt_preview: nil,
        token_hash: token_hash,
      )
      incomplete_compressed = ML::GGUF::Qwen35PromptCache::Entry.new(
        runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
        session_id: "s",
        turn_id: nil,
        model_id: "m",
        tokenizer_id: "t",
        prompt_hash: prompt_hash,
        prefix_len: 1,
        max_seq: snapshot.max_seq,
        layer_count: snapshot.layer_count,
        artifact_path: artifact.path,
        artifact_sha256: artifact.sha256,
        artifact_byte_size: artifact.byte_size,
        state_byte_size: snapshot.byte_size,
        created_at_unix: 2_i64,
        prompt_preview: nil,
        token_hash: token_hash,
        artifact_codec: "recurrent-int8",
        artifact_codec_block: 8,
      )
      File.open(store.manifest_path, "w") do |file|
        raw.to_json(file)
        file << '\n'
        incomplete_compressed.to_json(file)
        file << '\n'
      end

      hit = store.lookup_exact("m", "t", prompt_hash, 1).not_nil!
      hit.artifact_codec.should be_nil
      store.lookup_session("s").try(&.artifact_codec).should be_nil
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "validates compressed-artifact metadata before allowing compressed-reader restore" do
    root = File.tempname("qwen35-prompt-cache")
    Dir.mkdir_p(root)
    snapshot = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
      max_seq: 8,
      layer_count: 1,
      positions: [0_i32],
      records: [
        ML::GGUF::Qwen35StateSnapshot::Record.new(
          0,
          ML::GGUF::Qwen35StateSnapshot::RecordKind::SsmState,
          qwen35_cache_f32_bytes([1.0_f32, -2.0_f32, 3.0_f32, -4.0_f32]),
          ML::StorageMode::Shared,
        ),
      ],
    )
    artifact = ML::GGUF::Qwen35StateSnapshot.write_artifact(
      snapshot,
      File.join(root, "compressed.qkv"),
      artifact_codec: "recurrent-int8",
      artifact_codec_block: 8,
    )
    entry = ML::GGUF::Qwen35PromptCache::Entry.new(
      runtime_id: ML::GGUF::Qwen35PromptCache::RUNTIME_ID,
      session_id: "s",
      turn_id: nil,
      model_id: "m",
      tokenizer_id: "t",
      prompt_hash: "p",
      prefix_len: 0,
      max_seq: snapshot.max_seq,
      layer_count: snapshot.layer_count,
      artifact_path: artifact.path,
      artifact_sha256: artifact.sha256,
      artifact_byte_size: artifact.byte_size,
      state_byte_size: snapshot.byte_size,
      created_at_unix: 1_i64,
      prompt_preview: nil,
      token_hash: "h",
      artifact_codec: "recurrent-int8",
      artifact_codec_block: 8,
      artifact_validation_kind: "free-run-top1",
      artifact_validation_steps: 32,
      artifact_validation_hash: "hash",
    )

    ML::GGUF::Qwen35PromptCache.artifact_trust_metadata_valid?(entry).should be_true
    ML::GGUF::Qwen35PromptCache.validate_restorable_artifact!(entry)
    loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact(
      entry.artifact_path,
      expected_sha256: entry.artifact_sha256,
      expected_codec: entry.artifact_codec,
      expected_codec_block: entry.artifact_codec_block,
    )
    loaded.records.size.should eq(1)
  ensure
    FileUtils.rm_rf(root) if root && Dir.exists?(root)
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

  it "can serve a hot restore from the resident state cache without rereading the artifact" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"
    prompt_text = "The capital of France is"

    root = File.tempname("qwen35-resident-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root, resident_state_cache_entries: 1)
      live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      prompt.each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
      end

      saved = store.save(
        session_id: "resident-session",
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

      store.restore(hit, hp)
      File.delete(saved.artifact_path)

      reuse = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(reuse, hp) if ML::GGUF::Qwen35Metal.available?
      restored = store.restore(hit, hp, reuse_state: reuse)

      live_top, live_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, live)
      restored_top, restored_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, restored)

      restored_top.should eq(live_top)
      restored_logit.should be_close(live_logit, 1e-4_f32)
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "does not reuse a session artifact validation memo after the artifact file changes" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"
    prompt_text = "The capital of France is"

    root = File.tempname("qwen35-artifact-validation-memo")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      prompt.each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
      end

      saved = store.save(
        session_id: "memo-session",
        model_id: "qwen35-9b-q4km-test",
        tokenizer_id: "qwen35-tokenizer-test",
        prompt_text: prompt_text,
        token_ids: prompt,
        state: live,
      )
      File.open(saved.artifact_path, "a") { |file| file.write_byte(0_u8) }

      expect_raises(ArgumentError, /byte-size mismatch|sha256 mismatch/) do
        store.restore(saved, hp)
      end
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "restores a validated BF16 recurrent prompt-cache artifact through the Metal encoded path" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"
    prompt_text = "The capital of France is"

    root = File.tempname("qwen35-bf16-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(live, hp)
      prompt.each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
      end

      saved = store.save(
        session_id: "bf16-session",
        model_id: "qwen35-9b-q4km-test",
        tokenizer_id: "qwen35-tokenizer-test",
        prompt_text: prompt_text,
        token_ids: prompt,
        state: live,
        artifact_codec: "recurrent-bf16",
        artifact_validation_kind: "prompt-cache-bf16-smoke",
        artifact_validation_steps: prompt.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(prompt),
      )
      saved.artifact_codec.should eq("recurrent-bf16")

      reuse = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(reuse, hp)
      restored = store.restore(saved, hp, reuse_state: reuse)

      live_top, live_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, live)
      restored_top, restored_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, restored)
      restored_top.should eq(live_top)
      restored_logit.should be_close(live_logit, 1e-2_f32)
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end

  it "keeps validated INT8 recurrent prompt-cache Metal restore behind an explicit gate" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"
    prompt_text = "The capital of France is"
    old_gate = ENV["QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE"]?
    ENV.delete("QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE")

    root = File.tempname("qwen35-i8-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(live, hp)
      prompt.each_with_index do |token_id, pos|
        ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
      end

      saved = store.save(
        session_id: "i8-session",
        model_id: "qwen35-9b-q4km-test",
        tokenizer_id: "qwen35-tokenizer-test",
        prompt_text: prompt_text,
        token_ids: prompt,
        state: live,
        artifact_codec: "recurrent-int8",
        artifact_codec_block: 8,
        artifact_validation_kind: "prompt-cache-i8-smoke",
        artifact_validation_steps: prompt.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(prompt),
      )

      reuse = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(reuse, hp)
      expect_raises(ArgumentError, /QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE=1/) do
        store.restore(saved, hp, reuse_state: reuse)
      end
    ensure
      if old_gate
        ENV["QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE"] = old_gate
      else
        ENV.delete("QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE")
      end
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

  it "restores an exact full-prompt hit without replaying the final prompt token" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_PROMPT_CACHE)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"

    root = File.tempname("qwen35-full-prompt-cache")
    Dir.mkdir_p(root)
    begin
      store = ML::GGUF::Qwen35PromptCache::Store.new(root)
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      top = 0_i32
      logit = 0.0_f32
      prompt.each_with_index do |token_id, pos|
        top, logit = ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, state)
      end
      store.save(
        session_id: "session-full",
        model_id: "qwen35-9b-q4km-test",
        tokenizer_id: "qwen35-tokenizer-test",
        prompt_text: "",
        token_ids: prompt,
        state: state,
        next_token_id: top,
        next_token_logit: logit,
      )

      hit = store.lookup_longest_prefix(
        "qwen35-9b-q4km-test",
        "qwen35-tokenizer-test",
        prompt,
        max_prefix_len: prompt.size,
      ).not_nil!
      replay = store.restore_and_replay_suffix(hit, w, prompt)

      replay.reused_prefix_len.should eq(prompt.size)
      replay.replayed_tokens.should eq(0)
      replay.next_token_id.should eq(top)
      replay.next_token_logit.should eq(logit)
    ensure
      FileUtils.rm_rf(root) if Dir.exists?(root)
    end
  end
end

private def qwen35_cache_f32_bytes(values : Array(Float32)) : Bytes
  bytes = Bytes.new(values.size * sizeof(Float32))
  values.each_with_index do |value, i|
    bits = value.unsafe_as(UInt32)
    offset = i * sizeof(Float32)
    bytes[offset] = (bits & 0xff).to_u8
    bytes[offset + 1] = ((bits >> 8) & 0xff).to_u8
    bytes[offset + 2] = ((bits >> 16) & 0xff).to_u8
    bytes[offset + 3] = ((bits >> 24) & 0xff).to_u8
  end
  bytes
end
