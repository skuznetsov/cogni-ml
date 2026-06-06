require "./spec_helper"
require "file_utils"
require "../src/ml/gguf/gemma4_prompt_cache"
require "../src/ml/gguf/qwen35_metal"

private def gemma4_prompt_cache_fill(state : ML::GGUF::Gemma4Metal::ResidentState) : Nil
  state.layers.each_with_index do |layer, il|
    count = layer.max_seq * layer.kv_dim
    layer.k_cache_buf.write(Array(Float32).new(count) { |i| (10_000 * (il + 1) + i).to_f32 })
    layer.v_cache_buf.write(Array(Float32).new(count) { |i| (-10_000 * (il + 1) - i).to_f32 })
  end
end

private def gemma4_prompt_cache_prefix_equal?(a : ML::GGUF::Gemma4Metal::ResidentState,
                                              b : ML::GGUF::Gemma4Metal::ResidentState,
                                              prefix_len : Int32) : Bool
  a.layers.each_with_index.all? do |layer, il|
    other = b.layers[il]
    live = prefix_len * layer.kv_dim
    layer.k_cache_buf.read(layer.max_seq * layer.kv_dim)[0, live] == other.k_cache_buf.read(other.max_seq * other.kv_dim)[0, live] &&
      layer.v_cache_buf.read(layer.max_seq * layer.kv_dim)[0, live] == other.v_cache_buf.read(other.max_seq * other.kv_dim)[0, live]
  end
end

describe "Gemma4PromptCache" do
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "saves, looks up, and restores an exact resident prompt prefix" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32, 20_i32, 30_i32]

      saved = store.save_resident_state(
        source,
        token_ids,
        model_id: "gemma4-test",
        tokenizer_id: "tok-test",
        prompt_text: "hello",
        session_id: "s1",
        turn_id: "t1",
        next_token_id: 123_i32,
      )
      saved.runtime_id.should eq(ML::GGUF::Gemma4PromptCache::RUNTIME_ID)
      saved.prefix_len.should eq(token_ids.size)
      saved.token_hash.should eq(ML::GGUF::Gemma4PromptCache.token_hash(token_ids))
      saved.next_token_id.should eq(123_i32)

      hit = store.lookup_prompt("gemma4-test", "tok-test", "hello", token_ids).not_nil!
      hit.artifact_sha256.should eq(saved.artifact_sha256)
      hit.next_token_id.should eq(123_i32)
      hit.next_token_id = -1
      ML::GGUF::Gemma4PromptCache.artifact_trust_metadata_valid?(hit).should be_false
      hit.next_token_id = 123_i32
      restored = store.restore(hit)

      gemma4_prompt_cache_prefix_equal?(restored, source, token_ids.size).should be_true
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "finds longest compatible token prefix and ignores model/tokenizer mismatches" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2], 8)
      gemma4_prompt_cache_fill(source)
      prefix = [2_i32, 10_i32, 20_i32]

      store.save_resident_state(source, prefix, model_id: "m", tokenizer_id: "t", prompt_text: "p")

      store.lookup_longest_prefix("m", "t", prefix + [30_i32, 40_i32]).not_nil!.prefix_len.should eq(prefix.size)
      store.lookup_longest_prefix("other", "t", prefix + [30_i32]).should be_nil
      store.lookup_longest_prefix("m", "other", prefix + [30_i32]).should be_nil
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "fails closed for corrupt manifest lines and mutated artifacts" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32]

      saved = store.save_resident_state(source, token_ids, model_id: "m", tokenizer_id: "t")
      File.open(File.join(dir, "manifest.jsonl"), "a") { |file| file.puts("{not-json") }
      store.entries.size.should eq(1)

      File.open(saved.artifact_path, "a") { |file| file.write_byte(0_u8) }
      hit = store.lookup_prompt("m", "t", "", token_ids).not_nil!
      expect_raises(ArgumentError, /byte-size mismatch|checksum mismatch|changed/) do
        store.restore(hit)
      end
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "restores into a reusable resident state without replacing the object" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      target = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32, 20_i32]

      saved = store.save_resident_state(source, token_ids, model_id: "m", tokenizer_id: "t")
      restored = store.restore(saved, reuse_state: target)

      restored.object_id.should eq(target.object_id)
      gemma4_prompt_cache_prefix_equal?(target, source, token_ids.size).should be_true
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "reuses validated snapshots in-process and invalidates them when the artifact changes" do
    dir = File.tempname("gemma4-prompt-cache")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(
        dir,
        snapshot_cache_byte_limit: 1_000_000_i64,
        snapshot_cache_entry_limit: 1,
      )
      source = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      gemma4_prompt_cache_fill(source)
      token_ids = [2_i32, 10_i32, 20_i32]

      saved = store.save_resident_state(source, token_ids, model_id: "m", tokenizer_id: "t")
      target1 = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      store.restore(saved, reuse_state: target1)
      store.snapshot_cache_misses.should eq(1)
      store.snapshot_cache_hits.should eq(0)
      store.snapshot_cache_bytes.should eq(saved.state_byte_size)

      target2 = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      store.restore(saved, reuse_state: target2)
      store.snapshot_cache_hits.should eq(1)
      gemma4_prompt_cache_prefix_equal?(target2, source, token_ids.size).should be_true

      sleep 20.milliseconds
      File.open(saved.artifact_path, "r+") do |file|
        file.seek(saved.artifact_byte_size - 1)
        byte = file.read_byte.not_nil!
        file.seek(saved.artifact_byte_size - 1)
        file.write_byte(byte ^ 0xff_u8)
      end

      target3 = ML::GGUF::Gemma4Metal::ResidentState.new([2, 3], 8)
      expect_raises(ArgumentError, /checksum mismatch/) do
        store.restore(saved, reuse_state: target3)
      end
    ensure
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "clamps resident snapshot cache budgets to preserve a memory floor" do
    ML::GGUF::Gemma4PromptCache.clamp_snapshot_cache_byte_limit(
      1_000_i64,
      0_i64,
      100_i64,
    ).should eq(1_000_i64)
    ML::GGUF::Gemma4PromptCache.clamp_snapshot_cache_byte_limit(
      1_000_i64,
      500_i64,
      nil,
    ).should eq(1_000_i64)
    ML::GGUF::Gemma4PromptCache.clamp_snapshot_cache_byte_limit(
      1_000_i64,
      500_i64,
      400_i64,
    ).should eq(0_i64)
    ML::GGUF::Gemma4PromptCache.clamp_snapshot_cache_byte_limit(
      1_000_i64,
      500_i64,
      900_i64,
    ).should eq(400_i64)
  end

  it "validates exact known-span fast-forward metadata strictly" do
    full_history = [10_i32, 20_i32, 30_i32, 40_i32]
    entry = ML::GGUF::Gemma4PromptCache::Entry.new(
      runtime_id: ML::GGUF::Gemma4PromptCache::RUNTIME_ID,
      session_id: "s",
      turn_id: nil,
      model_id: "m",
      tokenizer_id: "t",
      prompt_hash: "p",
      token_hash: ML::GGUF::Gemma4PromptCache.token_hash(full_history, full_history.size - 1),
      prefix_len: full_history.size - 1,
      max_seq: 16,
      layer_count: 1,
      kv_dims: [2_i32],
      artifact_path: "a",
      artifact_sha256: "0" * 64,
      artifact_byte_size: 0_i64,
      state_byte_size: 0_i64,
      created_at_unix: 1_i64,
      prompt_preview: nil,
      next_token_id: full_history[-1],
      artifact_validation_kind: ML::GGUF::Gemma4PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
      artifact_validation_steps: 2,
      artifact_validation_hash: ML::GGUF::Gemma4PromptCache.token_hash(full_history),
    )

    ML::GGUF::Gemma4PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_true

    entry.artifact_validation_hash = ML::GGUF::Gemma4PromptCache.token_hash(full_history[0, 3])
    ML::GGUF::Gemma4PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
    entry.artifact_validation_hash = ML::GGUF::Gemma4PromptCache.token_hash(full_history)

    entry.next_token_id = 999_i32
    ML::GGUF::Gemma4PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
    entry.next_token_id = full_history[-1]

    entry.prefix_len = full_history.size - 2
    ML::GGUF::Gemma4PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false
    entry.prefix_len = full_history.size - 1

    entry.artifact_validation_steps = 1
    ML::GGUF::Gemma4PromptCache.exact_known_span_entry_valid?(entry, full_history, 2).should be_false

    entry.artifact_validation_steps = 2
    entry.artifact_validation_hash = nil
    ML::GGUF::Gemma4PromptCache.artifact_trust_metadata_valid?(entry).should be_false
  end

  it "stores direct output fast-forward certificates with hash validation" do
    dir = File.tempname("gemma4-output-fast-forward")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      prompt_ids = [10_i32, 20_i32]
      output_ids = [30_i32, 40_i32, 50_i32]
      full_history = prompt_ids + output_ids
      exact_entry = ML::GGUF::Gemma4PromptCache::Entry.new(
        runtime_id: ML::GGUF::Gemma4PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: "t1",
        model_id: "model-a",
        tokenizer_id: "tok-a",
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

      ML::GGUF::Gemma4PromptCache.output_fast_forward_entry_valid?(
        saved,
        "model-a",
        "s1",
        "prompt text",
        output_ids.size,
        tokenizer_id: "tok-a",
        turn_id: "t1",
      ).should be_true
      ML::GGUF::Gemma4PromptCache.output_fast_forward_entry_valid?(
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
      ML::GGUF::Gemma4PromptCache.output_fast_forward_entry_valid?(
        saved,
        "model-a",
        "s1",
        "prompt text",
        output_ids.size,
        turn_id: "t1",
      ).should be_false
      store.lookup_output_fast_forward("model-a", "s1", "other prompt", output_ids.size, turn_id: "t1").should be_nil

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
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end

  it "finds shorter terminal direct output certificates with an EOS guard" do
    dir = File.tempname("gemma4-output-fast-forward-at-most")
    FileUtils.mkdir_p(dir)
    begin
      store = ML::GGUF::Gemma4PromptCache::Store.new(dir)
      prompt_ids = [10_i32, 20_i32]
      terminal_output = [30_i32, 40_i32, 50_i32]
      nonterminal_output = [31_i32, 41_i32]
      terminal_history = prompt_ids + terminal_output
      nonterminal_history = prompt_ids + nonterminal_output
      terminal_entry = ML::GGUF::Gemma4PromptCache::Entry.new(
        runtime_id: ML::GGUF::Gemma4PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: nil,
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_hash: "unused",
        token_hash: ML::GGUF::Gemma4PromptCache.token_hash(terminal_history, terminal_history.size - 1),
        prefix_len: terminal_history.size - 1,
        max_seq: 16,
        layer_count: 1,
        kv_dims: [2_i32],
        artifact_path: "terminal.gkv",
        artifact_sha256: "0" * 64,
        artifact_byte_size: 0_i64,
        state_byte_size: 0_i64,
        created_at_unix: Time.utc.to_unix,
        prompt_preview: nil,
        next_token_id: terminal_output[-1],
        artifact_validation_kind: ML::GGUF::Gemma4PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: terminal_output.size,
        artifact_validation_hash: ML::GGUF::Gemma4PromptCache.token_hash(terminal_history),
      )
      nonterminal_entry = ML::GGUF::Gemma4PromptCache::Entry.new(
        runtime_id: ML::GGUF::Gemma4PromptCache::RUNTIME_ID,
        session_id: "s1",
        turn_id: nil,
        model_id: "model-a",
        tokenizer_id: "tok-a",
        prompt_hash: "unused",
        token_hash: ML::GGUF::Gemma4PromptCache.token_hash(nonterminal_history, nonterminal_history.size - 1),
        prefix_len: nonterminal_history.size - 1,
        max_seq: 16,
        layer_count: 1,
        kv_dims: [2_i32],
        artifact_path: "nonterminal.gkv",
        artifact_sha256: "0" * 64,
        artifact_byte_size: 0_i64,
        state_byte_size: 0_i64,
        created_at_unix: Time.utc.to_unix,
        prompt_preview: nil,
        next_token_id: nonterminal_output[-1],
        artifact_validation_kind: ML::GGUF::Gemma4PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: nonterminal_output.size,
        artifact_validation_hash: ML::GGUF::Gemma4PromptCache.token_hash(nonterminal_history),
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
      FileUtils.rm_rf(dir) if File.exists?(dir)
    end
  end
end
