require "digest/sha256"
require "./spec_helper"
require "../src/ml/gguf/qwen35_native_runtime"

private alias QwenEngine = ML::GGUF::Qwen35Engine
private alias NativeRuntime = ML::GGUF::Qwen35NativeRuntime

private def reference_top_two(logits : Array(Float32), token_ids : Array(Int32)) : {Int32, Int32}
  ranked = token_ids.sort do |left, right|
    if logits[left] == logits[right]
      left <=> right
    else
      logits[right] <=> logits[left]
    end
  end
  {ranked[0], ranked[1]}
end

describe ML::GGUF::Qwen35NativeRuntime do
  it "exposes zero-cost QBit phase timing defaults" do
    stats = NativeRuntime::QBitCacheStats.new(
      hits: 0,
      misses: 0,
      rejections: 0,
      transport_failures: 0,
      restore_failures: 0,
      writes: 0,
      write_failures: 0,
    )
    stats.lookup_time.should eq(Time::Span.zero)
    stats.restore_time.should eq(Time::Span.zero)
    stats.write_back_time.should eq(Time::Span.zero)
  end

  it "reports the selected backend without claiming CUDA or silent fallback" do
    auto_metal = NativeRuntime.backend_identity_for(
      QwenEngine::Backend::Auto,
      "model:one",
      metal_available: true,
      decode_wave_forced_off: false,
    )
    auto_metal.requested.should eq(QwenEngine::Backend::Auto)
    auto_metal.primary.should eq(QwenEngine::Backend::Metal)
    auto_metal.components.should eq([QwenEngine::Backend::Metal, QwenEngine::Backend::CPU])
    auto_metal.attribution.should eq(QwenEngine::Attribution::Planned)

    wave_off = NativeRuntime.backend_identity_for(
      QwenEngine::Backend::Auto,
      "model:one",
      metal_available: true,
      decode_wave_forced_off: true,
    )
    wave_off.primary.should eq(QwenEngine::Backend::Metal)
    wave_off.components.should eq([QwenEngine::Backend::Metal, QwenEngine::Backend::CPU])
    wave_off.attribution.should eq(QwenEngine::Attribution::Planned)

    expect_raises(QwenEngine::BackendMismatch, /required CPU/) do
      NativeRuntime.backend_identity_for(
        QwenEngine::Backend::CPU,
        "model:one",
        metal_available: true,
        decode_wave_forced_off: false,
      )
    end
    cpu_only = NativeRuntime.backend_identity_for(
      QwenEngine::Backend::CPU,
      "model:one",
      metal_available: false,
      decode_wave_forced_off: false,
    )
    cpu_only.primary.should eq(QwenEngine::Backend::CPU)
    cpu_only.components.should eq([QwenEngine::Backend::CPU])
    cpu_only.attribution.should eq(QwenEngine::Attribution::Observed)

    expect_raises(QwenEngine::BackendMismatch, /required Metal/) do
      NativeRuntime.backend_identity_for(
        QwenEngine::Backend::Metal,
        "model:one",
        metal_available: false,
        decode_wave_forced_off: false,
      )
    end
    expect_raises(QwenEngine::BackendMismatch, /attribution/) do
      NativeRuntime.backend_identity_for(
        QwenEngine::Backend::Metal,
        "model:one",
        metal_available: true,
        decode_wave_forced_off: false,
      )
    end
    expect_raises(QwenEngine::BackendMismatch, /CUDA/) do
      NativeRuntime.backend_identity_for(
        QwenEngine::Backend::CUDA,
        "model:one",
        metal_available: true,
        decode_wave_forced_off: false,
      )
    end
  end

  it "resolves labels to unique single tokens and preserves optional caller ids" do
    labels = [
      QwenEngine::Label.new("allow", "A"),
      QwenEngine::Label.new("ask", "B", 12_i32),
    ]
    NativeRuntime.resolve_label_ids(labels, [[11_i32], [12_i32]]).should eq([11_i32, 12_i32])
  end

  it "rejects unresolved, multi-token, duplicate, and mismatched labels before state creation" do
    labels = [QwenEngine::Label.new("allow", "A"), QwenEngine::Label.new("ask", "B")]
    expect_raises(ArgumentError, /exactly one token/) do
      NativeRuntime.resolve_label_ids(labels, [[11_i32, 12_i32], [13_i32]])
    end
    expect_raises(ArgumentError, /unique token ids/) do
      NativeRuntime.resolve_label_ids(labels, [[11_i32], [11_i32]])
    end
    expect_raises(ArgumentError, /token id mismatch/) do
      NativeRuntime.resolve_label_ids(
        [QwenEngine::Label.new("allow", "A", 99_i32), QwenEngine::Label.new("ask", "B")],
        [[11_i32], [12_i32]],
      )
    end
    expect_raises(ArgumentError, /label count/) do
      NativeRuntime.resolve_label_ids(labels, [[11_i32]])
    end
  end

  it "bounds request sequence capacity" do
    NativeRuntime.effective_max_seq(128_i32, nil).should eq(128_i32)
    NativeRuntime.effective_max_seq(128_i32, 64_i32).should eq(64_i32)
    expect_raises(ArgumentError, /max_seq/) { NativeRuntime.effective_max_seq(0_i32, nil) }
    expect_raises(ArgumentError, /exceeds runtime capacity/) { NativeRuntime.effective_max_seq(128_i32, 129_i32) }
  end

  it "fails closed on unsupported non-none reasoning before state creation" do
    NativeRuntime.validate_reasoning_effort_supported!(
      QwenEngine::ReasoningEffort::None,
      false,
    )
    NativeRuntime.validate_reasoning_effort_supported!(
      QwenEngine::ReasoningEffort::XHigh,
      true,
    )
    expect_raises(ArgumentError, /does not support reasoning_effort/) do
      NativeRuntime.validate_reasoning_effort_supported!(
        QwenEngine::ReasoningEffort::Low,
        false,
      )
    end
  end

  it "runs optional model-backed greedy and label-score parity" do
    pending!("set QWEN35_NATIVE_RUNTIME_MODEL_SMOKE=1") unless ENV["QWEN35_NATIVE_RUNTIME_MODEL_SMOKE"]? == "1"
    model_path = ENV["QWEN35_NATIVE_RUNTIME_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    pending!("Qwen model not present") unless File.exists?(model_path)

    prefix_messages = [QwenEngine::Message.new("system", "Answer tersely.")]
    messages = prefix_messages + [QwenEngine::Message.new("user", "Reply with one word: hello")]
    labels = [QwenEngine::Label.new("a", "A"), QwenEngine::Label.new("b", "B")]

    cache_root = File.tempname("qwen35-native-runtime-cache")
    Dir.mkdir_p(cache_root)
    runtime = NativeRuntime.new(
      model_path,
      max_seq: 256,
      prompt_cache_root: cache_root,
      prompt_cache_resident_states: 1,
    )
    warm_entry = runtime.prewarm_prefix(prefix_messages, max_seq: 256)
    warm_entry.prefix_len.should be > 0
    warm_entry.max_seq.should eq(256)
    generation_route = runtime.preflight(QwenEngine::Route::GenerateGreedy, QwenEngine::Backend::Auto)
    expect_raises(ArgumentError, /max_tokens/) do
      runtime.generate(
        QwenEngine::GenerateRequest.new(messages: messages, max_tokens: 0),
        generation_route,
      )
    end
    score_route = runtime.preflight(QwenEngine::Route::ScoreLabels, QwenEngine::Backend::Auto)
    expect_raises(ArgumentError, /names require uniqueness/) do
      runtime.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "Choose one letter.",
          labels: [QwenEngine::Label.new("same", "A"), QwenEngine::Label.new("same", "B")],
        ),
        score_route,
      )
    end

    engine = QwenEngine.new(runtime)
    result = nil.as(QwenEngine::GenerateResult?)
    low_result = nil.as(QwenEngine::GenerateResult?)
    scores = nil.as(QwenEngine::ScoreLabelsResult?)
    begin
      result = engine.generate(QwenEngine::GenerateRequest.new(messages: messages, max_tokens: 1, max_seq: 256))
      result.not_nil!.token_ids.size.should eq(1)
      result.not_nil!.completion_tokens.should eq(1)
      result.not_nil!.backend.primary.should eq(QwenEngine::Backend::Metal)
      result.not_nil!.backend.components.should eq([QwenEngine::Backend::Metal, QwenEngine::Backend::CPU])
      result.not_nil!.backend.attribution.should eq(QwenEngine::Attribution::Planned)
      first_cache_stats = runtime.prompt_cache_stats
      first_cache_stats.hits.should eq(1)
      first_cache_stats.misses.should eq(0)
      first_cache_stats.restore_failures.should eq(0)
      first_cache_stats.reused_prefix_tokens.should eq(warm_entry.prefix_len)
      first_cache_stats.replayed_suffix_tokens.should be > 0

      resident_result = engine.generate(QwenEngine::GenerateRequest.new(messages: messages, max_tokens: 1, max_seq: 256))
      resident_result.token_ids.should eq(result.not_nil!.token_ids)
      runtime.prompt_cache_stats.hits.should eq(2)

      if runtime.reasoning_effort_supported
        low_entry = runtime.prewarm_prefix(
          prefix_messages,
          max_seq: 256,
          reasoning_effort: QwenEngine::ReasoningEffort::Low,
        )
        low_entry.token_hash.should_not eq(warm_entry.token_hash)
        low_result = engine.generate(
          QwenEngine::GenerateRequest.new(
            messages: messages,
            max_tokens: 1,
            max_seq: 256,
            reasoning_effort: QwenEngine::ReasoningEffort::Low,
          )
        )
        low_result.not_nil!.reasoning_effort.should eq(QwenEngine::ReasoningEffort::Low)
        runtime.prompt_cache_stats.hits.should eq(3)
      end

      scores = engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "Choose one letter.",
          labels: labels,
          max_seq: 256,
        )
      )
      scores.not_nil!.best.token_id.should_not eq(scores.not_nil!.second.token_id)
      scores.not_nil!.best.logit.should be >= scores.not_nil!.second.logit
      scores.not_nil!.backend.attribution.should eq(QwenEngine::Attribution::Planned)

      # The second live runtime is rejected before it can replace the global
      # Metal mmap registration used by the first one.
      expect_raises(QwenEngine::BackendMismatch, /another Qwen35NativeRuntime/) do
        NativeRuntime.new(model_path, max_seq: 256)
      end
    ensure
      engine.close
    end

    # A new process/store must reject a corrupted on-disk artifact and fall
    # back to ordinary prefill without changing the public result.
    File.open(warm_entry.artifact_path, "a") { |file| file.write_byte(0_u8) }
    fallback_runtime = NativeRuntime.new(
      model_path,
      max_seq: 256,
      prompt_cache_root: cache_root,
      prompt_cache_resident_states: 0,
    )
    begin
      fallback_route = fallback_runtime.preflight(QwenEngine::Route::GenerateGreedy, QwenEngine::Backend::Auto)
      fallback_result = fallback_runtime.generate(
        QwenEngine::GenerateRequest.new(messages: messages, max_tokens: 1, max_seq: 256),
        fallback_route,
      )
      fallback_result.token_ids.should eq(result.not_nil!.token_ids)
      fallback_stats = fallback_runtime.prompt_cache_stats
      fallback_stats.hits.should eq(0)
      fallback_stats.restore_failures.should eq(1)

      if expected_low = low_result
        cold_low = fallback_runtime.generate(
          QwenEngine::GenerateRequest.new(
            messages: messages,
            max_tokens: 1,
            max_seq: 256,
            reasoning_effort: QwenEngine::ReasoningEffort::Low,
          ),
          fallback_route,
        )
        cold_low.token_ids.should eq(expected_low.token_ids)
        cold_low.reasoning_effort.should eq(QwenEngine::ReasoningEffort::Low)
        cold_low_stats = fallback_runtime.prompt_cache_stats
        cold_low_stats.hits.should eq(1)
        cold_low_stats.restore_failures.should eq(1)
      end
    ensure
      fallback_runtime.close
    end

    # Compare the public results against the previous low-level CPU/Metal
    # primitives after the runtime has released its mmap registration. This is
    # a real one-token/logit parity check, not self-consistency of the wrapper.
    gguf = ML::GGUF::GGUFFile.new(model_path)
    tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(
      gguf,
      model_path,
      ENV["LLAMA_TOKENIZE_BIN"]? || "",
    )
    gguf.close
    weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
    begin
      rendered = ML::GGUF::Qwen35Chat.render(
        messages.map { |message| ML::GGUF::Qwen35Chat::Message.new(message.role, message.content) },
        add_generation_prompt: true,
        enable_thinking: false,
      )
      prompt_ids = tokenizer.encode(rendered, add_bos_override: false)
      state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: 256)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)
      expected_token, _expected_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, state)
      result.not_nil!.token_ids[0].should eq(expected_token)

      if actual_low = low_result
        low_rendered = ML::GGUF::Qwen35Chat.render(
          messages.map { |message| ML::GGUF::Qwen35Chat::Message.new(message.role, message.content) },
          add_generation_prompt: true,
          reasoning_effort: QwenEngine::ReasoningEffort::Low,
        )
        low_prompt_ids = tokenizer.encode(low_rendered, add_bos_override: false)
        low_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: 256)
        ML::GGUF::Qwen35CPU.prepare_state_metal!(low_state, weights.hparams)
        expected_low_token, _expected_low_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, low_prompt_ids, 0, low_state)
        actual_low.token_ids[0].should eq(expected_low_token)
      end

      score_prompt_ids = tokenizer.encode("Choose one letter.", add_bos_override: false)
      encoded_labels = labels.map { |label| tokenizer.encode(label.text, add_bos_override: false) }
      encoded_labels.each { |tokens| tokens.size.should eq(1) }
      label_ids = encoded_labels.map { |tokens| tokens[0] }
      score_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: 256)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(score_state, weights.hparams)
      if score_prompt_ids.size > 1
        ML::GGUF::Qwen35CPU.prefill_tokens(weights, score_prompt_ids[0...-1], 0, score_state)
      end
      logits = ML::GGUF::Qwen35CPU.forward(weights, score_prompt_ids[-1], score_prompt_ids.size - 1, score_state)
      expected_best, expected_second = reference_top_two(logits, label_ids)
      scores.not_nil!.best.token_id.should eq(expected_best)
      scores.not_nil!.second.token_id.should eq(expected_second)
      scores.not_nil!.best.logit.should be_close(logits[expected_best], 1.0e-3_f32)
      scores.not_nil!.second.logit.should be_close(logits[expected_second], 1.0e-3_f32)
    ensure
      weights.close
      FileUtils.rm_rf(cache_root) if Dir.exists?(cache_root)
    end
  end
end
