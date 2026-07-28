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

  it "runs optional model-backed greedy and label-score parity" do
    pending!("set QWEN35_NATIVE_RUNTIME_MODEL_SMOKE=1") unless ENV["QWEN35_NATIVE_RUNTIME_MODEL_SMOKE"]? == "1"
    model_path = ENV["QWEN35_NATIVE_RUNTIME_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    pending!("Qwen model not present") unless File.exists?(model_path)

    messages = [QwenEngine::Message.new("user", "Reply with one word: hello")]
    labels = [QwenEngine::Label.new("a", "A"), QwenEngine::Label.new("b", "B")]

    runtime = NativeRuntime.new(model_path, max_seq: 256)
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
    scores = nil.as(QwenEngine::ScoreLabelsResult?)
    begin
      result = engine.generate(QwenEngine::GenerateRequest.new(messages: messages, max_tokens: 1, max_seq: 256))
      result.not_nil!.token_ids.size.should eq(1)
      result.not_nil!.completion_tokens.should eq(1)
      result.not_nil!.backend.primary.should eq(QwenEngine::Backend::Metal)
      result.not_nil!.backend.components.should eq([QwenEngine::Backend::Metal, QwenEngine::Backend::CPU])
      result.not_nil!.backend.attribution.should eq(QwenEngine::Attribution::Planned)

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
    end
  end
end
