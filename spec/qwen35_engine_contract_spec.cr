require "./spec_helper"
require "../src/ml/gguf/qwen35_engine_contract"

private alias QwenEngine = ML::GGUF::Qwen35Engine

private class FakeQwen35EngineRuntime < QwenEngine::Runtime
  getter generate_calls = 0
  getter score_labels_calls = 0
  getter preflight_calls = [] of QwenEngine::Route
  getter received_generate_routes = [] of QwenEngine::PreflightRoute
  getter received_score_labels_routes = [] of QwenEngine::PreflightRoute
  getter preflight_requested_backends = [] of QwenEngine::Backend
  getter close_calls = 0
  property generate_backend : QwenEngine::BackendIdentity
  property score_labels_backend : QwenEngine::BackendIdentity
  property generate_result_backend : QwenEngine::BackendIdentity?
  property score_labels_result_backend : QwenEngine::BackendIdentity?
  property generate_result_route : QwenEngine::Route = QwenEngine::Route::GenerateGreedy
  property score_labels_result_route : QwenEngine::Route = QwenEngine::Route::ScoreLabels
  property preflight_result_operation : QwenEngine::Route?
  property generate_token_ids : Array(Int32) = [7_i32]
  property generate_completion_tokens : Int32 = 1
  property score_best_token_id : Int32 = 11_i32
  property score_second_token_id : Int32 = 12_i32
  property close_failures_remaining : Int32 = 0
  property generate_entered : Channel(Nil)?
  property generate_release : Channel(Nil)?

  def initialize(
    @generate_backend : QwenEngine::BackendIdentity,
    score_labels_backend : QwenEngine::BackendIdentity? = nil,
  )
    @score_labels_backend = score_labels_backend || @generate_backend
  end

  def preflight(operation : QwenEngine::Route, requested_backend : QwenEngine::Backend) : QwenEngine::PreflightRoute
    @preflight_calls << operation
    @preflight_requested_backends << requested_backend
    backend = operation == QwenEngine::Route::GenerateGreedy ? @generate_backend : @score_labels_backend
    QwenEngine::PreflightRoute.new(@preflight_result_operation || operation, backend)
  end

  def generate(request : QwenEngine::GenerateRequest, route : QwenEngine::PreflightRoute) : QwenEngine::GenerateResult
    @generate_calls += 1
    @received_generate_routes << route
    @generate_entered.try(&.send(nil))
    @generate_release.try(&.receive)
    QwenEngine::GenerateResult.new(
      text: "ok",
      token_ids: @generate_token_ids,
      prompt_tokens: request.messages.size,
      completion_tokens: @generate_completion_tokens,
      backend: @generate_result_backend || route.backend,
      route: @generate_result_route,
    )
  end

  def score_labels(request : QwenEngine::ScoreLabelsRequest, route : QwenEngine::PreflightRoute) : QwenEngine::ScoreLabelsResult
    @score_labels_calls += 1
    @received_score_labels_routes << route
    QwenEngine::ScoreLabelsResult.new(
      best: QwenEngine::LabelScore.new(request.labels[0], @score_best_token_id, 3.5_f32),
      second: QwenEngine::LabelScore.new(request.labels[1], @score_second_token_id, 2.0_f32),
      backend: @score_labels_result_backend || route.backend,
      route: @score_labels_result_route,
    )
  end

  def close : Nil
    @close_calls += 1
    if @close_failures_remaining > 0
      @close_failures_remaining -= 1
      raise "fake runtime close failure"
    end
  end
end

private def fake_qwen_identity(
  primary : QwenEngine::Backend = QwenEngine::Backend::CPU,
  components : Array(QwenEngine::Backend) = [QwenEngine::Backend::CPU],
  requested : QwenEngine::Backend = QwenEngine::Backend::Auto,
  attribution : QwenEngine::Attribution = QwenEngine::Attribution::Observed,
) : QwenEngine::BackendIdentity
  QwenEngine::BackendIdentity.new(
    requested: requested,
    primary: primary,
    components: components,
    model_id: "fake:model",
    attribution: attribution,
  )
end

describe ML::GGUF::Qwen35Engine do
  it "preflights and forwards bounded generation with actual backend identity" do
    runtime = FakeQwen35EngineRuntime.new(
      fake_qwen_identity(
        QwenEngine::Backend::Metal,
        [QwenEngine::Backend::Metal, QwenEngine::Backend::CPU],
        QwenEngine::Backend::Metal,
      )
    )
    engine = QwenEngine.new(runtime, required_backend: QwenEngine::Backend::Metal)

    result = engine.generate(
      QwenEngine::GenerateRequest.new(
        messages: [QwenEngine::Message.new("user", "hello")],
        max_tokens: 8,
      )
    )

    result.text.should eq("ok")
    result.backend.primary.should eq(QwenEngine::Backend::Metal)
    result.backend.components.should eq([QwenEngine::Backend::Metal, QwenEngine::Backend::CPU])
    result.route.should eq(QwenEngine::Route::GenerateGreedy)
    runtime.preflight_calls.should eq([QwenEngine::Route::GenerateGreedy])
    runtime.preflight_requested_backends.should eq([QwenEngine::Backend::Metal])
    runtime.received_generate_routes.size.should eq(1)
    runtime.received_generate_routes[0].operation.should eq(QwenEngine::Route::GenerateGreedy)
    runtime.generate_calls.should eq(1)
  end

  it "rejects invalid or currently unsupported generation requests before calling the runtime" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    engine = QwenEngine.new(runtime)

    expect_raises(ArgumentError, /at least one message/) do
      engine.generate(QwenEngine::GenerateRequest.new(messages: [] of QwenEngine::Message, max_tokens: 8))
    end
    expect_raises(ArgumentError, /max_tokens/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 0,
        )
      )
    end
    expect_raises(ArgumentError, /deterministic/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 8,
          temperature: 0.5,
        )
      )
    end

    runtime.generate_calls.should eq(0)
    runtime.preflight_calls.should be_empty
  end

  it "scores constrained labels through a preflight route" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    engine = QwenEngine.new(runtime)
    labels = [
      QwenEngine::Label.new("allow", "A", 11_i32),
      QwenEngine::Label.new("ask", "B", 12_i32),
    ]

    result = engine.score_labels(
      QwenEngine::ScoreLabelsRequest.new(
        prompt: "Choose one label.",
        labels: labels,
      )
    )

    result.best.label.should eq(labels[0])
    result.second.label.should eq(labels[1])
    result.margin.should be_close(1.5_f32, 1.0e-6_f32)
    result.route.should eq(QwenEngine::Route::ScoreLabels)
    runtime.score_labels_calls.should eq(1)
    runtime.preflight_calls.should eq([QwenEngine::Route::ScoreLabels])
    runtime.received_score_labels_routes[0].operation.should eq(QwenEngine::Route::ScoreLabels)
  end

  it "allows each operation to preflight a distinct backend component route" do
    generate_backend = fake_qwen_identity(
      QwenEngine::Backend::Metal,
      [QwenEngine::Backend::Metal, QwenEngine::Backend::CPU],
    )
    score_backend = fake_qwen_identity(
      QwenEngine::Backend::CPU,
      [QwenEngine::Backend::CPU],
    )
    runtime = FakeQwen35EngineRuntime.new(generate_backend, score_labels_backend: score_backend)
    engine = QwenEngine.new(runtime)

    generation = engine.generate(
      QwenEngine::GenerateRequest.new(
        messages: [QwenEngine::Message.new("user", "hello")],
        max_tokens: 1,
      )
    )
    scores = engine.score_labels(
      QwenEngine::ScoreLabelsRequest.new(
        prompt: "Choose one label.",
        labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("ask", "B", 12_i32)],
      )
    )

    generation.backend.components.should eq([QwenEngine::Backend::Metal, QwenEngine::Backend::CPU])
    scores.backend.components.should eq([QwenEngine::Backend::CPU])
    runtime.received_generate_routes[0].backend.should eq(generate_backend)
    runtime.received_score_labels_routes[0].backend.should eq(score_backend)
  end

  it "rejects empty, duplicate, or underspecified labels before runtime execution" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    engine = QwenEngine.new(runtime)

    expect_raises(ArgumentError, /prompt/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("ask", "B", 12_i32)],
        )
      )
    end
    expect_raises(ArgumentError, /at least two/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "classify",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32)],
        )
      )
    end
    expect_raises(ArgumentError, /names require uniqueness/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "classify",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("allow", "B", 12_i32)],
        )
      )
    end
    expect_raises(ArgumentError, /texts require uniqueness/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "classify",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("ask", "A", 12_i32)],
        )
      )
    end
    expect_raises(ArgumentError, /unique token ids/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "classify",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("ask", "B", 11_i32)],
        )
      )
    end

    expect_raises(ArgumentError, /max_seq/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "classify",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("ask", "B", 12_i32)],
          max_seq: 0,
        )
      )
    end

    runtime.score_labels_calls.should eq(0)
    runtime.preflight_calls.should be_empty
  end

  it "rejects a required-backend mismatch after preflight but before execution" do
    changing_runtime = FakeQwen35EngineRuntime.new(
      fake_qwen_identity(
        QwenEngine::Backend::Metal,
        [QwenEngine::Backend::Metal, QwenEngine::Backend::CPU],
        QwenEngine::Backend::CPU,
      )
    )
    engine = QwenEngine.new(changing_runtime, required_backend: QwenEngine::Backend::Metal)

    expect_raises(QwenEngine::BackendMismatch, /required Metal/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
    end
    changing_runtime.generate_calls.should eq(0)
    changing_runtime.preflight_calls.should eq([QwenEngine::Route::GenerateGreedy])
    changing_runtime.preflight_requested_backends.should eq([QwenEngine::Backend::Metal])
  end

  it "rejects route drift returned by the runtime" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    runtime.generate_result_route = QwenEngine::Route::ScoreLabels
    engine = QwenEngine.new(runtime)

    expect_raises(QwenEngine::RouteMismatch, /GenerateGreedy/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
    end
    runtime.generate_calls.should eq(1)
  end

  it "rejects result backend identity drift from the preflight route" do
    runtime = FakeQwen35EngineRuntime.new(
      fake_qwen_identity(
        QwenEngine::Backend::Metal,
        [QwenEngine::Backend::Metal],
        QwenEngine::Backend::Auto,
      )
    )
    runtime.generate_result_backend = fake_qwen_identity(QwenEngine::Backend::CPU)
    engine = QwenEngine.new(runtime)

    expect_raises(QwenEngine::RouteMismatch, /backend identity/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
    end
  end

  it "accepts an observed result that refines a planned backend envelope" do
    planned = fake_qwen_identity(
      QwenEngine::Backend::Metal,
      [QwenEngine::Backend::Metal, QwenEngine::Backend::CPU],
      attribution: QwenEngine::Attribution::Planned,
    )
    observed = fake_qwen_identity(
      QwenEngine::Backend::CPU,
      [QwenEngine::Backend::CPU],
      attribution: QwenEngine::Attribution::Observed,
    )
    runtime = FakeQwen35EngineRuntime.new(planned)
    runtime.generate_result_backend = observed
    engine = QwenEngine.new(runtime)

    result = engine.generate(
      QwenEngine::GenerateRequest.new(
        messages: [QwenEngine::Message.new("user", "hello")],
        max_tokens: 1,
      )
    )

    result.backend.should eq(observed)
  end

  it "rejects a preflight route drift before runtime mutation" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    runtime.preflight_result_operation = QwenEngine::Route::ScoreLabels
    engine = QwenEngine.new(runtime)

    expect_raises(QwenEngine::RouteMismatch, /requested GenerateGreedy/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
    end
    runtime.generate_calls.should eq(0)
  end

  it "rejects malformed runtime results" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    engine = QwenEngine.new(runtime)
    runtime.generate_completion_tokens = 2
    runtime.generate_token_ids = [7_i32, 8_i32]

    expect_raises(Exception, /exceeds max_tokens/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
    end

    runtime.generate_completion_tokens = 1
    runtime.generate_token_ids = [7_i32]
    expect_raises(Exception, /exceeds max_seq/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
          max_seq: 1,
        )
      )
    end
  end

  it "rejects malformed label scores" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    runtime.score_second_token_id = 11_i32
    engine = QwenEngine.new(runtime)

    expect_raises(Exception, /same token id/) do
      engine.score_labels(
        QwenEngine::ScoreLabelsRequest.new(
          prompt: "classify",
          labels: [QwenEngine::Label.new("allow", "A", 11_i32), QwenEngine::Label.new("ask", "B", 12_i32)],
        )
      )
    end
  end

  it "closes the runtime exactly once and rejects operations after close" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    engine = QwenEngine.new(runtime)

    engine.close
    engine.close
    engine.closed?.should be_true
    runtime.close_calls.should eq(1)

    expect_raises(QwenEngine::Closed, /closed/) do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
    end
    runtime.generate_calls.should eq(0)
  end

  it "keeps the engine open when runtime close fails so cleanup can be retried" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    runtime.close_failures_remaining = 1
    engine = QwenEngine.new(runtime)

    expect_raises(Exception, /close failure/) { engine.close }
    engine.closed?.should be_false
    runtime.close_calls.should eq(1)

    engine.close
    engine.closed?.should be_true
    runtime.close_calls.should eq(2)
    engine.close
    runtime.close_calls.should eq(2)
  end

  it "serializes close behind an in-flight generation" do
    runtime = FakeQwen35EngineRuntime.new(fake_qwen_identity)
    runtime.generate_entered = Channel(Nil).new(1)
    runtime.generate_release = Channel(Nil).new(1)
    engine = QwenEngine.new(runtime)
    generation_done = Channel(Nil).new(1)
    close_done = Channel(Nil).new(1)

    spawn do
      engine.generate(
        QwenEngine::GenerateRequest.new(
          messages: [QwenEngine::Message.new("user", "hello")],
          max_tokens: 1,
        )
      )
      generation_done.send(nil)
    end
    runtime.generate_entered.not_nil!.receive

    spawn do
      engine.close
      close_done.send(nil)
    end
    select
    when close_done.receive
      fail "close completed while generation was still in flight"
    when timeout(10.milliseconds)
    end

    runtime.generate_release.not_nil!.send(nil)
    generation_done.receive
    close_done.receive
    engine.closed?.should be_true
    runtime.close_calls.should eq(1)
  end
end
