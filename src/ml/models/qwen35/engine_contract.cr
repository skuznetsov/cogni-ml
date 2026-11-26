module ML::GGUF
  # Stable request/result boundary for Qwen 3.5/3.6 inference.
  #
  # Backend implementations own weights, tokenizers, and mutable decode state.
  # Consumers should depend on this contract rather than Qwen35CPU, Metal, or
  # CUDA runner internals.
  class Qwen35Engine
    enum Backend
      Auto
      CPU
      Metal
      CUDA
    end

    # Planned identities describe a preflight execution envelope. Observed
    # identities are backed by runtime attribution for the completed operation.
    enum Attribution
      Planned
      Observed
    end

    # An operation selected by the caller. The runtime may reject an operation
    # during preflight when the selected backend cannot provide it.
    enum Route
      GenerateGreedy
      ScoreLabels
    end

    # Qwen 3.8 accepts only low, medium, and xhigh in its tokenizer template.
    # None is an engine compatibility mode that preserves the historical
    # enable_thinking=false generation suffix; it is not passed as a template
    # reasoning_effort string.
    enum ReasoningEffort
      None
      Low
      Medium
      XHigh
    end

    class BackendMismatch < Exception
    end

    class RouteMismatch < Exception
    end

    class Closed < Exception
    end

    record BackendIdentity,
      requested : Backend,
      primary : Backend,
      components : Array(Backend),
      model_id : String,
      attribution : Attribution = Attribution::Observed do
      def initialize(
        @requested : Backend,
        @primary : Backend,
        @components : Array(Backend),
        @model_id : String,
        @attribution : Attribution = Attribution::Observed,
      )
        raise ArgumentError.new("backend primary cannot be Auto") if @primary.auto?
        raise ArgumentError.new("backend components must not be empty") if @components.empty?
        raise ArgumentError.new("backend components cannot contain Auto") if @components.any?(&.auto?)
        raise ArgumentError.new("backend components must include primary #{@primary}") unless @components.includes?(@primary)
        raise ArgumentError.new("backend components must be unique") unless @components.uniq.size == @components.size
        raise ArgumentError.new("backend model_id must not be empty") if @model_id.strip.empty?
      end

      def satisfies?(required : Backend) : Bool
        required.auto? || (@attribution.observed? && @primary == required)
      end
    end

    # The preflight result is passed back into the runtime operation. Keeping
    # this value separate from the public result makes route drift observable.
    record PreflightRoute,
      operation : Route,
      backend : BackendIdentity

    record Message,
      role : String,
      content : String

    record GenerateRequest,
      messages : Array(Message),
      max_tokens : Int32,
      temperature : Float64 = 0.0,
      max_seq : Int32? = nil,
      reasoning_effort : ReasoningEffort = ReasoningEffort::None,
      session_id : String? = nil,
      checkpoint_id : String? = nil

    record GenerateResult,
      text : String,
      token_ids : Array(Int32),
      prompt_tokens : Int32,
      completion_tokens : Int32,
      backend : BackendIdentity,
      route : Route,
      reasoning_effort : ReasoningEffort = ReasoningEffort::None,
      checkpoint_id : String? = nil,
      checkpoint_pending : Bool = false do
      def checkpoint_pending? : Bool
        checkpoint_pending
      end
    end

    record Label,
      name : String,
      text : String,
      token_id : Int32? = nil

    record ScoreLabelsRequest,
      prompt : String,
      labels : Array(Label),
      max_seq : Int32? = nil

    # Keep the old type names as source-compatible aliases while the operation
    # itself is intentionally named score_labels.
    alias ClassificationRequest = ScoreLabelsRequest

    record LabelScore,
      label : Label,
      token_id : Int32,
      logit : Float32

    record ScoreLabelsResult,
      best : LabelScore,
      second : LabelScore,
      backend : BackendIdentity,
      route : Route do
      def margin : Float32
        @best.logit - @second.logit
      end
    end

    alias ClassificationResult = ScoreLabelsResult

    abstract class Runtime
      # Preflight must be side-effect free with respect to mutable decode state.
      # It selects a backend execution envelope before an operation can mutate
      # state. The result may refine a planned envelope to observed execution.
      abstract def preflight(operation : Route, requested_backend : Backend) : PreflightRoute

      abstract def generate(request : GenerateRequest, route : PreflightRoute) : GenerateResult
      abstract def score_labels(request : ScoreLabelsRequest, route : PreflightRoute) : ScoreLabelsResult

      # Runtime implementations must make this operation idempotent. The
      # engine may retry it when a cleanup attempt raises.
      abstract def close : Nil
    end

    getter required_backend : Backend

    @mutex = Mutex.new
    @closed = false

    def initialize(@runtime : Runtime, @required_backend : Backend = Backend::Auto)
    end

    def closed? : Bool
      @mutex.synchronize { @closed }
    end

    # Closing is deliberately idempotent so provider shutdown paths can call it
    # from multiple ownership layers without double-freeing backend resources.
    def close : Nil
      @mutex.synchronize do
        return if @closed

        # Keep the engine open when cleanup fails so the caller can retry. The
        # runtime contract requires its close operation to be idempotent.
        @runtime.close
        @closed = true
      end
    end

    def generate(request : GenerateRequest) : GenerateResult
      @mutex.synchronize do
        ensure_open!
        validate_generate_request!(request)
        route = preflight!(Route::GenerateGreedy)
        result = @runtime.generate(request, route)
        validate_backend!(result.backend)
        validate_result_route!(result.route, result.backend, route)
        validate_generate_result!(request, result)
        result
      end
    end

    def score_labels(request : ScoreLabelsRequest) : ScoreLabelsResult
      @mutex.synchronize do
        ensure_open!
        validate_score_labels_request!(request)
        route = preflight!(Route::ScoreLabels)
        result = @runtime.score_labels(request, route)
        validate_backend!(result.backend)
        validate_result_route!(result.route, result.backend, route)
        validate_score_labels_result!(request, result)
        result
      end
    end

    private def ensure_open! : Nil
      raise Closed.new("Qwen35Engine is closed") if @closed
    end

    private def preflight!(operation : Route) : PreflightRoute
      route = @runtime.preflight(operation, @required_backend)
      unless route.operation == operation
        raise RouteMismatch.new(
          "runtime preflight selected #{route.operation} for requested #{operation}"
        )
      end
      validate_backend!(route.backend)
      route
    end

    private def validate_backend!(identity : BackendIdentity) : Nil
      unless identity.requested == @required_backend
        raise BackendMismatch.new(
          "required #{@required_backend}, but preflight requested #{identity.requested} " \
          "for #{identity.model_id}"
        )
      end
      return if identity.satisfies?(@required_backend)

      raise BackendMismatch.new(
        "required #{@required_backend} with observed attribution, but runtime reported " \
        "#{identity.attribution} #{identity.primary} for #{identity.model_id}"
      )
    end

    private def validate_result_route!(
      result_route : Route,
      result_backend : BackendIdentity,
      expected : PreflightRoute,
    ) : Nil
      unless result_route == expected.operation
        raise RouteMismatch.new(
          "runtime result reported #{result_route}, expected #{expected.operation}"
        )
      end
      return if result_backend == expected.backend

      planned = expected.backend
      refined = planned.attribution.planned? &&
                result_backend.attribution.observed? &&
                result_backend.requested == planned.requested &&
                result_backend.model_id == planned.model_id &&
                result_backend.components.all? { |component| planned.components.includes?(component) }
      return if refined

      raise RouteMismatch.new("runtime result backend identity drifted outside the preflight route")
    end

    private def validate_generate_request!(request : GenerateRequest) : Nil
      raise ArgumentError.new("generation requires at least one message") if request.messages.empty?
      raise ArgumentError.new("generation max_tokens must be positive") unless request.max_tokens > 0
      raise ArgumentError.new("only deterministic temperature=0 generation is admitted") unless request.temperature == 0.0
      validate_max_seq!(request.max_seq)
      raise ArgumentError.new("generation requires non-empty message content") if request.messages.all? { |message| message.content.strip.empty? }

      request.messages.each do |message|
        raise ArgumentError.new("message role must not be empty") if message.role.strip.empty?
      end
      if session_id = request.session_id
        unless session_id.bytesize > 0 && session_id.bytesize <= 1024
          raise ArgumentError.new("generation session_id is outside 1..1024 bytes")
        end
      elsif request.checkpoint_id
        raise ArgumentError.new("generation checkpoint_id requires session_id")
      end
      if checkpoint_id = request.checkpoint_id
        unless checkpoint_id.matches?(/\A[0-9a-f]{64}\z/)
          raise ArgumentError.new("generation checkpoint_id is invalid")
        end
      end
    end

    private def validate_generate_result!(request : GenerateRequest, result : GenerateResult) : Nil
      unless result.reasoning_effort == request.reasoning_effort
        raise "runtime reasoning effort #{result.reasoning_effort} does not match requested #{request.reasoning_effort}"
      end
      if result.checkpoint_pending? && (request.session_id.nil? || result.checkpoint_id.nil?)
        raise "runtime returned a pending checkpoint outside a session checkpoint result"
      end
      if request.session_id
        checkpoint_id = result.checkpoint_id
        unless checkpoint_id && checkpoint_id.matches?(/\A[0-9a-f]{64}\z/)
          raise "runtime did not return a valid session checkpoint"
        end
      elsif result.checkpoint_id
        raise "runtime returned a checkpoint for a request without session_id"
      end
      raise "runtime returned a negative prompt token count" if result.prompt_tokens < 0
      raise "runtime returned a negative completion token count" if result.completion_tokens < 0
      if result.completion_tokens > request.max_tokens
        raise "runtime completion token count #{result.completion_tokens} exceeds max_tokens #{request.max_tokens}"
      end
      unless result.completion_tokens == result.token_ids.size
        raise "runtime completion token count #{result.completion_tokens} does not match #{result.token_ids.size} token ids"
      end
      raise "runtime returned a negative token id" if result.token_ids.any? { |id| id < 0 }
      if max_seq = request.max_seq
        total_tokens = result.prompt_tokens.to_i64 + result.completion_tokens.to_i64
        if total_tokens > max_seq
          raise "runtime token count #{total_tokens} exceeds max_seq #{max_seq}"
        end
      end
    end

    private def validate_score_labels_request!(request : ScoreLabelsRequest) : Nil
      raise ArgumentError.new("label scoring prompt must not be empty") if request.prompt.strip.empty?
      raise ArgumentError.new("label scoring requires at least two labels") if request.labels.size < 2
      validate_max_seq!(request.max_seq)

      names = request.labels.map do |label|
        name = label.name.strip
        raise ArgumentError.new("label scoring name must not be empty") if name.empty?
        name
      end
      texts = request.labels.map do |label|
        text = label.text.strip
        raise ArgumentError.new("label scoring text must not be empty") if text.empty?
        text
      end

      raise ArgumentError.new("label scoring names require uniqueness") unless names.uniq.size == names.size
      raise ArgumentError.new("label scoring texts require uniqueness") unless texts.uniq.size == texts.size

      token_ids = request.labels.compact_map(&.token_id)
      unless token_ids.empty? || token_ids.size == request.labels.size
        raise ArgumentError.new("label scoring labels must either all provide token ids or none")
      end
      raise ArgumentError.new("label scoring labels require non-negative token ids") if token_ids.any? { |id| id < 0 }
      raise ArgumentError.new("label scoring labels require unique token ids") unless token_ids.uniq.size == token_ids.size
    end

    private def validate_score_labels_result!(
      request : ScoreLabelsRequest,
      result : ScoreLabelsResult,
    ) : Nil
      raise "runtime returned an unknown best label" unless request.labels.includes?(result.best.label)
      raise "runtime returned an unknown second label" unless request.labels.includes?(result.second.label)
      raise "runtime returned the same best and second label" if result.best.label == result.second.label
      raise "runtime returned the same token id for best and second labels" if result.best.token_id == result.second.token_id
      raise "runtime returned a negative label token id" if result.best.token_id < 0 || result.second.token_id < 0
      raise "runtime returned a non-finite label logit" unless result.best.logit.finite? && result.second.logit.finite?
      raise "runtime returned labels out of descending logit order" if result.best.logit < result.second.logit

      if expected = result.best.label.token_id
        raise "runtime best token id does not match requested label" unless expected == result.best.token_id
      end
      if expected = result.second.label.token_id
        raise "runtime second token id does not match requested label" unless expected == result.second.token_id
      end
    end

    private def validate_max_seq!(max_seq : Int32?) : Nil
      return unless max_seq
      raise ArgumentError.new("max_seq must be positive") unless max_seq.not_nil! > 0
    end
  end
end
