require "http/client"
require "json"
require "uri"
require "time"
require "./gemma4_chat"
require "./gemma4_energy_profile"

module ML::GGUF
  # Runs two explicit, bounded Gemma 4 phases against a local llama.cpp server.
  # Reasoning is kept only in memory to form the answer prompt and never enters
  # the returned report.
  class Gemma4EnergyRunner
    class Error < Exception
    end

    alias RequestTransport = Proc(String, JSON::Any?, JSON::Any)

    REQUEST_TIMEOUT         = Time::Span.new(seconds: 600)
    THINK_PREFIX            = "<|channel>thought\n"
    CHANNEL_END             = "<channel|>"
    TURN_END                = "<turn|>"
    CONTROL_MARKERS         = ["<|think|>", "<|channel>", CHANNEL_END, "<|turn>", TURN_END]
    PRIVATE_THOUGHT_MARKERS = ["<|think|>", "<|channel>", CHANNEL_END]

    getter model_identity : String?

    @server_uri : URI
    @base_path : String
    @request_transport : RequestTransport?

    def initialize(server_url : String, @request_transport : RequestTransport? = nil)
      begin
        uri = URI.parse(server_url)
      rescue
        raise ArgumentError.new("server_url must be a valid local HTTP URL")
      end

      host = uri.host
      unless {"http", "https"}.includes?(uri.scheme) && host && !host.empty?
        raise ArgumentError.new("server_url must be a valid local HTTP URL")
      end
      if uri.user || uri.password
        raise ArgumentError.new("server_url must not embed credentials")
      end
      if uri.query || uri.fragment
        raise ArgumentError.new("server_url must not include a query or fragment")
      end

      @server_uri = uri
      @base_path = uri.path.to_s.rstrip("/")
      @model_identity = nil
    end

    def run(prompt : String, profile : Gemma4EnergyProfile) : JSON::Any
      raise ArgumentError.new("prompt must not be empty") if prompt.strip.empty?

      started_at_epoch_ms = Time.utc.to_unix_ms
      monotonic_start = Time.instant

      props = request("/props")
      model_identity = validate_model_identity(props)
      context_limit = required_positive_integer(
        nested(props, "default_generation_settings", "n_ctx"),
        "server context limit",
      )
      validate_gemma4_markers
      @model_identity = model_identity

      reasoning_enabled = profile.reasoning_tokens > 0
      rendered_prompt = Gemma4Chat.render_user_prompt(prompt, enable_thinking: reasoning_enabled)
      thought_prompt = reasoning_enabled ? rendered_prompt + THINK_PREFIX : rendered_prompt

      reasoning_content = ""
      reasoning_phase = if profile.reasoning_tokens == 0
                          phase_report(
                            budget: 0,
                            prompt_tokens: 0,
                            predicted_tokens: 0,
                            returned_token_ids: 0,
                            outcome: "skipped_zero_budget",
                            forced_close: false,
                            truncated: false,
                            stop_type: "skipped",
                            stopping_word: "",
                          )
                        else
                          response = complete(thought_prompt, profile.reasoning_tokens, context_limit)
                          reasoning_content = response.content

                          outcome, forced_close = case response.stop_type
                                                  when "limit"
                                                    if response.predicted_tokens != profile.reasoning_tokens
                                                      raise Error.new("reasoning usage did not match its token limit")
                                                    end
                                                    {"forced_close", true}
                                                  when "word"
                                                    if response.stopping_word == CHANNEL_END
                                                      {"natural_channel_close", false}
                                                    elsif response.stopping_word == TURN_END
                                                      raise Error.new("reasoning ended the turn before the answer phase")
                                                    else
                                                      raise Error.new("reasoning stopped on an unexpected marker")
                                                    end
                                                  when "eos"
                                                    raise Error.new("reasoning ended with EOS before the answer phase")
                                                  else
                                                    raise Error.new("reasoning returned an unexpected stop outcome")
                                                  end

                          phase_report(
                            budget: profile.reasoning_tokens,
                            prompt_tokens: response.prompt_tokens,
                            predicted_tokens: response.predicted_tokens,
                            returned_token_ids: response.returned_token_ids,
                            outcome: outcome,
                            forced_close: forced_close,
                            truncated: false,
                            stop_type: response.stop_type,
                            stopping_word: response.stopping_word,
                          )
                        end

      # Re-submit the completed prompt so llama.cpp accounts for prefill time
      # and validates the answer context independently of the reasoning cap.
      answer_prompt = if reasoning_enabled
                        thought_prompt + reasoning_content + CHANNEL_END
                      else
                        # Gemma4Chat already closes the empty thought channel
                        # when thinking is disabled.
                        thought_prompt
                      end
      answer_response = complete(answer_prompt, profile.answer_tokens, context_limit)
      # Private control markers in answer content could reopen or expose the
      # thought channel, so treat them as a malformed backend response.
      if PRIVATE_THOUGHT_MARKERS.any? { |marker| answer_response.content.includes?(marker) }
        raise Error.new("answer reopened a private Gemma 4 thought channel")
      end
      if answer_response.content.strip.empty?
        raise Error.new("llama.cpp returned an empty answer")
      end
      answer_outcome, answer_truncated = case answer_response.stop_type
                                         when "limit"
                                           if answer_response.predicted_tokens != profile.answer_tokens
                                             raise Error.new("answer usage did not match its token limit")
                                           end
                                           {"answer_limit", true}
                                         when "eos"
                                           {"eos", false}
                                         when "word"
                                           if answer_response.stopping_word == TURN_END
                                             {"turn_end", false}
                                           elsif answer_response.stopping_word == CHANNEL_END
                                             raise Error.new("answer stopped at a private channel boundary")
                                           else
                                             raise Error.new("answer stopped on an unexpected marker")
                                           end
                                         else
                                           raise Error.new("answer returned an unexpected stop outcome")
                                         end
      answer_phase = phase_report(
        budget: profile.answer_tokens,
        prompt_tokens: answer_response.prompt_tokens,
        predicted_tokens: answer_response.predicted_tokens,
        returned_token_ids: answer_response.returned_token_ids,
        outcome: answer_outcome,
        forced_close: false,
        truncated: answer_truncated,
        stop_type: answer_response.stop_type,
        stopping_word: answer_response.stopping_word,
      )

      ended_at_epoch_ms = Time.utc.to_unix_ms
      elapsed_ms = (Time.instant - monotonic_start).total_milliseconds.to_i64

      JSON::Any.new({
        "status"                  => JSON::Any.new("completed"),
        "usage_complete"          => JSON::Any.new(true),
        "comparison_eligible"     => JSON::Any.new(false),
        "quality_review_required" => JSON::Any.new(true),
        "model_identity"          => JSON::Any.new(model_identity),
        "profile"                 => JSON::Any.new({
          "name"             => JSON::Any.new(profile.name),
          "reasoning_tokens" => JSON::Any.new(profile.reasoning_tokens),
          "answer_tokens"    => JSON::Any.new(profile.answer_tokens),
        }),
        "answer" => JSON::Any.new(answer_response.content),
        "phases" => JSON::Any.new({
          "reasoning" => JSON::Any.new(reasoning_phase),
          "answer"    => JSON::Any.new(answer_phase),
        }),
        "started_at_epoch_ms" => JSON::Any.new(started_at_epoch_ms),
        "ended_at_epoch_ms"   => JSON::Any.new(ended_at_epoch_ms),
        "elapsed_ms"          => JSON::Any.new(elapsed_ms),
      })
    end

    private def validate_model_identity(props : JSON::Any) : String
      model_path = nested(props, "model_path")
      identity = model_path.try(&.as_s?)
      raise Error.new("llama.cpp /props did not identify a loaded model") unless identity && !identity.empty?

      normalized = identity.downcase.gsub(/[^a-z0-9]/, "")
      unless normalized.includes?("gemma4")
        raise Error.new("loaded model path does not identify Gemma 4")
      end

      identity
    end

    private def validate_gemma4_markers : Nil
      marker_source = CONTROL_MARKERS.join("\n")
      token_response = request("/tokenize", JSON::Any.new({
        "content"       => JSON::Any.new(marker_source),
        "parse_special" => JSON::Any.new(true),
        "with_pieces"   => JSON::Any.new(true),
      }))
      token_rows = nested(token_response, "tokens").try(&.as_a?)
      raise Error.new("Gemma 4 marker tokenization returned no token list") unless token_rows

      pieces = {} of String => Int64
      marker_counts = Hash(String, Int32).new(0)
      token_rows.each do |row|
        row_hash = row.as_h?
        next unless row_hash
        piece = row_hash["piece"]?.try(&.as_s?)
        token_id = row_hash["id"]?.try(&.as_i64?)
        if piece && token_id && CONTROL_MARKERS.includes?(piece)
          marker_counts[piece] += 1
          pieces[piece] = token_id
        end
      end

      CONTROL_MARKERS.each do |marker|
        unless pieces.has_key?(marker) && marker_counts[marker] == 1
          raise Error.new("loaded tokenizer did not recognize a Gemma 4 control marker as one token")
        end
      end
      ids = CONTROL_MARKERS.map { |marker| pieces[marker] }
      unless ids.uniq.size == CONTROL_MARKERS.size
        raise Error.new("Gemma 4 control markers did not tokenize as distinct special tokens")
      end
    end

    private record Completion,
      content : String,
      stop_type : String,
      stopping_word : String,
      prompt_tokens : Int64,
      predicted_tokens : Int64,
      returned_token_ids : Int64

    private def complete(prompt : String, budget : Int32, context_limit : Int64) : Completion
      response = request("/completion", JSON::Any.new({
        "prompt"            => JSON::Any.new(prompt),
        "n_predict"         => JSON::Any.new(budget),
        "temperature"       => JSON::Any.new(0.0),
        "seed"              => JSON::Any.new(1),
        "cache_prompt"      => JSON::Any.new(false),
        "return_tokens"     => JSON::Any.new(true),
        "timings_per_token" => JSON::Any.new(true),
        "stop"              => JSON::Any.new([CHANNEL_END, TURN_END].map { |token| JSON::Any.new(token) }),
      }))

      content = nested(response, "content").try(&.as_s?)
      stop_type = nested(response, "stop_type").try(&.as_s?)
      stopping_word = nested(response, "stopping_word").try(&.as_s?)
      timings = nested(response, "timings")
      predicted_tokens = nested(timings, "predicted_n").try(&.as_i64?)
      prompt_tokens = nested(response, "tokens_evaluated").try(&.as_i64?)
      truncated = nested(response, "truncated").try(&.as_bool?)
      raw_tokens = nested(response, "tokens").try(&.as_a?)

      if content.nil? || stop_type.nil? || stopping_word.nil? || predicted_tokens.nil? || prompt_tokens.nil? || truncated.nil? || raw_tokens.nil?
        raise Error.new("llama.cpp returned an invalid completion response")
      end
      raise Error.new("llama.cpp reported context truncation") if truncated
      raise Error.new("llama.cpp reported invalid token usage") if predicted_tokens <= 0 || prompt_tokens <= 0
      raise Error.new("llama.cpp exceeded the phase token budget") if predicted_tokens > budget
      raise Error.new("llama.cpp exceeded the loaded context limit") if prompt_tokens + predicted_tokens > context_limit

      raw_tokens.each do |token|
        unless token.as_i64?
          raise Error.new("llama.cpp returned invalid token IDs")
        end
      end
      returned_token_ids = raw_tokens.size.to_i64
      if returned_token_ids > predicted_tokens
        raise Error.new("llama.cpp token IDs exceed its reported generation count")
      end

      Completion.new(
        content,
        stop_type,
        stopping_word,
        prompt_tokens,
        predicted_tokens,
        returned_token_ids,
      )
    end

    private def phase_report(budget : Int32,
                             prompt_tokens : Int64,
                             predicted_tokens : Int64,
                             returned_token_ids : Int64,
                             outcome : String,
                             forced_close : Bool,
                             truncated : Bool,
                             stop_type : String,
                             stopping_word : String) : Hash(String, JSON::Any)
      {
        "budget_tokens"      => JSON::Any.new(budget),
        "prompt_tokens"      => JSON::Any.new(prompt_tokens),
        "predicted_tokens"   => JSON::Any.new(predicted_tokens),
        "returned_token_ids" => JSON::Any.new(returned_token_ids),
        "control_overhead"   => JSON::Any.new(predicted_tokens - returned_token_ids),
        "outcome"            => JSON::Any.new(outcome),
        "forced_close"       => JSON::Any.new(forced_close),
        "truncated"          => JSON::Any.new(truncated),
        "stop_type"          => JSON::Any.new(stop_type),
        "stopping_word"      => JSON::Any.new(stopping_word),
      }
    end

    private def request(path : String, payload : JSON::Any? = nil) : JSON::Any
      if transport = @request_transport
        return transport.call(path, payload)
      end

      client = HTTP::Client.new(@server_uri)
      client.connect_timeout = REQUEST_TIMEOUT
      client.read_timeout = REQUEST_TIMEOUT
      client.write_timeout = REQUEST_TIMEOUT
      request_path = "#{@base_path}#{path}"
      response = if body = payload
                   client.post(request_path, HTTP::Headers{"Content-Type" => "application/json"}, body.to_json)
                 else
                   client.get(request_path)
                 end
      unless response.success?
        raise Error.new("llama.cpp request failed with HTTP status #{response.status_code}")
      end
      JSON.parse(response.body)
    rescue ex : Error
      raise ex
    rescue
      # Never bubble up client exceptions which may include the server URL.
      raise Error.new("llama.cpp backend request failed")
    ensure
      client.try(&.close) if @request_transport.nil?
    end

    private def nested(value : JSON::Any?, *keys : String) : JSON::Any?
      current = value
      keys.each do |key|
        current = current.try(&.as_h?).try(&.[key]?)
      end
      current
    end

    private def required_positive_integer(value : JSON::Any?, field : String) : Int64
      integer = value.try(&.as_i64?)
      unless integer && integer > 0
        raise Error.new("llama.cpp returned an invalid #{field}")
      end
      integer
    end
  end
end
