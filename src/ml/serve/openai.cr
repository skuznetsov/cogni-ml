require "http/server"
require "json"
require "uri"
require "uuid"
require "crypto/subtle"

module ML::Serve::OpenAI
  MAX_BODY_BYTES = 8 * 1024 * 1024
  MAX_STOP_BYTES = 256

  record Message, role : String, content : String | Array(String)
  alias Input = String | Array(Int32) | Array(Message)
  record Options, max_tokens : Int32? = nil,
    temperature : Float64 = 0.0,
    top_p : Float64 = 1.0,
    seed : Int64? = nil,
    stop : Array(String) = [] of String,
    reasoning_effort : String? = nil
  record Result, prompt_tokens : Int32, completion_tokens : Int32, finish_reason : String

  class RequestError < Exception
    getter param : String?
    getter code : String?

    def initialize(message : String, *, @param : String? = nil, @code : String? = nil)
      super(message)
    end
  end

  abstract class Generation
    abstract def run(& : String -> Nil) : Result
  end

  abstract class Backend
    abstract def prepare(input : Input, options : Options) : Generation
  end

  private enum CompletionKind
    Chat
    Text
  end

  private record ParsedRequest,
    kind : CompletionKind,
    input : Input,
    options : Options,
    stream : Bool,
    include_usage : Bool

  class Handler
    include HTTP::Handler

    def initialize(@backend : Backend, @model : String, @api_key : String? = nil)
      @mutex = Mutex.new
    end

    def call(context : HTTP::Server::Context) : Nil
      return unless authorized?(context)

      request = context.request
      path = request.path
      case
      when request.method == "GET" && path == "/v1/models"
        write_model_list(context)
      when request.method == "GET" && path.starts_with?("/v1/models/")
        requested_model = begin
          URI.decode(path.byte_slice(11..-1) || "")
        rescue URI::Error
          ""
        end
        if requested_model == @model
          write_model(context, @model)
        else
          write_error(context, 404, "The requested model does not exist.", "invalid_request_error", code: "model_not_found")
        end
      when request.method == "POST" && path == "/v1/chat/completions"
        handle_completion(context, CompletionKind::Chat)
      when request.method == "POST" && path == "/v1/completions"
        handle_completion(context, CompletionKind::Text)
      else
        write_error(context, 404, "The requested endpoint does not exist.", "invalid_request_error", code: "not_found")
      end
    rescue ex : RequestError
      status = ex.code == "model_not_found" ? 404 : 400
      write_error(context, status, ex.message || "Invalid request.", "invalid_request_error", ex.param, ex.code)
    rescue ex : BodyHandled
      nil
    rescue ex : IO::Error
      # A disconnected client must unwind a generation instead of letting the
      # backend continue behind a dead stream.
      nil
    rescue Exception
      write_error(context, 500, "The request could not be completed.", "server_error", code: "internal_error")
    end

    private def authorized?(context : HTTP::Server::Context) : Bool
      return true unless api_key = @api_key

      value = context.request.headers["Authorization"]?
      parts = value.try(&.split(' ', 2))
      return true if parts && parts.size == 2 && parts[0].downcase == "bearer" && Crypto::Subtle.constant_time_compare(parts[1], api_key)

      context.response.headers["WWW-Authenticate"] = "Bearer"
      write_error(context, 401, "Invalid API key provided.", "invalid_request_error", code: "invalid_api_key")
      false
    end

    private def write_model_list(context : HTTP::Server::Context) : Nil
      created = Time.utc.to_unix
      json = JSON.build do |builder|
        builder.object do
          builder.field("object", "list")
          builder.field("data") do
            builder.array do
              write_model_object(builder, @model, created)
            end
          end
        end
      end
      write_json(context, 200, json)
    end

    private def write_model(context : HTTP::Server::Context, model : String) : Nil
      json = JSON.build do |builder|
        write_model_object(builder, model, Time.utc.to_unix)
      end
      write_json(context, 200, json)
    end

    private def write_model_object(builder : JSON::Builder, model : String, created : Int64) : Nil
      builder.object do
        builder.field("id", model)
        builder.field("object", "model")
        builder.field("created", created)
        builder.field("owned_by", "local")
      end
    end

    private def handle_completion(context : HTTP::Server::Context, kind : CompletionKind) : Nil
      root = read_json_body(context)
      parsed = parse_request(root, kind)
      request_id = "#{kind == CompletionKind::Chat ? "chatcmpl" : "cmpl"}-#{UUID.random}"
      created = Time.utc.to_unix

      @mutex.synchronize do
        generation = begin
          @backend.prepare(parsed.input, parsed.options)
        rescue ex : RequestError
          write_error(context, 400, ex.message || "Invalid request.", "invalid_request_error", ex.param, ex.code)
          return
        rescue ex : IO::Error
          return
        rescue Exception
          write_error(context, 500, "The request could not be completed.", "server_error", code: "internal_error")
          return
        end

        if parsed.stream
          begin
            stream_completion(context, generation, parsed, request_id, created)
          rescue IO::Error
            # Writing a delta failed, so the generation callback unwinds and
            # the server drops the partial stream without exposing internals.
            nil
          rescue Exception
            # Once SSE headers are sent, an OpenAI JSON error cannot replace
            # the response. Closing the partial stream is the honest outcome.
            nil
          end
        else
          result = nil.as(Result?)
          output = String.build do |text|
            result = generation.run { |fragment| text << fragment }
          end
          completion_result = result.not_nil!
          if parsed.kind == CompletionKind::Chat
            write_chat_completion(context, request_id, created, @model, output, completion_result)
          else
            write_text_completion(context, request_id, created, @model, output, completion_result)
          end
        end
      end
    end

    private def stream_completion(context : HTTP::Server::Context, generation : Generation, parsed : ParsedRequest, request_id : String, created : Int64) : Nil
      response = context.response
      response.status_code = 200
      response.content_type = "text/event-stream; charset=utf-8"
      response.headers["Cache-Control"] = "no-cache"
      response.headers["Connection"] = "keep-alive"

      if parsed.kind == CompletionKind::Chat
        write_chat_chunk(context, request_id, created, @model, {"role" => "assistant"}, nil, parsed.include_usage)
      end

      result = generation.run do |fragment|
        if parsed.kind == CompletionKind::Chat
          write_chat_chunk(context, request_id, created, @model, {"content" => fragment}, nil, parsed.include_usage)
        else
          write_text_chunk(context, request_id, created, @model, fragment, nil, parsed.include_usage)
        end
      end

      if parsed.kind == CompletionKind::Chat
        write_chat_chunk(context, request_id, created, @model, {} of String => String, result.finish_reason, parsed.include_usage)
      else
        write_text_chunk(context, request_id, created, @model, "", result.finish_reason, parsed.include_usage)
      end

      if parsed.include_usage
        write_usage_chunk(context, request_id, created, @model, parsed.kind, result)
      end
      write_sse_event(context, "[DONE]")
    end

    private def write_chat_completion(context : HTTP::Server::Context, id : String, created : Int64, model : String, content : String, result : Result) : Nil
      json = JSON.build do |builder|
        builder.object do
          builder.field("id", id)
          builder.field("object", "chat.completion")
          builder.field("created", created)
          builder.field("model", model)
          builder.field("choices") do
            builder.array do
              builder.object do
                builder.field("index", 0)
                builder.field("message") do
                  builder.object do
                    builder.field("role", "assistant")
                    builder.field("content", content)
                  end
                end
                builder.field("logprobs", nil)
                builder.field("finish_reason", result.finish_reason)
              end
            end
          end
          builder.field("usage") { write_usage(builder, result) }
        end
      end
      write_json(context, 200, json)
    end

    private def write_text_completion(context : HTTP::Server::Context, id : String, created : Int64, model : String, content : String, result : Result) : Nil
      json = JSON.build do |builder|
        builder.object do
          builder.field("id", id)
          builder.field("object", "text_completion")
          builder.field("created", created)
          builder.field("model", model)
          builder.field("choices") do
            builder.array do
              builder.object do
                builder.field("index", 0)
                builder.field("text", content)
                builder.field("logprobs", nil)
                builder.field("finish_reason", result.finish_reason)
              end
            end
          end
          builder.field("usage") { write_usage(builder, result) }
        end
      end
      write_json(context, 200, json)
    end

    private def write_usage(builder : JSON::Builder, result : Result) : Nil
      builder.object do
        builder.field("prompt_tokens", result.prompt_tokens)
        builder.field("completion_tokens", result.completion_tokens)
        builder.field("total_tokens", result.prompt_tokens.to_i64 + result.completion_tokens.to_i64)
      end
    end

    private def write_chat_chunk(context : HTTP::Server::Context, id : String, created : Int64, model : String, delta : Hash(String, String), finish_reason : String?, include_usage : Bool) : Nil
      json = JSON.build do |builder|
        builder.object do
          builder.field("id", id)
          builder.field("object", "chat.completion.chunk")
          builder.field("created", created)
          builder.field("model", model)
          builder.field("choices") do
            builder.array do
              builder.object do
                builder.field("index", 0)
                builder.field("delta") do
                  builder.object do
                    delta.each { |key, value| builder.field(key, value) }
                  end
                end
                builder.field("logprobs", nil)
                builder.field("finish_reason", finish_reason)
              end
            end
          end
          builder.field("usage", nil) if include_usage
        end
      end
      write_sse_event(context, json)
    end

    private def write_text_chunk(context : HTTP::Server::Context, id : String, created : Int64, model : String, text : String, finish_reason : String?, include_usage : Bool) : Nil
      json = JSON.build do |builder|
        builder.object do
          builder.field("id", id)
          builder.field("object", "text_completion")
          builder.field("created", created)
          builder.field("model", model)
          builder.field("choices") do
            builder.array do
              builder.object do
                builder.field("index", 0)
                builder.field("text", text)
                builder.field("logprobs", nil)
                builder.field("finish_reason", finish_reason)
              end
            end
          end
          builder.field("usage", nil) if include_usage
        end
      end
      write_sse_event(context, json)
    end

    private def write_usage_chunk(context : HTTP::Server::Context, id : String, created : Int64, model : String, kind : CompletionKind, result : Result) : Nil
      json = JSON.build do |builder|
        builder.object do
          builder.field("id", id)
          builder.field("object", kind == CompletionKind::Chat ? "chat.completion.chunk" : "text_completion")
          builder.field("created", created)
          builder.field("model", model)
          builder.field("choices") { builder.array { } }
          builder.field("usage") { write_usage(builder, result) }
        end
      end
      write_sse_event(context, json)
    end

    private def write_sse_event(context : HTTP::Server::Context, data : String) : Nil
      context.response << "data: " << data << "\n\n"
      context.response.flush
    end

    private def write_json(context : HTTP::Server::Context, status : Int32, json : String) : Nil
      context.response.status_code = status
      context.response.content_type = "application/json; charset=utf-8"
      context.response << json
    end

    private def write_error(context : HTTP::Server::Context, status : Int32, message : String, type : String, param : String? = nil, code : String? = nil) : Nil
      json = JSON.build do |builder|
        builder.object do
          builder.field("error") do
            builder.object do
              builder.field("message", message)
              builder.field("type", type)
              builder.field("param", param)
              builder.field("code", code)
            end
          end
        end
      end
      write_json(context, status, json)
    end

    private def read_json_body(context : HTTP::Server::Context) : JSON::Any
      if content_length = context.request.headers["Content-Length"]?
        parsed_length = content_length.to_i64?
        if parsed_length && parsed_length > MAX_BODY_BYTES
          context.response.headers["Connection"] = "close"
          write_error(context, 413, "Request body exceeds the size limit.", "invalid_request_error", code: "request_too_large")
          raise BodyHandled.new
        end
      end

      too_large = false
      body = String.build do |output|
        if request_body = context.request.body
          buffer = Bytes.new(16 * 1024)
          size = 0
          loop do
            read = request_body.read(buffer)
            break if read == 0
            size += read
            if size > MAX_BODY_BYTES
              too_large = true
              break
            end
            output.write(buffer[0, read])
          end
        end
      end
      if too_large
        context.response.headers["Connection"] = "close"
        write_error(context, 413, "Request body exceeds the size limit.", "invalid_request_error", code: "request_too_large")
        raise BodyHandled.new
      end

      JSON.parse(body)
    rescue ex : JSON::ParseException
      raise RequestError.new("Request body must contain valid JSON.", code: "invalid_json")
    end

    private class BodyHandled < Exception
    end

    private def parse_request(root : JSON::Any, kind : CompletionKind) : ParsedRequest
      object = root.as_h? || invalid("Request body must be a JSON object.")
      model_value = object["model"]? || invalid("A model is required.", "model", "missing_required_parameter")
      model = model_value.as_s? || invalid("The model parameter must be a string.", "model", "invalid_type")
      unless model == @model
        raise RequestError.new("The requested model does not exist.", param: "model", code: "model_not_found")
      end

      input = if kind == CompletionKind::Chat
                parse_messages(object["messages"]?)
              else
                parse_prompt(object["prompt"]?)
              end

      parse_unsupported_options(object, kind)
      stream = optional_bool(object, "stream", false)
      include_usage = parse_stream_options(object, stream)
      max_tokens = parse_max_tokens(object)
      temperature = optional_number(object, "temperature", 0.0, 0.0, 2.0)
      top_p = optional_number(object, "top_p", 1.0, 0.0, 1.0)
      seed = optional_int64(object, "seed")
      stop = parse_stop(object["stop"]?)
      reasoning_effort = optional_string(object, "reasoning_effort")

      options = Options.new(max_tokens, temperature, top_p, seed, stop, reasoning_effort)
      ParsedRequest.new(kind, input, options, stream, include_usage)
    end

    private def parse_messages(value : JSON::Any?) : Input
      messages_value = value || invalid("At least one message is required.", "messages", "missing_required_parameter")
      messages = messages_value.as_a? || invalid("Messages must be an array.", "messages", "invalid_type")
      invalid("At least one message is required.", "messages", "empty_array") if messages.empty?

      parsed = messages.map_with_index do |message, index|
        object = message.as_h? || invalid("Each message must be an object.", "messages[#{index}]", "invalid_type")
        role_value = object["role"]? || invalid("Each message must have a role.", "messages[#{index}].role", "missing_required_parameter")
        role = role_value.as_s? || invalid("Message role must be a string.", "messages[#{index}].role", "invalid_type")
        unless {"system", "developer", "user", "assistant"}.includes?(role)
          invalid("Message role is not supported.", "messages[#{index}].role", "unsupported_value")
        end
        content_value = object["content"]? || invalid("Message content is required.", "messages[#{index}].content", "missing_required_parameter")
        content = parse_message_content(content_value, index)
        Message.new(role, content)
      end
      parsed
    end

    private def parse_message_content(value : JSON::Any, index : Int32) : String | Array(String)
      if text = value.as_s?
        return text
      end

      parts = value.as_a? || invalid("Message content must be text or an array of text parts.", "messages[#{index}].content", "invalid_type")
      invalid("Message content parts cannot be empty.", "messages[#{index}].content", "empty_array") if parts.empty?
      parts.map_with_index do |part, part_index|
        object = part.as_h? || invalid("Content parts must be text objects.", "messages[#{index}].content[#{part_index}]", "invalid_type")
        type = object["type"]?.try(&.as_s?)
        unless type == "text"
          invalid("Only text content parts are supported.", "messages[#{index}].content[#{part_index}].type", "unsupported_value")
        end
        text_value = object["text"]? || invalid("Text content parts require text.", "messages[#{index}].content[#{part_index}].text", "missing_required_parameter")
        text_value.as_s? || invalid("Content part text must be a string.", "messages[#{index}].content[#{part_index}].text", "invalid_type")
      end
    end

    private def parse_prompt(value : JSON::Any?) : Input
      prompt = value || invalid("A prompt is required.", "prompt", "missing_required_parameter")
      if text = prompt.as_s?
        return text
      end

      tokens = prompt.as_a? || invalid("Prompt must be a string or an array of token IDs.", "prompt", "invalid_type")
      tokens.map_with_index do |token, index|
        integer = integer_value(token) || invalid("Prompt token IDs must be integers.", "prompt[#{index}]", "invalid_type")
        unless integer >= 0 && integer <= Int32::MAX
          invalid("Prompt token ID is out of range.", "prompt[#{index}]", "invalid_value")
        end
        integer.to_i32
      end
    end

    private def parse_max_tokens(object : Hash(String, JSON::Any)) : Int32?
      completion_value = object["max_completion_tokens"]?
      legacy_value = object["max_tokens"]?
      if completion_value && legacy_value && !completion_value.raw.nil? && !legacy_value.raw.nil?
        invalid("Use only one of max_completion_tokens or max_tokens.", "max_completion_tokens", "duplicate_parameter")
      end
      value = completion_value && !completion_value.raw.nil? ? completion_value : legacy_value
      param = completion_value && !completion_value.raw.nil? ? "max_completion_tokens" : "max_tokens"
      return nil unless value && !value.raw.nil?

      integer = integer_value(value) || invalid("Maximum tokens must be an integer.", param, "invalid_type")
      unless integer > 0 && integer <= Int32::MAX
        invalid("Maximum tokens must be a positive 32-bit integer.", param, "invalid_value")
      end
      integer.to_i32
    end

    private def optional_number(object : Hash(String, JSON::Any), name : String, default : Float64, minimum : Float64, maximum : Float64) : Float64
      value = object[name]?
      return default unless value && !value.raw.nil?

      number = case raw = value.raw
               when Int64
                 raw.to_f64
               when Float64
                 raw
               else
                 invalid("Parameter must be a number.", name, "invalid_type")
               end
      unless number.finite? && number >= minimum && number <= maximum
        invalid("Parameter is outside its supported range.", name, "invalid_value")
      end
      number
    end

    private def optional_int64(object : Hash(String, JSON::Any), name : String) : Int64?
      value = object[name]?
      return nil unless value && !value.raw.nil?
      integer_value(value) || invalid("Parameter must be an integer.", name, "invalid_type")
    end

    private def optional_bool(object : Hash(String, JSON::Any), name : String, default : Bool) : Bool
      value = object[name]?
      return default unless value && !value.raw.nil?
      bool_value(value, name)
    end

    private def bool_value(value : JSON::Any, name : String) : Bool
      case raw = value.raw
      when Bool
        raw
      else
        invalid("Parameter must be a boolean.", name, "invalid_type")
      end
    end

    private def optional_string(object : Hash(String, JSON::Any), name : String) : String?
      value = object[name]?
      return nil unless value && !value.raw.nil?
      value.as_s? || invalid("Parameter must be a string.", name, "invalid_type")
    end

    private def parse_stop(value : JSON::Any?) : Array(String)
      return [] of String unless value && !value.raw.nil?

      if text = value.as_s?
        validate_stop(text, "stop")
        [text]
      else
        entries = value.as_a? || invalid("Stop must be a string or an array of strings.", "stop", "invalid_type")
        invalid("Stop array cannot be empty.", "stop", "empty_array") if entries.empty?
        invalid("At most four stop sequences are supported.", "stop", "invalid_value") if entries.size > 4
        entries.map_with_index do |entry, index|
          text = entry.as_s? || invalid("Stop values must be strings.", "stop[#{index}]", "invalid_type")
          validate_stop(text, "stop[#{index}]")
          text
        end
      end
    end

    private def validate_stop(value : String, param : String) : Nil
      invalid("Stop sequences cannot be empty.", param, "invalid_value") if value.empty?
      invalid("Stop sequences cannot exceed #{MAX_STOP_BYTES} UTF-8 bytes.", param, "invalid_value") if value.bytesize > MAX_STOP_BYTES
    end

    private def parse_stream_options(object : Hash(String, JSON::Any), stream : Bool) : Bool
      value = object["stream_options"]?
      return false unless value && !value.raw.nil?
      invalid("stream_options requires stream=true.", "stream_options", "invalid_request_error") unless stream

      options = value.as_h? || invalid("stream_options must be an object.", "stream_options", "invalid_type")
      optional_bool(options, "include_obfuscation", false)
      optional_bool(options, "include_usage", false)
    end

    private def parse_unsupported_options(object : Hash(String, JSON::Any), kind : CompletionKind) : Nil
      if value = object["tools"]?
        unless value.raw.nil?
          tools = value.as_a? || invalid("Tools must be an array.", "tools", "invalid_type")
          invalid("Tool calls are not supported by this model endpoint.", "tools", "unsupported_value") unless tools.empty?
        end
      end

      if value = object["tool_choice"]?
        unless value.raw.nil? || value.as_s? == "none"
          invalid("Only tool_choice=none is supported.", "tool_choice", "unsupported_value")
        end
      end

      if value = object["functions"]?
        unless value.raw.nil?
          functions = value.as_a? || invalid("functions must be an array.", "functions", "invalid_type")
          invalid("Function calls are not supported by this model endpoint.", "functions", "unsupported_value") unless functions.empty?
        end
      end

      if value = object["function_call"]?
        unless value.raw.nil? || value.as_s? == "none"
          invalid("Only function_call=none is supported.", "function_call", "unsupported_value")
        end
      end

      if value = object["response_format"]?
        unless value.raw.nil?
          format = value.as_h? || invalid("Only text response_format is supported.", "response_format", "invalid_type")
          type = format["type"]?.try(&.as_s?)
          invalid("Only text response_format is supported.", "response_format", "unsupported_value") unless type == "text"
        end
      end

      {"presence_penalty", "frequency_penalty"}.each do |name|
        value = object[name]?
        if value && !value.raw.nil? && optional_number(object, name, 0.0, -2.0, 2.0) != 0.0
          invalid("Nonzero penalties are not supported.", name, "unsupported_value")
        end
      end

      if value = object["logprobs"]?
        unless value.raw.nil?
          enabled = bool_value(value, "logprobs")
          invalid("Log probabilities are not supported.", "logprobs", "unsupported_value") if enabled
        end
      end

      if value = object["top_logprobs"]?
        unless value.raw.nil?
          count = integer_value(value) || invalid("top_logprobs must be an integer.", "top_logprobs", "invalid_type")
          invalid("Log probabilities are not supported.", "top_logprobs", "unsupported_value") unless count == 0
        end
      end

      if value = object["n"]?
        unless value.raw.nil?
          count = integer_value(value) || invalid("n must be an integer.", "n", "invalid_type")
          invalid("Only n=1 is supported.", "n", "unsupported_value") unless count == 1
        end
      end

      if value = object["logit_bias"]?
        unless value.raw.nil?
          biases = value.as_h? || invalid("logit_bias must be an object.", "logit_bias", "invalid_type")
          invalid("Logit bias is not supported.", "logit_bias", "unsupported_value") unless biases.empty?
        end
      end

      if kind == CompletionKind::Text
        if value = object["echo"]?
          unless value.raw.nil?
            echo = bool_value(value, "echo")
            invalid("echo=true is not supported.", "echo", "unsupported_value") if echo
          end
        end
        if value = object["best_of"]?
          unless value.raw.nil?
            count = integer_value(value) || invalid("best_of must be an integer.", "best_of", "invalid_type")
            invalid("Only best_of=1 is supported.", "best_of", "unsupported_value") unless count == 1
          end
        end
        if value = object["suffix"]?
          invalid("suffix is not supported.", "suffix", "unsupported_value") unless value.raw.nil?
        end
      end

      if value = object["modalities"]?
        unless value.raw.nil?
          modalities = value.as_a? || invalid("modalities must be an array.", "modalities", "invalid_type")
          unless modalities.all? { |item| item.as_s? == "text" }
            invalid("Only text output modalities are supported.", "modalities", "unsupported_value")
          end
        end
      end
    end

    private def invalid(message : String, param : String? = nil, code : String? = "invalid_request_error") : NoReturn
      raise RequestError.new(message, param: param, code: code)
    end

    private def integer_value(value : JSON::Any) : Int64?
      value.raw.as?(Int64)
    end
  end
end
