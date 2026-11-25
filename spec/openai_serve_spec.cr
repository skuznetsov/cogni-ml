require "spec"
require "http/client"
require "uri"
require "../src/ml/serve/openai"

private class OpenAISpecGeneration < ML::Serve::OpenAI::Generation
  def initialize(@fragments : Array(String), @result : ML::Serve::OpenAI::Result = ML::Serve::OpenAI::Result.new(3, 2, "stop"))
  end

  def run(&block : String -> Nil) : ML::Serve::OpenAI::Result
    @fragments.each { |fragment| yield fragment }
    @result
  end
end

private class OpenAISpecBackend < ML::Serve::OpenAI::Backend
  getter last_input : ML::Serve::OpenAI::Input?
  getter last_options : ML::Serve::OpenAI::Options?
  getter prepare_count : Int32

  def initialize(@fragments : Array(String) = ["Hel", "lo"], @result : ML::Serve::OpenAI::Result = ML::Serve::OpenAI::Result.new(3, 2, "stop"), @fail_preflight : Bool = false)
    @last_input = nil
    @last_options = nil
    @prepare_count = 0
  end

  def prepare(input : ML::Serve::OpenAI::Input, options : ML::Serve::OpenAI::Options) : ML::Serve::OpenAI::Generation
    @prepare_count += 1
    @last_input = input
    @last_options = options
    if @fail_preflight
      raise ML::Serve::OpenAI::RequestError.new("Prompt exceeds model context.", param: "messages", code: "context_length_exceeded")
    end
    OpenAISpecGeneration.new(@fragments, @result)
  end
end

private def with_openai_server(backend : ML::Serve::OpenAI::Backend, api_key : String? = nil, model : String = "inkling-local", &block : URI ->)
  server = HTTP::Server.new([ML::Serve::OpenAI::Handler.new(backend, model, api_key)])
  address = server.bind_unused_port
  spawn { server.listen }
  uri = URI.parse("http://127.0.0.1:#{address.port}")
  begin
    yield uri
  ensure
    server.close
  end
end

private def openai_request(uri : URI, method : String, path : String, body : String? = nil, headers : HTTP::Headers = HTTP::Headers.new) : HTTP::Client::Response
  client = HTTP::Client.new(uri)
  begin
    client.exec(method, path, headers, body)
  ensure
    client.close
  end
end

private def json_headers : HTTP::Headers
  HTTP::Headers{"Content-Type" => "application/json"}
end

describe ML::Serve::OpenAI::Handler do
  it "discovers a model and passes text parts and duplicate messages without rewriting the input" do
    backend = OpenAISpecBackend.new
    with_openai_server(backend) do |uri|
      models = openai_request(uri, "GET", "/v1/models")
      models.status_code.should eq(200)
      parsed_models = JSON.parse(models.body)
      parsed_models["object"].as_s.should eq("list")
      parsed_models["data"][0]["id"].as_s.should eq("inkling-local")

      request = {
        "model"    => "inkling-local",
        "messages" => [
          {"role" => "developer", "content" => "Be concise."},
          {"role" => "user", "content" => [{"type" => "text", "text" => "First."}, {"type" => "text", "text" => "Second."}]},
          {"role" => "user", "content" => [{"type" => "text", "text" => "First."}, {"type" => "text", "text" => "Second."}]},
        ],
        "temperature"           => 0.4,
        "top_p"                 => 0.8,
        "seed"                  => 42,
        "max_completion_tokens" => 17,
        "reasoning_effort"      => "low",
        "stream"                => false,
        "logprobs"              => false,
        "top_logprobs"          => 0,
        "presence_penalty"      => 0,
        "frequency_penalty"     => 0,
      }.to_json
      response = openai_request(uri, "POST", "/v1/chat/completions", request, json_headers)
      response.status_code.should eq(200)
      body = JSON.parse(response.body)
      body["object"].as_s.should eq("chat.completion")
      body["choices"][0]["message"]["content"].as_s.should eq("Hello")
      body["choices"][0]["finish_reason"].as_s.should eq("stop")
      body["usage"]["prompt_tokens"].as_i.should eq(3)
      body["usage"]["completion_tokens"].as_i.should eq(2)

      captured = backend.last_input.as(Array(ML::Serve::OpenAI::Message))
      captured.size.should eq(3)
      captured[0].role.should eq("developer")
      captured[1].content.should eq(["First.", "Second."])
      captured[2].should eq(captured[1])
      options = backend.last_options.not_nil!
      options.max_tokens.should eq(17)
      options.temperature.should eq(0.4)
      options.top_p.should eq(0.8)
      options.seed.should eq(42_i64)
      options.reasoning_effort.should eq("low")
      options.stop.should be_empty
    end
  end

  it "accepts token-id prompts on the text completion endpoint" do
    backend = OpenAISpecBackend.new(["ok"], ML::Serve::OpenAI::Result.new(3, 1, "length"))
    with_openai_server(backend) do |uri|
      response = openai_request(uri, "POST", "/v1/completions", {"model" => "inkling-local", "prompt" => [7, 11, 7], "max_tokens" => 1, "echo" => false, "best_of" => 1}.to_json, json_headers)
      response.status_code.should eq(200)
      body = JSON.parse(response.body)
      body["object"].as_s.should eq("text_completion")
      body["choices"][0]["text"].as_s.should eq("ok")
      body["choices"][0]["finish_reason"].as_s.should eq("length")
      body["usage"]["total_tokens"].as_i.should eq(4)
      backend.last_input.should eq([7, 11, 7])
      backend.last_options.not_nil!.temperature.should eq(0.0)
      backend.last_options.not_nil!.top_p.should eq(1.0)
      backend.last_options.not_nil!.reasoning_effort.should be_nil

      streamed = openai_request(uri, "POST", "/v1/completions", {"model" => "inkling-local", "prompt" => "Hi", "stream" => true, "stream_options" => {"include_usage" => true}}.to_json, json_headers)
      streamed.status_code.should eq(200)
      stream_events = streamed.body.lines.select(&.starts_with?("data: ")).map { |line| line[6..] }
      JSON.parse(stream_events[0])["object"].as_s.should eq("text_completion")
      JSON.parse(stream_events[2])["object"].as_s.should eq("text_completion")
      JSON.parse(stream_events[2])["choices"].as_a.should be_empty
      stream_events.last.should eq("[DONE]")
    end
  end

  it "retrieves model IDs that contain URL-encoded path characters" do
    backend = OpenAISpecBackend.new
    with_openai_server(backend, model: "org/inkling") do |uri|
      response = openai_request(uri, "GET", "/v1/models/org%2Finkling")
      response.status_code.should eq(200)
      JSON.parse(response.body)["id"].as_s.should eq("org/inkling")
    end
  end

  it "streams OpenAI chat chunk shapes, one request id, optional usage, and DONE" do
    backend = OpenAISpecBackend.new(["Hel", "lo"], ML::Serve::OpenAI::Result.new(3, 2, "stop"))
    with_openai_server(backend) do |uri|
      body = {"model" => "inkling-local", "messages" => [{"role" => "user", "content" => "Hi"}], "stream" => true, "stream_options" => {"include_usage" => true}}.to_json
      response = openai_request(uri, "POST", "/v1/chat/completions", body, json_headers)
      response.status_code.should eq(200)
      response.headers["Content-Type"].should start_with("text/event-stream")

      events = response.body.lines.select(&.starts_with?("data: ")).map do |line|
        line[6..]
      end
      events.size.should eq(6)
      events[0].should_not eq("[DONE]")
      role_chunk = JSON.parse(events[0])
      role_chunk["object"].as_s.should eq("chat.completion.chunk")
      role_chunk["choices"][0]["delta"]["role"].as_s.should eq("assistant")
      role_chunk["choices"][0]["finish_reason"].raw.should be_nil
      role_chunk["usage"].raw.should be_nil

      first_content = JSON.parse(events[1])
      second_content = JSON.parse(events[2])
      first_content["choices"][0]["delta"]["content"].as_s.should eq("Hel")
      second_content["choices"][0]["delta"]["content"].as_s.should eq("lo")
      [role_chunk, first_content, second_content].each do |chunk|
        chunk["id"].as_s.should eq(role_chunk["id"].as_s)
        chunk["model"].as_s.should eq("inkling-local")
        chunk["usage"].raw.should be_nil
      end

      final_choice = JSON.parse(events[3])
      final_choice["choices"][0]["delta"].as_h.should be_empty
      final_choice["choices"][0]["finish_reason"].as_s.should eq("stop")
      usage = JSON.parse(events[4])
      usage["choices"].as_a.should be_empty
      usage["usage"]["total_tokens"].as_i.should eq(5)
      usage["id"].as_s.should eq(role_chunk["id"].as_s)
      events[5].should eq("[DONE]")
    end
  end

  it "returns structured preflight and request errors before starting a stream" do
    backend = OpenAISpecBackend.new(fail_preflight: true)
    with_openai_server(backend) do |uri|
      body = {"model" => "inkling-local", "messages" => [{"role" => "user", "content" => "large prompt"}], "stream" => true}.to_json
      response = openai_request(uri, "POST", "/v1/chat/completions", body, json_headers)
      response.status_code.should eq(400)
      response.headers["Content-Type"].should start_with("application/json")
      error = JSON.parse(response.body)["error"]
      error["param"].as_s.should eq("messages")
      error["code"].as_s.should eq("context_length_exceeded")
      response.body.should_not contain("large prompt")
      backend.prepare_count.should eq(1)
    end

    backend = OpenAISpecBackend.new
    with_openai_server(backend) do |uri|
      malformed = openai_request(uri, "POST", "/v1/chat/completions", "{", json_headers)
      malformed.status_code.should eq(400)
      JSON.parse(malformed.body)["error"]["type"].as_s.should eq("invalid_request_error")

      unsupported = openai_request(uri, "POST", "/v1/chat/completions", {"model" => "inkling-local", "messages" => [{"role" => "user", "content" => "Hi"}], "tools" => [{"type" => "function"}]}.to_json, json_headers)
      unsupported.status_code.should eq(400)
      JSON.parse(unsupported.body)["error"]["param"].as_s.should eq("tools")
      backend.prepare_count.should eq(0)

      empty_stop = openai_request(uri, "POST", "/v1/completions", {"model" => "inkling-local", "prompt" => "Hi", "stop" => ""}.to_json, json_headers)
      empty_stop.status_code.should eq(400)
      JSON.parse(empty_stop.body)["error"]["param"].as_s.should eq("stop")

      long_stop = openai_request(uri, "POST", "/v1/completions", {"model" => "inkling-local", "prompt" => "Hi", "stop" => "x" * (ML::Serve::OpenAI::MAX_STOP_BYTES + 1)}.to_json, json_headers)
      long_stop.status_code.should eq(400)
      JSON.parse(long_stop.body)["error"]["param"].as_s.should eq("stop")
      backend.prepare_count.should eq(0)
    end
  end

  it "checks bearer auth, model identity, and the bounded request body" do
    backend = OpenAISpecBackend.new
    with_openai_server(backend, "secret-test-key") do |uri|
      auth = openai_request(uri, "GET", "/v1/models")
      auth.status_code.should eq(401)
      JSON.parse(auth.body)["error"]["code"].as_s.should eq("invalid_api_key")

      headers = HTTP::Headers{"Authorization" => "Bearer secret-test-key"}
      unknown = openai_request(uri, "GET", "/v1/models/other", nil, headers)
      unknown.status_code.should eq(404)
      JSON.parse(unknown.body)["error"]["code"].as_s.should eq("model_not_found")

      auth_json_headers = HTTP::Headers{"Authorization" => "Bearer secret-test-key", "Content-Type" => "application/json"}
      unknown_completion = openai_request(uri, "POST", "/v1/completions", {"model" => "other", "prompt" => "Hi"}.to_json, auth_json_headers)
      unknown_completion.status_code.should eq(404)
      JSON.parse(unknown_completion.body)["error"]["code"].as_s.should eq("model_not_found")

      backend.prepare_count.should eq(0)
    end

    headers = json_headers
    request = HTTP::Request.new("POST", "/v1/chat/completions", headers, " " * (ML::Serve::OpenAI::MAX_BODY_BYTES + 1))
    response_io = IO::Memory.new
    response = HTTP::Server::Response.new(response_io)
    context = HTTP::Server::Context.new(request, response)
    ML::Serve::OpenAI::Handler.new(backend, "inkling-local").call(context)
    response.close
    response.status_code.should eq(413)
    JSON.parse(response_io.to_s.split("\r\n\r\n", 2)[1].split("\r\n0\r\n\r\n", 2)[0])["error"]["code"].as_s.should eq("request_too_large")
    backend.prepare_count.should eq(0)
  end
end
