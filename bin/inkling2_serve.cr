require "option_parser"
require "../src/ml/serve/inkling2"

model_path = nil.as(String?)
model_id = "inkling-2-mini"
host = "127.0.0.1"
port = 8080
context = 16_384
profile = "fast"
ngram = nil.as(Bool?)
dspark_path = nil.as(String?)
use_metal = false
{% if flag?(:darwin) && !flag?(:cpu_only) %}
  use_metal = true
{% end %}

parser = OptionParser.new do |options|
  options.banner = "Usage: ./build/inkling2_serve --model PATH [options]"
  options.on("--model PATH", "Inkling 2 Mini IQ4_NL GGUF path (required)") { |value| model_path = value }
  options.on("--model-id ID", "Model ID exposed over HTTP (default: inkling-2-mini)") { |value| model_id = value }
  options.on("--host HOST", "Listen address (default: 127.0.0.1)") { |value| host = value }
  options.on("--port PORT", "Listen port (default: 8080)") { |value| port = value.to_i }
  options.on("--context N", "Allocated model context (default: 16384)") { |value| context = value.to_i }
  options.on("--profile PROFILE", "Prefill profile: fast or ordered (default: fast)") { |value| profile = value }
  options.on("--dspark PATH", "Use native DSpark greedy drafting (mtp.safetensors)") { |value| dspark_path = value }
  options.on("--no-ngram", "Disable n-gram-assisted greedy verification") { ngram = false }
  options.on("--cpu", "Run inference on the CPU") { use_metal = false }
  options.on("--help", "Show this help") do
    puts options
    exit 0
  end
end

engine = nil.as(ML::GGUF::Inkling2Engine?)
begin
  parser.parse
  path = model_path
  raise ArgumentError.new("--model is required") unless path
  raise ArgumentError.new("model file does not exist: #{path}") unless File.file?(path)
  raise ArgumentError.new("--model-id and --host must be nonempty") if model_id.empty? || host.empty?
  raise ArgumentError.new("--port must be between 1 and 65535") unless port.in?(1..65_535)
  raise ArgumentError.new("--profile must be fast or ordered") unless profile.in?({"fast", "ordered"})
  raise ArgumentError.new("--context must be positive") unless context > 0

  ENV["INKLING2_PREFILL_ATTENTION"] = "ordered" if profile == "ordered"
  engine = ML::GGUF::Inkling2Engine.new(path, context, metal: use_metal,
    fast_prefill: profile == "fast", greedy_verifier: ngram, dspark_path: dspark_path)
  backend = ML::Serve::Inkling2Backend.new(engine)
  api_key = ENV["COGNI_INKLING2_API_KEY"]?.presence
  handler = ML::Serve::OpenAI::Handler.new(backend, model_id, api_key)
  server = HTTP::Server.new(handler)
  address = server.bind_tcp(host, port)
  active_profile = engine.fast_prefill? ? "fast" : (engine.resident? ? "ordered" : "scalar")
  STDERR.puts("inkling2_serve listening on http://#{address}/v1 model=#{model_id} " +
              "context=#{engine.max_seq} passes=#{engine.passes} prefill=#{active_profile} ngram=#{engine.greedy_verifier?} dspark=#{engine.dspark?}")
  server.listen
rescue ex : Exception
  STDERR.puts("inkling2_serve: #{ex.message}")
  exit 1
ensure
  engine.try(&.close)
end
