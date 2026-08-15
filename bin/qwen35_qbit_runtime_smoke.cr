# Guarded end-to-end probe for the native runtime's exact full-prompt QBit
# cache. Run each phase in a separate process so a hit cannot be satisfied by
# runtime-resident state or a previous model instance.

require "json"
require "option_parser"

require "../src/ml/gguf/qwen35_native_runtime"

DEFAULT_MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
MAX_SOURCE_BYTES   = 192_i64 * 1024 * 1024

phase = "baseline"
model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
endpoint = ENV["QWEN_QBIT_CLICKHOUSE_ENDPOINT"]? || "http://127.0.0.1:18123"
table_prefix = ENV["QWEN_QBIT_TABLE_PREFIX"]? || "qwen_qbit_runtime_smoke"
prompt = "Continue the sequence for ten more items, no explanation: 1, 2, 3, 4, 5,"
max_seq = 64
max_tokens = 4

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_runtime_smoke [options]"
  parser.on("--phase NAME", "baseline, seed, or hit") { |value| phase = value }
  parser.on("--model PATH", "Target Qwen GGUF path") { |value| model_path = value }
  parser.on("--endpoint URL", "ClickHouse HTTP endpoint") { |value| endpoint = value }
  parser.on("--table-prefix NAME", "Isolated ClickHouse table prefix") { |value| table_prefix = value }
  parser.on("--prompt TEXT", "User prompt") { |value| prompt = value }
  parser.on("--max-seq N", "Decode state capacity (default: 64)") { |value| max_seq = value.to_i }
  parser.on("--max-tokens N", "Generated tokens (default: 4)") { |value| max_tokens = value.to_i }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

raise "phase must be baseline, seed, or hit" unless phase.in?("baseline", "seed", "hit")
raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "prompt must not be empty" if prompt.strip.empty?
raise "max-seq must be positive" unless max_seq > 0
raise "max-tokens must be positive" unless max_tokens > 0

store = nil.as(ML::GGUF::QwenQBitClickHouseCache::Store?)
unless phase == "baseline"
  config = ML::GGUF::QwenQBitClickHouseCache::Config.new(
    endpoint: endpoint,
    table_prefix: table_prefix,
    connect_timeout: 2.seconds,
    read_timeout: 120.seconds,
    write_timeout: 120.seconds,
    max_recurrent_bytes: 96_i64 * 1024 * 1024,
    max_kv_bytes: 32_i64 * 1024 * 1024,
    max_total_artifact_bytes: 128_i64 * 1024 * 1024,
  )
  store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config)
  store.not_nil!.create_schema if phase == "seed"
end

runtime = nil.as(ML::GGUF::Qwen35NativeRuntime?)
engine = nil.as(ML::GGUF::Qwen35Engine?)
begin
  load_started = Time.instant
  runtime = ML::GGUF::Qwen35NativeRuntime.new(
    model_path,
    max_seq: max_seq,
    qbit_clickhouse_cache: store,
    qbit_cache_ttl: 1.hour,
    qbit_cache_write_back_max_source_bytes: phase == "seed" ? MAX_SOURCE_BYTES : 0_i64,
  )
  load_elapsed = Time.instant - load_started
  engine = ML::GGUF::Qwen35Engine.new(runtime.not_nil!)
  request = ML::GGUF::Qwen35Engine::GenerateRequest.new(
    messages: [ML::GGUF::Qwen35Engine::Message.new("user", prompt)],
    max_tokens: max_tokens,
    max_seq: max_seq,
  )

  generate_started = Time.instant
  result = engine.not_nil!.generate(request)
  generate_elapsed = Time.instant - generate_started
  stats = runtime.not_nil!.qbit_cache_stats

  phase_failure = case phase
                  when "baseline"
                    clean = stats.hits == 0 && stats.misses == 0 && stats.rejections == 0 &&
                            stats.transport_failures == 0 && stats.restore_failures == 0 &&
                            stats.writes == 0 && stats.write_failures == 0 && stats.last_failure.nil?
                    "baseline unexpectedly touched QBit cache" unless clean
                  when "seed"
                    clean = stats.hits == 0 && stats.misses == 1 && stats.rejections == 0 &&
                            stats.transport_failures == 0 && stats.restore_failures == 0 &&
                            stats.writes == 1 && stats.write_failures == 0 && stats.last_failure.nil?
                    "seed did not produce one clean miss and write" unless clean
                  when "hit"
                    clean = stats.hits == 1 && stats.misses == 0 && stats.rejections == 0 &&
                            stats.transport_failures == 0 && stats.restore_failures == 0 &&
                            stats.writes == 0 && stats.write_failures == 0 && stats.last_failure.nil?
                    "hit did not restore exactly one cache entry" unless clean
                  end

  output = JSON.build do |json|
    json.object do
      json.field "phase", phase
      json.field "model", File.basename(model_path)
      json.field "max_seq", max_seq
      json.field "max_tokens", max_tokens
      json.field "load_ms", load_elapsed.total_milliseconds.round(3)
      json.field "generate_ms", generate_elapsed.total_milliseconds.round(3)
      json.field "prompt_tokens", result.prompt_tokens
      json.field "completion_tokens", result.completion_tokens
      json.field "token_ids" do
        json.array { result.token_ids.each { |token_id| json.number token_id } }
      end
      json.field "text", result.text
      json.field "backend", result.backend.primary.to_s
      json.field "qbit" do
        json.object do
          json.field "hits", stats.hits
          json.field "misses", stats.misses
          json.field "rejections", stats.rejections
          json.field "transport_failures", stats.transport_failures
          json.field "restore_failures", stats.restore_failures
          json.field "writes", stats.writes
          json.field "write_failures", stats.write_failures
          json.field "last_failure", stats.last_failure
        end
      end
    end
  end
  puts "QBIT_RUNTIME_SMOKE_JSON=#{output}"
  raise phase_failure if phase_failure
ensure
  if active_engine = engine
    active_engine.close
  elsif active_runtime = runtime
    active_runtime.close
  end
end
