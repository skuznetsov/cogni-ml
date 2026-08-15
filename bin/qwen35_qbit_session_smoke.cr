# Guarded multi-turn QBit session probe. The client transcript remains the
# authority for chat messages; ClickHouse stores exact model state at each
# full-prompt boundary.

require "json"
require "option_parser"

require "../src/ml/gguf/qwen35_native_runtime"

alias QwenSessionEngine = ML::GGUF::Qwen35Engine
alias QwenSessionRuntime = ML::GGUF::Qwen35NativeRuntime

SESSION_MAX_SEQ        = 512
SESSION_MAX_ACTIONS    =   8
SESSION_MAX_SOURCE_MIB = 256
DEFAULT_MODEL_PATH     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"

struct QBitSessionMessage
  include JSON::Serializable

  getter role : String
  getter content : String

  def initialize(@role : String, @content : String)
  end
end

class QBitSessionTranscript
  include JSON::Serializable

  getter messages : Array(QBitSessionMessage)
  getter actions : Int32
  getter last_token_ids : Array(Int32)
  getter max_seq : Int32
  getter max_tokens : Int32

  def initialize(@messages : Array(QBitSessionMessage),
                 @actions : Int32,
                 @last_token_ids : Array(Int32),
                 @max_seq : Int32,
                 @max_tokens : Int32)
  end
end

record QBitSessionDelta,
  hits : Int64,
  misses : Int64,
  rejections : Int64,
  transport_failures : Int64,
  restore_failures : Int64,
  writes : Int64,
  write_failures : Int64,
  lookup_time : Time::Span,
  restore_time : Time::Span,
  write_back_time : Time::Span,
  last_failure : String?

record QBitSessionObservation,
  action : Int32,
  generate_time : Time::Span,
  prompt_tokens : Int32,
  completion_tokens : Int32,
  token_ids : Array(Int32),
  text : String,
  backend : String,
  qbit : QBitSessionDelta

def qbit_session_action(index : Int32, repetitions : Int32) : String
  segments = (1..repetitions).map do |segment|
    "segment #{index}.#{segment} preserves ordering ownership timestamps exact token history and checkpoint integrity"
  end
  "Action #{index} extends the bounded engineering session. #{segments.join("; ")}. " \
  "Reply only with checkpoint-#{index}."
end

def qbit_session_delta(after_stats : QwenSessionRuntime::QBitCacheStats,
                       before_stats : QwenSessionRuntime::QBitCacheStats) : QBitSessionDelta
  QBitSessionDelta.new(
    hits: after_stats.hits - before_stats.hits,
    misses: after_stats.misses - before_stats.misses,
    rejections: after_stats.rejections - before_stats.rejections,
    transport_failures: after_stats.transport_failures - before_stats.transport_failures,
    restore_failures: after_stats.restore_failures - before_stats.restore_failures,
    writes: after_stats.writes - before_stats.writes,
    write_failures: after_stats.write_failures - before_stats.write_failures,
    lookup_time: after_stats.lookup_time - before_stats.lookup_time,
    restore_time: after_stats.restore_time - before_stats.restore_time,
    write_back_time: after_stats.write_back_time - before_stats.write_back_time,
    last_failure: after_stats.last_failure,
  )
end

def qbit_session_run(engine : QwenSessionEngine,
                     runtime : QwenSessionRuntime,
                     messages : Array(QwenSessionEngine::Message),
                     action : Int32,
                     max_tokens : Int32,
                     max_seq : Int32) : QBitSessionObservation
  before_stats = runtime.qbit_cache_stats
  started = Time.instant
  result = engine.generate(
    QwenSessionEngine::GenerateRequest.new(
      messages: messages,
      max_tokens: max_tokens,
      max_seq: max_seq,
    )
  )
  elapsed = Time.instant - started
  delta = qbit_session_delta(runtime.qbit_cache_stats, before_stats)
  QBitSessionObservation.new(
    action: action,
    generate_time: elapsed,
    prompt_tokens: result.prompt_tokens,
    completion_tokens: result.completion_tokens,
    token_ids: result.token_ids,
    text: result.text,
    backend: result.backend.primary.to_s,
    qbit: delta,
  )
end

def qbit_session_clean?(delta : QBitSessionDelta) : Bool
  delta.rejections == 0 && delta.transport_failures == 0 &&
    delta.restore_failures == 0 && delta.write_failures == 0 &&
    delta.last_failure.nil?
end

def qbit_session_assert!(phase : String, observation : QBitSessionObservation) : Nil
  delta = observation.qbit
  expected = case phase
             when "seed", "continue"
               delta.hits == 0 && delta.misses == 1 && delta.writes == 1
             when "hit"
               delta.hits == 1 && delta.misses == 0 && delta.writes == 0
             when "baseline"
               delta.hits == 0 && delta.misses == 0 && delta.writes == 0 &&
                 delta.lookup_time == Time::Span.zero && delta.restore_time == Time::Span.zero &&
                 delta.write_back_time == Time::Span.zero
             else
               false
             end
  unless expected && qbit_session_clean?(delta)
    raise "#{phase} action #{observation.action} did not produce the expected clean QBit transition: #{delta.inspect}"
  end
end

def qbit_session_messages(messages : Array(QBitSessionMessage)) : Array(QwenSessionEngine::Message)
  messages.map { |message| QwenSessionEngine::Message.new(message.role, message.content) }
end

def qbit_session_serialized(messages : Array(QwenSessionEngine::Message)) : Array(QBitSessionMessage)
  messages.map { |message| QBitSessionMessage.new(message.role, message.content) }
end

def qbit_session_read(path : String) : QBitSessionTranscript
  raise "session transcript does not exist: #{path}" unless File.file?(path)
  QBitSessionTranscript.from_json(File.read(path))
end

def qbit_session_validate_transcript!(transcript : QBitSessionTranscript,
                                      max_seq : Int32,
                                      max_tokens : Int32) : Nil
  raise "transcript action count is outside 1..#{SESSION_MAX_ACTIONS}" unless transcript.actions.in?(1..SESSION_MAX_ACTIONS)
  raise "transcript max_seq mismatch" unless transcript.max_seq == max_seq
  raise "transcript max_tokens mismatch" unless transcript.max_tokens == max_tokens
  raise "transcript does not end with an assistant result" unless transcript.messages.last?.try(&.role) == "assistant"
end

def qbit_session_write(path : String, transcript : QBitSessionTranscript) : Nil
  parent = File.dirname(path)
  raise "session transcript directory does not exist: #{parent}" unless Dir.exists?(parent)
  temporary = "#{path}.tmp.#{Process.pid}"
  begin
    File.open(temporary, "w") { |file| transcript.to_json(file) }
    File.rename(temporary, path)
  ensure
    File.delete(temporary) if File.exists?(temporary)
  end
end

def qbit_session_emit(phase : String,
                      model_path : String,
                      max_seq : Int32,
                      load_time : Time::Span,
                      transcript_actions : Int32,
                      observations : Array(QBitSessionObservation)) : Nil
  output = JSON.build do |json|
    json.object do
      json.field "phase", phase
      json.field "model", File.basename(model_path)
      json.field "max_seq", max_seq
      json.field "load_ms", load_time.total_milliseconds.round(3)
      json.field "transcript_actions", transcript_actions
      json.field "observations" do
        json.array do
          observations.each do |observation|
            delta = observation.qbit
            json.object do
              json.field "action", observation.action
              json.field "generate_ms", observation.generate_time.total_milliseconds.round(3)
              json.field "prompt_tokens", observation.prompt_tokens
              json.field "completion_tokens", observation.completion_tokens
              json.field "token_ids" do
                json.array { observation.token_ids.each { |token_id| json.number token_id } }
              end
              json.field "text", observation.text
              json.field "backend", observation.backend
              json.field "qbit" do
                json.object do
                  json.field "hits", delta.hits
                  json.field "misses", delta.misses
                  json.field "rejections", delta.rejections
                  json.field "transport_failures", delta.transport_failures
                  json.field "restore_failures", delta.restore_failures
                  json.field "writes", delta.writes
                  json.field "write_failures", delta.write_failures
                  json.field "lookup_ms", delta.lookup_time.total_milliseconds.round(3)
                  json.field "restore_ms", delta.restore_time.total_milliseconds.round(3)
                  json.field "write_back_ms", delta.write_back_time.total_milliseconds.round(3)
                  json.field "last_failure", delta.last_failure
                end
              end
            end
          end
        end
      end
    end
  end
  puts "QBIT_SESSION_SMOKE_JSON=#{output}"
end

phase = "seed"
model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
endpoint = ENV["QWEN_QBIT_CLICKHOUSE_ENDPOINT"]? || "http://127.0.0.1:18123"
table_prefix = ENV["QWEN_QBIT_TABLE_PREFIX"]? || "qwen_qbit_session_smoke"
transcript_path = "/private/tmp/qwen_qbit_session_transcript.json"
actions = 6
payload_repetitions = 2
max_seq = SESSION_MAX_SEQ
max_tokens = 4
max_source_mib = SESSION_MAX_SOURCE_MIB

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_session_smoke [options]"
  parser.on("--phase NAME", "seed, baseline, hit, or continue") { |value| phase = value }
  parser.on("--model PATH", "Target Qwen GGUF path") { |value| model_path = value }
  parser.on("--endpoint URL", "ClickHouse HTTP endpoint") { |value| endpoint = value }
  parser.on("--table-prefix NAME", "Isolated ClickHouse table prefix") { |value| table_prefix = value }
  parser.on("--transcript PATH", "Benchmark transcript path") { |value| transcript_path = value }
  parser.on("--actions N", "Seed action count (default: 6)") { |value| actions = value.to_i }
  parser.on("--payload-repetitions N", "Payload segments per action (default: 2)") { |value| payload_repetitions = value.to_i }
  parser.on("--max-seq N", "State capacity, at most 512") { |value| max_seq = value.to_i }
  parser.on("--max-tokens N", "Generated tokens per action (default: 4)") { |value| max_tokens = value.to_i }
  parser.on("--max-source-mib N", "Write-back source admission, at most 256 MiB") { |value| max_source_mib = value.to_i }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

raise "phase must be seed, baseline, hit, or continue" unless phase.in?("seed", "baseline", "hit", "continue")
raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "actions must be within 1..#{SESSION_MAX_ACTIONS}" unless actions.in?(1..SESSION_MAX_ACTIONS)
raise "payload repetitions must be within 1..4" unless payload_repetitions.in?(1..4)
raise "max-seq must be within 64..#{SESSION_MAX_SEQ}" unless max_seq.in?(64..SESSION_MAX_SEQ)
raise "max-tokens must be within 1..8" unless max_tokens.in?(1..8)
raise "max-source-mib must be within 1..#{SESSION_MAX_SOURCE_MIB}" unless max_source_mib.in?(1..SESSION_MAX_SOURCE_MIB)
raise "seed refuses to overwrite transcript: #{transcript_path}" if phase == "seed" && File.exists?(transcript_path)

write_back = phase.in?("seed", "continue")
store = nil.as(ML::GGUF::QwenQBitClickHouseCache::Store?)
unless phase == "baseline"
  config = ML::GGUF::QwenQBitClickHouseCache::Config.new(
    endpoint: endpoint,
    table_prefix: table_prefix,
    connect_timeout: 2.seconds,
    read_timeout: 180.seconds,
    write_timeout: 180.seconds,
    max_recurrent_bytes: 96_i64 * 1024 * 1024,
    max_kv_bytes: 128_i64 * 1024 * 1024,
    max_total_artifact_bytes: 224_i64 * 1024 * 1024,
  )
  store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config)
  store.not_nil!.create_schema if phase == "seed"
end

runtime = nil.as(QwenSessionRuntime?)
engine = nil.as(QwenSessionEngine?)
begin
  load_started = Time.instant
  runtime = QwenSessionRuntime.new(
    model_path,
    max_seq: max_seq,
    qbit_clickhouse_cache: store,
    qbit_cache_ttl: 1.hour,
    qbit_cache_write_back_max_source_bytes: write_back ? max_source_mib.to_i64 * 1024 * 1024 : 0_i64,
  )
  load_time = Time.instant - load_started
  engine = QwenSessionEngine.new(runtime.not_nil!)
  observations = [] of QBitSessionObservation
  transcript_actions = actions

  case phase
  when "seed"
    messages = [
      QwenSessionEngine::Message.new(
        "system",
        "You are a deterministic session checkpoint probe. Follow every response constraint exactly.",
      ),
    ]
    (1..actions).each do |action|
      messages << QwenSessionEngine::Message.new("user", qbit_session_action(action, payload_repetitions))
      observation = qbit_session_run(engine.not_nil!, runtime.not_nil!, messages, action, max_tokens, max_seq)
      qbit_session_assert!(phase, observation)
      observations << observation
      messages << QwenSessionEngine::Message.new("assistant", observation.text)
    end
    last = observations.last
    qbit_session_write(
      transcript_path,
      QBitSessionTranscript.new(
        qbit_session_serialized(messages),
        actions,
        last.token_ids,
        max_seq,
        max_tokens,
      ),
    )
  when "baseline", "hit"
    transcript = qbit_session_read(transcript_path)
    qbit_session_validate_transcript!(transcript, max_seq, max_tokens)
    messages = qbit_session_messages(transcript.messages[0, transcript.messages.size - 1])
    observation = qbit_session_run(
      engine.not_nil!,
      runtime.not_nil!,
      messages,
      transcript.actions,
      max_tokens,
      max_seq,
    )
    qbit_session_assert!(phase, observation)
    unless observation.token_ids == transcript.last_token_ids
      raise "#{phase} token parity mismatch: #{observation.token_ids} != #{transcript.last_token_ids}"
    end
    observations << observation
    transcript_actions = transcript.actions
  when "continue"
    transcript = qbit_session_read(transcript_path)
    qbit_session_validate_transcript!(transcript, max_seq, max_tokens)
    messages = qbit_session_messages(transcript.messages)
    action = transcript.actions + 1
    raise "continued action exceeds #{SESSION_MAX_ACTIONS}" if action > SESSION_MAX_ACTIONS
    messages << QwenSessionEngine::Message.new("user", qbit_session_action(action, payload_repetitions))
    observation = qbit_session_run(engine.not_nil!, runtime.not_nil!, messages, action, max_tokens, max_seq)
    qbit_session_assert!(phase, observation)
    observations << observation
    messages << QwenSessionEngine::Message.new("assistant", observation.text)
    qbit_session_write(
      transcript_path,
      QBitSessionTranscript.new(
        qbit_session_serialized(messages),
        action,
        observation.token_ids,
        max_seq,
        max_tokens,
      ),
    )
    transcript_actions = action
  end

  qbit_session_emit(phase, model_path, max_seq, load_time, transcript_actions, observations)
ensure
  if active_engine = engine
    active_engine.close
  elsif active_runtime = runtime
    active_runtime.close
  end
end
