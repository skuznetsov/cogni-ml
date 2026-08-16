# Cross-session QBit checkpoint barrier probe.
#
# The session smoke drains every publication before the next action, so it
# never observes what a deferred publication does to an unrelated request.
# This probe interleaves two sessions and one sessionless request through a
# single runtime with deferred publication and asserts the barrier policy:
# a request waits for, and inherits the failure of, only a publication it
# actually depends on.
#
# The wait accounting is the discriminator. `async_checkpoint_wait_time` grows
# only inside a drain, so a request that must not be serialized has to show an
# exactly zero wait delta, while a request that depends on the in-flight row
# has to show a non-zero one.

require "http/client"
require "json"
require "option_parser"
require "uri"

require "../src/ml/gguf/qwen35_native_runtime"

alias BarrierEngine = ML::GGUF::Qwen35Engine
alias BarrierRuntime = ML::GGUF::Qwen35NativeRuntime

BARRIER_MAX_SEQ        = 512
BARRIER_MAX_SOURCE_MIB = 256
DEFAULT_MODEL_PATH     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"

record BarrierDelta,
  hits : Int64,
  misses : Int64,
  rejections : Int64,
  transport_failures : Int64,
  restore_failures : Int64,
  writes : Int64,
  write_failures : Int64,
  reused_prefix_tokens : Int64,
  lookup_time : Time::Span,
  async_enqueued : Int64,
  async_completed : Int64,
  async_pending : Int32,
  async_wait_time : Time::Span,
  last_failure : String?

record BarrierObservation,
  step : String,
  session : String?,
  requested_checkpoint_id : String?,
  checkpoint_id : String?,
  checkpoint_pending : Bool,
  generate_time : Time::Span,
  prompt_tokens : Int32,
  completion_tokens : Int32,
  text : String,
  delta : BarrierDelta

def barrier_delta(after : BarrierRuntime::QBitCacheStats,
                  before : BarrierRuntime::QBitCacheStats) : BarrierDelta
  BarrierDelta.new(
    hits: after.hits - before.hits,
    misses: after.misses - before.misses,
    rejections: after.rejections - before.rejections,
    transport_failures: after.transport_failures - before.transport_failures,
    restore_failures: after.restore_failures - before.restore_failures,
    writes: after.writes - before.writes,
    write_failures: after.write_failures - before.write_failures,
    reused_prefix_tokens: after.reused_prefix_tokens - before.reused_prefix_tokens,
    lookup_time: after.lookup_time - before.lookup_time,
    async_enqueued: after.async_checkpoint_enqueued - before.async_checkpoint_enqueued,
    async_completed: after.async_checkpoint_completed - before.async_checkpoint_completed,
    async_pending: after.async_checkpoint_pending,
    async_wait_time: after.async_checkpoint_wait_time - before.async_checkpoint_wait_time,
    last_failure: after.last_failure,
  )
end

def barrier_run(engine : BarrierEngine,
                runtime : BarrierRuntime,
                step : String,
                messages : Array(BarrierEngine::Message),
                max_tokens : Int32,
                max_seq : Int32,
                session_id : String? = nil,
                checkpoint_id : String? = nil) : BarrierObservation
  before = runtime.qbit_cache_stats
  started = Time.instant
  result = engine.generate(
    BarrierEngine::GenerateRequest.new(
      messages: messages,
      max_tokens: max_tokens,
      max_seq: max_seq,
      session_id: session_id,
      checkpoint_id: checkpoint_id,
    )
  )
  elapsed = Time.instant - started
  BarrierObservation.new(
    step: step,
    session: session_id,
    requested_checkpoint_id: checkpoint_id,
    checkpoint_id: result.checkpoint_id,
    checkpoint_pending: result.checkpoint_pending?,
    generate_time: elapsed,
    prompt_tokens: result.prompt_tokens,
    completion_tokens: result.completion_tokens,
    text: result.text,
    delta: barrier_delta(runtime.qbit_cache_stats, before),
  )
end

def barrier_assert!(condition : Bool, message : String) : Nil
  raise "barrier probe assertion failed: #{message}" unless condition
end

def barrier_assert_clean!(observation : BarrierObservation) : Nil
  delta = observation.delta
  barrier_assert!(delta.rejections == 0, "#{observation.step} recorded a checkpoint rejection")
  barrier_assert!(delta.transport_failures == 0, "#{observation.step} recorded a transport failure")
  barrier_assert!(delta.restore_failures == 0, "#{observation.step} recorded a restore failure")
  barrier_assert!(delta.write_failures == 0, "#{observation.step} recorded a write failure")
  barrier_assert!(delta.last_failure.nil?, "#{observation.step} left a failure message: #{delta.last_failure}")
end

def barrier_action(index : Int32, repetitions : Int32) : String
  segments = (1..repetitions).map do |segment|
    "segment #{index}.#{segment} keeps ordering ownership and exact token history stable"
  end
  "Action #{index} extends the bounded barrier session. #{segments.join("; ")}. " \
  "Reply only with barrier-#{index}."
end

def barrier_system_message : BarrierEngine::Message
  BarrierEngine::Message.new(
    "system",
    "You are a deterministic session checkpoint probe. Follow every response constraint exactly.",
  )
end

# Reads the published parent chain straight from ClickHouse: the probe asserts
# the durable row, not the runtime's own memory of what it wrote.
def barrier_published_parents(endpoint : String, table_prefix : String) : Hash(String, String)
  uri = URI.parse(endpoint)
  query = "SELECT checkpoint_id, parent_checkpoint_id FROM #{table_prefix}_checkpoints FORMAT TabSeparated"
  response = HTTP::Client.post(uri, body: query)
  raise "ClickHouse chain query failed: #{response.status_code} #{response.body}" unless response.success?
  parents = {} of String => String
  response.body.each_line do |line|
    next if line.empty?
    columns = line.split('\t')
    next unless columns.size == 2
    parents[columns[0]] = columns[1]
  end
  parents
end

def barrier_emit(phase : String,
                 model_path : String,
                 max_seq : Int32,
                 load_time : Time::Span,
                 observations : Array(BarrierObservation),
                 final_stats : BarrierRuntime::QBitCacheStats,
                 notes : Array(String)) : Nil
  output = JSON.build do |json|
    json.object do
      json.field "phase", phase
      json.field "model", File.basename(model_path)
      json.field "max_seq", max_seq
      json.field "load_ms", load_time.total_milliseconds.round(3)
      json.field "observations" do
        json.array do
          observations.each do |observation|
            delta = observation.delta
            json.object do
              json.field "step", observation.step
              json.field "session", observation.session
              json.field "requested_checkpoint_id", observation.requested_checkpoint_id
              json.field "checkpoint_id", observation.checkpoint_id
              json.field "checkpoint_pending", observation.checkpoint_pending
              json.field "generate_ms", observation.generate_time.total_milliseconds.round(3)
              json.field "prompt_tokens", observation.prompt_tokens
              json.field "completion_tokens", observation.completion_tokens
              json.field "text", observation.text
              json.field "hits", delta.hits
              json.field "misses", delta.misses
              json.field "rejections", delta.rejections
              json.field "transport_failures", delta.transport_failures
              json.field "restore_failures", delta.restore_failures
              json.field "writes", delta.writes
              json.field "write_failures", delta.write_failures
              json.field "reused_prefix_tokens", delta.reused_prefix_tokens
              json.field "lookup_ms", delta.lookup_time.total_milliseconds.round(3)
              json.field "async_enqueued", delta.async_enqueued
              json.field "async_completed", delta.async_completed
              json.field "async_pending", delta.async_pending
              json.field "async_wait_ms", delta.async_wait_time.total_milliseconds.round(6)
              json.field "last_failure", delta.last_failure
            end
          end
        end
      end
      json.field "final" do
        json.object do
          json.field "async_enqueued", final_stats.async_checkpoint_enqueued
          json.field "async_completed", final_stats.async_checkpoint_completed
          json.field "async_pending", final_stats.async_checkpoint_pending
          json.field "write_failures", final_stats.write_failures
          json.field "last_failure", final_stats.last_failure
        end
      end
      json.field "notes" do
        json.array { notes.each { |note| json.string note } }
      end
    end
  end
  puts "QBIT_BARRIER_PROBE_JSON=#{output}"
end

phase = "interleave"
model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
endpoint = ENV["QWEN_QBIT_CLICKHOUSE_ENDPOINT"]? || "http://127.0.0.1:18123"
table_prefix = ENV["QWEN_QBIT_TABLE_PREFIX"]? || "qwen_qbit_barrier_probe"
session_prefix = "qwen-qbit-barrier-probe"
payload_repetitions = 2
max_seq = BARRIER_MAX_SEQ
max_tokens = 4
max_source_mib = BARRIER_MAX_SOURCE_MIB

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_session_barrier_probe [options]"
  parser.on("--phase NAME", "interleave or failure") { |value| phase = value }
  parser.on("--model PATH", "Target Qwen GGUF path") { |value| model_path = value }
  parser.on("--endpoint URL", "ClickHouse HTTP endpoint") { |value| endpoint = value }
  parser.on("--table-prefix NAME", "Isolated ClickHouse table prefix") { |value| table_prefix = value }
  parser.on("--session-prefix NAME", "Session identity prefix") { |value| session_prefix = value }
  parser.on("--payload-repetitions N", "Payload segments per action (default: 2)") { |value| payload_repetitions = value.to_i }
  parser.on("--max-seq N", "State capacity, at most 512") { |value| max_seq = value.to_i }
  parser.on("--max-tokens N", "Generated tokens per action (default: 4)") { |value| max_tokens = value.to_i }
  parser.on("--max-source-mib N", "Write-back source admission, at most 256 MiB") { |value| max_source_mib = value.to_i }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

raise "phase must be interleave or failure" unless phase.in?("interleave", "failure")
raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "payload repetitions must be within 1..4" unless payload_repetitions.in?(1..4)
raise "max-seq must be within 64..#{BARRIER_MAX_SEQ}" unless max_seq.in?(64..BARRIER_MAX_SEQ)
raise "max-tokens must be within 1..8" unless max_tokens.in?(1..8)
raise "max-source-mib must be within 1..#{BARRIER_MAX_SOURCE_MIB}" unless max_source_mib.in?(1..BARRIER_MAX_SOURCE_MIB)

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
store.create_schema

if phase == "failure"
  # Reads still resolve, but the terminal checkpoint insert has nowhere to
  # land, so the failure is raised by the writer thread during publication
  # rather than by the request that enqueued it.
  drop = HTTP::Client.post(URI.parse(endpoint), body: "DROP TABLE IF EXISTS #{table_prefix}_checkpoints")
  raise "failed to drop the checkpoint table: #{drop.status_code} #{drop.body}" unless drop.success?
end

session_a = "#{session_prefix}-a"
session_b = "#{session_prefix}-b"
notes = [] of String

runtime = nil.as(BarrierRuntime?)
engine = nil.as(BarrierEngine?)
begin
  load_started = Time.instant
  runtime = BarrierRuntime.new(
    model_path,
    max_seq: max_seq,
    qbit_clickhouse_cache: store,
    qbit_cache_ttl: 1.hour,
    qbit_cache_write_back_max_source_bytes: max_source_mib.to_i64 * 1024 * 1024,
    qbit_async_checkpoint_writes: true,
  )
  load_time = Time.instant - load_started
  engine = BarrierEngine.new(runtime.not_nil!)
  active_engine = engine.not_nil!
  active_runtime = runtime.not_nil!
  observations = [] of BarrierObservation

  messages_a = [barrier_system_message]
  messages_b = [barrier_system_message]

  case phase
  when "interleave"
    # 1. First checkpoint of session A: a full anchor, published in the
    #    background, so the runtime now carries a pending row owned by A.
    messages_a << BarrierEngine::Message.new("user", barrier_action(1, payload_repetitions))
    a1 = barrier_run(active_engine, active_runtime, "a1", messages_a, max_tokens, max_seq, session_id: session_a)
    barrier_assert_clean!(a1)
    barrier_assert!(a1.delta.misses == 1 && a1.delta.hits == 0, "a1 was expected to miss the empty session cache")
    barrier_assert!(a1.checkpoint_pending, "a1 was expected to defer its anchor publication")
    barrier_assert!(a1.delta.async_enqueued == 1, "a1 was expected to enqueue exactly one publication")
    barrier_assert!(a1.checkpoint_id.try(&.matches?(/\A[0-9a-f]{64}\z/)) == true, "a1 checkpoint identity is malformed")
    observations << a1
    messages_a << BarrierEngine::Message.new("assistant", a1.text)

    # 2. A sessionless request must not be serialized behind A's publication.
    sessionless = barrier_run(
      active_engine, active_runtime, "sessionless",
      [barrier_system_message, BarrierEngine::Message.new("user", "Reply only with sessionless-ok.")],
      max_tokens, max_seq,
    )
    barrier_assert_clean!(sessionless)
    barrier_assert!(sessionless.checkpoint_id.nil?, "a sessionless request must not produce a checkpoint")
    barrier_assert!(sessionless.delta.async_enqueued == 0, "a sessionless request must not enqueue a publication")
    barrier_assert!(
      sessionless.delta.async_wait_time == Time::Span.zero,
      "a sessionless request waited #{sessionless.delta.async_wait_time.total_milliseconds}ms for another session's publication",
    )
    observations << sessionless

    # 3. First checkpoint of session B: another anchor, so its enqueue has to
    #    drain A's publication to free the single-flight slot.
    messages_b << BarrierEngine::Message.new("user", barrier_action(1, payload_repetitions))
    b1 = barrier_run(active_engine, active_runtime, "b1", messages_b, max_tokens, max_seq, session_id: session_b)
    barrier_assert_clean!(b1)
    barrier_assert!(b1.delta.misses == 1 && b1.delta.hits == 0, "b1 was expected to miss the empty session cache")
    barrier_assert!(b1.checkpoint_pending, "b1 was expected to defer its anchor publication")
    barrier_assert!(b1.delta.async_enqueued == 1, "b1 was expected to enqueue exactly one publication")
    barrier_assert!(b1.checkpoint_id != a1.checkpoint_id, "b1 reused session A's checkpoint identity")
    observations << b1
    messages_b << BarrierEngine::Message.new("assistant", b1.text)

    # 4. Session A continues without naming a parent. The barrier must not fire
    #    for B's pending row, and the latest-checkpoint lookup must still land
    #    on A1 rather than forking the chain from an older boundary.
    messages_a << BarrierEngine::Message.new("user", barrier_action(2, payload_repetitions))
    a2 = barrier_run(active_engine, active_runtime, "a2", messages_a, max_tokens, max_seq, session_id: session_a)
    barrier_assert_clean!(a2)
    barrier_assert!(a2.delta.hits == 1 && a2.delta.misses == 0, "a2 did not resolve session A's own checkpoint")
    barrier_assert!(a2.delta.reused_prefix_tokens > 0, "a2 reused no prefix tokens")
    barrier_assert!(
      a2.delta.async_wait_time == Time::Span.zero,
      "a2 waited #{a2.delta.async_wait_time.total_milliseconds}ms for session B's publication",
    )
    barrier_assert!(a2.checkpoint_id != a1.checkpoint_id, "a2 did not advance session A's checkpoint")
    observations << a2
    messages_a << BarrierEngine::Message.new("assistant", a2.text)

    # 5. Session B continues from its own pending row by identity. The lookup is
    #    fail-closed, so the barrier has to publish B1 before it resolves.
    messages_b << BarrierEngine::Message.new("user", barrier_action(2, payload_repetitions))
    b2 = barrier_run(
      active_engine, active_runtime, "b2", messages_b, max_tokens, max_seq,
      session_id: session_b, checkpoint_id: b1.checkpoint_id,
    )
    barrier_assert_clean!(b2)
    barrier_assert!(b2.delta.hits == 1 && b2.delta.misses == 0, "b2 did not resolve its named parent checkpoint")
    barrier_assert!(
      b2.delta.async_wait_time > Time::Span.zero,
      "b2 did not wait for the publication it depends on",
    )
    observations << b2
    messages_b << BarrierEngine::Message.new("assistant", b2.text)

    # 6. Session A continues by identity from a synchronously published parent.
    messages_a << BarrierEngine::Message.new("user", barrier_action(3, payload_repetitions))
    a3 = barrier_run(
      active_engine, active_runtime, "a3", messages_a, max_tokens, max_seq,
      session_id: session_a, checkpoint_id: a2.checkpoint_id,
    )
    barrier_assert_clean!(a3)
    barrier_assert!(a3.delta.hits == 1 && a3.delta.misses == 0, "a3 did not resolve its named parent checkpoint")
    observations << a3

    active_runtime.flush_qbit_checkpoint_writes
    final_stats = active_runtime.qbit_cache_stats
    barrier_assert!(final_stats.async_checkpoint_pending == 0, "a publication was still pending after the barrier")
    barrier_assert!(
      final_stats.async_checkpoint_completed == final_stats.async_checkpoint_enqueued,
      "publication accounting does not balance: " \
      "#{final_stats.async_checkpoint_completed} of #{final_stats.async_checkpoint_enqueued} completed",
    )
    barrier_assert!(final_stats.write_failures == 0, "the run recorded #{final_stats.write_failures} write failures")
    barrier_assert!(final_stats.last_failure.nil?, "the run left a failure message: #{final_stats.last_failure}")

    parents = barrier_published_parents(endpoint, table_prefix)
    expected_parents = {
      "a1" => {a1.checkpoint_id.not_nil!, ""},
      "b1" => {b1.checkpoint_id.not_nil!, ""},
      "a2" => {a2.checkpoint_id.not_nil!, a1.checkpoint_id.not_nil!},
      "b2" => {b2.checkpoint_id.not_nil!, b1.checkpoint_id.not_nil!},
      "a3" => {a3.checkpoint_id.not_nil!, a2.checkpoint_id.not_nil!},
    }
    expected_parents.each do |step, (checkpoint_id, expected_parent)|
      published = parents[checkpoint_id]?
      barrier_assert!(!published.nil?, "#{step} checkpoint #{checkpoint_id} was never published")
      barrier_assert!(
        published == expected_parent,
        "#{step} published parent #{published.inspect} instead of #{expected_parent.inspect}",
      )
    end
    notes << "verified #{expected_parents.size} published parent links against #{table_prefix}_checkpoints"

    barrier_emit(phase, model_path, max_seq, load_time, observations, final_stats, notes)
  when "failure"
    # 1. Session A enqueues an anchor whose publication cannot land. The
    #    request itself must succeed: durability is not its concern yet.
    messages_a << BarrierEngine::Message.new("user", barrier_action(1, payload_repetitions))
    a1 = barrier_run(active_engine, active_runtime, "a1", messages_a, max_tokens, max_seq, session_id: session_a)
    barrier_assert!(a1.checkpoint_pending, "a1 was expected to defer its anchor publication")
    barrier_assert!(a1.delta.async_enqueued == 1, "a1 was expected to enqueue exactly one publication")
    observations << a1

    # 2. Session B needs the single-flight slot, so its enqueue drains session
    #    A's failed publication. The failure belongs to A's durability, not to
    #    B's generation, so B has to record it and still return normally.
    messages_b << BarrierEngine::Message.new("user", barrier_action(1, payload_repetitions))
    b1 = barrier_run(active_engine, active_runtime, "b1", messages_b, max_tokens, max_seq, session_id: session_b)
    barrier_assert!(b1.checkpoint_pending, "b1 was expected to defer its anchor publication")
    barrier_assert!(
      b1.delta.write_failures > 0,
      "b1 drained session A's failed publication without recording it",
    )
    observations << b1
    notes << "b1 returned normally after draining session A's failure: #{b1.delta.last_failure}"

    # 3. An unrelated request must neither wait for nor inherit that failure.
    sessionless = barrier_run(
      active_engine, active_runtime, "sessionless",
      [barrier_system_message, BarrierEngine::Message.new("user", "Reply only with sessionless-ok.")],
      max_tokens, max_seq,
    )
    barrier_assert!(
      sessionless.delta.async_wait_time == Time::Span.zero,
      "a sessionless request waited for another session's failing publication",
    )
    observations << sessionless
    notes << "sessionless request returned normally while publications were failing"

    # 4. Durability barriers are where the failures surface. Each one is
    #    reported once, and the barrier has to stop raising once drained.
    raised_messages = [] of String
    barriers = 0
    loop do
      barriers += 1
      barrier_assert!(barriers <= 8, "durability barriers kept raising after #{raised_messages.size} reports")
      begin
        active_runtime.flush_qbit_checkpoint_writes
        break
      rescue ex
        raised_messages << (ex.message || "")
      end
    end
    barrier_assert!(!raised_messages.empty?, "no durability barrier reported the failed publications")
    raised_messages.each do |reported|
      barrier_assert!(
        reported.includes?("async QBit checkpoint") && reported.includes?("failed"),
        "a durability barrier reported an unexpected error: #{reported}",
      )
    end
    barrier_assert!(
      raised_messages.any?(&.includes?("async QBit checkpoint publication failed")),
      "the retained failure from session A was never surfaced: #{raised_messages.inspect}",
    )
    raised_messages.each { |reported| notes << "durability barrier raised: #{reported}" }
    notes << "barrier stopped raising after #{raised_messages.size} report(s)"

    final_stats = active_runtime.qbit_cache_stats
    barrier_assert!(final_stats.async_checkpoint_pending == 0, "a publication was still pending after the barrier")
    barrier_assert!(final_stats.write_failures > 0, "the failed publications were not counted")
    notes << "pre-close accounting: enqueued=#{final_stats.async_checkpoint_enqueued} " \
             "completed=#{final_stats.async_checkpoint_completed} " \
             "pending=#{final_stats.async_checkpoint_pending} " \
             "write_failures=#{final_stats.write_failures}"

    # 5. Teardown must complete whatever close reports. A durability failure is
    #    allowed to surface here, but only after the runtime is fully closed and
    #    only once.
    close_error = nil.as(Exception?)
    begin
      active_engine.close
    rescue ex
      close_error = ex
    end
    engine = nil
    runtime = nil
    notes << (close_error ? "close reported: #{close_error.message}" : "close reported no failure")

    closed_error = nil.as(Exception?)
    begin
      active_runtime.flush_qbit_checkpoint_writes
    rescue ex
      closed_error = ex
    end
    barrier_assert!(
      closed_error.is_a?(BarrierEngine::Closed),
      "teardown did not complete: the runtime still accepts durability barriers (#{closed_error.inspect})",
    )
    active_engine.close
    notes << "runtime reached the closed state and a repeated close did not raise again"

    barrier_emit(phase, model_path, max_seq, load_time, observations, final_stats, notes)
  end
ensure
  # Teardown reports a retained durability failure by raising, which would
  # replace whatever the probe was already failing on. Keep the original
  # diagnosis and report the teardown outcome separately.
  begin
    if active_engine = engine
      active_engine.close
    elsif active_runtime = runtime
      active_runtime.close
    end
  rescue teardown_error
    STDERR.puts "[probe] teardown reported: #{teardown_error.message}"
  end
end
