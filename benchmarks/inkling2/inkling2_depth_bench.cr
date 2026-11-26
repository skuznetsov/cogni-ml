require "digest/sha256"
require "json"
require "option_parser"
require "../../src/ml/gguf/inkling2_engine"
require "../../src/ml/gguf/inkling2_greedy_decode"
require "../../src/ml/gguf/inkling2_tokenizer"

private alias Engine = ML::GGUF::Inkling2Engine
private alias GreedyDecode = ML::GGUF::Inkling2GreedyDecode
private PREFILL_TILE          =  128
private BLOCK_SIZE            =    4
private SPLIT_BOUNDARY_PREFIX =   62
private WRAP_BOUNDARY_PREFIX  = 1022

private record CaseData,
  path : String,
  vocab_size : Int32,
  prefix : Array(Int32),
  continuation : Array(Int32),
  expected_outputs : Int32

private record KVSnapshot,
  per_visit : Array(String),
  entries : Int64,
  float_values : Int64,
  sha256 : String

private record ScalarReference,
  logits : Array(Array(Float32)),
  target_kv : Array(KVSnapshot)

private record P3Reference,
  epi_kv : Array(KVSnapshot)

private record LaneResult,
  lane : String,
  ids : Array(Int32),
  record : NamedTuple(
    event: String,
    case: String,
    case_id: String,
    scenario: String,
    passes: Int32,
    prefill_tile: Int32,
    lane: String,
    pair: Int32,
    order: String,
    requested_tokens: Int32,
    generated_tokens: Int32,
    committed_rows: Int32,
    blocks: Int32,
    accepted_rows: Int32,
    rejected_rows: Int32,
    reached_eos: Bool,
    prefill_start_unix_ns: Int64,
    prefill_end_unix_ns: Int64,
    prefill_elapsed_seconds: Float64,
    start_unix_ns: Int64,
    end_unix_ns: Int64,
    elapsed_seconds: Float64)

private def unix_ns : Int64
  Time.utc.to_unix_ns.to_i64
end

private def parse_integer(text : String, option : String) : Int32
  text.to_i32? || raise ArgumentError.new("#{option} expects a 32-bit integer, got #{text.inspect}")
end

private def read_case(path : String) : CaseData
  lines = File.read_lines(path).reject(&.empty?)
  raise ArgumentError.new("#{path}: expected a header, prefix, and continuation on separate lines") unless lines.size == 3

  header, prefix, continuation = lines.map { |line| line.split.map(&.to_i32) }
  raise ArgumentError.new("#{path}: inconsistent case header") unless header.size == 3 &&
                                                                      header[0] == prefix.size &&
                                                                      header[2] == continuation.size - 1
  raise ArgumentError.new("#{path}: empty prefix or negative token ID") if prefix.empty? ||
                                                                           prefix.any?(&.negative?) ||
                                                                           continuation.empty? ||
                                                                           continuation.any?(&.negative?)

  CaseData.new(path, header[1], prefix, continuation, header[2])
end

private def scenario_prefix(prefix : Array(Int32), scenario : String) : Array(Int32)
  case scenario
  when "corpus"
    prefix
  when "mixed"
    raise ArgumentError.new("mixed scenario needs at least #{SPLIT_BOUNDARY_PREFIX} prefix tokens") if prefix.size < SPLIT_BOUNDARY_PREFIX
    prefix.first(SPLIT_BOUNDARY_PREFIX)
  when "wrap"
    Array(Int32).new(WRAP_BOUNDARY_PREFIX) { |index| prefix[index % prefix.size] }
  when "physical_wrap"
    Array(Int32).new(WRAP_BOUNDARY_PREFIX + BLOCK_SIZE) { |index| prefix[index % prefix.size] }
  else
    raise ArgumentError.new("unknown scenario #{scenario.inspect}")
  end
end

private def float_bytes(values : Array(Float32)) : Bytes
  values.to_unsafe.as(UInt8*).to_slice(values.size * sizeof(Float32))
end

private def update_i32(digest : Digest::SHA256, value : Int32) : Nil
  bits = value.unsafe_as(UInt32)
  digest.update(Bytes.new(4) { |index| ((bits >> (index * 8)) & 0xff).to_u8 })
end

private def digest_visit(count : Int32, &entry : Int32 -> Tuple(Array(Float32), Array(Float32), Int32)) : {String, Int64}
  digest = Digest::SHA256.new
  update_i32(digest, count)
  float_values = 0_i64
  count.times do |index|
    key, value, position = yield index
    update_i32(digest, position)
    update_i32(digest, key.size)
    update_i32(digest, value.size)
    digest.update(float_bytes(key))
    digest.update(float_bytes(value))
    float_values += key.size + value.size
  end
  {digest.hexfinal, float_values}
end

private def finish_snapshot(digests : Array(String), entries : Int64, float_values : Int64) : KVSnapshot
  digest = Digest::SHA256.new
  digests.each { |value| digest.update(value) }
  KVSnapshot.new(digests, entries, float_values, digest.hexfinal)
end

private def target_snapshot(engine : Engine) : KVSnapshot
  digests = [] of String
  entries = 0_i64
  float_values = 0_i64
  engine.layer_visits.times do |visit|
    count = engine.kv_count(visit)
    digest, values = digest_visit(count) do |index|
      engine.key_value_at(visit, index)
    end
    digests << digest
    entries += count
    float_values += values
  end
  finish_snapshot(digests, entries, float_values)
end

private def target_tail_snapshot(engine : Engine, visits : Int32) : KVSnapshot
  raise ArgumentError.new("invalid target tail visit count #{visits}") unless visits.in?(1..engine.layer_visits)
  first_visit = engine.layer_visits - visits
  digests = [] of String
  entries = 0_i64
  float_values = 0_i64
  first_visit.upto(engine.layer_visits - 1) do |visit|
    count = engine.kv_count(visit)
    digest, values = digest_visit(count) do |index|
      engine.key_value_at(visit, index)
    end
    digests << digest
    entries += count
    float_values += values
  end
  finish_snapshot(digests, entries, float_values)
end

private def draft_snapshot(engine : Engine) : KVSnapshot
  visits = engine.draft_layer_visits
  raise "depth engine exposed no draft epilogue visits" unless visits > 0
  digests = [] of String
  entries = 0_i64
  float_values = 0_i64
  visits.times do |visit|
    count = engine.draft_kv_count(visit)
    digest, values = digest_visit(count) do |index|
      engine.draft_key_value_at(visit, index)
    end
    digests << digest
    entries += count
    float_values += values
  end
  finish_snapshot(digests, entries, float_values)
end

private def assert_logits!(expected : Array(Float32), actual : Array(Float32), label : String) : Nil
  raise "#{label}: logit vector size differs (expected #{expected.size}, got #{actual.size})" unless expected.size == actual.size
  return if float_bytes(expected) == float_bytes(actual)
  first = expected.each_index.find { |index| expected[index].unsafe_as(UInt32) != actual[index].unsafe_as(UInt32) }
  raise "#{label}: logit bits differ (first index: #{first})"
end

private def assert_snapshot!(expected : KVSnapshot, actual : KVSnapshot, label : String) : Nil
  unless expected.entries == actual.entries && expected.float_values == actual.float_values && expected.per_visit == actual.per_visit
    first = expected.per_visit.each_index.find { |index| expected.per_visit[index]? != actual.per_visit[index]? }
    raise "#{label}: retained KV differs (first visit: #{first}, expected entries #{expected.entries}, got #{actual.entries})"
  end
end

private def make_engine(model_path : String, context : Int32, passes : Int32, depth_speculation : Bool = false) : Engine
  Engine.new(model_path, context, metal: true, resident: true, passes: passes,
    prefill_tile: PREFILL_TILE, depth_speculation: depth_speculation)
end

private def validate_model_case!(engine : Engine, data : CaseData, scenario : String) : Nil
  config = engine.weights.config
  raise ArgumentError.new("#{data.path}: expected vocabulary size #{data.vocab_size}, model has #{config.vocab_size}") unless config.vocab_size == data.vocab_size
  if scenario.in?({"wrap", "physical_wrap"}) && config.sliding_window != WRAP_BOUNDARY_PREFIX + 2
    raise ArgumentError.new("wrap scenario requires a 1024-token sliding window")
  end
end

private def scalar_reference(model_path : String, context : Int32, prefix : Array(Int32), tokens : Array(Int32),
                             data : CaseData, scenario : String) : ScalarReference
  engine = make_engine(model_path, context, 10)
  begin
    validate_model_case!(engine, data, scenario)
    logits = [engine.prefill(prefix)]
    target = [] of KVSnapshot
    tokens.each do |token|
      logits << engine.forward(token).not_nil!
      target << target_snapshot(engine)
    end
    ScalarReference.new(logits, target)
  ensure
    engine.close
  end
end

private def p3_reference(model_path : String, context : Int32, prefix : Array(Int32), tokens : Array(Int32),
                         data : CaseData, scenario : String, epi_visits : Int32) : P3Reference
  engine = make_engine(model_path, context, 3)
  begin
    validate_model_case!(engine, data, scenario)
    raise ArgumentError.new("p3 model exposes fewer than #{epi_visits} layer visits") if engine.layer_visits < epi_visits
    engine.prefill(prefix)
    snapshots = [] of KVSnapshot
    tokens.each do |token|
      engine.forward(token, logits: false)
      snapshots << target_tail_snapshot(engine, epi_visits)
    end
    P3Reference.new(snapshots)
  ensure
    engine.close
  end
end

private def verify_depth!(model_path : String, data : CaseData, prefix : Array(Int32), scenario : String,
                          context : Int32) : Nil
  forced = data.continuation
  raise ArgumentError.new("#{data.path}: forced verification needs four inputs and one follow-on token") if forced.size < BLOCK_SIZE + 1

  # The scalar p10 path is the reference for every forced row, every retained
  # target state, and the scalar continuation after each possible rollback.
  reference = scalar_reference(model_path, context, prefix, forced, data, scenario)

  depth = make_engine(model_path, context, 10, depth_speculation: true)
  begin
    validate_model_case!(depth, data, scenario)
    epi_visits = depth.draft_layer_visits
    raise "depth speculation exposed no draft epilogue visits" unless epi_visits > 0

    # An independent serial p3 run supplies the draft epilogue cache reference.
    p3 = p3_reference(model_path, context, prefix, forced, data, scenario, epi_visits)

    block_count = 0
    offset = 0
    while offset + BLOCK_SIZE < forced.size
      block = StaticArray[forced[offset], forced[offset + 1], forced[offset + 2], forced[offset + 3]]
      (1..BLOCK_SIZE).each do |commit|
        depth.reset
        prefix_logits = depth.prefill(prefix)
        assert_logits!(reference.logits[0], prefix_logits, "#{scenario} prefix logits")
        offset.times do |index|
          depth.forward(forced[index], logits: false)
        end
        if offset > 0
          assert_snapshot!(reference.target_kv[offset - 1], target_snapshot(depth),
            "#{scenario} block #{block_count}, pre-block p10 state")
        end

        rows = depth.decode_depth_block4(block, commit: commit)
        rows.each_with_index do |row, row_index|
          assert_logits!(reference.logits[offset + row_index + 1], row,
            "#{scenario} block #{block_count}, candidate row #{row_index + 1} with commit #{commit}")
        end
        raise "depth block committed #{depth.position - prefix.size - offset} rows, expected #{commit}" unless depth.position == prefix.size + offset + commit

        assert_snapshot!(reference.target_kv[offset + commit - 1], target_snapshot(depth),
          "#{scenario} block #{block_count}, p10 target KV after commit #{commit}")
        assert_snapshot!(p3.epi_kv[offset + commit - 1], draft_snapshot(depth),
          "#{scenario} block #{block_count}, p3 draft epilogue KV after commit #{commit}")

        followup = depth.forward(forced[offset + commit]).not_nil!
        assert_logits!(reference.logits[offset + commit + 1], followup,
          "#{scenario} block #{block_count}, scalar follow-on after commit #{commit}")
        assert_snapshot!(reference.target_kv[offset + commit], target_snapshot(depth),
          "#{scenario} block #{block_count}, scalar follow-on target KV after commit #{commit}")
        assert_snapshot!(p3.epi_kv[offset + commit], draft_snapshot(depth),
          "#{scenario} block #{block_count}, scalar follow-on p3 draft KV after commit #{commit}")

        puts({
          event:                   "verify",
          case:                    data.path,
          case_id:                 "#{File.basename(data.path)}:#{scenario}",
          scenario:                scenario,
          passes:                  10,
          prefill_tile:            PREFILL_TILE,
          prefix_tokens:           prefix.size,
          block_offset:            offset,
          retained_rows:           commit,
          forced_logits_rows:      rows.size,
          target_kv_visits:        depth.layer_visits,
          target_kv_entries:       reference.target_kv[offset + commit - 1].entries,
          target_kv_sha256:        reference.target_kv[offset + commit - 1].sha256,
          draft_epi_visits:        depth.draft_layer_visits,
          draft_epi_kv_entries:    p3.epi_kv[offset + commit - 1].entries,
          draft_epi_kv_sha256:     p3.epi_kv[offset + commit - 1].sha256,
          followup_logits_checked: true,
          followup_kv_checked:     true,
        }.to_json)
        STDOUT.flush
      end
      block_count += 1
      offset += BLOCK_SIZE
    end
    raise "#{data.path}: no complete four-row block had a follow-on token" if block_count == 0
  ensure
    depth.close
  end
end

private def run_lane(engine : Engine, data : CaseData, prefix : Array(Int32), scenario : String,
                     limit : Int32, eos_id : Int32, lane : String, pair : Int32, order : String) : LaneResult
  depth_speculation = lane == "depth_spec"
  passes = 10
  engine.reset
  validate_model_case!(engine, data, scenario)

  prefill_start = unix_ns
  prefill_monotonic = Time.instant
  first_logits = engine.prefill(prefix)
  first_id = Engine.greedy_token_id(first_logits)
  prefill_elapsed = (Time.instant - prefill_monotonic).total_seconds
  prefill_end = unix_ns

  started_unix_ns = unix_ns
  monotonic_start = Time.instant
  result = GreedyDecode.generate(engine, prefix, first_id, limit, eos_id,
    ngram: false, depth_speculation: depth_speculation)
  elapsed = (Time.instant - monotonic_start).total_seconds
  ended_unix_ns = unix_ns

  committed_rows = engine.position - prefix.size
  raise "#{lane}: committed-row count #{committed_rows} differs from generated_tokens - 1 (#{result.ids.size - 1})" unless committed_rows == result.ids.size - 1
  rejected_rows = result.blocks * BLOCK_SIZE - result.accepted_rows
  raise "#{lane}: speculative row accounting does not balance" unless result.blocks * BLOCK_SIZE == result.accepted_rows + rejected_rows

  record = {
    event:                   "timing",
    case:                    data.path,
    case_id:                 "#{File.basename(data.path)}:#{scenario}",
    scenario:                scenario,
    passes:                  passes,
    prefill_tile:            PREFILL_TILE,
    lane:                    lane,
    pair:                    pair,
    order:                   order,
    requested_tokens:        limit,
    generated_tokens:        result.ids.size,
    committed_rows:          committed_rows,
    blocks:                  result.blocks,
    accepted_rows:           result.accepted_rows,
    rejected_rows:           rejected_rows,
    reached_eos:             result.ids.last? == eos_id,
    prefill_start_unix_ns:   prefill_start,
    prefill_end_unix_ns:     prefill_end,
    prefill_elapsed_seconds: prefill_elapsed,
    start_unix_ns:           started_unix_ns,
    end_unix_ns:             ended_unix_ns,
    elapsed_seconds:         elapsed,
  }
  LaneResult.new(lane, result.ids, record)
end

private def check_transcript!(actual : Array(Int32), expected : Array(Int32), label : String) : Nil
  return if actual == expected
  first = (0...Math.min(actual.size, expected.size)).find { |index| actual[index] != expected[index] }
  raise "#{label}: transcript differs at token #{first || Math.min(actual.size, expected.size)} (expected #{expected.size} tokens, got #{actual.size})"
end

private def benchmark!(model_path : String, data : CaseData, prefix : Array(Int32), scenario : String,
                       limit : Int32, pairs : Int32, warmup_pairs : Int32, initial_order : String) : Nil
  context = prefix.size + limit + BLOCK_SIZE
  scalar_engine = make_engine(model_path, context, 10)
  begin
    depth_engine = make_engine(model_path, context, 10, depth_speculation: true)
    begin
      validate_model_case!(scalar_engine, data, scenario)
      validate_model_case!(depth_engine, data, scenario)
      scalar_eos = ML::GGUF::Inkling2Tokenizer.from_gguf(scalar_engine.weights.gguf).eos_id
      depth_eos = ML::GGUF::Inkling2Tokenizer.from_gguf(depth_engine.weights.gguf).eos_id
      raise "scalar and depth engines disagree on EOS" unless scalar_eos == depth_eos

      (warmup_pairs + pairs).times do |index|
        pair = index - warmup_pairs
        order = pair.even? ? initial_order : (initial_order == "AB" ? "BA" : "AB")
        lanes = order == "AB" ? {"scalar_p10", "depth_spec"} : {"depth_spec", "scalar_p10"}
        results = [] of LaneResult
        lanes.each do |lane|
          engine = lane == "scalar_p10" ? scalar_engine : depth_engine
          eos_id = lane == "scalar_p10" ? scalar_eos : depth_eos
          result = run_lane(engine, data, prefix, scenario, limit, eos_id, lane, pair + 1, order)
          results << result if pair >= 0
        end
        next if pair < 0

        scalar = results.find { |result| result.lane == "scalar_p10" }.not_nil!
        speculative = results.find { |result| result.lane == "depth_spec" }.not_nil!
        check_transcript!(speculative.ids, scalar.ids, "pair #{pair + 1} scalar/depth transcript")
        if scenario == "corpus" && limit == data.continuation.size
          check_transcript!(scalar.ids, data.continuation, "pair #{pair + 1} fixture transcript")
        end
        results.each do |result|
          puts result.record.to_json
          STDOUT.flush
        end
      end
    ensure
      depth_engine.close
    end
  ensure
    scalar_engine.close
  end
end

model_path = nil.as(String?)
case_paths = [] of String
mode = "all"
scenario_option = "corpus"
token_limit = nil.as(Int32?)
pairs = 2
warmup_pairs = 1
order = "AB"

OptionParser.parse do |parser|
  parser.banner = "Usage: inkling2_depth_bench --model PATH --case PATH [options]"
  parser.on("--model PATH", "TML-d24-varloop-research-IQ4_NL GGUF; resident Metal") { |value| model_path = value }
  parser.on("--case PATH", "Three-line token fixture; repeat for more cases") { |value| case_paths << value }
  parser.on("--mode MODE", "verify, bench, or all (default: all)") { |value| mode = value }
  parser.on("--scenario NAME", "corpus, mixed, wrap, physical_wrap, or all (default: corpus)") { |value| scenario_option = value }
  parser.on("--tokens N", "Requested generated tokens; default is continuation fixture length") { |value| token_limit = parse_integer(value, "--tokens") }
  parser.on("--pairs N", "Timed scalar/depth pairs (default: 2)") { |value| pairs = parse_integer(value, "--pairs") }
  parser.on("--warmup-pairs N", "Untimed warmup pairs (default: 1)") { |value| warmup_pairs = parse_integer(value, "--warmup-pairs") }
  parser.on("--order ORDER", "AB or BA for first pair (default: AB)") { |value| order = value.upcase }
  parser.on("-h", "--help", "Show options") do
    puts parser
    exit
  end
end

begin
  raise ArgumentError.new("--model and at least one --case are required") unless model_path && !case_paths.empty?
  raise ArgumentError.new("model not found: #{model_path}") unless File.exists?(model_path.not_nil!)
  raise ArgumentError.new("--mode must be verify, bench, or all") unless mode.in?({"verify", "bench", "all"})
  raise ArgumentError.new("--scenario must be corpus, mixed, wrap, physical_wrap, or all") unless scenario_option.in?({"corpus", "mixed", "wrap", "physical_wrap", "all"})
  raise ArgumentError.new("--order must be AB or BA") unless order.in?({"AB", "BA"})
  raise ArgumentError.new("--tokens must be positive") if token_limit && token_limit.not_nil! <= 0
  raise ArgumentError.new("--pairs must be positive and --warmup-pairs non-negative") unless pairs > 0 && warmup_pairs >= 0

  scenarios = scenario_option == "all" ? ["corpus", "mixed", "wrap", "physical_wrap"] : [scenario_option]
  case_paths.each do |path|
    data = read_case(path)
    scenarios.each do |scenario|
      prefix = scenario_prefix(data.prefix, scenario)
      limit = token_limit ? token_limit.not_nil! : data.continuation.size
      raise ArgumentError.new("requested tokens exceed the fixture continuation; supply a longer case") if limit > data.continuation.size
      context = prefix.size + Math.max(data.continuation.size, limit) + BLOCK_SIZE

      puts({
        event:                         "case",
        case:                          path,
        case_id:                       "#{File.basename(path)}:#{scenario}",
        scenario:                      scenario,
        model:                         model_path,
        passes:                        10,
        prefill_tile:                  PREFILL_TILE,
        prefix_tokens:                 prefix.size,
        expected_outputs:              data.expected_outputs,
        requested_tokens:              limit,
        context:                       context,
        backend:                       "resident_metal",
        fixture_transcript_validation: mode != "verify" && scenario == "corpus" && limit == data.continuation.size,
      }.to_json)
      STDOUT.flush

      verify_depth!(model_path.not_nil!, data, prefix, scenario, context) if mode != "bench"
      benchmark!(model_path.not_nil!, data, prefix, scenario, limit, pairs, warmup_pairs, order) if mode != "verify"
    end
  end
rescue ex : Exception
  STDERR.puts("inkling2_depth_bench: #{ex.message}")
  exit 1
end
