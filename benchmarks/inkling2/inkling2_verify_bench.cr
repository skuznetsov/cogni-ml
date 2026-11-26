require "json"
require "option_parser"
require "../../src/ml/gguf/inkling2_engine"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  class ML::Metal::CommandBuffer
    class_property submissions : Int64 = 0_i64

    def commit_and_wait : Nil
      @@submissions += 1
      previous_def
    end
  end
{% end %}

private alias Engine = ML::GGUF::Inkling2Engine
private PREFILL_TILE          =  128
private BLOCK_SIZE            =    4
private SPLIT_BOUNDARY_PREFIX =   62
private WRAP_BOUNDARY_PREFIX  = 1022

private def read_case(path : String)
  lines = File.read_lines(path).reject(&.empty?)
  raise ArgumentError.new("#{path}: expected a header, prefix, and continuation on separate lines") unless lines.size == 3
  header, prefix, continuation = lines.map { |line| line.split.map(&.to_i32) }
  raise ArgumentError.new("#{path}: inconsistent case header") unless header.size == 3 &&
                                                                      header[0] == prefix.size &&
                                                                      header[2] == continuation.size - 1
  raise ArgumentError.new("#{path}: empty prefix or negative token ID") if prefix.empty? ||
                                                                           prefix.any?(&.negative?) ||
                                                                           continuation.any?(&.negative?)
  {header[1], prefix, continuation}
end

private def raw(values : Array(Float32)) : Bytes
  values.to_unsafe.as(UInt8*).to_slice(values.size * sizeof(Float32))
end

private def assert_logits!(expected : Array(Float32), actual : Array(Float32), label : String) : Nil
  return if raw(expected) == raw(actual)
  first = expected.each_index.find { |index| expected[index].unsafe_as(UInt32) != actual[index]?.try(&.unsafe_as(UInt32)) }
  raise "#{label}: logit bits differ (first index: #{first})"
end

private def greedy(logits : Array(Float32)) : Int32
  best_id = nil
  best = -Float32::INFINITY
  logits.each_with_index do |value, index|
    next unless value.finite? && value > best
    best_id = index
    best = value
  end
  best_id || raise "Inkling 2 returned no finite logits"
end

private def block_at(inputs : Array(Int32), offset : Int32) : StaticArray(Int32, 4)
  StaticArray[inputs[offset], inputs[offset + 1], inputs[offset + 2], inputs[offset + 3]]
end

# Read the retained cache in logical position order, so the comparison also
# detects bad KV addresses when a block crosses the end of the sliding ring.
private def kv_snapshot!(io : IO, engine : Engine, *, compare : Bool) : {Int64, Int64}
  byte_order = IO::ByteFormat::LittleEndian
  scratch = Bytes.empty
  entries = 0_i64
  float_values = 0_i64
  engine.layer_visits.times do |visit|
    count = engine.kv_count(visit)
    if compare
      expected_count = io.read_bytes(Int32, byte_order)
      raise "KV count mismatch at visit #{visit}" unless expected_count == count
    else
      io.write_bytes(count, byte_order)
    end
    count.times do |index|
      key, value, position = engine.key_value_at(visit, index)
      if compare
        raise "KV position mismatch at visit #{visit}, row #{index}" unless io.read_bytes(Int32, byte_order) == position
      else
        io.write_bytes(position, byte_order)
      end
      {key, value}.each_with_index do |values, part|
        data = raw(values)
        if compare
          scratch = Bytes.new(data.size) if scratch.size != data.size
          io.read_fully(scratch)
          label = part == 0 ? "K" : "V"
          raise "#{label} bits differ at visit #{visit}, position #{position}" unless scratch == data
        else
          io.write(data)
        end
        float_values += values.size
      end
      entries += 1
    end
  end
  {entries, float_values}
end

private def verify!(engine : Engine, prefix : Array(Int32), inputs : Array(Int32), continuation : Array(Int32), corpus : Bool)
  reference = [] of Array(Float32)
  witness = continuation[inputs.size]
  entries = 0_i64
  float_values = 0_i64
  snapshots = 0
  path = File.tempname("inkling2-kv-parity", ".bin")

  begin
    File.open(path, "w+") do |snapshot|
      engine.reset
      reference << engine.prefill(prefix)
      kv_snapshot!(snapshot, engine, compare: false)
      inputs.each_with_index do |token, index|
        reference << engine.forward(token).not_nil!
        if index == BLOCK_SIZE - 1 || index == inputs.size - 1
          kv_snapshot!(snapshot, engine, compare: false)
        end
      end
      witness_logits = engine.forward(witness).not_nil!

      snapshot.rewind
      engine.reset
      assert_logits!(reference[0], engine.prefill(prefix), "prefill")
      count, floats = kv_snapshot!(snapshot, engine, compare: true)
      entries += count
      float_values += floats
      snapshots += 1
      (0...inputs.size).step(BLOCK_SIZE) do |offset|
        logits = engine.decode_block4(block_at(inputs, offset))
        logits.each_with_index do |row, index|
          assert_logits!(reference[offset + index + 1], row, "decode row #{offset + index + 1}")
        end
        consumed = offset + BLOCK_SIZE
        if consumed == BLOCK_SIZE || consumed == inputs.size
          count, floats = kv_snapshot!(snapshot, engine, compare: true)
          entries += count
          float_values += floats
          snapshots += 1
        end
      end
      assert_logits!(witness_logits, engine.forward(witness).not_nil!, "continuation after block decode")
      raise "KV snapshot has trailing bytes" unless snapshot.read_byte.nil?
    end
  ensure
    File.delete?(path)
  end

  drift = if corpus
            reference.each_index.count { |index| greedy(reference[index]) != continuation[index] }
          else
            0
          end
  {
    raw_logit_rows:                    reference.size + 1,
    raw_logit_values:                  reference.sum(&.size) + reference[0].size,
    kv_snapshots:                      snapshots,
    kv_entries:                        entries,
    kv_float_values:                   float_values,
    known_continuation_checked:        corpus,
    known_continuation_mismatch_count: drift,
  }
end

private def command_counter : Int64
  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    ML::Metal::CommandBuffer.submissions
  {% else %}
    0_i64
  {% end %}
end

private def execute_lane(engine : Engine, prefix : Array(Int32), inputs : Array(Int32), lane : String)
  engine.reset
  engine.prefill(prefix)
  raise "unexpected prefill position #{engine.position}" unless engine.position == prefix.size
  rows = Array(Array(Float32)).new(inputs.size)
  before_commands = command_counter
  started_ms = Time.utc.to_unix_ms
  started = Time.instant
  (0...inputs.size).step(BLOCK_SIZE) do |offset|
    if lane == "scalar"
      BLOCK_SIZE.times { |row| rows << engine.forward(inputs[offset + row]).not_nil! }
    else
      engine.decode_block4(block_at(inputs, offset)).each { |row| rows << row }
    end
  end
  elapsed = (Time.instant - started).total_seconds
  stopped_ms = Time.utc.to_unix_ms
  commands = command_counter - before_commands
  raise "unexpected Metal command count for #{lane}: #{commands}" unless commands == (lane == "scalar" ? inputs.size : inputs.size // BLOCK_SIZE)
  {rows, elapsed, started_ms, stopped_ms, commands}
end

private def benchmark!(engine : Engine, prefix : Array(Int32), inputs : Array(Int32), case_path : String, passes : Int32,
                       scenario : String, pairs : Int32, warmup_pairs : Int32, initial_order : String) : Nil
  (warmup_pairs + pairs).times do |index|
    pair = index - warmup_pairs
    order = pair.even? ? initial_order : (initial_order == "AB" ? "BA" : "AB")
    lanes = order == "AB" ? {"scalar", "block4"} : {"block4", "scalar"}
    reference = nil
    lanes.each do |lane|
      rows, elapsed, started_ms, stopped_ms, commands = execute_lane(engine, prefix, inputs, lane)
      next if pair < 0
      if other = reference
        rows.each_with_index { |row, row_index| assert_logits!(other[row_index], row, "timed pair #{pair + 1}, row #{row_index + 1}") }
      end
      reference = rows
      puts({
        event:           "timing",
        case:            case_path,
        passes:          passes,
        scenario:        scenario,
        order:           order,
        prefill_tile:    PREFILL_TILE,
        lane:            lane,
        pair:            pair + 1,
        blocks:          inputs.size // BLOCK_SIZE,
        tokens:          inputs.size,
        logits_copied:   rows.size,
        logit_values:    rows.sum(&.size),
        elapsed_seconds: elapsed,
        started_ms:      started_ms,
        stopped_ms:      stopped_ms,
        commands:        commands,
      }.to_json)
      STDOUT.flush
    end
  end
end

model = nil
paths = [] of String
requested_passes = nil
mode = "all"
scenario_option = "corpus"
decode_tokens = 16
pairs = 2
warmup_pairs = 1
order = "AB"

OptionParser.parse do |parser|
  parser.banner = "Usage: inkling2_verify_bench --model PATH --case PATH [options]"
  parser.on("--model PATH", "Fixed-depth resident Metal GGUF") { |value| model = value }
  parser.on("--case PATH", "Three-line token fixture; repeat to compare more cases") { |value| paths << value }
  parser.on("--passes N", "Fixed depth 3 or 10; default from -3/-10.txt case suffix") { |value| requested_passes = value.to_i32 }
  parser.on("--mode MODE", "verify, bench, or all (default: all)") { |value| mode = value }
  parser.on("--scenario NAME", "corpus, mixed, wrap, or all (default: corpus)") { |value| scenario_option = value }
  parser.on("--decode-tokens N", "Forced inputs, a multiple of four (default: 16)") { |value| decode_tokens = value.to_i32 }
  parser.on("--pairs N", "Timed scalar/block4 pairs (default: 2)") { |value| pairs = value.to_i32 }
  parser.on("--warmup-pairs N", "Untimed warmup pairs (default: 1)") { |value| warmup_pairs = value.to_i32 }
  parser.on("--order ORDER", "AB or BA for the first timed pair (default: AB)") { |value| order = value.upcase }
  parser.on("-h", "--help", "Show options") do
    puts parser
    exit
  end
end

begin
  raise ArgumentError.new("--model and --case are required") if model.nil? || paths.empty?
  model_path = model.not_nil!
  raise ArgumentError.new("--mode must be verify, bench, or all") unless mode.in?({"verify", "bench", "all"})
  raise ArgumentError.new("--order must be AB or BA") unless order.in?({"AB", "BA"})
  raise ArgumentError.new("--pairs must be positive and --warmup-pairs non-negative") unless pairs > 0 && warmup_pairs >= 0
  raise ArgumentError.new("--decode-tokens must be a positive multiple of four") unless decode_tokens > 0 && decode_tokens % BLOCK_SIZE == 0
  raise ArgumentError.new("--scenario must be corpus, mixed, wrap, or all") unless scenario_option.in?({"corpus", "mixed", "wrap", "all"})
  scenarios = scenario_option == "all" ? ["corpus", "mixed", "wrap"] : [scenario_option]

  paths.each do |path|
    vocab, prefix, continuation = read_case(path)
    passes = requested_passes || /-(3|10)\.txt$/.match(path).try(&.[1].to_i32)
    raise ArgumentError.new("#{path}: specify --passes 3 or 10") unless passes && passes.in?({3, 10})
    raise ArgumentError.new("#{path}: expected #{decode_tokens} forced IDs and a continuation witness") unless decode_tokens < continuation.size
    inputs = continuation.first(decode_tokens)
    scenarios.each do |scenario|
      prompt = case scenario
               when "mixed"
                 raise ArgumentError.new("#{path}: mixed scenario needs at least 62 prefix tokens") if prefix.size < SPLIT_BOUNDARY_PREFIX
                 prefix.first(SPLIT_BOUNDARY_PREFIX)
               when "wrap"
                 Array(Int32).new(WRAP_BOUNDARY_PREFIX) { |index| prefix[index % prefix.size] }
               else
                 prefix
               end
      # Reserve one extra slot for a scalar step after all four-row blocks.
      context = prompt.size + inputs.size + 1
      engine = Engine.new(model_path, context, metal: true, resident: true, passes: passes, prefill_tile: PREFILL_TILE)
      begin
        config = engine.weights.config
        raise ArgumentError.new("#{path}: expected vocabulary size #{vocab}, model has #{config.vocab_size}") unless config.vocab_size == vocab
        if scenario == "wrap" && config.sliding_window != WRAP_BOUNDARY_PREFIX + 2
          raise ArgumentError.new("wrap scenario requires a 1024-token sliding window")
        end
        puts({event: "case", case: path, passes: passes, scenario: scenario, prefill_tile: PREFILL_TILE,
              prefix_tokens: prompt.size, decode_tokens: inputs.size, context: context,
              vocab_size: vocab, sliding_window: config.sliding_window, backend: "resident_metal"}.to_json)
        if mode != "bench"
          result = verify!(engine, prompt, inputs, continuation, scenario == "corpus")
          puts({event: "verify", case: path, passes: passes, scenario: scenario}.merge(result).to_json)
        end
        STDOUT.flush
        benchmark!(engine, prompt, inputs, path, passes, scenario, pairs, warmup_pairs, order) if mode != "verify"
      ensure
        engine.close
      end
    end
  end
rescue ex : Exception
  STDERR.puts("inkling2_verify_bench: #{ex.message}")
  exit 1
end
