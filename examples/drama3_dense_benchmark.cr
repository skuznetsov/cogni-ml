# Paired microbenchmark for the four dense BF16 projections in DRAMA-3 layer 0.
# This measures kernels only; it is not a full-model throughput benchmark.
# Build with the repository's normal Metal flags, then run:
#   DRAMA3_BACKEND=metal /tmp/drama3_dense_benchmark MODEL_DIR [options]
# An upstream input bundle can be supplied with --dense-fixture=PATH.
require "json"
require "random/secure"
require "set"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3DenseBenchmark
    include ML::Audio::Drama3

    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer

    PROJECTIONS = [
      {name: "qkv", rows: 8192, weight: "backbone.layers.0.linear_attn.in_proj_qkv.weight"},
      {name: "z", rows: 4096, weight: "backbone.layers.0.linear_attn.in_proj_z.weight"},
      {name: "b", rows: 32, weight: "backbone.layers.0.linear_attn.in_proj_b.weight"},
      {name: "a", rows: 32, weight: "backbone.layers.0.linear_attn.in_proj_a.weight"},
    ]
    WIDTH             = 2048
    WARMUPS           =    2
    MAX_LARGE_REPEATS =    8
    MAX_TINY_REPEATS  = 4096

    record Input, values : Vector, id : String, source : String, row : Int32
    record Arm, projection : String, pair : Int32, warmup : Bool, name : String,
      command_index : Int64, outputs : Array(Buffer)

    def self.input_id(values : Vector) : String
      # Stable content identity over F32 bit patterns; this is a pairing label,
      # not a cryptographic integrity claim.
      hash = 0xcbf29ce484222325_u64
      values.each do |value|
        hash = (hash ^ value.unsafe_as(UInt32).to_u64) &* 0x100000001b3_u64
      end
      "fnv64:%016x" % hash
    end

    def self.generated_input : Input
      values = Array(Float32).new(WIDTH) do |index|
        (((index * 17 % 101) - 50).to_f32 / 83_f32)
      end
      Input.new(values, input_id(values), "deterministic-generated-f32", 0)
    end

    def self.fixture_input(path : String, selected_row : Int32) : Input
      value = JSON.parse(File.read(path))
      raise "dense fixture format" unless value["format"].as_s == "drama3-dense-projection-inputs-v1"
      raise "dense fixture projection" unless value["projection"].as_s == "backbone.layers.0.linear_attn"
      raise "dense fixture input dtype" unless value["input_dtype"].as_s == "float32"
      shape = value["input_shape"].as_a.map(&.as_i)
      raise "dense fixture input shape" unless shape == [WIDTH]
      rows = value["rows"].as_a
      raise "dense fixture rows missing" if rows.empty?
      seen = Set(Int32).new
      selected : Vector? = nil
      rows.each do |row|
        index = row["row"].as_i
        raise "dense fixture row index" unless index >= 0 && !seen.includes?(index)
        seen << index
        values = row["values"].as_a.map(&.as_f.to_f32)
        raise "dense fixture row #{index} shape" unless values.size == WIDTH
        raise "dense fixture row #{index} contains non-finite values" unless values.all?(&.finite?)
        selected = values if index == selected_row
      end
      values = selected || raise "dense fixture row #{selected_row} missing"
      Input.new(values, input_id(values), "#{File.expand_path(path)}#row=#{selected_row}", selected_row)
    end

    def self.load_matrices(model_path : String) : Hash(String, Matrix)
      store = WeightStore.new(model_path)
      matrices = {} of String => Matrix
      PROJECTIONS.each do |projection|
        matrix = store.matrix(projection[:weight])
        raise "#{projection[:name]} matrix shape" unless matrix.rows == projection[:rows] && matrix.cols == WIDTH
        raise "#{projection[:name]} must use dense weights without scales/biases" if matrix.scales || matrix.biases
        raise "#{projection[:name]} weight dtype must be BF16, got #{matrix.tensor.dtype}" unless matrix.tensor.dtype == "BF16"
        matrices[projection[:name]] = matrix
      end
      matrices
    end

    def self.repeat_count(name : String, large : Int32, tiny : Int32) : Int32
      name.in?("qkv", "z") ? large : tiny
    end

    def self.compare_terminal!(ctx : Context, expected : Buffer, actual : Buffer,
                               projection : String, repeat : Int32) : Nil
      raise "#{projection} output shape mismatch" unless expected.count == actual.count
      # The pair has already been drained. Read shared contents directly here
      # rather than calling Context#read, which invokes finish on every buffer.
      ctx.readbacks["terminal"] += expected.count.to_i64 * 2 * 4
      left = expected.buffer.contents.as(Float32*) + expected.offset // 4
      right = actual.buffer.contents.as(Float32*) + actual.offset // 4
      expected.count.times do |index|
        lhs, rhs = left[index], right[index]
        raise "#{projection} non-finite terminal output" unless lhs.finite? && rhs.finite?
        unless lhs.unsafe_as(UInt32) == rhs.unsafe_as(UInt32)
          raise "#{projection}[repeat=#{repeat},row=#{index}] bitwise mismatch"
        end
      end
    end

    def self.pair(ctx : Context, matrices : Hash(String, Matrix), input : Buffer,
                  projection : NamedTuple(name: String, rows: Int32, weight: String),
                  repetitions : Int32, candidate_mode : String, pair_index : Int32,
                  warmup : Bool, input_identity : String) : Int64
      name = projection[:name]
      before = ctx.graph_stats.dup
      reads_before = ctx.readbacks.dup
      arms = [] of Arm
      started = Time.instant
      order = pair_index.even? ? ["baseline", "candidate"] : ["candidate", "baseline"]
      order.each do |arm_name|
        command_index = ctx.graph_stats["submitted"]
        reduction = arm_name == "baseline" ? "tree" : candidate_mode
        outputs = Array(Buffer).new(repetitions) do
          ctx.matvec(matrices[name], input, dense_reduction: reduction)
        end
        ctx.submit
        raise "completion wait before final drain" unless ctx.graph_stats["completion_waits"] == before["completion_waits"]
        raise "readback before final drain" unless ctx.readbacks == reads_before
        arms << Arm.new(name, pair_index, warmup, arm_name, command_index, outputs)
      end
      queued = ctx.graph_stats
      raise "expected two queued command buffers" unless queued["pending"] == 2 && queued["submitted"] == before["submitted"] + 2
      ctx.finish
      pair_wall_seconds = (Time.instant - started).total_seconds
      drained = ctx.graph_stats
      raise "incomplete final drain" unless drained["completion_waits"] == before["completion_waits"] + 2 &&
                                            drained["completed"] == before["completed"] + 2 && drained["pending"] == 0 && drained["retained_bytes"] == 0

      baseline = arms.find { |arm| arm.name == "baseline" }.not_nil!
      candidate = arms.find { |arm| arm.name == "candidate" }.not_nil!
      verified_elements = 0_i64
      baseline.outputs.zip(candidate.outputs).each_with_index do |(expected_buffer, actual_buffer), repeat|
        raise "#{name} output shape mismatch" unless expected_buffer.count == projection[:rows]
        compare_terminal!(ctx, expected_buffer, actual_buffer, name, repeat)
        verified_elements += projection[:rows]
      end
      after = ctx.graph_stats
      raise "verification submitted or waited for extra work" unless after == drained
      expected_bytes = verified_elements * 2 * 4
      terminal_before = reads_before.fetch("terminal", 0_i64)
      terminal_after = ctx.readbacks.fetch("terminal", 0_i64)
      raise "unexpected terminal readback" unless terminal_after - terminal_before == expected_bytes
      raise "unexpected control readback" unless ctx.readbacks.fetch("control", 0_i64) == reads_before.fetch("control", 0_i64)
      raise "unexpected diagnostic readback" unless ctx.readbacks.fetch("diagnostic", 0_i64) == reads_before.fetch("diagnostic", 0_i64)

      arms.each do |arm|
        puts({event: "dense_benchmark_arm", projection: name, pair: pair_index, warmup: warmup,
              arm: arm.name, command_index: arm.command_index, repetitions: repetitions,
              shape: [projection[:rows], WIDTH], weight_dtype: "BF16", input_id: input_identity}.to_json)
      end
      puts({event: "dense_benchmark_pair", projection: name, pair: pair_index, warmup: warmup,
            order: arms.map(&.name), wall_seconds: pair_wall_seconds,
            waits_before_drain: queued["completion_waits"] - before["completion_waits"],
            readbacks_before_drain: 0, submissions: after["submitted"] - before["submitted"],
            final_drain_waits: after["completion_waits"] - before["completion_waits"],
            exact_bitwise: true, verified_elements: verified_elements,
            terminal_readback_bytes: expected_bytes, graph: after}.to_json)
      verified_elements
    end

    def self.run : Nil
      raise "usage: drama3_dense_benchmark MODEL_DIR [--pairs=N] [--large-repeats=8] [--tiny-repeats=1024] [--candidate=simd|tree] [--dense-fixture=PATH] [--input-row=N]" if ARGV.empty?
      model_path = ARGV.shift
      pairs, large_repeats, tiny_repeats = 7, 8, 1024
      candidate_mode, fixture_path, input_row = "simd", nil.as(String?), 0
      ARGV.each do |argument|
        if argument.starts_with?("--pairs=")
          pairs = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--large-repeats=")
          large_repeats = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--tiny-repeats=")
          tiny_repeats = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--candidate=")
          candidate_mode = argument.split('=', 2)[1]
        elsif argument.starts_with?("--dense-fixture=")
          fixture_path = argument.split('=', 2)[1]
        elsif argument.starts_with?("--input-row=")
          input_row = argument.split('=', 2)[1].to_i
        else
          raise "unknown argument #{argument}"
        end
      end
      raise "pairs must be 1..15" unless pairs.in?(1..15)
      raise "large repeats must be 1..#{MAX_LARGE_REPEATS}" unless large_repeats.in?(1..MAX_LARGE_REPEATS)
      raise "tiny repeats must be 1..#{MAX_TINY_REPEATS}" unless tiny_repeats.in?(1..MAX_TINY_REPEATS)
      raise "candidate must be tree or simd" unless candidate_mode.in?("tree", "simd")
      raise "input row must be non-negative" unless input_row >= 0
      raise "--input-row requires --dense-fixture" if fixture_path.nil? && input_row != 0
      input_fixture = fixture_path ? fixture_input(fixture_path.not_nil!, input_row) : generated_input
      matrices = load_matrices(model_path)

      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
      ENV["DRAMA3_METAL_TIMELINE"] = "1"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV.delete("DRAMA3_PROFILE")
      Profile.configure_from_env!
      ctx = Context.new
      input_buffer = ctx.upload(input_fixture.values)
      invocation = Random::Secure.hex(16)
      stamp = {event: "dense_benchmark_invocation", id: invocation}.to_json
      puts stamp
      STDERR.puts stamp

      repeats = {} of String => Int32
      PROJECTIONS.each { |projection| repeats[projection[:name]] = repeat_count(projection[:name], large_repeats, tiny_repeats) }
      verified_elements = 0_i64
      PROJECTIONS.each do |projection|
        WARMUPS.times do |index|
          verified_elements += pair(ctx, matrices, input_buffer, projection, repeats[projection[:name]],
            candidate_mode, index, true, input_fixture.id)
        end
        pairs.times do |index|
          verified_elements += pair(ctx, matrices, input_buffer, projection, repeats[projection[:name]],
            candidate_mode, index, false, input_fixture.id)
        end
      end
      final = ctx.graph_stats
      raise "benchmark left work pending" unless final["pending"] == 0 && final["retained_bytes"] == 0
      readbacks = {control:    ctx.readbacks.fetch("control", 0_i64),
                   diagnostic: ctx.readbacks.fetch("diagnostic", 0_i64),
                   terminal:   ctx.readbacks.fetch("terminal", 0_i64)}
      puts({event: "dense_benchmark_summary", invocation_id: invocation,
            model: File.expand_path(model_path), input_id: input_fixture.id,
            input_source: input_fixture.source, input_row: input_fixture.row,
            candidate_mode: candidate_mode, projections: PROJECTIONS.map(&.[:name]),
            shapes: PROJECTIONS.to_h { |projection| {projection[:name], [projection[:rows], WIDTH]} },
            pairs_per_projection: pairs, warmups_per_projection: WARMUPS,
            large_repeats: large_repeats, tiny_repeats: tiny_repeats, repeats: repeats,
            weight_dtypes: PROJECTIONS.to_h { |projection| {projection[:name], "BF16"} },
            graph: final, readbacks: readbacks, verified_elements: verified_elements}.to_json)
    rescue ex
      ctx.try(&.abort)
      raise ex
    end
  end

  Drama3DenseBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
