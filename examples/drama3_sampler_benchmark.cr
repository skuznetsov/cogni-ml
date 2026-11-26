# Fixed-input diagnostic, not a full-model throughput benchmark.
# Build from the repository root with the normal Metal link flags:
#   COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh crystal 180 8192 \
#     build --release -Ddrama3_metal examples/drama3_sampler_benchmark.cr \
#     --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++ -fuse-ld=/usr/bin/ld" \
#     -o /tmp/drama3_sampler_benchmark
# Run with a fixture from scripts/drama3_sampler_reference.py. GPU timestamps
# are emitted by the existing Context timeline after the final pair drain.
require "json"
require "random/secure"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3SamplerBenchmark
    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer
    MAIN_CALLS  =  36
    TOWER_CALLS = 153

    class FixtureCase
      getter name : String, role : String, profile : String
      getter logits : Array(Float32), draws : Array(Float32), expected_ids : Array(Int32)
      getter retry_expected_ids : Array(Int32)
      getter temperature : Float64, top_k : Int32, top_p : Float64

      def initialize(value : JSON::Any)
        @name = value["name"].as_s
        @role = value["role"].as_s
        @profile = value["profile"].as_s
        @logits = value["logits"].as_a.map(&.as_f.to_f32)
        @draws = value["draws"].as_a.map(&.as_f.to_f32)
        @expected_ids = value["expected_ids"].as_a.map(&.as_i)
        @retry_expected_ids = value["retry_expected_ids"]?.try(&.as_a.map(&.as_i)) || [] of Int32
        @temperature = value["temperature"].as_f
        @top_k = value["top_k"].as_i
        @top_p = value["top_p"].as_f
        raise "invalid fixture role #{@role}" unless @role.in?("main", "tower")
        expected_size = @role == "main" ? 4097 : 4096
        raise "fixture shape #{@name}" unless @logits.size == expected_size
        raise "fixture logits #{@name}" unless @logits.all?(&.finite?)
        raise "fixture draws #{@name}" unless !@draws.empty? && @draws.all? { |v| v.finite? && v >= 0 && v <= 1 }
        raise "fixture expected IDs #{@name}" unless @expected_ids.size == @draws.size && @expected_ids.all? { |v| v >= 0 && v < expected_size }
        if @role == "main"
          raise "fixture retry IDs #{@name}" unless @retry_expected_ids.size == @draws.size && @retry_expected_ids.all? { |v| v >= 0 && v < expected_size }
        end
        raise "fixture parameters #{@name}" unless @temperature == 0.8 && @top_k == 30 && @top_p == 0.9
      end
    end

    record Inputs, fixture : FixtureCase, logits : Buffer, draws : Buffer
    record Group, name : String, role : String, command_index : Int64,
      outputs : Array(Buffer), inputs : Inputs

    def self.batch(ctx : Context, input : Inputs) : Array(Buffer)
      count = input.fixture.role == "main" ? MAIN_CALLS : TOWER_CALLS
      Array(Buffer).new(count) do |index|
        retry_sample = input.fixture.role == "main" && index.odd?
        ctx.sample(input.logits, input.draws.slice(index % input.fixture.draws.size, 1),
          retry_sample ? 1.0 : input.fixture.temperature, input.fixture.top_k,
          retry_sample ? 0.95 : input.fixture.top_p)
      end
    end

    def self.pair(ctx : Context, profile : String, main : Inputs, tower : Inputs,
                  pair : Int32, warmup : Bool) : Nil
      before = ctx.graph_stats
      reads_before = ctx.readbacks.dup
      groups = [] of Group
      started = Time.instant
      order = pair.even? ? [main, tower] : [tower, main]
      order.each do |input|
        index = ctx.graph_stats["submitted"]
        outputs = batch(ctx, input)
        ctx.submit
        raise "completion wait before final drain" unless ctx.graph_stats["completion_waits"] == before["completion_waits"]
        raise "readback before final drain" unless ctx.readbacks == reads_before
        groups << Group.new(input.fixture.name, input.fixture.role, index, outputs, input)
      end
      queued = ctx.graph_stats
      raise "expected two queued command buffers" unless queued["pending"] == 2 && queued["submitted"] == before["submitted"] + 2
      # This is the only completion boundary in a measured pair. Context.read
      # below sees an empty queue; counter checks reject any additional wait.
      ctx.finish
      pair_wall_seconds = (Time.instant - started).total_seconds
      drained = ctx.graph_stats
      raise "incomplete final drain" unless drained["completion_waits"] == before["completion_waits"] + 2 &&
        drained["completed"] == before["completed"] + 2 && drained["pending"] == 0 && drained["retained_bytes"] == 0

      groups.each do |group|
        group.outputs.each_with_index do |output, index|
          actual = ctx.read(output, "terminal").first
          expected_ids = group.role == "main" && index.odd? ? group.inputs.fixture.retry_expected_ids : group.inputs.fixture.expected_ids
          expected = expected_ids[index % group.inputs.fixture.draws.size]
          raise "sample mismatch #{group.name}[#{index}]: expected #{expected}, got #{actual}" unless actual == expected.to_f32
        end
        puts({event: "sampler_benchmark_group", profile: profile, pair: pair, warmup: warmup,
              name: group.name, role: group.role, command_index: group.command_index,
              calls: group.outputs.size, vocab: group.inputs.fixture.logits.size}.to_json)
      end
      after = ctx.graph_stats
      raise "verification submitted or waited for extra work" unless after == drained
      expected_bytes = (MAIN_CALLS + TOWER_CALLS).to_i64 * 4
      raise "unexpected terminal readback" unless ctx.readbacks["terminal"] - reads_before.fetch("terminal", 0_i64) == expected_bytes
      raise "unexpected control readback" unless ctx.readbacks["control"] - reads_before.fetch("control", 0_i64) == expected_bytes
      raise "unexpected diagnostic readback" unless ctx.readbacks.fetch("diagnostic", 0_i64) == reads_before.fetch("diagnostic", 0_i64)
      puts({event: "sampler_benchmark_pair", profile: profile, pair: pair, warmup: warmup,
            order: groups.map(&.role), wall_seconds: pair_wall_seconds,
            waits_before_drain: queued["completion_waits"] - before["completion_waits"],
            final_drain_waits: after["completion_waits"] - before["completion_waits"],
            submissions: after["submitted"] - before["submitted"],
            readbacks_before_drain: 0, verified_outputs: MAIN_CALLS + TOWER_CALLS,
            graph: after}.to_json)
    end

    def self.run : Nil
      raise "usage: drama3_sampler_benchmark FIXTURE.json [--pairs=N] [--depth=2]" if ARGV.empty?
      path = ARGV.shift
      pairs, depth = 7, "2"
      ARGV.each do |argument|
        if argument.starts_with?("--pairs=")
          pairs = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--depth=")
          depth = argument.split('=', 2)[1]
        else
          raise "unknown argument #{argument}"
        end
      end
      raise "pairs must be 1..15" unless pairs.in?(1..15)
      raise "depth must be 1 or 2" unless depth.in?("1", "2")
      invocation = Random::Secure.hex(16)
      stamp = {event: "sampler_benchmark_invocation", id: invocation}.to_json
      puts stamp
      STDERR.puts stamp
      fixture = JSON.parse(File.read(path))
      raise "fixture format" unless fixture["format"].as_s == "drama3-sampler-fixture-v1"
      cases = fixture["cases"].as_a.map { |value| FixtureCase.new(value) }
      raise "empty fixture" if cases.empty?
      profiles = cases.group_by(&.profile)
      profiles.each do |name, values|
        raise "profile #{name} needs exactly one main and one tower case" unless values.size == 2 && values.map(&.role).sort == ["main", "tower"]
      end
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth
      ENV["DRAMA3_METAL_TIMELINE"] = "1"
      # Component profiling would add boundaries and perturb the schedule.
      ENV.delete("DRAMA3_PROFILE")
      ML::Audio::Drama3::Profile.configure_from_env!
      ctx = Context.new
      input_sets = profiles.transform_values do |values|
        values.map { |value| Inputs.new(value, ctx.upload(value.logits), ctx.upload(value.draws)) }
      end
      # Warm both shapes and pipeline/guard setup before measured repetitions.
      input_sets.each do |profile, inputs|
        main = inputs.find { |input| input.fixture.role == "main" }.not_nil!
        tower = inputs.find { |input| input.fixture.role == "tower" }.not_nil!
        2.times { |warmup| pair(ctx, profile, main, tower, warmup, true) }
      end
      pairs.times do |index|
        schedule = index.even? ? input_sets.to_a : input_sets.to_a.reverse
        schedule.each do |profile, inputs|
          main = inputs.find { |input| input.fixture.role == "main" }.not_nil!
          tower = inputs.find { |input| input.fixture.role == "tower" }.not_nil!
          pair(ctx, profile, main, tower, index, false)
        end
      end
      puts({event: "sampler_benchmark_summary", invocation_id: invocation, fixture: File.expand_path(path),
            profiles: profiles.keys, pairs_per_profile: pairs, warmups_per_profile: 2,
            main_calls: MAIN_CALLS, tower_calls: TOWER_CALLS, finite_guard_included: true,
            main_normal_calls: MAIN_CALLS // 2, main_retry_calls: MAIN_CALLS // 2,
            graph: ctx.graph_stats, readbacks: ctx.readbacks}.to_json)
    rescue ex
      ctx.try(&.abort)
      raise ex
    end
  end

  Drama3SamplerBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
