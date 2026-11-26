# Compare the production native sampler against pinned MLX on common fixed
# float32 logits. This probe reads no model weights or activations.
# Build from the repository root on macOS:
#   crystal build -Ddrama3_metal examples/drama3_sampler_compare.cr \
#     --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++ -fuse-ld=/usr/bin/ld" \
#     -o /tmp/drama3_sampler_compare
# Run:
#   /tmp/drama3_sampler_compare INPUT.json OUTPUT.json
require "digest/sha256"
require "json"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3SamplerCompare
    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer

    SAMPLER_COMPARE_SOURCE = {{ read_file("#{__DIR__}/support/drama3_sampler_compare.metal") }}
    MAX_TOP_K               = 1024

    class FixtureCase
      getter name : String
      getter role : String
      getter source_arm : String
      getter step : Int32?
      getter case_kind : String
      getter logits : Array(Float32)
      getter draws : Array(Float32)
      getter noise_uniforms : Array(Array(Float32))
      getter temperature : Float64
      getter top_k : Int32
      getter top_p : Float64

      def initialize(value : JSON::Any)
        @name = value["name"].as_s
        @role = value["role"].as_s
        @source_arm = value["source_arm"]?.try(&.as_s) || "unspecified"
        step_value = value["step"]?
        @step = step_value && !step_value.raw.nil? ? step_value.as_i.to_i32 : nil
        @case_kind = value["case_kind"]?.try(&.as_s) || "trace_backed"
        @logits = value["logits"].as_a.map(&.as_f.to_f32)
        @draws = value["draws"].as_a.map(&.as_f.to_f32)
        @noise_uniforms = value["noise_uniforms"].as_a.map { |row| row.as_a.map(&.as_f.to_f32) }
        @temperature = value["temperature"].as_f
        @top_k = value["top_k"].as_i.to_i32
        @top_p = value["top_p"].as_f
        raise "empty fixture case name" if @name.empty?
        raise "empty fixture role #{@name}" if @role.empty?
        raise "fixture logits #{@name}" unless @logits.size > 0 && @logits.all?(&.finite?)
        raise "fixture draws #{@name}" unless !@draws.empty? && @draws.all? { |v| v.finite? && v >= 0_f32 && v <= 1_f32 }
        raise "fixture noise row count #{@name}" unless @noise_uniforms.size == @draws.size
        raise "fixture noise width #{@name}" unless @noise_uniforms.all? { |row| row.size == @top_k }
        raise "fixture noise values #{@name}" unless @noise_uniforms.flatten.all? { |v| v.finite? && v > 0_f32 && v < 1_f32 }
        raise "fixture parameters #{@name}" unless @temperature.finite? && @temperature >= 0 && @top_p.finite? && @top_p > 0 && @top_p <= 1
        raise "fixture top_k #{@name}" unless @top_k.in?(1..Math.min(@logits.size, MAX_TOP_K))
      end

      def output_count : Int32
        7 * top_k + 3 + draws.size
      end
    end

    record Inputs, fixture : FixtureCase, logits : Buffer, draws : Buffer,
      noise_uniforms : Buffer, diagnostics : Buffer, packed_samples : Buffer

    def self.diagnostic_params(fixture : FixtureCase) : Array(UInt64)
      [fixture.logits.size.to_u64, fixture.top_k.to_u64, fixture.draws.size.to_u64,
       Math.max(fixture.temperature, 1e-5).to_f32.unsafe_as(UInt32).to_u64,
       fixture.top_p.to_f32.unsafe_as(UInt32).to_u64]
    end

    def self.gpu_outputs(ctx : Context, input : Inputs) : Array(Float32)
      ctx.read(input.diagnostics, "diagnostic")
    end

    def self.ids(values : Array(Float32), start : Int32, count : Int32) : Array(Int32)
      Array(Int32).new(count) { |i| values[start + i].to_i32 }
    end

    def self.segment(values : Array(Float32), start : Int32, count : Int32) : Array(Float32)
      Array(Float32).new(count) { |i| values[start + i] }
    end

    def self.run : Nil
      raise "usage: drama3_sampler_compare INPUT.json OUTPUT.json" unless ARGV.size == 2
      input_path = File.expand_path(ARGV[0])
      output_path = File.expand_path(ARGV[1])
      raise "fixture is missing: #{input_path}" unless File.file?(input_path)
      raise "output already exists: #{output_path}" if File.exists?(output_path)

      fixture_bytes = File.read(input_path)
      fixture_json = JSON.parse(fixture_bytes)
      raise "fixture format" unless fixture_json["format"].as_s == "drama3-sampler-common-input-v1"
      cases = fixture_json["cases"].as_a.map { |value| FixtureCase.new(value) }
      raise "empty fixture" if cases.empty?
      raise "duplicate fixture case names" unless cases.map(&.name).uniq.size == cases.size
      raise "endpoint draws missing" unless cases.any? { |item| item.draws.includes?(0_f32) && item.draws.includes?(1_f32) }
      raise "boundary tie case missing" unless cases.any? { |item| item.case_kind == "synthetic_topk_boundary_tie" }

      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
      ENV.delete("DRAMA3_METAL_TIMELINE")
      ENV.delete("DRAMA3_PROFILE")
      ML::Audio::Drama3::Profile.configure_from_env!
      ctx = Context.new
      before = ctx.graph_stats
      reads_before = ctx.readbacks.dup
      raise "graph depth must be 2" unless before["depth"] == 2
      raise "context must start without work" unless before["submitted"] == 0 && before["completion_waits"] == 0
      raise "context must start without readbacks" unless reads_before.values.all?(&.zero?)

      # Every host input is uploaded once; all following work stays on the
      # resident queue until one terminal submission and final drain.
      inputs = cases.map do |fixture|
        logits = ctx.upload(fixture.logits)
        draws = ctx.upload(fixture.draws)
        noise = ctx.upload(fixture.noise_uniforms.flatten)
        diagnostics = ctx.alloc(fixture.output_count)
        ctx.dispatch("dr3_sampler_compare", SAMPLER_COMPARE_SOURCE,
          [logits, noise, diagnostics], diagnostic_params(fixture), 1, 1)
        samples = Array(Buffer).new(fixture.draws.size) do |draw_index|
          ctx.sample(logits, draws.slice(draw_index, 1), fixture.temperature,
            fixture.top_k, fixture.top_p)
        end
        packed_samples = ctx.alloc(fixture.draws.size)
        samples.each_with_index do |sample, draw_index|
          ctx.copy(sample, packed_samples.slice(draw_index, 1))
        end
        Inputs.new(fixture, logits, draws, noise, diagnostics, packed_samples)
      end

      raise "readback before final drain" unless ctx.readbacks == reads_before
      raise "completion wait before final drain" unless ctx.graph_stats["completion_waits"] == before["completion_waits"]
      ctx.submit
      before_drain = ctx.graph_stats
      raise "expected one queued GPU command" unless before_drain["pending"] == 1 && before_drain["submitted"] == 1
      raise "pre-drain completion wait" unless before_drain["completion_waits"] == before["completion_waits"]
      raise "pre-drain readback" unless ctx.readbacks == reads_before

      ctx.finish
      after_drain = ctx.graph_stats
      raise "final drain left pending work" unless after_drain["pending"] == 0 && after_drain["completed"] == after_drain["submitted"]
      raise "final drain did not wait exactly once" unless after_drain["completion_waits"] == before_drain["completion_waits"] + 1

      output_cases = inputs.map do |input|
        fixture = input.fixture
        values = gpu_outputs(ctx, input)
        actual_ids = ctx.read(input.packed_samples, "terminal").map do |value|
          raise "non-integral sample ID #{fixture.name}: #{value}" unless value.finite? && value == value.to_i32.to_f32
          value.to_i32
        end
        k = fixture.top_k
        offset_gumbel_ids = 7 * k + 3
        {
          name: fixture.name,
          role: fixture.role,
          source_arm: fixture.source_arm,
          step: fixture.step,
          case_kind: fixture.case_kind,
          temperature: fixture.temperature,
          top_k: fixture.top_k,
          top_p: fixture.top_p,
          draws: fixture.draws,
          native_sample_ids: actual_ids,
          native_topk_ids: ids(values, 0, k),
          native_topk_scaled_logits: segment(values, k, k),
          native_kept_mask: segment(values, 2 * k, k).map { |v| v != 0_f32 },
          native_full_probabilities: segment(values, 3 * k, k),
          native_full_cdf: segment(values, 4 * k, k),
          native_retained_conditional_probabilities: segment(values, 5 * k, k),
          native_retained_conditional_cdf: segment(values, 6 * k, k),
          native_kept_count: values[7 * k].to_i32,
          native_full_vocab_normalization: values[7 * k + 1],
          native_retained_mass: values[7 * k + 2],
          native_shared_gumbel_ids: ids(values, offset_gumbel_ids, fixture.draws.size),
        }
      end
      after_reads = ctx.graph_stats
      unless after_reads["submitted"] == after_drain["submitted"] &&
             after_reads["completed"] == after_drain["completed"] &&
             after_reads["completion_waits"] == after_drain["completion_waits"]
        raise "terminal reads added a submission or completion wait"
      end
      expected_terminal_bytes = inputs.sum { |input| input.packed_samples.count.to_i64 * 4 }
      expected_diagnostic_bytes = inputs.sum { |input| input.diagnostics.count.to_i64 * 4 }
      # Context.read checks the shared finite-status word once per read after
      # finish. There are exactly two post-drain reads per case: diagnostics
      # and packed production sample IDs. These are existing control reads,
      # not predrain synchronization or activation readbacks.
      expected_finite_status_reads = (inputs.size * 2).to_i64
      expected_control_bytes = expected_finite_status_reads * 4
      raise "terminal readback ledger mismatch" unless ctx.readbacks["terminal"] - reads_before.fetch("terminal", 0_i64) == expected_terminal_bytes
      raise "diagnostic readback ledger mismatch" unless ctx.readbacks["diagnostic"] - reads_before.fetch("diagnostic", 0_i64) == expected_diagnostic_bytes
      raise "finite-status readback ledger mismatch" unless ctx.readbacks.fetch("control", 0_i64) - reads_before.fetch("control", 0_i64) == expected_control_bytes

      File.write(output_path, {
        format: "drama3-sampler-common-output-v1",
        fixture_sha256: Digest::SHA256.hexdigest(fixture_bytes),
        resident_metal_source_sha256: Digest::SHA256.hexdigest(ML::Audio::Drama3::Resident::SOURCE),
        diagnostic_metal_source_sha256: Digest::SHA256.hexdigest(SAMPLER_COMPARE_SOURCE),
        sampler_kernel: "dr3_sample",
        diagnostic_kernel: "dr3_sampler_compare",
        cases: output_cases,
        queue: {
          graph_before_work: before,
          graph_before_final_drain: before_drain,
          graph_after_final_drain: after_drain,
          graph_after_terminal_reads: after_reads,
          readbacks_before_final_drain: reads_before,
          readbacks_after_terminal_reads: ctx.readbacks,
          submissions_before_final_drain: before_drain["submitted"] - before["submitted"],
          predrain_waits: before_drain["completion_waits"] - before["completion_waits"],
          final_drain_waits: after_drain["completion_waits"] - before_drain["completion_waits"],
          terminal_bytes: expected_terminal_bytes,
          diagnostic_bytes: expected_diagnostic_bytes,
          finite_status_reads_after_drain: expected_finite_status_reads,
          finite_status_readback_bytes: expected_control_bytes,
        },
      }.to_json)
      puts({event: "drama3_sampler_compare_complete", output: output_path,
            fixture_sha256: Digest::SHA256.hexdigest(fixture_bytes), cases: output_cases.size,
            queue: after_reads, readbacks: ctx.readbacks}.to_json)
    rescue ex
      ctx.try(&.abort)
      raise ex
    end
  end

  Drama3SamplerCompare.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
