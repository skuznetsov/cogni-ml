# Paired benchmark for the tower layer-0 feed-forward dense projections.
# This measures resident dense kernels on actual mapped checkpoint weights;
# it is not a full-model throughput benchmark.
# Build with the repository's normal Metal flags, then run:
#   DRAMA3_BACKEND=metal /tmp/drama3_tower_dense_benchmark MODEL_DIR [--candidate=simd|simd32] [--pairs=N] [--large-repeats=N]
# Each arm queues one homogeneous command buffer. Existing completion
# telemetry is emitted after the final drain; there is no per-operation fence.
require "json"
require "random/secure"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/tower"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3TowerDenseBenchmark
    include ML::Audio::Drama3

    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer

    record Projection, name : String, rows : Int32, cols : Int32, weight : String
    record Compared, elements : Int64, max_abs_error : Float64, bitwise_mismatches : Int64

    WARMUPS     =  2
    MAX_PAIRS   = 15
    MAX_REPEATS =  8
    ENVIRONMENT = [
      "DRAMA3_BACKEND",
      "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
      "DRAMA3_METAL_TIMELINE",
      "DRAMA3_DENSE_REDUCTION",
      "DRAMA3_HOST_PROFILE",
      "DRAMA3_MEMORY_PROFILE",
      "DRAMA3_ALLOCATION_PROFILE",
      "DRAMA3_PROFILE",
    ]

    def self.projections(config : TowerConfig) : Array(Projection)
      raise "tower dense benchmark expects dim=2048, got #{config.dim}" unless config.dim == 2048
      raise "tower dense benchmark expects intermediate_size=6144, got #{config.intermediate_size}" unless config.intermediate_size == 6144
      prefix = "tower.shared_layers.0.transformer_layer.feed_forward"
      [
        Projection.new("w1", config.intermediate_size, config.dim, "#{prefix}.w1.weight"),
        Projection.new("w3", config.intermediate_size, config.dim, "#{prefix}.w3.weight"),
        Projection.new("w2", config.dim, config.intermediate_size, "#{prefix}.w2.weight"),
      ]
    end

    def self.generated_input(width : Int32) : Vector
      Array(Float32).new(width) do |index|
        (((index * 17 % 101) - 50).to_f32 / 83_f32)
      end
    end

    def self.input_id(values : Vector) : String
      # Stable content identity over F32 bit patterns; this is a pairing label,
      # not a cryptographic integrity claim.
      hash = 0xcbf29ce484222325_u64
      values.each do |value|
        hash = (hash ^ value.unsafe_as(UInt32).to_u64) &* 0x100000001b3_u64
      end
      "fnv64:%016x" % hash
    end

    def self.load_matrices(model_path : String, projections : Array(Projection)) : Hash(String, Matrix)
      store = WeightStore.new(model_path)
      matrices = {} of String => Matrix
      projections.each do |projection|
        matrix = store.matrix(projection.weight)
        raise "#{projection.name} matrix shape" unless matrix.rows == projection.rows && matrix.cols == projection.cols
        raise "#{projection.name} must use dense weights without scales/biases" if matrix.scales || matrix.biases
        raise "#{projection.name} weight dtype must be BF16, got #{matrix.tensor.dtype}" unless matrix.tensor.dtype == "BF16"
        raise "#{projection.name} must come from a mapped safetensors tensor" unless matrix.tensor.owner && matrix.tensor.raw_bytes
        expected_weight_bytes = projection.rows.to_i64 * projection.cols * 2
        raise "#{projection.name} mapped weight byte count" unless matrix.tensor.raw_bytes.not_nil!.size.to_i64 == expected_weight_bytes
        matrices[projection.name] = matrix
      end
      matrices
    end

    def self.compare_outputs(ctx : Context, expected : Array(Buffer), actual : Array(Buffer),
                             projection : Projection) : Compared
      raise "#{projection.name} output count mismatch" unless expected.size == actual.size
      verified = 0_i64
      max_abs_error = 0.0
      bitwise_mismatches = 0_i64
      expected.zip(actual).each_with_index do |(left, right), repeat|
        raise "#{projection.name} output width mismatch" unless left.count == right.count && left.count == projection.rows
        # The pair has already been drained. Read shared contents directly
        # rather than calling Context#read, which would invoke another finish.
        lhs = left.buffer.contents.as(Float32*) + left.offset // 4
        rhs = right.buffer.contents.as(Float32*) + right.offset // 4
        left.count.times do |index|
          a, b = lhs[index], rhs[index]
          raise "#{projection.name} non-finite terminal output" unless a.finite? && b.finite?
          max_abs_error = Math.max(max_abs_error, (a.to_f64 - b.to_f64).abs)
          bitwise_mismatches += 1 if a.unsafe_as(UInt32) != b.unsafe_as(UInt32)
        end
        verified += left.count
      end
      # The outputs are only inspected after the existing final pair drain.
      ctx.readbacks["terminal"] += verified * 2 * 4
      Compared.new(verified, max_abs_error, bitwise_mismatches)
    end

    def self.pair(ctx : Context, matrix : Matrix, input : Buffer, input_identity : String,
                  projection : Projection, repetitions : Int32, pair_index : Int32,
                  warmup : Bool, mapped_offset : Int64, candidate : String) : Int64
      before = ctx.graph_stats.dup
      reads_before = ctx.readbacks.dup
      raise "tower dense benchmark requires queue depth 2" unless before["depth"] == 2
      order = pair_index.even? ? ["tree", candidate] : [candidate, "tree"]
      outputs = {} of String => Array(Buffer)
      command_indices = {} of String => Int64
      allocation_counts = {} of String => Int64
      started = Time.instant

      order.each_with_index do |mode, arm_index|
        command_indices[mode] = ctx.graph_stats["submitted"]
        before_allocations = ctx.allocation_stats[:allocations]
        outputs[mode] = Array(Buffer).new(repetitions) do
          ctx.matvec(matrix, input, dense_reduction: mode)
        end
        allocation_counts[mode] = ctx.allocation_stats[:allocations] - before_allocations
        ctx.submit

        raise "completion wait before final drain" unless ctx.graph_stats["completion_waits"] == before["completion_waits"]
        raise "readback before final drain" unless ctx.readbacks == reads_before
        stats = ctx.graph_stats
        raise "unexpected queued command count" unless stats["pending"] == arm_index + 1 && stats["submitted"] == before["submitted"] + arm_index + 1
      end

      queued = ctx.graph_stats
      raise "expected two queued command buffers" unless queued["pending"] == 2 && queued["submitted"] == before["submitted"] + 2
      ctx.finish
      pair_wall_seconds = (Time.instant - started).total_seconds
      drained = ctx.graph_stats
      raise "pair did not drain exactly two commands" unless drained["submitted"] == before["submitted"] + 2 &&
                                                             drained["completed"] == before["completed"] + 2 &&
                                                             drained["completion_waits"] == before["completion_waits"] + 2 &&
                                                             drained["pending"] == 0 && drained["retained_bytes"] == 0

      compared = compare_outputs(ctx, outputs["tree"], outputs[candidate], projection)
      after = ctx.graph_stats
      raise "verification submitted or waited for extra work" unless after == drained
      expected_bytes = compared.elements * 2 * 4
      terminal_before = reads_before.fetch("terminal", 0_i64)
      terminal_after = ctx.readbacks.fetch("terminal", 0_i64)
      raise "unexpected terminal readback ledger" unless terminal_after - terminal_before == expected_bytes
      raise "unexpected control readback" unless ctx.readbacks.fetch("control", 0_i64) == reads_before.fetch("control", 0_i64)
      raise "unexpected diagnostic readback" unless ctx.readbacks.fetch("diagnostic", 0_i64) == reads_before.fetch("diagnostic", 0_i64)

      order.each do |mode|
        pipeline = mode == "tree" ? "dr3_dense" : (mode == "simd32" ? "dr3_dense_simd32" : "dr3_dense_simd")
        puts({event: "tower_dense_benchmark_arm", projection: projection.name,
              pair: pair_index, warmup: warmup, mode: mode, requested_pipeline: pipeline,
              command_index: command_indices[mode], repetitions: repetitions,
              shape: [projection.rows, projection.cols], weight_dtype: "BF16",
              weight_mapping: "safetensors-mmap-no-copy", weight_offset: mapped_offset,
              input_width: input.count, input_id: input_identity,
              buffer_allocations: allocation_counts[mode]}.to_json)
      end
      puts({event: "tower_dense_benchmark_pair", projection: projection.name,
            pair: pair_index, warmup: warmup, order: order,
            wall_seconds: pair_wall_seconds,
            waits_before_drain: queued["completion_waits"] - before["completion_waits"],
            readbacks_before_drain: 0, submissions: after["submitted"] - before["submitted"],
            final_drain_waits: after["completion_waits"] - before["completion_waits"],
            exact_bitwise: compared.bitwise_mismatches == 0,
            bitwise_mismatches: compared.bitwise_mismatches,
            max_abs_error: compared.max_abs_error, verified_elements: compared.elements,
            terminal_readback_bytes: expected_bytes, graph: after}.to_json)
      raise "#{projection.name} tree/#{candidate} bitwise mismatch" unless compared.bitwise_mismatches == 0
      compared.elements
    end

    def self.run : Nil
      raise "usage: drama3_tower_dense_benchmark MODEL_DIR [--candidate=simd|simd32] [--pairs=N] [--large-repeats=N]" if ARGV.empty?
      model_path = File.expand_path(ARGV.shift)
      pairs, repetitions = 7, 8
      candidate = "simd"
      ARGV.each do |argument|
        if argument.starts_with?("--candidate=")
          candidate = argument.split('=', 2)[1]
        elsif argument.starts_with?("--pairs=")
          pairs = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--large-repeats=")
          repetitions = argument.split('=', 2)[1].to_i
        else
          raise "unknown argument #{argument}"
        end
      end
      raise "pairs must be 1..#{MAX_PAIRS}" unless pairs.in?(1..MAX_PAIRS)
      raise "large repeats must be 1..#{MAX_REPEATS}" unless repetitions.in?(1..MAX_REPEATS)
      raise "candidate must be simd or simd32" unless candidate.in?("simd", "simd32")

      config = TowerConfig.from_json(File.read(File.join(model_path, "config.json")))
      selected = projections(config)
      matrices = load_matrices(model_path, selected)
      old_environment = {} of String => String?
      ENVIRONMENT.each { |name| old_environment[name] = ENV[name]? }
      context : Context? = nil
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ENV["DRAMA3_METAL_TIMELINE"] = "1"
        ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
        ENV["DRAMA3_HOST_PROFILE"] = "0"
        ENV["DRAMA3_MEMORY_PROFILE"] = "0"
        ENV["DRAMA3_ALLOCATION_PROFILE"] = "0"
        ENV.delete("DRAMA3_PROFILE")
        Profile.configure_from_env!
        context = Context.new

        # Build each distinct input once and upload it before any benchmark
        # pairs. Both reductions reuse the exact same device-resident input.
        inputs = {} of Int32 => Buffer
        input_ids = {} of Int32 => String
        [config.dim, config.intermediate_size].uniq.each do |width|
          values = generated_input(width)
          inputs[width] = context.not_nil!.upload(values)
          input_ids[width] = input_id(values)
        end

        mapped_offsets = {} of String => Int64
        selected.each do |projection|
          # Materialize the existing no-copy Metal view before timed pairs.
          # This binds the tensor's mapped shard and does not submit or wait.
          view = MetalBackend.tensor_buffer(matrices[projection.name].tensor, true)
          raise "#{projection.name} mapped Metal offset" unless view.offset >= 0
          mapped_offsets[projection.name] = view.offset
        end
        invocation = Random::Secure.hex(16)
        puts({event: "tower_dense_benchmark_invocation", id: invocation,
              model: model_path, queue_depth: 2, dense_modes: ["tree", candidate],
              requested_pipelines: {tree: "dr3_dense", candidate: candidate == "simd32" ? "dr3_dense_simd32" : "dr3_dense_simd"},
              weight_dtype: "BF16", weight_mapping: "safetensors-mmap-no-copy",
              input_residency: "one-upload-per-distinct-width"}.to_json)
        STDERR.puts({event: "tower_dense_benchmark_invocation", id: invocation}.to_json)

        verified_elements = 0_i64
        selected.each do |projection|
          input = inputs[projection.cols]
          identity = input_ids[projection.cols]
          (WARMUPS + pairs).times do |index|
            warmup = index < WARMUPS
            verified_elements += pair(context.not_nil!, matrices[projection.name], input, identity,
              projection, repetitions, index, warmup, mapped_offsets[projection.name], candidate)
          end
        end

        final = context.not_nil!.graph_stats
        raise "benchmark left work pending" unless final["pending"] == 0 && final["retained_bytes"] == 0
        readbacks = {control:    context.not_nil!.readbacks.fetch("control", 0_i64),
                     diagnostic: context.not_nil!.readbacks.fetch("diagnostic", 0_i64),
                     terminal:   context.not_nil!.readbacks.fetch("terminal", 0_i64)}
        puts({event: "tower_dense_benchmark_summary", invocation_id: invocation,
              model: model_path, tower_dim: config.dim, intermediate_size: config.intermediate_size,
              candidate_mode: candidate, projections: selected.map(&.name),
              shapes: selected.to_h { |projection| {projection.name, [projection.rows, projection.cols]} },
              pairs_per_projection: pairs, warmups_per_projection: WARMUPS,
              large_repeats: repetitions, queue_depth: 2,
              weight_dtypes: selected.to_h { |projection| {projection.name, "BF16"} },
              graph: final, allocation_stats: context.not_nil!.allocation_stats,
              readbacks: readbacks, verified_elements: verified_elements}.to_json)
      rescue ex
        context.try(&.abort)
        raise ex
      ensure
        begin
          context.try(&.finish)
        rescue
        end
        MetalBackend.reset!
        old_environment.each do |name, value|
          if value
            ENV[name] = value.not_nil!
          else
            ENV.delete(name)
          end
        end
        Profile.configure_from_env!
      end
    end
  end

  Drama3TowerDenseBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
