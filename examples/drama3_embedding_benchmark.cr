# Paired benchmark of the resident tower's unseen codebook sum. It maps only
# the codebook tensor's safetensors shard and does not materialize model weights.
# Build with the repository's normal Metal flags, then run:
#   DRAMA3_BACKEND=metal /tmp/drama3_embedding_benchmark MODEL_DIR [--pairs=5] [--ops=16] [--depth=2]
# GPU timing comes from Context's existing command-completion timeline; there
# is no per-operation fence. This measures repeated fixed-input embedding sums,
# not end-to-end generation throughput.
require "json"
require "random/secure"
require "../src/ml/audio/drama3/tower_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3EmbeddingBenchmark
    include ML::Audio::Drama3

    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer

    WARMUPS = 2

    def self.load_codebook_matrix(model_path : String)
      index_path = File.join(model_path, "model.safetensors.index.json")
      shard_name = if File.file?(index_path)
                     index = JSON.parse(File.read(index_path))
                     index["weight_map"]["tower.codebook_embeddings.weight"].as_s
                   else
                     "model.safetensors"
                   end
      raise "invalid safetensors shard path" unless File.basename(shard_name) == shard_name
      owner = MappedFile.new(File.join(model_path, shard_name))
      header_size = IO::ByteFormat::LittleEndian.decode(UInt64, owner.slice(0_i64, 8_i64)).to_i64
      raise "invalid safetensors header size" unless header_size > 1 && header_size <= owner.size - 8
      header = JSON.parse(String.new(owner.slice(8_i64, header_size))).as_h
      tensor_header = header["tower.codebook_embeddings.weight"].as_h
      dtype = tensor_header["dtype"].as_s
      shape = tensor_header["shape"].as_a.map(&.as_i)
      offsets = tensor_header["data_offsets"].as_a.map(&.as_i64)
      raise "invalid codebook safetensors offsets" unless offsets.size == 2 && offsets[0] >= 0 && offsets[1] >= offsets[0]
      data_offset = 8_i64 + header_size + offsets[0]
      tensor = Tensor.new(shape, dtype, owner.slice(data_offset, offsets[1] - offsets[0]), owner)
      {owner: owner, matrix: Matrix.new(tensor), shard: shard_name, data_offset: data_offset}
    end

    def self.compare_outputs!(context : Context, expected : Array(Buffer), actual : Array(Buffer),
                              pair : Int32, ops : Int32) : Int64
      raise "output count mismatch" unless expected.size == actual.size && expected.size == ops
      verified = 0_i64
      expected.zip(actual).each_with_index do |(left, right), op|
        raise "output width mismatch" unless left.count == right.count
        lhs = left.buffer.contents.as(Float32*) + left.offset // 4
        rhs = right.buffer.contents.as(Float32*) + right.offset // 4
        left.count.times do |index|
          a, b = lhs[index], rhs[index]
          raise "non-finite benchmark output" unless a.finite? && b.finite?
          unless a.unsafe_as(UInt32) == b.unsafe_as(UInt32)
            raise "codebook sum mismatch at pair=#{pair}, op=#{op}, dimension=#{index}"
          end
        end
        verified += left.count
      end
      context.readbacks["terminal"] += verified * 2 * 4
      verified
    end

    def self.pair(context : Context, matrix : Matrix, frame : Buffer, config : TowerConfig,
                  first_codebook : Int32, end_codebook : Int32, depth : Int32,
                  pair_index : Int32, warmup : Bool, ops : Int32,
                  table_offset : Int64, shard_name : String) : Int64
      before = context.graph_stats.dup
      reads_before = context.readbacks.dup
      orders = pair_index.even? ? ["legacy", "fused"] : ["fused", "legacy"]
      outputs = {} of String => Array(Buffer)
      allocation_counts = {} of String => Int64
      command_indices = {} of String => Int64
      started = Time.instant

      orders.each_with_index do |arm, arm_index|
        command_indices[arm] = context.graph_stats["submitted"]
        before_allocations = context.allocation_stats[:allocations]
        outputs[arm] = Array(Buffer).new(ops) do
          context.codebook_sum(matrix, frame, first_codebook, end_codebook,
            config.vocab_size, config.semantic_begin, config.semantic_end,
            RESIDENT_TOWER_SOURCE, arm == "fused")
        end
        allocation_counts[arm] = context.allocation_stats[:allocations] - before_allocations
        context.submit

        raise "readback before pair drain" unless context.readbacks == reads_before
        stats = context.graph_stats
        if depth == 0
          raise "depth-0 submit missed completion" unless stats["completion_waits"] == before["completion_waits"] + arm_index + 1
        else
          raise "queued arm waited before final drain" unless stats["completion_waits"] == before["completion_waits"]
          raise "unexpected queued command count" unless stats["pending"] == arm_index + 1
        end
      end

      queued = context.graph_stats
      raise "expected two queued command buffers" if depth == 2 && queued["pending"] != 2
      context.finish
      pair_wall_seconds = (Time.instant - started).total_seconds
      drained = context.graph_stats
      raise "pair did not drain exactly two commands" unless drained["submitted"] == before["submitted"] + 2 &&
                                                             drained["completed"] == before["completed"] + 2 && drained["completion_waits"] == before["completion_waits"] + 2 &&
                                                             drained["pending"] == 0 && drained["retained_bytes"] == 0

      baseline = outputs["legacy"]
      candidate = outputs["fused"]
      verified = compare_outputs!(context, baseline, candidate, pair_index, ops)
      after = context.graph_stats
      raise "verification submitted or waited for extra work" unless after == drained
      raise "unexpected control or diagnostic readback" unless context.readbacks["control"] == reads_before.fetch("control", 0_i64) &&
                                                               context.readbacks["diagnostic"] == reads_before.fetch("diagnostic", 0_i64)
      expected_allocations = (end_codebook - first_codebook) == 0 ? ops.to_i64 : ops.to_i64 * (3 * (end_codebook - first_codebook) + 2).to_i64
      raise "legacy allocation count mismatch" unless allocation_counts["legacy"] == expected_allocations
      raise "fused allocation count mismatch" unless allocation_counts["fused"] == ops

      puts({event: "embedding_benchmark_arm", pair: pair_index, warmup: warmup,
            arm_order: orders, command_indices: command_indices, operations: ops,
            output_dim: config.dim, codebooks: [first_codebook, end_codebook],
            allocations: allocation_counts, weight_dtype: matrix.tensor.dtype,
            table_shape: matrix.tensor.shape, shard: shard_name, table_data_offset: table_offset}.to_json)
      puts({event: "embedding_benchmark_pair", pair: pair_index, warmup: warmup,
            wall_seconds: pair_wall_seconds, queue_depth: depth,
            commands: after["submitted"] - before["submitted"], final_drain_waits: after["completion_waits"] - before["completion_waits"],
            readbacks_before_drain: 0, exact_bitwise: true, verified_elements: verified,
            terminal_readback_bytes: verified * 2 * 4, graph: after}.to_json)
      verified
    end

    def self.run : Nil
      raise "usage: drama3_embedding_benchmark MODEL_DIR [--pairs=5] [--ops=16] [--depth=0|2]" if ARGV.empty?
      model_path = File.expand_path(ARGV.shift)
      pairs, ops, depth = 5, 16, 2
      ARGV.each do |argument|
        if argument.starts_with?("--pairs=")
          pairs = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--ops=")
          ops = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--depth=")
          depth = argument.split('=', 2)[1].to_i
        else
          raise "unknown argument #{argument}"
        end
      end
      raise "pairs must be 1..15" unless pairs.in?(1..15)
      raise "ops must be 1..128" unless ops.in?(1..128)
      raise "depth must be 0 or 2" unless depth == 0 || depth == 2

      config = TowerConfig.from_json(File.read(File.join(model_path, "config.json")))
      loaded = load_codebook_matrix(model_path)
      matrix = loaded[:matrix]
      raise "codebook embedding table shape mismatch" unless matrix.rows == config.num_codebooks * config.vocab_size && matrix.cols == config.dim
      raise "codebook embedding fusion requires unscaled BF16/F16/F32 weights, got #{matrix.tensor.dtype}" unless !matrix.scales && matrix.tensor.dtype.in?("BF16", "F16", "F32")

      old_backend = ENV["DRAMA3_BACKEND"]?
      old_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      old_timeline = ENV["DRAMA3_METAL_TIMELINE"]?
      old_memory_profile = ENV["DRAMA3_MEMORY_PROFILE"]?
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
      ENV["DRAMA3_METAL_TIMELINE"] = "1"
      ENV["DRAMA3_MEMORY_PROFILE"] = "0"
      context : Context? = nil
      begin
        context = Context.new
        mapped = MetalBackend.tensor_buffer(matrix.tensor, true)
        table_offset = mapped.offset
        frame_values = Array(Float32).new(config.num_codebooks + 1) do |index|
          index == 0 ? config.semantic_begin.to_f32 : ((index * 499 + 11) % config.vocab_size).to_f32
        end
        frame = context.upload(frame_values)
        first_codebook = 1 + config.vq_backbone_codebooks
        end_codebook = config.num_codebooks
        raise "invalid production unseen-codebook range" unless first_codebook <= end_codebook

        invocation = Random::Secure.hex(16)
        puts({event: "embedding_benchmark_invocation", id: invocation,
              model: model_path, dtype: matrix.tensor.dtype,
              table_shape: matrix.tensor.shape, shard: loaded[:shard],
              table_data_offset: table_offset, queue_depth: depth,
              input: "fixed-semantic-frame"}.to_json)
        verified_elements = 0_i64
        (WARMUPS + pairs).times do |index|
          verified_elements += pair(context.not_nil!, matrix, frame, config,
            first_codebook, end_codebook, depth, index, index < WARMUPS,
            ops, table_offset, loaded[:shard])
        end
        final = context.graph_stats
        raise "benchmark left work pending" unless final["pending"] == 0 && final["retained_bytes"] == 0
        readbacks = {control:    context.readbacks.fetch("control", 0_i64),
                     diagnostic: context.readbacks.fetch("diagnostic", 0_i64),
                     terminal:   context.readbacks.fetch("terminal", 0_i64)}
        puts({event: "embedding_benchmark_summary", invocation_id: invocation,
              model: model_path, dtype: matrix.tensor.dtype, table_shape: matrix.tensor.shape,
              codebooks: [first_codebook, end_codebook], operations_per_arm: ops,
              pairs: pairs, warmups: WARMUPS, queue_depth: depth,
              graph: final, readbacks: readbacks, verified_elements: verified_elements}.to_json)
      rescue ex
        context.try(&.abort)
        raise ex
      ensure
        begin
          context.try(&.finish)
        rescue
        end
        MetalBackend.reset!
        {"DRAMA3_BACKEND"                 => old_backend,
         "DRAMA3_COGNIGRAPH_MAX_INFLIGHT" => old_depth,
         "DRAMA3_METAL_TIMELINE"          => old_timeline,
         "DRAMA3_MEMORY_PROFILE"          => old_memory_profile}.each do |name, value|
          if value
            ENV[name] = value
          else
            ENV.delete(name)
          end
        end
      end
    end
  end

  Drama3EmbeddingBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
