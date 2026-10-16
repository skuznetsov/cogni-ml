# Paired native-Q4 packed-weight load probe for actual layer-39 MoE operands.
#
# Build: crystal build examples/drama3_expert_load_bench.cr -Ddrama3_metal \
#          -o /tmp/drama3_expert_load_bench
# Run:   DRAMA3_BACKEND=metal /tmp/drama3_expert_load_bench MODEL_DIR FULL_MOE_TRACE --candidate=packed-rows --repeats=32
# The default --candidate=word-loader retains the prior aligned-word probe.
#
# The baseline calls Context#experts; candidates isolate packed-word loading or
# output-row packing. This is a kernel probe, not an end-to-end throughput claim.
require "json"
require "digest/sha256"
require "random/secure"
require "set"
require "../src/ml/audio/drama3/backbone_resident"
require "./support/drama3_moe_provenance"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  LOAD_SOURCE = {{ read_file("#{__DIR__}/support/drama3_expert_load.metal") }}

  # Diagnostic-only variants of Context#experts' affine path. The baseline
  # selector is pinned to the legacy native kernel regardless of its default.
  module ML::Audio::Drama3::Resident
    class Context
      def experts_word_load_probe(matrix : ML::Audio::Drama3::Matrix, input : Buffer, ids : Buffer,
                                  source : String, *, per_rank_input : Bool = false) : Buffer
        check_owner!(input)
        check_owner!(ids)
        raise ArgumentError.new("resident batched experts require affine weights") unless matrix.scales && matrix.biases
        expected = matrix.cols.to_i64 * (per_rank_input ? ids.count : 1)
        raise ArgumentError.new("resident batched expert input width") unless input.count == expected
        raise ArgumentError.new("resident affine groups") if matrix.cols // matrix.group_size > 8192
        count = matrix.rows.to_i64 * ids.count
        raise ArgumentError.new("resident batched expert output too large") if count > Int32::MAX

        weight = weight_buffer(matrix.tensor)
        scales_tensor, biases_tensor = matrix.scales.not_nil!, matrix.biases.not_nil!
        scales, biases = weight_buffer(scales_tensor), weight_buffer(biases_tensor)
        output = alloc(count.to_i32)
        params = [matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64,
                  weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64,
                  encoding(scales_tensor), encoding(biases_tensor),
                  (matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1).to_u64,
                  per_rank_input ? 1_u64 : 0_u64]
        enc = encoder
        enc.set_pipeline(pipeline("dr3_experts_word_load", source))
        enc.set_buffer(weight.buffer, 0)
        enc.set_buffer(scales.buffer, 1)
        enc.set_buffer(biases.buffer, 2)
        [input, output, ids].each_with_index do |buffer, index|
          enc.set_buffer(buffer.buffer, index + 3, offset: buffer.offset)
        end
        enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 6)
        enc.set_threadgroup_memory(matrix.cols // matrix.group_size * 4, 0)
        enc.dispatch_threadgroups({matrix.rows, ids.count, 1}, {64, 1, 1})
        @pending.concat([input, output, ids])
        output
      end

      # Diagnostic row-packing copy of dr3_experts. Supported shapes share one
      # 64-lane group across several rows; unsupported shapes use production's
      # exact native fallback. active_rows exists only for the untimed odd-tail
      # guard and retains matrix.rows as the checkpoint storage stride.
      def experts_packed_rows_probe(matrix : ML::Audio::Drama3::Matrix, input : Buffer, ids : Buffer,
                                    source : String, *, per_rank_input : Bool = false,
                                    active_rows : Int32? = nil) : Buffer
        check_owner!(input)
        check_owner!(ids)
        raise ArgumentError.new("resident batched experts require affine weights") unless matrix.scales && matrix.biases
        expected = matrix.cols.to_i64 * (per_rank_input ? ids.count : 1)
        raise ArgumentError.new("resident batched expert input width") unless input.count == expected
        raise ArgumentError.new("resident affine groups") if matrix.cols // matrix.group_size > 8192
        rows = active_rows || matrix.rows
        raise ArgumentError.new("resident active expert rows") unless rows > 0 && rows <= matrix.rows
        count = rows.to_i64 * ids.count
        raise ArgumentError.new("resident batched expert output too large") if count > Int32::MAX

        groups = matrix.cols // matrix.group_size
        unless matrix.group_size == 64 && groups > 0 && groups <= 32 && 64 % groups == 0
          raise ArgumentError.new("packed-row tail guard requires a supported shape") if rows != matrix.rows
          return experts(matrix, input, ids, per_rank_input, "native")
        end

        weight = weight_buffer(matrix.tensor)
        scales_tensor, biases_tensor = matrix.scales.not_nil!, matrix.biases.not_nil!
        scales, biases = weight_buffer(scales_tensor), weight_buffer(biases_tensor)
        output = alloc(count.to_i32)
        params = [matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64,
                  weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64,
                  encoding(scales_tensor), encoding(biases_tensor),
                  (matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1).to_u64,
                  per_rank_input ? 1_u64 : 0_u64, rows.to_u64]
        rows_per_threadgroup = 64 // groups
        enc = encoder
        enc.set_pipeline(pipeline("dr3_experts_packed_rows", source))
        enc.set_buffer(weight.buffer, 0)
        enc.set_buffer(scales.buffer, 1)
        enc.set_buffer(biases.buffer, 2)
        [input, output, ids].each_with_index do |buffer, index|
          enc.set_buffer(buffer.buffer, index + 3, offset: buffer.offset)
        end
        enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 6)
        enc.set_threadgroup_memory(64 * 4, 0)
        enc.dispatch_threadgroups({(rows + rows_per_threadgroup - 1) // rows_per_threadgroup, ids.count, 1}, {64, 1, 1})
        @pending.concat([input, output, ids])
        output
      end
    end
  end

  module Drama3ExpertLoadBenchmark
    include ML::Audio::Drama3

    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer

    MODEL_REPOSITORY    = "cguldogan/drama-3-mlx-q4"
    MODEL_REVISION      = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
    UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
    UPSTREAM_REVISION   = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"

    TRACE_SHA256      = "d3e6e455dd4ab0c987a500cb0d3c807db23003e383725f4b4f6c49fecb64030f"
    INPUT_SHA256      = "625d01eba26f0338a1ea2fd0938c23abc57f05ba739b2d565db791c937279e61"
    ACTIVATION_SHA256 = "8b9086fd8e4a2ab31549429513dbe482d8abb289fae70a3be4289318bfd77244"
    DOWN_SHA256       = "475ff7f7f4560f99d08bac46e89d2ea124e62a78ca18755f85fd763ab5a295db"

    WARMUPS = 2
    PAIRS   = 7
    MAX_REPEATS = 128
    ORACLE_OFFSETS = [0, 1, 3, 4, 7, 8, 11, 12]
    ENVIRONMENT = [
      "DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_METAL_TIMELINE",
      "DRAMA3_HOST_PROFILE", "DRAMA3_MEMORY_PROFILE", "DRAMA3_ALLOCATION_PROFILE",
      "DRAMA3_METAL_RESOURCES", "DRAMA3_EXPERT_THREADS", "DRAMA3_EXPERT_PACKED_ROWS", "DRAMA3_PROFILE",
    ]

    record Fixture, reference : JSON::Any, input : Vector, ids : Array(Int32), activation : Vector,
      input_sha : String, ids_sha : String, ids_upload_sha : String, activation_sha : String,
      trace_sha : String
    record Projection, name : String, matrix : Matrix, input : Buffer, per_rank_input : Bool,
      weight_offset_mod4 : Int64
    record Arm, name : String, command_index : Int64, outputs : Array(Buffer)
    record OracleBuffers, input : Buffer
    record ReceiptControl, expected : Buffer, actual : Buffer, receipt : Buffer
    record TailGuard, active_rows : Int32, receipt : Buffer

    def self.float_stage(stages : JSON::Any, name : String, shape : Array(Int32), kind : String) : Vector
      stage = stages[name]
      raise ArgumentError.new("#{name}: shape mismatch") unless stage["shape"].as_a.map(&.as_i) == shape
      raise ArgumentError.new("#{name}: dtype mismatch") unless stage["dtype"].as_s == "float32"
      raise ArgumentError.new("#{name}: capture kind mismatch") unless stage["capture_kind"].as_s == kind
      values = stage["values"].as_a.map do |value|
        raw = value.raw
        number = case raw
                 when Float64 then raw.to_f32
                 when Int64   then raw.to_f32
                 else              raise ArgumentError.new("#{name}: non-numeric value")
                 end
        raise ArgumentError.new("#{name}: non-finite value") unless number.finite?
        number
      end
      expected = shape.reduce(1) { |product, dimension| product * dimension }
      raise ArgumentError.new("#{name}: value count mismatch") unless values.size == expected
      values
    end

    def self.f32_sha(values : Vector) : String
      io = IO::Memory.new
      values.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def self.i32_sha(values : Array(Int32)) : String
      io = IO::Memory.new
      values.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
      Digest::SHA256.hexdigest(io.to_slice)
    end

    def self.load_fixture(path : String) : Fixture
      bytes = File.read(path)
      trace_sha = Digest::SHA256.hexdigest(bytes)
      raise ArgumentError.new("unexpected pinned full-trace digest") unless trace_sha == TRACE_SHA256
      reference = JSON.parse(bytes)
      raise ArgumentError.new("unexpected upstream trace format") unless reference["format"].as_s == "drama3-moe-layer-diagnostic-v1"
      raise ArgumentError.new("expected full-run upstream capture") unless reference["mode"].as_s == "full_run"
      raise ArgumentError.new("expected row 1, layer 39") unless reference["coordinate"]["row"].as_i == 1 && reference["coordinate"]["layer"].as_i == 39
      raise ArgumentError.new("upstream capture control failed") unless reference["capture_control"]["passed"].as_bool
      model = reference["model"]
      raise ArgumentError.new("model pin mismatch") unless model["repository"].as_s == MODEL_REPOSITORY && model["revision"].as_s == MODEL_REVISION
      upstream = reference["upstream"]
      raise ArgumentError.new("upstream pin mismatch") unless upstream["repository"].as_s == UPSTREAM_REPOSITORY && upstream["revision"].as_s == UPSTREAM_REVISION

      stages = reference["stages"]
      input = float_stage(stages, "normalized_input", [2048], "upstream_sparse_argument")
      raise ArgumentError.new("pinned normalized-input mismatch") unless f32_sha(input) == INPUT_SHA256
      id_stage = stages["expert_ids_argpartition_order"]
      unless id_stage["shape"].as_a.map(&.as_i) == [8] && id_stage["dtype"].as_s == "int32" && id_stage["capture_kind"].as_s == "upstream_local"
        raise ArgumentError.new("expert ID shape/dtype/capture mismatch")
      end
      ids = id_stage["values"].as_a.map(&.as_i)
      raise ArgumentError.new("invalid expert IDs") unless ids.size == 8 && ids.uniq.size == 8 && ids.all? { |id| 0 <= id < 256 }
      raise ArgumentError.new("unexpected pinned expert order") unless ids == [159, 194, 245, 154, 230, 191, 171, 108]
      id_map = reference["expert_rank_to_id"].as_a.sort_by { |entry| entry["rank"].as_i }
      unless id_map.map { |entry| entry["rank"].as_i } == (0...8).to_a && id_map.map { |entry| entry["expert_id"].as_i } == ids
        raise ArgumentError.new("rank-to-expert map mismatch")
      end
      activation = float_stage(stages, "expert_activation_outputs", [8, 512], "upstream_swiglu_return")
      down = float_stage(stages, "expert_down_outputs", [8, 2048], "upstream_switch_mlp_return")
      activation_sha = f32_sha(activation)
      raise ArgumentError.new("pinned activation mismatch") unless activation_sha == ACTIVATION_SHA256
      raise ArgumentError.new("pinned down-output mismatch") unless f32_sha(down) == DOWN_SHA256
      id_values = ids.map(&.to_f32)
      Fixture.new(reference, input, ids, activation, f32_sha(input), i32_sha(ids), f32_sha(id_values), activation_sha, trace_sha)
    end

    def self.load_matrices(model_path : String, source_hashes : Hash(String, JSON::Any)) : Tuple(Hash(String, Matrix), Drama3MoeProvenance::Record)
      store = WeightStore.new(model_path)
      provenance = Drama3MoeProvenance.verify!(model_path, store, source_hashes)
      definitions = {
        "gate" => {name: "backbone.layers.39.mlp.switch_mlp.gate_proj.weight", rows: 512, cols: 2048},
        "up" => {name: "backbone.layers.39.mlp.switch_mlp.up_proj.weight", rows: 512, cols: 2048},
        "down" => {name: "backbone.layers.39.mlp.switch_mlp.down_proj.weight", rows: 2048, cols: 512},
      }
      matrices = {} of String => Matrix
      definitions.each do |label, definition|
        matrix = store.matrix(definition[:name], 0)
        raise ArgumentError.new("#{label}: matrix shape mismatch") unless matrix.rows == definition[:rows] && matrix.cols == definition[:cols]
        unless matrix.bits == 4 && matrix.group_size == 64 && matrix.scales && matrix.biases
          raise ArgumentError.new("#{label}: expected affine Q4/group64 weights")
        end
        raise ArgumentError.new("#{label}: expected mapped checkpoint tensor") unless matrix.tensor.owner && matrix.tensor.raw_bytes
        matrices[label] = matrix
      end
      {matrices, provenance}
    end

    def self.load_identity_hashes : Hash(String, String)
      root = File.expand_path("..", __DIR__)
      {
        "benchmark_cr" => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "drama3_expert_load_bench.cr"))),
        "support_metal_compiled" => Digest::SHA256.hexdigest(LOAD_SOURCE),
        "support_metal_file" => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "support/drama3_expert_load.metal"))),
        "resident_cr" => Digest::SHA256.hexdigest(File.read(File.join(root, "src/ml/audio/drama3/resident.cr"))),
        "resident_metal" => Digest::SHA256.hexdigest(File.read(File.join(root, "src/ml/audio/drama3/kernels/resident.metal"))),
      }
    end

    def self.read_receipt!(ctx : Context, receipt : Buffer) : Array(UInt32)
      values = receipt.buffer.contents.as(UInt32*) + receipt.offset // 4
      result = Array(UInt32).new(3) { |index| values[index] }
      ctx.readbacks["terminal"] += 3_i64 * 4_i64
      result
    end

    def self.encode_projection(ctx : Context, projection : Projection, ids : Buffer,
                               repeats : Int32, candidate : Bool, candidate_mode : String) : Array(Buffer)
      Array(Buffer).new(repeats) do
        if candidate
          case candidate_mode
          when "word-loader"
            ctx.experts_word_load_probe(projection.matrix, projection.input, ids, LOAD_SOURCE,
              per_rank_input: projection.per_rank_input)
          when "packed-rows"
            ctx.experts_packed_rows_probe(projection.matrix, projection.input, ids, LOAD_SOURCE,
              per_rank_input: projection.per_rank_input)
          else
            raise ArgumentError.new("unknown candidate mode #{candidate_mode}")
          end
        else
          ctx.experts(projection.matrix, projection.input, ids,
            per_rank_input: projection.per_rank_input, reduction: "native")
        end
      end
    end

    def self.pair(ctx : Context, projection : Projection, ids : Buffer, fixture : Fixture,
                  oracle : OracleBuffers, negative : ReceiptControl, tail_guard : TailGuard?,
                  repeats : Int32, candidate_mode : String, pair_index : Int32) : Int64
      before = ctx.graph_stats.dup
      reads_before = ctx.readbacks.dup
      raise "expert load benchmark requires queue depth 2" unless before["depth"] == 2
      warmup = pair_index < WARMUPS
      order = pair_index.even? ? ["baseline", "candidate"] : ["candidate", "baseline"]
      arms = {} of String => Arm
      started = Time.instant

      order.each_with_index do |arm_name, arm_index|
        command_index = ctx.graph_stats["submitted"]
        outputs = encode_projection(ctx, projection, ids, repeats, arm_name == "candidate", candidate_mode)
        ctx.submit
        raise "completion wait before receipt command" unless ctx.graph_stats["completion_waits"] == before["completion_waits"]
        raise "readback before receipt command" unless ctx.readbacks == reads_before
        queued = ctx.graph_stats
        expected_pending = arm_index + 1
        raise "unexpected queued expert load commands" unless queued["pending"] == expected_pending && queued["submitted"] == before["submitted"] + expected_pending
        arms[arm_name] = Arm.new(arm_name, command_index, outputs)
      end

      queued = ctx.graph_stats
      raise "expected exactly two queued projection commands" unless queued["pending"] == 2 && queued["submitted"] == before["submitted"] + 2

      # This third, untimed command performs all output validation on the GPU.
      # Opening it causes the depth-2 queue's documented oldest-flight wait;
      # the final finish below drains the remaining two commands together.
      receipt = ctx.alloc(3)
      control_pair = projection.name == "gate" && pair_index == 0
      ctx.dispatch("dr3_expert_load_receipt_clear", LOAD_SOURCE, [receipt], [] of UInt64, 3)
      receipt_open = ctx.graph_stats
      unless receipt_open["submitted"] == before["submitted"] + 2 && receipt_open["completed"] == before["completed"] + 1 &&
             receipt_open["completion_waits"] == before["completion_waits"] + 1 && receipt_open["pending"] == 1
        raise "expected exactly one queue auto-wait before receipt encoding"
      end
      raise "readback before receipt command" unless ctx.readbacks == reads_before

      baseline_outputs = arms["baseline"].outputs
      candidate_outputs = arms["candidate"].outputs
      raise "#{projection.name} output repetition count mismatch" unless baseline_outputs.size == candidate_outputs.size
      elements = 0_i64
      baseline_outputs.zip(candidate_outputs).each do |expected, actual|
        raise "#{projection.name} output shape mismatch" unless expected.count == actual.count
        ctx.dispatch("dr3_expert_load_compare", LOAD_SOURCE, [expected, actual, receipt], [expected.count.to_u64], expected.count)
        elements += expected.count.to_i64
      end
      if control_pair
        ctx.dispatch("dr3_expert_load_receipt_clear", LOAD_SOURCE, [negative.receipt], [] of UInt64, 3)
        ctx.dispatch("dr3_expert_load_compare", LOAD_SOURCE,
          [negative.expected, negative.actual, negative.receipt], [2_u64], 2)
        ctx.dispatch("dr3_expert_word_load_oracle", LOAD_SOURCE,
          [oracle.input, receipt], [] of UInt64, ORACLE_OFFSETS.size)
      end

      if control_pair
        if tail = tail_guard
          raise "odd tail guard must be an odd partial row group" unless tail.active_rows.odd? && tail.active_rows < projection.matrix.rows
          ctx.dispatch("dr3_expert_load_receipt_clear", LOAD_SOURCE, [tail.receipt], [] of UInt64, 3)
          tail_output = ctx.experts_packed_rows_probe(projection.matrix, projection.input, ids, LOAD_SOURCE,
            per_rank_input: projection.per_rank_input, active_rows: tail.active_rows)
          ctx.dispatch("dr3_expert_load_compare_rows", LOAD_SOURCE,
            [baseline_outputs.first, tail_output, tail.receipt],
            [tail.active_rows.to_u64, projection.matrix.rows.to_u64, ids.count.to_u64],
            tail.active_rows * ids.count)
        end
      end

      receipt_command_index = before["submitted"] + 2
      ctx.submit
      receipt_submitted = ctx.graph_stats
      unless receipt_submitted["submitted"] == before["submitted"] + 3 && receipt_submitted["pending"] == 2 &&
             receipt_submitted["completion_waits"] == before["completion_waits"] + 1
        raise "receipt command did not submit after the single depth-2 auto-wait"
      end
      ctx.finish
      pair_wall_seconds = (Time.instant - started).total_seconds
      drained = ctx.graph_stats
      unless drained["submitted"] == before["submitted"] + 3 && drained["completed"] == before["completed"] + 3 &&
             drained["completion_waits"] == before["completion_waits"] + 3 && drained["pending"] == 0 && drained["retained_bytes"] == 0
        raise "expert load pair did not perform one auto-wait and a two-command terminal drain"
      end

      receipt_counts = read_receipt!(ctx, receipt)
      control_counts = control_pair ? read_receipt!(ctx, negative.receipt) : nil
      tail_counts : Array(UInt32)? = nil
      if control_pair
        if tail = tail_guard
          tail_counts = read_receipt!(ctx, tail.receipt)
        end
      end
      after = ctx.graph_stats
      raise "terminal receipt read submitted or waited for more work" unless after == drained
      expected_receipts = 1_i64 + (control_pair ? 1_i64 : 0_i64) + (control_pair && !tail_guard.nil? ? 1_i64 : 0_i64)
      expected_terminal_delta = 3_i64 * 4_i64 * expected_receipts
      terminal_before = reads_before.fetch("terminal", 0_i64)
      terminal_after = ctx.readbacks.fetch("terminal", 0_i64)
      raise "unexpected terminal readback bytes" unless terminal_after - terminal_before == expected_terminal_delta
      raise "unexpected control readback" unless ctx.readbacks.fetch("control", 0_i64) == reads_before.fetch("control", 0_i64)
      raise "unexpected diagnostic readback" unless ctx.readbacks.fetch("diagnostic", 0_i64) == reads_before.fetch("diagnostic", 0_i64)

      arms.each_value do |arm|
        puts({event: "drama3_expert_load_arm", projection: projection.name, pair: pair_index,
              warmup: warmup, arm: arm.name, command_index: arm.command_index,
              candidate_mode: arm.name == "candidate" ? candidate_mode : "production-native",
              repetitions: repeats, input_shape: [projection.input.count // (projection.per_rank_input ? 8 : 1)],
              input_sha256: projection.per_rank_input ? fixture.activation_sha : fixture.input_sha,
              expert_ids_stage_sha256: fixture.ids_sha, expert_ids_upload_sha256: fixture.ids_upload_sha,
              output_shape: [projection.matrix.rows, 8], per_rank_input: projection.per_rank_input,
              mapped_weight_offset_mod4: projection.weight_offset_mod4}.to_json)
      end
      puts({event: "drama3_expert_load_pair", projection: projection.name, pair: pair_index,
            warmup: warmup, order: order, command_indices: arms.transform_values(&.command_index),
            receipt_command_index: receipt_command_index,
            gpu_duration_source: "stderr drama3_gpu_command joined by command_index",
            pair_wall_seconds: pair_wall_seconds, pre_receipt_pair_waits: 0,
            readbacks_before_receipt_drain: 0, submissions: after["submitted"] - before["submitted"],
            auto_waits_before_receipt: receipt_open["completion_waits"] - before["completion_waits"],
            final_drain_waits: after["completion_waits"] - receipt_open["completion_waits"],
            exact_bitwise: receipt_counts[0] == 0 && receipt_counts[1] == 0,
            gpu_receipt: {bit_mismatches: receipt_counts[0], nonfinite_pairs: receipt_counts[1],
                          unaligned_oracle_mismatches: receipt_counts[2]},
            negative_receipt: control_counts.try { |counts| {bit_mismatches: counts[0], nonfinite_pairs: counts[1],
                                                              unaligned_oracle_mismatches: counts[2]} },
            odd_tail_guard: tail_counts.try { |counts| {candidate: "packed-rows", projection: projection.name,
                                                         active_rows: tail_guard.not_nil!.active_rows,
                                                         storage_row_stride: projection.matrix.rows,
                                                         bit_mismatches: counts[0], nonfinite_pairs: counts[1]} },
            verified_elements_per_arm: elements,
            terminal_readback_bytes: expected_terminal_delta, graph: after}.to_json)
      unless receipt_counts[0] == 0 && receipt_counts[1] == 0 && receipt_counts[2] == 0
        raise "#{projection.name} GPU receipt mismatch: #{receipt_counts}"
      end
      if counts = control_counts
        unless counts == [2_u32, 1_u32, 0_u32]
          raise "negative GPU receipt oracle failed: expected bit_mismatches=2 nonfinite_pairs=1 unaligned=0 got=#{counts}"
        end
      end
      if counts = tail_counts
        unless counts[0] == 0 && counts[1] == 0 && counts[2] == 0
          raise "odd packed-row GPU receipt mismatch: #{counts}"
        end
      end
      elements
    end

    def self.run : Nil
      unless ARGV.size.in?(2..4)
        raise "usage: drama3_expert_load_bench MODEL_DIR PINNED_FULL_MOE_TRACE [--candidate=word-loader|packed-rows] [--repeats=32]"
      end
      model_path = File.expand_path(ARGV.shift)
      trace_path = File.expand_path(ARGV.shift)
      repeats = 32
      candidate_mode = "word-loader"
      ARGV.each do |argument|
        if argument.starts_with?("--repeats=")
          repeats = argument.split('=', 2)[1].to_i
        elsif argument.starts_with?("--candidate=")
          candidate_mode = argument.split('=', 2)[1]
        else
          raise "unknown argument #{argument}"
        end
      end
      raise "candidate must be word-loader or packed-rows" unless candidate_mode.in?("word-loader", "packed-rows")
      raise "repetitions must be 1..#{MAX_REPEATS}" unless repeats.in?(1..MAX_REPEATS)

      fixture = load_fixture(trace_path)
      matrices, provenance = load_matrices(model_path, fixture.reference["source_hashes"].as_h)
      source_hashes = load_identity_hashes
      old_environment = {} of String => String?
      ENVIRONMENT.each { |name| old_environment[name] = ENV[name]? }
      context : Context? = nil
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ENV["DRAMA3_METAL_TIMELINE"] = "1"
        ENV["DRAMA3_HOST_PROFILE"] = "0"
        ENV["DRAMA3_MEMORY_PROFILE"] = "0"
        ENV["DRAMA3_ALLOCATION_PROFILE"] = "0"
        ENV["DRAMA3_METAL_RESOURCES"] = "0"
        ENV["DRAMA3_EXPERT_THREADS"] = "64"
        ENV["DRAMA3_EXPERT_PACKED_ROWS"] = "0"
        ENV.delete("DRAMA3_PROFILE")
        Profile.configure_from_env!
        context = Context.new
        ctx = context.not_nil!
        raise "expert load benchmark requires queue depth 2" unless ctx.graph_stats["depth"] == 2
        raise "expert load benchmark requires the production 64-lane width" unless ctx.expert_threads == 64
        raise "host profiling must be disabled" if ctx.host_profile.enabled?

        input = ctx.upload(fixture.input)
        activation = ctx.upload(fixture.activation)
        ids = ctx.upload(fixture.ids.map(&.to_f32))
        oracle_input = ctx.upload([
          0x3f800001_u32.unsafe_as(Float32), 0x40000003_u32.unsafe_as(Float32),
          0xbf800005_u32.unsafe_as(Float32), 0x40400007_u32.unsafe_as(Float32),
        ])
        oracle = OracleBuffers.new(oracle_input)
        negative = ReceiptControl.new(
          ctx.upload([1.0_f32, 0.0_f32]),
          ctx.upload([0x3f800001_u32.unsafe_as(Float32), 0x7fc00000_u32.unsafe_as(Float32)]),
          ctx.alloc(3))
        gate_rows = matrices["gate"].rows
        tail_guard = if candidate_mode == "packed-rows"
                       TailGuard.new(gate_rows - 1, ctx.alloc(3))
                     end

        mapped_offsets = {} of String => Hash(String, Int64)
        matrices.each do |name, matrix|
          mapped_offsets[name] = {
            "weight" => MetalBackend.tensor_buffer(matrix.tensor, true).offset,
            "scales" => MetalBackend.tensor_buffer(matrix.scales.not_nil!, true).offset,
            "biases" => MetalBackend.tensor_buffer(matrix.biases.not_nil!, true).offset,
          }
        end
        projections = [
          Projection.new("gate", matrices["gate"], input, false, mapped_offsets["gate"]["weight"] % 4),
          Projection.new("up", matrices["up"], input, false, mapped_offsets["up"]["weight"] % 4),
          Projection.new("down", matrices["down"], activation, true, mapped_offsets["down"]["weight"] % 4),
        ]

        invocation = Random::Secure.hex(16)
        puts({event: "drama3_expert_load_invocation", id: invocation, model: model_path,
              trace: trace_path, trace_sha256: fixture.trace_sha,
              model_pin: fixture.reference["model"], upstream_pin: fixture.reference["upstream"],
              model_weight_provenance: provenance, source_hashes: source_hashes,
              queue_depth: 2, launch_width: 64, repetitions: repeats,
              warmup_pairs_per_projection: WARMUPS, measured_pairs_per_projection: PAIRS,
              fixture_input_sha256: fixture.input_sha, activation_sha256: fixture.activation_sha,
              expert_ids_sha256: fixture.ids_sha, expert_ids_upload_sha256: fixture.ids_upload_sha,
              candidate_mode: candidate_mode,
              candidate: candidate_mode == "word-loader" ?
                "aligned device uint packed-word read with exact byte-assembly fallback for odd offsets" :
                "multiple output rows packed into one 64-thread group; same byte loads and native arithmetic order",
              candidate_arithmetic: "same byte loads, group-wise dequantization, sequential F32 per-group subtotal, and ascending sequential F32 group-sum order",
              packed_rows_mapping: "groups=cols/64; when groups<=32 and 64%groups==0, rows_per_threadgroup=64/groups",
              unsupported_shape_behavior: "exact Context#experts native fallback",
              odd_tail_guard: candidate_mode == "packed-rows" ?
                {projection: "gate", active_rows: gate_rows - 1, checkpoint_storage_rows: gate_rows,
                 timing: "excluded first warmup pair receipt command"} : nil,
              unaligned_oracle_offsets: ORACLE_OFFSETS, mapped_offsets: mapped_offsets,
              baseline: "Context#experts production dr3_experts",
              receipt: "device-side exact-bit/nonfinite counters; compact UInt32 counters read after the pair terminal drain",
              negative_receipt_control: "finite one-bit difference plus NaN-versus-finite pair; expected mismatch=2 nonfinite=1",
              native_vs_upstream_equality_claimed: false}.to_json)
        STDERR.puts({event: "drama3_expert_load_invocation", id: invocation}.to_json)

        verified_elements = 0_i64
        projections.each do |projection|
          (WARMUPS + PAIRS).times do |pair_index|
            verified_elements += pair(ctx, projection, ids, fixture, oracle, negative, tail_guard,
              repeats, candidate_mode, pair_index)
          end
        end

        final = ctx.graph_stats
        expected_pairs = projections.size.to_i64 * (WARMUPS + PAIRS)
        expected_commands = expected_pairs * 3_i64
        unless final["pending"] == 0 && final["retained_bytes"] == 0 && final["max_pending"] == 2 &&
               final["submitted"] == expected_commands && final["completed"] == expected_commands &&
               final["completion_waits"] == expected_commands
          raise "unexpected expert load command/wait ledger"
        end
        readbacks = {control: ctx.readbacks.fetch("control", 0_i64),
                     diagnostic: ctx.readbacks.fetch("diagnostic", 0_i64),
                     terminal: ctx.readbacks.fetch("terminal", 0_i64)}
        expected_terminal = expected_pairs * 3_i64 * 4_i64 + 3_i64 * 4_i64
        expected_terminal += 3_i64 * 4_i64 unless tail_guard.nil?
        raise "unexpected total terminal readback bytes" unless readbacks[:terminal] == expected_terminal
        raise "unexpected non-terminal readback bytes" unless readbacks[:control] == 0 && readbacks[:diagnostic] == 0
        puts({event: "drama3_expert_load_summary", invocation_id: invocation,
              model: model_path, trace_sha256: fixture.trace_sha, source_hashes: source_hashes,
              repetitions: repeats, projections: projections.map(&.name), warmup_pairs_per_projection: WARMUPS,
              measured_pairs_per_projection: PAIRS, queue_depth: 2, candidate_mode: candidate_mode,
              odd_tail_guard: unless tail_guard.nil?
                                {candidate: "packed-rows", projection: "gate",
                                 active_rows: tail_guard.not_nil!.active_rows,
                                 checkpoint_storage_rows: matrices["gate"].rows,
                                 matched_bitwise: true}
                              end,
              graph: final,
              readbacks: readbacks, verified_elements_total_per_arm: verified_elements,
              expected_terminal_bytes: expected_terminal,
              output_scope: "production-native dr3_experts versus selected diagnostic candidate; exact bitwise GPU receipt, nonfinite pairs reported separately"}.to_json)
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

  Drama3ExpertLoadBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
