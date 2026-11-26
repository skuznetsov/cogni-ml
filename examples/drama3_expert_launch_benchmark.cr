# Paired 32/64-thread launch probe for the native affine-Q4 expert kernel.
#
# The experiment changes only the requested threadgroup width. It uses the
# pinned layer-39 upstream activation fixture and actual checkpoint weights;
# it does not claim that native MoE outputs match the upstream implementation.
# Each arm queues one homogeneous command buffer, then the pair drains once.
require "json"
require "digest/sha256"
require "random/secure"
require "../src/ml/audio/drama3/backbone_resident"
require "./support/drama3_moe_provenance"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  # Diagnostic-only reopening: this is a copy of Context#experts' native path,
  # with a test-only launch width. The resident kernel and production selector
  # remain untouched. The fixed g += 64 loop requires groups <= 32 for a
  # 32-thread launch; 64 threads retain the original multi-iteration partition.
  module ML::Audio::Drama3::Resident
    class Context
      def experts_launch_probe(matrix : ML::Audio::Drama3::Matrix, x : Buffer, ids : Buffer,
                               *, per_rank_input : Bool = false, thread_width : Int32) : Buffer
        check_owner!(x)
        check_owner!(ids)
        raise ArgumentError.new("expert launch width must be 32 or 64") unless thread_width == 32 || thread_width == 64
        raise ArgumentError.new("resident batched experts require affine weights") unless matrix.scales && matrix.biases
        expected = matrix.cols.to_i64 * (per_rank_input ? ids.count : 1)
        raise ArgumentError.new("resident batched expert input width") unless x.count == expected
        groups = matrix.cols // matrix.group_size
        raise ArgumentError.new("resident affine groups") if groups > 8192
        raise ArgumentError.new("32-thread expert launch requires at most 32 groups") if thread_width == 32 && groups > 32
        count = matrix.rows.to_i64 * ids.count
        raise ArgumentError.new("resident batched expert output too large") if count > Int32::MAX

        weight = MetalBackend.tensor_buffer(matrix.tensor, true)
        scales_tensor, biases_tensor = matrix.scales.not_nil!, matrix.biases.not_nil!
        scales, biases = MetalBackend.tensor_buffer(scales_tensor, true), MetalBackend.tensor_buffer(biases_tensor, true)
        output = alloc(count.to_i32)
        params = [matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64,
                  weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64,
                  encoding(scales_tensor), encoding(biases_tensor),
                  (matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1).to_u64,
                  per_rank_input ? 1_u64 : 0_u64]
        enc = encoder
        enc.set_pipeline(pipeline("dr3_experts", SOURCE))
        enc.set_buffer(weight.buffer, 0)
        enc.set_buffer(scales.buffer, 1)
        enc.set_buffer(biases.buffer, 2)
        [x, output, ids].each_with_index { |buffer, index| enc.set_buffer(buffer.buffer, index + 3, offset: buffer.offset) }
        enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 6)
        enc.set_threadgroup_memory(groups * 4, 0)
        enc.dispatch_threadgroups({matrix.rows, ids.count, 1}, {thread_width, 1, 1})
        @pending.concat([x, output, ids])
        output
      end
    end
  end

  module Drama3ExpertLaunchBenchmark
    include ML::Audio::Drama3

    alias Context = ML::Audio::Drama3::Resident::Context
    alias Buffer = ML::Audio::Drama3::Resident::Buffer

    record StageBuffers, gate : Buffer, up : Buffer, down : Buffer
    record Matrices, gate : Matrix, up : Matrix, down : Matrix
    record Compared, elements_per_arm : Int64, bitwise_mismatches : Int64

    TRACE_SHA256        = "d3e6e455dd4ab0c987a500cb0d3c807db23003e383725f4b4f6c49fecb64030f"
    INPUT_SHA256        = "625d01eba26f0338a1ea2fd0938c23abc57f05ba739b2d565db791c937279e61"
    ACTIVATION_SHA256   = "8b9086fd8e4a2ab31549429513dbe482d8abb289fae70a3be4289318bfd77244"
    DOWN_SHA256         = "475ff7f7f4560f99d08bac46e89d2ea124e62a78ca18755f85fd763ab5a295db"
    MODEL_REPOSITORY    = "cguldogan/drama-3-mlx-q4"
    MODEL_REVISION      = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
    UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
    UPSTREAM_REVISION   = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"

    WARMUPS     = 2
    PAIRS       = 7
    MAX_REPEATS = 8

    ENVIRONMENT = [
      "DRAMA3_BACKEND",
      "DRAMA3_COGNIGRAPH_MAX_INFLIGHT",
      "DRAMA3_METAL_TIMELINE",
      "DRAMA3_HOST_PROFILE",
      "DRAMA3_MEMORY_PROFILE",
      "DRAMA3_ALLOCATION_PROFILE",
      "DRAMA3_EXPERT_THREADS",
      "DRAMA3_PROFILE",
    ]

    def self.stage_vector(stages : JSON::Any, name : String, shape : Array(Int32), kind : String) : Vector
      stage = stages[name]
      raise ArgumentError.new("#{name}: shape mismatch") unless stage["shape"].as_a.map(&.as_i) == shape
      raise ArgumentError.new("#{name}: dtype mismatch") unless stage["dtype"].as_s == "float32"
      raise ArgumentError.new("#{name}: capture kind mismatch") unless stage["capture_kind"].as_s == kind
      values = stage["values"].as_a.map do |value|
        number = case raw = value.raw
                 when Float64 then raw.to_f32
                 when Int64   then raw.to_f32
                 else              raise ArgumentError.new("#{name}: non-numeric value")
                 end
        raise ArgumentError.new("#{name}: non-finite value") unless number.finite?
        number
      end
      expected_count = shape.reduce(1) { |product, dimension| product * dimension }
      raise ArgumentError.new("#{name}: value count mismatch") unless values.size == expected_count
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

    def self.load_capture(path : String) : Tuple(JSON::Any, Vector, Array(Int32), Vector, String, String, String, String)
      bytes = File.read(path)
      raise ArgumentError.new("unexpected pinned full trace digest") unless Digest::SHA256.hexdigest(bytes) == TRACE_SHA256
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
      input = stage_vector(stages, "normalized_input", [2048], "upstream_sparse_argument")
      input_sha = f32_sha(input)
      raise ArgumentError.new("pinned normalized-input mismatch") unless input_sha == INPUT_SHA256

      id_stage = stages["expert_ids_argpartition_order"]
      raise ArgumentError.new("expert ID shape/dtype/capture mismatch") unless id_stage["shape"].as_a.map(&.as_i) == [8] && id_stage["dtype"].as_s == "int32" && id_stage["capture_kind"].as_s == "upstream_local"
      ids = id_stage["values"].as_a.map(&.as_i)
      raise ArgumentError.new("invalid expert IDs") unless ids.size == 8 && ids.uniq.size == 8 && ids.all? { |id| 0 <= id < 256 }
      raise ArgumentError.new("unexpected pinned expert order") unless ids == [159, 194, 245, 154, 230, 191, 171, 108]
      ids_sha = i32_sha(ids)
      rank_map = reference["expert_rank_to_id"].as_a.sort_by { |entry| entry["rank"].as_i }
      raise ArgumentError.new("rank-to-expert map mismatch") unless rank_map.map { |entry| entry["rank"].as_i } == (0...8).to_a && rank_map.map { |entry| entry["expert_id"].as_i } == ids

      activation = stage_vector(stages, "expert_activation_outputs", [8, 512], "upstream_swiglu_return")
      down = stage_vector(stages, "expert_down_outputs", [8, 2048], "upstream_switch_mlp_return")
      activation_sha = f32_sha(activation)
      raise ArgumentError.new("pinned SwiGLU activation mismatch") unless activation_sha == ACTIVATION_SHA256
      raise ArgumentError.new("pinned upstream down-output mismatch") unless f32_sha(down) == DOWN_SHA256
      {reference, input, ids, activation, input_sha, ids_sha, activation_sha, Digest::SHA256.hexdigest(bytes)}
    end

    def self.load_matrices(model_path : String, source_hashes : Hash(String, JSON::Any)) : Tuple(Matrices, Drama3MoeProvenance::Record)
      store = WeightStore.new(model_path)
      provenance = Drama3MoeProvenance.verify!(model_path, store, source_hashes)
      names = {
        gate: "backbone.layers.39.mlp.switch_mlp.gate_proj.weight",
        up:   "backbone.layers.39.mlp.switch_mlp.up_proj.weight",
        down: "backbone.layers.39.mlp.switch_mlp.down_proj.weight",
      }
      gate = store.matrix(names[:gate], 0)
      up = store.matrix(names[:up], 0)
      down = store.matrix(names[:down], 0)
      [
        {gate, {rows: 512, cols: 2048}, "gate"},
        {up, {rows: 512, cols: 2048}, "up"},
        {down, {rows: 2048, cols: 512}, "down"},
      ].each do |matrix, shape, label|
        raise ArgumentError.new("#{label}: matrix shape mismatch") unless matrix.rows == shape[:rows] && matrix.cols == shape[:cols]
        raise ArgumentError.new("#{label}: expected affine Q4/group64 weights") unless matrix.bits == 4 && matrix.group_size == 64 && matrix.scales && matrix.biases
        raise ArgumentError.new("#{label}: expected mapped checkpoint tensor") unless matrix.tensor.owner && matrix.tensor.raw_bytes
      end
      {Matrices.new(gate, up, down), provenance}
    end

    def self.encode_arm(ctx : Context, matrices : Matrices, input : Buffer, activation : Buffer, ids : Buffer,
                        thread_width : Int32, repetitions : Int32) : Array(StageBuffers)
      Array(StageBuffers).new(repetitions) do
        gate = ctx.experts_launch_probe(matrices.gate, input, ids, thread_width: thread_width)
        up = ctx.experts_launch_probe(matrices.up, input, ids, thread_width: thread_width)
        down = ctx.experts_launch_probe(matrices.down, activation, ids, per_rank_input: true, thread_width: thread_width)
        StageBuffers.new(gate, up, down)
      end
    end

    def self.compare_outputs(ctx : Context, left : Array(StageBuffers), right : Array(StageBuffers)) : Compared
      raise "expert launch repetition count mismatch" unless left.size == right.size
      elements = 0_i64
      mismatches = 0_i64
      fields = ["gate", "up", "down"]
      left.zip(right).each do |(lhs_stages, rhs_stages)|
        fields.each do |field|
          lhs_buffer = case field
                       when "gate" then lhs_stages.gate
                       when "up"   then lhs_stages.up
                       else             lhs_stages.down
                       end
          rhs_buffer = case field
                       when "gate" then rhs_stages.gate
                       when "up"   then rhs_stages.up
                       else             rhs_stages.down
                       end
          raise "#{field} output width mismatch" unless lhs_buffer.count == rhs_buffer.count
          lhs = lhs_buffer.buffer.contents.as(Float32*) + lhs_buffer.offset // 4
          rhs = rhs_buffer.buffer.contents.as(Float32*) + rhs_buffer.offset // 4
          lhs_buffer.count.times do |index|
            a, b = lhs[index], rhs[index]
            raise "#{field} output contains non-finite value" unless a.finite? && b.finite?
            mismatches += 1_i64 if a.unsafe_as(UInt32) != b.unsafe_as(UInt32)
          end
          elements += lhs_buffer.count.to_i64
        end
      end
      ctx.readbacks["terminal"] += elements * 2_i64 * 4_i64
      Compared.new(elements, mismatches)
    end

    def self.pair(ctx : Context, matrices : Matrices, input : Buffer, ids : Buffer,
                  activation : Buffer, input_sha : String, activation_sha : String,
                  ids_stage_sha : String, ids_upload_sha : String,
                  repetitions : Int32, pair_index : Int32,
                  warmup : Bool) : Int64
      before = ctx.graph_stats.dup
      reads_before = ctx.readbacks.dup
      raise "expert launch benchmark requires queue depth 2" unless before["depth"] == 2
      order = pair_index.even? ? [32, 64] : [64, 32]
      arms = {} of Int32 => Array(StageBuffers)
      command_indices = {} of Int32 => Int64
      started = Time.instant

      order.each_with_index do |thread_width, arm_index|
        command_indices[thread_width] = ctx.graph_stats["submitted"]
        arms[thread_width] = encode_arm(ctx, matrices, input, activation, ids, thread_width, repetitions)
        ctx.submit

        raise "completion wait before pair drain" unless ctx.graph_stats["completion_waits"] == before["completion_waits"]
        raise "readback before pair drain" unless ctx.readbacks == reads_before
        queued = ctx.graph_stats
        expected_pending = arm_index + 1
        raise "unexpected queued expert launch commands" unless queued["pending"] == expected_pending && queued["submitted"] == before["submitted"] + expected_pending
      end

      queued = ctx.graph_stats
      raise "expected exactly two queued expert launch commands" unless queued["pending"] == 2 && queued["submitted"] == before["submitted"] + 2
      ctx.finish
      pair_wall_seconds = (Time.instant - started).total_seconds
      drained = ctx.graph_stats
      raise "expert launch pair did not drain exactly two commands" unless drained["submitted"] == before["submitted"] + 2 &&
                                                                           drained["completed"] == before["completed"] + 2 &&
                                                                           drained["completion_waits"] == before["completion_waits"] + 2 &&
                                                                           drained["pending"] == 0 && drained["retained_bytes"] == 0

      compared = compare_outputs(ctx, arms[32], arms[64])
      after = ctx.graph_stats
      raise "terminal comparison submitted or waited for more work" unless after == drained
      per_rep_elements = (matrices.gate.rows + matrices.up.rows + matrices.down.rows).to_i64 * ids.count
      expected_elements_per_arm = per_rep_elements * repetitions
      raise "unexpected compared expert output count" unless compared.elements_per_arm == expected_elements_per_arm
      expected_readback_delta = expected_elements_per_arm * 2_i64 * 4_i64
      raise "unexpected terminal readback ledger" unless ctx.readbacks.fetch("terminal", 0_i64) - reads_before.fetch("terminal", 0_i64) == expected_readback_delta
      raise "unexpected control readback" unless ctx.readbacks.fetch("control", 0_i64) == reads_before.fetch("control", 0_i64)
      raise "unexpected diagnostic readback" unless ctx.readbacks.fetch("diagnostic", 0_i64) == reads_before.fetch("diagnostic", 0_i64)

      order.each do |thread_width|
        puts({event: "drama3_expert_launch_arm", pair: pair_index, warmup: warmup,
              thread_width: thread_width, command_index: command_indices[thread_width],
              repetitions: repetitions, input_shape: [2048], expert_ids_shape: [8],
              gate_shape: [8, matrices.gate.rows], up_shape: [8, matrices.up.rows],
              down_input_shape: [8, matrices.down.cols],
              down_shape: [8, matrices.down.rows], gate_groups: matrices.gate.cols // matrices.gate.group_size,
              up_groups: matrices.up.cols // matrices.up.group_size, down_groups: matrices.down.cols // matrices.down.group_size,
              input_sha256: input_sha, activation_sha256: activation_sha,
              expert_ids_stage_sha256: ids_stage_sha,
              expert_ids_upload_sha256: ids_upload_sha}.to_json)
      end
      puts({event: "drama3_expert_launch_pair", pair: pair_index, warmup: warmup,
            order: order, pair_wall_scope: "both_arms_plus_terminal_drain",
            wall_seconds: pair_wall_seconds, waits_before_drain: queued["completion_waits"] - before["completion_waits"],
            readbacks_before_drain: 0, submissions: after["submitted"] - before["submitted"],
            final_drain_waits: after["completion_waits"] - before["completion_waits"],
            exact_bitwise: compared.bitwise_mismatches == 0,
            bitwise_mismatches: compared.bitwise_mismatches,
            verified_elements_per_arm: compared.elements_per_arm,
            terminal_readback_bytes: expected_readback_delta, graph: after}.to_json)
      raise "native 32/64 expert outputs differ bitwise" unless compared.bitwise_mismatches == 0
      compared.elements_per_arm
    end

    def self.run : Nil
      raise "usage: drama3_expert_launch_benchmark MODEL_DIR PINNED_FULL_MOE_TRACE [--repeats=1..8]" unless ARGV.size >= 2 && ARGV.size <= 3
      model_path = File.expand_path(ARGV.shift)
      trace_path = File.expand_path(ARGV.shift)
      repetitions = 8
      if argument = ARGV.shift?
        raise "unknown argument #{argument}" unless argument.starts_with?("--repeats=")
        repetitions = argument.split('=', 2)[1].to_i
      end
      raise "repeats must be 1..#{MAX_REPEATS}" unless repetitions.in?(1..MAX_REPEATS)

      reference, input_values, expert_ids, activation_values, input_sha, ids_sha, activation_sha, trace_sha = load_capture(trace_path)
      matrices, provenance = load_matrices(model_path, reference["source_hashes"].as_h)
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
        ENV["DRAMA3_EXPERT_THREADS"] = "64"
        ENV.delete("DRAMA3_PROFILE")
        Profile.configure_from_env!
        context = Context.new
        ctx = context.not_nil!
        raise "expert launch benchmark requires graph depth 2" unless ctx.graph_stats["depth"] == 2
        raise "host profiling must be disabled" if ctx.host_profile.enabled?

        # The fixture values and expert IDs are each uploaded once, before any
        # timed pair. All projection outputs remain device-resident until the
        # pair's existing depth-2 drain has completed.
        input = ctx.upload(input_values)
        expert_id_values = expert_ids.map(&.to_f32)
        expert_id_upload_sha = f32_sha(expert_id_values)
        ids = ctx.upload(expert_id_values)
        activation = ctx.upload(activation_values)
        mapped_offsets = {} of String => Int64
        {"gate" => matrices.gate, "up" => matrices.up, "down" => matrices.down}.each do |name, matrix|
          [matrix.tensor, matrix.scales.not_nil!, matrix.biases.not_nil!].each do |tensor|
            view = MetalBackend.tensor_buffer(tensor, true)
            raise "#{name} mapped Metal view offset" unless view.offset >= 0
          end
          mapped_offsets[name] = MetalBackend.tensor_buffer(matrix.tensor, true).offset
        end

        invocation = Random::Secure.hex(16)
        puts({event: "drama3_expert_launch_invocation", id: invocation,
              model: model_path, trace: trace_path, trace_sha256: trace_sha,
              model_pin: reference["model"], upstream_pin: reference["upstream"],
              model_weight_provenance: provenance, queue_depth: 2,
              launch_widths: [32, 64], repetitions: repetitions,
              warmup_pairs: WARMUPS, measured_pairs: PAIRS,
              input_sha256: input_sha, expert_ids_stage_sha256: ids_sha,
              expert_ids_upload_sha256: expert_id_upload_sha,
              reference_activation_sha256: activation_sha,
              reference_down_sha256: DOWN_SHA256,
              matrices: {gate: [matrices.gate.rows, matrices.gate.cols],
                         up: [matrices.up.rows, matrices.up.cols],
                         down: [matrices.down.rows, matrices.down.cols]},
              groups: {gate: matrices.gate.cols // matrices.gate.group_size,
                       up: matrices.up.cols // matrices.up.group_size,
                       down: matrices.down.cols // matrices.down.group_size},
              mapped_weight_offsets: mapped_offsets,
              native_output_reference_equality_claimed: false}.to_json)
        STDERR.puts({event: "drama3_expert_launch_invocation", id: invocation}.to_json)

        verified_elements = 0_i64
        (WARMUPS + PAIRS).times do |pair_index|
          verified_elements += pair(ctx, matrices, input, ids, activation, input_sha, activation_sha,
            ids_sha, expert_id_upload_sha,
            repetitions, pair_index, pair_index < WARMUPS)
        end

        final = ctx.graph_stats
        raise "expert launch benchmark left work pending" unless final["pending"] == 0 && final["retained_bytes"] == 0
        raise "expected depth-2 queue saturation" unless final["max_pending"] == 2
        expected_commands = (WARMUPS + PAIRS) * 2
        raise "unexpected submission/completion/wait ledger" unless final["submitted"] == expected_commands &&
                                                                    final["completed"] == expected_commands && final["completion_waits"] == expected_commands
        readbacks = {control:    ctx.readbacks.fetch("control", 0_i64),
                     diagnostic: ctx.readbacks.fetch("diagnostic", 0_i64),
                     terminal:   ctx.readbacks.fetch("terminal", 0_i64)}
        expected_terminal_bytes = verified_elements * 2_i64 * 4_i64
        raise "unexpected total terminal readback ledger" unless readbacks[:terminal] == expected_terminal_bytes
        raise "unexpected non-terminal readback" unless readbacks[:control] == 0 && readbacks[:diagnostic] == 0
        puts({event: "drama3_expert_launch_summary", invocation_id: invocation,
              model: model_path, trace_sha256: trace_sha,
              repetitions: repetitions, warmup_pairs: WARMUPS, measured_pairs: PAIRS,
              queue_depth: 2, graph: final, allocation_stats: ctx.allocation_stats,
              readbacks: readbacks, verified_elements_total_per_arm: verified_elements,
              output_scope: "native_32_vs_native_64_bitwise_only"}.to_json)
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

  Drama3ExpertLaunchBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
