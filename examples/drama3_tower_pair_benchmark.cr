# Paired production w1/w3 dispatches on captured upstream tower inputs.
# Arithmetic and comparisons stay on GPU; only compact receipts are read.
# MODEL_DIR PAIR_FIXTURE EXPECTED_FIXTURE_SHA256 [--repeats=8]
require "json"
require "digest/sha256"
require "random/secure"
require "../src/ml/audio/drama3/resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  PAIR_RECEIPT_SOURCE = {{ read_file("#{__DIR__}/support/drama3_dense_fusion.metal") }}
  PAIR_TOLERANCE_SOURCE = {{ read_file("#{__DIR__}/support/drama3_prefill_compare.metal") }}

  module Drama3TowerPairBenchmark
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer
    alias Outputs = Tuple(Buffer, Buffer)
    alias Matrices = Tuple(Matrix, Matrix)
    NAMES = {"w1", "w3"}
    WARMUPS = 2
    PAIRS = 7
    MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
    MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
    UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
    record Arm, command_index : Int64, outputs : Array(Outputs), encode_seconds : Float64

    def self.sha(path : String) : String
      Digest::SHA256.hexdigest(File.read(path))
    end

    def self.receipt(ctx : Context, expected : Buffer, actual : Buffer, counters : Buffer) : Nil
      raise "receipt shape" unless expected.count == actual.count
      ctx.dispatch("dr3_dense_four_compare", PAIR_RECEIPT_SOURCE, [expected, actual, counters],
        [expected.count.to_u64], expected.count)
    end

    def self.read_receipt(ctx : Context, counters : Buffer) : Array(UInt32)
      pointer = counters.buffer.contents.as(UInt32*) + counters.offset // 4
      ctx.readbacks["terminal"] += 8_i64
      [pointer[0], pointer[1]]
    end

    def self.pair(ctx : Context, matrices : Matrices, input : Buffer, repeats : Int32,
                  row : Int32, label : String, pair_index : Int32, negative : Tuple(Buffer, Buffer, Buffer)) : Int64
      before, reads_before = ctx.graph_stats.dup, ctx.readbacks.dup
      order = pair_index.even? ? ["separate", "fused"] : ["fused", "separate"]
      arms = {} of String => Arm
      order.each do |name|
        command_index = ctx.graph_stats["submitted"]
        started = Time.instant
        outputs = Array(Outputs).new(repeats) { ctx.matvec_pair(matrices, input, dense_reduction: "tree", fused: name == "fused") }
        encode_seconds = (Time.instant - started).total_seconds
        ctx.submit
        raise "unexpected early read/wait" unless ctx.readbacks == reads_before && ctx.graph_stats["completion_waits"] == before["completion_waits"]
        arms[name] = Arm.new(command_index, outputs, encode_seconds)
      end
      queued = ctx.graph_stats
      raise "expected two pending commands" unless queued["pending"] == 2 && queued["submitted"] == before["submitted"] + 2
      counters = ctx.alloc(2)
      ctx.dispatch("dr3_dense_four_receipt_clear", PAIR_RECEIPT_SOURCE, [counters], [] of UInt64, 2)
      elements = 0_i64
      arms["separate"].outputs.zip(arms["fused"].outputs).each do |expected, actual|
        2.times do |projection|
          receipt(ctx, expected[projection], actual[projection], counters)
          elements += expected[projection].count
        end
      end
      negative_control = row == 0 && pair_index == 0
      if negative_control
        ctx.dispatch("dr3_dense_four_receipt_clear", PAIR_RECEIPT_SOURCE, [negative[2]], [] of UInt64, 2)
        receipt(ctx, negative[0], negative[1], negative[2])
      end
      ctx.finish
      counts = read_receipt(ctx, counters)
      raise "tower pair bit/nonfinite mismatch #{counts}" unless counts == [0_u32, 0_u32]
      negative_counts = negative_control ? read_receipt(ctx, negative[2]) : nil
      raise "negative control failed #{negative_counts}" if negative_control && negative_counts != [2_u32, 1_u32]
      after = ctx.graph_stats
      raise "unexpected diagnostic queue accounting" unless after["submitted"] == before["submitted"] + 3 &&
                                                               after["completed"] == before["completed"] + 3 &&
                                                               after["completion_waits"] == before["completion_waits"] + 3 &&
                                                               after["pending"] == 0 && after["retained_bytes"] == 0
      raise "unexpected control read" unless ctx.readbacks.fetch("control", 0_i64) == reads_before.fetch("control", 0_i64) &&
                                             ctx.readbacks.fetch("diagnostic", 0_i64) == reads_before.fetch("diagnostic", 0_i64)
      puts({event: "tower_pair", row: row, label: label, pair: pair_index, warmup: pair_index < WARMUPS, repeats: repeats,
            order: order, command_indices: arms.transform_values(&.command_index),
            host_encode_seconds: arms.transform_values(&.encode_seconds), bit_mismatches: counts[0], nonfinite_pairs: counts[1],
            negative_control: negative_counts, values_per_arm: elements, terminal_receipt_bytes: negative_control ? 16 : 8,
            graph: after}.to_json)
      elements
    end

    # The fixture serializer hashes exact immutable BF16 payloads, before any
    # conversion or GPU context exists. This is model identity, not inference.
    def self.verify_model(store : WeightStore, model_path : String, fixture : JSON::Any) : String
      hashes = fixture["source_hashes"].as_h
      ["config.json", "model.safetensors.index.json"].each do |name|
        raise "model #{name} changed" unless sha(File.join(model_path, name)) == hashes[name].as_s
      end
      names = fixture["cases"].as_a.flat_map { |row| NAMES.map { |name| "#{row["weight_prefix"].as_s}.#{name}.weight" }.to_a }.uniq.sort
      digest = Digest::SHA256.new
      names.each do |name|
        tensor = store.tensor(name)
        raise "expected mapped BF16 weight" unless tensor.dtype == "BF16" && tensor.shape == [6144, 2048]
        digest.update("#{name}\0torch.bfloat16\0")
        shape = IO::Memory.new
        tensor.shape.each { |dimension| shape.write_bytes(dimension.to_i64, IO::ByteFormat::LittleEndian) }
        digest.update(shape.to_slice)
        digest.update(tensor.raw_bytes || raise "missing mapped weight payload")
      end
      actual = digest.hexfinal
      raise "mapped pair weights changed" unless actual == hashes["actual_projection_tensors"].as_s
      actual
    end

    def self.upstream(ctx : Context, matrices : Matrices, input : Buffer, row : JSON::Any, ordinal : Int32) : Nil
      actual = ctx.matvec_pair(matrices, input, dense_reduction: "tree", fused: true)
      receipts = NAMES.map_with_index do |name, index|
        record = row["projections"][name]
        raise "upstream shape/dtype" unless record["dtype"].as_s == "float32" && record["shape"].as_a.map(&.as_i) == [6144]
        values = record["values"].as_a.map(&.as_f.to_f32)
        raise "upstream count/finite" unless values.size == 6144 && values.all?(&.finite?)
        expected = ctx.upload(values)
        tiles = (values.size + 1023) // 1024
        parts, result = ctx.alloc(tiles * 5), ctx.alloc(5)
        ctx.dispatch("dr3_prefill_compare", PAIR_TOLERANCE_SOURCE, [actual[index], expected, parts],
          [values.size.to_u64, 6e-4_f32.unsafe_as(UInt32).to_u64, 3e-4_f32.unsafe_as(UInt32).to_u64], tiles)
        ctx.dispatch("dr3_prefill_compare_reduce", PAIR_TOLERANCE_SOURCE, [parts, result], [tiles.to_u64], 1)
        result
      end
      ctx.finish
      metrics = NAMES.map_with_index do |name, index|
        pointer = receipts[index].buffer.contents.as(Float32*) + receipts[index].offset // 4
        values = Array(Float32).new(5) { |column| pointer[column] }
        ctx.readbacks["terminal"] += 20_i64
        raise "upstream tolerance/nonfinite failure #{name}: #{values}" unless values.all?(&.finite?) && values[1] == 0 && values[4] == 0
        {name: name, bit_differences: values[0].to_i, tolerance_failures: values[1].to_i,
         max_abs: values[2], max_ratio: values[3], nonfinite: values[4].to_i}
      end
      puts({event: "tower_pair_upstream", row: ordinal, label: row["label"].as_s, atol: 6e-4, rtol: 3e-4,
            metrics: metrics, terminal_receipt_bytes: 40}.to_json)
    end

    def self.run : Nil
      raise "usage: tower pair benchmark MODEL_DIR PAIR_FIXTURE EXPECTED_FIXTURE_SHA256 [--repeats=8]" unless ARGV.size.in?(3..4)
      model_path, fixture_path, expected_sha = ARGV[0], ARGV[1], ARGV[2]
      repeats = ARGV[3]?.try { |value| value.starts_with?("--repeats=") ? value.split('=', 2)[1].to_i : raise "unknown argument" } || 8
      raise "repeats must be 1..32" unless repeats.in?(1..32)
      raise "fixture digest changed" unless expected_sha.size == 64 && sha(fixture_path) == expected_sha
      fixture = JSON.parse(File.read(fixture_path))
      raise "fixture format/pins" unless fixture["format"].as_s == "drama3-tower-pair-fixture-v1" &&
                                        fixture["model_repository"].as_s == MODEL_REPOSITORY && fixture["model_revision"].as_s == MODEL_REVISION &&
                                        fixture["upstream_revision"].as_s == UPSTREAM_REVISION
      cases = fixture["cases"].as_a
      raise "three distinct case identities required" unless cases.size == 3 && cases.map { |row| row["label"].as_s }.uniq.size == 3
      store = WeightStore.new(model_path)
      weights_sha256 = verify_model(store, model_path, fixture)
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
      ENV["DRAMA3_METAL_TIMELINE"] = "1"
      ENV["DRAMA3_DENSE_REDUCTION"] = "tree"
      ENV.delete("DRAMA3_PROFILE")
      Profile.configure_from_env!
      ctx = Context.new
      negative = {ctx.upload([1_f32, 0_f32]), ctx.upload([1_f32.unsafe_as(UInt32).+(1_u32).unsafe_as(Float32), Float32::NAN]), ctx.alloc(2)}
      root = File.expand_path("..", __DIR__)
      invocation = Random::Secure.hex(16)
      puts({event: "tower_pair_invocation", id: invocation, model: File.expand_path(model_path),
            fixture: File.expand_path(fixture_path), fixture_sha256: expected_sha, mapped_weights_sha256: weights_sha256,
            model_repository: MODEL_REPOSITORY, model_revision: MODEL_REVISION, upstream_revision: UPSTREAM_REVISION,
            source_sha256: {benchmark: sha(__FILE__), receipt: Digest::SHA256.hexdigest(PAIR_RECEIPT_SOURCE),
                            tolerance: Digest::SHA256.hexdigest(PAIR_TOLERANCE_SOURCE), resident_cr: sha(File.join(root, "src/ml/audio/drama3/resident.cr")),
                            resident_metal: Digest::SHA256.hexdigest(Resident::SOURCE)},
            queue_depth: 2, warmups: WARMUPS, measured_pairs: PAIRS, repeats: repeats,
            timing_scope: "paired overlapping command intervals, not isolated kernels or additive family costs"}.to_json)
      STDERR.puts({event: "tower_pair_invocation", id: invocation}.to_json)
      total = 0_i64
      cases.each_with_index do |row, ordinal|
        prefix = row["weight_prefix"].as_s
        matrices = NAMES.map { |name| store.matrix("#{prefix}.#{name}.weight") }
        matrices.each do |matrix|
          raise "matrix shape/dtype" unless matrix.rows == 6144 && matrix.cols == 2048 && matrix.tensor.dtype == "BF16" && !matrix.scales && !matrix.biases
        end
        raise "input shape/dtype" unless row["input"]["dtype"].as_s == "float32" && row["input"]["shape"].as_a.map(&.as_i) == [2048]
        values = row["input"]["values"].as_a.map(&.as_f.to_f32)
        raise "input width/finite" unless values.size == 2048 && values.all?(&.finite?)
        input = ctx.upload(values)
        (WARMUPS + PAIRS).times { |index| total += pair(ctx, matrices, input, repeats, ordinal, row["label"].as_s, index, negative) }
        upstream(ctx, matrices, input, row, ordinal)
      end
      puts({event: "tower_pair_summary", id: invocation, verified_values_per_arm: total, input_cases: cases.size,
            graph: ctx.graph_stats, readbacks: ctx.readbacks, full_upstream_gate_claimed: false}.to_json)
    rescue ex
      ctx.try(&.abort)
      raise ex
    end
  end
  Drama3TowerPairBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
