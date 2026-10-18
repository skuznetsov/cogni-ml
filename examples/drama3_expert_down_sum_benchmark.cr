# Full-generation candidate capture and paired local down-plus-sum diagnostic.
# The generation run preserves the ordinary terminal-code/WAV boundary; all
# MoE activations and arithmetic comparisons stay on Metal.
require "digest/sha256"
require "json"
require "set"
require "random/secure"
require "../src/ml/audio/drama3/generator_resident"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/prompt"
require "../src/ml/audio/drama3/wav"
require "./support/drama3_expert_down_sum_benchmark"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3ExpertDownSumBenchmark
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer

    TEXT           = "Hello, this is a voice test."
    SEED           = 1_u64
    MAX_FRAMES     =    64
    TEMPERATURE    =   0.8
    TOP_K          =    30
    TOP_P          =   0.9
    CAPTURE_FRAMES = [0, 29, 59] of Int32
    WARMUPS        = 2
    PAIRS          = 8
    REPEATS        = 2
    ORDER_CYCLE    = [
      ["unfused", "fused"],
      ["fused", "unfused"],
      ["fused", "unfused"],
      ["unfused", "fused"],
    ] of Array(String)

    CONTROLS = {
      "DRAMA3_BACKEND"                   => "metal",
      "DRAMA3_DAC_SCRATCH"               => "0",
      "DRAMA3_MEMORY_PROFILE"             => "0",
      "DRAMA3_ALLOCATION_PROFILE"         => "0",
      "DRAMA3_METAL_TIMELINE"             => "0",
      "DRAMA3_METAL_RESOURCES"            => "0",
      "DRAMA3_MOE_BATCH"                  => "1",
      "DRAMA3_EXPERT_THREADS"             => "64",
      "DRAMA3_DENSE_REDUCTION"            => "tree",
      "DRAMA3_BACKBONE_DENSE_REDUCTION"   => "tree",
      "DRAMA3_DELTA_TRANSPOSE"            => "1",
      "DRAMA3_EXPERT_DOWN_REDUCTION"      => "native",
      "DRAMA3_MMAP_SHARED"                => "0",
      "DRAMA3_COGNIGRAPH_MAX_INFLIGHT"    => "2",
      "DRAMA3_DAC_CONV_LAYOUT"             => "coalesced",
      "DRAMA3_TOWER_EMBEDDING_FUSION"     => "0",
      "DRAMA3_BACKBONE_DENSE_FUSION"      => "0",
      "DRAMA3_EXPERT_PACKED_ROWS"         => "0",
      "DRAMA3_TOWER_FFN_PAIR_FUSION"      => "0",
      "DRAMA3_EXPERT_DOWN_SUM_FUSION"     => "1",
      "DRAMA3_PREFILL_CHUNK"             => "1",
      "DRAMA3_GENERATION_SCRATCH"         => "0",
      "DRAMA3_HOST_PROFILE"                => "0",
    }

    record Arm, command_index : Int64, outputs : Array(Buffer), encode_seconds : Float64,
      fused_dispatches : Int64

    def self.configure_environment : Nil
      ENV.keys.each { |key| ENV.delete(key) if key.starts_with?("DRAMA3_") }
      ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
      CONTROLS.each { |key, value| ENV[key] = value }
      ENV.delete("DRAMA3_PROFILE")
      Profile.configure_from_env!
      raise "inference profiling must be disabled" if Profile.enabled?
    end

    def self.sha256(path : String) : String
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        bytes = Bytes.new(64 * 1024)
        while (count = io.read(bytes)) > 0
          digest.update(bytes[0, count])
        end
      end
      digest.hexfinal
    end

    def self.read_baseline(path : String) : NamedTuple(record: JSON::Any, report: JSON::Any)
      root = JSON.parse(File.read(path))
      record = if raw = root.as_a?
                 raise ArgumentError.new("baseline run list is empty") if raw.empty?
                 raw.first
               else
                 root
               end
      report = record["report"]? || record
      raise ArgumentError.new("baseline report is not resident Metal") unless report["backend"].as_s == "metal_resident"
      {record: record, report: report}
    end

    def self.assert_baseline_input!(report : JSON::Any) : Nil
      raise ArgumentError.new("baseline input text mismatch") unless report["text"].as_s == TEXT
      raise ArgumentError.new("baseline seed mismatch") unless report["seed"].as_i64 == SEED.to_i64
      raise ArgumentError.new("baseline max_frames mismatch") unless report["max_frames"].as_i == MAX_FRAMES
      raise ArgumentError.new("baseline temperature mismatch") unless report["temperature"].as_f == TEMPERATURE
      raise ArgumentError.new("baseline top_k mismatch") unless report["top_k"].as_i == TOP_K
      raise ArgumentError.new("baseline top_p mismatch") unless report["top_p"].as_f == TOP_P
      raise ArgumentError.new("baseline reference audio must be absent") unless report["reference_sha256"].raw.nil?
      raise ArgumentError.new("baseline expert down-sum selector must be frozen off") unless
        report["resident_expert_down_sum_fusion"].as_bool == false
    end

    def self.baseline_frames(report : JSON::Any) : Array(Array(Int32))
      report["frames"].as_a.map_with_index do |frame, row|
        frame.as_a.map_with_index do |value, column|
          raw = value.raw
          unless raw.is_a?(Int64) && raw.in?(Int32::MIN.to_i64..Int32::MAX.to_i64)
            raise ArgumentError.new("baseline frame #{row} column #{column} is not int32")
          end
          raw.to_i32
        end
      end
    end

    def self.baseline_reads(report : JSON::Any) : Hash(String, Int64)
      report["resident_readbacks"].as_h.transform_values do |value|
        raw = value.raw
        raise ArgumentError.new("baseline readback count must be an integer") unless raw.is_a?(Int64)
        raw
      end
    end

    def self.assert_graph!(actual : Hash(String, Int64), expected : JSON::Any) : Nil
      {"submitted", "completed", "completion_waits", "pending"}.each do |key|
        expected_value = expected[key].as_i64
        raise "generation graph #{key} changed: #{actual[key]} != #{expected_value}" unless actual[key] == expected_value
      end
      raise "generation graph must contain 122 submissions" unless actual["submitted"] == 122
      raise "generation graph must have no pending work" unless actual["pending"] == 0
      raise "generation graph must have no retained bytes" unless actual["retained_bytes"] == 0
      raise "generation graph depth must be two" unless actual["depth"] == 2
      raise "generation graph must saturate depth two" unless actual["max_pending"] == 2
    end

    def self.assert_reads!(actual : Hash(String, Int64), expected : Hash(String, Int64)) : Nil
      raise "resident readback ledger changed: #{actual} != #{expected}" unless actual == expected
      raise "unexpected diagnostic activation readback" unless actual.fetch("diagnostic", 0_i64) == 0
      raise "resident readback control/terminal bytes changed" unless actual.fetch("control", 0_i64) == 496 &&
        actual.fetch("terminal", 0_i64) == 494_160
    end

    def self.progress_frame(message : String) : Int32?
      return nil unless message.starts_with?("frame ")
      frame = message.byte_slice(6, message.bytesize - 6).not_nil!.split('/', 2).first
      frame.to_i?
    end

    def self.read_counters(ctx : Context, counters : Buffer) : Array(UInt32)
      raise "counter receipt must contain two words" unless counters.count == 2
      values = counters.buffer.contents.as(UInt32*) + counters.offset // 4
      ctx.readbacks["terminal"] += 8_i64
      [values[0], values[1]]
    end

    def self.compare(ctx : Context, expected : Buffer, actual : Buffer, counters : Buffer) : Nil
      raise "comparison shape mismatch" unless expected.count == actual.count
      ctx.dispatch("dr3_dense_four_compare", Drama3ExpertDownSum::RECEIPT_SOURCE,
        [expected, actual, counters], [expected.count.to_u64], expected.count)
    end

    def self.clear(ctx : Context, counters : Buffer) : Nil
      ctx.dispatch("dr3_fill", Resident::SOURCE, [counters], [counters.count.to_u64, 0_u64], counters.count)
    end

    def self.reference_sum(ctx : Context, record : Drama3ExpertDownSum::DownProjection) : Buffer
      down = ctx.experts(record.matrix, record.input, record.ids, per_rank_input: true, reduction: "native")
      routing = ctx.alloc(record.ids.count * 2)
      ctx.copy(record.ids, routing.slice(0, record.ids.count))
      ctx.copy(record.weights, routing.slice(record.ids.count, record.ids.count))
      output = ctx.alloc(record.matrix.rows)
      ctx.dispatch("dr3_expert_sum", BackboneResident::SOURCE, [down, routing, output],
        [record.matrix.rows.to_u64, record.ids.count.to_u64], record.matrix.rows)
      record.reference_down = down
      record.reference_output = output
      output
    end

    def self.verify_capture(ctx : Context, capture : Drama3ExpertDownSum::Capture) : JSON::Any
      capture.assert_inventory!
      before, reads_before = ctx.graph_stats.dup, ctx.readbacks.dup
      capture.records.each { |record| reference_sum(ctx, record) }
      raise "reference construction submitted or read back" unless ctx.graph_stats == before && ctx.readbacks == reads_before

      correct = ctx.alloc(2)
      clear(ctx, correct)
      operands = ctx.alloc(2)
      clear(ctx, operands)
      capture.records.each do |record|
        compare(ctx, record.reference_down.not_nil!, record.candidate_down, correct)
        compare(ctx, record.reference_output.not_nil!, record.candidate_output, correct)
        compare(ctx, record.input, record.input, operands)
        compare(ctx, record.ids, record.ids, operands)
        compare(ctx, record.weights, record.weights, operands)
      end

      variations = capture.input_variations
      variation_counters = ctx.alloc(variations.size * 2)
      clear(ctx, variation_counters)
      variations.each_with_index do |variation, index|
        compare(ctx, variation.reference, variation.actual, variation_counters.slice(index * 2, 2))
      end
      raise "capture validation added an early wait or read" unless ctx.readbacks == reads_before &&
        ctx.graph_stats["completion_waits"] == before["completion_waits"]

      ctx.finish
      bit_counts = read_counters(ctx, correct)
      operand_counts = read_counters(ctx, operands)
      raise "candidate/reference down or sum mismatch #{bit_counts}" unless bit_counts == [0_u32, 0_u32]
      raise "non-finite captured input/ID/weight #{operand_counts}" unless operand_counts == [0_u32, 0_u32]
      variation_values = Array(Array(UInt32)).new(variations.size) do |index|
        offset = index * 2
        pointer = variation_counters.buffer.contents.as(UInt32*) + (variation_counters.offset // 4) + offset
        [pointer[0], pointer[1]]
      end
      ctx.readbacks["terminal"] += variation_counters.count.to_i64 * 4
      down_input_variations_valid = variations.each_with_index.all? do |variation, index|
        variation.kind != "down_input" || (variation_values[index][0] > 0 && variation_values[index][1] == 0)
      end
      raise "a selected layer down input did not vary across rows" unless down_input_variations_valid
      raise "down-input variation contains non-finite values" unless variation_values.all? { |value| value[1] == 0 }

      after = ctx.graph_stats.dup
      raise "capture verification queue did not drain" unless after["pending"] == 0 && after["retained_bytes"] == 0
      JSON.parse({bit_mismatches: bit_counts[0], nonfinite_pairs: bit_counts[1],
                  operand_bit_mismatches: operand_counts[0], operand_nonfinite_pairs: operand_counts[1],
                  varying_layer_inputs_0_to_29: variations.count { |item| item.kind == "down_input" && item.from_frame == 0 },
                  varying_layer_inputs_29_to_59: variations.count { |item| item.kind == "down_input" && item.from_frame == 29 },
                  variation_receipts: variations.each_with_index.map { |item, index|
                    {name: item.name, kind: item.kind, from_frame: item.from_frame, to_frame: item.to_frame,
                     bit_mismatches: variation_values[index][0], nonfinite_pairs: variation_values[index][1]}
                  }, graph_before: before, graph_after: after, readbacks_before: reads_before,
                  readbacks_after: ctx.readbacks}.to_json)
    end

    private def self.encode_arm(ctx : Context, records : Array(Drama3ExpertDownSum::DownProjection),
                                arm_name : String, repeats : Int32) : Arm
      command_index = ctx.graph_stats["submitted"]
      fused_before = ctx.expert_down_sum_fused_dispatches
      started = Time.instant
      outputs = [] of Buffer
      repeats.times do
        records.each do |record|
          if arm_name == "fused"
            result = ctx.experts_down_sum(record.matrix, record.input, record.ids, record.weights,
              reduction: "native", capture_down: false)
            outputs << result[:output]
          else
            # Match the production flag-off fallback: per-rank down projection
            # followed by the existing weighted-sum kernel. The explicit
            # dr3_expert_sum routing-buffer reference stays capture-only above.
            down = ctx.experts(record.matrix, record.input, record.ids,
              per_rank_input: true, reduction: "native")
            output = ctx.alloc(record.matrix.rows)
            ctx.dispatch("dr3_experts_weighted_sum", Resident::SOURCE,
              [down, record.weights, output],
              [record.matrix.rows.to_u64, record.ids.count.to_u64], record.matrix.rows)
            outputs << output
          end
        end
      end
      encode_seconds = (Time.instant - started).total_seconds
      after_fused = ctx.expert_down_sum_fused_dispatches
      expected_fused = arm_name == "fused" ? (records.size * repeats).to_i64 : 0_i64
      raise "#{arm_name} arm used #{after_fused - fused_before} fused dispatches, expected #{expected_fused}" unless
        after_fused - fused_before == expected_fused
      ctx.submit
      Arm.new(command_index, outputs, encode_seconds, after_fused - fused_before)
    end

    def self.pair(ctx : Context, records : Array(Drama3ExpertDownSum::DownProjection), signature : Array(Int32),
                  pair_index : Int32, negative : Tuple(Buffer, Buffer, Buffer)) : Int64
      before, reads_before = ctx.graph_stats.dup, ctx.readbacks.dup
      raise "paired replay requires a drained queue" unless before["pending"] == 0 && before["retained_bytes"] == 0
      warmup = pair_index < WARMUPS
      order = ORDER_CYCLE[pair_index % ORDER_CYCLE.size]
      arms = {} of String => Arm
      order.each do |name|
        arm = encode_arm(ctx, records, name, REPEATS)
        raise "unexpected early read/wait" unless ctx.readbacks == reads_before &&
          ctx.graph_stats["completion_waits"] == before["completion_waits"]
        arms[name] = arm
      end
      queued = ctx.graph_stats.dup
      raise "expected two pending paired commands" unless queued["pending"] == 2 &&
        queued["submitted"] == before["submitted"] + 2

      counters = ctx.alloc(2)
      clear(ctx, counters)
      unfused, fused = arms["unfused"], arms["fused"]
      raise "paired output count mismatch" unless unfused.outputs.size == fused.outputs.size
      values = 0_i64
      unfused.outputs.zip(fused.outputs).each do |expected, actual|
        compare(ctx, expected, actual, counters)
        values += expected.count
      end
      negative_control = pair_index == 0
      if negative_control
        clear(ctx, negative[2])
        compare(ctx, negative[0], negative[1], negative[2])
      end
      ctx.finish
      counts = read_counters(ctx, counters)
      raise "paired down-sum bit/nonfinite mismatch #{counts}" unless counts == [0_u32, 0_u32]
      negative_counts = negative_control ? read_counters(ctx, negative[2]) : nil
      raise "seeded bit/NaN negative control was not rejected: #{negative_counts}" if
        negative_control && negative_counts != [2_u32, 1_u32]

      after = ctx.graph_stats.dup
      raise "unexpected paired completion accounting" unless after["submitted"] == before["submitted"] + 3 &&
        after["completed"] == before["completed"] + 3 && after["completion_waits"] == before["completion_waits"] + 3 &&
        after["pending"] == 0 && after["retained_bytes"] == 0
      expected_receipt_bytes = negative_control ? 16 : 8
      raise "unexpected paired readback" unless ctx.readbacks.fetch("terminal", 0_i64) ==
        reads_before.fetch("terminal", 0_i64) + expected_receipt_bytes &&
        ctx.readbacks.reject("terminal") == reads_before.reject("terminal")
      puts({event: "expert_down_sum_pair", signature: signature, pair: pair_index, warmup: warmup,
            repeats_per_arm: REPEATS, order: order, command_indices: arms.transform_values(&.command_index),
            host_encode_seconds: arms.transform_values(&.encode_seconds),
            actual_candidate_dispatches: arms.transform_values(&.fused_dispatches),
            records: records.size, values_per_arm: values, bit_mismatches: counts[0],
            nonfinite_pairs: counts[1], negative_control: negative_counts,
            terminal_receipt_bytes: expected_receipt_bytes, graph_before: before, graph_after: after,
            readbacks_before: reads_before, readbacks_after: ctx.readbacks,
            gpu_timing_source: "existing drama3_gpu_command stderr events after waits; correlate index with command_indices"}.to_json)
      values
    end

    def self.run_pairs(ctx : Context, capture : Drama3ExpertDownSum::Capture) : JSON::Any
      capture.active = false
      groups = capture.records.group_by(&.signature).to_a.sort_by { |entry| entry[0].to_a.join(":") }
      raise "no eligible homogeneous down-sum groups" if groups.empty?
      raise "paired benchmark requires all captured candidates eligible" unless capture.records.all?(&.eligible)
      negative = {ctx.upload([1_f32, 2_f32]), ctx.upload([1.25_f32, Float32::NAN]), ctx.alloc(2)}
      # Fixture uploads happen before the replay timeline and are not timed.
      setup_reads = ctx.readbacks.dup
      ctx.finish
      raise "diagnostic fixture setup changed readbacks" unless ctx.readbacks == setup_reads
      raise "diagnostic fixture setup did not drain" unless ctx.graph_stats["pending"] == 0 && ctx.graph_stats["retained_bytes"] == 0
      ctx.enable_expert_down_sum_timeline!
      total_values = 0_i64
      group_reports = [] of JSON::Any
      groups.each do |signature, records|
        start_graph = ctx.graph_stats.dup
        start_reads = ctx.readbacks.dup
        expected = {signature[0], signature[1], signature[2], signature[3], signature[4]}
        raise "homogeneous replay signature changed" unless records.all? { |record| record.signature == expected }
        (WARMUPS + PAIRS).times do |index|
          total_values += pair(ctx, records, signature.to_a, index, negative)
        end
        group_reports << JSON.parse({signature: signature.to_a, records: records.size,
          warmups: WARMUPS, measured_pairs: PAIRS, repeats_per_arm: REPEATS,
          graph_before: start_graph, graph_after: ctx.graph_stats,
          readbacks_before: start_reads, readbacks_after: ctx.readbacks}.to_json)
      end
      JSON.parse({groups: group_reports, verified_values_per_arm: total_values,
        pair_order_cycle: ORDER_CYCLE, warmups: WARMUPS, measured_pairs: PAIRS,
        repeats_per_arm: REPEATS, negative_control: "first warmup pair of each group requires [2 bit mismatches, 1 nonfinite pair]",
        graph: ctx.graph_stats, readbacks: ctx.readbacks, activation_readbacks: 0}.to_json)
    end

    def self.run : Nil
      ctx = nil.as(Context?)
      raise ArgumentError.new("usage: drama3_expert_down_sum_benchmark MODEL_DIR BASELINE_JSON OUTPUT_DIR") unless ARGV.size == 3
      model_dir, baseline_path, output_dir = ARGV.map { |value| File.expand_path(value) }
      raise ArgumentError.new("model directory is missing: #{model_dir}") unless Dir.exists?(model_dir)
      config_path = File.join(model_dir, "config.json")
      tokenizer_path = File.join(model_dir, "tokenizer.json")
      raise ArgumentError.new("model config is missing: #{config_path}") unless File.file?(config_path)
      raise ArgumentError.new("model tokenizer is missing: #{tokenizer_path}") unless File.file?(tokenizer_path)
      raise ArgumentError.new("baseline JSON is missing: #{baseline_path}") unless File.file?(baseline_path)

      Dir.mkdir_p(output_dir)
      wav_path = File.join(output_dir, "drama3-expert-down-sum.wav")
      manifest_path = File.join(output_dir, "capture.json")
      raise ArgumentError.new("refusing existing output WAV: #{wav_path}") if File.exists?(wav_path)
      raise ArgumentError.new("refusing existing output manifest: #{manifest_path}") if File.exists?(manifest_path)

      baseline = read_baseline(baseline_path)
      assert_baseline_input!(baseline[:report])
      expected_frames = baseline_frames(baseline[:report])
      expected_reads = baseline_reads(baseline[:report])
      raise ArgumentError.new("baseline must contain generated frames") if expected_frames.empty?
      raise ArgumentError.new("baseline frame count exceeds request bound") if expected_frames.size > MAX_FRAMES

      configure_environment
      config_json = JSON.parse(File.read(config_path))
      raise ArgumentError.new("only fish_omni at 44100 Hz is supported") unless
        config_json["model_type"].as_s == "fish_omni" && config_json["sample_rate"].as_i == 44_100
      tower_config = TowerConfig.from_json(config_json.to_json)
      raise ArgumentError.new("ten codec codebooks are required") unless tower_config.num_codebooks == 10
      tokenizer = Tokenizer.new(tokenizer_path, config_json["eos_token_id"].as_i, config_json["pad_token_id"].as_i)
      prompt = Prompt.build(TEXT, tokenizer, tower_config.num_codebooks)
      capacity = prompt.size + MAX_FRAMES - 1
      raise ArgumentError.new("baseline prompt row count mismatch") unless baseline[:report]["prompt_rows"].as_i == prompt.size

      store = WeightStore.new(model_dir)
      context = Resident::Context.new
      ctx = context
      raise "candidate selector did not freeze on" unless ctx.expert_down_sum_fusion_enabled
      raise "candidate requires 64 expert threads" unless ctx.expert_threads == 64
      raise "candidate requires packed rows disabled" if ctx.expert_packed_rows_enabled
      raise "candidate requires graph depth two" unless ctx.graph_stats["depth"] == 2
      # The selector belongs to the context and must not follow later environment edits.
      ENV["DRAMA3_EXPERT_DOWN_SUM_FUSION"] = "0"
      raise "candidate selector was not frozen at context construction" unless ctx.expert_down_sum_fusion_enabled
      capture = Drama3ExpertDownSum::Capture.new(ctx, store)
      ctx.expert_down_sum_capture = capture

      generator = ResidentGenerator.new(config_json, store, ctx, capacity)
      raise "backbone selector must preserve native expert down reduction" unless generator.backbone.expert_down_reduction == "native"
      raise "backbone batching selector mismatch" unless generator.backbone.batch_experts
      raise "other backbone dense fusion must remain off" if generator.backbone.dense_fusion_enabled
      generator.device_trace = ->(name : String, _buffer : Resident::Buffer) do
        if name == "prompt.main_logits"
          capture.frame = 0
          capture.active = CAPTURE_FRAMES.includes?(0)
        end
        nil
      end
      generator.progress = ->(message : String) do
        if frame = progress_frame(message)
          capture.frame = frame
          capture.active = CAPTURE_FRAMES.includes?(frame)
        end
        nil
      end

      generation_fused_before = ctx.expert_down_sum_fused_dispatches
      result = generator.generate(prompt, MAX_FRAMES, SEED, TEMPERATURE, TOP_K, TOP_P)
      capture.active = false
      frames = result.generation.frames
      raise "generated frames differ from baseline" unless frames == expected_frames
      raise "generation stop state differs from baseline" unless result.generation.stopped == baseline[:report]["stopped_on_eos"].as_bool
      raise "generation prompt row count differs from baseline" unless result.generation.prompt_rows == prompt.size
      sampler_inputs = generator.sampler_inputs || raise "resident sampler tape metadata is missing"
      baseline_sampler = baseline[:report]["sampler_inputs"]
      raise "sampler tape dtype differs from baseline" unless sampler_inputs[:dtype] == baseline_sampler["dtype"].as_s
      raise "sampler tape size differs from baseline" unless sampler_inputs[:uniform_count] == baseline_sampler["uniform_count"].as_i
      raise "sampler tape hash differs from baseline" unless sampler_inputs[:sha256] == baseline_sampler["sha256"].as_s
      raise "sampler tape provenance differs from baseline" unless sampler_inputs[:provenance] == baseline_sampler["provenance"].as_s
      expected_selected_dispatches = CAPTURE_FRAMES.size.to_i64 * 40
      generation_fused_dispatches = ctx.expert_down_sum_fused_dispatches - generation_fused_before
      raise "selected decode rows did not capture all 40 candidate calls" unless capture.records.size.to_i64 == expected_selected_dispatches
      raise "total generation candidate calls are below selected captured calls" unless
        generation_fused_dispatches >= expected_selected_dispatches

      codec = CodecResident.new(store, ctx)
      waveform = codec.decode(result.deviceframes.slice(0, frames.size * (tower_config.num_codebooks + 1)), frames.size)
      samples = ctx.read(waveform, "terminal")
      Wav.write(wav_path, samples)
      wav_sha = sha256(wav_path)
      expected_wav_sha = baseline[:record]["wav_sha256"].as_s
      raise "decoded WAV SHA differs from baseline: #{wav_sha} != #{expected_wav_sha}" unless wav_sha == expected_wav_sha

      generation_graph = ctx.graph_stats.dup
      generation_reads = ctx.readbacks.dup
      assert_graph!(generation_graph, baseline[:report]["resident_graph"])
      assert_reads!(generation_reads, expected_reads)
      raise "profiling must be disabled" if Profile.enabled?
      raise "generation scratch must remain disabled" if generator.workspace_stats[:enabled]
      capture.assert_inventory!

      capture_report = verify_capture(ctx, capture)
      pair_report = run_pairs(ctx, capture)
      root = File.expand_path("..", __DIR__)
      invocation = Random::Secure.hex(16)
      wav_report = {path: wav_path, sha256: wav_sha, expected_sha256: expected_wav_sha,
                    sample_rate: 44_100, samples: samples.size}
      generation_report = {frames: frames, frame_count: frames.size,
        stopped_on_eos: result.generation.stopped, prompt_rows: prompt.size, sampler_inputs: sampler_inputs}
      manifest = {
        format: "drama3-expert-down-sum-benchmark-v1", invocation: invocation,
        requested_arm: 1, baseline_arm: 0,
        model_dir: model_dir, baseline_path: baseline_path, baseline_sha256: sha256(baseline_path),
        input: {text: TEXT, seed: SEED, max_frames: MAX_FRAMES, temperature: TEMPERATURE, top_k: TOP_K, top_p: TOP_P},
        selected_decode_rows: CAPTURE_FRAMES, generation: generation_report,
        frozen_selector: {requested: true, context_value: ctx.expert_down_sum_fusion_enabled,
                          selected_capture_dispatches: capture.records.size,
                          total_generation_fused_dispatches: generation_fused_dispatches},
        eligible_inventory: capture.inventory,
        effective_controls: CONTROLS,
        generation_graph: generation_graph, generation_readbacks: generation_reads,
        capture_verification: capture_report, local_pairs: pair_report,
        final_graph: ctx.graph_stats, final_readbacks: ctx.readbacks,
        wav: wav_report,
        capture_records: capture.records.map(&.metadata),
        timing: {warmups: WARMUPS, measured_pairs: PAIRS, repeats_per_arm: REPEATS,
                 order_cycle: ORDER_CYCLE,
                 scope: "homogeneous local down-plus-sum commands; GPU command durations are emitted by existing timeline receipts after waits",
                 fused_arm: "experts_down_sum with candidate selector frozen on",
                 unfused_timed_arm: "production flag-off fallback: experts(per_rank_input: true) plus dr3_experts_weighted_sum; no legacy routing-buffer copies",
                 untimed_capture_reference: "explicit dr3_expert_sum using a copied IDs/weights routing buffer"},
        source_sha256: {
          benchmark: sha256(__FILE__), support: Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/support/drama3_expert_down_sum_benchmark.cr") }}),
          receipt_metal: Digest::SHA256.hexdigest(Drama3ExpertDownSum::RECEIPT_SOURCE),
          resident_cr: sha256(File.join(root, "src/ml/audio/drama3/resident.cr")),
          backbone_resident_cr: sha256(File.join(root, "src/ml/audio/drama3/backbone_resident.cr")),
          generator_resident_cr: sha256(File.join(root, "src/ml/audio/drama3/generator_resident.cr")),
          resident_metal: Digest::SHA256.hexdigest(Resident::SOURCE),
          backbone_metal: Digest::SHA256.hexdigest(BackboneResident::SOURCE),
        },
      }
      File.write(manifest_path, manifest.to_pretty_json + "\n")
      STDOUT.puts({event: "expert_down_sum_benchmark", manifest: manifest_path, generation: generation_report,
        frozen_selector: manifest[:frozen_selector], inventory: capture.inventory,
        generation_graph: generation_graph, generation_readbacks: generation_reads,
        final_graph: ctx.graph_stats, final_readbacks: ctx.readbacks,
        wav: wav_report, capture_verification: capture_report, local_pairs: pair_report}.to_json)
      STDOUT.flush
    rescue ex
      ctx.try(&.abort)
      raise ex
    end
  end
  Drama3ExpertDownSumBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
