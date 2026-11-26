# Full-generation candidate capture and paired local GDN-post diagnostic.
# Generation, codes, EOS, WAV, graph, and readback ledgers are checked against
# the qualified flag-off worker before the timing timeline is enabled.
require "digest/sha256"
require "json"
require "random/secure"
require "../src/ml/audio/drama3/generator_resident"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/prompt"
require "../src/ml/audio/drama3/wav"
require "./support/drama3_gdn_post_benchmark"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3GDNPostBenchmark
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer
    alias Capture = Drama3GDNPostBenchmarkSupport::Capture
    alias PostCall = Drama3GDNPostBenchmarkSupport::PostCall

    TEXT           = "Hello, this is a voice test."
    SEED           = 1_u64
    MAX_FRAMES     = 64
    TEMPERATURE    = 0.8
    TOP_K          = 30
    TOP_P          = 0.9
    CAPTURE_ROWS   = [0, 29, 59] of Int32
    WARMUPS        = 2
    PAIRS          = 8
    REPEATS        = 2
    ORDER_CYCLE    = [
      ["unfused", "candidate"],
      ["candidate", "unfused"],
      ["candidate", "unfused"],
      ["unfused", "candidate"],
    ] of Array(String)

    CONTROLS = {
      "DRAMA3_BACKEND"                    => "metal",
      "DRAMA3_DAC_SCRATCH"                => "0",
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
      "DRAMA3_DAC_CONV_LAYOUT"            => "coalesced",
      "DRAMA3_TOWER_EMBEDDING_FUSION"     => "0",
      "DRAMA3_BACKBONE_DENSE_FUSION"      => "0",
      "DRAMA3_EXPERT_PACKED_ROWS"         => "0",
      "DRAMA3_TOWER_FFN_PAIR_FUSION"      => "0",
      "DRAMA3_EXPERT_DOWN_SUM_FUSION"     => "0",
      "DRAMA3_EXPERT_SWIGLU_FUSION"       => "0",
      "DRAMA3_GDN_POST_FUSION"            => "1",
      "DRAMA3_PREFILL_CHUNK"              => "1",
      "DRAMA3_GENERATION_SCRATCH"         => "0",
      "DRAMA3_HOST_PROFILE"               => "0",
    }

    COMPILED_SOURCE_SHA256 = {
      "src/ml/audio/drama3/resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/resident.cr") }}),
      "src/ml/audio/drama3/backbone_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/backbone_resident.cr") }}),
      "src/ml/audio/drama3/generator_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/generator_resident.cr") }}),
      "src/ml/audio/drama3/prompt.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/prompt.cr") }}),
      "src/ml/audio/drama3/codec_resident.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/codec_resident.cr") }}),
      "src/ml/audio/drama3/wav.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/../src/ml/audio/drama3/wav.cr") }}),
      "examples/drama3_gdn_post_benchmark.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/drama3_gdn_post_benchmark.cr") }}),
      "examples/support/drama3_gdn_post_benchmark.cr" => Digest::SHA256.hexdigest({{ read_file("#{__DIR__}/support/drama3_gdn_post_benchmark.cr") }}),
    }

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

    def self.digest_model(model_dir : String) : NamedTuple(files: Hash(String, String), aggregate_sha256: String)
      paths = ["config.json", "tokenizer.json", "model.safetensors.index.json"] of String
      index_path = File.join(model_dir, "model.safetensors.index.json")
      if File.file?(index_path)
        shards = JSON.parse(File.read(index_path))["weight_map"].as_h.values.map(&.as_s)
        paths.concat(shards)
      elsif File.file?(File.join(model_dir, "model.safetensors"))
        paths << "model.safetensors"
      end
      paths << "codec.safetensors" if File.file?(File.join(model_dir, "codec.safetensors"))
      paths = paths.select { |relative| File.file?(File.join(model_dir, relative)) }.uniq.sort
      files = paths.to_h { |relative| {relative, sha256(File.join(model_dir, relative))} }
      aggregate = Digest::SHA256.hexdigest(files.map { |name, digest| "#{name}\n#{digest}\n" }.join)
      {files: files, aggregate_sha256: aggregate}
    end

    def self.verify_compiled_sources!(root : String) : Hash(String, String)
      COMPILED_SOURCE_SHA256.each do |relative, compiled_sha256|
        path = File.join(root, relative)
        raise ArgumentError.new("compiled source is missing at runtime: #{relative}") unless File.file?(path)
        runtime_sha256 = sha256(path)
        unless runtime_sha256 == compiled_sha256
          raise ArgumentError.new("runtime source differs from compile-time bytes: #{relative}")
        end
      end
      COMPILED_SOURCE_SHA256
    end

    def self.read_baseline(path : String) : NamedTuple(record: JSON::Any, report: JSON::Any)
      root = JSON.parse(File.read(path))
      record = if raw = root.as_a?
                 raise ArgumentError.new("baseline run list is empty") if raw.empty?
                 raw.first
               else
                 root
               end
      arm = record["arm"]?
      raise ArgumentError.new("qualified GDN control must be arm 0") unless arm && arm.as_i == 0
      baseline_controls = record["effective_controls"].as_h
      expected_baseline = CONTROLS.dup
      expected_baseline["DRAMA3_GDN_POST_FUSION"] = "0"
      expected_baseline_keys = expected_baseline.keys.sort
      raise ArgumentError.new("arm-0 baseline control inventory differs") unless baseline_controls.keys.sort == expected_baseline_keys
      expected_baseline_keys.each do |key|
        expected = expected_baseline[key]
        actual = baseline_controls[key].as_s
        raise ArgumentError.new("arm-0 baseline control #{key}=#{actual}, expected #{expected}") unless actual == expected
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
      raise ArgumentError.new("baseline GDN post selector must be explicitly off") unless report["resident_gdn_post_fusion"].as_bool == false
      raise ArgumentError.new("baseline GDN post fused dispatch count must be zero") unless report["resident_gdn_post_fused_dispatches"].as_i64 == 0_i64
      if down_sum = report["resident_expert_down_sum_fusion"]?
        raise ArgumentError.new("baseline expert down-sum selector must be off") unless down_sum.as_bool == false
      end
      if swiglu = report["resident_expert_swiglu_fusion"]?
        raise ArgumentError.new("baseline expert SwiGLU selector must be off") unless swiglu.as_bool == false
      end
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

    def self.run : Nil
      context = nil.as(Context?)
      raise ArgumentError.new("usage: drama3_gdn_post_benchmark MODEL_DIR BASELINE_JSON OUTPUT_DIR") unless ARGV.size == 3
      model_dir, baseline_path, output_dir = ARGV.map { |value| File.expand_path(value) }
      raise ArgumentError.new("model directory is missing: #{model_dir}") unless Dir.exists?(model_dir)
      config_path, tokenizer_path = File.join(model_dir, "config.json"), File.join(model_dir, "tokenizer.json")
      raise ArgumentError.new("model config is missing: #{config_path}") unless File.file?(config_path)
      raise ArgumentError.new("model tokenizer is missing: #{tokenizer_path}") unless File.file?(tokenizer_path)
      raise ArgumentError.new("baseline JSON is missing: #{baseline_path}") unless File.file?(baseline_path)
      Dir.mkdir_p(output_dir)
      wav_path = File.join(output_dir, "drama3-gdn-post.wav")
      manifest_path = File.join(output_dir, "capture.json")
      raise ArgumentError.new("refusing existing output WAV: #{wav_path}") if File.exists?(wav_path)
      raise ArgumentError.new("refusing existing output manifest: #{manifest_path}") if File.exists?(manifest_path)
      root = File.expand_path("..", __DIR__)
      checked_source_sha256 = verify_compiled_sources!(root)

      baseline = read_baseline(baseline_path)
      assert_baseline_input!(baseline[:report])
      expected_frames, expected_reads = baseline_frames(baseline[:report]), baseline_reads(baseline[:report])
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
      raise "GDN candidate selector did not freeze on" unless ctx.gdn_post_fusion_enabled
      raise "other expert down-sum fusion must remain off" if ctx.expert_down_sum_fusion_enabled
      raise "other expert SwiGLU fusion must remain off" if ctx.expert_swiglu_fusion_enabled
      raise "candidate requires graph depth two" unless ctx.graph_stats["depth"] == 2
      ENV["DRAMA3_GDN_POST_FUSION"] = "0"
      raise "GDN selector changed after context construction" unless ctx.gdn_post_fusion_enabled
      capture = Capture.new(ctx, store, prompt.size)
      ctx.gdn_post_benchmark_capture = capture

      generator = ResidentGenerator.new(config_json, store, ctx, capacity)
      raise "backbone selector must preserve native expert down reduction" unless generator.backbone.expert_down_reduction == "native"
      raise "backbone batching selector mismatch" unless generator.backbone.batch_experts
      raise "other backbone dense fusion must remain off" if generator.backbone.dense_fusion_enabled
      generator.backbone.gdn_post_benchmark_capture = capture

      generation_dispatches_before = ctx.gdn_post_fused_dispatches
      result = generator.generate(prompt, MAX_FRAMES, SEED, TEMPERATURE, TOP_K, TOP_P)
      capture.finish_generation!
      frames = result.generation.frames
      raise "generated frames differ from arm-0 baseline" unless frames == expected_frames
      raise "generation stop state differs from baseline" unless result.generation.stopped == baseline[:report]["stopped_on_eos"].as_bool
      raise "generation prompt row count differs from baseline" unless result.generation.prompt_rows == prompt.size
      sampler_inputs = generator.sampler_inputs || raise "resident sampler tape metadata is missing"
      baseline_sampler = baseline[:report]["sampler_inputs"]
      raise "sampler tape dtype differs from baseline" unless sampler_inputs[:dtype] == baseline_sampler["dtype"].as_s
      raise "sampler tape size differs from baseline" unless sampler_inputs[:uniform_count] == baseline_sampler["uniform_count"].as_i
      raise "sampler tape hash differs from baseline" unless sampler_inputs[:sha256] == baseline_sampler["sha256"].as_s
      raise "sampler tape provenance differs from baseline" unless sampler_inputs[:provenance] == baseline_sampler["provenance"].as_s
      capture.assert_inventory!
      backbone_rows = prompt.size + Math.min(frames.size, MAX_FRAMES - 1)
      expected_dispatches = backbone_rows.to_i64 * Drama3GDNPostBenchmarkSupport::LAYER_COUNT
      actual_dispatches = ctx.gdn_post_fused_dispatches - generation_dispatches_before
      raise "generation GDN candidate dispatch count #{actual_dispatches} != #{expected_dispatches}" unless actual_dispatches == expected_dispatches

      codec = CodecResident.new(store, ctx)
      waveform = codec.decode(result.deviceframes.slice(0, frames.size * (tower_config.num_codebooks + 1)), frames.size)
      samples = ctx.read(waveform, "terminal")
      Wav.write(wav_path, samples)
      wav_sha = sha256(wav_path)
      expected_wav_sha = baseline[:record]["wav_sha256"].as_s
      raise "decoded WAV SHA differs from baseline: #{wav_sha} != #{expected_wav_sha}" unless wav_sha == expected_wav_sha

      generation_graph, generation_reads = ctx.graph_stats.dup, ctx.readbacks.dup
      assert_graph!(generation_graph, baseline[:report]["resident_graph"])
      assert_reads!(generation_reads, expected_reads)
      raise "profiling must be disabled" if Profile.enabled?
      raise "generation scratch must remain disabled" if generator.workspace_stats[:enabled]
      capture_report = Drama3GDNPostBenchmarkSupport.verify_capture(ctx, capture)
      pair_report = Drama3GDNPostBenchmarkSupport.run_pairs(ctx, capture, WARMUPS, PAIRS, REPEATS)

      invocation = Random::Secure.hex(16)
      model_digest = digest_model(model_dir)
      executable = Process.executable_path.not_nil!
      wav_report = {path: wav_path, sha256: wav_sha, expected_sha256: expected_wav_sha,
        sample_rate: 44_100, samples: samples.size}
      generation_report = {frames: frames, frame_count: frames.size, stopped_on_eos: result.generation.stopped,
        prompt_rows: prompt.size, sampler_inputs: sampler_inputs}
      gdn_dispatch_report = {expected_all_generation_rows: expected_dispatches,
        actual_all_generation_dispatches: actual_dispatches,
        selected_rows: CAPTURE_ROWS,
        selected_captured_candidate_dispatches: capture.selected_candidate_dispatches,
        context_fused_dispatch_count_after_generation_capture_validation_and_local_replay: ctx.gdn_post_fused_dispatches}
      manifest = {
        format: "drama3-gdn-post-benchmark-v1", invocation: invocation,
        requested_arm: 1, baseline_arm: 0, model_dir: model_dir, baseline_path: baseline_path,
        baseline_sha256: sha256(baseline_path), executable_path: executable, executable_sha256: sha256(executable),
        model_files_baseline_authenticated: false,
        baseline_gdn_post: {fusion_enabled: baseline[:report]["resident_gdn_post_fusion"].as_bool,
          fused_dispatches: baseline[:report]["resident_gdn_post_fused_dispatches"].as_i64},
        model_sha256: {aggregate: model_digest[:aggregate_sha256], files: model_digest[:files]},
        input: {text: TEXT, seed: SEED, max_frames: MAX_FRAMES,
          temperature: TEMPERATURE, top_k: TOP_K, top_p: TOP_P},
        selected_generation_rows: CAPTURE_ROWS, generation: generation_report,
        frozen_selector: {requested: true, context_value: ctx.gdn_post_fusion_enabled,
          all_other_fusions_off: true, expected_candidate_dispatches: expected_dispatches,
          actual_generation_dispatches: actual_dispatches},
        gdn_post_dispatches: gdn_dispatch_report, effective_controls: CONTROLS,
        generation_graph: generation_graph, generation_readbacks: generation_reads,
        capture_inventory: capture.inventory, capture_verification: capture_report,
        local_pairs: pair_report, final_graph: ctx.graph_stats, final_readbacks: ctx.readbacks,
        wav: wav_report, capture_records: capture.records.map(&.metadata),
        timing: {warmups: WARMUPS, measured_pairs: PAIRS, repeats_per_arm: REPEATS,
          order_cycle: ORDER_CYCLE,
          scope: "homogeneous GDN post commands; GPU durations correlate with existing command timestamps by command index",
          candidate_arm: "frozen gdn_post fused dispatch",
          unfused_arm: "existing RMS, SiLU, then multiply sequence",
          capture_reference: "existing RMS, SiLU, then multiply sequence; GPU finite-bit and nonfinite-class receipts before timing"},
        source_sha256: {
          runtime_checked_compile_time_sha256: checked_source_sha256,
          resident_metal: Digest::SHA256.hexdigest(Resident::SOURCE),
          backbone_metal: Digest::SHA256.hexdigest(BackboneResident::SOURCE),
          receipt_metal: Digest::SHA256.hexdigest(Drama3GDNPostBenchmarkSupport::RECEIPT_SOURCE),
        },
      }
      File.write(manifest_path, manifest.to_pretty_json + "\n")
      STDOUT.puts({event: "drama3_gdn_post_benchmark", manifest: manifest_path,
        generation: generation_report, frozen_selector: manifest[:frozen_selector],
        dispatches: gdn_dispatch_report, inventory: capture.inventory,
        generation_graph: generation_graph, generation_readbacks: generation_reads,
        final_graph: ctx.graph_stats, final_readbacks: ctx.readbacks,
        wav: wav_report, capture_verification: capture_report, local_pairs: pair_report}.to_json)
      STDOUT.flush
    rescue ex
      context.try(&.abort)
      raise ex
    end
  end
  Drama3GDNPostBenchmark.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
