require "digest/sha256"
require "json"
require "../src/ml/audio/drama3/generator_resident"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/prompt"
require "../src/ml/audio/drama3/wav"
require "./support/drama3_decode_replay"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3DecodeReplayDriver
    include ML::Audio::Drama3

    TEXT           = "Hello, this is a voice test."
    SEED           = 1_u64
    MAX_FRAMES     =    64
    TEMPERATURE    =   0.8
    TOP_K          =    30
    TOP_P          =   0.9
    CAPTURE_FRAMES = [0, 29, 59] of Int32

    CONTROLS = {
      "DRAMA3_BACKEND"                  => "metal",
      "DRAMA3_DAC_SCRATCH"              => "0",
      "DRAMA3_MEMORY_PROFILE"           => "0",
      "DRAMA3_ALLOCATION_PROFILE"       => "0",
      "DRAMA3_METAL_TIMELINE"           => "0",
      "DRAMA3_METAL_RESOURCES"          => "0",
      "DRAMA3_MOE_BATCH"                => "1",
      "DRAMA3_EXPERT_THREADS"           => "64",
      "DRAMA3_DENSE_REDUCTION"          => "tree",
      "DRAMA3_BACKBONE_DENSE_REDUCTION" => "tree",
      "DRAMA3_DELTA_TRANSPOSE"          => "1",
      "DRAMA3_EXPERT_DOWN_REDUCTION"    => "native",
      "DRAMA3_MMAP_SHARED"              => "0",
      "DRAMA3_COGNIGRAPH_MAX_INFLIGHT"  => "2",
      "DRAMA3_DAC_CONV_LAYOUT"          => "coalesced",
      "DRAMA3_TOWER_EMBEDDING_FUSION"   => "0",
      "DRAMA3_BACKBONE_DENSE_FUSION"    => "0",
      "DRAMA3_EXPERT_PACKED_ROWS"       => "0",
      "DRAMA3_TOWER_FFN_PAIR_FUSION"    => "0",
      "DRAMA3_GENERATION_SCRATCH"       => "0",
      "DRAMA3_HOST_PROFILE"             => "0",
    }

    def self.configure_environment : Nil
      ENV.keys.each { |key| ENV.delete(key) if key.starts_with?("DRAMA3_") }
      ENV.delete("QWEN35_NATIVE_TOKENIZER_OFF")
      CONTROLS.each { |key, value| ENV[key] = value }
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
        raise "generation/decode graph #{key} changed: #{actual[key]} != #{expected_value}" unless actual[key] == expected_value
      end
      raise "generation/decode graph must contain 122 submissions" unless actual["submitted"] == 122
      raise "generation/decode graph must have no pending work" unless actual["pending"] == 0
      raise "generation/decode graph must have no retained bytes" unless actual["retained_bytes"] == 0
      raise "generation/decode graph depth must be two" unless actual["depth"] == 2
      raise "generation/decode graph must saturate depth two" unless actual["max_pending"] == 2
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

    def self.run : Nil
      raise ArgumentError.new("usage: drama3_decode_replay MODEL_DIR BASELINE_JSON OUTPUT_DIR") unless ARGV.size == 3
      model_dir, baseline_path, output_dir = ARGV.map { |value| File.expand_path(value) }
      raise ArgumentError.new("model directory is missing: #{model_dir}") unless Dir.exists?(model_dir)
      config_path = File.join(model_dir, "config.json")
      tokenizer_path = File.join(model_dir, "tokenizer.json")
      raise ArgumentError.new("model config is missing: #{config_path}") unless File.file?(config_path)
      raise ArgumentError.new("model tokenizer is missing: #{tokenizer_path}") unless File.file?(tokenizer_path)
      raise ArgumentError.new("baseline JSON is missing: #{baseline_path}") unless File.file?(baseline_path)

      Dir.mkdir_p(output_dir)
      wav_path = File.join(output_dir, "drama3-decode-capture.wav")
      manifest_path = File.join(output_dir, "capture.json")
      raise ArgumentError.new("refusing existing output WAV: #{wav_path}") if File.exists?(wav_path)
      raise ArgumentError.new("refusing existing output manifest: #{manifest_path}") if File.exists?(manifest_path)

      configure_environment
      config = JSON.parse(File.read(config_path))
      raise ArgumentError.new("only fish_omni at 44100 Hz is supported") unless config["model_type"].as_s == "fish_omni" && config["sample_rate"].as_i == 44_100
      tower_config = TowerConfig.from_json(config.to_json)
      raise ArgumentError.new("ten codec codebooks are required") unless tower_config.num_codebooks == 10
      tokenizer = Tokenizer.new(tokenizer_path, config["eos_token_id"].as_i, config["pad_token_id"].as_i)
      prompt = Prompt.build(TEXT, tokenizer, tower_config.num_codebooks)
      capacity = prompt.size + MAX_FRAMES - 1

      baseline = read_baseline(baseline_path)
      assert_baseline_input!(baseline[:report])
      expected_frames = baseline_frames(baseline[:report])
      expected_reads = baseline_reads(baseline[:report])
      raise ArgumentError.new("baseline must contain generated frames") if expected_frames.empty?
      raise ArgumentError.new("baseline prompt row count mismatch") unless baseline[:report]["prompt_rows"].as_i == prompt.size
      raise ArgumentError.new("baseline frame count exceeds request bound") if expected_frames.size > MAX_FRAMES

      store = WeightStore.new(model_dir)
      context = Resident::Context.new
      capture = Drama3DecodeReplay::Capture.new(context, store)
      capture.frame = 0
      capture.active = false
      context.decode_replay_capture = capture
      generator = ResidentGenerator.new(config, store, context, capacity)
      generator.device_trace = ->(name : String, _buffer : Resident::Buffer) do
        if name == "prompt.main_logits"
          # Row zero's lm_head dispatch has already been encoded at this callback;
          # the observer captures the remaining selected decode work.
          capture.frame = 0
          capture.active = true
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

      codec = CodecResident.new(store, context)
      waveform = codec.decode(result.deviceframes.slice(0, frames.size * (tower_config.num_codebooks + 1)), frames.size)
      samples = context.read(waveform, "terminal")
      Wav.write(wav_path, samples)
      wav_sha = sha256(wav_path)
      expected_wav_sha = baseline[:record]["wav_sha256"].as_s
      raise "decoded WAV SHA differs from baseline: #{wav_sha} != #{expected_wav_sha}" unless wav_sha == expected_wav_sha

      graph = context.graph_stats
      assert_graph!(graph, baseline[:report]["resident_graph"])
      assert_reads!(context.readbacks, expected_reads)
      raise "allocation or inference profiling must be disabled" if Profile.enabled?
      raise "generation scratch must remain disabled" if generator.workspace_stats[:enabled]
      Drama3DecodeReplay.assert_coverage!(capture.records.map(&.metadata))

      wav_report = {
        path:            wav_path,
        sha256:          wav_sha,
        expected_sha256: expected_wav_sha,
        sample_rate:     44_100,
        samples:         samples.size,
      }
      generation_report = {
        frames:         frames,
        frame_count:    frames.size,
        stopped_on_eos: result.generation.stopped,
        prompt_rows:    prompt.size,
        sampler_inputs: sampler_inputs,
      }
      summary = capture.summary
      records = capture.records.map(&.metadata)
      manifest = {
        format:          "drama3-decode-replay-capture-v1",
        model_dir:       model_dir,
        baseline_path:   baseline_path,
        baseline_sha256: sha256(baseline_path),
        input:           {text: TEXT, seed: SEED, max_frames: MAX_FRAMES, temperature: TEMPERATURE, top_k: TOP_K, top_p: TOP_P},
        selected_frames: CAPTURE_FRAMES,
        generation:      generation_report,
        graph:           graph,
        readbacks:       context.readbacks,
        wav:             wav_report,
        capture_summary: summary,
        capture_records: records,
        environment:     CONTROLS,
      }
      File.write(manifest_path, manifest.to_pretty_json + "\n")
      STDOUT.puts({event: "capture", manifest: manifest_path, generation: generation_report,
                   graph: graph, readbacks: context.readbacks, wav: wav_report,
                   capture_summary: summary, capture_records: records.size}.to_json)
      STDOUT.flush

      # The ordinary full-generation and DAC result is checked before the
      # homogeneous decode replay starts. Capture was inactive for DAC decode.
      capture.run
    rescue ex
      context.try(&.abort)
      raise ex
    end
  end

  Drama3DecodeReplayDriver.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
