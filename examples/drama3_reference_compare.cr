# Reference-voice parity probe: upload immutable MLX fixtures, observe actual
# native GPU activations, and read only terminal reduced comparison metrics.
require "digest/sha256"
require "option_parser"
require "../src/ml/audio/drama3/codec_encoder_resident"
require "../src/ml/audio/drama3/reference_audio"
require "../src/ml/audio/drama3/prompt_resident"
require "../src/ml/audio/drama3/tower"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3ReferenceCompare
    include ML::Audio::Drama3
    alias Buffer = Resident::Buffer
    alias Context = Resident::Context
    SOURCE = {{ read_file("#{__DIR__}/support/drama3_codec_compare.metal") + read_file("#{__DIR__}/support/drama3_reference_compare.metal") }}
    METRICS = 14
    CODEC_SHA256 = "543445a2ebc5224d2af5956d422f88c2cd21f4018c1a26d81b51ec231bb630f2"
    UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
    record Stage, name : String, rows : Int32, channels : Int32, reference : Buffer

    def self.file_sha(path : String) : String
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        chunk = Bytes.new(1024 * 1024)
        while (count = io.read(chunk)) > 0
          digest.update(chunk[0, count])
        end
      end
      digest.hexfinal
    end

    def self.load_stage(context : Context, directory : String, item : JSON::Any) : Stage
      name = item["name"].as_s
      shape = item["shape"].as_a.map(&.as_i)
      axis = name == "prompt.rows" ? "prompt_row" : "time"
      raise "stage layout #{name}" unless shape.size == 2 && item["row_axis"].as_s == axis && item["dtype"].as_s == "float32-le"
      rows, channels = shape[0], shape[1]
      count = rows.to_i64 * channels
      raise "stage dimensions #{name}" unless rows > 0 && channels > 0 && count <= Int32::MAX
      filename = item["file"].as_s
      raise "invalid stage filename" unless File.basename(filename) == filename && !filename.in?(".", "..")
      path = File.join(directory, filename)
      raise "stage byte count #{name}" unless File.size(path) == count * 4
      buffer = context.alloc(count.to_i32)
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        offset = 0_i64
        while offset < count * 4
          length = Math.min(count * 4 - offset, 1024_i64 * 1024).to_i32
          bytes = Bytes.new(buffer.buffer.contents.as(UInt8*) + offset, length)
          io.read_fully(bytes)
          digest.update(bytes)
          offset += length
        end
      end
      raise "stage hash mismatch #{name}" unless digest.hexfinal == item["sha256"].as_s
      Stage.new(name, rows, channels, buffer)
    end

    def self.compare(context : Context, actual : Buffer, expected : Buffer) : Buffer
      raise "comparison length mismatch" unless actual.count == expected.count
      parts_count = (actual.count + 1023) // 1024
      parts = context.alloc(parts_count * METRICS)
      output = context.alloc(METRICS)
      context.dispatch("dr3_codec_compare_parts", SOURCE, [actual, expected, parts],
        [actual.count.to_u64, 0_u64], parts_count)
      context.dispatch("dr3_codec_compare_reduce", SOURCE, [parts, output],
        [actual.count.to_u64, parts_count.to_u64], 1, 1)
      output
    end

    def self.stage_result(stage : Stage, values : Array(Float32))
      exact_required = stage.name == "final.codes" || stage.name == "prompt.rows" || stage.name.ends_with?(".selected_ids")
      finite = values[7] == 0
      passes = finite && values[0] <= 1e-3_f32 && values[1] <= 2e-4_f32 && values[13] >= 0 && values[13] <= 2e-4_f32
      passes &&= values[0] == 0 if exact_required
      {name: stage.name, shape: [stage.rows, stage.channels], max_abs: values[0], rms: values[1],
       reference_rms: values[2], actual_rms: values[3], reference_peak: values[4], actual_peak: values[5],
       nonfinite: values[7].to_i64, normalized_rms: values[13] >= 0 ? values[13] : nil,
       worst_index: (values[8].to_i64 << 16) | values[9].to_i64,
       exact_required: exact_required, elementwise_exact: finite && values[0] == 0, passes: passes}
    end

    def self.self_test : Nil
      context = Context.new
      reference = context.upload([0_f32, 1_f32, 4095_f32])
      stage = Stage.new("final.codes", 3, 1, reference)
      exact = compare(context, reference, reference)
      poisoned = compare(context, context.upload([0_f32, 2_f32, 4095_f32]), reference)
      pointwise_exact = pointwise_gate(context, reference, reference)
      pointwise_poisoned = pointwise_gate(context, context.upload([0_f32, 2_f32, 4095_f32]), reference)
      context.finish
      raise "ID equality oracle" unless stage_result(stage, context.read(exact, "diagnostic"))[:passes]
      raise "ID poison was admitted" if stage_result(stage, context.read(poisoned, "diagnostic"))[:passes]
      raise "pointwise equality oracle" unless context.read(pointwise_exact, "diagnostic")[0] == 0
      raise "pointwise poison was admitted" unless context.read(pointwise_poisoned, "diagnostic")[0] == 1
      puts({self_test: "PASS", scope: "GPU exact-ID and original backbone pointwise gates"}.to_json)
    end

    def self.pointwise_gate(context : Context, actual : Buffer, expected : Buffer) : Buffer
      raise "pointwise comparison length mismatch" unless actual.count == expected.count
      output = context.alloc(3)
      context.dispatch("dr3_reference_pointwise_gate", SOURCE, [actual, expected, output],
        [actual.count.to_u64], 1, 1)
      output
    end

    def self.run : Nil
      model_path = ""
      manifest_path = ""
      wav_path = ""
      reference_text = ""
      text = ""
      output_path = ""
      prefill_path = ""
      self_test = false
      OptionParser.parse do |parser|
        parser.on("--model DIR", "Pinned model directory") { |v| model_path = File.expand_path(v) }
        parser.on("--manifest FILE", "Upstream encoder fixture manifest") { |v| manifest_path = File.expand_path(v) }
        parser.on("--wav FILE", "The same mono PCM16 reference WAV") { |v| wav_path = File.expand_path(v) }
        parser.on("--reference-text TEXT", "Exact reference transcript") { |v| reference_text = v }
        parser.on("--text TEXT", "Target text from the prompt fixture") { |v| text = v }
        parser.on("--output FILE", "New comparison JSON") { |v| output_path = File.expand_path(v) }
        parser.on("--prefill-reference FILE", "Optional terminal MLX hidden/logit fixture manifest") { |v| prefill_path = File.expand_path(v) }
        parser.on("--self-test", "Check exact-ID reducer rejection") { self_test = true }
      end
      return self.self_test if self_test
      raise "all reference comparison arguments are required" if [model_path, manifest_path, wav_path, reference_text, text, output_path].any?(&.empty?)
      raise "output already exists" if File.exists?(output_path)
      input = JSON.parse(File.read(manifest_path))
      raise "manifest format" unless input["format"].as_s == "drama3-reference-voice-manifest-v1"
      raise "upstream identity" unless input["upstream"]["revision"].as_s == UPSTREAM_REVISION && input["upstream"]["worktree_clean"].as_bool
      raise "codec fixture identity" unless input["codec"]["sha256"].as_s == CODEC_SHA256
      raise "native codec identity" unless file_sha(File.join(model_path, "codec.safetensors")) == CODEC_SHA256
      raise "observer changed upstream IDs" unless input["encode"]["unobserved_vs_observed_gpu_code_comparison"]["status"].as_s == "PASS"
      raise "reference waveform identity" unless input["input"]["wav_sha256"].as_s == file_sha(wav_path)
      raise "reference transcript identity" unless input["input"]["transcript_utf8_sha256"].as_s == Digest::SHA256.hexdigest(reference_text)
      audio_input = Wav.read_reference(wav_path)
      raise "reference sample count" unless audio_input.samples == input["input"]["samples"].as_i
      config = JSON.parse(File.read(File.join(model_path, "config.json")))
      tokenizer = Tokenizer.new(File.join(model_path, "tokenizer.json"), config["eos_token_id"].as_i, config["pad_token_id"].as_i)
      plan = Prompt.reference_plan(text, reference_text, audio_input.frames, tokenizer,
        10, config["semantic_start_token_id"].as_i, TowerConfig.from_json(config.to_json).vocab_size)

      context = Context.new
      stages = {} of String => Stage
      input["stages"].as_a.each do |item|
        stage = load_stage(context, File.dirname(manifest_path), item)
        raise "duplicate reference stage #{stage.name}" if stages.has_key?(stage.name)
        stages[stage.name] = stage
      end
      required = ["audio.padded", "encoder.output", "quantizer.pre_module", "final.codes"]
      raise "incomplete encoder fixture" unless required.all? { |name| stages.has_key?(name) }
      prompt_stage = load_stage(context, File.dirname(manifest_path), input["prompt"])
      raise "target text identity" unless input["prompt"]["target_text_utf8_sha256"].as_s == Digest::SHA256.hexdigest(text)
      raise "prompt fixture identity" unless prompt_stage.name == "prompt.rows" && prompt_stage.rows == plan.row_count && prompt_stage.channels == plan.row_width
      prefill_stages = {} of String => Stage
      model_files = {} of String => String
      unless prefill_path.empty?
        prefill = JSON.parse(File.read(prefill_path))
        raise "prefill fixture format" unless prefill["format"].as_s == "drama3-prefill-reference-v1"
        raise "prefill text identity" unless prefill["text_sha256"].as_s == Digest::SHA256.hexdigest(text)
        raise "prefill waveform identity" unless prefill["wav_sha256"].as_s == file_sha(wav_path)
        raise "prefill transcript identity" unless prefill["transcript_sha256"].as_s == Digest::SHA256.hexdigest(reference_text)
        raise "prefill upstream identity" unless prefill["upstream_revision"].as_s == UPSTREAM_REVISION
        index = JSON.parse(File.read(File.join(model_path, "model.safetensors.index.json")))
        required_files = ["config.json", "model.safetensors.index.json", "tokenizer.json", "codec.safetensors"] +
                         index["weight_map"].as_h.values.map(&.as_s).uniq
        prefill["model_files"].as_a.each do |entry|
          filename = entry["file"].as_s
          raise "invalid prefill model filename" unless File.basename(filename) == filename && required_files.includes?(filename)
          raise "duplicate prefill model file" if model_files.has_key?(filename)
          digest = file_sha(File.join(model_path, filename))
          raise "prefill checkpoint identity #{filename}" unless digest == entry["sha256"].as_s
          model_files[filename] = digest
        end
        raise "incomplete prefill checkpoint identity" unless model_files.keys.sort == required_files.sort
        prefill["stages"].as_a.each do |item|
          stage = load_stage(context, File.dirname(prefill_path), item)
          raise "duplicate prefill stage" if prefill_stages.has_key?(stage.name)
          prefill_stages[stage.name] = stage
        end
        raise "prefill stage contract" unless prefill_stages.keys.sort == ["prompt.backbone.hidden", "prompt.main_logits"] &&
                                             prefill_stages["prompt.backbone.hidden"].rows == 1 && prefill_stages["prompt.backbone.hidden"].channels == 2048 &&
                                             prefill_stages["prompt.main_logits"].rows == 1 && prefill_stages["prompt.main_logits"].channels == 4097
      end
      diagnostics = {} of String => Buffer
      prefill_gates = {} of String => Buffer
      observed = [] of String
      store = WeightStore.new(model_path)
      encoder = CodecEncoderResident.new(store, context)
      encoder.device_trace = ->(name : String, actual : Buffer, rows : Int32, channels : Int32) {
        stage = stages[name]? || raise "unexpected encoder stage #{name}"
        raise "duplicate observed stage #{name}" if diagnostics.has_key?(name)
        raise "encoder shape #{name}: #{rows}x#{channels} != #{stage.rows}x#{stage.channels}" unless rows == stage.rows && channels == stage.channels
        raise "intermediate encoder host read" unless context.readbacks.empty?
        diagnostics[name] = compare(context, actual, stage.reference)
        observed << name
        nil
      }
      started = Time.instant
      encoded = encoder.encode(ReferenceAudio.upload(audio_input, context), audio_input.samples)
      raise "encoder frame count" unless encoded.frames == audio_input.frames
      prompt = PromptResident.build(plan, encoded.codes, context)
      raise "encoder/prompt handoff host read" unless context.readbacks.empty?
      diagnostics["prompt.rows"] = compare(context, prompt.rows, prompt_stage.reference)
      raise "missing encoder stages #{stages.keys - observed}" unless stages.keys.sort == observed.sort
      unless prefill_stages.empty?
        generator = ResidentGenerator.new(config, store, context, plan.row_count)
        generator.device_trace = ->(name : String, actual : Buffer) {
          stage = prefill_stages[name]? || raise "unexpected prefill stage #{name}"
          raise "duplicate prefill stage #{name}" if prefill_gates.has_key?(name)
          raise "prefill dimensions #{name}" unless actual.count == stage.reference.count
          diagnostics[name] = compare(context, actual, stage.reference)
          prefill_gates[name] = pointwise_gate(context, actual, stage.reference)
          nil
        }
        # One ordinary attempted frame supplies the existing EOS/control and
        # terminal boundaries; the diagnostic callbacks add no synchronization.
        generator.generate(prompt, 1, 1_u64, 0.8, 30, 0.9)
        raise "missing prefill trace" unless prefill_gates.keys.sort == prefill_stages.keys.sort
      end
      context.finish
      elapsed = (Time.instant - started).total_seconds
      before_reads = context.graph_stats
      raise "pending reference work" unless before_reads["pending"] == 0
      raise "intermediate diagnostic host read" if context.readbacks.has_key?("diagnostic")
      results = observed.map { |name| stage_result(stages[name], context.read(diagnostics[name], "diagnostic")) }
      prompt_result = stage_result(prompt_stage, context.read(diagnostics["prompt.rows"], "diagnostic"))
      prefill_results = prefill_stages.values.map do |stage|
        values = context.read(diagnostics[stage.name], "diagnostic")
        pointwise = context.read(prefill_gates[stage.name], "diagnostic")
        stage_result(stage, values).merge({passes: pointwise[0] == 0 && pointwise[1] == 0,
          pointwise_failures: pointwise[0].to_i, nonfinite: pointwise[1].to_i64,
          worst_error_to_allowance: pointwise[2], atol: 3e-5, rtol: 3e-4})
      end
      after_reads = context.graph_stats
      raise "terminal metrics introduced GPU work/waits" unless before_reads == after_reads
      code_result = results.find { |result| result[:name] == "final.codes" }.not_nil!
      first_failed = results.find { |result| !result[:passes] }.try(&.[:name]) ||
                     (prompt_result[:passes] ? nil : "prompt.rows") ||
                     prefill_results.find { |result| !result[:passes] }.try(&.[:name])
      report = {
        format: "drama3-native-reference-comparison-v1", manifest: manifest_path, manifest_sha256: file_sha(manifest_path),
        codec_sha256: CODEC_SHA256, waveform_sha256: file_sha(wav_path), samples: audio_input.samples, frames: encoded.frames,
        reference_text_sha256: Digest::SHA256.hexdigest(reference_text), text_sha256: Digest::SHA256.hexdigest(text),
        gates: {max_abs: 1e-3, rms: 2e-4, normalized_rms: 2e-4, code_ids: "elementwise exact", prompt_rows: "elementwise exact"},
        stages: results, prompt: prompt_result, code_ids_exact: code_result[:elementwise_exact],
        gate_scope: prefill_path.empty? ? "encoder and prompt only" : "encoder, prompt, and first prefill hidden/logits",
        prefill: {requested: !prefill_path.empty?, fixture: prefill_path.empty? ? nil : prefill_path,
                  fixture_sha256: prefill_path.empty? ? nil : file_sha(prefill_path),
                  model_files: model_files, stages: prefill_results,
                  passes: prefill_path.empty? ? nil : prefill_results.all?(&.[:passes])},
        first_failed_stage: first_failed,
        full_gate_passes: results.all?(&.[:passes]) && prompt_result[:passes] && prefill_results.all?(&.[:passes]),
        execution: {backend: "resident-metal", precision: "F32 checkpoint and loaded rotary coefficients", seconds: elapsed,
                    activation_readback_before_completion: false, code_read_before_prefill: false,
                    comparison: "GPU F32 compensated reductions", resident_readbacks: context.readbacks,
                    graph_before_terminal_reads: before_reads, graph_after_terminal_reads: after_reads},
      }
      File.write(output_path, report.to_pretty_json)
      puts({frames: encoded.frames, stages: results.size, first_failed_stage: report[:first_failed_stage],
            code_ids_exact: report[:code_ids_exact], prompt_exact: prompt_result[:elementwise_exact], full_gate_passes: report[:full_gate_passes], seconds: elapsed}.to_json)
      exit 2 unless report[:full_gate_passes]
    end
  end

  Drama3ReferenceCompare.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
