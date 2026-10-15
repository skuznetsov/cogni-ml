# Full-stream DAC comparison. Reference activations are uploaded once; model
# work and all numerical comparisons run on Metal. Only final diagnostics and
# the completed waveform are read by the host.
require "digest/sha256"
require "json"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/wav"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module Drama3CodecCrossCompare
    include ML::Audio::Drama3
    alias Buffer = Resident::Buffer
    alias Context = Resident::Context
    SOURCE = {{ read_file("#{__DIR__}/support/drama3_codec_compare.metal") }}
    CODEC_SHA256 = "543445a2ebc5224d2af5956d422f88c2cd21f4018c1a26d81b51ec231bb630f2"
    UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
    FISH_CODEC_SHA256 = "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
    FISH_REVISION     = "541566932abbaebcb0a40e562a1f846738383f38"
    METRICS = 14

    record Stage, name : String, rows : Int32, channels : Int32, reference : Buffer

    def self.expected_names : Array(String)
      names = [] of String
      10.times do |book|
        names << "quantizer.book.#{book}.embedding" << "quantizer.book.#{book}.projection"
      end
      names << "quantizer.codebook_sum"
      8.times { |layer| names << "quantizer.post_module.layers.#{layer}" }
      names << "quantizer.post_module"
      2.times do |stage|
        names << "quantizer.upsample.#{stage}.deconv" << "quantizer.upsample.#{stage}.convnext"
      end
      names << "decoder.model.0"
      4.times { |stage| 5.times { |block| names << "decoder.model.#{stage + 1}.block.#{block}" } }
      names << "decoder.model.5" << "decoder.model.6" << "waveform"
      names
    end

    def self.file_sha(path : String) : String
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        chunk = Bytes.new(1024 * 1024)
        while (size = io.read(chunk)) > 0
          digest.update(chunk[0, size])
        end
      end
      digest.hexfinal
    end

    def self.load_stage(context : Context, directory : String, item : JSON::Any) : Stage
      name = item["name"].as_s
      rows = item["rows"].as_i
      channels = item["channels"].as_i
      count = rows.to_i64 * channels
      raise "invalid stage dimensions #{name}" unless rows > 0 && channels > 0 && count <= Int32::MAX && item["count"].as_i64 == count
      filename = item["file"].as_s
      raise "invalid stage filename" unless File.basename(filename) == filename && !filename.in?(".", "..")
      path = File.join(directory, filename)
      raise "stage byte count #{name}" unless File.size(path) == count * 4
      buffer = context.alloc(count.to_i32)
      digest = Digest::SHA256.new
      File.open(path, "rb") do |io|
        offset = 0_i64
        while offset < count * 4
          size = Math.min(count * 4 - offset, 1024_i64 * 1024).to_i32
          bytes = Bytes.new(buffer.buffer.contents.as(UInt8*) + offset, size)
          io.read_fully(bytes)
          digest.update(bytes)
          offset += size
        end
      end
      raise "stage hash mismatch #{name}" unless digest.hexfinal == item["sha256"].as_s
      Stage.new(name, rows, channels, buffer)
    end

    def self.compare(context : Context, actual : Buffer, reference : Buffer, waveform : Bool) : Buffer
      raise "comparison length mismatch" unless actual.count == reference.count
      parts_count = (actual.count + 1023) // 1024
      parts = context.alloc(parts_count * METRICS)
      output = context.alloc(METRICS)
      context.dispatch("dr3_codec_compare_parts", SOURCE, [actual, reference, parts],
        [actual.count.to_u64, waveform ? 1_u64 : 0_u64], parts_count)
      context.dispatch("dr3_codec_compare_reduce", SOURCE, [parts, output],
        [actual.count.to_u64, parts_count.to_u64], 1, 1)
      output
    end

    def self.load_reference(path : String)
      input = JSON.parse(File.read(path))
      revision, codec_sha = case input["format"].as_s
                            when "drama3-codec-cross-reference-v1"
                              {UPSTREAM_REVISION, CODEC_SHA256}
                            when "drama3-fish-codec-cross-reference-v1"
                              {FISH_REVISION, FISH_CODEC_SHA256}
                            else
                              raise "reference format"
                            end
      raise "upstream identity" unless input["upstream"]["revision"].as_s == revision && input["upstream"]["worktree_clean"].as_bool
      raise "reference observer changed official DAC output" unless input["decode"]["official_decode_comparison"]["elementwise_exact"].as_bool && input["decode"]["official_decode_comparison"]["status"].as_s == "PASS"
      raise "reference codec identity" unless input["codec"]["sha256"].as_s == codec_sha
      frames = input["frames"].as_i
      raise "frame count" unless frames.in?(1..1024)
      codes = input["codes"].as_a.map { |book| book.as_a.map(&.as_i) }
      raise "ten complete DAC codebooks required" unless codes.size == 10 && codes.all? { |book| book.size == frames }
      codes.each_with_index do |book, index|
        vocab = index == 0 ? 4096 : 1024
        raise "codebook id range #{index}" unless book.all? { |value| value >= 0 && value < vocab }
      end
      raise "complete stage set required" unless input["stages"].as_a.map { |item| item["name"].as_s }.sort == expected_names.sort
      {input, frames, codes}
    end

    # Fixture transfers precede all GPU comparisons. This generic mode compares
    # completed decodes (or an identity control) without replaying either one.
    def self.compare_fixtures(actual_path : String, reference_path : String, output_path : String) : Nil
      raise "output already exists" if File.exists?(output_path)
      actual_input, frames, codes = load_reference(actual_path)
      reference_input, reference_frames, reference_codes = load_reference(reference_path)
      raise "fixtures must contain identical complete codes" unless frames == reference_frames && codes == reference_codes
      context = Context.new
      actual_stages = {} of String => Stage
      reference_stages = {} of String => Stage
      actual_input["stages"].as_a.each do |item|
        stage = load_stage(context, File.dirname(actual_path), item)
        actual_stages[stage.name] = stage
      end
      reference_input["stages"].as_a.each do |item|
        stage = load_stage(context, File.dirname(reference_path), item)
        reference_stages[stage.name] = stage
      end
      diagnostics = {} of String => Buffer
      expected_names.each do |name|
        a, r = actual_stages[name], reference_stages[name]
        raise "fixture stage shape #{name}" unless a.rows == r.rows && a.channels == r.channels
        diagnostics[name] = compare(context, a.reference, r.reference, name == "waveform")
      end
      a, r = actual_stages["waveform"], reference_stages["waveform"]
      raise "waveform shape" unless a.rows == frames * 2048 && a.channels == 1
      windows = (a.rows + 22049) // 22050
      window_buffer = context.alloc(windows * 6)
      context.dispatch("dr3_codec_wave_windows", SOURCE, [a.reference, r.reference, window_buffer], [a.rows.to_u64], windows)
      context.finish
      raise "comparison introduced intermediate host reads" unless context.readbacks.empty?
      before_reads = context.graph_stats
      results = expected_names.map do |name|
        stage_result(actual_stages[name], context.read(diagnostics[name], "diagnostic"))
      end
      window_values = context.read(window_buffer, "diagnostic")
      after_reads = context.graph_stats
      raise "terminal diagnostics added GPU work or waits" unless before_reads == after_reads
      wave = results.last
      report = {
        format: "drama3-codec-fixture-comparison-v1", actual_json: actual_path, reference_json: reference_path,
        actual_sha256: file_sha(actual_path), reference_sha256: file_sha(reference_path),
        actual_format: actual_input["format"].as_s, reference_format: reference_input["format"].as_s,
        actual_source_revision: actual_input["upstream"]["revision"].as_s, reference_source_revision: reference_input["upstream"]["revision"].as_s,
        actual_codec_sha256: actual_input["codec"]["sha256"].as_s, reference_codec_sha256: reference_input["codec"]["sha256"].as_s,
        frames: frames, samples: frames * 2048, sample_rate: 44100, stages: results,
        stage_gate_passes: results.all?(&.[:passes]), first_failed_stage: results.find { |item| !item[:passes] }.try(&.[:name]),
        one_lsb_gate_passes: wave[:within_one_pcm_lsb],
        full_gate_passes: results.all?(&.[:passes]) && wave[:within_one_pcm_lsb] == true && wave[:actual_peak] > 1e-7 && wave[:reference_peak] > 1e-7,
        gates: gates,
        waveform_windows: window_results(window_values, frames * 2048),
        execution: {backend: "resident-metal", comparison_reduction: "GPU F32 compensated sums",
                    activation_readback_before_completion: false, graph_before_terminal_reads: before_reads, graph_after_terminal_reads: after_reads,
                    resident_readbacks: context.readbacks},
      }
      File.write(output_path, report.to_json)
      puts({frames: frames, stages: results.size, first_failed_stage: report[:first_failed_stage], full_gate_passes: report[:full_gate_passes], max_pcm_error: wave[:max_abs_pcm_error]}.to_json)
    end

    def self.gates
      {stage_max_abs: 1e-3, stage_rms: 2e-4, stage_normalized_rms: 2e-4,
       waveform_atol: 5e-5, waveform_rtol: 1e-3, waveform_rule: "max(atol, abs(reference)*rtol)", pcm_max_lsb: 1}
    end

    def self.stage_result(stage : Stage, values : Array(Float32))
      name = stage.name
      norm = values[13]
      finite = values[7] == 0
      passes = if name == "waveform"
                 finite && values[6] == 0
               else
                 finite && values[0] <= 1e-3_f32 && values[1] <= 2e-4_f32 && norm >= 0 && norm <= 2e-4_f32
               end
      {name: name, rows: stage.rows, channels: stage.channels,
       max_abs: values[0], rms: values[1], reference_rms: values[2], actual_rms: values[3],
       reference_peak: values[4], actual_peak: values[5], point_failures: values[6].to_i64,
       nonfinite: values[7].to_i64, normalized_rms: norm >= 0 ? norm : nil,
       worst_index: (values[8].to_i64 << 16) | values[9].to_i64,
       max_abs_pcm_error: name == "waveform" ? values[10].to_i32 : nil,
       rms_pcm_error: name == "waveform" ? values[11] : nil,
       exact_pcm_samples: name == "waveform" ? values[12].to_i64 : nil,
       within_one_pcm_lsb: name == "waveform" ? finite && values[10] <= 1 : nil,
       passes: passes}
    end

    def self.window_results(values : Array(Float32), samples : Int32)
      values.each_slice(6).with_index.map { |item, index|
        {start_seconds: index * 0.5, end_seconds: Math.min((index + 1) * 0.5, samples / 44100.0),
         actual_rms: item[0], reference_rms: item[1], actual_peak: item[2], reference_peak: item[3],
         actual_pcm_peak: item[4].to_i32, reference_pcm_peak: item[5].to_i32}
      }.to_a
    end

    # A deliberately incorrect value must fail the diagnostic. These small
    # fixtures validate the GPU reducer against a host F64 oracle; the host
    # oracle is never used for model activations.
    def self.self_test : Nil
      context = Context.new
      reference = Array(Float32).new(2051) { |i| ((i % 23) - 11).to_f32 / 32 }
      actual = reference.dup
      actual[1024] += 0.002_f32
      actual[2050] -= 0.004_f32
      diagnostic = compare(context, context.upload(actual), context.upload(reference), true)
      values = context.read(diagnostic, "diagnostic")
      errors = actual.zip(reference).map { |a, r| a.to_f64 - r.to_f64 }
      expected_rms = Math.sqrt(errors.sum { |error| error * error } / errors.size)
      raise "GPU RMS oracle" unless (values[1] - expected_rms).abs < 1e-9
      raise "GPU maximum oracle" unless values[0] == errors.map(&.abs).max.to_f32
      raise "GPU waveform gate failed to reject poison" unless values[6] == 2 && values[10] > 1
      raise "GPU worst index" unless ((values[8].to_i64 << 16) | values[9].to_i64) == 2050
      exact = context.read(compare(context, context.upload(reference), context.upload(reference), true), "diagnostic")
      raise "GPU equality gate" unless exact[0] == 0 && exact[1] == 0 && exact[6] == 0 && exact[10] == 0 && exact[12] == reference.size

      invalid = [Float32::NAN, Float32::INFINITY, 0.0_f32]
      bad = context.read(compare(context, context.upload(invalid), context.upload([0.0_f32, 0.0_f32, 0.0_f32]), true), "diagnostic")
      raise "GPU nonfinite guard" unless bad[7] == 2
      zero = context.read(compare(context, context.upload([0.01_f32]), context.upload([0.0_f32]), false), "diagnostic")
      raise "GPU zero reference guard" unless zero[13] == -1

      # Preserve indices beyond F32's exact integer range in two 16-bit limbs.
      large = Array(Float32).new(16_777_221, 0.0_f32)
      baseline = context.upload(large)
      large[-1] = 0.5_f32
      index_result = context.read(compare(context, context.upload(large), baseline, false), "diagnostic")
      raise "GPU large worst index" unless ((index_result[8].to_i64 << 16) | index_result[9].to_i64) == 16_777_220

      wave = [0.25_f32, -0.5_f32, 0.0_f32]
      wave_buffer = context.upload(wave)
      window = context.alloc(6)
      context.dispatch("dr3_codec_wave_windows", SOURCE, [wave_buffer, wave_buffer, window], [3_u64], 1)
      statistics = context.read(window, "diagnostic")
      raise "GPU window RMS" unless (statistics[0] - Math.sqrt(0.3125 / 3)).abs < 1e-7
      raise "GPU window peak" unless statistics[2] == 0.5_f32 && statistics[4] == 16384
      puts({format: "drama3-codec-compare-self-test-v1", passes: true,
            checks: ["F64 RMS oracle", "cross-part poison", "exact equality", "nonfinite", "zero reference", "large index", "waveform windows"]}.to_json)
    end

    def self.run : Nil
      if ARGV == ["--self-test"]
        self_test
        return
      end
      if ARGV.size == 4 && ARGV[0] == "--compare-fixtures"
        actual, reference, output = ARGV[1..].map { |path| File.expand_path(path) }
        compare_fixtures(actual, reference, output)
        return
      end
      raise "usage: drama3_codec_cross_compare MODEL_DIR REFERENCE.json OUTPUT.json OUTPUT.wav" unless ARGV.size == 4
      model_dir, reference_path, output_path, wav_path = ARGV.map { |path| File.expand_path(path) }
      raise "output already exists" if File.exists?(output_path) || File.exists?(wav_path)
      input, frames, codes = load_reference(reference_path)
      raise "local codec identity" unless file_sha(File.join(model_dir, "codec.safetensors")) == CODEC_SHA256
      items = input["stages"].as_a
      names = items.map { |item| item["name"].as_s }
      raise "complete stage set required" unless names.sort == expected_names.sort

      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
      ENV.delete("DRAMA3_PROFILE")
      ENV.delete("DRAMA3_METAL_TIMELINE")
      Profile.configure_from_env!
      context = Context.new
      stages = {} of String => Stage
      items.each do |item|
        stage = load_stage(context, File.dirname(reference_path), item)
        stages[stage.name] = stage
      end
      raise "waveform shape" unless stages["waveform"].rows == frames * 2048 && stages["waveform"].channels == 1
      flat = codes.flatten.map(&.to_f32)
      code_buffer = context.upload(flat)
      diagnostics = {} of String => Buffer
      observed_order = [] of String
      waveform_windows : Buffer? = nil
      decoder = CodecResident.new(WeightStore.new(model_dir), context)
      decoder.device_trace = ->(name : String, values : Buffer, rows : Int32, channels : Int32) {
        raise "unexpected or duplicate stage #{name}" unless stages.has_key?(name) && !diagnostics.has_key?(name)
        stage = stages[name]
        raise "stage shape #{name}: #{rows}x#{channels} != #{stage.rows}x#{stage.channels}" unless rows == stage.rows && channels == stage.channels
        raise "mid-decode host readback" unless context.readbacks.empty?
        diagnostics[name] = compare(context, values, stage.reference, name == "waveform")
        observed_order << name
        if name == "waveform"
          windows = (rows + 22049) // 22050
          window_buffer = context.alloc(windows * 6)
          context.dispatch("dr3_codec_wave_windows", SOURCE, [values, stage.reference, window_buffer], [rows.to_u64], windows)
          waveform_windows = window_buffer
        end
        nil
      }

      started = Time.instant
      output = decoder.decode(code_buffer, frames, true)
      decode_seconds = (Time.instant - started).total_seconds
      raise "incomplete observed stages" unless observed_order == expected_names
      raise "decode introduced host readbacks" unless context.readbacks.empty?
      before_reads = context.graph_stats
      raise "decode left GPU work pending" unless before_reads["pending"] == 0
      results = observed_order.map do |name|
        stage_result(stages[name], context.read(diagnostics[name], "diagnostic"))
      end
      window_values = context.read(waveform_windows.not_nil!, "diagnostic")
      samples = context.read(output, "terminal")
      raise "terminal output length" unless samples.size == frames * 2048
      Wav.write(wav_path, samples)
      after_reads = context.graph_stats
      raise "terminal diagnostics added GPU work or waits" unless after_reads == before_reads
      wave = results.last
      report = {
        format: "drama3-codec-cross-comparison-v1", reference_json: reference_path,
        reference_sha256: file_sha(reference_path), codec_sha256: CODEC_SHA256,
        reference_codec_sha256: input["codec"]["sha256"].as_s,
        frames: frames, samples: samples.size, sample_rate: 44100, stages: results,
        stage_gate_passes: results.all?(&.[:passes]), first_failed_stage: results.find { |item| !item[:passes] }.try(&.[:name]),
        one_lsb_gate_passes: wave[:within_one_pcm_lsb],
        full_gate_passes: results.all?(&.[:passes]) && wave[:within_one_pcm_lsb] == true && wave[:actual_peak] > 1e-7 && wave[:reference_peak] > 1e-7,
        nonzero_waveform: {actual: wave[:actual_peak] > 1e-7, reference: wave[:reference_peak] > 1e-7},
        gates: gates,
        waveform_windows: window_results(window_values, samples.size),
        execution: {backend: "resident-metal", precision: "F32", decode_seconds: decode_seconds,
                    reference_upload_bytes: stages.values.sum { |stage| stage.reference.count.to_i64 * 4 },
                    activation_readback_before_decode_completion: false, comparison_reduction: "GPU F32 compensated sums",
                    graph_before_terminal_reads: before_reads, graph_after_terminal_reads: after_reads,
                    resident_readbacks: context.readbacks},
        wav: {path: wav_path, sha256: file_sha(wav_path)},
      }
      File.write(output_path, report.to_json)
      puts({frames: frames, samples: samples.size, stages: results.size, stage_gate_passes: report[:stage_gate_passes],
            first_failed_stage: report[:first_failed_stage], one_lsb_gate_passes: report[:one_lsb_gate_passes],
            max_pcm_error: wave[:max_abs_pcm_error], decode_seconds: decode_seconds}.to_json)
    end
  end

  Drama3CodecCrossCompare.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
