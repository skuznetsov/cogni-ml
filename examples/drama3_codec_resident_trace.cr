require "json"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/wav"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3
  abort "usage: drama3_codec_resident_trace MODEL_DIR REFERENCE_JSON OUTPUT_JSON [WAV_OUTPUT]" unless (3..4).includes?(ARGV.size)

  model_dir = File.expand_path(ARGV[0])
  reference_path = File.expand_path(ARGV[1])
  output_path = File.expand_path(ARGV[2])
  wav_path = ARGV[3]?.try { |path| File.expand_path(path) }
  abort "output already exists: #{output_path}" if File.exists?(output_path)
  abort "WAV output already exists: #{wav_path}" if wav_path && File.exists?(wav_path)

  ENV["DRAMA3_BACKEND"] = "metal"
  Profile.configure_from_env!
  abort "DRAMA-3 Metal backend is unavailable" unless MetalBackend.enabled?

  reference = JSON.parse(File.read(reference_path))
  codebooks = reference["codes"].as_a.map do |book|
    book.as_a.map(&.as_i64.to_i32)
  end
  abort "reference must contain ten DAC codebooks" unless codebooks.size == 10
  frames = codebooks[0].size
  abort "reference codebooks must contain at least one frame" unless frames > 0
  abort "reference DAC codebooks have inconsistent frame counts" unless codebooks.all? { |book| book.size == frames }

  # CodecResident's codebook-major layout is [c0..c9, frames]. The reference
  # helper already emits this exact ordering; Generator's main-token column is
  # intentionally absent from the DAC input.
  flat_codes = Array(Float32).new(frames * codebooks.size, 0_f32)
  codebooks.each_with_index do |book, book_index|
    book.each_with_index { |token, frame| flat_codes[book_index * frames + frame] = token.to_f32 }
  end

  expected = reference["traces"].as_h
  actual = {} of String => Vector
  context = Resident::Context.new
  decoder = CodecResident.new(WeightStore.new(model_dir), context)
  decoder.trace = ->(name : String, values : Vector) {
    # Internal resident-layer callbacks are useful when explicitly requested,
    # but the pinned MLX helper's public gates cover its stage boundary names.
    actual[name] = values.dup if expected.has_key?(name)
    nil
  }

  started = Time.instant
  output = decoder.decode(context.upload(flat_codes), frames, true)
  samples = context.read(output, "terminal")
  raise "codec waveform is empty" if samples.empty?
  raise "codec waveform contains non-finite values" unless samples.all?(&.finite?)
  raise "codec reference stage set mismatch; missing=#{(expected.keys - actual.keys).sort}, unexpected=#{(actual.keys - expected.keys).sort}" unless expected.keys.sort == actual.keys.sort

  comparisons = [] of NamedTuple(name: String, max_abs: Float64, rms: Float64, normalized_rms: Float64, worst_index: Int32)
  failures = [] of String
  expected.each do |name, expected_values|
    reference_values = expected_values.as_a
    observed = actual[name]
    raise "#{name}: length mismatch #{observed.size} != #{reference_values.size}" unless observed.size == reference_values.size
    max_error = 0.0_f64
    squared_error_sum = 0.0_f64
    reference_squared_sum = 0.0_f64
    worst_index = 0
    reference_values.each_with_index do |item, index|
      target = case raw = item.raw
               when Float64 then raw.to_f32
               when Int64   then raw.to_f32
               else              raise ArgumentError.new("#{name}[#{index}] is not numeric")
               end
      value = observed[index]
      error = (value - target).abs.to_f64
      squared_error_sum += error * error
      reference_squared_sum += target.to_f64 * target
      if error > max_error
        max_error = error
        worst_index = index
      end
      if name == "waveform"
        tolerance = {5.0e-5_f32, target.abs * 1.0e-3_f32}.max
        failures << "waveform[#{index}] error #{error} exceeds tolerance #{tolerance}" if error > tolerance
      end
    end
    rms = Math.sqrt(squared_error_sum / observed.size).to_f64
    reference_rms = Math.sqrt(reference_squared_sum / observed.size).to_f64
    normalized_rms = reference_rms > 0.0 ? rms / reference_rms : (rms == 0.0 ? 0.0 : Float64::INFINITY)
    comparisons << {name: name, max_abs: max_error, rms: rms, normalized_rms: normalized_rms, worst_index: worst_index}
    if name != "waveform" && (max_error > 1.0e-3 || rms > 2.0e-4 || normalized_rms > 2.0e-4)
      failures << "#{name} aggregate error max_abs=#{max_error}, rms=#{rms}, normalized_rms=#{normalized_rms}"
    end
  end
  raise "resident codec parity failures:\n#{failures.join("\n")}" unless failures.empty?

  if path = wav_path
    Wav.write(path, samples)
  end
  File.write(output_path, JSON.build do |json|
    json.object do
      json.field "mode", "resident-metal"
      json.field "codes", codebooks
      json.field "traces" do
        json.object do
          actual.keys.sort.each do |name|
            json.field name do
              json.array { actual[name].each { |value| json.number value.to_f64 } }
            end
          end
        end
      end
      json.field "comparisons", comparisons
      json.field "samples", samples.size
      json.field "elapsed_seconds", (Time.instant - started).total_seconds
      json.field "resident_readbacks", context.readbacks
      json.field "resident_submissions", context.submissions
    end
  end)
  Profile.write_from_env
  puts({frames: frames, samples: samples.size, elapsed_seconds: (Time.instant - started).total_seconds,
        trace_count: comparisons.size, max_stage_error: comparisons.reject { |entry| entry[:name] == "waveform" }.max_of(&.[:max_abs]),
        max_waveform_error: comparisons.find { |entry| entry[:name] == "waveform" }.not_nil![:max_abs],
        resident_readbacks: context.readbacks}.to_json)
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
