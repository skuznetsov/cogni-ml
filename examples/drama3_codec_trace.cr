require "../src/ml/audio/drama3/codec"
require "../src/ml/audio/drama3/wav"

# Decode identical code frames on either backend, without rerunning sampling.
# Use '-' for TRACE_JSON when only profiling / WAV output is required.
abort "usage: drama3_codec_trace MODEL_DIR CODES_JSON TRACE_JSON [WAV_OUTPUT]" unless (3..4).includes?(ARGV.size)
model_path, codes_path, trace_path = ARGV
report = JSON.parse(File.read(codes_path))
frames = report["frames"].as_a.map { |frame| frame.as_a.map(&.as_i) }
abort "no code frames" if frames.empty?
abort "inconsistent code frame widths" unless frames.all? { |frame| frame.size == frames[0].size }
books = Array(Array(Int32)).new(frames[0].size) { |book| frames.map(&.[book]) }
records = [] of NamedTuple(name: String, shape: Array(Int32), dtype: String, values: ML::Audio::Drama3::Vector)
ML::Audio::Drama3::Profile.configure_from_env!
ML::Audio::Drama3::MetalBackend.enabled?
started = Time.instant
store = ML::Audio::Drama3::WeightStore.new(model_path)
codec = ML::Audio::Drama3::Codec.new(store)
unless trace_path == "-"
  codec.trace = ->(name : String, values : ML::Audio::Drama3::Vector) {
    records << {name: name, shape: [values.size], dtype: "float32", values: values}
    nil
  }
end
samples = codec.decode(books)
puts({samples: samples.size, elapsed_seconds: (Time.instant - started).total_seconds}.to_json)
File.write(trace_path, {format: "drama3-reference-traces-v1", traces: records}.to_json) unless trace_path == "-"
if wav_path = ARGV[3]?
  abort "WAV output already exists" if File.exists?(wav_path)
  ML::Audio::Drama3::Wav.write(wav_path, samples)
end
ML::Audio::Drama3::Profile.write_from_env
