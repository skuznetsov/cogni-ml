# Full-length DAC memory probe on explicitly replayed saved codes. This is not
# text generation, listening evidence or a longer-context language-model gate.
require "digest/sha256"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/window_memory"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3
  abort "usage: drama3_window_probe MODEL SAVED_NATIVE_REPORT FRAMES OUTPUT_JSON" unless ARGV.size == 4
  model_path, input_path, frame_text, output_path = ARGV
  abort "output exists" if File.exists?(output_path)
  frames = frame_text.to_i
  abort "frames must be in 1..4096" unless frames.in?(1..4096)
  input_bytes = File.read(input_path)
  input = JSON.parse(input_bytes)
  saved = input["frames"].as_a.map { |row| row.as_a.map(&.as_i) }
  abort "saved codes must contain 11 IDs per frame" unless !saved.empty? && saved.all? { |row| row.size == 11 }
  codes = Array(Float32).new(frames * 10) do |i|
    book = i // frames
    frame = i % frames
    code = saved[frame % saved.size][book + 1]
    limit = book == 0 ? 4096 : 1024
    abort "invalid code" unless code >= 0 && code < limit
    code.to_f32
  end
  before = WindowMemory.snapshot
  store = WeightStore.new(model_path)
  context = Resident::Context.new
  decoder = CodecResident.new(store, context)
  started = Time.instant
  output = decoder.decode(context.upload(codes), frames, true)
  seconds = (Time.instant - started).total_seconds
  after_decode = WindowMemory.snapshot
  graph_before_read = context.graph_stats
  samples = context.read(output, "terminal")
  graph_after_read = context.graph_stats
  raise "terminal read submitted work" unless graph_before_read == graph_after_read
  raise "wrong output length" unless samples.size == frames * 2048
  hash = Digest::SHA256.new
  word = Bytes.new(sizeof(Float32))
  samples.each do |value|
    IO::ByteFormat::LittleEndian.encode(value, word)
    hash.update(word)
  end
  report = {
    format: "drama3-window-dac-probe-v1", model_path: model_path,
    input_report: input_path, input_sha256: Digest::SHA256.hexdigest(input_bytes),
    input_mode: frames <= saved.size ? "saved prefix" : "cyclic replay of saved codes",
    saved_frames: saved.size, frames: frames, samples: samples.size, seconds: seconds,
    output_f32_sha256: hash.hexfinal, readbacks: context.readbacks,
    graph_before_read: graph_before_read, graph_after_read: graph_after_read,
    allocation_stats: context.allocation_stats,
    dac_scratch: decoder.dac_scratch_stats,
    memory_before: before, memory_after_decode: after_decode, memory_after_read: WindowMemory.snapshot,
    scratch_requested: ENV["DRAMA3_DAC_SCRATCH"]? || "0",
    scope: "Full-length DAC allocation and terminal output on saved code replay; no new inference synchronization. No language-model or voice-quality claim.",
  }
  File.write(output_path, report.to_pretty_json)
  puts report.to_json
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
