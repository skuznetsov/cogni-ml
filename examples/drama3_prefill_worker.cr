# Serial diagnostic worker: change only the native prefill chunk between jobs.
# Static GPU weights/pipelines stay warm; each Engine job creates fresh states.
require "../src/ml/audio/drama3/studio/engine"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  begin
    raise "usage: drama3_prefill_worker MODEL_DIR" unless ARGV.size == 1
    raise "component profiling introduces waits" if ML::Audio::Drama3::Profile.enabled?
    engine = ML::Audio::Drama3::Studio::Engine.new(File.expand_path(ARGV[0]))
    STDOUT.puts({event: "ready", model_load_seconds: engine.load_seconds}.to_json)
    STDOUT.flush
    while line = STDIN.gets
      raise "worker request too large" if line.bytesize > 128 * 1024
      request = JSON.parse(line)
      chunk = request["chunk"].as_i
      raise "chunk must be 1..32" unless chunk.in?(1..32)
      ENV["DRAMA3_PREFILL_CHUNK"] = chunk.to_s
      STDERR.puts({event: "prefill_request_begin", id: request["id"], chunk: chunk}.to_json)
      report = engine.generate(request["input"], request["output"].as_s) do |message|
        STDOUT.puts({event: "progress", message: message}.to_json)
        STDOUT.flush
      end
      # Charge terminal collection to its own request. Publishing a result
      # before collecting would leak this job's cost into the next job's timer.
      GC.collect
      STDERR.puts({event: "prefill_request_end", id: request["id"], chunk: chunk}.to_json)
      STDOUT.puts({event: "result", id: request["id"], chunk: chunk, report: report}.to_json)
      STDOUT.flush
    end
  rescue ex
    STDOUT.puts({event: "error", message: ex.message || "prefill probe failed"}.to_json)
    STDOUT.flush
    exit 1
  end
{% else %}
  abort "prefill probe requires -Ddrama3_metal on macOS"
{% end %}
