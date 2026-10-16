require "option_parser"
require "../src/ml/audio/drama3/studio/server"
require "../src/ml/audio/drama3/studio/engine"

model = ""
data = File.expand_path("~/.local/share/cogni-ml/drama3-studio")
port = 8787
worker = false
OptionParser.parse do |parser|
  parser.banner = "usage: drama3_studio --model DIR [--data DIR] [--port 8787]"
  parser.on("--model DIR", "Pinned MLX-converted DRAMA-3 checkpoint") { |value| model = File.expand_path(value) }
  parser.on("--data DIR", "Local sessions, voice references and audio library") { |value| data = File.expand_path(value) }
  parser.on("--port N", "Loopback HTTP port") { |value| port = value.to_i }
  parser.on("--worker", "Internal persistent native worker") { worker = true }
  parser.on("-h", "--help", "Show usage") { puts parser; exit }
end
abort "--model must contain config.json" unless File.file?(File.join(model, "config.json"))
abort "port must be in 1024..65535" unless port.in?(1024..65535)
ENV["DRAMA3_BACKEND"] ||= "metal"
abort "studio requires DRAMA3_BACKEND=metal" unless ENV["DRAMA3_BACKEND"] == "metal"
ENV["DRAMA3_MOE_BATCH"] ||= "1"
ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] ||= "2"

if worker
  {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
    begin
      engine = ML::Audio::Drama3::Studio::Engine.new(model)
      STDOUT.puts({event: "ready", model_load_seconds: engine.load_seconds}.to_json)
      STDOUT.flush
      while line = STDIN.gets
        raise "worker request too large" if line.bytesize > 128 * 1024
        input = JSON.parse(line)
        report = engine.generate(input["input"], input["output"].as_s) do |message|
          STDOUT.puts({event: "progress", message: message}.to_json)
          STDOUT.flush
        end
        # Only between completed jobs. Inference kernels never wait for GC.
        gc_before = ML::Audio::Drama3::HostProfile.gc_snapshot
        gc_started = Time.instant
        GC.collect
        gc_seconds = (Time.instant - gc_started).total_seconds
        gc_after = ML::Audio::Drama3::HostProfile.gc_snapshot
        STDOUT.puts({event: "result", report: report,
                     worker_gc: {wall_seconds: gc_seconds,
                                 stats: ML::Audio::Drama3::HostProfile.gc_delta(gc_before, gc_after)}}.to_json)
        STDOUT.flush
      end
    rescue ex
      STDOUT.puts({event: "error", message: ex.message || "native inference failed"}.to_json)
      STDOUT.flush
      exit 1
    end
  {% else %}
    abort "studio worker requires a Darwin build with -Ddrama3_metal"
  {% end %}
else
  Dir.mkdir_p(data)
  File.open(File.join(data, ".studio.lock"), "a+") do |lock|
    begin
      lock.flock_exclusive(blocking: false)
    rescue IO::Error
      abort "another studio is already using this data directory"
    end
    store = ML::Audio::Drama3::Studio::Store.new(data)
    backend = ML::Audio::Drama3::Studio::WorkerBackend.new(Process.executable_path.not_nil!, model)
    app = ML::Audio::Drama3::Studio::App.new(store, backend, port, File.basename(model))
    server = HTTP::Server.new { |context| app.call(context) }
    server.bind_tcp("127.0.0.1", port)
    app.start
    {Signal::INT, Signal::TERM}.each do |signal|
      signal.trap do
        app.close
        server.close
      end
    end
    puts "DRAMA Studio: http://127.0.0.1:#{port}"
    puts "Local library: #{data}"
    server.listen
  ensure
    app.try(&.close)
  end
end
