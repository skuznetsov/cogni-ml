require "spec"
require "file_utils"
require "../src/ml/audio/drama3/studio/backend"

private def with_studio_worker_dir(& : String ->)
  dir = File.tempname("drama3-studio-worker")
  Dir.mkdir(dir)
  yield dir
ensure
  FileUtils.rm_rf(dir) if dir
end

private def write_studio_fake_worker(dir : String) : String
  path = File.join(dir, "fake-worker")
  File.write(path, <<-SH)
    #!/bin/sh
    model_dir="$3"
    touch "$model_dir/spawned"
    printf '%s\\n' '{"event":"ready"}'
    IFS= read -r request || exit 1
    if [ ! -e "$model_dir/hung-once" ]; then
      touch "$model_dir/hung-once"
      printf '%s\\n' "$$" > "$model_dir/hung.pid"
      exec /bin/sleep 30
    fi
    printf '%s\\n' '{"event":"result","report":{"ok":true}}'
    SH
  File.chmod(path, 0o700)
  path
end

private def write_studio_reusable_fake_worker(dir : String) : String
  path = File.join(dir, "reusable-fake-worker")
  File.write(path, <<-SH)
    #!/bin/sh
    model_dir="$3"
    printf '%s\\n' "$$" >> "$model_dir/pids"
    printf '%s\\n' '{"event":"ready"}'
    while IFS= read -r request; do
      printf '%s\\n' '{"event":"result","report":{"ok":true}}'
    done
    SH
  File.chmod(path, 0o700)
  path
end

describe ML::Audio::Drama3::Studio::WorkerBackend do
  it "kills and reaps a timed-out worker, then starts a clean one for the next job" do
    with_studio_worker_dir do |dir|
      worker = write_studio_fake_worker(dir)
      backend = ML::Audio::Drama3::Studio::WorkerBackend.new(worker, dir, 1.second)

      expect_raises(ML::Audio::Drama3::Studio::WorkerTimeoutError) do
        backend.generate(JSON.parse("{}"), File.join(dir, "first.wav")) { |_| }
      end

      pid = File.read(File.join(dir, "hung.pid")).strip
      Process.run("/bin/kill", ["-0", pid], output: Process::Redirect::Close,
        error: Process::Redirect::Close).success?.should be_false

      report = backend.generate(JSON.parse("{}"), File.join(dir, "second.wav")) { |_| }
      report["ok"].as_bool.should be_true
    ensure
      backend.try(&.close)
    end
  end

  it "validates a positive response timeout and never spawns after close" do
    with_studio_worker_dir do |dir|
      worker = write_studio_fake_worker(dir)
      expect_raises(ArgumentError) do
        ML::Audio::Drama3::Studio::WorkerBackend.new(worker, dir, Time::Span.zero)
      end

      backend = ML::Audio::Drama3::Studio::WorkerBackend.new(worker, dir, 1.second)
      backend.close
      expect_raises(Exception, "native worker is closed") do
        backend.generate(JSON.parse("{}"), File.join(dir, "unused.wav")) { |_| }
      end
      File.exists?(File.join(dir, "spawned")).should be_false
    ensure
      backend.try(&.close)
    end
  end

  it "reuses the same healthy worker PID across successful requests" do
    with_studio_worker_dir do |dir|
      worker = write_studio_reusable_fake_worker(dir)
      backend = ML::Audio::Drama3::Studio::WorkerBackend.new(worker, dir, 1.second)

      2.times do |index|
        report = backend.generate(JSON.parse({index: index}.to_json), File.join(dir, "take-#{index}.wav")) { |_| }
        report["ok"].as_bool.should be_true
      end

      pids = File.read(File.join(dir, "pids")).lines.map(&.strip)
      pids.size.should eq(1)
      pids.first.to_i64.should be > 0
    ensure
      backend.try(&.close)
    end
  end
end
