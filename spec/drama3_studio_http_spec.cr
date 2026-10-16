require "spec"
require "http/client"
require "file_utils"
require "../src/ml/audio/drama3/studio/server"

private class StudioTestBackend < ML::Audio::Drama3::Studio::Backend
  property fail = false
  getter calls = 0

  def generate(input : JSON::Any, output : String, &progress : String -> Nil) : JSON::Any
    @calls += 1
    raise "injected backend failure" if @fail
    progress.call("test progress")
    ML::Audio::Drama3::Wav.write(output, Array(Float32).new(4410, 0.25_f32))
    JSON.parse({seconds: 0.1, stopped_on_eos: false, frames: [[1, 2]], sampler_inputs: {sha256: "fixture"}}.to_json)
  end

  def close : Nil
  end
end

private class StudioQueueTestStore < ML::Audio::Drama3::Studio::Store
  property claim_failures = 0
  property record_failures = 0

  def claim_job : JSON::Any?
    if @claim_failures > 0
      @claim_failures -= 1
      raise IO::Error.new("injected claim persistence failure")
    end
    super
  end

  def fail_job(job_id : String, message : String) : Nil
    if @record_failures > 0
      @record_failures -= 1
      raise IO::Error.new("injected failure persistence failure")
    end
    super
  end
end

private class StudioTestServer
  getter base : String
  getter backend : StudioTestBackend
  getter store : StudioQueueTestStore
  @app : ML::Audio::Drama3::Studio::App
  @server : HTTP::Server
  @dir : String

  def initialize
    @dir = File.tempname("drama3-studio-http")
    @store = StudioQueueTestStore.new(@dir)
    @backend = StudioTestBackend.new
    app = nil.as(ML::Audio::Drama3::Studio::App?)
    @server = HTTP::Server.new { |context| app.not_nil!.call(context) }
    address = @server.bind_unused_port("127.0.0.1")
    @base = "http://127.0.0.1:#{address.port}"
    @app = app = ML::Audio::Drama3::Studio::App.new(@store, @backend, address.port, "test-model")
    @app.start
    spawn { @server.listen }
  end

  def request(method, path, value = "{}", headers = HTTP::Headers.new)
    headers["Origin"] = @base unless headers.has_key?("Origin")
    headers["Content-Type"] = "application/json" unless headers.has_key?("Content-Type")
    HTTP::Client.exec(method, @base + path, headers, value)
  end

  def wait_job(id : String) : JSON::Any
    300.times do
      job = @store.state["jobs"].as_a.find { |item| item["id"].as_s == id }.not_nil!
      return job if job["status"].as_s.in?("complete", "failed")
      sleep 10.milliseconds
    end
    raise "test job timed out"
  end

  def close
    @app.close
    @server.close
    FileUtils.rm_rf(@dir)
  end
end

describe ML::Audio::Drama3::Studio::App do
  it "recovers its queue after claim and failure-record I/O errors without retrying inference" do
    server = StudioTestServer.new
    server.store.claim_failures = 1
    server.store.record_failures = 1
    server.backend.fail = true
    session = server.store.create_session("Recovery")
    sid = session["id"].as_s
    segment = server.store.save_segment(sid, nil, JSON.parse({text: "Hello.", seed: 1, max_frames: 32,
                                                              temperature: 0.8, top_k: 30, top_p: 0.9, voice_id: nil}.to_json))
    path = "/api/sessions/#{sid}/segments/#{segment["id"].as_s}/generate"
    failed = JSON.parse(server.request("POST", path).body)
    server.wait_job(failed["id"].as_s)["status"].as_s.should eq("failed")
    server.backend.calls.should eq(1)
    server.backend.fail = false
    next_job = JSON.parse(server.request("POST", path).body)
    server.wait_job(next_job["id"].as_s)["status"].as_s.should eq("complete")
    server.backend.calls.should eq(2)
  ensure
    server.try(&.close)
  end

  it "does not spawn a native worker after closing its backend" do
    backend = ML::Audio::Drama3::Studio::WorkerBackend.new("/usr/bin/false", "/missing")
    backend.close
    expect_raises(Exception, "native worker is closed") do
      backend.generate(JSON.parse("{}"), "/unused.wav") { |_| }
    end
  end

  it "rejects cross-origin mutation, DNS rebinding, traversal and excessive requests" do
    server = StudioTestServer.new
    server.request("POST", "/api/sessions", "{\"name\":\"safe\"}", HTTP::Headers{"Origin" => "http://attacker.invalid"}).status_code.should eq(403)
    server.request("GET", "/api/state", "", HTTP::Headers{"Host" => "attacker.invalid"}).status_code.should eq(403)
    server.request("GET", "/api/audio/%2e%2e%2fdb.json", "").status_code.should_not eq(200)
    server.request("POST", "/api/sessions", {name: "x" * 131_073}.to_json).status_code.should eq(400)
    server.store.state["sessions"].as_a.should be_empty
    server.request("GET", "/", "").body.should contain("html")
  ensure
    server.try(&.close)
  end

  it "keeps generation, references, range playback and ordered export usable through HTTP" do
    server = StudioTestServer.new
    session = JSON.parse(server.request("POST", "/api/sessions", {name: "First session"}.to_json).body)
    sid = session["id"].as_s
    segment = JSON.parse(server.request("POST", "/api/sessions/#{sid}/segments",
      {text: "Hello.", seed: 1, max_frames: 32, temperature: 0.8, top_k: 30, top_p: 0.9, voice_id: nil}.to_json).body)
    gid = segment["id"].as_s
    job = JSON.parse(server.request("POST", "/api/sessions/#{sid}/segments/#{gid}/generate").body)
    server.wait_job(job["id"].as_s)["status"].as_s.should eq("complete")
    state = server.store.state
    asset = state["assets"].as_a.find { |item| item["kind"].as_s == "take" }.not_nil!
    aid = asset["id"].as_s
    audio = server.request("GET", "/api/audio/#{aid}", "", HTTP::Headers{"Range" => "bytes=0-11"})
    audio.status_code.should eq(206)
    audio.body.bytesize.should eq(12)
    audio.body.should start_with("RIFF")
    server.request("GET", "/api/audio/#{aid}", "", HTTP::Headers{"Range" => "bytes=999999999-"}).status_code.should eq(416)
    server.request("DELETE", "/api/assets/#{aid}").status_code.should eq(409)
    wav = File.read(server.store.asset_path(aid))
    voice = JSON.parse(server.request("POST", "/api/voices?name=Reference&transcript=Hello", wav,
      HTTP::Headers{"Content-Type" => "audio/wav"}).body)
    voice["transcript"].as_s.should eq("Hello")
    server.request("GET", "/api/voice-audio/#{voice["id"].as_s}", "").status_code.should eq(200)
    exported = JSON.parse(server.request("POST", "/api/sessions/#{sid}/export", {format: "mp3"}.to_json).body)
    server.wait_job(exported["id"].as_s)["status"].as_s.should eq("complete")
    state = server.store.state
    state["assets"].as_a.count { |item| item["kind"].as_s == "export" && item["format"].as_s == "mp3" }.should eq(1)
    server.backend.fail = true
    failed = JSON.parse(server.request("POST", "/api/sessions/#{sid}/segments/#{gid}/generate").body)
    server.wait_job(failed["id"].as_s)["status"].as_s.should eq("failed")
    server.store.state["assets"].as_a.count { |item| item["kind"].as_s == "take" }.should eq(1)
  ensure
    server.try(&.close)
  end
end
