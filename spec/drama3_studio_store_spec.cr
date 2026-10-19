require "./spec_helper"
require "file_utils"
require "../src/ml/audio/drama3/studio/store"

private def drama3_studio_input(text : String, revision : Int32? = nil) : JSON::Any
  fields = {
    "text"        => JSON::Any.new(text),
    "voice_id"    => JSON::Any.new(nil),
    "seed"        => JSON::Any.new(7_i64),
    "max_frames"  => JSON::Any.new(128_i64),
    "temperature" => JSON::Any.new(0.8_f64),
    "top_k"       => JSON::Any.new(40_i64),
    "top_p"       => JSON::Any.new(0.9_f64),
  } of String => JSON::Any
  fields["revision"] = JSON::Any.new(revision.to_i64) if revision
  JSON::Any.new(fields)
end

private def drama3_studio_json_fields(**values) : JSON::Any
  JSON.parse(values.to_h.to_json)
end

private def drama3_studio_finish(store : ML::Audio::Drama3::Studio::Store, job : JSON::Any, filename : String, stopped_on_eos : Bool = true) : JSON::Any
  path = File.join(store.root, "assets", filename)
  File.write(path, "test audio")
  store.finish_job(
    job["id"].as_s,
    filename,
    1.25_f64,
    JSON.parse({"stopped_on_eos" => stopped_on_eos}.to_json),
  )
end

describe ML::Audio::Drama3::Studio::Store do
  it "persists automatic windows and snapshots their frame ceiling without changing old defaults" do
    root = File.tempname("drama3-studio-auto-window")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      sid = store.create_session("Windows")["id"].as_s
      segment = store.save_segment(sid, nil, drama3_studio_input("Manual", 0))
      segment["auto_frames"].as_bool.should be_false
      gid = segment["id"].as_s
      segment = store.save_segment(sid, gid, drama3_studio_json_fields(
        text: "Automatic", revision: 1, auto_frames: true, max_frames: 4096))
      job = store.enqueue_generation(sid, gid)
      store.save_segment(sid, gid, drama3_studio_json_fields(text: "Changed", revision: 2, auto_frames: false))
      job["input"]["auto_frames"].as_bool.should be_true
      job["input"]["max_frames"].as_i.should eq(4096)
      reloaded = ML::Audio::Drama3::Studio::Store.new(root)
      reloaded.state["sessions"][0]["segments"][0]["auto_frames"].as_bool.should be_false
      expect_raises(ArgumentError, /auto_frames/) do
        reloaded.save_segment(sid, gid, drama3_studio_json_fields(text: "Invalid", revision: 3, auto_frames: "yes"))
      end
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "checks segment revisions and rejects out-of-range sampling settings" do
    root = File.tempname("drama3-studio-revision")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      session = store.create_session("Session")
      session_id = session["id"].as_s
      segment = store.save_segment(session_id, nil, drama3_studio_input("First take", 0))
      segment_id = segment["id"].as_s
      segment["revision"].as_i.should eq(1)
      segment["max_frames"].as_i.should eq(128)

      expect_raises(ML::Audio::Drama3::Studio::Store::ConflictError, /revision conflict/) do
        store.save_segment(session_id, segment_id, drama3_studio_input("Stale edit", 0))
      end
      expect_raises(ArgumentError, /max_frames/) do
        store.save_segment(session_id, segment_id, drama3_studio_json_fields(
          text: "Invalid frames",
          revision: 1,
          max_frames: 4097,
        ))
      end

      store.state["sessions"][0]["segments"][0]["revision"].as_i.should eq(1)
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "retains late results without selecting them over a newer edit" do
    root = File.tempname("drama3-studio-late-result")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      session_id = store.create_session("Session")["id"].as_s
      original = store.save_segment(session_id, nil, drama3_studio_input("Original", 0))
      segment_id = original["id"].as_s
      queued = store.enqueue_generation(session_id, segment_id)
      running = store.claim_job.not_nil!
      running["id"].as_s.should eq(queued["id"].as_s)
      expect_raises(ML::Audio::Drama3::Studio::Store::ConflictError, /already queued/) do
        store.enqueue_generation(session_id, segment_id)
      end
      store.update_job_progress(running["id"].as_s, "Decoding frame 12")["detail"].as_s.should eq("Decoding frame 12")

      edited = store.save_segment(session_id, segment_id, drama3_studio_input("Edited", 1))
      edited["revision"].as_i.should eq(2)
      edited["selected_take_id"].raw.should be_nil

      late_take = drama3_studio_finish(store, running, "late-take.wav", false)
      late_take["kind"].as_s.should eq("take")
      late_take["revision"].as_i.should eq(1)
      late_take["stopped_on_eos"].as_bool.should be_false
      late_take["truncated"].as_bool.should be_true
      store.state["sessions"][0]["segments"][0]["selected_take_id"].raw.should be_nil
      store.state["jobs"][0]["status"].as_s.should eq("complete")

      selected = store.select_take(session_id, segment_id, late_take["id"].as_s)
      selected["selected_take_id"].as_s.should eq(late_take["id"].as_s)
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "snapshots exports in segment order and protects selected takes from deletion" do
    root = File.tempname("drama3-studio-export-order")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      session_id = store.create_session("Ordered session")["id"].as_s
      first = store.save_segment(session_id, nil, drama3_studio_input("First", 0))
      second = store.save_segment(session_id, nil, drama3_studio_input("Second", 0))

      first_job = store.claim_job
      first_job.should be_nil
      store.enqueue_generation(session_id, first["id"].as_s)
      generated_first = drama3_studio_finish(store, store.claim_job.not_nil!, "take-first.wav")
      store.enqueue_generation(session_id, second["id"].as_s)
      generated_second = drama3_studio_finish(store, store.claim_job.not_nil!, "take-second.wav")

      expect_raises(ML::Audio::Drama3::Studio::Store::ConflictError, /selected by a segment/) do
        store.delete_asset(generated_first["id"].as_s)
      end

      export = store.enqueue_export(session_id, "wav")
      export["input"]["asset_ids"].as_a.map(&.as_s).should eq([
        generated_first["id"].as_s,
        generated_second["id"].as_s,
      ])
      # The immutable export snapshot must keep a source alive even after the
      # user edits the segment and clears its selection.
      store.save_segment(session_id, second["id"].as_s, drama3_studio_input("Second edited", 1))
      export_job = store.claim_job.not_nil!
      expect_raises(ML::Audio::Drama3::Studio::Store::ConflictError, /active job/) do
        store.delete_asset(generated_second["id"].as_s)
      end
      export_job["input"]["format"].as_s.should eq("wav")
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "marks queued and running jobs failed on restart and reloads committed state" do
    root = File.tempname("drama3-studio-restart")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      session_id = store.create_session("Restart session")["id"].as_s
      first = store.save_segment(session_id, nil, drama3_studio_input("Queued", 0))
      second = store.save_segment(session_id, nil, drama3_studio_input("Running", 0))
      queued = store.enqueue_generation(session_id, first["id"].as_s)
      pending = store.enqueue_generation(session_id, second["id"].as_s)
      store.claim_job.not_nil!["id"].as_s.should eq(queued["id"].as_s)
      store.claim_job.not_nil!["id"].as_s.should eq(pending["id"].as_s)

      restarted = ML::Audio::Drama3::Studio::Store.new(root)
      jobs = restarted.state["jobs"].as_a
      jobs.map { |job| job["status"].as_s }.should eq(["failed", "failed"])
      jobs.all? { |job| job["error"].as_s == "interrupted by restart" }.should be_true
      restarted.enqueue_generation(session_id, second["id"].as_s)["status"].as_s.should eq("queued")
      restarted.state["sessions"][0]["id"].as_s.should eq(session_id)
      pending["status"].as_s.should eq("queued")
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "does not persist an empty claim when no job is queued" do
    root = File.tempname("drama3-studio-empty-claim")
    manifest_path = File.join(root, "studio.json")
    original_manifest : String? = nil
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      store.create_session("No queued work")
      original_manifest = File.read(manifest_path)

      # A manifest obstruction makes any unnecessary persistence fail. An idle
      # claim should return nil without touching the durable manifest.
      File.delete(manifest_path)
      Dir.mkdir(manifest_path)
      store.claim_job.should be_nil
    ensure
      if File.exists?(root)
        FileUtils.rm_rf(manifest_path) if File.directory?(manifest_path)
        File.write(manifest_path, original_manifest.not_nil!) if original_manifest && !File.exists?(manifest_path)
        FileUtils.rm_rf(root)
      end
    end
  end

  it "rejects traversal IDs and prevents deleting active reference assets" do
    root = File.tempname("drama3-studio-paths")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      expect_raises(ArgumentError, /invalid asset id/) { store.asset_path("../outside.wav") }
      expect_raises(ArgumentError, /safe basename/) do
        store.create_voice("Reference", "A short transcript", "../outside.wav", 1.0_f64)
      end

      filename = "reference-voice.wav"
      File.write(File.join(store.root, "assets", filename), "reference audio")
      voice = store.create_voice("Reference", "A short transcript", filename, 1.0_f64)
      asset_id = store.state["assets"][0]["id"].as_s
      store.voice_path(voice["id"].as_s).should eq(File.join(store.root, "assets", filename))
      expect_raises(ML::Audio::Drama3::Studio::Store::ConflictError, /active voice/) do
        store.delete_asset(asset_id)
      end

      store.archive_voice(voice["id"].as_s)
      blocked_trash = File.join(store.root, "trash")
      File.write(blocked_trash, "cannot create trash directory")
      expect_raises(Exception) { store.delete_asset(asset_id) }
      store.state["assets"].as_a.any? { |asset| asset["id"].as_s == asset_id }.should be_true
      File.read(File.join(store.root, "assets", filename)).should eq("reference audio")

      File.delete(blocked_trash)
      store.delete_asset(asset_id)
      expect_raises(KeyError, /unknown asset/) { store.asset_path(asset_id) }
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "restores a staged asset deletion when the manifest still owns the asset on restart" do
    root = File.tempname("drama3-studio-delete-recovery")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      filename = "recovery-voice.wav"
      File.write(File.join(store.root, "assets", filename), "reference audio")
      voice = store.create_voice("Reference", "A short transcript", filename, 1.0_f64)
      store.archive_voice(voice["id"].as_s)
      asset_id = store.state["assets"][0]["id"].as_s

      # Model a process stop after staging the file but before the manifest
      # transaction commits. The manifest remains the authority for recovery.
      trash_dir = File.join(store.root, "trash")
      FileUtils.mkdir_p(trash_dir)
      tombstone = File.join(trash_dir, "#{asset_id}.pending")
      File.rename(File.join(store.root, "assets", filename), tombstone)

      restarted = ML::Audio::Drama3::Studio::Store.new(root)
      File.read(restarted.asset_path(asset_id)).should eq("reference audio")
      File.exists?(tombstone).should be_false
      restarted.state["assets"].as_a.any? { |asset| asset["id"].as_s == asset_id }.should be_true
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "persists atomically and returns detached state snapshots" do
    root = File.tempname("drama3-studio-atomic")
    begin
      store = ML::Audio::Drama3::Studio::Store.new(root)
      created = store.create_session("Durable session")
      created["created_at"].as_i64.should be > 1_000_000_000_000_i64
      snapshot = store.state
      snapshot["sessions"][0].as_h["name"] = JSON::Any.new("caller mutation")
      store.state["sessions"][0]["name"].as_s.should eq("Durable session")

      restarted = ML::Audio::Drama3::Studio::Store.new(root)
      restarted.state["sessions"][0]["id"].as_s.should eq(created["id"].as_s)
      JSON.parse(File.read(File.join(root, "studio.json"))).as_h["sessions"].as_a.size.should eq(1)
      Dir.children(root).any? { |name| name.starts_with?("studio.json.tmp-") }.should be_false
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end

  it "rejects an oversized persisted document before accepting more edits" do
    root = File.tempname("drama3-studio-bounds")
    begin
      FileUtils.mkdir_p(root)
      segments = JSON::Any.new(Array.new(1_001) { JSON::Any.new({} of String => JSON::Any) })
      session = JSON::Any.new({
        "id"       => JSON::Any.new("00000000-0000-0000-0000-000000000001"),
        "segments" => segments,
      } of String => JSON::Any)
      manifest = JSON::Any.new({
        "version"  => JSON::Any.new(1_i64),
        "sessions" => JSON::Any.new([session]),
        "voices"   => JSON::Any.new([] of JSON::Any),
        "assets"   => JSON::Any.new([] of JSON::Any),
        "jobs"     => JSON::Any.new([] of JSON::Any),
      } of String => JSON::Any)
      File.write(File.join(root, "studio.json"), manifest.to_json)

      expect_raises(ArgumentError, /segments per session limit/) do
        ML::Audio::Drama3::Studio::Store.new(root)
      end
    ensure
      FileUtils.rm_rf(root) if File.exists?(root)
    end
  end
end
