require "spec"
require "file_utils"
require "../src/ml/audio/drama3/studio/model_lease"

private def with_drama3_model_lease_root(& : String, String, String ->)
  root = File.tempname("drama3-model-lease")
  model = File.join(root, "model")
  other_model = File.join(root, "other-model")
  FileUtils.mkdir_p(model)
  FileUtils.mkdir_p(other_model)
  yield root, model, other_model
ensure
  FileUtils.rm_rf(root) if root
end

describe ML::Audio::Drama3::Studio::ModelLease do
  it "blocks another lease for the same canonical model until close" do
    with_drama3_model_lease_root do |root, model, _|
      lock_dir = File.join(root, "leases")
      lease = ML::Audio::Drama3::Studio::ModelLease.new(model, lock_dir: lock_dir)
      begin
        lease.held?.should be_true
        expect_raises(ML::Audio::Drama3::Studio::ModelLease::AlreadyOwned, /model already owned/) do
          ML::Audio::Drama3::Studio::ModelLease.acquire(model, lock_dir: lock_dir)
        end
        File.exists?(lease.lock_path).should be_true
        (File.info(lock_dir).permissions.to_i & 0o077).should eq(0)
        (File.info(lease.lock_path).permissions.to_i & 0o077).should eq(0)
      ensure
        lease.close
      end

      ML::Audio::Drama3::Studio::ModelLease.acquire(model, lock_dir: lock_dir).close
    end
  end

  it "uses the real directory for symlink aliases and permits a different model" do
    with_drama3_model_lease_root do |root, model, other_model|
      alias_path = File.join(root, "model-alias")
      File.symlink(model, alias_path)
      lock_dir = File.join(root, "leases")

      real_lease = ML::Audio::Drama3::Studio::ModelLease.acquire(model, lock_dir: lock_dir)
      alias_path_lease = nil.as(ML::Audio::Drama3::Studio::ModelLease?)
      other_lease = nil.as(ML::Audio::Drama3::Studio::ModelLease?)
      begin
        ML::Audio::Drama3::Studio::ModelLease.lock_path_for(model, lock_dir).should eq(
          ML::Audio::Drama3::Studio::ModelLease.lock_path_for(alias_path, lock_dir)
        )
        expect_raises(ML::Audio::Drama3::Studio::ModelLease::AlreadyOwned, /model already owned/) do
          alias_path_lease = ML::Audio::Drama3::Studio::ModelLease.acquire(alias_path, lock_dir: lock_dir)
        end

        other_lease = ML::Audio::Drama3::Studio::ModelLease.acquire(other_model, lock_dir: lock_dir)
        other_lease.not_nil!.held?.should be_true
      ensure
        alias_path_lease.try(&.close)
        other_lease.try(&.close)
        real_lease.close
      end
    end
  end

  it "blocks acquisition across processes and recovers after the holder exits" do
    with_drama3_model_lease_root do |root, model, _|
      lock_dir = File.join(root, "leases")
      FileUtils.mkdir_p(lock_dir, 0o700)
      lock_path = ML::Audio::Drama3::Studio::ModelLease.lock_path_for(model, lock_dir)
      holder = Process.new("/usr/bin/lockf", ["-k", lock_path, "/bin/sleep", "30"])

      begin
        locked = false
        100.times do
          probe = File.open(lock_path, "a+")
          begin
            probe.flock_exclusive(blocking: false)
            probe.flock_unlock
          rescue IO::Error
            locked = true
            break
          ensure
            probe.close
          end
          sleep 10.milliseconds
        end
        locked.should be_true

        expect_raises(ML::Audio::Drama3::Studio::ModelLease::AlreadyOwned, /model already owned/) do
          ML::Audio::Drama3::Studio::ModelLease.acquire(model, lock_dir: lock_dir)
        end

        holder.terminate
        holder.wait
        lease = ML::Audio::Drama3::Studio::ModelLease.acquire(model, lock_dir: lock_dir)
        lease.held?.should be_true
        lease.close
      ensure
        if holder.exists?
          holder.terminate
          holder.wait
        end
      end
    end
  end

  it "defaults to a stable per-user cache directory" do
    ML::Audio::Drama3::Studio::ModelLease.default_lock_dir.should eq(
      File.join(Path.home.to_s, ".cache", "cogni-ml", "drama3-model-leases")
    )
  end
end
