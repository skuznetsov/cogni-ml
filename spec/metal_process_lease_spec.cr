require "./spec_helper"
require "../src/ml/metal/process_lease"

describe ML::Metal::ProcessLease do
  it "fails closed while another process owns the GPU lease and recovers after exit" do
    path = File.tempname("cogni-metal-process", ".lock")
    holder = Process.new("/usr/bin/lockf", ["-k", path, "/bin/sleep", "5"])

    begin
      externally_locked = false
      100.times do
        probe = File.open(path, "a+")
        begin
          probe.flock_exclusive(blocking: false)
          probe.flock_unlock
        rescue IO::Error
          externally_locked = true
          break
        ensure
          probe.close
        end
        sleep 10.milliseconds
      end
      externally_locked.should be_true

      expect_raises(ML::Metal::ProcessLease::Unavailable, /Metal inference lease/) do
        ML::Metal::ProcessLease.acquire(path: path, wait_timeout: 50.milliseconds)
      end

      holder.terminate
      holder.wait
      lease = ML::Metal::ProcessLease.acquire(path: path, wait_timeout: 250.milliseconds)
      lease.held?.should be_true
      lease.close
      lease.held?.should be_false
    ensure
      if holder.exists?
        holder.terminate
        holder.wait
      end
      File.delete(path) if File.exists?(path)
    end
  end

  it "is reentrant inside one process and releases after the final holder" do
    path = File.tempname("cogni-metal-reentrant", ".lock")
    first = ML::Metal::ProcessLease.acquire(path: path, wait_timeout: 50.milliseconds)
    second = ML::Metal::ProcessLease.acquire(path: path, wait_timeout: 50.milliseconds)

    begin
      first.close
      second.held?.should be_true
    ensure
      second.close
      File.delete(path) if File.exists?(path)
    end
  end
end
