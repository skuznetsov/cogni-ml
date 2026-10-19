require "digest/sha256"
require "file_utils"

module ML::Audio::Drama3::Studio
  # Process-wide singleton guard for one canonical DRAMA3 model directory.
  # The open descriptor owns the advisory lock until close or process exit;
  # the lock file itself is intentionally retained for stable cross-process identity.
  class ModelLease
    class AlreadyOwned < Exception
    end

    DEFAULT_LOCK_DIR = File.join(Path.home.to_s, ".cache", "cogni-ml", "drama3-model-leases")

    getter model_dir : String
    getter lock_path : String

    @file : File?
    @closed : Bool
    @mutex : Mutex

    def initialize(model_dir : String, lock_dir : String = DEFAULT_LOCK_DIR)
      @mutex = Mutex.new
      @closed = true
      canonical = self.class.canonical_model_dir(model_dir)
      lock_directory = self.class.expanded_lock_dir(lock_dir)
      FileUtils.mkdir_p(lock_directory, 0o700)
      File.chmod(lock_directory, 0o700)
      lock_path = File.join(lock_directory, "#{Digest::SHA256.hexdigest(canonical)}.lock")

      file = File.open(lock_path, "a+", perm: 0o600)
      begin
        file.chmod(0o600)
        begin
          file.flock_exclusive(blocking: false)
        rescue ex : IO::Error
          if ex.os_error == Errno::EWOULDBLOCK || ex.os_error == Errno::EAGAIN
            raise AlreadyOwned.new("DRAMA3 model already owned (lease unavailable): #{canonical}")
          end
          raise ex
        end
      rescue ex
        file.close rescue nil
        raise ex
      end

      @model_dir = canonical
      @lock_path = lock_path
      @file = file
      @closed = false
    end

    def self.default_lock_dir : String
      DEFAULT_LOCK_DIR
    end

    # Exposed so CPU-only callers and tests can identify the stable lock path.
    # The filename contains a digest rather than the potentially sensitive model path.
    def self.lock_path_for(model_dir : String, lock_dir : String = default_lock_dir) : String
      canonical = canonical_model_dir(model_dir)
      expanded_lock_dir = expanded_lock_dir(lock_dir)
      File.join(expanded_lock_dir, "#{Digest::SHA256.hexdigest(canonical)}.lock")
    end

    def self.acquire(model_dir : String, lock_dir : String = default_lock_dir) : self
      new(model_dir, lock_dir)
    end

    def held? : Bool
      @mutex.synchronize { !@closed && !@file.nil? }
    end

    def close : Nil
      @mutex.synchronize do
        return if @closed
        @closed = true
        if file = @file
          @file = nil
          begin
            file.flock_unlock
          ensure
            file.close
          end
        end
      end
    end

    def finalize
      close
    rescue
    end

    def self.canonical_model_dir(model_dir : String) : String
      raise ArgumentError.new("DRAMA3 model directory must not be empty") if model_dir.strip.empty?
      raise ArgumentError.new("DRAMA3 model directory does not exist: #{model_dir}") unless Dir.exists?(model_dir)

      canonical = File.realpath(model_dir)
      raise ArgumentError.new("DRAMA3 model path is not a directory: #{canonical}") unless File.directory?(canonical)
      canonical
    end

    def self.expanded_lock_dir(lock_dir : String) : String
      raise ArgumentError.new("DRAMA3 model lease directory must not be empty") if lock_dir.strip.empty?
      File.expand_path(lock_dir)
    end
  end
end
