module ML::Metal
  # Cross-process single-flight guard for heavy Metal inference.
  #
  # BSD flock owns the actual lease, so an exited or killed process releases it
  # automatically. The small in-process reference count makes the guard
  # reentrant for callers that share one inference process.
  class ProcessLease
    class Unavailable < Exception
    end

    DEFAULT_WAIT_MS = 30_000_i64

    @@mutex = Mutex.new
    @@file : File? = nil
    @@path : String? = nil
    @@holders = 0

    getter path : String
    @closed = false

    private def initialize(@path : String)
    end

    def self.default_path : String
      path = ENV["COGNI_METAL_LEASE_PATH"]? || File.join(Dir.tempdir, "cogni-ml-metal-inference.lock")
      raise ArgumentError.new("COGNI_METAL_LEASE_PATH must not be empty") if path.strip.empty?
      File.expand_path(path)
    end

    def self.default_wait_timeout : Time::Span
      raw = ENV["COGNI_METAL_LEASE_WAIT_MS"]? || DEFAULT_WAIT_MS.to_s
      milliseconds = raw.to_i64?
      unless milliseconds && milliseconds >= 0
        raise ArgumentError.new("COGNI_METAL_LEASE_WAIT_MS must be a non-negative integer")
      end
      milliseconds.milliseconds
    end

    def self.acquire(path : String = default_path,
                     wait_timeout : Time::Span = default_wait_timeout) : self
      raise ArgumentError.new("Metal inference lease wait timeout must not be negative") if wait_timeout < Time::Span.zero
      expanded_path = File.expand_path(path)

      @@mutex.synchronize do
        if active_file = @@file
          unless @@path == expanded_path
            raise ArgumentError.new("Metal inference lease already uses #{@@path}; cannot also use #{expanded_path}")
          end
          @@holders += 1
          return new(expanded_path)
        end

        file = File.open(expanded_path, "a+")
        begin
          deadline = Time.instant + wait_timeout
          loop do
            begin
              file.flock_exclusive(blocking: false)
              break
            rescue ex : IO::Error
              if Time.instant >= deadline
                raise Unavailable.new("Metal inference lease #{expanded_path} remained busy for #{wait_timeout.total_milliseconds.to_i64}ms")
              end
              sleep 10.milliseconds
            end
          end
        rescue ex
          file.close
          raise ex
        end

        @@file = file
        @@path = expanded_path
        @@holders = 1
        new(expanded_path)
      end
    end

    def held? : Bool
      !@closed
    end

    def close : Nil
      return if @closed

      @@mutex.synchronize do
        return if @closed
        unless @@path == @path && @@holders > 0
          raise "Metal inference lease ownership is inconsistent"
        end

        @closed = true
        @@holders -= 1
        if @@holders == 0
          if file = @@file
            file.flock_unlock
            file.close
          end
          @@file = nil
          @@path = nil
        end
      end
    end

    def finalize
      close
    rescue
    end
  end
end
