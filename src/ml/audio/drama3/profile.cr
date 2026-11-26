require "json"

module ML::Audio::Drama3
  # Optional synchronous wall-time profiler for native DRAMA-3 execution.
  # Set DRAMA3_PROFILE to a JSON output path to enable collection.
  module Profile
    struct SpanStat
      getter path : String
      getter category : String
      getter calls : Int64
      getter inclusive_ns : Int64
      getter exclusive_ns : Int64

      def initialize(@path : String, @category : String, @calls : Int64,
                     @inclusive_ns : Int64, @exclusive_ns : Int64)
      end
    end

    private class Frame
      property path : String
      getter category : String
      property child_ns : Int64

      def initialize(@path : String, @category : String)
        @child_ns = 0_i64
      end
    end

    private class MutableStat
      getter path : String
      getter category : String
      property calls : Int64
      property inclusive_ns : Int64
      property exclusive_ns : Int64

      def initialize(@path : String, @category : String)
        @calls = 0_i64
        @inclusive_ns = 0_i64
        @exclusive_ns = 0_i64
      end
    end

    @@enabled = ENV["DRAMA3_PROFILE"]?.try { |path| !path.empty? } || false
    @@mutex = Mutex.new
    @@stacks = {} of UInt64 => Array(Frame)
    @@stats = {} of Tuple(String, String) => MutableStat

    # Call once before work when a process changes DRAMA3_PROFILE at runtime.
    # The CLI uses this before loading weights; normal library use reads the
    # environment when the module initializes.
    def self.configure_from_env! : Nil
      @@mutex.synchronize do
        path = ENV["DRAMA3_PROFILE"]?
        @@enabled = !!path && !path.empty?
        @@stacks.clear
        @@stats.clear
      end
    end

    def self.enabled? : Bool
      @@enabled
    end

    # Disabled calls return directly to the block. Enabled measurements keep a
    # per-fiber scope stack so nested inclusive time can be subtracted exactly
    # once from its parent's exclusive time.
    def self.measure(name : String, category : String = "compute", &block : -> T) : T forall T
      return yield unless @@enabled

      fiber_id = Fiber.current.object_id.to_u64
      frame = Frame.new(name, category)
      @@mutex.synchronize do
        stack = @@stacks[fiber_id]? || begin
          created = [] of Frame
          @@stacks[fiber_id] = created
          created
        end
        parent_path = stack.empty? ? "" : stack.last.path
        frame.path = parent_path.empty? ? name : "#{parent_path}/#{name}"
        stack << frame
      end

      started_at = Time.instant
      begin
        yield
      ensure
        finish_measurement(fiber_id, frame, started_at)
      end
    end

    def self.snapshot : Array(SpanStat)
      @@mutex.synchronize do
        @@stats.values.map do |stat|
          SpanStat.new(stat.path, stat.category, stat.calls, stat.inclusive_ns, stat.exclusive_ns)
        end.sort_by { |stat| {stat.path, stat.category} }
      end
    end

    # Emit a compact, machine-readable summary. The timestamps are monotonic
    # elapsed nanoseconds and have no relation to wall-clock time of day.
    def self.write_from_env : Nil
      path = ENV["DRAMA3_PROFILE"]?
      return if path.nil? || path.empty?

      spans = snapshot.map do |stat|
        {
          path:         stat.path,
          category:     stat.category,
          calls:        stat.calls,
          inclusive_ns: stat.inclusive_ns,
          exclusive_ns: stat.exclusive_ns,
        }
      end
      report = {
        schema_version: 1,
        clock:          "monotonic",
        unit:           "nanoseconds",
        spans:          spans,
      }
      File.write(path, report.to_pretty_json)
    end

    private def self.finish_measurement(fiber_id : UInt64, frame : Frame, started_at : Time::Instant) : Nil
      elapsed_ns = (Time.instant - started_at).total_nanoseconds.to_i64
      @@mutex.synchronize do
        stack = @@stacks[fiber_id]?
        if stack && !stack.empty?
          stack.pop
          exclusive_ns = elapsed_ns - frame.child_ns
          stat = @@stats[{frame.path, frame.category}]? || begin
            created = MutableStat.new(frame.path, frame.category)
            @@stats[{frame.path, frame.category}] = created
            created
          end
          stat.calls += 1
          stat.inclusive_ns += elapsed_ns
          stat.exclusive_ns += exclusive_ns
          if parent = stack.last?
            parent.child_ns += elapsed_ns
          else
            @@stacks.delete(fiber_id)
          end
        end
      end
    end
  end
end
