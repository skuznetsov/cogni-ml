require "fiber/execution_context/isolated"

module ML::GGUF
  record QwenQBitAsyncWriterStats,
    enqueued : Int64,
    completed : Int64,
    failures : Int64,
    pending : Int32,
    work_time : Time::Span,
    last_failure : String?

  record QwenQBitAsyncCompletion(Result),
    result : Result?,
    error_class : String?,
    error_message : String?,
    elapsed : Time::Span do
    def failed? : Bool
      !error_class.nil?
    end
  end

  # One-slot isolated execution-context handoff for CPU-heavy QBit serialization
  # and I/O.
  # The resident job remains counted until work finishes, so the caller cannot
  # accidentally retain two large host snapshots while one encode is active.
  class QwenQBitAsyncWriter(Job, Result)
    @mutex = Thread::Mutex.new
    @condition = Thread::ConditionVariable.new
    @job : Job? = nil
    @completion : QwenQBitAsyncCompletion(Result)? = nil
    @reserved = false
    @closed = false
    @stop = false
    @stopped = false
    @active = false
    @enqueued = 0_i64
    @completed = 0_i64
    @failures = 0_i64
    @work_time = Time::Span.zero
    @last_failure : String? = nil

    def initialize(name : String = "qwen-qbit-writer", &@work : Job -> Result)
      raise ArgumentError.new("QBit async writer name must not be empty") if name.empty?
      @worker = Fiber::ExecutionContext::Isolated.new(name) { run }
    end

    def enqueue(job : Job) : Nil
      enqueue_with { job }
    end

    # Claims the only resident slot before the caller materializes a potentially
    # large job. A concurrent close stops new reservations, then waits for an
    # existing preparation and its publication to finish.
    def enqueue_with(&prepare : -> Job) : Job
      reserved = false
      @mutex.synchronize do
        raise ArgumentError.new("QBit async writer is closed") if @closed
        if @reserved || @job || @active || @completion
          raise ArgumentError.new("QBit async writer single-flight slot is occupied")
        end
        @reserved = true
        reserved = true
      end

      job = prepare.call
      @mutex.synchronize do
        @job = job
        @reserved = false
        reserved = false
        @enqueued += 1
        @condition.broadcast
      end
      job
    rescue ex
      if reserved
        @mutex.synchronize do
          @reserved = false
          @condition.broadcast
        end
      end
      raise ex
    end

    # Waits for the resident job and consumes its completion. Persistent totals
    # and the last failure remain available through #stats.
    def flush : QwenQBitAsyncCompletion(Result)?
      @mutex.lock
      begin
        while @reserved || @job || @active
          @condition.wait(@mutex)
        end
        completion = @completion
        @completion = nil
        completion
      ensure
        @mutex.unlock
      end
    end

    def stats : QwenQBitAsyncWriterStats
      @mutex.synchronize do
        QwenQBitAsyncWriterStats.new(
          enqueued: @enqueued,
          completed: @completed,
          failures: @failures,
          pending: (@reserved || @job || @active) ? 1 : 0,
          work_time: @work_time,
          last_failure: @last_failure,
        )
      end
    end

    # Admission stops before the current reservation/job is drained. Returning
    # its completion lets the runtime surface a durability failure before
    # teardown.
    def close : QwenQBitAsyncCompletion(Result)?
      close_owner = false
      @mutex.synchronize do
        unless @closed
          @closed = true
          close_owner = true
        end
      end

      unless close_owner
        @mutex.lock
        begin
          until @stopped
            @condition.wait(@mutex)
          end
        ensure
          @mutex.unlock
        end
        return nil
      end

      completion = flush
      @mutex.synchronize do
        @stop = true
        @condition.broadcast
      end
      begin
        @worker.join
      ensure
        @mutex.synchronize do
          @stopped = true
          @condition.broadcast
        end
      end
      completion
    end

    private def run : Nil
      loop do
        job = nil.as(Job?)
        stop = false
        @mutex.lock
        begin
          while @job.nil? && !@stop
            @condition.wait(@mutex)
          end
          if @stop
            stop = true
          else
            job = @job
            @active = true
          end
        ensure
          @mutex.unlock
        end
        break if stop

        started = Time.instant
        result = nil.as(Result?)
        error_class = nil.as(String?)
        error_message = nil.as(String?)
        begin
          result = @work.call(job.not_nil!)
        rescue ex
          error_class = ex.class.to_s
          error_message = ex.message || "unknown error"
        ensure
          elapsed = Time.instant - started
          @mutex.synchronize do
            @work_time += elapsed
            if error_class
              @failures += 1
              @last_failure = "#{error_class}: #{error_message}"
            else
              @completed += 1
            end
            @completion = QwenQBitAsyncCompletion(Result).new(
              result,
              error_class,
              error_message,
              elapsed,
            )
            @job = nil
            @active = false
            @condition.broadcast
          end
        end
      end
    end
  end
end
