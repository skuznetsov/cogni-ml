require "json"

module ML::Audio::Drama3::Studio
  class WorkerTimeoutError < Exception
  end

  abstract class Backend
    abstract def generate(input : JSON::Any, output : String, &progress : String -> Nil) : JSON::Any
    abstract def close : Nil
  end

  # The HTTP process remains responsive while the separate native worker owns
  # the GPU. There is one pipe transaction at a time, enforced by App's queue.
  class WorkerBackend < Backend
    DEFAULT_RESPONSE_TIMEOUT = 5.minutes

    @process : Process?
    @closed = false

    def initialize(@executable : String, @model : String, @response_timeout : Time::Span = DEFAULT_RESPONSE_TIMEOUT)
      raise ArgumentError.new("response timeout must be positive") unless @response_timeout > Time::Span.zero
    end

    def generate(input : JSON::Any, output : String, &progress : String -> Nil) : JSON::Any
      raise "native worker is closed" if @closed
      process = @process
      unless process
        progress.call("Загрузка модели")
        raise "native worker is closed" if @closed
        process = Process.new(@executable, ["--worker", "--model", @model],
          input: Process::Redirect::Pipe, output: Process::Redirect::Pipe, error: Process::Redirect::Inherit)
        @process = process
        raise "native worker is closed" if @closed
        ready = message(process)
        raise "native worker failed to start" unless ready["event"].as_s == "ready"
      end
      process.input.puts({input: input, output: output}.to_json)
      process.input.flush
      loop do
        event = message(process)
        case event["event"].as_s
        when "progress" then progress.call(event["message"].as_s)
        when "result"   then return event["report"]
        when "error"    then raise event["message"].as_s
        else                 raise "invalid native worker response"
        end
      end
    rescue ex
      discard_worker(force: ex.is_a?(WorkerTimeoutError))
      raise ex
    end

    private def message(process : Process) : JSON::Any
      response = Channel(NamedTuple(line: String?, error: Exception?)).new(1)
      spawn do
        begin
          response.send({line: process.output.gets, error: nil})
        rescue ex
          response.send({line: nil, error: ex})
        end
      end

      line = nil.as(String?)
      select
      when result = response.receive
        raise result[:error].not_nil! if result[:error]
        line = result[:line]
      when timeout(@response_timeout)
        raise WorkerTimeoutError.new("native worker response timed out")
      end

      raise "native worker exited before completing the job" unless line
      raise "native worker response is too large" if line.bytesize > 8 * 1024 * 1024
      JSON.parse(line)
    end

    def close : Nil
      @closed = true
      discard_worker
    end

    private def discard_worker(force : Bool = false) : Nil
      process = @process
      @process = nil
      return unless process

      process.input.close rescue nil
      begin
        process.terminate(graceful: !force) unless process.terminated?
      rescue
        # Still wait below in case the process exited between checking and signaling.
      ensure
        begin
          process.wait
        rescue
        ensure
          process.close rescue nil
        end
      end
    end
  end
end
