require "file_utils"
require "uuid"
require "../wav"

module ML::Audio::Drama3::Studio
  # File-format conversion is an input/output boundary, outside model inference.
  class Media
    MAX_DIAGNOSTIC_BYTES  = 4 * 1024
    PROCESS_POLL_INTERVAL = 5.milliseconds
    TERMINATION_GRACE     = 100.milliseconds
    STDERR_DRAIN_GRACE    = 100.milliseconds

    getter ffmpeg : String

    def initialize(@ffmpeg : String = Process.find_executable("ffmpeg") || "",
                   @conversion_timeout : Time::Span = 30.seconds)
      raise ArgumentError.new("audio conversion timeout must be positive") unless @conversion_timeout > Time::Span.zero
    end

    def available? : Bool
      !@ffmpeg.empty? && File.file?(@ffmpeg)
    end

    def prepare_reference(input : String, output : String) : Float64
      owned_output = false
      check_output!(output)
      owned_output = true
      args = ["-protocol_whitelist", "file,pipe", "-format_whitelist", "wav,mp3,ogg,flac,mov,matroska,aac",
              "-i", input, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "44100",
              "-t", "10.1", "-c:a", "pcm_s16le", "-f", "wav", output]
      convert(args)
      # Decode slightly beyond the admitted length so long inputs fail rather
      # than silently becoming a truncated ten-second voice reference.
      reference = Wav.read_reference(output)
      reference.samples / 44100.0
    rescue ex
      File.delete(output) if owned_output && File.exists?(output)
      raise ex
    end

    def export(paths : Array(String), output : String, format : String) : Float64
      raise ArgumentError.new("unsupported export format") unless format.in?("wav", "mp3", "opus")
      owned_output = false
      check_output!(output)
      owned_output = true
      raise ArgumentError.new("export requires at least one take") if paths.empty?
      parent = File.dirname(output)
      paths.each do |path|
        raise ArgumentError.new("missing or invalid export take") unless File.file?(path) && File.dirname(path) == parent && File.basename(path).matches?(/\A[a-zA-Z0-9_-]+\.wav\z/)
      end
      list = File.join(parent, "concat-#{UUID.random}.txt")
      File.write(list, paths.map { |path| "file '#{File.basename(path)}'\n" }.join)
      codec = case format
              when "wav"  then ["-c:a", "pcm_s16le", "-f", "wav"]
              when "mp3"  then ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]
              when "opus" then ["-c:a", "libopus", "-b:a", "128k", "-f", "opus"]
              else             raise ArgumentError.new("unsupported export format")
              end
      convert(["-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i", list,
               "-map", "0:a:0", "-vn"] + codec + [output])
      # Duration is taken from the uncompressed inputs, avoiding lossy codec
      # priming/padding as an accidental change to the document duration.
      paths.sum { |path| wav_seconds(path) }
    rescue ex
      File.delete(output) if owned_output && File.exists?(output)
      raise ex
    ensure
      File.delete(list) if list && File.exists?(list)
    end

    private def check_output!(output : String) : Nil
      raise ArgumentError.new("ffmpeg is unavailable") unless available?
      raise ArgumentError.new("output already exists") if File.exists?(output)
    end

    private def convert(args : Array(String)) : Nil
      started_at = Time.instant
      process = Process.new(@ffmpeg, ["-nostdin", "-hide_banner", "-loglevel", "error", "-n"] + args,
        output: Process::Redirect::Close, error: Process::Redirect::Pipe)
      errors = IO::Memory.new(MAX_DIAGNOSTIC_BYTES)
      stderr_drained = Channel(Nil).new(1)
      drainer_started = false
      waited = false

      begin
        spawn do
          buffer = Bytes.new(8192)
          begin
            loop do
              bytes_read = process.error.read(buffer)
              break if bytes_read == 0
              remaining = MAX_DIAGNOSTIC_BYTES - errors.size
              errors.write(buffer[0, Math.min(bytes_read, remaining)]) if remaining > 0
            end
          rescue IO::Error
            # Closing the pipe after its bounded drain grace ends this reader.
          ensure
            stderr_drained.send(nil)
          end
        end
        drainer_started = true

        deadline = started_at + @conversion_timeout
        timed_out = false
        until process.terminated?
          if Time.instant >= deadline
            timed_out = true
            terminate_conversion!(process)
            break
          end
          sleep PROCESS_POLL_INTERVAL
        end

        await_stderr_drain!(process, stderr_drained)
        status = process.wait
        waited = true

        if timed_out
          message = "audio conversion timed out after #{@conversion_timeout.total_milliseconds.to_i64}ms"
          diagnostic = errors.to_s
          message += ": #{diagnostic}" unless diagnostic.empty?
          raise ArgumentError.new(message)
        end
        raise ArgumentError.new("audio conversion failed: #{errors.to_s[0, 800]}") unless status.success?
      ensure
        unless waited
          terminate_conversion!(process) unless process.terminated?
          await_stderr_drain!(process, stderr_drained) if drainer_started
          process.wait
        end
      end
    end

    private def terminate_conversion!(process : Process) : Nil
      return if process.terminated?

      process.terminate rescue nil
      deadline = Time.instant + TERMINATION_GRACE
      until process.terminated? || Time.instant >= deadline
        sleep PROCESS_POLL_INTERVAL
      end
      process.terminate(graceful: false) unless process.terminated?
    rescue
      # A concurrent exit can race either signal; the caller still reaps it.
      nil
    end

    private def await_stderr_drain!(process : Process, drained : Channel(Nil)) : Nil
      select
      when drained.receive
      when timeout(STDERR_DRAIN_GRACE)
        # A descendant may have inherited stderr. Do not let that extend the
        # conversion deadline; close only our read end and leave it unowned.
        process.error.close rescue nil
      end
    end

    private def wav_seconds(path : String) : Float64
      # The selected takes are native mono PCM16 WAVs. ffmpeg verifies the
      # stream on export; scan RIFF chunks to skip optional metadata chunks.
      File.open(path) do |io|
        raise ArgumentError.new("invalid take WAV") unless io.read_string(4) == "RIFF"
        io.skip(4)
        raise ArgumentError.new("invalid take WAV") unless io.read_string(4) == "WAVE"
        until io.pos >= io.size
          name = io.read_string(4)
          size = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i64
          raise ArgumentError.new("truncated take WAV") if size > io.size - io.pos
          return size / 88200.0 if name == "data"
          io.skip(size + size % 2)
        end
      end
      raise ArgumentError.new("take has no sample data")
    end
  end
end
