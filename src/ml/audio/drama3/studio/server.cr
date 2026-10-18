require "http/server"
require "uri/params"
require "uuid"
require "./store"
require "./media"
require "./backend"

module ML::Audio::Drama3::Studio
  class App
    JSON_LIMIT   = 128 * 1024
    UPLOAD_LIMIT = 20 * 1024 * 1024
    @wake = Channel(Nil).new(1)
    @closed = false
    @started = false
    @queue_error : String?

    def initialize(@store : Store, @backend : Backend, @port : Int32, @model_name : String,
                   @media : Media = Media.new)
      Dir.mkdir_p(File.join(@store.root, "tmp"))
    end

    def start : Nil
      return if @started
      @started = true
      spawn do
        pending_failure = nil.as(Tuple(String, String)?)
        until @closed
          begin
            if failure = pending_failure
              # Do not claim another job until the previous failure is durable.
              @store.fail_job(failure[0], failure[1])
              pending_failure = nil
            elsif job = @store.claim_job
              break if @closed
              begin
                perform(job)
              rescue ex
                message = (ex.message || "job failed").chars.first(1_000).join
                pending_failure = {job["id"].as_s, message.empty? ? "job failed" : message}
              end
            else
              select
              when @wake.receive
              when timeout(250.milliseconds)
              end
            end
            @queue_error = nil
          rescue ex
            # A transient manifest I/O failure must not kill the sole consumer.
            message = (ex.message || "queue persistence failed").chars.first(1_000).join
            STDERR.puts "studio queue paused: #{message}" unless @queue_error == message
            @queue_error = message
            sleep 1.second unless @closed
          end
        end
      end
    end

    def close : Nil
      @closed = true
      @backend.close
    end

    def call(context : HTTP::Server::Context) : Nil
      response = context.response
      response.headers["X-Content-Type-Options"] = "nosniff"
      response.headers["Referrer-Policy"] = "no-referrer"
      response.headers["Cache-Control"] = "no-store"
      response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
      request = context.request
      host = request.headers["Host"]?
      unless host.in?("127.0.0.1:#{@port}", "localhost:#{@port}")
        return error(context, 403, "local Host required")
      end
      unless request.method.in?("GET", "HEAD")
        unless request.headers["Origin"]? == "http://#{host}"
          return error(context, 403, "same-origin request required")
        end
      end
      route(context)
    rescue ex : Store::ConflictError
      error(context, 409, ex.message || "conflict")
    rescue ex : KeyError
      error(context, 404, "item not found")
    rescue ex : ArgumentError | JSON::ParseException | TypeCastError | OverflowError
      error(context, 400, ex.message || "invalid request")
    rescue ex
      STDERR.puts "studio request failed: #{ex.message}"
      error(context, 500, "request failed; see the local server log")
    end

    private def route(context : HTTP::Server::Context) : Nil
      request = context.request
      path = request.path
      method = request.method
      if method == "GET" && path.in?("/", "/app.js", "/i18n.js", "/style.css", "/assets/app.js", "/assets/i18n.js", "/assets/style.css", "/assets/voice-recorder.js", "/assets/voice-recorder-worklet.js")
        mime, data = case path
                     when "/app.js", "/assets/app.js"       then {"text/javascript; charset=utf-8", {{ read_file("#{__DIR__}/assets/app.js") }}}
                     when "/i18n.js", "/assets/i18n.js"     then {"text/javascript; charset=utf-8", {{ read_file("#{__DIR__}/assets/i18n.js") }}}
                     when "/style.css", "/assets/style.css" then {"text/css; charset=utf-8", {{ read_file("#{__DIR__}/assets/style.css") }}}
                     when "/assets/voice-recorder.js"         then {"text/javascript; charset=utf-8", {{ read_file("#{__DIR__}/assets/voice-recorder.js") }}}
                     when "/assets/voice-recorder-worklet.js" then {"text/javascript; charset=utf-8", {{ read_file("#{__DIR__}/assets/voice-recorder-worklet.js") }}}
                     else                                        {"text/html; charset=utf-8", {{ read_file("#{__DIR__}/assets/index.html") }}}
                     end
        context.response.content_type = mime
        context.response.print(data)
      elsif method == "GET" && path == "/api/state"
        state = @store.state
        state.as_h["capabilities"] = JSON.parse({backend: "metal_resident", model: @model_name,
                                                 ffmpeg: @media.available?, formats: @media.available? ? ["wav", "mp3", "opus"] : [] of String,
                                                 max_reference_seconds: 10, queue_error: @queue_error}.to_json)
        json(context, state)
      elsif method == "POST" && path == "/api/sessions"
        json(context, @store.create_session(body(context)["name"].as_s))
      elsif match = path.match(/\A\/api\/sessions\/([^\/]+)\z/)
        if method == "PATCH"
          input = body(context)
          json(context, @store.update_session(match[1], input["name"]?.try(&.as_s), input["archived"]?.try(&.as_bool)))
        else
          error(context, 405, "method not allowed")
        end
      elsif match = path.match(/\A\/api\/sessions\/([^\/]+)\/segments(?:\/([^\/]+))?(?:\/(generate|select))?\z/)
        session, segment, action = match[1], match[2]?, match[3]?
        if method == "POST" && segment && action == "generate"
          body(context)
          job = @store.enqueue_generation(session, segment)
          wake
          json(context, job)
        elsif method == "POST" && segment && action == "select"
          json(context, @store.select_take(session, segment, body(context)["asset_id"].as_s))
        elsif method == "POST" && !segment && !action
          json(context, @store.save_segment(session, nil, body(context)))
        elsif method == "PATCH" && segment && !action
          json(context, @store.save_segment(session, segment, body(context)))
        elsif method == "DELETE" && segment && !action
          revision = URI::Params.parse(request.query || "")["revision"].to_i
          @store.remove_segment(session, segment, revision)
          json(context, JSON.parse("{\"ok\":true}"))
        else
          error(context, 405, "method not allowed")
        end
      elsif method == "POST" && (match = path.match(/\A\/api\/sessions\/([^\/]+)\/export\z/))
        job = @store.enqueue_export(match[1], body(context)["format"].as_s)
        wake
        json(context, job)
      elsif method == "POST" && path == "/api/voices"
        upload_voice(context)
      elsif method == "DELETE" && (match = path.match(/\A\/api\/voices\/([^\/]+)\z/))
        @store.archive_voice(match[1])
        json(context, JSON.parse("{\"ok\":true}"))
      elsif method.in?("GET", "HEAD") && (match = path.match(/\A\/api\/(audio|voice-audio)\/([^\/]+)\z/))
        file = match[1] == "audio" ? @store.asset_path(match[2]) : @store.voice_path(match[2])
        serve_audio(context, file)
      elsif method == "DELETE" && (match = path.match(/\A\/api\/assets\/([^\/]+)\z/))
        @store.delete_asset(match[1])
        json(context, JSON.parse("{\"ok\":true}"))
      else
        error(context, 404, "route not found")
      end
    end

    private def read_bounded(context : HTTP::Server::Context, limit : Int32) : Bytes
      if length = context.request.content_length
        raise ArgumentError.new("request body is too large") if length > limit
      end
      io = context.request.body
      raise ArgumentError.new("request body required") unless io
      buffer = IO::Memory.new
      chunk = Bytes.new(16_384)
      while (count = io.read(chunk)) > 0
        raise ArgumentError.new("request body is too large") if buffer.size + count > limit
        buffer.write(chunk[0, count])
      end
      buffer.to_slice.dup
    end

    private def body(context : HTTP::Server::Context) : JSON::Any
      raise ArgumentError.new("JSON content type required") unless context.request.headers["Content-Type"]?.try(&.starts_with?("application/json"))
      input = JSON.parse(String.new(read_bounded(context, JSON_LIMIT)))
      input.as_h
      input
    end

    private def upload_voice(context : HTTP::Server::Context) : Nil
      params = URI::Params.parse(context.request.query || "")
      name, transcript = params["name"], params["transcript"]
      raise ArgumentError.new("voice name and transcript required") if name.strip.empty? || transcript.strip.empty? || name.bytesize > 200 || transcript.bytesize > 16_384
      bytes = read_bounded(context, UPLOAD_LIMIT)
      input = File.join(@store.root, "tmp", "#{UUID.random}.upload")
      filename = "reference-#{UUID.random}.wav"
      output = File.join(@store.root, "assets", filename)
      File.write(input, bytes)
      seconds = @media.prepare_reference(input, output)
      voice = @store.create_voice(name, transcript, filename, seconds)
      registered = true
      json(context, voice)
    ensure
      File.delete(input) if input && File.exists?(input)
      File.delete(output) if !registered && output && File.exists?(output)
    end

    private def perform(job : JSON::Any) : Nil
      id = job["id"].as_s
      input = JSON.parse(job["input"].to_json)
      kind = job["kind"].as_s
      format = kind == "generate" ? "wav" : input["format"].as_s
      filename = "#{kind}-#{UUID.random}.#{format}"
      output = File.join(@store.root, "assets", filename)
      report = if kind == "generate"
                 if reference = input["reference_filename"]?.try(&.as_s?)
                   input.as_h["reference_audio"] = JSON::Any.new(File.join(@store.root, "assets", reference))
                 end
                 @backend.generate(input, output) do |message|
                   @store.update_job_progress(id, message)
                 end
               else
                 paths = input["asset_ids"].as_a.map { |asset| @store.asset_path(asset.as_s) }
                 seconds = @media.export(paths, output, format)
                 JSON.parse({seconds: seconds, format: format, asset_ids: input["asset_ids"]}.to_json)
               end
      # Full terminal codes remain in the take's adjacent JSON report. Keeping
      # them out of the frequently persisted manifest bounds UI/progress I/O.
      summary = JSON.parse(report.to_json)
      summary.as_h.delete("frames")
      @store.finish_job(id, filename, report["seconds"].as_f, summary)
      registered = true
    ensure
      if output && !registered
        [output, output + ".json"].each do |path|
          begin
            File.delete(path) if File.exists?(path)
          rescue ex
            STDERR.puts "studio output cleanup failed: #{ex.message}"
          end
        end
      end
    end

    private def serve_audio(context : HTTP::Server::Context, path : String) : Nil
      raise KeyError.new("audio missing") unless File.file?(path)
      response = context.response
      response.content_type = case File.extname(path)
                              when ".mp3"  then "audio/mpeg"
                              when ".opus" then "audio/ogg"
                              else              "audio/wav"
                              end
      response.headers["Accept-Ranges"] = "bytes"
      if URI::Params.parse(context.request.query || "")["download"]? == "1"
        response.headers["Content-Disposition"] = "attachment; filename=\"#{File.basename(path)}\""
      end
      size = File.size(path)
      start = 0_i64
      count = size
      if range = context.request.headers["Range"]?
        match = range.match(/\Abytes=(\d+)-(\d*)\z/)
        unless match
          response.headers["Content-Range"] = "bytes */#{size}"
          response.status_code = 416
          return
        end
        start = match[1].to_i64
        last = match[2].empty? ? size - 1 : match[2].to_i64
        if start >= size || last < start
          response.headers["Content-Range"] = "bytes */#{size}"
          response.status_code = 416
          return
        end
        last = Math.min(last, size - 1)
        count = last - start + 1
        response.status_code = 206
        response.headers["Content-Range"] = "bytes #{start}-#{last}/#{size}"
      end
      response.content_length = count
      return if context.request.method == "HEAD"
      File.open(path) do |file|
        file.seek(start)
        IO.copy(file, response, count)
      end
    end

    private def wake : Nil
      select
      when @wake.send(nil)
      else
      end
    end

    private def json(context, value) : Nil
      context.response.content_type = "application/json; charset=utf-8"
      context.response.print(value.to_json)
    end

    private def error(context, status, message) : Nil
      context.response.status_code = status
      json(context, {error: message})
    end
  end
end
