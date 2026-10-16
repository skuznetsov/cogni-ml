require "file_utils"
require "json"
require "uuid"

module ML::Audio::Drama3::Studio
  # Persistent document, reference audio, immutable take, and serialized job store.
  class Store
    getter root : String

    class ConflictError < ArgumentError
    end

    private ID_PATTERN             = /\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\z/
    private TOMBSTONE_PATTERN      = /\A([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.pending\z/
    private FILENAME_PATTERN       = /\A[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}\z/
    private MANIFEST_VERSION       =  1_i64
    private MAX_SESSIONS           =  1_000
    private MAX_VOICES             =  1_000
    private MAX_ASSETS             =  1_000
    private MAX_JOBS               =  1_000
    private MAX_SEGMENTS           =  1_000
    private MAX_NAME_BYTES         =    200
    private MAX_SEGMENT_TEXT_BYTES = 16_384
    private MAX_TRANSCRIPT_BYTES   = 16_384

    @mutex : Mutex
    @database : Hash(String, JSON::Any)
    @manifest_path : String
    @asset_dir : String
    @trash_dir : String

    def initialize(root : String)
      @root = File.expand_path(root)
      @manifest_path = File.join(@root, "studio.json")
      @asset_dir = File.join(@root, "assets")
      @trash_dir = File.join(@root, "trash")
      @mutex = Mutex.new
      FileUtils.mkdir_p(@asset_dir)
      @database = empty_database

      if File.exists?(@manifest_path)
        load_database!
        recover_interrupted_jobs!
      else
        persist!
      end
      recover_asset_deletions!
    end

    # Returns a detached public snapshot, so a caller cannot mutate persisted state.
    def state : JSON::Any
      @mutex.synchronize do
        JSON.parse(JSON::Any.new({
          "sessions" => @database["sessions"],
          "voices"   => @database["voices"],
          "assets"   => @database["assets"],
          "jobs"     => @database["jobs"],
        } of String => JSON::Any).to_json)
      end
    end

    def create_session(name : String) : JSON::Any
      validate_label!(name, "session name")
      write do
        ensure_capacity!(sessions, MAX_SESSIONS, "sessions")
        now = timestamp
        session = object({
          "id"         => string(new_id),
          "name"       => string(name),
          "archived"   => boolean(false),
          "created_at" => integer(now),
          "updated_at" => integer(now),
          "segments"   => array([] of JSON::Any),
        } of String => JSON::Any)
        sessions << session
        copy(session)
      end
    end

    def update_session(id : String, name : String? = nil, archived : Bool? = nil) : JSON::Any
      validate_id!(id, "session id")
      validate_label!(name, "session name") if name
      write do
        session = find!(sessions, id, "session").as_h
        session["name"] = string(name.not_nil!) if name
        session["archived"] = boolean(archived.not_nil!) unless archived.nil?
        session["updated_at"] = integer(timestamp)
        copy(JSON::Any.new(session))
      end
    end

    # Creates a new segment when segment_id is nil, or updates one with an
    # optimistic revision check. Every edit clears the current selection while
    # preserving old take assets for explicit replay or reselection.
    def save_segment(session_id : String, segment_id : String?, input : JSON::Any) : JSON::Any
      validate_id!(session_id, "session id")
      validate_id!(segment_id, "segment id") if segment_id
      fields = input.as_h? || invalid("segment input must be an object")
      text = required_string(fields, "text")
      validate_text!(text, "segment text", MAX_SEGMENT_TEXT_BYTES)

      write do
        session = find!(sessions, session_id, "session").as_h
        ensure_editable_session!(session)
        segment_list = session["segments"].as_a
        existing = segment_id ? find_optional!(segment_list, segment_id) : nil

        if segment_id && existing.nil?
          raise KeyError.new("unknown segment")
        end

        revision = if current = existing
                     expected = required_i32(fields, "revision")
                     actual = required_i32(current.as_h, "revision")
                     raise ConflictError.new("segment revision conflict") unless expected == actual
                     raise ConflictError.new("segment revision exhausted") if actual == Int32::MAX
                     actual + 1
                   else
                     if raw_revision = fields["revision"]?
                       raise ConflictError.new("new segment revision must be zero") unless integer_value(raw_revision, "revision") == 0
                     end
                     1_i32
                   end

        old = existing.try(&.as_h)
        voice_id = if raw_voice = fields["voice_id"]?
                     parse_voice_id(raw_voice)
                   else
                     old.try { |values| nullable_string(values["voice_id"]?) }
                   end
        if voice_id
          voice = find!(voices, voice_id, "voice").as_h
          raise ConflictError.new("voice is archived") if voice["archived"].as_bool
        end

        seed = optional_i64(fields, "seed", old, 0_i64)
        max_frames = optional_bounded_i32(fields, "max_frames", old, 1024_i32, 1_i32, 4096_i32)
        temperature = optional_number(fields, "temperature", old, 1.0_f64)
        top_k = optional_bounded_i32(fields, "top_k", old, 50_i32, 1_i32, 1024_i32)
        top_p = optional_number(fields, "top_p", old, 1.0_f64)
        validate_sampling!(temperature, top_p)

        segment = object({
          "id"               => string(segment_id || new_id),
          "text"             => string(text),
          "voice_id"         => nullable(voice_id),
          "seed"             => integer(seed),
          "max_frames"       => integer(max_frames.to_i64),
          "temperature"      => number(temperature),
          "top_k"            => integer(top_k.to_i64),
          "top_p"            => number(top_p),
          "revision"         => integer(revision.to_i64),
          "selected_take_id" => nullable(nil),
        } of String => JSON::Any)

        if existing
          index = segment_list.index(existing.not_nil!) || raise KeyError.new("unknown segment")
          segment_list[index] = segment
        else
          ensure_capacity!(segment_list, MAX_SEGMENTS, "segments per session")
          segment_list << segment
        end
        session["updated_at"] = integer(timestamp)
        copy(segment)
      end
    end

    def remove_segment(session_id : String, segment_id : String, revision : Int32) : Nil
      validate_id!(session_id, "session id")
      validate_id!(segment_id, "segment id")
      write do
        session = find!(sessions, session_id, "session").as_h
        ensure_editable_session!(session)
        segment_list = session["segments"].as_a
        segment = find!(segment_list, segment_id, "segment").as_h
        raise ConflictError.new("segment revision conflict") unless required_i32(segment, "revision") == revision
        index = segment_list.index { |row| row["id"].as_s == segment_id } || raise KeyError.new("unknown segment")
        segment_list.delete_at(index)
        session["updated_at"] = integer(timestamp)
        nil
      end
    end

    def select_take(session_id : String, segment_id : String, asset_id : String) : JSON::Any
      validate_id!(session_id, "session id")
      validate_id!(segment_id, "segment id")
      validate_id!(asset_id, "asset id")
      write do
        session = find!(sessions, session_id, "session").as_h
        ensure_editable_session!(session)
        segment = find!(session["segments"].as_a, segment_id, "segment").as_h
        asset = find!(assets, asset_id, "asset").as_h
        belongs = asset["kind"].as_s == "take" &&
                  asset["session_id"]?.try(&.as_s?) == session_id &&
                  asset["segment_id"]?.try(&.as_s?) == segment_id
        raise ConflictError.new("take does not belong to segment") unless belongs
        segment["selected_take_id"] = string(asset_id)
        session["updated_at"] = integer(timestamp)
        copy(JSON::Any.new(segment))
      end
    end

    def create_voice(name : String, transcript : String, filename : String, seconds : Float64) : JSON::Any
      validate_label!(name, "voice name")
      validate_text!(transcript, "voice transcript", MAX_TRANSCRIPT_BYTES)
      validate_filename!(filename)
      validate_seconds!(seconds)

      write do
        ensure_capacity!(voices, MAX_VOICES, "voices")
        ensure_asset_capacity!(1)
        ensure_unused_filename!(filename)
        path = path_for_filename(filename)
        raise ArgumentError.new("reference audio file is missing") unless File.file?(path)
        voice_id = new_id
        asset_id = new_id
        now = timestamp
        reference = object({
          "id"         => string(asset_id),
          "name"       => string(name),
          "kind"       => string("reference"),
          "filename"   => string(filename),
          "format"     => string(file_format(filename)),
          "seconds"    => number(seconds),
          "created_at" => integer(now),
          "voice_id"   => string(voice_id),
        } of String => JSON::Any)
        voice = object({
          "id"         => string(voice_id),
          "name"       => string(name),
          "transcript" => string(transcript),
          "filename"   => string(filename),
          "asset_id"   => string(asset_id),
          "seconds"    => number(seconds),
          "archived"   => boolean(false),
          "created_at" => integer(now),
        } of String => JSON::Any)
        assets << reference
        voices << voice
        copy(voice)
      end
    end

    def archive_voice(id : String) : Nil
      validate_id!(id, "voice id")
      write do
        voice = find!(voices, id, "voice").as_h
        voice["archived"] = boolean(true)
        nil
      end
    end

    def asset_path(id : String) : String
      validate_id!(id, "asset id")
      @mutex.synchronize do
        asset = find!(assets, id, "asset").as_h
        path_for_filename(required_string(asset, "filename"))
      end
    end

    def voice_path(id : String) : String
      validate_id!(id, "voice id")
      @mutex.synchronize do
        voice = find!(voices, id, "voice").as_h
        raise KeyError.new("voice is archived") if voice["archived"].as_bool
        path_for_filename(required_string(voice, "filename"))
      end
    end

    def delete_asset(id : String) : Nil
      validate_id!(id, "asset id")
      @mutex.synchronize do
        snapshot = JSON.parse(JSON::Any.new(@database).to_json).as_h
        asset = find!(assets, id, "asset").as_h
        if asset_is_selected?(id)
          raise ConflictError.new("asset is selected by a segment")
        end
        if active_job_uses_asset?(id)
          raise ConflictError.new("asset is used by an active job")
        end
        if active_voice_uses_asset?(id)
          raise ConflictError.new("asset is used by an active voice")
        end
        filename = required_string(asset, "filename")
        path = path_for_filename(filename)
        tombstone = File.join(@trash_dir, "#{id}.pending")
        moved = false

        if File.exists?(path)
          FileUtils.mkdir_p(@trash_dir)
          raise ConflictError.new("asset deletion is already pending") if File.exists?(tombstone)
          File.rename(path, tombstone)
          moved = true
        end

        begin
          index = assets.index { |row| row["id"].as_s == id } || raise KeyError.new("unknown asset")
          assets.delete_at(index)
          persist!
        rescue error
          @database = snapshot
          if moved && File.exists?(tombstone) && !File.exists?(path)
            begin
              File.rename(tombstone, path)
            rescue
              # The active manifest still names this asset; startup recovery
              # will restore its pending tombstone if rollback cannot do so.
            end
          end
          raise error
        end

        # Once the manifest no longer references this asset, a failed unlink
        # is only cleanup debt and must not be reported as a failed deletion.
        remove_tombstone_best_effort(tombstone) if moved
      end
      nil
    end

    def enqueue_generation(session_id : String, segment_id : String) : JSON::Any
      validate_id!(session_id, "session id")
      validate_id!(segment_id, "segment id")
      write do
        ensure_capacity!(jobs, MAX_JOBS, "jobs")
        ensure_asset_capacity!(1)
        session = find!(sessions, session_id, "session").as_h
        ensure_editable_session!(session)
        segment = find!(session["segments"].as_a, segment_id, "segment").as_h
        revision = required_i32(segment, "revision")
        if jobs.any? do |job|
             values = job.as_h
             values["kind"].as_s == "generate" &&
             values["status"].as_s.in?("queued", "running") &&
             values["session_id"].as_s == session_id &&
             values["segment_id"].as_s == segment_id &&
             values["revision"].as_i == revision
           end
          raise ConflictError.new("generation already queued for this revision")
        end

        input = copy(JSON::Any.new(segment)).as_h
        voice_id = nullable_string(segment["voice_id"]?)
        if voice_id
          voice = find!(voices, voice_id, "voice").as_h
          raise ConflictError.new("voice is archived") if voice["archived"].as_bool
          input["reference_filename"] = string(required_string(voice, "filename"))
          input["reference_text"] = string(required_string(voice, "transcript"))
          input["reference_asset_id"] = string(required_string(voice, "asset_id"))
        else
          input["reference_filename"] = nullable(nil)
          input["reference_text"] = nullable(nil)
          input["reference_asset_id"] = nullable(nil)
        end

        job = object({
          "id"         => string(new_id),
          "kind"       => string("generate"),
          "status"     => string("queued"),
          "session_id" => string(session_id),
          "segment_id" => string(segment_id),
          "revision"   => integer(revision.to_i64),
          "input"      => JSON::Any.new(input),
          "progress"   => integer(0_i64),
          "created_at" => integer(timestamp),
        } of String => JSON::Any)
        jobs << job
        copy(job)
      end
    end

    def enqueue_export(session_id : String, format : String) : JSON::Any
      validate_id!(session_id, "session id")
      normalized_format = format.downcase
      raise ArgumentError.new("export format must be wav, mp3, or opus") unless normalized_format.in?("wav", "mp3", "opus")
      write do
        ensure_capacity!(jobs, MAX_JOBS, "jobs")
        ensure_asset_capacity!(1)
        session = find!(sessions, session_id, "session").as_h
        ensure_editable_session!(session)
        segment_list = session["segments"].as_a
        raise ArgumentError.new("cannot export an empty session") if segment_list.empty?
        selected_ids = segment_list.map do |raw_segment|
          segment = raw_segment.as_h
          selected_id = nullable_string(segment["selected_take_id"]?) || raise ConflictError.new("every segment needs a selected take")
          asset = find!(assets, selected_id, "asset").as_h
          valid = asset["kind"].as_s == "take" &&
                  asset["session_id"]?.try(&.as_s?) == session_id &&
                  asset["segment_id"]?.try(&.as_s?) == segment["id"].as_s
          raise ConflictError.new("selected take does not belong to segment") unless valid
          string(selected_id)
        end
        job = object({
          "id"         => string(new_id),
          "kind"       => string("export"),
          "status"     => string("queued"),
          "session_id" => string(session_id),
          "input"      => object({
            "format"    => string(normalized_format),
            "asset_ids" => array(selected_ids),
          } of String => JSON::Any),
          "created_at" => integer(timestamp),
        } of String => JSON::Any)
        jobs << job
        copy(job)
      end
    end

    def claim_job : JSON::Any?
      @mutex.synchronize do
        job = jobs.find { |row| row["status"].as_s == "queued" }
        if job.nil?
          nil
        else
          snapshot = JSON.parse(JSON::Any.new(@database).to_json).as_h
          begin
            values = job.as_h
            values["status"] = string("running")
            values["started_at"] = integer(timestamp)
            values["progress"] = integer(0_i64)
            result = copy(job)
            persist!
            result
          rescue error
            @database = snapshot
            raise error
          end
        end
      end
    end

    def update_job_progress(job_id : String, progress : Int32, detail : String? = nil) : JSON::Any
      validate_id!(job_id, "job id")
      raise ArgumentError.new("progress must be between 0 and 100") unless progress.in?(0..100)
      validate_text!(detail, "job detail", 2_000, allow_empty: true) if detail
      write do
        job = find!(jobs, job_id, "job").as_h
        ensure_active_job!(job)
        job["progress"] = integer(progress.to_i64)
        job["detail"] = string(detail.not_nil!) if detail
        copy(JSON::Any.new(job))
      end
    end

    # Worker callbacks provide descriptive progress rather than a measured
    # percentage. Keep the numeric progress field for percentage-aware callers.
    def update_job_progress(job_id : String, message : String) : JSON::Any
      validate_id!(job_id, "job id")
      validate_text!(message, "job detail", 2_000, allow_empty: true)
      write do
        job = find!(jobs, job_id, "job").as_h
        ensure_active_job!(job)
        job["detail"] = string(message)
        copy(JSON::Any.new(job))
      end
    end

    def finish_job(job_id : String, filename : String, seconds : Float64, report : JSON::Any) : JSON::Any
      validate_id!(job_id, "job id")
      validate_filename!(filename)
      validate_seconds!(seconds)
      raise ArgumentError.new("job report must be an object") unless report.as_h?

      write do
        job = find!(jobs, job_id, "job").as_h
        ensure_running_job!(job)
        ensure_unused_filename!(filename)
        path = path_for_filename(filename)
        raise ArgumentError.new("job output file is missing") unless File.file?(path)

        kind = job["kind"].as_s
        raise ArgumentError.new("unsupported job kind") unless kind.in?("generate", "export")
        raise ArgumentError.new("asset capacity exhausted") if assets.size >= MAX_ASSETS
        asset_id = new_id
        asset = object({
          "id"         => string(asset_id),
          "name"       => string(filename),
          "kind"       => string(kind == "generate" ? "take" : "export"),
          "filename"   => string(filename),
          "format"     => string(file_format(filename)),
          "seconds"    => number(seconds),
          "created_at" => integer(timestamp),
        } of String => JSON::Any)

        if session_id = job["session_id"]?.try(&.as_s?)
          asset.as_h["session_id"] = string(session_id)
        end
        if kind == "generate"
          segment_id = required_string(job, "segment_id")
          revision = required_i32(job, "revision")
          asset.as_h["segment_id"] = string(segment_id)
          asset.as_h["revision"] = integer(revision.to_i64)
          if report_value = report.as_h["stopped_on_eos"]?
            stopped = required_bool(report_value, "stopped_on_eos")
            asset.as_h["stopped_on_eos"] = boolean(stopped)
            asset.as_h["truncated"] = boolean(!stopped)
          elsif report_value = report.as_h["truncated"]?
            asset.as_h["truncated"] = boolean(required_bool(report_value, "truncated"))
          end
        else
          selected = job["input"].as_h["asset_ids"].as_a
          asset.as_h["source_asset_ids"] = array(selected.map { |value| copy(value) })
        end

        assets << asset
        if kind == "generate"
          session_id = required_string(job, "session_id")
          segment_id = required_string(job, "segment_id")
          if session = find_optional!(sessions, session_id)
            if segment = find_optional!(session.as_h["segments"].as_a, segment_id)
              if required_i32(segment.as_h, "revision") == required_i32(job, "revision")
                segment.as_h["selected_take_id"] = string(asset_id)
              end
            end
          end
        end

        values = job
        values["status"] = string("complete")
        values["progress"] = integer(100_i64)
        values["asset_id"] = string(asset_id)
        values["report"] = copy(report)
        values["finished_at"] = integer(timestamp)
        copy(asset)
      end
    end

    def fail_job(job_id : String, message : String) : Nil
      validate_id!(job_id, "job id")
      validate_text!(message, "job error", 4_000)
      write do
        job = find!(jobs, job_id, "job").as_h
        ensure_active_job!(job)
        job["status"] = string("failed")
        job["error"] = string(message)
        job["finished_at"] = integer(timestamp)
        nil
      end
    end

    private def empty_database : Hash(String, JSON::Any)
      {
        "version"  => integer(MANIFEST_VERSION),
        "sessions" => array([] of JSON::Any),
        "voices"   => array([] of JSON::Any),
        "assets"   => array([] of JSON::Any),
        "jobs"     => array([] of JSON::Any),
      }
    end

    private def load_database! : Nil
      parsed = JSON.parse(File.read(@manifest_path))
      values = parsed.as_h? || raise ArgumentError.new("studio manifest must be an object")
      version = values["version"]?.try(&.as_i64?) || raise ArgumentError.new("studio manifest is missing version")
      raise ArgumentError.new("unsupported studio manifest version") unless version == MANIFEST_VERSION
      %w(sessions voices assets jobs).each do |key|
        rows = values[key]?.try(&.as_a?) || raise ArgumentError.new("studio manifest is missing #{key}")
        limit = case key
                when "sessions" then MAX_SESSIONS
                when "voices"   then MAX_VOICES
                when "assets"   then MAX_ASSETS
                else                 MAX_JOBS
                end
        raise ArgumentError.new("studio manifest exceeds #{key} limit") if rows.size > limit
      end
      values["sessions"].as_a.each do |session|
        segments = session.as_h["segments"]?.try(&.as_a?) || raise ArgumentError.new("studio session is missing segments")
        raise ArgumentError.new("studio manifest exceeds segments per session limit") if segments.size > MAX_SEGMENTS
      end
      @database = values
    end

    private def recover_interrupted_jobs! : Nil
      changed = false
      jobs.each do |raw_job|
        job = raw_job.as_h
        status = job["status"]?.try(&.as_s?)
        if status && status.in?("queued", "running")
          job["status"] = string("failed")
          job["error"] = string("interrupted by restart")
          job["finished_at"] = integer(timestamp)
          changed = true
        end
      end
      persist! if changed
    end

    # Tombstones are named by asset ID. If the manifest still has that ID,
    # deletion did not commit and startup restores the original file. Otherwise
    # only cleanup remains; failed unlinks stay discoverable for next startup.
    private def recover_asset_deletions! : Nil
      return unless File.directory?(@trash_dir)

      Dir.children(@trash_dir).each do |name|
        match = TOMBSTONE_PATTERN.match(name)
        next unless match
        id = match[1]
        tombstone = File.join(@trash_dir, name)

        if asset = find_optional!(assets, id)
          path = path_for_filename(required_string(asset.as_h, "filename"))
          next if File.exists?(path)

          begin
            File.rename(tombstone, path)
          rescue
            # Keep it for a future startup attempt; the manifest still owns it.
          end
        else
          remove_tombstone_best_effort(tombstone)
        end
      end
    end

    private def remove_tombstone_best_effort(path : String) : Nil
      File.delete(path) if File.exists?(path)
    rescue
      # A committed deletion is not rolled back because filesystem cleanup failed.
    end

    private def persist! : Nil
      tmp = "#{@manifest_path}.tmp-#{Process.pid}-#{UUID.random}"
      begin
        File.open(tmp, "w") do |file|
          file.print(JSON::Any.new(@database).to_json)
          file << '\n'
          file.flush
        end
        File.rename(tmp, @manifest_path)
      ensure
        File.delete(tmp) if File.exists?(tmp)
      end
    end

    private def write(&block : -> T) : T forall T
      @mutex.synchronize do
        snapshot = JSON.parse(JSON::Any.new(@database).to_json).as_h
        begin
          result = yield
          persist!
          result
        rescue error
          @database = snapshot
          raise error
        end
      end
    end

    private def sessions : Array(JSON::Any)
      @database["sessions"].as_a
    end

    private def voices : Array(JSON::Any)
      @database["voices"].as_a
    end

    private def assets : Array(JSON::Any)
      @database["assets"].as_a
    end

    private def jobs : Array(JSON::Any)
      @database["jobs"].as_a
    end

    private def find!(rows : Array(JSON::Any), id : String, label : String) : JSON::Any
      find_optional!(rows, id) || raise KeyError.new("unknown #{label}")
    end

    private def find_optional!(rows : Array(JSON::Any), id : String) : JSON::Any?
      rows.find { |row| row.as_h["id"]?.try(&.as_s?) == id }
    end

    private def asset_is_selected?(asset_id : String) : Bool
      sessions.any? do |session|
        session.as_h["segments"].as_a.any? do |segment|
          segment.as_h["selected_take_id"]?.try(&.as_s?) == asset_id
        end
      end
    end

    private def active_job_uses_asset?(asset_id : String) : Bool
      jobs.any? do |raw_job|
        job = raw_job.as_h
        next false unless job["status"].as_s.in?("queued", "running")
        if job["kind"].as_s == "export"
          if input = job["input"]?.try(&.as_h?)
            input["asset_ids"]?.try(&.as_a?).try(&.any? { |value| value.as_s? == asset_id }) || false
          else
            false
          end
        elsif input = job["input"]?.try(&.as_h?)
          input["reference_asset_id"]?.try(&.as_s?) == asset_id
        else
          false
        end
      end
    end

    private def active_voice_uses_asset?(asset_id : String) : Bool
      voices.any? do |raw_voice|
        voice = raw_voice.as_h
        !voice["archived"].as_bool && voice["asset_id"]?.try(&.as_s?) == asset_id
      end
    end

    private def ensure_editable_session!(session : Hash(String, JSON::Any)) : Nil
      raise ConflictError.new("session is archived") if session["archived"].as_bool
    end

    private def ensure_active_job!(job : Hash(String, JSON::Any)) : Nil
      status = required_string(job, "status")
      raise ConflictError.new("job is not active") unless status.in?("queued", "running")
    end

    private def ensure_running_job!(job : Hash(String, JSON::Any)) : Nil
      raise ConflictError.new("job is not running") unless required_string(job, "status") == "running"
    end

    private def ensure_unused_filename!(filename : String) : Nil
      raise ConflictError.new("asset filename already exists") if assets.any? { |asset| asset.as_h["filename"].as_s == filename }
    end

    private def ensure_capacity!(rows : Array(JSON::Any), limit : Int32, label : String) : Nil
      raise ArgumentError.new("#{label} limit reached (#{limit})") if rows.size >= limit
    end

    # Every active generation or export can create one immutable asset.
    # Reserve the slot at enqueue time so later completion cannot be lost.
    private def ensure_asset_capacity!(additional_reservations : Int32) : Nil
      reserved = jobs.count do |raw_job|
        job = raw_job.as_h
        job["status"].as_s.in?("queued", "running") && job["kind"].as_s.in?("generate", "export")
      end
      if assets.size + reserved + additional_reservations > MAX_ASSETS
        raise ArgumentError.new("asset capacity exhausted")
      end
    end

    private def parse_voice_id(value : JSON::Any) : String?
      return nil if value.raw.nil?
      id = value.as_s? || invalid("voice_id must be a string or null")
      validate_id!(id, "voice id")
      id
    end

    private def validate_id!(id : String, label : String) : Nil
      raise ArgumentError.new("invalid #{label}") unless ID_PATTERN.matches?(id)
    end

    private def validate_filename!(filename : String) : Nil
      unless FILENAME_PATTERN.matches?(filename) && File.basename(filename) == filename && filename != "." && filename != ".."
        raise ArgumentError.new("filename must be a safe basename")
      end
      raise ArgumentError.new("filename must have a format extension") if file_format(filename).empty?
    end

    private def path_for_filename(filename : String) : String
      validate_filename!(filename)
      File.join(@asset_dir, filename)
    end

    private def file_format(filename : String) : String
      extension = File.extname(filename)
      extension.empty? ? "" : extension[1..].downcase
    end

    private def validate_label!(value : String, label : String) : Nil
      validate_text!(value, label, MAX_NAME_BYTES)
    end

    private def validate_text!(value : String, label : String, max_bytes : Int32, allow_empty : Bool = false) : Nil
      raise ArgumentError.new("#{label} must not be empty") if !allow_empty && value.strip.empty?
      raise ArgumentError.new("#{label} exceeds #{max_bytes} bytes") if value.bytesize > max_bytes
    end

    private def validate_seconds!(seconds : Float64) : Nil
      raise ArgumentError.new("seconds must be finite and non-negative") unless seconds.finite? && seconds >= 0.0
    end

    private def validate_sampling!(temperature : Float64, top_p : Float64) : Nil
      unless temperature.finite? && temperature >= 0.0 && temperature <= 2.0 && top_p.finite? && top_p > 0.0 && top_p <= 1.0
        raise ArgumentError.new("invalid sampling parameters")
      end
    end

    private def required_string(fields : Hash(String, JSON::Any), key : String) : String
      value = fields[key]? || invalid("missing #{key}")
      value.as_s? || invalid("#{key} must be a string")
    end

    private def required_bool(value : JSON::Any, label : String) : Bool
      case raw = value.raw
      when Bool
        raw
      else
        invalid("#{label} must be a boolean")
      end
    end

    private def required_i32(fields : Hash(String, JSON::Any), key : String) : Int32
      value = fields[key]? || invalid("missing #{key}")
      parsed = integer_value(value, key)
      raise ArgumentError.new("#{key} out of range") unless parsed >= 0 && parsed <= Int32::MAX
      parsed.to_i32
    end

    private def integer_value(value : JSON::Any, label : String) : Int64
      value.as_i64? || invalid("#{label} must be an integer")
    end

    private def optional_i64(fields : Hash(String, JSON::Any), key : String, old : Hash(String, JSON::Any)?, default : Int64) : Int64
      if value = fields[key]?
        integer_value(value, key)
      elsif old
        integer_value(old.not_nil![key], key)
      else
        default
      end
    end

    private def optional_bounded_i32(fields : Hash(String, JSON::Any), key : String, old : Hash(String, JSON::Any)?, default : Int32, low : Int32, high : Int32) : Int32
      raw = if value = fields[key]?
              value
            elsif old
              old.not_nil![key]
            else
              integer(default.to_i64)
            end
      parsed = integer_value(raw, key)
      raise ArgumentError.new("#{key} must be #{low}..#{high}") unless parsed >= low && parsed <= high
      parsed.to_i32
    end

    private def optional_number(fields : Hash(String, JSON::Any), key : String, old : Hash(String, JSON::Any)?, default : Float64) : Float64
      raw = if value = fields[key]?
              value
            elsif old
              old.not_nil![key]
            else
              number(default)
            end
      raw.as_f? || raw.as_i64?.try(&.to_f64) || invalid("#{key} must be a number")
    end

    private def nullable_string(value : JSON::Any?) : String?
      return nil unless value
      return nil if value.not_nil!.raw.nil?
      value.not_nil!.as_s? || invalid("expected a string or null")
    end

    private def timestamp : Int64
      Time.utc.to_unix_ms
    end

    private def new_id : String
      UUID.random.to_s.downcase
    end

    private def copy(value : JSON::Any) : JSON::Any
      JSON.parse(value.to_json)
    end

    private def string(value : String) : JSON::Any
      JSON::Any.new(value)
    end

    private def boolean(value : Bool) : JSON::Any
      JSON::Any.new(value)
    end

    private def integer(value : Int64) : JSON::Any
      JSON::Any.new(value)
    end

    private def number(value : Float64) : JSON::Any
      JSON::Any.new(value)
    end

    private def nullable(value : String?) : JSON::Any
      JSON::Any.new(value)
    end

    private def array(value : Array(JSON::Any)) : JSON::Any
      JSON::Any.new(value)
    end

    private def object(values : Hash(String, JSON::Any)) : JSON::Any
      JSON::Any.new(values)
    end

    private def invalid(message : String) : NoReturn
      raise ArgumentError.new(message)
    end
  end
end
