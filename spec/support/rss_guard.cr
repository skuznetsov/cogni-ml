# The watchdog runs outside the spec process so it still runs when the Crystal
# scheduler is blocked in a native call. Its preflight runs before discovery.
module CogniSpecRSSGuard
  extend self

  DEFAULT_MAX_RSS_MB = 16 * 1024
  DEFAULT_POLL_SEC   = 0.5
  DEFAULT_TIMEOUT    = 120
  DEFAULT_MIN_FREE   = {{ flag?(:darwin) ? 25 : 0 }}
  EXIT_CODE          = 99

  WATCHDOG = <<-'BASH'
    pid=$1
    state=$2
    max_kb=$3
    min_free=$4
    poll=$5
    timeout=$6
    ready=0
    previous=
    export LC_ALL=C

    trap 'rm -f "$state" "$state.next"; rmdir "${state%/*}" 2>/dev/null' EXIT

    owns_spec() {
      local parent
      read -r parent <<< "$(/bin/ps -o ppid= -p "$$" 2>/dev/null)"
      [[ "$parent" == "$pid" ]]
    }

    fail() {
      owns_spec || exit 0
      printf '[SPEC_%s_GUARD] %s: %s\n' "$1" "$phase" "$2" >&2
      # After preflight the spec may be blocked inside the driver. Do not wait
      # for its signal handlers or run further examples in this process.
      if (( ready )); then
        kill -KILL "$pid" 2>/dev/null
      fi
      exit 99
    }

    while owns_spec; do
      { IFS= read -r sequence && IFS= read -r phase; } < "$state" || exit 0
      if [[ "$sequence" != "$previous" ]]; then
        started=$SECONDS
        previous=$sequence
      fi
      if (( timeout > 0 && SECONDS - started > timeout )); then
        # The example may have finished while we sampled its last state.
        IFS= read -r current < "$state" || exit 0
        if [[ "$sequence" == "$current" ]]; then
          fail TIMEOUT "exceeded ${timeout}s"
        fi
        continue
      fi
      if (( max_kb > 0 )); then
        read -r rss <<< "$(/bin/ps -o rss= -p "$pid" 2>/dev/null)"
        [[ "$rss" =~ ^[0-9]+$ ]] || fail RSS "cannot read process RSS"
        if (( rss > max_kb )); then
          fail RSS "RSS ${rss} KiB > $((max_kb / 1024)) MiB"
        fi
      fi
      if (( min_free > 0 )); then
        snapshot=$(memory_pressure -Q 2>/dev/null) || fail PRESSURE "cannot read system memory headroom"
        free=$(awk '/System-wide memory free percentage:/ { sub(/%$/, "", $NF); print $NF }' <<< "$snapshot")
        [[ "$free" =~ ^(100|[1-9]?[0-9])$ ]] || fail PRESSURE "cannot read system memory headroom"
        if (( free <= min_free )); then
          fail PRESSURE "system free ${free}% <= ${min_free}%"
        fi
      fi
      if (( !ready )); then
        printf 'ready\n'
        ready=1
      fi
      sleep "$poll"
    done
    BASH

  @@watchdog : Process? = nil
  @@state : String? = nil
  @@sequence = 0_i64

  private def setting(name : String, default : Int32, max = Int32::MAX) : Int32
    return default unless raw = ENV[name]?
    value = raw.to_i?
    abort_guard("#{name} must be an integer between 0 and #{max}") unless value && value.in?(0..max)
    value
  end

  private def poll_sec : Float64
    return DEFAULT_POLL_SEC unless raw = ENV["COGNI_SPEC_RSS_POLL_SEC"]?
    value = raw.to_f?
    abort_guard("COGNI_SPEC_RSS_POLL_SEC must be between 0.1 and 1") unless value && value.in?(0.1..1.0)
    value
  end

  private def abort_guard(message : String) : NoReturn
    STDERR.puts "[SPEC_RESOURCE_GUARD] #{message}"
    exit EXIT_CODE
  end

  def start! : Nil
    return if ENV["COGNI_SPEC_RSS_GUARD"]? == "0"
    max_rss_mb = setting("COGNI_SPEC_MAX_RSS_MB", DEFAULT_MAX_RSS_MB)
    min_free = setting("COGNI_SPEC_MIN_FREE_PCT", DEFAULT_MIN_FREE, 100)
    timeout = setting("COGNI_SPEC_EXAMPLE_TIMEOUT_SEC", DEFAULT_TIMEOUT)
    return if max_rss_mb == 0 && min_free == 0 && timeout == 0
    poll = poll_sec

    dir = File.tempname("cogni-spec-watchdog", nil)
    Dir.mkdir(dir, 0o700)
    state = File.join(dir, "activity")
    @@state = state
    record!("spec discovery or suite hooks")

    watchdog = Process.new("/bin/bash", ["-c", WATCHDOG, "cogni-spec-watchdog",
                                         Process.pid.to_s, state, (max_rss_mb.to_i64 * 1024).to_s,
                                         min_free.to_s, poll.to_s, timeout.to_s],
      output: Process::Redirect::Pipe, error: Process::Redirect::Inherit)
    @@watchdog = watchdog
    watchdog.output.read_timeout = 5.seconds
    unless watchdog.output.gets == "ready"
      stop!
      abort_guard("preflight failed")
    end
    watchdog.output.close
    STDERR.puts "[SPEC_RESOURCE_GUARD] RSS=#{max_rss_mb} MiB, system free>#{min_free}%, example timeout=#{timeout}s, poll=#{poll}s"
  rescue ex : IO::Error
    stop!
    abort_guard("could not start watchdog: #{ex.message}")
  end

  def record!(phase : String) : Nil
    if state = @@state
      @@sequence += 1
      # Publish a whole record even if the observer reads during this update.
      File.write("#{state}.next", "#{@@sequence}\n#{phase}\n")
      File.rename("#{state}.next", state)
    end
  end

  def stop! : Nil
    if watchdog = @@watchdog
      watchdog.terminate(graceful: false)
      watchdog.wait
      @@watchdog = nil
    end
    if state = @@state
      File.delete?(state)
      File.delete?("#{state}.next")
      dir = File.dirname(state)
      Dir.delete(dir) if Dir.exists?(dir)
      @@state = nil
    end
  end
end

Spec.around_each do |example|
  item = example.example
  location = "#{item.file}:#{item.line} #{item.description.inspect}"
  CogniSpecRSSGuard.record!(location)
  begin
    example.run
  ensure
    CogniSpecRSSGuard.record!("suite hooks after #{location}")
  end
end

Spec.after_suite do
  CogniSpecRSSGuard.stop!
end

CogniSpecRSSGuard.start!
