# Process and system-pressure guard for specs.
#
# RSS limiting remains opt-in:
#   COGNI_SPEC_MAX_RSS_MB=12288 crystal spec ...
# System memory-pressure limiting is enabled by default on macOS:
#   COGNI_SPEC_MIN_FREE_PCT=12
#   COGNI_SPEC_MIN_FREE_PCT=0   # explicit opt-out
# Optional:
#   COGNI_SPEC_RSS_POLL_SEC=1
#   COGNI_SPEC_RSS_GUARD=0   # disable even if max is set
#
# This runs inside `crystal-run-spec.tmp`, complementing scripts/run_safe.sh.
# The outer script bounds the parent/process tree. This guard also lives inside
# `crystal-run-spec.tmp`, so a direct `crystal spec` still terminates before
# unified-memory/Metal/compressor pressure becomes unrecoverable.
module CogniSpecRSSGuard
  extend self

  DEFAULT_POLL_SEC = 1.0
  EXIT_CODE        =  99

  @@watchdog : Process? = nil
  @@enabled = false

  def enabled? : Bool
    return false if ENV["COGNI_SPEC_RSS_GUARD"]? == "0"
    limit_mb? > 0 || min_free_pct > 0
  end

  def limit_mb? : Int32
    raw = ENV["COGNI_SPEC_MAX_RSS_MB"]?
    return 0 unless raw
    raw.to_i? || 0
  end

  def poll_sec : Float64
    raw = ENV["COGNI_SPEC_RSS_POLL_SEC"]?
    value = raw.try(&.to_f?) || DEFAULT_POLL_SEC
    value > 0 ? value : DEFAULT_POLL_SEC
  end

  def min_free_pct : Int32
    raw = ENV["COGNI_SPEC_MIN_FREE_PCT"]?
    value = raw.try(&.to_i?) || default_min_free_pct
    value.clamp(0, 100)
  end

  private def default_min_free_pct : Int32
    {% if flag?(:darwin) %}
      12
    {% else %}
      0
    {% end %}
  end

  def current_rss_kb : Int64?
    rss_out = IO::Memory.new
    status = Process.run("ps", ["-o", "rss=", "-p", Process.pid.to_s], output: rss_out, error: Process::Redirect::Close)
    return nil unless status.success?
    rss_out.to_s.strip.to_i64?
  rescue IO::Error
    nil
  end

  def check!(label : String) : Nil
    max_kb = limit_mb?.to_i64 * 1024_i64
    if max_kb > 0
      rss = current_rss_kb
      if rss && rss > max_kb
        STDERR.puts "[SPEC_RSS_GUARD] #{label}: RSS #{rss}KB > #{limit_mb?}MB; exiting #{EXIT_CODE}"
        STDERR.flush
        exit EXIT_CODE
      end
    end

    if min_free_pct > 0
      free_pct = current_free_pct
      if free_pct && free_pct <= min_free_pct
        STDERR.puts "[SPEC_PRESSURE_GUARD] #{label}: system free #{free_pct}% <= #{min_free_pct}%; exiting #{EXIT_CODE}"
        STDERR.flush
        exit EXIT_CODE
      end
    end
  end

  def current_free_pct : Int32?
    output = IO::Memory.new
    status = Process.run("memory_pressure", ["-Q"], output: output, error: Process::Redirect::Close)
    return nil unless status.success?
    match = output.to_s.match(/System-wide memory free percentage:\s*(\d+)%/)
    match.try { |m| m[1].to_i? }
  rescue IO::Error
    nil
  end

  def start! : Nil
    return unless enabled?
    return if @@enabled
    @@enabled = true

    pid = Process.pid
    max_kb = limit_mb?.to_i64 * 1024_i64
    min_free = min_free_pct
    poll = poll_sec
    script = String.build do |s|
      s << "pid=" << pid << "\n"
      s << "max_kb=" << max_kb << "\n"
      s << "min_free=" << min_free << "\n"
      s << "poll=" << poll << "\n"
      s << "while kill -0 \"$pid\" 2>/dev/null; do\n"
      s << "  rss=$(ps -o rss= -p \"$pid\" 2>/dev/null | tr -d ' ')\n"
      s << "  if [ \"$max_kb\" -gt 0 ] && [ -n \"$rss\" ] && [ \"$rss\" -gt \"$max_kb\" ]; then\n"
      s << "    echo \"[SPEC_RSS_GUARD] async: RSS ${rss}KB > $((max_kb / 1024))MB; terminating pid $pid\" >&2\n"
      s << "    kill -TERM \"$pid\" 2>/dev/null || true\n"
      s << "    sleep 1\n"
      s << "    kill -KILL \"$pid\" 2>/dev/null || true\n"
      s << "    exit 0\n"
      s << "  fi\n"
      s << "  if [ \"$min_free\" -gt 0 ] && command -v memory_pressure >/dev/null 2>&1; then\n"
      s << "    free_pct=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/, \"\", $2); print $2; exit}')\n"
      s << "    if [ -n \"$free_pct\" ] && [ \"$free_pct\" -le \"$min_free\" ]; then\n"
      s << "      echo \"[SPEC_PRESSURE_GUARD] async: system free ${free_pct}% <= ${min_free}%; terminating pid $pid\" >&2\n"
      s << "      kill -TERM \"$pid\" 2>/dev/null || true\n"
      s << "      sleep 1\n"
      s << "      kill -KILL \"$pid\" 2>/dev/null || true\n"
      s << "      exit 0\n"
      s << "    fi\n"
      s << "  fi\n"
      s << "  sleep \"$poll\"\n"
      s << "done\n"
    end

    @@watchdog = Process.new("/bin/sh", ["-c", script],
      input: Process::Redirect::Close,
      output: Process::Redirect::Inherit,
      error: Process::Redirect::Inherit)

    STDERR.puts "[SPEC_RESOURCE_GUARD] enabled: max_rss=#{limit_mb?}MB min_free=#{min_free_pct}% poll=#{poll_sec}s pid=#{pid}"
    STDERR.flush
    check!("before_suite")
  end

  def stop! : Nil
    if process = @@watchdog
      process.terminate
      process.wait rescue nil
      @@watchdog = nil
    end
  end
end

Spec.before_suite do
  CogniSpecRSSGuard.start!
end

Spec.before_each do
  CogniSpecRSSGuard.check!("before_each")
end

Spec.after_each do
  CogniSpecRSSGuard.check!("after_each")
end

Spec.after_suite do
  CogniSpecRSSGuard.check!("after_suite")
  CogniSpecRSSGuard.stop!
end
