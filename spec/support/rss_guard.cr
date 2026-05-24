# Opt-in RSS guard for heavy specs.
#
# Enable with:
#   COGNI_SPEC_MAX_RSS_MB=12288 crystal spec ...
# Optional:
#   COGNI_SPEC_RSS_POLL_SEC=1
#   COGNI_SPEC_RSS_GUARD=0   # disable even if max is set
#
# This runs inside `crystal-run-spec.tmp`, complementing scripts/run_safe.sh.
# The outer script bounds the parent/process tree; this guard watches the
# actual spec executable's RSS and terminates it before machine-wide pressure
# becomes unrecoverable.
module CogniSpecRSSGuard
  extend self

  DEFAULT_POLL_SEC = 1.0
  EXIT_CODE        =  99

  @@watchdog : Process? = nil
  @@enabled = false

  def enabled? : Bool
    return false if ENV["COGNI_SPEC_RSS_GUARD"]? == "0"
    limit_mb? > 0
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

  def current_rss_kb : Int64?
    rss_out = IO::Memory.new
    status = Process.run("ps", ["-o", "rss=", "-p", Process.pid.to_s], output: rss_out, error: Process::Redirect::Close)
    return nil unless status.success?
    rss_out.to_s.strip.to_i64?
  end

  def check!(label : String) : Nil
    max_kb = limit_mb?.to_i64 * 1024_i64
    return if max_kb <= 0
    rss = current_rss_kb
    return unless rss && rss > max_kb
    STDERR.puts "[SPEC_RSS_GUARD] #{label}: RSS #{rss}KB > #{limit_mb?}MB; exiting #{EXIT_CODE}"
    STDERR.flush
    exit EXIT_CODE
  end

  def start! : Nil
    return unless enabled?
    return if @@enabled
    @@enabled = true

    pid = Process.pid
    max_kb = limit_mb?.to_i64 * 1024_i64
    poll = poll_sec
    script = String.build do |s|
      s << "pid=" << pid << "\n"
      s << "max_kb=" << max_kb << "\n"
      s << "poll=" << poll << "\n"
      s << "while kill -0 \"$pid\" 2>/dev/null; do\n"
      s << "  rss=$(ps -o rss= -p \"$pid\" 2>/dev/null | tr -d ' ')\n"
      s << "  if [ -n \"$rss\" ] && [ \"$rss\" -gt \"$max_kb\" ]; then\n"
      s << "    echo \"[SPEC_RSS_GUARD] async: RSS ${rss}KB > $((max_kb / 1024))MB; terminating pid $pid\" >&2\n"
      s << "    kill -TERM \"$pid\" 2>/dev/null || true\n"
      s << "    sleep 1\n"
      s << "    kill -KILL \"$pid\" 2>/dev/null || true\n"
      s << "    exit 0\n"
      s << "  fi\n"
      s << "  sleep \"$poll\"\n"
      s << "done\n"
    end

    @@watchdog = Process.new("/bin/sh", ["-c", script],
      input: Process::Redirect::Close,
      output: Process::Redirect::Inherit,
      error: Process::Redirect::Inherit)

    STDERR.puts "[SPEC_RSS_GUARD] enabled: max=#{limit_mb?}MB poll=#{poll_sec}s pid=#{pid}"
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
