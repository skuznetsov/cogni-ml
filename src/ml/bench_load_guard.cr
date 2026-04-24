module ML::BenchLoadGuard
  record ProcessLoad, pid : Int64, cpu : Float64, command : String

  def self.busy_processes(threshold_pct : Float64,
                          self_pid : Int64 = Process.pid) : Array(ProcessLoad)
    return [] of ProcessLoad if threshold_pct <= 0.0

    output = IO::Memory.new
    status = Process.run("ps", args: ["-Ao", "pid=,pcpu=,comm="], output: output, error: Process::Redirect::Close)
    return [] of ProcessLoad unless status.success?

    loads = [] of ProcessLoad
    output.to_s.each_line do |line|
      fields = line.strip.split(/\s+/, 3)
      next unless fields.size >= 3

      pid = fields[0].to_i?
      cpu = fields[1].to_f?
      command = fields[2]
      next unless pid && cpu
      next if pid == self_pid
      next if cpu < threshold_pct

      loads << ProcessLoad.new(pid, cpu, command)
    end
    loads.sort_by { |load| -load.cpu }
  rescue
    [] of ProcessLoad
  end

  def self.warn_if_busy(threshold_pct : Float64, io : IO = STDERR) : Nil
    busy = busy_processes(threshold_pct)
    report_busy(busy, threshold_pct, io)
  end

  def self.require_quiet!(threshold_pct : Float64, io : IO = STDERR) : Nil
    busy = busy_processes(threshold_pct)
    return if busy.empty?

    report_busy(busy, threshold_pct, io)
    raise "benchmark host is not quiet; lower load or pass --load-warning-threshold=0 to bypass"
  end

  private def self.report_busy(busy : Array(ProcessLoad),
                               threshold_pct : Float64,
                               io : IO) : Nil
    return if busy.empty?

    io.puts "WARNING: host CPU load may contaminate benchmark results."
    io.puts "         Processes above #{threshold_pct.round(1)}% CPU:"
    busy.first(5).each do |load|
      io.printf "         pid=%d cpu=%.1f%% cmd=%s\n", load.pid, load.cpu, load.command
    end
    io.puts "         Re-run with --load-warning-threshold=0 to suppress this warning."
  end
end
