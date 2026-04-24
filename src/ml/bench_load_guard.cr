module ML::BenchLoadGuard
  record ProcessLoad, pid : Int64, cpu : Float64, command : String
  record HostLoad, busy_processes : Array(ProcessLoad), total_cpu : Float64,
    per_process_threshold : Float64, total_threshold : Float64 do
    def busy? : Bool
      !busy_processes.empty? ||
        (total_threshold > 0.0 && total_cpu >= total_threshold)
    end
  end

  def self.process_loads(self_pid : Int64 = Process.pid) : Array(ProcessLoad)
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

      loads << ProcessLoad.new(pid, cpu, command)
    end
    loads.sort_by { |load| -load.cpu }
  rescue
    [] of ProcessLoad
  end

  def self.sample(per_process_threshold : Float64,
                  total_threshold : Float64,
                  self_pid : Int64 = Process.pid) : HostLoad
    loads = process_loads(self_pid)
    busy = if per_process_threshold > 0.0
             loads.select { |load| load.cpu >= per_process_threshold }
           else
             [] of ProcessLoad
           end
    HostLoad.new(
      busy,
      loads.sum(0.0) { |load| load.cpu },
      per_process_threshold,
      total_threshold
    )
  end

  def self.warn_if_busy(per_process_threshold : Float64,
                        total_threshold : Float64 = 0.0,
                        io : IO = STDERR) : Nil
    report(sample(per_process_threshold, total_threshold), io)
  end

  def self.require_quiet!(per_process_threshold : Float64,
                          total_threshold : Float64 = 0.0,
                          io : IO = STDERR) : Nil
    host_load = sample(per_process_threshold, total_threshold)
    return unless host_load.busy?

    report(host_load, io)
    raise "benchmark host is not quiet; lower load or pass --load-warning-threshold=0 --load-total-warning-threshold=0 to bypass"
  end

  private def self.report(host_load : HostLoad, io : IO) : Nil
    return unless host_load.busy?

    io.puts "WARNING: host CPU load may contaminate benchmark results."
    if host_load.total_threshold > 0.0 && host_load.total_cpu >= host_load.total_threshold
      io.printf "         Total observed CPU %.1f%% exceeds %.1f%%.\n",
        host_load.total_cpu, host_load.total_threshold
    end
    unless host_load.busy_processes.empty?
      io.puts "         Processes above #{host_load.per_process_threshold.round(1)}% CPU:"
      host_load.busy_processes.first(5).each do |load|
        io.printf "         pid=%d cpu=%.1f%% cmd=%s\n", load.pid, load.cpu, load.command
      end
    end
    io.puts "         Re-run with --load-warning-threshold=0 --load-total-warning-threshold=0 to suppress this warning."
  end
end
