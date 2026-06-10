#!/usr/bin/env crystal

def usage
  STDERR.puts "usage: crystal scripts/plan_rpi5_from_frontier_trace.cr TRACE.log [TRACE2.log ...]"
  STDERR.puts "       cat TRACE.log | crystal scripts/plan_rpi5_from_frontier_trace.cr -"
  STDERR.puts
  STDERR.puts "Environment:"
  STDERR.puts "  MAX_BATCH=3       Max V3D frontiers per grouped submit."
  STDERR.puts "  CPU_TINY_MAX=3    Route frontiers at or below this size to CPU."
  exit 2
end

record TraceRow,
  stage : String,
  pos : Int32,
  allowed : Int32,
  ids_csv : String

paths = ARGV
usage if paths.empty?

max_batch = (ENV["MAX_BATCH"]? || "3").to_i
cpu_tiny_max = (ENV["CPU_TINY_MAX"]? || "3").to_i
raise "MAX_BATCH must be positive" unless max_batch > 0
raise "CPU_TINY_MAX must be non-negative" unless cpu_tiny_max >= 0

rows = [] of TraceRow
pattern = /constraint frontier trace:\s+stage=([^\s]+)\s+pos=(\d+)\s+allowed=(\d+)\s+ids_csv=([0-9,]+)/

paths.each do |path|
  input = path == "-" ? STDIN.gets_to_end : File.read(path)
  input.each_line do |line|
    next unless match = line.match(pattern)

    rows << TraceRow.new(
      stage: match[1],
      pos: match[2].to_i,
      allowed: match[3].to_i,
      ids_csv: match[4],
    )
  end
end

if rows.empty?
  STDERR.puts "no constraint frontier trace rows found"
  exit 1
end

PI_ROUTE_POINTS = [
  {3, 0.180, 0.105},
  {8, 0.186, 0.299},
  {13, 0.189, 0.475},
  {16, 0.191, 0.604},
  {32, 0.328, 1.188},
  {64, 0.380, 2.383},
  {128, 0.610, 4.815},
  {256, 0.922, 9.642},
  {1024, 2.734, 38.846},
  {4096, 8.621, 155.674},
  {8192, 12.865, 309.318},
  {16384, 22.124, 621.220},
]

def interpolate_pi_route(count : Int32) : {Float64, Float64}
  return {0.0, 0.0} if count <= 0
  points = PI_ROUTE_POINTS
  if count <= points[0][0]
    scale = count.to_f / points[0][0].to_f
    return {points[0][1] * scale, points[0][2] * scale}
  end
  points.each_cons(2) do |pair|
    x0, v3d0, cpu0 = pair[0]
    x1, v3d1, cpu1 = pair[1]
    if count <= x1
      t = (count - x0).to_f / (x1 - x0).to_f
      return {v3d0 + t * (v3d1 - v3d0), cpu0 + t * (cpu1 - cpu0)}
    end
  end
  x, v3d, cpu = points[-1]
  scale = count.to_f / x.to_f
  {v3d * scale, cpu * scale}
end

BATCH_TOTAL_FACTORS = {
  1 => 1.0,
  2 => 0.286 / 0.168,
  4 => 0.496 / 0.168,
  8 => 1.009 / 0.168,
}

def batch_factor(batch : Int32) : Float64
  return BATCH_TOTAL_FACTORS[batch] if BATCH_TOTAL_FACTORS.has_key?(batch)
  keys = BATCH_TOTAL_FACTORS.keys.sort
  keys.each_cons(2) do |pair|
    lo, hi = pair[0], pair[1]
    if batch <= hi
      t = (batch - lo).to_f / (hi - lo).to_f
      return BATCH_TOTAL_FACTORS[lo] + t * (BATCH_TOTAL_FACTORS[hi] - BATCH_TOTAL_FACTORS[lo])
    end
  end
  BATCH_TOTAL_FACTORS[keys.last] * batch.to_f / keys.last.to_f
end

def estimate_group_gpu_ms(group : Array(TraceRow)) : Float64
  max_allowed = group.max_of(&.allowed)
  return 0.281 if group.size == 2 && max_allowed <= 13
  return 0.393 if group.size == 3 && max_allowed <= 13
  max_v3d = group.max_of { |row| interpolate_pi_route(row.allowed)[0] }
  max_v3d * batch_factor(group.size)
end

cpu_rows = rows.select { |row| row.allowed <= cpu_tiny_max }
v3d_rows = rows.reject { |row| row.allowed <= cpu_tiny_max }
v3d_groups = v3d_rows.each_slice(max_batch).to_a

single_cpu_ms = rows.sum { |row| interpolate_pi_route(row.allowed)[1] }
hybrid_cpu_ms = cpu_rows.sum { |row| interpolate_pi_route(row.allowed)[1] }
hybrid_v3d_ms = v3d_groups.sum { |group| estimate_group_gpu_ms(group) }
hybrid_total_ms = hybrid_cpu_ms + hybrid_v3d_ms

puts "trace_rows=#{rows.size}"
puts "max_batch=#{max_batch}"
puts "cpu_tiny_max=#{cpu_tiny_max}"
puts "single_cpu_ms=#{single_cpu_ms.round(4)}"
puts "hybrid_cpu_tiny_ms=#{hybrid_cpu_ms.round(4)}"
puts "hybrid_grouped_v3d_ms=#{hybrid_v3d_ms.round(4)}"
puts "hybrid_total_ms=#{hybrid_total_ms.round(4)}"
puts "hybrid_vs_cpu=#{(single_cpu_ms / hybrid_total_ms).round(3)}x" if hybrid_total_ms > 0
puts

rows.each_with_index do |row, i|
  v3d, cpu = interpolate_pi_route(row.allowed)
  route = row.allowed <= cpu_tiny_max ? "CPU" : "V3D"
  puts "trace_plan_row\tidx=#{i + 1}\troute=#{route}\tstage=#{row.stage}\tpos=#{row.pos}\tallowed=#{row.allowed}\test_v3d_ms=#{v3d.round(4)}\test_cpu_ms=#{cpu.round(4)}\tids_csv=#{row.ids_csv}"
end

puts
v3d_groups.each_with_index do |group, i|
  labels = group.map { |row| "#{row.stage}@#{row.pos}" }.join(",")
  ids_groups = group.map(&.ids_csv).join(":")
  cpu_ms = group.sum { |row| interpolate_pi_route(row.allowed)[1] }
  grouped_ms = estimate_group_gpu_ms(group)
  puts "trace_v3d_group\tidx=#{i + 1}\tbatch=#{group.size}\tmax_allowed=#{group.max_of(&.allowed)}\tcpu_ms=#{cpu_ms.round(4)}\tgrouped_v3d_ms=#{grouped_ms.round(4)}\tspeedup_vs_cpu=#{(cpu_ms / grouped_ms).round(3)}x\tlabels=#{labels}\tids_groups=#{ids_groups}"
end
