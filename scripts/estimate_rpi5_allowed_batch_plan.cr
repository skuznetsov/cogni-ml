#!/usr/bin/env crystal

def usage
  STDERR.puts "usage: crystal scripts/estimate_rpi5_allowed_batch_plan.cr MODEL.gguf"
  STDERR.puts
  STDERR.puts "Environment:"
  STDERR.puts "  LABEL_REGEX='REGEX'  Frontier labels to include."
  STDERR.puts "  MAX_BATCH=3          Max frontiers per grouped submit."
  STDERR.puts "  MIN_ALLOWED=1        Skip frontiers below this size."
  exit 2
end

record Frontier,
  label : String,
  allowed : Int32,
  route : String,
  v3d_ms : Float64,
  cpu_ms : Float64,
  ids_csv : String

model_path = ARGV[0]? || usage
label_regex = Regex.new(ENV["LABEL_REGEX"]? || "^(tool_call_prefix:start|finite_values:read_file\\.limit|finite_values:edit_mode\\.mode)$")
max_batch = (ENV["MAX_BATCH"]? || "3").to_i
min_allowed = (ENV["MIN_ALLOWED"]? || "1").to_i
raise "MAX_BATCH must be positive" unless max_batch > 0
raise "MIN_ALLOWED must be positive" unless min_allowed > 0

estimator = "scripts/estimate_qwen35_allowed_frontiers.cr"
unless File.exists?(estimator)
  STDERR.puts "run from the cogni-ml repository root"
  exit 2
end

output = IO::Memory.new
status = Process.run("crystal", [estimator, model_path], output: output, error: STDERR)
exit status.exit_code unless status.success?

frontiers = [] of Frontier
output.to_s.each_line do |line|
  next if line.empty? || line.includes?("=") && !line.includes?("\tallowed=")
  fields = line.split('\t')
  next if fields.size < 2
  label = fields[0]
  next unless label =~ label_regex

  allowed = nil
  route = ""
  v3d_ms = nil
  cpu_ms = nil
  ids_csv = ""
  fields[1..].each do |field|
    key, value = field.split("=", 2)
    case key
    when "allowed"
      allowed = value.to_i
    when "route"
      route = value
    when "est_v3d_ms"
      v3d_ms = value.to_f64
    when "est_cpu_ms"
      cpu_ms = value.to_f64
    when "ids_csv"
      ids_csv = value
    end
  end
  next unless allowed && v3d_ms && cpu_ms && !ids_csv.empty?
  next if allowed.not_nil! < min_allowed
  frontiers << Frontier.new(label, allowed.not_nil!, route, v3d_ms.not_nil!, cpu_ms.not_nil!, ids_csv)
end

if frontiers.empty?
  STDERR.puts "no frontiers matched label_regex=#{label_regex.inspect} min_allowed=#{min_allowed}"
  exit 1
end

# Measured warmed q6idx batch proxy, allowed=8 read_file.limit:
# batch1 0.168ms, batch2 0.286ms, batch4 0.496ms, batch8 1.009ms.
# Multi-frontier proof for allowed 3/8/13 in q6idx13_l256 measured 0.393ms,
# which matches interpolation between batch2 and batch4. Use these only as a
# route-planning model; product promotion still needs adapter measurements.
BATCH_TOTAL_FACTORS = {
  1 => 1.0,
  2 => 0.286 / 0.168,
  4 => 0.496 / 0.168,
  8 => 1.009 / 0.168,
}

def batch_factor(batch : Int32) : Float64
  return BATCH_TOTAL_FACTORS[batch] if BATCH_TOTAL_FACTORS.has_key?(batch)

  keys = BATCH_TOTAL_FACTORS.keys.sort
  if batch <= keys.first
    return BATCH_TOTAL_FACTORS[keys.first] * batch.to_f / keys.first.to_f
  end
  keys.each_cons(2) do |pair|
    lo, hi = pair[0], pair[1]
    if batch <= hi
      t = (batch - lo).to_f / (hi - lo).to_f
      return BATCH_TOTAL_FACTORS[lo] + t * (BATCH_TOTAL_FACTORS[hi] - BATCH_TOTAL_FACTORS[lo])
    end
  end
  last = keys.last
  BATCH_TOTAL_FACTORS[last] * batch.to_f / last.to_f
end

def estimate_group_gpu_ms(group : Array(Frontier)) : Float64
  max_allowed = group.max_of(&.allowed)
  return 0.393 if group.size == 3 && max_allowed <= 13

  max_v3d = group.max_of(&.v3d_ms)
  max_v3d * batch_factor(group.size)
end

groups = frontiers.each_slice(max_batch).to_a
single_cpu_ms = frontiers.sum(&.cpu_ms)
single_v3d_ms = frontiers.sum(&.v3d_ms)
grouped_gpu_ms = groups.sum { |g| estimate_group_gpu_ms(g) }

puts "model=#{model_path}"
puts "label_regex=#{label_regex.inspect}"
puts "max_batch=#{max_batch}"
puts "frontiers=#{frontiers.size}"
puts "batch_cost_model=measured warmed q6idx proxy; product adapter still required"
puts "single_cpu_ms=#{single_cpu_ms.round(4)}"
puts "single_v3d_ms=#{single_v3d_ms.round(4)}"
puts "grouped_v3d_ms=#{grouped_gpu_ms.round(4)}"
puts "grouped_vs_cpu=#{(single_cpu_ms / grouped_gpu_ms).round(3)}x"
puts "grouped_vs_unbatched_v3d=#{(single_v3d_ms / grouped_gpu_ms).round(3)}x"
puts

groups.each_with_index do |group, idx|
  max_allowed = group.max_of(&.allowed)
  cpu_ms = group.sum(&.cpu_ms)
  single_gpu = group.sum(&.v3d_ms)
  grouped_gpu = estimate_group_gpu_ms(group)
  labels = group.map(&.label).join(",")
  ids_groups = group.map(&.ids_csv).join(":")
  puts "batch_group\tidx=#{idx + 1}\tbatch=#{group.size}\tmax_allowed=#{max_allowed}\tcpu_ms=#{cpu_ms.round(4)}\tsingle_v3d_ms=#{single_gpu.round(4)}\tgrouped_v3d_ms=#{grouped_gpu.round(4)}\tspeedup_vs_cpu=#{(cpu_ms / grouped_gpu).round(3)}x\tlabels=#{labels}\tids_groups=#{ids_groups}"
end
