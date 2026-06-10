#!/usr/bin/env crystal

require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_constraints"

def usage
  STDERR.puts "usage: crystal scripts/estimate_qwen35_tool_frontier_trace.cr MODEL.gguf"
  STDERR.puts
  STDERR.puts "Environment:"
  STDERR.puts "  QWEN35_TOOLS_JSON='[...]'  Tool schema JSON. Defaults to the RPi5 sample tools."
  STDERR.puts "  TOOL_NAME=edit_mode        Chosen function for the deterministic trace."
  STDERR.puts "  MAX_BATCH=3                Max frontiers per planned V3D submit."
  STDERR.puts "  CPU_TINY_MAX=3             Route frontiers at or below this size to CPU."
  exit 2
end

record TraceFrontier,
  stage : String,
  label : String,
  allowed : Int32,
  ids_csv : String

model_path = ARGV[0]? || usage
tool_name = ENV["TOOL_NAME"]? || "edit_mode"
max_batch = (ENV["MAX_BATCH"]? || "3").to_i
cpu_tiny_max = (ENV["CPU_TINY_MAX"]? || "3").to_i
raise "MAX_BATCH must be positive" unless max_batch > 0
raise "CPU_TINY_MAX must be non-negative" unless cpu_tiny_max >= 0

default_tools_json = %([
  {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
    "path":{"type":"string"},
    "limit":{"type":"integer","minimum":1,"maximum":8},
    "exact":{"type":"boolean"}
  },"required":["path"]}}},
  {"type":"function","function":{"name":"list_directory","parameters":{"type":"object","properties":{
    "path":{"type":"string"},
    "recursive":{"type":"boolean"}
  },"required":["path"]}}},
  {"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{
    "pattern":{"type":"string"},
    "path":{"type":"string"},
    "case_sensitive":{"type":"boolean"}
  },"required":["pattern","path"]}}},
  {"type":"function","function":{"name":"edit_mode","parameters":{"type":"object","properties":{
    "mode":{"type":"string","enum":["fast","safe","minimal"]},
    "dry_run":{"type":"boolean"}
  },"required":["mode","dry_run"]}}}
])

tools = ML::GGUF::Qwen35Chat.parse_tools_json(ENV["QWEN35_TOOLS_JSON"]? || default_tools_json)
names = ML::GGUF::Qwen35Constraints.tool_function_names(tools)
unless names.includes?(tool_name)
  STDERR.puts "TOOL_NAME=#{tool_name.inspect} not found in tools: #{names.join(",")}"
  exit 1
end

gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path)
index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tokenizer)
required_by_name = ML::GGUF::Qwen35Constraints.tool_required_parameters(tools)
optional_by_name = ML::GGUF::Qwen35Constraints.tool_optional_parameters(tools)
finite_by_name = ML::GGUF::Qwen35Constraints.tool_finite_parameter_value_options(tools)

def frontier(stage : String,
             label : String,
             literals : Array(String),
             index : ML::GGUF::Qwen35Constraints::TokenTextIndex) : TraceFrontier?
  ids = ML::GGUF::Qwen35Constraints.literal_frontier_ids(index, literals).sort
  return nil if ids.empty?

  TraceFrontier.new(stage, label, ids.size, ids.join(","))
end

trace = [] of TraceFrontier

if row = frontier("function_prefix", "tool_call_prefix:start",
     ML::GGUF::Qwen35Constraints.qwen_tool_call_prefix_options(names), index)
  trace << row
end

required = required_by_name[tool_name]? || [] of String
optional = optional_by_name[tool_name]? || [] of String
finite_values = finite_by_name[tool_name]? || {} of String => Array(String)
parameters = required.dup

if parameters.empty?
  unless optional.empty?
    if row = frontier("optional_or_close", "optional_params:#{tool_name}:continue",
         ML::GGUF::Qwen35Constraints.qwen_parameter_continue_options(optional) +
           ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options, index)
      trace << row
    end
  end
else
  parameters.each_with_index do |parameter, i|
    open_literals = if i == 0
                      ML::GGUF::Qwen35Constraints.qwen_parameter_open_options([parameter])
                    else
                      ML::GGUF::Qwen35Constraints.qwen_parameter_continue_options([parameter])
                    end
    if row = frontier(i == 0 ? "parameter_open" : "parameter_separator",
         "parameter:#{tool_name}.#{parameter}:open", open_literals, index)
      trace << row
    end

    values = finite_values[parameter]?
    if values && !values.empty?
      if row = frontier("value_literal", "finite_values:#{tool_name}.#{parameter}", values, index)
        trace << row
      end
    else
      trace << TraceFrontier.new("freeform_value", "freeform:#{tool_name}.#{parameter}", 0, "")
    end
  end

  close_literals = if optional.empty?
                     ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options
                   else
                     ML::GGUF::Qwen35Constraints.qwen_parameter_continue_options(optional) +
                       ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options
                   end
  if row = frontier(optional.empty? ? "closing_parameter" : "optional_or_close",
       optional.empty? ? "single_parameter_close" : "optional_params:#{tool_name}:continue",
       close_literals, index)
    trace << row
  end
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

def estimate_group_gpu_ms(group : Array(TraceFrontier)) : Float64
  max_allowed = group.max_of(&.allowed)
  return 0.281 if group.size == 2 && max_allowed <= 13
  return 0.393 if group.size == 3 && max_allowed <= 13
  max_v3d = group.max_of { |row| interpolate_pi_route(row.allowed)[0] }
  max_v3d * batch_factor(group.size)
end

ranked = trace.reject { |row| row.allowed <= 0 }
groups = ranked.each_slice(max_batch).to_a
cpu_rows = ranked.select { |row| row.allowed <= cpu_tiny_max }
v3d_rows = ranked.reject { |row| row.allowed <= cpu_tiny_max }
hybrid_groups = v3d_rows.each_slice(max_batch).to_a
single_cpu_ms = ranked.sum { |row| interpolate_pi_route(row.allowed)[1] }
single_v3d_ms = ranked.sum { |row| interpolate_pi_route(row.allowed)[0] }
grouped_v3d_ms = groups.sum { |group| estimate_group_gpu_ms(group) }
hybrid_cpu_ms = cpu_rows.sum { |row| interpolate_pi_route(row.allowed)[1] }
hybrid_v3d_ms = hybrid_groups.sum { |group| estimate_group_gpu_ms(group) }
hybrid_ms = hybrid_cpu_ms + hybrid_v3d_ms

puts "model=#{model_path}"
puts "tool_name=#{tool_name}"
puts "max_batch=#{max_batch}"
puts "cpu_tiny_max=#{cpu_tiny_max}"
puts "frontier_trace_rows=#{ranked.size}"
puts "single_cpu_ms=#{single_cpu_ms.round(4)}"
puts "single_v3d_ms=#{single_v3d_ms.round(4)}"
puts "grouped_v3d_ms=#{grouped_v3d_ms.round(4)}"
puts "grouped_vs_cpu=#{(single_cpu_ms / grouped_v3d_ms).round(3)}x" if grouped_v3d_ms > 0
puts "hybrid_cpu_tiny_ms=#{hybrid_cpu_ms.round(4)}"
puts "hybrid_grouped_v3d_ms=#{hybrid_v3d_ms.round(4)}"
puts "hybrid_total_ms=#{hybrid_ms.round(4)}"
puts "hybrid_vs_cpu=#{(single_cpu_ms / hybrid_ms).round(3)}x" if hybrid_ms > 0
puts

trace.each_with_index do |row, i|
  v3d, cpu = interpolate_pi_route(row.allowed)
  puts "frontier_trace\tidx=#{i + 1}\tstage=#{row.stage}\tlabel=#{row.label}\tallowed=#{row.allowed}\test_v3d_ms=#{v3d.round(4)}\test_cpu_ms=#{cpu.round(4)}\tids_csv=#{row.ids_csv}"
end

puts
groups.each_with_index do |group, i|
  labels = group.map(&.label).join(",")
  ids_groups = group.map(&.ids_csv).join(":")
  cpu_ms = group.sum { |row| interpolate_pi_route(row.allowed)[1] }
  grouped_ms = estimate_group_gpu_ms(group)
  puts "trace_batch_group\tidx=#{i + 1}\tbatch=#{group.size}\tmax_allowed=#{group.max_of(&.allowed)}\tcpu_ms=#{cpu_ms.round(4)}\tgrouped_v3d_ms=#{grouped_ms.round(4)}\tspeedup_vs_cpu=#{(cpu_ms / grouped_ms).round(3)}x\tlabels=#{labels}\tids_groups=#{ids_groups}"
end

puts
cpu_rows.each do |row|
  cpu_ms = interpolate_pi_route(row.allowed)[1]
  puts "hybrid_cpu_row\tstage=#{row.stage}\tlabel=#{row.label}\tallowed=#{row.allowed}\tcpu_ms=#{cpu_ms.round(4)}"
end

hybrid_groups.each_with_index do |group, i|
  labels = group.map(&.label).join(",")
  ids_groups = group.map(&.ids_csv).join(":")
  cpu_ms = group.sum { |row| interpolate_pi_route(row.allowed)[1] }
  grouped_ms = estimate_group_gpu_ms(group)
  puts "hybrid_v3d_group\tidx=#{i + 1}\tbatch=#{group.size}\tmax_allowed=#{group.max_of(&.allowed)}\tcpu_ms=#{cpu_ms.round(4)}\tgrouped_v3d_ms=#{grouped_ms.round(4)}\tspeedup_vs_cpu=#{(cpu_ms / grouped_ms).round(3)}x\tlabels=#{labels}\tids_groups=#{ids_groups}"
end
