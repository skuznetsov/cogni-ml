#!/usr/bin/env crystal

require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_constraints"

def usage
  STDERR.puts "usage: crystal scripts/estimate_qwen35_allowed_frontiers.cr MODEL.gguf"
  exit 2
end

model_path = ARGV[0]? || usage
gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path)
index = ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tokenizer)

PI_ROUTE_CPU_MAX = (ENV["QWEN35_ALLOWED_HEAD_CPU_MAX"]? || "7").to_i
PI_ALLOWED_ROUTE_POINTS = [
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

tools = ML::GGUF::Qwen35Chat.parse_tools_json(%([
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
]))

def interpolate_pi_route(count : Int32) : {Float64, Float64}
  return {0.0, 0.0} if count <= 0

  points = PI_ALLOWED_ROUTE_POINTS
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

def measured_pi_ms(count : Int32) : Float64
  interpolate_pi_route(count)[0]
end

def pi_route(count : Int32) : String
  return "none" if count <= 0
  return "CPU" if count <= PI_ROUTE_CPU_MAX

  v3d, cpu = interpolate_pi_route(count)
  v3d < cpu ? "V3D" : "CPU"
end

def show(label : String, index : ML::GGUF::Qwen35Constraints::TokenTextIndex, literals : Array(String),
         tokenizer : ML::GGUF::Qwen35Tokenizer)
  ids = ML::GGUF::Qwen35Constraints.literal_frontier_ids(index, literals).sort
  samples = ids.first(8).map { |id| "#{id}:#{tokenizer.decode_single(id).inspect}" }.join(",")
  v3d_ms, cpu_ms = interpolate_pi_route(ids.size)
  ids_csv = ids.join(",")
  puts "#{label}\tliterals=#{literals.size}\tallowed=#{ids.size}\troute=#{pi_route(ids.size)}\test_v3d_ms=#{v3d_ms.round(4)}\test_cpu_ms=#{cpu_ms.round(4)}\tids_csv=#{ids_csv}\tsample=#{samples}"
end

tool_names = ML::GGUF::Qwen35Constraints.tool_function_names(tools)
required = ML::GGUF::Qwen35Constraints.tool_required_parameters(tools)
optional = ML::GGUF::Qwen35Constraints.tool_optional_parameters(tools)
finite_values = ML::GGUF::Qwen35Constraints.tool_finite_parameter_value_options(tools)

puts "model=#{model_path}"
puts "vocab=#{tokenizer.vocab.size}"
puts "pi_cost_model=measured indexed Q6 2B tied-head sweep"
puts "policy_cpu_max_allowed=#{PI_ROUTE_CPU_MAX}"
puts

show("tool_call_prefix:start",
  index,
  ML::GGUF::Qwen35Constraints.qwen_tool_call_prefix_options(tool_names),
  tokenizer)

tool_names.each do |name|
  show("function_name:#{name}:start",
    index,
    ["<function=#{name}>\n"],
    tokenizer)
  ["<", "<function=", "<function=#{name[0, [name.size, 4].min]}"].each do |emitted|
    remaining = ML::GGUF::Qwen35Constraints.advance_literal_options(["<function=#{name}>\n"], emitted)
    show("function_name:#{name}:after:#{emitted.inspect}", index, remaining, tokenizer) unless remaining.empty?
  end
end

required.each do |name, params|
  next if params.empty?
  show("required_params:#{name}:open",
    index,
    ML::GGUF::Qwen35Constraints.qwen_parameter_open_options(params),
    tokenizer)
end

optional.each do |name, params|
  next if params.empty?
  show("optional_params:#{name}:continue",
    index,
    ML::GGUF::Qwen35Constraints.qwen_parameter_continue_options(params),
    tokenizer)
end

finite_values.each do |name, by_param|
  by_param.each do |param, values|
    show("finite_values:#{name}.#{param}",
      index,
      ML::GGUF::Qwen35Constraints.qwen_parameter_value_options(values.map(&.strip)),
      tokenizer)
  end
end

show("single_parameter_close",
  index,
  ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options,
  tokenizer)
