require "json"
require "option_parser"

require "../src/ml/three_d/trellis2"

source_root = nil.as(String?)
output_root = nil.as(String?)
plan_path = "conversion.json"

parser = OptionParser.new do |options|
  options.banner =
    "Usage: trellis2_convert --source PATH --output PATH [--plan RELATIVE_PATH]"
  options.on("--source=PATH", "Immutable source directory") do |path|
    source_root = File.expand_path(path)
  end
  options.on("--output=PATH", "New output pack directory") do |path|
    output_root = File.expand_path(path)
  end
  options.on(
    "--plan=RELATIVE_PATH",
    "Strict conversion plan relative to the source directory"
  ) do |path|
    plan_path = path
  end
  options.on("-h", "--help", "Show help") do
    puts options
    exit
  end
end

begin
  parser.parse
  raise ArgumentError.new("--source is required") unless source_root
  raise ArgumentError.new("--output is required") unless output_root
  source = source_root.not_nil!
  output = output_root.not_nil!
  manifest = ML::ThreeD::Trellis2::Converter.convert!(
    source,
    output,
    plan_path
  )

  puts JSON.build { |json|
    json.object do
      json.field "converted", true
      json.field "schema_version", manifest.schema_version
      json.field "pack_id", manifest.pack_id
      json.field "output", output
      json.field "stages", manifest.stages.size
      json.field(
        "tensors",
        manifest.stages.sum { |stage| stage.tensors.size }
      )
      json.field "runtime_admitted", false
    end
  }
rescue ex : ML::ThreeD::Trellis2::ConverterError | ArgumentError
  STDERR.puts "trellis2_convert: #{ex.message}"
  exit 1
end
