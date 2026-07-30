require "json"
require "option_parser"

require "../src/ml/three_d/trellis2"

pack_root = nil.as(String?)

parser = OptionParser.new do |options|
  options.banner = "Usage: trellis2_inspect --pack PATH"
  options.on("--pack=PATH", "Path to a materialized TRELLIS.2 stage pack") do |path|
    pack_root = File.expand_path(path)
  end
  options.on("-h", "--help", "Show help") do
    puts options
    exit
  end
end

begin
  parser.parse
  root = pack_root || raise ArgumentError.new("--pack is required")
  manifest = ML::ThreeD::Trellis2::Manifest.load(root.not_nil!)

  puts JSON.build { |json|
    json.object do
      json.field "validated", true
      json.field "schema_version", manifest.schema_version
      json.field "pack_id", manifest.pack_id
      json.field "source_revision", manifest.source.revision
      json.field "model_revision", manifest.model.revision
      json.field "files", manifest.files.size
      json.field "stages" do
        json.array do
          manifest.stages.sort_by(&.order).each do |stage|
            json.object do
              json.field "id", stage.id
              json.field "order", stage.order
              json.field "tensors", stage.tensors.size
            end
          end
        end
      end
    end
  }
rescue ex : ML::ThreeD::Trellis2::ManifestError | ArgumentError
  STDERR.puts "trellis2_inspect: #{ex.message}"
  exit 1
end
