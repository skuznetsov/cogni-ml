require "option_parser"
{% unless flag?(:cpu_only) %}
  require "../src/ml/gguf/metal_backend"
{% end %}
require "../src/ml/gguf/nomic_bert"

DEFAULT_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

record InputRow, id : String, text : String

def load_tsv(path : String, id_col : Int32, text_col : Int32) : Array(InputRow)
  rows = [] of InputRow
  File.each_line(path) do |line|
    stripped = line.chomp
    next if stripped.empty? || stripped.starts_with?("#")
    parts = stripped.split('\t')
    max_col = {id_col, text_col}.max
    raise "bad TSV row in #{path}: need column #{max_col}, got #{parts.size}: #{line.inspect}" unless parts.size > max_col
    rows << InputRow.new(parts[id_col], parts[text_col])
  end
  rows
end

def vector_norm(vec : Array(Float32)) : Float64
  sum = 0.0_f64
  vec.each do |v|
    return Float64::NAN unless v.finite?
    f = v.to_f64
    sum += f * f
  end
  Math.sqrt(sum)
end

def vector_literal(vec : Array(Float32)) : String
  String.build do |io|
    io << '['
    vec.each_with_index do |v, i|
      io << ',' if i > 0
      io << v
    end
    io << ']'
  end
end

model_path = DEFAULT_MODEL
input_tsv = nil.as(String?)
out_path = nil.as(String?)
backend_name = "metal"
id_col = 0
text_col = 1
depth = 0
use_batched = true
batch_size_override = nil.as(Int32?)
limit = 0

OptionParser.parse do |p|
  p.banner = "Usage: nomic_embedding_export --input-tsv PATH --out PATH [options]"
  p.on("--model=PATH", "GGUF model path") { |v| model_path = v }
  p.on("--input-tsv=PATH", "Input TSV") { |v| input_tsv = v }
  p.on("--out=PATH", "Output TSV: id<TAB>dim<TAB>norm<TAB>vector_literal") { |v| out_path = v }
  p.on("--id-col=N", "Zero-based id column (default: 0)") { |v| id_col = v.to_i }
  p.on("--text-col=N", "Zero-based text column (default: 1; queries TSV usually 2)") { |v| text_col = v.to_i }
  p.on("--depth=N", "Encoder depth to export; default full depth") { |v| depth = v.to_i }
  p.on("--backend=NAME", "metal | f32 | f16sim (default: metal)") { |v| backend_name = v }
  p.on("--batch-size=N", "Override Metal microbatch size") { |v| batch_size_override = v.to_i }
  p.on("--no-batch", "Embed rows one by one") { use_batched = false }
  p.on("--limit=N", "Limit input rows") { |v| limit = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "--input-tsv is required" unless input = input_tsv
abort "--out is required" unless output_path = out_path
abort "model not found: #{model_path}" unless File.exists?(model_path)
abort "--id-col and --text-col must be non-negative" if id_col < 0 || text_col < 0

rows = load_tsv(input, id_col, text_col)
rows = rows.first(limit) if limit > 0
abort "input produced no rows" if rows.empty?

started = Time.instant
case backend_name
when "metal"
  {% if flag?(:cpu_only) %}
    raise "backend=metal is unavailable in -Dcpu_only builds"
  {% else %}
    ML::Metal::Device.init!
    model = ML::GGUF::NomicBertMoE.from_gguf(model_path, ML::GGUF::MetalBackend.new)
  {% end %}
when "f32"
  model = ML::GGUF::NomicBertMoE.from_gguf(model_path, ML::GGUF::F32Backend.new)
when "f16sim"
  model = ML::GGUF::NomicBertMoE.from_gguf(model_path, ML::GGUF::F16SimBackend.new)
else
  raise "unknown backend: #{backend_name}"
end
load_ms = (Time.instant - started).total_milliseconds
export_depth = depth > 0 ? depth : model.n_layers
abort "--depth must be in 1..#{model.n_layers}, got #{export_depth}" unless 1 <= export_depth <= model.n_layers

texts = rows.map(&.text)
embed_started = Time.instant
vectors = if use_batched
            model.embed_batch_depth(texts, export_depth, batch_size_override)
          else
            texts.map { |text| model.embed_depth(text, export_depth) }
          end
embed_ms = (Time.instant - embed_started).total_milliseconds

invalid = 0
zeroish = 0
File.open(output_path, "w") do |io|
  io.puts "id\tdim\tnorm\tvector"
  rows.each_with_index do |row, i|
    vec = vectors[i]
    norm = vector_norm(vec)
    if norm.nan?
      invalid += 1
    elsif norm < 1.0e-6
      zeroish += 1
    end
    io.puts "#{row.id}\t#{vec.size}\t#{norm}\t#{vector_literal(vec)}"
  end
end

puts "model=#{model_path}"
puts "backend=#{backend_name} rows=#{rows.size} depth=#{export_depth} dim=#{model.dim} batched=#{use_batched} batch_size=#{batch_size_override || "auto"}"
puts "load_ms=#{load_ms.round(3)} embed_ms=#{embed_ms.round(3)} ms_per_text=#{(embed_ms / rows.size).round(3)} invalid=#{invalid} zeroish=#{zeroish} out=#{output_path}"
