require "option_parser"
{% unless flag?(:cpu_only) %}
require "../src/ml/gguf/metal_backend"
{% end %}
require "../src/ml/gguf/nomic_bert"

DEFAULT_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

record TextItem, name : String, text : String
record QueryItem, name : String, text : String, expected_doc : String

DOCS = [
  TextItem.new("crystal_generics", "Crystal generics and macros let developers write reusable statically typed code with Ruby-like syntax while compiling to native LLVM binaries."),
  TextItem.new("metal_kernels", "Apple Metal compute kernels use command buffers, threadgroups, simdgroups, and GPU-resident buffers to accelerate matrix operations."),
  TextItem.new("postgres_hnsw", "PostgreSQL vector search can use HNSW indexes, cosine distance, and compressed embeddings for fast approximate nearest neighbor retrieval."),
  TextItem.new("csv_parser", "A streaming CSV parser handles delimiters, quoted fields, doubled quotes, CRLF endings, incremental chunks, and unterminated quote errors."),
  TextItem.new("rate_limiter", "A token bucket rate limiter tracks capacity, refill rate, monotonic time, and acquisition cost to control request throughput."),
  TextItem.new("moe_encoder", "A mixture-of-experts BERT encoder routes tokens through top experts and mean-pools hidden states into normalized embeddings."),
  TextItem.new("flash_hadamard", "Hadamard rotations, quantization, and packed asymmetric distance computation can speed vector retrieval over compressed sketches."),
  TextItem.new("compiler_debug", "Compiler debug information maps optimized machine code back to source locations, variables, scopes, and stack frames."),
]

QUERIES = [
  QueryItem.new("q_crystal", "How do Crystal macros and generic types help native compiled code?", "crystal_generics"),
  QueryItem.new("q_metal", "GPU command buffers and simdgroup matrix kernels on Apple Metal", "metal_kernels"),
  QueryItem.new("q_postgres", "nearest neighbor search with PostgreSQL vectors and HNSW cosine index", "postgres_hnsw"),
  QueryItem.new("q_csv", "incremental CSV reader for quoted fields and CRLF line endings", "csv_parser"),
  QueryItem.new("q_rate", "token bucket limiter with refill per second and monotonic clock", "rate_limiter"),
  QueryItem.new("q_hadamard", "compressed vector search using Hadamard transform and ADC", "flash_hadamard"),
]

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.each_index do |i|
    av = a[i].to_f64
    bv = b[i].to_f64
    dot += av * bv
    na += av * av
    nb += bv * bv
  end
  return 0.0 if na <= 0.0 || nb <= 0.0
  dot / Math.sqrt(na * nb)
end

def mean_pool_l2(hidden : Array(Float32), seq_len : Int32, dim : Int32) : Array(Float32)
  out = Array(Float32).new(dim, 0.0_f32)
  seq_len.times do |pos|
    off = pos * dim
    dim.times { |j| out[j] += hidden[off + j] }
  end
  inv_len = 1.0_f32 / seq_len.to_f32
  norm = 0.0_f32
  dim.times do |j|
    out[j] *= inv_len
    norm += out[j] * out[j]
  end
  norm = Math.sqrt(norm)
  out.map! { |v| norm > 1.0e-8_f32 ? v / norm : v }
end

def top_doc(query_vec : Array(Float32), docs : Array(Array(Float32)), names : Array(String)) : {String, Float64}
  best_name = ""
  best_score = -Float64::INFINITY
  docs.each_with_index do |doc_vec, i|
    score = cosine(query_vec, doc_vec)
    if score > best_score
      best_score = score
      best_name = names[i]
    end
  end
  {best_name, best_score}
end

model_path = DEFAULT_MODEL
limit_docs = DOCS.size
limit_queries = QUERIES.size
backend_name = "metal"

OptionParser.parse do |p|
  p.banner = "Usage: nomic_encoder_tricks_probe [options]"
  p.on("--model=PATH", "GGUF model path") { |v| model_path = v }
  p.on("--limit-docs=N", "Limit built-in docs") { |v| limit_docs = v.to_i }
  p.on("--limit-queries=N", "Limit built-in queries") { |v| limit_queries = v.to_i }
  p.on("--backend=NAME", "metal | f32 | f16sim (default: metal)") { |v| backend_name = v }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

unless File.exists?(model_path)
  STDERR.puts "model not found: #{model_path}"
  exit 2
end

docs = DOCS.first(limit_docs)
queries = QUERIES.first(limit_queries)
texts = docs.map(&.text) + queries.map(&.text)
names = docs.map(&.name)

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

layer_vectors = Array(Array(Array(Float32))).new(texts.size)
token_counts = Array(Int32).new(texts.size, 0)
depth_ms = Array(Float64).new(model.n_layers, 0.0)

texts.each_with_index do |text, i|
  tokens = model.tokenize(text)
  token_counts[i] = tokens.size
  layer_vectors << (1..model.n_layers).map do |depth|
    t_depth = Time.instant
    vec = model.embed_depth(text, depth)
    depth_ms[depth - 1] += (Time.instant - t_depth).total_milliseconds
    vec
  end.to_a
end

full_depth = model.n_layers
full_docs = layer_vectors[0, docs.size].map { |layers| layers[full_depth - 1] }
full_queries = layer_vectors[docs.size, queries.size].map { |layers| layers[full_depth - 1] }

puts "model=#{model_path}"
puts "backend=#{backend_name}"
puts "load_ms=#{load_ms.round(3)} docs=#{docs.size} queries=#{queries.size} layers=#{model.n_layers} dim=#{model.dim} tokens=#{token_counts.join(",")}"
puts "depth\tms_total\tms_per_text\tfull_cos_mean\tfull_cos_min\tlabel_top1\tfull_top1_agree\tquery_results"

(1..model.n_layers).each do |depth|
  doc_vecs = layer_vectors[0, docs.size].map { |layers| layers[depth - 1] }
  query_vecs = layer_vectors[docs.size, queries.size].map { |layers| layers[depth - 1] }

  cosines = [] of Float64
  label_hits = 0
  full_agree = 0
  result_parts = [] of String

  query_vecs.each_with_index do |qv, qi|
    cosines << cosine(qv, full_queries[qi])
    pred, score = top_doc(qv, doc_vecs, names)
    full_pred = top_doc(full_queries[qi], full_docs, names)[0]
    label_hits += 1 if pred == queries[qi].expected_doc
    full_agree += 1 if pred == full_pred
    result_parts << "#{queries[qi].name}:#{pred}:#{score.round(4)}"
  end

  mean_cos = cosines.sum / cosines.size
  min_cos = cosines.min
  puts "#{depth}\t#{depth_ms[depth - 1].round(3)}\t#{(depth_ms[depth - 1] / texts.size).round(3)}\t#{mean_cos.round(6)}\t#{min_cos.round(6)}\t#{label_hits}/#{queries.size}\t#{full_agree}/#{queries.size}\t#{result_parts.join(",")}"
end
