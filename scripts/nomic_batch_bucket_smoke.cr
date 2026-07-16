require "json"
require "option_parser"
require "../src/ml/gguf/metal_backend"
require "../src/ml/gguf/nomic_bert"

model_path = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

OptionParser.parse do |p|
  p.banner = "Usage: nomic_batch_bucket_smoke [--model PATH]"
  p.on("--model=PATH", "Path to Nomic GGUF model") { |v| model_path = v }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

texts = [
  "Crystal generics macros fibers channels typed native code " * 24,
  "Metal compute graph cache batch sequence padding valid lengths " * 27,
  "PostgreSQL vector search sorted heap code graph memory facts " * 18,
  "short control text",
]

def max_abs_delta(a : Array(Array(Float32)), b : Array(Array(Float32))) : Float64
  raise "row count mismatch" unless a.size == b.size
  max_delta = 0.0_f64
  a.each_with_index do |row, i|
    raise "dimension mismatch at row #{i}" unless row.size == b[i].size
    row.each_with_index do |v, j|
      delta = (v.to_f64 - b[i][j].to_f64).abs
      max_delta = delta if delta > max_delta
    end
  end
  max_delta
end

raise "model not found: #{model_path}" unless File.exists?(model_path)

ML::Metal::Device.init!
backend = ML::GGUF::MetalBackend.new
model = ML::GGUF::NomicBertMoE.from_gguf(model_path, backend)

ENV["NOMIC_BATCH_SEQ_BUCKET_OFF"] = "1"
exact_profile = model.profile_embed_batch(texts)
exact_embeddings = model.embed_batch(texts)
exact_cache_size = backend.compiled_batch_graph_cache_size
exact_cache_hits = backend.compiled_batch_graph_cache_hits
exact_cache_misses = backend.compiled_batch_graph_cache_misses

ENV.delete("NOMIC_BATCH_SEQ_BUCKET_OFF")
bucket_profile = model.profile_embed_batch(texts)
bucket_embeddings = model.embed_batch(texts)
bucket_logical_shape = bucket_profile.max_seq_len
bucket_physical_shape = ML::GGUF::NomicBatchShape.bucket_seq_len(bucket_logical_shape, model.max_seq_len)
first_bucket_cache_size = backend.compiled_batch_graph_cache_size
first_bucket_cache_hits = backend.compiled_batch_graph_cache_hits
first_bucket_cache_misses = backend.compiled_batch_graph_cache_misses

model.embed_batch(texts)
second_bucket_cache_size = backend.compiled_batch_graph_cache_size
second_bucket_cache_hits = backend.compiled_batch_graph_cache_hits
second_bucket_cache_misses = backend.compiled_batch_graph_cache_misses

JSON.build(STDOUT) do |json|
  json.object do
    json.field "exact_logical_shape", exact_profile.max_seq_len
    json.field "bucket_logical_shape", bucket_logical_shape
    json.field "bucket_physical_shape", bucket_physical_shape
    json.field "bucketed", bucket_physical_shape != bucket_logical_shape
    json.field "max_delta", max_abs_delta(exact_embeddings, bucket_embeddings)
    json.field "exact_cache_size", exact_cache_size
    json.field "exact_cache_hits", exact_cache_hits
    json.field "exact_cache_misses", exact_cache_misses
    json.field "first_bucket_cache_size", first_bucket_cache_size
    json.field "first_bucket_cache_hits", first_bucket_cache_hits
    json.field "first_bucket_cache_misses", first_bucket_cache_misses
    json.field "second_bucket_cache_size", second_bucket_cache_size
    json.field "second_bucket_cache_hits", second_bucket_cache_hits
    json.field "second_bucket_cache_misses", second_bucket_cache_misses
  end
end
STDOUT.puts
