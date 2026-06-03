require "option_parser"
require "set"
{% unless flag?(:cpu_only) %}
  require "../src/ml/gguf/metal_backend"
{% end %}
require "../src/ml/gguf/nomic_bert"

DEFAULT_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

record TextItem, name : String, text : String
record QueryItem, name : String, text : String, expected_doc : String
record DepthRetrievalMetrics,
  depth : Int32,
  ms_total : Float64,
  ms_per_text : Float64,
  invalid_vecs : Int32,
  zeroish_vecs : Int32,
  shallow_has_full_k : Int32,
  full_rerank_full_k : Int32,
  candidate_union_size : Int32

TOY_DOCS = [
  TextItem.new("crystal_generics", "Crystal generics and macros let developers write reusable statically typed code with Ruby-like syntax while compiling to native LLVM binaries."),
  TextItem.new("metal_kernels", "Apple Metal compute kernels use command buffers, threadgroups, simdgroups, and GPU-resident buffers to accelerate matrix operations."),
  TextItem.new("postgres_hnsw", "PostgreSQL vector search can use HNSW indexes, cosine distance, and compressed embeddings for fast approximate nearest neighbor retrieval."),
  TextItem.new("csv_parser", "A streaming CSV parser handles delimiters, quoted fields, doubled quotes, CRLF endings, incremental chunks, and unterminated quote errors."),
  TextItem.new("rate_limiter", "A token bucket rate limiter tracks capacity, refill rate, monotonic time, and acquisition cost to control request throughput."),
  TextItem.new("moe_encoder", "A mixture-of-experts BERT encoder routes tokens through top experts and mean-pools hidden states into normalized embeddings."),
  TextItem.new("flash_hadamard", "Hadamard rotations, quantization, and packed asymmetric distance computation can speed vector retrieval over compressed sketches."),
  TextItem.new("compiler_debug", "Compiler debug information maps optimized machine code back to source locations, variables, scopes, and stack frames."),
]

TOY_QUERIES = [
  QueryItem.new("q_crystal", "How do Crystal macros and generic types help native compiled code?", "crystal_generics"),
  QueryItem.new("q_metal", "GPU command buffers and simdgroup matrix kernels on Apple Metal", "metal_kernels"),
  QueryItem.new("q_postgres", "nearest neighbor search with PostgreSQL vectors and HNSW cosine index", "postgres_hnsw"),
  QueryItem.new("q_csv", "incremental CSV reader for quoted fields and CRLF line endings", "csv_parser"),
  QueryItem.new("q_rate", "token bucket limiter with refill per second and monotonic clock", "rate_limiter"),
  QueryItem.new("q_hadamard", "compressed vector search using Hadamard transform and ADC", "flash_hadamard"),
]

HARD_DOCS = [
  TextItem.new("crystal_macro_dsl", "Crystal macros expand at compile time, inspect AST nodes, generate overloads, and build internal DSLs while preserving static type checking."),
  TextItem.new("crystal_fibers_io", "Crystal fibers multiplex blocking-looking IO over an event loop with channels, scheduler yields, timeouts, and lightweight concurrency."),
  TextItem.new("crystal_c_bindings", "Crystal C bindings use lib declarations, pointer types, structs, callbacks, and link flags to call native libraries through a stable ABI."),

  TextItem.new("metal_compute_buffers", "Metal compute workloads encode command buffers, bind MTLBuffer objects, choose threadgroup sizes, and dispatch GPU kernels for data-parallel math."),
  TextItem.new("metal_render_pipeline", "Metal rendering configures vertex descriptors, render pass attachments, shaders, depth state, textures, and draw calls for graphics pipelines."),
  TextItem.new("cuda_kernel_occupancy", "CUDA kernel performance depends on blocks, warps, shared memory, occupancy, memory coalescing, streams, and asynchronous copies on NVIDIA GPUs."),

  TextItem.new("pg_hnsw_index", "PostgreSQL HNSW indexes organize vectors into navigable small-world graphs and tune ef_search for approximate nearest-neighbor recall."),
  TextItem.new("pg_ivfpq_index", "IVF-PQ vector search partitions embeddings into coarse lists and compresses residual subvectors with product quantization codes."),
  TextItem.new("pg_full_text", "PostgreSQL full-text search tokenizes documents into lexemes, stores tsvectors, ranks tsquery matches, and supports stemming dictionaries."),

  TextItem.new("csv_stream_parser", "A streaming CSV parser tracks quote state across chunks, handles doubled quotes, delimiters, CRLF boundaries, and malformed unterminated rows."),
  TextItem.new("json_pointer_patch", "JSON Pointer and JSON Patch address nested values by escaped path segments and apply add, remove, replace, move, copy, and test operations."),
  TextItem.new("ini_config_parser", "An INI parser reads sections, keys, comments, duplicate entries, quoted values, and interpolation rules from line-oriented config files."),

  TextItem.new("token_bucket", "A token bucket limiter accumulates tokens at a refill rate up to capacity and spends tokens immediately for burst-tolerant request control."),
  TextItem.new("leaky_bucket", "A leaky bucket limiter drains queued work at a fixed rate to smooth bursts and bound output throughput over time."),
  TextItem.new("retry_backoff", "Retry backoff policies use exponential delay, jitter, maximum attempts, and retryable error classification to avoid thundering herds."),

  TextItem.new("hadamard_adc", "Hadamard-rotated sketches with packed four-bit codes can support fast asymmetric distance computation for vector retrieval."),
  TextItem.new("pca_lowrank", "PCA low-rank compression projects vectors onto principal components, stores compact coefficients, and reconstructs approximate embeddings."),
  TextItem.new("pq_codebooks", "Product quantization splits vectors into subspaces, assigns codebook centroids, and evaluates approximate distances through lookup tables."),

  TextItem.new("debug_dwarf_locations", "DWARF debug info records source lines, lexical scopes, variable location lists, inlined calls, and stack unwinding metadata."),
  TextItem.new("compiler_optimizer_passes", "Compiler optimization passes transform IR with inlining, constant folding, loop unrolling, vectorization, and dead-code elimination."),
  TextItem.new("runtime_stack_traces", "Runtime stack traces capture call frames, function names, instruction pointers, exception context, and symbolized source locations."),

  TextItem.new("moe_bert_router", "A MoE BERT encoder routes token hidden states to top experts, combines expert outputs, and pools final hidden vectors for embeddings."),
  TextItem.new("dense_bert_encoder", "A dense BERT encoder applies the same transformer feed-forward layers to every token before pooling sentence embeddings."),
  TextItem.new("decoder_llm_generate", "A decoder-only language model autoregressively predicts next tokens from causal attention and maintains KV cache during generation."),
]

HARD_QUERIES = [
  QueryItem.new("q_macro_ast", "compile-time AST expansion for typed Crystal DSL overload generation", "crystal_macro_dsl"),
  QueryItem.new("q_fiber_channels", "lightweight Crystal concurrency with channels and scheduler-driven IO timeouts", "crystal_fibers_io"),
  QueryItem.new("q_ffi_callbacks", "Crystal lib declarations for native ABI pointers structs and callbacks", "crystal_c_bindings"),

  QueryItem.new("q_metal_compute", "MTLBuffer binding and threadgroup dispatch for Metal compute math kernels", "metal_compute_buffers"),
  QueryItem.new("q_metal_render", "render pass attachments vertex descriptors textures and draw calls in Metal", "metal_render_pipeline"),
  QueryItem.new("q_cuda_occupancy", "NVIDIA warps occupancy coalesced memory streams and shared memory copies", "cuda_kernel_occupancy"),

  QueryItem.new("q_hnsw_ef", "ef_search tuning in navigable small-world graph vector index for Postgres", "pg_hnsw_index"),
  QueryItem.new("q_ivfpq_residual", "coarse inverted lists and product quantized residual subvectors", "pg_ivfpq_index"),
  QueryItem.new("q_tsquery", "lexemes stemming dictionaries tsvector and tsquery document ranking", "pg_full_text"),

  QueryItem.new("q_csv_chunks", "quoted CSV field continues across chunks with CRLF and doubled quotes", "csv_stream_parser"),
  QueryItem.new("q_json_patch", "escaped JSON Pointer path segments used by replace move copy and test", "json_pointer_patch"),
  QueryItem.new("q_ini_sections", "line config parser for sections comments duplicate keys and interpolation", "ini_config_parser"),

  QueryItem.new("q_token_bucket", "burst tolerant limiter spends accumulated refill tokens up to capacity", "token_bucket"),
  QueryItem.new("q_leaky_bucket", "queue drains at fixed rate to smooth bursty request output", "leaky_bucket"),
  QueryItem.new("q_retry_jitter", "exponential retry delay with jitter maximum attempts and retryable errors", "retry_backoff"),

  QueryItem.new("q_hadamard_adc", "packed four-bit Hadamard sketches for asymmetric distance retrieval", "hadamard_adc"),
  QueryItem.new("q_pca_coeffs", "principal component coefficients reconstruct approximate embedding vectors", "pca_lowrank"),
  QueryItem.new("q_pq_lut", "subspace codebook centroids and lookup-table approximate vector distances", "pq_codebooks"),

  QueryItem.new("q_dwarf_locations", "DWARF variable location lists lexical scopes inlined calls and unwinding", "debug_dwarf_locations"),
  QueryItem.new("q_opt_passes", "IR inlining constant folding loop unrolling vectorization dead-code elimination", "compiler_optimizer_passes"),
  QueryItem.new("q_stack_symbols", "exception call frames instruction pointers and symbolized stack source locations", "runtime_stack_traces"),

  QueryItem.new("q_moe_experts", "top expert routing in BERT encoder before pooled embedding output", "moe_bert_router"),
  QueryItem.new("q_dense_bert", "same transformer feed-forward layers for every token in sentence encoder", "dense_bert_encoder"),
  QueryItem.new("q_decoder_kv", "causal autoregressive next-token model maintains KV cache while generating", "decoder_llm_generate"),
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
  score = dot / Math.sqrt(na * nb)
  score.nan? ? 0.0 : score
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

def ranked_docs(query_vec : Array(Float32), docs : Array(Array(Float32)), names : Array(String)) : Array(Tuple(String, Float64))
  rows = [] of Tuple(String, Float64)
  docs.each_with_index do |doc_vec, i|
    rows << {names[i], cosine(query_vec, doc_vec)}
  end
  rows.sort_by! { |pair| -pair[1] }
  rows
end

def vector_norm(a : Array(Float32)) : Float64
  sum = 0.0_f64
  a.each do |v|
    return Float64::NAN unless v.finite?
    vf = v.to_f64
    sum += vf * vf
  end
  Math.sqrt(sum)
end

def ranked_docs_subset(query_vec : Array(Float32), docs : Array(Array(Float32)), names : Array(String), candidate_names : Array(String)) : Array(Tuple(String, Float64))
  candidates = Set(String).new(candidate_names)
  rows = [] of Tuple(String, Float64)
  docs.each_with_index do |doc_vec, i|
    name = names[i]
    next unless candidates.includes?(name)
    rows << {name, cosine(query_vec, doc_vec)}
  end
  rows.sort_by! { |pair| -pair[1] }
  rows
end

def load_docs_tsv(path : String) : Array(TextItem)
  rows = [] of TextItem
  File.each_line(path) do |line|
    stripped = line.strip
    next if stripped.empty? || stripped.starts_with?("#")
    parts = stripped.split('	', 2)
    raise "bad docs TSV row in #{path}: expected id<TAB>text, got #{line.inspect}" unless parts.size == 2
    rows << TextItem.new(parts[0], parts[1])
  end
  rows
end

def load_queries_tsv(path : String) : Array(QueryItem)
  rows = [] of QueryItem
  File.each_line(path) do |line|
    stripped = line.strip
    next if stripped.empty? || stripped.starts_with?("#")
    parts = stripped.split('	', 3)
    raise "bad queries TSV row in #{path}: expected id<TAB>expected_doc<TAB>text, got #{line.inspect}" unless parts.size == 3
    rows << QueryItem.new(parts[0], parts[2], parts[1])
  end
  rows
end

def parse_depths(raw : String, max_depth : Int32) : Array(Int32)
  depths = [] of Int32
  raw.split(",").each do |part|
    value = part.strip
    next if value.empty?
    depth = value.to_i
    raise "depth must be in 1..#{max_depth}, got #{depth}" unless 1 <= depth <= max_depth
    depths << depth
  end
  raise "--depths produced no depths" if depths.empty?
  depths.uniq.sort
end

model_path = DEFAULT_MODEL
limit_docs = 0
limit_queries = 0
backend_name = "metal"
suite = "toy"
show_results = false
summary_only = false
use_batched = false
batch_size_override = nil.as(Int32?)
docs_tsv = nil.as(String?)
queries_tsv = nil.as(String?)
rerank_k = 3
depths_arg = nil.as(String?)
show_economics = false

OptionParser.parse do |p|
  p.banner = "Usage: nomic_encoder_tricks_probe [options]"
  p.on("--model=PATH", "GGUF model path") { |v| model_path = v }
  p.on("--suite=NAME", "toy | hard | combined (default: toy)") { |v| suite = v }
  p.on("--docs-tsv=PATH", "External docs TSV: id<TAB>text") { |v| docs_tsv = v }
  p.on("--queries-tsv=PATH", "External queries TSV: id<TAB>expected_doc<TAB>text") { |v| queries_tsv = v }
  p.on("--rerank-k=N", "Candidate count for shallow->full rerank metrics (default: 3)") { |v| rerank_k = v.to_i }
  p.on("--depths=LIST", "Comma-separated depths to evaluate; full depth is always computed for rerank") { |v| depths_arg = v }
  p.on("--limit-docs=N", "Limit built-in docs") { |v| limit_docs = v.to_i }
  p.on("--limit-queries=N", "Limit built-in queries") { |v| limit_queries = v.to_i }
  p.on("--backend=NAME", "metal | f32 | f16sim (default: metal)") { |v| backend_name = v }
  p.on("--batched", "Use embed_batch_depth for each requested depth") { use_batched = true }
  p.on("--batch-size=N", "Override embed_batch_depth microbatch size for --batched probes") { |v| batch_size_override = v.to_i }
  p.on("--economics", "Print rough two-stage retrieval economics after quality metrics") { show_economics = true }
  p.on("--summary-only", "Suppress per-query mismatch details") { summary_only = true }
  p.on("--show-results", "Print per-query top1/top3 details instead of compact mismatches") { show_results = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

unless File.exists?(model_path)
  STDERR.puts "model not found: #{model_path}"
  exit 2
end

docs_all, queries_all = if docs_tsv || queries_tsv
                          raise "--docs-tsv and --queries-tsv must be provided together" unless docs_tsv && queries_tsv
                          {load_docs_tsv(docs_tsv.not_nil!), load_queries_tsv(queries_tsv.not_nil!)}
                        else
                          case suite
                          when "toy"
                            {TOY_DOCS, TOY_QUERIES}
                          when "hard"
                            {HARD_DOCS, HARD_QUERIES}
                          when "combined"
                            {TOY_DOCS + HARD_DOCS, TOY_QUERIES + HARD_QUERIES}
                          else
                            raise "unknown suite: #{suite}"
                          end
                        end
docs = limit_docs > 0 ? docs_all.first(limit_docs) : docs_all
queries = limit_queries > 0 ? queries_all.first(limit_queries) : queries_all
raise "docs set is empty" if docs.empty?
raise "queries set is empty" if queries.empty?
name_set = Set(String).new(docs.map(&.name))
queries.each do |query|
  raise "query #{query.name.inspect} expects missing doc #{query.expected_doc.inspect}" unless name_set.includes?(query.expected_doc)
end
rerank_k = rerank_k.clamp(1, docs.size)
texts = docs.map(&.text) + queries.map(&.text)
names = docs.map(&.name)
text_names = docs.map(&.name) + queries.map(&.name)

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

full_depth = model.n_layers
eval_depths = depths_arg ? parse_depths(depths_arg.not_nil!, model.n_layers) : (1..model.n_layers).to_a
compute_depths = (eval_depths + [full_depth]).uniq.sort
depth_index = {} of Int32 => Int32
compute_depths.each_with_index { |depth, i| depth_index[depth] = i }

layer_vectors = Array(Array(Array(Float32))).new(texts.size)
token_counts = Array(Int32).new(texts.size, 0)
depth_ms = Array(Float64).new(model.n_layers, 0.0)

texts.each_with_index do |text, i|
  tokens = model.tokenize(text)
  token_counts[i] = tokens.size
end

if use_batched
  depth_results = {} of Int32 => Array(Array(Float32))
  compute_depths.each do |depth|
    t_depth = Time.instant
    vecs = model.embed_batch_depth(texts, depth, batch_size_override)
    depth_ms[depth - 1] += (Time.instant - t_depth).total_milliseconds
    depth_results[depth] = vecs
  end
  texts.each_index do |i|
    layer_vectors << compute_depths.map { |depth| depth_results[depth][i] }.to_a
  end
else
  texts.each_with_index do |text, _i|
    layer_vectors << compute_depths.map do |depth|
      t_depth = Time.instant
      vec = model.embed_depth(text, depth)
      depth_ms[depth - 1] += (Time.instant - t_depth).total_milliseconds
      vec
    end.to_a
  end
end

full_depth_i = depth_index[full_depth]
full_docs = layer_vectors[0, docs.size].map { |layers| layers[full_depth_i] }
full_queries = layer_vectors[docs.size, queries.size].map { |layers| layers[full_depth_i] }

puts "model=#{model_path}"
puts "backend=#{backend_name}"
puts "suite=#{suite}"
puts "docs_tsv=#{docs_tsv || ""} queries_tsv=#{queries_tsv || ""}"
puts "load_ms=#{load_ms.round(3)} docs=#{docs.size} queries=#{queries.size} layers=#{model.n_layers} eval_depths=#{eval_depths.join(",")} computed_depths=#{compute_depths.join(",")} dim=#{model.dim} rerank_k=#{rerank_k} batched=#{use_batched} batch_size=#{batch_size_override || "auto"} tokens=#{token_counts.join(",")}"
puts "depth\tms_total\tms_per_text\tfull_cos_mean\tfull_cos_min\tinvalid_vecs\tzeroish_vecs\tlabel_top1\tlabel_top3\tfull_top1_agree\tfull_top3_contains_depth_top1\tshallow_has_label@k\tshallow_has_full@k\tfull_rerank_label@k\tfull_rerank_full@k\tcandidate_union@k\tcandidate_doc_pct\tlazy_full_doc_savings_pct\tdetails"

retrieval_metrics = [] of DepthRetrievalMetrics

eval_depths.each do |depth|
  current_depth_i = depth_index[depth]
  doc_vecs = layer_vectors[0, docs.size].map { |layers| layers[current_depth_i] }
  query_vecs = layer_vectors[docs.size, queries.size].map { |layers| layers[current_depth_i] }
  all_vecs = doc_vecs + query_vecs
  invalid_vecs = 0
  zeroish_vecs = 0
  health_parts = [] of String
  all_vecs.each_with_index do |vec, vi|
    norm = vector_norm(vec)
    if norm.nan?
      invalid_vecs += 1
      health_parts << "invalid=#{text_names[vi]}" if health_parts.size < 8
    elsif norm < 1.0e-6
      zeroish_vecs += 1
      health_parts << "zeroish=#{text_names[vi]}" if health_parts.size < 8
    end
  end

  cosines = [] of Float64
  label_hits = 0
  label_top3_hits = 0
  full_agree = 0
  full_top3_contains_depth_top1 = 0
  shallow_has_label_k = 0
  shallow_has_full_k = 0
  full_rerank_label_k = 0
  full_rerank_full_k = 0
  result_parts = [] of String
  mismatch_parts = [] of String
  candidate_union = Set(String).new

  query_vecs.each_with_index do |qv, qi|
    cosines << cosine(qv, full_queries[qi])
    rank = ranked_docs(qv, doc_vecs, names)
    full_rank = ranked_docs(full_queries[qi], full_docs, names)
    pred = rank[0][0]
    score = rank[0][1]
    full_pred = full_rank[0][0]
    top3 = rank.first(3).map { |pair| pair[0] }
    topk = rank.first(rerank_k).map { |pair| pair[0] }
    topk.each { |name| candidate_union.add(name) }
    full_top3 = full_rank.first(3).map { |pair| pair[0] }
    rerank = ranked_docs_subset(full_queries[qi], full_docs, names, topk)
    rerank_pred = rerank.empty? ? "" : rerank[0][0]
    label_hits += 1 if pred == queries[qi].expected_doc
    label_top3_hits += 1 if top3.includes?(queries[qi].expected_doc)
    full_agree += 1 if pred == full_pred
    full_top3_contains_depth_top1 += 1 if full_top3.includes?(pred)
    shallow_has_label_k += 1 if topk.includes?(queries[qi].expected_doc)
    shallow_has_full_k += 1 if topk.includes?(full_pred)
    full_rerank_label_k += 1 if rerank_pred == queries[qi].expected_doc
    full_rerank_full_k += 1 if rerank_pred == full_pred
    result_parts << "#{queries[qi].name}:#{pred}:#{score.round(4)}:top3=#{top3.join("|")}:rerank=#{rerank_pred}"
    if pred != queries[qi].expected_doc || pred != full_pred || rerank_pred != full_pred
      mismatch_parts << "#{queries[qi].name}:pred=#{pred}:expected=#{queries[qi].expected_doc}:full=#{full_pred}:rerank=#{rerank_pred}:top3=#{top3.join("|")}:topk=#{topk.join("|")}"
    end
  end

  mean_cos = cosines.sum / cosines.size
  min_cos = cosines.min
  candidate_doc_pct = 100.0 * candidate_union.size / docs.size
  lazy_full_doc_savings_pct = 100.0 * (docs.size - candidate_union.size) / docs.size
  ms_total = depth_ms[depth - 1]
  ms_per_text = ms_total / texts.size
  details = if show_results
              result_parts.join(",")
            elsif summary_only
              parts = [] of String
              parts << "mismatches=#{mismatch_parts.size}" unless mismatch_parts.empty?
              parts.concat(health_parts)
              parts.empty? ? "ok" : parts.join(",")
            else
              parts = mismatch_parts.dup
              parts.concat(health_parts)
              parts.empty? ? "ok" : parts.join(",")
            end
  puts "#{depth}\t#{ms_total.round(3)}\t#{ms_per_text.round(3)}\t#{mean_cos.round(6)}\t#{min_cos.round(6)}\t#{invalid_vecs}\t#{zeroish_vecs}\t#{label_hits}/#{queries.size}\t#{label_top3_hits}/#{queries.size}\t#{full_agree}/#{queries.size}\t#{full_top3_contains_depth_top1}/#{queries.size}\t#{shallow_has_label_k}/#{queries.size}\t#{shallow_has_full_k}/#{queries.size}\t#{full_rerank_label_k}/#{queries.size}\t#{full_rerank_full_k}/#{queries.size}\t#{candidate_union.size}/#{docs.size}\t#{candidate_doc_pct.round(2)}\t#{lazy_full_doc_savings_pct.round(2)}\t#{details}"
  retrieval_metrics << DepthRetrievalMetrics.new(
    depth: depth,
    ms_total: ms_total,
    ms_per_text: ms_per_text,
    invalid_vecs: invalid_vecs,
    zeroish_vecs: zeroish_vecs,
    shallow_has_full_k: shallow_has_full_k,
    full_rerank_full_k: full_rerank_full_k,
    candidate_union_size: candidate_union.size,
  )
end

if show_economics
  full_total_ms = depth_ms[full_depth - 1]
  full_ms_per_text = full_total_ms / texts.size
  full_query_ms = full_ms_per_text * queries.size
  puts
  puts "economics_note=rough_embedding_only_estimate full_depth=#{full_depth} full_ms_per_text=#{full_ms_per_text.round(3)} full_baseline_ms=#{full_total_ms.round(3)}"
  puts "economics_depth\tk\tquality_gate\tcandidate_docs\tcandidate_doc_pct\tcold_lazy_ms\tcold_lazy_speedup\thot_query_extra_embed_ms\treduced_doc_scan_pct\tverdict"

  retrieval_metrics.each do |m|
    next if m.depth == full_depth
    candidate_doc_pct = 100.0 * m.candidate_union_size / docs.size
    reduced_doc_scan_pct = 100.0 * (docs.size - m.candidate_union_size) / docs.size
    lazy_full_docs_ms = full_ms_per_text * m.candidate_union_size
    cold_lazy_ms = m.ms_total + full_query_ms + lazy_full_docs_ms
    cold_lazy_speedup = full_total_ms / cold_lazy_ms
    hot_query_extra_embed_ms = m.ms_per_text * queries.size

    quality_ok = m.invalid_vecs == 0 &&
                 m.zeroish_vecs == 0 &&
                 m.shallow_has_full_k == queries.size &&
                 m.full_rerank_full_k == queries.size
    verdict =
      if !quality_ok
        "quality_fail"
      elsif cold_lazy_speedup > 1.05
        "cold_lazy_candidate"
      elsif reduced_doc_scan_pct >= 10.0
        "hot_index_gate_only"
      else
        "weak_economics"
      end

    quality_gate = "#{m.full_rerank_full_k}/#{queries.size}"
    puts "#{m.depth}\t#{rerank_k}\t#{quality_gate}\t#{m.candidate_union_size}/#{docs.size}\t#{candidate_doc_pct.round(2)}\t#{cold_lazy_ms.round(3)}\t#{cold_lazy_speedup.round(3)}\t#{hot_query_extra_embed_ms.round(3)}\t#{reduced_doc_scan_pct.round(2)}\t#{verdict}"
  end
end
