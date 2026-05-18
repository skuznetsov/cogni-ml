# CUDA mixed recurrent/full-attention stack scaffold probe for Qwen GGUF weights.
#
# This is a correctness scaffold, not an end-to-end decoder: it composes
# recurrent-layer and full-attention-layer CUDA runners in model layer order
# with device-resident hidden handoff between layers.

require "json"
require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/ngram_draft"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/cuda/qwen_recurrent_layer_runner"
require "../src/ml/cuda/qwen_full_attn_layer_runner"
require "../src/ml/cuda/qwen_output_head_runner"
require "../src/ml/cuda/qwen_mixed_stack_runner"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def parse_layers(value : String) : Array(Int32)
  value.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
end

def parse_i32_list(value : String) : Array(Int32)
  value.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
end

def parse_f32_list(value : String) : Array(Float32)
  value.split(",").map(&.strip).reject(&.empty?).map(&.to_f32)
end

alias FfnPcaUpdownAdapter = NamedTuple(
  rank: Int32,
  x_mean: Array(Float32),
  c_mean: Array(Float32),
  coeff_w: Array(Float32),
  down: Array(Float32))

def json_f32_array(value : JSON::Any, label : String) : Array(Float32)
  value.as_a.map { |item| item.as_f.to_f32 }
rescue ex
  raise ArgumentError.new("invalid #{label}: #{ex.message}")
end

def load_ffn_pca_updown_adapters(path : String, hidden : Int32) : Hash(Int32, FfnPcaUpdownAdapter)
  root = JSON.parse(File.read(path))
  format = root["format"]?.try(&.as_s?) || ""
  raise "unsupported adapter format: #{format}" unless format == "qwen35_ffn_updown_adapter_v1"
  file_hidden = root["hidden_dim"]?.try(&.as_i?) || hidden
  raise "adapter hidden #{file_hidden} does not match model hidden #{hidden}" unless file_hidden == hidden

  adapters = {} of Int32 => FfnPcaUpdownAdapter
  root["layers"].as_a.each do |entry|
    layer_id = entry["layer"].as_i
    rank = entry["rank"].as_i
    raise "adapter layer #{layer_id} rank must be in 1..64" unless rank > 0 && rank <= 64
    x_mean = json_f32_array(entry["x_mean"], "layer #{layer_id} x_mean")
    c_mean = json_f32_array(entry["c_mean"], "layer #{layer_id} c_mean")
    coeff_w = json_f32_array(entry["coeff_w"], "layer #{layer_id} coeff_w")
    down = json_f32_array(entry["down"], "layer #{layer_id} down")
    raise "adapter layer #{layer_id} x_mean size mismatch" unless x_mean.size == hidden
    raise "adapter layer #{layer_id} c_mean size mismatch" unless c_mean.size >= rank
    raise "adapter layer #{layer_id} coeff_w size mismatch" unless coeff_w.size >= rank * hidden
    raise "adapter layer #{layer_id} down size mismatch" unless down.size >= rank * hidden
    adapters[layer_id] = {
      rank:    rank,
      x_mean:  x_mean,
      c_mean:  c_mean,
      coeff_w: coeff_w,
      down:    down,
    }
  end
  adapters
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise ArgumentError.new("size mismatch") unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |v, i|
    d = (v - b[i]).abs
    max = d if d > max
  end
  max
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.each_with_index do |v, i|
    av = v.to_f64
    bv = b[i].to_f64
    dot += av * bv
    na += av * av
    nb += bv * bv
  end
  dot / Math.sqrt(na * nb)
end

def report_pair(name : String, gpu : Array(Float32), cpu : Array(Float32), lines : Array(String), max_allowed : Float32) : Bool
  cos = cosine(gpu, cpu)
  max_diff = max_abs_diff(gpu, cpu)
  ok = cos >= 0.99999 && max_diff <= max_allowed
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

def top2_from_logits(logits : Array(Float32), tok : Int32, vocab : Int32)
  offset = tok * vocab
  best_id = 0
  second_id = 0
  best = -Float32::INFINITY
  second = -Float32::INFINITY

  vocab.times do |i|
    v = logits[offset + i]
    if v > best
      second = best
      second_id = best_id
      best = v
      best_id = i
    elsif v > second
      second = v
      second_id = i
    end
  end

  {best_id, best, second_id, second, best - second}
end

def append_margin_bucket_report(lines : Array(String),
                                exact_top1_ids : Array(Int32),
                                proposal_top1_ids : Array(Int32),
                                exact_margins : Array(Float32),
                                edges : Array(Float32)) : Nil
  raise ArgumentError.new("top1 size mismatch") unless exact_top1_ids.size == proposal_top1_ids.size
  raise ArgumentError.new("margin size mismatch") unless exact_top1_ids.size == exact_margins.size

  sorted_edges = edges.sort
  bucket_count = sorted_edges.size + 1
  counts = Array(Int32).new(bucket_count, 0)
  accepts = Array(Int32).new(bucket_count, 0)

  exact_top1_ids.each_with_index do |target_id, i|
    margin = exact_margins[i]
    bucket = sorted_edges.index { |edge| margin < edge } || sorted_edges.size
    counts[bucket] += 1
    accepts[bucket] += 1 if proposal_top1_ids[i] == target_id
  end

  total = counts.sum
  total_accepts = accepts.sum
  total_rejects = total - total_accepts
  accept_rate = total > 0 ? (100.0 * total_accepts / total) : 0.0
  lines << "proposal_accept_tokens=#{total_accepts}"
  lines << "proposal_reject_tokens=#{total_rejects}"
  lines << "proposal_accept_rate=#{accept_rate.round(2)}"
  lines << "margin_bucket_edges=#{sorted_edges.join(",")}"

  bucket_count.times do |bucket|
    lower = bucket == 0 ? nil : sorted_edges[bucket - 1]?
    upper = sorted_edges[bucket]?
    label = if lower.nil?
              "lt_#{upper}"
            elsif upper.nil?
              "ge_#{lower}"
            else
              "#{lower}_to_#{upper}"
            end
    count = counts[bucket]
    accepted = accepts[bucket]
    rejected = count - accepted
    bucket_rate = count > 0 ? (100.0 * accepted / count) : 0.0
    lines << "margin_bucket_#{bucket}_#{label} count=#{count} accept=#{accepted} reject=#{rejected} accept_rate=#{bucket_rate.round(2)}"
  end
end

def rope_tables(tokens : Int32, start_pos : Int32, rope_dim : Int32, freq_base : Float32) : {Array(Float32), Array(Float32)}
  half = rope_dim // 2
  cos_table = Array(Float32).new(tokens * half, 0.0_f32)
  sin_table = Array(Float32).new(tokens * half, 0.0_f32)
  tokens.times do |tok|
    pos = start_pos + tok
    half.times do |i|
      freq = 1.0_f32 / (freq_base ** (2.0_f32 * i / rope_dim))
      theta = pos.to_f32 * freq
      cos_table[tok * half + i] = Math.cos(theta).to_f32
      sin_table[tok * half + i] = Math.sin(theta).to_f32
    end
  end
  {cos_table, sin_table}
end

def load_quant_weight(gguf : ML::GGUF::GGUFFile, name : String) : ML::GGUF::QuantWeight
  info = gguf.tensor(name) || raise "missing #{name}"
  raw = gguf.read_tensor_raw(info)
  in_dim = info.dims[0].to_i32
  out_dim = info.dims.size >= 2 ? info.dims[1].to_i32 : 1
  ML::GGUF::QuantWeight.new(raw, info.type, out_dim, in_dim)
end

class CudaQ4KTokenEmbedder
  EMBED_Q4K_PTX = {{ read_file("src/ml/cuda/kernels/embed_q4k_probe.ptx") }}

  def initialize(token_embd : ML::GGUF::QuantWeight,
                 @token_ids_device_ptr : ML::CUDA::DevicePtr,
                 @output_device_ptr : ML::CUDA::DevicePtr,
                 @history_tokens : Int32)
    raise "GPU token embedding currently requires Q4_K token embeddings" unless token_embd.type.q4_k?
    raise "embedding dim #{token_embd.in_dim} must be divisible by 256" unless token_embd.in_dim % 256 == 0
    raise "history tokens must be positive" unless @history_tokens > 0

    @hidden = token_embd.in_dim
    @vocab = token_embd.out_dim
    @module = ML::CUDA::CUDAModule.load(EMBED_Q4K_PTX, "embed_q4k")
    @fn = @module.function("embed_q4k_f32_from_token_id_cuda")
    @weight = ML::CUDA::DeviceBuffer.new(token_embd.raw.size.to_u64)
    @history = ML::CUDA::DeviceBuffer.new(bytesize_i32(@history_tokens))
    @history_gpu = Array(Int32).new(@history_tokens, 0)
    @token_index = Pointer(UInt32).malloc(1)
    @param_keepalive = [] of Void*
    @params = Pointer(Void*).malloc(7)
    @params[0] = box_ptr(@weight.ptr).as(Void*)
    @params[1] = box_ptr(@token_ids_device_ptr).as(Void*)
    @params[2] = box_ptr(@output_device_ptr).as(Void*)
    @params[3] = box_ptr(@history.ptr).as(Void*)
    @params[4] = @token_index.as(Void*)
    @params[5] = box_u32(@hidden.to_u32).as(Void*)
    @params[6] = box_u32(@vocab.to_u32).as(Void*)
    @closed = false

    ML::CUDA.copy_htod!(@weight.ptr, token_embd.raw.to_unsafe.as(Void*), token_embd.raw.size.to_u64, "token_embd_q4k")
  end

  def record_and_embed(token_index : Int32) : Nil
    raise "token index out of range" if token_index < 0 || token_index >= @history_tokens

    @token_index.value = token_index.to_u32
    grid = ((@hidden + 255) // 256).to_u32
    ML::CUDA.launch!(@fn, grid, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, @params, "embed q4k token id")
  end

  def read_history : Array(Int32)
    ML::CUDA.copy_dtoh!(@history_gpu.to_unsafe.as(Void*), @history.ptr, bytesize_i32(@history_tokens), "gpu_embedding_history")
    @history_gpu
  end

  def close : Nil
    return if @closed

    @weight.close
    @history.close
    @module.close
    @closed = true
  end

  private def box_ptr(value : ML::CUDA::DevicePtr) : Pointer(ML::CUDA::DevicePtr)
    ptr = Pointer(ML::CUDA::DevicePtr).malloc(1)
    ptr.value = value
    @param_keepalive << ptr.as(Void*)
    ptr
  end

  private def box_u32(value : UInt32) : Pointer(UInt32)
    ptr = Pointer(UInt32).malloc(1)
    ptr.value = value
    @param_keepalive << ptr.as(Void*)
    ptr
  end

  private def bytesize_i32(elements : Int32) : LibC::SizeT
    (elements * sizeof(Int32)).to_u64
  end
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layers = [0, 1, 2, 3, 4]
seed = 41_u64
tokens = 2
start_pos = 0
max_seq = 16
warmup = 0
steady_reps = 0
steady_graph_reps = 0
read_logits = false
read_top2 = false
gpu_logits_only = false
margin_bucket_report = false
margin_bucket_edges = [0.05_f32, 0.1_f32, 0.25_f32, 0.5_f32, 1.0_f32, 2.0_f32, 5.0_f32]
profile_phases = false
debug_readback = true
perf_only = false
skip_output_head = false
all_layers = false
greedy_loop_tokens = 0
greedy_loop_graph = false
greedy_loop_no_graph = false
greedy_loop_graph_device_ready = false
greedy_loop_gpu_embedding = false
greedy_loop_cpu_embedding = false
greedy_loop_read_logits = false
greedy_loop_probe_restore = false
greedy_loop_probe_restore_kv = false
greedy_loop_probe_pca_updown = false
greedy_loop_probe_pca_updown_raw_q8_rest = false
greedy_loop_probe_chunk_gamma = 0
greedy_loop_probe_chunk_margin = 0.03_f32
greedy_loop_probe_chunk_batched_verify = false
greedy_loop_probe_chunk_fast_verify_top1 = false
greedy_loop_probe_ngram = false
greedy_loop_probe_ngram_min = 2
greedy_loop_probe_ngram_max = 8
greedy_loop_probe_ngram_recursive = true
greedy_loop_probe_ngram_min_candidates = 0
greedy_loop_probe_ngram_risk_gate = true
greedy_loop_probe_ngram_risk_min_size = 16
greedy_loop_probe_ngram_history = [] of Int32
greedy_loop_probe_ngram_replay_start = -1
greedy_loop_probe_ngram_cursor_only = false
greedy_loop_prefix_tokens = [] of Int32
runtime_raw_q8 = false
runtime_skip_recurrent_ffn = false
runtime_skip_recurrent_ffn_layers = nil.as(Array(Int32)?)
runtime_pca_updown_zero = false
runtime_pca_updown_rank = 32
runtime_pca_updown_layers = nil.as(Array(Int32)?)
runtime_pca_updown_adapters_path : String? = nil
seed_token = 0
input_token = -1
input_tokens = [] of Int32
input_tokens_provided = false
known_replay_candidates = [] of Int32
known_replay_history = [] of Int32
known_replay_start = -1
known_replay_tokens = 0
known_replay_recover_on_reject = false
gpu_final_dump_path : String? = nil

OptionParser.parse do |p|
  p.banner = "Usage: cuda_mixed_stack_probe [--model PATH] [--layers LIST] [--tokens N] [--start-pos N] [--max-seq N] [--seed N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layers LIST", "Comma-separated layer ids in model order") { |v| layers = parse_layers(v) }
  p.on("--tokens N", "Sequence length for state progression") { |v| tokens = v.to_i }
  p.on("--start-pos N", "Starting decode position for full-attention KV cache") { |v| start_pos = v.to_i }
  p.on("--max-seq N", "KV cache capacity") { |v| max_seq = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--warmup N", "Untimed warmup stack runs") { |v| warmup = v.to_i }
  p.on("--steady-reps N", "After one reset priming run, time N runs without sequence/state reset; requires --perf-only") { |v| steady_reps = v.to_i }
  p.on("--steady-graph-reps N", "Capture one reset-free steady wave as a CUDA graph and replay it N times; requires --perf-only") { |v| steady_graph_reps = v.to_i }
  p.on("--read-logits", "Read full logits back for attribution; default reads resident CUDA top1 only") { read_logits = true }
  p.on("--gpu-logits-only", "Perf-only diagnostic: read CUDA logits/top2 margins without the CPU oracle") { gpu_logits_only = true; perf_only = true; read_logits = true }
  p.on("--margin-bucket-report", "In --read-logits oracle mode, bucket proposal top1 acceptance by exact top1/top2 margin") { margin_bucket_report = true }
  p.on("--margin-buckets LIST", "Comma-separated exact-margin bucket upper bounds for --margin-bucket-report") { |v| margin_bucket_edges = parse_f32_list(v) }
  p.on("--profile-phases", "Synchronize after each runner and print attribution timings; slower than default") { profile_phases = true }
  p.on("--skip-debug-readback", "Read only output-head results; skip final hidden/state/KV debug buffers for perf attribution") { debug_readback = false }
  p.on("--perf-only", "Skip CPU reference and hidden/state checks; reports CUDA timing/top1 only") { perf_only = true }
  p.on("--skip-output-head", "Perf-only body-floor probe: run model layers but skip output norm/lm-head/top1") { skip_output_head = true }
  p.on("--all-layers", "Run all model layers instead of the explicit/default layer slice") { all_layers = true }
  p.on("--greedy-loop-tokens N", "Run an embedding-driven greedy decode loop for N generated tokens; forces --tokens=1") { |v| greedy_loop_tokens = v.to_i }
  p.on("--greedy-loop-graph", "Capture the reset-free greedy-loop body as a CUDA graph and replay it after the first token") { greedy_loop_graph = true }
  p.on("--greedy-loop-no-graph", "Force direct per-token launches in --greedy-loop-tokens mode") { greedy_loop_no_graph = true }
  p.on("--greedy-loop-graph-device-ready", "Instantiate the greedy-loop CUDA graph with DEVICE_LAUNCH constraints; still host-launched by this probe") { greedy_loop_graph_device_ready = true }
  p.on("--greedy-loop-gpu-embedding", "Feed greedy-loop top1 ids back through a CUDA Q4_K token-embedding kernel instead of per-token CPU readback/embedding upload") { greedy_loop_gpu_embedding = true }
  p.on("--greedy-loop-cpu-embedding", "Force per-token CPU top1 readback and embedding upload in --greedy-loop-tokens mode") { greedy_loop_cpu_embedding = true }
  p.on("--greedy-loop-read-logits", "Diagnostic: read CUDA logits/top2 margins after each greedy-loop token; forces CPU feedback and no graph") { greedy_loop_read_logits = true; perf_only = true; read_logits = true; greedy_loop_cpu_embedding = true; greedy_loop_no_graph = true }
  p.on("--greedy-loop-probe-restore", "Diagnostic: run raw-Q8 proposal, restore recurrent state, then run exact verifier each token") { greedy_loop_probe_restore = true; perf_only = true; read_logits = true; greedy_loop_cpu_embedding = true; greedy_loop_no_graph = true }
  p.on("--greedy-loop-probe-restore-kv", "Also snapshot/restore full-attention KV caches in --greedy-loop-probe-restore") { greedy_loop_probe_restore_kv = true }
  p.on("--greedy-loop-probe-pca-updown", "Use loaded PCA-updown adapters as the discardable proposal route in --greedy-loop-probe-restore") { greedy_loop_probe_pca_updown = true }
  p.on("--greedy-loop-probe-pca-updown-raw-q8-rest", "With --greedy-loop-probe-pca-updown, use raw-Q8 recurrent FFN for non-PCA proposal layers") { greedy_loop_probe_pca_updown_raw_q8_rest = true }
  p.on("--greedy-loop-probe-chunk-gamma N", "Diagnostic: guarded chunk proposals with sequential/exact verification; forces CPU feedback and no graph") { |v| greedy_loop_probe_chunk_gamma = v.to_i; perf_only = true; read_top2 = true; greedy_loop_cpu_embedding = true; greedy_loop_no_graph = true }
  p.on("--greedy-loop-probe-chunk-margin F", "Raw-Q8 top1/top2 margin threshold for --greedy-loop-probe-chunk-gamma") { |v| greedy_loop_probe_chunk_margin = v.to_f32 }
  p.on("--greedy-loop-probe-chunk-batched-verify", "Use a second tokens=gamma stack to verify full raw-Q8 chunks and copy back only on full accept") { greedy_loop_probe_chunk_batched_verify = true }
  p.on("--greedy-loop-probe-chunk-fast-verify-top1", "Diagnostic: in batched chunk verification, trust resident CUDA top1 ids instead of copying/scanning full logits") { greedy_loop_probe_chunk_fast_verify_top1 = true }
  p.on("--greedy-loop-probe-ngram", "Use history n-gram candidates as the guarded chunk proposal source instead of raw-Q8 proposal") { greedy_loop_probe_ngram = true; perf_only = true; greedy_loop_cpu_embedding = true; greedy_loop_no_graph = true }
  p.on("--greedy-loop-probe-ngram-min N", "Minimum n-gram length for --greedy-loop-probe-ngram, default 2") { |v| greedy_loop_probe_ngram_min = v.to_i }
  p.on("--greedy-loop-probe-ngram-max N", "Maximum n-gram length for --greedy-loop-probe-ngram, default 8") { |v| greedy_loop_probe_ngram_max = v.to_i }
  p.on("--greedy-loop-probe-ngram-nonrecursive", "Disable recursive expansion for --greedy-loop-probe-ngram") { greedy_loop_probe_ngram_recursive = false }
  p.on("--greedy-loop-probe-ngram-min-candidates N", "Require at least N proposed tokens from --greedy-loop-probe-ngram") { |v| greedy_loop_probe_ngram_min_candidates = v.to_i }
  p.on("--greedy-loop-probe-ngram-no-risk-gate", "Disable n-gram risky-shape fallback gate") { greedy_loop_probe_ngram_risk_gate = false }
  p.on("--greedy-loop-probe-ngram-risk-min-size N", "Minimum candidate size for the risky-shape gate, default 16") { |v| greedy_loop_probe_ngram_risk_min_size = v.to_i }
  p.on("--greedy-loop-probe-ngram-history LIST", "Comma-separated proposal history token IDs; default is --seed-token only") { |v| greedy_loop_probe_ngram_history = parse_i32_list(v) }
  p.on("--greedy-loop-probe-ngram-replay-start N", "Start n-gram proposals at an aligned history cursor instead of suffix search; use for prompt/session cache hits") { |v| greedy_loop_probe_ngram_replay_start = v.to_i }
  p.on("--greedy-loop-probe-ngram-cursor-only", "Only propose from an active replay cursor; fallback instead of suffix-searching") { greedy_loop_probe_ngram_cursor_only = true }
  p.on("--greedy-loop-prefix-tokens LIST", "Comma-separated prompt/prefix token IDs to prefill before timed greedy-loop generation") { |v| greedy_loop_prefix_tokens = parse_i32_list(v) }
  p.on("--runtime-raw-q8", "Diagnostic: enable recurrent FFN raw-Q8 through the runtime stack switch instead of the environment default") { runtime_raw_q8 = true }
  p.on("--runtime-skip-recurrent-ffn", "Diagnostic proposal route: skip recurrent-layer FFNs and forward the post-attention residual") { runtime_skip_recurrent_ffn = true }
  p.on("--runtime-skip-recurrent-ffn-layers LIST", "Diagnostic proposal route: skip recurrent-layer FFNs only for comma-separated layer ids") { |v| runtime_skip_recurrent_ffn_layers = parse_layers(v) }
  p.on("--runtime-pca-updown-zero", "Diagnostic plumbing route: replace recurrent FFNs with zero PCA-updown adapters plus residual add") { runtime_pca_updown_zero = true; perf_only = true }
  p.on("--runtime-pca-updown-rank N", "Rank for --runtime-pca-updown-zero, default 32") { |v| runtime_pca_updown_rank = v.to_i }
  p.on("--runtime-pca-updown-layers LIST", "Apply --runtime-pca-updown-zero only to comma-separated recurrent layer ids") { |v| runtime_pca_updown_layers = parse_layers(v) }
  p.on("--runtime-pca-updown-adapters PATH", "Diagnostic proposal route: load real FFN PCA-updown adapters exported by qwen35_deltanet_fixed_basis_probe") { |v| runtime_pca_updown_adapters_path = v; perf_only = true }
  p.on("--seed-token ID", "Seed token id for --greedy-loop-tokens") { |v| seed_token = v.to_i }
  p.on("--input-token ID", "Use token_embd[ID] as the single non-greedy oracle input and zero recurrent states") { |v| input_token = v.to_i }
  p.on("--input-tokens LIST", "Use comma-separated token_embd IDs as the non-greedy semantic input sequence") { |v| input_tokens_provided = true; input_tokens = parse_i32_list(v) }
  p.on("--known-replay-candidates LIST", "For non-greedy --input-tokens, compare resident top1 rows against expected next-token candidates") { |v| known_replay_candidates = parse_i32_list(v); perf_only = true }
  p.on("--known-replay-history LIST", "Derive --input-tokens and --known-replay-candidates from a cached token history") { |v| known_replay_history = parse_i32_list(v); perf_only = true }
  p.on("--known-replay-start N", "Candidate start index inside --known-replay-history; inputs start at N-1") { |v| known_replay_start = v.to_i }
  p.on("--known-replay-tokens N", "Number of replay candidates to verify from --known-replay-history") { |v| known_replay_tokens = v.to_i }
  p.on("--known-replay-recover-on-reject", "Diagnostic: on known-replay reject, run a fresh short verifier for accepted-prefix+reject rows") { known_replay_recover_on_reject = true }
  p.on("--gpu-final-dump PATH", "Diagnostic: dump final hidden rows as raw little-endian f32 after a GPU run; works with --perf-only") { |v| gpu_final_dump_path = v }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

if !known_replay_history.empty?
  raise "--known-replay-history is incompatible with --input-token" if input_token >= 0
  raise "--known-replay-history is incompatible with --input-tokens" if input_tokens_provided || !input_tokens.empty?
  raise "--known-replay-history is incompatible with --known-replay-candidates" unless known_replay_candidates.empty?
  raise "--known-replay-start must be positive because inputs start at start-1" unless known_replay_start > 0
  raise "--known-replay-tokens must be positive" unless known_replay_tokens > 0
  raise "--known-replay-history too short for requested span" unless known_replay_start + known_replay_tokens <= known_replay_history.size

  input_tokens = known_replay_history[known_replay_start - 1, known_replay_tokens]
  known_replay_candidates = known_replay_history[known_replay_start, known_replay_tokens]
  input_tokens_provided = true
end

tokens = input_tokens.size unless input_tokens.empty?
raise "model not found: #{model}" unless File.exists?(model)
raise "layers must not be empty" if layers.empty?
raise "layers must be non-negative" unless layers.all? { |layer| layer >= 0 }
raise "layers must be strictly increasing for this probe" unless layers.each_cons(2).all? { |pair| pair[0] < pair[1] }
raise "tokens must be positive" unless tokens > 0
raise "start-pos must be non-negative" unless start_pos >= 0
raise "max-seq must cover start-pos + tokens" unless max_seq >= start_pos + tokens
raise "warmup must be non-negative" unless warmup >= 0
raise "steady-reps must be non-negative" unless steady_reps >= 0
raise "steady-graph-reps must be non-negative" unless steady_graph_reps >= 0
raise "--steady-reps requires --perf-only" if steady_reps > 0 && !perf_only
raise "--steady-reps does not support --profile-phases" if steady_reps > 0 && profile_phases
raise "--steady-graph-reps requires --perf-only" if steady_graph_reps > 0 && !perf_only
raise "--steady-graph-reps does not support --profile-phases" if steady_graph_reps > 0 && profile_phases
raise "use either --steady-reps or --steady-graph-reps, not both" if steady_reps > 0 && steady_graph_reps > 0
raise "--skip-output-head requires --perf-only" if skip_output_head && !perf_only
raise "--skip-output-head is incompatible with --read-logits" if skip_output_head && read_logits
raise "--gpu-logits-only is incompatible with --skip-output-head" if gpu_logits_only && skip_output_head
raise "--greedy-loop-tokens must be non-negative" unless greedy_loop_tokens >= 0
# CUDA graph replay is intentionally opt-in. On the RTX 5060 Ti / driver 595
# test node the current captured greedy body can segfault inside
# cuGraphInstantiate before CUDA returns an error code. Keep the explicit flag
# for graph debugging, but default to the stable direct-launch semantic path.
if greedy_loop_tokens > 1 && !greedy_loop_no_graph && ENV["QWEN_CUDA_GREEDY_GRAPH_DEFAULT"]? == "1"
  greedy_loop_graph = true
end
raise "use either --greedy-loop-graph or --greedy-loop-no-graph, not both" if greedy_loop_graph && greedy_loop_no_graph
raise "--greedy-loop-graph requires --greedy-loop-tokens" if greedy_loop_graph && greedy_loop_tokens == 0
raise "--greedy-loop-graph-device-ready requires --greedy-loop-graph" if greedy_loop_graph_device_ready && !greedy_loop_graph
raise "--greedy-loop-gpu-embedding requires --greedy-loop-tokens" if greedy_loop_gpu_embedding && greedy_loop_tokens == 0
raise "use either --greedy-loop-gpu-embedding or --greedy-loop-cpu-embedding, not both" if greedy_loop_gpu_embedding && greedy_loop_cpu_embedding
raise "--greedy-loop-read-logits requires --greedy-loop-tokens" if greedy_loop_read_logits && greedy_loop_tokens == 0
raise "--greedy-loop-read-logits is incompatible with --skip-output-head" if greedy_loop_read_logits && skip_output_head
raise "--greedy-loop-read-logits is incompatible with --greedy-loop-graph" if greedy_loop_read_logits && greedy_loop_graph
raise "--greedy-loop-probe-restore requires --greedy-loop-tokens" if greedy_loop_probe_restore && greedy_loop_tokens == 0
raise "--greedy-loop-probe-restore is incompatible with --skip-output-head" if greedy_loop_probe_restore && skip_output_head
raise "--greedy-loop-probe-restore is incompatible with --greedy-loop-graph" if greedy_loop_probe_restore && greedy_loop_graph
raise "--greedy-loop-probe-restore-kv requires --greedy-loop-probe-restore" if greedy_loop_probe_restore_kv && !greedy_loop_probe_restore
raise "--greedy-loop-probe-pca-updown requires --greedy-loop-probe-restore" if greedy_loop_probe_pca_updown && !greedy_loop_probe_restore
raise "--greedy-loop-probe-pca-updown requires --runtime-pca-updown-adapters" if greedy_loop_probe_pca_updown && !runtime_pca_updown_adapters_path
raise "--greedy-loop-probe-pca-updown-raw-q8-rest requires --greedy-loop-probe-pca-updown" if greedy_loop_probe_pca_updown_raw_q8_rest && !greedy_loop_probe_pca_updown
raise "--greedy-loop-probe-chunk-gamma must be non-negative" unless greedy_loop_probe_chunk_gamma >= 0
raise "--greedy-loop-probe-chunk-gamma requires --greedy-loop-tokens" if greedy_loop_probe_chunk_gamma > 0 && greedy_loop_tokens == 0
raise "--greedy-loop-probe-chunk-gamma is incompatible with --skip-output-head" if greedy_loop_probe_chunk_gamma > 0 && skip_output_head
raise "--greedy-loop-probe-chunk-gamma is incompatible with --greedy-loop-graph" if greedy_loop_probe_chunk_gamma > 0 && greedy_loop_graph
raise "--greedy-loop-probe-chunk-gamma is incompatible with --greedy-loop-probe-restore" if greedy_loop_probe_chunk_gamma > 0 && greedy_loop_probe_restore
raise "--greedy-loop-probe-chunk-margin must be non-negative" unless greedy_loop_probe_chunk_margin >= 0.0_f32
raise "--greedy-loop-probe-chunk-batched-verify requires --greedy-loop-probe-chunk-gamma" if greedy_loop_probe_chunk_batched_verify && greedy_loop_probe_chunk_gamma == 0
raise "--greedy-loop-probe-chunk-fast-verify-top1 requires --greedy-loop-probe-chunk-batched-verify" if greedy_loop_probe_chunk_fast_verify_top1 && !greedy_loop_probe_chunk_batched_verify
raise "--greedy-loop-probe-ngram requires --greedy-loop-probe-chunk-gamma" if greedy_loop_probe_ngram && greedy_loop_probe_chunk_gamma == 0
raise "--greedy-loop-probe-ngram-min must be positive" unless greedy_loop_probe_ngram_min > 0
raise "--greedy-loop-probe-ngram-max must be >= min" unless greedy_loop_probe_ngram_max >= greedy_loop_probe_ngram_min
raise "--greedy-loop-probe-ngram-min-candidates must be non-negative" unless greedy_loop_probe_ngram_min_candidates >= 0
raise "--greedy-loop-probe-ngram-risk-min-size must be positive" unless greedy_loop_probe_ngram_risk_min_size > 0
raise "--greedy-loop-probe-ngram-replay-start requires --greedy-loop-probe-ngram" if greedy_loop_probe_ngram_replay_start >= 0 && !greedy_loop_probe_ngram
raise "--greedy-loop-probe-ngram-cursor-only requires --greedy-loop-probe-ngram" if greedy_loop_probe_ngram_cursor_only && !greedy_loop_probe_ngram
raise "--greedy-loop-probe-ngram-replay-start must be >= -1" unless greedy_loop_probe_ngram_replay_start >= -1
raise "--greedy-loop-prefix-tokens requires --greedy-loop-tokens" if !greedy_loop_prefix_tokens.empty? && greedy_loop_tokens == 0
raise "--greedy-loop-prefix-tokens must contain at least one token when provided" if greedy_loop_prefix_tokens.empty? && ARGV.any? { |arg| arg.starts_with?("--greedy-loop-prefix-tokens") }
raise "--greedy-loop-prefix-tokens is incompatible with --greedy-loop-graph" if !greedy_loop_prefix_tokens.empty? && greedy_loop_graph
if !greedy_loop_prefix_tokens.empty? && !greedy_loop_probe_ngram_history.empty?
  raise "--greedy-loop-probe-ngram-history must end with --greedy-loop-prefix-tokens last token" unless greedy_loop_probe_ngram_history.last == greedy_loop_prefix_tokens.last
end
raise "--runtime-skip-recurrent-ffn is incompatible with --runtime-raw-q8" if runtime_skip_recurrent_ffn && runtime_raw_q8
raise "use either --runtime-skip-recurrent-ffn or --runtime-skip-recurrent-ffn-layers, not both" if runtime_skip_recurrent_ffn && runtime_skip_recurrent_ffn_layers
raise "--runtime-skip-recurrent-ffn-layers is incompatible with --runtime-raw-q8" if runtime_skip_recurrent_ffn_layers && runtime_raw_q8
raise "--runtime-pca-updown-rank must be in 1..64" unless runtime_pca_updown_rank > 0 && runtime_pca_updown_rank <= 64
raise "--runtime-pca-updown-layers requires --runtime-pca-updown-zero or --runtime-pca-updown-adapters" if runtime_pca_updown_layers && !runtime_pca_updown_zero && !runtime_pca_updown_adapters_path
raise "use either --runtime-pca-updown-zero or --runtime-pca-updown-adapters, not both" if runtime_pca_updown_zero && runtime_pca_updown_adapters_path
raise "--runtime-pca-updown-zero is incompatible with --runtime-raw-q8" if runtime_pca_updown_zero && runtime_raw_q8
raise "--runtime-pca-updown-zero is incompatible with --runtime-skip-recurrent-ffn" if runtime_pca_updown_zero && runtime_skip_recurrent_ffn
raise "--runtime-pca-updown-zero is incompatible with --runtime-skip-recurrent-ffn-layers" if runtime_pca_updown_zero && runtime_skip_recurrent_ffn_layers
raise "--runtime-pca-updown-adapters is incompatible with --runtime-raw-q8" if runtime_pca_updown_adapters_path && runtime_raw_q8
raise "--runtime-pca-updown-adapters is incompatible with --runtime-skip-recurrent-ffn" if runtime_pca_updown_adapters_path && runtime_skip_recurrent_ffn
raise "--runtime-pca-updown-adapters is incompatible with --runtime-skip-recurrent-ffn-layers" if runtime_pca_updown_adapters_path && runtime_skip_recurrent_ffn_layers
raise "adapter file not found: #{runtime_pca_updown_adapters_path}" if runtime_pca_updown_adapters_path && !File.file?(runtime_pca_updown_adapters_path.not_nil!)
raise "--input-token must be non-negative" if input_token < -1
raise "--input-token is incompatible with --greedy-loop-tokens; use --seed-token there" if input_token >= 0 && greedy_loop_tokens > 0
raise "--input-tokens must not be empty when provided" if input_tokens_provided && input_tokens.empty?
raise "--input-tokens is incompatible with --input-token" if !input_tokens.empty? && input_token >= 0
raise "--input-tokens is incompatible with --greedy-loop-tokens" if !input_tokens.empty? && greedy_loop_tokens > 0
raise "--known-replay-candidates requires --input-tokens" if !known_replay_candidates.empty? && input_tokens.empty?
raise "--known-replay-candidates is incompatible with --greedy-loop-tokens" if !known_replay_candidates.empty? && greedy_loop_tokens > 0
raise "--known-replay-candidates size must match --input-tokens size" if !known_replay_candidates.empty? && known_replay_candidates.size != input_tokens.size
raise "--known-replay-start requires --known-replay-history" if known_replay_start >= 0 && known_replay_history.empty?
raise "--known-replay-tokens requires --known-replay-history" if known_replay_tokens > 0 && known_replay_history.empty?
raise "--known-replay-recover-on-reject requires --known-replay-candidates or --known-replay-history" if known_replay_recover_on_reject && known_replay_candidates.empty?
if known_replay_recover_on_reject && (runtime_raw_q8 || runtime_skip_recurrent_ffn || runtime_pca_updown_zero || runtime_pca_updown_adapters_path)
  raise "--known-replay-recover-on-reject is only supported for the exact default runner route"
end
raise "--gpu-logits-only is incompatible with --greedy-loop-tokens" if gpu_logits_only && greedy_loop_tokens > 0
raise "--gpu-final-dump is incompatible with --greedy-loop-tokens" if gpu_final_dump_path && greedy_loop_tokens > 0
raise "--gpu-final-dump is incompatible with --steady-reps/--steady-graph-reps" if gpu_final_dump_path && (steady_reps > 0 || steady_graph_reps > 0)
raise "--margin-bucket-report requires --read-logits" if margin_bucket_report && !read_logits
raise "--margin-bucket-report is incompatible with --perf-only/--gpu-logits-only" if margin_bucket_report && perf_only
raise "--margin-buckets must not be empty" if margin_bucket_report && margin_bucket_edges.empty?
raise "--input-token requires --tokens=1" if input_token >= 0 && tokens != 1
if greedy_loop_tokens > 0
  raise "--skip-output-head is incompatible with --greedy-loop-tokens" if skip_output_head
  raise "--greedy-loop-tokens currently requires --perf-only; it is a semantic timing harness, not a CPU oracle" unless perf_only
  raise "--greedy-loop-tokens is incompatible with --steady-reps/--steady-graph-reps" if steady_reps > 0 || steady_graph_reps > 0
  raise "--greedy-loop-tokens is incompatible with --profile-phases" if profile_phases
  raise "--greedy-loop-tokens requires --tokens=1" unless tokens == 1
  greedy_loop_input_positions = greedy_loop_tokens + Math.max(greedy_loop_prefix_tokens.size - 1, 0)
  raise "max-seq must cover start-pos + prefix + greedy-loop-tokens" unless max_seq >= start_pos + greedy_loop_input_positions
end

eps = 1.0e-6_f32
gguf = ML::GGUF::GGUFFile.new(model)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
layers = (0...hparams.n_layer).map(&.to_i32) if all_layers
layers.each { |layer| raise "layer #{layer} out of range" unless layer < hparams.n_layer }
hidden = hparams.n_embd
debug_readback = false if perf_only
read_logits = false if perf_only && !gpu_logits_only && !greedy_loop_read_logits && !greedy_loop_probe_restore && greedy_loop_probe_chunk_gamma == 0
debug_readback = true if gpu_final_dump_path

rng = Random.new(seed)
token_embd = load_quant_weight(gguf, "token_embd.weight")
raise "seed-token #{seed_token} out of range" if seed_token < 0 || seed_token >= token_embd.out_dim
raise "input-token #{input_token} out of range" if input_token >= token_embd.out_dim
input_tokens.each { |tok| raise "input-tokens contains out-of-range token #{tok}" if tok < 0 || tok >= token_embd.out_dim }
known_replay_candidates.each { |tok| raise "known-replay-candidates contains out-of-range token #{tok}" if tok < 0 || tok >= token_embd.out_dim }
known_replay_history.each { |tok| raise "known-replay-history contains out-of-range token #{tok}" if tok < 0 || tok >= token_embd.out_dim }
greedy_loop_prefix_tokens.each { |tok| raise "greedy-loop-prefix-tokens contains out-of-range token #{tok}" if tok < 0 || tok >= token_embd.out_dim }
if greedy_loop_tokens > 0 && !greedy_loop_cpu_embedding && token_embd.type.q4_k?
  greedy_loop_gpu_embedding = true
end
semantic_input = greedy_loop_tokens > 0 || input_token >= 0 || !input_tokens.empty?
xs = if greedy_loop_tokens > 0
       ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, seed_token)
     elsif input_token >= 0
       ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, input_token)
     elsif !input_tokens.empty?
       seq = Array(Float32).new(input_tokens.size * hidden, 0.0_f32)
       input_tokens.each_with_index do |tok, i|
         emb = ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, tok)
         hidden.times { |j| seq[i * hidden + j] = emb[j] }
       end
       seq
     else
       Array(Float32).new(tokens * hidden) { ((rng.next_float - 0.5) * 0.2).to_f32 }
     end

recurrent_weights = {} of Int32 => ML::CUDA::QwenRecurrentLayerRunner::Weights
full_weights = {} of Int32 => ML::CUDA::QwenFullAttnLayerRunner::Weights
head_weights = ML::CUDA::QwenOutputHeadRunner::Weights.load(gguf)
conv_state_inits = {} of Int32 => Array(Float32)
ssm_state_inits = {} of Int32 => Array(Float32)

layers.each do |layer|
  if hparams.full_attention?(layer)
    full_weights[layer] = ML::CUDA::QwenFullAttnLayerRunner::Weights.load(gguf, layer)
  else
    weights = ML::CUDA::QwenRecurrentLayerRunner::Weights.load(gguf, layer, eps)
    recurrent_weights[layer] = weights
    conv_state_inits[layer] = Array(Float32).new((weights.conv_k - 1) * weights.qkv_dim) do
      semantic_input ? 0.0_f32 : ((rng.next_float - 0.5) * 0.05).to_f32
    end
    ssm_state_inits[layer] = Array(Float32).new(weights.h_v * weights.s * weights.s) do
      semantic_input ? 0.0_f32 : ((rng.next_float - 0.5) * 0.05).to_f32
    end
  end
end

cpu_current = xs.dup
cpu_states = Array(ML::GGUF::Qwen35CPU::LayerState).new(hparams.n_layer) { ML::GGUF::Qwen35CPU::LayerState.new }
cpu_ms = 0.0
cpu_top1_ids = [] of Int32
cpu_top2_ids = [] of Int32
cpu_top1_values = [] of Float32
cpu_top2_values = [] of Float32
cpu_top_margins = [] of Float32
cpu_logits_all = [] of Float32
cpu_weights = nil.as(ML::GGUF::Qwen35Weights?)

unless perf_only
  cpu_weights = ML::GGUF::Qwen35Weights.new(gguf, hparams)
  recurrent_weights.each_key do |layer|
    cpu_states[layer].conv_state = conv_state_inits[layer].dup
    cpu_states[layer].ssm_state = ssm_state_inits[layer].dup
  end

  cpu_t0 = Time.instant
  layers.each do |layer|
    lw = cpu_weights.not_nil!.layers[layer]
    out = Array(Float32).new(tokens * hidden, 0.0_f32)
    tokens.times do |tok|
      row = cpu_current[tok * hidden, hidden]
      y = case lw
          in ML::GGUF::Qwen35FullAttnWeights
            ML::GGUF::Qwen35CPU.forward_full_attn_layer(row, start_pos + tok, lw, cpu_states[layer], hparams, max_seq)
          in ML::GGUF::Qwen35RecurrentWeights
            ML::GGUF::Qwen35CPU.forward_recurrent_layer(row, 0, lw, cpu_states[layer], hparams, max_seq)
          end
      hidden.times { |i| out[tok * hidden + i] = y[i] }
    end
    cpu_current = out
  end
  cpu_ms = (Time.instant - cpu_t0).total_milliseconds
  cpu_logits_all = read_logits ? Array(Float32).new(tokens * head_weights.vocab, 0.0_f32) : [] of Float32
  cpu_top1_ids = Array(Int32).new(tokens)
  cpu_top2_ids = Array(Int32).new(tokens)
  cpu_top1_values = Array(Float32).new(tokens)
  cpu_top2_values = Array(Float32).new(tokens)
  cpu_top_margins = Array(Float32).new(tokens)
  tokens.times do |tok|
    row = cpu_current[tok * hidden, hidden]
    normed = ML::GGUF::Qwen35CPU.rms_norm(row, head_weights.norm, hparams.rms_eps)
    logits = ML::GGUF::QuantMatmul.matmul_add(normed, 1, head_weights.hidden,
      head_weights.output_raw, head_weights.output_type, head_weights.vocab,
      Array(Float32).new(head_weights.vocab, 0.0_f32))
    best_id = 0
    second_id = 0
    best = -Float32::INFINITY
    second = -Float32::INFINITY
    head_weights.vocab.times do |i|
      v = logits[i]
      cpu_logits_all[tok * head_weights.vocab + i] = v if read_logits
      if v > best
        second = best
        second_id = best_id
        best = v
        best_id = i
      elsif v > second
        second = v
        second_id = i
      end
    end
    cpu_top1_ids << best_id
    cpu_top2_ids << second_id
    cpu_top1_values << best
    cpu_top2_values << second
    cpu_top_margins << (best - second)
  end
end

cuda_ctx = nil.as(ML::CUDA::Context?)
runners = [] of ML::CUDA::QwenMixedStackRunner::LayerRunner
head = nil.as(ML::CUDA::QwenOutputHeadRunner?)
stack = nil.as(ML::CUDA::QwenMixedStackRunner?)
verifier_stack = nil.as(ML::CUDA::QwenMixedStackRunner?)
known_replay_recovery_stack = nil.as(ML::CUDA::QwenMixedStackRunner?)
final_gpu_all = Array(Float32).new(tokens * hidden, 0.0_f32)

begin
  cuda_ctx = ML::CUDA::Context.create
  # Full-attention CUDA kernels index RoPE tables by absolute decode position,
  # so upload a resident table once and update only start_pos in token loops.
  rope_table_tokens = start_pos + (greedy_loop_tokens > 0 ? greedy_loop_tokens + Math.max(greedy_loop_prefix_tokens.size - 1, 0) : tokens)
  cos_table, sin_table = rope_tables(rope_table_tokens, 0, hparams.rope_dim_count, hparams.rope_freq_base)

  layers.each_with_index do |layer, idx|
    layer_input = idx == 0 ? xs : Array(Float32).new(tokens * hidden, 0.0_f32)
    if hparams.full_attention?(layer)
      runners << ML::CUDA::QwenFullAttnLayerRunner.from_weights(full_weights[layer], tokens, max_seq, start_pos,
        hparams.n_head, hparams.n_head_kv, hparams.head_dim, hparams.rope_dim_count, hparams.rms_eps,
        layer_input, cos_table, sin_table)
    else
      runners << ML::CUDA::QwenRecurrentLayerRunner.from_weights(recurrent_weights[layer], tokens, layer_input,
        conv_state_inits[layer], ssm_state_inits[layer])
    end
  end
  head = ML::CUDA::QwenOutputHeadRunner.from_weights(head_weights, tokens,
    Array(Float32).new(tokens * hidden, 0.0_f32), hparams.rms_eps, read_logits: read_logits, read_top2: read_top2)
  output_head = head.not_nil!
  stack = ML::CUDA::QwenMixedStackRunner.new(layers, runners, output_head, tokens, hidden, xs)
  mixed_stack = stack.not_nil!
  mixed_stack.set_recurrent_ffn_raw_q8(true) if runtime_raw_q8
  mixed_stack.set_recurrent_ffn_skip(true) if runtime_skip_recurrent_ffn
  if skip_layers = runtime_skip_recurrent_ffn_layers
    mixed_stack.set_recurrent_ffn_skip_layers(skip_layers)
  end
  if runtime_pca_updown_zero
    mixed_stack.set_recurrent_ffn_pca_updown_zero(runtime_pca_updown_rank, runtime_pca_updown_layers)
  end
  if adapter_path = runtime_pca_updown_adapters_path
    adapters = load_ffn_pca_updown_adapters(adapter_path, hidden)
    selected = runtime_pca_updown_layers.try(&.to_set)
    layers.zip(runners).each do |layer_id, runner|
      next if selected && !selected.not_nil!.includes?(layer_id)
      case runner
      in ML::CUDA::QwenRecurrentLayerRunner
        adapter = adapters[layer_id]? || raise "missing PCA-updown adapter for recurrent layer #{layer_id}"
        runner.set_ffn_pca_updown_adapter(adapter[:x_mean], adapter[:c_mean], adapter[:coeff_w], adapter[:down], adapter[:rank])
      in ML::CUDA::QwenFullAttnLayerRunner
        # Full-attention layers keep their exact FFN path in this probe.
      end
    end
    if greedy_loop_probe_pca_updown
      mixed_stack.set_recurrent_ffn_pca_updown_enabled(false, runtime_pca_updown_layers)
    end
  end

  weight_upload_ms = mixed_stack.upload_weights(profile: profile_phases)

  verifier_weight_upload_ms = 0.0
  if greedy_loop_probe_chunk_batched_verify
    verifier_tokens = greedy_loop_probe_chunk_gamma
    verifier_xs = Array(Float32).new(verifier_tokens * hidden, 0.0_f32)
    verifier_runners = [] of ML::CUDA::QwenMixedStackRunner::LayerRunner
    layers.each_with_index do |layer, idx|
      layer_input = idx == 0 ? verifier_xs : Array(Float32).new(verifier_tokens * hidden, 0.0_f32)
      if hparams.full_attention?(layer)
        verifier_runners << ML::CUDA::QwenFullAttnLayerRunner.from_weights(full_weights[layer], verifier_tokens, max_seq, start_pos,
          hparams.n_head, hparams.n_head_kv, hparams.head_dim, hparams.rope_dim_count, hparams.rms_eps,
          layer_input, cos_table, sin_table)
      else
        verifier_runners << ML::CUDA::QwenRecurrentLayerRunner.from_weights(recurrent_weights[layer], verifier_tokens, layer_input,
          conv_state_inits[layer], ssm_state_inits[layer])
      end
    end
    verifier_head = ML::CUDA::QwenOutputHeadRunner.from_weights(head_weights, verifier_tokens,
      Array(Float32).new(verifier_tokens * hidden, 0.0_f32), hparams.rms_eps,
      read_logits: !greedy_loop_probe_chunk_fast_verify_top1)
    verifier_stack = ML::CUDA::QwenMixedStackRunner.new(layers, verifier_runners, verifier_head, verifier_tokens, hidden, verifier_xs)
    verifier_weight_upload_ms = verifier_stack.not_nil!.upload_weights(profile: false)
  end

  measured_tokens = tokens
  greedy_gpu_ids = [] of Int32
  greedy_position_ms = 0.0
  greedy_embedding_ms = 0.0
  greedy_body_ms = 0.0
  greedy_read_ms = 0.0
  greedy_prefix_ms = 0.0
  greedy_logits_top1_ids = [] of Int32
  greedy_top2_ids = [] of Int32
  greedy_top1_values = [] of Float32
  greedy_top2_values = [] of Float32
  greedy_top_margins = [] of Float32
  greedy_logits_mismatches = [] of String
  probe_raw_ids = [] of Int32
  probe_raw_top2_ids = [] of Int32
  probe_raw_margins = [] of Float32
  probe_raw_ms = 0.0
  probe_exact_ms = 0.0
  chunk_proposal_ids = [] of Int32
  chunk_exact_ids = [] of Int32
  chunk_raw_margins = [] of Float32
  chunk_raw_top2_ids = [] of Int32
  chunk_count = 0
  chunk_full_accepts = 0
  chunk_rejects = 0
  chunk_margin_fallbacks = 0
  chunk_raw_tokens = 0
  chunk_verify_tokens = 0
  chunk_accepted_tokens = 0
  chunk_raw_ms = 0.0
  chunk_verify_ms = 0.0
  chunk_batched_verify_ms = 0.0
  chunk_batched_verify_chunks = 0
  chunk_batched_verify_accepts = 0
  chunk_batched_verify_rejects = 0
  chunk_batched_verify_last_reject_commits = 0
  chunk_ngram_empty_fallbacks = 0
  chunk_ngram_risk_fallbacks = 0
  chunk_ngram_cursor_hits = 0
  chunk_ngram_match_lens = [] of Int32
  if greedy_loop_tokens > 0
    warmup.times do
      warm_token = seed_token
      Math.min(greedy_loop_tokens, 2).times do |i|
        pos = start_pos + i
        mixed_stack.update_decode_position(pos)
        mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, warm_token))
        mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: i == 0)
        mixed_stack.read_head_outputs
        warm_token = output_head.top1_ids[0]
      end
    end

    greedy_loop_start_token = greedy_loop_prefix_tokens.empty? ? seed_token : greedy_loop_prefix_tokens.last
    greedy_loop_decode_base_pos = start_pos + Math.max(greedy_loop_prefix_tokens.size - 1, 0)
    greedy_loop_reset_on_first_generated = greedy_loop_prefix_tokens.size <= 1
    if greedy_loop_prefix_tokens.size > 1
      prefix_t0 = Time.instant
      greedy_loop_prefix_tokens[0, greedy_loop_prefix_tokens.size - 1].each_with_index do |tok, i|
        mixed_stack.update_decode_position(start_pos + i)
        mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, tok))
        mixed_stack.run_sequence(profile_phases: false, debug_readback: false,
          reset_sequence: i == 0, sync_end: true, read_head_outputs: false, run_head: false)
      end
      greedy_prefix_ms = (Time.instant - prefix_t0).total_milliseconds
    end

    if greedy_loop_probe_chunk_gamma > 0
      ngram_index = nil.as(ML::GGUF::NgramDraft::IndexedHistory?)
      ngram_replay_cursor = nil.as(Int32?)
      ngram_replay_limit = 0
      gpu_token = greedy_loop_start_token
      if greedy_loop_probe_ngram
        initial_history = if !greedy_loop_probe_ngram_history.empty?
                            greedy_loop_probe_ngram_history
                          elsif !greedy_loop_prefix_tokens.empty?
                            greedy_loop_prefix_tokens
                          else
                            [greedy_loop_start_token]
                          end
        ngram_index = ML::GGUF::NgramDraft::IndexedHistory.new(initial_history, greedy_loop_probe_ngram_max, greedy_loop_probe_ngram_min)
        ngram_replay_limit = initial_history.size
        if greedy_loop_probe_ngram_replay_start >= 0
          raise "--greedy-loop-probe-ngram-replay-start out of history range" unless greedy_loop_probe_ngram_replay_start < ngram_replay_limit

          ngram_replay_cursor = greedy_loop_probe_ngram_replay_start
        end
        gpu_token = initial_history.last
      end
      generated = 0
      gpu_t0 = Time.instant
      while generated < greedy_loop_tokens
        chunk_count += 1
        chunk_start = generated
        chunk_limit = Math.min(greedy_loop_probe_chunk_gamma, greedy_loop_tokens - generated)
        proposal_ids = [] of Int32
        proposal_margins = [] of Float32
        proposal_top2 = [] of Int32
        fallback_due_margin = false
        ngram_pending_replay_cursor = nil.as(Int32?)
        ngram_match_len_for_gate = 0
        if greedy_loop_probe_ngram
          index = ngram_index.not_nil!
          if cursor = ngram_replay_cursor
            replay_count = Math.min(chunk_limit, ngram_replay_limit - cursor)
            if replay_count > 0
              proposal_ids = index.history[cursor, replay_count]
              ngram_pending_replay_cursor = cursor + proposal_ids.size
              chunk_ngram_cursor_hits += 1
              chunk_ngram_match_lens << -1
              ngram_match_len_for_gate = greedy_loop_probe_ngram_max
            end
          end
          if proposal_ids.empty? && !greedy_loop_probe_ngram_cursor_only
            span = index.candidate_span(chunk_limit,
              recursive: greedy_loop_probe_ngram_recursive,
              min_candidates: greedy_loop_probe_ngram_min_candidates)
            if span
              proposal_ids = span.ids
              ngram_pending_replay_cursor = span.source_start + span.match_len + proposal_ids.size
              chunk_ngram_match_lens << span.match_len
              ngram_match_len_for_gate = span.match_len
            else
              chunk_ngram_match_lens << 0
            end
          end
          if greedy_loop_probe_ngram_risk_gate && ML::GGUF::NgramDraft.risky_candidate_shape?(proposal_ids, greedy_loop_probe_ngram_risk_min_size, ngram_match_len_for_gate)
            chunk_ngram_risk_fallbacks += 1
            proposal_ids = [] of Int32
          end
          if proposal_ids.empty?
            chunk_ngram_empty_fallbacks += 1
          else
            proposal_ids.each do |id|
              chunk_proposal_ids << id
              chunk_raw_tokens += 1
            end
          end
        else
          snapshot = mixed_stack.snapshot_decode_state(include_kv: greedy_loop_probe_restore_kv)
          begin
            mixed_stack.set_recurrent_ffn_raw_q8(true)
            proposal_token = gpu_token
            chunk_limit.times do |j|
              pos = greedy_loop_decode_base_pos + generated + j
              t_position = Time.instant
              mixed_stack.update_decode_position(pos)
              greedy_position_ms += (Time.instant - t_position).total_milliseconds

              t_embedding = Time.instant
              mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, proposal_token))
              greedy_embedding_ms += (Time.instant - t_embedding).total_milliseconds

              t_raw = Time.instant
              mixed_stack.run_sequence(profile_phases: false, debug_readback: false,
                reset_sequence: greedy_loop_reset_on_first_generated && generated == 0 && j == 0, sync_end: true, read_head_outputs: true)
              chunk_raw_ms += (Time.instant - t_raw).total_milliseconds
              raw_best_id = output_head.top1_ids[0]
              raw_second_id = output_head.top2_ids_gpu[0]
              raw_margin = output_head.top1_values_gpu[0] - output_head.top2_values_gpu[0]
              proposal_ids << raw_best_id
              proposal_top2 << raw_second_id
              proposal_margins << raw_margin
              chunk_proposal_ids << raw_best_id
              chunk_raw_top2_ids << raw_second_id
              chunk_raw_margins << raw_margin
              chunk_raw_tokens += 1
              proposal_token = raw_best_id
              if raw_margin < greedy_loop_probe_chunk_margin
                fallback_due_margin = true
                break
              end
            end
            mixed_stack.restore_decode_state(snapshot)
          ensure
            mixed_stack.set_recurrent_ffn_raw_q8(false)
            snapshot.close
          end
        end

        if fallback_due_margin || proposal_ids.empty?
          chunk_margin_fallbacks += 1
          pos = greedy_loop_decode_base_pos + generated
          t_position = Time.instant
          mixed_stack.update_decode_position(pos)
          greedy_position_ms += (Time.instant - t_position).total_milliseconds
          t_embedding = Time.instant
          mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, gpu_token))
          greedy_embedding_ms += (Time.instant - t_embedding).total_milliseconds
          t_exact = Time.instant
          mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback,
            reset_sequence: greedy_loop_reset_on_first_generated && generated == 0, sync_end: true, read_head_outputs: true)
          chunk_verify_ms += (Time.instant - t_exact).total_milliseconds
          exact_id = output_head.top1_ids[0]
          chunk_exact_ids << exact_id
          greedy_gpu_ids << exact_id
          ngram_index.try(&.append(exact_id))
          gpu_token = exact_id
          generated += 1
          chunk_verify_tokens += 1
          next
        end

        accepted_this_chunk = 0
        rejected = false
        used_batched_verify = false

        if greedy_loop_probe_chunk_batched_verify && proposal_ids.size == greedy_loop_probe_chunk_gamma && generated + proposal_ids.size <= greedy_loop_tokens
          verify_stack = verifier_stack.not_nil!
          verify_xs = Array(Float32).new(greedy_loop_probe_chunk_gamma * hidden, 0.0_f32)
          verify_inputs = [gpu_token] + proposal_ids[0, proposal_ids.size - 1]
          verify_inputs.each_with_index do |tok, row|
            emb = ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, tok)
            hidden.times { |h| verify_xs[row * hidden + h] = emb[h] }
          end
          mixed_stack.copy_decode_state_to!(verify_stack, include_kv: true)
          verify_stack.update_decode_position(greedy_loop_decode_base_pos + generated)
          verify_stack.upload_first_sequence_input(verify_xs)
          t_batched = Time.instant
          verify_stack.run_sequence(profile_phases: false, debug_readback: false,
            reset_sequence: false, sync_end: true, read_head_outputs: true)
          chunk_batched_verify_ms += (Time.instant - t_batched).total_milliseconds
          chunk_batched_verify_chunks += 1
          exact_ids = if greedy_loop_probe_chunk_fast_verify_top1
                        verify_stack.head.top1_ids
                      else
                        Array(Int32).new(proposal_ids.size) do |j|
                          best_id, _, _, _, _ = top2_from_logits(verify_stack.head.logits_gpu_all, j, head_weights.vocab)
                          best_id
                        end
                      end
          chunk_verify_tokens += proposal_ids.size
          proposal_ids.each_with_index do |proposal_id, j|
            exact_id = exact_ids[j]
            chunk_exact_ids << exact_id
            greedy_gpu_ids << exact_id
            if exact_id == proposal_id
              accepted_this_chunk += 1
              chunk_accepted_tokens += 1
            else
              rejected = true
              break
            end
          end
          if rejected
            chunk_batched_verify_rejects += 1
            if accepted_this_chunk == proposal_ids.size - 1
              # If only the final proposed token rejected, the batched verifier
              # state is still canonical: all verifier inputs were accepted.
              verify_stack.copy_decode_state_to!(mixed_stack, include_kv: true)
              generated += proposal_ids.size
              gpu_token = exact_ids[accepted_this_chunk]
              ngram_index.try(&.append(exact_ids[0, proposal_ids.size]))
              ngram_replay_cursor = nil
              used_batched_verify = true
              chunk_batched_verify_last_reject_commits += 1
            else
              greedy_gpu_ids = greedy_gpu_ids[0, greedy_gpu_ids.size - accepted_this_chunk - 1]
              chunk_exact_ids = chunk_exact_ids[0, chunk_exact_ids.size - accepted_this_chunk - 1]
              chunk_verify_tokens -= proposal_ids.size
              chunk_accepted_tokens -= accepted_this_chunk
              accepted_this_chunk = 0
            end
          else
            chunk_batched_verify_accepts += 1
            verify_stack.copy_decode_state_to!(mixed_stack, include_kv: true)
            generated += proposal_ids.size
            gpu_token = proposal_ids.last
            ngram_index.try(&.append(proposal_ids))
            ngram_replay_cursor = ngram_pending_replay_cursor
            used_batched_verify = true
          end
        end

        unless used_batched_verify
          proposal_ids.each_with_index do |proposal_id, j|
            pos = greedy_loop_decode_base_pos + generated
            t_position = Time.instant
            mixed_stack.update_decode_position(pos)
            greedy_position_ms += (Time.instant - t_position).total_milliseconds
            t_embedding = Time.instant
            mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, gpu_token))
            greedy_embedding_ms += (Time.instant - t_embedding).total_milliseconds
            t_exact = Time.instant
            mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback,
              reset_sequence: greedy_loop_reset_on_first_generated && generated == 0 && j == 0 && chunk_start == 0, sync_end: true, read_head_outputs: true)
            chunk_verify_ms += (Time.instant - t_exact).total_milliseconds
            exact_id = output_head.top1_ids[0]
            chunk_exact_ids << exact_id
            greedy_gpu_ids << exact_id
            ngram_index.try(&.append(exact_id))
            generated += 1
            chunk_verify_tokens += 1
            gpu_token = exact_id
            if exact_id == proposal_id
              accepted_this_chunk += 1
              chunk_accepted_tokens += 1
            else
              rejected = true
              break
            end
            break if generated >= greedy_loop_tokens
          end
        end

        if rejected
          chunk_rejects += 1
          ngram_replay_cursor = nil if greedy_loop_probe_ngram
        elsif accepted_this_chunk == proposal_ids.size
          chunk_full_accepts += 1
          ngram_replay_cursor = ngram_pending_replay_cursor if greedy_loop_probe_ngram && !proposal_ids.empty?
        end
      end
      gpu_ms = (Time.instant - gpu_t0).total_milliseconds
      measured_tokens = greedy_loop_tokens
    elsif greedy_loop_probe_restore
      gpu_token = greedy_loop_start_token
      gpu_t0 = Time.instant
      greedy_loop_tokens.times do |i|
        pos = greedy_loop_decode_base_pos + i
        t_position = Time.instant
        mixed_stack.update_decode_position(pos)
        greedy_position_ms += (Time.instant - t_position).total_milliseconds

        t_embedding = Time.instant
        mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, gpu_token))
        greedy_embedding_ms += (Time.instant - t_embedding).total_milliseconds

        snapshot = mixed_stack.snapshot_decode_state(include_kv: greedy_loop_probe_restore_kv)
        begin
          if greedy_loop_probe_pca_updown
            mixed_stack.set_recurrent_ffn_raw_q8(true) if greedy_loop_probe_pca_updown_raw_q8_rest
            mixed_stack.set_recurrent_ffn_pca_updown_enabled(true, runtime_pca_updown_layers)
          else
            mixed_stack.set_recurrent_ffn_raw_q8(true)
          end
          t_raw = Time.instant
          mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: greedy_loop_reset_on_first_generated && i == 0,
            sync_end: true, read_head_outputs: true)
          probe_raw_ms += (Time.instant - t_raw).total_milliseconds
          raw_best_id, _, raw_second_id, _, raw_margin = top2_from_logits(output_head.logits_gpu_all, 0, head_weights.vocab)
          probe_raw_ids << raw_best_id
          probe_raw_top2_ids << raw_second_id
          probe_raw_margins << raw_margin

          mixed_stack.restore_decode_state(snapshot)
        ensure
          mixed_stack.set_recurrent_ffn_pca_updown_enabled(false, runtime_pca_updown_layers) if greedy_loop_probe_pca_updown
          mixed_stack.set_recurrent_ffn_raw_q8(false) if greedy_loop_probe_pca_updown_raw_q8_rest
          snapshot.close
        end

        mixed_stack.set_recurrent_ffn_raw_q8(false) unless greedy_loop_probe_pca_updown
        t_exact = Time.instant
        mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback, reset_sequence: greedy_loop_reset_on_first_generated && i == 0,
          sync_end: true, read_head_outputs: true)
        probe_exact_ms += (Time.instant - t_exact).total_milliseconds
        gpu_token = output_head.top1_ids[0]
        greedy_gpu_ids << gpu_token
      end
      gpu_ms = (Time.instant - gpu_t0).total_milliseconds
      measured_tokens = greedy_loop_tokens
    else
      graph_stream = nil.as(ML::CUDA::CUDAStream?)
      graph = nil.as(ML::CUDA::CUDAGraph?)
      graph_exec = nil.as(ML::CUDA::CUDAGraphExec?)
      gpu_embedder = nil.as(CudaQ4KTokenEmbedder?)
      if greedy_loop_graph && greedy_loop_tokens > 1
        graph_stream = ML::CUDA::CUDAStream.new
        ML::CUDA.with_stream(graph_stream.not_nil!) do
          graph_stream.not_nil!.begin_capture
          mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: false,
            sync_end: false, read_head_outputs: false)
          mixed_stack.increment_decode_position
          graph = graph_stream.not_nil!.end_capture
        end
        graph_exec = if greedy_loop_graph_device_ready
                       graph.not_nil!.instantiate_device_launch(graph_stream.not_nil!)
                     else
                       graph.not_nil!.instantiate
                     end
        graph.not_nil!.close
        graph = nil
        graph_exec.not_nil!.upload(graph_stream.not_nil!)
        graph_stream.not_nil!.synchronize
      end

      if greedy_loop_gpu_embedding
        gpu_embedder = CudaQ4KTokenEmbedder.new(token_embd, mixed_stack.top1_ids_device_ptr,
          mixed_stack.first_sequence_input_device_ptr, greedy_loop_tokens)
      end

      gpu_token = greedy_loop_start_token
      gpu_t0 = Time.instant
      begin
        greedy_loop_tokens.times do |i|
          pos = greedy_loop_decode_base_pos + i
          t_position = Time.instant
          if !greedy_loop_graph || i <= 1
            mixed_stack.update_decode_position(pos)
          end
          greedy_position_ms += (Time.instant - t_position).total_milliseconds

          t_embedding = Time.instant
          unless greedy_loop_gpu_embedding && i > 0
            mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, gpu_token))
          end
          greedy_embedding_ms += (Time.instant - t_embedding).total_milliseconds

          t_body = Time.instant
          if greedy_loop_graph && i > 0
            graph_exec.not_nil!.launch(graph_stream.not_nil!)
            graph_stream.not_nil!.synchronize unless greedy_loop_gpu_embedding
          else
            mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback, reset_sequence: greedy_loop_reset_on_first_generated && i == 0,
              sync_end: !greedy_loop_gpu_embedding, read_head_outputs: false)
          end
          greedy_body_ms += (Time.instant - t_body).total_milliseconds

          if greedy_loop_gpu_embedding
            t_feedback = Time.instant
            if stream = graph_stream
              ML::CUDA.with_stream(stream) { gpu_embedder.not_nil!.record_and_embed(i) }
            else
              gpu_embedder.not_nil!.record_and_embed(i)
            end
            greedy_embedding_ms += (Time.instant - t_feedback).total_milliseconds
          else
            t_read = Time.instant
            mixed_stack.read_head_outputs
            greedy_read_ms += (Time.instant - t_read).total_milliseconds
            if greedy_loop_read_logits
              best_id, best, second_id, second, margin = top2_from_logits(output_head.logits_gpu_all, 0, head_weights.vocab)
              greedy_logits_top1_ids << best_id
              greedy_top2_ids << second_id
              greedy_top1_values << best
              greedy_top2_values << second
              greedy_top_margins << margin
              greedy_logits_mismatches << "#{i}:#{best_id}:#{output_head.top1_ids[0]}" if best_id != output_head.top1_ids[0]
            end
            gpu_token = output_head.top1_ids[0]
            greedy_gpu_ids << gpu_token
          end
        end
        if greedy_loop_gpu_embedding
          t_read = Time.instant
          if stream = graph_stream
            stream.synchronize
          else
            ML::CUDA.synchronize!("cuCtxSynchronize(gpu embedding greedy loop)")
          end
          greedy_gpu_ids = gpu_embedder.not_nil!.read_history
          greedy_read_ms += (Time.instant - t_read).total_milliseconds
        end
      ensure
        gpu_embedder.try(&.close)
        graph_exec.try(&.close)
        graph.try(&.close)
        graph_stream.try(&.close)
      end
      gpu_ms = (Time.instant - gpu_t0).total_milliseconds
    end
    measured_tokens = greedy_loop_tokens
  else
    warmup.times { mixed_stack.run_sequence(profile_phases: false, debug_readback: false, run_head: !skip_output_head) }

    if steady_graph_reps > 0
      mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: true, run_head: !skip_output_head)
      stream = ML::CUDA::CUDAStream.new
      graph = nil.as(ML::CUDA::CUDAGraph?)
      graph_exec = nil.as(ML::CUDA::CUDAGraphExec?)
      begin
        ML::CUDA.with_stream(stream) do
          stream.begin_capture
          mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: false,
            sync_end: false, read_head_outputs: false, run_head: !skip_output_head)
          graph = stream.end_capture
        end
        graph_exec = graph.not_nil!.instantiate
        graph.not_nil!.close
        graph_exec.not_nil!.upload(stream)
        stream.synchronize
        t_graph = Time.instant
        steady_graph_reps.times do
          graph_exec.not_nil!.launch(stream)
        end
        stream.synchronize
        gpu_ms = (Time.instant - t_graph).total_milliseconds
        measured_tokens = tokens * steady_graph_reps
        mixed_stack.read_head_outputs unless skip_output_head
      ensure
        graph_exec.try(&.close)
        graph.try(&.close)
        stream.close
      end
    elsif steady_reps > 0
      # Prime device inputs and decode states once, then measure the steady path
      # where recurrent/KV state stays resident across decode steps.
      mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: true, run_head: !skip_output_head)
      t_steady = Time.instant
      steady_reps.times do
        mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback, reset_sequence: false,
          run_head: !skip_output_head)
      end
      gpu_ms = (Time.instant - t_steady).total_milliseconds
      measured_tokens = tokens * steady_reps
    else
      gpu_ms = mixed_stack.run_sequence(profile_phases: profile_phases, debug_readback: debug_readback,
        run_head: !skip_output_head)
    end
  end
  final_gpu_all = mixed_stack.final_gpu_all if debug_readback
  if path = gpu_final_dump_path
    File.open(path, "wb") do |io|
      final_gpu_all.each do |value|
        io.write_bytes(value, IO::ByteFormat::LittleEndian)
      end
    end
  end

  lines = [] of String
  ok = true
  if greedy_loop_tokens > 0
    if perf_only
      lines << "perf_only=true"
    elsif debug_readback
      lines << "final_all_check=skipped_for_greedy_loop"
    else
      lines << "debug_readback=false"
    end
    if greedy_loop_read_logits
      lines << "greedy_top1_logits_gpu=#{greedy_logits_top1_ids.join(",")}"
      lines << "greedy_top2_gpu=#{greedy_top2_ids.join(",")}"
      lines << "greedy_top1_values_gpu=#{greedy_top1_values.map { |v| v.round(6) }.join(",")}"
      lines << "greedy_top2_values_gpu=#{greedy_top2_values.map { |v| v.round(6) }.join(",")}"
      lines << "greedy_top_margin_gpu=#{greedy_top_margins.map { |v| v.round(6) }.join(",")}"
      lines << "greedy_logits_top1_mismatches=#{greedy_logits_mismatches.join(",")}"
    end
    if greedy_loop_probe_restore
      probe_route = if greedy_loop_probe_pca_updown_raw_q8_rest
                      "pca_updown_raw_q8_rest"
                    elsif greedy_loop_probe_pca_updown
                      "pca_updown"
                    else
                      "raw_q8"
                    end
      lines << "probe_route=#{probe_route}"
      lines << "probe_exact_top1_gpu=#{greedy_gpu_ids.join(",")}"
      lines << "probe_raw_top1_gpu=#{probe_raw_ids.join(",")}"
      lines << "probe_raw_top2_gpu=#{probe_raw_top2_ids.join(",")}"
      lines << "probe_raw_margin_gpu=#{probe_raw_margins.map { |v| v.round(6) }.join(",")}"
      lines << "probe_raw_ms=#{probe_raw_ms.round(3)}"
      lines << "probe_raw_ms_per_token=#{(probe_raw_ms / greedy_loop_tokens).round(3)}"
      lines << "probe_exact_ms=#{probe_exact_ms.round(3)}"
      lines << "probe_exact_ms_per_token=#{(probe_exact_ms / greedy_loop_tokens).round(3)}"
    end
    if greedy_loop_probe_chunk_gamma > 0
      full_accept_rate = chunk_count > 0 ? (100.0 * chunk_full_accepts / chunk_count) : 0.0
      token_accept_rate = chunk_raw_tokens > 0 ? (100.0 * chunk_accepted_tokens / chunk_raw_tokens) : 0.0
      lines << "chunk_probe_gamma=#{greedy_loop_probe_chunk_gamma}"
      lines << "chunk_probe_margin=#{greedy_loop_probe_chunk_margin.round(6)}"
      lines << "chunk_probe_count=#{chunk_count}"
      lines << "chunk_probe_full_accepts=#{chunk_full_accepts}"
      lines << "chunk_probe_rejects=#{chunk_rejects}"
      lines << "chunk_probe_margin_fallbacks=#{chunk_margin_fallbacks}"
      lines << "chunk_probe_raw_tokens=#{chunk_raw_tokens}"
      lines << "chunk_probe_verify_tokens=#{chunk_verify_tokens}"
      lines << "chunk_probe_accepted_tokens=#{chunk_accepted_tokens}"
      lines << "chunk_probe_full_accept_rate_pct=#{full_accept_rate.round(2)}"
      lines << "chunk_probe_token_accept_rate_pct=#{token_accept_rate.round(2)}"
      lines << "chunk_probe_raw_top1_gpu=#{chunk_proposal_ids.join(",")}"
      lines << "chunk_probe_raw_top2_gpu=#{chunk_raw_top2_ids.join(",")}"
      lines << "chunk_probe_raw_margin_gpu=#{chunk_raw_margins.map { |v| v.round(6) }.join(",")}"
      lines << "chunk_probe_exact_top1_gpu=#{chunk_exact_ids.join(",")}"
      lines << "chunk_probe_ngram=#{greedy_loop_probe_ngram}"
      if greedy_loop_probe_ngram
        lines << "chunk_probe_ngram_min=#{greedy_loop_probe_ngram_min}"
        lines << "chunk_probe_ngram_max=#{greedy_loop_probe_ngram_max}"
        lines << "chunk_probe_ngram_recursive=#{greedy_loop_probe_ngram_recursive}"
        lines << "chunk_probe_ngram_min_candidates=#{greedy_loop_probe_ngram_min_candidates}"
        lines << "chunk_probe_ngram_risk_gate=#{greedy_loop_probe_ngram_risk_gate}"
        lines << "chunk_probe_ngram_replay_start=#{greedy_loop_probe_ngram_replay_start}"
        lines << "chunk_probe_ngram_cursor_only=#{greedy_loop_probe_ngram_cursor_only}"
        lines << "chunk_probe_ngram_history_tokens=#{(greedy_loop_probe_ngram_history.empty? ? 1 : greedy_loop_probe_ngram_history.size)}"
        lines << "chunk_probe_ngram_match_lens=#{chunk_ngram_match_lens.join(",")}"
        lines << "chunk_probe_ngram_empty_fallbacks=#{chunk_ngram_empty_fallbacks}"
        lines << "chunk_probe_ngram_risk_fallbacks=#{chunk_ngram_risk_fallbacks}"
        lines << "chunk_probe_ngram_cursor_hits=#{chunk_ngram_cursor_hits}"
      end
      lines << "chunk_probe_raw_ms=#{chunk_raw_ms.round(3)}"
      lines << "chunk_probe_raw_ms_per_raw_token=#{(chunk_raw_ms / Math.max(chunk_raw_tokens, 1)).round(3)}"
      lines << "chunk_probe_verify_ms=#{chunk_verify_ms.round(3)}"
      lines << "chunk_probe_verify_ms_per_verify_token=#{(chunk_verify_ms / Math.max(chunk_verify_tokens, 1)).round(3)}"
      lines << "chunk_probe_batched_verify=#{greedy_loop_probe_chunk_batched_verify}"
      lines << "chunk_probe_batched_verify_chunks=#{chunk_batched_verify_chunks}"
      lines << "chunk_probe_batched_verify_accepts=#{chunk_batched_verify_accepts}"
      lines << "chunk_probe_batched_verify_rejects=#{chunk_batched_verify_rejects}"
      lines << "chunk_probe_batched_verify_last_reject_commits=#{chunk_batched_verify_last_reject_commits}"
      lines << "chunk_probe_batched_verify_ms=#{chunk_batched_verify_ms.round(3)}"
      lines << "chunk_probe_batched_verify_ms_per_chunk=#{(chunk_batched_verify_ms / Math.max(chunk_batched_verify_chunks, 1)).round(3)}"
    end
  elsif perf_only
    lines << "perf_only=true"
  elsif debug_readback
    ok = report_pair("final_all", final_gpu_all, cpu_current, lines, 1.0e-2_f32)
  else
    lines << "debug_readback=false"
  end
  gpu_top1_ids = skip_output_head ? [] of Int32 : (greedy_loop_tokens > 0 ? greedy_gpu_ids : output_head.top1_ids)
  unless known_replay_candidates.empty?
    accepted = 0
    reject_index = -1
    known_replay_candidates.each_with_index do |expected, i|
      if gpu_top1_ids[i]? == expected
        accepted += 1
      else
        reject_index = i
        break
      end
    end
    full_accept = accepted == known_replay_candidates.size
    commit_tokens = full_accept ? accepted : 0
    discarded_accept_prefix = full_accept ? 0 : accepted
    reject_recovery_required = !full_accept && accepted > 0
    recovery_rows = 0
    recovery_build_ms = 0.0
    recovery_weight_upload_ms = 0.0
    recovery_run_ms = 0.0
    recovery_top1_ids = [] of Int32
    recovery_prefix_match = false
    recovery_released_main_stack = false
    if known_replay_recover_on_reject && !full_accept
      recovery_rows = accepted + 1
      recovery_inputs = input_tokens[0, recovery_rows]
      recovery_xs = Array(Float32).new(recovery_rows * hidden, 0.0_f32)
      recovery_inputs.each_with_index do |tok, row|
        emb = ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, tok)
        hidden.times { |h| recovery_xs[row * hidden + h] = emb[h] }
      end

      # This diagnostic measures the short-stack recovery lower bound. Keeping
      # the full-span stack resident can OOM on small/fragmented CUDA hosts.
      mixed_stack.close
      recovery_released_main_stack = true

      t_recovery_build = Time.instant
      recovery_runners = [] of ML::CUDA::QwenMixedStackRunner::LayerRunner
      layers.each_with_index do |layer, idx|
        layer_input = idx == 0 ? recovery_xs : Array(Float32).new(recovery_rows * hidden, 0.0_f32)
        if hparams.full_attention?(layer)
          recovery_runners << ML::CUDA::QwenFullAttnLayerRunner.from_weights(full_weights[layer], recovery_rows, max_seq, start_pos,
            hparams.n_head, hparams.n_head_kv, hparams.head_dim, hparams.rope_dim_count, hparams.rms_eps,
            layer_input, cos_table, sin_table)
        else
          recovery_runners << ML::CUDA::QwenRecurrentLayerRunner.from_weights(recurrent_weights[layer], recovery_rows, layer_input,
            conv_state_inits[layer], ssm_state_inits[layer])
        end
      end
      recovery_head = ML::CUDA::QwenOutputHeadRunner.from_weights(head_weights, recovery_rows,
        Array(Float32).new(recovery_rows * hidden, 0.0_f32), hparams.rms_eps, read_logits: false, read_top2: false)
      known_replay_recovery_stack = ML::CUDA::QwenMixedStackRunner.new(layers, recovery_runners, recovery_head, recovery_rows, hidden, recovery_xs)
      recovery_build_ms = (Time.instant - t_recovery_build).total_milliseconds
      recovery_weight_upload_ms = known_replay_recovery_stack.not_nil!.upload_weights(profile: false)
      t_recovery_run = Time.instant
      known_replay_recovery_stack.not_nil!.run_sequence(profile_phases: false, debug_readback: false,
        reset_sequence: true, sync_end: true, read_head_outputs: true)
      recovery_run_ms = (Time.instant - t_recovery_run).total_milliseconds
      recovery_top1_ids = known_replay_recovery_stack.not_nil!.head.top1_ids
      recovery_prefix_match = recovery_top1_ids == gpu_top1_ids[0, recovery_rows]
    end
    lines << "known_replay_candidates=#{known_replay_candidates.join(",")}"
    lines << "known_replay_policy=full_accept_only"
    lines << "known_replay_commit_tokens=#{commit_tokens}"
    lines << "known_replay_discarded_accept_prefix=#{discarded_accept_prefix}"
    lines << "known_replay_reject_recovery_required=#{reject_recovery_required}"
    lines << "known_replay_recover_on_reject=#{known_replay_recover_on_reject}"
    lines << "known_replay_recovery_rows=#{recovery_rows}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_build_ms=#{recovery_build_ms.round(3)}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_weight_upload_ms=#{recovery_weight_upload_ms.round(3)}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_run_ms=#{recovery_run_ms.round(3)}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_run_ms_per_token=#{(recovery_run_ms / Math.max(recovery_rows, 1)).round(3)}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_top1=#{recovery_top1_ids.join(",")}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_prefix_match=#{recovery_prefix_match}" if known_replay_recover_on_reject
    lines << "known_replay_recovery_released_main_stack=#{recovery_released_main_stack}" if known_replay_recover_on_reject
    lines << "known_replay_accepted=#{accepted}"
    lines << "known_replay_total=#{known_replay_candidates.size}"
    lines << "known_replay_reject_index=#{reject_index}"
    lines << "known_replay_full_accept=#{full_accept}"
    lines << "known_replay_accept_rate_pct=#{(100.0 * accepted / Math.max(known_replay_candidates.size, 1)).round(2)}"
    unless known_replay_history.empty?
      lines << "known_replay_history_tokens=#{known_replay_history.size}"
      lines << "known_replay_start=#{known_replay_start}"
      lines << "known_replay_tokens=#{known_replay_tokens}"
    end
  end
  top1_ok = skip_output_head || perf_only || gpu_top1_ids == cpu_top1_ids
  if read_logits && greedy_loop_tokens == 0
    unless gpu_logits_only
      logits_ok = report_pair("logits", output_head.logits_gpu_all, cpu_logits_all, lines, 5.0e-3_f32)
      ok = ok && logits_ok
    end
    gpu_logits_top1_ids = [] of Int32
    gpu_top2_ids = [] of Int32
    gpu_top1_values = [] of Float32
    gpu_top2_values = [] of Float32
    gpu_top_margins = [] of Float32
    tokens.times do |tok|
      gpu_best_id, gpu_best, gpu_second_id, gpu_second, gpu_margin =
        top2_from_logits(output_head.logits_gpu_all, tok, head_weights.vocab)
      gpu_logits_top1_ids << gpu_best_id
      gpu_top2_ids << gpu_second_id
      gpu_top1_values << gpu_best
      gpu_top2_values << gpu_second
      gpu_top_margins << gpu_margin
      if gpu_best_id != gpu_top1_ids[tok]
        lines << "logits_top1_scan_mismatch_tok#{tok}=#{gpu_best_id}:#{gpu_top1_ids[tok]}"
        ok = false
      end
    end
    cuda_top_margins = Array(Float32).new(tokens) do |tok|
      output_head.top1_values_gpu[tok] - output_head.top2_values_gpu[tok]
    end
    cuda_top2_ok = output_head.top1_ids == gpu_top1_ids &&
                   output_head.top2_ids_gpu == gpu_top2_ids &&
                   max_abs_diff(cuda_top_margins, gpu_top_margins) <= 5.0e-3_f32
    if gpu_logits_only
      gpu_top1_ids = gpu_logits_top1_ids
      top1_ok = true
    else
      top1_ok = gpu_logits_top1_ids == cpu_top1_ids
    end
    lines << "top1_logits_gpu=#{gpu_logits_top1_ids.join(",")}"
    lines << "top1_cuda_gpu=#{output_head.top1_ids.join(",")}"
    lines << "top2_gpu=#{gpu_top2_ids.join(",")}"
    lines << "top2_cpu=#{gpu_logits_only ? "skipped" : cpu_top2_ids.join(",")}"
    lines << "top2_cuda_gpu=#{output_head.top2_ids_gpu.join(",")}"
    lines << "top1_values_logits_gpu=#{gpu_top1_values.map { |v| v.round(6) }.join(",")}"
    lines << "top1_values_cpu=#{gpu_logits_only ? "skipped" : cpu_top1_values.map { |v| v.round(6) }.join(",")}"
    lines << "top2_values_gpu=#{gpu_top2_values.map { |v| v.round(6) }.join(",")}"
    lines << "top2_values_cpu=#{gpu_logits_only ? "skipped" : cpu_top2_values.map { |v| v.round(6) }.join(",")}"
    lines << "top2_values_cuda_gpu=#{output_head.top2_values_gpu.map { |v| v.round(6) }.join(",")}"
    lines << "top_margin_gpu=#{gpu_top_margins.map { |v| v.round(6) }.join(",")}"
    lines << "top_margin_cpu=#{gpu_logits_only ? "skipped" : cpu_top_margins.map { |v| v.round(6) }.join(",")}"
    lines << "top_margin_cuda_gpu=#{cuda_top_margins.map { |v| v.round(6) }.join(",")}"
    unless gpu_logits_only
      margin_max_diff = max_abs_diff(gpu_top_margins, cpu_top_margins)
      margin_ok = margin_max_diff <= 5.0e-3_f32
      lines << "top_margin_max_diff=#{margin_max_diff}"
      lines << "top_margin_ok=#{margin_ok}"
      ok = ok && margin_ok
    end
    lines << "top2_cuda_ok=#{cuda_top2_ok}"
    append_margin_bucket_report(lines, cpu_top1_ids, gpu_top1_ids, cpu_top_margins, margin_bucket_edges) if margin_bucket_report
    ok = ok && cuda_top2_ok unless gpu_logits_only
  else
    lines << "logits_readback=false"
  end
  if path = gpu_final_dump_path
    lines << "gpu_final_dump=#{path}"
    lines << "gpu_final_count=#{final_gpu_all.size}"
  end
  lines << "top1_gpu=#{gpu_top1_ids.join(",")}"
  lines << "top1_cpu=#{perf_only ? "skipped" : cpu_top1_ids.join(",")}"
  lines << "top1_values_gpu=#{output_head.top1_values_gpu.map { |v| v.round(6) }.join(",")}"
  lines << "top1_ok=#{perf_only ? "skipped" : top1_ok}"
  ok = ok && top1_ok
  if debug_readback && !perf_only && greedy_loop_tokens == 0
    runners.each_with_index do |runner, idx|
      layer = layers[idx]
      case runner
      in ML::CUDA::QwenRecurrentLayerRunner
        conv_ok = report_pair("layer#{layer}_conv_state", runner.conv_state_gpu, cpu_states[layer].conv_state.not_nil!, lines, 2.0e-5_f32)
        ssm_ok = report_pair("layer#{layer}_ssm_state", runner.ssm_state_gpu, cpu_states[layer].ssm_state.not_nil!, lines, 1.0e-3_f32)
        ok = ok && conv_ok && ssm_ok
      in ML::CUDA::QwenFullAttnLayerRunner
        kv = runner.kv
        k_cpu = cpu_states[layer].k_cache || Array(Float32).new(max_seq * hparams.n_head_kv * hparams.head_dim, 0.0_f32)
        v_cpu = cpu_states[layer].v_cache || Array(Float32).new(max_seq * hparams.n_head_kv * hparams.head_dim, 0.0_f32)
        k_ok = report_pair("layer#{layer}_k_cache", kv.k_cache_gpu, k_cpu, lines, 2.0e-4_f32)
        v_ok = report_pair("layer#{layer}_v_cache", kv.v_cache_gpu, v_cpu, lines, 1.0e-3_f32)
        ok = ok && k_ok && v_ok
      end
    end
  end

  puts "device=#{cuda_ctx.device_name}"
  puts "compute_capability=#{cuda_ctx.compute_capability_major}.#{cuda_ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layers=#{layers.join(",")}"
  puts "tokens=#{tokens}"
  puts "start_pos=#{start_pos}"
  puts "max_seq=#{max_seq}"
  puts "warmup=#{warmup}"
  puts "steady_reps=#{steady_reps}"
  puts "steady_graph_reps=#{steady_graph_reps}"
  puts "greedy_loop_tokens=#{greedy_loop_tokens}"
  puts "greedy_loop_graph=#{greedy_loop_graph}"
  puts "greedy_loop_no_graph=#{greedy_loop_no_graph}"
  puts "greedy_loop_gpu_embedding=#{greedy_loop_gpu_embedding}"
  puts "greedy_loop_cpu_embedding=#{greedy_loop_cpu_embedding}"
  puts "greedy_loop_read_logits=#{greedy_loop_read_logits}"
  puts "greedy_loop_probe_restore=#{greedy_loop_probe_restore}"
  puts "greedy_loop_probe_restore_kv=#{greedy_loop_probe_restore_kv}"
  puts "greedy_loop_probe_pca_updown=#{greedy_loop_probe_pca_updown}"
  puts "greedy_loop_probe_pca_updown_raw_q8_rest=#{greedy_loop_probe_pca_updown_raw_q8_rest}"
  puts "greedy_loop_probe_chunk_gamma=#{greedy_loop_probe_chunk_gamma}"
  puts "greedy_loop_probe_chunk_margin=#{greedy_loop_probe_chunk_margin}"
  puts "greedy_loop_probe_chunk_batched_verify=#{greedy_loop_probe_chunk_batched_verify}"
  puts "greedy_loop_probe_chunk_fast_verify_top1=#{greedy_loop_probe_chunk_fast_verify_top1}"
  puts "greedy_loop_probe_ngram=#{greedy_loop_probe_ngram}"
  puts "greedy_loop_probe_ngram_min=#{greedy_loop_probe_ngram_min}"
  puts "greedy_loop_probe_ngram_max=#{greedy_loop_probe_ngram_max}"
  puts "greedy_loop_probe_ngram_recursive=#{greedy_loop_probe_ngram_recursive}"
  puts "greedy_loop_probe_ngram_min_candidates=#{greedy_loop_probe_ngram_min_candidates}"
  puts "greedy_loop_probe_ngram_risk_gate=#{greedy_loop_probe_ngram_risk_gate}"
  puts "greedy_loop_probe_ngram_history=#{greedy_loop_probe_ngram_history.join(",")}"
  puts "greedy_loop_probe_ngram_replay_start=#{greedy_loop_probe_ngram_replay_start}"
  puts "greedy_loop_probe_ngram_cursor_only=#{greedy_loop_probe_ngram_cursor_only}"
  puts "greedy_loop_prefix_tokens=#{greedy_loop_prefix_tokens.join(",")}"
  puts "seed_token=#{seed_token}"
  puts "input_token=#{input_token}"
  puts "input_tokens=#{input_tokens.join(",")}"
  puts "known_replay_candidates_arg=#{known_replay_candidates.join(",")}"
  puts "known_replay_history_tokens=#{known_replay_history.size}"
  puts "known_replay_start_arg=#{known_replay_start}"
  puts "known_replay_tokens_arg=#{known_replay_tokens}"
  puts "known_replay_recover_on_reject_arg=#{known_replay_recover_on_reject}"
  puts "read_logits=#{read_logits}"
  puts "read_top2=#{read_top2}"
  puts "gpu_logits_only=#{gpu_logits_only}"
  puts "margin_bucket_report=#{margin_bucket_report}"
  puts "profile_phases=#{profile_phases}"
  puts "debug_readback=#{debug_readback}"
  puts "perf_only=#{perf_only}"
  puts "skip_output_head=#{skip_output_head}"
  puts "runtime_raw_q8=#{runtime_raw_q8}"
  puts "runtime_skip_recurrent_ffn=#{runtime_skip_recurrent_ffn}"
  puts "runtime_skip_recurrent_ffn_layers=#{runtime_skip_recurrent_ffn_layers.try(&.join(",")) || ""}"
  puts "runtime_pca_updown_zero=#{runtime_pca_updown_zero}"
  puts "runtime_pca_updown_rank=#{runtime_pca_updown_rank}"
  puts "runtime_pca_updown_layers=#{runtime_pca_updown_layers.try(&.join(",")) || ""}"
  puts "runtime_pca_updown_adapters=#{runtime_pca_updown_adapters_path || ""}"
  puts "q4_raw_q8_ffn=#{ENV["QWEN_CUDA_Q4_RAW_Q8_FFN"]? == "1"}"
  puts "batched_ffn=#{ENV["QWEN_CUDA_BATCHED_FFN_OFF"]? != "1"}"
  puts "batched_projections=#{ENV["QWEN_CUDA_BATCHED_PROJECTIONS_OFF"]? != "1"}"
  puts "batched_ssm_out=#{ENV["QWEN_CUDA_BATCHED_SSM_OUT_OFF"]? != "1"}"
  puts "batched_norms=#{ENV["QWEN_CUDA_BATCHED_NORMS_OFF"]? != "1"}"
  puts "batched_alpha_beta_transform=#{ENV["QWEN_CUDA_BATCHED_ALPHA_BETA_TRANSFORM"]? == "1"}"
  puts "q5_tbatch4=#{ENV["QWEN_CUDA_Q5_TBATCH4_OFF"]? != "1"}"
  puts "q4_gate_tbatch4=#{ENV["QWEN_CUDA_Q4_GATE_TBATCH4_OFF"]? != "1"}"
  puts "q4_ssm_out_tbatch4=#{ENV["QWEN_CUDA_Q4_SSM_OUT_TBATCH4_OFF"]? != "1"}"
  puts "q4_down_add_tbatch4=#{ENV["QWEN_CUDA_Q4_DOWN_ADD_TBATCH4_OFF"]? != "1"}"
  puts "q4_tbatch4=#{ENV["QWEN_CUDA_Q4_TBATCH4_OFF"]? != "1"}"
  puts "q6_tbatch4=#{ENV["QWEN_CUDA_Q6_TBATCH4_OFF"]? != "1"}"
  puts "head_q6_tbatch4=#{ENV["QWEN_CUDA_HEAD_Q6_TBATCH4_OFF"]? != "1"}"
  puts "head_top2_batched=#{ENV["QWEN_CUDA_HEAD_TOP2_BATCHED_OFF"]? != "1"}"
  puts "full_attn_qkv_tbatch4=#{ENV["QWEN_CUDA_FULL_ATTN_QKV_TBATCH4_OFF"]? != "1"}"
  puts "full_attn_output_tbatch4=#{ENV["QWEN_CUDA_FULL_ATTN_OUTPUT_TBATCH4_OFF"]? != "1"}"
  puts "full_attn_batched_norms=#{ENV["QWEN_CUDA_FULL_ATTN_BATCHED_NORMS_OFF"]? != "1"}"
  puts "hidden=#{hidden}"
  puts "vocab=#{head_weights.vocab}"
  puts "weight_upload_ms=#{weight_upload_ms.round(3)}"
  puts "verifier_weight_upload_ms=#{verifier_weight_upload_ms.round(3)}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cuda_ms_per_token=#{(gpu_ms / measured_tokens).round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cpu_ms_per_token=#{(cpu_ms / (greedy_loop_tokens > 0 ? greedy_loop_tokens : tokens)).round(3)}"
  if greedy_loop_tokens > 0
    denom = greedy_loop_tokens.to_f64
    puts "greedy_prefix_tokens=#{greedy_loop_prefix_tokens.size}"
    puts "greedy_prefix_ms=#{greedy_prefix_ms.round(3)}"
    puts "greedy_position_ms=#{greedy_position_ms.round(3)}"
    puts "greedy_position_ms_per_token=#{(greedy_position_ms / denom).round(3)}"
    puts "greedy_embedding_ms=#{greedy_embedding_ms.round(3)}"
    puts "greedy_embedding_ms_per_token=#{(greedy_embedding_ms / denom).round(3)}"
    puts "greedy_body_ms=#{greedy_body_ms.round(3)}"
    puts "greedy_body_ms_per_token=#{(greedy_body_ms / denom).round(3)}"
    puts "greedy_read_ms=#{greedy_read_ms.round(3)}"
    puts "greedy_read_ms_per_token=#{(greedy_read_ms / denom).round(3)}"
  end
  mixed_stack.phase_lines.each { |line| puts line }
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  known_replay_recovery_stack.try(&.close)
  stack.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
