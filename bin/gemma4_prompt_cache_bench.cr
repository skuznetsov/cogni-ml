require "file_utils"
require "option_parser"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/gemma4_prompt_cache"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
prompt_len = 256
tokens_arg = nil.as(String?)
max_seq = 0
prefill_chunk = 512
warmups = 1
runs = 3
cache_root = nil.as(String?)
keep_cache = false
snapshot_cache_mib = 0
snapshot_cache_min_free_mib = (ENV["GEMMA4_PROMPT_CACHE_SNAPSHOT_MIN_FREE_MIB"]? || "4096").to_i
snapshot_cache_entries = 1

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_prompt_cache_bench [--prompt-len 256] [--runs 3]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt token count") { |v| prompt_len = v.to_i }
  p.on("--tokens IDS", "Comma-separated prompt token ids; overrides --prompt-len") { |v| tokens_arg = v }
  p.on("--max-seq N", "KV cache sequence capacity; default prompt_len+8") { |v| max_seq = v.to_i }
  p.on("--prefill-chunk N", "Row prefill chunk size") { |v| prefill_chunk = v.to_i }
  p.on("--warmups N", "Warmup iterations for each measured route") { |v| warmups = v.to_i }
  p.on("--runs N", "Measured iterations for each route") { |v| runs = v.to_i }
  p.on("--cache-root PATH", "Prompt-cache root; default temp dir") { |v| cache_root = v }
  p.on("--snapshot-cache-mib N", "Enable Store resident snapshot cache with this byte budget in MiB") { |v| snapshot_cache_mib = v.to_i }
  p.on("--snapshot-cache-min-free-mib N", "Clamp snapshot cache to leave at least this much available memory; default env GEMMA4_PROMPT_CACHE_SNAPSHOT_MIN_FREE_MIB or 4096") { |v| snapshot_cache_min_free_mib = v.to_i }
  p.on("--snapshot-cache-entries N", "Resident snapshot cache entry limit") { |v| snapshot_cache_entries = v.to_i }
  p.on("--keep-cache", "Do not remove the temp cache root") { keep_cache = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "prompt_len must be positive" unless prompt_len > 0
raise "prefill_chunk must be positive" unless prefill_chunk > 0
raise "runs must be positive" unless runs > 0
raise "warmups must be non-negative" unless warmups >= 0
raise "snapshot_cache_mib must be non-negative" unless snapshot_cache_mib >= 0
raise "snapshot_cache_min_free_mib must be non-negative" unless snapshot_cache_min_free_mib >= 0
raise "snapshot_cache_entries must be non-negative" unless snapshot_cache_entries >= 0

prompt = if arg = tokens_arg
           arg.split(',').reject(&.empty?).map(&.to_i32)
         else
           Array(Int32).new(prompt_len) { |i| (42 + i).to_i32 }
         end
raise "prompt tokens must not be empty" if prompt.empty?
max_seq = prompt.size + 8 if max_seq <= 0
raise "max_seq too small" if max_seq < prompt.size

def percentile(sorted : Array(Float64), p : Float64) : Float64
  return 0.0 if sorted.empty?

  idx = ((sorted.size - 1).to_f64 * p).round.to_i
  sorted[idx]
end

def summarize(label : String, samples : Array(Float64), token_count : Int32) : Float64
  sorted = samples.sort
  mean = samples.sum / samples.size
  p50 = percentile(sorted, 0.50)
  p90 = percentile(sorted, 0.90)
  tok_s = token_count.to_f64 / (p50 / 1000.0)
  puts "#{label}_runs=#{samples.map { |v| v.round(3) }.join(',')}"
  puts "#{label}_mean_ms=#{mean.round(3)} #{label}_p50_ms=#{p50.round(3)} #{label}_p90_ms=#{p90.round(3)} #{label}_p50_effective_tok_s=#{tok_s.round(3)}"
  p50
end

def prefill_once(weights : ML::GGUF::Gemma4Weights,
                 prompt : Array(Int32),
                 max_seq : Int32,
                 prefill_chunk : Int32) : NamedTuple(ms: Float64, state: ML::GGUF::Gemma4Metal::ResidentState)
  state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
  t0 = Time.instant
  ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(
    weights,
    prompt,
    0,
    state,
    chunk_size: prefill_chunk,
    stop_layer: weights.hparams.n_layer,
  ).not_nil!
  {ms: (Time.instant - t0).total_milliseconds, state: state}
end

started = Time.instant
weights = ML::GGUF::Gemma4Weights.from_gguf(model)
load_ms = (Time.instant - started).total_milliseconds
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

root = (cache_root || File.tempname("gemma4-prompt-cache-bench")).not_nil!
FileUtils.mkdir_p(root)

begin
  store = ML::GGUF::Gemma4PromptCache::Store.new(root)
  seed = prefill_once(weights, prompt, max_seq, prefill_chunk)
  save_t0 = Time.instant
  entry = store.save_resident_state(
    seed[:state],
    prompt,
    model_id: File.basename(model),
    tokenizer_id: "synthetic-token-ids",
    prompt_text: "",
    session_id: "bench",
  )
  save_ms = (Time.instant - save_t0).total_milliseconds
  hit = store.lookup_prompt(File.basename(model), "synthetic-token-ids", "", prompt).not_nil!

  # Warm both restore paths before measuring. Artifact restore validates and
  # rereads the durable file; snapshot restore is the hot in-memory lower bound.
  warmups.times do
    target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
    store.restore(hit, reuse_state: target)
  end

  cached_store = ML::GGUF::Gemma4PromptCache::Store.new(
    root,
    snapshot_cache_byte_limit: snapshot_cache_mib.to_i64 * 1024_i64 * 1024_i64,
    snapshot_cache_min_free_bytes: snapshot_cache_min_free_mib.to_i64 * 1024_i64 * 1024_i64,
    snapshot_cache_entry_limit: snapshot_cache_entries,
  )
  cached_hit = cached_store.lookup_prompt(File.basename(model), "synthetic-token-ids", "", prompt)
  cached_prime_ms = nil.as(Float64?)
  if cached_store.snapshot_cache_enabled? && cached_hit
    target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
    cached_t0 = Time.instant
    cached_store.restore(cached_hit, reuse_state: target)
    cached_prime_ms = (Time.instant - cached_t0).total_milliseconds
  end

  snapshot = ML::GGUF::Gemma4StateSnapshot.read_artifact(hit.artifact_path, expected_sha256: hit.artifact_sha256)
  warmups.times do
    target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, target)
  end
  if cached_store.snapshot_cache_enabled? && (cached = cached_hit)
    warmups.times do
      target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
      cached_store.restore(cached, reuse_state: target)
    end
  end

  warmups.times { prefill_once(weights, prompt, max_seq, prefill_chunk) }

  prefill_samples = [] of Float64
  artifact_restore_samples = [] of Float64
  cached_store_restore_samples = [] of Float64
  snapshot_restore_samples = [] of Float64

  runs.times do
    prefill_samples << prefill_once(weights, prompt, max_seq, prefill_chunk)[:ms]

    artifact_target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
    artifact_t0 = Time.instant
    store.restore(hit, reuse_state: artifact_target)
    artifact_restore_samples << (Time.instant - artifact_t0).total_milliseconds

    if cached_store.snapshot_cache_enabled? && (cached = cached_hit)
      cached_target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
      cached_t0 = Time.instant
      cached_store.restore(cached, reuse_state: cached_target)
      cached_store_restore_samples << (Time.instant - cached_t0).total_milliseconds
    end

    snapshot_target = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)
    snapshot_t0 = Time.instant
    ML::GGUF::Gemma4StateSnapshot.restore_into(snapshot, snapshot_target)
    snapshot_restore_samples << (Time.instant - snapshot_t0).total_milliseconds
  end

  puts "model=#{File.basename(model)} prompt_len=#{prompt.size} max_seq=#{max_seq} prefill_chunk=#{prefill_chunk} warmups=#{warmups} runs=#{runs} load_ms=#{load_ms.round(3)}"
  puts "cache_root=#{root} artifact_path=#{entry.artifact_path} artifact_bytes=#{entry.artifact_byte_size} state_bytes=#{entry.state_byte_size} save_ms=#{save_ms.round(3)}"
  puts "snapshot_cache_mib=#{snapshot_cache_mib} snapshot_cache_min_free_mib=#{snapshot_cache_min_free_mib} snapshot_cache_entries=#{snapshot_cache_entries} snapshot_cache_enabled=#{cached_store.snapshot_cache_enabled?} snapshot_cache_requested_bytes=#{cached_store.snapshot_cache_requested_byte_limit} snapshot_cache_effective_byte_limit=#{cached_store.snapshot_cache_byte_limit} cached_prime_ms=#{cached_prime_ms.try(&.round(3)) || "disabled"} snapshot_cache_bytes=#{cached_store.snapshot_cache_bytes} snapshot_cache_hits=#{cached_store.snapshot_cache_hits} snapshot_cache_misses=#{cached_store.snapshot_cache_misses}"
  prefill_p50 = summarize("cold_prefill", prefill_samples, prompt.size)
  artifact_p50 = summarize("artifact_restore", artifact_restore_samples, prompt.size)
  cached_store_p50 = cached_store_restore_samples.empty? ? nil : summarize("cached_store_restore", cached_store_restore_samples, prompt.size)
  snapshot_p50 = summarize("snapshot_restore", snapshot_restore_samples, prompt.size)
  puts "artifact_restore_speedup_vs_cold=#{(prefill_p50 / artifact_p50).round(4)}"
  if cached = cached_store_p50
    puts "cached_store_restore_speedup_vs_cold=#{(prefill_p50 / cached).round(4)}"
    puts "cached_store_restore_speedup_vs_artifact=#{(artifact_p50 / cached).round(4)}"
  end
  puts "snapshot_restore_speedup_vs_cold=#{(prefill_p50 / snapshot_p50).round(4)}"
ensure
  FileUtils.rm_rf(root) if !keep_cache && cache_root.nil? && File.exists?(root)
end
