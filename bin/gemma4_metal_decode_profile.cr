require "option_parser"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/gemma4_prompt_cache"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
prompt = [42, 43, 44, 45, 46, 47, 48, 49]
generate = 8
max_seq = 1024
warmups = 1
runs = 3
profile = false
profile_decode_only = false
decode_mode = "top1"
decode_wave = false
prefill_mode = "serial"
prefill_chunk = 8
prefill_head = true
stop_layer = nil.as(Int32?)
prompt_cache_root = nil.as(String?)
prompt_cache_session = "profile"
prompt_cache_snapshot_mib = 0
prompt_cache_snapshot_entries = 1

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_metal_decode_profile [--tokens 42,43] [--generate 8] [--max-seq 1024] [--runs 3]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids") { |v| prompt = v.split(',').reject(&.empty?).map(&.to_i) }
  p.on("--generate N", "Measured generated tokens per run") { |v| generate = v.to_i }
  p.on("--max-seq N", "KV cache sequence capacity") { |v| max_seq = v.to_i }
  p.on("--warmups N", "Warmup runs") { |v| warmups = v.to_i }
  p.on("--runs N", "Measured runs") { |v| runs = v.to_i }
  p.on("--profile", "Print shared Metal matmul/profile attribution for the final measured run") { profile = true }
  p.on("--profile-decode-only", "Reset profile counters after prefill and report only generated-token work") { profile_decode_only = true }
  p.on("--decode-wave", "Use one command buffer per decode token instead of one wait per layer") { decode_wave = true }
  p.on("--prefill-mode MODE", "Prompt prefill mode: serial or rows (default: serial)") { |v| prefill_mode = v }
  p.on("--prefill-chunk N", "Row prefill chunk size; exact path clamps above 8; GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 defaults cap to 512") { |v| prefill_chunk = v.to_i }
  p.on("--prefill-no-head", "Measure pure prompt body only; requires --body-only because no next-token seed is computed") { prefill_head = false }
  p.on("--stop-layer N", "Run only the first N layers for attribution (default: all)") { |v| stop_layer = v.to_i }
  p.on("--prompt-cache-root PATH", "Enable exact Gemma prompt-cache save/restore at this root") { |v| prompt_cache_root = v }
  p.on("--prompt-cache-session ID", "Prompt-cache session id") { |v| prompt_cache_session = v }
  p.on("--prompt-cache-snapshot-mib N", "Enable Store resident snapshot cache with this MiB budget") { |v| prompt_cache_snapshot_mib = v.to_i }
  p.on("--prompt-cache-snapshot-entries N", "Store resident snapshot cache entry limit") { |v| prompt_cache_snapshot_entries = v.to_i }
  p.on("--body-only", "During measured generation, update resident state from fixed synthetic tokens without final logits/top1") { decode_mode = "body" }
  p.on("--top1", "During measured generation, run exact greedy top1 chain (default)") { decode_mode = "top1" }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "prompt tokens must not be empty" if prompt.empty?
raise "generate must be positive" unless generate > 0
raise "runs must be positive" unless runs > 0
raise "max-seq too small" if max_seq < prompt.size + generate
raise "prompt-cache snapshot MiB must be non-negative" unless prompt_cache_snapshot_mib >= 0
raise "prompt-cache snapshot entries must be non-negative" unless prompt_cache_snapshot_entries >= 0

def percentile(sorted : Array(Float64), p : Float64) : Float64
  return 0.0 if sorted.empty?
  idx = ((sorted.size - 1).to_f64 * p).round.to_i
  sorted[idx]
end

def summarize(label : String, samples : Array(Float64), tokens : Int32) : Nil
  sorted = samples.sort
  mean = samples.sum / samples.size
  p50 = percentile(sorted, 0.50)
  p90 = percentile(sorted, 0.90)
  tok_s = tokens.to_f64 / (p50 / 1000.0)
  puts "#{label}_runs=#{samples.map { |v| v.round(3) }.join(',')}"
  puts "#{label}_mean_ms=#{mean.round(3)} #{label}_p50_ms=#{p50.round(3)} #{label}_p90_ms=#{p90.round(3)} #{label}_p50_tok_s=#{tok_s.round(3)}"
end

def top1_id(logits : Array(Float32)) : Int32
  best_id = 0
  best = logits[0]
  logits.each_with_index do |v, i|
    if v > best
      best = v
      best_id = i
    end
  end
  best_id.to_i32
end

def forward_top1(weights : ML::GGUF::Gemma4Weights, token_id : Int32, pos : Int32,
                 state : ML::GGUF::Gemma4Metal::ResidentState,
                 decode_wave : Bool = false) : Int32
  hidden = if decode_wave
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(weights, token_id, pos, state, weights.hparams.n_layer).not_nil!
           else
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos, state, weights.hparams.n_layer).not_nil!
           end
  logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
  top1_id(logits)
end

def synthetic_decode_token(i : Int32) : Int32
  ((i * 13 + 11751) % 32000).to_i32
end

def prefill_prompt_hidden(weights : ML::GGUF::Gemma4Weights,
                          prompt : Array(Int32),
                          state : ML::GGUF::Gemma4Metal::ResidentState,
                          prefill_mode : String,
                          prefill_chunk : Int32,
                          stop_layer : Int32?) : Array(Float32)
  case prefill_mode
  when "serial"
    last = [] of Float32
    prompt.each_with_index do |token_id, pos|
      last = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos.to_i32, state, stop_layer || weights.hparams.n_layer).not_nil!
    end
    last
  when "rows"
    ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(weights, prompt, 0, state, chunk_size: prefill_chunk, stop_layer: stop_layer || weights.hparams.n_layer).not_nil!
  else
    raise "prefill mode must be serial or rows"
  end
end

def run_once(weights : ML::GGUF::Gemma4Weights, prompt : Array(Int32), generate : Int32,
             max_seq : Int32, decode_mode : String, prefill_mode : String, prefill_chunk : Int32,
             prefill_head : Bool = true,
             stop_layer : Int32? = nil,
             profile : Bool = false,
             profile_decode_only : Bool = false,
             decode_wave : Bool = false,
             prompt_cache_store : ML::GGUF::Gemma4PromptCache::Store? = nil,
             prompt_cache_model_id : String = "",
             prompt_cache_tokenizer_id : String = "synthetic-token-ids",
             prompt_cache_session_id : String = "profile") : NamedTuple(prefill_ms: Float64, decode_ms: Float64, first_id: Int32, last_id: Int32, cache_route: String, cache_restore_ms: Float64)
  state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)

  ML::GGUF::Qwen35Metal::Profile.reset if profile
  ML::GGUF::Qwen35Metal::Profile.enable! if profile

  prefill_t0 = Time.instant
  hidden = [] of Float32
  next_id = -1_i32
  cache_route = prompt_cache_store ? "miss" : "none"
  cache_restore_ms = 0.0
  save_cache_after_head = false
  if store = prompt_cache_store
    if hit = store.lookup_prompt(prompt_cache_model_id, prompt_cache_tokenizer_id, "", prompt)
      restore_t0 = Time.instant
      store.restore(hit, reuse_state: state)
      cache_restore_ms = (Time.instant - restore_t0).total_milliseconds
      if prefill_head && (cached_next = hit.next_token_id)
        next_id = cached_next
        cache_route = "hit_next_id"
      elsif prefill_head
        # Older cache entries store authoritative K/V rows, not final prompt
        # logits. Replaying the last prompt token recovers exact logits while
        # rewriting the same K/V row.
        hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(
          weights,
          prompt.last,
          (prompt.size - 1).to_i32,
          state,
          stop_layer || weights.hparams.n_layer
        ).not_nil!
        cache_route = "hit_replay_last"
      else
        cache_route = "hit_restore_only"
      end
    end
  end

  if cache_route == "miss"
    hidden = prefill_prompt_hidden(weights, prompt, state, prefill_mode, prefill_chunk, stop_layer)
    save_cache_after_head = true
    cache_route = "miss_save"
  elsif cache_route == "none"
    hidden = prefill_prompt_hidden(weights, prompt, state, prefill_mode, prefill_chunk, stop_layer)
  end
  if prefill_head && next_id < 0
    logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
    next_id = top1_id(logits)
  end
  if save_cache_after_head
    prompt_cache_store.not_nil!.save_resident_state(
      state,
      prompt,
      model_id: prompt_cache_model_id,
      tokenizer_id: prompt_cache_tokenizer_id,
      prompt_text: "",
      session_id: prompt_cache_session_id,
      next_token_id: prefill_head ? next_id : nil,
    )
  end
  prefill_ms = (Time.instant - prefill_t0).total_milliseconds

  if profile && profile_decode_only
    ML::GGUF::Qwen35Metal::Profile.reset
    ML::GGUF::Qwen35Metal::Profile.enable!
  end

  decode_t0 = Time.instant
  cur = prefill_head ? next_id : synthetic_decode_token(0)
  generate.times do |i|
    pos = (prompt.size + i).to_i32
    if decode_mode == "body"
      cur = synthetic_decode_token(i)
      if decode_wave
        ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(weights, cur, pos, state, stop_layer || weights.hparams.n_layer).not_nil!
      else
        ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, cur, pos, state, stop_layer || weights.hparams.n_layer).not_nil!
      end
    else
      cur = forward_top1(weights, cur, pos, state, decode_wave)
    end
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  if profile
    ML::GGUF::Qwen35Metal::Profile.disable!
    puts ML::GGUF::Qwen35Metal::Profile.report_io
  end

  {prefill_ms: prefill_ms, decode_ms: decode_ms, first_id: next_id, last_id: cur, cache_route: cache_route, cache_restore_ms: cache_restore_ms}
end

started = Time.instant
weights = ML::GGUF::Gemma4Weights.from_gguf(model)
load_ms = (Time.instant - started).total_milliseconds
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

raise "decode mode must be top1 or body" unless {"top1", "body"}.includes?(decode_mode)
raise "prefill mode must be serial or rows" unless {"serial", "rows"}.includes?(prefill_mode)
raise "prefill chunk must be positive" unless prefill_chunk > 0
raise "--prefill-no-head requires --body-only" if !prefill_head && decode_mode != "body"
if sl = stop_layer
  raise "--stop-layer must be non-negative" if sl < 0
  raise "--stop-layer exceeds model layer count" if sl > weights.hparams.n_layer
end
if prompt_cache_root && stop_layer && stop_layer != weights.hparams.n_layer
  raise "--prompt-cache-root currently requires full-layer prefill/decode"
end

cache_store = nil.as(ML::GGUF::Gemma4PromptCache::Store?)
cache_model_id = File.basename(model)
cache_tokenizer_id = "synthetic-token-ids"
if root = prompt_cache_root
  cache_store = ML::GGUF::Gemma4PromptCache::Store.new(
    root,
    snapshot_cache_byte_limit: prompt_cache_snapshot_mib.to_i64 * 1024_i64 * 1024_i64,
    snapshot_cache_entry_limit: prompt_cache_snapshot_entries,
  )
end

puts "model=#{File.basename(model)} prompt_tokens=#{prompt.join(',')} prompt_len=#{prompt.size} generate=#{generate} max_seq=#{max_seq} warmups=#{warmups} runs=#{runs} mode=#{decode_mode} decode_wave=#{decode_wave} prefill_mode=#{prefill_mode} prefill_chunk=#{prefill_chunk} prefill_head=#{prefill_head} stop_layer=#{stop_layer || weights.hparams.n_layer} profile=#{profile} profile_decode_only=#{profile_decode_only} prompt_cache_enabled=#{!cache_store.nil?} prompt_cache_root=#{prompt_cache_root || ""} prompt_cache_snapshot_mib=#{prompt_cache_snapshot_mib} prompt_cache_snapshot_entries=#{prompt_cache_snapshot_entries} load_ms=#{load_ms.round(3)}"

warmups.times do
  run_once(
    weights, prompt, generate, max_seq, decode_mode, prefill_mode, prefill_chunk, prefill_head, stop_layer,
    profile_decode_only: false,
    decode_wave: decode_wave,
    prompt_cache_store: cache_store,
    prompt_cache_model_id: cache_model_id,
    prompt_cache_tokenizer_id: cache_tokenizer_id,
    prompt_cache_session_id: prompt_cache_session,
  )
end

prefill_samples = [] of Float64
decode_samples = [] of Float64
cache_restore_samples = [] of Float64
cache_route_counts = Hash(String, Int32).new(0)
first_id = 0_i32
last_id = 0_i32
runs.times do
  result = run_once(
    weights, prompt, generate, max_seq, decode_mode, prefill_mode, prefill_chunk, prefill_head, stop_layer, profile && runs == 1,
    profile_decode_only: profile_decode_only,
    decode_wave: decode_wave,
    prompt_cache_store: cache_store,
    prompt_cache_model_id: cache_model_id,
    prompt_cache_tokenizer_id: cache_tokenizer_id,
    prompt_cache_session_id: prompt_cache_session,
  )
  prefill_samples << result[:prefill_ms]
  decode_samples << result[:decode_ms]
  cache_route_counts[result[:cache_route]] += 1
  cache_restore_samples << result[:cache_restore_ms] if result[:cache_restore_ms] > 0.0
  first_id = result[:first_id]
  last_id = result[:last_id]
end

summarize("prefill", prefill_samples, prompt.size)
summarize("decode", decode_samples, generate)
summarize("cache_restore", cache_restore_samples, prompt.size) unless cache_restore_samples.empty?
decode_p50 = percentile(decode_samples.sort, 0.50)
puts "decode_ms_per_token_p50=#{(decode_p50 / generate).round(3)} first_id=#{first_id} last_id=#{last_id}"
unless cache_route_counts.empty?
  route_summary = cache_route_counts.map { |route, count| "#{route}:#{count}" }.join(",")
  puts "prompt_cache_routes=#{route_summary}"
end
if store = cache_store
  puts "prompt_cache_snapshot_bytes=#{store.snapshot_cache_bytes} prompt_cache_snapshot_hits=#{store.snapshot_cache_hits} prompt_cache_snapshot_misses=#{store.snapshot_cache_misses}"
end
