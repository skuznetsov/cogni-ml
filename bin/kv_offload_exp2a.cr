#!/usr/bin/env crystal
# Experiment 2A: Exact spill/restore — no routing, no sketch.
#
# Proves that the offload/restore concept doesn't break generation quality.
# Simulates exact KV offload by re-evaluating from stored token sequences:
#
#   1. Process a long text, split into blocks
#   2. "Spill" oldest blocks (store token IDs in PG)
#   3. For evaluation: restore by fetching tokens from PG, re-eval prefix + tail
#   4. Compare perplexity vs full-context baseline
#
# This doesn't serialize actual KV tensors — it re-evaluates from tokens.
# This is expensive but tests correctness of the restore concept.

require "../src/ml/llm/llama_ffi"
require "../src/ml/llm/llama"
require "json"

LLM_MODEL = ENV["LLM_MODEL"]? || (Path.home / ".cache/lm-studio/models/google/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf").to_s
BLOCK_SIZE  = (ENV["BLOCK_SIZE"]? || "64").to_i
TAIL_SIZE   = (ENV["TAIL_SIZE"]? || "128").to_i
EVAL_TOKENS = (ENV["EVAL_TOKENS"]? || "32").to_i

TEXT = ENV["EVAL_TEXT"]? || <<-TEXT
  Prince Vasili always spoke lazily, like an actor repeating his part in an old play.
  Anna Pavlovna Scherer, on the contrary, despite her forty years, was brimming over
  with excitement and impulsiveness. To be an enthusiast had become her social vocation,
  and sometimes even when she did not feel like it, she became enthusiastic in order not
  to disappoint the expectations of those who knew her. The subdued smile which, though
  it did not suit her faded features, always played round her lips expressed, as in a
  spoiled child, a continual consciousness of her charming defect, which she neither
  wished, nor could, nor considered it necessary to correct.

  In the midst of a conversation about political affairs Anna Pavlovna burst out:
  "Oh, don't speak to me of Austria! Perhaps I don't understand things, but Austria
  never has wished, and does not wish, for war. She is betraying us! Russia alone must
  save Europe. Our gracious sovereign recognizes his high vocation and will be true to
  it. That is the one thing I have faith in! Our good and wonderful sovereign has to
  perform the noblest role on earth, and he is so virtuous and noble that God will not
  forsake him. He will fulfill his vocation and crush the hydra of revolution, which has
  become more terrible than ever in the person of this murderer and villain!"

  "But, my dear," said Prince Vasili with a quiet smile, "with all my respect for your
  enthusiasm, I would suggest that you consider the matter more calmly. The whole
  situation is not as simple as it appears at first glance."

  "The whole situation is perfectly clear!" retorted Anna Pavlovna. "Napoleon is the
  Antichrist, I am convinced of it. That is not mere words, Prince, it is a deep
  conviction. The allied monarchs must not submit to this tyrant, they must unite to
  overthrow him. If we had a leader worthy of the name, this whole terrible episode
  would have ended long ago."

  At this point the door opened and the late arrival, a young officer, entered the room.
  He was a stout, heavily built young man with close-cropped hair, spectacles, light
  breeches fashionable at that time, a very high ruffle, and a brown dress coat. The
  young man was the illegitimate son of Count Bezukhov, a well-known grandee of
  Catherine's time. He had not yet entered either the military or civil service, as he
  had only just returned from abroad where he had been educated, and this was his first
  appearance in society. Anna Pavlovna greeted him with the nod she accorded to the
  lowest hierarchy in her drawing room.

  Pierre was clumsy. Stout, above the average height, broad, with huge red hands; he
  did not know, as the saying is, how to enter a drawing room and still less how to
  leave one. He was absent-minded, and whenever he tried to be attentive his face
  involuntarily assumed an expression of good-natured, simple-minded enjoyment.
  TEXT

def compute_perplexity(model : ML::LLM::Model, prefix_tokens : Array(Int32),
                       eval_tokens : Array(Int32)) : {Float64, Float64}
  ctx = model.create_context(n_ctx: 4096, n_batch: 512, flash_attn: false)
  t0 = Time.instant

  unless prefix_tokens.empty?
    unless ctx.eval(prefix_tokens)
      ctx.free; return {-1.0, 0.0}
    end
  end

  unless ctx.eval([eval_tokens[0]])
    ctx.free; return {-1.0, 0.0}
  end

  total_logprob = 0.0_f64
  n = 0
  (1...eval_tokens.size).each do |i|
    logits = ctx.get_logits
    true_token = eval_tokens[i]
    max_l = logits.max
    lse = 0.0_f64
    logits.each { |l| lse += Math.exp((l - max_l).to_f64) }
    lse = max_l.to_f64 + Math.log(lse)
    total_logprob += logits[true_token].to_f64 - lse
    n += 1
    unless ctx.eval([true_token])
      break
    end
  end

  elapsed_ms = (Time.instant - t0).total_milliseconds
  ctx.free
  return {-1.0, elapsed_ms} if n == 0
  {Math.exp(-total_logprob / n), elapsed_ms}
end

# --- Main ---
STDERR.puts "=== Experiment 2A: Exact spill/restore ==="
STDERR.puts "LLM: #{File.basename(LLM_MODEL)}"
STDERR.puts "block_size=#{BLOCK_SIZE} tail=#{TAIL_SIZE} eval=#{EVAL_TOKENS}"

ML::LLM.init
model = ML::LLM::Model.new(LLM_MODEL, n_gpu_layers: 99)

# Tokenize
all_tokens = model.tokenize(TEXT, add_bos: true)
STDERR.puts "Total tokens: #{all_tokens.size}"

eval_start = all_tokens.size - EVAL_TOKENS
tail_start = eval_start - TAIL_SIZE
eval_toks = all_tokens[eval_start..]
tail_toks = all_tokens[tail_start...eval_start]
prefix_toks = all_tokens[0...tail_start]

n_full_blocks = prefix_toks.size // BLOCK_SIZE
remainder = prefix_toks.size % BLOCK_SIZE
n_blocks = n_full_blocks + (remainder > 0 ? 1 : 0)
STDERR.puts "Prefix: #{prefix_toks.size} tokens (#{n_full_blocks} full + #{remainder > 0 ? 1 : 0} partial blocks), Tail: #{tail_toks.size}, Eval: #{eval_toks.size}"

# Spill blocks to file (PG latency already validated in Experiment 0)
spill_path = "/tmp/kv_offload_exp2a.json"
STDERR.puts "\nSpilling #{n_blocks} blocks to #{spill_path}..."
blocks_data = (0...n_blocks).map do |bi|
  start = bi * BLOCK_SIZE
  len = Math.min(BLOCK_SIZE, prefix_toks.size - start)
  prefix_toks[start, len]
end

spill_times = [] of Float64
t0 = Time.instant
File.write(spill_path, blocks_data.to_json)
spill_times << (Time.instant - t0).total_milliseconds
STDERR.puts "  spill: #{spill_times[0].round(3)}ms for #{n_blocks} blocks"

# Restore: read all blocks back
STDERR.puts "\nRestoring #{n_blocks} blocks from file..."
t0 = Time.instant
restored_blocks = Array(Array(Int32)).from_json(File.read(spill_path))
restore_ms = (Time.instant - t0).total_milliseconds
restored_prefix = [] of Int32
restored_blocks.each { |b| restored_prefix.concat(b) }
STDERR.puts "  restore: #{restore_ms.round(3)}ms"

# Verify restored == original (now includes partial block)
if restored_prefix == prefix_toks
  STDERR.puts "  integrity: OK (#{restored_prefix.size} restored == #{prefix_toks.size} original)"
else
  STDERR.puts "  integrity: FAIL (restored=#{restored_prefix.size} vs original=#{prefix_toks.size})"
  exit 1
end
File.delete(spill_path) rescue nil

# Compute perplexity for three configs
STDERR.puts "\nComputing perplexity (#{EVAL_TOKENS} eval tokens)..."

STDERR.print "  full context... "
ppl_full, ms_full = compute_perplexity(model, prefix_toks + tail_toks, eval_toks)
STDERR.puts "ppl=#{ppl_full.round(2)} (#{ms_full.round(0)}ms)"

STDERR.print "  recency only... "
ppl_recency, ms_recency = compute_perplexity(model, tail_toks, eval_toks)
STDERR.puts "ppl=#{ppl_recency.round(2)} (#{ms_recency.round(0)}ms)"

STDERR.print "  exact restore... "
ppl_restore, ms_restore = compute_perplexity(model, restored_prefix + tail_toks, eval_toks)
STDERR.puts "ppl=#{ppl_restore.round(2)} (#{ms_restore.round(0)}ms)"

# Results
STDERR.puts "\n=== Results ==="
STDERR.puts "  #{"Context".ljust(20)} #{"Tokens".rjust(8)} #{"PPL".rjust(8)} #{"Eval ms".rjust(10)}"
STDERR.puts "  #{"-" * 50}"
STDERR.puts "  #{"full".ljust(20)} #{(prefix_toks.size + tail_toks.size).to_s.rjust(8)} #{ppl_full.round(2).to_s.rjust(8)} #{ms_full.round(0).to_s.rjust(10)}"
STDERR.puts "  #{"recency (tail only)".ljust(20)} #{tail_toks.size.to_s.rjust(8)} #{ppl_recency.round(2).to_s.rjust(8)} #{ms_recency.round(0).to_s.rjust(10)}"
STDERR.puts "  #{"exact restore".ljust(20)} #{(restored_prefix.size + tail_toks.size).to_s.rjust(8)} #{ppl_restore.round(2).to_s.rjust(8)} #{ms_restore.round(0).to_s.rjust(10)}"

delta = (ppl_restore - ppl_full).abs
STDERR.puts "\n  Full vs restore delta: #{delta.round(4)} ppl"

if delta < 0.1
  STDERR.puts "  VERDICT: PASS (exact restore matches full context)"
elsif delta < 1.0
  STDERR.puts "  VERDICT: SOFT (small delta #{delta.round(2)})"
else
  STDERR.puts "  VERDICT: FAIL (large delta #{delta.round(2)})"
end

model.free
ML::LLM.cleanup
