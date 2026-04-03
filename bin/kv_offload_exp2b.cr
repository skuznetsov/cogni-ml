#!/usr/bin/env crystal
# Experiment 2B: Selective context routing proxy.
#
# Tests whether FlashHadamard sketch can select relevant text blocks
# for context construction. This is a CONTEXT SELECTION experiment,
# not a live KV-cache spill/restore test — it re-evaluates from tokens,
# not from serialized KV tensors.
#
# Four-way perplexity comparison:
#   a. Full context (baseline)
#   b. Recency only (tail)
#   c. Full prefix + tail (same as full — correctness check)
#   d. Sketch-selected blocks + tail (selective routing)
#
# Primary metric: ppl of sketch-selected vs full context.
# Fixed setup: one block_size, one tail, one k, one text.

require "../src/ml/llm/llama_ffi"
require "../src/ml/llm/llama"
require "../src/ml/gguf/nomic_bert"
require "../src/ml/gguf/metal_backend"
require "../src/ml/metal/compute_graph"

LLM_MODEL = ENV["LLM_MODEL"]? || (Path.home / ".cache/lm-studio/models/google/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf").to_s
EMBED_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

BLOCK_SIZE  = (ENV["BLOCK_SIZE"]? || "64").to_i
TAIL_SIZE   = (ENV["TAIL_SIZE"]? || "128").to_i
EVAL_TOKENS = (ENV["EVAL_TOKENS"]? || "32").to_i
SKETCH_K    = (ENV["SKETCH_K"]? || "2").to_i

TEXT = <<-TEXT
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

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64; na = 0.0_f64; nb = 0.0_f64
  a.size.times do |i|
    dot += a[i].to_f64 * b[i].to_f64
    na += a[i].to_f64 ** 2
    nb += b[i].to_f64 ** 2
  end
  na = Math.sqrt(na); nb = Math.sqrt(nb)
  return 0.0 if na < 1e-12 || nb < 1e-12
  dot / (na * nb)
end

# --- Main ---
STDERR.puts "=== Experiment 2B: Sketch-routed restore ==="
STDERR.puts "LLM: #{File.basename(LLM_MODEL)}"
STDERR.puts "Embed: #{File.basename(EMBED_MODEL)}"
STDERR.puts "block_size=#{BLOCK_SIZE} tail=#{TAIL_SIZE} eval=#{EVAL_TOKENS} sketch_k=#{SKETCH_K}"

# Init
ML::LLM.init
model = ML::LLM::Model.new(LLM_MODEL, n_gpu_layers: 99)
ML::Metal::Device.init!
embedder = ML::GGUF::NomicBertMoE.from_gguf(EMBED_MODEL, ML::GGUF::MetalBackend.new)

# Tokenize and split
all_tokens = model.tokenize(TEXT, add_bos: true)
eval_start = all_tokens.size - EVAL_TOKENS
tail_start = eval_start - TAIL_SIZE
eval_toks = all_tokens[eval_start..]
tail_toks = all_tokens[tail_start...eval_start]
prefix_toks = all_tokens[0...tail_start]

# Split prefix into blocks (including partial)
n_full = prefix_toks.size // BLOCK_SIZE
remainder = prefix_toks.size % BLOCK_SIZE
n_blocks = n_full + (remainder > 0 ? 1 : 0)

blocks = (0...n_blocks).map do |bi|
  start = bi * BLOCK_SIZE
  len = Math.min(BLOCK_SIZE, prefix_toks.size - start)
  prefix_toks[start, len]
end

STDERR.puts "Total: #{all_tokens.size} tokens"
STDERR.puts "Prefix: #{prefix_toks.size} tokens (#{n_blocks} blocks), Tail: #{tail_toks.size}, Eval: #{eval_toks.size}"

# Embed blocks for sketch routing
STDERR.puts "\nEmbedding #{n_blocks} blocks..."
block_texts = blocks.map do |toks|
  toks.map { |t| model.token_to_piece(t) }.join
end
block_embeddings = block_texts.map { |t| embedder.embed(t) }

# Embed tail as routing query
tail_text = tail_toks.map { |t| model.token_to_piece(t) }.join
query_emb = embedder.embed(tail_text)

# Find top-k blocks by sketch similarity
sims = block_embeddings.map_with_index { |emb, idx| {cosine(emb, query_emb), idx} }
sims.sort_by! { |s, _| -s }
topk_ids = sims[0, SKETCH_K].map { |_, idx| idx }
STDERR.puts "Sketch top-#{SKETCH_K}: blocks #{topk_ids} (sims: #{sims[0, SKETCH_K].map { |s, _| s.round(3) }})"

# Build four context variants
full_prefix = prefix_toks + tail_toks

recency_prefix = tail_toks

exact_prefix = prefix_toks + tail_toks  # same as full (all blocks restored)

routed_toks = [] of Int32
topk_ids.sort.each { |bi| routed_toks.concat(blocks[bi]) }  # keep original order
routed_prefix = routed_toks + tail_toks

routed_token_count = routed_toks.size
STDERR.puts "Routed: #{SKETCH_K} blocks = #{routed_token_count} tokens + #{tail_toks.size} tail = #{routed_prefix.size} total"
STDERR.puts "Memory saving: #{((1.0 - routed_prefix.size.to_f / full_prefix.size) * 100).round(1)}% fewer tokens"

# Compute perplexity
STDERR.puts "\nComputing perplexity..."
STDERR.print "  full...    "
ppl_full, ms_full = compute_perplexity(model, full_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_full.round(2)}"

STDERR.print "  recency... "
ppl_rec, ms_rec = compute_perplexity(model, recency_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_rec.round(2)}"

STDERR.print "  exact...   "
ppl_exact, ms_exact = compute_perplexity(model, exact_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_exact.round(2)}"

STDERR.print "  routed...  "
ppl_routed, ms_routed = compute_perplexity(model, routed_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_routed.round(2)}"

# Results
STDERR.puts "\n=== Results ==="
STDERR.puts "  #{"Context".ljust(20)} #{"Tokens".rjust(8)} #{"PPL".rjust(8)} #{"ms".rjust(8)}"
STDERR.puts "  #{"-" * 48}"
STDERR.puts "  #{"full".ljust(20)} #{full_prefix.size.to_s.rjust(8)} #{ppl_full.round(2).to_s.rjust(8)} #{ms_full.round(0).to_s.rjust(8)}"
STDERR.puts "  #{"recency".ljust(20)} #{recency_prefix.size.to_s.rjust(8)} #{ppl_rec.round(2).to_s.rjust(8)} #{ms_rec.round(0).to_s.rjust(8)}"
STDERR.puts "  #{"exact restore".ljust(20)} #{exact_prefix.size.to_s.rjust(8)} #{ppl_exact.round(2).to_s.rjust(8)} #{ms_exact.round(0).to_s.rjust(8)}"
STDERR.puts "  #{"routed restore".ljust(20)} #{routed_prefix.size.to_s.rjust(8)} #{ppl_routed.round(2).to_s.rjust(8)} #{ms_routed.round(0).to_s.rjust(8)}"

gap_rec = ppl_rec - ppl_exact
gap_routed = ppl_routed - ppl_exact
recovery = gap_rec > 0.1 ? ((gap_rec - gap_routed) / gap_rec * 100).round(1) : 100.0

STDERR.puts "\n  Recency gap vs exact:  +#{gap_rec.round(2)} ppl"
STDERR.puts "  Routed gap vs exact:   +#{gap_routed.round(2)} ppl"
STDERR.puts "  Recovery: #{recovery}% of recency-to-exact gap closed"
STDERR.puts "  Memory: #{routed_prefix.size}/#{full_prefix.size} tokens (#{((routed_prefix.size.to_f / full_prefix.size) * 100).round(1)}%)"

if gap_routed < 0.5 && recovery > 80
  STDERR.puts "\n  VERDICT: PASS"
elsif gap_routed < 2.0 && recovery > 50
  STDERR.puts "\n  VERDICT: SOFT"
else
  STDERR.puts "\n  VERDICT: FAIL"
end

model.free
ML::LLM.cleanup
