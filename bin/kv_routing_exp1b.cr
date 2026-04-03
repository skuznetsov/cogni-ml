#!/usr/bin/env crystal
# Experiment 1B: LLM-closer routing quality for FlashHadamard KV sketches.
#
# Measures perplexity on suffix tokens with three context configs:
#   a. Full prefix (ground truth)
#   b. Truncated to last M tokens (recency-only)
#   c. Truncated + top-k blocks retrieved by embedding sketch
#
# Uses Qwen 3.5 9B via llama.cpp FFI for actual autoregressive evaluation.

require "../src/ml/llm/llama_ffi"
require "../src/ml/llm/llama"
require "../src/ml/gguf/nomic_bert"
require "../src/ml/gguf/metal_backend"
require "../src/ml/metal/compute_graph"

LLM_MODEL = ENV["LLM_MODEL"]? || (Path.home / ".cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf").to_s
EMBED_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

BLOCK_SIZE   = (ENV["BLOCK_SIZE"]? || "128").to_i    # tokens per block
TAIL_SIZE    = (ENV["TAIL_SIZE"]? || "256").to_i      # hot tail tokens (recency window)
EVAL_TOKENS  = (ENV["EVAL_TOKENS"]? || "64").to_i     # suffix tokens to measure perplexity on
SKETCH_K     = (ENV["SKETCH_K"]? || "3").to_i         # blocks to retrieve via sketch
TOTAL_TOKENS = (ENV["TOTAL_TOKENS"]? || "1024").to_i  # total context length

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

def compute_perplexity(model : ML::LLM::Model,
                       prefix_tokens : Array(Int32), eval_tokens : Array(Int32)) : Float64
  # Create a fresh context for each evaluation (avoids stale KV state)
  ctx = model.create_context(n_ctx: 4096, n_batch: 512, flash_attn: false)

  # Eval prefix in one batch
  unless prefix_tokens.empty?
    STDERR.puts "    eval prefix (#{prefix_tokens.size} tokens)..."
    unless ctx.eval(prefix_tokens)
      STDERR.puts "    prefix eval FAILED"
      ctx.free
      return -1.0
    end
  end

  # Eval first eval token
  unless ctx.eval([eval_tokens[0]])
    STDERR.puts "    first token eval FAILED"
    ctx.free
    return -1.0
  end

  STDERR.puts "    getting logits..."
  total_logprob = 0.0_f64
  n = 0
  (1...eval_tokens.size).each do |i|
    true_token = eval_tokens[i]
    logits = ctx.get_logits

    max_logit = logits.max
    log_sum_exp = 0.0_f64
    logits.each { |l| log_sum_exp += Math.exp((l - max_logit).to_f64) }
    log_sum_exp = max_logit.to_f64 + Math.log(log_sum_exp)

    logprob = logits[true_token].to_f64 - log_sum_exp
    total_logprob += logprob
    n += 1

    unless ctx.eval([true_token])
      STDERR.puts "    eval token #{i} FAILED"
      break
    end
  end

  ctx.free
  return -1.0 if n == 0
  Math.exp(-total_logprob / n)
end

def embed_text_blocks(embedder : ML::GGUF::NomicBertMoE(ML::GGUF::MetalBackend),
                      text : String, block_size : Int32) : Array(Array(Float32))
  # Split text into chunks of ~block_size words
  words = text.split
  blocks = [] of String
  i = 0
  while i < words.size
    chunk = words[i, block_size].join(" ")
    blocks << chunk
    i += block_size
  end
  blocks.map { |b| embedder.embed(b) }
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.size.times do |i|
    dot += a[i].to_f64 * b[i].to_f64
    na += a[i].to_f64 ** 2
    nb += b[i].to_f64 ** 2
  end
  na = Math.sqrt(na)
  nb = Math.sqrt(nb)
  return 0.0 if na < 1e-12 || nb < 1e-12
  dot / (na * nb)
end

# --- Main ---
STDERR.puts "=== Experiment 1B: LLM-closer routing quality ==="
STDERR.puts "LLM: #{File.basename(LLM_MODEL)}"
STDERR.puts "Embed: #{File.basename(EMBED_MODEL)}"
STDERR.puts "block_size=#{BLOCK_SIZE} tail=#{TAIL_SIZE} eval=#{EVAL_TOKENS} sketch_k=#{SKETCH_K}"

# Init LLM
ML::LLM.init
llm_model = ML::LLM::Model.new(LLM_MODEL, n_gpu_layers: 99)
# Context created fresh per evaluation inside compute_perplexity

# Init embedder
ML::Metal::Device.init!
embedder = ML::GGUF::NomicBertMoE.from_gguf(EMBED_MODEL, ML::GGUF::MetalBackend.new)

# Tokenize
all_tokens = llm_model.tokenize(TEXT, add_bos: true)
STDERR.puts "Total tokens: #{all_tokens.size}"

if all_tokens.size < TAIL_SIZE + EVAL_TOKENS + BLOCK_SIZE * 2
  STDERR.puts "Text too short for experiment. Need at least #{TAIL_SIZE + EVAL_TOKENS + BLOCK_SIZE * 2} tokens."
  exit 1
end

# Split: prefix blocks | tail | eval
eval_start = all_tokens.size - EVAL_TOKENS
tail_start = eval_start - TAIL_SIZE
eval_toks = all_tokens[eval_start..]
tail_toks = all_tokens[tail_start...eval_start]
prefix_toks = all_tokens[0...tail_start]

n_prefix_blocks = prefix_toks.size // BLOCK_SIZE
STDERR.puts "Prefix: #{prefix_toks.size} tokens (#{n_prefix_blocks} blocks), Tail: #{tail_toks.size}, Eval: #{eval_toks.size}"

# Embed each prefix block for sketch routing
STDERR.puts "Embedding #{n_prefix_blocks} prefix blocks..."
block_texts = (0...n_prefix_blocks).map do |bi|
  start = bi * BLOCK_SIZE
  block_token_ids = prefix_toks[start, BLOCK_SIZE]
  # Decode tokens back to text for embedding
  block_token_ids.map { |t| llm_model.token_to_piece(t) }.join
end
block_embeddings = block_texts.map { |t| embedder.embed(t) }

# Embed the tail (query for routing)
tail_text = tail_toks.map { |t| llm_model.token_to_piece(t) }.join
query_embedding = embedder.embed(tail_text)

# Find top-k blocks by sketch similarity
similarities = block_embeddings.map_with_index { |emb, idx| {cosine(emb, query_embedding), idx} }
similarities.sort_by! { |s, _| -s }
topk_block_ids = similarities[0, SKETCH_K].map { |_, idx| idx }
STDERR.puts "Sketch top-#{SKETCH_K} blocks: #{topk_block_ids} (similarities: #{similarities[0, SKETCH_K].map { |s, _| s.round(3) }})"

# Build context variants
# a. Full prefix + tail
full_prefix = prefix_toks + tail_toks

# b. Tail only (recency)
recency_prefix = tail_toks

# c. Tail + sketch-retrieved blocks
sketch_blocks_toks = [] of Int32
topk_block_ids.sort.each do |bi|  # keep original order
  start = bi * BLOCK_SIZE
  sketch_blocks_toks.concat(prefix_toks[start, BLOCK_SIZE])
end
sketch_prefix = sketch_blocks_toks + tail_toks

STDERR.puts "\nContext sizes: full=#{full_prefix.size}, recency=#{recency_prefix.size}, sketch=#{sketch_prefix.size}"

# Compute perplexity for each
STDERR.puts "\nComputing perplexity (#{EVAL_TOKENS} eval tokens)..."

STDERR.print "  full context... "
ppl_full = compute_perplexity(llm_model, full_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_full.round(2)}"

STDERR.print "  recency only... "
ppl_recency = compute_perplexity(llm_model, recency_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_recency.round(2)}"

STDERR.print "  sketch+tail...  "
ppl_sketch = compute_perplexity(llm_model, sketch_prefix, eval_toks)
STDERR.puts "ppl=#{ppl_sketch.round(2)}"

# Results
STDERR.puts "\n=== Results ==="
STDERR.puts "  #{"Context".ljust(20)} #{"Tokens".rjust(8)} #{"Perplexity".rjust(12)}"
STDERR.puts "  #{"-" * 42}"
STDERR.puts "  #{"full".ljust(20)} #{full_prefix.size.to_s.rjust(8)} #{ppl_full.round(2).to_s.rjust(12)}"
STDERR.puts "  #{"recency (tail only)".ljust(20)} #{recency_prefix.size.to_s.rjust(8)} #{ppl_recency.round(2).to_s.rjust(12)}"
STDERR.puts "  #{"sketch + tail".ljust(20)} #{sketch_prefix.size.to_s.rjust(8)} #{ppl_sketch.round(2).to_s.rjust(12)}"

# Verdict
gap_recency = ppl_recency - ppl_full
gap_sketch = ppl_sketch - ppl_full
recovery = if gap_recency > 0.1
             ((gap_recency - gap_sketch) / gap_recency * 100).round(1)
           else
             0.0
           end

STDERR.puts "\n  Recency gap from full: +#{gap_recency.round(2)} ppl"
STDERR.puts "  Sketch gap from full:  +#{gap_sketch.round(2)} ppl"
STDERR.puts "  Recovery: #{recovery}% of recency gap closed by sketch routing"

if recovery > 30
  STDERR.puts "\n  VERDICT: PASS (sketch recovers #{recovery}% of context gap)"
elsif recovery > 10
  STDERR.puts "\n  VERDICT: SOFT (sketch recovers #{recovery}% — weak but non-zero signal)"
else
  STDERR.puts "\n  VERDICT: FAIL (sketch recovery #{recovery}% — near zero)"
end

# Cleanup
llm_model.free
ML::LLM.cleanup
