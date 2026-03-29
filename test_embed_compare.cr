# Compare embeddings: native Crystal BERT vs llama.cpp FFI
# Requires: -Duse_gguf flag for llama.cpp linking

require "./src/ml/gguf/nomic_bert"
require "./src/ml/llm/llama"

MODEL_PATH = "/Users/sergey/.cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf"

def cosine(a : Array(Float32) | Pointer(Float32), b : Array(Float32), n : Int32) : Float64
  dot = 0.0_f64
  n.times do |i|
    ai = a.is_a?(Pointer(Float32)) ? a[i] : a[i]
    dot += ai.to_f64 * b[i].to_f64
  end
  dot
end

STDERR.puts "=== Loading native model ==="
native = ML::GGUF::NomicBertMoE.from_gguf(MODEL_PATH)
STDERR.puts "Native ready (#{native.dim}d)"

STDERR.puts "\n=== Loading FFI model ==="
ML::LLM.init

model_params = ML::LLM::LlamaFFI.llama_model_default_params
model_params.n_gpu_layers = 99
ffi_model = ML::LLM::LlamaFFI.llama_model_load_from_file(MODEL_PATH, model_params)
raise "FFI model load failed" if ffi_model.null?

ctx_params = ML::LLM::LlamaFFI.llama_context_default_params
ctx_params.n_ctx = 512
ctx_params.n_batch = 512
ctx_params.embeddings = true  # Enable embedding mode
ffi_ctx = ML::LLM::LlamaFFI.llama_init_from_model(ffi_model, ctx_params)
raise "FFI context init failed" if ffi_ctx.null?

ffi_vocab = ML::LLM::LlamaFFI.llama_model_get_vocab(ffi_model)
dim = ML::LLM::LlamaFFI.llama_model_n_embd(ffi_model)
STDERR.puts "FFI ready (#{dim}d, Metal GPU)"

texts = ["Hello world", "Crystal programming language", "def method_name(arg)"]

texts.each do |text|
  # Native embedding
  native_emb = native.embed(text)

  # FFI tokenize
  max_tokens = 512
  tokens = Array(Int32).new(max_tokens, 0)
  n_tokens = ML::LLM::LlamaFFI.llama_tokenize(ffi_vocab, text, text.bytesize, tokens.to_unsafe, max_tokens, true, true)
  raise "Tokenize failed: #{n_tokens}" if n_tokens < 0

  STDERR.puts "\n\"#{text}\""
  STDERR.puts "  Native tokens: #{native.tokenize(text)}"
  STDERR.puts "  FFI tokens:    #{tokens[0, n_tokens].to_a}"

  # FFI encode — use llama_batch_init for proper embedding support
  # Note: BERT models may not have KV cache, skip llama_memory_clear

  batch = ML::LLM::LlamaFFI.llama_batch_init(n_tokens, 0, 1)
  n_tokens.times do |i|
    batch.token[i] = tokens[i]
    batch.pos[i] = i
    batch.n_seq_id[i] = 1
    batch.seq_id[i][0] = 0
    batch.logits[i] = 1_i8  # Request output for all positions
  end
  batch.n_tokens = n_tokens

  ret = ML::LLM::LlamaFFI.llama_encode(ffi_ctx, batch)
  if ret != 0
    STDERR.puts "  Encode failed: #{ret}"
    ML::LLM::LlamaFFI.llama_batch_free(batch)
    next
  end

  # Get pooled embedding (llama.cpp does mean pooling internally for BERT)
  ffi_emb = Array(Float32).new(dim, 0.0_f32)

  # Try sequence-level pooled embedding first
  seq_ptr = ML::LLM::LlamaFFI.llama_get_embeddings_seq(ffi_ctx, 0)
  if !seq_ptr.null?
    dim.times { |j| ffi_emb[j] = seq_ptr[j] }
    STDERR.puts "  (got seq-level embedding)"
  else
    # Fallback: try per-token
    got_any = false
    n_tokens.times do |i|
      ptr = ML::LLM::LlamaFFI.llama_get_embeddings_ith(ffi_ctx, i)
      next if ptr.null?
      got_any = true
      dim.times { |j| ffi_emb[j] += ptr[j] }
    end
    if got_any
      ffi_emb.map! { |v| v / n_tokens }
      STDERR.puts "  (got per-token embedding, mean pooled)"
    else
      # Last resort: llama_get_embeddings (returns pointer to first)
      ptr = ML::LLM::LlamaFFI.llama_get_embeddings(ffi_ctx)
      if !ptr.null?
        dim.times { |j| ffi_emb[j] = ptr[j] }
        STDERR.puts "  (got raw embedding pointer)"
      else
        STDERR.puts "  WARNING: all embedding pointers null!"
      end
    end
  end

  # L2 normalize (llama.cpp may already normalize for BERT)
  norm = Math.sqrt(ffi_emb.sum { |v| v * v })
  ffi_emb.map! { |v| v / norm } if norm > 1e-8

  ML::LLM::LlamaFFI.llama_batch_free(batch)

  # L2 normalize
  norm = Math.sqrt(ffi_emb.sum { |v| v * v })
  ffi_emb.map! { |v| v / norm } if norm > 1e-8

  # Compare
  cos_native_ffi = cosine(native_emb, ffi_emb, dim)
  STDERR.puts "  cosine(native, ffi): #{cos_native_ffi.round(6)}"
  STDERR.puts "  native[0..2]: #{native_emb[0, 3].map(&.round(6))}"
  STDERR.puts "  ffi[0..2]:    #{ffi_emb[0, 3].map(&.round(6))}"
end

ML::LLM::LlamaFFI.llama_free(ffi_ctx)
ML::LLM::LlamaFFI.llama_model_free(ffi_model)
ML::LLM.cleanup
STDERR.puts "\n=== Done ==="
