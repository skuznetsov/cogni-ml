# Per-phase timing for a single decode step, to decide where to invest next.
# Splits: full_attn layers, recurrent layers, output head, CPU non-matmul.
#
# Runs forward N times on a prefilled sequence and prints averages.

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

w = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
hp = w.hparams
state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)

# Prefill 5 tokens so KV and conv/SSM state are non-trivial
prompt_ids = [785_i32, 6722_i32, 315_i32, 9625_i32, 374_i32]
prompt_ids.each_with_index { |t, i| ML::GGUF::Qwen35CPU.forward(w, t, i, state) }

# Warm up
ML::GGUF::Qwen35CPU.forward(w, 11751_i32, 5, state)

# Reset state for a clean run? No — keep warm, just measure steady state.
# Run profiled forward (pos=6 now) a few times.

n_runs = 5
total_full = 0.0
total_rec  = 0.0
total_head = 0.0
total_emb  = 0.0
total_all  = 0.0
# Per-layer accumulators for layer 0..n_layer-1 (breakdown across one run)

n_layer = hp.n_layer

# Per-token pos grows each run; that's intended (realistic).
base_pos = 6
n_runs.times do |r|
  pos = base_pos + r

  all0 = Time.instant

  emb0 = Time.instant
  x = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 11751_i32)
  total_emb += (Time.instant - emb0).total_milliseconds

  w.layers.each_with_index do |lw, il|
    t0 = Time.instant
    case lw
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, lw, state.layers[il], hp, state.max_seq)
      total_full += (Time.instant - t0).total_milliseconds
    in ML::GGUF::Qwen35RecurrentWeights
      x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, lw, state.layers[il], hp, state.max_seq)
      total_rec  += (Time.instant - t0).total_milliseconds
    end
  end

  h0 = Time.instant
  ML::GGUF::Qwen35CPU.rms_norm!(x, w.output_norm, hp.rms_eps)
  _ = ML::GGUF::Qwen35CPU.qmatvec_nobias(w.output, x)
  total_head += (Time.instant - h0).total_milliseconds

  total_all += (Time.instant - all0).total_milliseconds
end

n = n_runs.to_f
printf "Averages over %d decode steps (pos starting at %d):\n", n_runs, base_pos
printf "  total forward:    %.1f ms\n",                         total_all  / n
printf "  embedding:        %.2f ms\n",                         total_emb  / n
printf "  full_attn layers (8 total):   %.1f ms  (%.2f ms/layer)\n",
  total_full / n, total_full / n / 8
printf "  recurrent layers (24 total):  %.1f ms  (%.2f ms/layer)\n",
  total_rec  / n, total_rec  / n / 24
printf "  output (rmsnorm + lm_head):   %.1f ms\n",              total_head / n
printf "\n"
printf "Allocation:\n"
printf "  full_attn share:  %5.1f%%\n", 100 * total_full / total_all
printf "  recurrent share:  %5.1f%%\n", 100 * total_rec  / total_all
printf "  head share:       %5.1f%%\n", 100 * total_head / total_all
