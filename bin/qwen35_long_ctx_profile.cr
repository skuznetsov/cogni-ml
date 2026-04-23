# Profile full-attn vs recurrent costs at longer context (pos ~ 256-512)
# to evaluate Phase 3a benefit where CPU attention is no longer trivial.

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

w  = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
hp = w.hparams

prefill = (ARGV[0]? || "256").to_i
n_runs  = (ARGV[1]? || "5").to_i

state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prefill + 16)

# Prefill with a deterministic token pattern
printf "Prefilling %d tokens...\n", prefill
t0 = Time.instant
prefill.times do |i|
  tok = ((i * 7 + 11) % 1000).to_i32
  ML::GGUF::Qwen35CPU.forward(w, tok, i, state)
end
printf "Prefill done in %.1fs\n\n", (Time.instant - t0).total_seconds

# Warm up
ML::GGUF::Qwen35CPU.forward(w, 11751_i32, prefill, state)

total_full = 0.0
total_rec  = 0.0
total_all  = 0.0

n_runs.times do |r|
  pos = prefill + 1 + r
  all0 = Time.instant
  x = ML::GGUF::Qwen35CPU.embedding_lookup(w.token_embd, 11751_i32)
  w.layers.each_with_index do |lw, il|
    t0 = Time.instant
    case lw
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, lw, state.layers[il], hp, prefill + 16)
      total_full += (Time.instant - t0).total_milliseconds
    in ML::GGUF::Qwen35RecurrentWeights
      x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, lw, state.layers[il], hp, prefill + 16)
      total_rec  += (Time.instant - t0).total_milliseconds
    end
  end
  ML::GGUF::Qwen35CPU.rms_norm!(x, w.output_norm, hp.rms_eps)
  _ = ML::GGUF::Qwen35CPU.qmatvec_nobias(w.output, x)
  total_all += (Time.instant - all0).total_milliseconds
end

n = n_runs.to_f
printf "Averages over %d decode steps at pos=%d:\n", n_runs, prefill + 1
printf "  total forward:    %.1f ms\n",         total_all / n
printf "  full_attn (8):    %.1f ms (%.2f ms/layer)\n", total_full / n, total_full / n / 8
printf "  recurrent (24):   %.1f ms (%.2f ms/layer)\n", total_rec / n,  total_rec  / n / 24
