# Phase 4.0 instrumentation probe.
#
# Enables Qwen35Metal.Profile, runs N decode steps, and reports:
#   encode vs wait vs read time per dispatch type,
#   CPU-fallback matmul count,
#   total sync count per token.
# Answers: is decode sync-bound or compute-bound?

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

w     = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
hp    = w.hparams
state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)

prefill = (ARGV[0]? || "5").to_i
n_runs  = (ARGV[1]? || "5").to_i

printf "Prefilling %d tokens...\n", prefill
prefill.times do |i|
  tok = ((i * 7 + 11) % 1000).to_i32
  ML::GGUF::Qwen35CPU.forward(w, tok, i, state)
end

# warm-up
ML::GGUF::Qwen35CPU.forward(w, 11751_i32, prefill, state)

# Enable profiling and run
ML::GGUF::Qwen35Metal::Profile.reset
ML::GGUF::Qwen35Metal::Profile.enable!

wall0 = Time.instant
n_runs.times do |r|
  ML::GGUF::Qwen35CPU.forward(w, 11751_i32, prefill + 1 + r, state)
end
wall_ms = (Time.instant - wall0).total_milliseconds

ML::GGUF::Qwen35Metal::Profile.disable!

printf "\n%s", ML::GGUF::Qwen35Metal::Profile.report_io
printf "  wall clock (n=%d): %.1f ms  (%.2f ms/tok)\n", n_runs, wall_ms, wall_ms / n_runs
