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
require "option_parser"

model_path = ENV["QWEN35_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
args = [] of String

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_sync_profile [--model PATH] [prefill] [n_runs]"
  parser.on("--model PATH", "GGUF model path (or QWEN35_MODEL)") { |path| model_path = path }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
  parser.unknown_args do |before_dash, after_dash|
    args.concat(before_dash)
    args.concat(after_dash)
  end
end

w = ML::GGUF::Qwen35Weights.from_gguf(model_path)
hp = w.hparams

prefill = (args[0]? || "5").to_i
n_runs = (args[1]? || "5").to_i
top1_only = ENV["QWEN35_PROFILE_TOP1"]? == "1"
# Include the warm-up token plus measured decode positions; otherwise 64/64
# writes past the KV cache allocation and corrupts benchmark evidence.
state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prefill + n_runs + 2)

def forward_for_profile(w, tok, pos, state, top1_only)
  if top1_only
    ML::GGUF::Qwen35CPU.forward_top1(w, tok, pos, state)
  else
    ML::GGUF::Qwen35CPU.forward(w, tok, pos, state)
  end
end

printf "Prefilling %d tokens...\n", prefill
prefill.times do |i|
  tok = ((i * 7 + 11) % 1000).to_i32
  forward_for_profile(w, tok, i, state, top1_only)
end

# warm-up
forward_for_profile(w, 11751_i32, prefill, state, top1_only)

# Enable profiling and run
ML::GGUF::Qwen35Metal::Profile.reset
ML::GGUF::Qwen35Metal::Profile.enable!

wall0 = Time.instant
n_runs.times do |r|
  forward_for_profile(w, 11751_i32, prefill + 1 + r, state, top1_only)
end
wall_ms = (Time.instant - wall0).total_milliseconds

ML::GGUF::Qwen35Metal::Profile.disable!

printf "\n%s", ML::GGUF::Qwen35Metal::Profile.report_io
printf "  wall clock (n=%d): %.1f ms  (%.2f ms/tok)\n", n_runs, wall_ms, wall_ms / n_runs
