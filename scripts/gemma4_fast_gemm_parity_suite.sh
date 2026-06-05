#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="$ROOT/scripts/run_safe.sh"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
PROFILE_BIN="${GEMMA4_PROFILE_BIN:-/tmp/gemma4_metal_decode_profile_parity_suite}"
OUT_DIR="${OUT_DIR:-/tmp/gemma4_fast_gemm_parity_$(date +%Y%m%d%H%M%S)}"
GENERATE=32
LIMIT=0
MAX_SEQ=1024
PREFILL_CHUNK=512
TIMEOUT_SEC=420
MAX_RSS_MB=8000
RUN_TIMEOUT_MS=80000
MODEL_ARG=()

usage() {
  cat <<'USAGE'
usage: scripts/gemma4_fast_gemm_parity_suite.sh [options]

Sequential strict-vs-fast CogniGemma parity smoke. It compares greedy token traces
for strict row-prefill and GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 on the same prompts.

Options:
  --generate N       Generated tokens per prompt (default: 32)
  --limit N          Limit prompt count (default: all built-in prompts)
  --max-seq N        KV max sequence length (default: 1024)
  --prefill-chunk N  Row-prefill chunk size (default: 512)
  --profile-bin P    Existing or output profile binary path
  --model P          Gemma4 GGUF model path passed to the profile binary
  --out-dir P        Output directory for logs
  --timeout-sec N    run_safe timeout seconds (default: 420)
  --max-rss-mb N     build RSS cap for run_safe (default: 8000)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate) GENERATE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-seq) MAX_SEQ="$2"; shift 2 ;;
    --prefill-chunk) PREFILL_CHUNK="$2"; shift 2 ;;
    --profile-bin) PROFILE_BIN="$2"; shift 2 ;;
    --model) MODEL_ARG=(--model "$2"); shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --max-rss-mb) MAX_RSS_MB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ ! -x "$PROFILE_BIN" ]]; then
  echo "building profile binary: $PROFILE_BIN" >&2
  CRYSTAL_CACHE_DIR="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_gemma4_parity_suite_build}" \
    "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT_SEC" "$MAX_RSS_MB" \
    build "$ROOT/bin/gemma4_metal_decode_profile.cr" \
    -o "$PROFILE_BIN" \
    --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
fi

prompts=(
  "Write a concise Crystal function that computes the top-k largest Int32 values from an Array(Int32) without sorting the full array. Include edge cases."
  "Explain why a batched matrix multiplication can be faster than a serial vector path, but can produce tiny numerical differences."
  "Draft a short test plan for a local LLM inference engine that compares prompt prefill speed and greedy decode parity."
  "Write a small recursive Fibonacci function in Python, then describe its time complexity and a faster alternative."
  "Summarize the tradeoff between exact hidden-state equality and output-level parity in one paragraph."
)

count=${#prompts[@]}
if [[ "$LIMIT" -gt 0 && "$LIMIT" -lt "$count" ]]; then
  count="$LIMIT"
fi

printf 'idx\tprompt_tokens\tstrict_pp_tps\tfast_pp_tps\tstrict_tg_tps\tfast_tg_tps\ttrace_equal\n'
fail=0

extract_value() {
  local key="$1" file="$2"
  rg -o "${key}=[^ ]+" "$file" | head -1 | cut -d= -f2 || true
}

for ((i=0; i<count; i++)); do
  prompt="${prompts[$i]}"
  strict_log="$OUT_DIR/prompt_${i}_strict.log"
  fast_log="$OUT_DIR/prompt_${i}_fast.log"

  env -u GEMMA4_ROW_PREFILL_ALLOW_GEMM COGNI_RUN_SAFE_MIN_FREE_PCT=12 \
    "$RUN_SAFE" "$PROFILE_BIN" "$TIMEOUT_SEC" "$RUN_TIMEOUT_MS" \
    "${MODEL_ARG[@]}" \
    --chat-user "$prompt" --generate "$GENERATE" --max-seq "$MAX_SEQ" --runs 1 --warmups 0 \
    --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --print-generated-ids \
    > "$strict_log" 2>&1

  env GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 COGNI_RUN_SAFE_MIN_FREE_PCT=12 \
    "$RUN_SAFE" "$PROFILE_BIN" "$TIMEOUT_SEC" "$RUN_TIMEOUT_MS" \
    "${MODEL_ARG[@]}" \
    --chat-user "$prompt" --generate "$GENERATE" --max-seq "$MAX_SEQ" --runs 1 --warmups 0 \
    --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --print-generated-ids \
    > "$fast_log" 2>&1

  strict_trace="$(extract_value token_trace "$strict_log")"
  fast_trace="$(extract_value token_trace "$fast_log")"
  equal="no"
  if [[ -n "$strict_trace" && "$strict_trace" == "$fast_trace" ]]; then
    equal="yes"
  else
    fail=1
  fi

  prompt_len="$(extract_value prompt_len "$strict_log")"
  strict_pp="$(extract_value prefill_p50_tok_s "$strict_log")"
  fast_pp="$(extract_value prefill_p50_tok_s "$fast_log")"
  strict_tg="$(extract_value decode_p50_tok_s "$strict_log")"
  fast_tg="$(extract_value decode_p50_tok_s "$fast_log")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$prompt_len" "$strict_pp" "$fast_pp" "$strict_tg" "$fast_tg" "$equal"
done

echo "logs=$OUT_DIR" >&2
exit "$fail"
