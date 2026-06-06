#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="$ROOT/scripts/run_safe.sh"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
PROFILE_BIN="${GEMMA4_PROFILE_BIN:-/tmp/gemma4_metal_decode_profile_cognigraph_matrix}"
MODEL="${GEMMA4_MODEL:-$HOME/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/cognigraph_gemma_matrix_$(date +%Y%m%d%H%M%S)}"
PROMPT=256
GEN=8
MAX_SEQ=512
PREFILL_CHUNK=512
TIMEOUT_SEC=420
MAX_RSS_MB=8000
RUN_TIMEOUT_MS=80000
BODY_CHAIN="${GEMMA4_BODY_CHAIN:-1}"

usage() {
  cat <<'USAGE'
usage: scripts/cognigraph_gemma_profile_matrix.sh [options]

Generate CogniGraph profile atlases for three Gemma4 corridors:
  pp_body  - fast numeric row-prefill body, no LM head
  tg_body  - decode body only, matched to llama-bench tg semantics
  top1     - greedy/top1 decode with resident vocab head

Options:
  --prompt N         Prompt tokens for pp_body (default: 256)
  --gen N            Decode tokens for tg_body/top1 (default: 8)
  --max-seq N        KV max sequence length (default: 512)
  --prefill-chunk N  Row-prefill chunk (default: 512)
  --body-chain N     Body-chain chunk for tg_body; default 1 for llama-bench parity
  --model PATH       Gemma4 GGUF path
  --profile-bin PATH Existing or output profile binary path
  --out-dir PATH     Output directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --gen) GEN="$2"; shift 2 ;;
    --max-seq) MAX_SEQ="$2"; shift 2 ;;
    --prefill-chunk) PREFILL_CHUNK="$2"; shift 2 ;;
    --body-chain) BODY_CHAIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --profile-bin) PROFILE_BIN="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

if [[ ! -x "$PROFILE_BIN" ]]; then
  echo "building profile binary: $PROFILE_BIN" >&2
  CRYSTAL_CACHE_DIR="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_cognigraph_gemma_matrix_build}" \
    "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT_SEC" "$MAX_RSS_MB" \
    build "$ROOT/bin/gemma4_metal_decode_profile.cr" \
    -o "$PROFILE_BIN" \
    --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
fi

TOKENS="$(python3 - <<PY
p=$PROMPT
print(','.join(str(42+i) for i in range(p)))
PY
)"

run_profile() {
  local label="$1"; shift
  local log="$OUT_DIR/${label}.profile.log"
  local atlas="$OUT_DIR/${label}.atlas.txt"
  COGNI_RUN_SAFE_MIN_FREE_PCT=12 "$RUN_SAFE" "$PROFILE_BIN" "$TIMEOUT_SEC" "$RUN_TIMEOUT_MS" \
    --model "$MODEL" "$@" > "$log" 2>&1
  "$ROOT/scripts/cognigraph_profile_atlas.cr" --log "$log" --limit 8 > "$atlas"
  echo "== $label =="
  rg -n 'Phi=|dominant_matmul=|totals:|traffic:' "$atlas" || true
}

GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 run_profile pp_body \
  --tokens "$TOKENS" --generate 1 --max-seq "$MAX_SEQ" --runs 1 --warmups 0 \
  --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --body-only --prefill-no-head --profile

run_profile tg_body \
  --tokens 42 --decode-only-seed 42 --generate "$GEN" --max-seq "$MAX_SEQ" --runs 1 --warmups 0 \
  --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --body-only --body-chain "$BODY_CHAIN" --profile --profile-decode-only

run_profile top1 \
  --tokens 42 --decode-only-seed 42 --generate "$GEN" --max-seq "$MAX_SEQ" --runs 1 --warmups 0 \
  --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --profile --profile-decode-only

echo "body_chain=$BODY_CHAIN (tg_body; 1 matches llama-bench tg)" >&2
echo "logs=$OUT_DIR" >&2
