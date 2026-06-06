#!/usr/bin/env bash
set -euo pipefail

# Sequential quiet-host ABBA runner for opt-in CogniGemma decode fusions.
# This is a promotion harness, not a broad benchmark. It fails closed by
# default when the host is noisy so that wall-time claims are not contaminated.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="${GEMMA4_FUSION_ABBA_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
MODEL="${GEMMA4_MODEL:-$HOME/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf}"
PROFILE_BIN="${GEMMA4_FUSION_ABBA_BIN:-/tmp/gemma4_metal_decode_profile_fusion_abba}"
OUT_DIR="${GEMMA4_FUSION_ABBA_OUT_DIR:-/tmp/gemma4_fusion_abba_$(date +%Y%m%d%H%M%S)}"
GEN="${GEMMA4_FUSION_ABBA_GEN:-128}"
WARMUPS="${GEMMA4_FUSION_ABBA_WARMUPS:-1}"
TIMEOUT_SEC="${GEMMA4_FUSION_ABBA_TIMEOUT:-420}"
MAX_MEM_MB="${GEMMA4_FUSION_ABBA_MAX_MEM_MB:-9000}"
MAX_SEQ="${GEMMA4_FUSION_ABBA_MAX_SEQ:-512}"
SEED="${GEMMA4_FUSION_ABBA_SEED:-42}"
SCHEDULE="${GEMMA4_FUSION_ABBA_SCHEDULE:-default,attnprep,default,ffnin,default,attnprep_ffnin,default}"
QUIET_WAIT_SEC="${COGNI_RUN_SAFE_WAIT_QUIET_SEC:-600}"
QUIET_REQUIRE="${COGNI_RUN_SAFE_REQUIRE_QUIET:-1}"
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-8}"

usage() {
  cat <<'USAGE'
usage: scripts/gemma4_fusion_abba.sh

Environment:
  GEMMA4_MODEL=PATH                         Gemma4 GGUF path
  GEMMA4_FUSION_ABBA_GEN=128                body decode tokens
  GEMMA4_FUSION_ABBA_WARMUPS=1              warmups per row
  GEMMA4_FUSION_ABBA_SCHEDULE=default,attnprep,default,ffnin,default,attnprep_ffnin,default
  GEMMA4_FUSION_ABBA_OUT_DIR=/tmp/...       output log directory
  GEMMA4_FUSION_ABBA_BIN=/tmp/...           profile binary path
  COGNI_RUN_SAFE_WAIT_QUIET_SEC=600         quiet-host wait before each row
  COGNI_RUN_SAFE_REQUIRE_QUIET=1            fail closed if still noisy
  COGNI_RUN_SAFE_MIN_FREE_PCT=8             memory-pressure kill threshold

Modes:
  default           no opt-in fusion
  attnprep          GEMMA4_ATTN_PREP_FUSE=1
  ffnin             GEMMA4_FFN_IN_RESIDUAL_NORM_FUSE=1
  attnprep_ffnin    both candidate fusions

Output:
  Prints TSV rows: index, mode, ms_per_token, tok_s, log.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 2; }
[[ -f "$ROOT/build/bridge.o" ]] || { echo "missing $ROOT/build/bridge.o" >&2; exit 2; }
mkdir -p "$OUT_DIR"

if [[ ! -x "$PROFILE_BIN" ]]; then
  echo "building $PROFILE_BIN" >&2
  (
    cd "$ROOT"
    CRYSTAL_CACHE_DIR="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_gemma4_fusion_abba_build}" \
    COGNI_RUN_SAFE_MIN_FREE_PCT="$MIN_FREE_PCT" \
    COGNI_RUN_SAFE_WAIT_QUIET_SEC="$QUIET_WAIT_SEC" \
    COGNI_RUN_SAFE_REQUIRE_QUIET="$QUIET_REQUIRE" \
      "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT_SEC" "$MAX_MEM_MB" \
      build bin/gemma4_metal_decode_profile.cr \
      -o "$PROFILE_BIN" \
      --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
  )
fi

mode_env() {
  case "$1" in
    default) ;;
    attnprep) echo "GEMMA4_ATTN_PREP_FUSE=1" ;;
    ffnin) echo "GEMMA4_FFN_IN_RESIDUAL_NORM_FUSE=1" ;;
    attnprep_ffnin)
      echo "GEMMA4_ATTN_PREP_FUSE=1"
      echo "GEMMA4_FFN_IN_RESIDUAL_NORM_FUSE=1"
      ;;
    *)
      echo "unknown mode: $1" >&2
      exit 2
      ;;
  esac
}

extract_ms() {
  python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r"decode_ms_per_token_p50=([0-9.]+)", s)
if not m:
    raise SystemExit("missing decode_ms_per_token_p50")
print(m.group(1))
PY
}

echo -e "index\tmode\tms_per_token\ttok_s\tlog"
IFS=',' read -r -a modes <<< "$SCHEDULE"
idx=0
for mode in "${modes[@]}"; do
  idx=$((idx + 1))
  log="$OUT_DIR/$(printf '%02d' "$idx")_${mode}.log"
  env_args=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && env_args+=("$line")
  done < <(mode_env "$mode")

  env "${env_args[@]}" \
    COGNI_RUN_SAFE_MIN_FREE_PCT="$MIN_FREE_PCT" \
    COGNI_RUN_SAFE_WAIT_QUIET_SEC="$QUIET_WAIT_SEC" \
    COGNI_RUN_SAFE_REQUIRE_QUIET="$QUIET_REQUIRE" \
    "$RUN_SAFE" "$PROFILE_BIN" "$TIMEOUT_SEC" "$MAX_MEM_MB" \
      --model "$MODEL" \
      --decode-wave \
      --body-only \
      --decode-only-seed="$SEED" \
      --generate="$GEN" \
      --max-seq="$MAX_SEQ" \
      --runs=1 \
      --warmups="$WARMUPS" \
      --body-chain=1 \
      > "$log" 2>&1

  ms="$(extract_ms "$log")"
  tok_s="$(python3 - <<PY
ms = float("$ms")
print(f"{1000.0 / ms:.6f}")
PY
)"
  echo -e "$idx\t$mode\t$ms\t$tok_s\t$log"
done

echo "logs=$OUT_DIR" >&2
