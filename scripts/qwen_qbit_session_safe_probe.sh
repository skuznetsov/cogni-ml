#!/usr/bin/env bash
# Build and run one large-session QBit phase under unified-memory guards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_SAFE="${COGNI_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
PROBE_BIN="${QWEN_QBIT_SESSION_PROBE_BIN:-/private/tmp/qwen35_qbit_session_smoke_release}"
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-25}"
BUILD_TIMEOUT_SEC="${QWEN_QBIT_BUILD_TIMEOUT_SEC:-600}"
RUN_TIMEOUT_SEC="${QWEN_QBIT_RUN_TIMEOUT_SEC:-1800}"
BUILD_MAX_TREE_MB="${QWEN_QBIT_BUILD_MAX_TREE_MB:-8192}"
RUN_MAX_TREE_MB="${QWEN_QBIT_RUN_MAX_TREE_MB:-28672}"
CRYSTAL_CACHE_DIR="${CRYSTAL_CACHE_DIR:-/private/tmp/cogni_ml_qbit_session_probe_cache}"

if [[ ! -x "$RUN_SAFE" || ! -x "$CRYSTAL_BIN" ]]; then
  echo "safe runner or Crystal compiler is not executable" >&2
  exit 2
fi
if [[ ! -f "$ROOT/build/bridge.o" ]]; then
  echo "missing $ROOT/build/bridge.o; run 'make build/bridge.o' first" >&2
  exit 2
fi

export COGNI_RUN_SAFE_MIN_FREE_PCT="$MIN_FREE_PCT"
export CRYSTAL_CACHE_DIR

needs_build=0
if [[ ! -x "$PROBE_BIN" || "${QWEN_QBIT_REBUILD:-0}" == "1" ]]; then
  needs_build=1
elif [[ "$ROOT/bin/qwen35_qbit_session_smoke.cr" -nt "$PROBE_BIN" ]]; then
  needs_build=1
else
  while IFS= read -r source_path; do
    if [[ "$source_path" -nt "$PROBE_BIN" ]]; then
      needs_build=1
      break
    fi
  done < <(find "$ROOT/src" -type f -name '*.cr')
fi

if [[ "$needs_build" == "1" ]]; then
  "$RUN_SAFE" "$CRYSTAL_BIN" "$BUILD_TIMEOUT_SEC" "$BUILD_MAX_TREE_MB" \
    build bin/qwen35_qbit_session_smoke.cr \
    -o "$PROBE_BIN" --release --no-debug --error-trace \
    --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -lc++"
fi

# Compiler RSS and Metal model state are never resident at the same time.
exec "$RUN_SAFE" "$PROBE_BIN" "$RUN_TIMEOUT_SEC" "$RUN_MAX_TREE_MB" "$@"
