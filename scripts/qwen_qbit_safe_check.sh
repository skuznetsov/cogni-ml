#!/usr/bin/env bash
# Focused, fail-closed verification for the experimental Qwen QBit cache path.
# This script intentionally accepts no spec-path arguments: widening it to the
# full Metal suite would defeat its resource-safety boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_SAFE="${COGNI_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
TIMEOUT_SEC="${QWEN_QBIT_SPEC_TIMEOUT_SEC:-420}"
MAX_TREE_MB="${QWEN_QBIT_SPEC_MAX_TREE_MB:-6144}"
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
SPEC_MAX_RSS_MB="${COGNI_SPEC_MAX_RSS_MB:-4096}"
CRYSTAL_CACHE_DIR="${CRYSTAL_CACHE_DIR:-/private/tmp/cogni_ml_qbit_safe_spec_cache}"

if [[ ! -x "$RUN_SAFE" ]]; then
  echo "missing executable safe runner: $RUN_SAFE" >&2
  exit 2
fi
if [[ ! -x "$CRYSTAL_BIN" ]]; then
  echo "missing Crystal compiler: $CRYSTAL_BIN" >&2
  exit 2
fi
if [[ ! -f "$ROOT/build/bridge.o" ]]; then
  echo "missing $ROOT/build/bridge.o; run 'make build/bridge.o' first" >&2
  exit 2
fi

export COGNI_RUN_SAFE_MIN_FREE_PCT="$MIN_FREE_PCT"
export COGNI_SPEC_MIN_FREE_PCT="$MIN_FREE_PCT"
export COGNI_SPEC_MAX_RSS_MB="$SPEC_MAX_RSS_MB"
export CRYSTAL_CACHE_DIR

exec "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT_SEC" "$MAX_TREE_MB" \
  spec \
  spec/qwen_qbit_gaussian_codec_spec.cr \
  spec/qwen_qbit_native_writer_spec.cr \
  spec/qwen_qbit_metal_restore_spec.cr \
  spec/qwen_qbit_native_restore_spec.cr \
  spec/qwen_qbit_state_snapshot_spec.cr \
  --error-trace \
  --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -lc++"
