#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_allowed_head_regression.sh [PROMPT]

Runs the local RPi5/Q6 allowed-head regression gate:
  1. marker smoke for threshold and Q6-off fallback reasons
  2. capture smoke with replay disabled

Set FULL_REPLAY=1 to let the capture smoke contact the Pi and run replay.

Environment:
  QWEN35_MODEL_PATH  Model path, default Qwen3.5-2B-Q4_K_M in LM Studio cache.
  N_GEN=64           Generation budget.
  FULL_REPLAY=0      Also run Pi replay from the fresh capture.
USAGE
  exit 2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

prompt="${1:-Set edit mode safe and dry run true.}"
full_replay="${FULL_REPLAY:-0}"

scripts/rpi5_q6_marker_smoke.sh "$prompt"

if [[ "$full_replay" == "1" ]]; then
  scripts/rpi5_q6_capture_smoke.sh "$prompt"
else
  SKIP_REPLAY=1 scripts/rpi5_q6_capture_smoke.sh "$prompt"
fi

printf "allowed_head_regression_result\tmarker=ok\tcapture=ok\tfull_replay=%s\n" "$full_replay"
