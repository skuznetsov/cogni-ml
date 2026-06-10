#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_allowed_head_regression.sh [PROMPT]

Runs the local RPi5/Q6 allowed-head regression gate:
  1. marker smoke for threshold and Q6-off fallback reasons
  2. capture smoke with replay disabled

Set FULL_REPLAY=1 to let the capture smoke contact the Pi, run replay, and run
the resident budget and resident stream gates.

Environment:
  QWEN35_MODEL_PATH  Model path, default Qwen3.5-2B-Q4_K_M in LM Studio cache.
  N_GEN=64           Generation budget.
  FULL_REPLAY=0      Also run Pi replay from the fresh capture.
  RESIDENT_BUDGET_GATE=1
                    With FULL_REPLAY=1, run resident upload budget gate.
  RESIDENT_STREAM_GATE=1
                    With FULL_REPLAY=1, run resident per-step stream gate.
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
resident_budget_gate="${RESIDENT_BUDGET_GATE:-1}"
resident_budget_gate_ran=0
resident_stream_gate="${RESIDENT_STREAM_GATE:-1}"
resident_stream_gate_ran=0

scripts/rpi5_q6_marker_smoke.sh "$prompt"

if [[ "$full_replay" == "1" ]]; then
  RESIDENT_BUDGET_GATE="$resident_budget_gate" \
  RESIDENT_STREAM_GATE="$resident_stream_gate" \
  scripts/rpi5_q6_capture_smoke.sh "$prompt"
  resident_budget_gate_ran="$resident_budget_gate"
  resident_stream_gate_ran="$resident_stream_gate"
else
  SKIP_REPLAY=1 scripts/rpi5_q6_capture_smoke.sh "$prompt"
fi

printf "allowed_head_regression_result\tmarker=ok\tcapture=ok\tfull_replay=%s\tresident_budget_gate=%s\tresident_stream_gate=%s\n" "$full_replay" "$resident_budget_gate_ran" "$resident_stream_gate_ran"
