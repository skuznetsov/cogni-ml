#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_capture_smoke.sh [PROMPT]

Runs a small Qwen3.5-2B constrained tool-call generation with
QWEN35_ALLOWED_HEAD_CAPTURE_PATH, then optionally replays the captured real
hidden rows on the Raspberry Pi 5 Q6 indexed-head probe.

Environment:
  QWEN35_MODEL_PATH  Model path, default Qwen3.5-2B-Q4_K_M in LM Studio cache.
  N_GEN=64           Generation budget.
  MIN_ALLOWED=4      Replay filter threshold for V3D rows.
  REPEATS=30         Pi replay repeats.
  RPI5_WARMUPS=3     Untimed Pi GPU dispatches before measurement.
  SKIP_REPLAY=1      Only generate capture; do not contact the Pi.
  RESIDENT_BUDGET_GATE=1
                    Also run resident upload budget gate after replay.
  RESIDENT_STREAM_GATE=1
                    Also run resident per-step stream gate after replay.
USAGE
  exit 2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

prompt="${1:-Set edit mode safe and dry run true.}"
model_path="${QWEN35_MODEL_PATH:-$HOME/.cache/lm-studio/models/lmstudio-community/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf}"
n_gen="${N_GEN:-64}"
min_allowed="${MIN_ALLOWED:-4}"
repeats="${REPEATS:-30}"
warmups="${RPI5_WARMUPS:-3}"
skip_replay="${SKIP_REPLAY:-0}"
resident_budget_gate="${RESIDENT_BUDGET_GATE:-0}"
resident_stream_gate="${RESIDENT_STREAM_GATE:-0}"

[[ -f "$model_path" ]] || {
  echo "model not found: $model_path" >&2
  exit 1
}

tools='[{"type":"function","function":{"name":"edit_mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]},"dry_run":{"type":"boolean"}},"required":["mode","dry_run"]}}}]'
stamp="$(date +%Y%m%d_%H%M%S)"
log_path="/tmp/qwen35_2b_frontier_capture_${stamp}.log"
capture_path="/tmp/qwen35_2b_allowed_head_capture_${stamp}.jsonl"

QWEN35_MODEL_PATH="$model_path" \
QWEN35_TOOLS_JSON="$tools" \
QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1 \
QWEN35_CONSTRAINT_FRONTIER_TRACE=1 \
QWEN35_ALLOWED_HEAD_CAPTURE_PATH="$capture_path" \
QWEN35_TOOL_RESPONSE_JSON=simple \
QWEN35_QUIET=1 \
scripts/run_safe.sh /opt/homebrew/bin/crystal 900 6000 run bin/qwen35_generate.cr \
  --link-flags="$repo_root/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++" \
  -- "$prompt" "$n_gen" >"$log_path" 2>&1

trace_rows="$(rg -c 'constraint frontier trace' "$log_path" || true)"
capture_rows="$(wc -l < "$capture_path" 2>/dev/null | tr -d ' ' || printf 0)"

printf "capture_smoke_log=%s\n" "$log_path"
printf "capture_smoke_capture=%s\n" "$capture_path"
printf "trace_rows=%s\n" "$trace_rows"
printf "capture_rows=%s\n" "$capture_rows"
rg -n 'Tool response JSON|tool_calls|request summary|Exception|ERROR' "$log_path" | tail -30 || true

if [[ "$trace_rows" == "0" || "$capture_rows" == "0" ]]; then
  echo "capture smoke produced no trace/capture rows" >&2
  exit 1
fi

if [[ "$skip_replay" == "1" ]]; then
  exit 0
fi

REPEATS="$repeats" \
RPI5_WARMUPS="$warmups" \
RAW_OUTPUT="${RAW_OUTPUT:-0}" \
bash scripts/rpi5_q6_capture_replay.sh "$capture_path" "$min_allowed"

if [[ "$resident_budget_gate" == "1" ]]; then
  REPEATS="$repeats" \
  RPI5_WARMUPS="$warmups" \
  scripts/rpi5_q6_resident_budget_gate.sh "$capture_path" "$min_allowed"
fi

if [[ "$resident_stream_gate" == "1" ]]; then
  REPEATS="$repeats" \
  scripts/rpi5_q6_resident_stream_gate.sh "$capture_path" "$min_allowed"
fi
