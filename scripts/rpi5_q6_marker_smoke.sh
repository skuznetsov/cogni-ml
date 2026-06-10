#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_marker_smoke.sh [PROMPT]

Runs two local Qwen3.5-2B structured constrained smokes and verifies
allowed-head profile markers:
  1. CPU threshold policy: QWEN35_ALLOWED_HEAD_CPU_MAX=7
  2. Q6 kill switch:     QWEN35_ALLOWED_HEAD_Q6_OFF=1

Environment:
  QWEN35_MODEL_PATH  Model path, default Qwen3.5-2B-Q4_K_M in LM Studio cache.
  N_GEN=64           Generation budget.
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
tools='[{"type":"function","function":{"name":"edit_mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]},"dry_run":{"type":"boolean"}},"required":["mode","dry_run"]}}}]'

[[ -f "$model_path" ]] || {
  echo "model not found: $model_path" >&2
  exit 1
}

run_case() {
  local label="$1"
  shift
  local log_path="/tmp/qwen35_${label}_marker_$(date +%Y%m%d_%H%M%S).log"

  env "$@" \
    QWEN35_MODEL_PATH="$model_path" \
    QWEN35_TOOLS_JSON="$tools" \
    QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1 \
    QWEN35_METAL_PROFILE=1 \
    QWEN35_TOOL_RESPONSE_JSON=simple \
    QWEN35_QUIET=1 \
    scripts/run_safe.sh /opt/homebrew/bin/crystal 900 6000 run bin/qwen35_generate.cr \
      --link-flags="$repo_root/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++" \
      -- "$prompt" "$n_gen" >"$log_path" 2>&1

  printf "marker_smoke_log[%s]=%s\n" "$label" "$log_path"
  rg -n 'allowed_head\.|Tool response JSON|tool_calls|request summary|Exception|ERROR' "$log_path" | tail -80 || true
  printf "%s\n" "$log_path"
}

threshold_output="$(run_case threshold QWEN35_ALLOWED_HEAD_CPU_MAX=7)"
threshold_log="$(tail -1 <<<"$threshold_output")"
printf "%s\n" "$threshold_output" | sed '$d'

q6_off_output="$(run_case q6_off QWEN35_ALLOWED_HEAD_Q6_OFF=1)"
q6_off_log="$(tail -1 <<<"$q6_off_output")"
printf "%s\n" "$q6_off_output" | sed '$d'

expect_in_log() {
  local log="$1"
  local pattern="$2"
  if ! rg -q "$pattern" "$log"; then
    echo "missing expected pattern in $log: $pattern" >&2
    exit 1
  fi
}

expect_fixed_in_log() {
  local log="$1"
  local pattern="$2"
  if ! rg -F -q "$pattern" "$log"; then
    echo "missing expected text in $log: $pattern" >&2
    exit 1
  fi
}

expect_in_log "$threshold_log" 'allowed_head\.cpu_threshold'
expect_in_log "$threshold_log" 'allowed_head\.cpu_selected_metal_hidden'
expect_in_log "$threshold_log" 'allowed_head\.metal_q6'
expect_fixed_in_log "$threshold_log" '"tool_calls":[{"name":"edit_mode","arguments":{"mode":"safe","dry_run":true}}]'

expect_in_log "$q6_off_log" 'allowed_head\.q6_off'
expect_in_log "$q6_off_log" 'allowed_head\.cpu_selected_metal_hidden'
expect_fixed_in_log "$q6_off_log" '"tool_calls":[{"name":"edit_mode","arguments":{"mode":"safe","dry_run":true}}]'

printf "marker_smoke_result\tthreshold=ok\tq6_off=ok\n"
