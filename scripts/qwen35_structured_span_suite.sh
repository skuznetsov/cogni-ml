#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${QWEN35_GENERATE_BIN:-/tmp/qwen35_generate_structured_span_suite}"
RUN_SAFE="${COGNI_RUN_SAFE:-../crystal_v2_repo/scripts/run_safe.sh}"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
TIMEOUT="${TIMEOUT:-420}"
RSS_MB="${RSS_MB:-12288}"
LOG_DIR="${LOG_DIR:-/tmp}"
REPS="${REPS:-2}"
GEN="${GEN:-80}"

if [[ ! -x "$BIN" || "${REBUILD:-0}" == "1" ]]; then
  COGNI_RUN_SAFE_MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}" \
  COGNI_SPEC_MAX_RSS_MB="$RSS_MB" \
  "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT" "$RSS_MB" \
    build bin/qwen35_generate.cr -o "$BIN" --release --no-debug --error-trace \
    --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
fi

COMMON_ENV=(
  QWEN35_QUIET=1
  QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1
  QWEN35_TOOL_RESPONSE_JSON=simple
  COGNI_RUN_SAFE_MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
  COGNI_SPEC_MAX_RSS_MB="$RSS_MB"
)

run_case() {
  local name="$1" prompt="$2" tools="$3" mode="$4" rep="$5"
  local log="$LOG_DIR/qwen35_span_suite_${name}_${mode}_${rep}_$(date +%Y%m%d_%H%M%S).log"

  if [[ "$mode" == "off" ]]; then
    env QWEN35_CONSTRAINED_FORCE_SPAN_OFF=1 QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
      "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
  else
    env QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
      "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
  fi

  local decode total spans free parsed
  decode="$(rg 'greedy summary:' "$log" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' || true)"
  total="$(rg 'request summary:' "$log" | sed -E 's/.*total_ms=([0-9.]+).*/\1/' || true)"
  spans="$(rg 'tool constraint summary:' "$log" | sed -E 's/.*forced_span_steps=([0-9]+).*/\1/' || true)"
  free="$(rg 'tool constraint summary:' "$log" | sed -E 's/.*freeform_value_steps=([0-9]+).*/\1/' || true)"
  parsed="$(rg -m1 -F '[{"name"' "$log" || true)"

  printf 'suite_row name=%s mode=%s rep=%s decode_ms=%s total_ms=%s forced_span=%s freeform=%s parsed=%s log=%s\n' \
    "$name" "$mode" "$rep" "$decode" "$total" "${spans:-na}" "${free:-na}" "${parsed:-MISSING}" "$log"
}

edit_tools='[{"type":"function","function":{"name":"edit_mode","description":"Choose edit mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]},"dry_run":{"type":"boolean"}},"required":["mode","dry_run"]}}}]'
read_tools='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":5}},"required":["path","limit"]}}}]'
optional_tools='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":5},"exact":{"type":"boolean"}},"required":["path"]}}}]'
multi_tools='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"edit_mode","description":"Choose edit mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]}},"required":["mode"]}}}]'

for rep in $(seq 1 "$REPS"); do
  run_case edit "Set edit mode to safe and dry run true." "$edit_tools" default "$rep"
  run_case edit "Set edit mode to safe and dry run true." "$edit_tools" off "$rep"
  run_case read "Read README.md with limit 3." "$read_tools" default "$rep"
  run_case read "Read README.md with limit 3." "$read_tools" off "$rep"
  run_case optional_select "Read README.md with limit 3." "$optional_tools" default "$rep"
  run_case optional_select "Read README.md with limit 3." "$optional_tools" off "$rep"
  run_case multi "Use edit mode safe." "$multi_tools" default "$rep"
  run_case multi "Use edit mode safe." "$multi_tools" off "$rep"
done
