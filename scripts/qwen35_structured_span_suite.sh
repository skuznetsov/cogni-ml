#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${QWEN35_GENERATE_BIN:-/tmp/qwen35_generate_structured_span_suite}"
RUN_SAFE="${COGNI_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
TIMEOUT="${TIMEOUT:-420}"
RSS_MB="${RSS_MB:-12288}"
LOG_DIR="${LOG_DIR:-/tmp}"
REPS="${REPS:-2}"
GEN="${GEN:-80}"
GATE_MODE="${GATE_MODE:-span}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}_$$"
MODEL_PATH="${QWEN35_MODEL_PATH:-}"
HOST_PATH="${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"
HOST_HOME="${HOME:?HOME is required}"
HOST_TMPDIR="${TMPDIR:-/tmp}"

if ! command -v jq >/dev/null 2>&1; then
  printf 'jq is required to validate parsed tool-call JSON\n' >&2
  exit 1
fi
if [[ ! "$REPS" =~ ^[0-9]+$ ]] || (( REPS < 2 || REPS % 2 != 0 )); then
  printf 'REPS must be an even integer >= 2 for balanced ABBA ordering\n' >&2
  exit 1
fi
if [[ -z "$MODEL_PATH" || ! -r "$MODEL_PATH" ]]; then
  printf 'QWEN35_MODEL_PATH must name a readable model\n' >&2
  exit 1
fi
if [[ "${REBUILD:-1}" != "0" && "${REBUILD:-1}" != "1" ]]; then
  printf 'REBUILD must be 0 or 1\n' >&2
  exit 1
fi
if [[ "$GATE_MODE" != "span" && "$GATE_MODE" != "token-options" && "$GATE_MODE" != "token-stage-span" ]]; then
  printf 'GATE_MODE must be span, token-options, or token-stage-span\n' >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
RUN_DIR="$LOG_DIR/qwen35_span_suite_${RUN_ID}"
if ! mkdir "$RUN_DIR"; then
  printf 'suite run directory already exists: %s\n' "$RUN_DIR" >&2
  exit 1
fi
PAIR_RESULTS="$RUN_DIR/pairs.tsv"
printf 'name\trep\tdecode_speedup_pct\ttotal_speedup_pct\tcandidate_steps\n' >"$PAIR_RESULTS"

SOURCE_INPUT_SHA256="$({
  printf '%s\n' 'build=qwen35_generate release no-debug error-trace bridge+Metal+Foundation+MPS+c++'
  "$CRYSTAL_BIN" --version
  while IFS= read -r path; do
    shasum -a 256 "$path"
  done < <(find bin/qwen35_generate.cr src lib -type f -print | LC_ALL=C sort)
  shasum -a 256 shard.yml shard.lock build/bridge.o
  shasum -a 256 "$ROOT/scripts/qwen35_structured_span_suite.sh" "$RUN_SAFE"
} | shasum -a 256 | awk '{print $1}')"

BUILD_ENV=(
  PATH="$HOST_PATH"
  HOME="$HOST_HOME"
  TMPDIR="$HOST_TMPDIR"
  COGNI_RUN_SAFE_WAIT_QUIET_SEC="${COGNI_RUN_SAFE_WAIT_QUIET_SEC:-0}"
  COGNI_RUN_SAFE_REQUIRE_QUIET="${COGNI_RUN_SAFE_REQUIRE_QUIET:-0}"
  COGNI_RUN_SAFE_MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-35}"
  COGNI_SPEC_MAX_RSS_MB="$RSS_MB"
)

PROVENANCE_MODE=prebuilt-explicit
if [[ ! -x "$BIN" || "${REBUILD:-1}" == "1" ]]; then
  PROVENANCE_MODE=fresh-build
  env -i "${BUILD_ENV[@]}" "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT" "$RSS_MB" \
    build bin/qwen35_generate.cr -o "$BIN" --release --no-debug --error-trace \
    --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
fi

SOURCE_REVISION="$(git rev-parse --verify HEAD)"
BIN_SHA256="$(shasum -a 256 "$BIN" | awk '{print $1}')"
MODEL_BYTES="$(stat -f %z "$MODEL_PATH")"
printf 'suite_config provenance=%s source_revision=%s source_input_sha256=%s binary_sha256=%s model_path=%s model_bytes=%s gate_mode=%s reps=%s gen=%s rss_mb=%s\n' \
  "$PROVENANCE_MODE" "$SOURCE_REVISION" "$SOURCE_INPUT_SHA256" "$BIN_SHA256" "$MODEL_PATH" "$MODEL_BYTES" "$GATE_MODE" "$REPS" "$GEN" "$RSS_MB"

COMMON_ENV=(
  PATH="$HOST_PATH"
  HOME="$HOST_HOME"
  TMPDIR="$HOST_TMPDIR"
  QWEN35_MODEL_PATH="$MODEL_PATH"
  QWEN35_QUIET=1
  QWEN35_DECODE_POLICY=greedy
  QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1
  QWEN35_TOOL_RESPONSE_JSON=simple
  COGNI_RUN_SAFE_WAIT_QUIET_SEC="${COGNI_RUN_SAFE_WAIT_QUIET_SEC:-0}"
  COGNI_RUN_SAFE_REQUIRE_QUIET="${COGNI_RUN_SAFE_REQUIRE_QUIET:-0}"
  COGNI_RUN_SAFE_MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-35}"
  COGNI_SPEC_MAX_RSS_MB="$RSS_MB"
)

run_case() {
  local name="$1" prompt="$2" tools="$3" mode="$4" rep="$5"
  local log="$RUN_DIR/${name}_${mode}_${rep}.log"

  if [[ "$mode" == "off" ]]; then
    if [[ "$GATE_MODE" == "span" ]]; then
      env -i QWEN35_CONSTRAINED_FORCE_SPAN_OFF=1 QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
        "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
    elif [[ "$GATE_MODE" == "token-options" ]]; then
      env -i QWEN35_CONSTRAINED_TOKEN_OPTIONS_OFF=1 QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
        "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
    else
      env -i QWEN35_CONSTRAINED_TOKEN_STAGE_SPAN_OFF=1 QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
        "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
    fi
  else
    if [[ "$GATE_MODE" == "token-stage-span" ]]; then
      env -i QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
        "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
    else
      env -i QWEN35_TOOLS_JSON="$tools" "${COMMON_ENV[@]}" \
        "$RUN_SAFE" "$BIN" 240 "$RSS_MB" "$prompt" "$GEN" >"$log" 2>&1
    fi
  fi

  if rg -q 'Token-option corridor unavailable' "$log"; then
    printf 'token-option fallback is not admissible in suite row: name=%s mode=%s rep=%s log=%s\n' \
      "$name" "$mode" "$rep" "$log" >&2
    return 1
  fi

  local decode total spans token_options stage_transitions free parsed_raw parsed tokens
  decode="$(rg 'greedy summary:' "$log" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' || true)"
  total="$(rg 'request summary:' "$log" | sed -E 's/.*total_ms=([0-9.]+).*/\1/' || true)"
  spans="$(rg 'tool constraint summary:' "$log" | sed -E 's/.*forced_span_steps=([0-9]+).*/\1/' || true)"
  token_options="$(rg 'tool constraint summary:' "$log" | sed -E 's/.*token_option_steps=([0-9]+).*/\1/' || true)"
  stage_transitions="$(rg 'greedy summary:' "$log" | sed -E 's/.*literal_token_stage_span_transitions=([0-9]+).*/\1/' || true)"
  free="$(rg 'tool constraint summary:' "$log" | sed -E 's/.*freeform_value_steps=([0-9]+).*/\1/' || true)"
  parsed_raw="$(awk '/^=== Parsed tool calls ===$/ { getline; print; exit }' "$log")"
  parsed="$(printf '%s\n' "$parsed_raw" | jq -ceS '
    if (type == "array" and length > 0 and
        all(.[]; type == "object" and (.name | type == "string") and (.arguments | type == "object")))
    then .
    else error("invalid parsed tool calls")
    end
  ' 2>/dev/null || true)"
  tokens="$(awk '/^=== Generated token ids ===$/ { getline; print; exit }' "$log")"

  if [[ -z "$decode" || -z "$total" || -z "$spans" || -z "$token_options" || -z "$stage_transitions" || -z "$tokens" || -z "$parsed" ]]; then
    printf 'incomplete suite row: name=%s mode=%s rep=%s log=%s\n' "$name" "$mode" "$rep" "$log" >&2
    return 1
  fi

  printf 'suite_row name=%s mode=%s rep=%s decode_ms=%s total_ms=%s forced_span=%s token_options=%s stage_transitions=%s freeform=%s parsed=%s log=%s\n' \
    "$name" "$mode" "$rep" "$decode" "$total" "$spans" "$token_options" "$stage_transitions" "${free:-na}" "$parsed" "$log"

  RUN_DECODE="$decode"
  RUN_TOTAL="$total"
  RUN_SPANS="$spans"
  RUN_TOKEN_OPTIONS="$token_options"
  RUN_STAGE_TRANSITIONS="$stage_transitions"
  RUN_PARSED="$parsed"
  RUN_TOKENS="$tokens"
}

run_pair() {
  local name="$1" prompt="$2" tools="$3" rep="$4" expected_raw="$5"
  local expected_parsed
  expected_parsed="$(printf '%s\n' "$expected_raw" | jq -ceS '.')"
  local first=default second=off
  if (( rep % 2 == 0 )); then
    first=off
    second=default
  fi

  local default_decode default_total default_spans default_token_options default_stage_transitions default_parsed default_tokens
  local off_decode off_total off_spans off_token_options off_stage_transitions off_parsed off_tokens

  run_case "$name" "$prompt" "$tools" "$first" "$rep"
  if [[ "$first" == "default" ]]; then
    default_decode="$RUN_DECODE"; default_total="$RUN_TOTAL"; default_spans="$RUN_SPANS"; default_token_options="$RUN_TOKEN_OPTIONS"; default_stage_transitions="$RUN_STAGE_TRANSITIONS"
    default_parsed="$RUN_PARSED"; default_tokens="$RUN_TOKENS"
  else
    off_decode="$RUN_DECODE"; off_total="$RUN_TOTAL"; off_spans="$RUN_SPANS"; off_token_options="$RUN_TOKEN_OPTIONS"; off_stage_transitions="$RUN_STAGE_TRANSITIONS"
    off_parsed="$RUN_PARSED"; off_tokens="$RUN_TOKENS"
  fi

  run_case "$name" "$prompt" "$tools" "$second" "$rep"
  if [[ "$second" == "default" ]]; then
    default_decode="$RUN_DECODE"; default_total="$RUN_TOTAL"; default_spans="$RUN_SPANS"; default_token_options="$RUN_TOKEN_OPTIONS"; default_stage_transitions="$RUN_STAGE_TRANSITIONS"
    default_parsed="$RUN_PARSED"; default_tokens="$RUN_TOKENS"
  else
    off_decode="$RUN_DECODE"; off_total="$RUN_TOTAL"; off_spans="$RUN_SPANS"; off_token_options="$RUN_TOKEN_OPTIONS"; off_stage_transitions="$RUN_STAGE_TRANSITIONS"
    off_parsed="$RUN_PARSED"; off_tokens="$RUN_TOKENS"
  fi

  if [[ "$default_tokens" != "$off_tokens" || "$default_parsed" != "$off_parsed" ]]; then
    printf 'parity failure: name=%s rep=%s\n' "$name" "$rep" >&2
    return 1
  fi
  if [[ "$default_parsed" != "$expected_parsed" ]]; then
    printf 'expected tool-call failure: name=%s rep=%s expected=%s actual=%s\n' \
      "$name" "$rep" "$expected_parsed" "$default_parsed" >&2
    return 1
  fi
  local candidate_steps off_steps
  if [[ "$GATE_MODE" == "span" ]]; then
    candidate_steps="$default_spans"
    off_steps="$off_spans"
  elif [[ "$GATE_MODE" == "token-options" ]]; then
    candidate_steps="$default_token_options"
    off_steps="$off_token_options"
  else
    candidate_steps="$default_stage_transitions"
    off_steps="$off_stage_transitions"
  fi
  if [[ ! "$candidate_steps" =~ ^[0-9]+$ || ! "$off_steps" =~ ^[0-9]+$ ]] ||
     (( candidate_steps <= 0 || off_steps != 0 )); then
    printf '%s activation failure: name=%s rep=%s candidate=%s off=%s\n' \
      "$GATE_MODE" "$name" "$rep" "$candidate_steps" "$off_steps" >&2
    return 1
  fi

  local decode_speedup total_speedup
  decode_speedup="$(awk -v candidate="$default_decode" -v baseline="$off_decode" \
    'BEGIN { if (baseline <= 0) exit 1; printf "%.3f", 100.0 * (baseline - candidate) / baseline }')"
  total_speedup="$(awk -v candidate="$default_total" -v baseline="$off_total" \
    'BEGIN { if (baseline <= 0) exit 1; printf "%.3f", 100.0 * (baseline - candidate) / baseline }')"

  printf 'suite_pair name=%s rep=%s order=%s-%s parity=exact gate_mode=%s decode_speedup_pct=%s total_speedup_pct=%s candidate_steps=%s\n' \
    "$name" "$rep" "$first" "$second" "$GATE_MODE" "$decode_speedup" "$total_speedup" "$candidate_steps"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$rep" "$decode_speedup" "$total_speedup" "$candidate_steps" >>"$PAIR_RESULTS"
}

edit_tools='[{"type":"function","function":{"name":"edit_mode","description":"Choose edit mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]},"dry_run":{"type":"boolean"}},"required":["mode","dry_run"]}}}]'
read_tools='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":5}},"required":["path","limit"]}}}]'
optional_tools='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":5},"exact":{"type":"boolean"}},"required":["path"]}}}]'
multi_tools='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"edit_mode","description":"Choose edit mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]}},"required":["mode"]}}}]'

for rep in $(seq 1 "$REPS"); do
  run_pair edit "Set edit mode to safe and dry run true." "$edit_tools" "$rep" \
    '[{"name":"edit_mode","arguments":{"mode":"safe","dry_run":true}}]'
  run_pair read "Read README.md with limit 3." "$read_tools" "$rep" \
    '[{"name":"read_file","arguments":{"path":"README.md","limit":3}}]'
  run_pair optional_select "Read README.md with limit 3." "$optional_tools" "$rep" \
    '[{"name":"read_file","arguments":{"path":"README.md","limit":3}}]'
  run_pair multi "Use edit mode safe." "$multi_tools" "$rep" \
    '[{"name":"edit_mode","arguments":{"mode":"safe"}}]'
done

awk -F '\t' '
  NR == 1 { next }
  count == 0 { min_decode = max_decode = $3; min_total = max_total = $4 }
  {
    count += 1
    sum_decode += $3
    sum_total += $4
    if ($3 < min_decode) min_decode = $3
    if ($3 > max_decode) max_decode = $3
    if ($4 < min_total) min_total = $4
    if ($4 > max_total) max_total = $4
  }
  END {
    if (count == 0) exit 1
    printf "suite_summary pairs=%d parity=exact decode_speedup_mean_pct=%.3f decode_speedup_range_pct=%.3f..%.3f total_speedup_mean_pct=%.3f total_speedup_range_pct=%.3f..%.3f\n", count, sum_decode / count, min_decode, max_decode, sum_total / count, min_total, max_total
  }
' "$PAIR_RESULTS"
printf 'suite_results path=%s\n' "$PAIR_RESULTS"
