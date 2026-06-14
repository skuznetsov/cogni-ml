#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

timestamp="$(date +%Y%m%d%H%M%S)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_prompt_artifact_suite_promotion_${timestamp}}"
prepare_dir="${SUITE_PREPARE_LOG_DIR:-$log_dir/prepare}"
gate_dir="${SUITE_GATE_LOG_DIR:-$log_dir/gate}"
artifact_dir="${SUITE_ARTIFACT_DIR:-$log_dir/artifacts}"

token_windows="${TOKEN_WINDOWS:-1:0,17:100,257:1000,4096:8192}"
prompt_len="${PROMPT_LEN:-16}"
canvas_len="${CANVAS_LEN:-8}"
max_layers="${MAX_LAYERS:-30}"
artifact_arms="${SUITE_ARTIFACT_ARMS:-variant}"
overwrite="${SUITE_ARTIFACT_OVERWRITE:-0}"
variant_profile="${VARIANT_PROFILE:-}"
prompt_ffn_resident_chunk_rows="${PROMPT_FFN_RESIDENT_GRAPH_CHUNK_ROWS:-8}"
base_extra_env="${BASE_EXTRA_ENV:-}"
variant_extra_env="${VARIANT_EXTRA_ENV:-}"

check_quiet="${CHECK_QUIET:-1}"
gate_check_quiet="${GATE_CHECK_QUIET:-$check_quiet}"
quiet_ms="${QUIET_MS:-3000}"
load_threshold="${LOAD_THRESHOLD:-40}"
total_threshold="${TOTAL_THRESHOLD:-${LOAD_TOTAL_THRESHOLD:-240}}"

suite_min_total_speedup="${SUITE_MIN_TOTAL_SPEEDUP:-${MIN_TOTAL_SPEEDUP:-1.10}}"
suite_window_min_total_speedup="${SUITE_WINDOW_MIN_TOTAL_SPEEDUP:-$suite_min_total_speedup}"
certificate_mode="${CERTIFICATE_MODE:-full-vocab-top1-metal}"
abba_warmups="${ABBA_WARMUPS:-1}"
abba_repeats="${ABBA_REPEATS:-3}"
abba_trim_per_arm="${ABBA_TRIM_PER_ARM:-1}"
abba_sequence="${ABBA_SEQUENCE:-base variant variant base}"
abba_mirror_sequence="${ABBA_MIRROR_SEQUENCE:-variant base base variant}"

base_map="${SUITE_BASE_ROUTE_ARTIFACT_MAP:-${CERT_BASE_ROUTE_ARTIFACT_MAP:-}}"
variant_map="${SUITE_VARIANT_ROUTE_ARTIFACT_MAP:-${CERT_VARIANT_ROUTE_ARTIFACT_MAP:-}}"
dry_run="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: diffusion_gemma_prompt_artifact_suite_promotion.sh

Fail-closed driver for the route-artifact prompt-suite promotion path:
  1. quiet precheck,
  2. shared-load route-artifact preparation unless maps are supplied,
  3. artifact-backed certified suite gate,
  4. one final artifact_suite_promotion decision row.

Important environment knobs:
  TOKEN_WINDOWS                         prompt:canvas windows, default four windows
  PROMPT_LEN / CANVAS_LEN / MAX_LAYERS  shape, defaults 16 / 8 / 30
  CHECK_QUIET=1                         run quiet precheck before model work
  GATE_CHECK_QUIET=1                    run quiet gate inside each child gate
  LOAD_THRESHOLD=40                     per-process quiet threshold
  LOAD_TOTAL_THRESHOLD=240              total quiet threshold fallback
  SUITE_ARTIFACT_ARMS=variant           variant, base, or base,variant
  SUITE_ARTIFACT_OVERWRITE=1            allow prepare to overwrite artifacts
  VARIANT_PROFILE=prompt-ffn-resident   append the certified prompt FFN resident profile
  PROMPT_FFN_RESIDENT_GRAPH_CHUNK_ROWS=8
                                         chunk size for that profile, default 8
  BASE_EXTRA_ENV / VARIANT_EXTRA_ENV    append profile envs to prepare and gate arms
  SUITE_*_ROUTE_ARTIFACT_MAP=SPEC       skip prepare and use existing maps
  SUITE_MIN_TOTAL_SPEEDUP=1.10          aggregate speedup floor
  SUITE_WINDOW_MIN_TOTAL_SPEEDUP=...    per-window speedup floor
  CERTIFICATE_MODE=full-vocab-top1-metal
  DRY_RUN=1                             print commands without running model work
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

bool_enabled() {
  case "$1" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *) die "invalid boolean value: $1" ;;
  esac
}

validate_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    die "$name must be a non-negative integer"
  fi
}

validate_positive_uint() {
  local name="$1"
  local value="$2"
  validate_uint "$name" "$value"
  if [[ "$value" -lt 1 ]]; then
    die "$name must be positive"
  fi
}

print_cmd() {
  local label="$1"
  shift
  printf '%s=' "$label"
  printf '%q ' "$@"
  printf '\n'
}

extract_map() {
  local key="$1"
  local file="$2"
  awk -v key="$key" -F= '$1 == key {print substr($0, length(key) + 2); found = 1} END {if (!found) exit 1}' "$file"
}

validate_positive_uint PROMPT_LEN "$prompt_len"
validate_positive_uint CANVAS_LEN "$canvas_len"
validate_positive_uint MAX_LAYERS "$max_layers"
validate_positive_uint PROMPT_FFN_RESIDENT_GRAPH_CHUNK_ROWS "$prompt_ffn_resident_chunk_rows"
bool_enabled "$check_quiet" >/dev/null || true
bool_enabled "$gate_check_quiet" >/dev/null || true
bool_enabled "$overwrite" >/dev/null || true
bool_enabled "$dry_run" >/dev/null || true
[[ -n "$token_windows" ]] || die "TOKEN_WINDOWS is required"

case "$variant_profile" in
  "")
    ;;
  prompt-ffn-resident)
    profile_env="DIFFUSION_GEMMA_PROMPT_FFN_RESIDENT_GRAPH=1 DIFFUSION_GEMMA_PROMPT_FFN_RESIDENT_GRAPH_MIN_ROWS=1 DIFFUSION_GEMMA_PROMPT_FFN_RESIDENT_GRAPH_CHUNK_ROWS=$prompt_ffn_resident_chunk_rows"
    if [[ -n "$variant_extra_env" ]]; then
      variant_extra_env="$profile_env $variant_extra_env"
    else
      variant_extra_env="$profile_env"
    fi
    ;;
  *)
    die "unsupported VARIANT_PROFILE: $variant_profile"
    ;;
esac

mkdir -p "$log_dir"
manifest="$log_dir/promotion_manifest.env"
{
  printf 'log_dir=%q\n' "$log_dir"
  printf 'prepare_dir=%q\n' "$prepare_dir"
  printf 'gate_dir=%q\n' "$gate_dir"
  printf 'artifact_dir=%q\n' "$artifact_dir"
  printf 'token_windows=%q\n' "$token_windows"
  printf 'prompt_len=%q\n' "$prompt_len"
  printf 'canvas_len=%q\n' "$canvas_len"
  printf 'max_layers=%q\n' "$max_layers"
  printf 'artifact_arms=%q\n' "$artifact_arms"
  printf 'variant_profile=%q\n' "$variant_profile"
  printf 'prompt_ffn_resident_chunk_rows=%q\n' "$prompt_ffn_resident_chunk_rows"
  printf 'base_extra_env=%q\n' "$base_extra_env"
  printf 'variant_extra_env=%q\n' "$variant_extra_env"
  printf 'check_quiet=%q\n' "$check_quiet"
  printf 'gate_check_quiet=%q\n' "$gate_check_quiet"
  printf 'quiet_ms=%q\n' "$quiet_ms"
  printf 'load_threshold=%q\n' "$load_threshold"
  printf 'total_threshold=%q\n' "$total_threshold"
  printf 'suite_min_total_speedup=%q\n' "$suite_min_total_speedup"
  printf 'suite_window_min_total_speedup=%q\n' "$suite_window_min_total_speedup"
  printf 'certificate_mode=%q\n' "$certificate_mode"
} >"$manifest"

printf 'artifact_suite_promotion_start log_dir=%s manifest=%s\n' "$log_dir" "$manifest"

quiet_log="$log_dir/quiet_precheck.log"
quiet_cmd=(
  env
  CHECK_QUIET_ONLY=1
  LOAD_THRESHOLD="$load_threshold"
  TOTAL_THRESHOLD="$total_threshold"
  QUIET_MS="$quiet_ms"
  SUMMARY_FORMAT=json
  "$repo_root/scripts/diffusion_gemma_quiet_gate_check.sh"
)

prepare_stdout="$log_dir/prepare.stdout"
prepare_stderr="$log_dir/prepare.stderr"
prepare_cmd=(
  env
  LOG_DIR="$prepare_dir"
  SUITE_ARTIFACT_DIR="$artifact_dir"
  TOKEN_WINDOWS="$token_windows"
  PROMPT_LEN="$prompt_len"
  CANVAS_LEN="$canvas_len"
  MAX_LAYERS="$max_layers"
  SUITE_ARTIFACT_ARMS="$artifact_arms"
  SUITE_ARTIFACT_OVERWRITE="$overwrite"
  BASE_EXTRA_ENV="$base_extra_env"
  VARIANT_EXTRA_ENV="$variant_extra_env"
  "$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_prepare.sh"
)

gate_stdout="$log_dir/gate.stdout"
gate_stderr="$log_dir/gate.stderr"
gate_cmd=(
  env
  LOG_DIR="$gate_dir"
  TOKEN_WINDOWS="$token_windows"
  PROMPT_LEN="$prompt_len"
  CANVAS_LEN="$canvas_len"
  MAX_LAYERS="$max_layers"
  CERTIFICATE_MODE="$certificate_mode"
  SUITE_MIN_TOTAL_SPEEDUP="$suite_min_total_speedup"
  SUITE_WINDOW_MIN_TOTAL_SPEEDUP="$suite_window_min_total_speedup"
  ABBA_WARMUPS="$abba_warmups"
  ABBA_REPEATS="$abba_repeats"
  ABBA_TRIM_PER_ARM="$abba_trim_per_arm"
  ABBA_SEQUENCE="$abba_sequence"
  ABBA_MIRROR_SEQUENCE="$abba_mirror_sequence"
  CHECK_QUIET="$gate_check_quiet"
  LOAD_THRESHOLD="$load_threshold"
  TOTAL_THRESHOLD="$total_threshold"
  QUIET_MS="$quiet_ms"
  BASE_EXTRA_ENV="$base_extra_env"
  VARIANT_EXTRA_ENV="$variant_extra_env"
)

if [[ -n "$base_map" ]]; then
  gate_cmd+=(SUITE_BASE_ROUTE_ARTIFACT_MAP="$base_map" CERT_BASE_ROUTE_ARTIFACT_MAP="$base_map")
fi
if [[ -n "$variant_map" ]]; then
  gate_cmd+=(SUITE_VARIANT_ROUTE_ARTIFACT_MAP="$variant_map" CERT_VARIANT_ROUTE_ARTIFACT_MAP="$variant_map")
fi
gate_cmd+=("$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_gate.sh")

if bool_enabled "$dry_run"; then
  print_cmd quiet_cmd "${quiet_cmd[@]}"
  print_cmd prepare_cmd "${prepare_cmd[@]}"
  print_cmd gate_cmd "${gate_cmd[@]}"
  printf 'artifact_suite_promotion decision=dry_run log_dir=%s\n' "$log_dir"
  exit 0
fi

if bool_enabled "$check_quiet"; then
  set +e
  "${quiet_cmd[@]}" >"$quiet_log" 2>&1
  quiet_rc=$?
  set -e
  cat "$quiet_log"
  if (( quiet_rc != 0 )); then
    printf 'artifact_suite_promotion decision=blocked reason=quiet_gate_failed rc=%s log=%s\n' "$quiet_rc" "$quiet_log"
    exit "$quiet_rc"
  fi
fi

if [[ -z "$base_map" && -z "$variant_map" ]]; then
  set +e
  "${prepare_cmd[@]}" >"$prepare_stdout" 2>"$prepare_stderr"
  prepare_rc=$?
  set -e
  cat "$prepare_stdout"
  if (( prepare_rc != 0 )); then
    printf 'artifact_suite_promotion decision=reject reason=prepare_failed rc=%s log=%s stderr=%s\n' "$prepare_rc" "$prepare_stdout" "$prepare_stderr"
    exit "$prepare_rc"
  fi

  if base_map_value="$(extract_map SUITE_BASE_ROUTE_ARTIFACT_MAP "$prepare_stdout" 2>/dev/null)"; then
    base_map="$base_map_value"
    gate_cmd=("${gate_cmd[@]:0:${#gate_cmd[@]}-1}" SUITE_BASE_ROUTE_ARTIFACT_MAP="$base_map" CERT_BASE_ROUTE_ARTIFACT_MAP="$base_map" "$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_gate.sh")
  fi
  if variant_map_value="$(extract_map SUITE_VARIANT_ROUTE_ARTIFACT_MAP "$prepare_stdout" 2>/dev/null)"; then
    variant_map="$variant_map_value"
    gate_cmd=("${gate_cmd[@]:0:${#gate_cmd[@]}-1}" SUITE_VARIANT_ROUTE_ARTIFACT_MAP="$variant_map" CERT_VARIANT_ROUTE_ARTIFACT_MAP="$variant_map" "$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_gate.sh")
  fi
fi

set +e
"${gate_cmd[@]}" >"$gate_stdout" 2>"$gate_stderr"
gate_rc=$?
set -e
cat "$gate_stdout"

decision="$(grep -E '^artifact_suite_gate decision=' "$gate_stdout" | tail -1 || true)"
if (( gate_rc != 0 )); then
  printf 'artifact_suite_promotion decision=reject reason=gate_failed rc=%s log=%s stderr=%s child_decision=%q\n' "$gate_rc" "$gate_stdout" "$gate_stderr" "$decision"
  exit "$gate_rc"
fi

prepare_log="<skipped>"
if [[ -f "$prepare_stdout" ]]; then
  prepare_log="$prepare_stdout"
fi
printf 'artifact_suite_promotion decision=candidate log_dir=%s prepare_log=%s gate_log=%s child_decision=%q\n' "$log_dir" "$prepare_log" "$gate_stdout" "$decision"
