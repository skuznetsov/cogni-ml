#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d%H%M%S)"

mixed_route_plan="${MIXED_ROUTE_PLAN:-${SUITE_MIXED_ROUTE_PLAN:-}}"
stage="${FALLBACK_REPLAY_STAGE:-all}"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_fallback_replay_${timestamp}}"
prepare_dir="${FALLBACK_PREPARE_LOG_DIR:-$log_dir/prepare}"
gate_dir="${FALLBACK_GATE_LOG_DIR:-$log_dir/gate}"
artifact_dir="${FALLBACK_ARTIFACT_DIR:-$log_dir/artifacts}"
attached_plan="${FALLBACK_ATTACHED_ROUTE_PLAN:-$log_dir/route_plan_with_base_fallback.jsonl}"
prepare_log="${FALLBACK_PREPARE_LOG:-}"
base_map="${FALLBACK_BASE_ROUTE_ARTIFACT_MAP:-${SUITE_BASE_ROUTE_ARTIFACT_MAP:-${CERT_BASE_ROUTE_ARTIFACT_MAP:-}}}"

fallback_windows="${FALLBACK_TOKEN_WINDOWS:-}"
plan_windows="${PLAN_TOKEN_WINDOWS:-}"
prompt_len="${PROMPT_LEN:-16}"
canvas_len="${CANVAS_LEN:-8}"
max_layers="${MAX_LAYERS:-30}"
variant_profile="${VARIANT_PROFILE:-prompt-ffn-resident}"

check_quiet="${CHECK_QUIET:-1}"
gate_check_quiet="${GATE_CHECK_QUIET:-$check_quiet}"
quiet_ms="${QUIET_MS:-3000}"
load_threshold="${LOAD_THRESHOLD:-40}"
total_threshold="${TOTAL_THRESHOLD:-${LOAD_TOTAL_THRESHOLD:-240}}"
overwrite="${SUITE_ARTIFACT_OVERWRITE:-0}"
dry_run="${DRY_RUN:-0}"
summary_enabled="${FALLBACK_SUMMARY:-1}"

suite_min_total_speedup="${SUITE_MIN_TOTAL_SPEEDUP:-${MIN_TOTAL_SPEEDUP:-1.10}}"
suite_window_min_total_speedup="${SUITE_WINDOW_MIN_TOTAL_SPEEDUP:-1.0}"
certificate_mode="${CERTIFICATE_MODE:-full-vocab-top1-metal}"

usage() {
  cat <<'EOF'
Usage: diffusion_gemma_fallback_replay_gate.sh

Prepare and measure exact fallback route replay for an existing mixed route
plan:
  1. derive base_exact fallback windows from MIXED_ROUTE_PLAN,
  2. prepare base route artifacts for those fallback windows,
  3. attach them to a derived route plan,
  4. run the mixed fast/exact suite gate from the derived plan.

Important environment knobs:
  MIXED_ROUTE_PLAN=PATH                 required source mixed route plan
  FALLBACK_REPLAY_STAGE=all             all, prepare, attach, or gate
  FALLBACK_PREPARE_LOG=PATH             reuse an existing base-artifact prepare log
  FALLBACK_BASE_ROUTE_ARTIFACT_MAP=SPEC reuse an existing fallback base map
  FALLBACK_ATTACHED_ROUTE_PLAN=PATH     output plan, or existing plan for gate
  FALLBACK_TOKEN_WINDOWS=prompt:canvas  optional fallback subset, default base_exact rows
  VARIANT_PROFILE=prompt-ffn-resident   profile used by existing fast artifacts
  CHECK_QUIET=1                         quiet precheck before prepare/gate work
  FALLBACK_SUMMARY=1                    write atlas and pp/tg summaries after gate
  DRY_RUN=1                             print commands without model work

The wrapper only prepares SUITE_ARTIFACT_ARMS=base and uses
diffusion_gemma_attach_fallback_artifacts.py to keep fallback artifacts in the
base_route_artifact slot.
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

validate_positive_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    die "$name must be a positive integer"
  fi
}

print_cmd() {
  local label="$1"
  shift
  printf '%s=' "$label"
  printf '%q ' "$@"
  printf '\n'
}

derive_windows() {
  local plan="$1"
  local mode="$2"
  python3 - "$plan" "$mode" <<'PY'
import json
import sys

plan, mode = sys.argv[1:3]
windows = []
with open(plan, encoding="utf-8") as handle:
    for lineno, raw in enumerate(handle, 1):
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("kind") != "diffusion_gemma_mixed_route_plan_window_v1":
            continue
        selected = row.get("selected_route")
        if mode == "fallback" and selected != "base_exact":
            continue
        windows.append(f"{int(row['prompt_token'])}:{int(row['canvas_token'])}")
if not windows:
    raise SystemExit(f"no {mode} windows found in {plan}")
print(",".join(windows))
PY
}

extract_map() {
  local key="$1"
  local file="$2"
  awk -v key="$key" -F= '$1 == key {print substr($0, length(key) + 2); found = 1} END {if (!found) exit 1}' "$file"
}

[[ -n "$mixed_route_plan" ]] || die "MIXED_ROUTE_PLAN is required"
[[ -f "$mixed_route_plan" ]] || die "MIXED_ROUTE_PLAN not found: $mixed_route_plan"
validate_positive_uint PROMPT_LEN "$prompt_len"
validate_positive_uint CANVAS_LEN "$canvas_len"
validate_positive_uint MAX_LAYERS "$max_layers"
bool_enabled "$check_quiet" >/dev/null || true
bool_enabled "$gate_check_quiet" >/dev/null || true
bool_enabled "$overwrite" >/dev/null || true
bool_enabled "$dry_run" >/dev/null || true
bool_enabled "$summary_enabled" >/dev/null || true

case "$stage" in
  all|prepare|attach|gate)
    ;;
  *)
    die "FALLBACK_REPLAY_STAGE must be all, prepare, attach, or gate"
    ;;
esac

if [[ -z "$fallback_windows" ]]; then
  fallback_windows="$(derive_windows "$mixed_route_plan" fallback)"
fi
if [[ -z "$plan_windows" ]]; then
  plan_windows="$(derive_windows "$mixed_route_plan" all)"
fi

if [[ -n "$prepare_log" ]]; then
  [[ -f "$prepare_log" ]] || die "FALLBACK_PREPARE_LOG not found: $prepare_log"
fi

if [[ "$stage" == "prepare" && -n "$prepare_log" ]]; then
  die "FALLBACK_REPLAY_STAGE=prepare cannot use FALLBACK_PREPARE_LOG"
fi
if [[ "$stage" == "prepare" && -n "$base_map" ]]; then
  die "FALLBACK_REPLAY_STAGE=prepare cannot use FALLBACK_BASE_ROUTE_ARTIFACT_MAP"
fi
if [[ "$stage" == "attach" || "$stage" == "gate" ]]; then
  if [[ "$stage" == "attach" || ! -f "$attached_plan" ]]; then
    [[ -n "$prepare_log" || -n "$base_map" ]] || die "$stage requires FALLBACK_PREPARE_LOG or FALLBACK_BASE_ROUTE_ARTIFACT_MAP unless FALLBACK_ATTACHED_ROUTE_PLAN already exists"
  fi
fi

should_prepare=0
if [[ "$stage" == "prepare" ]]; then
  should_prepare=1
elif [[ "$stage" == "all" && -z "$prepare_log" && -z "$base_map" ]]; then
  should_prepare=1
fi

mkdir -p "$log_dir"
manifest="$log_dir/fallback_replay_manifest.env"
{
  printf 'mixed_route_plan=%q\n' "$mixed_route_plan"
  printf 'stage=%q\n' "$stage"
  printf 'log_dir=%q\n' "$log_dir"
  printf 'prepare_dir=%q\n' "$prepare_dir"
  printf 'gate_dir=%q\n' "$gate_dir"
  printf 'artifact_dir=%q\n' "$artifact_dir"
  printf 'attached_plan=%q\n' "$attached_plan"
  printf 'fallback_windows=%q\n' "$fallback_windows"
  printf 'plan_windows=%q\n' "$plan_windows"
  printf 'prompt_len=%q\n' "$prompt_len"
  printf 'canvas_len=%q\n' "$canvas_len"
  printf 'max_layers=%q\n' "$max_layers"
  printf 'variant_profile=%q\n' "$variant_profile"
  printf 'check_quiet=%q\n' "$check_quiet"
  printf 'gate_check_quiet=%q\n' "$gate_check_quiet"
  printf 'quiet_ms=%q\n' "$quiet_ms"
  printf 'load_threshold=%q\n' "$load_threshold"
  printf 'total_threshold=%q\n' "$total_threshold"
  printf 'summary_enabled=%q\n' "$summary_enabled"
} >"$manifest"

printf 'fallback_replay_start log_dir=%s manifest=%s\n' "$log_dir" "$manifest"
printf 'fallback_replay_windows fallback=%s plan=%s\n' "$fallback_windows" "$plan_windows"

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
  TOKEN_WINDOWS="$fallback_windows"
  PROMPT_LEN="$prompt_len"
  CANVAS_LEN="$canvas_len"
  MAX_LAYERS="$max_layers"
  SUITE_ARTIFACT_ARMS=base
  SUITE_ARTIFACT_OVERWRITE="$overwrite"
  "$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_prepare.sh"
)

attach_cmd=(
  python3
  "$repo_root/scripts/diffusion_gemma_attach_fallback_artifacts.py"
  "$mixed_route_plan"
  --out "$attached_plan"
)

if [[ -n "$prepare_log" ]]; then
  attach_cmd+=(--prepare-log "$prepare_log")
elif [[ -n "$base_map" ]]; then
  attach_cmd+=(--base-map "$base_map")
else
  attach_cmd+=(--prepare-log "$prepare_stdout")
fi

gate_stdout="$log_dir/gate.stdout"
gate_stderr="$log_dir/gate.stderr"
summary_route_plan="${FALLBACK_SUMMARY_ROUTE_PLAN:-$gate_dir/route_plan.jsonl}"
summary_atlas="$log_dir/fallback_replay_route_plan_atlas.txt"
summary_atlas_tsv="$log_dir/fallback_replay_route_plan_atlas.tsv"
summary_pp_tg="$log_dir/fallback_replay_pp_tg.tsv"
gate_cmd=(
  env
  LOG_DIR="$gate_dir"
  PROMOTION_STAGE=gate
  SUITE_MIXED_ROUTE_PLAN="$attached_plan"
  TOKEN_WINDOWS="$plan_windows"
  PROMPT_LEN="$prompt_len"
  CANVAS_LEN="$canvas_len"
  MAX_LAYERS="$max_layers"
  VARIANT_PROFILE="$variant_profile"
  CHECK_QUIET="$gate_check_quiet"
  LOAD_THRESHOLD="$load_threshold"
  TOTAL_THRESHOLD="$total_threshold"
  QUIET_MS="$quiet_ms"
  SUITE_MIXED_FALLBACK_GATE=1
  SUITE_MIN_TOTAL_SPEEDUP="$suite_min_total_speedup"
  SUITE_WINDOW_MIN_TOTAL_SPEEDUP="$suite_window_min_total_speedup"
  CERTIFICATE_MODE="$certificate_mode"
  "$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_promotion.sh"
)

atlas_cmd=(
  python3
  "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py"
  "$summary_route_plan"
)

atlas_tsv_cmd=(
  python3
  "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py"
  "$summary_route_plan"
  --tsv
)

pp_tg_cmd=(
  python3
  "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py"
  "$gate_stdout"
)

write_summaries() {
  if ! bool_enabled "$summary_enabled"; then
    printf 'fallback_replay_summary skipped reason=disabled\n'
    return 0
  fi

  if [[ ! -f "$summary_route_plan" ]]; then
    printf 'fallback_replay_summary warning=missing_route_plan route_plan=%s\n' "$summary_route_plan"
  else
    if "${atlas_cmd[@]}" >"$summary_atlas" 2>"$summary_atlas.stderr"; then
      cat "$summary_atlas"
    else
      printf 'fallback_replay_summary warning=atlas_failed route_plan=%s stderr=%s\n' "$summary_route_plan" "$summary_atlas.stderr"
    fi
    if "${atlas_tsv_cmd[@]}" >"$summary_atlas_tsv" 2>"$summary_atlas_tsv.stderr"; then
      :
    else
      printf 'fallback_replay_summary warning=atlas_tsv_failed route_plan=%s stderr=%s\n' "$summary_route_plan" "$summary_atlas_tsv.stderr"
    fi
  fi

  if "${pp_tg_cmd[@]}" >"$summary_pp_tg" 2>"$summary_pp_tg.stderr"; then
    cat "$summary_pp_tg"
  else
    printf 'fallback_replay_summary warning=pp_tg_failed log=%s stderr=%s\n' "$gate_stdout" "$summary_pp_tg.stderr"
  fi

  printf 'fallback_replay_summary route_plan=%s atlas=%s atlas_tsv=%s pp_tg=%s\n' \
    "$summary_route_plan" "$summary_atlas" "$summary_atlas_tsv" "$summary_pp_tg"
}

if bool_enabled "$dry_run"; then
  if bool_enabled "$check_quiet" && [[ "$stage" != "attach" ]]; then
    print_cmd quiet_cmd "${quiet_cmd[@]}"
  fi
  if [[ "$should_prepare" == "1" ]]; then
    print_cmd prepare_cmd "${prepare_cmd[@]}"
  fi
  if [[ "$stage" == "all" || "$stage" == "attach" || ( "$stage" == "gate" && ! -f "$attached_plan" ) ]]; then
    print_cmd attach_cmd "${attach_cmd[@]}"
  fi
  if [[ "$stage" == "all" || "$stage" == "gate" ]]; then
    print_cmd gate_cmd "${gate_cmd[@]}"
    if bool_enabled "$summary_enabled"; then
      print_cmd summary_atlas_cmd "${atlas_cmd[@]}"
      print_cmd summary_atlas_tsv_cmd "${atlas_tsv_cmd[@]}"
      print_cmd summary_pp_tg_cmd "${pp_tg_cmd[@]}"
    fi
  fi
  printf 'fallback_replay decision=dry_run stage=%s log_dir=%s attached_plan=%s\n' "$stage" "$log_dir" "$attached_plan"
  exit 0
fi

if bool_enabled "$check_quiet" && [[ "$stage" != "attach" ]]; then
  set +e
  "${quiet_cmd[@]}" >"$quiet_log" 2>&1
  quiet_rc=$?
  set -e
  cat "$quiet_log"
  if (( quiet_rc != 0 )); then
    printf 'fallback_replay decision=blocked reason=quiet_gate_failed rc=%s log=%s\n' "$quiet_rc" "$quiet_log"
    exit "$quiet_rc"
  fi
fi

if [[ "$should_prepare" == "1" ]]; then
  set +e
  "${prepare_cmd[@]}" >"$prepare_stdout" 2>"$prepare_stderr"
  prepare_rc=$?
  set -e
  cat "$prepare_stdout"
  if (( prepare_rc != 0 )); then
    printf 'fallback_replay decision=reject reason=prepare_failed rc=%s log=%s stderr=%s\n' "$prepare_rc" "$prepare_stdout" "$prepare_stderr"
    exit "$prepare_rc"
  fi
  if [[ "$stage" == "prepare" ]]; then
    printf 'fallback_replay decision=prepared log_dir=%s prepare_log=%s\n' "$log_dir" "$prepare_stdout"
    exit 0
  fi
fi

if [[ "$stage" == "all" || "$stage" == "attach" || ! -f "$attached_plan" ]]; then
  "${attach_cmd[@]}"
  if [[ "$stage" == "attach" ]]; then
    printf 'fallback_replay decision=attached log_dir=%s attached_plan=%s\n' "$log_dir" "$attached_plan"
    exit 0
  fi
fi

if [[ "$stage" == "all" || "$stage" == "gate" ]]; then
  set +e
  "${gate_cmd[@]}" >"$gate_stdout" 2>"$gate_stderr"
  gate_rc=$?
  set -e
  cat "$gate_stdout"
  if (( gate_rc != 0 )); then
    printf 'fallback_replay decision=reject reason=gate_failed rc=%s log=%s stderr=%s\n' "$gate_rc" "$gate_stdout" "$gate_stderr"
    exit "$gate_rc"
  fi
  decision="$(grep -E '^artifact_suite_promotion decision=' "$gate_stdout" | tail -1 || true)"
  write_summaries
  printf 'fallback_replay decision=complete stage=%s log_dir=%s attached_plan=%s child_decision=%q\n' "$stage" "$log_dir" "$attached_plan" "$decision"
fi
