#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d%H%M%S)"

mixed_route_plan="${MIXED_ROUTE_PLAN:-${SUITE_MIXED_ROUTE_PLAN:-}}"
replay_mode="${FALLBACK_REPLAY_MODE:-selected}"
stage="${FALLBACK_REPLAY_STAGE:-all}"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_fallback_replay_${timestamp}}"
prepare_dir="${FALLBACK_PREPARE_LOG_DIR:-$log_dir/prepare}"
gate_dir="${FALLBACK_GATE_LOG_DIR:-$log_dir/gate}"
selected_gate_dir="${FALLBACK_SELECTED_GATE_LOG_DIR:-$log_dir/gate_selected}"
foreign_gate_dir="${FALLBACK_FOREIGN_GATE_LOG_DIR:-$log_dir/gate_foreign}"
artifact_dir="${FALLBACK_ARTIFACT_DIR:-$log_dir/artifacts}"
attached_plan="${FALLBACK_ATTACHED_ROUTE_PLAN:-$log_dir/route_plan_with_base_fallback.jsonl}"
prepare_log="${FALLBACK_PREPARE_LOG:-}"
base_map="${FALLBACK_BASE_ROUTE_ARTIFACT_MAP:-${SUITE_BASE_ROUTE_ARTIFACT_MAP:-${CERT_BASE_ROUTE_ARTIFACT_MAP:-}}}"
foreign_base_map_override="${FALLBACK_FOREIGN_BASE_ROUTE_ARTIFACT_MAP:-}"

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
fallback_compare_min_foreign_speedup="${FALLBACK_COMPARE_MIN_FOREIGN_SPEEDUP:-1.0}"
fallback_compare_require_foreign="${FALLBACK_COMPARE_REQUIRE_FOREIGN:-0}"
fallback_compare_write_promoted_route_plan="${FALLBACK_COMPARE_WRITE_PROMOTED_ROUTE_PLAN:-$fallback_compare_require_foreign}"

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
  FALLBACK_REPLAY_MODE=selected         selected, foreign, or compare
                                           selected: attach base artifacts into a derived mixed plan
                                           foreign: run fallback windows through the variant side
                                                    while loading base/env-bound route artifacts
                                           compare: run selected and foreign gates from the same
                                                    prepared/reused base artifacts
  FALLBACK_REPLAY_STAGE=all             all, prepare, attach, or gate
  FALLBACK_PREPARE_LOG=PATH             reuse an existing base-artifact prepare log
  FALLBACK_BASE_ROUTE_ARTIFACT_MAP=SPEC reuse an existing fallback base map
  FALLBACK_FOREIGN_BASE_ROUTE_ARTIFACT_MAP=SPEC
                                        reuse an existing fallback base map only for
                                        foreign variant-side replay
  FALLBACK_ATTACHED_ROUTE_PLAN=PATH     output plan, or existing plan for gate
  FALLBACK_TOKEN_WINDOWS=prompt:canvas  optional fallback subset, default base_exact rows
  VARIANT_PROFILE=prompt-ffn-resident   profile used by existing fast artifacts
  CHECK_QUIET=1                         quiet precheck before prepare/gate work
  FALLBACK_SUMMARY=1                    write atlas and pp/tg summaries after gate
  FALLBACK_COMPARE_MIN_FOREIGN_SPEEDUP=1.0
                                        minimum foreign-vs-selected speedup for compare summary
  FALLBACK_COMPARE_REQUIRE_FOREIGN=0    when 1, compare mode rejects unless foreign beats threshold
  FALLBACK_COMPARE_WRITE_PROMOTED_ROUTE_PLAN
                                        when 1, write a mixed route plan with winning
                                        foreign fallback rows promoted; defaults to
                                        FALLBACK_COMPARE_REQUIRE_FOREIGN
  DRY_RUN=1                             print commands without model work

The wrapper only prepares SUITE_ARTIFACT_ARMS=base and uses
diffusion_gemma_attach_fallback_artifacts.py to keep fallback artifacts in the
base_route_artifact slot. In foreign mode, the same prepared base artifacts are
fed to SUITE_VARIANT_ROUTE_ARTIFACT_MAP with expected arm/env role set to base.
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

base_artifact_map_for_windows() {
  local windows="$1"
  python3 - "$windows" "$artifact_dir" "$prompt_len" "$max_layers" <<'PY'
import sys

windows_raw, artifact_dir, prompt_len, max_layers = sys.argv[1:5]
entries = []
seen = set()
for entry in windows_raw.replace(",", " ").split():
    try:
        prompt, canvas = entry.split(":", 1)
        prompt_i = int(prompt)
        canvas_i = int(canvas)
    except ValueError as exc:
        raise SystemExit(f"fallback window must be prompt:canvas, got {entry!r}") from exc
    if prompt_i < 0 or canvas_i < 0:
        raise SystemExit("fallback window prompt/canvas tokens must be non-negative")
    key = (prompt_i, canvas_i)
    if key in seen:
        raise SystemExit(f"duplicate fallback window {prompt_i}:{canvas_i}")
    seen.add(key)
    path = f"{artifact_dir}/base_p{prompt_i}_c{canvas_i}_pl{prompt_len}_l{max_layers}.tsv"
    entries.append(f"{prompt_i}:{canvas_i}={path}")
if not entries:
    raise SystemExit("fallback windows must not be empty")
print(",".join(entries))
PY
}

check_map_files() {
  local raw_map="$1"
  local label="$2"
  python3 - "$raw_map" "$label" <<'PY'
import os
import sys

raw_map, label = sys.argv[1:3]
missing = []
for entry in raw_map.replace(",", " ").split():
    try:
        window, path = entry.split("=", 1)
    except ValueError as exc:
        raise SystemExit(f"{label} entry must be prompt:canvas=PATH, got {entry!r}") from exc
    if not path:
        raise SystemExit(f"{label} path must not be empty for {window}")
    if not os.path.isfile(path):
        missing.append(f"{window}={path}")
if missing:
    raise SystemExit(f"{label} missing artifact(s): {' '.join(missing)}")
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
bool_enabled "$fallback_compare_require_foreign" >/dev/null || true
bool_enabled "$fallback_compare_write_promoted_route_plan" >/dev/null || true

case "$stage" in
  all|prepare|attach|gate)
    ;;
  *)
    die "FALLBACK_REPLAY_STAGE must be all, prepare, attach, or gate"
    ;;
esac
case "$replay_mode" in
  selected|foreign|compare)
    ;;
  *)
    die "FALLBACK_REPLAY_MODE must be selected, foreign, or compare"
    ;;
esac
if [[ ( "$replay_mode" == "foreign" || "$replay_mode" == "compare" ) && "$stage" == "attach" ]]; then
  die "FALLBACK_REPLAY_MODE=$replay_mode does not use FALLBACK_REPLAY_STAGE=attach"
fi
if [[ "$replay_mode" == "compare" && -n "$foreign_base_map_override" ]]; then
  die "FALLBACK_REPLAY_MODE=compare uses the same base artifacts for selected and foreign; use FALLBACK_BASE_ROUTE_ARTIFACT_MAP"
fi

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
if [[ "$stage" == "prepare" && -n "$foreign_base_map_override" ]]; then
  die "FALLBACK_REPLAY_STAGE=prepare cannot use FALLBACK_FOREIGN_BASE_ROUTE_ARTIFACT_MAP"
fi
if [[ ( "$replay_mode" == "selected" || "$replay_mode" == "compare" ) && ( "$stage" == "attach" || "$stage" == "gate" ) ]]; then
  if [[ "$stage" == "attach" || ! -f "$attached_plan" ]]; then
    [[ -n "$prepare_log" || -n "$base_map" ]] || die "$stage requires FALLBACK_PREPARE_LOG or FALLBACK_BASE_ROUTE_ARTIFACT_MAP unless FALLBACK_ATTACHED_ROUTE_PLAN already exists"
  fi
fi

should_prepare=0
if [[ "$stage" == "prepare" ]]; then
  should_prepare=1
elif [[ "$stage" == "all" && -z "$prepare_log" && -z "$base_map" ]]; then
  if [[ "$replay_mode" != "foreign" || -z "$foreign_base_map_override" ]]; then
    should_prepare=1
  fi
fi

foreign_base_map=""
if [[ "$replay_mode" == "foreign" || "$replay_mode" == "compare" ]]; then
  if [[ -n "$foreign_base_map_override" ]]; then
    foreign_base_map="$foreign_base_map_override"
  elif [[ -n "$base_map" ]]; then
    foreign_base_map="$base_map"
  elif [[ -n "$prepare_log" ]]; then
    foreign_base_map="$(extract_map SUITE_BASE_ROUTE_ARTIFACT_MAP "$prepare_log")"
  else
    foreign_base_map="$(base_artifact_map_for_windows "$fallback_windows")"
  fi
fi

compare_promoted_route_plan="${FALLBACK_COMPARE_PROMOTED_ROUTE_PLAN:-$log_dir/fallback_replay_promoted_route_plan.jsonl}"
compare_promoted_pp_tg="${FALLBACK_COMPARE_PROMOTED_PP_TG:-$log_dir/fallback_replay_promoted_pp_tg.tsv}"

mkdir -p "$log_dir"
manifest="$log_dir/fallback_replay_manifest.env"
{
  printf 'mixed_route_plan=%q\n' "$mixed_route_plan"
  printf 'replay_mode=%q\n' "$replay_mode"
  printf 'stage=%q\n' "$stage"
  printf 'log_dir=%q\n' "$log_dir"
  printf 'prepare_dir=%q\n' "$prepare_dir"
  printf 'gate_dir=%q\n' "$gate_dir"
  printf 'selected_gate_dir=%q\n' "$selected_gate_dir"
  printf 'foreign_gate_dir=%q\n' "$foreign_gate_dir"
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
  printf 'fallback_compare_min_foreign_speedup=%q\n' "$fallback_compare_min_foreign_speedup"
  printf 'fallback_compare_require_foreign=%q\n' "$fallback_compare_require_foreign"
  printf 'fallback_compare_write_promoted_route_plan=%q\n' "$fallback_compare_write_promoted_route_plan"
  printf 'compare_promoted_route_plan=%q\n' "$compare_promoted_route_plan"
  printf 'compare_promoted_pp_tg=%q\n' "$compare_promoted_pp_tg"
  printf 'foreign_base_map_override=%q\n' "$foreign_base_map_override"
  printf 'foreign_base_map=%q\n' "$foreign_base_map"
} >"$manifest"

printf 'fallback_replay_start log_dir=%s manifest=%s\n' "$log_dir" "$manifest"
printf 'fallback_replay_windows fallback=%s plan=%s mode=%s\n' "$fallback_windows" "$plan_windows" "$replay_mode"

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
selected_gate_stdout="$log_dir/gate_selected.stdout"
selected_gate_stderr="$log_dir/gate_selected.stderr"
foreign_gate_stdout="$log_dir/gate_foreign.stdout"
foreign_gate_stderr="$log_dir/gate_foreign.stderr"
selected_runtime_gate_dir="$gate_dir"
foreign_runtime_gate_dir="$gate_dir"
if [[ "$replay_mode" == "compare" ]]; then
  selected_runtime_gate_dir="$selected_gate_dir"
  foreign_runtime_gate_dir="$foreign_gate_dir"
fi
summary_route_plan="${FALLBACK_SUMMARY_ROUTE_PLAN:-$gate_dir/route_plan.jsonl}"
selected_summary_route_plan="${FALLBACK_SELECTED_SUMMARY_ROUTE_PLAN:-$selected_runtime_gate_dir/route_plan.jsonl}"
foreign_summary_route_plan="${FALLBACK_FOREIGN_SUMMARY_ROUTE_PLAN:-$foreign_runtime_gate_dir/route_plan.jsonl}"
compare_summary="$log_dir/fallback_replay_compare_summary.txt"
compare_summary_tsv="$log_dir/fallback_replay_compare_summary.tsv"
gate_cmd=(
  env
  LOG_DIR="$selected_runtime_gate_dir"
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

foreign_gate_cmd=(
  env
  LOG_DIR="$foreign_runtime_gate_dir"
  PROMOTION_STAGE=gate
  TOKEN_WINDOWS="$fallback_windows"
  PROMPT_LEN="$prompt_len"
  CANVAS_LEN="$canvas_len"
  MAX_LAYERS="$max_layers"
  VARIANT_PROFILE="$variant_profile"
  CHECK_QUIET="$gate_check_quiet"
  LOAD_THRESHOLD="$load_threshold"
  TOTAL_THRESHOLD="$total_threshold"
  QUIET_MS="$quiet_ms"
  SUITE_MIN_TOTAL_SPEEDUP="$suite_min_total_speedup"
  SUITE_WINDOW_MIN_TOTAL_SPEEDUP="$suite_window_min_total_speedup"
  CERTIFICATE_MODE="$certificate_mode"
  SUITE_VARIANT_ROUTE_ARTIFACT_MAP="$foreign_base_map"
  CERT_VARIANT_ROUTE_ARTIFACT_MAP="$foreign_base_map"
  VARIANT_ROUTE_ARTIFACT_EXPECTED_ARM=base
  VARIANT_ROUTE_ARTIFACT_ENV_ROLE=base
  "$repo_root/scripts/diffusion_gemma_prompt_artifact_suite_promotion.sh"
)

write_summaries() {
  local label="$1"
  local route_plan="$2"
  local gate_log="$3"
  local prefix="$log_dir/fallback_replay"
  if [[ "$label" != "default" ]]; then
    prefix="$log_dir/fallback_replay_${label}"
  fi
  local summary_atlas="${prefix}_route_plan_atlas.txt"
  local summary_atlas_tsv="${prefix}_route_plan_atlas.tsv"
  local summary_pp_tg="${prefix}_pp_tg.tsv"
  local atlas_cmd=(
    python3
    "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py"
    "$route_plan"
  )
  local atlas_tsv_cmd=(
    python3
    "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py"
    "$route_plan"
    --tsv
  )
  local pp_tg_cmd=(
    python3
    "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py"
    "$gate_log"
  )

  if ! bool_enabled "$summary_enabled"; then
    printf 'fallback_replay_summary label=%s skipped reason=disabled\n' "$label"
    return 0
  fi

  if [[ ! -f "$route_plan" ]]; then
    printf 'fallback_replay_summary label=%s warning=missing_route_plan route_plan=%s\n' "$label" "$route_plan"
  else
    if "${atlas_cmd[@]}" >"$summary_atlas" 2>"$summary_atlas.stderr"; then
      cat "$summary_atlas"
    else
      printf 'fallback_replay_summary label=%s warning=atlas_failed route_plan=%s stderr=%s\n' "$label" "$route_plan" "$summary_atlas.stderr"
    fi
    if "${atlas_tsv_cmd[@]}" >"$summary_atlas_tsv" 2>"$summary_atlas_tsv.stderr"; then
      :
    else
      printf 'fallback_replay_summary label=%s warning=atlas_tsv_failed route_plan=%s stderr=%s\n' "$label" "$route_plan" "$summary_atlas_tsv.stderr"
    fi
  fi

  if "${pp_tg_cmd[@]}" >"$summary_pp_tg" 2>"$summary_pp_tg.stderr"; then
    cat "$summary_pp_tg"
  else
    printf 'fallback_replay_summary label=%s warning=pp_tg_failed log=%s stderr=%s\n' "$label" "$gate_log" "$summary_pp_tg.stderr"
  fi

  printf 'fallback_replay_summary label=%s route_plan=%s atlas=%s atlas_tsv=%s pp_tg=%s\n' \
    "$label" "$route_plan" "$summary_atlas" "$summary_atlas_tsv" "$summary_pp_tg"
}

write_compare_summary() {
  if ! bool_enabled "$summary_enabled"; then
    if bool_enabled "$fallback_compare_require_foreign"; then
      printf 'fallback_replay_compare_summary summary=disabled guard=required\n'
    else
      printf 'fallback_replay_compare_summary skipped reason=disabled\n'
      return 0
    fi
  fi
  if [[ ! -f "$selected_summary_route_plan" || ! -f "$foreign_summary_route_plan" ]]; then
    printf 'fallback_replay_compare_summary warning=missing_route_plan selected=%s foreign=%s\n' \
      "$selected_summary_route_plan" "$foreign_summary_route_plan"
    return 0
  fi

  local compare_cmd=(
    python3
    "$repo_root/scripts/diffusion_gemma_fallback_compare_summary.py"
    --selected-route-plan "$selected_summary_route_plan"
    --foreign-route-plan "$foreign_summary_route_plan"
    --min-foreign-speedup "$fallback_compare_min_foreign_speedup"
  )
  if bool_enabled "$fallback_compare_require_foreign"; then
    compare_cmd+=(--require-foreign)
  fi
  local compare_tsv_cmd=("${compare_cmd[@]}")
  if bool_enabled "$fallback_compare_write_promoted_route_plan"; then
    compare_cmd+=(--promoted-route-plan-out "$compare_promoted_route_plan")
  fi

  local compare_rc=0
  set +e
  "${compare_cmd[@]}" >"$compare_summary" 2>"$compare_summary.stderr"
  compare_rc=$?
  set -e
  if (( compare_rc == 0 )); then
    cat "$compare_summary"
  else
    cat "$compare_summary"
    if bool_enabled "$fallback_compare_require_foreign"; then
      printf 'fallback_replay_compare_summary reject=compare_failed rc=%s stderr=%s\n' "$compare_rc" "$compare_summary.stderr"
      cat "$compare_summary.stderr" >&2
      return "$compare_rc"
    fi
    printf 'fallback_replay_compare_summary warning=compare_failed rc=%s stderr=%s\n' "$compare_rc" "$compare_summary.stderr"
    return 0
  fi
  if "${compare_tsv_cmd[@]}" --tsv >"$compare_summary_tsv" 2>"$compare_summary_tsv.stderr"; then
    :
  else
    printf 'fallback_replay_compare_summary warning=compare_tsv_failed stderr=%s\n' "$compare_summary_tsv.stderr"
    return 0
  fi
  if bool_enabled "$fallback_compare_write_promoted_route_plan"; then
    if bool_enabled "$summary_enabled"; then
      if [[ -f "$compare_promoted_route_plan" ]]; then
        local promoted_pp_tg_cmd=(
          python3
          "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py"
          --prompt-len "$prompt_len"
          --canvas-len "$canvas_len"
          --max-layers "$max_layers"
          "$compare_promoted_route_plan"
        )
        if "${promoted_pp_tg_cmd[@]}" >"$compare_promoted_pp_tg" 2>"$compare_promoted_pp_tg.stderr"; then
          cat "$compare_promoted_pp_tg"
        else
          printf 'fallback_replay_compare_summary warning=promoted_pp_tg_failed promoted_route_plan=%s stderr=%s\n' \
            "$compare_promoted_route_plan" "$compare_promoted_pp_tg.stderr"
        fi
      else
        printf 'fallback_replay_compare_summary warning=missing_promoted_route_plan promoted_route_plan=%s\n' "$compare_promoted_route_plan"
      fi
    fi
    printf 'fallback_replay_compare_summary text=%s tsv=%s promoted_route_plan=%s promoted_pp_tg=%s\n' \
      "$compare_summary" "$compare_summary_tsv" "$compare_promoted_route_plan" "$compare_promoted_pp_tg"
  else
    printf 'fallback_replay_compare_summary text=%s tsv=%s\n' "$compare_summary" "$compare_summary_tsv"
  fi
}

if bool_enabled "$dry_run"; then
  if bool_enabled "$check_quiet" && [[ "$stage" != "attach" ]]; then
    print_cmd quiet_cmd "${quiet_cmd[@]}"
  fi
  if [[ "$should_prepare" == "1" ]]; then
    print_cmd prepare_cmd "${prepare_cmd[@]}"
  fi
  if [[ ( "$replay_mode" == "selected" || "$replay_mode" == "compare" ) && ( "$stage" == "all" || "$stage" == "attach" || ( "$stage" == "gate" && ! -f "$attached_plan" ) ) ]]; then
    print_cmd attach_cmd "${attach_cmd[@]}"
  fi
  if [[ "$stage" == "all" || "$stage" == "gate" ]]; then
    if [[ "$replay_mode" == "compare" ]]; then
      printf 'foreign_base_map=%s\n' "$foreign_base_map"
      print_cmd selected_gate_cmd "${gate_cmd[@]}"
      print_cmd foreign_gate_cmd "${foreign_gate_cmd[@]}"
    elif [[ "$replay_mode" == "foreign" ]]; then
      printf 'foreign_base_map=%s\n' "$foreign_base_map"
      print_cmd foreign_gate_cmd "${foreign_gate_cmd[@]}"
    else
      print_cmd gate_cmd "${gate_cmd[@]}"
    fi
    if bool_enabled "$summary_enabled" || { [[ "$replay_mode" == "compare" ]] && bool_enabled "$fallback_compare_require_foreign"; }; then
      if [[ "$replay_mode" == "compare" ]]; then
        print_cmd selected_summary_atlas_cmd python3 "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py" "$selected_summary_route_plan"
        print_cmd selected_summary_pp_tg_cmd python3 "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py" "$selected_gate_stdout"
        print_cmd foreign_summary_atlas_cmd python3 "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py" "$foreign_summary_route_plan"
        print_cmd foreign_summary_pp_tg_cmd python3 "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py" "$foreign_gate_stdout"
        compare_dry_cmd=(python3 "$repo_root/scripts/diffusion_gemma_fallback_compare_summary.py" --selected-route-plan "$selected_summary_route_plan" --foreign-route-plan "$foreign_summary_route_plan" --min-foreign-speedup "$fallback_compare_min_foreign_speedup")
        if bool_enabled "$fallback_compare_require_foreign"; then
          compare_dry_cmd+=(--require-foreign)
        fi
        compare_dry_tsv_cmd=("${compare_dry_cmd[@]}")
        if bool_enabled "$fallback_compare_write_promoted_route_plan"; then
          compare_dry_cmd+=(--promoted-route-plan-out "$compare_promoted_route_plan")
        fi
        print_cmd compare_summary_cmd "${compare_dry_cmd[@]}"
        print_cmd compare_summary_tsv_cmd "${compare_dry_tsv_cmd[@]}" --tsv
        if bool_enabled "$fallback_compare_write_promoted_route_plan" && bool_enabled "$summary_enabled"; then
          print_cmd compare_promoted_pp_tg_cmd \
            python3 "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py" \
            --prompt-len "$prompt_len" \
            --canvas-len "$canvas_len" \
            --max-layers "$max_layers" \
            "$compare_promoted_route_plan"
        fi
      elif [[ "$replay_mode" == "foreign" ]]; then
        print_cmd summary_atlas_cmd python3 "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py" "$foreign_summary_route_plan"
        print_cmd summary_atlas_tsv_cmd python3 "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py" "$foreign_summary_route_plan" --tsv
        print_cmd summary_pp_tg_cmd python3 "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py" "$gate_stdout"
      else
        print_cmd summary_atlas_cmd python3 "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py" "$selected_summary_route_plan"
        print_cmd summary_atlas_tsv_cmd python3 "$repo_root/scripts/diffusion_gemma_mixed_route_plan_atlas.py" "$selected_summary_route_plan" --tsv
        print_cmd summary_pp_tg_cmd python3 "$repo_root/scripts/diffusion_gemma_pp_tg_summary.py" "$gate_stdout"
      fi
    fi
  fi
  printf 'fallback_replay decision=dry_run mode=%s stage=%s log_dir=%s attached_plan=%s\n' "$replay_mode" "$stage" "$log_dir" "$attached_plan"
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

if [[ ( "$replay_mode" == "selected" || "$replay_mode" == "compare" ) && ( "$stage" == "all" || "$stage" == "attach" || ! -f "$attached_plan" ) ]]; then
  "${attach_cmd[@]}"
  if [[ "$stage" == "attach" ]]; then
    printf 'fallback_replay decision=attached log_dir=%s attached_plan=%s\n' "$log_dir" "$attached_plan"
    exit 0
  fi
fi

if [[ "$stage" == "all" || "$stage" == "gate" ]]; then
  if [[ "$replay_mode" == "compare" ]]; then
    if ! check_map_files "$foreign_base_map" "FALLBACK_FOREIGN_BASE_ROUTE_ARTIFACT_MAP"; then
      exit 2
    fi

    set +e
    "${gate_cmd[@]}" >"$selected_gate_stdout" 2>"$selected_gate_stderr"
    selected_rc=$?
    set -e
    cat "$selected_gate_stdout"
    selected_decision="$(grep -E '^artifact_suite_promotion decision=' "$selected_gate_stdout" | tail -1 || true)"
    if (( selected_rc == 0 )); then
      write_summaries selected "$selected_summary_route_plan" "$selected_gate_stdout"
    else
      printf 'fallback_replay decision=reject mode=compare arm=selected reason=gate_failed rc=%s log=%s stderr=%s child_decision=%q\n' \
        "$selected_rc" "$selected_gate_stdout" "$selected_gate_stderr" "$selected_decision"
    fi

    set +e
    "${foreign_gate_cmd[@]}" >"$foreign_gate_stdout" 2>"$foreign_gate_stderr"
    foreign_rc=$?
    set -e
    cat "$foreign_gate_stdout"
    foreign_decision="$(grep -E '^artifact_suite_promotion decision=' "$foreign_gate_stdout" | tail -1 || true)"
    if (( foreign_rc == 0 )); then
      write_summaries foreign "$foreign_summary_route_plan" "$foreign_gate_stdout"
    else
      printf 'fallback_replay decision=reject mode=compare arm=foreign reason=gate_failed rc=%s log=%s stderr=%s child_decision=%q\n' \
        "$foreign_rc" "$foreign_gate_stdout" "$foreign_gate_stderr" "$foreign_decision"
    fi

    if (( selected_rc != 0 || foreign_rc != 0 )); then
      printf 'fallback_replay decision=reject mode=compare reason=gate_failed selected_rc=%s foreign_rc=%s log_dir=%s\n' \
        "$selected_rc" "$foreign_rc" "$log_dir"
      exit 4
    fi
    set +e
    write_compare_summary
    compare_rc=$?
    set -e
    if (( compare_rc != 0 )); then
      printf 'fallback_replay decision=reject mode=compare reason=compare_summary_failed rc=%s log_dir=%s compare_summary=%s\n' \
        "$compare_rc" "$log_dir" "$compare_summary"
      exit "$compare_rc"
    fi
    printf 'fallback_replay decision=complete mode=compare stage=%s log_dir=%s selected_log=%s foreign_log=%s selected_decision=%q foreign_decision=%q\n' \
      "$stage" "$log_dir" "$selected_gate_stdout" "$foreign_gate_stdout" "$selected_decision" "$foreign_decision"
    exit 0
  fi

  if [[ "$replay_mode" == "foreign" ]]; then
    if ! check_map_files "$foreign_base_map" "FALLBACK_FOREIGN_BASE_ROUTE_ARTIFACT_MAP"; then
      exit 2
    fi
    set +e
    "${foreign_gate_cmd[@]}" >"$gate_stdout" 2>"$gate_stderr"
    gate_rc=$?
    set -e
    cat "$gate_stdout"
    if (( gate_rc != 0 )); then
      printf 'fallback_replay decision=reject mode=foreign reason=gate_failed rc=%s log=%s stderr=%s\n' "$gate_rc" "$gate_stdout" "$gate_stderr"
      exit "$gate_rc"
    fi
    decision="$(grep -E '^artifact_suite_promotion decision=' "$gate_stdout" | tail -1 || true)"
    write_summaries default "$foreign_summary_route_plan" "$gate_stdout"
    printf 'fallback_replay decision=complete mode=foreign stage=%s log_dir=%s foreign_base_map=%q child_decision=%q\n' "$stage" "$log_dir" "$foreign_base_map" "$decision"
    exit 0
  fi

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
  write_summaries default "$selected_summary_route_plan" "$gate_stdout"
  printf 'fallback_replay decision=complete mode=selected stage=%s log_dir=%s attached_plan=%s child_decision=%q\n' "$stage" "$log_dir" "$attached_plan" "$decision"
fi
