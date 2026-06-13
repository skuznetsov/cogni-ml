#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rev="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"

log_dir="${LOG_DIR:-/tmp/diffusiongemma_phase_mirror_abba_${rev}_$$_$(date +%Y%m%d%H%M%S)}"
forward_sequence="${FORWARD_SEQUENCE:-base variant variant base}"
mirror_sequence="${MIRROR_SEQUENCE:-variant base base variant}"
base_env="${BASE_ENV:-}"
variant_env="${VARIANT_ENV:-}"
min_speedup="${MIN_SPEEDUP:-1.02}"

prompt_len="${PROMPT_LEN:-16}"
canvas_len="${CANVAS_LEN:-8}"
prompt_token="${PROMPT_TOKEN:-1}"
canvas_token="${CANVAS_TOKEN:-0}"
max_layers="${MAX_LAYERS:-1}"
warmups="${WARMUPS:-1}"
repeats="${REPEATS:-8}"
trim_per_arm="${TRIM_PER_ARM:-1}"
full_routes="${FULL_ROUTES:-1}"
require_quiet="${REQUIRE_QUIET:-0}"
quiet_ms="${QUIET_MS:-15000}"
load_threshold="${LOAD_THRESHOLD:-40}"
total_threshold="${TOTAL_THRESHOLD:-90}"

phase_runner="${PHASE_RUNNER:-}"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
model="${MODEL:-${DIFFUSION_GEMMA_MODEL:-}}"
forward_tsv="${FORWARD_TSV:-}"
mirror_tsv="${MIRROR_TSV:-}"

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got %s\n' "$name" "$value" >&2
    exit 2
  fi
}

validate_env() {
  local name="$1"
  local raw="$2"
  local token
  for token in $raw; do
    if [[ ! "$token" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      printf '%s token must be KEY=VALUE, got %s\n' "$name" "$token" >&2
      exit 2
    fi
  done
}

validate_sequence() {
  local name="$1"
  local raw="$2"
  local token
  local count=0
  local base_count=0
  local variant_count=0
  for token in $raw; do
    count=$((count + 1))
    case "$token" in
      base) base_count=$((base_count + 1)) ;;
      variant) variant_count=$((variant_count + 1)) ;;
      *)
        printf '%s items must be base or variant, got %s\n' "$name" "$token" >&2
        exit 2
        ;;
    esac
  done
  if (( count == 0 || base_count == 0 || variant_count == 0 )); then
    printf '%s must contain at least one base and one variant arm\n' "$name" >&2
    exit 2
  fi
}

validate_float() {
  local name="$1"
  local value="$2"
  python3 - "$name" "$value" <<'PY'
import math
import sys

name = sys.argv[1]
value = sys.argv[2]
try:
    parsed = float(value)
except ValueError:
    print(f"{name} must be a finite float, got {value}", file=sys.stderr)
    sys.exit(2)
if not math.isfinite(parsed) or parsed <= 0:
    print(f"{name} must be a positive finite float, got {value}", file=sys.stderr)
    sys.exit(2)
PY
}

print_config() {
  if [[ "${compare_only:-0}" == "1" ]]; then
    printf 'MODE=compare_only\n'
  else
    printf 'MODE=run\n'
  fi
  printf 'LOG_DIR=%s\n' "$log_dir"
  printf 'FORWARD_SEQUENCE=%s\n' "$forward_sequence"
  printf 'MIRROR_SEQUENCE=%s\n' "$mirror_sequence"
  printf 'BASE_ENV=%s\n' "${base_env:-<empty>}"
  printf 'VARIANT_ENV=%s\n' "${variant_env:-<empty>}"
  printf 'PROMPT_LEN=%s\n' "$prompt_len"
  printf 'CANVAS_LEN=%s\n' "$canvas_len"
  printf 'MAX_LAYERS=%s\n' "$max_layers"
  printf 'WARMUPS=%s\n' "$warmups"
  printf 'REPEATS=%s\n' "$repeats"
  printf 'TRIM_PER_ARM=%s\n' "$trim_per_arm"
  printf 'FULL_ROUTES=%s\n' "$full_routes"
  printf 'REQUIRE_QUIET=%s\n' "$require_quiet"
  printf 'MIN_SPEEDUP=%s\n' "$min_speedup"
  printf 'PHASE_RUNNER=%s\n' "${phase_runner:-<build>}"
  printf 'FORWARD_TSV=%s\n' "${forward_tsv:-<run>}"
  printf 'MIRROR_TSV=%s\n' "${mirror_tsv:-<run>}"
}

bridge_object() {
  if [[ -n "${COGNI_ML_BRIDGE_O:-}" ]]; then
    printf '%s\n' "$COGNI_ML_BRIDGE_O"
  elif [[ -f "$repo_root/build/bridge.o" ]]; then
    printf '%s/build/bridge.o\n' "$repo_root"
  else
    printf '/Users/sergey/Projects/Crystal/cogni-ml/build/bridge.o\n'
  fi
}

ensure_phase_runner() {
  if [[ -n "$phase_runner" ]]; then
    if [[ ! -x "$phase_runner" ]]; then
      printf 'PHASE_RUNNER is not executable: %s\n' "$phase_runner" >&2
      exit 2
    fi
    return
  fi

  local bridge_o
  bridge_o="$(bridge_object)"
  if [[ ! -f "$bridge_o" ]]; then
    printf 'Set COGNI_ML_BRIDGE_O=/path/to/bridge.o or build bridge.o first\n' >&2
    exit 2
  fi

  phase_runner="/tmp/diffusion_gemma_phase_abba_mirror_${rev}"
  "$crystal_bin" build "$repo_root/scripts/diffusion_gemma_phase_abba.cr" \
    -o "$phase_runner" \
    --release \
    --error-trace \
    --link-flags="$bridge_o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
}

quiet_gate() {
  local label="$1"
  if [[ "$require_quiet" != "1" ]]; then
    return
  fi
  local gate_dir="$log_dir/quiet_${label}"
  LOG_DIR="$gate_dir" \
    QUIET_MS="$quiet_ms" \
    LOAD_THRESHOLD="$load_threshold" \
    TOTAL_THRESHOLD="$total_threshold" \
    SUMMARY_FORMAT=kv \
    "$repo_root/scripts/diffusion_gemma_quiet_gate_check.sh" | tee "$log_dir/quiet_${label}.txt"
}

run_sequence() {
  local label="$1"
  local sequence="$2"
  local out_tsv="$log_dir/${label}.tsv"
  local atlas_out="$log_dir/${label}_atlas.txt"
  local -a args=(
    --prompt-len "$prompt_len"
    --canvas-len "$canvas_len"
    --prompt-token "$prompt_token"
    --canvas-token "$canvas_token"
    --max-layers "$max_layers"
    --warmups "$warmups"
    --repeats "$repeats"
    --trim-per-arm "$trim_per_arm"
  )
  if [[ -n "$model" ]]; then
    args+=(--model "$model")
  fi
  if [[ "$full_routes" == "1" ]]; then
    args+=(--full-routes)
  fi

  quiet_gate "$label"
  "$phase_runner" "${args[@]}" --sequence "$sequence" --base-env "$base_env" --variant-env "$variant_env" >"$out_tsv"
  python3 "$repo_root/scripts/diffusion_gemma_phase_atlas.py" "$out_tsv" >"$atlas_out"
  printf 'phase_mirror_abba_run label=%s tsv=%s atlas=%s\n' "$label" "$out_tsv" "$atlas_out"
}

compare_pair() {
  local forward_path="$1"
  local mirror_path="$2"
  python3 - "$forward_path" "$mirror_path" "$min_speedup" <<'PY'
import csv
import math
import pathlib
import statistics
import sys

forward_path = pathlib.Path(sys.argv[1])
mirror_path = pathlib.Path(sys.argv[2])
min_speedup = float(sys.argv[3])

def as_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")

def as_int(row, key):
    try:
        return int(row.get(key, ""))
    except ValueError:
        return None

def median(values):
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan")
    return statistics.median(clean)

def summarize(path):
    data_lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    rows = [
        row
        for row in csv.DictReader(data_lines, delimiter="\t")
        if row.get("kind") == "sample"
        and row.get("measured") == "true"
        and row.get("arm") in {"base", "variant"}
    ]
    if not rows:
        raise SystemExit(f"no measured sample rows in {path}")
    base_values = [as_float(row, "total_ms") for row in rows if row.get("arm") == "base"]
    variant_values = [as_float(row, "total_ms") for row in rows if row.get("arm") == "variant"]
    if not base_values or not variant_values:
        raise SystemExit(f"missing base or variant samples in {path}")

    base_median = median(base_values)
    variant_median = median(variant_values)
    delta = base_median - variant_median
    speedup = base_median / variant_median if variant_median > 0 else float("nan")
    positions = sorted({idx for row in rows if (idx := as_int(row, "sequence_index")) is not None})
    position_medians = {
        idx: median(as_float(row, "total_ms") for row in rows if as_int(row, "sequence_index") == idx)
        for idx in positions
    }
    position_values = [value for value in position_medians.values() if math.isfinite(value)]
    position_span = max(position_values) - min(position_values) if position_values else float("nan")
    position_warning = math.isfinite(position_span) and position_span > abs(delta)
    checksums = {round(as_float(row, "checksum"), 6) for row in rows if math.isfinite(as_float(row, "checksum"))}
    checksum_ok = len(checksums) <= 1
    return {
        "base": base_median,
        "variant": variant_median,
        "delta": delta,
        "speedup": speedup,
        "position_span": position_span,
        "position_warning": position_warning,
        "checksum_ok": checksum_ok,
        "sample_count": len(rows),
    }

def fmt(value):
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"

forward = summarize(forward_path)
mirror = summarize(mirror_path)
for label, row in (("forward", forward), ("mirror", mirror)):
    print(
        "phase_mirror_abba_summary "
        f"label={label} "
        f"samples={row['sample_count']} "
        f"base_median_ms={fmt(row['base'])} "
        f"variant_median_ms={fmt(row['variant'])} "
        f"speedup={fmt(row['speedup'])} "
        f"delta_ms={fmt(row['delta'])} "
        f"position_span_ms={fmt(row['position_span'])} "
        f"position_warning={str(row['position_warning']).lower()} "
        f"checksum_ok={str(row['checksum_ok']).lower()}"
    )

if not forward["checksum_ok"] or not mirror["checksum_ok"]:
    decision = "reject_checksum_mismatch"
elif forward["position_warning"] or mirror["position_warning"]:
    decision = "blocked_by_sequence_position_bias"
elif forward["delta"] <= 0.0 or mirror["delta"] <= 0.0:
    decision = "reject_no_consistent_speedup"
elif min(forward["speedup"], mirror["speedup"]) < min_speedup:
    decision = "weak_below_min_speedup"
else:
    decision = "candidate_speedup_mirrored"

print(
    "phase_mirror_abba_result "
    f"decision={decision} "
    f"min_speedup={min_speedup:.6f} "
    f"forward_speedup={fmt(forward['speedup'])} "
    f"mirror_speedup={fmt(mirror['speedup'])}"
)
PY
}

mkdir -p "$log_dir"

require_positive_int PROMPT_LEN "$prompt_len"
require_positive_int CANVAS_LEN "$canvas_len"
require_positive_int MAX_LAYERS "$max_layers"
require_nonnegative_int WARMUPS "$warmups"
require_positive_int REPEATS "$repeats"
require_nonnegative_int TRIM_PER_ARM "$trim_per_arm"
require_nonnegative_int PROMPT_TOKEN "$prompt_token"
require_nonnegative_int CANVAS_TOKEN "$canvas_token"
require_nonnegative_int QUIET_MS "$quiet_ms"
validate_float MIN_SPEEDUP "$min_speedup"

if [[ "$full_routes" != "0" && "$full_routes" != "1" ]]; then
  printf 'FULL_ROUTES must be 0 or 1, got %s\n' "$full_routes" >&2
  exit 2
fi
if [[ "$require_quiet" != "0" && "$require_quiet" != "1" ]]; then
  printf 'REQUIRE_QUIET must be 0 or 1, got %s\n' "$require_quiet" >&2
  exit 2
fi

compare_only=0
if [[ -n "$forward_tsv" || -n "$mirror_tsv" ]]; then
  compare_only=1
  if [[ -z "$forward_tsv" || -z "$mirror_tsv" ]]; then
    printf 'FORWARD_TSV and MIRROR_TSV must be supplied together\n' >&2
    exit 2
  fi
  if [[ ! -f "$forward_tsv" || ! -f "$mirror_tsv" ]]; then
    printf 'FORWARD_TSV and MIRROR_TSV must point to existing files\n' >&2
    exit 2
  fi
else
  validate_env BASE_ENV "$base_env"
  validate_env VARIANT_ENV "$variant_env"
  validate_sequence FORWARD_SEQUENCE "$forward_sequence"
  validate_sequence MIRROR_SEQUENCE "$mirror_sequence"
fi

print_config

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if (( compare_only == 1 )); then
  python3 "$repo_root/scripts/diffusion_gemma_phase_atlas.py" "$forward_tsv" >"$log_dir/forward_atlas.txt"
  python3 "$repo_root/scripts/diffusion_gemma_phase_atlas.py" "$mirror_tsv" >"$log_dir/mirror_atlas.txt"
  compare_pair "$forward_tsv" "$mirror_tsv"
  exit 0
fi

ensure_phase_runner
run_sequence forward "$forward_sequence"
run_sequence mirror "$mirror_sequence"
compare_pair "$log_dir/forward.tsv" "$log_dir/mirror.tsv"
