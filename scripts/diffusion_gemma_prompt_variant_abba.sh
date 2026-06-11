#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_prompt_variant_abba_$(date +%Y%m%d%H%M%S)}"
sequence="${SEQUENCE:-base variant variant base}"
base_env="${BASE_ENV:-}"
variant_env="${VARIANT_ENV:-}"
quiet_ms="${QUIET_MS:-0}"
quiet_poll_ms="${QUIET_POLL_MS:-1000}"
load_threshold="${LOAD_THRESHOLD:-50}"
total_threshold="${TOTAL_THRESHOLD:-100}"
require_quiet="${REQUIRE_QUIET:-0}"

mkdir -p "$log_dir"

printf 'log_dir=%s\n' "$log_dir"
printf 'sequence=%s\n' "$sequence"
printf 'base_env=%s\n' "${base_env:-<empty>}"
printf 'variant_env=%s\n' "${variant_env:-<empty>}"
printf 'quiet_ms=%s\n' "$quiet_ms"
printf 'load_threshold=%s\n' "$load_threshold"
printf 'total_threshold=%s\n' "$total_threshold"
printf 'require_quiet=%s\n' "$require_quiet"

env_tokens() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ -z "$raw" ]]; then
    return
  fi

  local token
  for token in $raw; do
    if [[ ! "$token" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      printf 'error: env token must be KEY=VALUE, got %s\n' "$token" >&2
      exit 2
    fi
    out_ref+=("$token")
  done
}

wait_quiet() {
  local label="$1"
  python3 - "$label" "$quiet_ms" "$quiet_poll_ms" "$load_threshold" "$total_threshold" "$require_quiet" <<'PY'
import subprocess
import sys
import time

label = sys.argv[1]
quiet_ms = int(sys.argv[2])
quiet_poll_ms = int(sys.argv[3])
process_limit = float(sys.argv[4])
total_limit = float(sys.argv[5])
require_quiet = sys.argv[6] == "1"

if quiet_ms < 0:
    print("quiet_gate_result status=error reason=negative_quiet_ms", file=sys.stderr)
    sys.exit(2)
if quiet_poll_ms <= 0:
    print("quiet_gate_result status=error reason=nonpositive_quiet_poll_ms", file=sys.stderr)
    sys.exit(2)

def sample():
    out = subprocess.check_output(["ps", "-axo", "pcpu=,comm="], text=True)
    max_cpu = 0.0
    max_comm = ""
    total_cpu = 0.0
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            cpu = float(parts[0])
        except ValueError:
            continue
        comm = parts[1] if len(parts) > 1 else ""
        total_cpu += cpu
        if cpu > max_cpu:
            max_cpu = cpu
            max_comm = comm
    return max_cpu, total_cpu, max_comm

deadline = time.monotonic() + quiet_ms / 1000.0
while True:
    max_cpu, total_cpu, max_comm = sample()
    quiet = (process_limit <= 0 or max_cpu < process_limit) and (total_limit <= 0 or total_cpu < total_limit)
    if quiet:
        print(f"quiet_gate_result status=ok label={label} max_process_cpu={max_cpu:.1f} total_cpu={total_cpu:.1f} max_process={max_comm}")
        sys.exit(0)
    if quiet_ms <= 0 or time.monotonic() >= deadline:
        status = "fail" if require_quiet else "warn"
        print(f"quiet_gate_result status={status} label={label} max_process_cpu={max_cpu:.1f} total_cpu={total_cpu:.1f} max_process={max_comm}")
        sys.exit(4 if require_quiet else 0)
    time.sleep(quiet_poll_ms / 1000.0)
PY
}

base_tokens=()
variant_tokens=()
env_tokens "$base_env" base_tokens
env_tokens "$variant_env" variant_tokens

for arm in $sequence; do
  case "$arm" in
    base|variant) ;;
    *) printf 'error: SEQUENCE items must be base or variant, got %s\n' "$arm" >&2; exit 2 ;;
  esac
done

idx=0
for arm in $sequence; do
  case "$arm" in
    base) tokens=("${base_tokens[@]}") ;;
    variant) tokens=("${variant_tokens[@]}") ;;
  esac

  idx=$((idx + 1))
  run_tsv="$log_dir/run_${idx}_${arm}.tsv"
  run_log="$log_dir/run_${idx}_${arm}.log"
  printf 'prompt_variant_abba_run index=%d arm=%s out=%s\n' "$idx" "$arm" "$run_tsv"
  wait_quiet "run_${idx}_${arm}" | tee -a "$run_log"
  env OUT="$run_tsv" "${tokens[@]}" "$repo_root/scripts/diffusion_gemma_prompt_perf_probe.sh" >>"$run_log" 2>&1
done

python3 - "$log_dir" <<'PY'
import csv
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("run_*.tsv")):
    match = re.match(r"run_(\d+)_(base|variant)\.tsv", path.name)
    if not match:
        continue
    index = int(match.group(1))
    arm = match.group(2)
    with path.open(newline="", encoding="utf-8") as io:
        for row in csv.DictReader(io, delimiter="\t"):
            row["_index"] = index
            row["_arm"] = arm
            row["_source"] = path.name
            rows.append(row)

if not rows:
    print("diffusion_gemma_prompt_variant_abba_result parse_error=1")
    sys.exit(3)

def as_float(row, key):
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")

def unique_value(rows, key):
    values = sorted({row.get(key, "") for row in rows if row.get(key, "") != ""})
    return ",".join(values) if values else "NA"

for row in rows:
    print(
        "row "
        f"arm={row['_arm']} case={row.get('case', '')} status={row.get('status', '')} "
        f"fused={row.get('prompt_projection_fused_norm_rope', '')} "
        f"cache_ms={as_float(row, 'prompt_cache_ms'):.3f} "
        f"projection_ms={as_float(row, 'prompt_projection_ms'):.3f} "
        f"head_norm_ms={as_float(row, 'prompt_projection_head_norm_ms'):.3f} "
        f"rope_ms={as_float(row, 'prompt_projection_rope_ms'):.3f} "
        f"source={row['_source']}"
    )

bad = [row for row in rows if row.get("status") != "ok"]
if bad:
    print(f"diffusion_gemma_prompt_variant_abba_result non_ok_rows={len(bad)}")
    sys.exit(4)

cases = sorted({row.get("case", "") for row in rows})
for case in cases:
    by_arm = {}
    for row in rows:
        if row.get("case") == case:
            by_arm.setdefault(row["_arm"], []).append(row)
    if {"base", "variant"} <= set(by_arm):
        base_cache = statistics.median(as_float(row, "prompt_cache_ms") for row in by_arm["base"])
        variant_cache = statistics.median(as_float(row, "prompt_cache_ms") for row in by_arm["variant"])
        base_projection = statistics.median(as_float(row, "prompt_projection_ms") for row in by_arm["base"])
        variant_projection = statistics.median(as_float(row, "prompt_projection_ms") for row in by_arm["variant"])
        cache_ratio = variant_cache / base_cache if base_cache else float("inf")
        projection_ratio = variant_projection / base_projection if base_projection else float("inf")
        print(
            "diffusion_gemma_prompt_variant_abba_case "
            f"case={case} "
            f"base_cache_ms={base_cache:.3f} variant_cache_ms={variant_cache:.3f} "
            f"variant_over_base_cache={cache_ratio:.4f} cache_speedup={1 / cache_ratio if cache_ratio else 0.0:.4f}x "
            f"base_projection_ms={base_projection:.3f} variant_projection_ms={variant_projection:.3f} "
            f"variant_over_base_projection={projection_ratio:.4f} projection_speedup={1 / projection_ratio if projection_ratio else 0.0:.4f}x"
        )
        print(
            "diffusion_gemma_prompt_variant_abba_route "
            f"case={case} "
            f"base_backend={unique_value(by_arm['base'], 'loop_context_backend')} "
            f"variant_backend={unique_value(by_arm['variant'], 'loop_context_backend')} "
            f"base_batch_rows={unique_value(by_arm['base'], 'loop_context_batch_rows')} "
            f"variant_batch_rows={unique_value(by_arm['variant'], 'loop_context_batch_rows')} "
            f"base_fixed_gqa2={unique_value(by_arm['base'], 'loop_context_fixed_gqa2')} "
            f"variant_fixed_gqa2={unique_value(by_arm['variant'], 'loop_context_fixed_gqa2')}"
        )
        for metric in (
            "loop_ms_median",
            "loop_decode_context_ms",
            "loop_decode_qkv_ms",
            "loop_decode_attention_out_ms",
        ):
            base_metric = statistics.median(as_float(row, metric) for row in by_arm["base"])
            variant_metric = statistics.median(as_float(row, metric) for row in by_arm["variant"])
            metric_ratio = variant_metric / base_metric if base_metric else float("inf")
            print(
                "diffusion_gemma_prompt_variant_abba_metric "
                f"case={case} metric={metric} "
                f"base_ms={base_metric:.3f} variant_ms={variant_metric:.3f} "
                f"variant_over_base={metric_ratio:.4f} speedup={1 / metric_ratio if metric_ratio else 0.0:.4f}x"
            )
    else:
        print(f"diffusion_gemma_prompt_variant_abba_case case={case} incomplete_arms=1")
        sys.exit(4)
PY
