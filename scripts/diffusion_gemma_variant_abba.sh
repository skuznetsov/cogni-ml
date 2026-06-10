#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_variant_abba_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$log_dir"

steps="${STEPS:-}"
sequence="${SEQUENCE:-base variant variant base}"
base_env="${BASE_ENV:-}"
variant_env="${VARIANT_ENV:-}"
quiet_ms="${QUIET_MS:-0}"
quiet_poll_ms="${QUIET_POLL_MS:-1000}"
load_threshold="${LOAD_THRESHOLD:-50}"
total_threshold="${TOTAL_THRESHOLD:-100}"
require_quiet="${REQUIRE_QUIET:-0}"

printf 'log_dir=%s\n' "$log_dir"
printf 'steps=%s\n' "${steps:-proto-default}"
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

def sample() -> tuple[float, float, str]:
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
  run_log="$log_dir/run_${idx}_${arm}.txt"
  printf 'variant_abba_run index=%d arm=%s\n' "$idx" "$arm"
  wait_quiet "run_${idx}_${arm}"
  env_args=(LOG_DIR="$log_dir")
  if [[ -n "$steps" ]]; then
    env_args+=(STEPS="$steps")
  fi
  env "${env_args[@]}" "${tokens[@]}" "$repo_root/scripts/diffusion_gemma_proto.sh" >"$run_log" 2>&1
  tail -n 16 "$run_log"
done

python3 - "$log_dir" <<'PY'
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("run_*.txt")):
    match = re.match(r"run_(\d+)_(base|variant)\.txt", path.name)
    if not match:
        continue
    text = path.read_text(errors="replace")
    total = re.search(r"total_ms=([0-9.]+)", text)
    step = re.search(r"step_ms=([0-9.]+)", text)
    answer_status = re.search(r"diffusion_gemma_answer_status=(ok|no_clean_answer)", text)
    if total and step:
        rows.append((match.group(2), float(total.group(1)), float(step.group(1)), answer_status.group(1) if answer_status else "unknown", path.name))

if not rows:
    print("diffusion_gemma_variant_abba_result parse_error=1")
    sys.exit(3)

for arm, total, step, answer_status, name in rows:
    print(f"row arm={arm} total_ms={total:.2f} step_ms={step:.2f} answer_status={answer_status} source={name}")

by_arm = {}
clean_by_arm = {}
for arm, total, _, answer_status, _ in rows:
    by_arm.setdefault(arm, []).append(total)
    clean_by_arm[arm] = clean_by_arm.get(arm, 0) + int(answer_status == "ok")

if {"base", "variant"} <= set(by_arm):
    base = statistics.median(by_arm["base"])
    variant = statistics.median(by_arm["variant"])
    ratio = variant / base if base else float("inf")
    print(f"diffusion_gemma_variant_abba_result base_median_ms={base:.2f} variant_median_ms={variant:.2f} variant_over_base={ratio:.4f} speedup={1/ratio:.4f}x")
    print(f"diffusion_gemma_variant_abba_quality base_clean={clean_by_arm.get('base', 0)}/{len(by_arm['base'])} variant_clean={clean_by_arm.get('variant', 0)}/{len(by_arm['variant'])}")
else:
    print("diffusion_gemma_variant_abba_result incomplete_arms=1")
    sys.exit(4)
PY
