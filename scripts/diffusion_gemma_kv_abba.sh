#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_kv_abba_$(date +%Y%m%d%H%M%S)}"
mkdir -p "$log_dir"

steps="${STEPS:-4}"
sequence="${SEQUENCE:-off on on off}"
quiet_ms="${QUIET_MS:-0}"
quiet_poll_ms="${QUIET_POLL_MS:-1000}"
load_threshold="${LOAD_THRESHOLD:-50}"
total_threshold="${TOTAL_THRESHOLD:-100}"
require_quiet="${REQUIRE_QUIET:-0}"

printf 'log_dir=%s\n' "$log_dir"
printf 'steps=%s\n' "$steps"
printf 'sequence=%s\n' "$sequence"
printf 'quiet_ms=%s\n' "$quiet_ms"
printf 'load_threshold=%s\n' "$load_threshold"
printf 'total_threshold=%s\n' "$total_threshold"
printf 'require_quiet=%s\n' "$require_quiet"

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

idx=0
for mode in $sequence; do
  idx=$((idx + 1))
  run_log="$log_dir/run_${idx}_${mode}.txt"
  printf 'abba_run index=%d kv_cache=%s\n' "$idx" "$mode"
  wait_quiet "run_${idx}_${mode}"
  LOG_DIR="$log_dir" KV_CACHE="$mode" STEPS="$steps" "$repo_root/scripts/diffusion_gemma_proto.sh" >"$run_log" 2>&1
  tail -n 12 "$run_log"
done

python3 - "$log_dir" <<'PY'
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("run_*.txt")):
    text = path.read_text(errors="replace")
    mode = re.search(r"kv_cache=(on|off)", text)
    total = re.search(r"total_ms=([0-9.]+)", text)
    step = re.search(r"step_ms=([0-9.]+)", text)
    if mode and total and step:
        rows.append((mode.group(1), float(total.group(1)), float(step.group(1)), path.name))

if not rows:
    print("diffusion_gemma_kv_abba_result parse_error=1")
    sys.exit(3)

for mode, total, step, name in rows:
    print(f"row mode={mode} total_ms={total:.2f} step_ms={step:.2f} source={name}")

by_mode = {}
for mode, total, step, _ in rows:
    by_mode.setdefault(mode, []).append(total)

if {"on", "off"} <= set(by_mode):
    off_med = statistics.median(by_mode["off"])
    on_med = statistics.median(by_mode["on"])
    ratio = on_med / off_med if off_med else float("inf")
    print(f"diffusion_gemma_kv_abba_result off_median_ms={off_med:.2f} on_median_ms={on_med:.2f} on_over_off={ratio:.4f} speedup={1/ratio:.4f}x")
else:
    print("diffusion_gemma_kv_abba_result incomplete_modes=1")
    sys.exit(4)
PY
