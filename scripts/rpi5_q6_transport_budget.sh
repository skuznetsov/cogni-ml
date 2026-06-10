#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_transport_budget.sh CAPTURE.jsonl [MIN_ALLOWED]

Measures the current non-product transport tax around the Raspberry Pi 5 Q6
indexed-head probe. It replays captured hidden rows through one remote probe
process, then reports the remote wall time alongside prepack load, GPU dispatch,
and CPU oracle time.

Environment:
  RPI5_HOST        SSH host, default raspberrypi.local
  RPI5_REMOTE_DIR Remote probe dir, default ~/cogni-ml-vulkan-probe
  REPEATS=30       Timed GPU submits inside the probe.
  RPI5_WARMUPS=3   Untimed GPU submits before measurement.
  MIN_ALLOWED=4    Filter threshold for the V3D-routed rows.
  RAW_OUTPUT=0     Suppress full probe output; keep summary rows.
USAGE
  exit 2
}

capture="${1:-}"
min_allowed="${2:-${MIN_ALLOWED:-4}}"
[[ -n "$capture" && -f "$capture" ]] || usage
[[ "$min_allowed" =~ ^[0-9]+$ ]] || {
  echo "MIN_ALLOWED must be non-negative" >&2
  exit 2
}

host="${RPI5_HOST:-raspberrypi.local}"
remote_dir="${RPI5_REMOTE_DIR:-~/cogni-ml-vulkan-probe}"
repeats="${REPEATS:-30}"
warmups="${RPI5_WARMUPS:-3}"
raw_output="${RAW_OUTPUT:-0}"
converter="scripts/export_allowed_head_capture_replay.cr"
[[ -f "$converter" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

metric_from_output() {
  local key="$1"
  awk -v key="$key" '
    /^max_abs_diff=/ || /^prepack_load_ms=/ || /^remote_wall_ms=/ {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == key) print kv[2]
      }
    }'
}

local_f32="/tmp/rpi5_transport_budget_$$.f32"
remote_base="$(basename "$local_f32")"
cleanup() {
  rm -f "$local_f32"
}
trap cleanup EXIT

plan="$(MIN_ALLOWED="$min_allowed" crystal "$converter" "$capture" "$local_f32")"
printf "%s\n" "$plan"

rows="$(awk -F= '$1=="replay_rows" {print $2}' <<<"$plan")"
max_allowed="$(awk -F= '$1=="max_allowed" {print $2}' <<<"$plan")"
ids_groups="$(awk -F= '$1=="ids_groups" {print $2}' <<<"$plan")"
first_ids="${ids_groups%%:*}"
mode="q6idx${max_allowed}_l256"

if [[ -z "$rows" || -z "$max_allowed" || -z "$ids_groups" || "$rows" == "0" ]]; then
  echo "no replay rows selected" >&2
  exit 1
fi

scp -q "$local_f32" "$host:$remote_dir/$remote_base"

output="$(
  ssh "$host" bash -s -- "$remote_dir" "$mode" "$first_ids" "$ids_groups" "$remote_base" "$rows" "$repeats" "$warmups" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
mode="$2"
ids_csv="$3"
ids_groups="$4"
x_f32_load="$5"
rows="$6"
repeats="$7"
warmups="$8"

root="$HOME/cogni-vulkan-runtime/root"
if [[ "$remote_dir" == "~/"* ]]; then
  remote_dir="$HOME/${remote_dir#~/}"
fi
export VK_ICD_FILENAMES="$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json"
export LD_LIBRARY_PATH="$root/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RPI5_Q6_PREPACK_LOAD="qwen35_2b_token_embd_q6.pre20"
export RPI5_ROW_IDS_CSV="$ids_csv"
export RPI5_ROW_IDS_CSV_BATCH="$ids_groups"
export RPI5_WARMUPS="$warmups"
export RPI5_BATCH="$rows"
export RPI5_X_F32_LOAD="$x_f32_load"

cd "$remote_dir"
t0="$(date +%s%N)"
./rpi5_vulkan_q4k_probe rpi5_q6_matvec_pre_idx_l256.spv file qwen35_2b_token_embd_q6.cvgp "$repeats" "$mode"
t1="$(date +%s%N)"
rm -f "$x_f32_load"
awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "remote_wall_ms=%.3f\n", (t1 - t0) / 1000000.0 }'
vcgencmd get_throttled || true
REMOTE
)"

if [[ "$raw_output" != "0" ]]; then
  printf "%s\n" "$output"
fi

gpu="$(metric_from_output gpu_ms_avg <<<"$output")"
cpu="$(metric_from_output cpu_ms <<<"$output")"
speedup="$(metric_from_output speedup <<<"$output")"
diff="$(metric_from_output max_abs_diff <<<"$output")"
prepack_load="$(metric_from_output prepack_load_ms <<<"$output")"
remote_wall="$(metric_from_output remote_wall_ms <<<"$output")"
throttled="$(awk '/^throttled=/ {print $1}' <<<"$output" | tail -1)"

awk -v rows="$rows" -v min="$min_allowed" -v max_allowed="$max_allowed" \
    -v repeats="$repeats" -v warmups="$warmups" -v gpu="$gpu" -v cpu="$cpu" \
    -v speedup="$speedup" -v diff="$diff" -v prepack="$prepack_load" \
    -v remote_wall="$remote_wall" -v throttled="$throttled" '
  BEGIN {
    dispatch_ms = gpu * (repeats + warmups)
    setup_ms = remote_wall - prepack - dispatch_ms - cpu
    overhead_vs_gpu = gpu > 0 ? setup_ms / gpu : 0
    printf "transport_budget_result\tmin_allowed=%s\tbatch=%s\tmax_allowed=%s\trepeats=%s\twarmups=%s\tgpu_ms=%s\tcpu_ms=%s\tspeedup=%s\tmax_abs_diff=%s\tprepack_load_ms=%.3f\tdispatch_total_ms=%.3f\tremote_wall_ms=%.3f\tsetup_overhead_ms=%.3f\tsetup_overhead_vs_gpu_avg=%.1fx\t%s\n",
      min, rows, max_allowed, repeats, warmups, gpu, cpu, speedup, diff, prepack, dispatch_ms, remote_wall, setup_ms, overhead_vs_gpu, throttled
  }'
