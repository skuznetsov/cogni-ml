#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_resident_stream_bench.sh CAPTURE.jsonl [MIN_ALLOWED]

Replays captured Qwen3.5-2B allowed-head hidden rows through one resident
Raspberry Pi 5 V3D probe process. Unlike capture replay, this keeps Vulkan
objects, the prepacked Q6 head, and buffers alive while issuing one request per
captured decode step.

Environment:
  RPI5_HOST        SSH host, default raspberrypi.local
  RPI5_REMOTE_DIR Remote probe dir, default ~/cogni-ml-vulkan-probe
  REPEATS=30       Resident stream passes over all selected rows.
  MIN_ALLOWED=4    Filter threshold for V3D-routed rows.
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
raw_output="${RAW_OUTPUT:-0}"
converter="scripts/export_allowed_head_capture_replay.cr"
[[ -f "$converter" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

metric_from_output() {
  local key="$1"
  awk -v key="$key" '
    /^resident_stream_ms_avg=/ || /^max_abs_diff=/ || /^prepack_load_ms=/ {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == key) print kv[2]
      }
    }'
}

local_f32="/tmp/rpi5_resident_stream_$$.f32"
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

first_ids="$(
  awk -v max="$max_allowed" -F, '
    {
      for (i = 1; i <= max; i++) {
        if (i <= NF) v = $i
        else v = 0
        if (i > 1) printf ","
        printf "%s", v
      }
      printf "\n"
    }' <<<"$first_ids"
)"

scp -q "$local_f32" "$host:$remote_dir/$remote_base"

output="$(
  ssh "$host" bash -s -- "$remote_dir" "$mode" "$first_ids" "$ids_groups" "$remote_base" "$rows" "$repeats" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
mode="$2"
ids_csv="$3"
ids_groups="$4"
x_f32_load="$5"
rows="$6"
repeats="$7"

root="$HOME/cogni-vulkan-runtime/root"
if [[ "$remote_dir" == "~/"* ]]; then
  remote_dir="$HOME/${remote_dir#~/}"
fi
export VK_ICD_FILENAMES="$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json"
export LD_LIBRARY_PATH="$root/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RPI5_Q6_PREPACK_LOAD="qwen35_2b_token_embd_q6.pre20"
export RPI5_ROW_IDS_CSV="$ids_csv"
export RPI5_BATCH=1
export RPI5_RESIDENT_REQUESTS="$rows"
export RPI5_RESIDENT_STREAM_X_F32_LOAD="$x_f32_load"
export RPI5_RESIDENT_ROW_IDS_CSV_BATCH="$ids_groups"

cd "$remote_dir"
./rpi5_vulkan_q4k_probe rpi5_q6_matvec_pre_idx_l256.spv file qwen35_2b_token_embd_q6.cvgp "$repeats" "$mode"
rm -f "$x_f32_load"
vcgencmd get_throttled || true
REMOTE
)"

if [[ "$raw_output" != "0" ]]; then
  printf "%s\n" "$output"
fi

stream_ms="$(metric_from_output resident_stream_ms_avg <<<"$output")"
cpu_ms="$(metric_from_output resident_stream_cpu_ms_avg <<<"$output")"
speedup="$(metric_from_output resident_stream_speedup <<<"$output")"
diff="$(metric_from_output resident_stream_max_abs_diff <<<"$output")"
mismatches="$(metric_from_output resident_stream_top1_mismatches <<<"$output")"
prepack_load="$(metric_from_output prepack_load_ms <<<"$output")"
throttled="$(awk '/^throttled=/ {print $1}' <<<"$output" | tail -1)"

awk -v rows="$rows" -v min="$min_allowed" -v max_allowed="$max_allowed" \
    -v repeats="$repeats" -v stream_ms="$stream_ms" -v cpu_ms="$cpu_ms" \
    -v speedup="$speedup" -v diff="$diff" -v mismatches="$mismatches" \
    -v prepack="$prepack_load" -v throttled="$throttled" '
  BEGIN {
    printf "resident_stream_bench_result\tmin_allowed=%s\trequests=%s\tmax_allowed=%s\trepeats=%s\tresident_stream_ms=%s\tcpu_selected_ms=%s\tspeedup=%s\tmax_abs_diff=%s\ttop1_mismatches=%s\tone_time_prepack_load_ms=%s\t%s\n",
      min, rows, max_allowed, repeats, stream_ms, cpu_ms, speedup, diff, mismatches, prepack, throttled
  }'
