#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_capture_replay.sh CAPTURE.jsonl [MIN_ALLOWED]

Converts QWEN35_ALLOWED_HEAD_CAPTURE_PATH JSONL into raw Float32 hidden batches,
copies them to the Raspberry Pi 5 probe directory, and replays them through the
Q6 indexed-head V3D probe.

Environment:
  RPI5_HOST        SSH host, default raspberrypi.local
  RPI5_REMOTE_DIR Remote probe dir, default ~/cogni-ml-vulkan-probe
  REPEATS=30       Probe repeats.
  RPI5_WARMUPS=3   Untimed GPU dispatches before measurement.
  MIN_ALLOWED=4    Filter threshold for the hybrid V3D replay.
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
probe="scripts/rpi5_q6_frontier_probe.sh"
[[ -f "$converter" && -f "$probe" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

metric_from_output() {
  local key="$1"
  awk -v key="$key" '
    /^max_abs_diff=/ {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == key) print kv[2]
      }
    }'
}

run_replay() {
  local min="$1"
  local tag="$2"
  local local_f32="/tmp/rpi5_allowed_head_${tag}_$$.f32"
  local remote_base
  remote_base="$(basename "$local_f32")"

  local plan
  plan="$(MIN_ALLOWED="$min" crystal "$converter" "$capture" "$local_f32")"
  printf "%s\n" "$plan"

  local rows max_allowed ids_groups first_ids mode
  rows="$(awk -F= '$1=="replay_rows" {print $2}' <<<"$plan")"
  max_allowed="$(awk -F= '$1=="max_allowed" {print $2}' <<<"$plan")"
  ids_groups="$(awk -F= '$1=="ids_groups" {print $2}' <<<"$plan")"
  first_ids="${ids_groups%%:*}"
  mode="q6idx${max_allowed}_l256"

  scp -q "$local_f32" "$host:$remote_dir/$remote_base"
  rm -f "$local_f32"

  local output
  output="$(
    RAW_OUTPUT=1 \
    RPI5_WARMUPS="$warmups" \
    RPI5_BATCH="$rows" \
    RPI5_MODE="$mode" \
    RPI5_ROW_IDS_CSV_BATCH="$ids_groups" \
    RPI5_X_F32_LOAD="$remote_base" \
    bash "$probe" "capture_replay:${tag}" "$first_ids" "$repeats"
  )"
  ssh "$host" "rm -f $remote_dir/$remote_base"

  if [[ "$raw_output" != "0" ]]; then
    printf "%s\n" "$output"
  fi

  local gpu cpu speedup diff throttled
  gpu="$(metric_from_output gpu_ms_avg <<<"$output")"
  cpu="$(metric_from_output cpu_ms <<<"$output")"
  speedup="$(metric_from_output speedup <<<"$output")"
  diff="$(metric_from_output max_abs_diff <<<"$output")"
  throttled="$(awk '/^throttled=/ {print $1}' <<<"$output" | tail -1)"
  printf "capture_replay_result\tlabel=%s\tmin_allowed=%s\tbatch=%s\tmax_allowed=%s\tgpu_ms=%s\tcpu_ms=%s\tspeedup=%s\tmax_abs_diff=%s\t%s\n" \
    "$tag" "$min" "$rows" "$max_allowed" "$gpu" "$cpu" "$speedup" "$diff" "$throttled"
}

all_output="$(run_replay 0 all)"
printf "%s\n" "$all_output"

if (( min_allowed > 0 )); then
  filtered_output="$(run_replay "$min_allowed" "min${min_allowed}")"
  printf "%s\n" "$filtered_output"

  all_cpu="$(awk -F '\t' '/^capture_replay_result/ && $2=="label=all" {for (i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="cpu_ms") print kv[2]}}' <<<"$all_output")"
  filtered_cpu="$(awk -F '\t' '/^capture_replay_result/ {for (i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="cpu_ms") print kv[2]}}' <<<"$filtered_output")"
  filtered_gpu="$(awk -F '\t' '/^capture_replay_result/ {for (i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="gpu_ms") print kv[2]}}' <<<"$filtered_output")"
  awk -v all_cpu="$all_cpu" -v filtered_cpu="$filtered_cpu" -v filtered_gpu="$filtered_gpu" -v min="$min_allowed" '
    BEGIN {
      if (all_cpu > 0 && filtered_cpu >= 0 && filtered_gpu > 0) {
        hybrid = all_cpu - filtered_cpu + filtered_gpu
        printf "capture_replay_hybrid_estimate\tmin_allowed=%s\tall_cpu_ms=%.3f\tfiltered_cpu_ms=%.3f\tfiltered_gpu_ms=%.3f\thybrid_ms=%.3f\tspeedup=%.3fx\n",
          min, all_cpu, filtered_cpu, filtered_gpu, hybrid, all_cpu / hybrid
      }
    }'
fi
