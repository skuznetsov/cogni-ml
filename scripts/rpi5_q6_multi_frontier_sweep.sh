#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_multi_frontier_sweep.sh MODEL.gguf

Runs several tokenizer-derived Q6 allowed-head frontiers in one Raspberry Pi 5
V3D q6idx submit using per-batch row-id metadata.

Environment:
  MAX_FRONTIERS=3       Number of matched frontiers to group.
  LABEL_REGEX=REGEX    Frontier labels to group.
  REPEATS=40           Probe repeats.
  RPI5_WARMUPS=3       Untimed GPU dispatches before measurement.
  RAW_OUTPUT=0         Suppress full probe output.
  DRY_RUN=1            Print the grouped command inputs without SSH.
USAGE
  exit 2
}

model_path="${1:-}"
[[ -n "$model_path" ]] || usage

max_frontiers="${MAX_FRONTIERS:-3}"
label_regex="${LABEL_REGEX:-^(tool_call_prefix:start|finite_values:read_file\\.limit|finite_values:edit_mode\\.mode)$}"
repeats="${REPEATS:-40}"
warmups="${RPI5_WARMUPS:-3}"
raw_output="${RAW_OUTPUT:-0}"
dry_run="${DRY_RUN:-0}"

estimator="scripts/estimate_qwen35_allowed_frontiers.cr"
probe="scripts/rpi5_q6_frontier_probe.sh"
[[ -f "$estimator" && -f "$probe" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

selected="$(
  crystal "$estimator" "$model_path" |
    awk -F '\t' -v label_regex="$label_regex" -v max_frontiers="$max_frontiers" '
      {
        label=$1
        allowed=""
        route=""
        ids_csv=""
        for (i = 2; i <= NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "allowed") allowed=kv[2]
          if (kv[1] == "route") route=kv[2]
          if (kv[1] == "ids_csv") ids_csv=kv[2]
        }
        if (ids_csv != "" && (label_regex == "" || label ~ label_regex)) {
          print label "\t" allowed "\t" route "\t" ids_csv
          count++
          if (count >= max_frontiers) exit
        }
      }'
)"

if [[ -z "$selected" ]]; then
  echo "no frontiers matched label_regex=${label_regex:-<none>}" >&2
  exit 1
fi

batch=0
max_allowed=0
labels_csv=""
batch_ids_csv=""
first_ids_csv=""
while IFS=$'\t' read -r label allowed route ids_csv; do
  [[ -n "$label" ]] || continue
  batch=$((batch + 1))
  if (( allowed > max_allowed )); then
    max_allowed="$allowed"
  fi
  labels_csv="${labels_csv}${labels_csv:+,}${label}"
  batch_ids_csv="${batch_ids_csv}${batch_ids_csv:+:}${ids_csv}"
  if [[ -z "$first_ids_csv" ]]; then
    first_ids_csv="$ids_csv"
  fi
  printf "multi_frontier_row\tidx=%s\tlabel=%s\tallowed=%s\test_route=%s\tids_csv=%s\n" \
    "$batch" "$label" "$allowed" "$route" "$ids_csv"
done <<<"$selected"

mode="q6idx${max_allowed}_l256"
printf "multi_frontier_begin\tmodel=%s\tbatch=%s\tmax_allowed=%s\tmode=%s\trepeats=%s\twarmups=%s\tlabels=%s\n" \
  "$model_path" "$batch" "$max_allowed" "$mode" "$repeats" "$warmups" "$labels_csv"

if [[ "$dry_run" == "1" ]]; then
  printf "multi_frontier_policy\tdry_run=true\tbatch_ids_csv=%s\n" "$batch_ids_csv"
  exit 0
fi

probe_output="$(
  RPI5_WARMUPS="$warmups" \
  RPI5_BATCH="$batch" \
  RPI5_MODE="$mode" \
  RPI5_ROW_IDS_CSV_BATCH="$batch_ids_csv" \
  bash "$probe" "multi_frontier:${labels_csv}" "$first_ids_csv" "$repeats"
)"

if [[ "$raw_output" != "0" ]]; then
  printf "%s\n" "$probe_output"
fi

awk -v batch="$batch" -v max_allowed="$max_allowed" -v labels="$labels_csv" '
  /^max_abs_diff=/ {
    for (i = 1; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == "max_abs_diff") diff=kv[2]
      if (kv[1] == "gpu_ms_avg") gpu=kv[2]
      if (kv[1] == "cpu_ms") cpu=kv[2]
      if (kv[1] == "speedup") speedup=kv[2]
    }
  }
  /^throttled=/ { throttled=$1 }
  END {
    gpu_per = (gpu != "" && batch > 0) ? sprintf("%.6f", gpu / batch) : ""
    cpu_per = (cpu != "" && batch > 0) ? sprintf("%.6f", cpu / batch) : ""
    printf "multi_frontier_result\tlabels=%s\tbatch=%s\tmax_allowed=%s\tgpu_ms=%s\tcpu_ms=%s\tgpu_ms_per_frontier=%s\tcpu_ms_per_frontier=%s\tspeedup=%s\tmax_abs_diff=%s\t%s\n",
      labels, batch, max_allowed, gpu, cpu, gpu_per, cpu_per, speedup, diff, throttled
  }' <<<"$probe_output"
