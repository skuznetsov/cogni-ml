#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_batch_sweep.sh MODEL.gguf

Runs tokenizer-derived Q6 allowed-head frontiers on the Raspberry Pi 5 while
varying RPI5_BATCH. This is a proxy for command-buffer amortization: each batch
row uses a distinct synthetic hidden vector but the same allowed-token frontier.

Environment:
  BATCH_COUNTS='1 2 4 8'  Batch values to test.
  LABEL_REGEX=REGEX       Frontier labels to test.
  REPEATS=40              Probe repeats per frontier/batch.
  RPI5_WARMUPS=3          Untimed GPU dispatches before measurement.
  RAW_OUTPUT=0            Suppress full probe output.
  DRY_RUN=1               Preview selected frontiers without SSH probes.
USAGE
  exit 2
}

model_path="${1:-}"
[[ -n "$model_path" ]] || usage

batch_counts="${BATCH_COUNTS:-1 2 4 8}"
label_regex="${LABEL_REGEX:-^(finite_values:read_file\\.limit|finite_values:edit_mode\\.mode)$}"
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
    awk -F '\t' -v label_regex="$label_regex" '
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
        }
      }'
)"

if [[ -z "$selected" ]]; then
  echo "no frontiers matched label_regex=${label_regex:-<none>}" >&2
  exit 1
fi

printf "batch_sweep_begin\tmodel=%s\tbatch_counts=%s\trepeats=%s\twarmups=%s\tlabel_regex=%s\n" \
  "$model_path" "$batch_counts" "$repeats" "$warmups" "$label_regex"

while IFS=$'\t' read -r label allowed route ids_csv; do
  [[ -n "$label" ]] || continue
  printf "batch_sweep_frontier\tlabel=%s\tallowed=%s\test_route=%s\tids_csv=%s\n" \
    "$label" "$allowed" "$route" "$ids_csv"

  for batch in $batch_counts; do
    printf "batch_sweep_batch\tlabel=%s\tallowed=%s\tbatch=%s\n" "$label" "$allowed" "$batch"
    if [[ "$dry_run" == "1" ]]; then
      continue
    fi

    probe_output="$(
      RPI5_WARMUPS="$warmups" \
      RPI5_BATCH="$batch" \
      bash "$probe" "$label" "$ids_csv" "$repeats"
    )"
    if [[ "$raw_output" != "0" ]]; then
      printf "%s\n" "$probe_output"
    fi

    awk -v label="$label" -v allowed="$allowed" -v route="$route" -v batch="$batch" '
      /^max_abs_diff=/ {
        for (i = 1; i <= NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "max_abs_diff") diff=kv[2]
          if (kv[1] == "gpu_ms_avg") gpu=kv[2]
          if (kv[1] == "cpu_ms") cpu=kv[2]
          if (kv[1] == "speedup") speedup=kv[2]
        }
      }
      /^top1_match=/ {
        for (i = 1; i <= NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "top1_match") top1=kv[2]
          if (kv[1] == "gpu_top1_src") gpu_top1_src=kv[2]
          if (kv[1] == "cpu_top1_src") cpu_top1_src=kv[2]
        }
      }
      /^throttled=/ { throttled=$1 }
      END {
        gpu_per = (gpu != "" && batch > 0) ? sprintf("%.6f", gpu / batch) : ""
        cpu_per = (cpu != "" && batch > 0) ? sprintf("%.6f", cpu / batch) : ""
        if (top1 == "") top1 = "n/a"
        if (gpu_top1_src == "") gpu_top1_src = "n/a"
        if (cpu_top1_src == "") cpu_top1_src = "n/a"
        printf "batch_sweep_result\tlabel=%s\tallowed=%s\test_route=%s\tbatch=%s\tgpu_ms=%s\tcpu_ms=%s\tgpu_ms_per_row=%s\tcpu_ms_per_row=%s\tspeedup=%s\tmax_abs_diff=%s\ttop1_match=%s\tgpu_top1_src=%s\tcpu_top1_src=%s\t%s\n",
          label, allowed, route, batch, gpu, cpu, gpu_per, cpu_per, speedup, diff, top1, gpu_top1_src, cpu_top1_src, throttled
      }' <<<"$probe_output"
  done
done <<<"$selected"
