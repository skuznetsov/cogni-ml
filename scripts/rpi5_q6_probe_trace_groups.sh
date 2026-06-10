#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_probe_trace_groups.sh TRACE.log [TRACE2.log ...]

Parses QWEN35_CONSTRAINT_FRONTIER_TRACE=1 logs, plans V3D frontier groups,
and runs each group on the Raspberry Pi 5 Q6 indexed-head probe.

Environment:
  MAX_BATCH=3       Max V3D frontiers per grouped submit.
  CPU_TINY_MAX=3    Route frontiers at or below this size to CPU.
  REPEATS=30        Probe repeats per V3D group.
  RPI5_WARMUPS=3    Untimed GPU dispatches before measurement.
  RAW_OUTPUT=0      Suppress full probe output; keep summary rows.
  DRY_RUN=1         Print planned groups without SSH probes.
USAGE
  exit 2
}

[[ "$#" -gt 0 ]] || usage

max_batch="${MAX_BATCH:-3}"
cpu_tiny_max="${CPU_TINY_MAX:-3}"
repeats="${REPEATS:-30}"
warmups="${RPI5_WARMUPS:-3}"
raw_output="${RAW_OUTPUT:-0}"
dry_run="${DRY_RUN:-0}"

planner="scripts/plan_rpi5_from_frontier_trace.cr"
probe="scripts/rpi5_q6_frontier_probe.sh"
[[ -f "$planner" && -f "$probe" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

plan_output="$(
  MAX_BATCH="$max_batch" \
  CPU_TINY_MAX="$cpu_tiny_max" \
  crystal "$planner" "$@"
)"
printf "%s\n" "$plan_output"

groups="$(
  awk -F '\t' '
    /^trace_v3d_group/ {
      idx = batch = max_allowed = labels = ids_groups = ""
      for (i = 2; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == "idx") idx = kv[2]
        if (kv[1] == "batch") batch = kv[2]
        if (kv[1] == "max_allowed") max_allowed = kv[2]
        if (kv[1] == "labels") labels = kv[2]
        if (kv[1] == "ids_groups") ids_groups = kv[2]
      }
      if (idx != "" && batch != "" && max_allowed != "" && ids_groups != "") {
        print idx "\t" batch "\t" max_allowed "\t" labels "\t" ids_groups
      }
    }' <<<"$plan_output"
)"

if [[ -z "$groups" ]]; then
  printf "trace_probe_policy\tv3d_groups=0\n"
  exit 0
fi

while IFS=$'\t' read -r idx batch max_allowed labels ids_groups; do
  [[ -n "$idx" ]] || continue
  first_ids="${ids_groups%%:*}"
  mode="q6idx${max_allowed}_l256"
  label="trace_group:${idx}:${labels}"
  printf "trace_probe_group\tidx=%s\tbatch=%s\tmax_allowed=%s\tmode=%s\tlabels=%s\tids_groups=%s\n" \
    "$idx" "$batch" "$max_allowed" "$mode" "$labels" "$ids_groups"
  if [[ "$dry_run" == "1" ]]; then
    continue
  fi

  probe_output="$(
    RPI5_WARMUPS="$warmups" \
    RPI5_BATCH="$batch" \
    RPI5_MODE="$mode" \
    RPI5_ROW_IDS_CSV_BATCH="$ids_groups" \
    bash "$probe" "$label" "$first_ids" "$repeats"
  )"
  if [[ "$raw_output" != "0" ]]; then
    printf "%s\n" "$probe_output"
  fi

  awk -v idx="$idx" -v labels="$labels" -v batch="$batch" -v max_allowed="$max_allowed" '
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
      printf "trace_probe_result\tidx=%s\tlabels=%s\tbatch=%s\tmax_allowed=%s\tgpu_ms=%s\tcpu_ms=%s\tgpu_ms_per_frontier=%s\tcpu_ms_per_frontier=%s\tspeedup=%s\tmax_abs_diff=%s\t%s\n",
        idx, labels, batch, max_allowed, gpu, cpu, gpu_per, cpu_per, speedup, diff, throttled
    }' <<<"$probe_output"
done <<<"$groups"
