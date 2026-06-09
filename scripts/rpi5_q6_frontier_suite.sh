#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_frontier_suite.sh MODEL.gguf [MAX_FRONTIERS]

Runs selected tokenizer-derived allowed-token frontiers on the Raspberry Pi 5
V3D Q6 tied-head probe. It uses estimate_qwen35_allowed_frontiers.cr, filters
rows by route, then calls scripts/rpi5_q6_frontier_probe.sh for each row.

Environment:
  ROUTE_FILTER=V3D       Route to run from estimator output.
  LABEL_REGEX=REGEX     Optional awk regex for frontier labels.
  REPEATS=40            Probe repeats per frontier.
  DRY_RUN=1             Print selected rows without running SSH probes.
  RAW_OUTPUT=0           Suppress full per-frontier probe output; keep summary.
  MIN_SPEEDUP=1.25       Minimum measured V3D speedup for V3D_CLEAR verdict.

Examples:
  DRY_RUN=1 scripts/rpi5_q6_frontier_suite.sh MODEL.gguf
  LABEL_REGEX='finite_values:edit_mode' scripts/rpi5_q6_frontier_suite.sh MODEL.gguf 2
USAGE
  exit 2
}

model_path="${1:-}"
max_frontiers="${2:-3}"
[[ -n "$model_path" ]] || usage

route_filter="${ROUTE_FILTER:-V3D}"
label_regex="${LABEL_REGEX:-}"
repeats="${REPEATS:-40}"
dry_run="${DRY_RUN:-0}"
raw_output="${RAW_OUTPUT:-1}"
min_speedup="${MIN_SPEEDUP:-1.25}"

estimator="scripts/estimate_qwen35_allowed_frontiers.cr"
probe="scripts/rpi5_q6_frontier_probe.sh"
[[ -f "$estimator" && -f "$probe" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

selected="$(
  crystal "$estimator" "$model_path" |
    awk -F '\t' -v route_filter="$route_filter" -v label_regex="$label_regex" '
      {
        label=$1
        allowed=""
        route=""
        est_v3d_ms=""
        est_cpu_ms=""
        ids_csv=""
        for (i = 2; i <= NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "allowed") allowed=kv[2]
          if (kv[1] == "route") route=kv[2]
          if (kv[1] == "est_v3d_ms") est_v3d_ms=kv[2]
          if (kv[1] == "est_cpu_ms") est_cpu_ms=kv[2]
          if (kv[1] == "ids_csv") ids_csv=kv[2]
        }
        if (route == route_filter && ids_csv != "" && (label_regex == "" || label ~ label_regex)) {
          print label "\t" allowed "\t" route "\t" est_v3d_ms "\t" est_cpu_ms "\t" ids_csv
        }
      }'
)"

if [[ -z "$selected" ]]; then
  echo "no frontiers matched route=$route_filter label_regex=${label_regex:-<none>}" >&2
  exit 1
fi

count=0
while IFS=$'\t' read -r label allowed route est_v3d_ms est_cpu_ms ids_csv; do
  [[ -n "$label" ]] || continue
  count=$((count + 1))
  if (( count > max_frontiers )); then
    break
  fi

  printf "frontier=%s allowed=%s route=%s est_v3d_ms=%s est_cpu_ms=%s ids_csv=%s\n" "$label" "$allowed" "$route" "$est_v3d_ms" "$est_cpu_ms" "$ids_csv"
  if [[ "$dry_run" == "1" ]]; then
    continue
  fi
  probe_output="$(bash "$probe" "$label" "$ids_csv" "$repeats")"
  if [[ "$raw_output" != "0" ]]; then
    printf "%s\n" "$probe_output"
  fi
  summary="$(
    awk -v label="$label" -v allowed="$allowed" -v route="$route" -v est_v3d_ms="$est_v3d_ms" -v est_cpu_ms="$est_cpu_ms" -v min_speedup="$min_speedup" '
      /^max_abs_diff=/ {
        for (i = 1; i <= NF; i++) {
          split($i, kv, "=")
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
        v3d_ratio = (est_v3d_ms > 0 && gpu != "") ? sprintf("%.3f", gpu / est_v3d_ms) : ""
        cpu_ratio = (est_cpu_ms > 0 && cpu != "") ? sprintf("%.3f", cpu / est_cpu_ms) : ""
        measured_speedup = (gpu > 0 && cpu > 0) ? cpu / gpu : 0
        verdict = "UNKNOWN"
        if (top1 != "true") {
          verdict = "MISMATCH"
        } else if (throttled != "" && throttled != "throttled=0x0") {
          verdict = "THROTTLED"
        } else if (gpu != "" && cpu != "" && gpu < cpu && measured_speedup >= min_speedup) {
          verdict = "V3D_CLEAR"
        } else if (gpu != "" && cpu != "" && gpu < cpu) {
          verdict = "V3D_NEAR"
        } else if (gpu != "" && cpu != "") {
          verdict = "CPU_WINS"
        }
        printf "frontier_result\tlabel=%s\tallowed=%s\troute=%s\tverdict=%s\test_v3d_ms=%s\test_cpu_ms=%s\tgpu_ms=%s\tcpu_ms=%s\tspeedup=%s\tv3d_est_ratio=%s\tcpu_est_ratio=%s\ttop1_match=%s\tgpu_top1_src=%s\tcpu_top1_src=%s\t%s\n",
          label, allowed, route, verdict, est_v3d_ms, est_cpu_ms, gpu, cpu, speedup, v3d_ratio, cpu_ratio, top1, gpu_top1_src, cpu_top1_src, throttled
      }' <<<"$probe_output"
  )"
  printf "%s\n" "$summary"
done <<<"$selected"
