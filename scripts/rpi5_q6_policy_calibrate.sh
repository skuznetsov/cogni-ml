#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_policy_calibrate.sh MODEL.gguf

Calibrates the Raspberry Pi 5 V3D Q6 allowed-head policy against real
tokenizer-derived frontiers from estimate_qwen35_allowed_frontiers.cr.

Environment:
  LABEL_REGEX='REGEX'   Frontier labels to calibrate.
  REPEATS=40            Probe repeats per frontier.
  MIN_SPEEDUP=1.25      Minimum measured V3D speedup for V3D_CLEAR.
  RAW_OUTPUT=0          Suppress full probe output; keep result rows.
  DRY_RUN=1             Print selected frontiers without SSH probes.

Default LABEL_REGEX covers:
  tool_call_prefix:start
  finite_values:read_file.limit
  finite_values:edit_mode.mode
USAGE
  exit 2
}

model_path="${1:-}"
[[ -n "$model_path" ]] || usage

label_regex="${LABEL_REGEX:-^(tool_call_prefix:start|finite_values:read_file\\.limit|finite_values:edit_mode\\.mode)$}"
repeats="${REPEATS:-40}"
min_speedup="${MIN_SPEEDUP:-1.25}"
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
        if (ids_csv != "" && (label_regex == "" || label ~ label_regex)) {
          print label "\t" allowed "\t" route "\t" est_v3d_ms "\t" est_cpu_ms "\t" ids_csv
        }
      }'
)"

if [[ -z "$selected" ]]; then
  echo "no frontiers matched label_regex=${label_regex:-<none>}" >&2
  exit 1
fi

tmp_results="$(mktemp)"
trap 'rm -f "$tmp_results"' EXIT

printf "calibration_begin\tmodel=%s\tmin_speedup=%s\trepeats=%s\tlabel_regex=%s\n" \
  "$model_path" "$min_speedup" "$repeats" "$label_regex"

while IFS=$'\t' read -r label allowed route est_v3d_ms est_cpu_ms ids_csv; do
  [[ -n "$label" ]] || continue
  printf "calibration_frontier\tlabel=%s\tallowed=%s\test_route=%s\test_v3d_ms=%s\test_cpu_ms=%s\tids_csv=%s\n" \
    "$label" "$allowed" "$route" "$est_v3d_ms" "$est_cpu_ms" "$ids_csv"
  if [[ "$dry_run" == "1" ]]; then
    continue
  fi

  probe_output="$(bash "$probe" "$label" "$ids_csv" "$repeats")"
  if [[ "$raw_output" != "0" ]]; then
    printf "%s\n" "$probe_output"
  fi

  result="$(
    awk -v label="$label" -v allowed="$allowed" -v est_route="$route" -v est_v3d_ms="$est_v3d_ms" -v est_cpu_ms="$est_cpu_ms" -v min_speedup="$min_speedup" '
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
        printf "calibration_result\tlabel=%s\tallowed=%s\test_route=%s\tverdict=%s\test_v3d_ms=%s\test_cpu_ms=%s\tgpu_ms=%s\tcpu_ms=%s\tspeedup=%s\tv3d_est_ratio=%s\tcpu_est_ratio=%s\ttop1_match=%s\tgpu_top1_src=%s\tcpu_top1_src=%s\t%s\n",
          label, allowed, est_route, verdict, est_v3d_ms, est_cpu_ms, gpu, cpu, speedup, v3d_ratio, cpu_ratio, top1, gpu_top1_src, cpu_top1_src, throttled
      }' <<<"$probe_output"
  )"
  printf "%s\n" "$result" | tee -a "$tmp_results"
done <<<"$selected"

if [[ "$dry_run" == "1" ]]; then
  printf "calibration_policy\tdry_run=true\n"
  exit 0
fi

awk -F '\t' '
  {
    allowed = ""
    verdict = ""
    for (i = 2; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == "allowed") allowed = kv[2] + 0
      if (kv[1] == "verdict") verdict = kv[2]
    }
    if (allowed > 0) {
      seen[allowed] = 1
      if (verdict == "V3D_CLEAR") {
        clear[allowed] = 1
      } else if (verdict == "V3D_NEAR") {
        near[allowed] = 1
      } else {
        cpu_or_blocked[allowed] = 1
      }
    }
  }
  END {
    cpu_max = 0
    clear_min = ""
    near_list = ""
    for (n in seen) {
      if (cpu_or_blocked[n] && n > cpu_max) cpu_max = n
      if (clear[n] && (clear_min == "" || n < clear_min)) clear_min = n
      if (near[n]) near_list = near_list (near_list == "" ? "" : ",") n
    }
    if (clear_min == "") clear_min = "none"
    if (near_list == "") near_list = "none"
    printf "calibration_policy\tpolicy_cpu_max_allowed=%s\tv3d_clear_min_allowed=%s\tnear_boundary_allowed=%s\n",
      cpu_max, clear_min, near_list
  }' "$tmp_results"
