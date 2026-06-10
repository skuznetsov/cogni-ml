#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_policy_repeat_sweep.sh MODEL.gguf

Runs rpi5_q6_policy_calibrate.sh across several REPEATS values to expose
near-boundary verdict stability. This is not command-buffer batching; it is a
noise/stability check for the current one-frontier submit path.

Environment:
  REPEAT_COUNTS='20 40 80'   Repeats values to test.
  LABEL_REGEX=REGEX          Frontier labels to calibrate.
  RPI5_WARMUPS=3             Untimed GPU dispatches before measurement.
  MIN_SPEEDUP=1.25           V3D_CLEAR threshold.
  RAW_OUTPUT=0               Suppress full probe output.
  DRY_RUN=1                  Preview selected calibration frontiers.
USAGE
  exit 2
}

model_path="${1:-}"
[[ -n "$model_path" ]] || usage

repeat_counts="${REPEAT_COUNTS:-20 40 80}"
label_regex="${LABEL_REGEX:-^(finite_values:read_file\\.limit|finite_values:edit_mode\\.mode)$}"
warmups="${RPI5_WARMUPS:-3}"
min_speedup="${MIN_SPEEDUP:-1.25}"
raw_output="${RAW_OUTPUT:-0}"
dry_run="${DRY_RUN:-0}"

calibrator="scripts/rpi5_q6_policy_calibrate.sh"
[[ -f "$calibrator" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}

tmp_results="$(mktemp)"
trap 'rm -f "$tmp_results"' EXIT

printf "repeat_sweep_begin\tmodel=%s\trepeat_counts=%s\twarmups=%s\tlabel_regex=%s\tmin_speedup=%s\n" \
  "$model_path" "$repeat_counts" "$warmups" "$label_regex" "$min_speedup"

for repeats in $repeat_counts; do
  printf "repeat_sweep_repeats\trepeats=%s\n" "$repeats"
  output="$(
    RAW_OUTPUT="$raw_output" \
    DRY_RUN="$dry_run" \
    LABEL_REGEX="$label_regex" \
    REPEATS="$repeats" \
    RPI5_WARMUPS="$warmups" \
    MIN_SPEEDUP="$min_speedup" \
    bash "$calibrator" "$model_path"
  )"
  printf "%s\n" "$output"
  if [[ "$dry_run" != "1" ]]; then
    awk -F '\t' -v repeats="$repeats" '
      /^calibration_result/ {
        label = allowed = verdict = gpu = cpu = speedup = top1 = throttled = ""
        for (i = 2; i <= NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "label") label = kv[2]
          if (kv[1] == "allowed") allowed = kv[2]
          if (kv[1] == "verdict") verdict = kv[2]
          if (kv[1] == "gpu_ms") gpu = kv[2]
          if (kv[1] == "cpu_ms") cpu = kv[2]
          if (kv[1] == "speedup") speedup = kv[2]
          if (kv[1] == "top1_match") top1 = kv[2]
          if (kv[1] == "throttled") throttled = kv[2]
        }
        printf "repeat_sweep_result\trepeats=%s\tlabel=%s\tallowed=%s\tverdict=%s\tgpu_ms=%s\tcpu_ms=%s\tspeedup=%s\ttop1_match=%s\tthrottled=%s\n",
          repeats, label, allowed, verdict, gpu, cpu, speedup, top1, throttled
      }' <<<"$output" | tee -a "$tmp_results"
  fi
done

if [[ "$dry_run" == "1" ]]; then
  printf "repeat_sweep_policy\tdry_run=true\n"
  exit 0
fi

awk -F '\t' '
  /^repeat_sweep_result/ {
    label = verdict = ""
    for (i = 2; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == "label") label = kv[2]
      if (kv[1] == "verdict") verdict = kv[2]
    }
    if (label != "" && verdict != "") {
      labels[label] = 1
      verdicts[label, verdict]++
    }
  }
  END {
    for (label in labels) {
      summary = ""
      split("CPU_WINS V3D_NEAR V3D_CLEAR MISMATCH THROTTLED UNKNOWN", order, " ")
      for (i = 1; i <= length(order); i++) {
        key = label SUBSEP order[i]
        if (verdicts[key] > 0) {
          summary = summary (summary == "" ? "" : ",") order[i] ":" verdicts[key]
        }
      }
      if (summary == "") summary = "none"
      printf "repeat_sweep_summary\tlabel=%s\tverdict_counts=%s\n", label, summary
    }
  }' "$tmp_results"
