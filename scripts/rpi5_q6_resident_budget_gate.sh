#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_resident_budget_gate.sh CAPTURE.jsonl [MIN_ALLOWED]

Runs the RPi5 Q6 allowed-head transport budget and fails unless the resident
upload request remains correct and faster than the CPU selected-row oracle.

Environment:
  MIN_UPLOAD_SPEEDUP=1.25  Required cpu_selected/resident_upload speedup.
  MAX_ABS_DIFF=0.0001      Maximum allowed CPU/GPU absolute diff.
  REPEATS=30
  RPI5_WARMUPS=3
USAGE
  exit 2
}

capture="${1:-}"
min_allowed="${2:-${MIN_ALLOWED:-4}}"
[[ -n "$capture" && -f "$capture" ]] || usage

min_speedup="${MIN_UPLOAD_SPEEDUP:-1.25}"
max_abs_diff="${MAX_ABS_DIFF:-0.0001}"

output="$(RAW_OUTPUT=0 scripts/rpi5_q6_transport_budget.sh "$capture" "$min_allowed")"
printf "%s\n" "$output"

budget_row="$(awk -F '\t' '/^resident_budget_result/ {print; found=1} END {exit found ? 0 : 1}' <<<"$output")" || {
  echo "resident_budget_result row missing" >&2
  exit 1
}
transport_row="$(awk -F '\t' '/^transport_budget_result/ {print; found=1} END {exit found ? 0 : 1}' <<<"$output")" || {
  echo "transport_budget_result row missing" >&2
  exit 1
}

BUDGET_ROW="$budget_row" TRANSPORT_ROW="$transport_row" \
awk -F '\t' -v min_speedup="$min_speedup" -v max_abs_diff="$max_abs_diff" '
  function field(row, key,    n, parts, kv) {
    n = split(row, parts, "\t")
    for (i = 1; i <= n; i++) {
      split(parts[i], kv, "=")
      if (kv[1] == key) return kv[2]
    }
    return ""
  }
  BEGIN {
    resident = ENVIRON["BUDGET_ROW"]
    transport = ENVIRON["TRANSPORT_ROW"]
    speedup = field(resident, "upload_request_speedup")
    diff = field(transport, "max_abs_diff")
    throttled = field(resident, "throttled")
    gsub(/x$/, "", speedup)
    if (speedup == "" || diff == "") {
      print "resident budget parse failed" > "/dev/stderr"
      exit 1
    }
    if ((speedup + 0.0) < (min_speedup + 0.0)) {
      printf "resident upload speedup %.3f below threshold %.3f\n", speedup, min_speedup > "/dev/stderr"
      exit 1
    }
    if ((diff + 0.0) > (max_abs_diff + 0.0)) {
      printf "max_abs_diff %.9f above threshold %.9f\n", diff, max_abs_diff > "/dev/stderr"
      exit 1
    }
    if (throttled != "" && throttled != "0x0") {
      printf "unexpected throttle state %s\n", throttled > "/dev/stderr"
      exit 1
    }
    printf "resident_budget_gate_result\tmin_allowed=%s\tupload_request_speedup=%s\tmax_abs_diff=%s\tmin_speedup=%s\tmax_abs_diff_limit=%s\tthrottled=%s\n",
      field(resident, "min_allowed"), speedup, diff, min_speedup, max_abs_diff, throttled
  }'
