#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_resident_stdin_gate.sh CAPTURE.jsonl [MIN_ALLOWED]

Runs the RPi5 Q6 resident stdin smoke and fails unless the binary stdin request
protocol preserves top1 parity, logit fields, diff bounds, and clean throttle
state.

Environment:
  MAX_ABS_DIFF=0.0001  Maximum allowed CPU/GPU absolute diff.
  MAX_ROWS=2           Number of captured rows to send as stdin requests.
  RPI5_WARMUPS=3
USAGE
  exit 2
}

capture="${1:-}"
min_allowed="${2:-${MIN_ALLOWED:-4}}"
[[ -n "$capture" && -f "$capture" ]] || usage

max_abs_diff="${MAX_ABS_DIFF:-0.0001}"

output="$(scripts/rpi5_q6_resident_stdin_smoke.sh "$capture" "$min_allowed")"
printf "%s\n" "$output"

summary_row="$(awk -F '\t' '/^resident_stdin_smoke_result/ {print; found=1} END {exit found ? 0 : 1}' <<<"$output")" || {
  echo "resident_stdin_smoke_result row missing" >&2
  exit 1
}

STDIN_OUTPUT="$output" SUMMARY_ROW="$summary_row" \
awk -F '\t' -v max_abs_diff="$max_abs_diff" '
  function field(row, key,    n, parts, kv) {
    n = split(row, parts, "\t")
    for (i = 1; i <= n; i++) {
      split(parts[i], kv, "=")
      if (kv[1] == key) return kv[2]
    }
    return ""
  }
  BEGIN {
    summary = ENVIRON["SUMMARY_ROW"]
    output = ENVIRON["STDIN_OUTPUT"]
    requests = field(summary, "requests")
    summary_mismatches = field(summary, "top1_mismatches")
    throttled = field(summary, "throttled")
    n = split(output, lines, "\n")
    seen = 0
    max_diff_seen = 0.0
    row_mismatches = 0
    for (li = 1; li <= n; li++) {
      if (lines[li] !~ /^resident_stdin_result/) continue
      seen++
      diff = field(lines[li], "max_abs_diff")
      top1_ok = field(lines[li], "top1_match")
      gpu_logit = field(lines[li], "gpu_top1_logit")
      cpu_logit = field(lines[li], "cpu_top1_logit")
      if (diff == "" || top1_ok == "" || gpu_logit == "" || cpu_logit == "") {
        print "resident stdin row parse failed" > "/dev/stderr"
        exit 1
      }
      if ((gpu_logit + 0.0) != gpu_logit || (cpu_logit + 0.0) != cpu_logit) {
        print "resident stdin logit parse failed" > "/dev/stderr"
        exit 1
      }
      if ((diff + 0.0) > max_diff_seen) max_diff_seen = diff + 0.0
      if (top1_ok != "true") row_mismatches++
    }
    if (requests == "" || summary_mismatches == "") {
      print "resident stdin summary parse failed" > "/dev/stderr"
      exit 1
    }
    if ((requests + 0) != seen) {
      printf "resident stdin result count %d does not match summary %s\n", seen, requests > "/dev/stderr"
      exit 1
    }
    if ((summary_mismatches + 0) != 0 || row_mismatches != 0) {
      printf "resident stdin top1 mismatches summary=%s rows=%d\n", summary_mismatches, row_mismatches > "/dev/stderr"
      exit 1
    }
    if (max_diff_seen > (max_abs_diff + 0.0)) {
      printf "max_abs_diff %.9f above threshold %.9f\n", max_diff_seen, max_abs_diff > "/dev/stderr"
      exit 1
    }
    if (throttled != "" && throttled != "0x0") {
      printf "unexpected throttle state %s\n", throttled > "/dev/stderr"
      exit 1
    }
    printf "resident_stdin_gate_result\trequests=%s\tmax_abs_diff=%g\ttop1_mismatches=%s\tmax_abs_diff_limit=%s\tthrottled=%s\n",
      requests, max_diff_seen, summary_mismatches, max_abs_diff, throttled
  }'
