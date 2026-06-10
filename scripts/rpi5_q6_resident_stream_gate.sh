#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_resident_stream_gate.sh CAPTURE.jsonl [MIN_ALLOWED]

Runs the RPi5 Q6 allowed-head resident stream bench and fails unless per-step
resident requests remain correct and faster than the CPU selected-row oracle.

Environment:
  MIN_STREAM_SPEEDUP=1.20  Required cpu_selected/resident_stream speedup.
  MIN_REPEATS=10           Minimum REPEATS allowed for this timing gate.
  MAX_ABS_DIFF=0.0001      Maximum allowed CPU/GPU absolute diff.
  REPEATS=30
USAGE
  exit 2
}

capture="${1:-}"
min_allowed="${2:-${MIN_ALLOWED:-4}}"
[[ -n "$capture" && -f "$capture" ]] || usage

min_speedup="${MIN_STREAM_SPEEDUP:-1.20}"
min_repeats="${MIN_REPEATS:-10}"
max_abs_diff="${MAX_ABS_DIFF:-0.0001}"
repeats="${REPEATS:-30}"

[[ "$min_repeats" =~ ^[0-9]+$ && "$repeats" =~ ^[0-9]+$ ]] || {
  echo "MIN_REPEATS and REPEATS must be non-negative integers" >&2
  exit 2
}
if (( repeats < min_repeats )); then
  printf "REPEATS=%s below resident stream gate minimum %s\n" "$repeats" "$min_repeats" >&2
  exit 2
fi

output="$(RAW_OUTPUT=0 scripts/rpi5_q6_resident_stream_bench.sh "$capture" "$min_allowed")"
printf "%s\n" "$output"

stream_row="$(awk -F '\t' '/^resident_stream_bench_result/ {print; found=1} END {exit found ? 0 : 1}' <<<"$output")" || {
  echo "resident_stream_bench_result row missing" >&2
  exit 1
}

STREAM_ROW="$stream_row" \
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
    row = ENVIRON["STREAM_ROW"]
    speedup = field(row, "speedup")
    diff = field(row, "max_abs_diff")
    mismatches = field(row, "top1_mismatches")
    throttled = field(row, "throttled")
    gsub(/x$/, "", speedup)
    if (speedup == "" || diff == "" || mismatches == "") {
      print "resident stream parse failed" > "/dev/stderr"
      exit 1
    }
    if ((speedup + 0.0) < (min_speedup + 0.0)) {
      printf "resident stream speedup %.3f below threshold %.3f\n", speedup, min_speedup > "/dev/stderr"
      exit 1
    }
    if ((diff + 0.0) > (max_abs_diff + 0.0)) {
      printf "max_abs_diff %.9f above threshold %.9f\n", diff, max_abs_diff > "/dev/stderr"
      exit 1
    }
    if ((mismatches + 0) != 0) {
      printf "top1 mismatches %s\n", mismatches > "/dev/stderr"
      exit 1
    }
    if (throttled != "" && throttled != "0x0") {
      printf "unexpected throttle state %s\n", throttled > "/dev/stderr"
      exit 1
    }
    printf "resident_stream_gate_result\tmin_allowed=%s\trequests=%s\tstream_speedup=%s\tmax_abs_diff=%s\ttop1_mismatches=%s\tmin_speedup=%s\tmax_abs_diff_limit=%s\tthrottled=%s\n",
      field(row, "min_allowed"), field(row, "requests"), speedup, diff, mismatches, min_speedup, max_abs_diff, throttled
  }'
