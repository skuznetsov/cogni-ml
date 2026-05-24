#!/bin/bash
# Safe runner for heavy Crystal/Metal commands.
# Usage: scripts/run_safe.sh <binary> [timeout_sec] [max_mem_mb] [args...]
#
# This is based on ../crystal_v2_repo/scripts/run_safe.sh, with one important
# addition for `crystal spec`: monitor and kill the child process tree too,
# because Crystal may spawn a `crystal-run-spec.tmp` process that can survive
# parent interruption and keep consuming memory/GPU.
set -u

BINARY="${1:-}"
TIMEOUT="${2:-120}"
MAX_MEM="${3:-8192}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

if [ -z "$BINARY" ]; then
  echo "Usage: $0 <binary> [timeout_sec=120] [max_mem_mb=8192] [args...]"
  exit 1
fi

STDOUT_TMP=$(mktemp /tmp/run_safe_stdout.XXXXXX)
STDERR_TMP=$(mktemp /tmp/run_safe_stderr.XXXXXX)
WATCHDOG_PID=""
PASSTHROUGH_STDIO="${RUN_SAFE_PASSTHROUGH_STDIO:-0}"
PID=""

log_line() {
  if [ "$PASSTHROUGH_STDIO" = "1" ]; then
    echo "$@" >&2
  else
    echo "$@"
  fi
}

dump_captured_output() {
  if [ "$PASSTHROUGH_STDIO" = "1" ]; then
    if [ -s "$STDERR_TMP" ]; then
      log_line "=== STDERR ==="
      cat "$STDERR_TMP" >&2
    fi
  else
    echo "=== STDOUT ==="
    cat "$STDOUT_TMP"
    echo "=== STDERR ==="
    cat "$STDERR_TMP"
  fi
}

cleanup() {
  if [ -n "$WATCHDOG_PID" ]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  rm -f "$STDOUT_TMP" "$STDERR_TMP"
}
trap cleanup EXIT
trap 'exit 1' TERM

children_of() {
  local root="$1"
  local frontier="$root"
  local all=""
  local next=""
  local p c
  while [ -n "$frontier" ]; do
    next=""
    for p in $frontier; do
      for c in $(pgrep -P "$p" 2>/dev/null || true); do
        all="$all $c"
        next="$next $c"
      done
    done
    frontier="$next"
  done
  echo "$all"
}

process_tree() {
  if [ -z "$PID" ]; then
    return 0
  fi
  echo "$PID $(children_of "$PID")"
}

rss_tree_kb() {
  local total=0
  local rss p
  for p in $(process_tree); do
    rss=$(ps -o rss= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -n "$rss" ]; then
      total=$((total + rss))
    fi
  done
  echo "$total"
}

fd_count_for_pid() {
  local target_pid="$1"
  local tmp
  tmp=$(mktemp /tmp/run_safe_lsof.XXXXXX) || return 0
  (lsof -n -P -p "$target_pid" 2>/dev/null | wc -l | tr -d ' ' >"$tmp") &
  local lsof_pid=$!
  local ticks=0
  while [ $ticks -lt 10 ]; do
    if ! kill -0 "$lsof_pid" 2>/dev/null; then
      wait "$lsof_pid" 2>/dev/null || true
      cat "$tmp"
      rm -f "$tmp"
      return 0
    fi
    sleep 0.1
    ticks=$((ticks + 1))
  done
  kill -9 "$lsof_pid" 2>/dev/null || true
  wait "$lsof_pid" 2>/dev/null || true
  rm -f "$tmp"
  echo ""
}

fd_tree_count() {
  local total=0
  local fds p
  for p in $(process_tree); do
    fds=$(fd_count_for_pid "$p")
    if [ -n "$fds" ]; then
      total=$((total + fds))
    fi
  done
  echo "$total"
}

kill_tree_briefly() {
  local pids
  pids=$(process_tree | tr ' ' '\n' | awk 'NF {print}' | sort -rn)
  if [ -n "$pids" ]; then
    kill -TERM $pids 2>/dev/null || true
    sleep 0.5
    kill -9 $pids 2>/dev/null || true
  fi
}

if [ "$PASSTHROUGH_STDIO" = "1" ]; then
  "$BINARY" "$@" <&0 >&1 2> "$STDERR_TMP" &
else
  "$BINARY" "$@" > "$STDOUT_TMP" 2> "$STDERR_TMP" &
fi
PID=$!
RUN_SAFE_PID=$$

(
  sleep $((TIMEOUT + 2))
  if kill -0 "$PID" 2>/dev/null; then
    FD_COUNT=$(fd_tree_count)
    RSS=$(rss_tree_kb)
    log_line "[KILL] Timeout after ${TIMEOUT}s (tree FDs: ${FD_COUNT:-?}, tree RSS: ${RSS:-?}KB)"
    kill_tree_briefly
    dump_captured_output
    kill -TERM "$RUN_SAFE_PID" 2>/dev/null || true
  fi
) &
WATCHDOG_PID=$!

HALF_SECS=0
MAX_HALF_SECS=$((TIMEOUT * 2))
while [ $HALF_SECS -lt $MAX_HALF_SECS ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID"
    EXIT=$?
    dump_captured_output
    if [ $EXIT -eq 139 ]; then log_line "[CRASH] Segfault (exit 139)"; fi
    if [ $EXIT -eq 134 ]; then log_line "[CRASH] Abort (exit 134)"; fi
    SECS=$((HALF_SECS / 2))
    log_line "[EXIT: $EXIT] after ~${SECS}s"
    exit $EXIT
  fi

  FD_COUNT=$(fd_tree_count)
  RSS=$(rss_tree_kb)

  if [ -n "$FD_COUNT" ] && [ "$FD_COUNT" -gt 1000 ]; then
    SECS=$((HALF_SECS / 2))
    log_line "[KILL] FD leak detected in process tree: $FD_COUNT FDs after ~${SECS}s"
    kill_tree_briefly
    dump_captured_output
    exit 1
  fi

  if [ -n "$RSS" ] && [ "$RSS" -gt $((MAX_MEM * 1024)) ]; then
    SECS=$((HALF_SECS / 2))
    log_line "[KILL] Memory limit for process tree: ${RSS}KB > ${MAX_MEM}MB after ~${SECS}s"
    kill_tree_briefly
    dump_captured_output
    exit 1
  fi

  sleep 0.5
  HALF_SECS=$((HALF_SECS + 1))
done

FD_COUNT=$(fd_tree_count)
RSS=$(rss_tree_kb)
log_line "[KILL] Timeout after ${TIMEOUT}s (tree FDs: ${FD_COUNT:-?}, tree RSS: ${RSS:-?}KB)"
kill_tree_briefly
dump_captured_output
exit 1
