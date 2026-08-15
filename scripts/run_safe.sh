#!/bin/bash
# Safe runner for heavy Crystal/Metal commands.
# Usage: scripts/run_safe.sh <binary> [timeout_sec] [max_mem_mb] [args...]
#
# This is based on ../crystal_v2_repo/scripts/run_safe.sh, with one important
# addition for `crystal spec`: monitor and kill the child process tree too,
# because Crystal may spawn a `crystal-run-spec.tmp` process that can survive
# parent interruption and keep consuming memory/GPU.
#
# System-pressure guard (enabled by default):
#   COGNI_RUN_SAFE_MIN_FREE_PCT=12
#   COGNI_RUN_SAFE_MIN_FREE_PCT=0   # explicit opt-out
# kills the process tree if `memory_pressure -Q` reports free memory at or
# below the threshold. This catches unified-memory/Metal/compressor pressure
# that is not always visible in the child RSS alone.
#
# Optional benchmark-noise preflight:
#   COGNI_RUN_SAFE_WAIT_QUIET_SEC=600 COGNI_RUN_SAFE_REQUIRE_QUIET=1
# waits for other-process CPU load to fall below thresholds before launching
# the child. This is fail-closed for perf runs: it does not kill or modify
# user processes, it only waits and optionally aborts before model load.
set -u
set -m 2>/dev/null || true

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
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
WAIT_QUIET_SEC="${COGNI_RUN_SAFE_WAIT_QUIET_SEC:-0}"
QUIET_POLL_SEC="${COGNI_RUN_SAFE_QUIET_POLL_SEC:-1}"
QUIET_PROC_PCT="${COGNI_RUN_SAFE_QUIET_PROC_PCT:-50}"
QUIET_TOTAL_PCT="${COGNI_RUN_SAFE_QUIET_TOTAL_PCT:-100}"
REQUIRE_QUIET="${COGNI_RUN_SAFE_REQUIRE_QUIET:-0}"
PID=""
PGID=""
CAN_KILL_PGID=0
SEEN_PIDS=""
PARENT_DONE=0
PARENT_EXIT=0

log_line() {
  if [ "$PASSTHROUGH_STDIO" = "1" ]; then
    echo "$@" >&2
  else
    echo "$@"
  fi
}

busy_report() {
  ps -Ao pid=,pcpu=,comm= 2>/dev/null | awk \
    -v self="$$" \
    -v proc_thr="$QUIET_PROC_PCT" \
    -v total_thr="$QUIET_TOTAL_PCT" '
      $1 == self { next }
      {
        pid=$1
        cpu=$2 + 0
        $1=""; $2=""
        sub(/^  */, "", $0)
        cmd=$0
        total += cpu
        if (cpu >= proc_thr && proc_thr > 0) {
          busy_count += 1
          busy = busy sprintf(" pid=%s cpu=%.1f cmd=%s\n", pid, cpu, cmd)
        }
      }
      END {
        total_busy = (total_thr > 0 && total >= total_thr)
        if (busy_count > 0 || total_busy) {
          printf("busy total_cpu=%.1f proc_threshold=%.1f total_threshold=%.1f\n", total, proc_thr, total_thr)
          if (busy != "") printf("%s", busy)
        }
      }'
}

wait_for_quiet_host() {
  local waited=0
  local report=""
  if [ "$WAIT_QUIET_SEC" -le 0 ] && [ "$REQUIRE_QUIET" != "1" ]; then
    return 0
  fi
  if [ "$QUIET_POLL_SEC" -le 0 ]; then
    QUIET_POLL_SEC=1
  fi
  while :; do
    report=$(busy_report)
    if [ -z "$report" ]; then
      return 0
    fi
    if [ "$WAIT_QUIET_SEC" -le 0 ] || [ "$waited" -ge "$WAIT_QUIET_SEC" ]; then
      break
    fi
    log_line "[WAIT] Host busy before run_safe launch (${waited}/${WAIT_QUIET_SEC}s):"
    printf '%s\n' "$report" >&2
    sleep "$QUIET_POLL_SEC"
    waited=$((waited + QUIET_POLL_SEC))
  done
  if [ "$REQUIRE_QUIET" = "1" ]; then
    log_line "[ABORT] Host still busy before run_safe launch:"
    printf '%s\n' "$report" >&2
    exit 75
  fi
  if [ -n "$report" ]; then
    log_line "[WARN] Host busy before run_safe launch:"
    printf '%s\n' "$report" >&2
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

pids_in_pgid() {
  if [ -z "$PGID" ]; then
    return 0
  fi
  ps -axo pid=,pgid= 2>/dev/null | awk -v pg="$PGID" '$2 == pg {print $1}'
}

process_tree() {
  local roots p c
  if [ -z "$PID" ]; then
    return 0
  fi
  roots="$PID $SEEN_PIDS $(pids_in_pgid)"
  for p in $roots; do
    if kill -0 "$p" 2>/dev/null; then
      echo "$p"
    fi
    for c in $(children_of "$p"); do
      if kill -0 "$c" 2>/dev/null; then
        echo "$c"
      fi
    done
  done | awk 'NF && !seen[$1]++ {print $1}'
}

remember_tree() {
  local p
  for p in $(process_tree); do
    case " $SEEN_PIDS " in
      *" $p "*) ;;
      *) SEEN_PIDS="$SEEN_PIDS $p" ;;
    esac
  done
}

live_tree_without_parent() {
  local p
  for p in $(process_tree); do
    if [ "$p" != "$PID" ] && kill -0 "$p" 2>/dev/null; then
      echo "$p"
    fi
  done
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

system_free_pct() {
  if ! command -v memory_pressure >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  memory_pressure -Q 2>/dev/null | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/, "", $2); print $2; exit}'
}

require_memory_headroom() {
  local free_pct
  if [ "$MIN_FREE_PCT" -le 0 ]; then
    return 0
  fi
  free_pct=$(system_free_pct)
  if [ -z "$free_pct" ]; then
    # `memory_pressure` is macOS-specific. Preserve portability when it is not
    # installed, but fail closed when the command exists and its output cannot
    # be interpreted.
    if command -v memory_pressure >/dev/null 2>&1; then
      log_line "[ABORT] Cannot read system memory pressure before launch"
      exit 75
    fi
    return 0
  fi
  if [ "$free_pct" -le "$MIN_FREE_PCT" ]; then
    log_line "[ABORT] Insufficient system memory headroom before launch: free ${free_pct}% <= ${MIN_FREE_PCT}%"
    exit 75
  fi
  log_line "[PREFLIGHT] System memory free ${free_pct}% > ${MIN_FREE_PCT}%"
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
  if [ "$CAN_KILL_PGID" -eq 1 ] && [ -n "$PGID" ]; then
    kill -TERM "-$PGID" 2>/dev/null || true
  fi
  if [ -n "$pids" ]; then
    kill -TERM $pids 2>/dev/null || true
    sleep 0.5
    if [ "$CAN_KILL_PGID" -eq 1 ] && [ -n "$PGID" ]; then
      kill -9 "-$PGID" 2>/dev/null || true
    fi
    kill -9 $pids 2>/dev/null || true
  fi
}

if [ "$PASSTHROUGH_STDIO" = "1" ]; then
  require_memory_headroom
  wait_for_quiet_host
  "$BINARY" "$@" <&0 >&1 2> "$STDERR_TMP" &
else
  require_memory_headroom
  wait_for_quiet_host
  "$BINARY" "$@" > "$STDOUT_TMP" 2> "$STDERR_TMP" &
fi
PID=$!
PGID="$PID"
actual_pgid=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
if [ -n "$actual_pgid" ]; then
  PGID="$actual_pgid"
fi
SELF_PGID=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')
if [ -n "$PGID" ] && [ "$PGID" != "$SELF_PGID" ]; then
  CAN_KILL_PGID=1
fi
set +m 2>/dev/null || true
RUN_SAFE_PID=$$

(
  sleep $((TIMEOUT + 2))
  remember_tree
  if [ -n "$(process_tree)" ]; then
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
  remember_tree

  if [ "$PARENT_DONE" -eq 0 ] && ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID"
    PARENT_EXIT=$?
    PARENT_DONE=1
  fi

  if [ "$PARENT_DONE" -eq 1 ] && [ -z "$(live_tree_without_parent)" ]; then
    dump_captured_output
    if [ $PARENT_EXIT -eq 139 ]; then log_line "[CRASH] Segfault (exit 139)"; fi
    if [ $PARENT_EXIT -eq 134 ]; then log_line "[CRASH] Abort (exit 134)"; fi
    SECS=$((HALF_SECS / 2))
    log_line "[EXIT: $PARENT_EXIT] after ~${SECS}s"
    exit $PARENT_EXIT
  fi

  FD_COUNT=$(fd_tree_count)
  RSS=$(rss_tree_kb)
  FREE_PCT=$(system_free_pct)

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

  if [ -n "$FREE_PCT" ] && [ "$MIN_FREE_PCT" -gt 0 ] && [ "$FREE_PCT" -le "$MIN_FREE_PCT" ]; then
    SECS=$((HALF_SECS / 2))
    log_line "[KILL] System memory pressure: free ${FREE_PCT}% <= ${MIN_FREE_PCT}% after ~${SECS}s (tree RSS: ${RSS:-?}KB)"
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
