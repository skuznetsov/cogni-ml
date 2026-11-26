#!/bin/bash
# Safe runner for heavy Crystal/Metal commands.
# Usage: scripts/run_safe.sh <binary> [timeout_sec] [max_mem_mb] [args...]
#
# The workload runs in its own process group so signals, timeouts, memory
# pressure, and wrapper cleanup stop the complete inherited process tree.
# A small watchdog also stops that group if the wrapper itself is killed.
# Deliberate daemonization via setsid() is outside this containment boundary.
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
# the child. It never kills or modifies unrelated processes.
set -u
set -m 2>/dev/null || true

BINARY="${1:-}"
TIMEOUT="${2:-120}"
MAX_MEM="${3:-8192}"
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
if [ "$#" -ge 3 ]; then
  shift 3
else
  shift "$#"
fi

if [ -z "$BINARY" ]; then
  echo "Usage: $0 <binary> [timeout_sec=120] [max_mem_mb=8192] [args...]"
  exit 1
fi
if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]{0,8}$ && "$MAX_MEM" =~ ^(0|[1-9][0-9]{0,8})$ ]]; then
  echo "[ABORT] run_safe needs a positive timeout and a non-negative RSS limit (0 disables RSS limiting)" >&2
  exit 2
fi
if ! [[ "$MIN_FREE_PCT" =~ ^(100|[1-9]?[0-9])$ ]]; then
  echo "[ABORT] COGNI_RUN_SAFE_MIN_FREE_PCT must be an integer between 0 and 100" >&2
  exit 2
fi

STDOUT_TMP=$(mktemp /tmp/run_safe_stdout.XXXXXX)
STDERR_TMP=$(mktemp /tmp/run_safe_stderr.XXXXXX)
WATCHDOG_PID=""
PRESSURE_TOOL=""
PASSTHROUGH_STDIO="${RUN_SAFE_PASSTHROUGH_STDIO:-0}"
WAIT_QUIET_SEC="${COGNI_RUN_SAFE_WAIT_QUIET_SEC:-0}"
QUIET_POLL_SEC="${COGNI_RUN_SAFE_QUIET_POLL_SEC:-1}"
QUIET_PROC_PCT="${COGNI_RUN_SAFE_QUIET_PROC_PCT:-50}"
QUIET_TOTAL_PCT="${COGNI_RUN_SAFE_QUIET_TOTAL_PCT:-100}"
REQUIRE_QUIET="${COGNI_RUN_SAFE_REQUIRE_QUIET:-0}"
PID=""
PGID=""
CAN_KILL_PGID=0
PARENT_DONE=0
PARENT_EXIT=0
TREE_STOPPED=0
LAUNCHING=0
PENDING_SIGNAL_NAME=""
PENDING_SIGNAL_STATUS=""

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

pids_in_pgid() {
  if [ "$CAN_KILL_PGID" -ne 1 ] || [ -z "$PGID" ]; then
    return 0
  fi
  ps -axo pid=,pgid= 2>/dev/null | awk -v pg="$PGID" '$2 == pg {print $1}'
}

process_tree() {
  if [ -z "$PID" ]; then
    return 0
  fi
  if [ "$CAN_KILL_PGID" -eq 1 ]; then
    pids_in_pgid
  elif kill -0 "$PID" 2>/dev/null; then
    echo "$PID"
  fi
}

workload_running() {
  if [ "$CAN_KILL_PGID" -eq 1 ] && kill -0 "-$PGID" 2>/dev/null; then
    return 0
  fi
  [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null
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
  local snapshot
  snapshot=$(LC_ALL=C "$PRESSURE_TOOL" -Q 2>/dev/null) || return 1
  awk '/System-wide memory free percentage:/ { sub(/%$/, "", $NF); print $NF }' <<< "$snapshot"
}

require_memory_headroom() {
  local free_pct
  if [ "$MIN_FREE_PCT" -le 0 ]; then
    return 0
  fi
  if ! PRESSURE_TOOL=$(command -v memory_pressure); then
    if [ "$(uname -s)" = Darwin ]; then
      log_line "[ABORT] Cannot find memory_pressure before launch"
      exit 75
    fi
    return 0
  fi
  free_pct=$(system_free_pct)
  if ! [[ "$free_pct" =~ ^(100|[1-9]?[0-9])$ ]]; then
    log_line "[ABORT] Cannot read system memory pressure before launch"
    exit 75
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

stop_watchdog() {
  if [ -n "$WATCHDOG_PID" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill -KILL "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
  WATCHDOG_PID=""
}

kill_tree_briefly() {
  local target="$PID"
  if [ "$TREE_STOPPED" -eq 1 ]; then
    return 0
  fi
  TREE_STOPPED=1
  if [ "$CAN_KILL_PGID" -eq 1 ]; then
    target="-$PGID"
  fi
  [ -n "$target" ] || return 0
  kill -TERM "$target" 2>/dev/null || true
  sleep 0.5
  kill -KILL "$target" 2>/dev/null || true
}

cleanup() {
  trap - EXIT HUP INT QUIT TERM USR1
  stop_watchdog
  if [ "$TREE_STOPPED" -eq 0 ] && workload_running; then
    kill_tree_briefly
  fi
  if [ -n "$PID" ]; then
    wait "$PID" 2>/dev/null || true
  fi
  rm -f "$STDOUT_TMP" "$STDERR_TMP"
}

terminate_for_signal() {
  local signal_name="$1"
  local status="$2"
  trap '' HUP INT QUIT TERM USR1
  log_line "[KILL] Received SIG${signal_name}; terminating process tree"
  kill_tree_briefly
  exit "$status"
}

handle_signal() {
  local signal_name="$1"
  local status="$2"
  if [ "$LAUNCHING" -eq 1 ]; then
    PENDING_SIGNAL_NAME="$signal_name"
    PENDING_SIGNAL_STATUS="$status"
    return 0
  fi
  terminate_for_signal "$signal_name" "$status"
}

finish_launch() {
  LAUNCHING=0
  if [ -n "$PENDING_SIGNAL_NAME" ]; then
    terminate_for_signal "$PENDING_SIGNAL_NAME" "$PENDING_SIGNAL_STATUS"
  fi
}

timeout_wrapper() {
  trap '' HUP INT QUIT TERM USR1
  log_line "[KILL] Timeout after ${TIMEOUT}s"
  kill_tree_briefly
  dump_captured_output
  exit 1
}

trap cleanup EXIT
trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT
trap 'handle_signal QUIT 131' QUIT
trap 'handle_signal TERM 143' TERM
trap timeout_wrapper USR1

require_memory_headroom
wait_for_quiet_host
LAUNCHING=1

if [ "$PASSTHROUGH_STDIO" = "1" ]; then
  "$BINARY" "$@" <&0 >&1 2> "$STDERR_TMP" &
else
  "$BINARY" "$@" > "$STDOUT_TMP" 2> "$STDERR_TMP" &
fi
PID=$!
actual_pgid=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
SELF_PGID=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')
if [ -n "$actual_pgid" ] && [ "$actual_pgid" = "$PID" ] && [ "$actual_pgid" != "$SELF_PGID" ]; then
  PGID="$actual_pgid"
  CAN_KILL_PGID=1
elif [ -z "$actual_pgid" ] && ! kill -0 "$PID" 2>/dev/null; then
  # Very short commands may finish before ps observes them. Bash job control
  # still assigns PID as the group ID; keep monitoring only if that group has
  # surviving descendants, otherwise return the already captured exit status.
  wait "$PID"
  PARENT_EXIT=$?
  PARENT_DONE=1
  PGID="$PID"
  if kill -0 "-$PGID" 2>/dev/null; then
    CAN_KILL_PGID=1
  else
    finish_launch
    dump_captured_output
    log_line "[EXIT: $PARENT_EXIT] after ~0s"
    exit "$PARENT_EXIT"
  fi
else
  finish_launch
  log_line "[ABORT] Cannot isolate the workload process group"
  kill_tree_briefly
  dump_captured_output
  exit 75
fi
finish_launch
set +m 2>/dev/null || true
RUN_SAFE_PID=$$
STARTED=$SECONDS
DEADLINE=$((STARTED + TIMEOUT))

(
  WATCHDOG_SELF=$(/bin/sh -c 'printf "%s\n" "$PPID"')
  while :; do
    WATCHDOG_PARENT=$(/bin/ps -o ppid= -p "$WATCHDOG_SELF" 2>/dev/null | tr -d ' ')
    if [ "$WATCHDOG_PARENT" != "$RUN_SAFE_PID" ]; then
      kill -KILL "-$PGID" 2>/dev/null || true
      exit 0
    fi
    if (( SECONDS >= DEADLINE )); then break; fi
    sleep 0.5
  done
  kill -USR1 "$RUN_SAFE_PID" 2>/dev/null || true
  kill -TERM "-$PGID" 2>/dev/null || true
  sleep 0.5
  kill -KILL "-$PGID" 2>/dev/null || true
) &
WATCHDOG_PID=$!

while (( SECONDS < DEADLINE )); do
  if [ "$PARENT_DONE" -eq 0 ] && ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID"
    PARENT_EXIT=$?
    PARENT_DONE=1
  fi

  if [ "$PARENT_DONE" -eq 1 ] && ! workload_running; then
    stop_watchdog
    dump_captured_output
    if [ $PARENT_EXIT -eq 139 ]; then log_line "[CRASH] Segfault (exit 139)"; fi
    if [ $PARENT_EXIT -eq 134 ]; then log_line "[CRASH] Abort (exit 134)"; fi
    SECS=$((SECONDS - STARTED))
    log_line "[EXIT: $PARENT_EXIT] after ~${SECS}s"
    exit "$PARENT_EXIT"
  fi

  if [ -n "$PRESSURE_TOOL" ]; then
    FREE_PCT=$(system_free_pct)
    if ! [[ "$FREE_PCT" =~ ^(100|[1-9]?[0-9])$ ]]; then
      log_line "[KILL] Cannot read system memory pressure during execution"
      kill_tree_briefly
      dump_captured_output
      exit 1
    fi
    if [ "$FREE_PCT" -le "$MIN_FREE_PCT" ]; then
      log_line "[KILL] System memory pressure: free ${FREE_PCT}% <= ${MIN_FREE_PCT}% after ~$((SECONDS - STARTED))s"
      kill_tree_briefly
      dump_captured_output
      exit 1
    fi
  fi

  FD_COUNT=$(fd_tree_count)

  if [ -n "$FD_COUNT" ] && [ "$FD_COUNT" -gt 1000 ]; then
    SECS=$((SECONDS - STARTED))
    log_line "[KILL] FD leak detected in process tree: $FD_COUNT FDs after ~${SECS}s"
    kill_tree_briefly
    dump_captured_output
    exit 1
  fi

  if [ "$MAX_MEM" -gt 0 ]; then
    RSS=$(rss_tree_kb)
    if [ -n "$RSS" ] && [ "$RSS" -gt $((MAX_MEM * 1024)) ]; then
      SECS=$((SECONDS - STARTED))
      log_line "[KILL] Memory limit for process tree: ${RSS}KB > ${MAX_MEM}MB after ~${SECS}s"
      kill_tree_briefly
      dump_captured_output
      exit 1
    fi
  fi

  sleep 0.5
done

timeout_wrapper
