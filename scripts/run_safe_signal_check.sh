#!/usr/bin/env bash
# Focused regression check for run_safe process-tree cleanup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="${COGNI_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
SELF="$ROOT/scripts/run_safe_signal_check.sh"

if [[ "${1:-}" == "--fixture" ]]; then
  PID_DIR="${2:?missing fixture PID directory}"
  trap '' HUP INT TERM
  printf '%s\n' "$$" > "$PID_DIR/payload.pid"
  /bin/bash -c 'trap "" HUP INT TERM; exec sleep 300' &
  printf '%s\n' "$!" > "$PID_DIR/child.pid"
  wait
  exit $?
fi

TMP_DIR="$(mktemp -d /tmp/run_safe_signal_check.XXXXXX)"

pid_from_file() {
  local path="$1"
  local pid=""
  if [[ -f "$path" ]]; then
    read -r pid < "$path" || true
  fi
  case "$pid" in
    ''|*[!0-9]*) return 1 ;;
    *) printf '%s\n' "$pid" ;;
  esac
}

pid_is_live() {
  local pid="$1"
  local state=""
  state="$(ps -o stat= -p "$pid" 2>/dev/null | awk 'NR == 1 {print $1}')"
  if [[ -n "$state" ]]; then
    [[ "$state" != Z* ]]
    return
  fi
  kill -0 "$pid" 2>/dev/null
}

cleanup() {
  local pid_file pid
  for pid_file in "$TMP_DIR"/*/*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid="$(pid_from_file "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && pid_is_live "$pid"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

wait_for_fixture() {
  local case_dir="$1"
  local ticks=0
  while [[ $ticks -lt 100 ]]; do
    if [[ -s "$case_dir/payload.pid" && -s "$case_dir/child.pid" ]]; then
      return 0
    fi
    sleep 0.05
    ticks=$((ticks + 1))
  done
  echo "fixture did not start for $(basename "$case_dir")" >&2
  return 1
}

wait_until_dead() {
  local pid="$1"
  local ticks=0
  while [[ $ticks -lt 60 ]]; do
    if ! pid_is_live "$pid"; then
      return 0
    fi
    sleep 0.05
    ticks=$((ticks + 1))
  done
  return 1
}

verify_case() {
  local case_name="$1"
  local expected_status="$2"
  local case_dir="$TMP_DIR/$case_name"
  local actual_status pid role failed=0
  read -r actual_status < "$case_dir/status"
  if [[ "$actual_status" -ne "$expected_status" ]]; then
    echo "$case_name: expected exit $expected_status, got $actual_status" >&2
    failed=1
  fi
  for role in payload child; do
    [[ -e "$case_dir/$role.pid" ]] || continue
    pid="$(pid_from_file "$case_dir/$role.pid")"
    if ! wait_until_dead "$pid"; then
      echo "$case_name: orphaned $role PID $pid" >&2
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]]
}

record_background_case() {
  local case_name="$1"
  local action="$2"
  local timeout="${3:-30}"
  local case_dir="$TMP_DIR/$case_name"
  local wrapper_pid status
  mkdir -p "$case_dir"
  COGNI_RUN_SAFE_MIN_FREE_PCT=0 \
    /bin/bash "$RUN_SAFE" /bin/bash "$timeout" 64 "$SELF" --fixture "$case_dir" \
    > "$case_dir/output.log" 2>&1 &
  wrapper_pid=$!
  printf '%s\n' "$wrapper_pid" > "$case_dir/wrapper.pid"
  wait_for_fixture "$case_dir"
  case "$action" in
    wait) ;;
    *) kill -s "$action" "$wrapper_pid" ;;
  esac
  set +e
  wait "$wrapper_pid" 2>/dev/null
  status=$?
  set -e
  printf '%s\n' "$status" > "$case_dir/status"
}

record_ctrl_c_case() {
  local case_dir="$TMP_DIR/int"
  mkdir -p "$case_dir"
  /usr/bin/expect -f - "$RUN_SAFE" "$SELF" "$case_dir" <<'EXPECT'
log_user 0
set timeout 10
set run_safe [lindex $argv 0]
set self [lindex $argv 1]
set case_dir [lindex $argv 2]
spawn /usr/bin/env COGNI_RUN_SAFE_MIN_FREE_PCT=0 /bin/bash $run_safe /bin/bash 30 64 $self --fixture $case_dir
for {set ticks 0} {$ticks < 100} {incr ticks} {
  if {[file exists "$case_dir/payload.pid"] && [file exists "$case_dir/child.pid"]} { break }
  after 50
}
send -- "\003"
expect {
  eof {
    set result [wait]
    if {[llength $result] > 4 && [lindex $result 4] == "CHILDKILLED"} {
      set status 130
    } else {
      set status [lindex $result 3]
    }
  }
  timeout { set status 124; catch {close}; catch {wait} }
}
set status_file [open "$case_dir/status" w]
puts $status_file $status
close $status_file
EXPECT
}

record_normal_exit_case() {
  local case_dir="$TMP_DIR/normal"
  local status
  mkdir -p "$case_dir"
  set +e
  COGNI_RUN_SAFE_MIN_FREE_PCT=0 \
    /bin/bash "$RUN_SAFE" /usr/bin/true 5 64 > "$case_dir/output.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$case_dir/status"
}

if [[ ! -x /usr/bin/expect ]]; then
  echo "missing /usr/bin/expect; cannot exercise a real PTY Ctrl-C" >&2
  exit 2
fi

failed=0
record_normal_exit_case
verify_case normal 0 || failed=1
record_ctrl_c_case
verify_case int 130 || failed=1
record_background_case term TERM
verify_case term 143 || failed=1
record_background_case timeout wait 1
verify_case timeout 1 || failed=1
record_background_case killed_wrapper KILL
verify_case killed_wrapper 137 || failed=1

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "run_safe signal cleanup: PASS"
