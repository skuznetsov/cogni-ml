#!/bin/bash
# Exercise watchdog failures without a model or a real memory-pressure event.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
tmp="$(mktemp -d /tmp/cogni-spec-watchdog-check.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT
crystal_bin="${CRYSTAL:-crystal}"
runner="$root/scripts/run_safe.sh"

cat > "$tmp/probe.cr" <<'CRYSTAL'
require "spec"
require "rss_guard"

lib WatchdogProbeLibC
  fun sleep(seconds : UInt32) : UInt32
end

WatchdogProbeLibC.sleep(30_u32) if ENV["WATCHDOG_PROBE_DISCOVERY"]? == "1"

describe "spec watchdog" do
  it("quick example one") { sleep 0.75.seconds }
  it("quick example two") { sleep 0.75.seconds }
  it("ordinary assertion failure") { 1.should eq(2) }
  it("blocked in native code") { WatchdogProbeLibC.sleep(30_u32) }

  it "changes memory telemetry" do
    File.write(ENV["WATCHDOG_PROBE_STATE"], ENV["WATCHDOG_PROBE_TELEMETRY"])
    WatchdogProbeLibC.sleep(30_u32)
  end
end
CRYSTAL

mkdir "$tmp/bin"
cat > "$tmp/bin/memory_pressure" <<'BASH'
#!/bin/bash
state=$(cat "$WATCHDOG_PROBE_STATE" 2>/dev/null || true)
case "$state" in
  low) echo "System-wide memory free percentage: 5%" ;;
  missing) exit 1 ;;
  *) echo "System-wide memory free percentage: 95%" ;;
esac
BASH
chmod +x "$tmp/bin/memory_pressure"

CRYSTAL_PATH="$root/spec/support:$("$crystal_bin" env CRYSTAL_PATH)" \
  "$runner" "$crystal_bin" 60 4096 build "$tmp/probe.cr" -o "$tmp/probe"

run_case() {
  local name="$1" expected="$2" diagnostic="$3" example="$4"
  shift 4
  local status=0 log="$tmp/$name.log"
  rm -f "$tmp/state"
  PATH="$tmp/bin:$PATH" WATCHDOG_PROBE_STATE="$tmp/state" \
    COGNI_RUN_SAFE_MIN_FREE_PCT=0 COGNI_SPEC_RSS_GUARD=1 \
    COGNI_SPEC_MAX_RSS_MB=0 COGNI_SPEC_MIN_FREE_PCT=0 \
    COGNI_SPEC_EXAMPLE_TIMEOUT_SEC=1 COGNI_SPEC_RSS_POLL_SEC=0.1 \
    env "$@" "$runner" "$tmp/probe" 8 256 --example "$example" > "$log" 2>&1 || status=$?
  if [[ "$status" != "$expected" ]] || ! grep -Eq "$diagnostic" "$log"; then
    printf '%s: expected exit %s and /%s/, got %s\n' "$name" "$expected" "$diagnostic" "$status" >&2
    cat "$log" >&2
    return 1
  fi
  printf '%s: PASS (exit %s)\n' "$name" "$status"
}

run_case success 0 '2 examples, 0 failures' 'quick example'
run_case assertion 1 '1 examples?, 1 failures?' 'ordinary assertion failure'
run_case timeout 137 'SPEC_TIMEOUT_GUARD.*blocked in native code' 'blocked in native code'
run_case discovery 137 'SPEC_TIMEOUT_GUARD.*spec discovery' 'quick example' WATCHDOG_PROBE_DISCOVERY=1
run_case rss 99 'SPEC_RSS_GUARD.*RSS .* > 1 MiB' 'quick example' COGNI_SPEC_MAX_RSS_MB=1
run_case pressure 137 'SPEC_PRESSURE_GUARD.*changes memory telemetry.*system free 5%' 'changes memory telemetry' \
  COGNI_SPEC_MIN_FREE_PCT=25 COGNI_SPEC_EXAMPLE_TIMEOUT_SEC=0 WATCHDOG_PROBE_TELEMETRY=low
run_case missing 137 'SPEC_PRESSURE_GUARD.*cannot read system memory headroom' 'changes memory telemetry' \
  COGNI_SPEC_MIN_FREE_PCT=25 COGNI_SPEC_EXAMPLE_TIMEOUT_SEC=0 WATCHDOG_PROBE_TELEMETRY=missing
run_case invalid 99 'COGNI_SPEC_MAX_RSS_MB must be an integer' 'quick example' COGNI_SPEC_MAX_RSS_MB=typo
run_case opt_out 0 '2 examples, 0 failures' 'quick example' COGNI_SPEC_RSS_GUARD=0 COGNI_SPEC_MAX_RSS_MB=1
run_case tree_pressure 1 'System memory pressure: free 5% <= 25%' 'changes memory telemetry' \
  COGNI_SPEC_RSS_GUARD=0 COGNI_RUN_SAFE_MIN_FREE_PCT=25 WATCHDOG_PROBE_TELEMETRY=low
run_case tree_missing 1 'Cannot read system memory pressure during execution' 'changes memory telemetry' \
  COGNI_SPEC_RSS_GUARD=0 COGNI_RUN_SAFE_MIN_FREE_PCT=25 WATCHDOG_PROBE_TELEMETRY=missing
