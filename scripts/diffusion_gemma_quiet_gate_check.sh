#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_quiet_check_$(date +%Y%m%d%H%M%S)}"
quiet_ms="${QUIET_MS:-15000}"
load_threshold="${LOAD_THRESHOLD:-30}"
total_threshold="${TOTAL_THRESHOLD:-90}"
summary_format="${SUMMARY_FORMAT:-json}"
require_candidate="${REQUIRE_CANDIDATE:-0}"

set +e
LOG_DIR="$log_dir" \
CHECK_QUIET_ONLY=1 \
REQUIRE_QUIET=1 \
REQUIRE_CANDIDATE="$require_candidate" \
QUIET_MS="$quiet_ms" \
LOAD_THRESHOLD="$load_threshold" \
TOTAL_THRESHOLD="$total_threshold" \
"$repo_root/scripts/diffusion_gemma_prompt_variant_abba.sh"
quiet_rc=$?
set -e

summary="$("$repo_root/scripts/diffusion_gemma_quiet_snapshot_summary.py" \
  --format "$summary_format" \
  --load-threshold "$load_threshold" \
  --total-threshold "$total_threshold" \
  "$log_dir")"
printf '%s\n' "$summary"

if (( quiet_rc != 0 )); then
  exit "$quiet_rc"
fi

if [[ "$require_candidate" == "1" ]]; then
  if [[ "$summary_format" == "json" ]]; then
    python3 -c 'import json,sys; sys.exit(0 if json.loads(sys.stdin.read()).get("quiet_candidate") else 4)' <<<"$summary"
  elif ! grep -q '^quiet_candidate=true$' <<<"$summary"; then
    exit 4
  fi
fi

exit 0
