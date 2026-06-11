#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_quiet_check_$(date +%Y%m%d%H%M%S)}"
quiet_ms="${QUIET_MS:-15000}"
load_threshold="${LOAD_THRESHOLD:-30}"
total_threshold="${TOTAL_THRESHOLD:-90}"
summary_format="${SUMMARY_FORMAT:-json}"

set +e
LOG_DIR="$log_dir" \
CHECK_QUIET_ONLY=1 \
REQUIRE_QUIET=1 \
QUIET_MS="$quiet_ms" \
LOAD_THRESHOLD="$load_threshold" \
TOTAL_THRESHOLD="$total_threshold" \
"$repo_root/scripts/diffusion_gemma_prompt_variant_abba.sh"
quiet_rc=$?
set -e

"$repo_root/scripts/diffusion_gemma_quiet_snapshot_summary.py" \
  --format "$summary_format" \
  --load-threshold "$load_threshold" \
  --total-threshold "$total_threshold" \
  "$log_dir"

exit "$quiet_rc"
