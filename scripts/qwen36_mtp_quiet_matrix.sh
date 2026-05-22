#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

: "${GEN:=128}"
: "${REPEATS:=3}"
: "${QUIET_MS:=600000}"
: "${LOAD_THRESHOLD:=50}"
: "${TOTAL_THRESHOLD:=100}"
: "${CRYSTAL_CACHE_DIR:=/tmp/cogni_ml_qwen36_mtp_quiet_matrix}"

export CRYSTAL_CACHE_DIR

crystal build --no-codegen bin/qwen36_mtp_baseline_matrix.cr --error-trace

exec crystal run bin/qwen36_mtp_baseline_matrix.cr -- \
  --gen="${GEN}" \
  --repeats="${REPEATS}" \
  --compare-plain \
  --run-available \
  --no-commands \
  --wait-quiet-ms="${QUIET_MS}" \
  --load-warning-threshold="${LOAD_THRESHOLD}" \
  --load-total-warning-threshold="${TOTAL_THRESHOLD}" \
  --require-quiet
