#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

min_canvas="${GROUPED_MOE_MIN_CANVAS:-4}"
max_canvas="${GROUPED_MOE_MAX_CANVAS:-8}"
prompt_lengths="${SYNTHETIC_PROMPT_LENGTHS:-128}"
canvas_lengths="${SYNTHETIC_CANVAS_LENGTHS:-4,8}"
rev="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 2
  fi
}

require_positive_int GROUPED_MOE_MIN_CANVAS "$min_canvas"
require_positive_int GROUPED_MOE_MAX_CANVAS "$max_canvas"
if (( min_canvas > max_canvas )); then
  printf 'GROUPED_MOE_MIN_CANVAS must be <= GROUPED_MOE_MAX_CANVAS\n' >&2
  exit 2
fi

base_env="DIFFUSION_GEMMA_SHARED_FFN_BATCH_ROWS=1 DIFFUSION_GEMMA_SHARED_FFN_BATCH_MIN_CANVAS=${min_canvas} DIFFUSION_GEMMA_SHARED_FFN_BATCH_MAX_CANVAS=${max_canvas} DIFFUSION_GEMMA_MOE_FFN_BATCH_ROWS=1"
variant_env="DIFFUSION_GEMMA_GROUPED_MOE_POLICY=1 DIFFUSION_GEMMA_GROUPED_MOE_POLICY_MIN_CANVAS=${min_canvas} DIFFUSION_GEMMA_GROUPED_MOE_POLICY_MAX_CANVAS=${max_canvas}"

export DIFFUSION_GEMMA_FULL_ROUTES="${DIFFUSION_GEMMA_FULL_ROUTES:-1}"
export DIFFUSION_GEMMA_CONTEXT_METAL="${DIFFUSION_GEMMA_CONTEXT_METAL:-1}"
export DIFFUSION_GEMMA_PROMPT_PROJ_METAL="${DIFFUSION_GEMMA_PROMPT_PROJ_METAL:-1}"
export DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH="${DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH:-1}"
export DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE="${DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE:-1}"
export DIFFUSION_GEMMA_SPARSE_LOOP_BIN="${DIFFUSION_GEMMA_SPARSE_LOOP_BIN:-/tmp/diffusion_gemma_sparse_loop_fullroute_grouped_${rev}}"
export BASE_ENV="$base_env"
export VARIANT_ENV="$variant_env"
export SYNTHETIC_PROMPT_LENGTHS="$prompt_lengths"
export SYNTHETIC_CANVAS_LENGTHS="$canvas_lengths"
export CACHE_WARMUPS="${CACHE_WARMUPS:-1}"
export CACHE_REPEATS="${CACHE_REPEATS:-4}"
export WARMUPS="${WARMUPS:-1}"
export REPEATS="${REPEATS:-2}"
export QUIET_MS="${QUIET_MS:-15000}"
export LOAD_THRESHOLD="${LOAD_THRESHOLD:-40}"
export TOTAL_THRESHOLD="${TOTAL_THRESHOLD:-90}"
export REQUIRE_QUIET="${REQUIRE_QUIET:-1}"
export PROMOTION_FORMAT="${PROMOTION_FORMAT:-kv}"

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="/tmp/diffusiongemma_grouped_moe_fullroutes_c${min_canvas}_${max_canvas}_$(date +%Y%m%d%H%M%S)"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'GROUPED_MOE_MIN_CANVAS=%s\n' "$min_canvas"
  printf 'GROUPED_MOE_MAX_CANVAS=%s\n' "$max_canvas"
  printf 'SYNTHETIC_PROMPT_LENGTHS=%s\n' "$SYNTHETIC_PROMPT_LENGTHS"
  printf 'SYNTHETIC_CANVAS_LENGTHS=%s\n' "$SYNTHETIC_CANVAS_LENGTHS"
  printf 'BASE_ENV=%s\n' "$BASE_ENV"
  printf 'VARIANT_ENV=%s\n' "$VARIANT_ENV"
  printf 'DIFFUSION_GEMMA_SPARSE_LOOP_BIN=%s\n' "$DIFFUSION_GEMMA_SPARSE_LOOP_BIN"
  printf 'LOG_DIR=%s\n' "$LOG_DIR"
  printf 'command=%s/scripts/diffusion_gemma_prompt_variant_abba.sh\n' "$repo_root"
  exit 0
fi

"$repo_root/scripts/diffusion_gemma_prompt_variant_abba.sh"
