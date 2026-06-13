#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grouped_min_canvas="${GROUPED_MOE_MIN_CANVAS:-4}"
grouped_max_canvas="${GROUPED_MOE_MAX_CANVAS:-16}"
residual_min_canvas="${ATTENTION_RESIDUAL_MIN_CANVAS:-8}"
residual_max_canvas="${ATTENTION_RESIDUAL_MAX_CANVAS:-8}"
prompt_lengths="${SYNTHETIC_PROMPT_LENGTHS:-128}"
canvas_lengths="${SYNTHETIC_CANVAS_LENGTHS:-8}"
rev="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "$name" "$value" >&2
    exit 2
  fi
}

require_range() {
  local min_name="$1"
  local min_value="$2"
  local max_name="$3"
  local max_value="$4"
  require_positive_int "$min_name" "$min_value"
  require_positive_int "$max_name" "$max_value"
  if (( min_value > max_value )); then
    printf '%s must be <= %s\n' "$min_name" "$max_name" >&2
    exit 2
  fi
}

require_range GROUPED_MOE_MIN_CANVAS "$grouped_min_canvas" GROUPED_MOE_MAX_CANVAS "$grouped_max_canvas"
require_range ATTENTION_RESIDUAL_MIN_CANVAS "$residual_min_canvas" ATTENTION_RESIDUAL_MAX_CANVAS "$residual_max_canvas"

base_env="DIFFUSION_GEMMA_SHARED_FFN_BATCH_ROWS=1 DIFFUSION_GEMMA_SHARED_FFN_BATCH_MIN_CANVAS=${grouped_min_canvas} DIFFUSION_GEMMA_SHARED_FFN_BATCH_MAX_CANVAS=${grouped_max_canvas} DIFFUSION_GEMMA_MOE_FFN_BATCH_ROWS=1 DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_ROWS=1 DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_MIN_CANVAS=${grouped_min_canvas} DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_MAX_CANVAS=${grouped_max_canvas}"
variant_env="${base_env} DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_ROWS=1 DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_MIN_CANVAS=${residual_min_canvas} DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_MAX_CANVAS=${residual_max_canvas} DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_ROWS=1 DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_MIN_ROWS=${residual_min_canvas} DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_MAX_ROWS=${residual_max_canvas}"

export DIFFUSION_GEMMA_FULL_ROUTES="${DIFFUSION_GEMMA_FULL_ROUTES:-1}"
export DIFFUSION_GEMMA_CONTEXT_METAL="${DIFFUSION_GEMMA_CONTEXT_METAL:-1}"
export DIFFUSION_GEMMA_PROMPT_PROJ_METAL="${DIFFUSION_GEMMA_PROMPT_PROJ_METAL:-1}"
export DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH="${DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH:-1}"
export DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE="${DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE:-1}"
export DIFFUSION_GEMMA_SPARSE_LOOP_BIN="${DIFFUSION_GEMMA_SPARSE_LOOP_BIN:-/tmp/diffusion_gemma_sparse_loop_grouped_attention_residual_${rev}}"
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
  export LOG_DIR="/tmp/diffusiongemma_grouped_attention_residual_c${residual_min_canvas}_${residual_max_canvas}_$(date +%Y%m%d%H%M%S)"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'GROUPED_MOE_MIN_CANVAS=%s\n' "$grouped_min_canvas"
  printf 'GROUPED_MOE_MAX_CANVAS=%s\n' "$grouped_max_canvas"
  printf 'ATTENTION_RESIDUAL_MIN_CANVAS=%s\n' "$residual_min_canvas"
  printf 'ATTENTION_RESIDUAL_MAX_CANVAS=%s\n' "$residual_max_canvas"
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
