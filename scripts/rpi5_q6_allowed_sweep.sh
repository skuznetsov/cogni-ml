#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/cogni-vulkan-runtime/root}"
ICD="${VK_ICD_FILENAMES:-$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json}"
PREPACK="${RPI5_Q6_PREPACK_LOAD:-qwen35_2b_token_embd_q6.pre20}"
SPV="${1:-rpi5_q6_matvec_pre_idx_l256.spv}"
TENSOR="${2:-qwen35_2b_token_embd_q6.cvgp}"
COUNTS="${COUNTS:-64 256 1024 4096 8192 16384}"
REPEATS="${REPEATS:-30}"

export VK_ICD_FILENAMES="$ICD"
export LD_LIBRARY_PATH="$ROOT/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RPI5_Q6_PREPACK_LOAD="$PREPACK"

for n in $COUNTS; do
  mode="q6idx${n}_l256"
  if [ "$n" -ge 4096 ]; then
    mode="q6idxs${n}_l256"
  fi
  ./rpi5_vulkan_q4k_probe "$SPV" file "$TENSOR" "$REPEATS" "$mode"
done

vcgencmd measure_temp || true
vcgencmd get_throttled || true
