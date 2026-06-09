#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/cogni-vulkan-runtime/root}"
ICD="${VK_ICD_FILENAMES:-$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json}"
PREPACK="${RPI5_Q6_PREPACK_LOAD:-qwen35_2b_token_embd_q6.pre20}"
SPV="${1:-rpi5_q6_matvec_pre_idx_l256.spv}"
TENSOR="${2:-qwen35_2b_token_embd_q6.cvgp}"
COUNTS="${COUNTS:-3 8 13 16 32 64 128 256}"
REPEATS="${REPEATS:-30}"

export VK_ICD_FILENAMES="$ICD"
export LD_LIBRARY_PATH="$ROOT/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RPI5_Q6_PREPACK_LOAD="$PREPACK"

printf "allowed\tmode\tgpu_ms\tcpu_ms\twinner\tspeedup\ttop1_match\n"
for n in $COUNTS; do
  mode="q6idx${n}_l256"
  out="$(./rpi5_vulkan_q4k_probe "$SPV" file "$TENSOR" "$REPEATS" "$mode")"
  metrics="$(printf "%s\n" "$out" | awk '
    /max_abs_diff=/ {
      gpu=""; cpu="";
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^gpu_ms_avg=/) { split($i, a, "="); gpu=a[2]; }
        if ($i ~ /^cpu_ms=/) { split($i, a, "="); cpu=a[2]; }
      }
      print gpu "\t" cpu;
    }')"
  top1="$(printf "%s\n" "$out" | awk '
    /top1_match=/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^top1_match=/) { split($i, a, "="); print a[2]; }
      }
    }')"
  gpu_ms="$(printf "%s" "$metrics" | cut -f1)"
  cpu_ms="$(printf "%s" "$metrics" | cut -f2)"
  winner="$(awk -v g="$gpu_ms" -v c="$cpu_ms" 'BEGIN { print (g < c ? "V3D" : "CPU") }')"
  speedup="$(awk -v g="$gpu_ms" -v c="$cpu_ms" 'BEGIN { if (g < c) printf "%.3fx", c / g; else printf "%.3fx", g / c; }')"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$n" "$mode" "$gpu_ms" "$cpu_ms" "$winner" "$speedup" "${top1:-unknown}"
done

vcgencmd measure_temp >&2 || true
vcgencmd get_throttled >&2 || true
