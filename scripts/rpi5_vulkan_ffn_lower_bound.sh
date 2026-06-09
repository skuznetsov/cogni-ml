#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GATE.cvgp UP.cvgp DOWN.cvgp [repeats]" >&2
  exit 2
fi

gate="$1"
up="$2"
down="$3"
repeats="${4:-7}"
matvec_spv="${5:-rpi5_q4_matvec_pre.spv}"
matvec_mode="${6:-pre}"

root="${HOME}/cogni-vulkan-runtime/root"
export VK_ICD_FILENAMES="${HOME}/cogni-vulkan-runtime/icd/broadcom_icd.user.json"
export LD_LIBRARY_PATH="${root}/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

run_one() {
  local label="$1"
  local tensor="$2"
  echo "=== ${label} ==="
  ./rpi5_vulkan_q4k_probe "${matvec_spv}" file "${tensor}" "${repeats}" "${matvec_mode}" |
    grep -E 'device=|prepack_ms|max_abs_diff|cpu_neon4_ms'
}

run_one gate "${gate}"
run_one up "${up}"
run_one down "${down}"

if command -v vcgencmd >/dev/null 2>&1; then
  vcgencmd measure_temp || true
  vcgencmd get_throttled || true
fi
