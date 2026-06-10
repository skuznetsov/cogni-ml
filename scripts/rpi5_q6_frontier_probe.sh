#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_frontier_probe.sh LABEL IDS_CSV [REPEATS]

Runs one exact tokenizer-derived allowed-token frontier on the Raspberry Pi 5
V3D Q6 tied-head probe. Use IDS_CSV from:

  crystal scripts/estimate_qwen35_allowed_frontiers.cr MODEL.gguf

Environment:
  RPI5_HOST        SSH host, default raspberrypi.local
  RPI5_REMOTE_DIR Remote probe dir, default $HOME/cogni-ml-vulkan-probe on the Pi
  RPI5_SPV         Shader, default rpi5_q6_matvec_pre_idx_l256.spv
  RPI5_TENSOR      Tensor artifact, default qwen35_2b_token_embd_q6.cvgp
  RPI5_PREPACK     Cached prepack, default qwen35_2b_token_embd_q6.pre20
  RPI5_MODE        Probe mode override, default q6idx${allowed}_l256
  RPI5_WARMUPS     Untimed GPU dispatches before measurement, default 0
  RPI5_BATCH       Hidden rows per submit for q6idx probe, default 1
  RPI5_ROW_IDS_CSV_BATCH
                   Colon-separated per-batch row-id frontiers.
USAGE
  exit 2
}

label="${1:-}"
ids_csv="${2:-}"
repeats="${3:-40}"

[[ -n "$label" && -n "$ids_csv" ]] || usage

allowed_count="$(awk -F, 'NF == 0 || $0 == "" { exit 1 } { print NF }' <<<"$ids_csv")" || {
  echo "invalid IDS_CSV: $ids_csv" >&2
  exit 2
}

host="${RPI5_HOST:-raspberrypi.local}"
remote_dir="${RPI5_REMOTE_DIR:-}"
remote_dir_arg="${remote_dir:-__DEFAULT__}"
spv="${RPI5_SPV:-rpi5_q6_matvec_pre_idx_l256.spv}"
tensor="${RPI5_TENSOR:-qwen35_2b_token_embd_q6.cvgp}"
prepack="${RPI5_PREPACK:-qwen35_2b_token_embd_q6.pre20}"
mode="${RPI5_MODE:-q6idx${allowed_count}_l256}"
warmups="${RPI5_WARMUPS:-0}"
batch="${RPI5_BATCH:-1}"
batch_ids_csv="${RPI5_ROW_IDS_CSV_BATCH:-}"

printf "label=%s allowed=%s mode=%s host=%s repeats=%s warmups=%s batch=%s ids_csv=%s\n" "$label" "$allowed_count" "$mode" "$host" "$repeats" "$warmups" "$batch" "$ids_csv"
if [[ -n "$batch_ids_csv" ]]; then
  printf "batch_ids_csv=%s\n" "$batch_ids_csv"
fi

ssh "$host" bash -s -- "$remote_dir_arg" "$spv" "$tensor" "$prepack" "$repeats" "$mode" "$ids_csv" "$warmups" "$batch" "$batch_ids_csv" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
spv="$2"
tensor="$3"
prepack="$4"
repeats="$5"
mode="$6"
ids_csv="$7"
warmups="$8"
batch="$9"
batch_ids_csv="${10:-}"

root="$HOME/cogni-vulkan-runtime/root"
if [[ "$remote_dir" == "__DEFAULT__" ]]; then
  remote_dir="$HOME/cogni-ml-vulkan-probe"
elif [[ "$remote_dir" == "~/"* ]]; then
  remote_dir="$HOME/${remote_dir#~/}"
fi
export VK_ICD_FILENAMES="$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json"
export LD_LIBRARY_PATH="$root/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RPI5_Q6_PREPACK_LOAD="$prepack"
export RPI5_ROW_IDS_CSV="$ids_csv"
export RPI5_ROW_IDS_CSV_BATCH="$batch_ids_csv"
export RPI5_WARMUPS="$warmups"
export RPI5_BATCH="$batch"

cd "$remote_dir"
./rpi5_vulkan_q4k_probe "$spv" file "$tensor" "$repeats" "$mode"
vcgencmd get_throttled || true
REMOTE
