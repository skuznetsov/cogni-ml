#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/rpi5_q6_resident_stdin_smoke.sh CAPTURE.jsonl [MIN_ALLOWED]

Runs a small stdin-driven resident probe smoke on the Raspberry Pi 5. The probe
keeps Vulkan objects and the prepacked Q6 head alive, then handles one
hidden-vector + row-id frontier per stdin line.

Environment:
  RPI5_HOST        SSH host, default raspberrypi.local
  RPI5_REMOTE_DIR Remote probe dir, default ~/cogni-ml-vulkan-probe
  RPI5_WARMUPS=3   Untimed GPU dispatches before stdin requests.
  MAX_ROWS=2       Number of captured rows to send as stdin requests.
  MIN_ALLOWED=4    Filter threshold for V3D-routed rows.
USAGE
  exit 2
}

capture="${1:-}"
min_allowed="${2:-${MIN_ALLOWED:-4}}"
[[ -n "$capture" && -f "$capture" ]] || usage
[[ "$min_allowed" =~ ^[0-9]+$ ]] || {
  echo "MIN_ALLOWED must be non-negative" >&2
  exit 2
}

host="${RPI5_HOST:-raspberrypi.local}"
remote_dir="${RPI5_REMOTE_DIR:-~/cogni-ml-vulkan-probe}"
warmups="${RPI5_WARMUPS:-3}"
max_rows="${MAX_ROWS:-2}"
converter="scripts/export_allowed_head_capture_replay.cr"
[[ -f "$converter" ]] || {
  echo "run from the cogni-ml repository root" >&2
  exit 2
}
[[ "$max_rows" =~ ^[0-9]+$ && "$max_rows" != "0" ]] || {
  echo "MAX_ROWS must be a positive integer" >&2
  exit 2
}

local_f32="/tmp/rpi5_resident_stdin_$$.f32"
remote_base="$(basename "$local_f32")"
cleanup() {
  rm -f "$local_f32"
}
trap cleanup EXIT

plan="$(MIN_ALLOWED="$min_allowed" MAX_ROWS="$max_rows" crystal "$converter" "$capture" "$local_f32")"
printf "%s\n" "$plan"

rows="$(awk -F= '$1=="replay_rows" {print $2}' <<<"$plan")"
hidden_dim="$(awk -F= '$1=="hidden_dim" {print $2}' <<<"$plan")"
max_allowed="$(awk -F= '$1=="max_allowed" {print $2}' <<<"$plan")"
ids_groups="$(awk -F= '$1=="ids_groups" {print $2}' <<<"$plan")"
first_ids="${ids_groups%%:*}"
mode="q6idx${max_allowed}_l256"

if [[ -z "$rows" || -z "$hidden_dim" || -z "$max_allowed" || -z "$ids_groups" || "$rows" == "0" ]]; then
  echo "no replay rows selected" >&2
  exit 1
fi

first_ids="$(
  awk -v max="$max_allowed" -F, '
    {
      for (i = 1; i <= max; i++) {
        if (i <= NF) v = $i
        else v = 0
        if (i > 1) printf ","
        printf "%s", v
      }
      printf "\n"
    }' <<<"$first_ids"
)"

scp -q "$local_f32" "$host:$remote_dir/$remote_base"

output="$(
  ssh "$host" bash -s -- "$remote_dir" "$mode" "$first_ids" "$ids_groups" "$remote_base" "$rows" "$hidden_dim" "$warmups" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
mode="$2"
ids_csv="$3"
ids_groups="$4"
x_f32_load="$5"
rows="$6"
hidden_dim="$7"
warmups="$8"

root="$HOME/cogni-vulkan-runtime/root"
if [[ "$remote_dir" == "~/"* ]]; then
  remote_dir="$HOME/${remote_dir#~/}"
fi
trap 'rm -f "$remote_dir"/rpi5_resident_stdin_req_*.f32 "$remote_dir/$x_f32_load"' EXIT
export VK_ICD_FILENAMES="$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json"
export LD_LIBRARY_PATH="$root/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RPI5_Q6_PREPACK_LOAD="qwen35_2b_token_embd_q6.pre20"
export RPI5_ROW_IDS_CSV="$ids_csv"
export RPI5_BATCH=1
export RPI5_RESIDENT_STDIN=1
export RPI5_WARMUPS="$warmups"

cd "$remote_dir"
row_bytes=$((hidden_dim * 4))
stdin_path="/tmp/rpi5_resident_stdin_$$.tsv"
: >"$stdin_path"
IFS=':' read -r -a groups <<<"$ids_groups"
for ((i = 0; i < rows; i++)); do
  req="rpi5_resident_stdin_req_$$_${i}.f32"
  dd if="$x_f32_load" of="$req" bs="$row_bytes" skip="$i" count=1 status=none
  printf "%s\t%s\n" "$req" "${groups[$i]}" >>"$stdin_path"
done

./rpi5_vulkan_q4k_probe rpi5_q6_matvec_pre_idx_l256.spv file qwen35_2b_token_embd_q6.cvgp 1 "$mode" <"$stdin_path"
rm -f "$stdin_path"
vcgencmd get_throttled || true
REMOTE
)"

printf "%s\n" "$output"

result_count="$(awk -F '\t' '/^resident_stdin_result/ {n++} END {print n + 0}' <<<"$output")"
mismatches="$(awk -F '\t' '/^resident_stdin_result/ {for (i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="top1_match" && kv[2]!="true") bad++}} END {print bad + 0}' <<<"$output")"
throttled="$(awk '/^throttled=/ {print $1}' <<<"$output" | tail -1)"

if [[ "$result_count" != "$rows" ]]; then
  echo "resident stdin result count $result_count does not match rows $rows" >&2
  exit 1
fi
if [[ "$mismatches" != "0" ]]; then
  echo "resident stdin top1 mismatches: $mismatches" >&2
  exit 1
fi
if [[ -n "$throttled" && "$throttled" != "throttled=0x0" ]]; then
  echo "unexpected throttle state $throttled" >&2
  exit 1
fi

printf "resident_stdin_smoke_result\trequests=%s\tmax_allowed=%s\ttop1_mismatches=%s\t%s\n" \
  "$rows" "$max_allowed" "$mismatches" "$throttled"
