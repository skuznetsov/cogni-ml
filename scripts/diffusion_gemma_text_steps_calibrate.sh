#!/usr/bin/env bash
set -euo pipefail

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
server="${LLAMA_DIFFUSION_GEMMA_SERVER:-$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-diffusion-gemma-server}"
log_dir="${LOG_DIR:-/tmp}"
ngl="${NGL:-99}"
fa="${FA:-1}"
maxtok="${MAXTOK:-512}"
text_steps="${TEXT_STEPS:-10}"
seed="${SEED:-123}"
steps_list="${STEPS_LIST:-8 10 12}"
prompts_file="${PROMPTS_FILE:-}"

prompt_names=("verify" "resident")
prompts=(
  "Write one concise sentence about why local verification matters."
  "Write one concise sentence about why resident inference helps."
)

if [[ -n "$prompts_file" ]]; then
  if [[ ! -f "$prompts_file" ]]; then
    printf 'error: PROMPTS_FILE not found: %s\n' "$prompts_file" >&2
    exit 2
  fi
  prompt_names=()
  prompts=()
  idx=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" == *$'\t'* ]]; then
      printf 'error: prompts must not contain tab characters\n' >&2
      exit 2
    fi
    if [[ "$line" == *"::"* ]]; then
      prompt_names+=("${line%%::*}")
      prompts+=("${line#*::}")
    else
      prompt_names+=("prompt${idx}")
      prompts+=("$line")
    fi
    idx=$((idx + 1))
  done <"$prompts_file"
fi

if [[ "${#prompts[@]}" -eq 0 ]]; then
  printf 'error: no prompts to calibrate\n' >&2
  exit 2
fi

read -r -a steps <<<"$steps_list"
if [[ "${#steps[@]}" -eq 0 ]]; then
  printf 'error: STEPS_LIST is empty\n' >&2
  exit 2
fi
for step in "${steps[@]}"; do
  if [[ ! "$step" =~ ^[1-9][0-9]*$ ]] || (( step > 128 )); then
    printf 'error: invalid STEPS_LIST value: %s\n' "$step" >&2
    exit 2
  fi
done

mkdir -p "$log_dir"
stamp="$(date +%Y%m%d%H%M%S)"
stderr_log="$log_dir/diffusiongemma_text_calibrate_${stamp}.stderr.log"
reply_log="$log_dir/diffusiongemma_text_calibrate_${stamp}.replies.log"
summary_tsv="$log_dir/diffusiongemma_text_calibrate_${stamp}.summary.tsv"

printf 'server=%s\n' "$server"
printf 'model=%s\n' "$model"
printf 'stderr_log=%s\n' "$stderr_log"
printf 'reply_log=%s\n' "$reply_log"
printf 'summary_tsv=%s\n' "$summary_tsv"
printf 'steps_list=%s\n' "$steps_list"
printf 'prompt_count=%s\n' "${#prompts[@]}"

if [[ ! -x "$server" ]]; then
  printf 'error: server not executable: %s\n' "$server" >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  printf 'error: model not found: %s\n' "$model" >&2
  exit 2
fi

field_value() {
  local line="$1"
  local key="$2"
  local IFS=$'\t'
  local parts=()
  read -r -a parts <<<"$line"
  for part in "${parts[@]:1}"; do
    if [[ "$part" == "$key="* ]]; then
      printf '%s\n' "${part#*=}"
      return 0
    fi
  done
  return 1
}

coproc DG_TEXT_SERVER {
  NGL="$ngl" FA="$fa" MAXTOK="$maxtok" TEXT_STEPS="$text_steps" TEXT_KVCACHE=1 SEED="$seed" \
    "$server" "$model" 2>"$stderr_log"
}
server_pid="$DG_TEXT_SERVER_PID"

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=""
if ! IFS= read -r -t "${READY_TIMEOUT:-180}" ready <&"${DG_TEXT_SERVER[0]}"; then
  printf 'diffusion_gemma_text_steps_calibrate_result ready_timeout=1 stderr_log=%s\n' "$stderr_log"
  exit 3
fi
printf 'server_line=%s\n' "$ready"
if [[ ! "$ready" =~ ^READY[[:space:]][0-9]+$ ]]; then
  printf 'diffusion_gemma_text_steps_calibrate_result bad_ready=1 stderr_log=%s\n' "$stderr_log"
  exit 4
fi

: >"$reply_log"
printf 'name\tstatus\tselected_steps\tselected_ms\tattempts\n' >"$summary_tsv"
clean_count=0
attempt_count=0
for i in "${!prompts[@]}"; do
  name="${prompt_names[$i]}"
  prompt="${prompts[$i]}"
  selected_steps=""
  selected_ms=""
  status="no_clean"
  attempts=0
  for step in "${steps[@]}"; do
    attempts=$((attempts + 1))
    attempt_count=$((attempt_count + 1))
    printf 'TEXT\tsteps=%s\t%s\n' "$step" "$prompt" >&"${DG_TEXT_SERVER[1]}"
    reply=""
    if ! IFS= read -r -t "${TEXT_TIMEOUT:-180}" reply <&"${DG_TEXT_SERVER[0]}"; then
      printf 'diffusion_gemma_text_steps_calibrate_result text_timeout=1 name=%s step=%s stderr_log=%s\n' "$name" "$step" "$stderr_log"
      exit 5
    fi
    printf '%s\n' "$reply" | tee -a "$reply_log"
    if [[ "$reply" == TEXT_OK$'\t'* ]]; then
      selected_steps="$(field_value "$reply" steps || true)"
      selected_ms="$(field_value "$reply" total_ms || true)"
      status="ok"
      clean_count=$((clean_count + 1))
      break
    fi
  done
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "${selected_steps:-NA}" "${selected_ms:-NA}" "$attempts" | tee -a "$summary_tsv"
done

printf 'QUIT\n' >&"${DG_TEXT_SERVER[1]}"
exec {DG_TEXT_SERVER[1]}>&-
wait "$server_pid"
trap - EXIT

ready_count="$(python3 - "$stderr_log" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).read_text(errors="replace").count("diffusion-gemma-server ready"))
PY
)"

printf 'diffusion_gemma_text_steps_calibrate_result clean=%s/%s attempts=%s server_ready_count=%s summary_tsv=%s reply_log=%s stderr_log=%s\n' \
  "$clean_count" "${#prompts[@]}" "$attempt_count" "$ready_count" "$summary_tsv" "$reply_log" "$stderr_log"

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/diffusion_gemma_validate_text_replies.py" --allow-no-clean "$reply_log"

if [[ "$ready_count" != "1" || "$clean_count" -ne "${#prompts[@]}" ]]; then
  exit 6
fi
